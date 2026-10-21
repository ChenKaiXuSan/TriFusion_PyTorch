#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Triangulate SAM3D 2D body keypoints into multi-view 3D GT."""

from __future__ import annotations

import argparse
import json
import logging
import itertools
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import yaml
import concurrent.futures
from tqdm import tqdm


LOGGER = logging.getLogger("sam3d_kpt_triangulation")
VIEW_NAMES = ("front", "left", "right")


def _as_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def load_config(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def reshape_k(raw: Iterable[float]) -> np.ndarray:
    return np.asarray(list(raw), dtype=np.float64).reshape(3, 3)


def _size_tuple(raw: Iterable[int]) -> Tuple[int, int]:
    size = tuple(int(x) for x in raw)
    if len(size) != 2:
        raise ValueError(f"Expected [width, height], got {raw}")
    return size


def view_size_maps(raw: Any) -> Dict[str, Tuple[int, int]]:
    if isinstance(raw, dict):
        return {view: _size_tuple(raw[view]) for view in VIEW_NAMES}
    size = _size_tuple(raw)
    return {view: size for view in VIEW_NAMES}


def resize_transform(
    old_size: Tuple[int, int],
    new_size: Tuple[int, int],
    mode: str = "letterbox",
) -> np.ndarray:
    old_w, old_h = old_size
    new_w, new_h = new_size
    if mode == "non_uniform":
        sx, sy = new_w / old_w, new_h / old_h
        tx, ty = 0.0, 0.0
    elif mode == "letterbox":
        s = min(new_w / old_w, new_h / old_h)
        sx = sy = s
        tx, ty = (new_w - s * old_w) / 2.0, (new_h - s * old_h) / 2.0
    elif mode == "center_crop":
        s = max(new_w / old_w, new_h / old_h)
        sx = sy = s
        tx, ty = -(s * old_w - new_w) / 2.0, -(s * old_h - new_h) / 2.0
    else:
        raise ValueError(f"Unknown resize mode: {mode}")
    return np.array([[sx, 0.0, tx], [0.0, sy, ty], [0.0, 0.0, 1.0]], dtype=np.float64)


def resize_k(
    k: np.ndarray,
    old_size: Tuple[int, int],
    new_size: Tuple[int, int],
    mode: str = "letterbox",
) -> np.ndarray:
    return resize_transform(old_size, new_size, mode) @ k


def transform_points_between_sizes(
    points: np.ndarray,
    from_size: Tuple[int, int],
    to_size: Tuple[int, int],
    mode: str = "letterbox",
) -> np.ndarray:
    if from_size == to_size:
        return points.astype(np.float32, copy=True)
    to_from = resize_transform(to_size, from_size, mode)
    from_to = np.linalg.inv(to_from)
    points64 = np.asarray(points, dtype=np.float64)
    ones = np.ones((points64.shape[0], 1), dtype=np.float64)
    points_h = np.concatenate([points64[:, :2], ones], axis=1)
    transformed = (from_to @ points_h.T).T[:, :2]
    return transformed.astype(np.float32)


def normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        raise ValueError("Cannot normalize zero-length vector.")
    return v / n


def _maybe_float(x: float):
    try:
        if np.isfinite(x):
            return float(x)
    except Exception:
        pass
    return ""


def lookat_rt(
    camera_center: Iterable[float],
    target: Iterable[float],
    up: Iterable[float] = (0.0, 0.0, 1.0),
    flip_y: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    c = np.asarray(camera_center, dtype=np.float64).reshape(3)
    t = np.asarray(target, dtype=np.float64).reshape(3)
    up_vec = normalize(np.asarray(up, dtype=np.float64).reshape(3))

    z_cam = normalize(t - c)
    x_cam = np.cross(z_cam, up_vec)
    if np.linalg.norm(x_cam) < 1e-6:
        alt_up = np.array([0.0, 1.0, 0.0]) if abs(z_cam[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
        x_cam = np.cross(z_cam, alt_up)
    x_cam = normalize(x_cam)

    y_cam = normalize(np.cross(x_cam, z_cam))
    if flip_y:
        y_cam = -y_cam

    r_cw = np.stack([x_cam, y_cam, z_cam], axis=1)
    r_wc = r_cw.T
    t_wc = -r_wc @ c
    return r_wc, t_wc


def build_camera_maps(config: Dict[str, Any]) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, np.ndarray]]]:
    cam_cfg = config["camera_position"]
    target = np.asarray(cam_cfg["T"], dtype=np.float64)
    z = float(cam_cfg["z"])
    baseline = float(cam_cfg["baseline"])
    x_half = baseline / 2.0
    y_side = math.sqrt(max(float(cam_cfg["dist_left"]) ** 2 - x_half**2, 0.0))

    centers = {
        "front": np.array([0.0, float(cam_cfg["dist_front"]), z], dtype=np.float64),
        "left": np.array([-x_half, y_side, z], dtype=np.float64),
        "right": np.array([x_half, y_side, z], dtype=np.float64),
    }

    k_cfg = config["camera_K"]
    k_maps = {view: reshape_k(k_cfg[view]) for view in VIEW_NAMES}

    tri_cfg = config.get("triangulation", {})
    if bool(tri_cfg.get("resize_intrinsics", True)):
        old_size = _size_tuple(tri_cfg.get("intrinsics_image_size", [2304, 1296]))
        target_size_raw = tri_cfg.get("triangulation_image_size", tri_cfg.get("keypoint_image_size", [332, 224]))
        new_size_maps = view_size_maps(target_size_raw)
        resize_mode = str(tri_cfg.get("intrinsics_resize_mode", tri_cfg.get("resize_mode", "letterbox")))
        k_maps = {
            view: resize_k(k, old_size, new_size_maps[view], resize_mode)
            for view, k in k_maps.items()
        }

    rt_maps: Dict[str, Dict[str, np.ndarray]] = {}
    for view, center in centers.items():
        r_wc, t_wc = lookat_rt(center, target)
        rt_maps[view] = {"R": r_wc, "t": t_wc, "C": center}

    return k_maps, rt_maps


def build_projection(k: np.ndarray, rt: Dict[str, np.ndarray]) -> np.ndarray:
    return k @ np.hstack([rt["R"], rt["t"].reshape(3, 1)])


def project(p: np.ndarray, x_world: np.ndarray) -> np.ndarray:
    x_h = np.append(x_world, 1.0)
    uvw = p @ x_h
    if abs(float(uvw[2])) < 1e-12:
        return np.array([np.nan, np.nan], dtype=np.float64)
    return uvw[:2] / uvw[2]


def triangulate_point(ps: List[np.ndarray], xs: List[np.ndarray]) -> np.ndarray:
    a_rows = []
    for p, x in zip(ps, xs):
        a_rows.append(x[0] * p[2, :] - p[0, :])
        a_rows.append(x[1] * p[2, :] - p[1, :])
    _, _, vt = np.linalg.svd(np.asarray(a_rows, dtype=np.float64))
    x_h = vt[-1]
    if abs(float(x_h[3])) < 1e-12:
        return np.full(3, np.nan, dtype=np.float32)
    return (x_h[:3] / x_h[3]).astype(np.float32)


def load_sam3d_npz(path: Path) -> Tuple[np.ndarray, Optional[int]]:
    with np.load(path, allow_pickle=True) as obj:
        if "output" not in obj:
            raise KeyError(f"Missing 'output' in {path}")
        output = obj["output"].item()
    kpt2d = np.asarray(output["pred_keypoints_2d"], dtype=np.float32)
    if kpt2d.ndim != 2 or kpt2d.shape[1] < 2:
        raise ValueError(f"Invalid pred_keypoints_2d shape {kpt2d.shape} in {path}")
    frame_idx = output.get("frame_idx")
    return kpt2d[:, :2], int(frame_idx) if frame_idx is not None else None


def frame_id(path: Path) -> str:
    match = re.search(r"(\d+)_sam3d_body\.npz$", path.name)
    return match.group(1) if match else path.stem


def collect_frame_map(view_dir: Path) -> Dict[str, Path]:
    return {frame_id(path): path for path in sorted(view_dir.glob("*_sam3d_body.npz"))}


def valid_observation(
    pt: np.ndarray,
    image_size: Optional[Tuple[int, int]],
    margin: float,
    filter_bounds: bool,
) -> bool:
    if not np.all(np.isfinite(pt[:2])):
        return False
    if not filter_bounds or image_size is None:
        return True
    w, h = image_size
    x, y = float(pt[0]), float(pt[1])
    return -margin <= x <= w + margin and -margin <= y <= h + margin


def triangulate_frame(
    points_by_view: Dict[str, np.ndarray],
    projection_maps: Dict[str, np.ndarray],
    rt_maps: Dict[str, Dict[str, np.ndarray]],
    image_sizes: Optional[Dict[str, Tuple[int, int]]],
    margin_px: float,
    filter_bounds: bool,
    max_reproj_error_px: float,
    triangulation_strategy: str = "all_visible",
    min_views: int = 2,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    n_keypoints = min(points.shape[0] for points in points_by_view.values())
    keypoints_3d = np.full((n_keypoints, 3), np.nan, dtype=np.float32)
    reproj_error = np.full(n_keypoints, np.nan, dtype=np.float32)
    valid_mask = np.zeros(n_keypoints, dtype=bool)

    reproj_per_view = np.full((n_keypoints, len(VIEW_NAMES)), np.nan, dtype=np.float32)
    depths_per_view = np.full((n_keypoints, len(VIEW_NAMES)), np.nan, dtype=np.float32)
    cond_nums = np.full(n_keypoints, np.nan, dtype=np.float32)
    loo_mean_disp = np.full(n_keypoints, np.nan, dtype=np.float32)
    loo_max_disp = np.full(n_keypoints, np.nan, dtype=np.float32)
    depth_var = np.full(n_keypoints, np.nan, dtype=np.float32)

    for idx in range(n_keypoints):
        used_views: List[str] = []
        used_ps: List[np.ndarray] = []
        used_xs: List[np.ndarray] = []
        for view in VIEW_NAMES:
            pt = points_by_view[view][idx]
            image_size = image_sizes.get(view) if image_sizes else None
            if valid_observation(
                pt,
                image_size=image_size,
                margin=margin_px,
                filter_bounds=filter_bounds,
            ):
                used_views.append(view)
                used_ps.append(projection_maps[view])
                used_xs.append(pt.astype(np.float64))
        if len(used_views) < 2:
            continue

        candidates = []
        if triangulation_strategy == "best_subset":
            min_subset = max(2, int(min_views))
            for n_views in range(min_subset, len(used_views) + 1):
                for combo in itertools.combinations(range(len(used_views)), n_views):
                    combo_views = [used_views[i] for i in combo]
                    combo_ps = [used_ps[i] for i in combo]
                    combo_xs = [used_xs[i] for i in combo]
                    x_candidate = triangulate_point(combo_ps, combo_xs)
                    if not np.all(np.isfinite(x_candidate)):
                        continue
                    combo_depths = [
                        float((rt_maps[v]["R"] @ x_candidate + rt_maps[v]["t"])[2])
                        for v in combo_views
                    ]
                    if not all(depth > 0.0 for depth in combo_depths):
                        continue
                    combo_errs = [
                        np.linalg.norm(project(p, x_candidate) - x)
                        for p, x in zip(combo_ps, combo_xs)
                    ]
                    candidates.append((float(np.mean(combo_errs)), -n_views, combo_views, combo_ps, combo_xs, x_candidate))
        elif triangulation_strategy == "all_visible":
            x_candidate = triangulate_point(used_ps, used_xs)
            if np.all(np.isfinite(x_candidate)):
                depths = [float((rt_maps[v]["R"] @ x_candidate + rt_maps[v]["t"])[2]) for v in used_views]
                if all(depth > 0.0 for depth in depths):
                    errs = [np.linalg.norm(project(p, x_candidate) - x) for p, x in zip(used_ps, used_xs)]
                    candidates.append((float(np.mean(errs)), -len(used_views), used_views, used_ps, used_xs, x_candidate))
        else:
            raise ValueError(f"Unknown triangulation_strategy: {triangulation_strategy}")

        if not candidates:
            continue
        mean_err, _, used_views, used_ps, used_xs, x_world = min(candidates, key=lambda item: (item[0], item[1]))
        if mean_err > max_reproj_error_px:
            continue

        errs = [np.linalg.norm(project(p, x_world) - x) for p, x in zip(used_ps, used_xs)]
        depths = [float((rt_maps[v]["R"] @ x_world + rt_maps[v]["t"])[2]) for v in used_views]

        # fill per-view reproj errors and depths into arrays (match VIEW_NAMES order)
        for vi, view in enumerate(VIEW_NAMES):
            if view in used_views:
                i = used_views.index(view)
                reproj_per_view[idx, vi] = float(errs[i])
                depths_per_view[idx, vi] = float((rt_maps[view]["R"] @ x_world + rt_maps[view]["t"])[2])

        # condition number of linear system A (stability indicator)
        try:
            a_rows = []
            for p, x in zip(used_ps, used_xs):
                a_rows.append(x[0] * p[2, :] - p[0, :])
                a_rows.append(x[1] * p[2, :] - p[1, :])
            a_mat = np.asarray(a_rows, dtype=np.float64)
            s = np.linalg.svd(a_mat, compute_uv=False)
            cond = float(s[0] / (s[-1] + 1e-12))
        except Exception:
            cond = float('nan')
        cond_nums[idx] = cond

        # leave-one-out stability (requires at least 3 views to omit one)
        loo_disps = []
        if len(used_views) >= 3:
            for omit_i in range(len(used_views)):
                sub_ps = [p for j, p in enumerate(used_ps) if j != omit_i]
                sub_xs = [x for j, x in enumerate(used_xs) if j != omit_i]
                sub_x = triangulate_point(sub_ps, sub_xs)
                if np.all(np.isfinite(sub_x)):
                    loo_disps.append(float(np.linalg.norm(sub_x - x_world)))
        if loo_disps:
            loo_mean_disp[idx] = float(np.mean(loo_disps))
            loo_max_disp[idx] = float(np.max(loo_disps))

        depth_var[idx] = float(np.var(depths))

        keypoints_3d[idx] = x_world
        reproj_error[idx] = mean_err
        valid_mask[idx] = True

    return (
        keypoints_3d,
        valid_mask,
        reproj_error,
        reproj_per_view,
        depths_per_view,
        cond_nums,
        loo_mean_disp,
        loo_max_disp,
        depth_var,
    )


def process_sequence(
    input_root: Path,
    output_root: Path,
    person_id: str,
    env_name: str,
    k_maps: Dict[str, np.ndarray],
    rt_maps: Dict[str, Dict[str, np.ndarray]],
    config: Dict[str, Any],
    max_frames: Optional[int] = None,
    show_progress: bool = False,
    position: int = 0,
) -> Dict[str, Any]:
    tri_cfg = config.get("triangulation", {})
    keypoint_size_raw = tri_cfg.get("keypoint_image_size")
    target_size_raw = tri_cfg.get("triangulation_image_size", keypoint_size_raw)
    keypoint_sizes = view_size_maps(keypoint_size_raw) if keypoint_size_raw else None
    image_sizes = view_size_maps(target_size_raw) if target_size_raw else None
    keypoint_to_target_mode = str(tri_cfg.get("keypoint_to_triangulation_resize_mode", tri_cfg.get("resize_mode", "letterbox")))
    margin_px = float(tri_cfg.get("valid_margin_px", 8.0))
    filter_bounds = bool(tri_cfg.get("filter_keypoints_by_image_bounds", False))
    max_reproj_error_px = float(tri_cfg.get("max_reproj_error_px", 25.0))
    triangulation_strategy = str(tri_cfg.get("triangulation_strategy", "all_visible"))
    min_views = int(tri_cfg.get("min_views", 2))

    sequence_dir = input_root / person_id / env_name
    view_dirs = {view: sequence_dir / view for view in VIEW_NAMES}
    missing_dirs = [str(path) for path in view_dirs.values() if not path.exists()]
    if missing_dirs:
        raise FileNotFoundError(f"Missing view directories: {missing_dirs}")

    frame_maps = {view: collect_frame_map(path) for view, path in view_dirs.items()}
    common_ids = sorted(set.intersection(*(set(m.keys()) for m in frame_maps.values())))
    if not common_ids:
        raise RuntimeError(f"No common SAM3D frames found in {sequence_dir}")
    if max_frames is not None and max_frames > 0:
        common_ids = common_ids[: int(max_frames)]

    projection_maps = {view: build_projection(k_maps[view], rt_maps[view]) for view in VIEW_NAMES}
    out_dir = output_root / person_id / env_name
    frame_out_dir = out_dir / "frames"
    frame_out_dir.mkdir(parents=True, exist_ok=True)

    all_kpts: List[np.ndarray] = []
    all_masks: List[np.ndarray] = []
    all_errors: List[np.ndarray] = []
    all_reproj_per_view: List[np.ndarray] = []
    all_depths_per_view: List[np.ndarray] = []
    all_cond_nums: List[np.ndarray] = []
    all_loo_mean: List[np.ndarray] = []
    all_loo_max: List[np.ndarray] = []
    all_depth_var: List[np.ndarray] = []
    source_frames: List[str] = []

    frame_iter = tqdm(
        common_ids,
        desc=f"{person_id}/{env_name}",
        unit="frame",
        leave=False,
        position=position,
        disable=not show_progress,
    )
    for frame_name in frame_iter:
        sam3d_points_by_view = {
            view: load_sam3d_npz(frame_maps[view][frame_name])[0] for view in VIEW_NAMES
        }
        if keypoint_sizes and image_sizes:
            points_by_view = {
                view: transform_points_between_sizes(
                    sam3d_points_by_view[view],
                    keypoint_sizes[view],
                    image_sizes[view],
                    keypoint_to_target_mode,
                )
                for view in VIEW_NAMES
            }
        else:
            points_by_view = sam3d_points_by_view
        (
            keypoints_3d,
            valid_mask,
            reproj_error,
            reproj_per_view,
            depths_per_view,
            cond_nums,
            loo_mean_disp,
            loo_max_disp,
            depth_var,
        ) = triangulate_frame(
            points_by_view,
            projection_maps,
            rt_maps,
            image_sizes=image_sizes,
            margin_px=margin_px,
            filter_bounds=filter_bounds,
            max_reproj_error_px=max_reproj_error_px,
            triangulation_strategy=triangulation_strategy,
            min_views=min_views,
        )

        frame_file = frame_out_dir / f"{frame_name}_triangulated_kpt.npz"
        np.savez_compressed(
            frame_file,
            keypoints_3d=keypoints_3d,
            valid_mask=valid_mask,
            reproj_error=reproj_error,
            reproj_per_view=reproj_per_view,
            depths_per_view=depths_per_view,
            cond_nums=cond_nums,
            loo_mean_disp=loo_mean_disp,
            loo_max_disp=loo_max_disp,
            depth_var=depth_var,
            pred_keypoints_2d_front=points_by_view["front"],
            pred_keypoints_2d_left=points_by_view["left"],
            pred_keypoints_2d_right=points_by_view["right"],
            sam3d_pred_keypoints_2d_front=sam3d_points_by_view["front"],
            sam3d_pred_keypoints_2d_left=sam3d_points_by_view["left"],
            sam3d_pred_keypoints_2d_right=sam3d_points_by_view["right"],
        )
        all_kpts.append(keypoints_3d)
        all_masks.append(valid_mask)
        all_errors.append(reproj_error)
        all_reproj_per_view.append(reproj_per_view)
        all_depths_per_view.append(depths_per_view)
        all_cond_nums.append(cond_nums)
        all_loo_mean.append(loo_mean_disp)
        all_loo_max.append(loo_max_disp)
        all_depth_var.append(depth_var)
        source_frames.append(frame_name)

    keypoints_3d_seq = np.stack(all_kpts, axis=0)
    valid_mask_seq = np.stack(all_masks, axis=0)
    reproj_error_seq = np.stack(all_errors, axis=0)
    reproj_per_view_seq = np.stack(all_reproj_per_view, axis=0)
    depths_per_view_seq = np.stack(all_depths_per_view, axis=0)
    cond_nums_seq = np.stack(all_cond_nums, axis=0)
    loo_mean_seq = np.stack(all_loo_mean, axis=0)
    loo_max_seq = np.stack(all_loo_max, axis=0)
    depth_var_seq = np.stack(all_depth_var, axis=0)
    np.savez_compressed(
        out_dir / "keypoints_3d.npz",
        keypoints_3d=keypoints_3d_seq,
        valid_mask=valid_mask_seq,
        reproj_error=reproj_error_seq,
        frame_ids=np.asarray(source_frames),
        K=np.asarray([k_maps[v] for v in VIEW_NAMES]),
        R=np.asarray([rt_maps[v]["R"] for v in VIEW_NAMES]),
        t=np.asarray([rt_maps[v]["t"] for v in VIEW_NAMES]),
        views=np.asarray(VIEW_NAMES),
        reproj_per_view=reproj_per_view_seq,
        depths_per_view=depths_per_view_seq,
        cond_nums=cond_nums_seq,
        loo_mean_disp=loo_mean_seq,
        loo_max_disp=loo_max_seq,
        depth_var=depth_var_seq,
    )

    # write per-sequence CSV summarizing per-frame per-keypoint metrics
    import csv

    csv_file = out_dir / "errors.csv"
    with open(csv_file, "w", encoding="utf-8", newline="") as cf:
        writer = csv.writer(cf)
        header = [
            "frame_id",
            "keypoint_idx",
            "valid",
            "reproj_error",
            "reproj_front",
            "reproj_left",
            "reproj_right",
            "depth_front",
            "depth_left",
            "depth_right",
            "cond_num",
            "loo_mean_disp",
            "loo_max_disp",
            "depth_var",
        ]
        writer.writerow(header)
        for fi, frame_id in enumerate(source_frames):
            for kp in range(keypoints_3d_seq.shape[1]):
                writer.writerow([
                    frame_id,
                    kp,
                    int(bool(valid_mask_seq[fi, kp])),
                    float(reproj_error_seq[fi, kp]) if np.isfinite(reproj_error_seq[fi, kp]) else "",
                    _maybe_float(reproj_per_view_seq[fi, kp, 0]),
                    _maybe_float(reproj_per_view_seq[fi, kp, 1]),
                    _maybe_float(reproj_per_view_seq[fi, kp, 2]),
                    _maybe_float(depths_per_view_seq[fi, kp, 0]),
                    _maybe_float(depths_per_view_seq[fi, kp, 1]),
                    _maybe_float(depths_per_view_seq[fi, kp, 2]),
                    _maybe_float(cond_nums_seq[fi, kp]),
                    _maybe_float(loo_mean_seq[fi, kp]),
                    _maybe_float(loo_max_seq[fi, kp]),
                    _maybe_float(depth_var_seq[fi, kp]),
                ])
    valid_ratio = float(valid_mask_seq.mean()) if valid_mask_seq.size else 0.0
    mean_reproj = float(np.nanmean(reproj_error_seq)) if np.isfinite(reproj_error_seq).any() else float("nan")
    summary = {
        "person_id": person_id,
        "env_name": env_name,
        "frames": len(source_frames),
        "keypoints": int(keypoints_3d_seq.shape[1]),
        "valid_ratio": valid_ratio,
        "mean_reproj_error_px": mean_reproj,
        "output": str(out_dir / "keypoints_3d.npz"),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def iter_sequences(input_root: Path) -> Iterable[Tuple[str, str]]:
    for person_dir in sorted(path for path in input_root.iterdir() if path.is_dir()):
        for env_dir in sorted(path for path in person_dir.iterdir() if path.is_dir()):
            if all((env_dir / view).exists() for view in VIEW_NAMES):
                yield person_dir.name, env_dir.name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("triangulation.yaml"),
        help="Triangulation yaml config.",
    )
    parser.add_argument("--input-root", type=Path, default=None, help="SAM3D body result root.")
    parser.add_argument("--output-root", type=Path, default=None, help="Output root for triangulated 3D GT.")
    parser.add_argument("--person-id", type=str, default=None, help="Process one person id, e.g. 07.")
    parser.add_argument("--env-name", type=str, default=None, help="Process one env folder, e.g. 昼多い.")
    parser.add_argument("--debug-one", action="store_true", help="Process the first available sequence only.")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit frames per sequence for smoke tests.")
    parser.add_argument("--num-workers", type=int, default=None, help="Override processing.num_workers.")
    parser.add_argument(
        "--extrinsics-json",
        type=Path,
        default=None,
        help="Override camera extrinsics with the JSON written by calibrate_extrinsics.py "
        "(keys: extrinsics.<view>.{R,t,C}). Intrinsics still come from the config.",
    )
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars.")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s")
    args = parse_args()
    config = load_config(args.config)

    paths_cfg = config.get("paths", {})
    input_root = _as_path(args.input_root or paths_cfg.get("sam3d_kpt_path", paths_cfg.get("sam3d_results_path")))
    output_root = _as_path(args.output_root or paths_cfg.get("triangulated_kpt_path", paths_cfg.get("output_path")))
    output_root.mkdir(parents=True, exist_ok=True)

    k_maps, rt_maps = build_camera_maps(config)

    def load_extrinsics(path: Path) -> Dict[str, Dict[str, np.ndarray]]:
        with open(path, "r", encoding="utf-8") as f:
            ext = json.load(f)["extrinsics"]
        out: Dict[str, Dict[str, np.ndarray]] = {}
        for view in VIEW_NAMES:
            if view not in ext:
                raise KeyError(f"Extrinsics JSON {path} is missing view {view!r}")
            out[view] = {
                "R": np.asarray(ext[view]["R"], dtype=np.float64),
                "t": np.asarray(ext[view]["t"], dtype=np.float64),
                "C": np.asarray(ext[view]["C"], dtype=np.float64),
            }
        return out

    extrinsics_dir: Optional[Path] = None
    if args.extrinsics_json is not None:
        if args.extrinsics_json.is_dir():
            # per-sequence extrinsics: <dir>/<person>_<env>.json (calibrate_extrinsics.py --per-sequence)
            extrinsics_dir = args.extrinsics_json
            LOGGER.info("Using per-sequence extrinsics from %s", extrinsics_dir)
        else:
            rt_maps = load_extrinsics(args.extrinsics_json)
            LOGGER.info("Loaded optimized extrinsics from %s", args.extrinsics_json)

    if args.person_id or args.env_name:
        if not (args.person_id and args.env_name):
            raise ValueError("--person-id and --env-name must be used together.")
        seqs_to_process = [(args.person_id, args.env_name)]
    else:
        seqs_to_process = list(iter_sequences(input_root))
        if args.debug_one:
            seqs_to_process = seqs_to_process[:1]

    if not seqs_to_process:
        raise RuntimeError(f"No complete front/left/right SAM3D sequences found in {input_root}")

    processing_cfg = config.get("processing", {})
    num_workers = args.num_workers if args.num_workers is not None else processing_cfg.get("num_workers", 4)
    num_workers = max(1, int(num_workers))
    num_workers = min(num_workers, len(seqs_to_process))

    show_progress = bool(config.get("processing", {}).get("show_progress", True)) and not args.no_progress

    if show_progress:
        tqdm.write(f"Processing {len(seqs_to_process)} sequence(s) with {num_workers} worker(s)")
    else:
        LOGGER.info("Processing %d sequence(s) with %d worker(s)", len(seqs_to_process), num_workers)
    summaries = []

    def run_one(seq: Tuple[str, str], position: int = 1) -> Dict[str, Any]:
        person_id, env_name = seq
        if not show_progress:
            LOGGER.info("Processing %s/%s", person_id, env_name)
        seq_rt = rt_maps
        if extrinsics_dir is not None:
            seq_file = extrinsics_dir / f"{person_id}_{env_name}.json"
            if seq_file.exists():
                seq_rt = load_extrinsics(seq_file)
            else:
                LOGGER.warning("No per-sequence extrinsics for %s/%s; using constructed layout", person_id, env_name)
        summary = process_sequence(
            input_root,
            output_root,
            person_id,
            env_name,
            k_maps,
            seq_rt,
            config,
            max_frames=args.max_frames,
            show_progress=show_progress and num_workers == 1,
            position=position,
        )
        done_msg = (
            f"Done {person_id}/{env_name}: frames={summary['frames']} "
            f"valid={summary['valid_ratio']:.3f} "
            f"mean_rpe={summary['mean_reproj_error_px']:.3f}px"
        )
        if show_progress:
            tqdm.write(done_msg)
        else:
            LOGGER.info(done_msg)
        return summary

    if num_workers == 1:
        for seq in seqs_to_process:
            summaries.append(run_one(seq, position=0))
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as ex:
            future_to_seq = {ex.submit(run_one, seq, 1): seq for seq in seqs_to_process}
            completed = concurrent.futures.as_completed(future_to_seq)
            completed = tqdm(
                completed,
                total=len(future_to_seq),
                desc="Sequences",
                unit="seq",
                disable=not show_progress,
            )
            for fut in completed:
                person_id, env_name = future_to_seq[fut]
                try:
                    summaries.append(fut.result())
                except Exception:
                    LOGGER.exception("Failed processing %s/%s", person_id, env_name)

    summaries.sort(key=lambda item: (str(item.get("person_id", "")), str(item.get("env_name", ""))))

    with open(output_root / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)
    LOGGER.info("Saved global summary to %s", output_root / "summary.json")


if __name__ == "__main__":
    main()
