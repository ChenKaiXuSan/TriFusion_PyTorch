#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from map_config import KEEP_KEYPOINT_INDICES


ENV_NAMES = {
    "夜多い": "Night_High",
    "夜少ない": "Night_Low",
    "昼多い": "Day_High",
    "昼少ない": "Day_Low",
}
CAMERAS = ("front", "left", "right")
METHODS = ("mean", "median", "confidence")


def frame_id(path: Path) -> str:
    match = re.search(r"(\d+)_sam3d_body\.npz$", path.name)
    return str(int(match.group(1))) if match else path.stem


def normalize_frame_id(value: Any) -> str:
    text = str(value).strip()
    try:
        return str(int(text))
    except ValueError:
        return text


def safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    return number


def load_sam3d_frame(file_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    with np.load(file_path, allow_pickle=True) as data:
        if "output" not in data:
            raise KeyError(f"Missing 'output' in {file_path}")
        output = data["output"].item()
        keypoints_3d = np.asarray(output.get("pred_keypoints_3d"), dtype=np.float32)
        if keypoints_3d.ndim != 2 or keypoints_3d.shape[1] < 3:
            raise ValueError(f"Invalid pred_keypoints_3d shape {keypoints_3d.shape} in {file_path}")

        confidence = output.get("confidence")
        pred_2d = output.get("pred_keypoints_2d")
        if confidence is None and pred_2d is not None:
            pred_2d = np.asarray(pred_2d, dtype=np.float32)
            if pred_2d.ndim == 2 and pred_2d.shape[1] >= 3:
                confidence = pred_2d[:, 2]
        if confidence is None:
            confidence = np.ones((keypoints_3d.shape[0],), dtype=np.float32)

    confidence = np.asarray(confidence, dtype=np.float32).reshape(-1)
    if KEEP_KEYPOINT_INDICES is not None:
        keypoints_3d = keypoints_3d[KEEP_KEYPOINT_INDICES]
        confidence = confidence[KEEP_KEYPOINT_INDICES]

    n_joints = min(keypoints_3d.shape[0], confidence.shape[0])
    return keypoints_3d[:n_joints, :3].astype(np.float32), confidence[:n_joints]


def load_sam3d_sequence(view_dir: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    files = sorted(view_dir.glob("*_sam3d_body.npz"), key=frame_id)
    if not files:
        raise FileNotFoundError(f"No SAM3D frame npz found in {view_dir}")

    poses = []
    confs = []
    ids = []
    for file_path in files:
        pose, conf = load_sam3d_frame(file_path)
        poses.append(pose)
        confs.append(conf)
        ids.append(frame_id(file_path))

    n_joints = min(pose.shape[0] for pose in poses)
    return (
        np.stack([pose[:n_joints] for pose in poses], axis=0),
        np.stack([conf[:n_joints] for conf in confs], axis=0),
        ids,
    )


def list_sam3d_files(view_dir: Path) -> Dict[str, Path]:
    files = sorted(view_dir.glob("*_sam3d_body.npz"), key=frame_id)
    if not files:
        raise FileNotFoundError(f"No SAM3D frame npz found in {view_dir}")
    return {frame_id(file_path): file_path for file_path in files}


def load_selected_sam3d_frames(
    file_map: Dict[str, Path],
    selected_ids: List[str],
    num_workers: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    paths = [file_map[fid] for fid in selected_ids]
    if num_workers > 1 and len(paths) > 1:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            loaded = list(executor.map(load_sam3d_frame, paths))
    else:
        loaded = [load_sam3d_frame(path) for path in paths]
    poses = [item[0] for item in loaded]
    confs = [item[1] for item in loaded]

    n_joints = min(pose.shape[0] for pose in poses)
    return (
        np.stack([pose[:n_joints] for pose in poses], axis=0),
        np.stack([conf[:n_joints] for conf in confs], axis=0),
    )


def load_gt_sequence(gt_path: Path) -> Tuple[np.ndarray, np.ndarray, List[str] | None]:
    if not gt_path.exists():
        raise FileNotFoundError(f"GT file not found: {gt_path}")
    with np.load(gt_path, allow_pickle=False) as data:
        key = next((k for k in ("keypoints_3d", "KPT_3D", "coords_3d") if k in data.files), None)
        if key is None:
            raise KeyError(f"No 3D keypoint array found in {gt_path}. Keys={data.files}")
        keypoints = np.asarray(data[key], dtype=np.float32)
        if "valid_mask" in data.files:
            valid_mask = np.asarray(data["valid_mask"], dtype=bool)
        else:
            valid_mask = np.isfinite(keypoints).all(axis=-1)
        frame_ids = (
            [normalize_frame_id(x) for x in np.asarray(data["frame_ids"]).tolist()]
            if "frame_ids" in data.files
            else None
        )

    if KEEP_KEYPOINT_INDICES is not None and keypoints.shape[1] != len(KEEP_KEYPOINT_INDICES):
        keypoints = keypoints[:, KEEP_KEYPOINT_INDICES]
        valid_mask = valid_mask[:, KEEP_KEYPOINT_INDICES]
    return keypoints[:, :, :3].astype(np.float32), valid_mask.astype(bool), frame_ids


def align_sequences(
    view_data: Dict[str, Tuple[np.ndarray, np.ndarray, List[str]]],
    gt_pose: np.ndarray,
    gt_valid: np.ndarray,
    gt_frame_ids: List[str] | None,
    max_frames: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    if gt_frame_ids is not None:
        common = set(gt_frame_ids)
        for _, _, ids in view_data.values():
            common &= {normalize_frame_id(fid) for fid in ids}
        aligned_ids = sorted(common, key=lambda x: int(x) if x.isdigit() else x)
        if max_frames is not None and max_frames > 0:
            aligned_ids = aligned_ids[:max_frames]
        if not aligned_ids:
            raise ValueError("No common frame ids across all views and GT")

        gt_lookup = {normalize_frame_id(fid): idx for idx, fid in enumerate(gt_frame_ids)}
        view_lookup = {
            cam: {normalize_frame_id(fid): idx for idx, fid in enumerate(ids)}
            for cam, (_, _, ids) in view_data.items()
        }
        view_pose = []
        view_conf = []
        for cam in CAMERAS:
            pose, conf, _ = view_data[cam]
            indices = [view_lookup[cam][fid] for fid in aligned_ids]
            view_pose.append(pose[indices])
            view_conf.append(conf[indices])
        gt_indices = [gt_lookup[fid] for fid in aligned_ids]
        return (
            np.stack(view_pose, axis=2),
            np.stack(view_conf, axis=2),
            gt_pose[gt_indices],
            gt_valid[gt_indices],
            aligned_ids,
        )

    n_frames = min(gt_pose.shape[0], *(view_data[cam][0].shape[0] for cam in CAMERAS))
    if max_frames is not None and max_frames > 0:
        n_frames = min(n_frames, max_frames)
    view_pose = [view_data[cam][0][:n_frames] for cam in CAMERAS]
    view_conf = [view_data[cam][1][:n_frames] for cam in CAMERAS]
    ids = [str(i) for i in range(n_frames)]
    return (
        np.stack(view_pose, axis=2),
        np.stack(view_conf, axis=2),
        gt_pose[:n_frames],
        gt_valid[:n_frames],
        ids,
    )


def select_common_frame_ids(
    view_files: Dict[str, Dict[str, Path]],
    gt_frame_ids: List[str] | None,
    gt_num_frames: int,
    max_frames: int | None = None,
    sampling: str = "uniform",
) -> Tuple[List[str], List[int]]:
    def _sample_ids(ids: List[str]) -> List[str]:
        if max_frames is None or max_frames <= 0 or len(ids) <= max_frames:
            return ids
        if sampling == "head":
            return ids[:max_frames]
        indices = np.linspace(0, len(ids) - 1, num=max_frames, dtype=np.int64)
        return [ids[int(idx)] for idx in indices]

    if gt_frame_ids is not None:
        common = {normalize_frame_id(fid) for fid in gt_frame_ids}
        normalized_view_files = {
            cam: {normalize_frame_id(fid): path for fid, path in files.items()}
            for cam, files in view_files.items()
        }
        for files in normalized_view_files.values():
            common &= set(files.keys())
        selected_ids = sorted(common, key=lambda x: int(x) if x.isdigit() else x)
        selected_ids = _sample_ids(selected_ids)
        if not selected_ids:
            raise ValueError("No common frame ids across all views and GT")

        gt_lookup = {normalize_frame_id(fid): idx for idx, fid in enumerate(gt_frame_ids)}
        return selected_ids, [gt_lookup[fid] for fid in selected_ids]

    n_frames = min(gt_num_frames, *(len(files) for files in view_files.values()))
    ids = [str(i) for i in range(n_frames)]
    selected_ids = _sample_ids(ids)
    return selected_ids, [int(fid) for fid in selected_ids]


def normalize(x: np.ndarray, eps: float) -> np.ndarray:
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norm, eps)


def canonicalize_pose(
    pose: np.ndarray,
    neck_index: int = 51,
    left_shoulder_index: int = 5,
    right_shoulder_index: int = 6,
    mid_hip_index: int = -1,
    eps: float = 1e-6,
) -> np.ndarray:
    if not (0 <= neck_index < pose.shape[1]):
        raise ValueError(f"neck index {neck_index} out of range for J={pose.shape[1]}")
    if not (0 <= left_shoulder_index < pose.shape[1]):
        raise ValueError(f"left shoulder index {left_shoulder_index} out of range for J={pose.shape[1]}")
    if not (0 <= right_shoulder_index < pose.shape[1]):
        raise ValueError(f"right shoulder index {right_shoulder_index} out of range for J={pose.shape[1]}")

    neck = pose[:, neck_index : neck_index + 1]
    left = pose[:, left_shoulder_index]
    right = pose[:, right_shoulder_index]

    # Same axis convention as the model-side RobustCanonicalization: x = right - left.
    x_raw = right - left
    shoulder_mid = 0.5 * (left + right)
    shoulder_dist = np.linalg.norm(x_raw, axis=-1)
    shoulder_prior = 0.5  # average shoulder width in meters
    valid = np.isfinite(shoulder_dist) & (shoulder_dist > eps) & (shoulder_dist < 2 * shoulder_prior)
    if not valid.all():
        # Degenerate/outlier frames reuse the axis of the temporally nearest
        # valid frame (forward fill; the first valid frame for a leading
        # invalid prefix). A zero shoulder vector would otherwise collapse the
        # rotation matrix to zero and the canonicalized pose to the origin.
        if valid.any():
            idx = np.arange(pose.shape[0])
            ffill = np.maximum.accumulate(np.where(valid, idx, -1))
            fallback = np.where(ffill >= 0, ffill, int(np.argmax(valid)))
            x_raw = np.where(valid[:, None], x_raw, x_raw[fallback])
            shoulder_mid = np.where(valid[:, None], shoulder_mid, shoulder_mid[fallback])
        else:
            x_raw = np.tile(np.array([1.0, 0.0, 0.0], dtype=np.float64), (pose.shape[0], 1))
    x_axis = normalize(x_raw, eps)

    if 0 <= mid_hip_index < pose.shape[1]:
        down = pose[:, mid_hip_index] - neck[:, 0]
    else:
        down = shoulder_mid - neck[:, 0]
    down_axis = normalize(down, eps)

    # Guard the cross product against x_axis (near-)parallel to down_axis so the
    # rotation always has full rank.
    z_raw = np.cross(x_axis, down_axis)
    ref_z = np.cross(x_axis, np.array([0.0, 0.0, 1.0]))
    ref_y = np.cross(x_axis, np.array([0.0, 1.0, 0.0]))
    ref = np.where(np.linalg.norm(ref_z, axis=-1, keepdims=True) > eps, ref_z, ref_y)
    z_raw = np.where(np.linalg.norm(z_raw, axis=-1, keepdims=True) > eps, z_raw, ref)
    z_axis = normalize(z_raw, eps)
    y_axis = normalize(np.cross(z_axis, x_axis), eps)
    rot = np.stack([x_axis, y_axis, z_axis], axis=-1)
    return np.einsum("tjc,tcd->tjd", pose - neck, rot).astype(np.float32)


def fuse_views(view_pose: np.ndarray, view_conf: np.ndarray, method: str, eps: float = 1e-8) -> np.ndarray:
    if method == "mean":
        return np.nanmean(view_pose, axis=2).astype(np.float32)
    if method == "median":
        return np.nanmedian(view_pose, axis=2).astype(np.float32)
    if method == "confidence":
        weights = np.nan_to_num(view_conf, nan=0.0, posinf=0.0, neginf=0.0)
        weights = np.clip(weights, 0.0, None)
        denom = weights.sum(axis=2, keepdims=True)
        weights = np.where(denom > eps, weights / np.maximum(denom, eps), 1.0 / view_pose.shape[2])
        return np.sum(view_pose * weights[..., None], axis=2).astype(np.float32)
    raise ValueError(f"Unsupported method: {method}")


def procrustes_align(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Least-squares similarity (Umeyama) alignment of source onto target.

    Row-vector convention: aligned = s * (source_c @ R) + target_mean with
    R = U diag(1,1,d) Vt from SVD(source_c.T @ target_c) and the LS-optimal
    scale s = sum(d-corrected singular values) / ||source_c||^2. The previous
    implementation applied the transposed rotation and a norm-ratio scale,
    which systematically over-estimates PA-MPJPE (it can even exceed MPJPE).
    Degenerate inputs fall back to translation-only alignment.
    """
    source_mean = np.mean(source, axis=0, keepdims=True)
    target_mean = np.mean(target, axis=0, keepdims=True)
    source_centered = source - source_mean
    target_centered = target - target_mean
    source_var = float(np.sum(source_centered**2))
    target_norm = float(np.linalg.norm(target_centered))
    if source_var < 1e-12 or target_norm < 1e-8:
        return source_centered + target_mean

    h = source_centered.T @ target_centered
    u, s, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(u @ vt))
    signs = np.ones(s.shape[0])
    signs[-1] = d if d != 0 else 1.0
    rotation = (u * signs) @ vt
    scale = float(np.sum(s * signs)) / source_var
    return scale * (source_centered @ rotation) + target_mean


def compute_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    valid_mask: np.ndarray,
    root_index: int = 0,
    pck_thresholds: Tuple[float, ...] = (0.02, 0.05, 0.10, 0.15),
) -> Dict[str, Any]:
    n_frames = min(pred.shape[0], gt.shape[0], valid_mask.shape[0])
    n_joints = min(pred.shape[1], gt.shape[1], valid_mask.shape[1])
    pred = pred[:n_frames, :n_joints, :3]
    gt = gt[:n_frames, :n_joints, :3]
    valid = valid_mask[:n_frames, :n_joints] & np.isfinite(pred).all(axis=-1) & np.isfinite(gt).all(axis=-1)
    if not np.any(valid):
        return {}

    dist = np.linalg.norm(pred - gt, axis=-1)
    valid_dist = dist[valid]

    root_index = min(max(root_index, 0), n_joints - 1)
    root_valid = valid & valid[:, root_index : root_index + 1]
    root_dist = np.linalg.norm(
        (pred - pred[:, root_index : root_index + 1]) - (gt - gt[:, root_index : root_index + 1]),
        axis=-1,
    )
    root_values = root_dist[root_valid]

    pa_values = []
    pa_frame_raw_values = []  # unaligned errors on exactly the PA point set (same-mask protocol)
    for frame_idx in range(n_frames):
        frame_valid = valid[frame_idx]
        if int(frame_valid.sum()) < 3:
            continue
        aligned = procrustes_align(pred[frame_idx, frame_valid], gt[frame_idx, frame_valid])
        pa_values.extend(np.linalg.norm(aligned - gt[frame_idx, frame_valid], axis=-1).tolist())
        pa_frame_raw_values.extend(dist[frame_idx, frame_valid].tolist())

    pck = {
        f"{threshold:.2f}": float(np.mean(valid_dist <= threshold))
        for threshold in pck_thresholds
    }
    auc_thresholds = np.linspace(0.0, 0.15, 31)
    auc = float(np.mean([np.mean(valid_dist <= threshold) for threshold in auc_thresholds]))
    per_axis_mae = np.mean(np.abs((pred - gt)[valid]), axis=0)
    per_joint_mpjpe = []
    for joint_idx in range(n_joints):
        joint_valid = valid[:, joint_idx]
        per_joint_mpjpe.append(float(np.mean(dist[:, joint_idx][joint_valid])) if np.any(joint_valid) else None)

    return {
        "num_frames": int(n_frames),
        "num_keypoints": int(n_joints),
        "num_valid_points": int(valid.sum()),
        "mpjpe_m": float(np.mean(valid_dist)),
        "median_error_m": float(np.median(valid_dist)),
        "root_mpjpe_m": float(np.mean(root_values)) if root_values.size else None,
        "pa_mpjpe_m": float(np.mean(pa_values)) if pa_values else None,
        "mpjpe_pa_frames_m": float(np.mean(pa_frame_raw_values)) if pa_frame_raw_values else None,
        "pck": pck,
        "auc_0.15": auc,
        "per_axis_mae_m": {
            "x": float(per_axis_mae[0]),
            "y": float(per_axis_mae[1]),
            "z": float(per_axis_mae[2]),
        },
        "per_joint_mpjpe_m": per_joint_mpjpe,
    }


def evaluate_subject_env(
    subject_id: str,
    env_folder: str,
    sam3d_root: Path,
    gt_root: Path,
    args: argparse.Namespace,
) -> Dict[str, Dict[str, Any]]:
    view_files = {}
    for cam in CAMERAS:
        view_dir = sam3d_root / subject_id / env_folder / cam
        if not view_dir.exists():
            raise FileNotFoundError(f"SAM3D directory not found: {view_dir}")
        view_files[cam] = list_sam3d_files(view_dir)

    gt_pose, gt_valid, gt_frame_ids = load_gt_sequence(gt_root / subject_id / env_folder / "keypoints_3d.npz")
    frame_ids, gt_indices = select_common_frame_ids(
        view_files=view_files,
        gt_frame_ids=gt_frame_ids,
        gt_num_frames=gt_pose.shape[0],
        max_frames=args.max_frames,
        sampling=args.sampling,
    )
    normalized_view_files = {
        cam: {normalize_frame_id(fid): path for fid, path in files.items()}
        for cam, files in view_files.items()
    }
    loaded_views = {
        cam: load_selected_sam3d_frames(
            normalized_view_files[cam],
            frame_ids,
            num_workers=args.num_workers,
        )
        for cam in CAMERAS
    }
    view_pose = np.stack([loaded_views[cam][0] for cam in CAMERAS], axis=2)
    view_conf = np.stack([loaded_views[cam][1] for cam in CAMERAS], axis=2)
    gt_pose = gt_pose[gt_indices]
    gt_valid = gt_valid[gt_indices]

    # 注意：load_sam3d_frame / load_gt_sequence 已把 70 关节原始布局按
    # KEEP_KEYPOINT_INDICES 映射到 52 关节模型空间，此处数组即 52 关节，
    # canonicalize 的锚点 (neck=51, shoulders=5/6) 因此是正确的。
    # 不要在这里再做 KEEP 切片（会二次映射）。
    n_joints = min(view_pose.shape[1], gt_pose.shape[1])
    view_pose = view_pose[:, :n_joints]
    view_conf = view_conf[:, :n_joints]
    gt_pose = gt_pose[:, :n_joints]
    gt_valid = gt_valid[:, :n_joints]

    if args.canonicalize:
        gt_pose = canonicalize_pose(
            gt_pose,
            neck_index=args.neck_index,
            left_shoulder_index=args.left_shoulder_index,
            right_shoulder_index=args.right_shoulder_index,
            mid_hip_index=args.mid_hip_index,
            eps=args.eps,
        )
        canonical_views = []
        for view_idx in range(view_pose.shape[2]):
            canonical_views.append(
                canonicalize_pose(
                    view_pose[:, :, view_idx],
                    neck_index=args.neck_index,
                    left_shoulder_index=args.left_shoulder_index,
                    right_shoulder_index=args.right_shoulder_index,
                    mid_hip_index=args.mid_hip_index,
                    eps=args.eps,
                )
            )
        view_pose = np.stack(canonical_views, axis=2)

    results = {}
    env_name = ENV_NAMES.get(env_folder, env_folder)
    for method in METHODS:
        pred = fuse_views(view_pose, view_conf, method)
        metrics = compute_metrics(
            pred=pred,
            gt=gt_pose,
            valid_mask=gt_valid,
            root_index=args.root_index,
            pck_thresholds=tuple(args.pck_thresholds),
        )
        if not metrics:
            continue
        results[method] = {
            "method": method,
            "person_id": subject_id,
            "environment": env_folder,
            "environment_name": env_name,
            "total_frames": metrics["num_frames"],
            "canonicalize": bool(args.canonicalize),
            "mean_view_confidence": {
                cam: float(np.mean(loaded_views[cam][1])) for cam in CAMERAS
            },
            "metrics": metrics,
            "cameras": {
                "fusion": {
                    "num_frames": metrics["num_frames"],
                    "num_keypoints": metrics["num_keypoints"],
                    "metrics": metrics,
                    "frame_ids": frame_ids,
                }
            },
        }
    return results


def mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return float(sum(values) / len(values)) if values else None


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    buckets: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    env_buckets: Dict[str, Dict[str, Dict[str, List[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for row in rows:
        method = row["method"]
        env = row["environment_name"]
        for metric in ("mpjpe_m", "median_error_m", "root_mpjpe_m", "pa_mpjpe_m", "mpjpe_pa_frames_m", "auc_0.15", "pck_0.05", "pck_0.10", "pck_0.15"):
            value = safe_float(row.get(metric))
            if value is None:
                continue
            buckets[method][metric].append(value)
            env_buckets[method][env][metric].append(value)

    return {
        "overall": {
            method: {
                "num_pairs": len([r for r in rows if r["method"] == method]),
                **{metric: mean(values) for metric, values in metrics.items()},
            }
            for method, metrics in sorted(buckets.items())
        },
        "by_environment": {
            method: {
                env: {metric: mean(values) for metric, values in metrics.items()}
                for env, metrics in sorted(envs.items())
            }
            for method, envs in sorted(env_buckets.items())
        },
    }


def save_outputs(output_dir: Path, payloads: List[Dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for payload in payloads:
        method = payload["method"]
        person_id = payload["person_id"]
        env_name = payload["environment_name"]
        method_dir = output_dir / method / person_id / env_name
        method_dir.mkdir(parents=True, exist_ok=True)
        with open(method_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        metrics = payload["metrics"]
        pck = metrics.get("pck", {})
        rows.append(
            {
                "method": method,
                "person_id": person_id,
                "environment": payload["environment"],
                "environment_name": env_name,
                "num_frames": metrics.get("num_frames"),
                "num_keypoints": metrics.get("num_keypoints"),
                "num_valid_points": metrics.get("num_valid_points"),
                "mpjpe_m": metrics.get("mpjpe_m"),
                "median_error_m": metrics.get("median_error_m"),
                "root_mpjpe_m": metrics.get("root_mpjpe_m"),
                "pa_mpjpe_m": metrics.get("pa_mpjpe_m"),
                "pck_0.02": pck.get("0.02"),
                "pck_0.05": pck.get("0.05"),
                "pck_0.10": pck.get("0.10"),
                "pck_0.15": pck.get("0.15"),
                "auc_0.15": metrics.get("auc_0.15"),
                "source_file": str(method_dir / "metrics.json"),
            }
        )

    csv_path = output_dir / "fusion_baseline_detailed.csv"
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "num_rows": len(rows),
        "methods": list(METHODS),
        "summary": summarize(rows),
        "output_files": {
            "detailed_csv": str(csv_path),
            "summary_json": str(output_dir / "fusion_baseline_summary.json"),
        },
    }
    with open(output_dir / "fusion_baseline_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate fixed tri-view fusion baselines against triangulated pseudo GT."
    )
    parser.add_argument("--sam3d-root", type=str, default="/home/data/xchen/drive/sam3d_body_results_right")
    parser.add_argument("--gt-root", type=str, default="/home/data/xchen/drive/sam3d_body_triangulated_gt")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/home/workspace/kaixu/code/MultiView_DriverAction_PyTorch/TriPoseFusion/eval/logs/comparison_fusion_baselines_pesudo_gt",
    )
    parser.add_argument("--subject", type=str, default=None, help="Optional subject id, e.g. 01")
    parser.add_argument("--env", type=str, default=None, help="Optional raw env folder name")
    parser.add_argument("--root-index", type=int, default=0)
    parser.add_argument("--neck-index", type=int, default=51)
    parser.add_argument("--left-shoulder-index", type=int, default=5)
    parser.add_argument("--right-shoulder-index", type=int, default=6)
    parser.add_argument("--mid-hip-index", type=int, default=-1)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--pck-thresholds", type=float, nargs="+", default=[0.02, 0.05, 0.10, 0.15])
    parser.add_argument("--max-frames", type=int, default=None, help="Optional debug limit after frame alignment.")
    parser.add_argument("--num-workers", type=int, default=4, help="Thread workers for selected SAM3D frame loading.")
    parser.add_argument(
        "--sampling",
        choices=("uniform", "head"),
        default="uniform",
        help="Frame selection strategy used only when --max-frames is set.",
    )
    parser.add_argument("--no-canonicalize", dest="canonicalize", action="store_false")
    parser.set_defaults(canonicalize=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sam3d_root = Path(args.sam3d_root)
    gt_root = Path(args.gt_root)
    output_dir = Path(args.output_dir)
    if not sam3d_root.exists():
        raise FileNotFoundError(f"SAM3D root does not exist: {sam3d_root}")
    if not gt_root.exists():
        raise FileNotFoundError(f"GT root does not exist: {gt_root}")

    subjects = sorted(path.name for path in sam3d_root.iterdir() if path.is_dir())
    if args.subject:
        subjects = [subject for subject in subjects if subject == args.subject]
    if not subjects:
        raise RuntimeError("No subjects selected")

    envs = [args.env] if args.env else list(ENV_NAMES.keys())
    payloads = []
    skipped = []
    for subject_id in subjects:
        for env_folder in envs:
            print(f"Processing {subject_id}/{ENV_NAMES.get(env_folder, env_folder)}...", flush=True)
            try:
                results = evaluate_subject_env(subject_id, env_folder, sam3d_root, gt_root, args)
            except Exception as exc:
                skipped.append({"person_id": subject_id, "environment": env_folder, "error": str(exc)})
                print(f"Skip {subject_id}/{env_folder}: {exc}")
                continue
            payloads.extend(results[method] for method in METHODS if method in results)
            print(f"Processed {subject_id}/{ENV_NAMES.get(env_folder, env_folder)}")

    save_outputs(output_dir, payloads)
    if skipped:
        with open(output_dir / "skipped.json", "w", encoding="utf-8") as f:
            json.dump(skipped, f, indent=2, ensure_ascii=False)

    print(f"Saved fusion baseline metrics to: {output_dir}")
    print(f"Evaluated rows: {len(payloads)}; skipped subject/env pairs: {len(skipped)}")


if __name__ == "__main__":
    main()
