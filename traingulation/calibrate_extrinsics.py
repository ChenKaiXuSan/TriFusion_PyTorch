"""Optimize front/right camera extrinsics against SAM3D 2D observations.

Gauge: left camera fixed at its constructed pose; scale pinned by a soft
baseline-length constraint. Effect is measured on held-out frames with the
production best_subset triangulation (same config as GT generation).
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

TRI_DIR = Path("/work/1/SKIING/chenkaixu/code/TriFusion/.claude/worktrees/review-fixes/traingulation")
sys.path.insert(0, str(TRI_DIR))
from sam3d_kpt_triangulation import (  # noqa: E402
    VIEW_NAMES,
    build_camera_maps,
    build_projection,
    collect_frame_map,
    load_config,
    load_sam3d_npz,
    transform_points_between_sizes,
    triangulate_frame,
    valid_observation,
    view_size_maps,
)

CFG = load_config(TRI_DIR / "triangulation.yaml")
DATA_ROOT = Path("/work/1/SKIING/chenkaixu/data/drive/sam3d_body_results_right")
OUT_JSON = Path("/home/SKIING/chenkaixu/.claude/jobs/ac1f81c6/tmp/optimized_extrinsics.json")

TRI_CFG = CFG["triangulation"]
KEYPOINT_SIZES = view_size_maps(TRI_CFG["keypoint_image_size"])
IMAGE_SIZES = view_size_maps(TRI_CFG["triangulation_image_size"])
RESIZE_MODE = str(TRI_CFG.get("keypoint_to_triangulation_resize_mode", "letterbox"))
MARGIN = float(TRI_CFG.get("valid_margin_px", 8.0))
MAX_RPE = float(TRI_CFG.get("max_reproj_error_px", 40.0))
MIN_VIEWS = int(TRI_CFG.get("min_views", 2))
BASELINE = float(CFG["camera_position"]["baseline"])

K_MAPS, RT0 = build_camera_maps(CFG)
rng = np.random.default_rng(0)


def load_frame_points(seq_dir: Path, frame_maps, frame_name):
    pts = {}
    for view in VIEW_NAMES:
        raw, _ = load_sam3d_npz(frame_maps[view][frame_name])
        pts[view] = transform_points_between_sizes(
            raw, KEYPOINT_SIZES[view], IMAGE_SIZES[view], RESIZE_MODE
        )
    return pts


def sample_frames():
    """Interleave calib/holdout frames across a spread of sequences."""
    calib, holdout = [], []
    subjects = sorted(p.name for p in DATA_ROOT.iterdir() if p.is_dir())
    for subject in subjects[::3]:  # every 3rd subject for coverage
        for env_dir in sorted((DATA_ROOT / subject).iterdir()):
            if not env_dir.is_dir():
                continue
            frame_maps = {v: collect_frame_map(env_dir / v) for v in VIEW_NAMES}
            if any(not m for m in frame_maps.values()):
                continue
            common = sorted(set.intersection(*(set(m.keys()) for m in frame_maps.values())))
            step = max(1, len(common) // 40)
            picked = common[::step][:40]
            for i, fid in enumerate(picked):
                entry = (env_dir, frame_maps, fid)
                (calib if i % 2 == 0 else holdout).append(entry)
    return calib, holdout


def collect_observations(entries):
    """Return pts2d (N,3,2) float64 with NaN for unobserved, per keypoint."""
    rows = []
    for env_dir, frame_maps, fid in entries:
        try:
            pts = load_frame_points(env_dir, frame_maps, fid)
        except Exception:
            continue
        n_kpt = min(p.shape[0] for p in pts.values())
        arr = np.full((n_kpt, 3, 2), np.nan)
        for vi, view in enumerate(VIEW_NAMES):
            for k in range(n_kpt):
                pt = pts[view][k]
                if valid_observation(pt, IMAGE_SIZES[view], MARGIN, True):
                    arr[k, vi] = pt[:2]
        rows.append(arr)
    return np.concatenate(rows, axis=0)


def make_rt(params):
    rt = {"left": RT0["left"]}
    for i, view in enumerate(("front", "right")):
        w = params[i * 6 : i * 6 + 3]
        dt = params[i * 6 + 3 : i * 6 + 6]
        r0, t0 = RT0[view]["R"], RT0[view]["t"]
        r_new = Rotation.from_rotvec(w).as_matrix() @ r0
        t_new = t0 + dt
        c_new = -r_new.T @ t_new
        rt[view] = {"R": r_new, "t": t_new, "C": c_new}
    return rt


def batched_residuals(params, pts2d):
    rt = make_rt(params)
    p_mats = {v: build_projection(K_MAPS[v], rt[v]) for v in VIEW_NAMES}
    obs_mask = np.isfinite(pts2d[..., 0])
    n_obs_total = int(obs_mask.sum())
    res = np.zeros(n_obs_total * 2)
    offset_map = np.zeros(pts2d.shape[:2], dtype=np.int64) - 1
    flat_index = 0
    for k in range(pts2d.shape[0]):
        pass  # offsets assigned per-group below

    # group points by view-combination
    combos = {}
    for idx in range(pts2d.shape[0]):
        key = tuple(np.nonzero(obs_mask[idx])[0].tolist())
        if len(key) >= 2:
            combos.setdefault(key, []).append(idx)

    # observation ordering: row-major over (point, observed view)
    obs_offsets = np.cumsum(obs_mask.sum(axis=1))
    starts = np.concatenate([[0], obs_offsets[:-1]])

    for key, idx_list in combos.items():
        idx_arr = np.asarray(idx_list)
        views = [VIEW_NAMES[i] for i in key]
        p_stack = np.stack([p_mats[v] for v in views])  # (k,3,4)
        obs = pts2d[idx_arr][:, list(key), :]  # (n,k,2)
        n, k = obs.shape[0], obs.shape[1]
        a = np.empty((n, 2 * k, 4))
        for j in range(k):
            a[:, 2 * j] = obs[:, j, 0, None] * p_stack[j, 2] - p_stack[j, 0]
            a[:, 2 * j + 1] = obs[:, j, 1, None] * p_stack[j, 2] - p_stack[j, 1]
        _, _, vt = np.linalg.svd(a)
        xh = vt[:, -1, :]
        bad = np.abs(xh[:, 3]) < 1e-12
        xh = np.where(bad[:, None], np.array([0, 0, 1.0, 1.0]), xh)
        x3 = xh[:, :3] / xh[:, 3:4]
        for j, view in enumerate(views):
            uvw = np.einsum("rc,nc->nr", p_mats[view], np.hstack([x3, np.ones((n, 1))]))
            depth = (x3 @ rt[view]["R"].T + rt[view]["t"])[:, 2]
            ok = (np.abs(uvw[:, 2]) > 1e-9) & (depth > 0) & ~bad
            uv = uvw[:, :2] / np.where(np.abs(uvw[:, 2:]) > 1e-9, uvw[:, 2:], 1.0)
            r = uv - obs[:, j]
            r = np.where(ok[:, None], r, 50.0)
            local_j = [list(key).index(kk) for kk in key]  # identity; kept for clarity
            pos = starts[idx_arr] + j
            res[pos * 2] = r[:, 0]
            res[pos * 2 + 1] = r[:, 1]

    # gauge/regularization residuals
    c_l = RT0["left"]["C"]
    c_r = make_rt(params)["right"]["C"]
    bl_res = (np.linalg.norm(c_r - c_l) - BASELINE) * 2000.0  # 1 mm = 2 px
    prior = params * 20.0  # weak prior: 0.05 rad or 5 cm = 1 px
    return np.concatenate([res, [bl_res], prior])


def pipeline_metrics(entries, rt_maps, label):
    p_maps = {v: build_projection(K_MAPS[v], rt_maps[v]) for v in VIEW_NAMES}
    stats = {"n_valid": 0, "n_total": 0, "errs": [], "front_used": 0}
    for env_dir, frame_maps, fid in entries:
        try:
            pts = load_frame_points(env_dir, frame_maps, fid)
        except Exception:
            continue
        out = triangulate_frame(
            pts, p_maps, rt_maps, IMAGE_SIZES, MARGIN, True, MAX_RPE,
            triangulation_strategy="best_subset", min_views=MIN_VIEWS,
        )
        kpts3d, valid, rpe, rpe_view = out[0], out[1], out[2], out[3]
        stats["n_total"] += int(valid.shape[0])
        stats["n_valid"] += int(valid.sum())
        stats["errs"].extend(rpe[valid].tolist())
        stats["front_used"] += int(np.isfinite(rpe_view[valid, 0]).sum())
    errs = np.asarray(stats["errs"])
    print(
        f"[{label}] valid_ratio={stats['n_valid']/max(stats['n_total'],1):.3f} "
        f"mean_rpe={errs.mean():.2f}px median={np.median(errs):.2f}px "
        f"p95={np.percentile(errs,95):.2f}px "
        f"front_coverage={stats['front_used']/max(stats['n_valid'],1):.3f} "
        f"(valid={stats['n_valid']:,}/{stats['n_total']:,})"
    )


def main():
    calib_entries, holdout_entries = sample_frames()
    print(f"calib frames={len(calib_entries)} holdout frames={len(holdout_entries)}")
    pts2d = collect_observations(calib_entries)
    keep = np.isfinite(pts2d[..., 0]).sum(axis=1) >= 2
    pts2d = pts2d[keep]
    if pts2d.shape[0] > 8000:
        pts2d = pts2d[rng.choice(pts2d.shape[0], 8000, replace=False)]
    print(f"calibration points={pts2d.shape[0]}")

    x0 = np.zeros(12)
    print("round 1 (huber f_scale=5)...")
    r1 = least_squares(batched_residuals, x0, args=(pts2d,), loss="huber", f_scale=5.0, max_nfev=400)
    # outlier rejection against round-1 model, then refine
    res1 = batched_residuals(r1.x, pts2d)[:-13].reshape(-1, 2)
    obs_err = np.linalg.norm(res1, axis=1)
    obs_mask = np.isfinite(pts2d[..., 0])
    per_pt_err = np.full(pts2d.shape[0], 0.0)
    starts = np.concatenate([[0], np.cumsum(obs_mask.sum(axis=1))[:-1]])
    for i in range(pts2d.shape[0]):
        n_o = int(obs_mask[i].sum())
        per_pt_err[i] = obs_err[starts[i] : starts[i] + n_o].mean()
    keep2 = per_pt_err < 30.0
    pts2d_f = pts2d[keep2]
    print(f"round 2 on {pts2d_f.shape[0]} inlier points...")
    r2 = least_squares(batched_residuals, r1.x, args=(pts2d_f,), loss="huber", f_scale=3.0, max_nfev=400)

    rt_new = make_rt(r2.x)
    print("\n--- optimized geometry ---")
    for view in ("front", "right"):
        w = r2.x[(0 if view == "front" else 1) * 6 :][:3]
        c0, c1 = RT0[view]["C"], rt_new[view]["C"]
        print(
            f"{view}: rot_delta={np.degrees(np.linalg.norm(w)):.2f}deg "
            f"C {np.round(c0,3).tolist()} -> {np.round(c1,3).tolist()} "
            f"(moved {np.linalg.norm(c1-c0)*100:.1f} cm)"
        )
    print(f"baseline: {np.linalg.norm(rt_new['right']['C']-RT0['left']['C']):.4f} m (target {BASELINE})")

    print("\n--- held-out frames, production best_subset triangulation ---")
    pipeline_metrics(holdout_entries, RT0, "original extrinsics")
    pipeline_metrics(holdout_entries, rt_new, "optimized extrinsics")

    OUT_JSON.write_text(json.dumps({
        "params": r2.x.tolist(),
        "extrinsics": {
            v: {"R": rt_new[v]["R"].tolist(), "t": rt_new[v]["t"].tolist(), "C": np.asarray(rt_new[v]["C"]).tolist()}
            for v in VIEW_NAMES
        },
    }, indent=1))
    print(f"\nsaved optimized extrinsics to {OUT_JSON}")


if __name__ == "__main__":
    main()
