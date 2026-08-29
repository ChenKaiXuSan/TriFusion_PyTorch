"""Optimize front/right camera extrinsics against SAM3D 2D observations.

Gauge: left camera fixed at its constructed pose; scale pinned by a soft
baseline-length constraint (physical prior). Two modes:

  global        one set of extrinsics fitted on frames spread over every 3rd
                subject (original experiment; writes optimized_extrinsics.json)
  --per-sequence
                one set per (person, env) sequence, fitted on that sequence's
                own frames; writes <out-dir>/<person>_<env>.json plus a
                per-sequence sanity report (held-out reprojection error before
                and after, and the triangulated shoulder width, which should
                be ~0.37 m for an adult; the constructed extrinsics gave
                0.46-2.10 m because the rig moved between sessions).

Effect is always measured on held-out frames with the production best_subset
triangulation (same config as GT generation).
"""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

TRI_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TRI_DIR))
sys.path.insert(0, str(TRI_DIR.parent / "TriPoseFusion"))
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
from map_config import KEEP_KEYPOINT_INDICES  # noqa: E402

CFG = load_config(TRI_DIR / "triangulation.yaml")
DATA_ROOT = Path("/work/1/SKIING/chenkaixu/data/drive/sam3d_body_results_right")
OUT_JSON = TRI_DIR / "optimized_extrinsics.json"

TRI_CFG = CFG["triangulation"]
KEYPOINT_SIZES = view_size_maps(TRI_CFG["keypoint_image_size"])
IMAGE_SIZES = view_size_maps(TRI_CFG["triangulation_image_size"])
RESIZE_MODE = str(TRI_CFG.get("keypoint_to_triangulation_resize_mode", "letterbox"))
MARGIN = float(TRI_CFG.get("valid_margin_px", 8.0))
MAX_RPE = float(TRI_CFG.get("max_reproj_error_px", 40.0))
MIN_VIEWS = int(TRI_CFG.get("min_views", 2))
BASELINE = float(CFG["camera_position"]["baseline"])
L_SHO, R_SHO = KEEP_KEYPOINT_INDICES[5], KEEP_KEYPOINT_INDICES[6]  # 70-joint space

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


def sequence_entries(env_dir: Path, n_frames: int):
    """Evenly spaced frames of one sequence, interleaved into calib / holdout."""
    frame_maps = {v: collect_frame_map(env_dir / v) for v in VIEW_NAMES}
    if any(not m for m in frame_maps.values()):
        return [], []
    common = sorted(set.intersection(*(set(m.keys()) for m in frame_maps.values())))
    step = max(1, len(common) // n_frames)
    picked = common[::step][:n_frames]
    calib, holdout = [], []
    for i, fid in enumerate(picked):
        (calib if i % 2 == 0 else holdout).append((env_dir, frame_maps, fid))
    return calib, holdout


def sample_frames():
    """Global mode: interleave calib/holdout frames across every 3rd subject."""
    calib, holdout = [], []
    subjects = sorted(p.name for p in DATA_ROOT.iterdir() if p.is_dir())
    for subject in subjects[::3]:
        for env_dir in sorted((DATA_ROOT / subject).iterdir()):
            if env_dir.is_dir():
                c, h = sequence_entries(env_dir, 40)
                calib += c
                holdout += h
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
    return np.concatenate(rows, axis=0) if rows else np.zeros((0, 3, 2))


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
    res = np.zeros(int(obs_mask.sum()) * 2)

    combos = {}
    for idx in range(pts2d.shape[0]):
        key = tuple(np.nonzero(obs_mask[idx])[0].tolist())
        if len(key) >= 2:
            combos.setdefault(key, []).append(idx)
    starts = np.concatenate([[0], np.cumsum(obs_mask.sum(axis=1))[:-1]])

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
            pos = starts[idx_arr] + j
            res[pos * 2] = r[:, 0]
            res[pos * 2 + 1] = r[:, 1]

    c_l = RT0["left"]["C"]
    c_r = rt["right"]["C"]
    bl_res = (np.linalg.norm(c_r - c_l) - BASELINE) * 2000.0  # 1 mm = 2 px
    prior = params * 20.0  # weak prior: 0.05 rad or 5 cm = 1 px
    return np.concatenate([res, [bl_res], prior])


def calibrate_points(pts2d: np.ndarray, max_points: int = 8000):
    """Two-round robust least squares; returns 12 params."""
    keep = np.isfinite(pts2d[..., 0]).sum(axis=1) >= 2
    pts2d = pts2d[keep]
    if pts2d.shape[0] > max_points:
        pts2d = pts2d[rng.choice(pts2d.shape[0], max_points, replace=False)]
    r1 = least_squares(batched_residuals, np.zeros(12), args=(pts2d,), loss="huber", f_scale=5.0, max_nfev=400)
    res1 = batched_residuals(r1.x, pts2d)[:-13].reshape(-1, 2)
    obs_err = np.linalg.norm(res1, axis=1)
    obs_mask = np.isfinite(pts2d[..., 0])
    starts = np.concatenate([[0], np.cumsum(obs_mask.sum(axis=1))[:-1]])
    per_pt_err = np.array([obs_err[starts[i] : starts[i] + int(obs_mask[i].sum())].mean() for i in range(pts2d.shape[0])])
    pts2d_f = pts2d[per_pt_err < 30.0]
    r2 = least_squares(batched_residuals, r1.x, args=(pts2d_f,), loss="huber", f_scale=3.0, max_nfev=400)
    return r2.x, int(pts2d.shape[0]), int(pts2d_f.shape[0])


def evaluate_rt(entries, rt_maps) -> dict:
    """Production best_subset triangulation on entries; reprojection stats + shoulder width."""
    p_maps = {v: build_projection(K_MAPS[v], rt_maps[v]) for v in VIEW_NAMES}
    n_valid = n_total = front_used = 0
    errs, widths = [], []
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
        n_total += int(valid.shape[0])
        n_valid += int(valid.sum())
        errs.extend(rpe[valid].tolist())
        front_used += int(np.isfinite(rpe_view[valid, 0]).sum())
        if valid[L_SHO] and valid[R_SHO]:
            widths.append(float(np.linalg.norm(kpts3d[L_SHO] - kpts3d[R_SHO])))
    errs = np.asarray(errs)
    return {
        "valid_ratio": n_valid / max(n_total, 1),
        "mean_rpe": float(errs.mean()) if errs.size else float("nan"),
        "median_rpe": float(np.median(errs)) if errs.size else float("nan"),
        "p95_rpe": float(np.percentile(errs, 95)) if errs.size else float("nan"),
        "front_coverage": front_used / max(n_valid, 1),
        "shoulder_width_m": float(np.median(widths)) if widths else float("nan"),
        "n_valid": n_valid,
    }


def rt_to_json(rt):
    return {v: {"R": rt[v]["R"].tolist(), "t": rt[v]["t"].tolist(), "C": np.asarray(rt[v]["C"]).tolist()} for v in VIEW_NAMES}


def geometry_report(params):
    rt_new = make_rt(params)
    rep = {}
    for view in ("front", "right"):
        w = params[(0 if view == "front" else 1) * 6 :][:3]
        rep[view] = {
            "rot_delta_deg": float(np.degrees(np.linalg.norm(w))),
            "moved_cm": float(np.linalg.norm(rt_new[view]["C"] - RT0[view]["C"]) * 100),
        }
    rep["baseline_m"] = float(np.linalg.norm(rt_new["right"]["C"] - RT0["left"]["C"]))
    return rep


# ---------------------------------------------------------------- per-sequence


def run_sequence(task):
    env_dir, out_path, n_frames = task
    person, env = env_dir.parent.name, env_dir.name
    calib, holdout = sequence_entries(env_dir, n_frames)
    if not calib:
        return f"{person}/{env}: no frames"
    pts2d = collect_observations(calib)
    params, n_pts, n_inl = calibrate_points(pts2d)
    rt_new = make_rt(params)
    before = evaluate_rt(holdout, RT0)
    after = evaluate_rt(holdout, rt_new)
    payload = {
        "person": person, "env": env, "params": params.tolist(), "n_points": n_pts, "n_inliers": n_inl,
        "geometry": geometry_report(params), "holdout_before": before, "holdout_after": after,
        "extrinsics": rt_to_json(rt_new),
    }
    out_path.write_text(json.dumps(payload, indent=1, ensure_ascii=False))
    return (f"{person}/{env}: rpe {before['mean_rpe']:.2f}->{after['mean_rpe']:.2f}px  "
            f"shoulder {before['shoulder_width_m']:.3f}->{after['shoulder_width_m']:.3f}m  "
            f"front_cov {before['front_coverage']:.2f}->{after['front_coverage']:.2f}")


def main_per_sequence(args):
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tasks = []
    for subject in sorted(p for p in DATA_ROOT.iterdir() if p.is_dir()):
        for env_dir in sorted(p for p in subject.iterdir() if p.is_dir()):
            out_path = args.out_dir / f"{subject.name}_{env_dir.name}.json"
            if out_path.exists() and not args.overwrite:
                continue
            tasks.append((env_dir, out_path, args.frames))
    print(f"{len(tasks)} sequences to calibrate with {args.workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for msg in ex.map(run_sequence, tasks):
            print(msg, flush=True)
    # summary
    rows = [json.loads(p.read_text()) for p in sorted(args.out_dir.glob("*.json"))]
    sw_b = np.array([r["holdout_before"]["shoulder_width_m"] for r in rows])
    sw_a = np.array([r["holdout_after"]["shoulder_width_m"] for r in rows])
    rpe_b = np.array([r["holdout_before"]["mean_rpe"] for r in rows])
    rpe_a = np.array([r["holdout_after"]["mean_rpe"] for r in rows])
    print(f"\n{len(rows)} sequences | shoulder width: before median {np.nanmedian(sw_b):.3f} (std {np.nanstd(sw_b):.3f}) "
          f"-> after median {np.nanmedian(sw_a):.3f} (std {np.nanstd(sw_a):.3f}) | "
          f"mean rpe: {np.nanmean(rpe_b):.2f} -> {np.nanmean(rpe_a):.2f} px")


def sam3d_shoulder_width(entries) -> float:
    """Median SAM3D metric shoulder width (pred_keypoints_3d) over frames and views."""
    widths = []
    for _env_dir, frame_maps, fid in entries:
        for view in VIEW_NAMES:
            with np.load(frame_maps[view][fid], allow_pickle=True) as obj:
                k3 = np.asarray(obj["output"].item()["pred_keypoints_3d"], dtype=np.float64)
            if k3.shape[0] > max(L_SHO, R_SHO):
                widths.append(float(np.linalg.norm(k3[L_SHO] - k3[R_SHO])))
    return float(np.median(widths)) if widths else float("nan")


def scale_rt(rt, s: float):
    """Scale the scene about the (fixed) left camera centre; reprojections are invariant."""
    c_l = np.asarray(rt["left"]["C"], dtype=np.float64)
    out = {"left": rt["left"]}
    for view in ("front", "right"):
        r = np.asarray(rt[view]["R"], dtype=np.float64)
        c = c_l + s * (np.asarray(rt[view]["C"], dtype=np.float64) - c_l)
        out[view] = {"R": r, "t": -r @ c, "C": c}
    return out


def anchor_sequence(task):
    """Post-process one per-sequence JSON: set the metric scale (a gauge freedom of
    uncalibrated multi-view geometry) so the triangulated shoulder width matches the
    monocular estimator's body scale for that sequence."""
    json_path, n_frames = task
    payload = json.loads(json_path.read_text())
    env_dir = DATA_ROOT / payload["person"] / payload["env"]
    calib, holdout = sequence_entries(env_dir, n_frames)
    entries = calib + holdout
    rt = {v: {k: np.asarray(x, dtype=np.float64) for k, x in payload["extrinsics"][v].items()} for v in VIEW_NAMES}
    w_sam = sam3d_shoulder_width(entries)
    w_tri = evaluate_rt(entries, rt)["shoulder_width_m"]
    if not (np.isfinite(w_sam) and np.isfinite(w_tri) and w_tri > 0):
        return f"{payload['person']}/{payload['env']}: cannot anchor (sam {w_sam}, tri {w_tri})"
    s = w_sam / w_tri
    rt_s = scale_rt(rt, s)
    after = evaluate_rt(holdout, rt_s)
    payload["scale_anchor"] = {"sam3d_shoulder_m": w_sam, "triangulated_shoulder_m_before": w_tri, "factor": s}
    payload["holdout_after_unanchored"] = payload.get("holdout_after")
    payload["holdout_after"] = after
    payload["extrinsics"] = rt_to_json(rt_s)
    payload["geometry"]["baseline_m"] = float(np.linalg.norm(rt_s["right"]["C"] - rt_s["left"]["C"]))
    json_path.write_text(json.dumps(payload, indent=1, ensure_ascii=False))
    return (f"{payload['person']}/{payload['env']}: scale x{s:.3f} shoulder {w_tri:.3f}->{after['shoulder_width_m']:.3f} "
            f"(sam3d {w_sam:.3f}) rpe {after['mean_rpe']:.2f}px baseline {payload['geometry']['baseline_m']:.3f}m")


def main_anchor(args):
    tasks = [(p, args.frames) for p in sorted(args.out_dir.glob("*.json"))
             if args.overwrite or "scale_anchor" not in json.loads(p.read_text())]
    print(f"{len(tasks)} sequences to scale-anchor with {args.workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for msg in ex.map(anchor_sequence, tasks):
            print(msg, flush=True)
    rows = [json.loads(p.read_text()) for p in sorted(args.out_dir.glob("*.json"))]
    sw = np.array([r["holdout_after"]["shoulder_width_m"] for r in rows])
    sam = np.array([r.get("scale_anchor", {}).get("sam3d_shoulder_m", np.nan) for r in rows])
    rpe = np.array([r["holdout_after"]["mean_rpe"] for r in rows])
    bl = np.array([r["geometry"]["baseline_m"] for r in rows])
    print(f"\n{len(rows)} sequences | shoulder width after anchoring: median {np.nanmedian(sw):.3f} "
          f"[{np.nanmin(sw):.3f},{np.nanmax(sw):.3f}] std {np.nanstd(sw):.3f} | sam3d median {np.nanmedian(sam):.3f} | "
          f"mean rpe {np.nanmean(rpe):.2f} px | implied baseline median {np.nanmedian(bl):.3f} [{np.nanmin(bl):.2f},{np.nanmax(bl):.2f}] m")


def main_global():
    calib_entries, holdout_entries = sample_frames()
    print(f"calib frames={len(calib_entries)} holdout frames={len(holdout_entries)}")
    pts2d = collect_observations(calib_entries)
    params, n_pts, n_inl = calibrate_points(pts2d)
    print(f"calibration points={n_pts} inliers(round2)={n_inl}")
    rt_new = make_rt(params)
    print("--- optimized geometry ---", json.dumps(geometry_report(params), indent=1))
    print("--- held-out frames, production best_subset triangulation ---")
    for label, rt in (("original", RT0), ("optimized", rt_new)):
        print(f"[{label}] {json.dumps(evaluate_rt(holdout_entries, rt))}")
    OUT_JSON.write_text(json.dumps({"params": params.tolist(), "extrinsics": rt_to_json(rt_new)}, indent=1))
    print(f"saved optimized extrinsics to {OUT_JSON}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-sequence", action="store_true")
    ap.add_argument("--anchor-scale", action="store_true",
                    help="post-process per-sequence JSONs in --out-dir: fix the metric scale gauge to the SAM3D body scale")
    ap.add_argument("--out-dir", type=Path, default=TRI_DIR / "optimized_extrinsics_per_sequence")
    ap.add_argument("--frames", type=int, default=80, help="frames sampled per sequence (half calib, half holdout)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    if args.anchor_scale:
        main_anchor(args)
    elif args.per_sequence:
        main_per_sequence(args)
    else:
        main_global()


if __name__ == "__main__":
    main()
