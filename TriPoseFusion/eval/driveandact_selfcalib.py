#!/usr/bin/env python3
"""Does the self-calibration recipe of the in-house reference recover a known rig?  Test on Drive&Act.

Drive&Act runs (dense clips: vp5, vp11, vp13 x 2 runs, 1,800 frames each) with cameras
front = inner_mirror (ids_1, world frame), left = a_column_driver (ids_5), right = a_column_co_driver (ids_2).
The official intrinsics (incl. distortion) are kept; SAM 3D Body 2D keypoints are undistorted to pinhole
pixel coordinates.  For every run and perturbation level:

 1. perturb the official extrinsics of front and right: rotation by `rot` deg about a random axis through
    the camera centre, centre shifted by `shift` m in a random direction.  left stays fixed (the gauge, as
    in traingulation/calibrate_extrinsics.py); the "tape-measured" left-right baseline is that of the
    perturbed rig.
 2. run the in-house recipe (calibrate_extrinsics.py --per-sequence, then --anchor-scale) unchanged:
    80 evenly spaced frames, half to calibrate and half held out; 12-parameter least squares on the DLT
    reprojection error of the 52 kept keypoints (Huber 5 px, then 3 px after dropping points > 30 px),
    soft baseline (1 mm = 2 px), weak prior (0.05 rad or 5 cm = 1 px); then scale the scene about the left
    camera so that the triangulated shoulder width equals SAM 3D Body's metric shoulder width.
    All pixel constants are multiplied by the focal-length ratio Drive&Act / in-house (~2.3), so that the
    recipe is identical in angular terms.
 3. compare with the official calibration: rotation error (deg) and centre error (mm) of front and right,
    left-right baseline error; and triangulate every frame with the production rule (best_subset,
    reprojection <= 40 px) under the official, the perturbed (start) and the self-calibrated extrinsics,
    scored against the official 3D annotation (body14 / upper12, metrics of eval_driveandact_triangulation.py).
    Plain three-view DLT with the official calibration reproduces that script's numbers (sanity check).

    python TriPoseFusion/eval/driveandact_selfcalib.py --out TriPoseFusion/eval/logs/driveandact_selfcalib/result.json
    python TriPoseFusion/eval/driveandact_selfcalib.py --runs vp11/run1_2018-05-24-13-44-01 --levels 30 --seeds 1 --out /tmp/x.json
"""
from __future__ import annotations

import argparse
import json
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve()
TPF = HERE.parents[1]
REPO = HERE.parents[2]
for p in (str(TPF), str(REPO / "traingulation")):
    if p not in sys.path:
        sys.path.insert(0, p)
from sam3d_kpt_triangulation import VIEW_NAMES, build_camera_maps, load_config, triangulate_frame  # noqa: E402
from map_config import KEEP_KEYPOINT_INDICES  # noqa: E402
from eval.eval_driveandact_baselines import (  # noqa: E402
    COMMON_NAMES, LHIP_IDX, LSHO_IDX, MHR_TO_BODY25, NECK_IDX, RHIP_IDX, RSHO_IDX, canon, load_gt_run, subset_metrics,
)
from eval.eval_driveandact_triangulation import VIEW_CAMS, Camera  # noqa: E402

DATA_ROOT = Path("/work/1/SKIING/chenkaixu/data/drive/driveandact/tripose_eval_dense")
DA_ROOT = Path("/work/1/SKIING/chenkaixu/data/drive/driveandact")
CFG = load_config(REPO / "traingulation" / "triangulation.yaml")
TRI = CFG["triangulation"]
K_IN, _ = build_camera_maps(CFG)
F_IN = float(np.mean([K_IN[v][0, 0] for v in VIEW_NAMES]))        # in-house focal, triangulation pixels
KEEP = np.asarray(KEEP_KEYPOINT_INDICES)
L_SHO, R_SHO = int(KEEP[5]), int(KEEP[6])                          # 70-joint space
EVAL_MHR = [i for i, _ in MHR_TO_BODY25]                           # body14; neck (69) -> shoulder midpoint
LEVELS = {0: 0.0, 10: 0.05, 20: 0.10, 30: 0.15, 45: 0.20}         # rotation deg -> centre shift m
CRIT = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 50, 1e-10)


# ------------------------------------------------------------------ geometry helpers

def rt_of(R, t):
    R, t = np.asarray(R, float), np.asarray(t, float)
    return {"R": R, "t": t, "C": -R.T @ t}


def rot_err_deg(Ra, Rb):
    return float(np.degrees(np.linalg.norm(Rotation.from_matrix(Ra @ Rb.T).as_rotvec())))


def rig_errors(rt, ref):
    out = {}
    for v in ("front", "right"):
        out[f"{v}_rot_deg"] = rot_err_deg(rt[v]["R"], ref[v]["R"])
        out[f"{v}_centre_mm"] = float(1000 * np.linalg.norm(rt[v]["C"] - ref[v]["C"]))
    bl = lambda r: np.linalg.norm(r["right"]["C"] - r["left"]["C"])  # noqa: E731
    out["baseline_mm"] = float(1000 * (bl(rt) - bl(ref)))
    out["baseline_rel"] = float(bl(rt) / bl(ref) - 1)
    return out


def perturb(rt, rot_deg, shift_m, rng):
    out = {"left": rt["left"]}
    for v in ("front", "right"):
        a = rng.normal(size=3); a /= np.linalg.norm(a)
        d = rng.normal(size=3); d /= np.linalg.norm(d)
        R = Rotation.from_rotvec(np.radians(rot_deg) * a).as_matrix() @ rt[v]["R"]
        C = rt[v]["C"] + shift_m * d
        out[v] = {"R": R, "t": -R @ C, "C": C}
    return out


def scale_rt(rt, s):
    c_l = rt["left"]["C"]
    out = {"left": rt["left"]}
    for v in ("front", "right"):
        C = c_l + s * (rt[v]["C"] - c_l)
        out[v] = {"R": rt[v]["R"], "t": -rt[v]["R"] @ C, "C": C}
    return out


# ------------------------------------------------------------------ the in-house recipe, parameterised

class Recipe:
    def __init__(self, Ks, px, baseline, rt0, seed):
        self.Ks, self.px, self.baseline, self.rt0 = Ks, px, baseline, rt0
        self.rng = np.random.default_rng(seed)

    def make_rt(self, params):
        rt = {"left": self.rt0["left"]}
        for i, v in enumerate(("front", "right")):
            w, dt = params[6 * i:6 * i + 3], params[6 * i + 3:6 * i + 6]
            R = Rotation.from_rotvec(w).as_matrix() @ self.rt0[v]["R"]
            rt[v] = rt_of(R, self.rt0[v]["t"] + dt)
        return rt

    def residuals(self, params, pts2d):
        rt = self.make_rt(params)
        P = {v: self.Ks[v] @ np.hstack([rt[v]["R"], rt[v]["t"][:, None]]) for v in VIEW_NAMES}
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
            p_stack = np.stack([P[v] for v in views])
            obs = pts2d[idx_arr][:, list(key), :]
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
            for j, v in enumerate(views):
                uvw = np.einsum("rc,nc->nr", P[v], np.hstack([x3, np.ones((n, 1))]))
                depth = (x3 @ rt[v]["R"].T + rt[v]["t"])[:, 2]
                ok = (np.abs(uvw[:, 2]) > 1e-9) & (depth > 0) & ~bad
                uv = uvw[:, :2] / np.where(np.abs(uvw[:, 2:]) > 1e-9, uvw[:, 2:], 1.0)
                r = np.where(ok[:, None], uv - obs[:, j], 50.0 * self.px)
                pos = starts[idx_arr] + j
                res[pos * 2] = r[:, 0]
                res[pos * 2 + 1] = r[:, 1]
        bl_res = (np.linalg.norm(rt["right"]["C"] - rt["left"]["C"]) - self.baseline) * 2000.0 * self.px
        return np.concatenate([res, [bl_res], params * 20.0 * self.px])

    def calibrate(self, pts2d, max_points=8000):
        keep = np.isfinite(pts2d[..., 0]).sum(axis=1) >= 2
        pts2d = pts2d[keep]
        if pts2d.shape[0] > max_points:
            pts2d = pts2d[self.rng.choice(pts2d.shape[0], max_points, replace=False)]
        r1 = least_squares(self.residuals, np.zeros(12), args=(pts2d,), loss="huber", f_scale=5.0 * self.px, max_nfev=400)
        err = np.linalg.norm(self.residuals(r1.x, pts2d)[:-13].reshape(-1, 2), axis=1)
        obs_mask = np.isfinite(pts2d[..., 0])
        starts = np.concatenate([[0], np.cumsum(obs_mask.sum(axis=1))[:-1]])
        per_pt = np.array([err[starts[i]:starts[i] + int(obs_mask[i].sum())].mean() for i in range(pts2d.shape[0])])
        pts_f = pts2d[per_pt < 30.0 * self.px]
        r2 = least_squares(self.residuals, r1.x, args=(pts_f,), loss="huber", f_scale=3.0 * self.px, max_nfev=400)
        return r2.x, int(pts2d.shape[0]), int(pts_f.shape[0]), int(r1.nfev + r2.nfev)


# ------------------------------------------------------------------ data + triangulation

def load_run(vp, run):
    cams = {v: Camera.from_json(DA_ROOT / vdir / vp / f"{run}.{ids}.calibration.json") for v, (vdir, ids) in VIEW_CAMS.items()}
    sizes = {}
    for v, (vdir, ids) in VIEW_CAMS.items():
        s = json.loads((DA_ROOT / vdir / vp / f"{run}.{ids}.calibration.json").read_text())["intrinsics"]["img_size"]
        sizes[v] = (int(s["width"]), int(s["height"]))
    gt_pose, gt_valid = load_gt_run(DATA_ROOT / "gt" / vp / f"{run}.npz")
    k2, k3 = {}, {}
    for v in VIEW_NAMES:
        d2, d3 = {}, {}
        for fp in sorted((DATA_ROOT / "sam3d" / vp / run / v).glob("*_sam3d_body.npz")):
            with np.load(fp, allow_pickle=True) as z:
                o = z["output"].item()
            fid = int(fp.name.split("_")[0])
            d2[fid] = np.asarray(o["pred_keypoints_2d"], np.float64)[:, :2]
            d3[fid] = np.asarray(o["pred_keypoints_3d"], np.float64)[:, :3]
        k2[v], k3[v] = d2, d3
    fids = sorted(set(gt_pose).intersection(*(set(k2[v]) for v in VIEW_NAMES)))
    return cams, sizes, gt_pose, gt_valid, k2, k3, fids


def undistort_all(cams, sizes, k2, fids, margin):
    U = {}
    for v in VIEW_NAMES:
        raw = np.stack([k2[v][f] for f in fids])                                # (F,70,2)
        und = cv2.undistortPointsIter(raw.reshape(-1, 1, 2), cams[v].K, cams[v].dist, None, cams[v].K, CRIT).reshape(raw.shape)
        w, h = sizes[v]
        inside = (raw[..., 0] >= -margin) & (raw[..., 0] <= w + margin) & (raw[..., 1] >= -margin) & (raw[..., 1] <= h + margin)
        und[~(inside & np.isfinite(raw).all(-1))] = np.nan
        U[v] = und
    return U


def triangulate(U, frames_idx, joints, Ks, rt, strategy, max_rpe):
    """(len(frames), len(joints), 3) points (NaN = rejected) and the number of views used per point."""
    P = {v: Ks[v] @ np.hstack([rt[v]["R"], rt[v]["t"][:, None]]) for v in VIEW_NAMES}
    rtm = {v: {"R": rt[v]["R"], "t": rt[v]["t"]} for v in VIEW_NAMES}
    X = np.full((len(frames_idx), len(joints), 3), np.nan)
    nv = np.zeros((len(frames_idx), len(joints)), np.int8)
    rpe = np.full((len(frames_idx), len(joints)), np.nan)
    for a, fi in enumerate(frames_idx):
        pts = {v: U[v][fi, joints] for v in VIEW_NAMES}
        out = triangulate_frame(pts, P, rtm, None, 0.0, False, max_rpe, triangulation_strategy=strategy, min_views=2)
        ok = out[1]
        X[a, ok] = out[0][ok]
        rpe[a, ok] = out[2][ok]
        nv[a] = np.isfinite(out[3]).sum(1) * ok
    return X, nv, rpe


def pose_metrics(X, nv, gt, valid):
    pred = np.full((X.shape[0], len(COMMON_NAMES), 3), np.nan)
    sel = [k for k, i in enumerate(EVAL_MHR) if k != NECK_IDX]
    pred[:, sel] = X[:, sel]
    pred[:, NECK_IDX] = 0.5 * (pred[:, LSHO_IDX] + pred[:, RSHO_IDX])
    m = subset_metrics(pred.astype(np.float32), gt, valid)
    with np.errstate(invalid="ignore", divide="ignore"):
        mc = subset_metrics(canon(pred.astype(np.float32)), canon(gt), valid)   # body-centred frame, as the paper's protocol
    tri = valid[:, sel]
    got = np.isfinite(X[:, sel]).all(-1) & tri
    out = {s: {k: m[s][k] for k in ("mpjpe_m", "pa_mpjpe_m") if k in m.get(s, {})} for s in ("body14", "upper12")}
    out["canonical"] = {k: mc["body14"][k] for k in ("mpjpe_m", "pa_mpjpe_m") if k in mc.get("body14", {})}
    out["coverage"] = float(got.sum() / max(tri.sum(), 1))
    out["two_view_share"] = float((nv[:, sel][got] == 2).mean()) if got.any() else float("nan")
    return out


def holdout_stats(U, idx, Ks, rt, max_rpe):
    X, nv, rpe = triangulate(U, idx, KEEP, Ks, rt, "best_subset", max_rpe)
    ok = np.isfinite(rpe)
    return {"valid_ratio": float(ok.mean()), "mean_rpe_px": float(np.nanmean(rpe)) if ok.any() else float("nan"),
            "three_view_share": float(np.mean(nv[ok] == 3)) if ok.any() else float("nan")}


# ------------------------------------------------------------------ one run

def run_one(task):
    vp, run, levels, n_seeds, frames_per_seq = task
    cams, sizes, gt_pose, gt_valid, k2, k3, fids = load_run(vp, run)
    Ks = {v: cams[v].K for v in VIEW_NAMES}
    px = float(np.mean([Ks[v][0, 0] for v in VIEW_NAMES])) / F_IN
    margin, max_rpe = float(TRI.get("valid_margin_px", 8.0)) * px, float(TRI.get("max_reproj_error_px", 40.0)) * px
    U = undistort_all(cams, sizes, k2, fids, margin)
    official = {v: rt_of(cams[v].R, cams[v].t) for v in VIEW_NAMES}
    bl_off = float(np.linalg.norm(official["right"]["C"] - official["left"]["C"]))

    gt = np.stack([gt_pose[f] for f in fids])
    valid = np.stack([gt_valid[f] for f in fids])
    valid &= valid[:, [LHIP_IDX, RHIP_IDX, LSHO_IDX, RSHO_IDX, NECK_IDX]].all(axis=1)[:, None]
    all_idx = np.arange(len(fids))
    step = max(1, len(fids) // frames_per_seq)
    picked = all_idx[::step][:frames_per_seq]
    calib_idx, hold_idx = picked[0::2], picked[1::2]
    w_sam = float(np.median([np.linalg.norm(k3[v][fids[i]][L_SHO] - k3[v][fids[i]][R_SHO]) for i in picked for v in VIEW_NAMES]))

    def evaluate(rt):
        Xb, nvb, _ = triangulate(U, all_idx, EVAL_MHR, Ks, rt, "best_subset", max_rpe)
        return pose_metrics(Xb, nvb, gt, valid)

    def evaluate_plain(rt):
        Xb, nvb, _ = triangulate(U, all_idx, EVAL_MHR, Ks, rt, "all_visible", np.inf)
        return pose_metrics(Xb, nvb, gt, valid)

    res = {"run": f"{vp}/{run}", "n_frames": len(fids), "px_scale": px, "sam3d_shoulder_m": w_sam}
    Xp, nvp, _ = triangulate(U, all_idx, EVAL_MHR, Ks, official, "all_visible", np.inf)
    res["official"] = {"plain_dlt": pose_metrics(Xp, nvp, gt, valid), "best_subset": evaluate(official),
                       "holdout": holdout_stats(U, hold_idx, Ks, official, max_rpe)}
    # tracer: anchoring the official rig to the estimator's shoulder width (isolates the scale anchor)
    Xs, _, _ = triangulate(U, picked, [L_SHO, R_SHO], Ks, official, "best_subset", max_rpe)
    w_off = float(np.nanmedian(np.linalg.norm(Xs[:, 0] - Xs[:, 1], axis=-1)))
    res["official"]["triangulated_shoulder_m"] = w_off
    off_anch = scale_rt(official, w_sam / w_off)
    res["official_scale_anchored"] = {"scale": w_sam / w_off, "rig": rig_errors(off_anch, official), "pose": evaluate(off_anch),
                                      "pose_plain": evaluate_plain(off_anch)}

    trials = []
    for rot in levels:
        for seed in range(1 if rot == 0 else n_seeds):
            rng = np.random.default_rng([zlib.crc32(f"{vp}/{run}".encode()), rot, seed])
            start = perturb(official, rot, LEVELS[rot], rng)
            baseline = float(np.linalg.norm(start["right"]["C"] - start["left"]["C"]))
            rec = Recipe(Ks, px, baseline, start, seed)
            pts2d = np.stack([np.stack([U[v][i, KEEP] for v in VIEW_NAMES], axis=1) for i in calib_idx]).reshape(-1, 3, 2)
            params, n_pts, n_inl, nfev = rec.calibrate(pts2d)
            cal = rec.make_rt(params)
            Xs, _, _ = triangulate(U, picked, [L_SHO, R_SHO], Ks, cal, "best_subset", max_rpe)
            w_tri = float(np.nanmedian(np.linalg.norm(Xs[:, 0] - Xs[:, 1], axis=-1)))
            anch = scale_rt(cal, w_sam / w_tri) if np.isfinite(w_tri) and w_tri > 0 else cal
            t = {"rot_deg": rot, "shift_m": LEVELS[rot], "seed": seed, "n_points": n_pts, "n_inliers": n_inl, "nfev": nfev,
                 "scale": w_sam / w_tri if np.isfinite(w_tri) else None,
                 "start": {"rig": rig_errors(start, official), "holdout": holdout_stats(U, hold_idx, Ks, start, max_rpe)},
                 "calibrated": {"rig": rig_errors(anch, official), "rig_before_anchor": rig_errors(cal, official),
                                "holdout": holdout_stats(U, hold_idx, Ks, anch, max_rpe), "pose": evaluate(anch),
                                "pose_plain": evaluate_plain(anch),
                                # oracle scale: the self-calibrated rig rescaled to the official baseline (isolates the anchor)
                                "pose_oracle_scale": evaluate(scale_rt(cal, bl_off / np.linalg.norm(cal["right"]["C"] - cal["left"]["C"])))}}
            t["extrinsics"] = {v: {"R": anch[v]["R"].tolist(), "t": anch[v]["t"].tolist()} for v in VIEW_NAMES}
            if rot >= 30:
                t["start"]["pose"] = evaluate(start)
            trials.append(t)
            c = t["calibrated"]
            print(f"{vp}/{run} rot {rot:2d} seed {seed}: rot err F {c['rig']['front_rot_deg']:.2f} R {c['rig']['right_rot_deg']:.2f} deg | "
                  f"centre F {c['rig']['front_centre_mm']:.0f} R {c['rig']['right_centre_mm']:.0f} mm | baseline {100 * c['rig']['baseline_rel']:+.1f}% | "
                  f"body14 {1000 * c['pose']['body14']['mpjpe_m']:.1f} mm (official {1000 * res['official']['best_subset']['body14']['mpjpe_m']:.1f})",
                  flush=True)
    res["trials"] = trials
    return res


def _g(d, *keys):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return float("nan")
        d = d[k]
    return float(d)


def summarise(results):
    mm = lambda xs: 1000 * float(np.nanmean(xs)) if len(xs) else float("nan")  # noqa: E731

    def row(label, poses):
        print(f"{label:44s} abs {mm([_g(p, 'body14', 'mpjpe_m') for p in poses]):6.1f} / {mm([_g(p, 'body14', 'pa_mpjpe_m') for p in poses]):5.1f}"
              f" | body-centred {mm([_g(p, 'canonical', 'mpjpe_m') for p in poses]):6.1f} / {mm([_g(p, 'canonical', 'pa_mpjpe_m') for p in poses]):5.1f}")

    print("\n==== summary: body14 MPJPE / PA-MPJPE (mm), mean over runs (trials pooled) ====")
    off = [r["official"] for r in results]
    oa = [r["official_scale_anchored"] for r in results]
    row("official, three-view DLT", [o["plain_dlt"] for o in off])
    row("official, best_subset (in-house rule)", [o["best_subset"] for o in off])
    row("official + shoulder anchor, three-view DLT", [o.get("pose_plain", {}) for o in oa])
    row("official + shoulder anchor, best_subset", [o["pose"] for o in oa])
    print(f"official: coverage {np.mean([o['best_subset']['coverage'] for o in off]):.3f}, two-view share {np.mean([o['best_subset']['two_view_share'] for o in off]):.3f}, "
          f"holdout reprojection {np.mean([o['holdout']['mean_rpe_px'] for o in off]):.2f} px; anchor scale x{np.mean([o['scale'] for o in oa]):.3f} "
          f"({min(o['scale'] for o in oa):.3f}-{max(o['scale'] for o in oa):.3f})")
    levels = sorted({t["rot_deg"] for r in results for t in r["trials"]})
    for lv in levels:
        ts = [t for r in results for t in r["trials"] if t["rot_deg"] == lv]
        rot = [t["calibrated"]["rig"][f"{v}_rot_deg"] for t in ts for v in ("front", "right")]
        cen = [t["calibrated"]["rig"][f"{v}_centre_mm"] for t in ts for v in ("front", "right")]
        bl = [100 * t["calibrated"]["rig"]["baseline_rel"] for t in ts]
        srot = [t["start"]["rig"][f"{v}_rot_deg"] for t in ts for v in ("front", "right")]
        print(f"\nstart {lv} deg / {100 * ts[0]['shift_m']:.0f} cm ({len(ts)} trials; start rotation error median {np.median(srot):.1f} deg, "
              f"start holdout reprojection {np.nanmean([t['start']['holdout']['mean_rpe_px'] for t in ts]):.1f} px, "
              f"valid {np.mean([t['start']['holdout']['valid_ratio'] for t in ts]):.2f})")
        print(f"   calibrated rig: rotation error median {np.median(rot):.2f} max {np.max(rot):.2f} deg | centre error median {np.median(cen):.0f} max {np.max(cen):.0f} mm | "
              f"baseline {np.median(bl):+.1f}% ({np.min(bl):+.1f}..{np.max(bl):+.1f}) | holdout reprojection {np.mean([t['calibrated']['holdout']['mean_rpe_px'] for t in ts]):.2f} px")
        row("   self-calibrated + anchor, best_subset", [t["calibrated"]["pose"] for t in ts])
        row("   self-calibrated + anchor, three-view DLT", [t["calibrated"].get("pose_plain", {}) for t in ts])
        row("   self-calibrated, oracle scale, best_subset", [t["calibrated"]["pose_oracle_scale"] for t in ts])
        st = [t["start"]["pose"] for t in ts if "pose" in t["start"]]
        if st:
            row("   start (uncalibrated), best_subset", st)
    print("\nper run (start 0 deg): rotation F/R deg, centre F/R mm, baseline %, anchor scale, abs / body-centred MPJPE (best_subset) vs official")
    for r in results:
        t = next(t for t in r["trials"] if t["rot_deg"] == min(levels))
        c = t["calibrated"]
        print(f"   {r['run']:32s} {c['rig']['front_rot_deg']:4.1f}/{c['rig']['right_rot_deg']:4.1f}  {c['rig']['front_centre_mm']:4.0f}/{c['rig']['right_centre_mm']:4.0f}  "
              f"{100 * c['rig']['baseline_rel']:+5.1f}  x{t['scale']:.3f}  {1000 * _g(c['pose'], 'body14', 'mpjpe_m'):5.1f}/{1000 * _g(c['pose'], 'canonical', 'mpjpe_m'):5.1f}"
              f"  vs {1000 * _g(r['official']['best_subset'], 'body14', 'mpjpe_m'):5.1f}/{1000 * _g(r['official']['best_subset'], 'canonical', 'mpjpe_m'):5.1f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="*", default=None, help="vp/run names (default: all dense runs)")
    ap.add_argument("--levels", nargs="*", type=int, default=sorted(LEVELS))
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--frames", type=int, default=80, help="frames per run, half calibrate / half hold out (as in-house)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--summary-only", action="store_true", help="re-print the summary of an existing --out file")
    ap.add_argument("--rigs-out", type=Path, default=None,
                    help="write the self-calibrated, scale-anchored extrinsics of seed 0 at the largest level, per run "
                         "(input of build_driveandact_cache_selftri.py --extrinsics-json)")
    args = ap.parse_args()
    if args.summary_only:
        summarise(json.loads(args.out.read_text()))
        return
    runs = args.runs or [f"{p.parent.name}/{p.stem}" for p in sorted((DATA_ROOT / "gt").glob("vp*/*.npz"))]
    tasks = [(r.split("/")[0], r.split("/")[1], args.levels, args.seeds, args.frames) for r in runs]
    print(f"{len(tasks)} runs, levels {args.levels}, {args.seeds} seeds; in-house focal {F_IN:.1f} px", flush=True)
    with ProcessPoolExecutor(max_workers=min(args.workers, len(tasks))) as ex:
        results = list(ex.map(run_one, tasks))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1))
    summarise(results)
    print(f"saved -> {args.out}")
    if args.rigs_out:
        lv = max(args.levels)
        rigs = {r["run"]: {"start": {"rot_deg": lv, "shift_m": LEVELS[lv], "seed": 0},
                           "extrinsics": next(t for t in r["trials"] if t["rot_deg"] == lv and t["seed"] == 0)["extrinsics"]}
                for r in results}
        args.rigs_out.write_text(json.dumps(rigs, indent=1))
        print(f"self-calibrated extrinsics -> {args.rigs_out}")


if __name__ == "__main__":
    main()
