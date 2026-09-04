#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""STEP1 head-movement analysis on the fused 3D keypoints (for non-CS readers).

Inputs (all already on disk):
  fused_keypoints_perseq/<person>_<env>.npz   fused_v3 (T,52,3) canonical frame, frame_ids
  sam3d_body_triangulated_gt_perseq/<p>/<env>/keypoints_3d.npz   70-joint world-frame reference
  label/person_XX_<day|night>_<high|low>_h265.json   3 annotators, head-turn direction ranges
  annotation/split_mid_end/full.json                  start / mid / end frame of each drive

Outputs (analysis_head_movement/, mirrors the dataset layout):
  per_sequence/<person>_<env>.json          counts, rates, angle statistics, sign check
  per_sequence/<person>_<env>_angles.csv    per-frame yaw/pitch (relative to the driver's own
                                            median facing direction) + majority labels
  summary_by_sequence.csv / summary_by_person.csv / summary_by_environment.csv
  annotator_agreement.csv, detection_accuracy.csv, environment_tests.csv
  figures/*.png
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import sys

import numpy as np

_EVAL_DIR = Path(__file__).resolve().parents[2] / "TriPoseFusion" / "eval"
for _p in (str(_EVAL_DIR), str(_EVAL_DIR.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from eval_fusion_baselines_pesudo_gt import canonicalize_pose  # noqa: E402  (same pipeline as the paper)

FPS = 29.97
# SAM3D / MHR 127-joint skeleton (same camera frame as the 70 keypoints)
MHR_NECK, MHR_LSHO, MHR_RSHO = 112, 75, 39
MHR_SKULL, MHR_NOSE, MHR_REYE, MHR_LEYE = 114, 121, 123, 125


def mhr_head_angles(J: np.ndarray):
    """Yaw (+ = driver's right) / pitch (+ = up) of the head from fused MHR head joints in the
    canonical frame (x = right, y = down, z = fore/aft with sign fixed by the median)."""
    eye_mid = 0.5 * (J[:, MHR_REYE] + J[:, MHR_LEYE])
    f_raw = eye_mid - J[:, MHR_SKULL]                    # skull centre -> eyes: forward (and a bit down)
    lat = J[:, MHR_REYE] - J[:, MHR_LEYE]                # right - left
    up = np.cross(f_raw, lat)
    up /= np.linalg.norm(up, axis=-1, keepdims=True) + 1e-9
    fwd = np.cross(lat, up)
    fwd /= np.linalg.norm(fwd, axis=-1, keepdims=True) + 1e-9
    s = np.sign(np.nanmedian(fwd[:, 2])) or 1.0
    yaw = np.degrees(np.arctan2(fwd[:, 0], s * fwd[:, 2]))
    pitch = np.degrees(np.arctan2(-fwd[:, 1], np.hypot(fwd[:, 0], fwd[:, 2])))
    return yaw, pitch
ENV_MAP = {"昼多い": "day_high", "昼少ない": "day_low", "夜多い": "night_high", "夜少ない": "night_low"}
ENV_ORDER = ["day_high", "day_low", "night_high", "night_low"]
H_CLASSES = {"left": -1, "right": +1, "left_up": -1, "left_down": -1, "right_up": +1, "right_down": +1}
V_CLASSES = {"up": +1, "down": -1, "left_up": +1, "left_down": -1, "right_up": +1, "right_down": -1}
# 52-joint layout: 0 nose, 1 l-eye, 2 r-eye, 3 l-ear, 4 r-ear, 5 l-shoulder, 6 r-shoulder, 49 l-acromion, 50 r-acromion, 51 neck
NOSE, LEAR, REAR, LSH, RSH, NECK = 0, 3, 4, 5, 6, 51
# 70-joint reference layout: same first 7 indices, neck = 69
NECK70 = 69


# ----------------------------------------------------------------- helpers

def runs(mask: np.ndarray, merge_gap: int = 2) -> list[tuple[int, int]]:
    """Contiguous True runs as (start, end) inclusive indices, merging gaps <= merge_gap."""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    out = []
    s = p = int(idx[0])
    for i in idx[1:]:
        i = int(i)
        if i - p > merge_gap + 1:
            out.append((s, p))
            s = i
        p = i
    out.append((s, p))
    return out


def smooth(x: np.ndarray, k_med: int = 5, k_mean: int = 9) -> np.ndarray:
    """Median then moving-average filter, NaN-aware."""
    from scipy.ndimage import median_filter
    y = x.copy()
    nan = ~np.isfinite(y)
    if nan.all():
        return y
    # interpolate NaNs for filtering, restore afterwards
    idx = np.arange(len(y))
    y[nan] = np.interp(idx[nan], idx[~nan], y[~nan])
    y = median_filter(y, size=k_med, mode="nearest")
    kern = np.ones(k_mean) / k_mean
    y = np.convolve(y, kern, mode="same")
    y[nan] = np.nan
    return y


def head_angles(nose, lear, rear, x_axis, up_axis, fwd_axis):
    """Yaw (+ = driver's right) and pitch (+ = up) of the head-forward vector, degrees."""
    f = nose - 0.5 * (lear + rear)
    fx = np.einsum("tc,tc->t", f, x_axis) if x_axis.ndim == 2 else f @ x_axis
    fu = np.einsum("tc,tc->t", f, up_axis) if up_axis.ndim == 2 else f @ up_axis
    ff = np.einsum("tc,tc->t", f, fwd_axis) if fwd_axis.ndim == 2 else f @ fwd_axis
    yaw = np.degrees(np.arctan2(fx, ff))
    pitch = np.degrees(np.arctan2(fu, np.sqrt(fx**2 + ff**2)))
    return yaw, pitch


def chordal_mean(M: np.ndarray) -> np.ndarray:
    """Nearest rotation matrices to per-frame sums of rotations (T,3,3); NaN where empty."""
    out = np.full_like(M, np.nan)
    ok = np.isfinite(M).all(axis=(1, 2)) & (np.abs(M).sum(axis=(1, 2)) > 0)
    if ok.any():
        U, _, Vt = np.linalg.svd(M[ok])
        d = np.sign(np.linalg.det(U @ Vt))
        U[:, :, -1] *= d[:, None]
        out[ok] = U @ Vt
    return out


def fleiss_kappa(binary_by_rater: np.ndarray) -> float:
    """binary_by_rater: (raters, frames) bool → Fleiss' kappa for 2 categories."""
    k, n = binary_by_rater.shape
    n1 = binary_by_rater.sum(0).astype(float)
    n0 = k - n1
    p_i = (n1**2 + n0**2 - k) / (k * (k - 1))
    p_bar = p_i.mean()
    p1 = n1.sum() / (k * n)
    p_e = p1**2 + (1 - p1) ** 2
    return float((p_bar - p_e) / (1 - p_e)) if p_e < 1 else float("nan")


def match_events(pred: list, gt: list) -> tuple[int, int, int]:
    """Count TP/FP/FN by overlap between (start,end) intervals."""
    used = set()
    tp = 0
    for ps, pe in pred:
        for j, (gs, ge) in enumerate(gt):
            if j in used:
                continue
            if ps <= ge and gs <= pe:
                used.add(j)
                tp += 1
                break
    return tp, len(pred) - tp, len(gt) - tp


# ----------------------------------------------------------------- loading

def load_labels(path: Path):
    d = json.load(open(path, encoding="utf-8"))
    per_rater = []
    for a in d["annotations"]:
        segs = []
        for vl in a.get("videoLabels", []):
            for r in vl["ranges"]:
                for lab in vl["timelinelabels"]:
                    segs.append((int(r["start"]), int(r["end"]), lab))
        per_rater.append(segs)
    return per_rater


def load_splits(path: Path) -> dict:
    out = {}
    for task in json.load(open(path, encoding="utf-8")):
        video = Path(task["data"]["video"]).name.replace("_h265.mp4", "")
        marks = {}
        for ann in task["annotations"][:1]:
            for r in ann["result"]:
                v = r["value"]
                for lab in v["timelinelabels"]:
                    marks[lab] = int(v["ranges"][0]["start"])
        out[video] = marks
    return out


# ----------------------------------------------------------------- per sequence

def analyse_sequence(name: str, args, splits: dict, det_grid: dict) -> dict:
    person, env_jp = name[:2], name[3:]
    env = ENV_MAP[env_jp]
    video = f"person_{person}_{env}"
    raters = load_labels(args.label_dir / f"{video}_h265.json")
    n_raters = len(raters)
    f = np.load(args.fused_dir / f"{name}.npz", allow_pickle=True)
    frame_ids = f["frame_ids"].astype(int)
    T = len(frame_ids)
    marks = splits.get(video, {})
    start = marks.get("start", int(frame_ids[0]))
    end = marks.get("end", int(frame_ids[-1]))
    duration_min = max(end - start, 1) / FPS / 60.0

    # --- label arrays over video frame numbers [0, maxf]
    maxf = max(end, int(frame_ids[-1]), max((s[1] for r in raters for s in r), default=0)) + 1
    H = np.zeros((n_raters, maxf), dtype=np.int8)   # -1 left, +1 right
    V = np.zeros((n_raters, maxf), dtype=np.int8)   # -1 down, +1 up
    for i, segs in enumerate(raters):
        for s, e, lab in segs:
            if lab in H_CLASSES:
                H[i, s:e + 1] = H_CLASSES[lab]
            if lab in V_CLASSES:
                V[i, s:e + 1] = V_CLASSES[lab]
    drive = np.zeros(maxf, dtype=bool)
    drive[start:end + 1] = True

    def majority(sign_arr):
        votes = (sign_arr != 0).sum(0) >= max(2, math.ceil(n_raters / 2))
        sgn = np.sign(sign_arr.sum(0))
        return votes & drive, sgn

    h_maj, h_sign = majority(H)
    v_maj, v_sign = majority(V)
    h_events = [(s, e, int(np.sign(h_sign[s:e + 1].sum()) or 0)) for s, e in runs(h_maj)]
    v_events = [(s, e, int(np.sign(v_sign[s:e + 1].sum()) or 0)) for s, e in runs(v_maj)]
    per_rater_counts = {
        "horizontal": [len(runs((H[i] != 0) & drive)) for i in range(n_raters)],
        "vertical": [len(runs((V[i] != 0) & drive)) for i in range(n_raters)],
    }
    kappa_h = fleiss_kappa((H[:, drive] != 0)) if n_raters >= 2 else float("nan")
    kappa_v = fleiss_kappa((V[:, drive] != 0)) if n_raters >= 2 else float("nan")

    # --- angles from the selected keypoint source (canonical frame: x = right, y = down, z = fore/aft)
    #     fused_v3 / fused_mean / fused_median: fused keypoints; view_<cam>: that camera's own SAM3D
    #     keypoints after its per-frame canonicalization (head-vs-torso angles are invariant to it).
    if args.source.startswith("view_"):
        cams = [str(c) for c in f["cameras"]]
        P = f["view_pose"][:, :, cams.index(args.source[5:]), :].astype(np.float64)
    else:
        P = f[args.source].astype(np.float64)
    fwd_sign = np.sign(np.nanmedian((P[:, NOSE] - 0.5 * (P[:, LEAR] + P[:, REAR]))[:, 2])) or 1.0
    ex, ey, ez = np.eye(3)
    yaw_c, pitch_c = head_angles(P[:, NOSE], P[:, LEAR], P[:, REAR], ex, -ey, fwd_sign * ez)

    # --- absolute angles from the world-frame reference (cabin-fixed cameras)
    g = np.load(args.gt_dir / person / env_jp / "keypoints_3d.npz", allow_pickle=True)
    G, gv = g["keypoints_3d"].astype(np.float64), g["valid_mask"]
    ok = gv[:, [NOSE, LEAR, REAR, LSH, RSH, NECK70]].all(1)
    x_raw = G[:, RSH] - G[:, LSH]
    down_raw = 0.5 * (G[:, LSH] + G[:, RSH]) - G[:, NECK70]
    x_axis = np.nanmedian(x_raw[ok], 0) if ok.any() else np.array([1, 0, 0.0])
    x_axis /= np.linalg.norm(x_axis)
    down = np.nanmedian(down_raw[ok], 0) if ok.any() else np.array([0, 1, 0.0])
    down -= down @ x_axis * x_axis
    down /= np.linalg.norm(down)
    fwd = np.cross(x_axis, down)
    nose_g, lear_g, rear_g = G[:, NOSE].copy(), G[:, LEAR].copy(), G[:, REAR].copy()
    for arr in (nose_g, lear_g, rear_g):
        arr[~ok] = np.nan
    fsign_g = np.sign(np.nanmedian((nose_g - 0.5 * (lear_g + rear_g)) @ fwd)) or 1.0
    yaw_a, pitch_a = head_angles(nose_g, lear_g, rear_g, x_axis, -down, fsign_g * fwd)
    g_ids = g["frame_ids"].astype(int)
    # align reference frames to fused frame ids
    pos = {fid: i for i, fid in enumerate(g_ids)}
    sel = np.array([pos.get(fid, -1) for fid in frame_ids])
    yaw_abs = np.where(sel >= 0, yaw_a[np.clip(sel, 0, None)], np.nan)
    pitch_abs = np.where(sel >= 0, pitch_a[np.clip(sel, 0, None)], np.nan)
    in_drive = (frame_ids >= start) & (frame_ids <= end)

    # --- primary: head orientation from SAM3D head-joint rotations, fused over views
    #     (camera frame -> world/cabin frame with the per-sequence extrinsics, chordal mean,
    #      relative to the driver's median orientation while driving)
    yaw_rot = pitch_rot = None
    n_views_used = np.zeros(T, dtype=np.int8)
    jc_views = []
    rot_dir = args.head_rot_dir / name
    if all((rot_dir / f"{v}.npz").exists() for v in ("front", "left", "right")):
        gviews = [str(v) for v in g["views"]]
        Rw_sum = np.zeros((T, 3, 3))
        cnt = np.zeros(T)
        jc_views = []
        for v in ("front", "left", "right"):
            z = np.load(rot_dir / f"{v}.npz")
            hr = z["head_rot"].astype(np.float64)
            if len(hr) != T:
                raise ValueError(f"head_rot length {len(hr)} != {T} for {name}/{v}")
            if "joint_coords" in z.files:
                jc_views.append(z["joint_coords"].astype(np.float32))
            Rv = g["R"][gviews.index(v)]                 # world -> camera (OpenCV: x right, y down, z forward)
            # SAM3D rotations are expressed in a y-up / z-backward camera frame; flip y,z to
            # OpenCV, then camera -> world with R_v^T.  Verified: inter-view disagreement of the
            # same head rotation drops from 50-140 deg to 8-22 deg with this convention.
            FLIP = np.diag([1.0, -1.0, -1.0])
            Rw = np.einsum("ji,jk,tkl->til", Rv, FLIP, hr)   # R_v^T @ FLIP @ head_rot
            okv = np.isfinite(Rw).all(axis=(1, 2))
            Rw_sum[okv] += Rw[okv]
            cnt += okv
        n_views_used = cnt.astype(np.int8)
        R_fused = chordal_mean(Rw_sum)
        ref_frames = in_drive & (cnt > 0)
        if ref_frames.any():
            R0 = chordal_mean(np.nansum(R_fused[ref_frames], axis=0)[None])[0]
            Rrel = R_fused @ R0.T
            fwd_h = fsign_g * fwd                        # cabin forward, sign fixed by the median nose-ear direction
            fvec = np.einsum("tij,j->ti", Rrel, fwd_h)
            fx, fu, ff = fvec @ x_axis, fvec @ (-down), fvec @ fwd_h
            yaw_rot = np.degrees(np.arctan2(fx, ff))
            pitch_rot = np.degrees(np.arctan2(fu, np.hypot(fx, ff)))

    # --- KEYPOINT path (primary): fused MHR head joints.  Each view's 127-joint skeleton is
    #     canonicalized exactly like the paper's pipeline (neck origin, shoulder axis) and the
    #     three views are mean-fused; head yaw/pitch follow from the fused head joints.
    yaw_mhr = pitch_mhr = None
    J_fused = None
    if len(jc_views) == 3:
        Jv = np.stack([canonicalize_pose(j, neck_index=MHR_NECK, left_shoulder_index=MHR_LSHO,
                                         right_shoulder_index=MHR_RSHO) for j in jc_views], axis=2)  # (T,127,V,3)
        with np.errstate(all="ignore"):
            J_fused = np.nanmean(Jv, axis=2).astype(np.float32)
        yaw_mhr, pitch_mhr = mhr_head_angles(J_fused.astype(np.float64))
        if args.export_mhr:
            fpath = args.fused_dir / f"{name}.npz"
            payload = dict(np.load(fpath, allow_pickle=True))
            payload["fused_mhr127_mean"] = J_fused
            payload["mhr127_n_views"] = n_views_used
            np.savez_compressed(fpath, **payload)

    # Primary = the fused 52-joint keypoints (fused_v3: nose vs ear midpoint).  Validated on
    # subject 01 against the SAM3D head-rotation reference (yaw corr ~0.8; ~25 deg during true
    # turns vs ~3 deg at rest).  Alternatives are kept as validation columns / opt-in.
    if args.prefer_mhr and yaw_mhr is not None:
        angle_source, primary_yaw, primary_pitch = "fused_mhr_head_keypoints", yaw_mhr, pitch_mhr
    else:
        angle_source, primary_yaw, primary_pitch = f"{args.source}_keypoints", yaw_c, pitch_c
    nanT = np.full(T, np.nan)

    series = {}
    for key, raw in (("yaw_rel", primary_yaw), ("pitch_rel", primary_pitch),
                     ("yaw_rot", yaw_rot if yaw_rot is not None else nanT), ("pitch_rot", pitch_rot if pitch_rot is not None else nanT),
                     ("yaw_kpt", yaw_c), ("pitch_kpt", pitch_c), ("yaw_abs", yaw_abs), ("pitch_abs", pitch_abs)):
        sm = smooth(raw)
        base = np.nanmedian(sm[in_drive]) if in_drive.any() else np.nanmedian(sm)
        series[key] = sm - base
    valid_frac = float(np.isfinite(series["yaw_rel"][in_drive]).mean()) if in_drive.any() else 0.0
    # validation: keypoint-derived yaw vs rotation-derived yaw should agree if the head joints are sound
    kv = in_drive & np.isfinite(series["yaw_rel"]) & np.isfinite(series["yaw_rot"])
    kpt_vs_rot_corr = round(float(np.corrcoef(series["yaw_rel"][kv], series["yaw_rot"][kv])[0, 1]), 3) if kv.sum() > 100 else None

    # --- peak angle inside each annotated event, and sign check
    fid_pos = {int(fid): i for i, fid in enumerate(frame_ids)}

    def event_peak(ev_list, key):
        peaks, signs_ok, n_sign = [], 0, 0
        for s, e, sgn in ev_list:
            idxs = [fid_pos[x] for x in range(s - 5, e + 6) if x in fid_pos]
            if not idxs:
                continue
            seg = series[key][idxs]
            if not np.isfinite(seg).any():
                continue
            k = int(np.nanargmax(np.abs(seg)))
            peaks.append(float(abs(seg[k])))
            if sgn != 0:
                n_sign += 1
                signs_ok += int(np.sign(seg[k]) == sgn)
        return peaks, (signs_ok / n_sign if n_sign else float("nan"))

    h_peaks, h_sign_ok = event_peak(h_events, "yaw_rel")
    v_peaks, v_sign_ok = event_peak(v_events, "pitch_rel")
    h_peaks_abs, _ = event_peak(h_events, "yaw_abs")

    # --- AI physical head turns from the primary angle series (fixed thresholds)
    AI_TH_H, AI_TH_V, AI_MIN_FRAMES = 15.0, 10.0, 6

    def ai_events(sig, th):
        m = (np.abs(np.nan_to_num(sig)) > th) & in_drive
        evs = []
        for s, e in runs(m, 3):
            if e - s + 1 >= AI_MIN_FRAMES:
                seg = sig[s:e + 1]
                k = int(np.nanargmax(np.abs(seg)))
                evs.append((s, e, float(seg[k])))
        return evs

    ai_h = ai_events(series["yaw_rel"], AI_TH_H)
    ai_v = ai_events(series["pitch_rel"], AI_TH_V)
    ai_h_frame = np.zeros(T, dtype=np.int8)
    ai_v_frame = np.zeros(T, dtype=np.int8)
    for s, e, pk in ai_h:
        ai_h_frame[s:e + 1] = 1 if pk > 0 else -1
    for s, e, pk in ai_v:
        ai_v_frame[s:e + 1] = 1 if pk > 0 else -1

    def overlap_frac(ai_list, label_events):
        if not ai_list:
            return None
        hit = 0
        for s, e, _ in ai_list:
            fs, fe = int(frame_ids[s]), int(frame_ids[e])
            hit += any(fs <= le and ls <= fe for ls, le, _ in label_events)
        return round(hit / len(ai_list), 2)

    ai_stats = {
        "ai_horizontal_turns": len(ai_h), "ai_vertical_turns": len(ai_v),
        "ai_horizontal_turns_per_min": round(len(ai_h) / duration_min, 2),
        "ai_vertical_turns_per_min": round(len(ai_v) / duration_min, 2),
        "ai_left_turns": sum(1 for *_, pk in ai_h if pk < 0), "ai_right_turns": sum(1 for *_, pk in ai_h if pk > 0),
        "ai_up_turns": sum(1 for *_, pk in ai_v if pk > 0), "ai_down_turns": sum(1 for *_, pk in ai_v if pk < 0),
        "ai_horizontal_peak_deg_mean": round(float(np.mean([abs(pk) for *_, pk in ai_h])), 1) if ai_h else None,
        "ai_horizontal_peak_deg_max": round(float(max(abs(pk) for *_, pk in ai_h)), 1) if ai_h else None,
        "ai_vertical_peak_deg_mean": round(float(np.mean([abs(pk) for *_, pk in ai_v])), 1) if ai_v else None,
        "ai_horizontal_turn_duration_s_mean": round(float(np.mean([(e - s + 1) / FPS for s, e, _ in ai_h])), 2) if ai_h else None,
        "ai_turns_overlapping_annotated_glance": overlap_frac(ai_h, h_events),
        "annotated_glances_with_head_turn_ge15": round(float(np.mean([p >= AI_TH_H for p in h_peaks])), 2) if h_peaks else None,
    }

    # --- threshold detection vs majority labels (for detection_accuracy.csv)
    det = {}
    for (theta, tmin) in det_grid["horizontal"]:
        m = np.abs(series["yaw_rel"]) > theta
        m &= in_drive
        pred = [(frame_ids[s], frame_ids[e]) for s, e in runs(m, 3) if e - s + 1 >= tmin]
        det[("horizontal", theta, tmin)] = match_events(pred, [(s, e) for s, e, _ in h_events])
    for (theta, tmin) in det_grid["vertical"]:
        m = np.abs(series["pitch_rel"]) > theta
        m &= in_drive
        pred = [(frame_ids[s], frame_ids[e]) for s, e in runs(m, 3) if e - s + 1 >= tmin]
        det[("vertical", theta, tmin)] = match_events(pred, [(s, e) for s, e, _ in v_events])

    yaw_d = series["yaw_rel"][in_drive]
    pitch_d = series["pitch_rel"][in_drive]
    rec = {
        "sequence": name, "person": person, "environment": env, "environment_jp": env_jp,
        "drive_start_frame": int(start), "drive_end_frame": int(end), "drive_minutes": round(duration_min, 2),
        "n_annotators": n_raters,
        "horizontal_turns": len(h_events), "vertical_turns": len(v_events),
        "horizontal_turns_per_min": round(len(h_events) / duration_min, 2),
        "vertical_turns_per_min": round(len(v_events) / duration_min, 2),
        "left_turns": sum(1 for *_, s in h_events if s < 0), "right_turns": sum(1 for *_, s in h_events if s > 0),
        "up_turns": sum(1 for *_, s in v_events if s > 0), "down_turns": sum(1 for *_, s in v_events if s < 0),
        "per_annotator_horizontal": per_rater_counts["horizontal"],
        "per_annotator_vertical": per_rater_counts["vertical"],
        "kappa_horizontal": round(kappa_h, 3), "kappa_vertical": round(kappa_v, 3),
        "mean_turn_duration_s": round(float(np.mean([(e - s + 1) / FPS for s, e, _ in h_events + v_events])), 2)
        if (h_events or v_events) else None,
        "yaw_std_deg": round(float(np.nanstd(yaw_d)), 1), "pitch_std_deg": round(float(np.nanstd(pitch_d)), 1),
        "yaw_p95_abs_deg": round(float(np.nanpercentile(np.abs(yaw_d), 95)), 1),
        "pitch_p95_abs_deg": round(float(np.nanpercentile(np.abs(pitch_d), 95)), 1),
        "horizontal_event_peak_yaw_deg_mean": round(float(np.mean(h_peaks)), 1) if h_peaks else None,
        "horizontal_event_peak_yaw_deg_median": round(float(np.median(h_peaks)), 1) if h_peaks else None,
        "horizontal_event_peak_yaw_abs_deg_mean": round(float(np.mean(h_peaks_abs)), 1) if h_peaks_abs else None,
        "vertical_event_peak_pitch_deg_mean": round(float(np.mean(v_peaks)), 1) if v_peaks else None,
        "sign_agreement_horizontal": round(h_sign_ok, 2) if h_sign_ok == h_sign_ok else None,
        "sign_agreement_vertical": round(v_sign_ok, 2) if v_sign_ok == v_sign_ok else None,
        "angle_valid_fraction": round(valid_frac, 3),
        **ai_stats,
        "angle_source": angle_source,
        "yaw_keypoint_vs_rotation_corr": kpt_vs_rot_corr,
        "n_views_mean": round(float(n_views_used[in_drive].mean()), 2) if in_drive.any() else None,
        "_det": det,
    }

    # per-frame CSV
    out_csv = args.out_dir / "per_sequence" / f"{name}_angles.csv"
    hm = np.array([h_sign[fid] if 0 <= fid < maxf and h_maj[fid] else 0 for fid in frame_ids])
    vm = np.array([v_sign[fid] if 0 <= fid < maxf and v_maj[fid] else 0 for fid in frame_ids])
    with open(out_csv, "w", encoding="utf-8") as fh:
        fh.write("frame_id,time_s,in_drive,yaw_deg,pitch_deg,n_views,ai_turn_horizontal,ai_turn_vertical,"
                 "label_horizontal,label_vertical,yaw_rot_deg,pitch_rot_deg,yaw_kpt52_deg,pitch_kpt52_deg,yaw_abs_deg,pitch_abs_deg\n")
        HV = {0: '', 1: 'right', -1: 'left'}
        VV = {0: '', 1: 'up', -1: 'down'}
        for i in range(T):
            fh.write(f"{frame_ids[i]},{frame_ids[i]/FPS:.2f},{int(in_drive[i])},"
                     f"{series['yaw_rel'][i]:.1f},{series['pitch_rel'][i]:.1f},{int(n_views_used[i])},"
                     f"{HV[int(ai_h_frame[i])]},{VV[int(ai_v_frame[i])]},{HV[int(hm[i])]},{VV[int(vm[i])]},"
                     f"{series['yaw_rot'][i]:.1f},{series['pitch_rot'][i]:.1f},"
                     f"{series['yaw_kpt'][i]:.1f},{series['pitch_kpt'][i]:.1f},{series['yaw_abs'][i]:.1f},{series['pitch_abs'][i]:.1f}\n")
    return rec


# ----------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    D = Path("/work/1/SKIING/chenkaixu/data/drive")
    ap.add_argument("--fused-dir", type=Path, default=D / "fused_keypoints_perseq")
    ap.add_argument("--gt-dir", type=Path, default=D / "sam3d_body_triangulated_gt_perseq")
    ap.add_argument("--label-dir", type=Path, default=D / "label")
    ap.add_argument("--split-json", type=Path, default=D / "annotation/split_mid_end/full.json")
    ap.add_argument("--out-dir", type=Path, default=D / "analysis_head_movement")
    ap.add_argument("--head-rot-dir", type=Path, default=D / "sam3d_head_rot_cache",
                    help="per-sequence SAM3D head-joint rotations (extract_sam3d_head_rotation.py)")
    ap.add_argument("--sequences", nargs="*", default=None, help="subset of <person>_<env> names (debug)")
    ap.add_argument("--source", default="fused_v3",
                    choices=["fused_v3", "fused_mean", "fused_median", "view_front", "view_left", "view_right"],
                    help="keypoint source for the head angles (all in the canonical body frame)")
    ap.add_argument("--prefer-mhr", action="store_true",
                    help="use fused MHR head joints as the primary angle source instead of the 52-joint fused_v3")
    ap.add_argument("--export-mhr", action="store_true",
                    help="write the fused 127-joint MHR skeleton (fused_mhr127_mean) into the fused_keypoints_perseq files")
    args = ap.parse_args()
    (args.out_dir / "per_sequence").mkdir(parents=True, exist_ok=True)
    (args.out_dir / "figures").mkdir(exist_ok=True)

    splits = load_splits(args.split_json)
    det_grid = {"horizontal": [(th, t) for th in (10, 15, 20, 25, 30, 40) for t in (3, 6, 9)],
                "vertical": [(th, t) for th in (5, 8, 10, 15, 20) for t in (3, 6, 9)]}
    names = sorted(p.stem for p in args.fused_dir.glob("*.npz"))
    if args.sequences:
        names = [n for n in names if n in set(args.sequences)]
    records = []
    for n in names:
        try:
            rec = analyse_sequence(n, args, splits, det_grid)
        except Exception as e:  # keep going, report at the end
            print(f"!! {n}: {e}", flush=True)
            continue
        records.append(rec)
        json.dump({k: v for k, v in rec.items() if k != "_det"},
                  open(args.out_dir / "per_sequence" / f"{n}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"{n}: {rec['drive_minutes']} min, H {rec['horizontal_turns']} ({rec['horizontal_turns_per_min']}/min), "
              f"V {rec['vertical_turns']} ({rec['vertical_turns_per_min']}/min), yaw p95 {rec['yaw_p95_abs_deg']}°, "
              f"sign ok H {rec['sign_agreement_horizontal']}", flush=True)

    # ---- summary tables
    cols = ["sequence", "person", "environment", "drive_minutes",
            "ai_horizontal_turns", "ai_vertical_turns", "ai_horizontal_turns_per_min", "ai_vertical_turns_per_min",
            "ai_left_turns", "ai_right_turns", "ai_up_turns", "ai_down_turns",
            "ai_horizontal_peak_deg_mean", "ai_horizontal_peak_deg_max", "ai_vertical_peak_deg_mean",
            "ai_horizontal_turn_duration_s_mean",
            "horizontal_turns", "vertical_turns",
            "horizontal_turns_per_min", "vertical_turns_per_min", "left_turns", "right_turns", "up_turns", "down_turns",
            "mean_turn_duration_s", "ai_turns_overlapping_annotated_glance", "annotated_glances_with_head_turn_ge15",
            "yaw_std_deg", "pitch_std_deg", "yaw_p95_abs_deg", "pitch_p95_abs_deg",
            "horizontal_event_peak_yaw_deg_mean", "horizontal_event_peak_yaw_abs_deg_mean",
            "vertical_event_peak_pitch_deg_mean", "sign_agreement_horizontal", "sign_agreement_vertical",
            "kappa_horizontal", "kappa_vertical", "angle_valid_fraction", "angle_source", "n_views_mean",
            "yaw_keypoint_vs_rotation_corr"]
    with open(args.out_dir / "summary_by_sequence.csv", "w", encoding="utf-8") as fh:
        fh.write(",".join(cols) + "\n")
        for r in records:
            fh.write(",".join("" if r.get(c) is None else str(r[c]) for c in cols) + "\n")

    def agg(rows, keys):
        out = {}
        for k in keys:
            vals = [r[k] for r in rows if r.get(k) is not None]
            out[k] = round(float(np.mean(vals)), 2) if vals else None
        return out

    num_keys = ["ai_horizontal_turns_per_min", "ai_vertical_turns_per_min", "ai_horizontal_peak_deg_mean",
                "ai_vertical_peak_deg_mean", "ai_horizontal_turn_duration_s_mean",
                "horizontal_turns_per_min", "vertical_turns_per_min", "yaw_p95_abs_deg", "pitch_p95_abs_deg",
                "horizontal_event_peak_yaw_deg_mean", "vertical_event_peak_pitch_deg_mean", "mean_turn_duration_s"]
    by_person = defaultdict(list)
    by_env = defaultdict(list)
    for r in records:
        by_person[r["person"]].append(r)
        by_env[r["environment"]].append(r)
    with open(args.out_dir / "summary_by_person.csv", "w", encoding="utf-8") as fh:
        fh.write("person,n_sequences," + ",".join(num_keys) + "," +
                 ",".join(f"horizontal_turns_per_min_{e}" for e in ENV_ORDER) + "\n")
        for p in sorted(by_person):
            a = agg(by_person[p], num_keys)
            per_env = {r["environment"]: r["horizontal_turns_per_min"] for r in by_person[p]}
            fh.write(f"{p},{len(by_person[p])}," + ",".join("" if a[k] is None else str(a[k]) for k in num_keys) + "," +
                     ",".join(str(per_env.get(e, "")) for e in ENV_ORDER) + "\n")
    with open(args.out_dir / "summary_by_environment.csv", "w", encoding="utf-8") as fh:
        fh.write("environment,environment_jp,n_sequences," + ",".join(num_keys) + "\n")
        for e in ENV_ORDER:
            a = agg(by_env[e], num_keys)
            jp = [k for k, v in ENV_MAP.items() if v == e][0]
            fh.write(f"{e},{jp},{len(by_env[e])}," + ",".join("" if a[k] is None else str(a[k]) for k in num_keys) + "\n")

    # ---- annotator agreement
    with open(args.out_dir / "annotator_agreement.csv", "w", encoding="utf-8") as fh:
        fh.write("sequence,kappa_horizontal,kappa_vertical,annotator_counts_horizontal,annotator_counts_vertical,majority_horizontal,majority_vertical\n")
        for r in records:
            fh.write(f"{r['sequence']},{r['kappa_horizontal']},{r['kappa_vertical']},"
                     f"\"{r['per_annotator_horizontal']}\",\"{r['per_annotator_vertical']}\",{r['horizontal_turns']},{r['vertical_turns']}\n")

    # ---- detection accuracy (pooled over all sequences)
    pooled = defaultdict(lambda: [0, 0, 0])
    for r in records:
        for k, (tp, fp, fn) in r["_det"].items():
            pooled[k][0] += tp; pooled[k][1] += fp; pooled[k][2] += fn
    best = {}
    with open(args.out_dir / "detection_accuracy.csv", "w", encoding="utf-8") as fh:
        fh.write("axis,threshold_deg,min_frames,TP,FP,FN,precision,recall,F1\n")
        for (axis, th, t), (tp, fp, fn) in sorted(pooled.items()):
            prec = tp / (tp + fp) if tp + fp else 0.0
            rec_ = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * prec * rec_ / (prec + rec_) if prec + rec_ else 0.0
            fh.write(f"{axis},{th},{t},{tp},{fp},{fn},{prec:.3f},{rec_:.3f},{f1:.3f}\n")
            if f1 > best.get(axis, (0, None))[0]:
                best[axis] = (f1, (th, t, prec, rec_))

    # ---- environment tests (paired across persons)
    tests = []
    try:
        from scipy import stats
        for key in ["ai_horizontal_turns_per_min", "ai_vertical_turns_per_min", "ai_horizontal_peak_deg_mean",
                    "horizontal_turns_per_min", "vertical_turns_per_min", "yaw_p95_abs_deg"]:
            table = {p: {r["environment"]: r[key] for r in rows} for p, rows in by_person.items()}
            full = [p for p in table if all(e in table[p] and table[p][e] is not None for e in ENV_ORDER)]
            arrs = [[table[p][e] for p in full] for e in ENV_ORDER]
            fr = stats.friedmanchisquare(*arrs)
            day = [(table[p]["day_high"] + table[p]["day_low"]) / 2 for p in full]
            night = [(table[p]["night_high"] + table[p]["night_low"]) / 2 for p in full]
            high = [(table[p]["day_high"] + table[p]["night_high"]) / 2 for p in full]
            low = [(table[p]["day_low"] + table[p]["night_low"]) / 2 for p in full]
            w_dn = stats.wilcoxon(day, night)
            w_hl = stats.wilcoxon(high, low)
            tests.append((key, len(full), fr.pvalue, np.mean(day), np.mean(night), w_dn.pvalue, np.mean(high), np.mean(low), w_hl.pvalue))
    except Exception as e:
        print("environment tests skipped:", e)
    with open(args.out_dir / "environment_tests.csv", "w", encoding="utf-8") as fh:
        fh.write("measure,n_persons,friedman_p_4env,mean_day,mean_night,wilcoxon_p_day_vs_night,mean_high_light,mean_low_light,wilcoxon_p_high_vs_low\n")
        for t in tests:
            fh.write(",".join(f"{x:.4g}" if isinstance(x, float) else str(x) for x in t) + "\n")

    # ---- figures
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        persons = sorted(by_person)
        AI_SUB = "AI-measured physical head turns (|yaw| > 15 deg / |pitch| > 10 deg, >= 0.2 s)"
        AN_SUB = "annotators' glance events (3-rater majority)"
        for key_ai, key_an, title, fname in [
                ("ai_horizontal_turns_per_min", "horizontal_turns_per_min", "Horizontal", "fig1_horizontal_turns_per_min.png"),
                ("ai_vertical_turns_per_min", "vertical_turns_per_min", "Vertical", "fig2_vertical_turns_per_min.png")]:
            fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
            w = 0.2
            for ax, key, sub in zip(axes, (key_ai, key_an), (AI_SUB, AN_SUB)):
                for j, e in enumerate(ENV_ORDER):
                    vals = [next((r[key] for r in by_person[p] if r["environment"] == e), np.nan) for p in persons]
                    ax.bar(np.arange(len(persons)) + (j - 1.5) * w, vals, w, label=e)
                ax.set_ylabel("per minute"); ax.set_title(f"{title}: {sub}")
            axes[1].set_xticks(range(len(persons))); axes[1].set_xticklabels(persons); axes[1].set_xlabel("participant")
            axes[0].legend(ncol=4, fontsize=8); fig.tight_layout(); fig.savefig(args.out_dir / "figures" / fname, dpi=150); plt.close(fig)
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        for ax, key, ttl in zip(axes, ["ai_horizontal_turns_per_min", "ai_vertical_turns_per_min", "ai_horizontal_peak_deg_mean", "horizontal_turns_per_min"],
                                ["AI horizontal turns / min", "AI vertical turns / min", "AI mean peak yaw per turn (deg)", "Annotated horizontal glances / min"]):
            ax.boxplot([[r[key] for r in by_env[e] if r.get(key) is not None] for e in ENV_ORDER], labels=ENV_ORDER)
            ax.set_title(ttl, fontsize=10); ax.tick_params(axis="x", rotation=20)
        fig.suptitle("By lighting environment (each point = one drive)"); fig.tight_layout()
        fig.savefig(args.out_dir / "figures" / "fig3_environment_boxplots.png", dpi=150); plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        hp = [r["ai_horizontal_peak_deg_mean"] for r in records if r.get("ai_horizontal_peak_deg_mean")]
        vp = [r["horizontal_event_peak_yaw_deg_mean"] for r in records if r.get("horizontal_event_peak_yaw_deg_mean")]
        axes[0].hist(hp, bins=20); axes[0].set_title("AI head turns: mean peak yaw per drive (deg)", fontsize=10)
        axes[1].hist(vp, bins=20); axes[1].set_title("Annotated glances: mean peak yaw per drive (deg)", fontsize=10)
        fig.tight_layout(); fig.savefig(args.out_dir / "figures" / "fig4_turn_angle_histograms.png", dpi=150); plt.close(fig)
        # example timeline
        ex = records[0]["sequence"]
        import csv
        rows = list(csv.DictReader(open(args.out_dir / "per_sequence" / f"{ex}_angles.csv", encoding="utf-8")))
        t = np.array([float(r["time_s"]) for r in rows]); y = np.array([float(r["yaw_deg"]) for r in rows])
        lab = np.array([r["label_horizontal"] for r in rows]); ai = np.array([r["ai_turn_horizontal"] for r in rows])
        fig, ax = plt.subplots(figsize=(13, 3.8))
        ax.plot(t, y, lw=0.8, color="k", label="head yaw from fused head keypoints (3 views, canonical frame; deg, + = right)")
        for s, e in runs(ai != ""):
            ax.axvspan(t[s], t[e], color="tab:green", alpha=0.35, lw=0)
        for s, e in runs(lab == "right"):
            ax.axvspan(t[s], t[e], ymin=0.0, ymax=0.08, color="tab:red", alpha=0.9, lw=0)
        for s, e in runs(lab == "left"):
            ax.axvspan(t[s], t[e], ymin=0.0, ymax=0.08, color="tab:blue", alpha=0.9, lw=0)
        ax.axhline(15, color="gray", ls="--", lw=0.6); ax.axhline(-15, color="gray", ls="--", lw=0.6)
        ax.set_xlabel("time (s)"); ax.set_ylabel("deg")
        ax.set_title(f"{ex}: green = AI head turns (>15 deg); bottom strip = annotators' glances (red = right, blue = left)", fontsize=10)
        ax.legend(loc="upper right", fontsize=8); fig.tight_layout()
        fig.savefig(args.out_dir / "figures" / "fig5_example_timeline.png", dpi=150); plt.close(fig)
        # detection F1 heatmaps
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
        for ax, axis in zip(axes, ["horizontal", "vertical"]):
            ths = sorted({th for (a, th, t) in pooled if a == axis}); ts = sorted({t for (a, th, t) in pooled if a == axis})
            M = np.zeros((len(ths), len(ts)))
            for i, th in enumerate(ths):
                for j, tt in enumerate(ts):
                    tp, fp, fn = pooled[(axis, th, tt)]
                    p_ = tp / (tp + fp) if tp + fp else 0; r_ = tp / (tp + fn) if tp + fn else 0
                    M[i, j] = 2 * p_ * r_ / (p_ + r_) if p_ + r_ else 0
            im = ax.imshow(M, vmin=0, vmax=1, cmap="viridis"); ax.set_xticks(range(len(ts))); ax.set_xticklabels(ts)
            ax.set_yticks(range(len(ths))); ax.set_yticklabels(ths); ax.set_xlabel("min duration (frames)"); ax.set_ylabel("angle threshold (deg)")
            ax.set_title(f"{axis}: overlap (F1) of threshold detections with annotators' glance labels", fontsize=9)
            for i in range(len(ths)):
                for j in range(len(ts)):
                    ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", color="w", fontsize=7)
        fig.tight_layout(); fig.savefig(args.out_dir / "figures" / "fig6_detection_f1.png", dpi=150); plt.close(fig)
    except Exception as e:
        print("figures skipped:", e)

    # ---- console summary
    print("\n==== SUMMARY ====")
    print(f"sequences analysed: {len(records)} / {len(names)}")
    for e in ENV_ORDER:
        a = agg(by_env[e], num_keys)
        print(f"{e:11s} AI turns H {a['ai_horizontal_turns_per_min']}/min V {a['ai_vertical_turns_per_min']}/min peak {a['ai_horizontal_peak_deg_mean']}°  | annotated glances H {a['horizontal_turns_per_min']}/min V {a['vertical_turns_per_min']}/min  | yaw p95 {a['yaw_p95_abs_deg']}°")
    print("kappa horizontal mean", round(float(np.nanmean([r['kappa_horizontal'] for r in records])), 3),
          "vertical", round(float(np.nanmean([r['kappa_vertical'] for r in records])), 3))
    print("sign agreement H mean", round(float(np.nanmean([r['sign_agreement_horizontal'] for r in records if r['sign_agreement_horizontal'] is not None])), 2),
          "V", round(float(np.nanmean([r['sign_agreement_vertical'] for r in records if r['sign_agreement_vertical'] is not None])), 2))
    for axis, (f1, (th, t, p_, r_)) in best.items():
        print(f"best detection {axis}: theta={th}° min={t} frames  F1={f1:.3f} (P {p_:.2f} R {r_:.2f})")
    for t in tests:
        print(f"env test {t[0]}: friedman p={t[2]:.3g}; day {t[3]:.2f} vs night {t[4]:.2f} p={t[5]:.3g}; high {t[6]:.2f} vs low {t[7]:.2f} p={t[8]:.3g}")
    json.dump({"best_detection": {a: {"f1": v[0], "theta": v[1][0], "min_frames": v[1][1], "precision": v[1][2], "recall": v[1][3]} for a, v in best.items()},
               "environment_tests": tests}, open(args.out_dir / "analysis_summary.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
