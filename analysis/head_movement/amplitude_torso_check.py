#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Does torso rotation bias the head-yaw check?  Torso-corrected check signals for every keypoint source.

keypoint yaw (yaw_deg)       head relative to the torso (body-centred frame of the keypoints)
check signal (yaw_rot_deg)   head relative to the cabin (estimator head rotation, fused over views)
torso yaw, two estimates     est: per-view MHR shoulder axis (joints 39/75; camera frame, OpenCV axes,
                                  verified by inter-view agreement 16-19 deg vs. 81-83 deg with the
                                  y/z flip used for the head rotations) -> cabin, averaged over views
                             tri: shoulder axis of the triangulated reference, widths >20 % off the
                                  drive median rejected and interpolated
                             each smoothed either like the head yaw (~0.3 s) or over 61 frames (~2 s)
torso-corrected check        check - torso yaw  (head relative to the torso, small-angle approximation)

Per drive (frames inside the annotated drive): torso-yaw spread (std), torso share of large turns
(slope of torso yaw on the check over frames with |check| > 20 deg), and for every source the
correlation of keypoint yaw with each reference and the amplitude ratio (median |kpt| / |ref| over
frames with |ref| > 20 deg, as in source_validity.py).  Reported as means (ratio, r) or medians
(spread, slope) over drives.

    python analysis/head_movement/amplitude_torso_check.py [--root <head_movement_analysis>] [--json out.json]
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter

D = Path("/work/1/SKIING/chenkaixu/data/drive")
SOURCES = ["view_front", "view_left", "view_right", "fused_median", "fused_mean", "fused_main"]
LSHO, RSHO = 75, 39            # MHR joints in the per-view estimator skeleton
LSH, RSH, NECK70 = 5, 6, 69    # keypoints of the triangulated reference
SMOOTH = {"0.3s": lambda t: np.convolve(median_filter(t, size=5, mode="nearest"), np.ones(9) / 9, mode="same"),
          "2s": lambda t: median_filter(t, size=61, mode="nearest")}


def unit(a):
    with np.errstate(invalid="ignore", divide="ignore"):
        return a / np.linalg.norm(a, axis=-1, keepdims=True)


def col(rows, k):
    return np.array([float(q[k]) if q[k] not in ("", "nan") else np.nan for q in rows])


def fill(t):
    idx = np.arange(len(t))
    f = np.isfinite(t)
    return np.interp(idx, idx[f], t[f]) if f.any() else t


def ratio(kpt, ref, mask):
    big = mask & np.isfinite(kpt) & np.isfinite(ref) & (np.abs(ref) > 20)
    return float(np.median(np.abs(kpt[big]) / np.abs(ref[big]))) if big.sum() > 30 else np.nan


def corr(a, b, mask):
    m = mask & np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() > 30 else np.nan


def raw_torso_yaws(seq, fids_csv):
    """Unsmoothed torso yaw (deg, about the cabin vertical) from both estimates, on the csv frames."""
    person, env = seq.split("_")
    g = np.load(D / "sam3d_body_triangulated_gt_perseq" / person / env / "keypoints_3d.npz", allow_pickle=True)
    gviews = [str(v) for v in g["views"]]
    G, gv, gf = g["keypoints_3d"].astype(float), g["valid_mask"], g["frame_ids"].astype(int)
    ok = gv[:, [LSH, RSH, NECK70]].all(1)
    xm = unit(np.nanmedian(G[ok, RSH] - G[ok, LSH], 0))          # cabin axes as in the analysis
    dm = np.nanmedian(0.5 * (G[ok, LSH] + G[ok, RSH]) - G[ok, NECK70], 0)
    dm = unit(dm - dm @ xm * xm)
    fm = np.cross(xm, dm)
    yaw_of = lambda x: np.degrees(np.arctan2(x @ fm, x @ xm))  # noqa: E731
    xs = []
    for v in ("front", "left", "right"):
        J = np.load(D / "sam3d_head_rot_cache" / seq / f"{v}.npz")["joint_coords"].astype(float)
        xs.append(unit(J[:, RSHO] - J[:, LSHO]) @ g["R"][gviews.index(v)])   # camera -> cabin: R_v^T x
    est = yaw_of(unit(np.nansum(xs, 0)))
    x = G[:, RSH] - G[:, LSH]
    w = np.linalg.norm(x, axis=1)
    good = ok & (np.abs(w / np.median(w[ok]) - 1) < 0.2)
    tri_ref = yaw_of(unit(x))
    tri_ref[~good] = np.nan
    pos = {f: i for i, f in enumerate(gf)}
    sel = np.array([pos.get(f, -1) for f in fids_csv])
    tri = np.where(sel >= 0, tri_ref[np.clip(sel, 0, None)], np.nan)
    return {"est": fill(est), "tri": fill(tri)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=D / "downstream_analysis/head_movement_analysis")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()
    refs_names = ["cabin"] + [f"{t}_{s}" for t in ("est", "tri") for s in SMOOTH]
    res = {src: {k: {"ratio": [], "r": []} for k in refs_names} for src in SOURCES}
    torso = {k: {"spread": [], "slope": []} for k in refs_names[1:]}
    seqs = sorted(p.name[:-len("_angles.csv")] for p in (args.root / "fused_main" / "per_sequence").glob("*_angles.csv"))
    for seq in seqs:
        base = list(csv.DictReader((args.root / "fused_main" / "per_sequence" / f"{seq}_angles.csv").open(encoding="utf-8")))
        dr = np.array([q["in_drive"] == "1" for q in base])
        yc = col(base, "yaw_rot_deg")
        raw = raw_torso_yaws(seq, np.array([int(q["frame_id"]) for q in base]))
        refs = {"cabin": yc}
        for t, s in ((t, s) for t in ("est", "tri") for s in SMOOTH):
            ty = SMOOTH[s](raw[t])
            ty = ty - np.nanmedian(ty[dr])
            m = dr & np.isfinite(yc) & np.isfinite(ty)
            ty = (np.sign(np.corrcoef(ty[m], yc[m])[0, 1]) or 1.0) * ty    # orient like the head yaw
            big = m & (np.abs(yc) > 20)
            torso[f"{t}_{s}"]["spread"].append(float(np.std(ty[m])))
            torso[f"{t}_{s}"]["slope"].append(float(np.polyfit(yc[big], ty[big], 1)[0]) if big.sum() > 30 else np.nan)
            refs[f"{t}_{s}"] = yc - ty
        for src in SOURCES:
            yk = col(list(csv.DictReader((args.root / src / "per_sequence" / f"{seq}_angles.csv").open(encoding="utf-8"))), "yaw_deg")
            for k, ref in refs.items():
                res[src][k]["ratio"].append(ratio(yk, ref, dr))
                res[src][k]["r"].append(corr(yk, ref, dr))
    out = {"n_drives": len(seqs), "torso": {}, "sources": {}}
    print(f"{len(seqs)} drives\n\ntorso yaw            spread (median deg)   share of turns >20 deg (median slope, p10-p90)")
    for k, v in torso.items():
        sl = np.array(v["slope"]); sl = sl[np.isfinite(sl)]
        out["torso"][k] = {"spread_median": float(np.median(v["spread"])), "slope_median": float(np.median(sl)),
                           "slope_p10": float(np.percentile(sl, 10)), "slope_p90": float(np.percentile(sl, 90)), "n_slope": int(sl.size)}
        print(f"{k:20s} {np.median(v['spread']):>8.1f}              {np.median(sl):.2f} ({np.percentile(sl, 10):.2f} to {np.percentile(sl, 90):.2f}; {sl.size} drives)")
    stats = {"ratio": ("amplitude ratio (mean over drives)", lambda a: np.nanmean(a), "{:>11.2f}"),
             "r": ("correlation r (mean over drives)", lambda a: np.nanmean(a), "{:>11.2f}"),
             "r_min": ("correlation r (minimum over drives)", lambda a: np.nanmin(a), "{:>11.2f}"),
             "n_fail": ("drives with r < 0.5", lambda a: float(np.sum(np.asarray(a) < 0.5)), "{:>11.0f}")}
    for metric, (title, fn, fmt) in stats.items():
        print(f"\n{title}")
        print(f"{'source':13s}" + "".join(f"{k:>11s}" for k in refs_names))
        for src in SOURCES:
            vals = [float(fn(res[src][k]["ratio" if metric == "ratio" else "r"])) for k in refs_names]
            out["sources"].setdefault(src, {})[metric] = dict(zip(refs_names, vals))
            print(f"{src:13s}" + "".join(fmt.format(x) for x in vals))
    out["per_drive"] = {"sequences": seqs, "values": res}
    if args.json:
        args.json.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
