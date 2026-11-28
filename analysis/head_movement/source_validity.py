#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Head-yaw measurement validity per keypoint source (the paper's downstream table).

For every source folder under <root> (written by run_head_movement_analysis.py) and over the
frames inside the annotated drive:

  r mean / r min / failed   per-drive Pearson correlation between keypoint yaw and the independent
                            head-rotation yaw (summary_by_sequence.csv), its minimum, and the number
                            of drives with r < 0.5
  amplitude ratio           median of |keypoint yaw| / |rotation yaw| on frames where the independent
                            signal exceeds 20 deg, averaged over drives (1 = same amplitude)
  rest noise                std of keypoint yaw on frames where |rotation yaw| < 3 deg (deg)
  pitch r                   per-drive correlation of keypoint pitch with rotation pitch, when present
  glance agreement          left/right direction agreement with the human glance annotations
                            (glance_direction_agreement.csv), if that file exists
  turns/min                 detected horizontal head turns per minute of covered driving

    python analysis/head_movement/source_validity.py [--root <head_movement_analysis>]
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

SOURCES = ["view_front", "view_left", "view_right", "fused_median", "fused_mean", "fused_main"]


def col(rows, key, mask):
    if key not in rows[0]:
        return None
    return np.array([float(r[key]) if r[key] not in ("", "nan", "None") else np.nan for r in rows])[mask]


def compute(root: Path) -> list[dict]:
    """One dict per source present under root (keys: source, r_mean, r_min, failed, amplitude, rest_sd,
    pitch_r, glance, turns_per_min, drives)."""
    glance = {}
    g = root / "glance_direction_agreement.csv"
    if g.exists():
        glance = {r["source"]: float(r["agreement"]) for r in csv.DictReader(g.open())}
    out = []
    for s in SOURCES:
        summ = root / s / "summary_by_sequence.csv"
        if not summ.exists():
            continue
        rows = list(csv.DictReader(summ.open(encoding="utf-8")))
        f = lambda k: np.array([float(r[k]) if r.get(k) not in (None, "", "None", "nan") else np.nan for r in rows])
        r = f("yaw_keypoint_vs_rotation_corr")
        turns = f("ai_horizontal_turns_per_min")
        amp, rest, pr = [], [], []
        for p in sorted((root / s / "per_sequence").glob("*_angles.csv")):
            a = list(csv.DictReader(p.open(encoding="utf-8")))
            d = np.array([x["in_drive"] == "1" for x in a])
            yaw, yr = col(a, "yaw_deg", d), col(a, "yaw_rot_deg", d)
            m = np.isfinite(yaw) & np.isfinite(yr)
            big, still = m & (np.abs(yr) > 20), m & (np.abs(yr) < 3)
            amp.append(float(np.median(np.abs(yaw[big]) / np.abs(yr[big]))) if big.sum() > 30 else np.nan)
            rest.append(float(np.nanstd(yaw[still])) if still.sum() > 100 else np.nan)
            pit, pitr = col(a, "pitch_deg", d), col(a, "pitch_rot_deg", d)
            if pit is not None and pitr is not None:
                mp = np.isfinite(pit) & np.isfinite(pitr)
                pr.append(float(np.corrcoef(pit[mp], pitr[mp])[0, 1]) if mp.sum() > 100 else np.nan)
        out.append({"source": s, "r_mean": float(np.nanmean(r)), "r_min": float(np.nanmin(r)), "failed": int((r < 0.5).sum()),
                    "amplitude": float(np.nanmean(amp)), "rest_sd": float(np.nanmean(rest)),
                    "pitch_r": float(np.nanmean(pr)) if pr else float("nan"), "glance": glance.get(s, float("nan")),
                    "turns_per_min": float(np.nanmean(turns)), "drives": len(rows)})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path,
                    default=Path("/work/1/SKIING/chenkaixu/data/drive/downstream_analysis/head_movement_analysis"))
    args = ap.parse_args()
    print(f"{'source':13s} {'r mean':>7s} {'r min':>6s} {'failed':>6s} {'ampl.':>6s} {'rest sd':>8s} {'pitch r':>8s} {'glance':>7s} {'turns/min':>9s}  drives")
    for d in compute(args.root):
        print(f"{d['source']:13s} {d['r_mean']:7.3f} {d['r_min']:6.2f} {d['failed']:6d} {d['amplitude']:6.2f} {d['rest_sd']:8.2f} "
              f"{d['pitch_r']:8.2f} {100 * d['glance']:6.1f}% {d['turns_per_min']:9.2f}  {d['drives']}")


if __name__ == "__main__":
    main()
