#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Qualitative figure for the paper: head-yaw trace from a single camera vs. fused keypoints,
against the independent head-rotation reference, over a short window of one drive.

Reads <root>/<source>/per_sequence/<seq>_angles.csv (columns time_s, yaw_deg, yaw_rot_deg,
n_views) for the requested sources and draws them on one axis; frames where the single view
has no valid head keypoints (yaw NaN) are shaded.

  python analysis/head_movement/plot_yaw_trace.py --sequence 12_夜少ない --start 120 --duration 40 \
      --fused fused_main --single view_front --out paper_tits/figures/yaw_trace.pdf
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/work/1/SKIING/chenkaixu/data/drive/downstream_analysis/head_movement_analysis")


def load(source: str, seq: str):
    t, y, yr = [], [], []
    with (ROOT / source / "per_sequence" / f"{seq}_angles.csv").open() as fh:
        for row in csv.DictReader(fh):
            t.append(float(row["time_s"]))
            y.append(float(row["yaw_deg"]) if row["yaw_deg"] else np.nan)
            yr.append(float(row["yaw_rot_deg"]) if row["yaw_rot_deg"] else np.nan)
    return np.array(t), np.array(y), np.array(yr)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sequence", required=True)
    ap.add_argument("--start", type=float, default=0.0, help="window start (s, time_s of the csv)")
    ap.add_argument("--duration", type=float, default=40.0)
    ap.add_argument("--fused", default="fused_main")
    ap.add_argument("--single", default="view_front")
    ap.add_argument("--legend-loc", default="lower right")
    ap.add_argument("--also", default=None, help="optional third source drawn thin (e.g. fused_mean)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    t, y_single, y_rot = load(args.single, args.sequence)
    _, y_fused, _ = load(args.fused, args.sequence)
    y_also = load(args.also, args.sequence)[1] if args.also else None
    w = (t >= args.start) & (t <= args.start + args.duration)
    if y_also is not None:
        y_also = y_also[w] - np.nanmedian(y_also[w])
    t, y_single, y_fused, y_rot = t[w] - args.start, y_single[w], y_fused[w], y_rot[w]
    # centre each trace on its window median so offsets between frames do not dominate
    y_single = y_single - np.nanmedian(y_single)
    y_fused = y_fused - np.nanmedian(y_fused)
    y_rot = y_rot - np.nanmedian(y_rot)

    fig, ax = plt.subplots(figsize=(7.0, 2.3))
    gap = np.isnan(y_single)
    for s, e in zip(*[np.flatnonzero(np.diff(np.concatenate([[0], gap.astype(int), [0]])))[i::2] for i in (0, 1)]):
        ax.axvspan(t[s], t[min(e, len(t) - 1)], color="0.85", lw=0, zorder=0)
    ax.plot(t, y_rot, color="0.35", lw=1.0, ls="--", label="head-rotation reference")
    ax.plot(t, y_single, color="#d62728", lw=1.0, alpha=0.9, label=f"single view ({args.single[5:]})")
    if y_also is not None:
        ax.plot(t, y_also, color="#2ca02c", lw=1.0, alpha=0.9, label="mean fusion" if args.also == "fused_mean" else args.also)
    ax.plot(t, y_fused, color="#1f77b4", lw=1.3, label="fused (TriPoseFusion)")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("head yaw (deg)")
    ax.set_xlim(0, args.duration)
    ax.legend(ncol=1, fontsize=7.5, loc=args.legend_loc, frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=300)
    r_s = np.corrcoef(np.nan_to_num(y_single), np.nan_to_num(y_rot))[0, 1]
    r_f = np.corrcoef(np.nan_to_num(y_fused), np.nan_to_num(y_rot))[0, 1]
    if y_also is not None:
        ok = np.isfinite(y_also) & np.isfinite(y_rot) & np.isfinite(y_fused)
        print(f"regression slope on the reference: fused {np.polyfit(y_rot[ok], y_fused[ok], 1)[0]:.2f}, "
              f"{args.also} {np.polyfit(y_rot[ok], y_also[ok], 1)[0]:.2f}; corr {args.also} {np.corrcoef(y_also[ok], y_rot[ok])[0, 1]:.2f}")
    print(f"saved {args.out}; window corr vs reference: single {r_s:.2f}, fused {r_f:.2f}; "
          f"single-view NaN frames {gap.mean()*100:.0f}%")


if __name__ == "__main__":
    main()
