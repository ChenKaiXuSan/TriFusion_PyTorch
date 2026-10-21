#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Direction agreement between keypoint head-yaw and human left/right glance events.

Human glance annotations are small-amplitude (median accompanying head turn ~8 deg) and
are NOT a head-turn ground truth; but their *direction* is unambiguous.  For every
majority-vote left/right glance event we test whether the keypoint yaw deviates from the
driver's resting yaw in the labelled direction:

    dyaw = mean(yaw during event) - median(yaw over the drive)     [deg]
    agree = sign(dyaw) == label sign   (and |dyaw| >= --min-deg to count as a decision)

Input: <root>/<source>/per_sequence/<seq>_angles.csv produced by
run_head_movement_analysis.py (columns yaw_deg, in_drive, label_horizontal).
Output: <root>/glance_direction_agreement.csv / .md (per source, pooled over 88 drives,
plus per-sequence agreement for the paired comparison).
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

SOURCES = ["fused_v3", "fused_full", "fused_mean", "fused_median", "view_front", "view_left", "view_right"]


def runs(mask: np.ndarray):
    idx = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(np.int8), [0]])))
    return list(zip(idx[::2], idx[1::2] - 1))


def load_angles(path: Path):
    yaw, drive, lab = [], [], []
    with path.open() as fh:
        for row in csv.DictReader(fh):
            yaw.append(float(row["yaw_deg"]) if row["yaw_deg"] else np.nan)
            drive.append(row["in_drive"] == "1")
            lab.append(row["label_horizontal"])
    return np.array(yaw), np.array(drive), np.array(lab)


def analyse_source(root: Path, source: str, min_deg: float, sign_right: float):
    per_seq = []
    n_ev = n_dec = n_agree = 0
    for p in sorted((root / source / "per_sequence").glob("*_angles.csv")):
        yaw, drive, lab = load_angles(p)
        base = np.nanmedian(yaw[drive])
        ev = n_d = n_a = 0
        for side, sgn in (("left", -sign_right), ("right", sign_right)):
            for s, e in runs((lab == side) & drive):
                seg = yaw[s:e + 1]
                if np.isfinite(seg).sum() < 2:
                    continue
                d = np.nanmean(seg) - base
                ev += 1
                if abs(d) >= min_deg:
                    n_d += 1
                    n_a += int(np.sign(d) == sgn)
        per_seq.append((p.name.replace("_angles.csv", ""), ev, n_d, n_a))
        n_ev += ev; n_dec += n_d; n_agree += n_a
    return {"source": source, "events": n_ev, "decided": n_dec, "agree": n_agree,
            "agreement": n_agree / max(n_dec, 1), "coverage": n_dec / max(n_ev, 1), "per_seq": per_seq}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path,
                    default=Path("/work/1/SKIING/chenkaixu/data/drive/downstream_analysis/head_movement_analysis"))
    ap.add_argument("--min-deg", type=float, default=3.0, help="min |dyaw| to count as a directional decision")
    ap.add_argument("--sources", nargs="*", default=None)
    args = ap.parse_args()
    sources = [s for s in (args.sources or SOURCES) if (args.root / s / "per_sequence").is_dir()]

    # yaw sign convention: determine once from the reference source (mean yaw on right-labelled
    # frames minus left-labelled frames), then apply the same convention to every source.
    ref = sources[0]
    r_vals, l_vals = [], []
    for p in sorted((args.root / ref / "per_sequence").glob("*_angles.csv")):
        yaw, drive, lab = load_angles(p)
        base = np.nanmedian(yaw[drive])
        r_vals.extend((yaw[(lab == "right") & drive] - base).tolist())
        l_vals.extend((yaw[(lab == "left") & drive] - base).tolist())
    sign_right = 1.0 if np.nanmean(r_vals) > np.nanmean(l_vals) else -1.0
    print(f"sign convention from {ref}: right-labelled mean dyaw {np.nanmean(r_vals):+.1f}, "
          f"left {np.nanmean(l_vals):+.1f} -> right = {'+' if sign_right > 0 else '-'}yaw")

    results = [analyse_source(args.root, s, args.min_deg, sign_right) for s in sources]
    with (args.root / "glance_direction_agreement.csv").open("w") as fh:
        fh.write("source,events,decided,agree,agreement,coverage\n")
        for r in results:
            fh.write(f"{r['source']},{r['events']},{r['decided']},{r['agree']},{r['agreement']:.4f},{r['coverage']:.4f}\n")
    with (args.root / "glance_direction_agreement_per_sequence.csv").open("w") as fh:
        fh.write("source,sequence,events,decided,agree\n")
        for r in results:
            for seq, ev, nd, na in r["per_seq"]:
                fh.write(f"{r['source']},{seq},{ev},{nd},{na}\n")
    lines = [f"# Left/right glance direction agreement (|dyaw| >= {args.min_deg} deg; 88 drives)", "",
             "| source | glance events | decided | agreement | decided fraction |", "|---|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r['source']} | {r['events']} | {r['decided']} | {100*r['agreement']:.1f}% | {100*r['coverage']:.0f}% |")
    (args.root / "glance_direction_agreement.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
