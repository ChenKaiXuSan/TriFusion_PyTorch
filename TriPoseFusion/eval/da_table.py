#!/usr/bin/env python3
"""Drive&Act comparison table: zero-shot and in-domain training under one protocol.

Protocol (all columns): leave-one-subject-out, 3 folds, whole validation sequences,
compute_metrics against the dataset reference.  Sources (pegasus/eqb_driveandact_job.sh):

    zero-shot     eqb_<name>/fold{0..4}/crosseval_driveandact.json   (mean over the 5 in-house folds)
    trained       daeq_<name>     dataset reference,  1500 steps x 32 x 64
    label-free    daeqlf_<name>   self-triangulated reference (gt_pose_tri9), same budget
    6000 steps    daeq6k_<name>   unchanged in-house recipe, validated every 250 steps

    python TriPoseFusion/eval/da_table.py [--out da_table.md]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from eqb_table import ROWS, VIEWS, load, mean

ROOT = Path(__file__).resolve().parent / "logs" / "learned_fusion_experiments"
FREE = {"single_front": "Single view: front", "single_left": "Single view: left", "single_right": "Single view: right",
        "best_single_oracle": "Best single view (oracle)", "fuse_median": "Median fusion", "fuse_mean": "Mean fusion"}
CURVE_STEPS = (250, 500, 1000, 1500, 2000, 3000, 4000, 6000)


def results(root: Path, tag: str) -> list[dict]:
    out = [load(fd / "result.json") for fd in sorted((root / tag).glob("fold[0-9]"))]
    return [r for r in out if r is not None]


def mm(rs, key="learned_last"):
    """Fold-mean (MPJPE, PA) in mm, or None."""
    if not rs:
        return None
    return 1000 * mean(r[key]["mpjpe_m"] for r in rs), 1000 * mean(r[key]["pa_mpjpe_m"] for r in rs)


def stress(root: Path, tag: str, mode: str):
    err, ref, orc, wt = [], [], [], []
    for fd in sorted((root / tag).glob("fold[0-9]")):
        d = load(fd / f"stress_{mode}.json")
        if d is None:
            continue
        err.append(mean(d[v]["learned"]["mpjpe_m"] for v in VIEWS))
        ref.append(mean(d[v]["fuse_mean_same_corruption"]["mpjpe_m"] for v in VIEWS))
        orc.append(mean(d[v]["fuse_mean_excluding_view_oracle"]["mpjpe_m"] for v in VIEWS))
        wt.append(mean(d[v]["weight_on_corrupted_view"] for v in VIEWS))
    return (1000 * mean(err), 1000 * mean(ref), 1000 * mean(orc), mean(wt), len(err)) if err else None


def curve(rs):
    """step -> fold-mean (MPJPE, PA) in mm, only steps present in every fold."""
    by = {}
    for r in rs:
        for h in r["history"]:
            by.setdefault(h["step"], []).append((h["mpjpe_m"], h["pa_mpjpe_m"]))
    return {s: tuple(1000 * np.mean(v, axis=0)) for s, v in sorted(by.items()) if len(v) == len(rs)}


def cell(x, n=None, want=None):
    if x is None:
        return ""
    flag = "" if n is None or n == want else f" ({n}f)"
    return f"{x[0]:.1f} / {x[1]:.1f}{flag}"


def render(root: Path) -> str:
    rows, anchor = [], None
    for name, label in ROWS.items():
        zs = [load(fd / "crosseval_driveandact.json") for fd in sorted((root / f"eqb_{name}").glob("fold[0-9]"))]
        zs = [z for z in zs if z is not None]
        tr, lf, k6 = results(root, f"daeq_{name}"), results(root, f"daeqlf_{name}"), results(root, f"daeq6k_{name}")
        if not (zs or tr or lf or k6):
            continue
        anchor = anchor or (tr or k6 or lf or None)
        params = next((r["params"] for r in (tr or k6 or lf)), None)
        if params is None:
            src = load(root / f"eqb_{name}" / "fold0" / "result.json")
            params = src["params"] if src else 0
        zero = ((1000 * mean(z["learned_mean"]["mpjpe_m"] for z in zs), 1000 * mean(z["learned_mean"]["pa_mpjpe_m"] for z in zs))
                if zs else None)
        zrange = (1000 * min(z["learned_mean"]["pa_mpjpe_m"] for z in zs), 1000 * max(z["learned_mean"]["pa_mpjpe_m"] for z in zs)) if zs else None
        cv = curve(k6) if k6 else {}
        best = min(cv.items(), key=lambda kv: kv[1][0]) if cv else None
        rows.append(dict(name=name, label=label, params=params, zero=zero, nz=len(zs), zrange=zrange,
                         tr=mm(tr), ntr=len(tr), lf=mm(lf), nlf=len(lf), k6=mm(k6), nk6=len(k6), best=best, cv=cv,
                         sz=stress(root, f"daeq_{name}", "zero"), sn=stress(root, f"daeq_{name}", "noise")))
    if not rows or anchor is None:
        return f"no Drive&Act results under {root}\n"
    base = {k: (1000 * mean(r["training_free"][k]["mpjpe_m"] for r in anchor),
                1000 * mean(r["training_free"][k]["pa_mpjpe_m"] for r in anchor)) for k in FREE}
    sref = next((r["sz"] for r in rows if r["sz"]), None)
    nref = next((r["sn"] for r in rows if r["sn"]), None)

    out = ["### Drive&Act: zero-shot vs trained in-domain (MPJPE / PA-MPJPE, mm; leave-one-subject-out, 3 folds)", "",
           "| Method | Params | Zero-shot (PA range over 5 source folds) | Trained, dataset ref. (1500) | Trained, label-free (1500) | "
           "Trained, 6000 steps | Best common checkpoint of the 6000-step run | zero-view MPJPE | noise-view MPJPE |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for key, label in FREE.items():
        tail = ""
        if key == "fuse_mean" and sref and nref:
            tail = f"{sref[1]:.1f} | {nref[1]:.1f}"
        else:
            tail = " | "
        out.append(f"| {label} | 0 | {cell(base[key])} | {cell(base[key])} | {cell(base[key])} | {cell(base[key])} | | {tail} |")
    if sref and nref:
        out.append(f"| Mean of the two clean views (oracle) | 0 | | | | | | {sref[2]:.1f} | {nref[2]:.1f} |")
    for r in rows:
        free = r["params"] == 0
        zr = f" ({r['zrange'][0]:.1f}–{r['zrange'][1]:.1f})" if r["zrange"] and not free else ""
        best = f"{r['best'][1][0]:.1f} / {r['best'][1][1]:.1f} @ {r['best'][0]}" if r["best"] else ""
        sz = f"{r['sz'][0]:.1f}" + ("" if r["sz"][4] == 3 else f" ({r['sz'][4]}f)") if r["sz"] else ""
        sn = f"{r['sn'][0]:.1f}" + ("" if r["sn"][4] == 3 else f" ({r['sn'][4]}f)") if r["sn"] else ""
        out.append(f"| {r['label']} | {r['params']:,} | {cell(r['zero'], r['nz'], 5)}{zr} | {cell(r['tr'], r['ntr'], 3)} | "
                   f"{cell(r['lf'], r['nlf'], 3)} | {cell(r['k6'], r['nk6'], 3)} | {best} | {sz} | {sn} |")

    out += ["", "Validation MPJPE (mm, fold mean) along the 6000-step run:", "",
            "| Method | " + " | ".join(str(s) for s in CURVE_STEPS) + " |", "|---|" + "---:|" * len(CURVE_STEPS)]
    for r in rows:
        if r["cv"]:
            out.append(f"| {r['label']} | " + " | ".join(f"{r['cv'][s][0]:.1f}" if s in r["cv"] else "" for s in CURVE_STEPS) + " |")
    out += ["", "Validation PA-MPJPE (mm, fold mean) along the 6000-step run:", "",
            "| Method | " + " | ".join(str(s) for s in CURVE_STEPS) + " |", "|---|" + "---:|" * len(CURVE_STEPS)]
    for r in rows:
        if r["cv"]:
            out.append(f"| {r['label']} | " + " | ".join(f"{r['cv'][s][1]:.1f}" if s in r["cv"] else "" for s in CURVE_STEPS) + " |")
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    text = render(args.root)
    print(text)
    if args.out:
        args.out.write_text(text)


if __name__ == "__main__":
    main()
