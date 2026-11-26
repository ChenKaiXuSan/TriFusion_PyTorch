#!/usr/bin/env python3
"""Aggregate the equal-budget comparison table from learned_fusion_experiments results.

Every configuration ``<prefix>_<name>`` is expected to hold ``fold{0..4}/result.json`` and,
optionally, ``stress_zero.json``, ``stress_noise.json`` and ``driveandact.json``.  All
numbers are means over the folds present; a row with fewer than five folds is marked.

    python TriPoseFusion/eval/eqb_table.py                     # prefix eqb
    python TriPoseFusion/eval/eqb_table.py --prefix eqbw16 --out table.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent / "logs" / "learned_fusion_experiments"
VIEWS = ("front", "left", "right")
# name -> label, ordered as in the paper table
ROWS = {
    "tf_geomedian": "Geometric median (Weiszfeld)",
    "tf_consensus": "Leave-one-out consensus",
    "tf_smooth": "Mean + temporal smoothing",
    "tf_gpa": "Procrustes re-alignment + mean",
    "concat_mlp": "Concat-MLP",
    "lift_mlp": "Lifting-MLP",
    "tcn": "Temporal-CNN",
    "skel_tr": "Skeleton-Transformer",
    "tripose64": "v1 backbone",
    "gating_nt": "Learned view weighting (convex)",
    "gating": "Gating-only (+ temporal conv)",
    "resid64": "Residual-only (8K)",
    "setres": "Hybrid set-residual",
    "setfusion": "Set fusion (proposed)",
    "setfusion_d64": "Set fusion, d=64 (capacity probe)",
    "setfusion_d128": "Set fusion, d=128 (capacity probe)",
}
FREE = {
    "single_front": "Single view: front",
    "single_left": "Single view: left",
    "single_right": "Single view: right",
    "best_single_oracle": "Best single view (oracle)",
    "fuse_median": "Median fusion",
    "fuse_mean": "Mean fusion",
}


def load(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else float("nan")


def stress_stats(folds, name):
    """Mean learned MPJPE / weight on the corrupted view, over folds and the three views."""
    err, wt, ref, oracle = [], [], [], []
    for fd in folds:
        d = load(fd / name)
        if d is None:
            continue
        err.append(mean(d[v]["learned"]["mpjpe_m"] for v in VIEWS))
        wt.append(mean(d[v]["weight_on_corrupted_view"] for v in VIEWS))
        ref.append(mean(d[v]["fuse_mean_same_corruption"]["mpjpe_m"] for v in VIEWS))
        oracle.append(mean(d[v]["fuse_mean_excluding_view_oracle"]["mpjpe_m"] for v in VIEWS))
    return mean(err), mean(wt), mean(ref), mean(oracle), len(err)


def collect(prefix: str, root: Path):
    rows, ref = [], None
    for name, label in ROWS.items():
        folds = sorted((root / f"{prefix}_{name}").glob("fold[0-9]"))
        res = [r for r in (load(fd / "result.json") for fd in folds) if r is not None]
        if not res:
            continue
        if ref is None or len(res) > len(ref):
            ref = res
        da = [load(fd / "driveandact.json") for fd in folds]
        da = [d["overall"] for d in da if d is not None and "overall" in d]
        zero, noise = stress_stats(folds, "stress_zero.json"), stress_stats(folds, "stress_noise.json")
        rows.append({
            "name": name, "label": label, "folds": len(res), "params": res[0]["params"],
            "mpjpe": mean(r["learned_last"]["mpjpe_m"] for r in res),
            "pa": mean(r["learned_last"]["pa_mpjpe_m"] for r in res),
            "pck10": mean(r["learned_last"]["pck"]["0.10"] for r in res),
            "per_fold": [r["learned_last"]["mpjpe_m"] for r in res],
            "zero": zero, "noise": noise,
            "da": 1000 * mean(d["learned"]["pa_mpjpe_m"] for d in da) if da else float("nan"),
            "da_mean": 1000 * mean(d["fuse_mean"]["pa_mpjpe_m"] for d in da) if da else float("nan"),
            "n_da": len(da),
            "train_s": mean(r["train_seconds"] for r in res),
        })
    return rows, ref


def render(prefix: str, root: Path) -> str:
    rows, ref = collect(prefix, root)
    if not rows:
        return f"no results for prefix {prefix!r} under {root}\n"
    base = {k: (mean(r["training_free"][k]["mpjpe_m"] for r in ref),
                mean(r["training_free"][k]["pa_mpjpe_m"] for r in ref),
                mean(r["training_free"][k]["pck"]["0.10"] for r in ref)) for k in FREE}
    m0, p0, _ = base["fuse_mean"]
    zero_ref = next((r["zero"] for r in rows if r["zero"][4]), None)
    noise_ref = next((r["noise"] for r in rows if r["noise"][4]), None)
    da_mean = next((r["da_mean"] for r in rows if r["n_da"]), float("nan"))

    def pct(x, ref_):
        return f"{100 * (x / ref_ - 1):+.1f}%"

    def part(n, r, free_ok=False):
        """Mark a cell that does not cover all folds (training-free rows need one Drive&Act run)."""
        if n == r["folds"] == 5 or (free_ok and n == 1 and r["params"] == 0):
            return ""
        return f" ({n}f)"

    out = [f"### Equal-budget table (`{prefix}_*`, mean over folds; errors in metres, Drive&Act in mm)", "",
           "| Method | Params | MPJPE | PA-MPJPE | dMPJPE | dPA | PCK@0.10 | zero-view MPJPE | w(bad) | "
           "noise-view MPJPE | w(bad) | D&A zero-shot PA |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for key, label in FREE.items():
        m, p, k = base[key]
        extra = ["", "", "", "", ""]
        if key == "fuse_mean":
            extra = [f"{zero_ref[2]:.4f}" if zero_ref else "", "0.333",
                     f"{noise_ref[2]:.4f}" if noise_ref else "", "0.333", f"{da_mean:.1f}"]
        out.append(f"| {label} | 0 | {m:.4f} | {p:.4f} | {pct(m, m0)} | {pct(p, p0)} | {k:.3f} | "
                   + " | ".join(extra) + " |")
    if zero_ref:
        out.append(f"| Mean of the two clean views (oracle) | 0 | | | | | | {zero_ref[3]:.4f} | 0 | "
                   f"{noise_ref[3]:.4f} | 0 | |" if noise_ref else
                   f"| Mean of the two clean views (oracle) | 0 | | | | | | {zero_ref[3]:.4f} | 0 | | | |")
    for r in rows:
        flag = "" if r["folds"] == 5 else f" ({r['folds']} folds)"
        z, n = r["zero"], r["noise"]
        out.append(
            f"| {r['label']}{flag} | {r['params']:,} | {r['mpjpe']:.4f} | {r['pa']:.4f} | {pct(r['mpjpe'], m0)} | "
            f"{pct(r['pa'], p0)} | {r['pck10']:.3f} | "
            + (f"{z[0]:.4f}{part(z[4], r)} | {z[1]:.3f}" if z[4] else " | ") + " | "
            + (f"{n[0]:.4f}{part(n[4], r)} | {n[1]:.3f}" if n[4] else " | ") + " | "
            + (f"{r['da']:.1f}{part(r['n_da'], r, free_ok=True)}" if r["n_da"] else "") + " |")
    out += ["", "Per-fold MPJPE (folds 0-4):", ""]
    out.append("| Method | " + " | ".join(f"fold {k}" for k in range(5)) + " |")
    out.append("|---|" + "---:|" * 5)
    out.append("| Mean fusion | " + " | ".join(f"{r['training_free']['fuse_mean']['mpjpe_m']:.4f}" for r in ref) + " |")
    for r in rows:
        out.append(f"| {r['label']} | " + " | ".join(f"{x:.4f}" for x in r["per_fold"]) + " |")
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prefix", default="eqb")
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--out", type=Path, default=None, help="also write the markdown table to this file")
    args = ap.parse_args()
    text = render(args.prefix, args.root)
    print(text)
    if args.out:
        args.out.write_text(text)


if __name__ == "__main__":
    main()
