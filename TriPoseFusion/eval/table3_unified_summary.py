#!/usr/bin/env python3
"""汇总 eval_table3_unified.py 各方法输出为 Markdown 表。

两种聚合口径都给出（同一份逐序列 CSV）：
  - seq-mean：先逐序列（person×env）求指标，再对序列取均值——与另一会话的同子集对照表一致；
  - fold-agg：eval_trifusion_pesudo_gt 的 fold 级聚合（unified_summary.json）。
best-single（oracle）：按序列取 MPJPE 最小视角，仅在 seq-mean 口径下定义。
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ORDER = ["single_front", "single_left", "single_right", "best_single", "mean", "median", "learned", "model"]
LABEL = {"single_front": "single front", "single_left": "single left", "single_right": "single right",
         "best_single": "best single (oracle, per-sequence)", "mean": "fuse mean (= uniform gate)",
         "median": "fuse median", "learned": "learned fusion baseline (1.9K, pseudo-GT supervised)",
         "model": "TriPoseFusion (self-supervised)"}
METRICS = ["mpjpe", "pa_mpjpe", "mpjpe_pa_frames", "pck@0.1"]


def _seq_rows(d: Path, fold: int):
    p = d / f"person_env_metrics_fold_{fold}.csv"
    if not p.exists():
        return {}
    rows = {}
    with open(p, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            key = (r.get("person_id"), r.get("env_folder"))
            rows[key] = {k: float(v) for k, v in r.items() if k not in ("person_id", "env_folder", "environment_name") and v not in ("", None)}
    return rows


def _seq_mean(rows):
    out = {}
    for m in METRICS:
        v = [r[m] for r in rows.values() if m in r and r[m] == r[m]]
        out[m] = sum(v) / len(v) if v else float("nan")
    out["n"] = len(rows)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--fold", type=int, default=0)
    a = ap.parse_args()

    fold_agg, seq_mean, groups, robust = {}, {}, {}, {}
    for m in ORDER:
        p = a.root / m / "unified_summary.json"
        if p.exists():
            s = json.load(open(p))
            fold_agg[m] = s["metrics"]; groups[m] = s.get("groups") or {}; robust[m] = s.get("robust_canonicalizer")
        rows = _seq_rows(a.root / m, a.fold)
        if rows:
            seq_mean[m] = _seq_mean(rows)

    singles = {m: _seq_rows(a.root / m, a.fold) for m in ("single_front", "single_left", "single_right")}
    if all(singles.values()):
        keys = set.intersection(*[set(v.keys()) for v in singles.values()])
        best_rows = {}
        for k in keys:
            best = min(singles, key=lambda m: singles[m][k].get("mpjpe", float("inf")))
            best_rows[k] = singles[best][k]
        seq_mean["best_single"] = _seq_mean(best_rows)
        robust["best_single"] = robust.get("single_front")

    f = lambda d, k: f"{d[k]:.4f}" if k in d and d[k] == d[k] else "–"
    print(f"### seq-mean 口径（逐序列均值，n={next(iter(seq_mean.values()))['n'] if seq_mean else '?'}）")
    print("| 方法 | canonicalizer | MPJPE | PA-MPJPE | MPJPE(PA 同掩码) | PCK@0.10 |")
    print("|---|---|---|---|---|---|")
    for m in ORDER:
        if m in seq_mean:
            d = seq_mean[m]
            print(f"| {LABEL[m]} | {'robust' if robust.get(m) else 'non-robust'} | {f(d,'mpjpe')} | {f(d,'pa_mpjpe')} | {f(d,'mpjpe_pa_frames')} | {f(d,'pck@0.1')} |")
    print()
    print("### fold-agg 口径（eval_trifusion fold 级聚合）+ 关节分组")
    print("| 方法 | MPJPE | PA-MPJPE | MPJPE(PA 同掩码) | PCK@0.10 | head | shoulders_neck | body | hands |")
    print("|---|---|---|---|---|---|---|---|---|")
    for m in ORDER:
        if m in fold_agg:
            d = fold_agg[m]; g = groups.get(m, {})
            gg = lambda k: f"{g[k]:.3f}" if g.get(k) is not None else "–"
            print(f"| {LABEL[m]} | {f(d,'mpjpe')} | {f(d,'pa_mpjpe')} | {f(d,'mpjpe_pa_frames')} | {f(d,'pck@0.1')} | {gg('head')} | {gg('shoulders_neck')} | {gg('body')} | {gg('hands')} |")
    (a.root / "table3_unified.json").write_text(json.dumps({"seq_mean": seq_mean, "fold_agg": fold_agg, "groups": groups, "robust": robust}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
