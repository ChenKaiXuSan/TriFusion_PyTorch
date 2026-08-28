#!/usr/bin/env python3
"""汇总 eval_table3_unified.py 各方法输出为一张 Markdown 表（含 best-single 行）。

best-single：对三个 single_* 行的 person_env_metrics_fold_{fold}.csv，按序列取 MPJPE 最小
视角，再对序列取均值（oracle）。其余行直接读 unified_summary.json（fold 级聚合）。
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ORDER = ["single_front", "single_left", "single_right", "best_single", "mean", "median", "learned", "model"]
LABEL = {"single_front": "single front", "single_left": "single left", "single_right": "single right",
         "best_single": "best single (oracle, per-sequence)", "mean": "fuse mean (= uniform gate)",
         "median": "fuse median", "learned": "learned fusion baseline (1.9K)", "model": "TriPoseFusion"}


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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True, help="eval_table3_unified 的 eval.output_dir")
    ap.add_argument("--fold", type=int, default=0)
    a = ap.parse_args()

    summ = {}
    for m in ORDER:
        p = a.root / m / "unified_summary.json"
        if p.exists():
            summ[m] = json.load(open(p))

    # best-single（oracle）
    singles = {m: _seq_rows(a.root / m, a.fold) for m in ("single_front", "single_left", "single_right") if (a.root / m).exists()}
    if len(singles) == 3 and all(singles.values()):
        keys = set.intersection(*[set(v.keys()) for v in singles.values()])
        agg = {"mpjpe": [], "pa_mpjpe": [], "mpjpe_pa_frames": [], "pck@0.1": []}
        for k in keys:
            best = min(singles, key=lambda m: singles[m][k].get("mpjpe", float("inf")))
            for metric in agg:
                if metric in singles[best][k]:
                    agg[metric].append(singles[best][k][metric])
        summ["best_single"] = {"method": "best_single", "robust_canonicalizer": summ.get("single_front", {}).get("robust_canonicalizer"),
                               "metrics": {m: sum(v) / len(v) for m, v in agg.items() if v}, "note": f"per-sequence oracle over {len(keys)} sequences (sequence-mean aggregation)"}

    print("| 方法 | canonicalizer | MPJPE | PA-MPJPE | MPJPE(PA 同掩码) | PCK@0.10 | body | hands |")
    print("|---|---|---|---|---|---|---|---|")
    for m in ORDER:
        s = summ.get(m)
        if not s:
            continue
        mt = s["metrics"]; g = s.get("groups") or {}
        rc = "robust" if s.get("robust_canonicalizer") else "non-robust"
        f = lambda k: f"{mt[k]:.4f}" if k in mt and mt[k] == mt[k] else "–"
        gb = f"{g['body']:.3f}" if g.get("body") is not None else "–"
        gh = f"{g['hands']:.3f}" if g.get("hands") is not None else "–"
        print(f"| {LABEL[m]} | {rc} | {f('mpjpe')} | {f('pa_mpjpe')} | {f('mpjpe_pa_frames')} | {f('pck@0.1')} | {gb} | {gh} |")
    (a.root / "table3_unified.json").write_text(json.dumps(summ, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
