#!/usr/bin/env python3
"""从 learned_fusion_fold{N}.json 的 per_sequence.per_joint_mpjpe_m 聚合关节分组 MPJPE。

分组（52 关节模型空间，与另一会话的基线分组一致）：
  head 0-4, shoulders_neck [5,6,49,50,51], body = 0-6 + 49-51, hands 7-48
口径：逐序列先对组内关节取 nan 均值，再对序列取均值。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

GROUPS = {
    "head": list(range(0, 5)),
    "shoulders_neck": [5, 6, 49, 50, 51],
    "body": list(range(0, 7)) + [49, 50, 51],
    "hands": list(range(7, 49)),
}


def _per_joint_array(metrics: dict) -> np.ndarray:
    pj = metrics.get("per_joint_mpjpe_m")
    if pj is None:
        raise KeyError("per_joint_mpjpe_m missing in metrics")
    if isinstance(pj, dict):
        n = max(int(k) for k in pj.keys()) + 1
        arr = np.full(n, np.nan)
        for k, v in pj.items():
            arr[int(k)] = np.nan if v is None else float(v)
        return arr
    return np.asarray([np.nan if v is None else float(v) for v in pj], dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=Path(__file__).resolve().parent / "logs" / "learned_fusion_baseline" / "learned_fusion_fold0.json",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    d = json.load(open(args.results, encoding="utf-8"))
    rows = d["per_sequence"]
    per_seq = {g: [] for g in GROUPS}
    for r in rows:
        arr = _per_joint_array(r["metrics"])
        for g, idx in GROUPS.items():
            vals = arr[[i for i in idx if i < arr.shape[0]]]
            if np.isfinite(vals).any():
                per_seq[g].append(float(np.nanmean(vals)))
    summary = {
        g: {"mpjpe_m": float(np.mean(v)), "num_sequences": len(v)} for g, v in per_seq.items()
    }
    out = {
        "source": str(args.results),
        "method": "learned_fusion_baseline",
        "aggregation": "per-sequence nanmean over group joints, then mean over sequences",
        "groups": {g: idx for g, idx in GROUPS.items()},
        "summary": summary,
    }
    out_path = args.output or args.results.with_name(args.results.stem + "_joint_groups.json")
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    for g, s in summary.items():
        print(f"{g:15s} MPJPE {s['mpjpe_m']:.4f}  (n={s['num_sequences']})")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
