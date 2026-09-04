#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare the STEP1 head-movement results obtained from different keypoint sources.

Reads <root>/<source>/summary_by_sequence.csv for every source folder present and writes
  <root>/method_comparison.csv   per-source means of the key measures + per-drive agreement with fused_v3
  <root>/method_comparison.md    plain-language table
  <root>/figures_method_comparison/*.png
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

SOURCES = ["fused_v3", "fused_mean", "fused_median", "view_front", "view_left", "view_right"]
LABEL = {"fused_v3": "v3 融合", "fused_mean": "均值融合", "fused_median": "中位数融合",
         "view_front": "单视角 front", "view_left": "单视角 left", "view_right": "单视角 right"}
KEYS = ["ai_horizontal_turns_per_min", "ai_horizontal_peak_deg_mean", "ai_vertical_turns_per_min",
        "ai_horizontal_turn_duration_s_mean", "yaw_p95_abs_deg", "annotated_glances_with_head_turn_ge15",
        "yaw_keypoint_vs_rotation_corr"]
NAMES = {"ai_horizontal_turns_per_min": "水平转头 次/分", "ai_horizontal_peak_deg_mean": "水平峰值角 (度)",
         "ai_vertical_turns_per_min": "垂直转头 次/分", "ai_horizontal_turn_duration_s_mean": "每次持续 (秒)",
         "yaw_p95_abs_deg": "yaw 95 分位 (度)", "annotated_glances_with_head_turn_ge15": "瞥视中伴随≥15°转头比例",
         "yaw_keypoint_vs_rotation_corr": "与旋转矩阵核对相关"}


def load(root: Path, src: str) -> dict:
    p = root / src / "summary_by_sequence.csv"
    if not p.exists():
        return {}
    out = {}
    for r in csv.DictReader(open(p, encoding="utf-8")):
        out[r["sequence"]] = {k: (float(r[k]) if r.get(k) not in (None, "", "None") else np.nan) for k in KEYS}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path,
                    default=Path("/work/1/SKIING/chenkaixu/data/drive/downstream_analysis/head_movement_analysis"))
    args = ap.parse_args()
    data = {s: load(args.root, s) for s in SOURCES}
    data = {s: d for s, d in data.items() if d}
    ref = data.get("fused_v3", {})
    seqs = sorted(set.intersection(*[set(d) for d in data.values()])) if data else []

    rows = []
    for s, d in data.items():
        row = {"source": s, "label": LABEL[s], "n_sequences": len(d)}
        for k in KEYS:
            row[f"mean_{k}"] = round(float(np.nanmean([d[q][k] for q in d])), 2)
        # agreement with v3 across drives (Spearman) for the two hypothesis measures
        for k in ("ai_horizontal_turns_per_min", "ai_horizontal_peak_deg_mean"):
            if ref and s != "fused_v3":
                a = np.array([d[q][k] for q in seqs]); b = np.array([ref[q][k] for q in seqs])
                m = np.isfinite(a) & np.isfinite(b)
                from scipy.stats import spearmanr
                row[f"spearman_vs_v3_{k}"] = round(float(spearmanr(a[m], b[m]).correlation), 3) if m.sum() > 5 else None
            else:
                row[f"spearman_vs_v3_{k}"] = 1.0 if s == "fused_v3" else None
        rows.append(row)

    cols = list(rows[0].keys())
    with open(args.root / "method_comparison.csv", "w", encoding="utf-8") as fh:
        fh.write(",".join(cols) + "\n")
        for r in rows:
            fh.write(",".join("" if r[c] is None else str(r[c]) for c in cols) + "\n")

    with open(args.root / "method_comparison.md", "w", encoding="utf-8") as fh:
        fh.write("# 不同关节点来源的头部运动结果对比（88 段驾驶）\n\n")
        fh.write("同一分析流程、同一阈值，只换头部关节点的来源。所有来源都在同一 canonical 身体坐标系中"
                 "（单视角 = 该视角 SAM3D 关节点经自身 canonicalize；均值/中位数 = 三视角 canonicalize 后融合）。\n\n")
        fh.write("| 来源 | " + " | ".join(NAMES[k] for k in KEYS) + " | 与 v3 的一致性（转头频率 / 峰值角，Spearman） |\n")
        fh.write("|---|" + "---|" * (len(KEYS) + 1) + "\n")
        for r in rows:
            fh.write(f"| {r['label']} | " + " | ".join(
                "" if np.isnan(r[f"mean_{k}"]) else f"{r[f'mean_{k}']:.2f}" for k in KEYS) +
                f" | {r['spearman_vs_v3_ai_horizontal_turns_per_min']} / {r['spearman_vs_v3_ai_horizontal_peak_deg_mean']} |\n")
        fh.write("\n说明：\n- 数值为 88 段的平均；每段的数字在各来源子目录的 `summary_by_sequence.csv`。\n"
                 "- \"与 v3 的一致性\"：按段比较两种来源给出的转头频率/峰值角的排序相关（1 = 完全一致）。\n"
                 "- \"与旋转矩阵核对相关\"只在已生成核对信号的段上有值（提取完成后填满）。\n")

    # figure: per-drive AI horizontal turns/min, each source vs v3
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        (args.root / "figures_method_comparison").mkdir(exist_ok=True)
        others = [s for s in data if s != "fused_v3"]
        fig, axes = plt.subplots(1, len(others), figsize=(3.6 * len(others), 3.6))
        for ax, s in zip(np.atleast_1d(axes), others):
            a = [data[s][q]["ai_horizontal_turns_per_min"] for q in seqs]
            b = [ref[q]["ai_horizontal_turns_per_min"] for q in seqs]
            ax.scatter(b, a, s=12); lim = max(max(a), max(b)) * 1.05
            ax.plot([0, lim], [0, lim], "k--", lw=0.7); ax.set_xlabel("fused_v3 turns/min"); ax.set_ylabel(f"{s} turns/min")
            ax.set_title(s)
        fig.suptitle("AI horizontal head turns per minute: each source vs fused_v3 (one point = one drive)")
        fig.tight_layout(); fig.savefig(args.root / "figures_method_comparison" / "turns_per_min_vs_v3.png", dpi=150)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(9, 4))
        x = np.arange(len(rows))
        ax.bar(x - 0.2, [r["mean_ai_horizontal_turns_per_min"] for r in rows], 0.4, label="horizontal turns / min")
        ax2 = ax.twinx()
        ax2.bar(x + 0.2, [r["mean_ai_horizontal_peak_deg_mean"] for r in rows], 0.4, color="tab:orange", label="mean peak yaw (deg)")
        ax.set_xticks(x); ax.set_xticklabels([r["source"] for r in rows], rotation=15)
        ax.set_ylabel("turns / min"); ax2.set_ylabel("deg"); ax.set_title("Mean over 88 drives by keypoint source")
        ax.legend(loc="upper left"); ax2.legend(loc="upper right"); fig.tight_layout()
        fig.savefig(args.root / "figures_method_comparison" / "means_by_source.png", dpi=150); plt.close(fig)
    except Exception as e:
        print("figures skipped:", e)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
