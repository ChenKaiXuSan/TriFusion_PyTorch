#!/usr/bin/env python3
"""伪 GT（三角化）可靠性统计，供 rebuttal F 点使用。

遍历 triangulated GT 根目录下所有 <person>/<env>/keypoints_3d.npz，输出：
  - 序列级：帧数、valid_ratio、平均重投影误差
  - 关节级（52 保留关节）：有效率、重投影误差分位数
  - 关节组级（head / shoulder_neck / hands）：同上聚合
结果写 JSON + Markdown 表格。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "TriPoseFusion"))

from map_config import KEEP_KEYPOINT_INDICES  # noqa: E402

# 70 关节空间中的分组（与 map_config 注释一致）
GROUPS_70 = {
    "head": list(range(0, 5)),                # nose, eyes, ears
    "shoulder_neck": [5, 6, 67, 68, 69],      # shoulders, acromions, neck
    "right_hand": list(range(21, 42)),
    "left_hand": list(range(42, 63)),
}


def joint_group(idx70: int) -> str:
    for name, members in GROUPS_70.items():
        if idx70 in members:
            return name
    return "other"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gt-root",
        type=Path,
        default=Path("/work/1/SKIING/chenkaixu/data/drive/sam3d_body_triangulated_gt"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "TriPoseFusion" / "eval" / "logs" / "pseudo_gt_reliability",
    )
    args = parser.parse_args()

    npz_paths = sorted(args.gt_root.glob("*/*/keypoints_3d.npz"))
    if not npz_paths:
        raise SystemExit(f"no keypoints_3d.npz under {args.gt_root}")

    keep = np.asarray(KEEP_KEYPOINT_INDICES, dtype=np.int64)
    n_keep = len(keep)

    total_frames = 0
    valid_count = np.zeros(n_keep, dtype=np.int64)
    # 重投影误差按关节累积（仅 valid 帧）
    err_sum = np.zeros(n_keep, dtype=np.float64)
    err_sq_sum = np.zeros(n_keep, dtype=np.float64)
    # 全体 valid 误差样本的分位数：用直方图近似成本高，直接下采样收集
    err_samples: list[np.ndarray] = []
    seq_rows = []

    for p in npz_paths:
        d = np.load(p, allow_pickle=True)
        vm = d["valid_mask"][:, keep]          # (T, 52)
        re = d["reproj_error"][:, keep]        # (T, 52)
        t = vm.shape[0]
        total_frames += t
        valid_count += vm.sum(axis=0)
        re_valid = np.where(vm, re, 0.0)
        err_sum += re_valid.sum(axis=0)
        err_sq_sum += (re_valid**2).sum(axis=0)
        flat = re[vm]
        if flat.size:
            step = max(1, flat.size // 20000)
            err_samples.append(flat[::step].astype(np.float64))
        seq_rows.append(
            {
                "person": p.parent.parent.name,
                "env": p.parent.name,
                "frames": int(t),
                "valid_ratio": float(vm.mean()),
                "mean_reproj_px": float(flat.mean()) if flat.size else None,
            }
        )

    valid_rate = valid_count / max(total_frames, 1)
    mean_err = np.divide(err_sum, valid_count, out=np.full(n_keep, np.nan), where=valid_count > 0)

    groups = [joint_group(int(i)) for i in keep]
    group_stats = {}
    for g in ["head", "shoulder_neck", "right_hand", "left_hand"]:
        sel = np.asarray([i for i, name in enumerate(groups) if name == g])
        gvalid = valid_count[sel].sum() / (max(total_frames, 1) * len(sel))
        gerr = err_sum[sel].sum() / max(valid_count[sel].sum(), 1)
        group_stats[g] = {
            "n_joints": int(len(sel)),
            "valid_rate": float(gvalid),
            "mean_reproj_px": float(gerr),
        }
    hands_valid = (
        valid_count[[i for i, g in enumerate(groups) if g.endswith("hand")]].sum()
        / (max(total_frames, 1) * sum(1 for g in groups if g.endswith("hand")))
    )
    body_idx = [i for i, g in enumerate(groups) if not g.endswith("hand")]
    body_valid = valid_count[body_idx].sum() / (max(total_frames, 1) * len(body_idx))

    all_err = np.concatenate(err_samples)
    percentiles = {
        f"p{q}": float(np.percentile(all_err, q)) for q in (50, 75, 90, 95, 99)
    }

    per_joint = [
        {
            "idx70": int(keep[i]),
            "idx52": i,
            "group": groups[i],
            "valid_rate": float(valid_rate[i]),
            "mean_reproj_px": None if np.isnan(mean_err[i]) else float(mean_err[i]),
        }
        for i in range(n_keep)
    ]

    summary = {
        "n_sequences": len(npz_paths),
        "total_frames": int(total_frames),
        "overall_mean_reproj_px": float(all_err.mean()),
        "reproj_percentiles_px": percentiles,
        "overall_valid_rate_52": float(valid_count.sum() / (total_frames * n_keep)),
        "body_head_valid_rate": float(body_valid),
        "hands_valid_rate": float(hands_valid),
        "group_stats": group_stats,
        "per_joint": per_joint,
        "sequences": seq_rows,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.output_dir / "pseudo_gt_reliability.json"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    md = ["# 伪 GT 可靠性统计（rebuttal F 点）", ""]
    md.append(f"- 序列数: {len(npz_paths)}，总帧数: {total_frames}")
    md.append(f"- 全体 52 关节有效率: {summary['overall_valid_rate_52']:.3f}")
    md.append(f"- 头/肩颈 10 关节有效率: {body_valid:.3f}，手部 42 关节有效率: {hands_valid:.3f}")
    md.append(
        f"- 重投影误差 (px, valid 关节): mean {all_err.mean():.2f}, "
        + ", ".join(f"{k}={v:.2f}" for k, v in percentiles.items())
    )
    md.append("")
    md.append("| 关节组 | 关节数 | 有效率 | 平均重投影误差(px) |")
    md.append("|---|---|---|---|")
    for g, s in group_stats.items():
        md.append(f"| {g} | {s['n_joints']} | {s['valid_rate']:.3f} | {s['mean_reproj_px']:.2f} |")
    (args.output_dir / "pseudo_gt_reliability.md").write_text("\n".join(md) + "\n")

    print(f"wrote {out_json}")
    print("\n".join(md))


if __name__ == "__main__":
    main()
