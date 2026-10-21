#!/usr/bin/env python3
"""只读统计：伪 GT 三角化中各视角的参与比例与逐视角重投影误差（52 关节子集）。

回应 LAYQ L9 / xS2o X6 关于 Table 2 各视角 valid 数极不均衡（前视 1.40M vs 左右 ~16M）
与前视 P95 42 px 的疑问。不修改任何 GT 文件。
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

GROUPS = {"head": list(range(0, 5)), "shoulders_neck": [5, 6, 49, 50, 51], "hands": list(range(7, 49))}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gt-root", type=Path, default=Path("/work/1/SKIING/chenkaixu/data/drive/sam3d_body_triangulated_gt"))
    p.add_argument("--output", type=Path, default=REPO_ROOT / "TriPoseFusion" / "eval" / "logs" / "pseudo_gt_reliability" / "view_participation.json")
    a = p.parse_args()
    keep = np.asarray(KEEP_KEYPOINT_INDICES)
    views = None
    part = None  # (52, V) 参与计数（valid 且该视角残差有限）
    valid_total = np.zeros(52)
    err_sum = None
    err_samples = []
    for f in sorted(a.gt_root.glob("*/*/keypoints_3d.npz")):
        z = np.load(f, allow_pickle=True)
        if views is None:
            views = [str(v) for v in z["views"]]
            part = np.zeros((52, len(views)))
            err_sum = np.zeros((52, len(views)))
        vm = z["valid_mask"][:, keep]  # (T,52)
        rpv = z["reproj_per_view"][:, keep]  # (T,52,V)
        used = np.isfinite(rpv) & vm[:, :, None]
        part += used.sum(0)
        valid_total += vm.sum(0)
        err_sum += np.where(used, rpv, 0.0).sum(0)
        flat = rpv[used]
        if flat.size:
            step = max(1, flat.size // 5000)
            err_samples.append(np.stack([np.where(used, rpv, np.nan)[:, :, v][used[:, :, v]][::step] for v in range(len(views))], axis=0) if False else flat[::step])
    rate = part / np.maximum(valid_total[:, None], 1)
    mean_err = err_sum / np.maximum(part, 1)
    out = {"views": views, "total_valid_joint_frames": float(valid_total.sum()),
           "participation_count": {v: float(part[:, i].sum()) for i, v in enumerate(views)},
           "participation_rate_over_valid": {v: float(part[:, i].sum() / max(valid_total.sum(), 1)) for i, v in enumerate(views)},
           "mean_reproj_px_per_view": {v: float(err_sum[:, i].sum() / max(part[:, i].sum(), 1)) for i, v in enumerate(views)},
           "per_group": {}}
    for g, idx in GROUPS.items():
        out["per_group"][g] = {v: {"participation_rate": float(part[idx, i].sum() / max(valid_total[idx].sum(), 1)),
                                    "mean_reproj_px": float(err_sum[idx, i].sum() / max(part[idx, i].sum(), 1))}
                               for i, v in enumerate(views)}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps({k: out[k] for k in ("participation_count", "participation_rate_over_valid", "mean_reproj_px_per_view")}, indent=1))
    for g, d in out["per_group"].items():
        print(g, {v: (round(x["participation_rate"], 3), round(x["mean_reproj_px"], 1)) for v, x in d.items()})


if __name__ == "__main__":
    main()
