#!/usr/bin/env python3
"""Table 3 统一口径评估：所有方法走 eval_trifusion_pesudo_gt 的同一度量路径。

GT canonicalize（模型端 canonicalizer）、掩码、Procrustes、同掩码 MPJPE、关节分组
对所有行完全一致，只替换"预测"的来源：
  +eval.method=model          TriPoseFusion（eval.ckpt_path）
  +eval.method=mean|median    canonicalized 视角均值 / 中位数
  +eval.method=single_front|single_left|single_right
  +eval.method=learned        E 点学习型融合基线（+eval.learned_ckpt=...pt）
best-single 由 table3_unified_summary.py 从三个 single 行按序列取最优得到。
其余 Hydra 覆盖与 eval_trifusion_pesudo_gt.py 相同（需 eval.ckpt_path 以构建模块，
canonicalizer 配置由 model.geofusion_use_robust_canonicalization 决定并写入输出）。
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
EVAL_DIR = Path(__file__).resolve().parent
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

import torch  # noqa: E402
import hydra  # noqa: E402
from omegaconf import DictConfig  # noqa: E402
from pytorch_lightning import seed_everything  # noqa: E402

import eval_trifusion_pesudo_gt as base  # noqa: E402
from train import load_fold_dataset_idx_from_json  # noqa: E402

logger = logging.getLogger(__name__)


def _make_forward(module, method: str, learned=None):
    orig = module.model.forward
    view_names = [str(v) for v in module.model.view_names]

    def fwd(pose3d, pose2d=None, **kw):
        out = orig(pose3d=pose3d, pose2d=pose2d, **kw)
        pv = out["P_views"]  # (B,T,J,V,3) canonicalized inputs
        if method == "model":
            return out
        if method == "mean":
            out["P_final"] = pv.mean(dim=3)
        elif method == "median":
            out["P_final"] = pv.median(dim=3).values
        elif method.startswith("single_"):
            out["P_final"] = pv[:, :, :, view_names.index(method.split("_", 1)[1])]
        elif method == "learned":
            conf = torch.ones(pv.shape[:-1], device=pv.device, dtype=pv.dtype)
            out["P_final"] = learned(pv, conf)[0]
        else:
            raise ValueError(f"unknown method {method}")
        out["alpha"] = torch.full_like(out["alpha"], 1.0 / out["alpha"].shape[-1])
        return out

    return fwd


@hydra.main(version_base=None, config_path="../configs", config_name="train.yaml")
def main(config: DictConfig) -> None:
    seed_everything(42, workers=True)
    method = str(base._cfg_get(config, "eval.method", "model"))
    output_dir = Path(str(base._cfg_get(config, "eval.output_dir", Path(config.log_path) / "table3_unified"))) / method
    gt_root = Path(str(base._cfg_get(config, "eval.triangulated_gt_root", "/home/data/xchen/drive/sam3d_body_triangulated_gt"))).expanduser().resolve()
    pck_thresholds = [float(x) for x in base._cfg_get(config, "eval.pck_thresholds", [0.02, 0.05, 0.1])]
    ckpt = base._resolve_ckpt(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module = base._load_module(config, ckpt, device)

    learned = None
    if method == "learned":
        from learned_fusion_baseline import LearnedFusion  # noqa: E402

        learned_ckpt = str(base._cfg_get(config, "eval.learned_ckpt", ""))
        if not learned_ckpt:
            raise ValueError("+eval.learned_ckpt is required for method=learned")
        learned = LearnedFusion()
        learned.load_state_dict(torch.load(learned_ckpt, map_location="cpu"))
        learned.to(device).eval()

    module.model.forward = _make_forward(module, method, learned)
    robust = bool(config.model.geofusion_use_robust_canonicalization)
    logger.info("Unified Table 3 eval: method=%s robust_canonicalizer=%s", method, robust)

    fold_dataset_idx = load_fold_dataset_idx_from_json(config)
    fold = base._selected_folds(config, fold_dataset_idx.keys())[0]
    split = base._selected_splits(config)[0]
    fold_metrics, _ = base._evaluate_fold(
        config=config, fold=fold, fold_dataset=fold_dataset_idx[fold], module=module, device=device,
        split=split, gt_root=gt_root, pck_thresholds=pck_thresholds, output_dir=output_dir,
    )
    summary = {
        "method": method,
        "robust_canonicalizer": robust,
        "ckpt": str(ckpt),
        "fold": fold,
        "split": split,
        "metrics": {k: float(v) for k, v in fold_metrics.items()},
    }
    jg = output_dir / f"joint_mpjpe_fold_{fold}.json"
    if jg.exists():
        summary["groups"] = {k: v.get("mpjpe") for k, v in json.load(open(jg)).get("groups", {}).items()}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "unified_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print("UNIFIED", json.dumps({k: summary[k] for k in ("method", "robust_canonicalizer")}),
          {k: round(v, 4) for k, v in summary["metrics"].items() if k in ("mpjpe", "pa_mpjpe", "mpjpe_pa_frames", "pck@0.05", "pck@0.1")},
          summary.get("groups"))


if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"
    main()
