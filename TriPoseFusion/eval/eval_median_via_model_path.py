#!/usr/bin/env python3
"""协议对照：用 eval_trifusion_pesudo_gt 的完全相同度量路径评估"三视角中位数融合"。

模型行与基线行的 GT canonicalize 实现不同（模型端 torch RobustCanonicalization vs
eval 脚本 numpy canonicalize_pose）。本脚本把模型 forward 替换为
P_final = median_v(P_views)（P_views 即模型端 canonicalize 后的各视角输入），
其余（GT canonicalize、掩码、指标、关节分组输出）与模型行逐位一致，
用来判断"手部误差差异"是模型能力还是协议差异。

用法：与 eval_trifusion_pesudo_gt.py 相同的 Hydra 覆盖（需 eval.ckpt_path 以构建模块）。
"""
from __future__ import annotations

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


@hydra.main(version_base=None, config_path="../configs", config_name="train.yaml")
def main(config: DictConfig) -> None:
    seed_everything(42, workers=True)
    output_dir = Path(str(base._cfg_get(config, "eval.output_dir", Path(config.log_path) / "median_protocol_check")))
    gt_root = Path(str(base._cfg_get(config, "eval.triangulated_gt_root", "/home/data/xchen/drive/sam3d_body_triangulated_gt"))).expanduser().resolve()
    pck_thresholds = [float(x) for x in base._cfg_get(config, "eval.pck_thresholds", [0.02, 0.05, 0.1])]
    ckpt = base._resolve_ckpt(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module = base._load_module(config, ckpt, device)

    orig_forward = module.model.forward

    def median_forward(pose3d, pose2d=None, **kw):
        out = orig_forward(pose3d=pose3d, pose2d=pose2d, **kw)
        out["P_final"] = out["P_views"].median(dim=3).values
        out["alpha"] = torch.full_like(out["alpha"], 1.0 / out["alpha"].shape[-1])
        return out

    module.model.forward = median_forward  # 实例属性覆盖，nn.Module.__call__ 会调用它
    logger.info("Model forward replaced by cross-view median of canonicalized inputs (robust=%s)",
                config.model.geofusion_use_robust_canonicalization)

    fold_dataset_idx = load_fold_dataset_idx_from_json(config)
    fold = base._selected_folds(config, fold_dataset_idx.keys())[0]
    split = base._selected_splits(config)[0]
    fold_metrics, _ = base._evaluate_fold(
        config=config, fold=fold, fold_dataset=fold_dataset_idx[fold], module=module, device=device,
        split=split, gt_root=gt_root, pck_thresholds=pck_thresholds, output_dir=output_dir,
    )
    keep = {k: round(float(v), 4) for k, v in fold_metrics.items() if k in ("mpjpe", "pa_mpjpe", "pck@0.1", "pck@0.10", "pck@0.05")}
    print("MEDIAN-VIA-MODEL-PATH", keep)
    jg = output_dir / f"joint_mpjpe_fold_{fold}.json"
    if jg.exists():
        import json
        groups = json.load(open(jg)).get("groups")
        print("GROUPS", groups)


if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"
    main()
