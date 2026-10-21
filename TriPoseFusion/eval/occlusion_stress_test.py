#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""单视角遮挡压力测试（rebuttal C 点）。

对每个条件（clean + 依次"遮挡"每个视角），在验证集上跑推理并统计：
  - 各视角平均 gate 权重（含被遮挡视角上的权重）
  - MPJPE / PA-MPJPE 相对 clean 的变化
遮挡模式：zero（关键点置零）或 noise（加 σ=0.1m 高斯噪声）。

用法（与 eval_trifusion_pesudo_gt.py 相同的 Hydra 覆盖，另加）：
  +eval.occlusion_mode=zero +eval.stress_max_batches=0(不限)
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

import torch  # noqa: E402  (torch 必须先于 hydra 导入以保持 CUDA 检测)
import hydra  # noqa: E402
import numpy as np  # noqa: E402
from omegaconf import DictConfig  # noqa: E402
from pytorch_lightning import seed_everything  # noqa: E402
from tqdm.auto import tqdm  # noqa: E402

import eval_trifusion_pesudo_gt as base  # noqa: E402
from dataloader.data_loader import DriverKPTDataModule  # noqa: E402
from train import load_fold_dataset_idx_from_json  # noqa: E402

logger = logging.getLogger(__name__)


def _occlude(
    kpt: dict[str, torch.Tensor], view: str, mode: str
) -> dict[str, torch.Tensor]:
    out = dict(kpt)
    if mode == "zero":
        out[view] = torch.zeros_like(kpt[view])
    elif mode == "noise":
        out[view] = kpt[view] + 0.1 * torch.randn_like(kpt[view])
    else:
        raise ValueError(f"Unknown occlusion mode: {mode}")
    return out


def _run_condition(
    config: DictConfig,
    module,
    device: torch.device,
    fold_dataset,
    split: str,
    gt_root: Path,
    pck_thresholds: list[float],
    occlude_view: str | None,
    mode: str,
    max_batches: int,
) -> dict:
    data_module = DriverKPTDataModule(config, fold_dataset)
    dataloader = base._build_dataloader(data_module, split)
    sample_lookup = base._build_sample_lookup(
        fold_dataset["train" if split == "train" else "val"]
    )
    source_view = base._resolve_source_view(config)
    sequence_cache: dict = {}

    view_names = [str(v) for v in module.model.view_names]
    item_metrics: list[dict[str, float]] = []
    alpha_sum = torch.zeros(len(view_names), dtype=torch.float64)
    alpha_frames = 0
    skipped = 0

    label = f"occlude={occlude_view or 'clean'}"
    progress = tqdm(dataloader, desc=label, leave=True)
    with torch.no_grad():
        for batch_idx, batch in enumerate(progress):
            if max_batches and batch_idx >= max_batches:
                break
            pose3d = {k: v.to(device) for k, v in batch["sam3d_kpt_3d"].items()}
            pose2d = {k: v.to(device) for k, v in batch["sam3d_kpt_2d"].items()}
            if occlude_view is not None:
                pose3d = _occlude(pose3d, occlude_view, mode)
                pose2d = _occlude(pose2d, occlude_view, mode)
            out = module.model(pose3d=pose3d, pose2d=pose2d)
            pred = out["P_final"]

            for sample_idx, meta in enumerate(batch.get("meta", [])):
                person_id = str(meta["person_id"])
                env_folder = str(meta["env_folder"])
                key = (person_id, env_folder)
                if key not in sequence_cache:
                    sequence_cache[key] = base._load_triangulated_sequence(
                        gt_root, person_id, env_folder
                    )
                try:
                    sample = sample_lookup.get(key)
                    if sample is None:
                        raise ValueError(f"No sample mapping for {key}")
                    source_frame_ids = base._selected_source_frame_ids(
                        sample=sample,
                        start_frame=int(meta.get("start_frame", 0)),
                        end_frame=(
                            int(meta["end_frame"])
                            if meta.get("end_frame") is not None
                            else None
                        ),
                        target_t=int(pred.shape[1]),
                        source_view=source_view,
                    )
                    gt_kpt, gt_valid = base._select_gt_by_frame_ids(
                        tri_seq=sequence_cache[key],
                        source_frame_ids=source_frame_ids,
                    )
                except ValueError:
                    skipped += 1
                    continue

                gt_kpt, gt_valid = base._apply_joint_selection(
                    gt_kpt, gt_valid, pred.shape[2]
                )
                gt_tensor = torch.from_numpy(gt_kpt).to(device)
                gt_tensor = module.model._canonicalize_pose(
                    gt_tensor.unsqueeze(0)
                ).squeeze(0)
                valid_tensor = torch.from_numpy(gt_valid.astype(np.bool_)).to(device)
                metrics = base._compute_sample_metrics(
                    pred[sample_idx], gt_tensor, valid_tensor, pck_thresholds
                )
                if metrics:
                    item_metrics.append(metrics)
                    alpha = out["alpha"][sample_idx]  # (T, J, V)
                    alpha_sum += alpha.mean(dim=(0, 1)).double().cpu()
                    alpha_frames += 1
            progress.set_postfix(items=len(item_metrics), skipped=skipped)
    progress.close()

    merged = base._merge_metric_lists(item_metrics)
    mean_alpha = (
        (alpha_sum / max(alpha_frames, 1)).tolist() if alpha_frames else None
    )
    return {
        "condition": occlude_view or "clean",
        "mode": mode if occlude_view else None,
        "num_items": len(item_metrics),
        "skipped": skipped,
        "mpjpe": merged.get("mpjpe"),
        "pa_mpjpe": merged.get("pa_mpjpe"),
        "mean_gate_weights": (
            dict(zip(view_names, mean_alpha)) if mean_alpha else None
        ),
        "gate_on_occluded_view": (
            dict(zip(view_names, mean_alpha)).get(occlude_view)
            if (mean_alpha and occlude_view)
            else None
        ),
    }


@hydra.main(version_base=None, config_path="../configs", config_name="train.yaml")
def main(config: DictConfig) -> None:
    seed_everything(42, workers=True)
    torch.set_float32_matmul_precision("high")

    output_dir = Path(
        str(
            base._cfg_get(
                config, "eval.output_dir", Path(config.log_path) / "occlusion_stress"
            )
        )
    )
    gt_root = Path(
        str(
            base._cfg_get(
                config,
                "eval.triangulated_gt_root",
                "/home/data/xchen/drive/sam3d_body_triangulated_gt",
            )
        )
    ).expanduser().resolve()
    mode = str(base._cfg_get(config, "eval.occlusion_mode", "zero"))
    max_batches = int(base._cfg_get(config, "eval.stress_max_batches", 0))
    pck_thresholds = [
        float(x) for x in base._cfg_get(config, "eval.pck_thresholds", [0.02, 0.05, 0.1])
    ]

    ckpt = base._resolve_ckpt(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Occlusion stress test on device=%s mode=%s", device, mode)
    module = base._load_module(config, ckpt, device)

    fold_dataset_idx = load_fold_dataset_idx_from_json(config)
    folds = base._selected_folds(config, fold_dataset_idx.keys())
    fold = folds[0]
    split = base._selected_splits(config)[0]
    fold_dataset = fold_dataset_idx[fold]

    view_names = [str(v) for v in module.model.view_names]
    conditions: list[str | None] = [None] + view_names

    results = []
    for cond in conditions:
        results.append(
            _run_condition(
                config=config,
                module=module,
                device=device,
                fold_dataset=fold_dataset,
                split=split,
                gt_root=gt_root,
                pck_thresholds=pck_thresholds,
                occlude_view=cond,
                mode=mode,
                max_batches=max_batches,
            )
        )
        logger.info("condition done: %s", results[-1])

    clean = results[0]
    for r in results[1:]:
        if clean.get("mpjpe") and r.get("mpjpe"):
            r["delta_mpjpe_vs_clean"] = r["mpjpe"] - clean["mpjpe"]

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"occlusion_stress_fold{fold}_{split}_{mode}.json"
    payload = {
        "ckpt": str(ckpt),
        "fold": fold,
        "split": split,
        "mode": mode,
        "max_batches": max_batches,
        "view_names": view_names,
        "results": results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    logger.info("Saved: %s", out_path)

    print("\ncondition | mpjpe | pa_mpjpe | gate(front/left/right顺序=view_names)")
    for r in results:
        gw = r["mean_gate_weights"]
        gw_s = (
            "/".join(f"{gw[v]:.3f}" for v in view_names) if gw else "-"
        )
        print(
            f"{r['condition']:>8} | {r['mpjpe']:.4f} | {r['pa_mpjpe']:.4f} | {gw_s}"
            if r["mpjpe"] is not None
            else f"{r['condition']:>8} | no items"
        )


if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"
    main()
