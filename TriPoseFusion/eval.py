#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from __future__ import annotations

import csv
import json
import logging
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import DeviceStatsMonitor, RichProgressBar
from pytorch_lightning.loggers import CSVLogger

from dataloader.data_loader import DriverKPTDataModule
from main import load_fold_dataset_idx_from_json
from trainer.train_triple_fusion import GeoFusionPoseTrainer

logger = logging.getLogger(__name__)


def _cfg_get(config: DictConfig, path: str, default: Any = None) -> Any:
    value = OmegaConf.select(config, path)
    return default if value is None else value


def _as_float(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value.detach().cpu().item())
        return value.detach().cpu().tolist()
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def _clean_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    return {key: _as_float(value) for key, value in metrics.items()}


def _parse_val_loss_from_name(path: Path) -> Optional[float]:
    """Parse val loss from names such as `12-0.34.ckpt` or `epoch=12-val/loss=0.34.ckpt`."""
    stem = path.stem
    for token in reversed(stem.replace("=", "-").split("-")):
        try:
            return float(token)
        except ValueError:
            continue
    return None


def _checkpoint_candidates(root: Path, fold: int) -> list[Path]:
    patterns = [
        root / "checkpoints" / f"fold_{fold}" / "*.ckpt",
        root / "**" / "checkpoints" / f"fold_{fold}" / "*.ckpt",
        root / f"fold_{fold}" / "**" / "*.ckpt",
    ]
    candidates: list[Path] = []
    for pattern in patterns:
        candidates.extend(Path(p) for p in root.glob(str(pattern.relative_to(root))))
    return sorted(set(candidates))


def _find_ckpt(config: DictConfig, fold: int) -> Optional[Path]:
    explicit = _cfg_get(config, "eval.ckpt_path")
    if explicit:
        ckpt = Path(str(explicit)).expanduser()
        if not ckpt.exists():
            raise FileNotFoundError(f"eval.ckpt_path does not exist: {ckpt}")
        return ckpt

    ckpt_dir = _cfg_get(config, "eval.ckpt_dir") or _cfg_get(config, "log_path")
    if not ckpt_dir:
        return None

    root = Path(str(ckpt_dir)).expanduser()
    if not root.exists():
        logger.warning("Checkpoint search directory does not exist: %s", root)
        return None

    candidates = _checkpoint_candidates(root, fold)
    if not candidates:
        logger.warning("No checkpoint found for fold %s under %s", fold, root)
        return None

    non_last = [p for p in candidates if p.name != "last.ckpt"]
    scored = [(p, _parse_val_loss_from_name(p)) for p in non_last]
    scored = [(p, loss) for p, loss in scored if loss is not None and math.isfinite(loss)]
    if scored:
        return min(scored, key=lambda item: item[1])[0]

    last = [p for p in candidates if p.name == "last.ckpt"]
    if last:
        return max(last, key=lambda p: p.stat().st_mtime)

    return max(candidates, key=lambda p: p.stat().st_mtime)


def _build_trainer(config: DictConfig) -> Trainer:
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    devices = [int(config.train.gpu)] if accelerator == "gpu" else 1
    output_dir = Path(str(_cfg_get(config, "eval.output_dir", Path(config.log_path) / "eval")))

    return Trainer(
        accelerator=accelerator,
        devices=devices,
        logger=CSVLogger(save_dir=str(output_dir), name="csv_logs"),
        callbacks=[RichProgressBar(leave=True), DeviceStatsMonitor()],
        enable_checkpointing=False,
    )


def _run_split(
    trainer: Trainer,
    module: GeoFusionPoseTrainer,
    data_module: DriverKPTDataModule,
    ckpt_path: Path,
    split: str,
) -> Dict[str, Any]:
    if split == "val":
        result = trainer.validate(module, datamodule=data_module, ckpt_path=str(ckpt_path))
    elif split == "test":
        result = trainer.test(module, datamodule=data_module, ckpt_path=str(ckpt_path))
    elif split == "train":
        data_module.setup("fit")
        result = trainer.validate(
            module,
            dataloaders=data_module.train_dataloader(),
            ckpt_path=str(ckpt_path),
        )
    else:
        raise ValueError(f"Unsupported eval.split={split!r}. Use train, val, or test.")

    merged: Dict[str, Any] = {}
    for item in result or []:
        merged.update(item)
    metrics = _clean_metrics(merged)
    if split == "train":
        metrics = {
            key.replace("val/", "train/", 1) if key.startswith("val/") else key: value
            for key, value in metrics.items()
        }
    return metrics


def _aggregate(per_fold: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    buckets: Dict[str, list[float]] = defaultdict(list)
    for metrics in per_fold.values():
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                buckets[key].append(float(value))

    aggregate: Dict[str, Dict[str, float]] = {}
    for key, values in buckets.items():
        if not values:
            continue
        mean = sum(values) / len(values)
        var = sum((value - mean) ** 2 for value in values) / len(values)
        aggregate[key] = {"mean": mean, "std": math.sqrt(var), "n": len(values)}
    return aggregate


def _save_results(
    output_dir: Path,
    per_fold: Dict[str, Dict[str, Any]],
    aggregate: Dict[str, Dict[str, float]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "eval_metrics.json"
    csv_path = output_dir / "eval_metrics.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"per_fold": per_fold, "aggregate": aggregate}, f, indent=2)

    metric_names = sorted({key for metrics in per_fold.values() for key in metrics})
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["fold"] + metric_names)
        for fold, metrics in sorted(per_fold.items(), key=lambda item: int(item[0])):
            writer.writerow([fold] + [metrics.get(name, "") for name in metric_names])
        writer.writerow([])
        writer.writerow(["metric", "mean", "std", "n"])
        for metric, stats in sorted(aggregate.items()):
            writer.writerow([metric, stats["mean"], stats["std"], stats["n"]])

    logger.info("Saved eval metrics JSON: %s", json_path)
    logger.info("Saved eval metrics CSV : %s", csv_path)


def _selected_folds(config: DictConfig, all_folds: Iterable[int]) -> list[int]:
    fold = _cfg_get(config, "eval.fold")
    if fold is None or str(fold).lower() == "all":
        return sorted(int(item) for item in all_folds)
    return [int(fold)]


@hydra.main(version_base=None, config_path="configs", config_name="train.yaml")
def main(config: DictConfig) -> None:
    seed_everything(42, workers=True)
    torch.set_float32_matmul_precision("high")

    split = str(_cfg_get(config, "eval.split", "val")).lower()
    output_dir = Path(str(_cfg_get(config, "eval.output_dir", Path(config.log_path) / "eval")))
    fold_dataset_idx = load_fold_dataset_idx_from_json(config)

    per_fold: Dict[str, Dict[str, Any]] = {}
    for fold in _selected_folds(config, fold_dataset_idx.keys()):
        if fold not in fold_dataset_idx:
            raise KeyError(f"Fold {fold} is not in the dataset index JSON.")

        ckpt = _find_ckpt(config, fold)
        if ckpt is None:
            raise FileNotFoundError(
                f"No checkpoint found for fold {fold}. Set eval.ckpt_path or eval.ckpt_dir."
            )

        logger.info("%s", "#" * 60)
        logger.info("Evaluating fold %s on %s split", fold, split)
        logger.info("Checkpoint: %s", ckpt)
        logger.info("%s", "#" * 60)

        module = GeoFusionPoseTrainer(config)
        data_module = DriverKPTDataModule(config, fold_dataset_idx[fold])
        trainer = _build_trainer(config)
        metrics = _run_split(trainer, module, data_module, ckpt, split)
        metrics["ckpt_path"] = str(ckpt)
        per_fold[str(fold)] = metrics

        numeric = {key: round(value, 6) for key, value in metrics.items() if isinstance(value, (int, float))}
        logger.info("Fold %s metrics: %s", fold, numeric)

    aggregate = _aggregate(per_fold)
    logger.info("%s", "#" * 60)
    logger.info("Aggregate metrics")
    for metric, stats in sorted(aggregate.items()):
        logger.info("%s: mean=%.6f, std=%.6f, n=%d", metric, stats["mean"], stats["std"], stats["n"])
    logger.info("%s", "#" * 60)

    _save_results(output_dir, per_fold, aggregate)


if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"
    main()
