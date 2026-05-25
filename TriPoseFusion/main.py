#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict

import hydra
import torch
from omegaconf import DictConfig
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
)
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

from dataloader.data_loader import DriverKPTDataModule
from map_config import VideoSample
from trainer.train_triple_fusion import GeoFusionPoseTrainer

logger = logging.getLogger(__name__)


def _selected_train_fold(config: DictConfig) -> int:
    fold = getattr(config.train, "fold", None)
    if fold is None:
        raise ValueError("Missing train.fold in config. Set train.fold to an integer fold id.")
    if str(fold).lower() == "all":
        raise ValueError("train.fold must be a single integer when using per-fold JSON files.")
    return int(fold)


def _resolve_fold_index_json(config: DictConfig, fold: int) -> Path:
    """Return the JSON file for the configured training fold."""
    index_mapping = Path(config.paths.index_mapping)
    if index_mapping.is_file():
        return index_mapping

    index_name = str(getattr(config.paths, "index_file", "fold_{fold}.json") or "fold_{fold}.json")
    index_file = index_mapping / index_name.format(fold=fold)

    if not index_file.exists():
        raise FileNotFoundError(
            f"Fold index JSON not found: {index_file}. "
            f"Please generate fold {fold} beforehand or set paths.index_file."
        )
    return index_file


def _sample_from_json(item: Dict[str, Any]) -> VideoSample:
    sam3d_kpts = item.get("sam3d_kpts")
    label_path = item.get("label_path") or item.get("label")
    return VideoSample(
        person_id=str(item["person_id"]),
        env_folder=str(item["env_folder"]),
        env_key=str(item["env_key"]),
        videos={key: Path(value) for key, value in item.get("videos", {}).items()},
        label_path=Path(label_path) if label_path else None,
        sam3d_kpts={key: Path(value) for key, value in sam3d_kpts.items()}
        if sam3d_kpts
        else None,
    )


def load_fold_dataset_idx_from_json(
    config: DictConfig,
) -> Dict[int, Dict[str, list[VideoSample]]]:
    fold = _selected_train_fold(config)
    index_file = _resolve_fold_index_json(config, fold)
    logger.info("Loading fold %d dataset index from JSON: %s", fold, index_file)

    with open(index_file, "r", encoding="utf-8") as f:
        serial = json.load(f)

    if "train" in serial or "val" in serial:
        return {
            fold: {
                split: [_sample_from_json(item) for item in serial.get(split, [])]
                for split in ("train", "val")
            }
        }

    fold_key = str(fold)
    if fold_key not in serial:
        raise KeyError(f"Fold {fold} not found in aggregate index JSON: {index_file}")

    split_dict = serial[fold_key]
    return {
        fold: {
            split: [_sample_from_json(item) for item in split_dict.get(split, [])]
            for split in ("train", "val")
        }
    }


def build_module(hparams: DictConfig) -> GeoFusionPoseTrainer:
    if hparams.model.backbone != "triple_fusion":
        raise ValueError(
            f"Unsupported model.backbone={hparams.model.backbone!r}. "
            "This project entry currently supports only 'triple_fusion'."
        )
    return GeoFusionPoseTrainer(hparams)


def train_one_fold(
    hparams: DictConfig, dataset_idx: Dict[str, list[VideoSample]], fold: int
) -> None:
    seed_everything(42, workers=True)

    module = build_module(hparams)
    data_module = DriverKPTDataModule(hparams, dataset_idx)

    tb_logger = TensorBoardLogger(
        save_dir=os.path.join(hparams.log_path, "tb_logs"),
        name=f"fold_{fold}",
    )
    csv_logger = CSVLogger(
        save_dir=os.path.join(hparams.log_path, "csv_logs"),
        name=f"fold_{fold}",
    )

    checkpoint = ModelCheckpoint(
        dirpath=os.path.join(hparams.log_path, "checkpoints", f"fold_{fold}"),
        filename="{epoch}-{val/loss:.2f}",
        auto_insert_metric_name=False,
        monitor="val/loss",
        mode="min",
        save_last=True,
        save_top_k=2,
    )

    trainer = Trainer(
        devices=str(hparams.train.devices),
        accelerator="gpu",
        strategy="auto",
        max_epochs=hparams.train.max_epochs,
        logger=[tb_logger, csv_logger],
        check_val_every_n_epoch=1,
        callbacks=[
            RichProgressBar(leave=True),
            RichModelSummary(max_depth=3),
            checkpoint,
            LearningRateMonitor(logging_interval="step"),
        ],
    )

    trainer.fit(module, data_module)
    test_metrics = trainer.test(
        module, data_module, ckpt_path="best", weights_only=False
    )
    logger.info("Test metrics for fold %d: %s", fold, test_metrics)
    with open(
        os.path.join(tb_logger.log_dir, "test_metrics.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(test_metrics, f, indent=4)


@hydra.main(version_base=None, config_path="configs", config_name="train.yaml")
def init_params(config: DictConfig) -> None:
    fold_dataset_idx = load_fold_dataset_idx_from_json(config)

    logger.info("%s", "#" * 50)
    logger.info("Start training selected fold")
    logger.info("%s", "#" * 50)

    for fold, dataset_value in fold_dataset_idx.items():
        logger.info("%s", "#" * 50)
        logger.info("Start train fold: %s", fold)
        logger.info("%s", "#" * 50)

        train_one_fold(config, dataset_value, fold)

        logger.info("%s", "#" * 50)
        logger.info("Finish train fold: %s", fold)
        logger.info("%s", "#" * 50)

    logger.info("%s", "#" * 50)
    logger.info("Finished training selected fold")
    logger.info("%s", "#" * 50)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    os.environ["HYDRA_FULL_ERROR"] = "1"
    init_params()
