#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from .kpt_dataset import KPTDataset


class DriverKPTDataModule(LightningDataModule):
    """LightningDataModule for synchronized multi-view SAM3D keypoints."""

    def __init__(self, opt, dataset_idx: Optional[Dict] = None) -> None:
        super().__init__()
        self.num_workers = int(opt.data.num_workers)
        self.batch_size = int(opt.data.batch_size)
        self.target_t = int(opt.data.uniform_temporal_subsample_num)
        self.dataset_idx = dataset_idx
        self.experiment = opt.experiment

        self.view_name = opt.train.view_name
        if isinstance(self.view_name, str):
            self.view_name = [self.view_name]

    def prepare_data(self) -> None:
        return None

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_gait_dataset = KPTDataset(
            experiment=self.experiment,
            index_mapping=self.dataset_idx["train"],
            view_name=self.view_name,
            target_t=self.target_t,
        )
        self.val_gait_dataset = KPTDataset(
            experiment=self.experiment,
            index_mapping=self.dataset_idx["val"],
            view_name=self.view_name,
            target_t=self.target_t,
        )
        self.test_gait_dataset = KPTDataset(
            experiment=self.experiment,
            index_mapping=self.dataset_idx["val"],
            view_name=self.view_name,
            target_t=self.target_t,
        )

    @staticmethod
    def _uniform_kpt_frames(tensor: torch.Tensor, target_t: int) -> torch.Tensor:
        if tensor.ndim != 3:
            raise ValueError(f"Expected kpt tensor shape (T,K,D), got {tuple(tensor.shape)}")

        frames, joints, dims = tensor.shape
        if target_t <= 0 or frames == target_t:
            return tensor
        if frames <= 0:
            return torch.zeros((target_t, joints, dims), dtype=tensor.dtype)
        if frames == 1:
            return tensor.repeat(target_t, 1, 1)

        indices = torch.linspace(0, frames - 1, steps=target_t).long()
        return tensor.index_select(0, indices)

    def _stack_view_dict(
        self,
        batch: list[Dict[str, Any]],
        key: str,
        target_t: int,
    ) -> Dict[str, Optional[torch.Tensor]]:
        by_view = {view: [] for view in self.view_name}
        for sample in batch:
            view_dict = sample.get(key)
            if not isinstance(view_dict, dict):
                continue
            for view in self.view_name:
                tensor = view_dict.get(view)
                if tensor is not None:
                    if target_t > 0 and tensor.shape[0] != target_t:
                        tensor = self._uniform_kpt_frames(tensor, target_t)
                    by_view[view].append(tensor)

        return {
            view: torch.stack(items, dim=0) if items else None
            for view, items in by_view.items()
        }

    def _collate_fn(self, batch: list[Dict[str, Any]]) -> Dict[str, Any]:
        if not batch:
            return {}

        labels = []
        label_info = []
        meta = []
        chunk_info = []

        for sample in batch:
            sample_labels = sample.get("label")
            segment_count = 1
            if sample_labels is not None:
                if sample_labels.ndim == 0:
                    sample_labels = sample_labels.view(1)
                labels.append(sample_labels)
                segment_count = int(sample_labels.shape[0])

            sample_label_info = sample.get("label_info")
            if sample_label_info:
                if isinstance(sample_label_info, list):
                    label_info.extend(sample_label_info)
                else:
                    label_info.append(sample_label_info)

            sample_meta = sample.get("meta")
            if sample_meta is not None:
                for segment_idx in range(segment_count):
                    meta_entry = dict(sample_meta)
                    meta_entry["segment_idx"] = segment_idx
                    meta_entry["segment_count"] = segment_count
                    meta.append(meta_entry)

                    if sample_meta.get("chunk_info") is not None:
                        chunk_entry = dict(sample_meta["chunk_info"])
                        chunk_entry["segment_idx"] = segment_idx
                        chunk_entry["segment_count"] = segment_count
                        chunk_info.append(chunk_entry)

        return {
            "sam3d_kpt_2d": self._stack_view_dict(batch, "sam3d_kpt_2d", self.target_t),
            "sam3d_kpt_3d": self._stack_view_dict(batch, "sam3d_kpt_3d", self.target_t),
            "label": torch.cat(labels, dim=0) if labels else None,
            "label_info": label_info,
            "meta": meta,
            "chunk_info": chunk_info,
        }

    def _make_loader(self, dataset, shuffle: bool, drop_last: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=shuffle,
            drop_last=drop_last,
            persistent_workers=self.num_workers > 0,
            collate_fn=self._collate_fn,
        )

    def train_dataloader(self) -> DataLoader:
        return self._make_loader(self.train_gait_dataset, shuffle=True, drop_last=True)

    def val_dataloader(self) -> DataLoader:
        return self._make_loader(self.val_gait_dataset, shuffle=False, drop_last=True)

    def test_dataloader(self) -> DataLoader:
        return self._make_loader(self.test_gait_dataset, shuffle=False, drop_last=True)
