#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from __future__ import annotations

import logging
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from map_config import KEEP_KEYPOINT_INDICES, VideoSample

logger = logging.getLogger(__name__)


class KPTDataset(Dataset):
    """
    Multi-view SAM3D keypoint dataset.

    Output:
        sample["sam3d_kpt_2d"][view] : Tensor (T, K, 2)
        sample["sam3d_kpt_3d"][view] : Tensor (T, K, 3)
        sample["video"][view]        : None  # reserved for rgb/rgb_kpt compatibility
        sample["label"]              : Optional[LongTensor]
        sample["meta"]               : dict
    """

    def __init__(
        self,
        experiment: str,
        index_mapping: List[VideoSample],
        transform: Any = None,
        view_name: Optional[List[str]] = None,
        target_t: Optional[int] = None,
    ) -> None:
        super().__init__()
        self._experiment = experiment
        self._index_mapping = index_mapping
        self._transform = transform
        self.view_name = view_name or ["front", "left", "right"]
        self.target_t = int(target_t) if target_t is not None and int(target_t) > 0 else None
        self.chunk_frames = self.target_t
        self.keep_keypoint_indices = (
            np.asarray(KEEP_KEYPOINT_INDICES, dtype=np.int64)
            if KEEP_KEYPOINT_INDICES is not None
            else None
        )

        self._kpt_cache: OrderedDict[Tuple[str, int, Optional[int]], Dict[str, torch.Tensor]] = OrderedDict()
        self._cache_max_size = 128
        self._file_list_cache: Dict[str, List[Path]] = {}

        self._chunked_index: List[Dict[str, Any]] = []
        if self.chunk_frames is not None:
            self._build_chunked_index()
            self._valid_source_indices = list(range(len(self._chunked_index)))
            logger.info(
                "SAM3D chunking enabled: %d samples -> %d chunks (chunk=%d frames)",
                len(self._index_mapping),
                len(self._chunked_index),
                int(self.chunk_frames),
            )
        else:
            self._valid_source_indices = list(range(len(self._index_mapping)))

    def __len__(self) -> int:
        return len(self._valid_source_indices)

    def _sorted_npz_files(self, kpt_dir: Path) -> List[Path]:
        cache_key = str(kpt_dir)
        if cache_key not in self._file_list_cache:
            self._file_list_cache[cache_key] = sorted(kpt_dir.glob("*_sam3d_body.npz"))
        return self._file_list_cache[cache_key]

    @staticmethod
    def _extract_frame_id(path: Path) -> str:
        stem = path.stem
        suffix = "_sam3d_body"
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
        return stem

    def _real_frame_interval_for_view(
        self,
        kpt_dir: Path,
        start_frame: int,
        end_frame: Optional[int],
    ) -> Dict[str, Any]:
        npz_files = self._sorted_npz_files(kpt_dir)
        start_idx = max(0, int(start_frame))
        end_idx = len(npz_files) if end_frame is None else min(len(npz_files), int(end_frame))
        if end_idx <= start_idx or len(npz_files) == 0:
            return {
                "start_idx": start_idx,
                "end_idx": end_idx,
                "start_frame_id": None,
                "end_frame_id": None,
                "num_frames": 0,
            }

        selected = npz_files[start_idx:end_idx]
        return {
            "start_idx": start_idx,
            "end_idx": end_idx,
            "start_frame_id": self._extract_frame_id(selected[0]),
            "end_frame_id": self._extract_frame_id(selected[-1]),
            "num_frames": len(selected),
        }

    def _count_frames_for_view(self, kpt_dir: Path) -> int:
        if not kpt_dir.exists():
            return 0
        return len(self._sorted_npz_files(kpt_dir))

    def _effective_frames(self, item: VideoSample) -> int:
        if item.sam3d_kpts is None:
            return 0

        counts: List[int] = []
        for view in self.view_name:
            kpt_dir = item.sam3d_kpts.get(view)
            if kpt_dir is None:
                continue
            cnt = self._count_frames_for_view(kpt_dir)
            if cnt > 0:
                counts.append(cnt)

        if len(counts) == 0:
            return 0

        return min(counts)

    def _build_chunked_index(self) -> None:
        if self.chunk_frames is None:
            return

        max_frames = int(self.chunk_frames)

        for item in self._index_mapping:
            total_frames = self._effective_frames(item)

            if total_frames <= 0:
                continue

            num_chunks = (total_frames + max_frames - 1) // max_frames
            for chunk_idx in range(num_chunks):
                chunk_start = chunk_idx * max_frames
                chunk_end = min(chunk_start + max_frames, total_frames)
                self._chunked_index.append(
                    {
                        "original_item": item,
                        "chunk_start_frame": chunk_start,
                        "chunk_end_frame": chunk_end,
                        "chunk_idx": chunk_idx,
                        "total_chunks": num_chunks,
                    }
                )

    def _load_one_view_kpts(
        self,
        kpt_dir: Path,
        start_frame: int,
        end_frame: Optional[int],
    ) -> Dict[str, torch.Tensor]:
        cache_key = (str(kpt_dir), start_frame, end_frame)
        if cache_key in self._kpt_cache:
            self._kpt_cache.move_to_end(cache_key)
            return self._kpt_cache[cache_key]

        npz_files = self._sorted_npz_files(kpt_dir)
        if len(npz_files) == 0:
            raise RuntimeError(f"No SAM3D npz found in {kpt_dir}")

        start_idx = max(0, int(start_frame))
        end_idx = len(npz_files) if end_frame is None else min(len(npz_files), int(end_frame))
        if end_idx <= start_idx:
            raise RuntimeError(
                f"Invalid frame range [{start_idx}, {end_idx}) for {kpt_dir} with {len(npz_files)} frames"
            )

        kpt2d_list: List[np.ndarray] = []
        kpt3d_list: List[np.ndarray] = []

        for path in npz_files[start_idx:end_idx]:
            with np.load(path, allow_pickle=True) as obj:
                if "output" not in obj:
                    raise RuntimeError(f"Missing 'output' key in {path}")

                output = obj["output"].item()
            if not isinstance(output, dict):
                raise RuntimeError(f"Invalid SAM3D output format in {path}")

            pred2d = output.get("pred_keypoints_2d")
            pred3d = output.get("pred_keypoints_3d")
            if pred2d is None or pred3d is None:
                raise RuntimeError(f"Missing pred_keypoints_2d/3d in {path}")

            pred2d = np.asarray(pred2d, dtype=np.float32)
            pred3d = np.asarray(pred3d, dtype=np.float32)

            if pred2d.ndim != 2 or pred2d.shape[1] != 2:
                raise RuntimeError(f"Invalid pred_keypoints_2d shape {pred2d.shape} in {path}")
            if pred3d.ndim != 2 or pred3d.shape[1] != 3:
                raise RuntimeError(f"Invalid pred_keypoints_3d shape {pred3d.shape} in {path}")

            k = min(pred2d.shape[0], pred3d.shape[0])
            if k <= 0:
                raise RuntimeError(f"Empty keypoints in {path}")

            if self.keep_keypoint_indices is not None:
                max_index = int(np.max(self.keep_keypoint_indices))
                if max_index >= k:
                    raise RuntimeError(
                        f"KEEP_KEYPOINT_INDICES requires keypoint index {max_index}, "
                        f"but {path} only has {k} aligned 2D/3D keypoints"
                    )
                pred2d = pred2d[self.keep_keypoint_indices]
                pred3d = pred3d[self.keep_keypoint_indices]
                k = len(self.keep_keypoint_indices)

            kpt2d_list.append(pred2d[:k])
            kpt3d_list.append(pred3d[:k])

        kpt2d = torch.from_numpy(np.stack(kpt2d_list, axis=0))
        kpt3d = torch.from_numpy(np.stack(kpt3d_list, axis=0))

        payload = {
            "kpt_2d": kpt2d,  # (T, K, 2)
            "kpt_3d": kpt3d,  # (T, K, 3)
        }

        self._kpt_cache[cache_key] = payload
        self._kpt_cache.move_to_end(cache_key)
        while len(self._kpt_cache) > self._cache_max_size:
            oldest_key = next(iter(self._kpt_cache))
            del self._kpt_cache[oldest_key]

        return payload

    @staticmethod
    def _uniform_temporal_sample(tensor: torch.Tensor, target_t: Optional[int]) -> torch.Tensor:
        if target_t is None:
            return tensor
        if tensor.ndim < 1:
            raise ValueError(f"Expected tensor with temporal dimension, got shape={tuple(tensor.shape)}")

        t = int(tensor.shape[0])
        if t == target_t:
            return tensor
        if t <= 0:
            raise ValueError("Empty temporal tensor cannot be sampled")
        if t == 1:
            repeat_shape = [target_t] + [1] * (tensor.ndim - 1)
            return tensor.repeat(*repeat_shape)

        idx = torch.linspace(0, t - 1, steps=target_t).long()
        return tensor.index_select(0, idx)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        source_index = self._valid_source_indices[index]

        if self.chunk_frames is not None:
            chunk_info = self._chunked_index[source_index]
            item = chunk_info["original_item"]
            start_frame = int(chunk_info["chunk_start_frame"])
            end_frame = int(chunk_info["chunk_end_frame"])
        else:
            chunk_info = None
            item = self._index_mapping[source_index]
            start_frame = 0
            end_frame = None

        if item.sam3d_kpts is None:
            raise RuntimeError(
                f"sam3d_kpts is missing for person={item.person_id}, env={item.env_folder}"
            )

        missing_views = [v for v in self.view_name if v not in item.sam3d_kpts]
        if missing_views:
            raise RuntimeError(
                f"Missing requested sam3d views {missing_views} for person={item.person_id}, env={item.env_folder}"
            )

        sam3d_kpt_2d: Dict[str, torch.Tensor] = {}
        sam3d_kpt_3d: Dict[str, torch.Tensor] = {}
        sam3d_real_frame_interval: Dict[str, Dict[str, Any]] = {}

        for view in self.view_name:
            sam3d_real_frame_interval[view] = self._real_frame_interval_for_view(
                item.sam3d_kpts[view],
                start_frame,
                end_frame,
            )
            payload = self._load_one_view_kpts(item.sam3d_kpts[view], start_frame, end_frame)
            sam3d_kpt_2d[view] = self._uniform_temporal_sample(payload["kpt_2d"], self.target_t)
            sam3d_kpt_3d[view] = self._uniform_temporal_sample(payload["kpt_3d"], self.target_t)

        return {
            "sam3d_kpt_2d": sam3d_kpt_2d,
            "sam3d_kpt_3d": sam3d_kpt_3d,
            "meta": {
                "experiment": self._experiment,
                "index": source_index,
                "person_id": item.person_id,
                "env_folder": item.env_folder,
                "env_key": item.env_key,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "sam3d_real_frame_interval": sam3d_real_frame_interval,
                "is_chunked": self.chunk_frames is not None,
                "chunk_info": {
                    "chunk_idx": chunk_info["chunk_idx"],
                    "total_chunks": chunk_info["total_chunks"],
                    "chunk_start_frame": start_frame,
                    "chunk_end_frame": end_frame,
                }
                if chunk_info is not None
                else None,
            },
        }
