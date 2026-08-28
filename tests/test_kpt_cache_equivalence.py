"""KPTDataset 关键点数组缓存与逐文件读取必须逐位等价。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "TriPoseFusion"))

from dataloader.build_kpt_cache import build_view  # noqa: E402
from dataloader.kpt_dataset import KPTDataset  # noqa: E402
from map_config import VideoSample  # noqa: E402


def _make_sam3d_dir(root: Path, n_frames: int, views=("front", "left", "right")) -> Path:
    rng = np.random.default_rng(0)
    for v in views:
        vd = root / "01" / "昼多い" / v
        vd.mkdir(parents=True)
        for i in range(n_frames):
            out = {
                "pred_keypoints_2d": rng.normal(size=(70, 2)).astype(np.float32),
                "pred_keypoints_3d": rng.normal(size=(70, 3)).astype(np.float32),
                "pred_vertices": np.zeros((10, 3), dtype=np.float32),
            }
            np.savez(vd / f"{i:06d}_sam3d_body.npz", output=np.array(out, dtype=object))
    return root / "01" / "昼多い"


def _sample(env_dir: Path) -> VideoSample:
    return VideoSample(
        person_id="01",
        env_folder="昼多い",
        env_key="day_high",
        videos={},
        label_path=None,
        sam3d_kpts={v: env_dir / v for v in ("front", "left", "right")},
    )


def test_cache_matches_per_file_loading(tmp_path: Path) -> None:
    sam3d_root = tmp_path / "sam3d"
    env_dir = _make_sam3d_dir(sam3d_root, n_frames=40)
    cache_root = tmp_path / "cache"
    for v in ("front", "left", "right"):
        msg = build_view((str(env_dir / v), str(cache_root), str(sam3d_root), False))
        assert msg.startswith("built"), msg

    ds_file = KPTDataset("exp", [_sample(env_dir)], view_name=["front", "left", "right"], target_t=16)
    ds_cache = KPTDataset(
        "exp", [_sample(env_dir)], view_name=["front", "left", "right"], target_t=16, kpt_cache_root=cache_root
    )
    assert ds_cache._view_cache_arrays(env_dir / "front") is not None
    assert ds_file._view_cache_arrays(env_dir / "front") is None
    assert len(ds_file) == len(ds_cache) == 3  # 40 帧 -> 16/16/8

    for i in range(len(ds_file)):
        a, b = ds_file[i], ds_cache[i]
        for v in ("front", "left", "right"):
            assert a["sam3d_kpt_3d"][v].shape == (16, 52, 3)
            assert torch.equal(a["sam3d_kpt_3d"][v], b["sam3d_kpt_3d"][v])
            assert torch.equal(a["sam3d_kpt_2d"][v], b["sam3d_kpt_2d"][v])
            assert a["meta"]["sam3d_real_frame_interval"][v] == b["meta"]["sam3d_real_frame_interval"][v]
        assert a["meta"]["chunk_info"] == b["meta"]["chunk_info"]


def test_missing_cache_falls_back_silently(tmp_path: Path) -> None:
    sam3d_root = tmp_path / "sam3d"
    env_dir = _make_sam3d_dir(sam3d_root, n_frames=20)
    ds = KPTDataset(
        "exp", [_sample(env_dir)], view_name=["front"], target_t=16, kpt_cache_root=tmp_path / "nonexistent"
    )
    item = ds[0]
    assert item["sam3d_kpt_3d"]["front"].shape == (16, 52, 3)
