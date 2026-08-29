"""守卫测试：baselines 评估链路的 70→52 关节空间映射不变量。

SAM3D pred_keypoints_3d 与三角化伪 GT 均为 70 关节原始布局；
canonicalize_pose 的默认锚点 (neck=51, shoulders=5/6) 是 52 关节模型空间的
索引（52 空间 51 ←→ 70 空间 69 = neck）。这些锚点之所以正确，是因为
load_sam3d_frame / load_gt_sequence 在加载时就按 KEEP_KEYPOINT_INDICES
做了 70→52 映射——本文件守住这一前提（若有人移除加载器里的 KEEP，或在
下游重复做 KEEP 二次映射，这里会失败）。

背景：70 布局的 51 号是左手指关节（伪 GT 中 ~80% 帧为 NaN），如果 70
布局数组未经映射直接进 canonicalize，NaN 会经 neck 平移传播导致整帧 NaN。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "TriPoseFusion"))
sys.path.insert(0, str(REPO_ROOT / "TriPoseFusion" / "eval"))

from map_config import KEEP_KEYPOINT_INDICES  # noqa: E402
from eval_fusion_baselines_pesudo_gt import (  # noqa: E402
    canonicalize_pose,
    load_gt_sequence,
    load_sam3d_frame,
)


def _make_pose70(t: int = 8, nan_left_hand: bool = True) -> np.ndarray:
    rng = np.random.default_rng(0)
    pose = rng.normal(size=(t, 70, 3)).astype(np.float32) * 0.1
    # 解剖学合理的锚点（70 布局）：肩 5/6、颈 69
    pose[:, 5] = [-0.2, 0.0, 0.0]
    pose[:, 6] = [0.2, 0.0, 0.0]
    pose[:, 69] = [0.0, 0.1, 0.0]
    if nan_left_hand:
        # 模拟伪 GT：左手关节（42-62，含 70 布局的 51）大部分帧无效 = NaN
        pose[:, 42:63] = np.nan
    return pose


def test_keep_maps_anchor_indices_to_model_space() -> None:
    keep = list(KEEP_KEYPOINT_INDICES)
    assert len(keep) == 52
    assert keep[5] == 5  # left shoulder
    assert keep[6] == 6  # right shoulder
    assert keep[51] == 69  # neck: 模型空间 51 ←→ 70 布局 69


def test_load_sam3d_frame_returns_52_joint_model_space(tmp_path: Path) -> None:
    pose70 = _make_pose70(t=1, nan_left_hand=False)[0]  # (70,3)
    out = {
        "pred_keypoints_3d": pose70,
        "pred_keypoints_2d": np.zeros((70, 2), dtype=np.float32),
    }
    npz = tmp_path / "000001_sam3d_body.npz"
    np.savez(npz, output=np.array(out, dtype=object))

    kpt, conf = load_sam3d_frame(npz)
    assert kpt.shape == (52, 3)
    assert conf.shape == (52,)
    # 模型空间 51 号必须是 70 布局的 69 号（neck），而非手指关节
    assert np.allclose(kpt[51], pose70[69])
    assert np.allclose(kpt[5], pose70[5]) and np.allclose(kpt[6], pose70[6])


def test_load_gt_sequence_returns_52_joint_model_space(tmp_path: Path) -> None:
    pose70 = _make_pose70(t=4, nan_left_hand=True)
    valid = np.isfinite(pose70).all(axis=-1)
    npz = tmp_path / "keypoints_3d.npz"
    np.savez(npz, keypoints_3d=pose70, valid_mask=valid, frame_ids=np.array(["1", "2", "3", "4"]))

    gt, gt_valid, _ = load_gt_sequence(npz)
    assert gt.shape == (4, 52, 3)
    assert np.allclose(gt[:, 51], pose70[:, 69])  # neck 映射正确
    # 经过映射的 GT 可直接用默认锚点 canonicalize：非手部关节全部有限
    canon = canonicalize_pose(gt)
    body_idx = list(range(0, 7)) + [49, 50, 51]
    assert np.isfinite(canon[:, body_idx]).all()
    assert np.allclose(canon[:, 51], 0.0, atol=1e-6)  # neck 为原点


def test_canonicalize_on_raw_70_layout_would_propagate_nan() -> None:
    """说明为什么加载器级 KEEP 是承重的：未映射的 70 布局直接 canonicalize
    时，NaN 手指关节（70 布局 51 号）会让整帧输出 NaN。"""
    pose70 = _make_pose70(nan_left_hand=True)
    canon = canonicalize_pose(pose70)
    assert not np.isfinite(canon).any(axis=(1, 2)).any()
