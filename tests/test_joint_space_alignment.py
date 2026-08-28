"""回归测试：baselines 评估必须在 52 关节模型空间做 canonicalize。

背景：SAM3D pred_keypoints_3d 与三角化伪 GT 都是 70 关节布局，而
canonicalize_pose 的默认锚点 (neck=51, shoulders=5/6) 是 52 关节模型空间的
索引。70 布局中 51 号是左手指关节（伪 GT 中 ~80% 帧为 NaN），直接在 70
布局上 canonicalize 会让绝大多数帧整帧变 NaN 且躯干轴退化。
修复：evaluate_subject_env 先按 KEEP_KEYPOINT_INDICES 选出 52 关节。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "TriPoseFusion"))
sys.path.insert(0, str(REPO_ROOT / "TriPoseFusion" / "eval"))

from map_config import KEEP_KEYPOINT_INDICES  # noqa: E402
from eval_fusion_baselines_pesudo_gt import canonicalize_pose  # noqa: E402


def _make_pose70(t: int = 8, nan_left_hand: bool = True) -> np.ndarray:
    rng = np.random.default_rng(0)
    pose = rng.normal(size=(t, 70, 3)).astype(np.float32) * 0.1
    # 解剖学合理的锚点（70 布局）：肩 5/6、颈 69
    pose[:, 5] = [-0.2, 0.0, 0.0]
    pose[:, 6] = [0.2, 0.0, 0.0]
    pose[:, 69] = [0.0, 0.1, 0.0]
    if nan_left_hand:
        # 模拟伪 GT：左手关节（42-62，含 51）大部分帧无效 = NaN
        pose[:, 42:63] = np.nan
    return pose


def test_keep_maps_anchor_indices_to_model_space() -> None:
    keep = list(KEEP_KEYPOINT_INDICES)
    assert len(keep) == 52
    assert keep[5] == 5  # left shoulder
    assert keep[6] == 6  # right shoulder
    assert keep[51] == 69  # neck: 模型空间 51 ←→ 70 布局 69


def test_canonicalize_on_keep_selected_pose_is_finite() -> None:
    pose70 = _make_pose70(nan_left_hand=True)
    keep = np.asarray(KEEP_KEYPOINT_INDICES, dtype=np.int64)
    pose52 = pose70[:, keep]

    canon = canonicalize_pose(pose52)  # 默认锚点 neck=51/shoulders=5,6（52 空间）
    # 非手部关节（0-6, 49-51）必须全部有限：NaN 不再经 neck 传播到整帧
    body_idx = list(range(0, 7)) + [49, 50, 51]
    assert np.isfinite(canon[:, body_idx]).all()
    # neck 为原点
    assert np.allclose(canon[:, 51], 0.0, atol=1e-6)


def test_canonicalize_on_raw_70_layout_propagates_nan() -> None:
    """记录 bug 行为：70 布局直接 canonicalize 时 NaN 手指关节整帧传播。"""
    pose70 = _make_pose70(nan_left_hand=True)
    canon = canonicalize_pose(pose70)  # neck=51 在 70 布局中是 NaN 手指
    assert not np.isfinite(canon).any(axis=(1, 2)).any()
