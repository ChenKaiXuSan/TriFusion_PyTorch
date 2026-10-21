"""留一视角（leave-one-view-out）自监督 teacher 的行为守卫。

- view_mask=None 时 forward 与旧行为逐位一致（回归）
- 屏蔽视角的 gate 权重严格为 0，其余视角权重和为 1
- 屏蔽视角的输入不泄漏：改变被屏蔽视角的输入不改变输出
- LOO 损失 = 融合输出 vs 被屏蔽视角 canonical 观测；val 阶段屏蔽顺序确定
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "TriPoseFusion"))

from trainer.train_triple_fusion import TriFusionPoseTrainer  # noqa: E402

VIEWS = ["front", "left", "right"]


def _module(loo: bool) -> TriFusionPoseTrainer:
    cfg = OmegaConf.load(REPO_ROOT / "TriPoseFusion" / "configs" / "train.yaml")
    cfg.model.loo_teacher = loo
    cfg.model.geofusion_use_robust_canonicalization = True
    torch.manual_seed(0)
    m = TriFusionPoseTrainer(cfg)
    m.eval()
    return m


def _batch(b: int = 2, t: int = 16, j: int = 52, seed: int = 1) -> dict:
    g = torch.Generator().manual_seed(seed)
    p3 = {v: torch.randn(b, t, j, 3, generator=g) * 0.2 for v in VIEWS}
    for v in VIEWS:  # 合理的肩/颈锚点，避免退化回退
        p3[v][:, :, 5] = torch.tensor([-0.2, 0.0, 0.0])
        p3[v][:, :, 6] = torch.tensor([0.2, 0.0, 0.0])
        p3[v][:, :, 51] = torch.tensor([0.0, 0.1, 0.0])
    p2 = {v: torch.randn(b, t, j, 2, generator=g) * 50 for v in VIEWS}
    return {"sam3d_kpt_3d": p3, "sam3d_kpt_2d": p2}


def test_no_mask_is_backward_compatible() -> None:
    m = _module(loo=False)
    batch = _batch()
    with torch.no_grad():
        a = m.forward(batch)
        b = m.forward(batch, view_mask=None)
        c = m.forward(batch, view_mask=torch.ones(2, 3, dtype=torch.bool))
    assert torch.equal(a["P_final"], b["P_final"])
    assert torch.allclose(a["P_final"], c["P_final"], atol=1e-6)
    assert a["view_mask"] is None


def test_masked_view_gets_zero_gate_and_does_not_leak() -> None:
    m = _module(loo=True)
    batch = _batch()
    mask = torch.tensor([[True, False, True], [False, True, True]])
    with torch.no_grad():
        out = m.forward(batch, view_mask=mask)
        alpha = out["alpha"]  # (B,T,J,V)
        assert torch.all(alpha[0, :, :, 1] == 0)
        assert torch.all(alpha[1, :, :, 0] == 0)
        assert torch.allclose(alpha.sum(-1), torch.ones_like(alpha[..., 0]), atol=1e-5)

        # 改变被屏蔽视角的输入（保持锚点），输出不得变化
        batch2 = _batch(seed=7)
        batch2["sam3d_kpt_3d"] = {v: batch["sam3d_kpt_3d"][v].clone() for v in VIEWS}
        batch2["sam3d_kpt_2d"] = {v: batch["sam3d_kpt_2d"][v].clone() for v in VIEWS}
        batch2["sam3d_kpt_3d"]["left"][0, :, 7:49] += 1.0  # 样本 0 屏蔽 left
        batch2["sam3d_kpt_2d"]["left"][0] += 10.0
        out2 = m.forward(batch2, view_mask=mask)
    assert torch.allclose(out["P_final"][0], out2["P_final"][0], atol=1e-6)
    assert not torch.allclose(out["P_views"][0, :, :, 1], out2["P_views"][0, :, :, 1])


def test_loo_loss_targets_held_out_view_and_val_is_deterministic() -> None:
    m = _module(loo=True)
    batch = _batch(b=3)
    with torch.no_grad():
        losses = m._losses(batch, stage="val", batch_idx=1)
    assert torch.isfinite(losses["loss"])
    assert losses["loo_mpjpe"] is not None and torch.isfinite(losses["loo_mpjpe"])

    # val: loo_index = (arange(B)+batch_idx) % V
    mask, idx = m._sample_view_mask(3, "val", 1, torch.device("cpu"))
    assert idx.tolist() == [1, 2, 0]
    assert mask.tolist() == [[True, False, True], [True, True, False], [False, True, True]]

    # L_tri 等于对被屏蔽视角 canonical 观测的 SmoothL1
    with torch.no_grad():
        out = m.forward(batch, view_mask=mask)
        out["loo_index"] = idx
        pred = out["P_final"]
        target = torch.stack([out["P_views"][i, :, :, idx[i]] for i in range(3)])
        expected = torch.nn.functional.smooth_l1_loss(pred, target)
        got = m._teacher_loss(pred, batch, out)
    assert torch.allclose(got, expected, atol=1e-6)

    # train: 概率 1 时每个样本都屏蔽一个视角
    mask_tr, idx_tr = m._sample_view_mask(8, "train", 0, torch.device("cpu"))
    assert bool((idx_tr >= 0).all()) and (mask_tr.sum(1) == 2).all()
