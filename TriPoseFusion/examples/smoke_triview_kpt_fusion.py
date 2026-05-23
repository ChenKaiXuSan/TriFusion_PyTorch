#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from TriPoseFusion.models.keypoint_mlp import GeoFusionPoseNet


def make_hparams() -> SimpleNamespace:
    return SimpleNamespace(
        model=SimpleNamespace(
            geofusion_view_names=["front", "left", "right"],
            geofusion_num_joints=70,
            geofusion_hidden_dim=32,
            geofusion_refiner_dim=64,
            geofusion_refiner_layers=2,
            geofusion_dropout=0.1,
            geofusion_use_2d=True,
            geofusion_use_conf=True,
            geofusion_use_reproj_error_feature=False,
            geofusion_canonicalize=True,
            geofusion_nce_dim=16,
            kpt_neck_index=69,
            kpt_left_shoulder_index=5,
            kpt_right_shoulder_index=6,
            kpt_mid_hip_index=-1,
        )
    )


def make_fake_views(batch_size: int = 2, frames: int = 8, joints: int = 70):
    pose3d = {
        "front": torch.randn(batch_size, 3, frames, joints, 1),
        "left": torch.randn(batch_size, 3, frames, joints, 1),
        "right": torch.randn(batch_size, 3, frames, joints, 1),
    }
    pose2d = {
        "front": torch.randn(batch_size, 2, frames, joints, 1),
        "left": torch.randn(batch_size, 2, frames, joints, 1),
        "right": torch.randn(batch_size, 2, frames, joints, 1),
    }
    conf2d = {
        "front": torch.rand(batch_size, 1, frames, joints, 1),
        "left": torch.rand(batch_size, 1, frames, joints, 1),
        "right": torch.rand(batch_size, 1, frames, joints, 1),
    }
    return pose3d, pose2d, conf2d


def main() -> None:
    torch.manual_seed(0)
    model = GeoFusionPoseNet(make_hparams())
    model.eval()

    pose3d, pose2d, conf2d = make_fake_views()
    with torch.no_grad():
        out = model(pose3d=pose3d, pose2d=pose2d, conf2d=conf2d)
        stacked = torch.stack([pose3d["front"], pose3d["left"], pose3d["right"]], dim=1)
        out_stacked = model(pose3d=stacked)

    assert out["P_final"].shape == (2, 8, 70, 3), out["P_final"].shape
    assert out["P_init"].shape == (2, 8, 70, 3), out["P_init"].shape
    assert out["alpha"].shape == (2, 8, 70, 3), out["alpha"].shape
    assert out["P_views"].shape == (2, 8, 70, 3, 3), out["P_views"].shape
    assert out["H_views"].shape[:4] == (2, 8, 70, 3), out["H_views"].shape
    assert torch.allclose(out["alpha"].sum(dim=-1), torch.ones_like(out["alpha"][..., 0]), atol=1e-5)
    assert torch.allclose(out["P_final"], out["P_init"], atol=1e-6)
    assert out_stacked["P_final"].shape == (2, 8, 70, 3), out_stacked["P_final"].shape

    print("GeoFusionPoseNet smoke sample passed")
    print(f"P_final: {tuple(out['P_final'].shape)}")
    print(f"P_init: {tuple(out['P_init'].shape)}")
    print(f"alpha: {tuple(out['alpha'].shape)}")
    print(f"P_views: {tuple(out['P_views'].shape)}")
    print(f"H_views: {tuple(out['H_views'].shape)}")
    print(f"alpha sum mean: {float(out['alpha'].sum(dim=-1).mean())}")


if __name__ == "__main__":
    main()
