#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

TensorDict = Dict[str, torch.Tensor]
KptInput = Union[TensorDict, torch.Tensor, Sequence[torch.Tensor]]


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ViewFeatureEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TemporalPoseRefiner(nn.Module):
    def __init__(self, joints: int, hidden_dim: int, layers: int, dropout: float) -> None:
        super().__init__()
        in_channels = joints * 3
        blocks = []
        channels = hidden_dim
        for layer_idx in range(layers):
            blocks.extend(
                [
                    nn.Conv1d(
                        in_channels if layer_idx == 0 else channels,
                        channels,
                        kernel_size=3,
                        padding=1,
                    ),
                    nn.BatchNorm1d(channels),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
        self.tcn = nn.Sequential(*blocks)
        self.out = nn.Conv1d(channels, in_channels, kernel_size=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, pose_btj3: torch.Tensor) -> torch.Tensor:
        bsz, frames, joints, _ = pose_btj3.shape
        expected_joints = self.out.out_channels // 3
        if joints != expected_joints:
            raise ValueError(
                f"TemporalPoseRefiner was initialized for J={expected_joints}, "
                f"but got J={joints}. Set model.geofusion_num_joints={joints} "
                "or filter keypoints in the dataloader."
            )
        x = pose_btj3.reshape(bsz, frames, joints * 3).transpose(1, 2)
        delta = self.out(self.tcn(x)).transpose(1, 2)
        return delta.reshape(bsz, frames, joints, 3)


class GeoFusionPoseNet(nn.Module):
    """Geometry-guided three-view 3D keypoint fusion network.

    This model does not do classification. It outputs fused 3D keypoints.

    Inputs:
        pose3d: dict/list/stacked tensor of synchronized views.
            Each view can be (B,T,J,3) or (B,3,T,J,1).
        pose2d: optional 2D keypoints, each view (B,T,J,2) or (B,2,T,J,1).
        conf2d: optional 2D confidence, each view (B,T,J,1) or (B,1,T,J,1).
        reproj_error: optional per-view reprojection error feature.

    Outputs:
        P_final: fused refined pose, (B,T,J,3)
        P_init: gated weighted pose before temporal refinement, (B,T,J,3)
        alpha: joint-wise view weights, (B,T,J,V)
        P_views: canonicalized input 3D poses, (B,T,J,V,3)
        H_views: encoded view features, (B,T,J,V,H)
    """

    def __init__(self, hparams) -> None:
        super().__init__()
        cfg = hparams.model
        self.view_names: List[str] = list(
            getattr(cfg, "geofusion_view_names", ["front", "left", "right"])
        )
        self.num_views = len(self.view_names)
        self.joints = int(getattr(cfg, "geofusion_num_joints", 70))
        self.hidden_dim = int(getattr(cfg, "geofusion_hidden_dim", 128))
        self.dropout = float(getattr(cfg, "geofusion_dropout", 0.1))
        self.use_2d = bool(getattr(cfg, "geofusion_use_2d", True))
        self.use_conf = bool(getattr(cfg, "geofusion_use_conf", True))
        self.use_reproj_error_feature = bool(
            getattr(cfg, "geofusion_use_reproj_error_feature", False)
        )
        self.neck_index = int(getattr(cfg, "kpt_neck_index", 69))
        self.left_shoulder_index = int(getattr(cfg, "kpt_left_shoulder_index", 5))
        self.right_shoulder_index = int(getattr(cfg, "kpt_right_shoulder_index", 6))
        self.mid_hip_index = int(getattr(cfg, "kpt_mid_hip_index", -1))
        self.canonicalize = bool(getattr(cfg, "geofusion_canonicalize", True))
        self.eps = float(getattr(cfg, "geofusion_eps", 1e-6))

        feature_dim = 3 + 3
        if self.use_2d:
            feature_dim += 2 + 2
        if self.use_conf:
            feature_dim += 1
        if self.use_reproj_error_feature:
            feature_dim += 1

        self.view_encoder = ViewFeatureEncoder(feature_dim, self.hidden_dim, self.dropout)
        self.gate_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.hidden_dim // 2, 1),
        )
        projection_dim = int(getattr(cfg, "geofusion_nce_dim", 64))
        self.nce_projector = ProjectionHead(
            in_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            out_dim=projection_dim,
            dropout=self.dropout,
        )
        self.refiner = TemporalPoseRefiner(
            joints=self.joints,
            hidden_dim=int(getattr(cfg, "geofusion_refiner_dim", 256)),
            layers=int(getattr(cfg, "geofusion_refiner_layers", 2)),
            dropout=self.dropout,
        )

    @staticmethod
    def _velocity(x: torch.Tensor) -> torch.Tensor:
        vel = torch.zeros_like(x)
        vel[:, 1:] = x[:, 1:] - x[:, :-1]
        return vel

    def _as_view_list(self, value: KptInput, dims: int) -> List[torch.Tensor]:
        if isinstance(value, dict):
            missing = [name for name in self.view_names if name not in value]
            if missing:
                raise KeyError(f"Missing views: {missing}")
            views = [value[name] for name in self.view_names]
        elif torch.is_tensor(value):
            if value.ndim < 1 or value.shape[1] != self.num_views:
                raise ValueError(
                    f"Stacked input must have view dimension at dim=1, got {tuple(value.shape)}"
                )
            views = [value[:, idx] for idx in range(self.num_views)]
        elif isinstance(value, (list, tuple)):
            views = list(value)
        else:
            raise TypeError("Expected dict, stacked tensor, list, or tuple input")

        if len(views) != self.num_views:
            raise ValueError(f"Expected {self.num_views} views, got {len(views)}")
        return [self._to_btjc(view, dims=dims) for view in views]

    def _to_btjc(self, x: torch.Tensor, dims: int) -> torch.Tensor:
        if x.ndim == 4 and x.shape[-1] == dims:
            return x.float()
        if x.ndim == 5 and x.shape[1] == dims:
            return x.squeeze(-1).permute(0, 2, 3, 1).contiguous().float()
        raise ValueError(
            f"Expected (B,T,J,{dims}) or (B,{dims},T,J,1), got {tuple(x.shape)}"
        )

    def _canonicalize_pose(self, pose: torch.Tensor) -> torch.Tensor:
        if not self.canonicalize:
            return pose
        if not (0 <= self.neck_index < pose.shape[2]):
            raise ValueError(f"neck index {self.neck_index} out of range for J={pose.shape[2]}")
        for idx in (self.left_shoulder_index, self.right_shoulder_index):
            if not (0 <= idx < pose.shape[2]):
                raise ValueError(f"shoulder index {idx} out of range for J={pose.shape[2]}")

        neck = pose[:, :, self.neck_index : self.neck_index + 1]
        left = pose[:, :, self.left_shoulder_index]
        right = pose[:, :, self.right_shoulder_index]
        x_axis = F.normalize(left - right, dim=-1, eps=self.eps)

        if 0 <= self.mid_hip_index < pose.shape[2]:
            down = pose[:, :, self.mid_hip_index] - neck.squeeze(2)
        else:
            shoulder_mid = 0.5 * (left + right)
            down = shoulder_mid - neck.squeeze(2)
        down_axis = F.normalize(down, dim=-1, eps=self.eps)
        z_axis = F.normalize(torch.cross(x_axis, down_axis, dim=-1), dim=-1, eps=self.eps)
        y_axis = F.normalize(torch.cross(z_axis, x_axis, dim=-1), dim=-1, eps=self.eps)
        rot = torch.stack([x_axis, y_axis, z_axis], dim=-1)
        return torch.einsum("btjc,btcd->btjd", pose - neck, rot)

    def _build_features(
        self,
        pose3d: torch.Tensor,
        pose2d: Optional[torch.Tensor],
        conf2d: Optional[torch.Tensor],
        reproj_error: Optional[torch.Tensor],
    ) -> torch.Tensor:
        features = [pose3d, self._velocity(pose3d)]
        if self.use_2d:
            if pose2d is None:
                pose2d = pose3d.new_zeros(*pose3d.shape[:-1], 2)
            features.extend([pose2d, self._velocity(pose2d)])
        if self.use_conf:
            if conf2d is None:
                conf2d = pose3d.new_ones(*pose3d.shape[:-1], 1)
            elif conf2d.ndim == 3:
                conf2d = conf2d.unsqueeze(-1)
            features.append(conf2d.float())
        if self.use_reproj_error_feature:
            if reproj_error is None:
                reproj_error = pose3d.new_zeros(*pose3d.shape[:-1], 1)
            elif reproj_error.ndim == 3:
                reproj_error = reproj_error.unsqueeze(-1)
            features.append(reproj_error.float())
        return torch.cat(features, dim=-1)

    def forward(
        self,
        pose3d: KptInput,
        pose2d: Optional[KptInput] = None,
        conf2d: Optional[KptInput] = None,
        reproj_error: Optional[KptInput] = None,
    ) -> Dict[str, torch.Tensor]:
        p3d_views = [self._canonicalize_pose(v) for v in self._as_view_list(pose3d, dims=3)]
        p2d_views = self._as_view_list(pose2d, dims=2) if pose2d is not None else [None] * self.num_views
        conf_views = self._as_view_list(conf2d, dims=1) if conf2d is not None else [None] * self.num_views
        err_views = self._as_view_list(reproj_error, dims=1) if reproj_error is not None else [None] * self.num_views

        encoded = []
        for idx in range(self.num_views):
            feat = self._build_features(p3d_views[idx], p2d_views[idx], conf_views[idx], err_views[idx])
            encoded.append(self.view_encoder(feat))
        hidden = torch.stack(encoded, dim=3)
        gate_logits = self.gate_head(hidden).squeeze(-1)
        alpha = F.softmax(gate_logits, dim=-1)

        pose_stack = torch.stack(p3d_views, dim=3)
        p_init = (alpha.unsqueeze(-1) * pose_stack).sum(dim=3)
        delta = self.refiner(p_init)
        p_final = p_init + delta

        return {
            "P_final": p_final,
            "P_init": p_init,
            "delta": delta,
            "alpha": alpha,
            "P_views": pose_stack,
            "H_views": hidden,
        }


# Backward-compatible names; all now point to the fusion model, not a classifier.
TriViewKeypointFusionNet = GeoFusionPoseNet
KeypointTemporalMLP = GeoFusionPoseNet
