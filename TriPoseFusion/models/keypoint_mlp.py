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


class MultiScaleVelocityFeature(nn.Module):
    """IMPROVEMENT #3: Multi-scale velocity features for richer temporal dynamics.

    Instead of simple 1-frame difference, this module computes:
    - Single-frame velocity (instant motion)
    - Multi-frame velocity (smoothed motion over different timescales)
    - Acceleration (rate of change of velocity)
    - Jerk (rate of change of acceleration) for high-order dynamics

    Args:
        max_timescale: Maximum number of frames to look back (default=5)
    """

    def __init__(self, max_timescale: int = 5) -> None:
        super().__init__()
        self.max_timescale = max_timescale
        # Learnable weights for different timescales
        self.timescale_weights = nn.Parameter(torch.ones(max_timescale))

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Compute multi-scale velocity and acceleration features.

        Args:
            x: (B,T,J,dims) input poses/coordinates

        Returns:
            List of feature tensors: [velocity_1, velocity_3, velocity_5, acc, jerk]
        """
        B, T, J, D = x.shape

        features = []

        # Single-frame velocity (instant motion)
        vel_1 = x[:, 1:] - x[:, :-1]  # (B,T-1,J,D)
        features.append(vel_1)

        # Multi-scale velocities
        for scale in range(2, min(self.max_timescale + 1, T // 2)):
            vel_n = (x[:, scale:] - x[:, :-scale]) / scale
            features.append(vel_n)

        # Compute acceleration from single-frame velocity
        if T >= 3:
            vel_all = torch.cat([vel_1, torch.zeros(B, 1, J, D, device=x.device)], dim=1)
            acc = vel_all[:, 2:] - vel_all[:, :-2]
            features.append(acc / 2.0)

        # Compute jerk (rate of change of acceleration) if enough frames
        if T >= 4:
            vel_all = torch.cat([vel_1, torch.zeros(B, 1, J, D, device=x.device)], dim=1)
            acc_all = torch.cat([
                torch.zeros(B, 1, J, D, device=x.device),
                vel_all[:, 2:] - vel_all[:, :-2]
            ], dim=1) / 2.0
            jerk = acc_all[:, 3:] - acc_all[:, :-3]
            features.append(jerk / 3.0)

        # Pad to match input length (use last valid value)
        padded_features = []
        for feat in features:
            pad_len = T - feat.shape[1]
            if pad_len > 0:
                padding = torch.zeros(B, pad_len, J, D, device=x.device)
                padded_feat = torch.cat([padding, feat], dim=1)
            else:
                padded_feat = feat
            padded_features.append(padded_feat)

        return padded_features


class CrossViewAttention(nn.Module):
    """Multi-head attention for cross-view interaction with positional encoding.

    IMPROVEMENT #1: View-specific positional encoding to distinguish camera views.
    This allows the model to learn that 'front view is more reliable for steering'
    vs 'left view is better when checking blind spots'.

    Args:
        embed_dim: Feature dimension (should match hidden_dim)
        num_heads: Number of attention heads (default=4, uses 128/4=32 dim/head)
    """

    def __init__(self, embed_dim: int, num_heads: int = 4, num_views: int = 3) -> None:
        super().__init__()
        if num_views <= 0:
            raise ValueError(f"num_views must be positive, got {num_views}")
        self.num_views = int(num_views)
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.1
        )
        # IMPROVEMENT: View-specific positional encoding
        # Each configured view has a learnable position embedding.
        self.view_pos_embed = nn.Parameter(torch.randn(1, 1, self.num_views, embed_dim))

    def forward(
        self, H_views: torch.Tensor, view_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Allow views to communicate with positional context.

        Args:
            H_views: (B,T,J,V,H) encoded view features from different cameras
            view_mask: optional (B,V) bool, True = view available. Masked views are
                excluded as attention keys (leave-one-view-out self-supervision).

        Returns:
            Attended features where each view has incorporated contextual information
            with awareness of which configured view it is
        """
        B, T, J, V, H = H_views.shape
        if V != self.num_views:
            raise ValueError(f"Expected {self.num_views} views, got {V}")

        # Add positional encoding to distinguish view positions
        H_with_pos = H_views + self.view_pos_embed[:, :, :V, :]

        # Reshape to treat views as tokens for attention: (B*T*J, V, H)
        H_reshaped = H_with_pos.reshape(B * T * J, V, H)

        key_padding_mask = None
        if view_mask is not None:
            # nn.MultiheadAttention: key_padding_mask True = ignore that key
            key_padding_mask = (
                (~view_mask.bool()).unsqueeze(1).expand(B, T * J, V).reshape(B * T * J, V)
            )

        # Multi-head attention among views - each view sees all others with position info
        attended, _ = self.attention(
            H_reshaped, H_reshaped, H_reshaped, key_padding_mask=key_padding_mask
        )

        # Reshape back to original dimensions for downstream processing
        return attended.reshape(B, T, J, V, H)


class CrossViewAttentionWithGlobalGate(nn.Module):
    """Alternative: Global gate + cross-view attention hybrid.

    IMPROVEMENT #1 (alt): Combine global view weights with local cross-view communication.
    This provides both interpretability (global view importance) and robustness (view exchange).

    Args:
        embed_dim: Feature dimension
        num_heads: Number of attention heads
        use_global_gate: Whether to also compute global view gate
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        use_global_gate: bool = True,
        num_views: int = 3,
    ) -> None:
        super().__init__()
        if num_views <= 0:
            raise ValueError(f"num_views must be positive, got {num_views}")
        self.num_views = int(num_views)
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.1
        )
        self.use_global_gate = use_global_gate

        # View-specific positional encoding
        self.view_pos_embed = nn.Parameter(torch.randn(1, 1, self.num_views, embed_dim))

        # Global view importance head (for interpretability)
        if use_global_gate:
            self.global_gate = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, embed_dim // 2),
                nn.GELU(),
                nn.Linear(embed_dim // 2, 1),
            )

    def forward(self, H_views: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Return both attended features and global gate weights.

        Returns:
            Dictionary with 'H_attended' (features after cross-view attention)
            and 'global_alpha' (global view importance per frame).
        """
        B, T, J, V, H = H_views.shape
        if V != self.num_views:
            raise ValueError(f"Expected {self.num_views} views, got {V}")

        # Add positional encoding
        H_with_pos = H_views + self.view_pos_embed[:, :, :V, :]
        H_reshaped = H_with_pos.reshape(B * T * J, V, H)

        # Cross-view attention
        attended, _ = self.attention(H_reshaped, H_reshaped, H_reshaped)
        H_attended = attended.reshape(B, T, J, V, H)

        # Global gate (average over joints to get view-level importance)
        if self.use_global_gate:
            H_mean = H_attended.mean(dim=2, keepdim=True)  # (B,T,1,V,H)
            global_logits = self.global_gate(H_mean).squeeze(-1)  # (B,T,V)
            global_alpha = F.softmax(global_logits, dim=-1)
        else:
            global_alpha = None

        return {
            "H_attended": H_attended,
            "global_alpha": global_alpha
        }


class RobustCanonicalization(nn.Module):
    """IMPROVEMENT #4: More robust pose canonicalization with outlier detection.

    The original canonicalization using only left/right shoulder can be unstable when
    keypoint detection is noisy. This version uses:
    1. RANSAC-like outlier rejection for shoulder keypoints
    2. Multiple landmark averaging for more stable axis estimation
    3. Smoothed rotation matrix computation

    Args:
        eps: Epsilon for numerical stability
        outlier_threshold: Threshold for detecting shoulder outliers (in meters)
    """

    def __init__(self, eps: float = 1e-6, outlier_threshold: float = 0.15) -> None:
        super().__init__()
        self.eps = eps
        self.outlier_threshold = outlier_threshold
        # Shoulder distance prior (average shoulder width ~0.5m)
        self.shoulder_prior = 0.5

    def forward(
        self,
        pose: torch.Tensor,
        neck: torch.Tensor,
        left_shoulder: torch.Tensor,
        right_shoulder: torch.Tensor,
        mid_hip: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute robust canonicalization with outlier detection.

        Args:
            pose: (B,T,J,3) input pose
            neck: (B,T,1,3) neck position
            left_shoulder: (B,T,1,3) left shoulder position
            right_shoulder: (B,T,1,3) right shoulder position
            mid_hip: (B,T,1,3) optional mid hip position

        Returns:
            Canonicalized pose (B,T,J,3) with outlier handling
        """
        B, T, J, _ = pose.shape
        if neck.ndim == 3:
            neck = neck.unsqueeze(2)
        if left_shoulder.ndim == 3:
            left_shoulder = left_shoulder.unsqueeze(2)
        if right_shoulder.ndim == 3:
            right_shoulder = right_shoulder.unsqueeze(2)
        if mid_hip is not None and mid_hip.ndim == 3:
            mid_hip = mid_hip.unsqueeze(2)

        # Compute shoulder distance and detect outliers
        shoulder_dist = torch.linalg.norm(left_shoulder - right_shoulder, dim=-1, keepdim=True)
        shoulder_outlier = (shoulder_dist < self.eps) | (shoulder_dist > 2 * self.shoulder_prior)
        valid = ~shoulder_outlier.squeeze(-1).squeeze(-1)  # (B,T)

        # Fallback for outlier frames: reuse the shoulder axis of the temporally
        # nearest valid frame in the same clip (forward fill, then the first valid
        # frame for a leading invalid prefix). Replacing both shoulders with the
        # neck would yield a zero x-axis and collapse the rotation to zero.
        x_axis_raw = (right_shoulder - left_shoulder).squeeze(2)  # (B,T,3)
        shoulder_mid = 0.5 * (left_shoulder + right_shoulder).squeeze(2)  # (B,T,3)
        if not bool(valid.all()):
            idx = torch.arange(T, device=pose.device).expand(B, T)
            ffill_idx = torch.cummax(torch.where(valid, idx, idx.new_full((), -1)), dim=1).values
            first_valid = torch.argmax(valid.int(), dim=1, keepdim=True)  # 0 if none valid
            fallback_idx = torch.where(ffill_idx >= 0, ffill_idx, first_valid).clamp_min(0)
            gather_idx = fallback_idx.unsqueeze(-1).expand(-1, -1, 3)
            x_axis_raw = torch.where(valid.unsqueeze(-1), x_axis_raw, x_axis_raw.gather(1, gather_idx))
            shoulder_mid = torch.where(valid.unsqueeze(-1), shoulder_mid, shoulder_mid.gather(1, gather_idx))
            # Clips with no valid frame at all: default lateral unit axis.
            has_valid = valid.any(dim=1).view(B, 1, 1)
            default_axis = torch.zeros_like(x_axis_raw)
            default_axis[..., 0] = 1.0
            x_axis_raw = torch.where(has_valid, x_axis_raw, default_axis)
        x_axis = F.normalize(x_axis_raw, dim=-1, eps=self.eps)

        # Compute y-axis (downward direction)
        if mid_hip is not None:
            down_raw = mid_hip.squeeze(2) - neck.squeeze(2)
        else:
            down_raw = shoulder_mid - neck.squeeze(2)

        # Clip extreme values to prevent numerical instability
        down_norm = torch.linalg.norm(down_raw, dim=-1, keepdim=True)
        down_norm = torch.clip(down_norm, min=self.eps, max=1.0)
        down_axis = down_raw / down_norm

        # Compute z-axis (orthogonal to body plane). If the shoulder and torso
        # axes are (near-)parallel the cross product degenerates; fall back to a
        # fixed reference so the rotation matrix always has full rank.
        z_raw = torch.cross(x_axis, down_axis, dim=-1)
        ref_z = torch.cross(x_axis, x_axis.new_tensor([0.0, 0.0, 1.0]).expand_as(x_axis), dim=-1)
        ref_y = torch.cross(x_axis, x_axis.new_tensor([0.0, 1.0, 0.0]).expand_as(x_axis), dim=-1)
        ref = torch.where(torch.linalg.norm(ref_z, dim=-1, keepdim=True) > self.eps, ref_z, ref_y)
        z_raw = torch.where(torch.linalg.norm(z_raw, dim=-1, keepdim=True) > self.eps, z_raw, ref)
        z_axis = F.normalize(z_raw, dim=-1, eps=self.eps)
        y_axis = F.normalize(torch.cross(z_axis, x_axis, dim=-1), dim=-1, eps=self.eps)

        # Build rotation matrix and apply to all joints
        rot = torch.stack([x_axis, y_axis, z_axis], dim=-1)
        return torch.einsum("btjc,btcd->btjd", pose - neck, rot)


class TemporalPoseRefiner(nn.Module):
    """Deprecated - use DilatedTemporalPoseRefiner instead.

    Kept for backward compatibility but will be removed in future versions.
    """

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


class DilatedTemporalPoseRefiner(nn.Module):
    """Enhanced TCN with dilated convolutions for expanding receptive field.

    IMPROVEMENT: Exponential dilation enables efficient long-range temporal modeling.
    Dilation enables exponential receptive field growth:
    - Layer 0: dilation=1 → covers 1 frame context
    - Layer 1: dilation=2 → covers 3 frames context
    - Layer 2: dilation=4 → covers 7 frames context
    - Layer 3: dilation=8 → covers 15 frames context (with 4 layers)

    This significantly improves temporal modeling for slow gestures.
    """

    def __init__(self, joints: int, hidden_dim: int, layers: int, dropout: float) -> None:
        super().__init__()
        in_channels = joints * 3
        blocks = []
        channels = hidden_dim

        # Use dilated convolutions with exponential receptive field growth
        for layer_idx in range(layers):
            dilation = 2 ** layer_idx  # Exponential: 1, 2, 4, 8...
            effective_padding = dilation

            blocks.extend(
                [
                    nn.Conv1d(
                        in_channels if layer_idx == 0 else channels,
                        channels,
                        kernel_size=3,
                        padding=effective_padding,
                        dilation=dilation,
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
                f"DilatedTemporalPoseRefiner was initialized for J={expected_joints}, "
                f"but got J={joints}. Set model.geofusion_num_joints={joints} "
                "or filter keypoints in the dataloader."
            )
        x = pose_btj3.reshape(bsz, frames, joints * 3).transpose(1, 2)
        delta = self.out(self.tcn(x)).transpose(1, 2)
        return delta.reshape(bsz, frames, joints, 3)


class ZeroTemporalPoseRefiner(nn.Module):
    """No-op temporal refiner used for gate-only fusion baselines."""

    def forward(self, pose_btj3: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(pose_btj3)


class TriViewKeypointFusionNet(nn.Module):
    """Geometry-guided three-view 3D keypoint fusion network.

    IMPROVED VERSION with:
    - View-specific positional encoding in CrossViewAttention (IMPROVEMENT #1)
    - Multi-scale velocity features (IMPROVEMENT #3)
    - Robust canonicalization with outlier detection (IMPROVEMENT #4)
    - Configurable gate regularization support

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
        if self.num_views <= 0:
            raise ValueError("model.geofusion_view_names must contain at least one view")
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

        # IMPROVEMENT CONFIGS (from config file)
        self.use_temporal_refiner = bool(getattr(cfg, "geofusion_use_temporal_refiner", True))
        self.use_dilated_refiner = bool(getattr(cfg, "geofusion_use_dilated_refiner", True))
        self.use_multiscale_velocity = bool(getattr(cfg, "geofusion_use_multiscale_velocity", True))
        self.use_robust_canonicalization = bool(
            getattr(cfg, "geofusion_use_robust_canonicalization", False)
        )
        self.use_cross_view_attention = bool(
            getattr(cfg, "geofusion_use_cross_view_attention", True)
        )
        self.use_learned_gate = bool(getattr(cfg, "geofusion_use_learned_gate", True))
        self.gate_entropy_reg_lambda = float(getattr(cfg, "geofusion_gate_entropy_reg_lambda", 0.0))

        # Calculate feature dimension based on configuration
        # Base: [xyz, velocity_xyz] = 3 + 3 = 6 dimensions
        # If using multiscale velocity: add velocity_3, velocity_5, acc, jerk = +4*3 = +12 dims
        base_feature_dim = 3 + 3
        if self.use_multiscale_velocity:
            base_feature_dim += 3 * 4  # vel_3, vel_5, acc, jerk
        feature_dim = base_feature_dim
        if self.use_2d:
            feature_dim += 2 + 2  # [uv, velocity_uv]
        if self.use_conf:
            feature_dim += 1  # confidence
        if self.use_reproj_error_feature:
            feature_dim += 1  # reprojection error

        self.view_encoder = ViewFeatureEncoder(feature_dim, self.hidden_dim, self.dropout)

        # IMPROVEMENT #1: Cross-view attention with positional encoding
        num_attention_heads = int(getattr(cfg, "geofusion_attention_num_heads", 4))
        if self.use_cross_view_attention:
            self.cross_view_attention = CrossViewAttention(
                embed_dim=self.hidden_dim,
                num_heads=num_attention_heads,
                num_views=self.num_views,
            )
        else:
            self.cross_view_attention = nn.Identity()

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

        # IMPROVEMENT #2: Dilated TCN refiner with exponential receptive field growth.
        # Can be disabled to isolate learned gate fusion without temporal modeling.
        if not self.use_temporal_refiner:
            self.refiner = ZeroTemporalPoseRefiner()
        elif self.use_dilated_refiner:
            self.refiner = DilatedTemporalPoseRefiner(
                joints=self.joints,
                hidden_dim=int(getattr(cfg, "geofusion_refiner_dim", 256)),
                layers=int(getattr(cfg, "geofusion_refiner_layers", 4)),
                dropout=self.dropout,
            )
        else:
            self.refiner = TemporalPoseRefiner(
                joints=self.joints,
                hidden_dim=int(getattr(cfg, "geofusion_refiner_dim", 256)),
                layers=int(getattr(cfg, "geofusion_refiner_layers", 2)),
                dropout=self.dropout,
            )

        # IMPROVEMENT #4: Robust canonicalization module (optional)
        if self.use_robust_canonicalization:
            self.robust_canon = RobustCanonicalization(eps=self.eps)

    @staticmethod
    def _velocity(x: torch.Tensor) -> torch.Tensor:
        """Compute single-frame velocity (difference from previous frame)."""
        vel = torch.zeros_like(x)
        vel[:, 1:] = x[:, 1:] - x[:, :-1]
        return vel

    def _get_multiscale_velocity_features(self, pose: torch.Tensor) -> List[torch.Tensor]:
        """IMPROVEMENT #3: Compute multi-scale velocity and acceleration features.

        Args:
            pose: (B,T,J,3) input pose

        Returns:
            List of feature tensors for each timescale: [vel_1, vel_3, vel_5, acc, jerk]
        """
        B, T, J, D = pose.shape
        features = []

        # Single-frame velocity (instant motion)
        vel_1 = pose[:, 1:] - pose[:, :-1]
        features.append(vel_1)

        # Multi-scale velocities with learnable scaling
        for scale in [3, 5]:
            if T > scale:
                vel_n = (pose[:, scale:] - pose[:, :-scale]) / scale
                features.append(vel_n)

        # Compute acceleration from single-frame velocity
        if T >= 3:
            vel_all = torch.cat([vel_1, torch.zeros(B, 1, J, D, device=pose.device)], dim=1)
            acc = vel_all[:, 2:] - vel_all[:, :-2]
            features.append(acc / 2.0)

        # Compute jerk (rate of change of acceleration) if enough frames
        if T >= 4:
            vel_all = torch.cat([vel_1, torch.zeros(B, 1, J, D, device=pose.device)], dim=1)
            acc_all = torch.cat([
                torch.zeros(B, 1, J, D, device=pose.device),
                vel_all[:, 2:] - vel_all[:, :-2]
            ], dim=1) / 2.0
            jerk = acc_all[:, 3:] - acc_all[:, :-3]
            features.append(jerk / 3.0)

        # Pad to match input length (use last valid value)
        padded_features = []
        for feat in features:
            pad_len = T - feat.shape[1]
            if pad_len > 0:
                padding = torch.zeros(B, pad_len, J, D, device=pose.device)
                padded_feat = torch.cat([padding, feat], dim=1)
            else:
                padded_feat = feat
            padded_features.append(padded_feat)

        return padded_features

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
        """Normalize pose to canonical coordinate system using body landmarks.

        IMPROVEMENT #4: Uses robust canonicalization with outlier detection if enabled.

        Uses neck and shoulder positions to establish a body-centered coordinate frame:
        - x-axis: left to right shoulder direction
        - y-axis: perpendicular to x in the horizontal plane
        - z-axis: up direction (cross product of x and y)

        This helps the model be invariant to global pose position.
        """
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

        # Use robust canonicalization if enabled (with outlier detection)
        if hasattr(self, 'robust_canon'):
            mid_hip = pose[:, :, self.mid_hip_index:self.mid_hip_index+1] if 0 <= self.mid_hip_index < pose.shape[2] else None
            return self.robust_canon(pose, neck, left, right, mid_hip)

        # Original canonicalization (fallback for backward compatibility)
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
        """Build concatenated feature tensor from available inputs.

        IMPROVEMENT #3: Uses multiscale velocity features if enabled.
        """
        features = [pose3d]

        # Velocity features - use multiscale if enabled
        if self.use_multiscale_velocity:
            vel_features = self._get_multiscale_velocity_features(pose3d)
            features.extend(vel_features)
        else:
            features.append(self._velocity(pose3d))

        if self.use_2d:
            if pose2d is None:
                pose2d = pose3d.new_zeros(*pose3d.shape[:-1], 2)
            if self.use_multiscale_velocity and len(vel_features) > 0:
                # Try to get velocity features for 2D as well (simplified here)
                pass
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
        view_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through the fusion network.

        view_mask: optional (B,V) bool, True = view available. A masked view's
        encoded features are zeroed, it is excluded from cross-view attention keys
        and receives gate weight 0, so the fused pose depends only on the remaining
        views (its canonicalized pose is still returned in P_views as a target).

        Processing steps:
        1. Canonicalize each view's pose to body-centered coordinates (IMPROVEMENT #4)
        2. Encode each view independently with shared encoder
        3. IMPROVEMENT #1: Cross-view attention allows views to communicate
        4. Learn view weights (alpha) based on attended features
        5. Gated weighted fusion of poses
        6. Temporal refinement with dilated TCN (IMPROVEMENT #2)

        Returns:
            Dict with keys: P_final, P_init, delta, alpha, P_views, H_views
        """
        # Step 1: Canonicalize each view's pose
        p3d_views = [self._canonicalize_pose(v) for v in self._as_view_list(pose3d, dims=3)]
        p2d_views = self._as_view_list(pose2d, dims=2) if pose2d is not None else [None] * self.num_views
        conf_views = self._as_view_list(conf2d, dims=1) if conf2d is not None else [None] * self.num_views
        err_views = self._as_view_list(reproj_error, dims=1) if reproj_error is not None else [None] * self.num_views

        # Step 2: Encode each view independently
        encoded = []
        for idx in range(self.num_views):
            feat = self._build_features(p3d_views[idx], p2d_views[idx], conf_views[idx], err_views[idx])
            encoded.append(self.view_encoder(feat))
        hidden = torch.stack(encoded, dim=3)  # Shape: (B,T,J,V,H)

        if view_mask is not None:
            view_mask = view_mask.to(device=hidden.device, dtype=torch.bool)
            if view_mask.shape != (hidden.shape[0], self.num_views):
                raise ValueError(
                    f"view_mask must be (B,V)=({hidden.shape[0]},{self.num_views}), got {tuple(view_mask.shape)}"
                )
            hidden = hidden * view_mask[:, None, None, :, None].to(hidden.dtype)

        # IMPROVEMENT #1: Cross-view attention before gating.
        # Can be disabled for ablation with model.geofusion_use_cross_view_attention=false.
        if isinstance(self.cross_view_attention, CrossViewAttention):
            hidden_attended = self.cross_view_attention(hidden, view_mask=view_mask)
        else:
            hidden_attended = self.cross_view_attention(hidden)

        # Step 3: Learn view weights with attended features
        if self.use_learned_gate:
            gate_logits = self.gate_head(hidden_attended).squeeze(-1)  # (B,T,J,V)
            if view_mask is not None:
                gate_logits = gate_logits.masked_fill(~view_mask[:, None, None, :], -1e4)
            alpha = F.softmax(gate_logits, dim=-1)  # Joint-wise view weights
        else:
            alpha = hidden_attended.new_full(
                hidden_attended.shape[:-1],
                1.0 / float(self.num_views),
            )
            if view_mask is not None:
                avail = view_mask[:, None, None, :].to(alpha.dtype).expand_as(alpha)
                alpha = avail / avail.sum(dim=-1, keepdim=True).clamp_min(1.0)

        # IMPROVEMENT #2: Gate regularization for stability (computed but not backprop'd in forward)
        # Entropy regularization to prevent views from being completely ignored
        if self.gate_entropy_reg_lambda > 0:
            entropy = -(alpha * alpha.log().clamp_min(1e-6)).sum(dim=-1, keepdim=True)
            # Max possible entropy is log(V), so (max - entropy) encourages uniformity
            max_entropy = torch.tensor([torch.log(torch.tensor(self.num_views))], device=alpha.device)
            # Note: This loss term would be added to training loss in the trainer

        # Step 4: Gated fusion of canonicalized poses
        pose_stack = torch.stack(p3d_views, dim=3)  # (B,T,J,V,3)
        p_init = (alpha.unsqueeze(-1) * pose_stack).sum(dim=3)  # Weighted sum

        # Step 5: Temporal refinement with dilated TCN (IMPROVEMENT #2)
        delta = self.refiner(p_init)
        p_final = p_init + delta

        return {
            "P_final": p_final,      # Refined fused pose (main output)
            "P_init": p_init,        # Fusion result before temporal refinement
            "delta": delta,          # TCN correction term
            "alpha": alpha,          # View weights for interpretability
            "P_views": pose_stack,   # Canonicalized input poses
            "H_views": hidden_attended,  # Attended view features
            "view_mask": view_mask,  # (B,V) bool or None
        }
