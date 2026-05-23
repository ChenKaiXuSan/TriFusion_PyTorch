#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Sequence

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule

try:
    from project.models.keypoint_mlp import GeoFusionPoseNet
except ImportError:
    try:
        from TriPoseFusion.models.keypoint_mlp import GeoFusionPoseNet
    except ImportError:
        from models.keypoint_mlp import GeoFusionPoseNet

logger = logging.getLogger(__name__)


class GeoFusionPoseTrainer(LightningModule):
    """Geometry-guided self-supervised multi-view 3D pose fusion trainer."""

    def __init__(self, hparams) -> None:
        super().__init__()
        self.save_hyperparameters()
        cfg = hparams.model
        loss_cfg = getattr(hparams, "loss", None)
        train_cfg = getattr(hparams, "train", None)

        self.model = GeoFusionPoseNet(hparams)
        self.lr = float(getattr(loss_cfg, "lr", 1e-3))
        self.weight_decay = float(getattr(loss_cfg, "weight_decay", 1e-5))
        self.grad_clip_val = float(getattr(train_cfg, "grad_clip_val", 1.0))
        self.view_names = self.model.view_names

        self.lambda_tri = float(getattr(cfg, "lambda_tri", 1.0))
        self.lambda_reproj = float(getattr(cfg, "lambda_reproj", 0.0))
        self.lambda_view = float(getattr(cfg, "lambda_view", 0.2))
        self.lambda_bone = float(getattr(cfg, "lambda_bone", 0.5))
        self.lambda_temp = float(getattr(cfg, "lambda_temp", 0.1))
        self.lambda_info_nce = float(getattr(cfg, "lambda_info_nce", 0.1))
        self.info_nce_temperature = float(getattr(cfg, "info_nce_temperature", 0.1))
        self.bones = list(getattr(cfg, "geofusion_bones", []))

    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        return self.model(
            pose3d=self._get_required(batch, ("kpt3d", "pose3d", "sam3d_kpt", "sam3d_kpt_3d")),
            pose2d=self._get_optional(batch, ("kpt2d", "pose2d", "sam2d_kpt", "sam2d_kpt_2d", "sam3d_kpt_2d")),
            conf2d=self._get_optional(batch, ("conf2d", "kpt2d_conf", "confidence2d")),
            reproj_error=self._get_optional(batch, ("reproj_error", "reprojection_error")),
        )

    @staticmethod
    def _get_optional(batch: Dict[str, Any], keys: Sequence[str]):
        for key in keys:
            value = batch.get(key)
            if value is not None:
                return value
        return None

    @staticmethod
    def _get_required(batch: Dict[str, Any], keys: Sequence[str]):
        value = GeoFusionPoseTrainer._get_optional(batch, keys)
        if value is None:
            raise KeyError(f"Missing required 3D keypoints. Expected one of: {keys}")
        return value

    def _teacher_loss(
        self,
        pred: torch.Tensor,
        batch: Dict[str, Any],
        out: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        teacher = self._get_optional(batch, ("P_teacher", "p_teacher", "teacher3d", "triangulated3d"))
        if teacher is not None:
            teacher = self.model._to_btjc(teacher, dims=3)
            teacher = self.model._canonicalize_pose(teacher)
        else:
            # No cameras / triangulation teacher: use robust canonical multi-view median.
            teacher = out["P_views"].detach().median(dim=3).values
        return F.smooth_l1_loss(pred, teacher)

    def _view_consistency_loss(self, pred: torch.Tensor, out: Dict[str, torch.Tensor]) -> torch.Tensor:
        diff = torch.linalg.norm(pred.unsqueeze(3) - out["P_views"], dim=-1)
        return (out["alpha"].detach() * diff).mean()

    def _bone_loss(self, pred: torch.Tensor, out: Dict[str, torch.Tensor]) -> torch.Tensor:
        if not self.bones:
            return pred.new_zeros(())
        losses = []
        ref = out["P_init"].detach()
        for a, b in self.bones:
            pred_len = torch.linalg.norm(pred[:, :, a] - pred[:, :, b], dim=-1)
            ref_len = torch.linalg.norm(ref[:, :, a] - ref[:, :, b], dim=-1)
            losses.append(torch.abs(pred_len - ref_len).mean())
        return torch.stack(losses).mean() if losses else pred.new_zeros(())

    @staticmethod
    def _temporal_loss(pred: torch.Tensor) -> torch.Tensor:
        if pred.shape[1] < 3:
            return pred.new_zeros(())
        acc = pred[:, 2:] - 2 * pred[:, 1:-1] + pred[:, :-2]
        return torch.linalg.norm(acc, dim=-1).mean()

    def _bidirectional_info_nce(self, anchor: torch.Tensor, positive: torch.Tensor) -> torch.Tensor:
        if anchor.shape[0] <= 1:
            return anchor.new_zeros(())
        anchor = F.normalize(anchor, dim=-1)
        positive = F.normalize(positive, dim=-1)
        logits = anchor @ positive.T / self.info_nce_temperature
        labels = torch.arange(anchor.shape[0], device=anchor.device)
        return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))

    def _info_nce_loss(self, out: Dict[str, torch.Tensor]) -> torch.Tensor:
        hidden = out["H_views"]  # (B,T,J,V,H)
        bsz, frames, joints, views, dim = hidden.shape
        projected = self.model.nce_projector(hidden.reshape(bsz * frames * joints * views, dim))
        projected = projected.reshape(bsz * frames * joints, views, -1)
        pair_losses = []
        for i in range(views):
            for j in range(i + 1, views):
                pair_losses.append(self._bidirectional_info_nce(projected[:, i], projected[:, j]))
        return torch.stack(pair_losses).mean() if pair_losses else hidden.new_zeros(())

    def _project_points(self, points: torch.Tensor, camera: Dict[str, torch.Tensor]) -> torch.Tensor:
        # points: (B,T,J,3), K: (B,3,3) or (3,3), R/t optional.
        K = camera["K"].to(points.device).float()
        if K.ndim == 2:
            K = K.unsqueeze(0).expand(points.shape[0], -1, -1)
        R = camera.get("R")
        t = camera.get("t")
        cam_points = points
        if R is not None:
            R = R.to(points.device).float()
            if R.ndim == 2:
                R = R.unsqueeze(0).expand(points.shape[0], -1, -1)
            cam_points = torch.matmul(cam_points, R.transpose(-1, -2).unsqueeze(1))
        if t is not None:
            t = t.to(points.device).float()
            if t.ndim == 1:
                t = t.unsqueeze(0).expand(points.shape[0], -1)
            cam_points = cam_points + t[:, None, None]
        proj = torch.matmul(cam_points, K.transpose(-1, -2).unsqueeze(1))
        return proj[..., :2] / proj[..., 2:].clamp_min(1e-6)

    def _reprojection_loss(self, pred: torch.Tensor, batch: Dict[str, Any]) -> torch.Tensor:
        pose2d = self._get_optional(batch, ("kpt2d", "pose2d", "sam2d_kpt", "sam2d_kpt_2d", "sam3d_kpt_2d"))
        cameras = self._get_optional(batch, ("cameras", "camera", "camera_params"))
        if pose2d is None or cameras is None:
            return pred.new_zeros(())
        pose2d_views = self.model._as_view_list(pose2d, dims=2)
        losses = []
        for idx, name in enumerate(self.view_names):
            cam = cameras[name] if isinstance(cameras, dict) and name in cameras else None
            if cam is None or "K" not in cam:
                continue
            pred2d = self._project_points(pred, cam)
            losses.append(F.smooth_l1_loss(pred2d, pose2d_views[idx]))
        return torch.stack(losses).mean() if losses else pred.new_zeros(())

    def _losses(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        out = self.forward(batch)
        pred = out["P_final"]
        loss_tri = self._teacher_loss(pred, batch, out)
        loss_reproj = self._reprojection_loss(pred, batch)
        loss_view = self._view_consistency_loss(pred, out)
        loss_bone = self._bone_loss(pred, out)
        loss_temp = self._temporal_loss(pred)
        loss_info_nce = self._info_nce_loss(out)
        loss = (
            self.lambda_tri * loss_tri
            + self.lambda_reproj * loss_reproj
            + self.lambda_view * loss_view
            + self.lambda_bone * loss_bone
            + self.lambda_temp * loss_temp
            + self.lambda_info_nce * loss_info_nce
        )
        return {
            "loss": loss,
            "loss_tri": loss_tri,
            "loss_reproj": loss_reproj,
            "loss_view": loss_view,
            "loss_bone": loss_bone,
            "loss_temp": loss_temp,
            "loss_info_nce": loss_info_nce,
            "alpha": out["alpha"],
            "P_final": pred,
        }

    def _shared_step(self, batch: Dict[str, Any], stage: str) -> torch.Tensor:
        losses = self._losses(batch)
        bsz = losses["P_final"].shape[0]
        self.log(f"{stage}/loss", losses["loss"], on_step=stage == "train", on_epoch=True, prog_bar=True, batch_size=bsz)
        self.log_dict(
            {
                f"{stage}/loss_tri": losses["loss_tri"],
                f"{stage}/loss_reproj": losses["loss_reproj"],
                f"{stage}/loss_view": losses["loss_view"],
                f"{stage}/loss_bone": losses["loss_bone"],
                f"{stage}/loss_temp": losses["loss_temp"],
                f"{stage}/loss_info_nce": losses["loss_info_nce"],
            },
            on_step=stage == "train",
            on_epoch=True,
            batch_size=bsz,
        )
        alpha_mean = losses["alpha"].mean(dim=(0, 1, 2))
        self.log_dict(
            {f"{stage}/alpha_{name}": alpha_mean[idx] for idx, name in enumerate(self.view_names)},
            on_step=False,
            on_epoch=True,
            batch_size=bsz,
        )
        return losses["loss"]

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "val")

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, int(getattr(self.trainer, "estimated_stepping_batches", 1))),
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "train/loss"}}

    def configure_gradient_clipping(
        self,
        optimizer,
        gradient_clip_val: Optional[float] = None,
        gradient_clip_algorithm: Optional[str] = None,
    ) -> None:
        clip_val = self.grad_clip_val if gradient_clip_val is None else gradient_clip_val
        if clip_val > 0:
            self.clip_gradients(
                optimizer,
                gradient_clip_val=clip_val,
                gradient_clip_algorithm=gradient_clip_algorithm or "norm",
            )


# Backward-compatible aliases for existing imports/configs.
TripleViewSelfSupervisedFusionTrainer = GeoFusionPoseTrainer
TripleFusionSelfSupervisedTrainer = GeoFusionPoseTrainer
MultiFusion3DCNNTrainer = GeoFusionPoseTrainer
