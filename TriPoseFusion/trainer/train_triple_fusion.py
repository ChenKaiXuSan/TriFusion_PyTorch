#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Sequence

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule

from models.keypoint_mlp import TriViewKeypointFusionNet  # For backward compatibility with existing configs that specify this class.   
logger = logging.getLogger(__name__)


def gate_entropy_loss(alpha: torch.Tensor) -> torch.Tensor:
    """Uniformity regularizer log(V) - H(alpha), averaged over batch/time/joints.

    alpha: (..., V) softmax gate weights. Returns a non-negative scalar that is
    zero iff alpha is uniform over the view dimension. alpha must be clamped
    before the log so near-zero weights keep a finite, non-zero gradient.
    """
    entropy = -(alpha * alpha.clamp_min(1e-6).log()).sum(dim=-1)
    max_entropy = torch.log(torch.tensor(float(alpha.shape[-1]), device=alpha.device))
    return (max_entropy - entropy).mean()


class TriFusionPoseTrainer(LightningModule):
    """Geometry-guided self-supervised multi-view 3D pose fusion trainer."""

    def __init__(self, hparams) -> None:
        super().__init__()
        self.save_hyperparameters()
        cfg = hparams.model
        loss_cfg = getattr(hparams, "loss", None)
        train_cfg = getattr(hparams, "train", None)

        self.model = TriViewKeypointFusionNet(hparams)
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
        # IMPROVEMENT #2: Gate entropy regularization lambda
        self.lambda_gate_entropy = float(getattr(cfg, "geofusion_gate_entropy_reg_lambda", 0.0))
        self.info_nce_temperature = float(getattr(cfg, "info_nce_temperature", 0.1))
        self.bones = list(getattr(cfg, "geofusion_bones", []))
        # Leave-one-view-out self-supervised teacher: mask one view per sample and use
        # that view's canonicalized observation as the L_tri target (median is the
        # trivial optimum of the original objective; a held-out view is not).
        self.loo_teacher = bool(getattr(cfg, "loo_teacher", False))
        self.loo_prob = float(getattr(cfg, "loo_prob", 1.0))

    def forward(
        self, batch: Dict[str, Any], view_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        return self.model(
            pose3d=self._get_required(batch, ("kpt3d", "pose3d", "sam3d_kpt", "sam3d_kpt_3d")),
            pose2d=self._get_optional(batch, ("kpt2d", "pose2d", "sam2d_kpt", "sam2d_kpt_2d", "sam3d_kpt_2d")),
            conf2d=self._get_optional(batch, ("conf2d", "kpt2d_conf", "confidence2d")),
            reproj_error=self._get_optional(batch, ("reproj_error", "reprojection_error")),
            view_mask=view_mask,
        )

    def _batch_size(self, batch: Dict[str, Any]) -> int:
        pose3d = self._get_required(batch, ("kpt3d", "pose3d", "sam3d_kpt", "sam3d_kpt_3d"))
        if isinstance(pose3d, dict):
            pose3d = next(iter(pose3d.values()))
        return int(pose3d.shape[0])

    def _sample_view_mask(
        self, bsz: int, stage: str, batch_idx: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (view_mask (B,V) bool, loo_index (B,) long; -1 = no view held out).

        train: random held-out view per sample (each sample with prob loo_prob);
        val/test: deterministic (sample + batch_idx) % V so every view is held out
        equally often and the metric is reproducible.
        """
        num_views = len(self.view_names)
        if stage == "train":
            loo_index = torch.randint(0, num_views, (bsz,), device=device)
            keep = torch.rand(bsz, device=device) >= self.loo_prob
            loo_index = loo_index.masked_fill(keep, -1)
        else:
            loo_index = (torch.arange(bsz, device=device) + int(batch_idx)) % num_views
        view_mask = torch.ones(bsz, num_views, dtype=torch.bool, device=device)
        rows = torch.nonzero(loo_index >= 0, as_tuple=True)[0]
        view_mask[rows, loo_index[rows]] = False
        return view_mask, loo_index

    @staticmethod
    def _get_optional(batch: Dict[str, Any], keys: Sequence[str]):
        for key in keys:
            value = batch.get(key)
            if value is not None:
                return value
        return None

    @staticmethod
    def _get_required(batch: Dict[str, Any], keys: Sequence[str]):
        value = TriFusionPoseTrainer._get_optional(batch, keys)
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
            return F.smooth_l1_loss(pred, teacher)

        p_views = out["P_views"].detach()  # (B,T,J,V,3)
        median = p_views.median(dim=3).values
        loo_index = out.get("loo_index")
        if loo_index is None or not bool((loo_index >= 0).any()):
            # No cameras / triangulation teacher: use robust canonical multi-view median.
            return F.smooth_l1_loss(pred, median)

        # Leave-one-view-out: samples with a held-out view regress that view's
        # canonicalized observation (not in the input); the rest keep the median.
        bsz = pred.shape[0]
        target = p_views[torch.arange(bsz, device=pred.device), :, :, loo_index.clamp_min(0)]
        sel = loo_index >= 0
        loss = F.smooth_l1_loss(pred[sel], target[sel], reduction="sum")
        if bool((~sel).any()):
            loss = loss + F.smooth_l1_loss(pred[~sel], median[~sel], reduction="sum")
        return loss / pred.numel()

    @staticmethod
    def _loo_error(pred: torch.Tensor, out: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        """Mean L2 distance (m) between the fused pose and the held-out view's
        canonicalized observation, over held-out samples. Self-supervised proxy
        for checkpoint selection."""
        loo_index = out.get("loo_index")
        if loo_index is None or not bool((loo_index >= 0).any()):
            return None
        sel = loo_index >= 0
        bsz = pred.shape[0]
        target = out["P_views"].detach()[torch.arange(bsz, device=pred.device), :, :, loo_index.clamp_min(0)]
        dist = torch.linalg.norm(pred[sel] - target[sel], dim=-1)
        return dist[torch.isfinite(dist)].mean()

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

    def _losses(
        self, batch: Dict[str, Any], stage: str = "train", batch_idx: int = 0
    ) -> Dict[str, torch.Tensor]:
        view_mask = loo_index = None
        if self.loo_teacher:
            view_mask, loo_index = self._sample_view_mask(
                self._batch_size(batch), stage, batch_idx, self.device
            )
        out = self.forward(batch, view_mask=view_mask)
        out["loo_index"] = loo_index
        pred = out["P_final"]
        loss_tri = self._teacher_loss(pred, batch, out)
        loss_reproj = self._reprojection_loss(pred, batch)
        loss_view = self._view_consistency_loss(pred, out)
        loss_bone = self._bone_loss(pred, out)
        loss_temp = self._temporal_loss(pred)
        loss_info_nce = self._info_nce_loss(out)

        # IMPROVEMENT #2: Gate entropy regularization - encourages uniform view usage
        # This prevents some views from being completely ignored during training
        if self.lambda_gate_entropy > 0:
            loss_gate_entropy = gate_entropy_loss(out["alpha"])  # alpha: (B,T,J,V)
        else:
            loss_gate_entropy = pred.new_zeros(())

        loss = (
            self.lambda_tri * loss_tri
            + self.lambda_reproj * loss_reproj
            + self.lambda_view * loss_view
            + self.lambda_bone * loss_bone
            + self.lambda_temp * loss_temp
            + self.lambda_info_nce * loss_info_nce
            + self.lambda_gate_entropy * loss_gate_entropy
        )
        return {
            "loss": loss,
            "loss_tri": loss_tri,
            "loss_reproj": loss_reproj,
            "loss_view": loss_view,
            "loss_bone": loss_bone,
            "loss_temp": loss_temp,
            "loss_info_nce": loss_info_nce,
            "loss_gate_entropy": loss_gate_entropy,
            "alpha": out["alpha"],
            "P_final": pred,
            "loo_mpjpe": self._loo_error(pred, out),
        }

    def _shared_step(self, batch: Dict[str, Any], stage: str, batch_idx: int = 0) -> torch.Tensor:
        losses = self._losses(batch, stage=stage, batch_idx=batch_idx)
        bsz = losses["P_final"].shape[0]
        self.log(f"{stage}/loss", losses["loss"], on_step=stage == "train", on_epoch=True, prog_bar=True, batch_size=bsz)
        if losses.get("loo_mpjpe") is not None:
            self.log(
                f"{stage}/loo_mpjpe", losses["loo_mpjpe"],
                on_step=False, on_epoch=True, prog_bar=stage != "train", batch_size=bsz,
            )
        self.log_dict(
            {
                f"{stage}/loss_tri": losses["loss_tri"],
                f"{stage}/loss_reproj": losses["loss_reproj"],
                f"{stage}/loss_view": losses["loss_view"],
                f"{stage}/loss_bone": losses["loss_bone"],
                f"{stage}/loss_temp": losses["loss_temp"],
                f"{stage}/loss_info_nce": losses["loss_info_nce"],
                f"{stage}/loss_gate_entropy": losses.get("loss_gate_entropy", torch.zeros(1)),
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
        return self._shared_step(batch, "train", batch_idx)

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "val", batch_idx)

    def test_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "test", batch_idx)

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
TripleViewSelfSupervisedFusionTrainer = TriFusionPoseTrainer
TripleFusionSelfSupervisedTrainer = TriFusionPoseTrainer
MultiFusion3DCNNTrainer = TriFusionPoseTrainer
