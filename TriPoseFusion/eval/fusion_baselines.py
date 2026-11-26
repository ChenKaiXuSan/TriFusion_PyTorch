#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Comparison baselines for keypoint-level multi-view fusion.

Every baseline has the interface of the fusion models in learned_fusion_experiments.py,

    forward(view_pose (B,T,J,V,3), view_conf (B,T,J,V)) -> (pred (B,T,J,3), weights (B,T,J,V)),

consumes the same cached canonicalised per-view keypoints (NaN = view missing for that
joint), and returns NaN for joints that no view observes, so the same train / stress /
driveandact / summarize code and the same compute_metrics protocol apply to all of them.

Training-free (no parameters; `train` only evaluates them):
  GeoMedianFusion    geometric median of the views (Weiszfeld), the classic robust average
  ConsensusFusion    leave-one-out consensus: drop the view farthest from the others when
                     it is an outlier, then average (RANSAC-style rule for V=3)
  SmoothMeanFusion   mean fusion followed by masked Gaussian temporal smoothing
  GPAFusion          generalised Procrustes: rigidly re-align every view to the running
                     mean shape per frame, then average

Learned (same supervision, corruption augmentation and step budget as the proposed model;
all predict a zero-initialised residual on top of the masked mean, i.e. each starts from
mean fusion and is given the same residual parameterisation as the proposed model):
  ConcatMLPFusion            per-joint MLP on the concatenated views (naive late fusion)
  LiftingMLPFusion           whole-skeleton residual MLP per frame (Martinez et al. 2017 style)
  TemporalCNNFusion          dilated temporal convolutions over the whole skeleton
                             (VideoPose3D style, Pavllo et al. 2019)
  SkeletonTransformerFusion  transformer across the joints of a frame (PoseFormer-style
                             spatial encoder)
"""
from __future__ import annotations

import torch
from torch import nn


def _prep(view_pose: torch.Tensor):
    finite = torch.isfinite(view_pose).all(dim=-1)  # (B,T,J,V)
    pose = torch.nan_to_num(view_pose, nan=0.0)
    fm = finite.to(pose.dtype)
    n = fm.sum(dim=-1, keepdim=True).clamp_min(1.0)
    mean = (pose * fm.unsqueeze(-1)).sum(dim=-2) / n  # (B,T,J,3)
    return finite, pose, fm, mean, fm / n, finite.any(dim=-1)


def _finish(pred: torch.Tensor, any_finite: torch.Tensor) -> torch.Tensor:
    return torch.where(any_finite.unsqueeze(-1).expand_as(pred), pred, torch.nan)


# ------------------------------------------------------------------ training-free


class GeoMedianFusion(nn.Module):
    def __init__(self, iters: int = 30, eps: float = 1e-6) -> None:
        super().__init__()
        self.iters, self.eps, self.last_logscale = iters, eps, None

    def forward(self, view_pose: torch.Tensor, view_conf: torch.Tensor):
        finite, pose, fm, mean, _, any_finite = _prep(view_pose)
        y, w = mean, fm
        for _ in range(self.iters):
            d = torch.linalg.norm(pose - y.unsqueeze(-2), dim=-1).clamp_min(self.eps)
            w = fm / d
            y = (pose * w.unsqueeze(-1)).sum(dim=-2) / w.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        weights = w / w.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        return _finish(y, any_finite), weights


class ConsensusFusion(nn.Module):
    """Drop the view farthest from the mean of the other views if that distance exceeds
    `tau` metres and `ratio` times the spread of the remaining views; average the rest."""

    def __init__(self, tau: float = 0.10, ratio: float = 2.0) -> None:
        super().__init__()
        self.tau, self.ratio, self.last_logscale = tau, ratio, None

    def forward(self, view_pose: torch.Tensor, view_conf: torch.Tensor):
        finite, pose, fm, mean, w_uniform, any_finite = _prep(view_pose)
        n = fm.sum(dim=-1, keepdim=True)  # (B,T,J,1)
        tot = (pose * fm.unsqueeze(-1)).sum(dim=-2, keepdim=True)  # (B,T,J,1,3)
        others = (tot - pose * fm.unsqueeze(-1)) / (n.unsqueeze(-1) - fm.unsqueeze(-1)).clamp_min(1.0)
        d = torch.linalg.norm(pose - others, dim=-1)  # distance of each view to the mean of the others
        d = torch.where(finite, d, torch.full_like(d, -1.0))
        dmax, worst = d.max(dim=-1, keepdim=True)
        keep = fm.clone().scatter_(-1, worst, 0.0)
        kn = keep.sum(dim=-1, keepdim=True).clamp_min(1.0)
        kmean = (pose * keep.unsqueeze(-1)).sum(dim=-2) / kn
        spread = 2.0 * (torch.linalg.norm(pose - kmean.unsqueeze(-2), dim=-1) * keep).max(dim=-1, keepdim=True).values
        drop = (n >= 3) & (dmax > self.tau) & (dmax > self.ratio * spread)
        weights = torch.where(drop, keep / kn, w_uniform)
        pred = torch.where(drop, kmean, mean)
        return _finish(pred, any_finite), weights


class SmoothMeanFusion(nn.Module):
    """Mean fusion + masked Gaussian smoothing along time (sigma in frames)."""

    def __init__(self, sigma: float = 2.0, radius: int = 4) -> None:
        super().__init__()
        x = torch.arange(-radius, radius + 1, dtype=torch.float32)
        k = torch.exp(-0.5 * (x / sigma) ** 2)
        self.register_buffer("kernel", (k / k.sum()).view(1, 1, -1), persistent=False)
        self.radius, self.last_logscale = radius, None

    def forward(self, view_pose: torch.Tensor, view_conf: torch.Tensor):
        finite, pose, fm, mean, w_uniform, any_finite = _prep(view_pose)
        b, t, j, _ = mean.shape
        m = any_finite.to(mean.dtype).permute(0, 2, 1).reshape(b * j, 1, t)
        x = (mean * any_finite.unsqueeze(-1)).permute(0, 2, 3, 1).reshape(b * j * 3, 1, t)
        k = self.kernel.to(mean.dtype)
        num = torch.nn.functional.conv1d(x, k, padding=self.radius).reshape(b, j, 3, t)
        den = torch.nn.functional.conv1d(m, k, padding=self.radius).reshape(b, j, 1, t)
        sm = (num / den.clamp_min(1e-6)).permute(0, 3, 1, 2)
        return _finish(sm, any_finite), w_uniform


def _rigid_align(x: torch.Tensor, y: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """Per-frame rigid (rotation + translation) alignment of x onto y over joints with m.
    x, y: (N,J,3); m: (N,J) float mask. Frames with <3 usable joints are returned unchanged.
    Row-vector convention: aligned = (x - mu_x) @ R + mu_y, R = U diag(1,1,d) Vt of
    SVD((x - mu_x)^T (y - mu_y))."""
    w = m.unsqueeze(-1)
    cnt = m.sum(dim=1, keepdim=True).clamp_min(1.0).unsqueeze(-1)
    mu_x = (x * w).sum(dim=1, keepdim=True) / cnt
    mu_y = (y * w).sum(dim=1, keepdim=True) / cnt
    xc, yc = (x - mu_x) * w, (y - mu_y) * w
    h = xc.transpose(1, 2) @ yc  # (N,3,3)
    ok = m.sum(dim=1) >= 3
    h = torch.where(ok.view(-1, 1, 1), h, torch.eye(3, dtype=x.dtype, device=x.device).expand_as(h))
    u, _, vt = torch.linalg.svd(h)
    d = torch.sign(torch.linalg.det(u @ vt))
    d = torch.where(d == 0, torch.ones_like(d), d)
    s = torch.ones(x.shape[0], 3, dtype=x.dtype, device=x.device)
    s[:, 2] = d
    r = (u * s.unsqueeze(1)) @ vt
    aligned = (x - mu_x) @ r + mu_y
    return torch.where(ok.view(-1, 1, 1), aligned, x)


class GPAFusion(nn.Module):
    def __init__(self, iters: int = 2) -> None:
        super().__init__()
        self.iters, self.last_logscale = iters, None

    def forward(self, view_pose: torch.Tensor, view_conf: torch.Tensor):
        finite, pose, fm, mean, w_uniform, any_finite = _prep(view_pose)
        b, t, j, v, _ = pose.shape
        n = fm.sum(dim=-1, keepdim=True).clamp_min(1.0)
        ref, cur = mean, pose
        for _ in range(self.iters):
            cols = []
            for vi in range(v):
                a = _rigid_align(cur[..., vi, :].reshape(b * t, j, 3).double(), ref.reshape(b * t, j, 3).double(),
                                 fm[..., vi].reshape(b * t, j).double())
                cols.append(a.reshape(b, t, j, 3).to(pose.dtype))
            cur = torch.stack(cols, dim=-2) * fm.unsqueeze(-1)
            ref = cur.sum(dim=-2) / n
        return _finish(ref, any_finite), w_uniform


# ------------------------------------------------------------------ learned


def _zero_last(layer: nn.Module) -> None:
    nn.init.zeros_(layer.weight)
    nn.init.zeros_(layer.bias)


class ConcatMLPFusion(nn.Module):
    def __init__(self, hidden: int = 64, num_joints: int = 52, num_views: int = 3, joint_emb: int = 8) -> None:
        super().__init__()
        self.joint_emb = nn.Embedding(num_joints, joint_emb)
        self.net = nn.Sequential(nn.Linear(num_views * 4 + joint_emb, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 3))
        _zero_last(self.net[-1])
        self.last_logscale = None

    def forward(self, view_pose: torch.Tensor, view_conf: torch.Tensor):
        finite, pose, fm, mean, w_uniform, any_finite = _prep(view_pose)
        b, t, j, v, _ = pose.shape
        emb = self.joint_emb.weight[None, None].expand(b, t, j, -1)
        x = torch.cat([pose.reshape(b, t, j, v * 3), fm, emb], dim=-1)
        return _finish(mean + self.net(x), any_finite), w_uniform


class LiftingMLPFusion(nn.Module):
    def __init__(self, hidden: int = 512, blocks: int = 2, dropout: float = 0.1,
                 num_joints: int = 52, num_views: int = 3) -> None:
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(num_joints * num_views * 4, hidden), nn.LayerNorm(hidden), nn.ReLU())
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Dropout(dropout),
                          nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Dropout(dropout))
            for _ in range(blocks)])
        self.out = nn.Linear(hidden, num_joints * 3)
        _zero_last(self.out)
        self.last_logscale = None

    def forward(self, view_pose: torch.Tensor, view_conf: torch.Tensor):
        finite, pose, fm, mean, w_uniform, any_finite = _prep(view_pose)
        b, t, j, v, _ = pose.shape
        h = self.inp(torch.cat([pose.reshape(b, t, j * v * 3), fm.reshape(b, t, j * v)], dim=-1))
        for blk in self.blocks:
            h = h + blk(h)
        return _finish(mean + self.out(h).reshape(b, t, j, 3), any_finite), w_uniform


class TemporalCNNFusion(nn.Module):
    """Receptive field 29 frames (~1 s at 30 fps): k=3 input conv + dilations 1, 3, 9."""

    def __init__(self, channels: int = 256, dilations=(1, 3, 9), dropout: float = 0.1,
                 num_joints: int = 52, num_views: int = 3) -> None:
        super().__init__()
        self.inp = nn.Sequential(nn.Conv1d(num_joints * num_views * 4, channels, 3, padding=1),
                                 nn.GroupNorm(1, channels), nn.ReLU())
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Conv1d(channels, channels, 3, padding=d, dilation=d), nn.GroupNorm(1, channels),
                          nn.ReLU(), nn.Dropout(dropout), nn.Conv1d(channels, channels, 1),
                          nn.GroupNorm(1, channels), nn.ReLU(), nn.Dropout(dropout))
            for d in dilations])
        self.out = nn.Conv1d(channels, num_joints * 3, 1)
        _zero_last(self.out)
        self.last_logscale = None

    def forward(self, view_pose: torch.Tensor, view_conf: torch.Tensor):
        finite, pose, fm, mean, w_uniform, any_finite = _prep(view_pose)
        b, t, j, v, _ = pose.shape
        x = torch.cat([pose.reshape(b, t, j * v * 3), fm.reshape(b, t, j * v)], dim=-1).transpose(1, 2)
        h = self.inp(x)
        for blk in self.blocks:
            h = h + blk(h)
        delta = self.out(h).transpose(1, 2).reshape(b, t, j, 3)
        return _finish(mean + delta, any_finite), w_uniform


class SkeletonTransformerFusion(nn.Module):
    def __init__(self, d: int = 64, layers: int = 3, heads: int = 4, num_joints: int = 52, num_views: int = 3) -> None:
        super().__init__()
        self.inp = nn.Linear(num_views * 4, d)
        self.joint_emb = nn.Embedding(num_joints, d)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d, heads, dim_feedforward=2 * d, dropout=0.0, batch_first=True)
            for _ in range(layers)])
        self.out = nn.Linear(d, 3)
        _zero_last(self.out)
        self.last_logscale = None

    def forward(self, view_pose: torch.Tensor, view_conf: torch.Tensor):
        finite, pose, fm, mean, w_uniform, any_finite = _prep(view_pose)
        b, t, j, v, _ = pose.shape
        h = self.inp(torch.cat([pose.reshape(b, t, j, v * 3), fm], dim=-1)) + self.joint_emb.weight[None, None]
        h = h.reshape(b * t, j, -1)
        for layer in self.layers:
            h = layer(h)
        return _finish(mean + self.out(h).reshape(b, t, j, 3), any_finite), w_uniform


BASELINES = {
    "tf_geomedian": GeoMedianFusion, "tf_consensus": ConsensusFusion,
    "tf_smooth": SmoothMeanFusion, "tf_gpa": GPAFusion,
    "concat_mlp": ConcatMLPFusion, "lift_mlp": LiftingMLPFusion,
    "tcn": TemporalCNNFusion, "skel_tr": SkeletonTransformerFusion,
}
