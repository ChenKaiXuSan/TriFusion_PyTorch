#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Learned confidence-weighted tri-view fusion baseline (rebuttal point E).

A small learnable-triangulation-style baseline on exactly the same inputs as
TriPoseFusion: canonicalized per-view SAM3D keypoints (52 joints) with per-joint
confidences. A shared per-joint MLP produces per-view gating logits (softmax
over views) plus a residual refinement on the fused pose. Trained against the
triangulated pseudo GT on valid joints only, GroupKFold fold-0 protocol
(train subjects -> fit, val subjects -> report), metrics via the corrected
``compute_metrics`` so numbers are directly comparable with Table 3.

Example (smoke test):
    python learned_fusion_baseline.py \
      --sam3d-root /work/SKIING/chenkaixu/data/drive/sam3d_body_results_right \
      --gt-root /work/SKIING/chenkaixu/data/drive/sam3d_body_triangulated_gt \
      --train-subjects 01 03 --val-subjects 02 --max-frames-per-seq 300 \
      --epochs 3 --output-dir logs/learned_fusion_baseline_smoke
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

EVAL_DIR = Path(__file__).resolve().parent
REPO_ROOT = EVAL_DIR.parents[0]
for p in (str(REPO_ROOT), str(EVAL_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from eval_fusion_baselines_pesudo_gt import (  # noqa: E402
    CAMERAS,
    ENV_NAMES,
    canonicalize_pose,
    compute_metrics,
    fuse_views,
    list_sam3d_files,
    load_gt_sequence,
    load_selected_sam3d_frames,
    normalize_frame_id,
    select_common_frame_ids,
)

import torch  # noqa: E402
from torch import nn  # noqa: E402


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_sequence_arrays(
    subject_id: str,
    env_folder: str,
    sam3d_root: Path,
    gt_root: Path,
    args: argparse.Namespace,
    stride: int,
) -> Dict[str, np.ndarray]:
    """Return canonicalized view poses/confs and GT for one subject/env."""
    view_files = {}
    for cam in CAMERAS:
        view_dir = sam3d_root / subject_id / env_folder / cam
        if not view_dir.exists():
            raise FileNotFoundError(f"SAM3D directory not found: {view_dir}")
        view_files[cam] = list_sam3d_files(view_dir)

    gt_pose, gt_valid, gt_frame_ids = load_gt_sequence(
        gt_root / subject_id / env_folder / "keypoints_3d.npz"
    )
    frame_ids, gt_indices = select_common_frame_ids(
        view_files=view_files,
        gt_frame_ids=gt_frame_ids,
        gt_num_frames=gt_pose.shape[0],
        max_frames=None,
    )
    if stride > 1:
        frame_ids = frame_ids[::stride]
        gt_indices = gt_indices[::stride]
    if args.max_frames_per_seq and len(frame_ids) > args.max_frames_per_seq:
        keep = np.linspace(0, len(frame_ids) - 1, num=args.max_frames_per_seq, dtype=np.int64)
        frame_ids = [frame_ids[i] for i in keep]
        gt_indices = [gt_indices[i] for i in keep]

    normalized_view_files = {
        cam: {normalize_frame_id(fid): path for fid, path in files.items()}
        for cam, files in view_files.items()
    }
    loaded = {
        cam: load_selected_sam3d_frames(normalized_view_files[cam], frame_ids, num_workers=args.num_workers)
        for cam in CAMERAS
    }
    view_pose = np.stack([loaded[cam][0] for cam in CAMERAS], axis=2)  # (T, J, V, 3)
    view_conf = np.stack([loaded[cam][1] for cam in CAMERAS], axis=2)  # (T, J, V)
    gt_pose = gt_pose[gt_indices]
    gt_valid = gt_valid[gt_indices]

    n_joints = min(view_pose.shape[1], gt_pose.shape[1])
    view_pose = view_pose[:, :n_joints]
    view_conf = view_conf[:, :n_joints]
    gt_pose = gt_pose[:, :n_joints]
    gt_valid = gt_valid[:, :n_joints]

    gt_pose = canonicalize_pose(gt_pose)
    view_pose = np.stack(
        [canonicalize_pose(view_pose[:, :, v]) for v in range(view_pose.shape[2])], axis=2
    )
    return {
        "view_pose": view_pose.astype(np.float32),
        "view_conf": np.nan_to_num(view_conf, nan=0.0).astype(np.float32),
        "gt_pose": gt_pose.astype(np.float32),
        "gt_valid": gt_valid,
        "frame_ids": frame_ids,
    }


def load_split(
    items: List[Dict[str, Any]],
    sam3d_root: Path,
    gt_root: Path,
    args: argparse.Namespace,
    stride: int,
    subjects_filter: List[str] | None,
) -> List[Dict[str, Any]]:
    sequences = []
    for item in items:
        subject_id = str(item["person_id"])
        env_folder = str(item["env_folder"])
        if subjects_filter and subject_id not in subjects_filter:
            continue
        try:
            t0 = time.time()
            arrays = load_sequence_arrays(subject_id, env_folder, sam3d_root, gt_root, args, stride)
            arrays["person_id"] = subject_id
            arrays["env_folder"] = env_folder
            sequences.append(arrays)
            print(
                f"  loaded {subject_id}/{ENV_NAMES.get(env_folder, env_folder)}: "
                f"{arrays['view_pose'].shape[0]} frames ({time.time() - t0:.1f}s)",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  skip {subject_id}/{env_folder}: {exc}", flush=True)
    return sequences


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class LearnedFusion(nn.Module):
    """Per-joint confidence-weighted view gating + residual refinement."""

    def __init__(self, num_joints: int = 52, num_views: int = 3, emb_dim: int = 16, hidden: int = 64):
        super().__init__()
        self.num_joints = num_joints
        self.num_views = num_views
        self.joint_emb = nn.Embedding(num_joints, emb_dim)
        in_dim = 3 + 1 + emb_dim
        self.view_encoder = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU()
        )
        self.gate_head = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )
        self.refine_head = nn.Sequential(
            nn.Linear(3 + emb_dim + hidden, hidden), nn.ReLU(), nn.Linear(hidden, 3)
        )

    def forward(self, view_pose: torch.Tensor, view_conf: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # view_pose: (B, J, V, 3); view_conf: (B, J, V)
        b, j, v, _ = view_pose.shape
        joint_ids = torch.arange(j, device=view_pose.device)
        emb = self.joint_emb(joint_ids)[None, :, None, :].expand(b, j, v, -1)
        feats = torch.cat([view_pose, view_conf.unsqueeze(-1), emb], dim=-1)
        enc = self.view_encoder(feats)  # (B, J, V, H)
        ctx = enc.mean(dim=2, keepdim=True).expand(-1, -1, v, -1)
        logits = self.gate_head(torch.cat([enc, ctx], dim=-1)).squeeze(-1)  # (B, J, V)
        weights = torch.softmax(logits, dim=-1)
        fused = (view_pose * weights.unsqueeze(-1)).sum(dim=2)  # (B, J, 3)
        refine_in = torch.cat([fused, emb[:, :, 0, :], enc.mean(dim=2)], dim=-1)
        out = fused + self.refine_head(refine_in)
        return out, weights


# ---------------------------------------------------------------------------
# Train / evaluate
# ---------------------------------------------------------------------------

def build_tensors(sequences: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    view_pose = np.concatenate([s["view_pose"] for s in sequences], axis=0)
    view_conf = np.concatenate([s["view_conf"] for s in sequences], axis=0)
    gt_pose = np.concatenate([s["gt_pose"] for s in sequences], axis=0)
    gt_valid = np.concatenate([s["gt_valid"] for s in sequences], axis=0)
    finite = np.isfinite(view_pose).all(axis=(2, 3)) & np.isfinite(gt_pose).all(axis=-1)
    gt_valid = gt_valid & finite
    return {
        "view_pose": torch.from_numpy(np.nan_to_num(view_pose)),
        "view_conf": torch.from_numpy(view_conf),
        "gt_pose": torch.from_numpy(np.nan_to_num(gt_pose)),
        "gt_valid": torch.from_numpy(gt_valid),
    }


def train_model(model: nn.Module, data: Dict[str, torch.Tensor], args: argparse.Namespace) -> List[float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    n = data["view_pose"].shape[0]
    losses = []
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(n)
        epoch_loss, batches = 0.0, 0
        for start in range(0, n, args.batch_size):
            idx = perm[start : start + args.batch_size]
            pred, _ = model(data["view_pose"][idx], data["view_conf"][idx])
            valid = data["gt_valid"][idx]
            if not valid.any():
                continue
            err = (pred - data["gt_pose"][idx]).abs().sum(dim=-1)
            loss = err[valid].mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss)
            batches += 1
        mean_loss = epoch_loss / max(batches, 1)
        losses.append(mean_loss)
        print(f"epoch {epoch + 1}/{args.epochs}: masked L1 loss {mean_loss:.4f}", flush=True)
    return losses


@torch.no_grad()
def evaluate_split(
    model: nn.Module, sequences: List[Dict[str, Any]], args: argparse.Namespace
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    model.eval()
    rows = []
    for seq in sequences:
        vp = torch.from_numpy(np.nan_to_num(seq["view_pose"]))
        vc = torch.from_numpy(seq["view_conf"])
        preds = []
        for start in range(0, vp.shape[0], args.batch_size):
            pred, _ = model(vp[start : start + args.batch_size], vc[start : start + args.batch_size])
            preds.append(pred)
        pred = torch.cat(preds, dim=0).numpy()
        metrics = compute_metrics(pred=pred, gt=seq["gt_pose"], valid_mask=seq["gt_valid"], root_index=args.root_index)
        if not metrics:
            continue
        rows.append(
            {
                "person_id": seq["person_id"],
                "environment": seq["env_folder"],
                "environment_name": ENV_NAMES.get(seq["env_folder"], seq["env_folder"]),
                "metrics": metrics,
            }
        )
    keys = ("mpjpe_m", "median_error_m", "pa_mpjpe_m", "mpjpe_pa_frames_m", "auc_0.15")
    summary = {}
    for key in keys:
        values = [row["metrics"].get(key) for row in rows]
        values = [v for v in values if v is not None]
        summary[key] = float(np.mean(values)) if values else None
    summary["num_rows"] = len(rows)
    return rows, summary


@torch.no_grad()
def evaluate_fixed_confidence(
    sequences: List[Dict[str, Any]], args: argparse.Namespace
) -> Dict[str, Any]:
    """Fixed confidence-weighted fusion on the same loaded frames (sanity anchor)."""
    values: Dict[str, List[float]] = {"mpjpe_m": [], "pa_mpjpe_m": []}
    for seq in sequences:
        pred = fuse_views(seq["view_pose"], seq["view_conf"], "confidence")
        metrics = compute_metrics(pred=pred, gt=seq["gt_pose"], valid_mask=seq["gt_valid"], root_index=args.root_index)
        for key in values:
            if metrics.get(key) is not None:
                values[key].append(metrics[key])
    return {key: (float(np.mean(vals)) if vals else None) for key, vals in values.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Learned confidence-weighted fusion baseline (rebuttal E).")
    parser.add_argument("--sam3d-root", type=str, default="/work/SKIING/chenkaixu/data/drive/sam3d_body_results_right")
    parser.add_argument("--gt-root", type=str, default="/work/SKIING/chenkaixu/data/drive/sam3d_body_triangulated_gt")
    parser.add_argument("--index-mapping", type=str, default="/work/SKIING/chenkaixu/data/drive/index_mapping/fold_0.json")
    parser.add_argument("--output-dir", type=str, default=str(EVAL_DIR / "logs" / "learned_fusion_baseline"))
    parser.add_argument("--train-stride", type=int, default=5, help="Frame stride for the training split.")
    parser.add_argument("--eval-stride", type=int, default=1, help="Frame stride for the val split (1 = all frames).")
    parser.add_argument("--max-frames-per-seq", type=int, default=None, help="Debug cap per sequence (uniform).")
    parser.add_argument("--train-subjects", type=str, nargs="*", default=None, help="Debug subject filter (train).")
    parser.add_argument("--val-subjects", type=str, nargs="*", default=None, help="Debug subject filter (val).")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--root-index", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(args.torch_threads)

    sam3d_root = Path(args.sam3d_root)
    gt_root = Path(args.gt_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.index_mapping, "r", encoding="utf-8") as f:
        fold = json.load(f)

    print("Loading train split...", flush=True)
    train_seqs = load_split(fold["train"], sam3d_root, gt_root, args, args.train_stride, args.train_subjects)
    print("Loading val split...", flush=True)
    val_seqs = load_split(fold["val"], sam3d_root, gt_root, args, args.eval_stride, args.val_subjects)
    if not train_seqs or not val_seqs:
        raise RuntimeError(f"Empty split: train={len(train_seqs)} val={len(val_seqs)}")

    train_persons = sorted({s["person_id"] for s in train_seqs})
    val_persons = sorted({s["person_id"] for s in val_seqs})
    assert not set(train_persons) & set(val_persons), "train/val subject leak"
    n_train_frames = sum(s["view_pose"].shape[0] for s in train_seqs)
    print(f"train: {len(train_seqs)} seqs / {n_train_frames} frames / persons {train_persons}", flush=True)
    print(f"val:   {len(val_seqs)} seqs / persons {val_persons}", flush=True)

    model = LearnedFusion(num_joints=train_seqs[0]["view_pose"].shape[1])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model parameters: {n_params}", flush=True)

    data = build_tensors(train_seqs)
    losses = train_model(model, data, args)

    rows, summary = evaluate_split(model, val_seqs, args)
    anchor = evaluate_fixed_confidence(val_seqs, args)
    print(f"val summary: {json.dumps(summary, indent=2)}", flush=True)
    print(f"fixed confidence-weighted anchor on same frames: {anchor}", flush=True)

    torch.save(
        {"state_dict": model.state_dict(), "num_params": n_params, "args": vars(args)},
        output_dir / "learned_fusion_baseline.ckpt",
    )
    payload = {
        "protocol": {
            "fold": args.index_mapping,
            "train_persons": train_persons,
            "val_persons": val_persons,
            "train_frames": n_train_frames,
            "train_stride": args.train_stride,
            "eval_stride": args.eval_stride,
            "canonicalize": True,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "num_params": n_params,
        },
        "train_losses": losses,
        "val_summary": summary,
        "fixed_confidence_anchor": anchor,
        "val_rows": rows,
    }
    with open(output_dir / "learned_fusion_baseline.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    md_lines = [
        "# Learned confidence-weighted fusion baseline (rebuttal E)",
        "",
        f"- params: {n_params}; fold-0 protocol; train persons {train_persons}; val persons {val_persons}",
        f"- train frames: {n_train_frames} (stride {args.train_stride}); epochs {args.epochs}",
        "",
        "| metric | learned fusion | fixed confidence (same frames) |",
        "|---|---|---|",
        f"| MPJPE (m) | {summary.get('mpjpe_m')} | {anchor.get('mpjpe_m')} |",
        f"| PA-MPJPE (m) | {summary.get('pa_mpjpe_m')} | {anchor.get('pa_mpjpe_m')} |",
        f"| MPJPE@PA-mask (m) | {summary.get('mpjpe_pa_frames_m')} | - |",
        f"| AUC@0.15 | {summary.get('auc_0.15')} | - |",
    ]
    with open(output_dir / "learned_fusion_baseline.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")
    print(f"Saved outputs to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
