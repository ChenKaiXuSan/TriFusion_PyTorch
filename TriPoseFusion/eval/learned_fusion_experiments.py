#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Triangulation-supervised learned fusion: experiments for the re-scoped paper.

Sub-commands (all CPU, a few minutes each):
  train        k-fold training with a FIXED number of steps (no test-fold model
               selection), plus training-free rows (single views / best single /
               mean / median) on exactly the same cached val sequences and the same
               compute_metrics protocol.
  stress       view-corruption robustness on the trained fold models: one view is
               dropped (NaN), zeroed, or noised; reports MPJPE / PA and the learned
               weight assigned to the corrupted view, next to mean fusion under the
               same corruption.
  driveandact  zero-shot transfer of a fold model to the Drive&Act dense clips
               (hs10 PA-MPJPE), next to single views and mean fusion in the same
               canonical space.
  summarize    aggregate fold results (mean +- std) into markdown.

Examples:
  python learned_fusion_experiments.py train --folds 0 1 2 3 4 --tag main
  python learned_fusion_experiments.py stress --folds 0 1 2 3 4 --tag main --mode zero
  python learned_fusion_experiments.py driveandact --tag main --fold 0
  python learned_fusion_experiments.py summarize --tag main
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for p in (str(SCRIPT_DIR), str(REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from eval_fusion_baselines_pesudo_gt import canonicalize_pose, compute_metrics, fuse_views  # noqa: E402
from learned_fusion_baseline import LearnedFusion, SeqStore, load_fold_pairs, masked_loss, summarize  # noqa: E402

VIEWS = ("front", "left", "right")
DEFAULT_CACHE = Path("/work/1/SKIING/chenkaixu/data/drive/learned_fusion_cache")
DEFAULT_INDEX = Path("/work/1/SKIING/chenkaixu/data/drive/index_mapping")
DEFAULT_DA_ROOT = Path("/work/1/SKIING/chenkaixu/data/drive/driveandact/tripose_eval_dense")
DEFAULT_OUT = SCRIPT_DIR / "logs" / "learned_fusion_experiments"


# ------------------------------------------------------------------ helpers


def _metrics_rows(store: SeqStore, predict, pck) -> list[dict]:
    rows = []
    for s in store.seqs:
        pred = predict(s)
        m = compute_metrics(pred=pred, gt=s["gt_pose"], valid_mask=s["gt_valid"], pck_thresholds=tuple(pck))
        if m:
            rows.append({"person": s["person"], "env": s["env"], "metrics": m})
    return rows


def training_free_rows(store: SeqStore, pck) -> dict[str, dict]:
    """Same cache, same compute_metrics: single views, oracle best single, mean, median."""
    out = {}

    def _summ(rows):
        agg = summarize(rows)
        agg["groups"] = group_mpjpe(rows)
        return agg

    for vi, view in enumerate(VIEWS):
        out[f"single_{view}"] = _summ(_metrics_rows(store, lambda s, vi=vi: s["view_pose"][:, :, vi], pck))
    out["fuse_mean"] = _summ(_metrics_rows(store, lambda s: fuse_views(s["view_pose"], s["view_conf"], "mean"), pck))
    out["fuse_median"] = _summ(_metrics_rows(store, lambda s: fuse_views(s["view_pose"], s["view_conf"], "median"), pck))
    # oracle best single view per sequence (lowest MPJPE)
    best_rows = []
    for s in store.seqs:
        cands = []
        for vi in range(len(VIEWS)):
            m = compute_metrics(pred=s["view_pose"][:, :, vi], gt=s["gt_pose"], valid_mask=s["gt_valid"], pck_thresholds=tuple(pck))
            if m:
                cands.append(m)
        if cands:
            best_rows.append({"person": s["person"], "env": s["env"], "metrics": min(cands, key=lambda m: m["mpjpe_m"])})
    out["best_single_oracle"] = summarize(best_rows)
    out["best_single_oracle"]["groups"] = group_mpjpe(best_rows)
    return out


JOINT_GROUPS = {"head": list(range(0, 5)), "shoulders_neck": [5, 6, 49, 50, 51],
                "body": list(range(0, 7)) + [49, 50, 51], "hands": list(range(7, 49))}


def group_mpjpe(rows: list[dict]) -> dict[str, float]:
    """Mean over sequences of the mean per-joint MPJPE within each joint group."""
    out = {}
    for g, idx in JOINT_GROUPS.items():
        vals = []
        for r in rows:
            pj = r["metrics"].get("per_joint_mpjpe_m")
            if pj:
                v = [pj[i] for i in idx if i < len(pj) and pj[i] is not None]
                if v:
                    vals.append(float(np.mean(v)))
        out[g] = float(np.mean(vals)) if vals else float("nan")
    return out


def learned_rows(model: LearnedFusion, store: SeqStore, pck, corrupt=None) -> tuple[list[dict], np.ndarray]:
    """Evaluate model on store; corrupt(view_pose)->view_pose optionally applied. Returns rows, mean weights (V,)."""
    model.eval()
    rows, wsum, wn = [], np.zeros(len(VIEWS)), 0
    with torch.no_grad():
        for s in store.seqs:
            vp = s["view_pose"] if corrupt is None else corrupt(s["view_pose"])
            pred, w = model(torch.from_numpy(vp).unsqueeze(0), torch.from_numpy(s["view_conf"]).unsqueeze(0))
            pred = pred.squeeze(0).numpy()
            w = w.squeeze(0).numpy()  # (T,J,V)
            finite_any = np.isfinite(vp).all(-1).any(-1)  # (T,J)
            wsum += w[finite_any].sum(axis=0)
            wn += int(finite_any.sum())
            m = compute_metrics(pred=pred, gt=s["gt_pose"], valid_mask=s["gt_valid"], pck_thresholds=tuple(pck))
            if m:
                rows.append({"person": s["person"], "env": s["env"], "metrics": m})
    model.train()
    return rows, wsum / max(wn, 1)


class ResidualFusion(LearnedFusion):
    """LearnedFusion + per-joint residual refinement head: after weighted fusion, an MLP
    predicts a 3D correction from [fused, per-view canonical coords, per-view disagreement,
    finite flags, joint embedding]. Tests whether capacity can correct systematic monocular
    errors (depth, hands) beyond view weighting, under correct pseudo-GT supervision."""

    def __init__(self, residual_hidden: int = 64, **kw) -> None:
        super().__init__(**kw)
        emb = self.joint_emb.embedding_dim if self.joint_emb is not None else 0
        in_dim = 3 + 3 * 3 + 3 + 3 + emb  # fused + 3 views*3 + disagreement + finite + emb
        self.residual = torch.nn.Sequential(
            torch.nn.Linear(in_dim, residual_hidden), torch.nn.GELU(),
            torch.nn.Linear(residual_hidden, residual_hidden), torch.nn.GELU(),
            torch.nn.Linear(residual_hidden, 3),
        )
        torch.nn.init.zeros_(self.residual[-1].weight)
        torch.nn.init.zeros_(self.residual[-1].bias)

    def forward(self, view_pose: torch.Tensor, view_conf: torch.Tensor):
        fused, weights = super().forward(view_pose, view_conf)
        finite = torch.isfinite(view_pose).all(dim=-1)  # (B,T,J,V)
        pose = torch.nan_to_num(view_pose, nan=0.0)
        fused0 = torch.nan_to_num(fused, nan=0.0)
        disagreement = torch.linalg.norm(pose - fused0.unsqueeze(-2), dim=-1) * finite  # (B,T,J,V)
        b, t, j, v, _ = pose.shape
        feats = [fused0, pose.reshape(b, t, j, v * 3), disagreement, finite.float()]
        if self.joint_emb is not None:
            feats.append(self.joint_emb(torch.arange(j, device=pose.device))[None, None].expand(b, t, j, -1))
        delta = self.residual(torch.cat(feats, dim=-1))
        return fused + delta, weights


class SetFusion(torch.nn.Module):
    """Triangulation-distilled set fusion (v2 model).

    Per joint, the V monocular estimates are tokens [canonical xyz, disagreement to the
    finite-view mean, finite flag] + view id + joint id embeddings. A masked transformer
    across the view set (permutation-equivariant, any subset of views) yields gate logits
    and a pooled feature; a transformer across the J joints propagates context along the
    skeleton (hands from wrists); a depthwise temporal conv adds motion context. Two heads:
    a 3D residual correction on top of the gated fusion and a per-joint log-scale of a
    Laplace likelihood (uncertainty), trained with masked NLL against the pseudo-GT.
    """

    def __init__(self, d: int = 64, heads: int = 4, view_layers: int = 2, joint_layers: int = 1,
                 num_joints: int = 52, num_views: int = 3, temporal: bool = True, kernel: int = 5,
                 uncertainty: bool = True) -> None:
        super().__init__()
        self.d, self.uncertainty = d, uncertainty
        self.in_proj = torch.nn.Linear(5, d)
        self.view_emb = torch.nn.Embedding(num_views, d)
        self.joint_emb = torch.nn.Embedding(num_joints, d)
        mk = lambda: torch.nn.TransformerEncoderLayer(d, heads, dim_feedforward=2 * d, dropout=0.0, batch_first=True)
        self.view_layers = torch.nn.ModuleList([mk() for _ in range(view_layers)])
        self.joint_layers = torch.nn.ModuleList([mk() for _ in range(joint_layers)])
        self.gate = torch.nn.Linear(d, 1)
        self.temporal = temporal
        if temporal:
            self.tconv = torch.nn.Conv1d(d, d, kernel, padding=kernel // 2, groups=d)
            torch.nn.init.zeros_(self.tconv.weight)
            torch.nn.init.zeros_(self.tconv.bias)
        self.head = torch.nn.Linear(d, 4 if uncertainty else 3)
        torch.nn.init.zeros_(self.head.weight)
        torch.nn.init.zeros_(self.head.bias)
        self.last_logscale = None

    def forward(self, view_pose: torch.Tensor, view_conf: torch.Tensor):
        b, t, j, v, _ = view_pose.shape
        finite = torch.isfinite(view_pose).all(dim=-1)  # (B,T,J,V)
        pose = torch.nan_to_num(view_pose, nan=0.0)
        n_finite = finite.sum(dim=-1, keepdim=True).clamp_min(1)
        mean_pose = (pose * finite.unsqueeze(-1)).sum(dim=-2) / n_finite
        disagreement = torch.linalg.norm(pose - mean_pose.unsqueeze(-2), dim=-1) * finite
        feats = torch.cat([pose, disagreement.unsqueeze(-1), finite.float().unsqueeze(-1)], dim=-1)  # (B,T,J,V,5)
        x = self.in_proj(feats) + self.view_emb.weight[None, None, None] + self.joint_emb.weight[None, None, :, None]
        x = x.reshape(b * t * j, v, self.d)
        any_finite = finite.any(dim=-1)  # (B,T,J)
        pad = (~finite).reshape(b * t * j, v)
        pad = pad & any_finite.reshape(-1, 1)  # rows with no finite view: keep unmasked to avoid NaN
        for layer in self.view_layers:
            x = layer(x, src_key_padding_mask=pad)
        logits = self.gate(x).squeeze(-1).masked_fill(pad, -1e4)
        weights = torch.softmax(logits, dim=-1)  # (BTJ,V)
        fused = (pose.reshape(b * t * j, v, 3) * weights.unsqueeze(-1)).sum(dim=1)  # (BTJ,3)
        pooled = (x * weights.unsqueeze(-1)).sum(dim=1)  # (BTJ,d)
        h = pooled.reshape(b * t, j, self.d)
        for layer in self.joint_layers:
            h = layer(h)
        h = h.reshape(b, t, j, self.d)
        if self.temporal:
            ht = h.permute(0, 2, 3, 1).reshape(b * j, self.d, t)
            h = h + self.tconv(ht).reshape(b, j, self.d, t).permute(0, 3, 1, 2)
        out = self.head(h)
        pred = fused.reshape(b, t, j, 3) + out[..., :3]
        self.last_logscale = out[..., 3] if self.uncertainty else None
        pred = torch.where(any_finite.unsqueeze(-1).expand_as(pred), pred, torch.nan)
        return pred, weights.reshape(b, t, j, v)


def masked_laplace_nll(pred, gt, valid, logscale):
    mask = valid & torch.isfinite(gt).all(dim=-1) & torch.isfinite(pred).all(dim=-1)
    if not mask.any():
        return pred.new_zeros(())
    s = logscale[mask].clamp(-6, 3)
    err = (pred[mask] - gt[mask]).abs().sum(dim=-1)
    return (err * torch.exp(-s) + 3 * s).mean()


def uncertainty_calibration(model, store: SeqStore) -> dict:
    """Spearman correlation of predicted scale vs. actual error, and error of most/least confident deciles."""
    if getattr(model, "last_logscale", None) is None and not isinstance(model, SetFusion):
        return {}
    errs, scales = [], []
    model.eval()
    with torch.no_grad():
        for s in store.seqs:
            pred, _ = model(torch.from_numpy(s["view_pose"]).unsqueeze(0), torch.from_numpy(s["view_conf"]).unsqueeze(0))
            ls = model.last_logscale
            if ls is None:
                return {}
            pred, ls = pred.squeeze(0).numpy(), ls.squeeze(0).numpy()
            gt, valid = s["gt_pose"], s["gt_valid"] & np.isfinite(pred).all(-1) & np.isfinite(s["gt_pose"]).all(-1)
            errs.append(np.linalg.norm(pred - gt, axis=-1)[valid])
            scales.append(ls[valid])
    model.train()
    e, sc = np.concatenate(errs), np.concatenate(scales)
    from scipy.stats import spearmanr
    rho = float(spearmanr(sc, e).correlation)
    order = np.argsort(sc)
    k = max(len(e) // 10, 1)
    return {"spearman_scale_vs_error": rho, "err_most_confident_decile": float(e[order[:k]].mean()),
            "err_least_confident_decile": float(e[order[-k:]].mean()), "err_all": float(e.mean())}


class TriPoseWrapper(torch.nn.Module):
    """The ORIGINAL TriPoseFusion architecture (view encoder -> cross-view attention ->
    joint-wise gate -> dilated TCN), re-trained here with triangulation supervision instead
    of the degenerate median teacher, without InfoNCE, on already-canonical cache inputs.
    Lets the paper keep the submitted architecture while fixing its training."""

    def __init__(self, hidden: int = 64, refiner_dim: int = 128, refiner_layers: int = 4,
                 velocity: bool = False, cross_view_attention: bool = True, temporal: bool = True) -> None:
        super().__init__()
        from omegaconf import OmegaConf
        from models.keypoint_mlp import TriViewKeypointFusionNet
        cfg = OmegaConf.load(REPO_ROOT / "configs" / "train.yaml")
        cfg.train.view_name = list(VIEWS)
        m = cfg.model
        m.geofusion_view_names = list(VIEWS)
        m.geofusion_use_2d = False
        m.geofusion_use_conf = False
        m.geofusion_use_reproj_error_feature = False
        m.geofusion_canonicalize = False  # cache inputs are already canonical
        m.geofusion_hidden_dim = int(hidden)
        m.geofusion_refiner_dim = int(refiner_dim)
        m.geofusion_refiner_layers = int(refiner_layers)
        m.geofusion_use_multiscale_velocity = bool(velocity)
        m.geofusion_use_cross_view_attention = bool(cross_view_attention)
        m.geofusion_use_temporal_refiner = bool(temporal)
        m.geofusion_use_dilated_refiner = True
        m.geofusion_use_learned_gate = True
        m.geofusion_dropout = 0.0
        self.net = TriViewKeypointFusionNet(cfg)
        self.last_logscale = None

    def forward(self, view_pose: torch.Tensor, view_conf: torch.Tensor):
        finite = torch.isfinite(view_pose).all(dim=-1)  # (B,T,J,V)
        pose = torch.nan_to_num(view_pose, nan=0.0)
        view_mask = finite.any(dim=(1, 2))  # (B,V): a view that is entirely missing is masked
        if bool(view_mask.all()):
            view_mask = None
        out = self.net(pose3d={v: pose[:, :, :, i] for i, v in enumerate(VIEWS)}, view_mask=view_mask)
        pred = out["P_final"]
        any_finite = finite.any(dim=-1, keepdim=True)
        pred = torch.where(any_finite.expand_as(pred), pred, torch.nan)
        return pred, out["alpha"]


class _ZeroHead(torch.nn.Module):
    """Replaces weight_mlp: constant logits -> uniform softmax over finite views."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.new_zeros(x.shape[:-1] + (1,))


def build_model(args) -> LearnedFusion:
    if getattr(args, "model", "learned") == "tripose":
        return TriPoseWrapper(hidden=args.hidden, refiner_dim=args.refiner_dim, velocity=args.tripose_velocity,
                              cross_view_attention=not args.no_cross_view_attention, temporal=args.temporal)
    if getattr(args, "model", "learned") == "setfusion":
        return SetFusion(d=args.hidden if args.hidden >= 16 else 64, temporal=args.temporal,
                         view_layers=args.view_layers, joint_layers=args.joint_layers,
                         uncertainty=not args.no_uncertainty)
    kw = dict(hidden=args.hidden, temporal=args.temporal, joint_emb=args.joint_emb)
    rh = getattr(args, "residual_hidden", 0)
    model = ResidualFusion(residual_hidden=rh, **kw) if rh > 0 else LearnedFusion(**kw)
    if getattr(args, "uniform_weights", False):
        model.weight_mlp = _ZeroHead()
        model.joint_emb = None
    return model


def augment_batch(batch: dict, prob: float, rng: np.random.Generator) -> dict:
    """Training-time view corruption: with prob per sample, corrupt one random view
    (drop -> NaN, zero, or noise sigma=0.1 m) so the gate can learn to detect it."""
    if prob <= 0:
        return batch
    vp = batch["view_pose"].clone()
    for i in range(vp.shape[0]):
        if rng.random() >= prob:
            continue
        v = int(rng.integers(vp.shape[3]))
        mode = ("drop", "zero", "noise")[int(rng.integers(3))]
        if mode == "drop":
            vp[i, :, :, v] = float("nan")
        elif mode == "zero":
            vp[i, :, :, v] = 0.0
        else:
            vp[i, :, :, v] = vp[i, :, :, v] + 0.1 * torch.randn_like(vp[i, :, :, v])
    out = dict(batch)
    out["view_pose"] = vp
    return out


def fold_dir(args, fold: int) -> Path:
    return Path(args.output_dir) / args.tag / f"fold{fold}"


def load_fold_model(args, fold: int) -> LearnedFusion:
    model = build_model(args)
    model.load_state_dict(torch.load(fold_dir(args, fold) / "model.pt", map_location="cpu"))
    model.eval()
    return model


# ------------------------------------------------------------------ train


def cmd_train(args) -> None:
    for fold in args.folds:
        torch.manual_seed(args.seed)
        rng = np.random.default_rng(args.seed + fold)
        train_pairs, val_pairs = load_fold_pairs(args.index_mapping, fold)
        train_store = SeqStore(args.cache_dir, train_pairs)
        val_store = SeqStore(args.cache_dir, val_pairs)
        print(f"[fold {fold}] train {len(train_store.seqs)} / val {len(val_store.seqs)} sequences", flush=True)

        model = build_model(args)
        n_params = sum(p.numel() for p in model.parameters())
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        history, best = [], None
        t0 = time.time()
        for step in range(1, args.steps + 1):
            batch = augment_batch(train_store.sample_windows(rng, args.batch, args.window), args.augment_prob, rng)
            pred, _ = model(batch["view_pose"], batch["view_conf"])
            if getattr(model, "last_logscale", None) is not None:
                loss = masked_laplace_nll(pred, batch["gt_pose"], batch["gt_valid"], model.last_logscale)
            else:
                loss = masked_loss(pred, batch["gt_pose"], batch["gt_valid"])
            opt.zero_grad()
            loss.backward()
            opt.step()
            if step % args.eval_every == 0 or step == args.steps:
                rows, wmean = learned_rows(model, val_store, args.pck)
                agg = summarize(rows)
                history.append({"step": step, "loss": float(loss.item()), "mpjpe_m": agg["mpjpe_m"], "pa_mpjpe_m": agg["pa_mpjpe_m"]})
                print(f"[fold {fold}] step {step} loss {loss.item():.4f} val MPJPE {agg['mpjpe_m']:.4f} PA {agg['pa_mpjpe_m']:.4f} ({time.time()-t0:.0f}s)", flush=True)
                if best is None or agg["mpjpe_m"] < best["mpjpe_m"]:
                    best = {"step": step, **agg}
        # fixed-step model (no selection on the evaluation fold)
        rows, wmean = learned_rows(model, val_store, args.pck)
        agg = summarize(rows)
        agg["groups"] = group_mpjpe(rows)
        anchors = training_free_rows(val_store, args.pck)

        out = fold_dir(args, fold)
        out.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), out / "model.pt")
        payload = {
            "fold": fold, "params": n_params, "steps": args.steps, "train_seconds": time.time() - t0,
            "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items() if k != "func"},
            "learned_last": agg,
            "uncertainty_calibration": uncertainty_calibration(model, val_store),
            "learned_best_val_for_reference": best,
            "mean_view_weights": dict(zip(VIEWS, wmean.tolist())),
            "training_free": anchors,
            "history": history,
            "per_sequence_learned": rows,
        }
        (out / "result.json").write_text(json.dumps(payload, indent=1, ensure_ascii=False))
        print(f"[fold {fold}] learned(last) {agg['mpjpe_m']:.4f}/{agg['pa_mpjpe_m']:.4f} | mean {anchors['fuse_mean']['mpjpe_m']:.4f}/{anchors['fuse_mean']['pa_mpjpe_m']:.4f} | best single {anchors['best_single_oracle']['mpjpe_m']:.4f} | weights {np.round(wmean, 3).tolist()}", flush=True)


# ------------------------------------------------------------------ stress


def _corruptor(view_idx: int, mode: str, rng: np.random.Generator):
    def f(vp: np.ndarray) -> np.ndarray:
        vp = vp.copy()
        if mode == "drop":
            vp[:, :, view_idx] = np.nan
        elif mode == "zero":
            vp[:, :, view_idx] = 0.0
        elif mode == "noise":
            vp[:, :, view_idx] = vp[:, :, view_idx] + rng.normal(scale=0.1, size=vp[:, :, view_idx].shape).astype(np.float32)
        else:
            raise ValueError(mode)
        return vp
    return f


def cmd_stress(args) -> None:
    for fold in args.folds:
        model = load_fold_model(args, fold)
        _, val_pairs = load_fold_pairs(args.index_mapping, fold)
        store = SeqStore(args.cache_dir, val_pairs)
        rng = np.random.default_rng(args.seed)
        res = {}
        conds = [("clean", None)] + [(v, _corruptor(i, args.mode, rng)) for i, v in enumerate(VIEWS)]
        for name, corrupt in conds:
            rows, wmean = learned_rows(model, store, args.pck, corrupt)
            learned = summarize(rows)
            mean_rows = _metrics_rows(store, lambda s, c=corrupt: fuse_views(s["view_pose"] if c is None else c(s["view_pose"]), s["view_conf"], "mean"), args.pck)
            excl = None
            if name != "clean":
                vi = VIEWS.index(name)
                keep = [i for i in range(len(VIEWS)) if i != vi]
                excl = summarize(_metrics_rows(store, lambda s, k=keep: np.nanmean(s["view_pose"][:, :, k], axis=2), args.pck))
            res[name] = {
                "learned": learned,
                "learned_mean_weights": dict(zip(VIEWS, wmean.tolist())),
                "weight_on_corrupted_view": (None if name == "clean" else float(wmean[VIEWS.index(name)])),
                "fuse_mean_same_corruption": summarize(mean_rows),
                "fuse_mean_excluding_view_oracle": excl,
            }
            print(f"[fold {fold}] {args.mode:5s} {name:6s} learned {learned['mpjpe_m']:.4f}/{learned['pa_mpjpe_m']:.4f} w={np.round(wmean,3).tolist()} | mean {res[name]['fuse_mean_same_corruption']['mpjpe_m']:.4f}", flush=True)
        (fold_dir(args, fold) / f"stress_{args.mode}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))


# ------------------------------------------------------------------ Drive&Act


def cmd_driveandact(args) -> None:
    from eval_driveandact_model import HS10_NAMES, build_windows, load_gt, load_view, metrics, to_hs10  # noqa: E402

    model = load_fold_model(args, args.fold)
    results, pooled, pooled_gt, pooled_valid = {}, {}, [], []
    wsum, wn = np.zeros(len(VIEWS)), 0
    for gt_npz in sorted((args.da_root / "gt").glob("vp*/*.npz")):
        vp, run = gt_npz.parent.name, gt_npz.stem
        sam_root = args.da_root / "sam3d" / vp / run
        if not all((sam_root / c).is_dir() for c in VIEWS):
            continue
        views = {c: load_view(sam_root / c) for c in VIEWS}
        gt, gt_valid = load_gt(gt_npz)
        ok = set(gt)
        for v in views.values():
            ok &= set(v)
        windows = build_windows(ok)
        if not windows:
            continue
        preds = {k: [] for k in ["learned", "fuse_mean", "fuse_median", *[f"single_{c}" for c in VIEWS]]}
        gts, valids = [], []
        with torch.no_grad():
            for w in windows:
                canon = {c: canonicalize_pose(np.stack([views[c][f][1] for f in w]).astype(np.float32)) for c in VIEWS}
                vp_arr = np.stack([canon[c] for c in VIEWS], axis=2)  # (T,J,V,3)
                conf = np.ones(vp_arr.shape[:3], dtype=np.float32)
                fused, wts = model(torch.from_numpy(vp_arr).unsqueeze(0), torch.from_numpy(conf).unsqueeze(0))
                fused = fused.squeeze(0).numpy()
                wts = wts.squeeze(0).numpy()
                wsum += wts.reshape(-1, len(VIEWS)).sum(axis=0)
                wn += wts.shape[0] * wts.shape[1]
                preds["learned"].append(to_hs10(fused[None]).reshape(-1, len(HS10_NAMES), 3))
                preds["fuse_mean"].append(to_hs10(np.nanmean(vp_arr, axis=2)[None]).reshape(-1, len(HS10_NAMES), 3))
                preds["fuse_median"].append(to_hs10(np.nanmedian(vp_arr, axis=2)[None]).reshape(-1, len(HS10_NAMES), 3))
                for c in VIEWS:
                    preds[f"single_{c}"].append(to_hs10(canon[c][None]).reshape(-1, len(HS10_NAMES), 3))
                gts.extend(gt[f] for f in w)
                valids.extend(gt_valid[f] for f in w)
        gt_arr, valid_arr = np.stack(gts), np.stack(valids)
        run_res = {"num_windows": len(windows), "num_frames": len(gts)}
        for k, chunks in preds.items():
            arr = np.concatenate(chunks)
            run_res[k] = metrics(arr, gt_arr, valid_arr)
            pooled.setdefault(k, []).append(arr)
        # scale diagnostic: mean radial extent of learned vs mean-fusion output (canonical space)
        l_arr, m_arr = np.concatenate(preds["learned"]), np.concatenate(preds["fuse_mean"])
        run_res["learned_over_mean_scale_ratio"] = float(
            np.nanmean(np.linalg.norm(l_arr, axis=-1)) / np.nanmean(np.linalg.norm(m_arr, axis=-1))
        )
        pooled_gt.append(gt_arr)
        pooled_valid.append(valid_arr)
        results[f"{vp}/{run}"] = run_res
        print(f"{vp}/{run}: learned PA={run_res['learned']['pa_mpjpe_m']*1000:.1f} mm  mean PA={run_res['fuse_mean']['pa_mpjpe_m']*1000:.1f} mm", flush=True)
    gt_all, valid_all = np.concatenate(pooled_gt), np.concatenate(pooled_valid)
    overall = {k: metrics(np.concatenate(v), gt_all, valid_all) for k, v in pooled.items()}
    overall["learned"]["mean_view_weights"] = dict(zip(VIEWS, (wsum / max(wn, 1)).tolist()))
    results["overall"] = overall
    results["protocol"] = {"joints": HS10_NAMES, "fold_model": args.fold, "canonicalize": "numpy canonicalize_pose (same as training cache)"}
    out = fold_dir(args, args.fold) / "driveandact.json"
    out.write_text(json.dumps(results, indent=1))
    print("== Drive&Act pooled hs10 ==")
    for k, m in overall.items():
        print(f"{k:13s} PA-MPJPE={m['pa_mpjpe_m']*1000:6.1f} mm  rigid-MPJPE={m['rigid_mpjpe_m']*1000:6.1f} mm")
    print("learned mean view weights:", overall["learned"]["mean_view_weights"])


# ------------------------------------------------------------------ summarize


def cmd_summarize(args) -> None:
    base = Path(args.output_dir) / args.tag
    folds = sorted(int(p.name[4:]) for p in base.glob("fold*") if (p / "result.json").exists())
    table: dict[str, dict[str, list]] = {}
    weights = []
    for f in folds:
        r = json.loads((base / f"fold{f}" / "result.json").read_text())
        rows = dict(r["training_free"])
        rows["learned (fixed steps)"] = r["learned_last"]
        weights.append([r["mean_view_weights"][v] for v in VIEWS])
        for name, agg in rows.items():
            t = table.setdefault(name, {"mpjpe": [], "pa": [], "pck10": []})
            t["mpjpe"].append(agg["mpjpe_m"])
            t["pa"].append(agg["pa_mpjpe_m"])
            t["pck10"].append(agg["pck"].get("0.10", float("nan")))
    groups: dict[str, dict[str, list]] = {}
    calib = []
    for f in folds:
        r = json.loads((base / f"fold{f}" / "result.json").read_text())
        rows = dict(r["training_free"])
        rows["learned (fixed steps)"] = r["learned_last"]
        for name, agg in rows.items():
            if "groups" in agg:
                g = groups.setdefault(name, {k: [] for k in JOINT_GROUPS})
                for k in JOINT_GROUPS:
                    g[k].append(agg["groups"].get(k, float("nan")))
        if r.get("uncertainty_calibration"):
            calib.append(r["uncertainty_calibration"])
    lines = [f"folds: {folds}", "", "| method | MPJPE (m) | PA-MPJPE (m) | PCK@0.10 |", "|---|---|---|---|"]
    for name, t in table.items():
        f = lambda v: f"{np.mean(v):.4f} ± {np.std(v):.4f}"
        lines.append(f"| {name} | {f(t['mpjpe'])} | {f(t['pa'])} | {f(t['pck10'])} |")
    if groups:
        lines += ["", "| method | head | shoulders/neck | body | hands |", "|---|---|---|---|---|"]
        for name, g in groups.items():
            lines.append(f"| {name} | " + " | ".join(f"{np.nanmean(g[k]):.4f}" for k in JOINT_GROUPS) + " |")
    if calib:
        lines += ["", f"uncertainty calibration (mean over folds): spearman(scale, error)={np.mean([c['spearman_scale_vs_error'] for c in calib]):.3f}, "
                  f"error most-confident decile {np.mean([c['err_most_confident_decile'] for c in calib]):.4f} vs least-confident {np.mean([c['err_least_confident_decile'] for c in calib]):.4f} (all {np.mean([c['err_all'] for c in calib]):.4f})"]
    w = np.array(weights)
    lines.append("")
    lines.append(f"learned mean view weights (front/left/right): {np.round(w.mean(0), 3).tolist()} ± {np.round(w.std(0), 3).tolist()}")
    for mode in ("drop", "zero", "noise"):
        files = [base / f"fold{f}" / f"stress_{mode}.json" for f in folds]
        files = [p for p in files if p.exists()]
        if not files:
            continue
        lines += ["", f"### view corruption: {mode} (mean over {len(files)} folds)", "", "| condition | learned MPJPE | learned PA | weight on corrupted view | mean-fusion MPJPE (same corruption) | mean excl. view (oracle) |", "|---|---|---|---|---|---|"]
        for cond in ("clean", *VIEWS):
            vals = [json.loads(p.read_text())[cond] for p in files]
            g = lambda key: np.mean([v["learned"][key] for v in vals])
            wv = [v["weight_on_corrupted_view"] for v in vals if v["weight_on_corrupted_view"] is not None]
            mm = np.mean([v["fuse_mean_same_corruption"]["mpjpe_m"] for v in vals])
            ex = [v["fuse_mean_excluding_view_oracle"]["mpjpe_m"] for v in vals if v["fuse_mean_excluding_view_oracle"]]
            lines.append(f"| {cond} | {g('mpjpe_m'):.4f} | {g('pa_mpjpe_m'):.4f} | {(np.mean(wv) if wv else float('nan')):.3f} | {mm:.4f} | {(np.mean(ex) if ex else float('nan')):.4f} |")
    da = [base / f"fold{f}" / "driveandact.json" for f in folds]
    da = [p for p in da if p.exists()]
    if da:
        lines += ["", "### Drive&Act zero-shot (hs10, pooled PA-MPJPE mm)", "", "| method | " + " | ".join(p.parent.name for p in da) + " |", "|---|" + "---|" * len(da)]
        keys = ["single_front", "single_left", "single_right", "fuse_mean", "fuse_median", "learned"]
        for k in keys:
            lines.append(f"| {k} | " + " | ".join(f"{json.loads(p.read_text())['overall'][k]['pa_mpjpe_m']*1000:.1f}" for p in da) + " |")
    text = "\n".join(lines)
    (base / "summary.md").write_text(text)
    print(text)


# ------------------------------------------------------------------ main


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--index-mapping", type=Path, default=DEFAULT_INDEX)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--tag", type=str, default="main")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pck", type=float, nargs="+", default=[0.02, 0.05, 0.10, 0.15])
    # model
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--joint-emb", type=int, default=8)
    ap.add_argument("--no-temporal", dest="temporal", action="store_false")
    ap.set_defaults(temporal=True)
    ap.add_argument("--uniform-weights", action="store_true", help="ablation: fixed uniform view weights (temporal head only)")
    ap.add_argument("--residual-hidden", type=int, default=0, help=">0: add a per-joint residual refinement MLP of this width")
    ap.add_argument("--model", choices=("learned", "setfusion", "tripose"), default="learned")
    ap.add_argument("--refiner-dim", type=int, default=128, help="tripose: TCN channel width")
    ap.add_argument("--tripose-velocity", action="store_true", help="tripose: keep multi-scale velocity features")
    ap.add_argument("--no-cross-view-attention", action="store_true", help="tripose: disable cross-view attention")
    ap.add_argument("--view-layers", type=int, default=2)
    ap.add_argument("--joint-layers", type=int, default=1)
    ap.add_argument("--no-uncertainty", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    t.add_argument("--augment-prob", type=float, default=0.0, help="per-sample prob of corrupting one random view during training")
    t.add_argument("--steps", type=int, default=3000)
    t.add_argument("--batch", type=int, default=32)
    t.add_argument("--window", type=int, default=64)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--eval-every", type=int, default=500)
    t.set_defaults(func=cmd_train)

    s = sub.add_parser("stress")
    s.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    s.add_argument("--mode", choices=("drop", "zero", "noise"), default="zero")
    s.set_defaults(func=cmd_stress)

    d = sub.add_parser("driveandact")
    d.add_argument("--fold", type=int, default=0)
    d.add_argument("--da-root", type=Path, default=DEFAULT_DA_ROOT)
    d.set_defaults(func=cmd_driveandact)

    m = sub.add_parser("summarize")
    m.set_defaults(func=cmd_summarize)

    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    args.func(args)


if __name__ == "__main__":
    main()
