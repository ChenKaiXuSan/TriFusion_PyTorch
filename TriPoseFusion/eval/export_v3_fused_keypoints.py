#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export TriPoseFusion v3 fused 3D keypoints for downstream analysis.

v3 = ResidualFusion(residual_hidden=64, uncertainty=True) trained with corruption
augmentation on the per-sequence-calibrated (v3) pseudo-GT, 5-fold cross-subject
(tag ``perseq_resid64_aug_unc``).  Every subject is routed to the fold model whose
*validation* split contains it, so no sequence is scored by a model that saw it.

For each cached sequence ``<person>_<env>.npz`` (learned_fusion_cache_perseq) the
script appends to the matching file in ``--out-dir`` (created from the mean-fusion
export) the fields

  fused_v3        (T,52,3) float32  fused canonical 3D keypoints, metres
  uncertainty_v3  (T,52)   float32  Laplace scale b per joint, metres (larger = less confident)
  gate_v3         (T,52,3) float32  learned per-view weights (front/left/right order = ``cameras``)
  fold_v3         ()       int      fold model used

and writes ``v3_export_manifest.json`` with per-sequence sanity metrics against the
v3 reference (MPJPE / PA on gt_valid joints) next to the numbers stored in the
training ``result.json`` so a wrong checkpoint or config is caught immediately.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REVIEW_FIXES_EVAL = Path(
    "/work/1/SKIING/chenkaixu/code/TriFusion/.claude/worktrees/review-fixes/TriPoseFusion/eval"
)
DEFAULT_MODEL_ROOT = REVIEW_FIXES_EVAL / "logs/learned_fusion_experiments/perseq_resid64_aug_unc"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--code-dir", type=Path, default=REVIEW_FIXES_EVAL,
                    help="eval dir containing learned_fusion_experiments.py (model definitions)")
    ap.add_argument("--cache-dir", type=Path,
                    default=Path("/work/1/SKIING/chenkaixu/data/drive/learned_fusion_cache_perseq"))
    ap.add_argument("--index-mapping", type=Path,
                    default=Path("/work/1/SKIING/chenkaixu/data/drive/index_mapping"))
    ap.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    ap.add_argument("--out-dir", type=Path,
                    default=Path("/work/1/SKIING/chenkaixu/data/drive/fused_keypoints_perseq"))
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--threads", type=int, default=4)
    return ap.parse_args()


def val_subjects(index_mapping: Path, fold: int) -> set[str]:
    fj = json.load(open(index_mapping / f"fold_{fold}.json", encoding="utf-8"))
    subs = set()
    for x in fj["val"]:
        pid = x.get("person_id") if isinstance(x, dict) else str(x)
        subs.add(str(pid)[:2])
    return subs


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.threads)
    sys.path.insert(0, str(args.code_dir))
    sys.path.insert(0, str(args.code_dir.parent))
    from learned_fusion_experiments import ResidualFusion  # noqa: E402
    from eval_fusion_baselines_pesudo_gt import compute_metrics  # noqa: E402

    cfg = json.load(open(args.model_root / "fold0" / "result.json", encoding="utf-8"))["config"]
    assert cfg["residual_hidden"] > 0 and cfg["residual_uncertainty"], cfg
    args.out_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    for fold in args.folds:
        model = ResidualFusion(
            residual_hidden=cfg["residual_hidden"], uncertainty=cfg["residual_uncertainty"],
            hidden=cfg["hidden"], temporal=cfg["temporal"], joint_emb=cfg["joint_emb"],
        )
        model.load_state_dict(torch.load(args.model_root / f"fold{fold}" / "model.pt", map_location="cpu"))
        model.eval()
        ref = {(r["person"], r["env"]): r["metrics"] for r in json.load(
            open(args.model_root / f"fold{fold}" / "result.json", encoding="utf-8"))["per_sequence_learned"]}
        subs = val_subjects(args.index_mapping, fold)
        files = sorted(p for p in args.cache_dir.glob("*.npz") if p.stem[:2] in subs)
        print(f"[fold {fold}] val subjects {sorted(subs)} -> {len(files)} sequences", flush=True)
        for p in files:
            with np.load(p, allow_pickle=False) as z:
                vp = z["view_pose"].astype(np.float32)
                vc = z["view_conf"].astype(np.float32)
                gt, gv = z["gt_pose"].astype(np.float32), z["gt_valid"].astype(bool)
            with torch.no_grad():
                pred, w = model(torch.from_numpy(vp)[None], torch.from_numpy(vc)[None])
                scale = torch.exp(model.last_logscale.clamp(-6, 3))
            pred = pred[0].numpy().astype(np.float32)
            gate = w[0].numpy().astype(np.float32)
            unc = scale[0].numpy().astype(np.float32)
            m = compute_metrics(pred=pred, gt=gt, valid_mask=gv, pck_thresholds=(0.05, 0.10)) or {}
            person, env = p.stem[:2], p.stem[3:]
            r = ref.get((person, env), {})
            row = {"sequence": p.stem, "fold": fold, "frames": int(pred.shape[0]),
                   "mpjpe_m": m.get("mpjpe_m"), "pa_mpjpe_m": m.get("pa_mpjpe_m"),
                   "ref_mpjpe_m": r.get("mpjpe_m"), "ref_pa_mpjpe_m": r.get("pa_mpjpe_m"),
                   "mean_gate": gate.reshape(-1, gate.shape[-1]).mean(0).tolist(),
                   "median_uncertainty_m": float(np.median(unc))}
            manifest.append(row)
            out_path = args.out_dir / p.name
            payload = dict(np.load(out_path, allow_pickle=True)) if out_path.exists() else {
                "view_pose": vp, "view_conf": vc, "gt_pose": gt, "gt_valid": gv,
                "frame_ids": np.load(p, allow_pickle=True)["frame_ids"],
                "cameras": np.load(p, allow_pickle=True)["cameras"]}
            payload.update(fused_v3=pred, uncertainty_v3=unc, gate_v3=gate, fold_v3=np.int64(fold))
            np.savez_compressed(out_path, **payload)
            print(f"  {p.stem}: T={pred.shape[0]} mpjpe={row['mpjpe_m']:.4f} (ref {r.get('mpjpe_m', float('nan')):.4f})", flush=True)
    json.dump(manifest, open(args.out_dir / "v3_export_manifest.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    mp = np.array([r["mpjpe_m"] for r in manifest if r["mpjpe_m"] is not None])
    print(f"done: {len(manifest)} sequences, mean per-sequence MPJPE {mp.mean():.4f} m", flush=True)


if __name__ == "__main__":
    main()
