#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export fused 3D keypoints from any ``learned_fusion_experiments`` model for downstream
analysis (generalises export_v3_fused_keypoints.py, which is ResidualFusion-only).

The model class is inferred from ``<model-root>/fold0/result.json`` (``config``):
SetFusion (``--model setfusion``), ResidualFusion (``--residual-hidden > 0``) or the
gating-only LearnedFusion.  Every subject is routed to the fold model whose *validation*
split contains it, so no sequence is scored by a model that saw it.

For each cached sequence ``<person>_<env>.npz`` the script appends to the matching file
in ``--out-dir`` the fields

  fused_<suffix>        (T,52,3) float32  fused canonical 3D keypoints, metres
  uncertainty_<suffix>  (T,52)   float32  Laplace scale per joint (absent if the model has none)
  gate_<suffix>         (T,52,3) float32  per-view weights (front/left/right = ``cameras``)
  fold_<suffix>         ()       int

and writes ``<suffix>_export_manifest.json`` with per-sequence MPJPE/PA against the v3
reference next to the numbers in the training ``result.json`` (checkpoint/config guard).

Example (paper's full model, 5-fold SetFusion + corruption augmentation):
  python TriPoseFusion/eval/export_fused_keypoints.py \
      --model-root TriPoseFusion/eval/logs/learned_fusion_experiments/perseq_setfusion \
      --suffix full
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

EVAL_DIR = Path(__file__).resolve().parent
D = Path("/work/1/SKIING/chenkaixu/data/drive")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-root", type=Path,
                    default=EVAL_DIR / "logs/learned_fusion_experiments/perseq_setfusion")
    ap.add_argument("--suffix", default="full", help="field suffix: fused_<suffix>, gate_<suffix>, ...")
    ap.add_argument("--cache-dir", type=Path, default=D / "learned_fusion_cache_perseq")
    ap.add_argument("--index-mapping", type=Path, default=D / "index_mapping")
    ap.add_argument("--out-dir", type=Path, default=D / "downstream_analysis/fused_keypoints")
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--threads", type=int, default=4)
    return ap.parse_args()


def val_subjects(index_mapping: Path, fold: int) -> set[str]:
    d = json.load(open(index_mapping / f"fold_{fold}.json", encoding="utf-8"))
    return {str(s["person_id"])[:2] for s in d["val"]}


def build_from_config(cfg: dict):
    from learned_fusion_experiments import LearnedFusion, ResidualFusion, SetFusion  # noqa: E402
    if cfg.get("model") == "setfusion":
        return SetFusion(d=cfg["hidden"] if cfg["hidden"] >= 16 else 64, temporal=cfg["temporal"],
                         view_layers=cfg["view_layers"], joint_layers=cfg["joint_layers"],
                         uncertainty=not cfg.get("no_uncertainty", False))
    kw = dict(hidden=cfg["hidden"], temporal=cfg["temporal"], joint_emb=cfg["joint_emb"])
    if cfg.get("residual_hidden", 0) > 0:
        return ResidualFusion(residual_hidden=cfg["residual_hidden"],
                              uncertainty=cfg.get("residual_uncertainty", False), **kw)
    return LearnedFusion(**kw)


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.threads)
    sys.path.insert(0, str(EVAL_DIR))
    sys.path.insert(0, str(EVAL_DIR.parent))
    from eval_fusion_baselines_pesudo_gt import compute_metrics  # noqa: E402

    cfg = json.load(open(args.model_root / "fold0" / "result.json", encoding="utf-8"))["config"]
    print(f"model config: model={cfg.get('model')} hidden={cfg['hidden']} residual={cfg.get('residual_hidden', 0)} "
          f"aug={cfg.get('augment_prob')} steps={cfg.get('steps')}", flush=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sfx = args.suffix

    manifest = []
    for fold in args.folds:
        model = build_from_config(cfg)
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
                ls = getattr(model, "last_logscale", None)
                unc = torch.exp(ls.clamp(-6, 3))[0].numpy().astype(np.float32) if ls is not None else None
            pred = pred[0].numpy().astype(np.float32)
            gate = w[0].numpy().astype(np.float32)
            m = compute_metrics(pred=pred, gt=gt, valid_mask=gv, pck_thresholds=(0.05, 0.10)) or {}
            person, env = p.stem[:2], p.stem[3:]
            r = ref.get((person, env), {})
            row = {"sequence": p.stem, "fold": fold, "frames": int(pred.shape[0]),
                   "mpjpe_m": m.get("mpjpe_m"), "pa_mpjpe_m": m.get("pa_mpjpe_m"),
                   "ref_mpjpe_m": r.get("mpjpe_m"), "ref_pa_mpjpe_m": r.get("pa_mpjpe_m"),
                   "mean_gate": gate.reshape(-1, gate.shape[-1]).mean(0).tolist(),
                   "median_uncertainty_m": float(np.median(unc)) if unc is not None else None}
            manifest.append(row)
            out_path = args.out_dir / p.name
            if out_path.exists():
                payload = dict(np.load(out_path, allow_pickle=True))
            else:
                with np.load(p, allow_pickle=True) as z:
                    payload = {"view_pose": vp, "view_conf": vc, "gt_pose": gt, "gt_valid": gv,
                               "frame_ids": z["frame_ids"], "cameras": z["cameras"]}
            payload[f"fused_{sfx}"] = pred
            payload[f"gate_{sfx}"] = gate
            payload[f"fold_{sfx}"] = np.int64(fold)
            if unc is not None:
                payload[f"uncertainty_{sfx}"] = unc
            np.savez_compressed(out_path, **payload)
            print(f"  {p.stem}: T={pred.shape[0]} mpjpe={row['mpjpe_m']:.4f} "
                  f"(train-log ref {r.get('mpjpe_m', float('nan')):.4f})", flush=True)
    json.dump(manifest, open(args.out_dir / f"{sfx}_export_manifest.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    mp = np.array([r["mpjpe_m"] for r in manifest if r["mpjpe_m"] is not None])
    print(f"done: {len(manifest)} sequences, mean per-sequence MPJPE {mp.mean():.4f} m", flush=True)


if __name__ == "__main__":
    main()
