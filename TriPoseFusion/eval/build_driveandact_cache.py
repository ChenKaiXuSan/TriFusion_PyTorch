#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a learned-fusion training cache from the Drive&Act dense clips (in-domain
adaptation experiment: does triangulation-supervised residual refinement work on a NEW rig
when trained on that rig's own reference?).

Per run: SAM3D 52-joint 3D per view -> canonicalize_pose (same as the main cache) ->
view_pose (T,52,3,3). Reference: official openpose_3d (BODY-25) on the hs10 joints; per
frame the GT hs10 set is similarity-aligned (Umeyama) onto the training-free mean-fusion
hs10 pose so that the reference lives in the prediction's canonical frame (the metric is
PA-MPJPE anyway); the BODY-25 neck (definition mismatch with SAM3D) is excluded, so 9
joints carry supervision/evaluation. Also writes leave-one-subject-out fold JSONs in the
index_mapping format used by learned_fusion_experiments.py.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
for p in (str(SCRIPT_DIR), str(SCRIPT_DIR.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

from eval_driveandact_model import HS10_IDX, HS10_NAMES, load_gt, load_view, to_hs10  # noqa: E402
from eval_fusion_baselines_pesudo_gt import canonicalize_pose  # noqa: E402

VIEWS = ("front", "left", "right")
NECK = HS10_NAMES.index("neck")


def umeyama(src: np.ndarray, dst: np.ndarray):
    """Similarity transform (s, R, t), row convention: dst ~= s * src @ R + t."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    sc, dc = src - mu_s, dst - mu_d
    var_s = float((sc ** 2).sum())
    u, s, vt = np.linalg.svd(sc.T @ dc)
    d = np.sign(np.linalg.det(u @ vt)) or 1.0
    signs = np.ones(len(s))
    signs[-1] = d
    r = (u * signs) @ vt
    scale = float((s * signs).sum()) / max(var_s, 1e-12)
    t = mu_d - scale * mu_s @ r
    return scale, r, t


def build_run(da_root: Path, vp: str, run: str, out_dir: Path) -> dict:
    gt, gt_valid = load_gt(da_root / "gt" / vp / f"{run}.npz")
    views = {c: load_view(da_root / "sam3d" / vp / run / c) for c in VIEWS}
    ok = set(gt)
    for v in views.values():
        ok &= set(v)
    frames = sorted(ok)
    raw = {c: np.stack([views[c][f][1] for f in frames]).astype(np.float32) for c in VIEWS}
    canon = np.stack([canonicalize_pose(raw[c]) for c in VIEWS], axis=2)  # (T,52,3,3)
    mean_hs10 = to_hs10(np.nanmean(canon, axis=2))  # (T,10,3)
    gt_hs10 = np.stack([gt[f] for f in frames])  # (T,10,3)
    valid_hs10 = np.stack([gt_valid[f] for f in frames])  # (T,10)
    valid_hs10[:, NECK] = False  # BODY-25 neck != SAM3D neck definition

    t_n = len(frames)
    gt52 = np.full((t_n, canon.shape[1], 3), np.nan, dtype=np.float32)
    valid52 = np.zeros((t_n, canon.shape[1]), dtype=bool)
    n_aligned = 0
    for t in range(t_n):
        m = valid_hs10[t] & np.isfinite(mean_hs10[t]).all(-1)
        if m.sum() < 4:
            continue
        s, r, tr = umeyama(gt_hs10[t][m], mean_hs10[t][m])
        aligned = s * gt_hs10[t] @ r + tr
        for j in np.nonzero(m)[0]:
            gt52[t, HS10_IDX[j]] = aligned[j]
            valid52[t, HS10_IDX[j]] = True
        n_aligned += 1
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / f"{vp}_{run}.npz",
        view_pose=canon.astype(np.float16),
        view_conf=np.ones(canon.shape[:3], dtype=np.float16),
        gt_pose=gt52.astype(np.float16),
        gt_valid=valid52,
        frame_ids=np.asarray(frames),
    )
    return {"vp": vp, "run": run, "frames": t_n, "aligned_frames": n_aligned, "valid_points": int(valid52.sum())}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--da-root", type=Path, default=Path("/work/1/SKIING/chenkaixu/data/drive/driveandact/tripose_eval_dense"))
    ap.add_argument("--out-dir", type=Path, default=Path("/work/1/SKIING/chenkaixu/data/drive/learned_fusion_cache_driveandact"))
    ap.add_argument("--index-dir", type=Path, default=Path("/work/1/SKIING/chenkaixu/data/drive/driveandact_index_mapping"))
    args = ap.parse_args()

    runs = []
    for gt_npz in sorted((args.da_root / "gt").glob("vp*/*.npz")):
        vp, run = gt_npz.parent.name, gt_npz.stem
        if not all((args.da_root / "sam3d" / vp / run / c).is_dir() for c in VIEWS):
            continue
        info = build_run(args.da_root, vp, run, args.out_dir)
        runs.append((vp, run))
        print(info, flush=True)

    subjects = sorted({vp for vp, _ in runs})
    args.index_dir.mkdir(parents=True, exist_ok=True)
    for k, held in enumerate(subjects):
        fold = {
            "train": [{"person_id": vp, "env_folder": run} for vp, run in runs if vp != held],
            "val": [{"person_id": vp, "env_folder": run} for vp, run in runs if vp == held],
        }
        (args.index_dir / f"fold_{k}.json").write_text(json.dumps(fold, indent=1))
        print(f"fold_{k}: held-out {held}: train {len(fold['train'])} / val {len(fold['val'])} runs")


if __name__ == "__main__":
    main()
