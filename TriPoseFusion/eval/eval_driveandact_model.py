#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TriPoseFusion checkpoint 在 Drive&Act 密集片段上的零样本评测。

输入由 driveandact/pipeline/extract_dense.py + run_sam3d_batch.py --root 产出
（每 run 3 段 × 600 帧连续 30fps），模型按训练同样的 16 帧连续窗口推理。

关节协议 hs10：模型输出为 52 关节（无肘/髋），与 BODY-25 GT 的公共集为
nose, 2 eye, 2 ear, 2 shoulder, 2 wrist, neck（预测侧 neck 取双肩中点以对齐
BODY-25 定义）。GT 无髋则无法与模型 canonical 系对齐，故主指标为 PA-MPJPE
（相似变换不变），另报刚性对齐（无尺度）MPJPE 以显示尺度差。
基线（单视角 / mean 融合）在模型自身 _canonicalize_pose 系内计算，与模型同协议。

用法:
  python TriPoseFusion/eval/eval_driveandact_model.py \
      --data-root .../driveandact/tripose_eval_dense \
      --config <run>/.hydra/config.yaml --ckpt <run>/checkpoints/fold_0/xx.ckpt \
      --output-dir TriPoseFusion/eval/logs/driveandact_model
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from map_config import KEEP_KEYPOINT_INDICES  # noqa: E402
from eval.eval_fusion_baselines_pesudo_gt import compute_metrics  # noqa: E402
from trainer.train_triple_fusion import TriFusionPoseTrainer  # noqa: E402

CAMERAS = ("front", "left", "right")
WINDOW = 16
KEEP = list(KEEP_KEYPOINT_INDICES)
# (52 关节空间索引, BODY-25 名称)
HS10: List[Tuple[int, str]] = [
    (KEEP.index(0), "nose"),
    (KEEP.index(1), "lEye"), (KEEP.index(2), "rEye"),
    (KEEP.index(3), "lEar"), (KEEP.index(4), "rEar"),
    (KEEP.index(5), "lShoulder"), (KEEP.index(6), "rShoulder"),
    (KEEP.index(62), "lWrist"), (KEEP.index(41), "rWrist"),
    (KEEP.index(69), "neck"),
]
HS10_IDX = np.array([i for i, _ in HS10])
HS10_NAMES = [n for _, n in HS10]
LSHO, RSHO, NECK = HS10_NAMES.index("lShoulder"), HS10_NAMES.index("rShoulder"), HS10_NAMES.index("neck")


def load_view(view_dir: Path) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    out = {}
    for fp in sorted(view_dir.glob("*_sam3d_body.npz")):
        fid = int(fp.name.split("_")[0])
        with np.load(fp, allow_pickle=True) as data:
            o = data["output"].item()
            k2 = np.asarray(o["pred_keypoints_2d"], dtype=np.float32)[KEEP]
            k3 = np.asarray(o["pred_keypoints_3d"], dtype=np.float32)[KEEP]
        out[fid] = (k2, k3)
    return out


def load_gt(gt_npz: Path):
    with np.load(gt_npz, allow_pickle=False) as data:
        frames = data["frames"]
        poses = data["poses"]
        names = [str(n) for n in data["joint_names"]]
    idx = np.array([names.index(n) for n in HS10_NAMES])
    gt, valid = {}, {}
    for k, fid in enumerate(frames):
        sel = poses[k][idx]
        gt[int(fid)] = sel[:, :3].astype(np.float32)
        valid[int(fid)] = (sel[:, 3] > 0) & (np.abs(sel[:, :3]).sum(-1) > 0)
    return gt, valid


def build_windows(fids_ok: set) -> List[List[int]]:
    """在所有视角与 GT 都存在的帧上取不重叠的 16 帧连续窗口。"""
    windows, sorted_f = [], sorted(fids_ok)
    i = 0
    while i + WINDOW <= len(sorted_f):
        w = sorted_f[i:i + WINDOW]
        if w[-1] - w[0] == WINDOW - 1:
            windows.append(w)
            i += WINDOW
        else:
            i += 1
    return windows


def to_hs10(pose52: np.ndarray) -> np.ndarray:
    p = pose52[..., HS10_IDX, :].copy()
    p[..., NECK, :] = 0.5 * (p[..., LSHO, :] + p[..., RSHO, :])
    return p


def rigid_mpjpe(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> float:
    """逐帧 Kabsch（旋转+平移、无尺度）对齐后的 MPJPE。"""
    errs = []
    for p, g, v in zip(pred, gt, valid):
        if v.sum() < 3:
            continue
        ps, gs = p[v], g[v]
        pc, gc = ps - ps.mean(0), gs - gs.mean(0)
        u, _, vt = np.linalg.svd(pc.T @ gc)
        d = np.sign(np.linalg.det(u @ vt))
        rot = (u * np.array([1, 1, d])) @ vt
        errs.extend(np.linalg.norm(pc @ rot - gc, axis=-1).tolist())
    return float(np.mean(errs)) if errs else float("nan")


def metrics(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> Dict:
    m = compute_metrics(pred, gt, valid, root_index=NECK)
    return {
        "num_frames": m.get("num_frames"),
        "num_valid_points": m.get("num_valid_points"),
        "pa_mpjpe_m": m.get("pa_mpjpe_m"),
        "rigid_mpjpe_m": rigid_mpjpe(pred, gt, valid),
        "mpjpe_m_frame_mismatch": m.get("mpjpe_m"),  # 不同 canonical 系,仅供参考
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    config = OmegaConf.load(args.config)
    payload = torch.load(str(args.ckpt), map_location="cpu", weights_only=False)
    state = payload["state_dict"] if "state_dict" in payload else payload
    module = TriFusionPoseTrainer(config)
    missing, unexpected = module.load_state_dict(state, strict=False)
    print(f"ckpt loaded: missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    module.to(device).eval()
    view_names = [str(v) for v in module.model.view_names]

    results: Dict[str, Dict] = {}
    pooled: Dict[str, List[np.ndarray]] = {}
    pooled_gt, pooled_valid, alpha_sum, alpha_n = [], [], None, 0

    for gt_npz in sorted((args.data_root / "gt").glob("vp*/*.npz")):
        vp, run = gt_npz.parent.name, gt_npz.stem
        sam_root = args.data_root / "sam3d" / vp / run
        if not all((sam_root / c).is_dir() for c in CAMERAS):
            print(f"skip {vp}/{run}: sam3d incomplete", flush=True)
            continue
        views = {c: load_view(sam_root / c) for c in CAMERAS}
        gt, gt_valid = load_gt(gt_npz)
        ok = set(gt)
        for v in views.values():
            ok &= set(v)
        windows = build_windows(ok)
        if not windows:
            print(f"skip {vp}/{run}: no full windows", flush=True)
            continue

        preds = {k: [] for k in ["model", "fuse_mean", *[f"single_{c}" for c in CAMERAS]]}
        gts, valids = [], []
        with torch.no_grad():
            for b in range(0, len(windows), args.batch_size):
                batch = windows[b:b + args.batch_size]
                p3d = {c: torch.from_numpy(np.stack([[views[c][f][1] for f in w] for w in batch])).to(device)
                       for c in view_names}
                p2d = {c: torch.from_numpy(np.stack([[views[c][f][0] for f in w] for w in batch])).to(device)
                       for c in view_names}
                out = module.model(pose3d=p3d, pose2d=p2d)
                preds["model"].append(to_hs10(out["P_final"].cpu().numpy()).reshape(-1, len(HS10), 3))
                canon = {c: module.model._canonicalize_pose(p3d[c]).cpu().numpy() for c in view_names}
                for c in view_names:
                    preds[f"single_{c}"].append(to_hs10(canon[c]).reshape(-1, len(HS10), 3))
                preds["fuse_mean"].append(
                    to_hs10(np.mean(np.stack([canon[c] for c in view_names]), axis=0)).reshape(-1, len(HS10), 3)
                )
                a = out["alpha"].detach().cpu().numpy()  # (B,T,J,V)
                alpha_sum = a.sum(axis=(0, 1, 2)) if alpha_sum is None else alpha_sum + a.sum(axis=(0, 1, 2))
                alpha_n += a.shape[0] * a.shape[1] * a.shape[2]
                for w in batch:
                    gts.extend(gt[f] for f in w)
                    valids.extend(gt_valid[f] for f in w)

        gt_arr, valid_arr = np.stack(gts), np.stack(valids)
        run_res = {"num_windows": len(windows), "num_frames": len(gts)}
        for k, chunks in preds.items():
            arr = np.concatenate(chunks)
            run_res[k] = metrics(arr, gt_arr, valid_arr)
            pooled.setdefault(k, []).append(arr)
        pooled_gt.append(gt_arr)
        pooled_valid.append(valid_arr)
        results[f"{vp}/{run}"] = run_res
        print(f"{vp}/{run}: {len(windows)} windows, model PA={run_res['model']['pa_mpjpe_m']:.4f} "
              f"fuse_mean PA={run_res['fuse_mean']['pa_mpjpe_m']:.4f}", flush=True)

    if pooled_gt:
        gt_all, valid_all = np.concatenate(pooled_gt), np.concatenate(pooled_valid)
        overall = {k: metrics(np.concatenate(v), gt_all, valid_all) for k, v in pooled.items()}
        overall["model"]["mean_view_alpha"] = {
            n: float(alpha_sum[i] / alpha_n) for i, n in enumerate(view_names)
        }
        results["overall"] = overall
        results["protocol"] = {"joints": HS10_NAMES, "window": WINDOW,
                               "ckpt": str(args.ckpt), "config": str(args.config)}
        print("\n== OVERALL (pooled, hs10) ==")
        for k, m in overall.items():
            print(f"{k:13s} PA-MPJPE={m['pa_mpjpe_m']*1000:6.1f} mm  rigid-MPJPE={m['rigid_mpjpe_m']*1000:6.1f} mm")
        print("model mean view alpha:", overall["model"]["mean_view_alpha"])

    out_path = args.output_dir / "driveandact_model.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
