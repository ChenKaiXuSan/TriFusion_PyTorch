#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Drive&Act 域内自适应缓存 —— **自三角化参考**（真正 label-free 的版本）。

与 build_driveandact_cache.py 的唯一区别：训练参考不再是官方 openpose_3d，而是
"SAM3D 2D + 官方标定 → 去畸变 DLT 三角化 → 重投影 ≤ 40 px 质量过滤"，即与自有数据
伪 GT 流水线同一配方，不使用任何人工/官方 3D 标注。官方 GT 仍写入 gt_pose/gt_valid，
**只用于验证**。每个 npz 含：
    view_pose / view_conf            同原缓存（逐视角 canonicalize）
    gt_pose / gt_valid               官方 hs10（去 neck → 9 关节）对齐到均值融合系【仅评估】
    gt_pose_tri52 / gt_valid_tri52   自三角化 52 关节参考（含手），对齐同上【训练】
    gt_pose_tri9 / gt_valid_tri9     同上但只在 9 个 hs 关节有效【训练，与官方 GT 监督同关节集】
对齐：三角化的 9 个 hs 关节 → 均值融合 canonical 系的逐帧 Umeyama 相似变换，应用到全部 52 关节。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
for p in (str(SCRIPT_DIR), str(SCRIPT_DIR.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

from build_driveandact_cache import NECK, VIEWS, umeyama  # noqa: E402
from eval_driveandact_model import HS10_IDX, load_gt, load_view, to_hs10  # noqa: E402
from eval_driveandact_triangulation import VIEW_CAMS, Camera  # noqa: E402
from eval_fusion_baselines_pesudo_gt import canonicalize_pose, procrustes_align  # noqa: E402

HS9 = np.array([i for k, i in enumerate(HS10_IDX) if k != NECK])  # 52 布局下的 9 个评估关节


def triangulate_batch(norm: dict, cams: dict) -> np.ndarray:
    """norm[v]: (N,J,2) 去畸变归一化坐标 → (N,J,3) 世界系 DLT（批量 SVD）。"""
    rows = []
    for v in VIEWS:
        P = cams[v].P
        xy = norm[v]
        rows.append(xy[..., 0:1] * P[2] - P[0])
        rows.append(xy[..., 1:2] * P[2] - P[1])
    A = np.stack(rows, axis=-2)  # (N,J,6,4)
    _, _, vt = np.linalg.svd(A)
    X = vt[..., -1, :]
    return X[..., :3] / X[..., 3:4]


def build_run(da_root: Path, calib_root: Path, vp: str, run: str, out_dir: Path, rpe_px: float) -> dict:
    gt, gt_valid = load_gt(da_root / "gt" / vp / f"{run}.npz")
    views = {c: load_view(da_root / "sam3d" / vp / run / c) for c in VIEWS}
    ok = set(gt)
    for v in views.values():
        ok &= set(v)
    frames = sorted(ok)
    t_n = len(frames)
    raw2d = {c: np.stack([views[c][f][0] for f in frames]).astype(np.float64) for c in VIEWS}
    raw3d = {c: np.stack([views[c][f][1] for f in frames]).astype(np.float32) for c in VIEWS}
    j_n = raw3d["front"].shape[1]

    # ---- 自三角化参考（世界系, 米）+ 重投影质量过滤 ----
    cams = {v: Camera.from_json(calib_root / vdir / vp / f"{run}.{ids}.calibration.json")
            for v, (vdir, ids) in VIEW_CAMS.items()}
    norm = {v: cams[v].undistort(raw2d[v].reshape(-1, 2)).reshape(t_n, j_n, 2) for v in VIEWS}
    tri = triangulate_batch(norm, cams)  # (T,52,3)
    res = np.zeros((t_n, j_n, len(VIEWS)))
    for k, v in enumerate(VIEWS):
        uv = cams[v].project(tri.reshape(-1, 3)).reshape(t_n, j_n, 2)
        res[..., k] = np.linalg.norm(uv - raw2d[v], axis=-1)
    valid_tri = np.isfinite(tri).all(-1) & (res.max(-1) <= rpe_px)

    # ---- 逐视角 canonicalize（同原缓存）----
    canon = np.stack([canonicalize_pose(raw3d[c]) for c in VIEWS], axis=2)  # (T,52,3,3)
    mean52 = np.nanmean(canon, axis=2)
    mean_hs10 = to_hs10(mean52)

    # ---- 官方 GT（仅评估）：与原缓存完全相同 ----
    gt_hs10 = np.stack([gt[f] for f in frames])
    valid_hs10 = np.stack([gt_valid[f] for f in frames])
    valid_hs10[:, NECK] = False
    gt52 = np.full((t_n, j_n, 3), np.nan, dtype=np.float32)
    valid52 = np.zeros((t_n, j_n), dtype=bool)
    for t in range(t_n):
        m = valid_hs10[t] & np.isfinite(mean_hs10[t]).all(-1)
        if m.sum() < 4:
            continue
        s, r, tr = umeyama(gt_hs10[t][m], mean_hs10[t][m])
        aligned = s * gt_hs10[t] @ r + tr
        for j in np.nonzero(m)[0]:
            gt52[t, HS10_IDX[j]] = aligned[j]
            valid52[t, HS10_IDX[j]] = True

    # ---- 自三角化参考对齐到均值融合 canonical 系（用 9 个 hs 关节求相似变换）----
    tri52 = np.full((t_n, j_n, 3), np.nan, dtype=np.float32)
    vtri52 = np.zeros((t_n, j_n), dtype=bool)
    ref_vs_gt = []  # 自参考 vs 官方 GT 的逐帧 PA 误差（9 关节）——自参考质量
    for t in range(t_n):
        m9 = valid_tri[t, HS9] & np.isfinite(mean52[t, HS9]).all(-1)
        if m9.sum() < 4:
            continue
        s, r, tr = umeyama(tri[t, HS9][m9], mean52[t, HS9][m9])
        aligned = (s * tri[t] @ r + tr).astype(np.float32)
        tri52[t] = aligned
        vtri52[t] = valid_tri[t]
        both = valid52[t, HS9] & valid_tri[t, HS9]
        if both.sum() >= 3:
            a = procrustes_align(aligned[HS9][both], gt52[t, HS9][both])
            ref_vs_gt.append(float(np.linalg.norm(a - gt52[t, HS9][both], axis=-1).mean()))
    vtri9 = np.zeros_like(vtri52)
    vtri9[:, HS9] = vtri52[:, HS9]

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / f"{vp}_{run}.npz",
        view_pose=canon.astype(np.float16),
        view_conf=np.ones(canon.shape[:3], dtype=np.float16),
        gt_pose=gt52.astype(np.float16), gt_valid=valid52,
        gt_pose_tri52=tri52.astype(np.float16), gt_valid_tri52=vtri52,
        gt_pose_tri9=tri52.astype(np.float16), gt_valid_tri9=vtri9,
        frame_ids=np.asarray(frames),
    )
    sw = np.linalg.norm(tri[:, HS10_IDX[5]] - tri[:, HS10_IDX[6]], axis=-1)  # lShoulder - rShoulder
    return {
        "vp": vp, "run": run, "frames": t_n,
        "tri_valid_ratio_all52": round(float(valid_tri.mean()), 3),
        "tri_valid_ratio_hs9": round(float(valid_tri[:, HS9].mean()), 3),
        "tri_median_rpe_px": round(float(np.median(res[valid_tri])), 2),
        "tri_shoulder_width_m": round(float(np.nanmedian(sw)), 3),
        "selfref_vs_officialGT_PA_mm": round(1000 * float(np.mean(ref_vs_gt)), 1) if ref_vs_gt else None,
        "official_valid_points": int(valid52.sum()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--da-root", type=Path, default=Path("/work/1/SKIING/chenkaixu/data/drive/driveandact/tripose_eval_dense"))
    ap.add_argument("--calib-root", type=Path, default=Path("/work/1/SKIING/chenkaixu/data/drive/driveandact"))
    ap.add_argument("--out-dir", type=Path, default=Path("/work/1/SKIING/chenkaixu/data/drive/learned_fusion_cache_driveandact_selftri"))
    ap.add_argument("--rpe-px", type=float, default=40.0, help="三角化质量过滤: 所有视角重投影残差 ≤ 此值 (自有流水线为 40 px)")
    args = ap.parse_args()
    infos = []
    for gt_npz in sorted((args.da_root / "gt").glob("vp*/*.npz")):
        vp, run = gt_npz.parent.name, gt_npz.stem
        if not all((args.da_root / "sam3d" / vp / run / c).is_dir() for c in VIEWS):
            continue
        info = build_run(args.da_root, args.calib_root, vp, run, args.out_dir, args.rpe_px)
        infos.append(info)
        print(info, flush=True)
    (args.out_dir / "summary.json").write_text(json.dumps(infos, indent=1))


if __name__ == "__main__":
    main()
