#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Drive&Act 公开数据集上的融合基线评测（回应评审 xS2o 的 Mandatory Evaluation）。

数据由 driveandact/pipeline/extract_frames.py + run_sam3d_batch.py 产出：
    <data_root>/sam3d/<vp>/<run>/{front,left,right}/<fid>_sam3d_body.npz  (MHR-70)
    <data_root>/gt/<vp>/<run>.npz  (frames, poses (N,25,4) BODY-25 xyz+conf, joint_names)

协议：与论文主实验一致——逐视角肩颈系 canonicalize 后融合，
compute_metrics（含修复后的 Umeyama PA）。关节取 MHR-70 与 BODY-25 的
同名公共集中 GT 实际有标注的 body14（头 5 + 肩肘腕 6 + 髋 2 + 颈；
Drive&Act 的 openpose_3d 里膝/踝置信度恒为 0），另报上半身子集 upper12。

与主实验协议的两处必要差异（BODY-25 定义所致）：
- BODY-25 的 neck 定义为双肩中点（实测距肩中点仅 ~2cm），用
  shoulder_mid-neck 做 down 轴会退化，故 down 轴改用 midHip-neck
  （双髋均值，两侧都合成），GT 髋无效的帧整帧剔除。
- 预测侧 neck 同样改为双肩中点，消除 MHR neck（C7 附近）与 BODY-25 的
  定义偏差。

用法:
    python TriPoseFusion/eval/eval_driveandact_baselines.py \
        --data-root /work/1/SKIING/chenkaixu/data/drive/driveandact/tripose_eval \
        --output-dir TriPoseFusion/eval/logs/driveandact_baselines
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.eval_fusion_baselines_pesudo_gt import (  # noqa: E402
    canonicalize_pose,
    compute_metrics,
    fuse_views,
)

CAMERAS = ("front", "left", "right")

# MHR-70 索引 → BODY-25 名称（同名公共关节）
MHR_TO_BODY25: List[Tuple[int, str]] = [
    (0, "nose"),
    (1, "lEye"), (2, "rEye"), (3, "lEar"), (4, "rEar"),
    (5, "lShoulder"), (6, "rShoulder"),
    (7, "lElbow"), (8, "rElbow"),
    (62, "lWrist"), (41, "rWrist"),
    (9, "lHip"), (10, "rHip"),
    (69, "neck"),
]
COMMON_NAMES = [name for _, name in MHR_TO_BODY25]  # body14
UPPER12 = ["nose", "lEye", "rEye", "lEar", "rEar", "lShoulder", "rShoulder",
           "lElbow", "rElbow", "lWrist", "rWrist", "neck"]
NECK_IDX = COMMON_NAMES.index("neck")
LSHO_IDX = COMMON_NAMES.index("lShoulder")
RSHO_IDX = COMMON_NAMES.index("rShoulder")
LHIP_IDX = COMMON_NAMES.index("lHip")
RHIP_IDX = COMMON_NAMES.index("rHip")
MIDHIP_IDX = len(COMMON_NAMES)  # 合成关节,仅用于 canonicalize 的 down 轴


def load_view_sequence(view_dir: Path) -> Dict[int, np.ndarray]:
    """帧号 → (18,3) 公共关节 3D。"""
    mhr_idx = np.array([i for i, _ in MHR_TO_BODY25])
    out: Dict[int, np.ndarray] = {}
    for fp in sorted(view_dir.glob("*_sam3d_body.npz")):
        fid = int(fp.name.split("_")[0])
        with np.load(fp, allow_pickle=True) as data:
            output = data["output"].item()
            kpt3d = np.asarray(output["pred_keypoints_3d"], dtype=np.float32)
        sel = kpt3d[mhr_idx, :3].copy()
        sel[NECK_IDX] = 0.5 * (sel[LSHO_IDX] + sel[RSHO_IDX])  # 对齐 BODY-25 neck 定义
        out[fid] = sel
    return out


def load_gt_run(gt_npz: Path) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """帧号 → (18,3) GT 与 (18,) 有效掩码。"""
    with np.load(gt_npz, allow_pickle=False) as data:
        frames = data["frames"]
        poses = data["poses"]  # (N,25,4)
        names = [str(n) for n in data["joint_names"]]
    b25_idx = np.array([names.index(name) for _, name in MHR_TO_BODY25])
    gt_pose: Dict[int, np.ndarray] = {}
    gt_valid: Dict[int, np.ndarray] = {}
    for k, fid in enumerate(frames):
        sel = poses[k][b25_idx]  # (18,4)
        gt_pose[int(fid)] = sel[:, :3].astype(np.float32)
        gt_valid[int(fid)] = (sel[:, 3] > 0) & (np.abs(sel[:, :3]).sum(axis=-1) > 0)
    return gt_pose, gt_valid


def canon(pose: np.ndarray) -> np.ndarray:
    """追加合成 midHip 做 down 轴,canonicalize 后再去掉。"""
    mid_hip = 0.5 * (pose[:, LHIP_IDX] + pose[:, RHIP_IDX])
    extended = np.concatenate([pose, mid_hip[:, None]], axis=1)
    out = canonicalize_pose(
        extended, neck_index=NECK_IDX,
        left_shoulder_index=LSHO_IDX, right_shoulder_index=RSHO_IDX,
        mid_hip_index=MIDHIP_IDX,
    )
    return out[:, :MIDHIP_IDX]


def subset_metrics(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> Dict:
    upper_sel = np.array([COMMON_NAMES.index(n) for n in UPPER12])
    res = {
        "body14": compute_metrics(pred, gt, valid, root_index=NECK_IDX),
        "upper12": compute_metrics(
            pred[:, upper_sel], gt[:, upper_sel], valid[:, upper_sel],
            root_index=UPPER12.index("neck"),
        ),
    }
    for sub in res.values():
        sub.pop("per_axis_mae_m", None)
    res["body14"]["per_joint_names"] = COMMON_NAMES
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Dict] = {}
    pooled: Dict[str, List[np.ndarray]] = {}  # method → [pred...] 跨 run 拼接
    pooled_gt: List[np.ndarray] = []
    pooled_valid: List[np.ndarray] = []

    for gt_npz in sorted((args.data_root / "gt").glob("vp*/*.npz")):
        vp = gt_npz.parent.name
        run = gt_npz.stem
        sam_root = args.data_root / "sam3d" / vp / run
        if not all((sam_root / cam).is_dir() for cam in CAMERAS):
            print(f"skip {vp}/{run}: sam3d incomplete", flush=True)
            continue

        views = {cam: load_view_sequence(sam_root / cam) for cam in CAMERAS}
        gt_pose, gt_valid = load_gt_run(gt_npz)
        common = set(gt_pose)
        for seq in views.values():
            common &= set(seq)
        fids = sorted(common)
        if len(fids) < 10:
            print(f"skip {vp}/{run}: only {len(fids)} common frames", flush=True)
            continue

        gt_arr = np.stack([gt_pose[f] for f in fids])
        valid_arr = np.stack([gt_valid[f] for f in fids])
        # GT 髋/肩/颈无效则 canonical 系不可靠,整帧剔除
        frame_ok = valid_arr[:, [LHIP_IDX, RHIP_IDX, LSHO_IDX, RSHO_IDX, NECK_IDX]].all(axis=1)
        valid_arr &= frame_ok[:, None]
        gt_canon = canon(gt_arr)
        view_canon = {
            cam: canon(np.stack([views[cam][f] for f in fids])) for cam in CAMERAS
        }
        stack = np.stack([view_canon[cam] for cam in CAMERAS], axis=2)  # (T,J,V,3)
        conf = np.ones(stack.shape[:3], dtype=np.float32)

        run_res = {"num_frames": len(fids), "num_frames_canon_valid": int(frame_ok.sum())}
        preds = {f"single_{cam}": view_canon[cam] for cam in CAMERAS}
        for method in ("mean", "median"):
            preds[f"fuse_{method}"] = fuse_views(stack, conf, method)
        for name, pred in preds.items():
            run_res[name] = subset_metrics(pred, gt_canon, valid_arr)
            pooled.setdefault(name, []).append(pred)
        pooled_gt.append(gt_canon)
        pooled_valid.append(valid_arr)
        results[f"{vp}/{run}"] = run_res
        print(f"{vp}/{run}: {len(fids)} frames, "
              f"fuse_mean MPJPE={run_res['fuse_mean']['body14']['mpjpe_m']:.4f} "
              f"PA={run_res['fuse_mean']['body14']['pa_mpjpe_m']:.4f}", flush=True)

    if pooled_gt:
        gt_all = np.concatenate(pooled_gt)
        valid_all = np.concatenate(pooled_valid)
        overall = {
            name: subset_metrics(np.concatenate(preds), gt_all, valid_all)
            for name, preds in pooled.items()
        }
        results["overall"] = overall
        print("\n== OVERALL (pooled) ==")
        for name, res in overall.items():
            c, u = res["body14"], res["upper12"]
            print(f"{name:14s} body14: MPJPE={c['mpjpe_m']:.4f} PA={c['pa_mpjpe_m']:.4f} | "
                  f"upper12: MPJPE={u['mpjpe_m']:.4f} PA={u['pa_mpjpe_m']:.4f}", flush=True)

    out = args.output_dir / "driveandact_baselines.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
