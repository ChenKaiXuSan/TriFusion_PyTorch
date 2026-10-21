#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Drive&Act 上的 DLT 三角化基线（LAYQ: direct 2D triangulation baseline）。

用每个 run 自带的 calibration.json（已验证 R,t 为 world→cam，世界系 = ids_1
即 inner_mirror 相机系）对三视角 SAM3D 2D 关键点做 DLT 三角化。报告两套坐标系：
- **absolute**：直接在 GT 世界系比较，未经任何对齐/规范化的绝对 MPJPE（毫米），
  与 Drive&Act 上 SOTA 的 22.6–30.4 mm 严格同口径；
- **canonical**：与 eval_driveandact_baselines.py 完全相同的 canonicalize 协议，
  可直接并入融合基线表。
关节协议复用 body14 / upper12；neck 与基线一致取三角化后双肩中点。
另报三个两视角组合，作为视角冗余/丢视角鲁棒性的参考。

用法:
    python TriPoseFusion/eval/eval_driveandact_triangulation.py \
        --data-root .../driveandact/tripose_eval \
        --da-root   .../driveandact \
        --output-dir TriPoseFusion/docs/driveandact_results
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.eval_driveandact_baselines import (  # noqa: E402
    COMMON_NAMES, LHIP_IDX, LSHO_IDX, MHR_TO_BODY25, NECK_IDX, RHIP_IDX, RSHO_IDX,
    canon, load_gt_run, subset_metrics,
)

VIEW_CAMS = {
    "front": ("inner_mirror", "ids_1"),
    "left": ("a_column_driver", "ids_5"),
    "right": ("a_column_co_driver", "ids_2"),
}
VIEW_SETS: Dict[str, Tuple[str, ...]] = {
    "tri_front_left_right": ("front", "left", "right"),
    "tri_left_right": ("left", "right"),
    "tri_front_left": ("front", "left"),
    "tri_front_right": ("front", "right"),
}


def quat_to_R(w: float, x: float, y: float, z: float) -> np.ndarray:
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


class Camera:
    def __init__(self, R: np.ndarray, t: np.ndarray, K: np.ndarray, dist: np.ndarray) -> None:
        self.R, self.t, self.K, self.dist = R, t, K, dist
        self.P = np.hstack([R, t[:, None]])  # 作用于去畸变归一化坐标

    @classmethod
    def from_json(cls, calib_json: Path) -> "Camera":
        c = json.loads(calib_json.read_text())
        e, i = c["extrinsics"], c["intrinsics"]
        r, t = e["rotation"], e["translation"]
        K = np.array([
            [i["focallength"]["fx"], 0, i["principal_point"]["cx"]],
            [0, i["focallength"]["fy"], i["principal_point"]["cy"]],
            [0, 0, 1],
        ])
        d = i["distortion"]
        return cls(
            quat_to_R(r["w"], r["x"], r["y"], r["z"]),
            np.array([t["x"], t["y"], t["z"]]),
            K, np.array([d["k1"], d["k2"], d["p1"], d["p2"], d["k3"]]),
        )

    _UNDIST_CRITERIA = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 50, 1e-10)

    def undistort(self, uv: np.ndarray) -> np.ndarray:
        # 默认 undistortPoints 仅 5 次迭代，大离轴角下未收敛（k1=-0.27 时可达 cm 级）
        pts = uv.reshape(-1, 1, 2).astype(np.float64)
        return cv2.undistortPointsIter(
            pts, self.K, self.dist, None, None, self._UNDIST_CRITERIA).reshape(-1, 2)

    def project(self, xyz: np.ndarray) -> np.ndarray:
        """世界系 (...,3) → 像素 (...,2)，含畸变。"""
        flat = xyz.reshape(-1, 1, 3).astype(np.float64)
        uv, _ = cv2.projectPoints(flat, cv2.Rodrigues(self.R)[0], self.t, self.K, self.dist)
        return uv.reshape(*xyz.shape[:-1], 2)


def triangulate(obs: Sequence[Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    rows = []
    for P, xy in obs:
        rows.append(xy[0] * P[2] - P[0])
        rows.append(xy[1] * P[2] - P[1])
    _, _, vt = np.linalg.svd(np.asarray(rows))
    X = vt[-1]
    return X[:3] / X[3]


def load_view_kpt2d(view_dir: Path) -> Dict[int, np.ndarray]:
    mhr_idx = np.array([i for i, _ in MHR_TO_BODY25])
    out: Dict[int, np.ndarray] = {}
    for fp in sorted(view_dir.glob("*_sam3d_body.npz")):
        fid = int(fp.name.split("_")[0])
        with np.load(fp, allow_pickle=True) as data:
            kpt2d = np.asarray(data["output"].item()["pred_keypoints_2d"], dtype=np.float64)
        out[fid] = kpt2d[mhr_idx, :2]
    return out


def self_test() -> None:
    """随机 3D → 三相机含畸变投影 → 三角化还原，误差应 <1e-3 m。"""
    rng = np.random.default_rng(0)
    K = np.array([[567.0, 0, 640.0], [0, 567.0, 512.0], [0, 0, 1]])
    dist = np.array([-0.2661, 0.0549, 0.0, 0.0, 0.0])
    cams = [
        Camera(quat_to_R(np.cos(a / 2), 0, np.sin(a / 2), 0),
               np.array([0.3 * k, 0.02 * k, 0.1 * k]), K, dist)
        for k, a in enumerate((-0.6, 0.0, 0.6))
    ]
    pts = rng.normal([0.3, 0.0, 1.5], 0.25, (200, 3))  # 覆盖到 ~45° 离轴
    errs = [
        np.linalg.norm(triangulate([(c.P, c.undistort(c.project(X))[0]) for c in cams]) - X)
        for X in pts
    ]
    assert max(errs) < 1e-3, f"self-test failed: max err {max(errs)}"
    print(f"[self-test] round-trip max error {max(errs):.2e} m  OK", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--da-root", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    self_test()

    results: Dict[str, Dict] = {}
    pooled: Dict[str, Dict[str, List[np.ndarray]]] = {"absolute": {}, "canonical": {}}
    pooled_gt = {"absolute": [], "canonical": []}
    pooled_valid: List[np.ndarray] = []
    reproj_px: List[np.ndarray] = []

    for gt_npz in sorted((args.data_root / "gt").glob("vp*/*.npz")):
        vp, run = gt_npz.parent.name, gt_npz.stem
        sam_root = args.data_root / "sam3d" / vp / run
        if not all((sam_root / v).is_dir() for v in VIEW_CAMS):
            print(f"skip {vp}/{run}: sam3d incomplete", flush=True)
            continue

        cams = {
            v: Camera.from_json(args.da_root / vdir / vp / f"{run}.{ids}.calibration.json")
            for v, (vdir, ids) in VIEW_CAMS.items()
        }
        kpt2d = {v: load_view_kpt2d(sam_root / v) for v in VIEW_CAMS}
        gt_pose, gt_valid = load_gt_run(gt_npz)
        fids = sorted(set(gt_pose).intersection(*(set(k) for k in kpt2d.values())))
        if len(fids) < 10:
            print(f"skip {vp}/{run}: only {len(fids)} common frames", flush=True)
            continue

        gt_arr = np.stack([gt_pose[f] for f in fids])
        valid_arr = np.stack([gt_valid[f] for f in fids])
        # 与基线脚本相同的帧筛选（canonical 系需要髋/肩/颈），两套坐标系共用同一帧集
        frame_ok = valid_arr[:, [LHIP_IDX, RHIP_IDX, LSHO_IDX, RSHO_IDX, NECK_IDX]].all(axis=1)
        valid_arr &= frame_ok[:, None]

        norm = {v: np.stack([cams[v].undistort(kpt2d[v][f]) for f in fids]) for v in VIEW_CAMS}
        J = len(COMMON_NAMES)
        preds: Dict[str, np.ndarray] = {}
        for name, views in VIEW_SETS.items():
            pred = np.zeros((len(fids), J, 3), dtype=np.float32)
            for k in range(len(fids)):
                for j in range(J):
                    pred[k, j] = triangulate([(cams[v].P, norm[v][k, j]) for v in views])
            pred[:, NECK_IDX] = 0.5 * (pred[:, LSHO_IDX] + pred[:, RSHO_IDX])
            preds[name] = pred

        # 标定/同步 sanity：GT 3D 投影 vs SAM3D 2D 的像素残差
        res = []
        for v in VIEW_CAMS:
            uv = cams[v].project(gt_arr)
            det = np.stack([kpt2d[v][f] for f in fids])
            res.append(np.linalg.norm(uv - det, axis=-1)[valid_arr])
        res_px = np.concatenate(res)
        reproj_px.append(res_px)

        gt_canon = canon(gt_arr)
        run_res: Dict = {
            "num_frames": len(fids),
            "num_frames_canon_valid": int(frame_ok.sum()),
            "gtproj_vs_sam2d_px": {"median": float(np.median(res_px)),
                                   "mean": float(np.mean(res_px))},
        }
        for name, pred in preds.items():
            run_res[name] = {
                "absolute": subset_metrics(pred, gt_arr, valid_arr),
                "canonical": subset_metrics(canon(pred), gt_canon, valid_arr),
            }
            pooled["absolute"].setdefault(name, []).append(pred)
            pooled["canonical"].setdefault(name, []).append(canon(pred))
        pooled_gt["absolute"].append(gt_arr)
        pooled_gt["canonical"].append(gt_canon)
        pooled_valid.append(valid_arr)
        results[f"{vp}/{run}"] = run_res
        a = run_res["tri_front_left_right"]["absolute"]["body14"]
        print(f"{vp}/{run}: {len(fids)} frames | 3-view abs MPJPE={a['mpjpe_m']*1000:.1f}mm "
              f"PA={a['pa_mpjpe_m']*1000:.1f}mm | GT-proj residual median {np.median(res_px):.1f}px",
              flush=True)

    if pooled_valid:
        valid_all = np.concatenate(pooled_valid)
        overall: Dict = {
            "gtproj_vs_sam2d_px": {"median": float(np.median(np.concatenate(reproj_px)))}
        }
        for space in ("absolute", "canonical"):
            gt_all = np.concatenate(pooled_gt[space])
            for name, plist in pooled[space].items():
                overall.setdefault(name, {})[space] = subset_metrics(
                    np.concatenate(plist), gt_all, valid_all)
        results["overall"] = overall
        print("\n== OVERALL DLT triangulation (pooled, mm) ==")
        print(f"{'method':22s} | abs body14 MPJPE / PA | abs upper12 MPJPE / PA | canon body14 MPJPE / PA")
        for name in VIEW_SETS:
            ab, au = overall[name]["absolute"]["body14"], overall[name]["absolute"]["upper12"]
            cb = overall[name]["canonical"]["body14"]
            print(f"{name:22s} | {ab['mpjpe_m']*1000:6.1f} / {ab['pa_mpjpe_m']*1000:5.1f}"
                  f"        | {au['mpjpe_m']*1000:6.1f} / {au['pa_mpjpe_m']*1000:5.1f}"
                  f"         | {cb['mpjpe_m']*1000:6.1f} / {cb['pa_mpjpe_m']*1000:5.1f}", flush=True)
        print(f"GT-projection vs SAM3D-2D residual median: "
              f"{overall['gtproj_vs_sam2d_px']['median']:.1f}px")

    out = args.output_dir / "driveandact_triangulation.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
