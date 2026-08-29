#!/usr/bin/env python3
"""为学习型融合基线（rebuttal E 点）构建缓存。

对每个 <person>/<env>：加载三视角 SAM3D 52 关节序列 + 三角化伪 GT，
按共同帧对齐，KEEP 映射到 52 关节模型空间，分别 canonicalize
（与 Table 3 基线协议一致），存为单个 npz：
  view_pose (T,52,V,3) fp16 canonical、view_conf (T,52,V) fp16、
  gt_pose (T,52,3) fp16 canonical、gt_valid (T,52) bool、frame_ids。
"""
from __future__ import annotations

import argparse
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from map_config import KEEP_KEYPOINT_INDICES  # noqa: E402
from eval_fusion_baselines_pesudo_gt import (  # noqa: E402
    CAMERAS,
    canonicalize_pose,
    list_sam3d_files,
    load_gt_sequence,
    load_selected_sam3d_frames,
    normalize_frame_id,
    select_common_frame_ids,
)


def build_one(
    subject_id: str,
    env_folder: str,
    sam3d_root: Path,
    gt_root: Path,
    out_dir: Path,
    num_workers: int,
) -> str:
    out_path = out_dir / f"{subject_id}_{env_folder}.npz"
    if out_path.exists():
        return f"skip (exists): {out_path.name}"

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
        sampling="uniform",
    )
    normalized_view_files = {
        cam: {normalize_frame_id(fid): path for fid, path in files.items()}
        for cam, files in view_files.items()
    }
    loaded_views = {
        cam: load_selected_sam3d_frames(
            normalized_view_files[cam], frame_ids, num_workers=num_workers
        )
        for cam in CAMERAS
    }
    view_pose = np.stack([loaded_views[cam][0] for cam in CAMERAS], axis=2)  # (T,J,V,3)
    view_conf = np.stack([loaded_views[cam][1] for cam in CAMERAS], axis=2)  # (T,J,V)
    gt_pose = gt_pose[gt_indices]
    gt_valid = gt_valid[gt_indices]

    keep = np.asarray(KEEP_KEYPOINT_INDICES, dtype=np.int64)
    if view_pose.shape[1] > keep.max():
        view_pose = view_pose[:, keep]
        view_conf = view_conf[:, keep]
    if gt_pose.shape[1] > keep.max():
        gt_pose = gt_pose[:, keep]
        gt_valid = gt_valid[:, keep]

    gt_canon = canonicalize_pose(gt_pose)
    canon_views = [
        canonicalize_pose(view_pose[:, :, v]) for v in range(view_pose.shape[2])
    ]
    view_canon = np.stack(canon_views, axis=2)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        view_pose=view_canon.astype(np.float16),
        view_conf=view_conf.astype(np.float16),
        gt_pose=gt_canon.astype(np.float16),
        gt_valid=gt_valid.astype(bool),
        frame_ids=np.asarray(frame_ids),
        cameras=np.asarray(CAMERAS),
    )
    return f"built: {out_path.name} T={view_canon.shape[0]}"


def _job(args_tuple):
    subject_id, env_folder, sam3d_root, gt_root, out_dir, workers = args_tuple
    try:
        return build_one(
            subject_id, env_folder, Path(sam3d_root), Path(gt_root), Path(out_dir), workers
        )
    except Exception as exc:  # noqa: BLE001
        return f"FAILED {subject_id}/{env_folder}: {exc}\n{traceback.format_exc()}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sam3d-root",
        type=Path,
        default=Path("/work/1/SKIING/chenkaixu/data/drive/sam3d_body_results_right"),
    )
    parser.add_argument(
        "--gt-root",
        type=Path,
        default=Path("/work/1/SKIING/chenkaixu/data/drive/sam3d_body_triangulated_gt"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/work/1/SKIING/chenkaixu/data/drive/learned_fusion_cache"),
    )
    parser.add_argument("--subject", type=str, default=None)
    parser.add_argument("--env", type=str, default=None)
    parser.add_argument("--num-procs", type=int, default=4)
    parser.add_argument("--io-workers", type=int, default=4)
    args = parser.parse_args()

    pairs = []
    for subj_dir in sorted(args.sam3d_root.iterdir()):
        if not subj_dir.is_dir():
            continue
        if args.subject and subj_dir.name != args.subject:
            continue
        for env_dir in sorted(subj_dir.iterdir()):
            if not env_dir.is_dir():
                continue
            if args.env and env_dir.name != args.env:
                continue
            pairs.append(
                (
                    subj_dir.name,
                    env_dir.name,
                    str(args.sam3d_root),
                    str(args.gt_root),
                    str(args.out_dir),
                    args.io_workers,
                )
            )

    print(f"{len(pairs)} sequences to cache -> {args.out_dir}", flush=True)
    if args.num_procs <= 1:
        for p in pairs:
            print(_job(p), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.num_procs) as pool:
            for msg in pool.map(_job, pairs):
                print(msg, flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
