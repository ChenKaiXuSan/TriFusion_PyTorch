#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from eval_fusion_baselines_pesudo_gt import (  # noqa: E402
    CAMERAS,
    ENV_NAMES,
    canonicalize_pose,
    compute_metrics,
    fuse_views,
    list_sam3d_files,
    load_gt_sequence,
    load_selected_sam3d_frames,
    mean,
    normalize_frame_id,
    safe_float,
    select_common_frame_ids,
)


FUSION_METHODS = ("mean", "median", "confidence")


def moving_average_pose(sequence: np.ndarray, window: int) -> np.ndarray:
    """Centered temporal moving average for (T,J,3) pose sequences."""
    if window <= 1:
        return sequence.astype(np.float32, copy=True)
    if window % 2 == 0:
        raise ValueError(f"smoothing window must be odd, got {window}")
    if sequence.ndim != 3:
        raise ValueError(f"expected pose sequence with shape (T,J,3), got {sequence.shape}")

    before = window // 2
    after = window - 1 - before
    padded = np.pad(sequence, ((before, after), (0, 0), (0, 0)), mode="edge")
    smoothed = np.stack(
        [np.mean(padded[idx : idx + window], axis=0) for idx in range(sequence.shape[0])],
        axis=0,
    )
    return smoothed.astype(np.float32)


def metric_payload(
    method: str,
    pred: np.ndarray,
    gt_pose: np.ndarray,
    gt_valid: np.ndarray,
    loaded_views: Dict[str, tuple[np.ndarray, np.ndarray]],
    frame_ids: List[str],
    subject_id: str,
    env_folder: str,
    args: argparse.Namespace,
    source: str,
    extra: Dict[str, Any] | None = None,
) -> Dict[str, Any] | None:
    metrics = compute_metrics(
        pred=pred,
        gt=gt_pose,
        valid_mask=gt_valid,
        root_index=args.root_index,
        pck_thresholds=tuple(args.pck_thresholds),
    )
    if not metrics:
        return None
    payload: Dict[str, Any] = {
        "method": method,
        "source": source,
        "person_id": subject_id,
        "environment": env_folder,
        "environment_name": ENV_NAMES.get(env_folder, env_folder),
        "total_frames": metrics["num_frames"],
        "canonicalize": bool(args.canonicalize),
        "mean_view_confidence": {
            cam: float(np.mean(loaded_views[cam][1])) for cam in CAMERAS
        },
        "metrics": metrics,
        "cameras": {
            source: {
                "num_frames": metrics["num_frames"],
                "num_keypoints": metrics["num_keypoints"],
                "metrics": metrics,
                "frame_ids": frame_ids,
            }
        },
    }
    if extra:
        payload.update(extra)
    return payload


def best_single_payload(payloads: Iterable[Dict[str, Any]]) -> Dict[str, Any] | None:
    candidates = [
        payload for payload in payloads
        if safe_float(payload.get("metrics", {}).get("mpjpe_m")) is not None
    ]
    if not candidates:
        return None
    best = min(candidates, key=lambda payload: float(payload["metrics"]["mpjpe_m"]))
    copied = json.loads(json.dumps(best))
    copied["method"] = "best_single"
    copied["source"] = "oracle_single_view"
    copied["selected_single_method"] = best["method"]
    copied["cameras"] = {
        "oracle_single_view": {
            **copied["cameras"][best["source"]],
            "selected_single_method": best["method"],
        }
    }
    return copied


def evaluate_subject_env(
    subject_id: str,
    env_folder: str,
    sam3d_root: Path,
    gt_root: Path,
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    view_files = {}
    for cam in CAMERAS:
        view_dir = sam3d_root / subject_id / env_folder / cam
        if not view_dir.exists():
            raise FileNotFoundError(f"SAM3D directory not found: {view_dir}")
        view_files[cam] = list_sam3d_files(view_dir)

    gt_pose, gt_valid, gt_frame_ids = load_gt_sequence(gt_root / subject_id / env_folder / "keypoints_3d.npz")
    frame_ids, gt_indices = select_common_frame_ids(
        view_files=view_files,
        gt_frame_ids=gt_frame_ids,
        gt_num_frames=gt_pose.shape[0],
        max_frames=args.max_frames,
        sampling=args.sampling,
    )
    normalized_view_files = {
        cam: {normalize_frame_id(fid): path for fid, path in files.items()}
        for cam, files in view_files.items()
    }
    loaded_views = {
        cam: load_selected_sam3d_frames(
            normalized_view_files[cam],
            frame_ids,
            num_workers=args.num_workers,
        )
        for cam in CAMERAS
    }

    view_pose = np.stack([loaded_views[cam][0] for cam in CAMERAS], axis=2)
    view_conf = np.stack([loaded_views[cam][1] for cam in CAMERAS], axis=2)
    gt_pose = gt_pose[gt_indices]
    gt_valid = gt_valid[gt_indices]

    n_joints = min(view_pose.shape[1], gt_pose.shape[1])
    view_pose = view_pose[:, :n_joints]
    view_conf = view_conf[:, :n_joints]
    gt_pose = gt_pose[:, :n_joints]
    gt_valid = gt_valid[:, :n_joints]

    if args.canonicalize:
        gt_pose = canonicalize_pose(
            gt_pose,
            neck_index=args.neck_index,
            left_shoulder_index=args.left_shoulder_index,
            right_shoulder_index=args.right_shoulder_index,
            mid_hip_index=args.mid_hip_index,
            eps=args.eps,
        )
        view_pose = np.stack(
            [
                canonicalize_pose(
                    view_pose[:, :, view_idx],
                    neck_index=args.neck_index,
                    left_shoulder_index=args.left_shoulder_index,
                    right_shoulder_index=args.right_shoulder_index,
                    mid_hip_index=args.mid_hip_index,
                    eps=args.eps,
                )
                for view_idx in range(view_pose.shape[2])
            ],
            axis=2,
        )

    payloads: List[Dict[str, Any]] = []
    single_payloads: List[Dict[str, Any]] = []
    if args.include_single:
        for view_idx, cam in enumerate(CAMERAS):
            payload = metric_payload(
                method=f"{cam}_single",
                pred=view_pose[:, :, view_idx],
                gt_pose=gt_pose,
                gt_valid=gt_valid,
                loaded_views=loaded_views,
                frame_ids=frame_ids,
                subject_id=subject_id,
                env_folder=env_folder,
                args=args,
                source=cam,
            )
            if payload is not None:
                single_payloads.append(payload)
                payloads.append(payload)
        best_payload = best_single_payload(single_payloads)
        if best_payload is not None:
            payloads.append(best_payload)

    if args.include_fusion:
        for method in FUSION_METHODS:
            pred = fuse_views(view_pose, view_conf, method)
            payload = metric_payload(
                method=method,
                pred=pred,
                gt_pose=gt_pose,
                gt_valid=gt_valid,
                loaded_views=loaded_views,
                frame_ids=frame_ids,
                subject_id=subject_id,
                env_folder=env_folder,
                args=args,
                source="fusion",
            )
            if payload is not None:
                payloads.append(payload)

    if args.include_smoothing:
        median_pred = fuse_views(view_pose, view_conf, "median")
        smoothed = moving_average_pose(median_pred, args.smoothing_window)
        payload = metric_payload(
            method=f"median_smooth_w{args.smoothing_window}",
            pred=smoothed,
            gt_pose=gt_pose,
            gt_valid=gt_valid,
            loaded_views=loaded_views,
            frame_ids=frame_ids,
            subject_id=subject_id,
            env_folder=env_folder,
            args=args,
            source="median_temporal_smoothing",
            extra={"smoothing_window": int(args.smoothing_window)},
        )
        if payload is not None:
            payloads.append(payload)

    return payloads


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    buckets: Dict[str, Dict[str, List[float]]] = {}
    env_buckets: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
    for row in rows:
        method = row["method"]
        env = row["environment_name"]
        buckets.setdefault(method, {})
        env_buckets.setdefault(method, {}).setdefault(env, {})
        for metric in (
            "mpjpe_m",
            "median_error_m",
            "root_mpjpe_m",
            "pa_mpjpe_m",
            "auc_0.15",
            "pck_0.05",
            "pck_0.10",
            "pck_0.15",
        ):
            value = safe_float(row.get(metric))
            if value is None:
                continue
            buckets[method].setdefault(metric, []).append(value)
            env_buckets[method][env].setdefault(metric, []).append(value)

    return {
        "overall": {
            method: {
                "num_pairs": len([row for row in rows if row["method"] == method]),
                **{metric: mean(values) for metric, values in metrics.items()},
            }
            for method, metrics in sorted(buckets.items())
        },
        "by_environment": {
            method: {
                env: {metric: mean(values) for metric, values in metrics.items()}
                for env, metrics in sorted(envs.items())
            }
            for method, envs in sorted(env_buckets.items())
        },
    }


def save_outputs(output_dir: Path, payloads: List[Dict[str, Any]], args: argparse.Namespace) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for payload in payloads:
        method = payload["method"]
        person_id = payload["person_id"]
        env_name = payload["environment_name"]
        method_dir = output_dir / method / person_id / env_name
        method_dir.mkdir(parents=True, exist_ok=True)
        with open(method_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        metrics = payload["metrics"]
        pck = metrics.get("pck", {})
        rows.append(
            {
                "method": method,
                "person_id": person_id,
                "environment": payload["environment"],
                "environment_name": env_name,
                "source": payload["source"],
                "selected_single_method": payload.get("selected_single_method"),
                "canonicalize": payload["canonicalize"],
                "smoothing_window": payload.get("smoothing_window"),
                "num_frames": metrics.get("num_frames"),
                "num_keypoints": metrics.get("num_keypoints"),
                "num_valid_points": metrics.get("num_valid_points"),
                "mpjpe_m": metrics.get("mpjpe_m"),
                "median_error_m": metrics.get("median_error_m"),
                "root_mpjpe_m": metrics.get("root_mpjpe_m"),
                "pa_mpjpe_m": metrics.get("pa_mpjpe_m"),
                "pck_0.02": pck.get("0.02"),
                "pck_0.05": pck.get("0.05"),
                "pck_0.10": pck.get("0.10"),
                "pck_0.15": pck.get("0.15"),
                "auc_0.15": metrics.get("auc_0.15"),
                "source_file": str(method_dir / "metrics.json"),
            }
        )

    fieldnames = sorted({key for row in rows for key in row.keys()})
    csv_path = output_dir / "additional_baseline_detailed.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    methods = sorted({row["method"] for row in rows})
    summary_path = output_dir / "additional_baseline_summary.json"
    summary = {
        "num_rows": len(rows),
        "methods": methods,
        "canonicalize": bool(args.canonicalize),
        "smoothing_window": int(args.smoothing_window),
        "summary": summarize(rows),
        "notes": {
            "best_single": "Oracle per subject/environment selection among front_single, left_single, and right_single by MPJPE.",
            "median_smooth": "Centered moving average applied after canonicalized median fusion.",
        },
        "output_files": {
            "detailed_csv": str(csv_path),
            "summary_json": str(summary_path),
        },
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate additional canonicalized single-view, fixed-fusion, and smoothing baselines."
    )
    parser.add_argument("--sam3d-root", type=str, default="/home/data/xchen/drive/sam3d_body_results_right")
    parser.add_argument("--gt-root", type=str, default="/home/data/xchen/drive/sam3d_body_triangulated_gt")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/home/workspace/kaixu/code/MultiView_DriverAction_PyTorch/TriPoseFusion/eval/logs/comparison_additional_baselines_pseudo_gt",
    )
    parser.add_argument("--subject", type=str, default=None, help="Optional subject id, e.g. 01")
    parser.add_argument("--env", type=str, default=None, help="Optional raw env folder name")
    parser.add_argument("--root-index", type=int, default=0)
    parser.add_argument("--neck-index", type=int, default=51)
    parser.add_argument("--left-shoulder-index", type=int, default=5)
    parser.add_argument("--right-shoulder-index", type=int, default=6)
    parser.add_argument("--mid-hip-index", type=int, default=-1)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--pck-thresholds", type=float, nargs="+", default=[0.02, 0.05, 0.10, 0.15])
    parser.add_argument("--max-frames", type=int, default=None, help="Optional debug limit after frame alignment.")
    parser.add_argument("--num-workers", type=int, default=4, help="Thread workers for selected SAM3D frame loading.")
    parser.add_argument("--smoothing-window", type=int, default=5, help="Odd moving-average window for median smoothing.")
    parser.add_argument("--sampling", choices=("uniform", "head"), default="uniform")
    parser.add_argument("--no-canonicalize", dest="canonicalize", action="store_false")
    parser.add_argument("--no-single", dest="include_single", action="store_false")
    parser.add_argument("--no-fusion", dest="include_fusion", action="store_false")
    parser.add_argument("--no-smoothing", dest="include_smoothing", action="store_false")
    parser.set_defaults(
        canonicalize=True,
        include_single=True,
        include_fusion=True,
        include_smoothing=True,
    )
    args = parser.parse_args()
    if args.smoothing_window < 1 or args.smoothing_window % 2 == 0:
        parser.error("--smoothing-window must be a positive odd integer")
    return args


def main() -> None:
    args = parse_args()
    sam3d_root = Path(args.sam3d_root)
    gt_root = Path(args.gt_root)

    if args.subject:
        subjects = [args.subject]
    else:
        subjects = sorted(path.name for path in gt_root.iterdir() if path.is_dir())

    envs = [args.env] if args.env else list(ENV_NAMES.keys())
    all_payloads: List[Dict[str, Any]] = []
    failures: List[Dict[str, str]] = []

    for subject_id in subjects:
        for env_folder in envs:
            try:
                payloads = evaluate_subject_env(subject_id, env_folder, sam3d_root, gt_root, args)
                all_payloads.extend(payloads)
                print(f"[OK] subject={subject_id} env={ENV_NAMES.get(env_folder, env_folder)} methods={len(payloads)}")
            except Exception as exc:  # noqa: BLE001
                failures.append({"subject": subject_id, "environment": env_folder, "error": str(exc)})
                print(f"[WARN] subject={subject_id} env={env_folder}: {exc}")

    output_dir = Path(args.output_dir)
    save_outputs(output_dir, all_payloads, args)
    if failures:
        with open(output_dir / "failures.json", "w", encoding="utf-8") as f:
            json.dump(failures, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(all_payloads)} result payloads to {output_dir}")
    if failures:
        print(f"Skipped {len(failures)} subject/env pairs; see {output_dir / 'failures.json'}")


if __name__ == "__main__":
    main()
