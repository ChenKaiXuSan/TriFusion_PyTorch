#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from __future__ import annotations

import csv
import json
import logging
import math
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

import hydra
import numpy as np
import torch
from dataloader.data_loader import DriverKPTDataModule
from map_config import KEEP_KEYPOINT_INDICES
from omegaconf import DictConfig, ListConfig, OmegaConf
from pytorch_lightning import seed_everything
from tqdm.auto import tqdm
from train import load_fold_dataset_idx_from_json
from trainer.train_triple_fusion import TriFusionPoseTrainer

logger = logging.getLogger(__name__)

ENV_NAMES = {
    "夜多い": "Night_High",
    "夜少ない": "Night_Low",
    "昼多い": "Day_High",
    "昼少ない": "Day_Low",
}


@dataclass
class TriangulatedSequence:
    keypoints_3d: np.ndarray  # (T, J, 3)
    valid_mask: np.ndarray  # (T, J)
    frame_ids: np.ndarray | None = None  # (T,)


def _cfg_get(config: DictConfig, path: str, default: Any = None) -> Any:
    value = OmegaConf.select(config, path)
    return default if value is None else value


def _selected_folds(config: DictConfig, all_folds: Iterable[int]) -> list[int]:
    fold = _cfg_get(config, "eval.fold")
    if fold is None or str(fold).lower() == "all":
        return sorted(int(item) for item in all_folds)
    return [int(fold)]


def _resolve_ckpt(config: DictConfig) -> Path:
    ckpt_path = _cfg_get(config, "eval.ckpt_path")
    if not ckpt_path:
        raise ValueError("eval.ckpt_path is required for triangulated GT evaluation.")
    ckpt = Path(str(ckpt_path)).expanduser().resolve()
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {ckpt}")
    return ckpt


def _load_module(
    config: DictConfig, ckpt: Path, device: torch.device
) -> TriFusionPoseTrainer:
    payload = torch.load(str(ckpt), map_location="cpu")
    state_dict = (
        payload["state_dict"]
        if isinstance(payload, dict) and "state_dict" in payload
        else payload
    )
    module = TriFusionPoseTrainer(config)
    missing, unexpected = module.load_state_dict(state_dict, strict=False)
    if missing:
        logger.warning("Missing keys when loading ckpt: %d", len(missing))
    if unexpected:
        logger.warning("Unexpected keys when loading ckpt: %d", len(unexpected))
    module.to(device)
    module.eval()
    return module


def _uniform_sample_indices(total_frames: int, target_t: int) -> np.ndarray:
    if total_frames <= 0:
        raise ValueError("Cannot sample indices from empty sequence")
    if target_t <= 0 or total_frames == target_t:
        return np.arange(total_frames, dtype=np.int64)
    if total_frames == 1:
        return np.zeros((target_t,), dtype=np.int64)
    # Keep indices consistent with dataset._uniform_temporal_sample(torch.linspace(...).long()).
    return torch.linspace(0, total_frames - 1, steps=target_t).long().cpu().numpy()


def _normalize_frame_id(frame_id: Any) -> str:
    text = str(frame_id).strip()
    if text == "":
        return text
    try:
        return str(int(text))
    except ValueError:
        return text


def _extract_frame_id_from_sam3d_file(path: Path) -> str:
    stem = path.stem
    suffix = "_sam3d_body"
    if stem.endswith(suffix):
        stem = stem[: -len(suffix)]
    return _normalize_frame_id(stem)


def _selected_source_frame_ids(
    sample: Any,
    start_frame: int,
    end_frame: int | None,
    target_t: int,
    source_view: str,
) -> List[str]:
    if sample.sam3d_kpts is None or source_view not in sample.sam3d_kpts:
        raise ValueError(
            f"Missing source view {source_view!r} for sample {sample.person_id}/{sample.env_folder}"
        )

    view_dir = Path(sample.sam3d_kpts[source_view])
    files = sorted(view_dir.glob("*_sam3d_body.npz"))
    safe_start = max(0, int(start_frame))
    safe_end = len(files) if end_frame is None else min(len(files), max(0, int(end_frame)))
    seg_files = files[safe_start:safe_end]
    if len(seg_files) <= 0:
        raise ValueError(
            f"Empty source segment for {sample.person_id}/{sample.env_folder} at view={source_view}. "
            f"requested=({start_frame}, {end_frame}), clamped=({safe_start}, {safe_end}), total={len(files)}"
        )

    indices = _uniform_sample_indices(len(seg_files), target_t)
    return [_extract_frame_id_from_sam3d_file(seg_files[int(i)]) for i in indices]


def _select_gt_by_frame_ids(
    tri_seq: TriangulatedSequence,
    source_frame_ids: List[str],
) -> tuple[np.ndarray, np.ndarray]:
    if tri_seq.frame_ids is None:
        raise ValueError("GT has no frame_ids; cannot enforce strict same-time alignment.")

    gt_map = {_normalize_frame_id(fid): idx for idx, fid in enumerate(tri_seq.frame_ids.tolist())}
    missing = [fid for fid in source_frame_ids if fid not in gt_map]
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(f"Missing {len(missing)} frame_ids in GT (first 5: {preview})")

    idx = np.asarray([gt_map[fid] for fid in source_frame_ids], dtype=np.int64)
    return tri_seq.keypoints_3d[idx], tri_seq.valid_mask[idx]


def _build_sample_lookup(samples: List[Any]) -> Dict[tuple[str, str], Any]:
    lookup: Dict[tuple[str, str], Any] = {}
    for sample in samples:
        lookup[(str(sample.person_id), str(sample.env_folder))] = sample
    return lookup


def _resolve_source_view(config: DictConfig) -> str:
    explicit = _cfg_get(config, "eval.source_view")
    if explicit:
        return str(explicit)

    view_name = config.train.view_name
    if isinstance(view_name, ListConfig):
        if len(view_name) == 0:
            raise ValueError("train.view_name is empty")
        return str(view_name[0])
    if isinstance(view_name, (list, tuple)):
        if len(view_name) == 0:
            raise ValueError("train.view_name is empty")
        return str(view_name[0])
    return str(view_name)


def _apply_joint_selection(
    gt_kpt: np.ndarray, gt_valid: np.ndarray, pred_joints: int
) -> tuple[np.ndarray, np.ndarray]:
    if gt_kpt.shape[1] == pred_joints:
        return gt_kpt, gt_valid

    keep = np.asarray(KEEP_KEYPOINT_INDICES, dtype=np.int64)
    if keep.size > 0 and keep.max() < gt_kpt.shape[1] and keep.size == pred_joints:
        return gt_kpt[:, keep], gt_valid[:, keep]

    joints = min(pred_joints, gt_kpt.shape[1])
    logger.warning(
        "Joint count mismatch pred=%d gt=%d. Fallback to first %d joints.",
        pred_joints,
        gt_kpt.shape[1],
        joints,
    )
    return gt_kpt[:, :joints], gt_valid[:, :joints]


def _procrustes_align(
    pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    pred_mean = pred.mean(dim=0, keepdim=True)
    gt_mean = gt.mean(dim=0, keepdim=True)
    pred_center = pred - pred_mean
    gt_center = gt - gt_mean

    cov = pred_center.transpose(0, 1) @ gt_center
    u, s, v_t = torch.linalg.svd(cov, full_matrices=False)
    r = v_t.transpose(0, 1) @ u.transpose(0, 1)

    if torch.det(r) < 0:
        v_t = v_t.clone()
        v_t[-1, :] *= -1
        r = v_t.transpose(0, 1) @ u.transpose(0, 1)

    var_pred = (pred_center**2).sum().clamp_min(eps)
    scale = s.sum() / var_pred
    aligned = scale * (pred_center @ r) + gt_mean
    return aligned


def _compute_sample_metrics(
    pred_btj3: torch.Tensor,
    gt_btj3: torch.Tensor,
    valid_btj: torch.Tensor,
    pck_thresholds: list[float],
) -> Dict[str, float]:
    dist = torch.linalg.norm(pred_btj3 - gt_btj3, dim=-1)
    valid = valid_btj.bool()
    valid_count = int(valid.sum().item())
    if valid_count == 0:
        return {}

    mpjpe = float(dist[valid].mean().item())

    pa_dist_values = []
    for frame_idx in range(pred_btj3.shape[0]):
        frame_valid = valid[frame_idx]
        if int(frame_valid.sum().item()) < 3:
            continue
        pred_frame = pred_btj3[frame_idx][frame_valid]
        gt_frame = gt_btj3[frame_idx][frame_valid]
        pred_aligned = _procrustes_align(pred_frame, gt_frame)
        pa_dist_values.append(torch.linalg.norm(pred_aligned - gt_frame, dim=-1))

    if pa_dist_values:
        pa_mpjpe = float(torch.cat(pa_dist_values).mean().item())
    else:
        pa_mpjpe = float("nan")

    metrics: Dict[str, float] = {
        "mpjpe": mpjpe,
        "pa_mpjpe": pa_mpjpe,
        "valid_joints": float(valid_count),
    }
    for thr in pck_thresholds:
        metrics[f"pck@{thr}"] = float((dist[valid] <= thr).float().mean().item())
    return metrics


def _merge_metric_lists(per_item_metrics: list[Dict[str, float]]) -> Dict[str, float]:
    buckets: Dict[str, list[float]] = defaultdict(list)
    for item in per_item_metrics:
        for key, value in item.items():
            if isinstance(value, float) and not math.isnan(value):
                buckets[key].append(value)
    out: Dict[str, float] = {}
    for key, values in buckets.items():
        out[key] = float(sum(values) / len(values))
    out["num_items"] = float(len(per_item_metrics))
    return out


def _aggregate_fold_metrics(
    per_fold: Dict[str, Dict[str, float]],
) -> Dict[str, Dict[str, float]]:
    buckets: Dict[str, list[float]] = defaultdict(list)
    for fold_metrics in per_fold.values():
        for key, value in fold_metrics.items():
            if isinstance(value, float) and not math.isnan(value):
                buckets[key].append(value)
    agg: Dict[str, Dict[str, float]] = {}
    for key, values in buckets.items():
        if not values:
            continue
        mean = sum(values) / len(values)
        var = sum((v - mean) ** 2 for v in values) / len(values)
        agg[key] = {
            "mean": float(mean),
            "std": float(math.sqrt(var)),
            "n": float(len(values)),
        }
    return agg


def _save_results(
    output_dir: Path,
    per_fold: Dict[str, Dict[str, float]],
    aggregate: Dict[str, Dict[str, float]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "triangulated_eval_metrics.json"
    csv_path = output_dir / "triangulated_eval_metrics.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"per_fold": per_fold, "aggregate": aggregate}, f, indent=2)

    metric_names = sorted(
        {name for metrics in per_fold.values() for name in metrics.keys()}
    )
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["fold"] + metric_names)
        for fold, metrics in sorted(per_fold.items(), key=lambda item: int(item[0])):
            writer.writerow([fold] + [metrics.get(name, "") for name in metric_names])
        writer.writerow([])
        writer.writerow(["metric", "mean", "std", "n"])
        for metric, stats in sorted(aggregate.items()):
            writer.writerow([metric, stats["mean"], stats["std"], stats["n"]])

    logger.info("Saved triangulated evaluation JSON: %s", json_path)
    logger.info("Saved triangulated evaluation CSV : %s", csv_path)


def _aggregate_person_env_metrics(
    item_records: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[tuple[str, str], List[Dict[str, float]]] = defaultdict(list)
    for record in item_records:
        grouped[(record["person_id"], record["env_folder"])].append(record["metrics"])

    out: Dict[str, Dict[str, Any]] = {}
    for (person_id, env_folder), metrics_list in grouped.items():
        merged = _merge_metric_lists(metrics_list)
        out[f"{person_id}/{env_folder}"] = {
            "person_id": person_id,
            "env_folder": env_folder,
            "metrics": merged,
        }
    return out


def _save_person_env_results(
    output_dir: Path,
    fold: int,
    person_env_results: Dict[str, Dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"person_env_metrics_fold_{fold}.json"
    csv_path = output_dir / f"person_env_metrics_fold_{fold}.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(person_env_results, f, indent=2, ensure_ascii=False)

    metric_names = sorted(
        {
            metric_name
            for entry in person_env_results.values()
            for metric_name in entry.get("metrics", {}).keys()
        }
    )
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["person_id", "env_folder"] + metric_names)
        for key in sorted(person_env_results.keys()):
            entry = person_env_results[key]
            metrics = entry.get("metrics", {})
            writer.writerow(
                [entry.get("person_id", ""), entry.get("env_folder", "")]
                + [metrics.get(name, "") for name in metric_names]
            )

    logger.info("Saved person/env metrics JSON: %s", json_path)
    logger.info("Saved person/env metrics CSV : %s", csv_path)


def _to_m(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    return number


def _pck_dict_from_metrics(metrics: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key, value in metrics.items():
        if not str(key).startswith("pck@"):
            continue
        thr = str(key).split("@", maxsplit=1)[1]
        try:
            out[f"{float(thr):.2f}"] = float(value)
        except (TypeError, ValueError):
            continue
    return out


def _build_comparison_summary(
    all_subjects: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    env_buckets: Dict[str, Dict[str, Any]] = {}
    for subject in all_subjects:
        person_id = subject.get("person_id", "")
        for env_name, env_stats in subject.get("environments", {}).items():
            bucket = env_buckets.setdefault(
                env_name,
                {
                    "subjects": [],
                    "mpjpe_m": [],
                    "median_error_m": [],
                    "root_mpjpe_m": [],
                    "pa_mpjpe_m": [],
                    "pck_0.05": [],
                    "pck_0.10": [],
                    "pck_0.15": [],
                    "auc_0.15": [],
                },
            )
            if person_id and person_id not in bucket["subjects"]:
                bucket["subjects"].append(person_id)

            metrics = env_stats.get("metrics", {})
            value_map = {
                "mpjpe_m": metrics.get("mpjpe_m"),
                "median_error_m": metrics.get("median_error_m"),
                "root_mpjpe_m": metrics.get("root_mpjpe_m"),
                "pa_mpjpe_m": metrics.get("pa_mpjpe_m"),
                "auc_0.15": metrics.get("auc_0.15"),
            }
            for name, value in value_map.items():
                if isinstance(value, (int, float)) and not math.isnan(float(value)):
                    bucket[name].append(float(value))

            pck = metrics.get("pck", {})
            for pck_key, bucket_key in (("0.05", "pck_0.05"), ("0.10", "pck_0.10"), ("0.15", "pck_0.15")):
                pck_val = pck.get(pck_key)
                if isinstance(pck_val, (int, float)):
                    bucket[bucket_key].append(float(pck_val))

    summary: Dict[str, Dict[str, Any]] = {}
    for env_name, bucket in env_buckets.items():
        def _avg(values: List[float]) -> float:
            return float(sum(values) / len(values)) if values else 0.0

        summary[env_name] = {
            "subjects": bucket["subjects"],
            "avg_mpjpe_m": _avg(bucket["mpjpe_m"]),
            "avg_median_error_m": _avg(bucket["median_error_m"]),
            "avg_root_mpjpe_m": _avg(bucket["root_mpjpe_m"]),
            "avg_pa_mpjpe_m": _avg(bucket["pa_mpjpe_m"]),
            "avg_pck_0.05": _avg(bucket["pck_0.05"]),
            "avg_pck_0.10": _avg(bucket["pck_0.10"]),
            "avg_pck_0.15": _avg(bucket["pck_0.15"]),
            "avg_auc_0.15": _avg(bucket["auc_0.15"]),
        }
    return summary


def _save_comparison_style_results(
    output_dir: Path,
    person_env_results: Dict[str, Dict[str, Any]],
) -> None:
    """Save outputs compatible with comparison_sam3d_3dkpt_with_pesudo_gt format."""
    output_dir.mkdir(parents=True, exist_ok=True)

    subjects_map: Dict[str, Dict[str, Any]] = {}
    for key in sorted(person_env_results.keys()):
        entry = person_env_results[key]
        person_id = str(entry.get("person_id", ""))
        env_folder = str(entry.get("env_folder", ""))
        metrics = entry.get("metrics", {})
        env_name = ENV_NAMES.get(env_folder, env_folder)

        num_items = int(round(float(metrics.get("num_items", 0.0))))
        pck = _pck_dict_from_metrics(metrics)
        env_payload = {
            "person_id": person_id,
            "environment": env_folder,
            "environment_name": env_name,
            "total_frames": num_items,
            "mean_sam3d_confidence": None,
            "mean_valid_ratio": None,
            "valid_ratio_range": None,
            "metrics": {
                "mpjpe_m": _to_m(metrics.get("mpjpe")),
                "median_error_m": None,
                "root_mpjpe_m": None,
                "pa_mpjpe_m": _to_m(metrics.get("pa_mpjpe")),
                "pck": pck,
                "auc_0.15": pck.get("0.15"),
            },
            "cameras": {
                "fusion": {
                    "num_frames": num_items,
                    "num_keypoints": None,
                    "mean_sam3d_confidence": None,
                    "valid_ratio": None,
                    "metrics": {
                        "num_valid_points": int(round(float(metrics.get("valid_joints", 0.0)))),
                        "mpjpe_m": _to_m(metrics.get("mpjpe")),
                        "median_error_m": None,
                        "root_mpjpe_m": None,
                        "pa_mpjpe_m": _to_m(metrics.get("pa_mpjpe")),
                        "pck": pck,
                        "auc_0.15": pck.get("0.15"),
                    },
                    "frame_ids": [],
                }
            },
        }

        env_dir = output_dir / person_id / env_name
        env_dir.mkdir(parents=True, exist_ok=True)
        with open(env_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(env_payload, f, indent=2, ensure_ascii=False)

        subject = subjects_map.setdefault(person_id, {"person_id": person_id, "environments": {}})
        subject["environments"][env_folder] = env_payload

    all_subjects = [subjects_map[k] for k in sorted(subjects_map.keys())]
    summary = _build_comparison_summary(all_subjects)
    with open(output_dir / "comparison_data.json", "w", encoding="utf-8") as f:
        json.dump({"subjects": all_subjects, "summary": summary}, f, indent=2, ensure_ascii=False)

    logger.info("Saved comparison-style outputs: %s", output_dir)


def _build_dataloader(data_module: DriverKPTDataModule, split: str):
    data_module.setup("fit")
    if split == "train":
        return data_module.train_dataloader()
    if split == "val":
        return data_module.val_dataloader()
    if split == "test":
        return data_module.test_dataloader()
    raise ValueError(f"Unsupported eval.split={split!r}. Use train/val/test.")


def _load_triangulated_sequence(
    gt_root: Path, person_id: str, env_folder: str
) -> TriangulatedSequence:
    seq_path = gt_root / str(person_id) / str(env_folder) / "keypoints_3d.npz"
    if not seq_path.exists():
        raise FileNotFoundError(f"Triangulated GT not found: {seq_path}")
    with np.load(seq_path, allow_pickle=False) as obj:
        keypoints_3d = np.asarray(obj["keypoints_3d"], dtype=np.float32)
        valid_mask = np.asarray(obj["valid_mask"], dtype=bool)
        frame_ids = (
            np.asarray(obj["frame_ids"]).astype(str)
            if "frame_ids" in obj.files
            else None
        )
    if keypoints_3d.ndim != 3 or keypoints_3d.shape[-1] != 3:
        raise ValueError(
            f"Invalid keypoints_3d shape in {seq_path}: {keypoints_3d.shape}"
        )
    if valid_mask.shape[:2] != keypoints_3d.shape[:2]:
        raise ValueError(
            f"valid_mask shape mismatch in {seq_path}: {valid_mask.shape} vs {keypoints_3d.shape}"
        )
    return TriangulatedSequence(
        keypoints_3d=keypoints_3d,
        valid_mask=valid_mask,
        frame_ids=frame_ids,
    )


def _evaluate_fold(
    config: DictConfig,
    fold: int,
    fold_dataset: Dict[str, Any],
    module: TriFusionPoseTrainer,
    device: torch.device,
    split: str,
    gt_root: Path,
    pck_thresholds: list[float],
    output_dir: Path,
) -> tuple[Dict[str, float], Dict[str, Dict[str, Any]]]:
    data_module = DriverKPTDataModule(config, fold_dataset)
    dataloader = _build_dataloader(data_module, split)
    sequence_cache: Dict[tuple[str, str], TriangulatedSequence] = {}
    item_metrics: list[Dict[str, float]] = []
    item_records: List[Dict[str, Any]] = []
    skipped_samples = 0
    split_key = "train" if split == "train" else "val"
    sample_lookup = _build_sample_lookup(fold_dataset[split_key])
    source_view = _resolve_source_view(config)
    logger.info("Using source view for frame-id alignment: %s", source_view)
    progress = tqdm(
        dataloader,
        desc=f"fold {fold} inference",
        total=len(dataloader),
        leave=True,
    )

    with torch.no_grad():
        for batch_idx, batch in enumerate(progress):
            sam3d_kpt_3d = {k: v.to(device) for k, v in batch["sam3d_kpt_3d"].items()}
            sam3d_kpt_2d = {k: v.to(device) for k, v in batch["sam3d_kpt_2d"].items()}
            out = module.model(pose3d=sam3d_kpt_3d, pose2d=sam3d_kpt_2d)
            pred = out["P_final"]

            meta_list = batch.get("meta", [])
            if len(meta_list) != pred.shape[0]:
                raise RuntimeError(
                    f"Batch meta size mismatch at batch {batch_idx}: meta={len(meta_list)} pred={pred.shape[0]}"
                )

            for sample_idx, meta in enumerate(meta_list):
                person_id = str(meta["person_id"])
                env_folder = str(meta["env_folder"])
                start_frame = int(meta.get("start_frame", 0))
                end_frame = meta.get("end_frame", None)
                end_frame = int(end_frame) if end_frame is not None else None

                cache_key = (person_id, env_folder)
                if cache_key not in sequence_cache:
                    sequence_cache[cache_key] = _load_triangulated_sequence(
                        gt_root, person_id, env_folder
                    )

                tri_seq = sequence_cache[cache_key]
                target_t = int(pred.shape[1])
                try:
                    sample = sample_lookup.get(cache_key)
                    if sample is None:
                        raise ValueError(f"No sample path mapping for {person_id}/{env_folder}")

                    source_frame_ids = _selected_source_frame_ids(
                        sample=sample,
                        start_frame=start_frame,
                        end_frame=end_frame,
                        target_t=target_t,
                        source_view=source_view,
                    )

                    gt_kpt, gt_valid = _select_gt_by_frame_ids(
                        tri_seq=tri_seq,
                        source_frame_ids=source_frame_ids,
                    )
                except ValueError as exc:
                    skipped_samples += 1
                    logger.warning(
                        "Skip sample due to invalid GT frame range person=%s env=%s start=%s end=%s: %s",
                        person_id,
                        env_folder,
                        start_frame,
                        end_frame,
                        exc,
                    )
                    continue

                gt_kpt, gt_valid = _apply_joint_selection(
                    gt_kpt, gt_valid, pred.shape[2]
                )

                gt_tensor = torch.from_numpy(gt_kpt).to(device)
                gt_tensor = module.model._canonicalize_pose(
                    gt_tensor.unsqueeze(0)
                ).squeeze(0)
                valid_tensor = torch.from_numpy(gt_valid.astype(np.bool_)).to(device)

                sample_metrics = _compute_sample_metrics(
                    pred[sample_idx],
                    gt_tensor,
                    valid_tensor,
                    pck_thresholds,
                )
                if sample_metrics:
                    item_metrics.append(sample_metrics)
                    item_records.append(
                        {
                            "person_id": person_id,
                            "env_folder": env_folder,
                            "metrics": sample_metrics,
                        }
                    )
            progress.set_postfix(items=len(item_metrics), skipped=skipped_samples, batch=batch_idx + 1)

    progress.close()

    fold_metrics = _merge_metric_lists(item_metrics)
    fold_metrics["skipped_samples"] = float(skipped_samples)

    person_env_results = _aggregate_person_env_metrics(item_records)
    _save_person_env_results(output_dir=output_dir, fold=fold, person_env_results=person_env_results)

    logger.info("Fold %d metrics: %s", fold, fold_metrics)
    return fold_metrics, person_env_results


@hydra.main(version_base=None, config_path="../configs", config_name="train.yaml")
def main(config: DictConfig) -> None:
    seed_everything(42, workers=True)
    torch.set_float32_matmul_precision("high")

    split = str(_cfg_get(config, "eval.split", "val")).lower()
    output_dir = Path(
        str(_cfg_get(config, "eval.output_dir", Path(config.log_path) / "eval"))
    )
    gt_root = (
        Path(
            str(
                _cfg_get(
                    config,
                    "eval.triangulated_gt_root",
                    "/home/data/xchen/drive/sam3d_body_triangulated_gt",
                )
            )
        )
        .expanduser()
        .resolve()
    )
    if not gt_root.exists():
        raise FileNotFoundError(f"Triangulated GT root does not exist: {gt_root}")

    pck_thresholds_raw = _cfg_get(config, "eval.pck_thresholds", [0.02, 0.05, 0.1])
    pck_thresholds = [float(x) for x in pck_thresholds_raw]

    config.eval.ckpt_path = "/home/workspace/kaixu/code/MultiView_DriverAction_PyTorch/logs/train/trifusionpose_['front', 'left', 'right']_16f/2026-06-07/18-33-58/checkpoints/fold_0/21-0.95.ckpt"
    ckpt = _resolve_ckpt(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Running triangulated evaluation on device: %s", device)
    logger.info("Using checkpoint: %s", ckpt)
    logger.info("Using triangulated GT root: %s", gt_root)

    module = _load_module(config, ckpt, device)
    fold_dataset_idx = load_fold_dataset_idx_from_json(config)

    per_fold: Dict[str, Dict[str, float]] = {}
    merged_person_env_results: Dict[str, Dict[str, Any]] = {}
    for fold in _selected_folds(config, fold_dataset_idx.keys()):
        if fold not in fold_dataset_idx:
            raise KeyError(f"Fold {fold} is not in dataset index JSON.")
        fold_metrics, fold_person_env_results = _evaluate_fold(
            config=config,
            fold=fold,
            fold_dataset=fold_dataset_idx[fold],
            module=module,
            device=device,
            split=split,
            gt_root=gt_root,
            pck_thresholds=pck_thresholds,
            output_dir=output_dir,
        )
        per_fold[str(fold)] = fold_metrics
        for person_env_key, entry in fold_person_env_results.items():
            if person_env_key in merged_person_env_results:
                existing_metrics = merged_person_env_results[person_env_key].get("metrics", {})
                new_metrics = entry.get("metrics", {})
                merged_person_env_results[person_env_key]["metrics"] = _merge_metric_lists(
                    [existing_metrics, new_metrics]
                )
            else:
                merged_person_env_results[person_env_key] = entry

    aggregate = _aggregate_fold_metrics(per_fold)
    _save_results(output_dir, per_fold, aggregate)
    _save_comparison_style_results(output_dir, merged_person_env_results)


if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"
    main()
