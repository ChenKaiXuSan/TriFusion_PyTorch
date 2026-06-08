#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_m_from_maybe_mm(value: Any, assume_mm: bool) -> float | None:
    number = _safe_float(value)
    if number is None:
        return None
    return number / 1000.0 if assume_mm else number


def _extract_pck(metrics: Dict[str, Any], target_m: float) -> float | None:
    pck = metrics.get("pck", {})
    if not isinstance(pck, dict):
        return None

    keys = [f"{target_m:.2f}", f"{int(round(target_m * 1000.0))}"]
    for key in keys:
        if key in pck:
            return _safe_float(pck.get(key))
    return None


def _extract_metric_m(metrics: Dict[str, Any], base: str) -> float | None:
    if f"{base}_m" in metrics:
        return _to_m_from_maybe_mm(metrics.get(f"{base}_m"), assume_mm=False)
    if f"{base}_mm" in metrics:
        return _to_m_from_maybe_mm(metrics.get(f"{base}_mm"), assume_mm=True)
    return None


def _iter_metrics_files(root: Path) -> Iterable[Path]:
    return sorted(root.glob("*/*/metrics.json"))


def _pair_key_from_metrics_path(path: Path) -> Tuple[str, str]:
    # .../<person>/<env>/metrics.json
    return path.parent.parent.name, path.parent.name


def _load_trifuse(root: Path) -> Dict[Tuple[str, str], Dict[str, Any]]:
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for fp in _iter_metrics_files(root):
        obj = json.loads(fp.read_text(encoding="utf-8"))
        metrics = obj.get("metrics", {})
        person, env = _pair_key_from_metrics_path(fp)
        out[(person, env)] = {
            "mpjpe_m": _extract_metric_m(metrics, "mpjpe"),
            "pa_mpjpe_m": _extract_metric_m(metrics, "pa_mpjpe"),
            "pck_0.10": _extract_pck(metrics, 0.10),
            "view_weights": obj.get("view_weights", {}),
            "source_file": str(fp),
        }
    return out


def _extract_single_cam(camera_metrics: Dict[str, Any]) -> Dict[str, float | None]:
    return {
        "mpjpe_m": _extract_metric_m(camera_metrics, "mpjpe"),
        "pa_mpjpe_m": _extract_metric_m(camera_metrics, "pa_mpjpe"),
        "pck_0.10": _extract_pck(camera_metrics, 0.10),
    }


def _load_single(root: Path) -> Dict[Tuple[str, str], Dict[str, Any]]:
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for fp in _iter_metrics_files(root):
        obj = json.loads(fp.read_text(encoding="utf-8"))
        person, env = _pair_key_from_metrics_path(fp)
        cams = obj.get("cameras", {})

        cam_data: Dict[str, Dict[str, float | None]] = {}
        for cam in ("front", "left", "right"):
            cam_metrics = cams.get(cam, {}).get("metrics", {}) if isinstance(cams, dict) else {}
            cam_data[cam] = _extract_single_cam(cam_metrics)

        out[(person, env)] = {
            "front": cam_data["front"],
            "left": cam_data["left"],
            "right": cam_data["right"],
            "source_file": str(fp),
        }
    return out


def _best_single(cam_values: Dict[str, Dict[str, float | None]], metric: str, larger_better: bool) -> float | None:
    vals = [cam_values[cam].get(metric) for cam in ("front", "left", "right")]
    vals = [v for v in vals if isinstance(v, float)]
    if not vals:
        return None
    return max(vals) if larger_better else min(vals)


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _mean_map_values(maps: list[Dict[str, Any]]) -> Dict[str, float]:
    buckets: Dict[str, list[float]] = defaultdict(list)
    for mapping in maps:
        if not isinstance(mapping, dict):
            continue
        for key, value in mapping.items():
            value_float = _safe_float(value)
            if value_float is not None:
                buckets[str(key)].append(value_float)

    return {key: float(sum(values) / len(values)) for key, values in buckets.items() if values}


def compare_all(trifuse_dir: Path, single_dir: Path, output_dir: Path) -> None:
    trifuse = _load_trifuse(trifuse_dir)
    single = _load_single(single_dir)
    common_keys = sorted(set(trifuse.keys()) & set(single.keys()))

    if not common_keys:
        raise RuntimeError(
            f"No common person/env found. trifuse={trifuse_dir}, single={single_dir}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    detailed_csv = output_dir / "single_vs_trifuse_detailed.csv"
    summary_json = output_dir / "single_vs_trifuse_summary.json"

    rows = []
    delta_mpjpe = []
    delta_pa = []
    delta_pck = []
    trifuse_better_mpjpe = 0
    trifuse_better_pa = 0
    trifuse_better_pck = 0
    trifuse_weight_maps: list[Dict[str, Any]] = []
    weight_keys: set[str] = set()

    for person, env in common_keys:
        tri = trifuse[(person, env)]
        sng = single[(person, env)]

        best_single_mpjpe = _best_single(sng, "mpjpe_m", larger_better=False)
        best_single_pa = _best_single(sng, "pa_mpjpe_m", larger_better=False)
        best_single_pck = _best_single(sng, "pck_0.10", larger_better=True)

        tri_mpjpe = tri.get("mpjpe_m")
        tri_pa = tri.get("pa_mpjpe_m")
        tri_pck = tri.get("pck_0.10")
        tri_weights = tri.get("view_weights", {})
        if isinstance(tri_weights, dict) and tri_weights:
            trifuse_weight_maps.append(tri_weights)
            weight_keys.update(str(key) for key in tri_weights.keys())

        d_mpjpe = (tri_mpjpe - best_single_mpjpe) if isinstance(tri_mpjpe, float) and isinstance(best_single_mpjpe, float) else None
        d_pa = (tri_pa - best_single_pa) if isinstance(tri_pa, float) and isinstance(best_single_pa, float) else None
        d_pck = (tri_pck - best_single_pck) if isinstance(tri_pck, float) and isinstance(best_single_pck, float) else None

        if isinstance(d_mpjpe, float):
            delta_mpjpe.append(d_mpjpe)
            if d_mpjpe < 0:
                trifuse_better_mpjpe += 1
        if isinstance(d_pa, float):
            delta_pa.append(d_pa)
            if d_pa < 0:
                trifuse_better_pa += 1
        if isinstance(d_pck, float):
            delta_pck.append(d_pck)
            if d_pck > 0:
                trifuse_better_pck += 1

        row = {
            "person_id": person,
            "environment": env,
            "trifuse_mpjpe_m": tri_mpjpe,
            "trifuse_pa_mpjpe_m": tri_pa,
            "trifuse_pck_0.10": tri_pck,
            "single_front_mpjpe_m": sng["front"].get("mpjpe_m"),
            "single_left_mpjpe_m": sng["left"].get("mpjpe_m"),
            "single_right_mpjpe_m": sng["right"].get("mpjpe_m"),
            "single_front_pa_mpjpe_m": sng["front"].get("pa_mpjpe_m"),
            "single_left_pa_mpjpe_m": sng["left"].get("pa_mpjpe_m"),
            "single_right_pa_mpjpe_m": sng["right"].get("pa_mpjpe_m"),
            "single_front_pck_0.10": sng["front"].get("pck_0.10"),
            "single_left_pck_0.10": sng["left"].get("pck_0.10"),
            "single_right_pck_0.10": sng["right"].get("pck_0.10"),
            "best_single_mpjpe_m": best_single_mpjpe,
            "best_single_pa_mpjpe_m": best_single_pa,
            "best_single_pck_0.10": best_single_pck,
            "delta_trifuse_minus_best_single_mpjpe_m": d_mpjpe,
            "delta_trifuse_minus_best_single_pa_mpjpe_m": d_pa,
            "delta_trifuse_minus_best_single_pck_0.10": d_pck,
            "trifuse_source_file": tri.get("source_file"),
            "single_source_file": sng.get("source_file"),
        }
        for key in sorted(weight_keys):
            row[f"trifuse_view_weight_{key}"] = tri_weights.get(key)
        rows.append(row)

    fieldnames = sorted({key for row in rows for key in row.keys()}) if rows else []
    with open(detailed_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "num_common_pairs": len(common_keys),
        "trifuse_dir": str(trifuse_dir),
        "single_dir": str(single_dir),
        "metrics": {
            "mean_delta_trifuse_minus_best_single_mpjpe_m": _mean(delta_mpjpe),
            "mean_delta_trifuse_minus_best_single_pa_mpjpe_m": _mean(delta_pa),
            "mean_delta_trifuse_minus_best_single_pck_0.10": _mean(delta_pck),
            "trifuse_better_count_mpjpe": trifuse_better_mpjpe,
            "trifuse_better_count_pa_mpjpe": trifuse_better_pa,
            "trifuse_better_count_pck_0.10": trifuse_better_pck,
        },
        "weights": _mean_map_values(trifuse_weight_maps),
        "output_files": {
            "detailed_csv": str(detailed_csv),
            "summary_json": str(summary_json),
        },
    }

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Saved detailed comparison: {detailed_csv}")
    print(f"Saved summary: {summary_json}")
    print(f"Common pairs: {len(common_keys)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare all single-view-vs-GT results against trifuse-vs-GT results."
    )
    parser.add_argument(
        "--trifuse-dir",
        type=str,
        default="/home/workspace/kaixu/code/MultiView_DriverAction_PyTorch/TriPoseFusion/eval/logs/eval_trifusion_pesudo_gt",
        help="Directory containing trifuse metrics in */*/metrics.json layout.",
    )
    parser.add_argument(
        "--single-dir",
        type=str,
        default="/home/workspace/kaixu/code/MultiView_DriverAction_PyTorch/TriPoseFusion/eval/logs/comparison_sam3d_3dkpt_with_pesudo_gt",
        help="Directory containing single-view metrics in */*/metrics.json layout.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/home/workspace/kaixu/code/MultiView_DriverAction_PyTorch/TriPoseFusion/eval/logs/comparison_single_vs_trifuse",
        help="Directory to save detailed.csv and summary.json.",
    )
    args = parser.parse_args()

    trifuse_dir = Path(args.trifuse_dir)
    single_dir = Path(args.single_dir)
    output_dir = Path(args.output_dir)

    if not trifuse_dir.exists():
        raise FileNotFoundError(f"trifuse-dir does not exist: {trifuse_dir}")
    if not single_dir.exists():
        raise FileNotFoundError(f"single-dir does not exist: {single_dir}")

    compare_all(trifuse_dir=trifuse_dir, single_dir=single_dir, output_dir=output_dir)


if __name__ == "__main__":
    main()