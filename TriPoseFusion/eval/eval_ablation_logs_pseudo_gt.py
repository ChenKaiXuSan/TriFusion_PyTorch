#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


@dataclass(frozen=True)
class EvalJob:
    run_dir: Path
    ckpt_path: Path
    output_dir: Path
    command: list[str]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def discover_runs(logs_root: Path, pattern: str, fold: int) -> list[Path]:
    """Return Hydra run directories that contain checkpoints for the requested fold."""
    runs: list[Path] = []
    for config_path in sorted(logs_root.glob(f"{pattern}/**/.hydra/config.yaml")):
        run_dir = config_path.parent.parent
        ckpt_dir = run_dir / "checkpoints" / f"fold_{fold}"
        if ckpt_dir.exists() and any(ckpt_dir.glob("*.ckpt")):
            runs.append(run_dir)
    return runs


def _checkpoint_metric(path: Path) -> float | None:
    match = re.search(r"-([0-9]+(?:\.[0-9]+)?)\.ckpt$", path.name)
    if match is None:
        return None
    return float(match.group(1))


def _checkpoint_dir(run_dir: Path, fold: int) -> Path:
    return run_dir / "checkpoints" / f"fold_{fold}"


def select_checkpoint(run_dir: Path, fold: int, policy: str) -> Path:
    ckpt_dir = _checkpoint_dir(run_dir, fold)
    if policy == "last":
        last = ckpt_dir / "last.ckpt"
        if not last.exists():
            raise FileNotFoundError(f"Missing checkpoint: {last}")
        return last

    if policy != "best":
        raise ValueError(f"select_checkpoint supports 'best' or 'last', got {policy!r}")

    scored = [
        (metric, path)
        for path in ckpt_dir.glob("*.ckpt")
        if path.name != "last.ckpt"
        for metric in [_checkpoint_metric(path)]
        if metric is not None
    ]
    if scored:
        return min(scored, key=lambda item: (item[0], item[1].name))[1]

    last = ckpt_dir / "last.ckpt"
    if last.exists():
        return last
    raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")


def select_checkpoints(run_dir: Path, fold: int, policy: str) -> list[Path]:
    if policy in {"best", "last"}:
        return [select_checkpoint(run_dir, fold=fold, policy=policy)]
    if policy != "all":
        raise ValueError(f"Unsupported ckpt policy: {policy!r}")

    ckpts = sorted(_checkpoint_dir(run_dir, fold).glob("*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {_checkpoint_dir(run_dir, fold)}")
    return ckpts


def _safe_name(run_dir: Path, logs_root: Path, ckpt_path: Path, policy: str) -> str:
    try:
        relative = run_dir.relative_to(logs_root)
    except ValueError:
        relative = run_dir
    base = "__".join(relative.parts)
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", base).strip("_")
    if policy == "all":
        safe_ckpt = re.sub(r"[^A-Za-z0-9_.-]+", "_", ckpt_path.stem).strip("_")
        return f"{safe}__{safe_ckpt}"
    return safe


def build_eval_command(
    python_executable: str,
    eval_script: Path,
    run_dir: Path,
    ckpt_path: Path,
    output_dir: Path,
    gt_root: Path,
    split: str,
    fold: str,
    extra_overrides: Sequence[str] = (),
) -> list[str]:
    return [
        python_executable,
        str(eval_script),
        "--config-path",
        str(run_dir / ".hydra"),
        "--config-name",
        "config",
        f"eval.ckpt_path={ckpt_path}",
        f"eval.output_dir={output_dir}",
        f"eval.triangulated_gt_root={gt_root}",
        f"eval.split={split}",
        f"eval.fold={fold}",
        f"hydra.run.dir={output_dir / 'hydra_run'}",
        *extra_overrides,
    ]


def build_jobs(
    logs_root: Path,
    output_root: Path,
    gt_root: Path,
    pattern: str,
    ckpt_policy: str,
    split: str,
    fold: int,
    python_executable: str,
    eval_script: Path,
    extra_overrides: Sequence[str],
) -> list[EvalJob]:
    jobs: list[EvalJob] = []
    for run_dir in discover_runs(logs_root=logs_root, pattern=pattern, fold=fold):
        for ckpt_path in select_checkpoints(run_dir, fold=fold, policy=ckpt_policy):
            output_dir = output_root / _safe_name(run_dir, logs_root, ckpt_path, ckpt_policy)
            command = build_eval_command(
                python_executable=python_executable,
                eval_script=eval_script,
                run_dir=run_dir,
                ckpt_path=ckpt_path,
                output_dir=output_dir,
                gt_root=gt_root,
                split=split,
                fold=str(fold),
                extra_overrides=extra_overrides,
            )
            jobs.append(EvalJob(run_dir=run_dir, ckpt_path=ckpt_path, output_dir=output_dir, command=command))
    return jobs


def _load_eval_metrics(output_dir: Path) -> dict[str, Any]:
    metrics_path = output_dir / "triangulated_eval_metrics.json"
    if not metrics_path.exists():
        return {}
    with open(metrics_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _flatten_summary(job: EvalJob, metrics: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "run_dir": str(job.run_dir),
        "ckpt_path": str(job.ckpt_path),
        "output_dir": str(job.output_dir),
    }
    aggregate = metrics.get("aggregate", {}) if isinstance(metrics, dict) else {}
    for metric_name, stats in aggregate.items():
        if not isinstance(stats, dict):
            continue
        row[f"{metric_name}_mean"] = stats.get("mean")
        row[f"{metric_name}_std"] = stats.get("std")
        row[f"{metric_name}_n"] = stats.get("n")
    return row


def write_summary(output_root: Path, rows: list[dict[str, Any]]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / "summary.json"
    csv_path = output_root / "summary.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _print_command(command: Iterable[str]) -> None:
    print(" ".join(str(part) for part in command))


def main() -> None:
    repo_root = _repo_root()
    parser = argparse.ArgumentParser(
        description="Evaluate each TriPoseFusion ablation run against triangulated pseudo GT."
    )
    parser.add_argument("--logs-root", type=Path, default=repo_root / "logs" / "train")
    parser.add_argument("--output-root", type=Path, default=repo_root / "logs" / "eval_ablation_pseudo_gt")
    parser.add_argument(
        "--gt-root",
        type=Path,
        default=Path("/home/data/xchen/drive/sam3d_body_triangulated_gt"),
    )
    parser.add_argument("--pattern", type=str, default="trifusion_*")
    parser.add_argument("--ckpt-policy", choices=("best", "last", "all"), default="best")
    parser.add_argument("--split", choices=("train", "val", "all"), default="val")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--python", dest="python_executable", default=sys.executable)
    parser.add_argument(
        "--eval-script",
        type=Path,
        default=repo_root / "TriPoseFusion" / "eval" / "eval_trifusion_pesudo_gt.py",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Additional Hydra overrides forwarded to eval_trifusion_pesudo_gt.py.",
    )
    args = parser.parse_args()

    jobs = build_jobs(
        logs_root=args.logs_root,
        output_root=args.output_root,
        gt_root=args.gt_root,
        pattern=args.pattern,
        ckpt_policy=args.ckpt_policy,
        split=args.split,
        fold=args.fold,
        python_executable=args.python_executable,
        eval_script=args.eval_script,
        extra_overrides=args.overrides,
    )
    if not jobs:
        raise RuntimeError(f"No eval jobs found under {args.logs_root} with pattern {args.pattern!r}")

    rows: list[dict[str, Any]] = []
    for job in jobs:
        print(f"Run: {job.run_dir}")
        print(f"Ckpt: {job.ckpt_path}")
        print(f"Out : {job.output_dir}")
        _print_command(job.command)
        if args.dry_run:
            rows.append(_flatten_summary(job, {}))
            continue

        subprocess.run(job.command, check=True)
        rows.append(_flatten_summary(job, _load_eval_metrics(job.output_dir)))

    write_summary(args.output_root, rows)
    print(f"Saved summary: {args.output_root / 'summary.csv'}")
    print(f"Saved summary: {args.output_root / 'summary.json'}")


if __name__ == "__main__":
    main()
