#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""Build TriPoseFusion cross-validation index files with plain argparse."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from sklearn.model_selection import GroupKFold

try:
    from map_config import CAM_NAMES, ENV_KEY_TO_FOLDER, VideoSample
except ImportError:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from TriPoseFusion.map_config import CAM_NAMES, ENV_KEY_TO_FOLDER, VideoSample


def _first_existing_dir(candidates: List[Path]) -> Optional[Path]:
    for path in candidates:
        if path.exists():
            return path
    return None


def _infer_video_path(root_path: Path, video_path: Optional[Path]) -> Path:
    if video_path is not None:
        return video_path
    found = _first_existing_dir([root_path / "videos_split", root_path / "videos"])
    return found if found is not None else root_path / "videos_split"


def _infer_annotation_path(root_path: Path, annotation_path: Optional[Path]) -> Path:
    if annotation_path is not None:
        return annotation_path
    found = _first_existing_dir([root_path / "label", root_path / "labels", root_path / "annotation"])
    return found if found is not None else root_path / "label"


class DefineCrossValidation(object):
    """Build samples from label/video/SAM3D paths and split them by person_id."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.root_path: Path = Path(args.root_path)
        self.video_path: Path = _infer_video_path(self.root_path, args.video_path)
        self.annotation_path: Path = _infer_annotation_path(
            self.root_path, args.annotation_path
        )
        self.sam3d_results_path: Path = Path(args.sam3d_results_path)
        self.index_mapping: Path = Path(args.index_mapping)
        self.fold_file_template: str = str(args.fold_file_template)

        self.fold_count: int = int(args.folds)
        self.overwrite: bool = bool(args.overwrite)

    @staticmethod
    def _parse_label_filename(path: Path) -> Tuple[str, str, str]:
        """person_01_night_high_h265.json -> ("01", "night", "high")."""
        parts = path.stem.split("_")
        if len(parts) < 5 or parts[0] != "person":
            raise ValueError(f"Unexpected label filename: {path.name}")
        return parts[1], parts[2], parts[3]

    def _collect_one_sample(self, label_path: Path) -> VideoSample | None:
        person_id, daynight, highlow = self._parse_label_filename(label_path)
        if (daynight, highlow) not in ENV_KEY_TO_FOLDER:
            return None

        env_folder = ENV_KEY_TO_FOLDER[(daynight, highlow)]
        env_key = f"{daynight}_{highlow}"
        video_dir = self.video_path / person_id / env_folder
        if not video_dir.exists():
            return None

        videos: Dict[str, Path] = {}
        for cam in CAM_NAMES:
            video_file = video_dir / f"{cam}.mp4"
            if video_file.exists():
                videos[cam] = video_file

        if not videos:
            return None

        sam3d_kpts: Dict[str, Path] = {}
        for cam in CAM_NAMES:
            kpt_dir = self.sam3d_results_path / person_id / env_folder / cam
            if kpt_dir.exists():
                sam3d_kpts[cam] = kpt_dir

        return VideoSample(
            person_id=person_id,
            env_folder=env_folder,
            env_key=env_key,
            label_path=label_path,
            videos=videos,
            sam3d_kpts=sam3d_kpts if sam3d_kpts else None,
        )

    def build_samples(self) -> List[VideoSample]:
        label_files = sorted(self.annotation_path.glob("person_*_*.json"))
        samples: List[VideoSample] = []

        for label_path in label_files:
            try:
                sample = self._collect_one_sample(label_path)
            except Exception as exc:
                print(f"Skip invalid label {label_path}: {exc}")
                sample = None

            if sample is not None:
                samples.append(sample)

        return samples

    def split_by_person(
        self, samples: List[VideoSample]
    ) -> Dict[int, Dict[str, List[VideoSample]]]:
        if self.fold_count <= 1:
            return {0: {"train": samples, "val": []}}

        groups = [sample.person_id for sample in samples]
        unique_groups = sorted(set(groups))
        if self.fold_count > len(unique_groups):
            raise ValueError(
                f"folds={self.fold_count} is larger than person count "
                f"({len(unique_groups)}). Please reduce --folds."
            )

        indices = list(range(len(samples)))
        group_kfold = GroupKFold(n_splits=self.fold_count)
        fold_samples: Dict[int, Dict[str, List[VideoSample]]] = {}

        for fold, (train_idx, val_idx) in enumerate(
            group_kfold.split(indices, groups=groups)
        ):
            fold_samples[fold] = {
                "train": [samples[i] for i in train_idx],
                "val": [samples[i] for i in val_idx],
            }

        return fold_samples

    def prepare(self) -> Dict[int, Dict[str, List[VideoSample]]]:
        samples = self.build_samples()
        if not samples:
            raise RuntimeError(
                "No valid samples found. Please check:\n"
                f"  video_path={self.video_path}\n"
                f"  annotation_path={self.annotation_path}\n"
                f"  sam3d_results_path={self.sam3d_results_path}\n"
                "  label filename format: person_XX_(day|night)_(high|low)_h265.json\n"
                "  video structure: videos/XX/(夜多い|夜少ない|昼多い|昼少ない)/(front|right|left).mp4"
            )

        return self.split_by_person(samples)

    @staticmethod
    def _sample_to_json(sample: VideoSample) -> Dict[str, Any]:
        return {
            "person_id": sample.person_id,
            "env_folder": sample.env_folder,
            "env_key": sample.env_key,
            "label_path": str(sample.label_path) if sample.label_path else None,
            "videos": {key: str(value) for key, value in sample.videos.items()},
            "sam3d_kpts": {
                key: str(value) for key, value in sample.sam3d_kpts.items()
            }
            if sample.sam3d_kpts
            else None,
        }

    @staticmethod
    def _sample_from_json(item: Dict[str, Any]) -> VideoSample:
        sam3d_kpts = item.get("sam3d_kpts")
        label_path = item.get("label_path") or item.get("label")
        return VideoSample(
            person_id=str(item["person_id"]),
            env_folder=str(item["env_folder"]),
            env_key=str(item["env_key"]),
            videos={key: Path(value) for key, value in item.get("videos", {}).items()},
            label_path=Path(label_path) if label_path else None,
            sam3d_kpts={key: Path(value) for key, value in sam3d_kpts.items()}
            if sam3d_kpts
            else None,
        )

    def _fold_file(self, fold: int) -> Path:
        return self.index_mapping / self.fold_file_template.format(fold=fold)

    def _existing_fold_files(self) -> List[Path]:
        if "{fold}" in self.fold_file_template:
            prefix, suffix = self.fold_file_template.split("{fold}", 1)
            return sorted(self.index_mapping.glob(f"{prefix}*{suffix}"))
        return sorted(self.index_mapping.glob(self.fold_file_template))

    def _save_fold_files(
        self, fold_samples: Dict[int, Dict[str, List[VideoSample]]]
    ) -> None:
        for fold, splits in fold_samples.items():
            fold_file = self._fold_file(fold)
            serial = {
                "train": [self._sample_to_json(s) for s in splits["train"]],
                "val": [self._sample_to_json(s) for s in splits["val"]],
            }
            with open(fold_file, "w", encoding="utf-8") as f:
                json.dump(serial, f, ensure_ascii=False, indent=2)
            print(f"Saved fold {fold}: {fold_file}")

    def _load_fold_files(self) -> Dict[int, Dict[str, List[VideoSample]]]:
        fold_samples: Dict[int, Dict[str, List[VideoSample]]] = {}
        for fold in range(self.fold_count):
            fold_file = self._fold_file(fold)
            if not fold_file.exists():
                raise FileNotFoundError(
                    f"Fold index JSON not found: {fold_file}. "
                    "Use --overwrite to regenerate fold files."
                )

            with open(fold_file, "r", encoding="utf-8") as f:
                serial = json.load(f)

            fold_samples[fold] = {"train": [], "val": []}
            for split in ["train", "val"]:
                for item in serial.get(split, []):
                    fold_samples[fold][split].append(self._sample_from_json(item))

        print(f"Loaded {len(fold_samples)} fold index files from: {self.index_mapping}")
        return fold_samples

    def __call__(self) -> Dict[int, Dict[str, List[VideoSample]]]:
        self.index_mapping.mkdir(parents=True, exist_ok=True)
        existing_fold_files = self._existing_fold_files()

        if self.overwrite or len(existing_fold_files) < self.fold_count:
            fold_samples = self.prepare()
            self._save_fold_files(fold_samples)
            return fold_samples

        return self._load_fold_files()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate or load TriPoseFusion GroupKFold index JSON."
    )
    parser.add_argument(
        "--root-path",
        type=Path,
        default=Path("/workspace/data/multi_view_driver_action"),
    )
    parser.add_argument("--video-path", type=Path, default="/workspace/data/videos_split")
    parser.add_argument("--annotation-path", type=Path, default="/workspace/data/multi_view_driver_action/label")
    parser.add_argument(
        "--sam3d-results-path",
        type=Path,
        default=Path("/workspace/data/sam3d_body_results_right"),
    )
    parser.add_argument(
        "--index-mapping",
        type=Path,
        default=Path("/workspace/data/multi_view_driver_action/index_mapping"),
    )
    parser.add_argument("--fold-file-template", type=str, default="fold_{fold}.json")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def print_summary(args: argparse.Namespace, fold_samples: Dict[int, Dict[str, List[VideoSample]]]) -> None:
    print("TriPoseFusion cross-validation")
    print(f"  root_path={args.root_path}")
    print(f"  video_path={_infer_video_path(args.root_path, args.video_path)}")
    print(f"  annotation_path={_infer_annotation_path(args.root_path, args.annotation_path)}")
    print(f"  sam3d_results_path={args.sam3d_results_path}")
    print(f"  index_mapping={args.index_mapping}")
    print(f"  fold_file_template={args.fold_file_template}")
    print(f"  folds={args.folds}")
    for fold, splits in fold_samples.items():
        print(f"fold {fold}: train={len(splits['train'])}, val={len(splits['val'])}")


def main() -> None:
    args = build_arg_parser().parse_args()
    fold_samples = DefineCrossValidation(args)()
    print_summary(args, fold_samples)


if __name__ == "__main__":
    main()
