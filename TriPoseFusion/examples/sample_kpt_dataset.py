#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Minimal sample for TriPoseFusion KPTDataset.

Run example:
python3 TriPoseFusion/examples/sample_kpt_dataset.py \
  --sam3d-root /workspace/data/sam3d_body_results_right_full \
  --person 01 \
  --env 夜多い \
  --views front,left,right \
  --max-video-frames 32
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

from torch.utils.data import DataLoader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample loader for SAM3D 2D/3D keypoints")
    parser.add_argument(
        "--sam3d-root",
        type=Path,
        default=Path("/workspace/data/sam3d_body_results_right_full"),
        help="Root directory of sam3d results",
    )
    parser.add_argument("--person", type=str, default="01", help="Person id")
    parser.add_argument("--env", type=str, default="夜多い", help="Environment folder name")
    parser.add_argument(
        "--views",
        type=str,
        default="front,left,right",
        help="Comma-separated views, e.g. front,left,right",
    )
    parser.add_argument(
        "--max-video-frames",
        type=int,
        default=32,
        help="Chunk size. Set <=0 to disable chunking",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    tri_pose_root = Path(__file__).resolve().parents[1]
    if str(tri_pose_root) not in sys.path:
        sys.path.insert(0, str(tri_pose_root))

    from dataloader.kpt_dataset import KPTDataset

    views = [v.strip() for v in args.views.split(",") if v.strip()]
    if not views:
        raise ValueError("views is empty")

    sam3d_kpts = {v: args.sam3d_root / args.person / args.env / v for v in views}
    for view, p in sam3d_kpts.items():
        if not p.exists():
            raise FileNotFoundError(f"Missing sam3d dir for view={view}: {p}")

    sample = SimpleNamespace(
        person_id=args.person,
        env_folder=args.env,
        env_key="sample",
        videos={},
        sam3d_kpts=sam3d_kpts,
    )

    dataset = KPTDataset(
        experiment="sample",
        index_mapping=[sample],
        view_name=views,
        target_t=16,
    )

    print(f"dataset length: {len(dataset)}")

    one = dataset[0]
    print("single sample meta:", one["meta"])
    for v in views:
        print(f"{v} kpt2d shape:", tuple(one["sam3d_kpt_2d"][v].shape))
        print(f"{v} kpt3d shape:", tuple(one["sam3d_kpt_3d"][v].shape))

    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda b: b[0])
    batch = next(iter(loader))
    print("\nfrom DataLoader(batch_size=1):")
    for v in views:
        print(f"{v} kpt2d shape:", tuple(batch["sam3d_kpt_2d"][v].shape))
        print(f"{v} kpt3d shape:", tuple(batch["sam3d_kpt_3d"][v].shape))


if __name__ == "__main__":
    main()
