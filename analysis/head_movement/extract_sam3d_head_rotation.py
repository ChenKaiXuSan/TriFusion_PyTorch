#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extract per-frame, per-view head rotation matrices from raw SAM3D outputs.

Why: head yaw/pitch computed from the 5 face keypoints is unreliable in this cabin
(masks, goggles, steering-arm occlusion of the front camera).  SAM3D also outputs
``pred_global_rots`` (127 MHR joints x 3x3, camera frame); joint 114 sits at the
ear midpoint and its rotation cleanly separates real head turns (~45 deg) from the
keypoint artefacts (<5 deg).  Nothing cached this field, so we read the raw npz.

Output: <out>/<person>_<env>/<view>.npz with head_rot (T,3,3) f32, root_rot (T,3) f32
(axis-angle global orientation), joint_coords (T,127,3) f32 (SAM3D/MHR 127-joint skeleton in
the same camera frame as pred_keypoints_3d: head joints 111-125, 114 = skull centre,
121 = nose, 123/125 = eyes, 39/75 = shoulders, 112 = neck), frame_ids; resumable.
Reading the pickled npz loads the whole record (image + mesh) - I/O bound; use workers.
"""
from __future__ import annotations

import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HEAD_JOINT = 114
VIEWS = ("front", "left", "right")
ROOT = Path("/work/1/SKIING/chenkaixu/data/drive")


def extract_one(task) -> str:
    seq, view, sam3d_root, out_dir, frame_ids = task
    person, env = seq[:2], seq[3:]
    out_path = out_dir / seq / f"{view}.npz"
    if out_path.exists():
        return f"skip {seq}/{view}"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    T = len(frame_ids)
    head = np.full((T, 3, 3), np.nan, dtype=np.float32)
    root = np.full((T, 3), np.nan, dtype=np.float32)
    joints = np.full((T, 127, 3), np.nan, dtype=np.float32)   # MHR 127-joint skeleton, camera frame
    ok = np.zeros(T, dtype=bool)
    vdir = sam3d_root / person / env / view
    for i, fid in enumerate(frame_ids):
        p = vdir / f"{int(fid):06d}_sam3d_body.npz"
        if not p.exists():
            continue
        try:
            o = np.load(p, allow_pickle=True)["output"].item()
            head[i] = o["pred_global_rots"][HEAD_JOINT]
            root[i] = o["global_rot"]
            joints[i] = o["pred_joint_coords"]
            ok[i] = True
        except Exception as e:  # corrupt file: leave NaN
            print(f"!! {p.name}: {e}", file=sys.stderr, flush=True)
    tmp = out_path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, head_rot=head, root_rot=root, joint_coords=joints, ok=ok,
                        frame_ids=np.asarray(frame_ids))
    tmp.rename(out_path)
    return f"done {seq}/{view} ok={int(ok.sum())}/{T}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sam3d-root", type=Path, default=ROOT / "sam3d_body_results_right")
    ap.add_argument("--fused-dir", type=Path, default=ROOT / "fused_keypoints_perseq",
                    help="defines the sequence list and the common frame ids")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "sam3d_head_rot_cache")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--sequences", nargs="*", default=None)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    names = args.sequences or sorted(p.stem for p in args.fused_dir.glob("*.npz"))
    tasks = []
    for seq in names:
        with np.load(args.fused_dir / f"{seq}.npz", allow_pickle=True) as z:
            fids = z["frame_ids"].astype(int)
        for view in VIEWS:
            tasks.append((seq, view, args.sam3d_root, args.out_dir, fids))
    print(f"{len(tasks)} (sequence, view) tasks, {args.workers} workers", flush=True)
    with Pool(args.workers) as pool:
        for msg in pool.imap_unordered(extract_one, tasks):
            print(msg, flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
