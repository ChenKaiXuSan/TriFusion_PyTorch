#!/usr/bin/env python3
"""把 SAM3D 逐帧 npz 里的 pred_keypoints_2d/3d 预抽成按视角的数组缓存。

训练/评估时每个 16 帧样本要读 48 个 ~358KB 的 npz（内含 mesh 顶点、图像等），
实际只用其中 ~1.4KB 的关键点；一个 epoch 读取量约 590GB，GPU 利用率 <30%。
本脚本一次性生成：
  <cache_root>/<person>/<env>/<view>/kpt_2d.npy   (N, K, 2) float32
  <cache_root>/<person>/<env>/<view>/kpt_3d.npy   (N, K, 3) float32
  <cache_root>/<person>/<env>/<view>/files.txt    N 行，与 KPTDataset 的 sorted(glob) 顺序一致
KPTDataset 传入 kpt_cache_root（或环境变量 TRIFUSION_KPT_CACHE_ROOT）后按位置切片读取，
语义与逐文件读取完全一致（保留全部 70 关节，KEEP 在加载时应用）。

用法:
  python TriPoseFusion/dataloader/build_kpt_cache.py \
      --sam3d-root /work/1/SKIING/chenkaixu/data/drive/sam3d_body_results_right \
      --cache-root /work/1/SKIING/chenkaixu/data/drive/sam3d_kpt_cache --num-procs 16
"""
from __future__ import annotations

import argparse
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np


def build_view(args_tuple) -> str:
    view_dir, cache_root, sam3d_root, overwrite = args_tuple
    view_dir, cache_root, sam3d_root = Path(view_dir), Path(cache_root), Path(sam3d_root)
    rel = view_dir.relative_to(sam3d_root)
    out_dir = cache_root / rel
    if not overwrite and (out_dir / "kpt_3d.npy").exists():
        return f"skip {rel}"
    try:
        files = sorted(view_dir.glob("*_sam3d_body.npz"))  # 与 KPTDataset._sorted_npz_files 相同
        if not files:
            return f"empty {rel}"
        k2, k3 = [], []
        for p in files:
            with np.load(p, allow_pickle=True) as obj:
                out = obj["output"].item()
            k2.append(np.asarray(out["pred_keypoints_2d"], dtype=np.float32))
            k3.append(np.asarray(out["pred_keypoints_3d"], dtype=np.float32))
        kmin2 = min(a.shape[0] for a in k2)
        kmin3 = min(a.shape[0] for a in k3)
        kpt_2d = np.stack([a[:kmin2] for a in k2], axis=0)
        kpt_3d = np.stack([a[:kmin3] for a in k3], axis=0)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "kpt_2d.npy", kpt_2d)
        np.save(out_dir / "kpt_3d.npy", kpt_3d)
        (out_dir / "files.txt").write_text("\n".join(p.name for p in files) + "\n", encoding="utf-8")
        return f"built {rel}: N={len(files)} K2={kmin2} K3={kmin3}"
    except Exception as exc:  # noqa: BLE001
        return f"FAILED {rel}: {exc}\n{traceback.format_exc()}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sam3d-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--views", nargs="+", default=["front", "left", "right"])
    parser.add_argument("--subject", type=str, default=None)
    parser.add_argument("--num-procs", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    jobs = []
    for subj in sorted(args.sam3d_root.iterdir()):
        if not subj.is_dir() or (args.subject and subj.name != args.subject):
            continue
        for env in sorted(subj.iterdir()):
            if not env.is_dir():
                continue
            for v in args.views:
                vd = env / v
                if vd.is_dir():
                    jobs.append((str(vd), str(args.cache_root), str(args.sam3d_root), args.overwrite))
    print(f"{len(jobs)} view dirs -> {args.cache_root}", flush=True)
    if args.num_procs <= 1:
        for j in jobs:
            print(build_view(j), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.num_procs) as pool:
            for msg in pool.map(build_view, jobs):
                print(msg, flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
