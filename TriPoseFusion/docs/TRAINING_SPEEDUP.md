# TriPoseFusion 训练提速（2026-08-28）

## 现象

2026-08-28 修正版重训（Pegasus 955575–79，A100 80GB，batch 32，`num_workers=32`）：

- 每 epoch 34,220 个 16 帧片段，**13–16 分钟/epoch**，50 epoch ≈ 11–14 小时。
- `qattach` 实测 GPU 利用率：多数采样 0%，偶发 ~95% 脉冲，**占空比约 20–30%**；
  显存 25 GB / 80 GB。节点 load average 仅 6–7（32 个 loader worker 在等 I/O，不是在算）。
- 所有 full 系 run 的 `val/loss` 在 **epoch 0–1 即最低**，之后单调上升
  （论文旧 ckpt 也是 epoch 0 的 `0-0.80`），后面 48 个 epoch 是空跑。

## 根因

`KPTDataset._load_one_view_kpts` 对每个片段逐帧 `np.load(allow_pickle=True)` SAM3D 的 npz：

- 每个 npz ≈ **358 KB**（含 `pred_vertices` 18439×3、224×332 `frame` 图像、全套 MHR 参数），
  训练只用 `pred_keypoints_2d/3d`（≈ 1.4 KB，0.4%）。
- 一个样本 = 16 帧 × 3 视角 = 48 个文件 ≈ 17 MB 读取 + 48 次 unpickle；
  一个 epoch ≈ **590 GB** 读取，再加每 epoch 全量 val（9,701 片段）≈ 300 GB。
- dataset 内置 LRU 只缓存 128 个片段，跨 epoch 完全不复用。

## 方案（均已在 `rebuttal-work` 分支实现，commit d55b4b8）

### 1. 关键点数组缓存（核心）

`TriPoseFusion/dataloader/build_kpt_cache.py`：一次性把每个 `(受试者, 环境, 视角)` 目录的
npz 抽成

```
<cache_root>/<person>/<env>/<view>/kpt_2d.npy   (N, 70, 2) float32
<cache_root>/<person>/<env>/<view>/kpt_3d.npy   (N, 70, 3) float32
<cache_root>/<person>/<env>/<view>/files.txt    与 KPTDataset 的 sorted(glob) 顺序一致
```

保留全部 70 关节，KEEP 映射仍在加载时应用，语义与逐文件读取完全一致。

`KPTDataset(kpt_cache_root=...)`（或环境变量 `TRIFUSION_KPT_CACHE_ROOT`，配置项
`paths.kpt_cache_root`）：有缓存则 mmap 切片；缺失自动回退逐文件读取；启动时也不再
glob 每个目录的 ~1 万个文件。

验证：
- `tests/test_kpt_cache_equivalence.py`：合成数据两条路径输出逐位相同（含 meta 帧区间）。
- 真实数据（受试者 01，2,879 片段随机 40 个）：逐位相同；单进程 **362 ms → 2.5 ms /片段（144×）**。

缓存位置：`/work/1/SKIING/chenkaixu/data/drive/sam3d_kpt_cache`（约 0.5 GB）。构建：

```bash
python TriPoseFusion/dataloader/build_kpt_cache.py \
  --sam3d-root /work/1/SKIING/chenkaixu/data/drive/sam3d_body_results_right \
  --cache-root /work/1/SKIING/chenkaixu/data/drive/sam3d_kpt_cache --num-procs 16
```

### 2. Early stopping / 验证频率

`train.py` 新增 `EarlyStopping(monitor="val/loss")`，由 `train.early_stop_patience` 控制
（默认 0 = 关闭，保持旧行为）；`train.check_val_every_n_epoch` 可配。

### 3. Pegasus 快速模板

`pegasus/corrected/train_template.sh.in` 默认 `MAX_EPOCHS=12`、`EARLY_STOP_PATIENCE=3`、
`KPT_CACHE_ROOT=/work/SKIING/chenkaixu/data/drive/sam3d_kpt_cache`、`NUM_WORKERS=16`，
`PROJECT_DIR` 可覆盖（快速版用 `/work/SKIING/chenkaixu/code/TriFusion_fast`）。

```bash
PROJECT_DIR=/work/SKIING/chenkaixu/code/TriFusion_fast ./pegasus/corrected/generate_and_submit.sh --submit
```

预期：I/O 不再是瓶颈后 GPU 可跑满，单 run 从 11–14 小时降到 **30 分钟量级**
（I/O 解决后可再把 batch 提到 128 进一步减少步数）。

## 相关观察

- 修正后重训的 gate 权重（epoch 4）：full / λ_gate=0 / uniform / base 全部精确 0.333；
  仅 no_cross_view_attention 略分化（0.326/0.335/0.339）——gate 塌缩是架构性的。
- `qattach -c "nvidia-smi ..." <request-id>` 可随时进正在跑的作业查看 GPU 状态。
