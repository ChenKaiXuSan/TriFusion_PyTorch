# TriPoseFusion 对比实验说明

本文档记录当前 TriPoseFusion 消融实验设计、Pegasus 脚本、提交方式、日志位置和结果查看方式。

## 实验目标

TriPoseFusion 的对比实验主要回答一个问题：

> 三视角 3D 姿态融合中，每个改进模块分别带来了多少贡献？

因此所有实验都固定同一套数据、同一个 fold、同样的训练轮数和 batch size，只改变模型模块开关。

当前实验固定：

| 项目 | 设置 |
|------|------|
| Fold | `train.fold=0` |
| Views | `front,left,right` |
| Backbone | `triple_fusion` |
| 默认 epoch | `50` |
| 默认 batch size | `32` |
| 默认采样帧数 | `16` |
| 默认 GPU 数 | `1` |

## 实验列表

| 实验名 | 脚本 | Dilated TCN | Multi-scale Velocity | Gate Entropy | Robust Canonicalization | Cross-view Attention | Learned Gate | 目的 |
|--------|------|-------------|----------------------|--------------|--------------------------|----------------------|--------------|------|
| `base_simple` | `pegasus/train_trid_pose_fusion_base_simple.sh` | 关 | 关 | 关 | 关 | 开 | 开 | 最基础 baseline |
| `dilated_tcn` | `pegasus/train_trid_pose_fusion_dilated_tcn.sh` | 开 | 关 | 关 | 关 | 开 | 开 | 单独验证膨胀 TCN |
| `multiscale_velocity` | `pegasus/train_trid_pose_fusion_multiscale_velocity.sh` | 关 | 开 | 关 | 关 | 开 | 开 | 单独验证多尺度速度特征 |
| `gate_entropy` | `pegasus/train_trid_pose_fusion_gate_entropy.sh` | 关 | 关 | 开，`lambda=0.01` | 关 | 开 | 开 | 单独验证视角门控熵正则 |
| `robust_canon` | `pegasus/train_trid_pose_fusion_robust_canon.sh` | 关 | 关 | 关 | 开 | 开 | 开 | 单独验证鲁棒 canonicalization |
| `full` | `pegasus/train_trid_pose_fusion_full.sh` | 开 | 开 | 开，`lambda=0.01` | 开 | 开 | 开 | 完整 TriPoseFusion |
| `no_cross_view_attention` | `pegasus/train_trid_pose_fusion_no_cross_view_attention.sh` | 开 | 开 | 开，`lambda=0.01` | 开 | 关 | 开 | 验证跨视角 attention 的贡献 |
| `uniform_gate` | `pegasus/train_trid_pose_fusion_uniform_gate.sh` | 开 | 开 | 关 | 开 | 开 | 关 | 验证 learned gate 是否优于均匀视角权重 |

## 单个实验提交

在 Pegasus 上单独提交某个实验：

```bash
qsub pegasus/train_trid_pose_fusion_base_simple.sh
qsub pegasus/train_trid_pose_fusion_dilated_tcn.sh
qsub pegasus/train_trid_pose_fusion_multiscale_velocity.sh
qsub pegasus/train_trid_pose_fusion_gate_entropy.sh
qsub pegasus/train_trid_pose_fusion_robust_canon.sh
qsub pegasus/train_trid_pose_fusion_full.sh
qsub pegasus/train_trid_pose_fusion_no_cross_view_attention.sh
qsub pegasus/train_trid_pose_fusion_uniform_gate.sh
```

## 批量提交全部实验

提交全部 6 个 fold 0 对比实验：

```bash
bash pegasus/submit_trid_pose_fusion_ablation.sh
```

快速检查流程时可以只跑 5 epoch：

```bash
MAX_EPOCHS=5 bash pegasus/submit_trid_pose_fusion_ablation.sh
```

也可以覆盖其他训练参数：

```bash
MAX_EPOCHS=20 BATCH_SIZE=16 NUM_FRAMES=16 DEVICES=1 \
  bash pegasus/submit_trid_pose_fusion_ablation.sh
```

## 脚本结构

当前 Pegasus 脚本分成两层：

| 文件 | 作用 |
|------|------|
| `pegasus/train_trid_pose_fusion_base_simple.sh` | 固定 `EXPERIMENT_NAME=base_simple`，固定 `FOLD=0` |
| `pegasus/train_trid_pose_fusion_dilated_tcn.sh` | 固定 `EXPERIMENT_NAME=dilated_tcn`，固定 `FOLD=0` |
| `pegasus/train_trid_pose_fusion_multiscale_velocity.sh` | 固定 `EXPERIMENT_NAME=multiscale_velocity`，固定 `FOLD=0` |
| `pegasus/train_trid_pose_fusion_gate_entropy.sh` | 固定 `EXPERIMENT_NAME=gate_entropy`，固定 `FOLD=0` |
| `pegasus/train_trid_pose_fusion_robust_canon.sh` | 固定 `EXPERIMENT_NAME=robust_canon`，固定 `FOLD=0` |
| `pegasus/train_trid_pose_fusion_full.sh` | 固定 `EXPERIMENT_NAME=full`，固定 `FOLD=0` |
| `pegasus/train_trid_pose_fusion_no_cross_view_attention.sh` | 固定 `EXPERIMENT_NAME=no_cross_view_attention`，固定 `FOLD=0` |
| `pegasus/train_trid_pose_fusion_uniform_gate.sh` | 固定 `EXPERIMENT_NAME=uniform_gate`，固定 `FOLD=0` |
| `pegasus/submit_trid_pose_fusion_ablation.sh` | 批量提交 8 个独立实验脚本 |

## 训练日志位置

每个实验的 PBS stdout/stderr 分开保存：

| 实验名 | stdout | stderr |
|--------|--------|--------|
| `base_simple` | `logs/pegasus/trid_base_simple.out` | `logs/pegasus/trid_base_simple.err` |
| `dilated_tcn` | `logs/pegasus/trid_dilated_tcn.out` | `logs/pegasus/trid_dilated_tcn.err` |
| `multiscale_velocity` | `logs/pegasus/trid_multiscale_velocity.out` | `logs/pegasus/trid_multiscale_velocity.err` |
| `gate_entropy` | `logs/pegasus/trid_gate_entropy.out` | `logs/pegasus/trid_gate_entropy.err` |
| `robust_canon` | `logs/pegasus/trid_robust_canon.out` | `logs/pegasus/trid_robust_canon.err` |
| `full` | `logs/pegasus/trid_full.out` | `logs/pegasus/trid_full.err` |
| `no_cross_view_attention` | `logs/pegasus/trid_no_cross_view_attention.out` | `logs/pegasus/trid_no_cross_view_attention.err` |
| `uniform_gate` | `logs/pegasus/trid_uniform_gate.out` | `logs/pegasus/trid_uniform_gate.err` |

训练产生的 TensorBoard、CSV 和 checkpoint 会进入：

```text
logs/train/trifusion_base_simple_fold0/...
logs/train/trifusion_dilated_tcn_fold0/...
logs/train/trifusion_multiscale_velocity_fold0/...
logs/train/trifusion_gate_entropy_fold0/...
logs/train/trifusion_robust_canon_fold0/...
logs/train/trifusion_full_fold0/...
```

## 日志中能看到的内容

Pegasus stdout 会打印本次实验配置，例如：

```text
Experiment: trifusion_full_fold0
Fold: 0
Dilated refiner: true
Multiscale velocity: true
Gate entropy lambda: 0.01
Robust canonicalization: true
```

Lightning 训练日志会记录：

```text
train/loss
val/loss
test/loss
train/loss_tri
val/loss_tri
test/loss_tri
train/loss_view
val/loss_view
test/loss_view
train/loss_bone
val/loss_bone
test/loss_bone
train/loss_temp
val/loss_temp
test/loss_temp
train/loss_info_nce
val/loss_info_nce
test/loss_info_nce
train/loss_gate_entropy
val/loss_gate_entropy
test/loss_gate_entropy
train/alpha_front
train/alpha_left
train/alpha_right
val/alpha_front
val/alpha_left
val/alpha_right
test/alpha_front
test/alpha_left
test/alpha_right
```

其中：

| 指标 | 含义 |
|------|------|
| `loss` | 总损失 |
| `loss_tri` | 融合结果与 teacher / median pseudo GT 的差距 |
| `loss_view` | 融合结果与各视角输入的一致性 |
| `loss_bone` | 骨骼长度约束 |
| `loss_temp` | 时间平滑约束 |
| `loss_info_nce` | 视角间 latent contrastive loss |
| `loss_gate_entropy` | 视角 gate 熵正则项 |
| `alpha_front/left/right` | 模型平均使用每个视角的权重 |

## 建议结果表

训练完成后，建议用如下表格汇总：

| Method | Dilated TCN | Multi-scale Vel | Gate Entropy | Robust Canon | Val Loss ↓ | Test Loss ↓ | MPJPE ↓ | PA-MPJPE ↓ | PCK ↑ |
|--------|-------------|-----------------|--------------|--------------|------------|-------------|---------|------------|-------|
| Base | 否 | 否 | 否 | 否 | | | | | |
| + Dilated TCN | 是 | 否 | 否 | 否 | | | | | |
| + Multi-scale Vel | 否 | 是 | 否 | 否 | | | | | |
| + Gate Entropy | 否 | 否 | 是 | 否 | | | | | |
| + Robust Canon | 否 | 否 | 否 | 是 | | | | | |
| Full | 是 | 是 | 是 | 是 | | | | | |

说明：

- `Val Loss` 和 `Test Loss` 可以直接从训练日志中读取。
- `MPJPE / PA-MPJPE / PCK` 需要训练后使用评估脚本基于 checkpoint 再计算。
- 如果只做初步判断，可以先比较 `base_simple` 和 `full`。

## 注意事项

1. 当前对比实验只跑 `fold 0`，适合快速完成一轮消融。
2. 如果要写论文级正式结果，建议后续扩展到 `fold 0-4`。
3. `model.geofusion_use_robust_canonicalization` 是当前模型实际读取的配置名。
4. `train.yaml` 里旧名字 `geofusion_use_robust_canon` 与当前模型读取名不一致，命令行和脚本里应使用 `geofusion_use_robust_canonicalization`。
5. 当前训练日志能反映 loss 和视角 alpha；如果需要最终几何指标，需要额外跑 eval。
