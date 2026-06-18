# TriPoseFusion 配置说明文档

## 📖 概述

本文档详细说明 `TriPoseFusion` 训练配置文件的所有参数及其作用。配置文件位于 `configs/train.yaml`，使用 Hydra 格式。

---

## 🚀 快速开始

### 基础训练命令

```bash
# 默认配置训练
python -m TriPoseFusion.trainer.train_triple_fusion

# 使用特定配置文件
python TriPoseFusion/train.py --config-name train.yaml
```

### 修改配置参数

```bash
# 运行时覆盖参数 (Hydra 风格)
python TriPoseFusion/train.py model.geofusion_refiner_layers=3 train.max_epochs=30

# 使用不同视图组合
python TriPoseFusion/train.py +view_name='[front]' view='single'
```

---

## 📑 配置结构总览

```yaml
hydra:              # Hydra 输出目录管理
paths:              # 数据路径
data:               # 数据加载配置
loss:               # 优化器超参数
model:              # 模型架构配置 ⭐ 核心区域
train:              # 训练设置
eval:               # 评估设置
experiment:         # 实验命名
log_path:           # 日志路径
```

---

## 🔧 核心参数详解

### 1. 数据路径 (paths)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `root_path` | `/home/data/xchen/drive/multi_view_driver_action` | 数据集根目录 |
| `index_mapping` | `${paths.root_path}/index_mapping` | 索引映射文件/目录 |
| `sam3d_results_path` | `/home/data/xchen/drive/sam3d_body_results_right` | SAM3D 关键点结果路径 |

### 2. 数据加载 (data)

| 参数 | 默认值 | 说明 | 建议值 |
|------|--------|------|--------|
| `num_workers` | 32 | DataLoader 工作进程数 | GPU 多时增加至 64+ |
| `batch_size` | 30 | 每 GPU batch size | OOM 时减小至 16-20 |
| `uniform_temporal_subsample_num` | 16 | 每视频抽帧数 | 根据显存调整 |

**性能提示**: `num_workers=32` 适合大多数系统；如果 CPU 瓶颈，可降低到 8-16。

### 3. 优化器 (loss)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `lr` | 0.0001 | AdamW 学习率 |
| `weight_decay` | 0.00001 | L2 正则化 |

### 4. 模型架构 (model) - **核心区域** ⭐

#### 视图与关节配置

```yaml
geofusion_view_names: ['front', 'left', 'right']  # 使用哪些视图
geofusion_num_joints: 52                          # 关键点数量
```

#### 关键地标索引 (用于坐标规范)

| 参数 | 默认值 | 用途 |
|------|--------|------|
| `kpt_neck_index` | 51 | 颈部作为坐标原点 |
| `kpt_left_shoulder_index` | 5 | 左肩定义 x 轴 |
| `kpt_right_shoulder_index` | 6 | 右肩定义 x 轴 |
| `kpt_mid_hip_index` | -1 | 髋部，-1 表示使用肩部中点 |

#### 网络核心参数

```yaml
geofusion_hidden_dim: 128       # 嵌入维度 (推荐：128-256)
geofusion_refiner_dim: 256      # TCN 通道数 (推荐：256-512)
```

#### ⭐ IMPROVEMENT 1: Dilated TCN Temporal Refiner

膨胀卷积层数影响时序建模能力：

| layers | Receptive Field | 适用场景 |
|--------|-----------------|----------|
| 2 | 3 frames | 快速手势 |
| **4** | **15 frames** | **慢速手势 (推荐)** |

```yaml
geofusion_refiner_layers: 4    # 建议值：2 或 4
geofusion_dropout: 0.1         # Dropout 率，防止过拟合
```

#### ⭐ IMPROVEMENT 1: Cross-View Attention

跨视图注意力机制让不同视角能够"互相交流":

```yaml
geofusion_attention_num_heads: 4    # 推荐：hidden_dim/32 = 4 或更大
```

**效果**: 当某个视图被遮挡时，模型可以依赖其他视图的信息。

#### Feature Engineering Options

| 参数 | 默认值 | 启用后增加输入维度 |
|------|--------|-------------------|
| `geofusion_use_2d` | true | +4 (uv+velocity) |
| `geofusion_use_conf` | false | +1 (confidence) |
| `geofusion_use_reproj_error_feature` | false | +1 (requires cameras) |

#### Canonicalization 配置

```yaml
geofusion_canonicalize: true      # 启用坐标规范 (推荐)
geofusion_eps: 0.000001           # 数值稳定性的 epsilon
geofusion_use_robust_canon: false # 使用鲁棒 canonicalization(实验性)
```

#### NCE Contrastive Learning

用于在潜在空间中对齐不同视图的特征：

```yaml
geofusion_nce_dim: 64             # 投影维度
info_nce_temperature: 0.1         # 温度参数 (越小越严格)
```

#### Loss Weights - 各损失项权重

```yaml
lambda_tri: 1.0           # Teacher forcing loss (主要监督信号)
lambda_reproj: 0.0        # Reprojection to 2D keypoints
lambda_view: 0.2          # View consistency with gated fusion
lambda_bone: 0.5          # Bone length preservation
lambda_temp: 0.1          # Temporal smoothness
lambda_info_nce: 0.1      # InfoNCE contrastive learning
```

**调参建议**: 主要关注 `lambda_tri` (监督强度) 和 `lambda_view` (融合约束)。

#### ⭐ IMPROVEMENT 2: Gate Regularization

防止某些视图被完全忽略的正则化：

```yaml
geofusion_gate_entropy_reg_lambda: 0.01   # 正则化强度
```

- `0.0`: 不使用 (旧版行为)
- `0.01`: 轻度正则化 (推荐)
- `0.1`: 强正则化 (所有视图使用均衡时)

#### Feature Engineering Switches (高级选项)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `geofusion_use_dilated_refiner` | true | 使用膨胀 TCN vs 标准 TCN |
| `geofusion_use_multiscale_velocity` | true | 多尺度速度特征 (+vel_3, vel_5, acc, jerk) |

#### Bone Constraints

保持关节间的骨骼长度约束：

```yaml
geofusion_bones:
    - [5, 6]      # Shoulder width
    - [51, 5]     # Neck-left shoulder
    - [51, 6]     # Neck-right shoulder
```

---

## 🏃 训练配置 (train)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `max_epochs` | 50 | 最大训练轮数 |
| `devices` | 2 | GPU 数量 |
| `grad_clip_val` | 1.0 | 梯度裁剪值 |
| `fold` | 0 | K-fold 交叉验证 fold ID |
| `view` | multi | 'single' 或 'multi' |
| `view_name` | ['front','left','right'] | 使用的视图列表 |

---

## 🔍 评估配置 (eval)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `ckpt_path` | null | 显式指定 checkpoint 文件 |
| `ckpt_dir` | null | 搜索 checkpoint 的目录 |
| `split` | val | 'train', 'val' 或 'test' |
| `fold` | all | 评估哪个 fold(s) |

---

## 📊 实验配置 (experiment, log_path)

```yaml
# 动态命名模式
experiment: trifusionpose_${train.view_name}_${data.uniform_temporal_subsample_num}f

# 日志输出路径结构
log_path: logs/train/${experiment}/${now:%Y-%m-%d}/${now:%H-%M-%S}
```

**实际输出**: `logs/train/trifusionpose_[view_name]_16f/2024-06-09/21-30-45/`

---

## 🧪 配置文件变体示例

### 版本 A: 基础配置 (标准训练)

```yaml
model:
  geofusion_refiner_layers: 4
  geofusion_attention_num_heads: 4
  geofusion_use_robust_canon: false
  geofusion_gate_entropy_reg_lambda: 0.01
```

### 版本 B: 快速手势优化 (减少时序依赖)

```yaml
model:
  geofusion_refiner_layers: 2       # 仅 3 frames RF
  geofusion_use_multiscale_velocity: false  # 简单速度特征
  geofusion_dropout: 0.2            # 更强正则化
```

### 版本 C: 最大性能配置 (推荐)

```yaml
model:
  geofusion_refiner_layers: 4       # 15 frames RF
  geofusion_attention_num_heads: 8  # 更多注意力头
  geofusion_use_multiscale_velocity: true
  geofusion_use_robust_canon: true  # 异常值鲁棒性
  geofusion_dropout: 0.1
```

---

## 🎯 常见配置场景

### 场景 1: GPU OOM (显存不足)

```yaml
data:
  batch_size: 16              # 降低 batch size
  uniform_temporal_subsample_num: 8     # 减少帧数

model:
  geofusion_hidden_dim: 64    # 减小模型宽度
```

### 场景 2: CPU 瓶颈 (数据加载慢)

```yaml
data:
  num_workers: 8              # 降低 worker 数量
```

### 场景 3: 仅使用一个视图 (快速测试)

```yaml
train:
  view: single                # Single-view mode
  view_name: '[front]'        # Only front camera
```

### 场景 4: 自定义视角组合

```yaml
train:
  view: multi
  view_name: '[front, left]'  # Only front and left views
```

---

## 🔧 参数验证与调试

### 检查配置加载

```python
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf

with initialize_config_dir(config_dir="./configs"):
    cfg = compose(config_name="train")
    print(OmegaConf.to_yaml(cfg))
```

### 打印关键参数

```bash
# 查看 model 配置
python TriPoseFusion/train.py --cfg model
```

---

## 📚 相关文档

- [CLAUDE.md](../CLAUDE.md) - 项目架构概览
- [OOM_SOLUTIONS_INDEX.md](../docs/OOM_SOLUTIONS_INDEX.md) - 显存问题解决方案
- [IMPROVEMENT_SUMMARY.md](../IMPROVEMENT_SUMMARY.md) - 改进内容汇总

---

## 💡 最佳实践建议

1. **从默认配置开始**: 先用 `model.geofusion_refiner_layers=4` 训练，观察收敛情况
2. **监控 Loss**: 注意 `loss_tri` 和 `loss_gate_entropy` 的平衡
3. **逐步调试**: 先确保 single-view 运行正常再切换 multi-view
4. **显存监控**: 使用 `watch -n 1 nvidia-smi` 实时查看显存占用

---

## ❓ FAQ

**Q: gate_entropy_reg_lambda 应该设多少？**  
A: 建议从 0.01 开始，如果训练中发现某些视图从未被使用，可增加到 0.1。

**Q: 为什么 use_reproj_error_feature=False?**  
A: 需要相机内参和位姿信息；如果没有这些标注数据，保持为 False。

**Q: canoticalization 关闭会影响性能吗？**  
A: 会！建议始终开启 canonicalization，它对位置不变性至关重要。
