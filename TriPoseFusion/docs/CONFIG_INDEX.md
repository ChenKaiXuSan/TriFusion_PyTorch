# TriPoseFusion 配置文件索引

本文档列出所有可用的配置文件及其用途。

---

## 📁 配置文件列表

| 文件名 | 描述 | 适用场景 |
|--------|------|----------|
| [train.yaml](../configs/train.yaml) | **标准完整配置** - 包含所有参数和详细注释 | 日常训练使用 (推荐) |
| [train_template.yaml](../configs/train_template.yaml) | **完整模板** - 所有可用参数的空白模板 | 自定义配置参考 |
| [train_minimal.yaml](../configs/train_minimal.yaml) | **最小化配置** - 仅必要参数，快速测试 | 快速原型验证 |

---

## 🚀 快速开始

### 使用标准配置 (推荐)

```bash
# 默认训练命令
python TriPoseFusion/train.py --config-name train

# 修改参数
python TriPoseFusion/train.py model.geofusion_refiner_layers=3 train.max_epochs=30
```

### 使用最小化配置 (快速测试)

```bash
# 快速验证训练流程
python TriPoseFusion/train.py --config-name train_minimal
```

---

## 📚 文档指南

| 文档 | 用途 | 链接 |
|------|------|------|
| **配置说明** | 详细参数解释 | [CONFIGURATION_GUIDE.md](./CONFIGURATION_GUIDE.md) ⭐ |
| **改进总结** | 模型改进说明 | [IMPROVEMENT_SUMMARY.md](../../IMPROVEMENT_SUMMARY.md) |
| **对比实验** | TriPoseFusion 消融实验与 Pegasus 脚本说明 | [ABLATION_EXPERIMENTS.md](./ABLATION_EXPERIMENTS.md) |
| **OOM 解决方案** | 显存问题处理 | [OOM_SOLUTIONS_INDEX.md](../docs/OOM_SOLUTIONS_INDEX.md) |

---

## ⚙️ 配置修改指南

### 常见调整场景

| 需求 | 推荐操作 | 配置文件建议 |
|------|----------|-------------|
| **快速调试** | 降低 batch_size，减少 epochs | train_minimal.yaml |
| **生产训练** | 使用完整优化设置 | train.yaml |
| **自定义模型** | 参考模板修改参数 | train_template.yaml |
| **仅改少数参数** | 运行时覆盖 | 任意 + `+param=value` |

### 运行时参数覆盖示例

```bash
# 标准配置下修改单一参数
python TriPoseFusion/train.py model.geofusion_refiner_layers=2

# 同时修改多个参数
python TriPoseFusion/train.py \
    model.geofusion_hidden_dim=64 \
    data.batch_size=16 \
    train.max_epochs=10
```

---

## 🔧 配置模板创建

### 从模板创建自定义配置

```bash
# 从模板复制
cp configs/train_template.yaml configs/train_custom.yaml

# 编辑自定义配置
vi configs/train_custom.yaml

# 使用自定义配置
python TriPoseFusion/train.py --config-name train_custom
```

---

## 🎓 参数参考

主要模型参数快速索引：

| 类别 | 关键参数 | 默认值 |
|------|---------|-- ------|
| **时序建模** | `geofusion_refiner_layers` | 4 (RF=15) |
| **跨视图注意力** | `geofusion_attention_num_heads` | 4 |
| **正则化强度** | `geofusion_dropout` | 0.1 |
| **门控正则化** | `geofusion_gate_entropy_reg_lambda` | 0.01 |
| **损失权重** | `lambda_tri`, `lambda_view`, etc. | 见配置 |

---

## 🆘 故障排查

| 问题 | 可能原因 | 解决方案 |
|------|----------|---------|
| OOM (显存不足) | batch_size 过大 | 降低 `data.batch_size` 或 `geofusion_hidden_dim` |
| 训练慢 | num_workers 过小 | 增加 `data.num_workers` 至 32-64 |
| Loss 不下降 | lambda_tri 过小 | 增加到 1.0-2.0 |
| 某些视图从未被使用 | gate_entropy_reg_lambda=0 | 设置为 0.01-0.1 |

---

## 📊 推荐配置组合

### 快速验证 (Quick Start)

```yaml
# configs/train_minimal.yaml
data:
  batch_size: 20
train:
  max_epochs: 20
  devices: 1
```

### 标准训练 (Standard Training)

```yaml
# configs/train.yaml (默认)
data:
  batch_size: 30
  num_workers: 32
train:
  max_epochs: 50
  devices: 2
model:
  geofusion_refiner_layers: 4
```

### 高性能训练 (High Performance)

```yaml
# 基于 train.yaml 修改
data:
  batch_size: 46          # GPU 显存足够时
  num_workers: 64         # CPU 充足时
model:
  geofusion_hidden_dim: 256
  geofusion_refiner_dim: 512
```

---

**最后更新**: 2024-06-09  
**维护者**: Kaixu Chen
