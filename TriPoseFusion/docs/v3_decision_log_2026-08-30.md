# TriPoseFusion v3 决策与实验记录（2026-08-30）

## 1. 论文主线决定

**主线 = 逐序列自标定＋尺度锚定伪 GT 流水线 ＋ 数据集/Drive&Act 基准 ＋ TriPoseFusion v3 模型章节**（三者并列，不只押模型）。

v3 的四条主张（均已有证据）：
1. 免训练 canonicalized 融合：跨机位零样本 Drive&Act PA 23.1 mm，优于任一单视角；
2. 损坏感知门控（增强训练）：坏视角权重 0.02–0.10，置零视角 +1–5% vs 均值融合 +18%，≈ oracle 剔除，干净精度不变，可迁移；
3. 机位内残差修正（8K）：自有数据 −14.3% MPJPE；Drive&Act 域内官方 GT 监督 −23% PA，**自三角化参考（label-free）−15% PA**（本日实验 4）；零样本跨机位失效（23→36 mm）→ 定位为"目标机位无标注自适应"；
4. 逐关节 Laplace 不确定性：SetFusion 上 Spearman 0.72，残差头合体版验证中（实验 1）。

必须收窄/放弃的叙事：视角加权或注意力提升精度（1.9K ≈ 均值，权重≈均匀）；残差头跨机位通用。

## 2. v3 结构（`learned_fusion_experiments.py --residual-hidden 64 --residual-uncertainty`，增强 p=0.5）

```
SAM3D 52 关节 × 3 视角
 → ① canonicalize（neck 原点，肩轴 x，肩中点−neck 为 down；退化前向填充）      免训练
 → ② 门控 MLP 11→32→32→1（[置信度, 与均值偏差, 有限] + 8-d 关节嵌入）softmax；     1.9K
      时序深度卷积 k=5 零初始化；训练时随机一视角 丢失/置零/加噪
 → ③ 残差头 MLP 26→64→64→3（[融合, 三视角坐标, 偏差, 有限, 关节嵌入]）零初始化   +6.1K
 → ④ 末层第 4 维 = Laplace log-scale（偏置初始 −3），掩码 NLL 训练                  +65
监督：v3 伪 GT（rpe 3.31 px）/ 目标机位自三角化参考；固定步数，不在评估折选模型。
```

删除且有依据：跨视角注意力（去掉 0.1359→0.136，无差）、视角编码器/原容量 TCN（870K 反而 −4.2%、折间 ±20%）、SetFusion 关节注意力/时序头（无贡献）。门控塌缩根因 = attention 后视角 token 不可辨识（架构级）。

## 3. 实验 4：Drive&Act 自三角化参考的 label-free 自适应（本日完成）

缓存 `learned_fusion_cache_driveandact_selftri`（`build_driveandact_cache_selftri.py`）：训练参考 = SAM3D 2D + 官方标定 去畸变 DLT + 重投影 ≤40 px 过滤（与自有伪 GT 同配方）；官方 GT 仅验证。参考质量：hs9 有效 99%，52 关节 84–95%，中位 rpe 8–13 px，三角化肩宽 0.32–0.34 m，自参考 vs 官方 GT PA 14–22 mm。LOSO 3 折，1500 步 / batch 16 / 窗口 32，与 `da_resid64` 完全一致。

| 训练参考 | 模型 | MPJPE | PA | Δ vs 均值融合 |
|---|---|---|---|---|
| — | 均值融合 | 0.0247 | 0.0243 | — |
| 官方 GT | 残差 64 | 0.0203 | 0.0186 | −18% / −23% |
| 官方 GT | 残差 64＋增强 | 0.0216 | 0.0193 | −13% / −21% |
| **自三角化 hs9** | **残差 64** | **0.0225** | **0.0206** | **−9% / −15%** |
| 自三角化 52 | 残差 64 | 0.0230 | 0.0209 | −7% / −14% |
| 自三角化 52＋增强 | 残差 64 | 0.0235 | 0.0212 | −5% / −13% |
| 自三角化 52＋增强 | 学习型 1.9K | 0.0247 | 0.0226 | 0% / −7% |

逐折（自三角化 hs9）：fold0 −16%/−24%，fold1 −1%/−7%，fold2 −12%/−17%。
结论：无 3D 标注的机位内自适应成立，收益 ≈ 官方 GT 监督版 2/3，随参考质量单调；增益来自残差修正而非加权。

## 4. 实验 1–3（进行中，登录节点 3 并行，预计 2026-08-30 ~18:30 完成）

- 实验 1 `perseq_resid64_aug_unc`：残差 64＋增强＋不确定性头，5 折 + 置零压力测试 + summarize；验收标准 MPJPE ≈0.128、Spearman ≈0.7。
- 实验 2 数据量曲线：Drive&Act `da_resid64_n1/n2`（已完成，n=4 即 da_resid64）；自有数据 `perseq_resid64_aug_n4/n16`（进行中，68 即 perseq_resid64_aug）。索引 `index_mapping_n{4,16}`、`driveandact_index_mapping_n{1,2}`。
- 实验 3 Drive&Act 压力测试：`da_resid64 / da_resid64_aug / da_learned_aug` × zero/noise/drop × 3 折（已完成，结果在各 fold 的 `stress_*.json`，待汇总）。早期读数：da_resid64 置零 front 视角 learned 0.0242 vs 均值融合 0.0844，坏视角权重 0.02。

## 5. 产物

- 提交（worktree-review-fixes）：`0a23c3b` 结果总表 `results_summary_2026-08-30.md`；`05a8a2a` ResidualFusion 不确定性头；`13bb369` 自三角化缓存构建器 + `--train-gt-key`。
- 数据：`learned_fusion_cache_driveandact_selftri/`（6 npz + summary.json）。
- 迁移清单：`data/drive/TRIFUSION_TRANSFER_MANIFEST.md`（两模型训练包 0.9 GB；v3 GT 11 GB）。

## 6. 待办

1. 实验 1–3 汇总后写入本文档与 `review_fix_plan.md`；决定不确定性头保留方式（合体 vs 两阶段）。
2. v3 内消融时序卷积（`--no-temporal`）；天然缺失帧（none_detected）上的门控测试。
3. rebuttal 正文数字替换为 v3 GT / 统一口径 / Drive&Act 四行。
4. 论文：分组（头/肩颈/躯干/手）报告残差头增益；limitation 写明输出在肩颈规范系、协议手部主导。
