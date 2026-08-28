# Rebuttal 草稿（2026-08-28）

对应三份评审：LAYQ / 19jf / xS2o（均 Weak Reject，置信度 4）。
回应策略见 `review_fix_plan.md`；本文件是可提交正文的工作稿。

**占位符约定**：`【待填:xxx】` = 依赖正在跑/排队中的实验，数字一出即可替换。

进度跟踪：

| 点 | 内容 | 状态 |
|---|---|---|
| A | Procrustes 缺陷披露 + 修正数字 | 文字 ✅，数字【待填:修正评估】 |
| B | 回退算法伪代码 | ✅ |
| C | gating 塌缩回应 | 框架 ✅，依赖 c_lam0 重训 + 压力测试 |
| D | 绝对误差 vs SOTA | ✅ 分解表已出（躯干 0.340 全场最优、手部 1.580 全场最差），讲法已改写 |
| E | 学习型基线 | ✅ 已出数（1.9K 参数：MPJPE 0.981 / PA 0.290，优于旧 ckpt 主模型） |
| F | 伪 GT 可靠性 | ✅（数字已产出） |
| G | 轻量主张 | ✅ |
| H | 小项（映射表/帧率/划分） | ✅ |
| I | Drive&Act 公开集评测（xS2o Mandatory Eval） | ✅ 基线数字已出；模型 ckpt 零样本待补 |

---

## 0. 致全体评审的共同说明（诚实披露）

We thank all reviewers for their careful reading. Prompted by R-LAYQ's observation
that PA-MPJPE exceeds MPJPE for the canonicalized baseline (Table 3), we audited our
evaluation code and found **an implementation error in the Procrustes alignment**:
the optimal rotation under the row-vector convention was applied transposed, and the
scale used a norm-ratio instead of the least-squares optimum. This inflated **all**
reported PA-MPJPE numbers (unaligned MPJPE was unaffected). We also found and fixed
two further defects the reviews pointed toward: a zero-gradient gate-entropy
regularizer (R-xS2o) and a degenerate shoulder-fallback in canonicalization that
collapsed entire frames to the origin (R-LAYQ). All numbers below are recomputed
with the corrected code; corrected checkpoints for the retraining-dependent rows are
reported where available. 【待填:哪些表格行已换新数字】

（注：三个修复均有数值回归测试；合成数据上修复后的 PA 误差回到噪声水平
0.04–0.05，旧实现为 0.7–2.4。）

**关节空间审计（2026-08-28 下午，结论：无第四缺陷）**：曾怀疑基线在 70 关节
原始布局上误用 52 空间锚点（70 布局的 51 号是手指关节）。经跨会话交叉核实，
load_sam3d_frame / load_gt_sequence 在加载时即按 KEEP 做 70→52 映射，锚点
(neck=51, shoulders=5/6) 在映射后的模型空间中完全正确，基线与模型行关节集
一致——**该怀疑不成立**。已将此前提固化为 4 个守卫测试
（tests/test_joint_space_alignment.py，防止未来移除加载器 KEEP 或二次映射）。
修复后代码的单序列快检（01/夜多い）：canonicalized 融合基线 MPJPE 0.304 m /
PA-MPJPE 0.062 m，PA < MPJPE 反常消失——归因于上述三处修复本身。

---

## 1. R-LAYQ

### A. "PA-MPJPE (1.664) > MPJPE (0.969) 反常" — 属实，已定位并修复

上面的共同说明 + 具体数字：

- 旧 Table 3 canonicalized 基线：MPJPE 0.969 / PA-MPJPE 1.664（异常来源 =
  转置旋转 + 非最优 scale + 塌缩帧）。
- **修正后 Table 3 基线（88 序列均值，canonicalized，修正 Procrustes）**：

  | 行 | MPJPE | PA-MPJPE | PCK@0.10 |
  |---|---|---|---|
  | single front | 0.987 | 0.600 | 0.116 |
  | single left | 0.961 | 0.536 | 0.110 |
  | single right | 0.974 | 0.552 | 0.107 |
  | best single | 0.952 | 0.565 | 0.113 |
  | fuse mean | 0.971 | 0.564 | 0.111 |
  | fuse median | 0.971 | 0.560 | 0.112 |
  | median + smooth (w5) | 0.971 | 0.559 | 0.112 |
  | fuse confidence | = mean（SAM3D 输出无逐关节置信度，退化为均匀权重；正文须注明或删行） |

  **PA < MPJPE 反常在所有行消失。**

- **同子集对照（fold-0 val 20 序列，修正管线，旧 ckpt；
  `corrected_fold0_val_comparison.json`）——这是论文 Table 3 应采用的口径：**

  | 方法 | MPJPE | PA-MPJPE | PCK@0.10 |
  |---|---|---|---|
  | single front | 1.087 | 0.381 | 0.120 |
  | single left | 1.057 | **0.350** | 0.113 |
  | single right | 1.081 | 0.356 | 0.105 |
  | best single | 1.040 | 0.376 | 0.110 |
  | fuse mean (= confidence) | 1.072 | 0.362 | 0.114 |
  | fuse median | 1.076 | 0.361 | 0.115 |
  | median + smooth (w5) | 1.076 | 0.361 | 0.115 |
  | **TriPoseFusion（旧 ckpt 0-0.80）** | **0.994** | 0.367 | **0.149** |

  ⚠️ **核心结论（必须如实写入 rebuttal）**：修正管线下，旧 checkpoint 的模型
  相对最佳基线仅 MPJPE −4.4%、PCK@0.10 +30%（相对），PA-MPJPE **持平**
  （0.367 vs 0.350）。论文 Table 3 的大幅领先是三处 bug（尤其 canonicalize
  塌缩同时作用于 GT 与预测）造成的假象。因此：
  1. 旧 ckpt 是用带 bug 的 canonicalizer 训练的，与修正评估存在训练/评估
     错配；**Pegasus 5 个修正重训（955575–79）的结果是 A 点主表的唯一
     依据**，rebuttal 主张的强弱取决于它们。
  2. 重训结果出来前，措辞保守为"MPJPE/PCK 小幅改善、PA 持平"，不再写
     "显著优于"。
  3. PCK 明显改善说明模型在"对得准的关节"上更好——配合 per-joint 躯干/手部
     分解（D 点）：若躯干关节差距更大，仍是可辩护的贡献。
  4. 若重训后优势仍小，则诚实收窄贡献为：canonicalization 协议 + 轻量融合
     + 手部关节鲁棒性（PCK）+ 公开集零样本评测（I 点），并主动承认 Table 3
     原数字错误。
  （口径细节：基线为逐帧-逐序列均值，模型为逐片段-逐序列均值，帧覆盖略异，
  不改变结论。fold_0.json 实为 train 68 / val 20 序列。）

- **修正代码重训（Pegasus 955575–79）的 ckpt 修正评估（fold-0 val，
  eval_trifusion_pesudo_gt 聚合口径；`eval/logs/ckpt_sweep_full`、`ckpt_sweep_ablation`）：**

  | 配置 / ckpt | MPJPE | PA-MPJPE |
  |---|---|---|
  | full — epoch 0 / epoch 1 / 最后 epoch | 0.9488 / 0.9481 / 0.9481 | 0.346 |
  | base_simple（无 robust canon、无 TCN、无多尺度速度） | 0.9478 | 0.346 |
  | uniform_gate | 0.9488 | 0.346 |
  | no_cross_view_attention | 0.9487 | 0.346 |
  | full_gate_lambda0（λ_gate=0，gate 仍精确 0.333） | 0.9478 | 0.344 |
  | （旧 ckpt，同口径） | 0.949 | 0.346 |

  **结论：架构开关与 epoch 均无关，所有配置收敛到同一解（≈ 中位数融合）。**
  论文 Table 4 "robust canonicalization 0.846 → 0.412（−51%）"确认为 bug 3
  的塌缩假象（不可靠帧 GT 与预测同时归零）；修正后 canonicalization 的
  **评估侧**贡献（raw 2.08 → canonicalized ~1.0）仍然成立，但作为**模型模块**
  的消融增益不存在。根因见 C 点第 4 条（训练目标的平凡解）。
  rebuttal 写法：撤回 Table 4 的模块增益主张；Table 3/4 改为修正数字；
  贡献重述为"鲁棒 canonicalization 协议 + 轻量融合（≈固定融合）+ 公开集验证"，
  除非留一视角目标（C 点第 4 条）的重训明显改善。
- **主结果（full 模型，旧 ckpt + 修正评估，fold0 val）：MPJPE 0.949 /
  PA-MPJPE 0.346**（论文原 0.412 / 0.275）。⚠️ 方向与直觉相反：旧
  canonicalization 塌缩 bug 使 GT 与预测同时塌缩到原点，**人为压低**了论文
  数字；修正后模型行数字变差。rebuttal 必须如实呈现，并把论证重心放在
  (i) 相对单视角 SAM3D 起点的改善幅度（同口径修正后重算）、(ii) 重训后
  的数字（修正代码训练的 ckpt 预计显著优于"旧 ckpt+新评估"的错配组合，
  因为 canonicalizer 行为已改变）、(iii) per-joint 分解。全 fold 汇总
  【待填:corrected_trifusion_full 其余 fold】。
- 新增与 PA-MPJPE 完全同掩码的 `mpjpe_pa_frames` 指标，排除 mask/聚合协议差异
  的解释（回应"aggregation protocol"疑问）。

### B. "肩部退化时的回退规则不清" — 给出精确算法

修复后的回退算法（与代码逐行对应，模型端与评估端 numpy 实现一致，
约定 x = right − left）：

```
输入: 每帧 left/right shoulder, neck; 阈值 0 < ‖L−R‖ ≤ 2·0.5m
1. valid[t] = eps < ‖L[t]−R[t]‖ ≤ 1.0m          # 肩宽合法性
2. 对 invalid 帧:
     x_axis[t] ← x_axis[nearest valid frame]     # 同片段内前向填充;
                                                 # 开头连续 invalid 用第一个 valid 帧回填
   若整个片段无 valid 帧: x_axis ← 默认单位侧向轴 [1,0,0]
3. z_axis = normalize(neck − mid_hip)（躯干轴）
   若 x_axis 与 z_axis 接近平行（叉积范数 < eps）: 改用参考轴兜底
4. y = normalize(z × x); x = normalize(y × z); R = [x y z]
```

旧实现把两肩都替换为 neck，导致 x 轴为零向量、旋转矩阵为零矩阵、整帧
canonicalized pose 塌缩到原点——这正是评审猜测的来源，已修复并有回归测试。

### 其他小点

- 52 关节映射表：见共通补充 H-1。
- GroupKFold：见共通补充 H-3（任何折的 train/val 无受试者重叠）。

---

## 2. R-19jf

### F. 伪 GT 可靠性 / 手部遮挡 — 用重投影统计正面回答

对全部 88 个 受试者×环境 序列（701,565 帧）的三角化伪 GT 统计
（修复后重新生成的伪 GT）：

| 关节组 | 关节数 | 有效率 | 平均重投影误差 (px) |
|---|---|---|---|
| 头部（鼻/眼/耳） | 5 | 1.000 | 8.29 |
| 肩/肩峰/颈 | 5 | 1.000 | 4.37 |
| 右手 | 21 | 0.366 | 12.16 |
| 左手 | 21 | 0.322 | 13.99 |

- 全体 52 关节有效率 0.470；重投影误差 mean 9.58 px
  （p50 = 7.79，p90 = 20.53，p99 = 33.09）。
- **头/肩颈 10 关节有效率 100%、重投影误差 4–8 px**：伪 GT 对身体主干高度可靠。
- 手部关节有效率仅 ~34%：正文将把手部误差单列并明示其参考不确定性；
  评估仅统计三角化 valid 的关节（评估协议已按 valid mask 掩码）。
- 备选协议：提供"仅头/肩颈关节"的补充评估行【待填:corrected 评估的
  per-joint 分解即可导出】。
- 每序列诊断量（LOO 视角位移、条件数、深度方差）已随伪 GT 一并发布，
  支持复核。

---

## 3. R-xS2o

### C. "gating 塌缩到均匀 → adaptive 主张不成立"

两层回应：

1. **塌缩不是熵正则强迫的**：我们发现熵正则实现存在 clamp 顺序错误
   （`log` 后 clamp），损失恒为 ln V、梯度为零——即训练中该正则从未生效
   （训练日志 loss_gate_entropy 恒 1.0986 = ln 3 可佐证）。旁证：关闭
   cross-view attention 的 run 学出了非均匀 gate (0.28/0.34/0.38)。
2. **重训验证**：修复后代码、λ_gate = 0 的重训 run（c_lam0）检验 gate 是否
   自发分化：【待填:c_lam0 gate 权重分布】。并补充单视角合成遮挡/置零压力
   测试，观察 gate 对视角质量骤降的响应：【待填:occlusion_stress_test 结果】。
   若 gate 仍趋均匀，我们将把主张收窄为 "canonicalization + uniform fusion"
   （注：ablation 中 uniform_gate 本就为最优 0.7891，方法的主要贡献不依赖
   adaptive gating 成立）。
4. **根因与修正方向（训练目标审计，2026-08-28 晚）**：论文 3.7 的自监督目标
   以三视角中位数为 $\mathcal{L}_{tri}$ 的 teacher，view/bone/temp 亦为自洽项——
   **均匀融合即全局最优**；训练日志显示 val/loss 的 99% 是 InfoNCE（0.1×7.8），
   几何项自 epoch 0 起 ≈0 且不变。因此 gate 均匀不是熵正则所致（该项梯度为零），
   而是目标函数的最优解；论文 3.5 节"entropy regularization keeps the average
   weights close to uniform"的解释需改写。已实现**留一视角（leave-one-view-out）
   自监督 teacher**：训练时随机屏蔽一个视角（特征置零 + attention mask + gate=0），
   $\mathcal{L}_{tri}$ 目标改为该视角的 canonical 观测——无平凡解、且给 gate
   分化提供梯度；配 λ_nce=0.01、λ_gate=0、按 val/loo_mpjpe 选 ckpt。
   仍是自监督、不使用伪 GT。结果：【待填:c_loo 957197 vs c_looctl 957198
   的修正 MPJPE/PA、gate 分布、遮挡压力测试】。
3. **旁证（E 点学习型基线）**：在完全相同的 keypoint 输入上，一个 1.9K 参数、
   仅用跨视角几何一致性特征的融合器学出了明显非均匀的视角权重
   （front 0.41 / left 0.32 / right 0.27）并因此优于固定融合——说明**自适应
   加权在此数据上是可学的**；主模型 gate 塌缩为均匀是训练目标（熵正则失效）
   /架构层面的问题，而非"数据不支持自适应加权"。这也正是 λ_gate=0 重训
   （c_lam0）要验证的。

   初步观察（旧 checkpoint，zero 遮挡，冒烟规模 2 batch/条件）：任一视角被
   置零后 gate 权重仍精确保持 1/3 均匀（含被遮挡视角），MPJPE 由 0.325 升至
   0.379–0.393——直接证实旧模型 gate 已塌缩且对输入质量无响应，作为重训
   前后的 "before" 对照。正式数字以全量验证集 + 重训 ckpt 为准。

### D. "绝对误差 0.412 m vs SOTA 22–30 mm" — 不可直接比较

- 关节集不同：本文 52 关节中 42 个为手指关节（SOTA 多为 14–17 身体关节）；
  伪 GT 手部有效率仅 ~34%（见 F 表），误差下限由伪 GT 噪声决定。
- 输入不同：SAM3D 单视输入的起点误差为 2.078 m；我们的贡献是相对改善
  【待填:改善百分比（修正后）】。
- per-joint 分解（fold-0 val 20 序列，逐序列均值再平均，52 空间分组：
  head 0–4 / shoulders_neck [5,6,49,50,51] / body = 两者合集 / hands 7–48）：

  | 方法 | head | shoulders_neck | **body (10)** | **hands (42)** | hands÷body |
  |---|---|---|---|---|---|
  | single front | 0.485 | 0.288 | 0.387 | 1.424 | 3.68× |
  | single left | 0.482 | 0.295 | 0.388 | 1.366 | 3.52× |
  | single right | 0.485 | 0.293 | 0.389 | 1.398 | 3.59× |
  | best single | 0.481 | 0.294 | 0.387 | **1.346** | 3.48× |
  | fuse mean / median / median+smooth | 0.484 | 0.292 | 0.388 | 1.390–1.393 | 3.58× |
  | 学习型融合基线（E 点，1.9K） | 0.456 | 0.265 | 0.360 | **1.294** | 3.59× |
  | **TriPoseFusion（旧 ckpt）** | **0.414** | **0.265** | **0.340** | 1.580 | 4.65× |
  | TriPoseFusion（重训 ckpt） | 【待填:955575 出 ckpt 后，同样输出四组】 | | | | |

  （合并表 `corrected_fold0_val_joint_groups.json`；口径：主模型行为 fold 内
  valid-point 加权，其余为逐序列均值再平均，差异不足以翻转下述模式，正文须注明。）

  **D 点讲法（替换笼统的"不可直接比较"）**：
  1. 绝对误差由 42 个手部关节主导——**所有方法**手部/躯干误差比 ≈ 3.5×
     （手部占 52 关节的 81%，伪 GT 手部有效率仅 34%）；Drive&Act 标准 body 协议
     下同一管线为厘米级（I 点）。这是与 SOTA 数字量级差异的来源。
  2. **主模型躯干误差 0.340 m 显著优于所有基线**（固定融合 0.388 −12%，
     学习型 0.360 −6%）——模型在可靠关节上的优势是真实的。
  3. **手部 1.580 m 是全场最差**（vs best single 1.346 +17%、学习型 1.294 +22%）：
     整体 MPJPE 的小幅领先全部来自躯干，手部被学习模块拉坏。这是当前模型的
     明确弱点，如实写出，并与 19jf 手部遮挡质疑衔接（手部伪 GT 本身 66% 无效，
     模型在此学到的很可能是噪声）。重训 ckpt 评估须同样输出四组，检查手部退化
     是否为旧 canonicalizer bug 造成的训练污染；bone/temporal 项需针对手部单独
     检查（可考虑修订版对手部关节降权或单独报告）。

### E. 缺学习型融合基线

新增学习型融合基线（相同 keypoint 输入、相同 GroupKFold 划分、与 Table 3
完全同一 compute_metrics/掩码协议）：逐帧·关节·视角特征
[与跨视角均值的距离（几何一致性）, 视角有限性, （SAM3D 置信度——实测恒为 1，
无信息，保留仅为接口一致）] + 关节索引 embedding → 小型 MLP 逐关节视角权重 →
softmax（掩掉无效视角）加权融合 → 深度可分离时间卷积残差头。模型 ~2K 参数
（刻意远小于主模型 0.87M，隔离"可学习加权"本身的贡献）。同时在同一数据上
报告固定加权融合（= mean，因置信度缺失）作对照锚点。

**结果（fold-0 val 20 序列，逐序列均值，`eval/logs/learned_fusion_baseline/learned_fusion_fold0.json`）：**

| 方法 | 参数量 | MPJPE | PA-MPJPE | PCK@0.10 |
|---|---|---|---|---|
| 固定均值融合（锚点，同数据） | 0 | 1.059 | 0.336 | – |
| **学习型融合基线（本节）** | 1.9K | **0.981** | **0.290** | **0.152** |
| TriPoseFusion（旧 ckpt，同子集，见 A 点） | 0.87M | 0.994 | 0.367 | 0.149 |

训练 3000 步内 val 指标稳定（MPJPE 0.981–0.989，PA 0.284–0.290），best step 1500。
学到的平均视角权重 front 0.41 / left 0.32 / right 0.27——**仅凭跨视角几何一致性
即可学出非均匀、有意义的视角权重**（对 C 点的旁证：自适应加权在此输入上可学）。

⚠️ **对 rebuttal 的含义**：修正管线下，1.9K 参数的学习型基线在三项指标上均
**优于旧 ckpt 的主模型**。与 A 点结论一致——旧 ckpt 与修正管线错配，
Pegasus 重训 ckpt 必须同时超过 (a) 固定融合基线 (b) 本学习型基线，主张才成立。
若重训后仍不及本基线，rebuttal 应诚实承认并把贡献收窄（见 A 点第 4 条），
同时可把本基线作为论文修订版的新对照行（评审明确要求的学习型基线）。

### I. "Mandatory Evaluation on Drive&Act" — 已完成（零样本 + 免训练融合）

详见 `docs/driveandact_results/README.md`（脚本 `eval/eval_driveandact_baselines.py`）。

设置：Drive&Act 官方 split_0 **test** 受试者 vp5/vp11/vp13（6 run，约 126 分钟），
1 fps 采样 7,855 个三视角同步时刻（±21 ms），NIR 1280×1024；每视角逐帧
SAM3D-Body（与主实验同模型同设置，检出 99.97%）；GT 为官方 `openpose_3d`
（BODY-25 多视角三角化伪 GT）。协议与主实验一致（逐视角 canonicalize → 融合 →
修复后的 compute_metrics），两处因 BODY-25 定义必须的调整：down 轴用
midHip−neck（BODY-25 neck≈肩中点，shoulder_mid−neck 会退化）、预测侧 neck 同取
肩中点。GT 膝/踝置信度恒 0，故报 **body14** 与 **upper12**；髋/肩/颈任一无效
整帧剔除，6,673/7,855 帧进入统计。

| 方法 | body14 MPJPE | body14 PA | upper12 MPJPE | upper12 PA |
|---|---|---|---|---|
| single front | 61.1 | 35.8 | 59.5 | 27.5 |
| single left | 64.8 | 37.8 | 62.9 | 31.9 |
| single right | 54.1 | 34.0 | 50.6 | 27.3 |
| **fuse mean** | **55.8** | **31.3** | **53.6** | **23.8** |
| fuse median | 57.8 | 32.9 | 55.4 | 24.6 |

（pooled，mm。按受试者 fuse-mean body14 MPJPE 51–62 mm，PA 28.7–33.5 mm，跨人稳定。
PCK@50mm 42%、@100mm 94%。）

论点：
- 评审引用的 SOTA 22.6–30.4 mm 是**在 Drive&Act 上全监督训练**的方法；本结果为
  **零样本 SAM3D + 免训练融合**，PA-MPJPE 31.3 mm（上半身 23.8 mm）已在同一量级。
- 融合一致优于最佳单视角：PA −8%（body14）/ −13%（upper12），与主实验方向一致。
- 直接回应 D 点：主实验 0.412 m 的绝对量级来自 52 关节协议（42 个手部关节）与
  自建伪 GT——同一管线在标准 body 协议下即为厘米级。
- 局限如实写：OpenPose 伪 GT 厘米级下限；SAM3D 默认体型躯干尺度比 GT 大 ~18%
  （留在未对齐 MPJPE 内，PA 消除）；1 fps 采样。
- 【待填:TriPoseFusion ckpt 零样本评测——需 30 fps 连续窗口，另一会话补抽密集片段后报告】

---

## 4. 共通补充（H / G）

### H-1. 52 关节映射表（SAM3D 70 关节空间 → 模型 52 关节）

| 模型索引 (52) | 源索引 (70) | 关节 |
|---|---|---|
| 0–4 | 0–4 | nose, left-eye, right-eye, left-ear, right-ear |
| 5–6 | 5–6 | left-shoulder, right-shoulder |
| 7–27 | 21–41 | 右手 21 关节（腕 + 5指×4） |
| 28–48 | 42–62 | 左手 21 关节（腕 + 5指×4） |
| 49–51 | 67–69 | left-acromion, right-acromion, neck |

（下半身关节因驾驶座遮挡不纳入评估。）

### H-2. 运动导数帧率归一化

如实说明（已核对代码）：视频 29.97 fps；训练/评估片段是 **16 个连续帧**（chunk=16，
uniform subsample 16→16 为恒等），运动导数为逐帧有限差分（步长 1/3/5），
**未做 Δt 归一化**。因此所有速度/加速度/急动度共享同一固定 Δt≈33 ms，在本数据集
内归一化只是常数缩放、不影响结果；但若换到不同帧率的数据，导数尺度会变化。
回应：在正文注明固定帧率与差分定义；修订版把差分除以 Δt（m/s、m/s²），并把
temporal loss 的加速度项同样按 Δt² 归一化——这是一个常数因子的改动，不改变
本文实验。

### H-3. GroupKFold 划分与序列统计

22 名受试者 × 4 环境（昼/夜 × 光量多/少）= 88 序列，共 701,565 帧
（~29.97 fps，序列长度 5,860–11,833 帧，中位 7,971 帧 ≈ 4.4 分钟）。
按受试者 5 折 GroupKFold（seed 42）：

| fold | train 受试者 | val 受试者 | val 序列 | val 帧数 |
|---|---|---|---|---|
| 0 | 17 | 5 (02,07,12,17,24) | 20 | 155,002 |
| 1 | 17 | 5 (01,06,11,16,21) | 20 | 171,654 |
| 2 | 18 | 4 (05,10,15,20) | 16 | 133,497 |
| 3 | 18 | 4 (04,09,14,19) | 16 | 112,379 |
| 4 | 18 | 4 (03,08,13,18) | 16 | 129,033 |

所有折均无 train/val 受试者重叠（已程序化验证）。

### G. 轻量化主张（已实测）

0.87 M 参数 / 3.33 MB (fp32) / CPU 单线程 6.2 ms/帧（99.4 ms / 16 帧片段，
登录节点 CPU）。

---

## 5. 评审逐条对照表（2026-08-28 晚核对原文）

图例：✅ 草稿已有完整回应；🟡 有回应但需补数字/措辞；❌ 尚未覆盖（下方已补初稿）。

### R-LAYQ
| # | 评审原文要点 | 状态 | 对应 |
|---|---|---|---|
| L1 | Table 3 PA-MPJPE > MPJPE 反常；说明 mask/单位/聚合/对齐代码，给修正结果 | ✅ | §0 三处缺陷 + A 点修正表 + 同掩码指标 |
| L2 | 回退规则不清（两肩都换成 neck → 零向量），要伪代码 | ✅ | B 点 |
| L3 | 参考来自同一组相机，非独立 3D 精度；建议 mocap 子集或直接三角化基线 | 🟡 | I 点 Drive&Act 独立 GT；补充见 5.1 |
| L4 | 0.412 m / PCK 0.141 仍高；Table 4 学习模块无增益、gate 均匀；收窄主张 | ✅ | A 点（修正重训表）+ C 点第 4 条 + D 点 |
| L5 | 52 关节映射表缺失 | ✅ | H-1 |
| L6 | 运动导数/temporal loss 无帧率归一化 | ✅ | H-2（已按代码如实改写） |
| L7 | 骨长正则只有 3 条；缺关节限制/生物力学合理性评估 | ❌ | 5.2 |
| L8 | 参数量/运行时/延迟未报 | ✅ | G 点 |
| L9 | GroupKFold/片段数/时长；Table 2 各视角 valid 数极不均衡 | 🟡 | H-3 已有划分统计；valid 不均衡解释见 5.3 |

### R-19jf
| # | 评审原文要点 | 状态 | 对应 |
|---|---|---|---|
| J1 | 严重遮挡下评估 21 关节手部；参考不可靠 | ✅ | F 点统计 + D 点手部/躯干分解 + 仅躯干协议 |
| J2 | canonicalization 机制未验证：是解决单目深度不一致，还是利用固定三视角配置？泛化性 | 🟡 | A 点（增益为假象）+ I 点（Drive&Act 不同相机配置下仍成立）；措辞见 5.4 |
| J3 | 标注生成/人工修正流程/不确定性细节 | 🟡 | F 点有重投影统计；**流程描述需作者确认**（见 5.5） |
| J4 | 其余模块贡献有限，讨论必要性 | ✅ | A/C 点收窄 |

### R-xS2o
| # | 评审原文要点 | 状态 | 对应 |
|---|---|---|---|
| X1 | gate 塌缩到 0.333；归因熵正则 | ✅ | C 点（真因是目标函数平凡解；LOO 重训进行中） |
| X2 | TCN/多尺度速度贡献仅 1 mm | ✅ | A 点修正重训表（所有配置相同） |
| X3 | 绝对误差 0.412 m vs Drive&Act SOTA 22.6–30.4 mm | ✅ | D 点分解 + I 点 Drive&Act 厘米级 |
| X4 | 缺深度学习基线（UPose3D / Learnable Triangulation / Faster VoxelPose） | 🟡 | E 点学习型基线；为何不比图像级方法见 5.6 |
| X5 | SAM3D 细节/领域偏差缺失；2.078 m 说明估计器灾难性失败 | ❌ | 5.7 |
| X6 | 伪 GT 重投影误差 26–42 px，指标循环 | ✅ | F 点 + 5.3 |
| X7 | PCK@0.10 阈值太松 | ❌ | 5.8 |
| Xs | 三条要求：修复塌缩 / Drive&Act / 深度基线 | 🟡/✅/🟡 | C（LOO 待出）/ I / E+5.6 |

### 5.1 L3：参考独立性
承认本数据集无 mocap；补充两点独立证据：(i) Drive&Act 上使用官方 OpenPose 三角化
GT（与本文伪 GT 生成流程无关、相机配置不同），同一管线 body14 PA 31.3 mm
（I 点）；(ii) 伪 GT 的重投影误差与 LOO 视角位移诊断量随数据发布，可复核。
"直接三角化基线"即伪 GT 本身，故不作为方法行。

### 5.2 L7：骨长正则与生物力学合理性（初稿）
承认仅约束 3 条上身骨长（左右肩、颈-左肩、颈-右肩），原因是 52 关节集缺乏
髋/膝等下身关节，肢体骨长受手部噪声主导。修订版可补充：预测序列的肩宽/上臂长
时间标准差（稳定性指标）与基线对比【待填:可从 corrected 评估输出计算】，
并明确不主张关节角度限制建模。

### 5.3 L9/X6：各视角 valid 数不均衡与前视 42 px
Table 2 前视 valid 关节 1.40M 远少于左/右 ~16M：三角化的逐视角有效性由重投影
残差门控决定，前视相机（内后视镜位置）对手部/肩部的透视缩短与遮挡最重，其 2D
检测更常被 outlier 拒绝，故参与三角化的比例低、参与时残差也更大（42 px P95）。
这正说明伪 GT 主要由左/右视角决定，对前视预测的评估更"独立"。
【待填:用重生成伪 GT 的 reproj_per_view / valid 统计给出精确比例——仅读取，不改生成】

### 5.4 J2：canonicalization 的机制与泛化（措辞）
诚实说明：修正评估后，canonicalization 作为**模型模块**的消融增益（0.846→0.412）
不成立（bug 假象）；其真实作用是**评估/融合前的坐标系对齐**：raw 单视角 2.08 m →
canonicalized 0.93–1.09 m（PA 0.35 m），即单目 3D 各视角的平移/尺度/朝向
不一致是主要误差源，canonicalization 消除的是这一项，而非"利用固定三视角配置"。
泛化证据：Drive&Act 的相机布局（A 柱/后视镜）、传感器（NIR）、GT 来源均不同，
同一 canonicalize→融合流程下融合仍优于最佳单视角 8–13%（I 点）。

### 5.5 J3：标注/人工修正流程（需作者确认）
论文 4.3 称 2D 关键点"manually checked and corrected"后三角化；rebuttal 描述必须
与实际流程一致（若人工修正为抽样核查而非逐帧修正，需如实改写并给出核查比例）。
**此项由作者定稿，本草稿不代写。**

### 5.6 X4：为何不与图像级深度方法直接比较，以及补了什么
UPose3D / Learnable Triangulation / VoxelPose 以图像特征或热图为输入并需标定
外参与 3D 监督，与本文"仅接收现成单目 3D 关键点、无 3D 标签"的问题设定不同，
同口径比较不可行；我们补充了同输入、同划分、同指标的**学习型融合基线**（E 点，
1.9K 参数、置信度/几何一致性加权回归，属 learnable-triangulation 风格的关键点级
对应物），并在 Drive&Act 上与全监督 SOTA 数字同表对照（I 点）。承认这不能替代
图像级基线，列为未来工作。

### 5.7 X5：SAM3D 细节与 2.078 m 的含义
补充：SAM3D-Body（sam-3d-body-dinov3 权重，MHR 参数化，70 关节输出），未在车内
数据微调；逐帧独立推理。**2.078 m 不是估计器失败**：该行是 raw 相机坐标系、未
canonicalize 的 MPJPE（root-MPJPE 0.9–1.2 m、PCK 0），反映单目预测与三角化参考
的原点/尺度不一致；同一批预测 canonicalize 后单视角 MPJPE 0.96–1.09 m、PA 0.35 m
（另一会话 corrected_single_sam3d 与 additional_baselines）。正文该行将标注
"raw camera frame, no canonicalization"。

### 5.8 X7：更严格的 PCK 阈值
修正评估已输出 PCK@0.02/0.05/0.10：主模型（旧 ckpt，fold-0 val）0.048 / 0.069 /
0.159；基线行【待填:fusion/additional summary 中的 pck_0.02/0.05】。补入表格，
并说明手部关节主导下严格阈值数值偏低（D 点分解）。

## 附：数字依赖清单

| 占位符 | 来源 | 状态（2026-08-28） |
|---|---|---|
| corrected Table 3 各行 | eval_fusion_baselines / eval_single_sam3d / eval_additional_baselines（修正代码，另一会话 15:01 启动的 run 经审计确认有效） | 本机运行中 |
| corrected full 模型（fold0 val） | eval_trifusion_pesudo_gt + 旧 ckpt | **已出**：MPJPE 0.949 / PA 0.346 / gate 0.3333 均匀 |
| per-joint 分解（D/F） | corrected_fold0_val_joint_groups.json（另一会话合并表） | **已出**（旧 ckpt）；重训 ckpt 四组待补 |
| c_lam0 gate 分布（C） | Pegasus job 955579 | 排队中 |
| 重训 ablation（Table 4） | Pegasus jobs 955575–955578 | 排队中 |
| 遮挡压力测试（C） | occlusion_stress_test.py | 脚本已验通，待重训 ckpt 跑全量 |
| 学习型基线（E） | learned_fusion_baseline.py | **已出**（fold-0：MPJPE 0.981 / PA 0.290 / PCK 0.152）；其余 fold 可按需补 |
| Drive&Act 公开集评测（xS2o Mandatory Eval） | docs/driveandact_results/（另一会话，已 merge 进 rebuttal-work） | **基线已出**（见 I 点）；TriPoseFusion ckpt 零样本待补 |
