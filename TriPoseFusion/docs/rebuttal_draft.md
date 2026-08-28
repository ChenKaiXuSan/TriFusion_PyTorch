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
| D | 绝对误差 vs SOTA | 框架 ✅，依赖 per-joint 分解数字 |
| E | 学习型基线 | 依赖新基线训练 |
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
- 修正后基线：MPJPE 【待填:corrected_fusion_baselines】 / PA-MPJPE 【待填】
  （单序列快检 0.304 / 0.062，全量运行中）。
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

   初步观察（旧 checkpoint，zero 遮挡，冒烟规模 2 batch/条件）：任一视角被
   置零后 gate 权重仍精确保持 1/3 均匀（含被遮挡视角），MPJPE 由 0.325 升至
   0.379–0.393——直接证实旧模型 gate 已塌缩且对输入质量无响应，作为重训
   前后的 "before" 对照。正式数字以全量验证集 + 重训 ckpt 为准。

### D. "绝对误差 0.412 m vs SOTA 22–30 mm" — 不可直接比较

- 关节集不同：本文 52 关节中 42 个为手指关节（SOTA 多为 14–17 身体关节）；
  伪 GT 手部有效率仅 ~34%（见 F 表），误差下限由伪 GT 噪声决定。
- 输入不同：SAM3D 单视输入的起点误差为 2.078 m；我们的贡献是相对改善
  【待填:改善百分比（修正后）】。
- per-joint 分解：躯干/头部关节 MPJPE = 【待填:corrected per_joint_mpjpe
  头/肩颈 组】，远小于手部 【待填】——绝对量级与 SOTA 场景可比性的差距主要
  来自手部与伪 GT 噪声。

### E. 缺学习型融合基线

新增学习型融合基线（相同 keypoint 输入、相同 GroupKFold 划分、与 Table 3
完全同一 compute_metrics/掩码协议）：逐帧·关节·视角特征
[SAM3D 置信度, 与跨视角均值的距离（几何一致性）, 视角有限性] + 关节索引
embedding → 小型 MLP 逐关节视角权重 → softmax（掩掉无效视角）加权融合 →
深度可分离时间卷积残差头。模型 ~2K 参数（刻意远小于主模型 0.87M，隔离
"可学习加权"本身的贡献）。同时在同一数据上报告固定 confidence 加权融合
作对照锚点。结果：【待填:learned_fusion_baseline MPJPE / PA-MPJPE vs 锚点】。

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

视频为 29.97 fps；片段按 uniform temporal subsample 取 16 帧后，速度/加速度
特征按实际帧间隔 Δt 归一化（单位 m/s、m/s²），与采样率解耦。
【核对代码后按实际实现微调措辞】

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

## 附：数字依赖清单

| 占位符 | 来源 | 状态（2026-08-28） |
|---|---|---|
| corrected Table 3 各行 | eval_fusion_baselines / eval_single_sam3d / eval_additional_baselines（修正代码，另一会话 15:01 启动的 run 经审计确认有效） | 本机运行中 |
| corrected full 模型（fold0 val） | eval_trifusion_pesudo_gt + 旧 ckpt | **已出**：MPJPE 0.949 / PA 0.346 / gate 0.3333 均匀 |
| per-joint 分解（D/F） | 上述 corrected 评估的 per_joint_mpjpe 输出 | 随评估产出 |
| c_lam0 gate 分布（C） | Pegasus job 955579 | 排队中 |
| 重训 ablation（Table 4） | Pegasus jobs 955575–955578 | 排队中 |
| 遮挡压力测试（C） | occlusion_stress_test.py | 脚本已验通，待重训 ckpt 跑全量 |
| 学习型基线（E） | learned_fusion_baseline.py | 缓存构建中，随后 fold-0 全量训练 |
| Drive&Act 公开集评测（xS2o Mandatory Eval） | docs/driveandact_results/（另一会话，已 merge 进 rebuttal-work） | **基线已出**（见 I 点）；TriPoseFusion ckpt 零样本待补 |
