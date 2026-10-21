# ACCV 2026 Rebuttal 工作包（2026-09-01）

> **定稿记录（2026-09-02）**：作者已确认 ① 披露策略（四缺陷公开披露 + 主张收窄）；
> ② 伪 GT 表述属实（`sam3d_body_results_right/` 的 2D 关键点经人工矫正，三角化本身自动）。
> 提交前审查修正 2 处无数据支撑表述（三版参考→两版；D&A 单视角对比改为 canonical 系）。
> 最终文件：`accv2026_rebuttal_official/rebuttal.pdf`（#802，1 页），副本在 `TriFusion_accv/rebuttal/`。
>
> **v2 修订（2026-09-02，依据外部审读意见）**：① 新增统一结果表 Tab. 1（A 自有数据 fold-0 片段级
> 聚合 / B 5 折诊断 / C Drive&Act 零样本），每块一个协议、一个 valid mask，正文数字改为片段级
> 口径（模型 0.189/0.063）；② LAYQ W3 改为"无独立 MoCap，不再主张绝对精度"，Drive&Act 只作
> 跨机位迁移证据（其 GT 本身是 OpenPose 三角化）；③ 门控段承认熵项若生效确会推向均匀，再说明
> 实现梯度为零、去掉后仍塌缩、因此移除该项；④ 补 Drive&Act 协议（split-0 test、10 关节、PA 主
> 指标原因、自适应 LOSO 受试者不相交）、手部 mask 全方法统一、尺度锚点来源（SAM3D 肩宽，PA/
> 重投影与尺度无关）、回退轴定义与前瞻说明、延迟仅融合阶段；⑤ 删除与 SOTA 区间的直接比较；
> 基线措辞改为 "not input-matched"，8K 残差头标注为 diagnostic，原架构改述为"同协议重训未优于
> 简单头"；⑥ 补 DLT 直接三角化行（Drive&Act 官方标定，32.5 mm 绝对 MPJPE）。
>
> **v2.1（同日，作者质询"为何加 8K 实验"）**：8K 残差头只保留其正当角色——xS2o request 3 要求的
> 同输入学习型基线家族中的一行（Tab. 1B 三行：1.9K / 8K / 原架构重训 870K）；删除 Drive&Act
> "机位内自适应 18.6 mm" 行与相关句（无评审要求、属新贡献、触碰模板红线、且把叙事带偏为"新模型"），
> 删除 A1 中"8K 头是 diagnostic 而非替换"的声明（不再需要）。18.6 mm 结果留给新论文。
>
> **v2.2（同日，第二轮外部审读）**：① A3 补 UPose3D（标定多视角 2D 关键点＋不确定性，与无标定
> 3D 关键点接口不匹配；set-attention＋不确定性头覆盖其思路）；② 19jf W2 重写：canonicalization 是
> 定义公共坐标系的前提步骤，raw-vs-canonicalized MPJPE 无意义（论文 2.078 m 是坐标系错配），可
> 量化效果 = 融合相对最佳单视角的增益（干净输入下很小：MPJPE −1% / PA −5% / PCK +2pt；D&A PA
> 24.0→23.1）＋视角丢失冗余＋跨机位迁移，明写"modest contribution"；③ A2 解释 D&A 上 rigid-MPJPE
> 融合不如最佳单视角的原因（SAM3D 逐视角尺度偏差 ≈18%，平均不消除）；④ 加回 hedge：零样本误差与
> 有监督 22.6–30.4 mm 同量级但不直接可比；⑤ 表格卫生：870K 补 PCK@0.05=0.35，DLT 移出 C 块改为
> 脚注（sanity anchor），删 1.9K 行（正文注明≈mean）。**无法照做的建议**：加 "raw mean fusion"
> 行——raw 各视角输出在各自相机系，MPJPE 无定义，PA 对刚体变换不变，该行要么是稻草人要么无信息。
> **作者须知**：修正后 canonicalization+uniform fusion 相对最佳固定单视角的干净增益仅 1–5%，
> 逐序列 oracle 最佳单视角（0.174 seq-mean）甚至优于均值融合（0.186）；这是论文剩余贡献的真实上限。
>
> **v2.3（2026-09-02 晚，版本合并）**：以 `TriFusion_accv/rebuttal/` 17:44 版为准（A2 的 PCK 引用改正为
> 0.05/0.10，与 Tab. 1 列一致），放弃 worktree 内 4 处未提交的措辞微调；补两个指针：19jf W2 末
> "Module necessity: W4, A1."、xS2o A2 末 "M2: W3, 19jf W1."；为回 1 页压缩 A3/A2 两处从句。
> 至此三份审稿 20 条意见每条都有显式编号回应或指针。最终稿 = `accv2026_rebuttal_official/rebuttal.tex`
> ≡ `TriFusion_accv/rebuttal/rebuttal.tex`（需在 TriFusion_accv 里 commit+push 到 Overleaf）。

材料来源：`reviews_ac1f.txt`（三份评审全文）、`main.tex`（论文）、
`results_summary_2026-08-30.md`（三阶段数字总表）、`review_fix_plan.md` §二·五~二·十、
`v3_decision_log_2026-08-30.md`。数字若与本文件冲突，以 `results_summary_2026-08-30.md` 为准。

---

## 阶段一：审稿意见分析

三位评审均 Weak Reject（2 分）、置信度均 4。没有任何一条核心技术指控是误读——
代码审计证实全部成立。因此本 rebuttal 的性质不是"澄清误解"，而是
"承认缺陷 → 展示修正 → 用修正后的更强证据重建主张"。

### R-LAYQ（标题："evaluation validity 存疑"；态度最可争取，明说修正数字 could materially change the assessment）

| # | 问题 | 分类 | 是否成立 | 优先级 |
|---|---|---|---|---|
| L1 | Table 3 反常：canonicalized 基线 PA 1.664 > MPJPE 0.969；要求核查 mask/单位/聚合/对齐代码并给修正数字 | evaluation concern | **成立**（Procrustes 转置旋转+非最优 scale+塌缩帧） | **Critical** |
| L2 | §3.3 肩部回退：双肩替换为 neck → 肩向量为零，缺前帧/单位轴规则 | methodological + missing clarification | **成立**（推断完全正确） | Important |
| L3 | 评估参考与被评系统同源（同相机），建议小规模 mocap 或独立参考 | evaluation concern | 成立（部分可满足） | Important |
| L4 | 0.412 m 绝对误差 + Table 4 学习模块无增益 + gate 均匀 → 收窄主张 | evaluation + novelty | **成立** | **Critical** |
| L5 | 小项：52 关节映射表缺失；运动导数无帧率归一化；骨长正则只 3 段；模型规模/延迟未报；GroupKFold/时长/逐视角有效数失衡未文档化 | presentation/minor | 成立 | Minor |

### R-19jf（标题：伪 GT 可靠性 + canonicalization 泛化性）

| # | 问题 | 分类 | 是否成立 | 优先级 |
|---|---|---|---|---|
| J1 | 严重遮挡下评估完整 21 手部关节；"manually corrected 三角化参考不能视为可靠 GT" | evaluation concern | **成立**（手部有效率仅 0.34） | **Critical** |
| J2 | canonicalization 增益机制未验证：是解决跨视角单目深度不一致，还是利用固定三视角配置 | methodological concern | 半成立（有正面证据可答） | Important |
| J3 | 标注生成/人工修正流程/不确定性估计细节不足 | missing clarification | 成立 | Minor |
| J4 | 其余模块（attention/temporal/gating）贡献有限，必要性需讨论 | novelty concern | 成立（与 L4/X1 同簇） | Important（并入 L4） |

### R-xS2o（标题："Mathematical Collapse"；最尖锐，rebuttal 三点点名要求）

| # | 问题 | 分类 | 是否成立 | 优先级 |
|---|---|---|---|---|
| X1 | gate 功能性塌缩 0.33334/0.33334/0.33335，网络退化为静态算术平均；损坏视角仍被注入 33.33% | methodological concern | **成立** | **Critical** |
| X2 | TCN / 多尺度速度惰性：各只 1 mm | evaluation/novelty | 成立（与 L4 同簇） | Important |
| X3 | 绝对误差 0.412 m vs Drive&Act SOTA 22.6–30.4 mm："functionally unusable" | evaluation concern | 成立（但根因是 GT 尺度） | **Critical** |
| X4 | Table 3 无深度学习基线；点名 UPose3D / Learnable Triangulation / Faster VoxelPose 至少一个 | missing experiment | 成立（可部分满足） | **Critical** |
| X5 | "Mandatory Evaluation on Drive&Act"：不能只用私有模拟器数据 | missing experiment | 成立（已满足） | **Critical** |
| X6 | SAM3D 无架构/训练域/偏差分析 | missing clarification | 成立 | Minor |
| X7 | 伪 GT P95 重投影 26/42 px → 循环论证 | evaluation concern | 成立（与 J1/L3 同簇） | Important |
| X8 | PCK@0.10（100 mm）阈值过松 | evaluation minor | 成立 | Minor |

### 跨评审聚类（rebuttal 分节依据）

- **簇 A 评估有效性 / 伪 GT**：L1 + L3 + J1 + X7（+X3 的量级部分）→ 一节回答
- **簇 B 学习模块失效 / gate 塌缩 / 主张收窄**：L4 + J4 + X1 + X2 → 一节回答
- **簇 C 外部验证与学习型基线**：X5 + X4 + X3 + J2 → 一节回答（Drive&Act 是同一组证据）
- **簇 D 澄清类小项**：L2 + L5 + J3 + X6 + X8 → 一节合并

### 直接影响 AC 决定的点

1. **L1**（LAYQ 原话 "could materially change the assessment"）——修正数字是翻盘的唯一入口。
2. **X1/X4/X5**（xS2o 的三点点名要求）——三点全部或大部分满足，才能中和最负面的一票。
3. **J1**（19jf 的核心疑虑）——伪 GT 可靠性同时被三人质疑，是共同底层问题。

---

## 阶段二：回答策略（Critical / Important 逐条）

### L1（Critical）：PA>MPJPE 反常
- 核心担忧：度量实现有错 → 全部数字不可信。
- 已有证据：无（论文中该反常无解释）。评审判断正确。
- 误解：无。
- 需补充：承认 + 根因（Procrustes 转置旋转、norm-ratio scale、塌缩帧）+ 合成数据验证（旧实现 PA 0.7–2.4，修复后回到噪声水平 0.04）+ 修正数字。
- 新实验：已完成（修正重评 + 逐序列标定参考）。
- 最有说服力数字：canonicalized mean 融合 0.971/0.564（反常消除）；标定参考下 0.186/0.060。
- 回答（2 句）："The reviewer is right; we found a transposed rotation and a norm-ratio scale in the Procrustes implementation plus a frame-collapse bug. With the corrected code the reversal disappears (mean fusion 0.971/0.564 m; 0.186/0.060 m on the calibrated reference), and we additionally report a same-mask unaligned MPJPE."

### L4 + J4 + X1 + X2（Critical 簇）：gate 塌缩与主张收窄
- 核心担忧：核心贡献（adaptive gating + 学习模块）不成立，"mathematical paralysis"。
- 已有证据：论文 Table 4 本身已透明显示 canonicalization 主导（LAYQ 也把这列为 strength）。
- 误解：一处——xS2o 把塌缩归因于熵正则的最大熵推动；实际熵正则因 clamp 顺序 bug 梯度为零，从未生效。塌缩另有根因。
- 需补充：根因链（正则死代码 → λ=0 重训仍 0.3333 → LOO teacher 重训仍 0.3333 → 结论：median 伪教师目标下均匀融合是平凡最优，塌缩是架构级）。
- 无法回避处：诚实收窄——修正后 full 模型 0.1857/0.0593 ≡ mean 融合 0.1858/0.0595，学习模块在干净输入下无精度增益。
- 新实验（挽回点）：损坏增强训练使门控真正生效——1.9K 参数头给置零视角权重 0.037，MPJPE 仅 +1–5%（均值融合 +18%）；8K 残差修正头域内 −14%。这直接回应 xS2o "损坏视角被强制注入 33.33%" 的后果描述。
- 回答（3 句）："The collapse is real but not caused by the entropy regularizer (it had zero gradient due to a clamp-order bug); retraining with the fix, with λ=0, and with a leave-one-view-out teacher all still give 0.3333 gates, showing the collapse is architectural under a median pseudo-teacher. We therefore narrow the claim: the working core is canonicalization + uniform fusion. As a new experiment, corruption-augmented training restores functional gating (zeroed-view weight 0.04, degradation +1–5% vs +18% for mean fusion)."

### X5（Critical）：Drive&Act 强制评测
- 核心担忧：私有模拟器数据不可信。
- 新实验：已完成。零样本（不在 D&A 训练）：canonicalize+mean 23.1 mm PA、full 模型 ckpt 23.7 mm——落在 xS2o 自己引用的 SOTA 区间 22.6–30.4 mm 内；机位内 label-free 自适应 18.6 mm（−23%）。
- 最有说服力数字：23.1/23.7 mm vs 他引用的 22.6–30.4 mm。
- 回答（2 句）："We evaluated zero-shot on Drive&Act against its official reference: 23.1 mm PA-MPJPE (canonicalize+mean) and 23.7 mm (full model) — within the 22.6–30.4 mm SOTA range the reviewer cites — and 18.6 mm after label-free per-rig adaptation. This also provides the independent-reference validation requested by R-LAYQ."

### X3（Critical）：绝对误差量级
- 核心担忧：0.412 m 物理上不可用。
- 根因：伪 GT 构造外参未标定 → 各场次度量尺度错误（中位 2.38×）。必须披露。
- 修正后：标定参考上 0.186 m / PA 0.060 m（52 关节含双手；手部 0.316 主导，躯干 0.067、肩颈 0.033）；D&A 上 23 mm 级。
- 回答（2 句）："The 0.412 m magnitude was dominated by a metric-scale error in the constructed extrinsics of our triangulation reference (median 2.4×), which we fixed by per-sequence self-calibration with anthropometric anchoring (reprojection 9.60→3.31 px, shoulder-width std 0.369→0.015 m). On the calibrated reference the system reaches 0.186 m MPJPE / 0.060 m PA over 52 joints including both hands, and 23 mm PA on Drive&Act."

### X4（Critical）：深度学习基线
- 核心担忧：无学习型对照，贡献无法定位。
- 部分不可满足：点名的 UPose3D / Iskakov LT / Faster VoxelPose 需要图像/体素输入与标定相机，与本文 keypoint-only 融合接口模态不匹配（论文的贡献恰恰是绕开图像级融合）；rebuttal 期内也不可能复现三个图像级系统。诚实说明，不硬辩。
- 可满足部分（新实验）：在完全相同的 keypoint 输入上补齐学习型融合家族——置信度加权回归（learnable-triangulation 式，1.9K）、set-attention＋不确定性头（28K，Spearman 0.72）、残差修正头（8K，−14%）、原架构（870K）。共 16 组、5 折。
- 回答（2 句）："Image/volumetric baselines (VoxelPose variants, UPose3D) are not applicable to our keypoint-only interface, whose purpose is precisely to avoid image-level fusion; instead we add a family of learned fusers on the identical input (confidence-weighted LT-style, set-attention with uncertainty, residual head, 1.9K–870K params). The best (8K residual head) improves MPJPE −14% in-domain and 18.6 mm PA after adaptation on Drive&Act."

### J1 + L3 + X7（Critical 簇）：伪 GT 可靠性
- 核心担忧：用不可靠参考评估 → 循环论证；手部关节尤甚。
- 已有证据：论文 Table 2 已报告重投影误差（LAYQ 列为 strength：没有伪装成 mocap）。
- 需补充：per-joint 有效率（头/肩颈 1.00，手 0.34）；手部单列（hands 0.316 vs body 0.067）；仅躯干协议；逐序列自标定使参考质量可量化提升（rpe 9.60→3.31 px）；外部独立验证 = Drive&Act 官方 GT。
- mocap 子集（L3）：舱内 mocap 不可行（xS2o 自己在 strengths 里承认 "logistical impossibility"），用 D&A 官方参考替代，明说。
- 回答（2 句）："We now quantify reference reliability per joint (head/shoulder–neck valid rate 1.00, hands 0.34), report hands separately (hands dominate: 0.316 m vs torso 0.067), add a torso-only protocol, and improve the reference itself by per-sequence self-calibration (reprojection 9.60→3.31 px). Independent validation against Drive&Act's official reference (§Drive&Act) breaks the same-camera circularity."

### J2（Important）：canonicalization 泛化机制
- 回答（2 句）："The gain comes from removing per-view depth/scale inconsistency rather than from the fixed rig: the same canonicalize+mean procedure transfers zero-shot to Drive&Act's different camera geometry (23.1 mm PA), and raw→canonicalized single-view error drops consistently on both datasets."

### L2（Important）：回退算法
- 回答（1 句）："The reviewer's inference is correct — the missing rule is: temporally nearest valid shoulder axis within the clip (forward-fill + prefix back-fill), else a default lateral axis, with a reference-axis guard for the parallel case; pseudocode added to §3.3."

### 小项（L5/J3/X6/X8）：全部一句带过 + 指向 revision
- 52 关节映射表、GroupKFold 划分与时长统计、帧率归一化、SAM3D 细节（架构/训练域/在 NIR 域的偏差讨论）、PCK@0.05/0.02、0.87M 参数 / 3.33 MB / 6.2 ms/帧 CPU。

### 策略红线
1. 不与任何评审争对错——三人核心指控全部属实，姿态是"你们的审稿触发了审计，我们修完了"。
2. 所有新数字标注 New experiment / Additional analysis。
3. 不承诺做不到的事（不承诺复现 UPose3D/VoxelPose；不承诺 mocap）。
4. 伪 GT 表述与论文一致：人工修核作用于三角化输入端 2D 关键点（作者已定口径）。

---

## 建议新增实验/分析

**无需再加**——支撑上述全部回答的实验已完成（见 `results_summary_2026-08-30.md`）。
可选加分项（CPU 小时级，非必需）：
1. v3 合体配置的去时序消融（决策记录待办 #2，供终稿 Table 用，rebuttal 不引用）。
2. 天然缺失帧（none_detected）上的门控测试——若跑出与合成 drop 一致的结论，可在 rebuttal
   §3 加半句 "including naturally missing detections"。

---

## 阶段三 A：Full reasoning version（英文全文，不限篇幅）

### To all reviewers (common statement)

We thank all reviewers for exceptionally careful reading. Prompted by R-LAYQ's observation
that PA-MPJPE exceeds MPJPE for canonicalized baselines in Tab. 3, we audited our entire
evaluation stack. The audit confirmed the reviewers' concerns and identified four
implementation defects:

(i) the Procrustes alignment applied the optimal rotation transposed and used a norm-ratio
instead of the least-squares scale; (ii) the gate-entropy regularizer had zero gradient
(clamp applied after log; its loss was constant at ln 3 throughout training); (iii) the
shoulder-degeneracy fallback in canonicalization produced a zero rotation and collapsed
whole frames to the origin — exactly as R-LAYQ inferred; (iv) the camera extrinsics used to
triangulate the pseudo ground truth were constructed from layout assumptions and were never
calibrated, producing per-session metric-scale errors (median 2.4×, per-subject 1.2–6×).

All four are fixed (numerical regression tests included). The reference is rebuilt by
per-sequence self-calibration with anthropometric scale anchoring: mean reprojection error
9.60→3.31 px, shoulder-width std across 88 sequences 0.369→0.015 m, front-view utilization
8%→45%. Every number below uses the corrected code and, where stated, the calibrated
reference. We report this transparently: the corrected results resolve the anomalies the
reviewers identified, and strengthen rather than weaken the paper's practical case. Code,
calibration, and tests will be released.

### R-LAYQ

**W1 (metric anomaly).** You are right, and this observation triggered the audit. Cause:
defects (i)+(iii). Synthetic check: for ground truth transformed by a known
rotation+scale+noise, the old code returns PA error 0.7–2.4 m where the true residual is
0.04 m; the fixed code recovers the noise level. Corrected Tab. 3 (identical masks;
we additionally report a same-mask unaligned MPJPE to exclude aggregation-protocol
explanations): canonicalized mean fusion 0.971 m MPJPE / 0.564 m PA on the original
reference — the reversal is gone; on the calibrated reference 0.186 / 0.060 m.

**W2 (fallback rule).** Your inference is exactly right — a rule was missing from the text.
The complete fallback: if the shoulder span is degenerate, use the temporally nearest valid
shoulder axis within the same clip (forward-fill, then prefix back-fill); if the whole clip
is invalid, use a default lateral unit axis; a reference-axis guard handles the
shoulder-axis-parallel-to-torso case. Pseudocode added to §3.3; the model-side and
evaluation-side implementations are now unified (x = right − left).

**W3 (independent reference).** In-cabin mocap remains infeasible (as R-xS2o notes), but we
added the closest available substitute: zero-shot evaluation on Drive&Act against its
official multi-view reference (different rig, different sensor domain) — see R-xS2o
response. We also improved the internal reference itself by per-sequence self-calibration
(above), and verified conclusions are stable across three reference versions.

**W4 (narrow the learned-component claims).** We do. Corrected ablation: the full model and
uniform mean fusion are indistinguishable (0.1857/0.0593 vs 0.1858/0.0595); the paper's
working core is robust canonicalization + uniform fusion, and we rewrite the claims
accordingly. The gate analysis and the new corruption-augmented variant that does gate are
summarized in the R-xS2o response. Efficiency numbers you asked for: 0.87M parameters,
3.33 MB fp32, 6.2 ms/frame single-thread CPU.

**Minor.** The revision adds: the exact SAM3D→reference 52-joint mapping table; frame-rate
normalization for all temporal derivatives; GroupKFold subject splits, clip counts and
durations; discussion of the per-view valid-count imbalance (front 8%→45% after
calibration); broader bone-length coverage remains future work.

### R-19jf

**W1 (reference reliability under occlusion).** We now quantify it per joint: valid-rate
head/shoulder–neck 1.00, hands 0.34; hands dominate the corrected error (hands 0.316 m vs
body 0.067, shoulder–neck 0.033, head 0.101). The revision (a) reports hand joints
separately with their validity, (b) adds a torso-only protocol, (c) improves the reference
by per-sequence self-calibration (reprojection 9.60→3.31 px), and (d) validates externally
on Drive&Act's official reference (23.1–23.7 mm PA zero-shot). Manual checks/corrections
apply to the 2D keypoints input to triangulation; the triangulation itself is automatic
DLT with a 40 px reprojection filter — the revision states this pipeline precisely, with
per-sequence uncertainty statistics.

**W2 (canonicalization mechanism).** Additional analysis: the gain comes from removing
per-view depth/scale inconsistency, not from the fixed three-view layout. Evidence: the
same canonicalize+mean procedure transfers zero-shot to Drive&Act's different camera
geometry (23.1 mm PA); raw→canonicalized single-view error drops consistently on both
datasets; and the improvement persists across all three reference versions.

**Minor (module necessity).** Agreed — see the narrowed claims (R-LAYQ W4). The corrected
ablation attributes clean-input accuracy to canonicalization + uniform fusion; the learned
components' value is robustness under view corruption (R-xS2o response, item 1), not
clean-input accuracy.

### R-xS2o

**W1 (gating collapse) — "Rectify the Neural Collapse".** The collapse is real. One
correction to the attribution: the entropy regularizer could not have forced it, because a
clamp-order bug made its gradient exactly zero (loss constant at ln 3 in all logs).
Retraining with the corrected regularizer, with λ_gate = 0, and with a leave-one-view-out
teacher all still produce 0.3333 gates: the collapse is architectural — under a
median-of-views pseudo-teacher, uniform fusion is a trivial optimum. We therefore narrow
the claims (see R-LAYQ W4). New experiment: corruption-augmented training restores
functional gating. A 1.9K-parameter fusion head assigns weight 0.037 to a zeroed view and
degrades only +1–5% MPJPE (mean fusion: +18%; excluding-the-view oracle: ±0%); with
naturally missing views (NaN) the weight is exactly 0. An 8K residual-correction head
further improves clean MPJPE by −14% (0.128 vs 0.149, 5-fold cross-subject).

**W2 (inert temporal/velocity modules).** Confirmed by the corrected ablation; claims
narrowed accordingly. In the corruption-augmented family, the temporal head does matter:
removing it cancels the learned advantage (0.872 vs mean 0.865 on the uncalibrated
reference; consistent on the calibrated one).

**W3 (absolute magnitude).** The 0.412 m scale was dominated by defect (iv) — an
uncalibrated-extrinsics metric-scale error in the reference (median 2.4×) — not by pose
error. On the calibrated reference the system reaches 0.186 m MPJPE / 0.060 m PA over 52
joints including both hands (hands dominate at 0.316; torso 0.067). On Drive&Act
(below) the same pipeline yields 23 mm-level PA, directly comparable to the SOTA numbers
you cite. We also add PCK@0.05/0.02 as you suggest.

**"Mandatory Evaluation on Drive&Act" — done (new experiment).** Zero-shot (no training on
Drive&Act, official reference, different rig): canonicalize+mean 23.1 mm PA, full model
23.7 mm — within the 22.6–30.4 mm range of the supervised SOTA you cite; geometric DLT
with the official calibration reaches 32.5 mm absolute MPJPE (body-14). With label-free
per-rig adaptation (self-triangulated reference, no manual labels), the 8K residual head
reaches 18.6 mm PA (−23% vs mean fusion). Gating collapse also reproduces on Drive&Act
for the original architecture (0.3333), confirming the architectural diagnosis.

**"Integrate Deep Learning Baselines".** The named methods (UPose3D, Learnable
Triangulation, Faster VoxelPose) consume images or calibrated volumetric features; our
interface is keypoint-only by design — its purpose is to avoid image-level fusion on
embedded hardware — so they are not applicable to the same input, and faithfully
re-implementing three image-based systems is beyond rebuttal scope. What is applicable we
added (new experiment): a family of learned fusers on the identical keypoint input —
confidence-weighted regression in the spirit of Learnable Triangulation's algebraic
weighting (1.9K params), set-attention with an uncertainty head (28K; uncertainty
Spearman 0.72), a residual-correction head (8K; best), and our original architecture
(870K; worst of the family). 16 configurations, 5-fold cross-subject.

**Minor.** SAM3D details (architecture, training domain, NIR-domain bias discussion) added;
pseudo-GT circularity addressed via calibration + external Drive&Act validation;
PCK@0.05/0.02 added.

---

## 阶段三 B：One-page ACCV rebuttal（英文，可直接排版）

最终稿见 `accv2026_rebuttal_official/rebuttal.tex`（官方 ACCV 模板，v2.3，实测编译 1 页；
需 `pdflatex` 三遍行号才正确）。早期单栏自制版 `accv2026_rebuttal_onepage.tex` 已作废。

---

## Confidential Comments to AC：判断

**不建议填写 Confidential Comments to AC。**

逐条对照四个触发条件：
1. 评审明显误读且正文难以解释？——没有。三人核心指控全部属实；唯一的小误读
   （xS2o 把塌缩归因于熵正则）在正文一句话即可澄清。
2. 评审意见与分数严重不一致？——没有。三份 2 分与其列出的缺陷相称。
3. 明显超出 scope 的要求？——xS2o 的图像级基线要求接近边界，但正文的
   模态不匹配论证足以应对，不需要也不宜升级到 AC。
4. 程序性/公平性问题？——没有。

此外主动向 AC 单独披露"我们修了四个缺陷"反而会把叙事焦点从"修正后的强结果"
移到"原稿有多错"，没有收益。所有披露都放在对全体评审的公开首段，语气与
正文一致即可。

---

## 数字速查（写作时防错）

| 用途 | 数字 | 来源 |
|---|---|---|
| 合成数据 Procrustes 验证 | 旧 0.7–2.4 → 新 0.04（噪声水平） | review_fix_plan §一 |
| 修正后 canonicalized mean（原参考，88 序列） | 0.971 / 0.564 | §二·六 |
| 标定参考 fold-0 val：模型 vs mean | 0.1857/0.0593 vs 0.1858/0.0595 | results_summary §1 |
| 关节分组（标定参考） | head 0.101 / 肩颈 0.033 / body 0.067 / hands 0.316 | results_summary §1 |
| 参考标定改善 | rpe 9.60→3.31 px；肩宽 std 0.369→0.015 m；front 8%→45% | §二·八 |
| 尺度错误幅度 | 中位 2.38×（1.24–5.99×） | §二·八 |
| per-joint 有效率 | 头/肩颈 1.00，手 0.34 | F 点统计 |
| 塌缩证据 | λ=0 / LOO 重训 gate 均 0.3333；D&A 上复现 | §二·五 + D&A |
| 损坏增强 1.9K | 置零视角权重 0.037；MPJPE +1–5% vs mean +18%；NaN 时权重 0 | results_summary §3 |
| 残差头 8K | 5 折 0.1284/0.0402（−14.3% vs mean 0.1486/0.0440） | results_summary §2 |
| SetFusion 不确定性 | Spearman 0.72 | results_summary §2 |
| D&A 零样本 | canon+mean 23.1 mm PA；full ckpt 23.7 mm；SOTA 引用区间 22.6–30.4 mm | results_summary §4 |
| D&A DLT 三角化 | 绝对 body14 32.5 mm | results_summary §6 |
| D&A 机位内自适应 | 残差头 18.6 mm PA（−23%） | results_summary §7 |
| 轻量数字 | 0.87M 参数 / 3.33 MB / 6.2 ms/帧 CPU 单线程 | G 点 |
| 去时序消融 | 0.872 ≈ mean 0.865（旧参考） | notemporal tag |
