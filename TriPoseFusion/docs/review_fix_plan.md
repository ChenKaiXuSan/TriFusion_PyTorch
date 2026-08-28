# 评审回应与修改计划（2026-08-28）

三份评审（LAYQ / 19jf / xS2o，均 Weak Reject，置信度 4）的核心技术指控经代码核实
**基本属实**。本文档整理已完成的代码修复、需要重跑/重训的实验，以及 rebuttal 中
各评审点的回应策略。

## 一、已核实并修复的三个 bug（本分支 `worktree-review-fixes`）

### 1. Procrustes 对齐使用了转置旋转（影响最大）

`eval_fusion_baselines_pesudo_gt.py` / `eval_single_sam3d_pesudo_gt.py` /
`eval_trifusion_pesudo_gt.py` 三处副本都把行向量约定下的最优旋转 `U @ Vt`
写成了它的转置 `Vt.T @ U.T`，且两个 numpy 副本还使用范数配比 scale
（`‖tgt‖/‖src‖`）而非最小二乘最优 scale。

数值验证（对"GT 经旋转+缩放+噪声"构造的数据，正确 PA 误差应 ≈ 噪声水平）：

| 旋转角 | 未对齐 | 旧实现 PA | 修复后 PA |
|---|---|---|---|
| 0.3 rad | 1.291 | 0.719 | 0.044 |
| 1.2 rad | 3.203 | 2.431 | 0.049 |
| 2.5 rad | 3.306 | 1.477 | 0.043 |

结论：**论文中所有 PA-MPJPE（含主结果 0.275 m）都被系统性高估**，修复后只会
变好。这同时直接回答 LAYQ 的第一大条（Table 3 canonicalized 基线
PA-MPJPE 1.664 > MPJPE 0.969 的反常）：转置旋转 + 非最优 scale + 塌缩帧
（见 bug 3）共同造成。

### 2. 门控熵正则是零梯度死代码

`trainer/train_triple_fusion.py` 原代码 `alpha.log().clamp_min(1e-6)` 把 clamp
加在 log 之后（α<1 时 log 全负，全部被抬到 1e-6），熵恒为 0、损失恒为
log V（训练日志中 loss_gate_entropy 恒为 1.0986=ln3 / 0.6931=ln2 即为此），
梯度为零。已抽出 `gate_entropy_loss()` 修复为 `alpha.clamp_min(1e-6).log()`。

对 xS2o "gating collapse" 的回应要点：塌缩**不是**熵正则强迫的（正则根本没
生效）；旁证是关掉 cross-view attention 的 run 学出了非均匀 gate
(0.28/0.34/0.38)。但"adaptive gating"的主张确实需要收窄或用重训+遮挡压力
测试支撑（见三、C）。

### 3. 肩部退化回退产生零旋转、整帧塌缩

`models/keypoint_mlp.py` RobustCanonicalization 在肩宽异常时把左右肩都替换为
neck → x 轴为零向量 → 两次叉积后旋转矩阵为零矩阵 → 整帧 canonicalized pose
塌缩为原点（LAYQ 第二大条猜测完全正确）。已改为：复用同片段内时间上最近
有效帧的肩轴（前向填充+前缀回填），整段无效时用默认单位侧向轴；并对
肩轴与躯干轴平行的叉积退化加参考轴兜底。eval 侧 numpy 版 `canonicalize_pose`
同样修复，并统一为与模型一致的 `x = right - left` 约定。

### 配套改动

- `compute_metrics` 新增 `mpjpe_pa_frames_m`（与 PA-MPJPE 完全同掩码的未对齐
  MPJPE），回应 LAYQ 对 mask/聚合协议的质疑。
- 回归测试 `tests/test_review_metric_fixes.py`（11 用例，全部通过）。

## 二、需要重跑的实验

本机数据根：`/work/1/SKIING/chenkaixu/data/drive/`（配置里的
`/home/data/xchen/drive` 是 bnode105/docker 路径）。Python 用
`~/miniconda3/envs/asd/bin/python`。

1. **伪 GT 重生成**（本机可跑，2026-08-28 已启动）：

   ```bash
   python traingulation/sam3d_kpt_triangulation.py \
     --config traingulation/triangulation.yaml \
     --input-root /work/1/SKIING/chenkaixu/data/drive/sam3d_body_results_right \
     --output-root /work/1/SKIING/chenkaixu/data/drive/sam3d_body_triangulated_gt \
     --num-workers 16 --no-progress
   ```

2. **Table 3 基线行重跑**（本机 CPU 即可，修复后的代码）：

   ```bash
   python TriPoseFusion/eval/eval_fusion_baselines_pesudo_gt.py \
     --sam3d-root /work/1/SKIING/chenkaixu/data/drive/sam3d_body_results_right \
     --gt-root /work/1/SKIING/chenkaixu/data/drive/sam3d_body_triangulated_gt \
     --output-dir TriPoseFusion/eval/logs/corrected_fusion_baselines
   python TriPoseFusion/eval/eval_single_sam3d_pesudo_gt.py --sam3d-root ... --gt-root ... \
     --output-dir TriPoseFusion/eval/logs/corrected_single_sam3d
   python TriPoseFusion/eval/eval_additional_baselines_pseudo_gt.py  # 同参数
   ```

3. **模型行重评**：老 checkpoint 的 MPJPE 不变，PA-MPJPE 会因 Procrustes 修复
   下降；可先用 `eval_trifusion_pesudo_gt.py` + 现有 ckpt 重评拿到修正数字。

4. **Table 4 重训**（必须 GPU 机器，pegasus 脚本）：canonicalizer 修复改变了
   模型输入处理，严格起见全部 ablation 需用修复后代码重训。优先级：
   `full`、`uniform_gate`、`base_simple`、`no_cross_view_attention`。

## 二·五、结论更新（2026-08-28 晚，两条独立评估链确认）

- 修复后重训的所有配置（full / base_simple / uniform_gate / gate_lambda0）与旧 ckpt
  在修正评估下全部收敛到 MPJPE ≈0.948 / PA ≈0.346（fold-0 val）。
- 模型输出 ≡ canonicalized **均值融合 + 约 6 mm 的 TCN 残差**（‖P_final − mean‖≈6 mm，
  ‖P_final − median‖≈26 mm），三者在 MPJPE 上不可区分。根因：L_tri 的 teacher 是三视角
  中位数，view/bone/temp 均为自洽项，均匀融合即最优（平凡解）；val loss 的 99% 是 InfoNCE。
- 论文 Table 4 "robust canonicalization 0.846→0.412" 为 bug 假象；本文档早期同子集表中的
  "MPJPE −4.4%"以及"躯干 −12% 最优 / 手部 +17% 最差"均为**跨协议假象**：模型行的 GT 走
  `module.model._canonicalize_pose`，基线行走 numpy `canonicalize_pose`，同协议下 median
  融合四组 = 0.412/0.252/0.332/1.582，与模型行相同。**Table 3 全部行必须走同一条
  canonicalize 路径**（手部对旋转轴差异最敏感，可差 0.23 m）。
- 对 rebuttal 的含义：路线 A（诚实披露 + 收窄为 canonicalization 协议 / 数据集 / 评估修正）
  是唯一站得住的主线；LOO teacher（rebuttal-work 分支）是方法改动，只作附加实验。
- 仍成立的结论：手部主导绝对误差（所有方法手部/躯干 ≈3.5–4.7×）；单视图 2.078 m 是原始相机
  坐标系未 canonicalize 的数字；模型 0.87M 参数 / CPU 6.2 ms/帧。

## 三、rebuttal 各点回应策略

- **A（LAYQ：PA>MPJPE 反常）**：如实说明发现的对齐实现缺陷 + 给修正数字 +
  同掩码指标。这是 LAYQ 明说"could materially change the assessment"的点。
- **B（LAYQ：回退规则不清）**：给出修复后的精确回退算法（时间最近有效帧 →
  默认轴），附伪代码。
- **C（xS2o：gating 塌缩）**：二选一——(i) λ_gate=0 + 修复后代码重训，展示
  gate 能否分化，并补"单视角合成遮挡/置零"压力测试看 gate 响应；(ii) 诚实
  收窄主张为 "canonicalization + uniform fusion"（ablation 里 uniform_gate
  本就是最优 0.7891）。倾向 (i) 失败则落到 (ii)。
- **D（xS2o：绝对误差 0.412m vs SOTA 22-30mm）**：说明不可直接比较——52 关节
  含双手 21+21、伪 GT 噪声下限、SAM3D 单视输入 2.078m 起点；用
  `per_joint_mpjpe` 做躯干/手部误差分解证明主干误差远小。需要重评后出数字。
- **E（xS2o：缺学习型基线）**：在相同 keypoint 输入上补至少一个学习型融合
  基线（如 learnable-triangulation 风格的置信度加权回归）。工作量最大，
  rebuttal 期内能做多少做多少。
- **F（19jf：伪 GT 可靠性/手部遮挡）**：报告伪 GT 的重投影误差分布与
  per-joint 有效率；把手部关节单独列出并在正文承认其参考不确定性；
  可加"仅躯干 31 关节"的备选评估协议。
- **G（轻量主张）**：已测得 0.87M 参数 / 3.33MB fp32 / CPU 单线程 6.2ms/帧
  （99.4ms/16 帧片段，登录节点 CPU），写入论文。
- **H（小项）**：52 关节映射表（SAM3D→triangulated）、运动导数帧率归一化
  说明、GroupKFold 划分与序列时长统计，均为文字/表格补充。
