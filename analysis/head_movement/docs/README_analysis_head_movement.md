# 头部运动分析结果（STEP1：全走行区间）— 使用说明

> **本分析给出两套互补的"头部运动"指标，请勿混用：**
> - **AI 物理转头**（列名以 `ai_` 开头）：由关节点的头部 5 点（鼻、双眼、双耳）算出头部朝向角，
>   水平角 |yaw| > 15°（垂直角 |pitch| > 10°）且持续 ≥ 0.2 秒记为一次转头。这是"头真的转了多少度"。
> - **人工瞥视事件**（列名不带前缀）：3 位标注者在视频上标出的"往哪里看"事件。经核对它们大多是
>   **小幅瞥视**（伴随的头部转动中位数仅约 8°，只有约 3 成达到 15°），并且漏掉了一部分明显的转头。
>   它反映的是视线/注意去向，不等于头部转动角度。
>
> 假说 1–3 讨论的"頭部運動による補償行動"应以 **AI 物理转头** 为主指标，人工瞥视作为补充。

本文件夹是"驾驶中头部回转"分析的结果。所有表格都是 CSV（可直接用 Excel 打开），
图是 PNG。不需要任何编程知识即可阅读；需要复算时脚本在代码仓库
`analysis/head_movement/run_head_movement_analysis.py`。

## 数据是怎么来的（一段话说明）

每位受试者在驾驶模拟器里开了 4 段（白天/夜间 × 光照多/少），三台摄像头同步拍摄。
我们用 AI 从三路视频估计出驾驶者的 3D 关节位置并融合（`fused_keypoints_perseq/` 中的
`fused_v3`），再由鼻尖和双耳的位置算出**头部朝向角**，从角度曲线中自动找出每一次转头。
另外，3 位标注者独立观看视频，在时间轴上标出了"往哪里看"的瞥视事件（左/右/上/下及斜向，
三人中至少两人一致才算一次）。

AI 角度的可信度用一个独立信号做了核对：SAM3D 模型对每个视角输出的头部旋转矩阵（不依赖
鼻/耳是否可见，三视角转到座舱系后平均）。全部 88 段的核对结果（主表 `yaw_keypoint_vs_rotation_corr` 列）：

| 关节点来源 | 水平角与核对信号的相关（平均 / 最低） | 相关 <0.5 的段数 |
|---|---|---|
| **fused_v3** | **0.86 / 0.56** | **0** |
| fused_mean | 0.86 / 0.50 | 1 |
| fused_median | 0.83 / 0.44 | 1 |
| 单视角 front | 0.76 / 0.16 | 4 |
| 单视角 left | 0.83 / 0.52 | 0 |
| 单视角 right | 0.80 / 0.21 | 3 |

结论：融合关节点的头部水平角在所有段上都可信，且比任一单视角更稳（单视角有个别段几乎失效）。
真转头（>20°）时 15° 阈值的捕获率约 60–90%，静止时误报 <1%。
**垂直角（抬头/低头）与核对信号相关较弱（0.3–0.75），俯仰类结论请谨慎。**

## 目录结构：按"头部关节点来源"分子目录，每个子目录内文件完全相同

```
head_movement_analysis/
├── README.md                    本说明
├── method_comparison.csv / .md  各来源结果对比（总表）
├── figures_method_comparison/   对比图
├── fused_v3/                    ★ 主结果：TriPoseFusion v3 融合关节点（含 report.md 通俗报告）
├── fused_mean/                  三视角均值融合（先统一到 canonical 身体系再取平均）
├── fused_median/                三视角中位数融合
├── view_front/                  仅 front 摄像头的 SAM3D 关节点（经该视角自身 canonicalize）
├── view_left/                   仅 left 摄像头
└── view_right/                  仅 right 摄像头
```

六个来源用**同一套分析流程、同一阈值**，只换头部关节点的来源；均在同一 canonical 身体坐标系中，
数字可直接横比。人工瞥视相关的列在各子目录中完全相同（与关节点来源无关）。

## 每个子目录内的文件

| 文件 | 内容 | 怎么用 |
|---|---|---|
| `summary_by_sequence.csv` | **主表**。88 行 = 22 人 × 4 光照环境，每行一段驾驶 | 群间比较、统计检验直接用这张表 |
| `summary_by_person.csv` | 22 行，每人 4 段的平均值，以及各环境的水平回转频率 | 按人比较时用 |
| `summary_by_environment.csv` | 4 行，各光照环境的平均值 | 看光照影响 |
| `environment_tests.csv` | 光照环境是否有显著影响的检验结果（p 值） | 见下文"怎么读 p 值" |
| `annotator_agreement.csv` | 3 位标注者之间的一致性 | 说明人工标注可靠程度 |
| `detection_accuracy.csv` | 用 AI 角度自动判定"转头"与人工标注的吻合度 | 说明 AI 角度可信到什么程度 |
| `per_sequence/<人>_<环境>.json` | 每段驾驶的全部指标（与主表同内容，逐段一个文件） | 查看某一段的细节 |
| `per_sequence/<人>_<环境>_angles.csv` | **逐帧**头部角度 + 人工标签（约 30 帧/秒） | 想看某段的时间曲线时用 |
| `figures/` | 图 1–6，见下 | 汇报用 |
| `analysis_summary.json` | 最优判定阈值、检验结果的机器可读版 | 复算用 |

文件名中的环境：`day_high`=昼多い（白天·光多）、`day_low`=昼少ない、`night_high`=夜多い、`night_low`=夜少ない。
人的编号 01–24（缺 22、23），共 22 人。

## 主表各列的意思（`summary_by_sequence.csv`）

| 列 | 意思 | 单位 |
|---|---|---|
| `drive_minutes` | 这段驾驶的时长（从标注的 start 到 end） | 分钟 |
| **`ai_horizontal_turns` / `ai_vertical_turns`** | **AI 物理转头次数**：水平角 >15°（垂直角 >10°）且持续 ≥0.2 秒 | 次 |
| **`ai_horizontal_turns_per_min` / `ai_vertical_turns_per_min`** | **AI 转头频率** = 次数 ÷ 时长 —— **假说 2 用这个** | 次/分 |
| `ai_left_turns` `ai_right_turns` `ai_up_turns` `ai_down_turns` | AI 转头按方向拆分 | 次 |
| **`ai_horizontal_peak_deg_mean`** / `_max` | 每次 AI 水平转头的峰值角度的平均 / 最大 —— **假说 3 用这个** | 度 |
| `ai_vertical_peak_deg_mean` | AI 垂直转头峰值俯仰角平均（可靠性较弱） | 度 |
| `ai_horizontal_turn_duration_s_mean` | 每次 AI 转头平均持续时间 | 秒 |
| `horizontal_turns` / `vertical_turns` | 人工标注的**瞥视事件**次数（三人多数一致；斜向同时计入水平和垂直） | 次 |
| `horizontal_turns_per_min` / `vertical_turns_per_min` | 瞥视事件频率 | 次/分 |
| `left_turns` `right_turns` `up_turns` `down_turns` | 瞥视事件按方向拆分 | 次 |
| `mean_turn_duration_s` | 每次瞥视事件平均持续时间 | 秒 |
| `ai_turns_overlapping_annotated_glance` | AI 转头中与某个人工瞥视事件在时间上重叠的比例 | 0–1 |
| `annotated_glances_with_head_turn_ge15` | 人工瞥视事件中伴随 ≥15° 头部转动的比例 | 0–1 |
| `yaw_keypoint_vs_rotation_corr` | 关节点角度与独立核对信号的相关系数（越接近 1 越可信；核对信号未生成的段为空） | — |
| `yaw_std_deg` / `pitch_std_deg` | 整段驾驶中头部水平角/俯仰角的波动幅度（标准差） | 度 |
| `yaw_p95_abs_deg` / `pitch_p95_abs_deg` | 整段中 95% 的时间头部偏离正面不超过这个角度 —— **假说 3 用这个**（反映"转得多大"，不受个别异常帧影响） | 度 |
| `horizontal_event_peak_yaw_deg_mean` | 每次人工标注的水平回转中，头部达到的**峰值角度**的平均 —— 假说 3 的另一个指标 | 度 |
| `horizontal_event_peak_yaw_abs_deg_mean` | 同上，但用"相对座舱"的绝对角度（含躯干转动） | 度 |
| `vertical_event_peak_pitch_deg_mean` | 垂直回转的峰值俯仰角平均 | 度 |
| `sign_agreement_horizontal` | AI 判断的转向方向与标注者一致的比例（1.0 = 全部一致）—— 用来确认角度方向没有算反 | 0–1 |
| `kappa_horizontal` / `kappa_vertical` | 3 位标注者的一致性（Fleiss' κ；0.6 以上算相当一致） | — |
| `angle_valid_fraction` | 这段中 AI 能算出角度的帧的比例 | 0–1 |

角度约定：**正面注视 = 0°**，零点取该段驾驶中头部朝向的中位数（即"平时看前方的方向"）。
水平角（yaw）正 = 向驾驶者右侧，负 = 向左；俯仰角（pitch）正 = 抬头，负 = 低头。
`yaw_deg`/`pitch_deg` 是相对躯干的角度（由融合关节点算），`yaw_abs_deg`/`pitch_abs_deg` 是相对座舱的角度（由三角化参考算，包含躯干转动）。

## 怎么读 p 值（`environment_tests.csv`）

`friedman_p_4env`：4 种光照下的差异是否超出偶然；`wilcoxon_p_day_vs_night`、`wilcoxon_p_high_vs_low`：
白天 vs 夜间、光多 vs 光少 的两两比较。**p < 0.05 通常视为"有显著差异"**。这些检验是按人配对的
（同一个人在不同光照下比较），所以不受个体差异影响。

## 图

- 图 1 / 图 2：每人在 4 种光照下的水平 / 垂直回转频率（柱状图）
- 图 3：按光照环境的箱线图（每个点 = 一段驾驶）
- 图 4：回转峰值角度的分布
- 图 5：一段驾驶的头部水平角时间曲线，红/蓝底色 = 标注者标的"右转/左转"——直观看 AI 角度和人工标注是否对得上
- 图 6：自动判定"转头"的准确度（F1，1.0 为完美）随角度阈值和最短持续时间的变化

## 注意事项（重要）

1. **群间比较还不能做**：目前没有受试者"视野障害群/正常群"及障害程度的表。有了这张表，
   把它按 `person` 列合并进主表即可做假说 1–3 的检验。
2. **角度精度**：头部关节点的 AI 误差约 7–10 cm，逐帧角度有噪声（已做 0.3 秒平滑），
   所以转头判定用了"角度阈值 + 最短持续时间"，并用独立信号核对过（见上文）。
   水平角可信，垂直角较弱。`detection_accuracy.csv` / 图 6 显示的是 AI 转头与**人工瞥视**的
   重叠度——两者本来就是不同的量，重叠度低不代表 AI 角度不准。
3. 4 种光照环境是同一批人重复测量，做统计时必须按人配对（本分析已这样做）。
4. 这里只做了 STEP1（全走行区间）。STEP2（红灯/急刹/U 转等事件）需要事件发生时刻表和
   各目标区域（信号机/前车/后视镜）的方向表，目前没有。
