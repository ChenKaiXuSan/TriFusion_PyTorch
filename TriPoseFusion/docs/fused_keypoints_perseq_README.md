# fused_keypoints — 融合后的 3D 关键点（下游分析的唯一输入）

**2026-11-27 定稿：下游一律使用 `fused_main`。** 旧版本 `fused_v3`（8K 残差模型，09-04 导出）和 `fused_full`
（28K 小训练量，10-21 导出）已从本目录的文件中移除，逐位校验过的副本在 `../_archive/`。

88 个序列（22 受试者 × 4 环境），701,565 帧，29.97 fps。文件名 `<受试者>_<环境>.npz`，环境 ∈ {昼多い, 昼少ない, 夜多い, 夜少ない}。

## 字段

| 字段 | 形状 | 说明 |
|---|---|---|
| **`fused_main`** | (T,52,3) float32 | **主模型的融合关键点（下游请用这个）**，米，canonical 身体系 |
| `uncertainty_main` | (T,52) float32 | 逐关节 Laplace 尺度 b（米），越大越不确定；与实际误差的 Spearman 相关 0.70 |
| `gate_main` | (T,52,3) float32 | 模型分给三个视角的权重，顺序同 `cameras` |
| `fold_main` | int | 生成该序列所用的折模型（该受试者不在其训练集中） |
| `fused_mean` | (T,52,3) float32 | 三视角 canonical 均值融合（免训练对照） |
| `fused_median` | (T,52,3) float32 | 坐标中位数融合（免训练对照） |
| `view_pose` | (T,52,3,3) float32 | 三个视角各自 canonicalize 后的 SAM3D 关键点 (T,J,V,xyz) |
| `view_conf` | (T,52,3) float32 | SAM3D 逐关节置信度 |
| `gt_pose` / `gt_valid` | (T,52,3) / (T,52) | 三角化参考（逐序列自标定 + 尺度锚定，重投影误差 3.31 px）与有效掩码 |
| `frame_ids` | (T,) str | 原视频帧号（三视角公共帧） |
| `cameras` | (3,) str | 视角名（front / left / right） |

## 必读约定

- **坐标与单位**：坐标以**米**存储，canonical 身体系——颈部为原点，肩轴为 x，肩中点−颈为 down，逐帧旋转。
  不是相机/世界坐标；需要世界坐标请用 `data/drive/sam3d_body_triangulated_gt_perseq/`（70 关节，世界系）。
  （论文与结果表里的**误差**统一以 mm 报告，见代码仓库 `TriPoseFusion/docs/results_summary_2026-11-27.md`。）
- **关节集（52）**：索引 0–4 鼻/左眼/右眼/左耳/右耳，5–6 左右肩，7–27 右手 21 点（含腕），
  28–48 左手 21 点（含腕），49–51 左肩峰/右肩峰/颈。**无肘、髋、躯干。**
- **NaN**：某关节三个视角都无检出时 `fused_main` 为 NaN（`fused_mean` 同）。
- **尺度**：SAM3D 度量尺度（肩宽中位约 0.37 m）；跨人比较建议按肩宽归一化。
- **精度**（5 折跨受试者，对三角化参考）：`fused_main` MPJPE 129.1 mm / PA-MPJPE 40.5 mm；均值融合 148.6 / 44.0 mm。
  分组：头 73.2 mm、肩颈 22.5 mm、手 201.6 mm（均值融合 81.2 / 22.0 / 233.5）。逐序列自检见 `main_export_manifest.json`
  （与训练记录逐段一致）。
- **头部朝向**（88 段，对 SAM3D 头部旋转矩阵核对）：水平角（yaw）相关 0.843，全部 88 段可信，转头幅度约为均值融合的 1.2 倍（核对信号含躯干转动，绝对幅度无法判定，见 `head_movement_analysis/README.md` §6.5）；
  **垂直角（pitch）相关只有 0.40，逐帧抖动大，俯仰类指标不建议用于结论。**
- **帧范围注意**：`frame_ids` 的范围取自标注的 start/end，而标注帧号实为 24 fps（视频 29.97 fps）。
  因此关键点只覆盖每段真实驾驶区间的前约 77%，并在开头多出约 2 秒驾驶前的帧。
  真实驾驶区间 = 标注帧号 × 1.249；最后约 20% 目前没有 SAM3D 推理结果。
- **受试者划分**：与 `../index_mapping/fold_{0..4}.json` 一致（按受试者分折），下游建模请沿用。

## 生成方式

```bash
# 代码仓库 TriFusion，分支 hybrid-fusion；Pegasus 作业：qsub pegasus/export_main_job.sh
python TriPoseFusion/eval/export_fused_keypoints.py \
    --model-root TriPoseFusion/eval/logs/learned_fusion_experiments/eqb_setfusion --suffix main --device auto
```

模型：`eqb_setfusion/fold{0..4}/model.pt`——SetFusion（d = 32，27.9K 参数；视角集合注意力门控 + 关节注意力 + 残差修正 +
Laplace 不确定性头），6000 步 × batch 32 × 64 帧窗口，损坏增强 p = 0.5，逐序列自标定三角化参考监督。
均值/中位数融合 = `view_pose` 沿视角维的均值/中位数。

## 旧版本

| 字段 | 模型 | 导出 | 现在的位置 |
|---|---|---|---|
| `fused_full`（及 `uncertainty_/gate_/fold_full`） | 同结构，小训练量 3000 步 × 16 × 32 帧 | 2026-10-21 | `../_archive/fused_keypoints_full_20261021/` |
| `fused_v3`（及 `uncertainty_/gate_/fold_v3`） | 8K 残差模型 | 2026-09-04 | `../_archive/fused_keypoints_v3_20260904/` |

归档文件与本目录同名、同帧顺序；需要时把对应数组加回即可。三个版本的对比见 `../_archive/README.md`。
