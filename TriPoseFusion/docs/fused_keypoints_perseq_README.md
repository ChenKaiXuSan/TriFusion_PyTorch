# fused_keypoints_perseq — 融合后的 3D 关键点（供下游分析）

生成日期：2026-09-04。88 个序列（22 受试者 × 4 光照环境），701,565 帧，29.97 fps。
文件名 `<受试者>_<环境>.npz`，环境 ∈ {昼多い, 昼少ない, 夜多い, 夜少ない}。

## 字段

| 字段 | 形状 | 说明 |
|---|---|---|
| `fused_v3` | (T,52,3) float32 | **TriPoseFusion v3 融合关键点**（推荐使用），米，canonical 身体系 |
| `uncertainty_v3` | (T,52) float32 | v3 逐关节 Laplace 尺度 b（米），越大越不确定；训练校准 Spearman(b, 误差)=0.72 |
| `gate_v3` | (T,52,3) float32 | v3 学到的视角权重，顺序同 `cameras` |
| `fold_v3` | int | 生成该序列所用的折模型（该受试者不在其训练集中） |
| `fused_mean` | (T,52,3) float32 | 三视角 canonical 均值融合（= 投稿模型的等价输出，Tab.1A） |
| `fused_median` | (T,52,3) float32 | 坐标中位数融合 |
| `view_pose` | (T,52,3,3) float32 | 三个视角各自 canonicalize 后的 SAM3D 关键点 (T,J,V,xyz) |
| `view_conf` | (T,52,3) float32 | SAM3D 逐关节置信度 |
| `gt_pose` / `gt_valid` | (T,52,3) / (T,52) | v3 三角化参考（逐序列自标定+尺度锚定，rpe 3.31 px）与有效掩码 |
| `frame_ids` | (T,) str | 原视频帧号（三视角公共帧） |
| `cameras` | (3,) str | 视角名（front / left / right） |

## 必读约定

- **坐标系**：canonical 身体系——颈部为原点，肩轴为 x，肩中点−颈为 down，逐帧旋转。
  不是相机/世界坐标；需要世界坐标请用 `../sam3d_body_triangulated_gt_perseq/`（70 关节，世界系）。
- **关节集（52）**：索引 0–4 鼻/左眼/右眼/左耳/右耳，5–6 左右肩，7–27 右手 21 点（含腕），
  28–48 左手 21 点（含腕），49–51 左肩峰/右肩峰/颈。**无肘、髋、躯干。**
- **NaN**：某关节三个视角都无检出时 `fused_v3` 为 NaN（`fused_mean` 同）。
- **尺度**：SAM3D 度量尺度（肩宽中位约 0.37 m）；跨人比较建议按肩宽归一化。
- **精度参考**（5 折跨受试者，v3 参考上）：v3 MPJPE 0.129 m / PA 0.040 m；均值融合 0.149 / 0.044；
  手部误差约为躯干的 4 倍（v3 手部 0.20 m）。逐序列自检见 `v3_export_manifest.json`。
- **受试者划分**：与 `../index_mapping/fold_{0..4}.json` 一致（GroupKFold），下游建模请沿用。

## 生成方式

```bash
# 均值/中位数融合 + 输入缓存整理（来自 learned_fusion_cache_perseq）
python -c "..."  # 见 TriPoseFusion/docs 记录；等价于 view_pose.mean(axis=2)
# v3 融合（rebuttal-work 分支）
python TriPoseFusion/eval/export_v3_fused_keypoints.py --threads 4
```

模型：`learned_fusion_experiments/perseq_resid64_aug_unc/fold{0..4}/model.pt`
（ResidualFusion: hidden 32, joint_emb 8, temporal k=5, residual 64, Laplace 不确定性头，
损坏增强 p=0.5，6000 步，v3 伪 GT 监督）。
