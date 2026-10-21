# 仓库索引（2026-10-21 归一后）

**主线 = IEEE T-ITS 投稿《TriPoseFusion v2: Self-Calibrated Multi-View Fusion of Monocular 3D Keypoints for In-Cabin Driver Pose Estimation》**，分支 `worktree-review-fixes`（→ 合并到 `master`）。ACCV 2026 投稿已结束，材料见 `archive/accv2026/`。

## 1. 论文

| 内容 | 位置 |
|---|---|
| T-ITS 稿（IEEEtran，9 页） | `paper_tits/main.tex`、`main.pdf`、`figures/` |
| 结果总表（三阶段对照、统一 Table 3、16 组学习型融合、Drive&Act 四项） | `TriPoseFusion/docs/results_summary_2026-08-30.md` |
| v3 决策与实验记录（结构、实验 1–4、下游节） | `TriPoseFusion/docs/v3_decision_log_2026-08-30.md` |
| 修复/发现全记录（§一～二·七 ACCV 期，§二·八 起 v3） | `TriPoseFusion/docs/review_fix_plan.md` |
| ACCV 归档 | `archive/accv2026/`（README 内有说明）+ Overleaf `code/TriFusion_accv/` |

## 2. 代码入口

| 用途 | 脚本 |
|---|---|
| v3 学习型融合全部实验（learned / residual / setfusion / tripose；train / stress / driveandact / summarize） | `TriPoseFusion/eval/learned_fusion_experiments.py` |
| 伪 GT：逐序列自标定 + 尺度锚定 → 三角化 | `traingulation/calibrate_extrinsics.py`、`traingulation/sam3d_kpt_triangulation.py` |
| 训练缓存构建 | `TriPoseFusion/eval/build_learned_fusion_cache.py`（自有）、`build_driveandact_cache.py`（官方 GT 监督）、`build_driveandact_cache_selftri.py`（label-free 自三角化） |
| Drive&Act 评测（免训练基线 / DLT 三角化 / 模型零样本） | `TriPoseFusion/eval/eval_driveandact_{baselines,triangulation,model}.py` |
| 统一 Table 3（v3 GT） | `TriPoseFusion/eval/eval_table3_unified.py` |
| 融合关键点导出（任意模型） | `TriPoseFusion/eval/export_fused_keypoints.py` |
| 下游：头部朝向测量、七来源对比、yaw 曲线图 | `analysis/head_movement/{run_head_movement_analysis,compare_sources,plot_yaw_trace}.py` |
| 原 v1 Lightning 训练（历史） | `TriPoseFusion/train.py` + `pegasus/` |

## 3. 结果日志（git 内）

- `TriPoseFusion/eval/logs/learned_fusion_experiments/<tag>/fold*/{result.json,model.pt,stress_*.json,driveandact.json}` — v3 全部实验（36 个 tag，见 results_summary）
- `TriPoseFusion/eval/logs/table3_unified_perseqGT/`、`perseqGT_trifusion_full_oldckpt/`、`driveandact_*` — v3 GT 口径评测
- `TriPoseFusion/docs/results_perseqGT/`、`docs/driveandact_results/` — 入稿用汇总
- `archive/accv2026/logs_oldGT/` — 旧 GT 时期（仅对照）

## 4. 数据（`/work/1/SKIING/chenkaixu/data/drive/`，详见该目录 README.md）

| 用途 | 目录 |
|---|---|
| **v3 伪 GT（正确）** | `sam3d_body_triangulated_gt_perseq/` |
| 训练缓存（v3） | `learned_fusion_cache_perseq/`、`sam3d_kpt_cache/`、`index_mapping*/` |
| Drive&Act | `driveandact/`（原始 + `tripose_eval/` 1 fps + `tripose_eval_dense/` 30 fps + `pipeline/`）、`learned_fusion_cache_driveandact{,_selftri}/`、`driveandact_index_mapping*/` |
| 下游分析（融合关键点 + 头部运动） | `downstream_analysis/` |
| v1 ckpt 与 ACCV 重训日志 | `checkpoints/v1_accv_*` |
| 过期（v1/v2 GT、v1 缓存） | `_legacy/` |
