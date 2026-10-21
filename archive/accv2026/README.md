# ACCV 2026 #802 归档（已结束）

**状态**：三位审稿人均 Weak Reject（LAYQ / 19jf / xS2o，置信度 4），rebuttal 于 2026-09-02 提交，
投稿已结束。本目录只做历史存档，**所有数字已被 T-ITS 主线（`paper_tits/`，v3 伪 GT）取代**。

## 内容

- `rebuttal/official/` — 官方模板版 rebuttal（`rebuttal.tex/.pdf`、`reviews_accv2026_802.txt` 审稿原文）
- `rebuttal/accv2026_rebuttal_onepage.*`、`accv2026_rebuttal_package.md`、`rebuttal_draft.md` — 工作稿
- `logs_oldGT/` — ACCV 时期的评估日志（**均基于 v1 名义外参伪 GT，尺度错 2.4×，仅作对照**）：
  - `corrected_*`：修正度量代码后、旧 GT 上的 Table 3 重评（fusion / additional / single / trifusion_full）
  - `table3_unified_fold0/`：旧 GT 统一口径 Table 3（来自 TriFusion_fast）
  - `learned_fusion_baseline/`：旧 GT 上的 1.9K 学习型基线（其 −12.5% 增益后证实为尺度假象）
  - `loo_runs/`、`ckpt_sweep_*`：留一视角 teacher 与 ckpt 扫描（门控塌缩验证）
  - `occlusion_stress_smoke/`、`occlusion_stress_cfull/`、`protocol_check_median/`、`pseudo_gt_reliability/`
  - `v1_train_EXPERIMENT_SUMMARY.md`：v1 原始训练汇总
- 大型二进制（ckpt、预测 npz）不入 git：`/work/1/SKIING/chenkaixu/data/drive/checkpoints/v1_accv_*`

## 相关位置

- ACCV 投稿正文（LLNCS）+ 补充材料：Overleaf 仓库 `code/TriFusion_accv/`（`main.tex`、`supplementary.tex`、`rebuttal/`）
- 修复与发现的完整记录：`TriPoseFusion/docs/review_fix_plan.md`（§一～二·七 为 ACCV 时期，§二·八 起为 v3 主线）
- 为什么旧数字不可信：`review_fix_plan.md` §二·八 与 T-ITS 稿 Sec. "How Reference-Scale Errors Manufacture Illusory Gains"
