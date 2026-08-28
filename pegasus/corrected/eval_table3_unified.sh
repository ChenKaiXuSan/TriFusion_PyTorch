#!/bin/bash
#PBS -A SKIING
#PBS -q gpu
#PBS -l elapstim_req=03:00:00
#PBS -N tbl3_uni
#PBS -o /work/SKIING/chenkaixu/code/TriFusion_fast/logs/pegasus/table3_unified.out
#PBS -e /work/SKIING/chenkaixu/code/TriFusion_fast/logs/pegasus/table3_unified.err
# Table 3 统一口径：所有行走 eval_trifusion_pesudo_gt 的同一度量路径（fold-0 val，robust canonicalizer）。
PROJECT_DIR=${PROJECT_DIR:-/work/SKIING/chenkaixu/code/TriFusion_fast}
cd "${PROJECT_DIR}"
set +u; source activate /home/SKIING/chenkaixu/miniconda3/envs/direction; set -u
export PYTHONPATH="${PROJECT_DIR}/TriPoseFusion:${PROJECT_DIR}:${PYTHONPATH:-}"
CKPT=${CKPT:-/work/SKIING/chenkaixu/code/TriFusion_corrected/logs/train/ctrifusion_full_viewsfront_left_right_16f_fold0/2026-08-28/19-08-58/checkpoints/fold_0/0-0.79.ckpt}
# 学习型基线权重未入库，直接引用 rebuttal-work worktree 下的文件
LEARNED=${LEARNED:-/work/SKIING/chenkaixu/code/TriFusion/.claude/worktrees/rebuttal-work/TriPoseFusion/eval/logs/learned_fusion_baseline/learned_fusion_fold0.pt}
OUT=${OUT:-${PROJECT_DIR}/logs/table3_unified_fold0}
COMMON="paths.root_path=/work/SKIING/chenkaixu/data/drive paths.sam3d_results_path=/work/SKIING/chenkaixu/data/drive/sam3d_body_results_right paths.kpt_cache_root=/work/SKIING/chenkaixu/data/drive/sam3d_kpt_cache model.geofusion_use_robust_canonicalization=true model.geofusion_gate_entropy_reg_lambda=0.0 +eval.triangulated_gt_root=/work/SKIING/chenkaixu/data/drive/sam3d_body_triangulated_gt eval.fold=0 eval.split=val data.num_workers=8 eval.ckpt_path=${CKPT} eval.output_dir=${OUT}"
for m in model mean median single_front single_left single_right learned; do
  echo "=== method=$m $(date +%H:%M)"
  python TriPoseFusion/eval/eval_table3_unified.py ${COMMON} +eval.method=$m +eval.learned_ckpt=${LEARNED} 2>&1 | grep -E "UNIFIED|Traceback|Error"
done
python TriPoseFusion/eval/table3_unified_summary.py --root ${OUT} --fold 0
echo ALL_DONE $(date +%H:%M)
