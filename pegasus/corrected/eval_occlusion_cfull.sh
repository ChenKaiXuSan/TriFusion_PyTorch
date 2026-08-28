#!/bin/bash
#PBS -A SKIING
#PBS -q gpu
#PBS -l elapstim_req=02:00:00
#PBS -N occl_cfull
#PBS -o /work/SKIING/chenkaixu/code/TriFusion_fast/logs/pegasus/occl_cfull.out
#PBS -e /work/SKIING/chenkaixu/code/TriFusion_fast/logs/pegasus/occl_cfull.err
# 修正重训 full ckpt 的全量单视角遮挡压力测试（C 点 "before" 数据）。登录节点长时间进程会被 watchdog 杀，故作为作业提交。
PROJECT_DIR=${PROJECT_DIR:-/work/SKIING/chenkaixu/code/TriFusion_fast}
cd "${PROJECT_DIR}"
set +u; source activate /home/SKIING/chenkaixu/miniconda3/envs/direction; set -u
export PYTHONPATH="${PROJECT_DIR}/TriPoseFusion:${PROJECT_DIR}:${PYTHONPATH:-}"
CKPT=${CKPT:-/work/SKIING/chenkaixu/code/TriFusion_corrected/logs/train/ctrifusion_full_viewsfront_left_right_16f_fold0/2026-08-28/19-08-58/checkpoints/fold_0/0-0.79.ckpt}
TAG=${TAG:-cfull}
python TriPoseFusion/eval/occlusion_stress_test.py \
  paths.root_path=/work/SKIING/chenkaixu/data/drive \
  paths.sam3d_results_path=/work/SKIING/chenkaixu/data/drive/sam3d_body_results_right \
  paths.kpt_cache_root=/work/SKIING/chenkaixu/data/drive/sam3d_kpt_cache \
  model.geofusion_use_robust_canonicalization=true model.geofusion_gate_entropy_reg_lambda=0.0 \
  eval.ckpt_path="${CKPT}" \
  +eval.triangulated_gt_root=/work/SKIING/chenkaixu/data/drive/sam3d_body_triangulated_gt \
  eval.fold=0 eval.split=val +eval.stress_max_batches=0 +eval.occlusion_mode=zero \
  eval.output_dir="${PROJECT_DIR}/logs/occlusion_stress_${TAG}" data.num_workers=8
