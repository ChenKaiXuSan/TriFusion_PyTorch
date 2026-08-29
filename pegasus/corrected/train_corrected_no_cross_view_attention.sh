#!/bin/bash
#PBS -A SKIING
#PBS -q gpu
#PBS -l elapstim_req=24:00:00
#PBS -N c_nocva
#PBS -o /work/SKIING/chenkaixu/code/TriFusion_corrected/logs/pegasus/no_cross_view_attention.out
#PBS -e /work/SKIING/chenkaixu/code/TriFusion_corrected/logs/pegasus/no_cross_view_attention.err

# 修正版重训（worktree-review-fixes 修复后代码）：canonicalizer 回退 / 门控熵 /
# Procrustes 修复后的 Table 4 重跑。代码 checkout: TriFusion_corrected（分支 rebuttal-work）。

PROJECT_DIR=${PROJECT_DIR:-/work/SKIING/chenkaixu/code/TriFusion_corrected}
cd "${PROJECT_DIR}"
mkdir -p "${PROJECT_DIR}/logs/pegasus"

set +u
source activate /home/SKIING/chenkaixu/miniconda3/envs/direction
set -u

echo "============================================================"
echo "TriPoseFusion corrected retrain: no_cross_view_attention"
echo "Project dir: ${PROJECT_DIR}"
echo "Git commit: $(git rev-parse --short HEAD)"
echo "Python: $(python --version)  ($(which python))"
echo "Start time: $(date)"
echo "============================================================"
nvidia-smi

export PYTHONPATH="${PROJECT_DIR}/TriPoseFusion:${PROJECT_DIR}:${PYTHONPATH:-}"

root_path=/work/SKIING/chenkaixu/data/drive
index_mapping=${root_path}/index_mapping
sam3d_results_path=${root_path}/sam3d_body_results_right

num_workers=${NUM_WORKERS:-16}
batch_size=${BATCH_SIZE:-32}
uniform_temporal_subsample_num=${NUM_FRAMES:-16}
# val/loss 在 epoch 0-1 即最优：默认 12 epoch + patience 3 early stopping（原 50 epoch 无 early stop）
max_epochs=${MAX_EPOCHS:-12}
early_stop_patience=${EARLY_STOP_PATIENCE:-3}
# 关键点数组缓存（dataloader/build_kpt_cache.py 生成）；设为空串则回退逐文件读取
kpt_cache_root=${KPT_CACHE_ROOT:-/work/SKIING/chenkaixu/data/drive/sam3d_kpt_cache}
devices=${DEVICES:-1}
fold=0
experiment_name=no_cross_view_attention
view_names=${VIEW_NAMES:-'["front","left","right"]'}
view_tag=${view_names//[\"\[\] ]/}
view_tag=${view_tag//,/_}

use_dilated_refiner=true
use_multiscale_velocity=true
gate_entropy_lambda=0.01
use_robust_canonicalization=true
use_cross_view_attention=false
use_learned_gate=true
run_name="ctrifusion_${experiment_name}_views${view_tag}_${uniform_temporal_subsample_num}f_fold${fold}"

echo "Experiment: ${run_name}"
echo "dilated=${use_dilated_refiner} msvel=${use_multiscale_velocity} gate_lambda=${gate_entropy_lambda} robust=${use_robust_canonicalization} cva=${use_cross_view_attention} learned_gate=${use_learned_gate}"
echo "max_epochs=${max_epochs} early_stop_patience=${early_stop_patience} kpt_cache_root=${kpt_cache_root:-<none>}"

python TriPoseFusion/train.py \
  paths.root_path="${root_path}" \
  paths.index_mapping="${index_mapping}" \
  paths.sam3d_results_path="${sam3d_results_path}" \
  paths.kpt_cache_root="${kpt_cache_root:-null}" \
  train.early_stop_patience="${early_stop_patience}" \
  data.num_workers="${num_workers}" \
  data.batch_size="${batch_size}" \
  data.uniform_temporal_subsample_num="${uniform_temporal_subsample_num}" \
  model.backbone=triple_fusion \
  model.geofusion_use_dilated_refiner="${use_dilated_refiner}" \
  model.geofusion_use_multiscale_velocity="${use_multiscale_velocity}" \
  model.geofusion_gate_entropy_reg_lambda="${gate_entropy_lambda}" \
  model.geofusion_use_robust_canonicalization="${use_robust_canonicalization}" \
  model.geofusion_use_cross_view_attention="${use_cross_view_attention}" \
  model.geofusion_use_learned_gate="${use_learned_gate}" \
  train.view=multi \
  train.view_name="${view_names}" \
  train.fold="${fold}" \
  train.max_epochs="${max_epochs}" \
  train.devices="${devices}" \
  experiment="${run_name}"

echo "============================================================"
echo "Finished ${run_name}"
echo "End time: $(date)"
echo "============================================================"
