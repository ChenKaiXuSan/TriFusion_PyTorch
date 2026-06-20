#!/bin/bash
#PBS -A SKIING
#PBS -q gpu
#PBS -l elapstim_req=24:00:00
#PBS -N uniform_gate
#PBS -o /work/SKIING/chenkaixu/code/MultiView_DriverAction_PyTorch/logs/pegasus/trid_uniform_gate.out
#PBS -e /work/SKIING/chenkaixu/code/MultiView_DriverAction_PyTorch/logs/pegasus/trid_uniform_gate.err

PROJECT_DIR=/work/SKIING/chenkaixu/code/MultiView_DriverAction_PyTorch
cd "${PROJECT_DIR}"
mkdir -p "${PROJECT_DIR}/logs/pegasus"

set +u
source activate /home/SKIING/chenkaixu/miniconda3/envs/direction
set -u

echo "============================================================"
echo "TriPoseFusion ablation job: uniform_gate"
echo "Project dir: ${PROJECT_DIR}"
echo "Python: $(python --version)"
echo "Python path: $(which python)"
echo "Start time: $(date)"
echo "============================================================"
nvidia-smi
conda env list

export PYTHONPATH="${PROJECT_DIR}/TriPoseFusion:${PROJECT_DIR}:${PYTHONPATH:-}"

root_path=/work/SKIING/chenkaixu/data/drive
index_mapping=${root_path}/index_mapping
sam3d_results_path=/work/SKIING/chenkaixu/data/drive/sam3d_body_results_right

num_workers=${NUM_WORKERS:-32}
batch_size=${BATCH_SIZE:-32}
uniform_temporal_subsample_num=${NUM_FRAMES:-16}
max_epochs=${MAX_EPOCHS:-50}
devices=${DEVICES:-1}
fold=0
experiment_name=uniform_gate
view_names=${VIEW_NAMES:-'["front","left","right"]'}
view_tag=${view_names//[\"\[\] ]/}
view_tag=${view_tag//,/_}

use_dilated_refiner=true
use_multiscale_velocity=true
gate_entropy_lambda=0.0
use_robust_canonicalization=true
use_cross_view_attention=true
use_learned_gate=false
run_name="trifusion_${experiment_name}_views${view_tag}_${uniform_temporal_subsample_num}f_fold${fold}_dilated${use_dilated_refiner}_msvel${use_multiscale_velocity}_gate${gate_entropy_lambda}_robust${use_robust_canonicalization}_attn${use_cross_view_attention}_learnedgate${use_learned_gate}"

echo "Experiment: ${run_name}"
echo "Fold: ${fold}"
echo "Views: ${view_names}"
echo "Index mapping: ${index_mapping}"
echo "SAM3D path: ${sam3d_results_path}"
echo "Dilated refiner: ${use_dilated_refiner}"
echo "Multiscale velocity: ${use_multiscale_velocity}"
echo "Gate entropy lambda: ${gate_entropy_lambda}"
echo "Robust canonicalization: ${use_robust_canonicalization}"
echo "Cross-view attention: ${use_cross_view_attention}"
echo "Learned gate: ${use_learned_gate}"

python TriPoseFusion/train.py \
  paths.root_path="${root_path}" \
  paths.index_mapping="${index_mapping}" \
  paths.sam3d_results_path="${sam3d_results_path}" \
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

# =============================================================================
# TriPoseFusion 消融实验：uniform_gate
# =============================================================================
# 目的：
#   验证 learned joint-wise view gate 是否优于固定均匀视角权重。
#
# 消融设置：
#   - 保持 full model 的 cross-view attention / dilated TCN /
#     multi-scale velocity / robust canonicalization
#   - 关闭 model.geofusion_use_learned_gate
#   - alpha 固定为 1 / num_views
# =============================================================================
