#!/bin/bash
#PBS -A SKIING
#PBS -q gpu
#PBS -l elapstim_req=24:00:00
#PBS -N gate_only
#PBS -o /work/SKIING/chenkaixu/code/MultiView_DriverAction_PyTorch/logs/pegasus/trid_gate_only.out
#PBS -e /work/SKIING/chenkaixu/code/MultiView_DriverAction_PyTorch/logs/pegasus/trid_gate_only.err

PROJECT_DIR=/work/SKIING/chenkaixu/code/MultiView_DriverAction_PyTorch
cd "${PROJECT_DIR}"
mkdir -p "${PROJECT_DIR}/logs/pegasus"

set +u
source activate /home/SKIING/chenkaixu/miniconda3/envs/direction
set -u

echo "============================================================"
echo "TriPoseFusion baseline job: gate_only"
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
fold=${FOLD:-0}
experiment_name=gate_only
view_names=${VIEW_NAMES:-'["front","left","right"]'}
view_tag=${view_names//[\"\[\] ]/}
view_tag=${view_tag//,/_}

use_temporal_refiner=false
use_dilated_refiner=false
use_multiscale_velocity=true
gate_entropy_lambda=0.01
use_robust_canonicalization=true
use_cross_view_attention=false
use_learned_gate=true
run_name="trifusion_${experiment_name}_views${view_tag}_${uniform_temporal_subsample_num}f_fold${fold}_temporal${use_temporal_refiner}_msvel${use_multiscale_velocity}_gate${gate_entropy_lambda}_robust${use_robust_canonicalization}_attn${use_cross_view_attention}_learnedgate${use_learned_gate}"

echo "Experiment: ${run_name}"
echo "Fold: ${fold}"
echo "Views: ${view_names}"
echo "Index mapping: ${index_mapping}"
echo "SAM3D path: ${sam3d_results_path}"
echo "Temporal refiner: ${use_temporal_refiner}"
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
  model.geofusion_use_temporal_refiner="${use_temporal_refiner}" \
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
# TriPoseFusion baseline: gate_only
# =============================================================================
# Purpose:
#   Evaluate learned joint-wise view gating after canonicalization, without
#   cross-view attention or temporal refinement.
#
# Setting:
#   - canonicalized 3-view input
#   - learned joint-wise gate enabled
#   - cross-view attention disabled
#   - temporal TCN/refiner disabled, so P_final == P_init
# =============================================================================
