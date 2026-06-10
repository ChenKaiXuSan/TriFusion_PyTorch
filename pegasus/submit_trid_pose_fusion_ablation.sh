#!/bin/bash

set -euo pipefail

# =============================================================================
# TriPoseFusion 消融实验批量提交脚本
# =============================================================================
# 这个脚本负责把 6 个独立实验脚本提交到 Pegasus。
# 每个实验脚本固定只跑 fold 0。
#
# 默认行为：
#   提交 fold 0 的 6 个 TriPoseFusion 对比实验。
#
# 快速 smoke test：
#   MAX_EPOCHS=5 bash pegasus/submit_trid_pose_fusion_ablation.sh
#
# 可选覆盖参数：
#   MAX_EPOCHS=50       训练 epoch 数
#   NUM_WORKERS=32      DataLoader worker 数量
#   BATCH_SIZE=32       batch size
#   NUM_FRAMES=16       uniform temporal subsample 帧数
#   DEVICES=1           GPU 数量
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
mkdir -p "${PROJECT_DIR}/logs/pegasus"

# === 独立实验脚本列表 ===
# base_simple:
#   关闭所有改进模块，作为最基础 baseline。
# dilated_tcn:
#   只打开 dilated temporal refiner。
# multiscale_velocity:
#   只打开多尺度速度特征。
# gate_entropy:
#   只打开 gate entropy regularization。
# robust_canon:
#   只打开 robust canonicalization。
# full:
#   打开全部改进模块，作为完整方法。
scripts=(
  train_trid_pose_fusion_base_simple.sh
  train_trid_pose_fusion_dilated_tcn.sh
  train_trid_pose_fusion_multiscale_velocity.sh
  train_trid_pose_fusion_gate_entropy.sh
  train_trid_pose_fusion_robust_canon.sh
  train_trid_pose_fusion_full.sh
)

# === 默认提交参数 ===
# 所有实验脚本内部都固定 FOLD=0。
max_epochs=${MAX_EPOCHS:-50}
num_workers=${NUM_WORKERS:-32}
batch_size=${BATCH_SIZE:-32}
num_frames=${NUM_FRAMES:-16}
devices=${DEVICES:-1}

# === 批量提交 PBS job ===
# 每个实验是一个独立 job，避免一个实验超时或失败影响其他实验。
for script_name in "${scripts[@]}"; do
  script_path="${SCRIPT_DIR}/${script_name}"
  echo "Submitting ${script_name}, FOLD=0, MAX_EPOCHS=${max_epochs}"
  qsub -v MAX_EPOCHS="${max_epochs}",NUM_WORKERS="${num_workers}",BATCH_SIZE="${batch_size}",NUM_FRAMES="${num_frames}",DEVICES="${devices}" "${script_path}"
done
