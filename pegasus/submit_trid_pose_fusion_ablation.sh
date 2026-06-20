#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

jobs=(
  train_trid_pose_fusion_base_simple.sh
  train_trid_pose_fusion_dilated_tcn.sh
  train_trid_pose_fusion_multiscale_velocity.sh
  train_trid_pose_fusion_gate_entropy.sh
  train_trid_pose_fusion_robust_canon.sh
  train_trid_pose_fusion_full.sh
  train_trid_pose_fusion_no_cross_view_attention.sh
  train_trid_pose_fusion_uniform_gate.sh
)

echo "Submitting ${#jobs[@]} TriPoseFusion ablation jobs..."
for job in "${jobs[@]}"; do
  echo "qsub ${SCRIPT_DIR}/${job}"
  qsub "${SCRIPT_DIR}/${job}"
done
