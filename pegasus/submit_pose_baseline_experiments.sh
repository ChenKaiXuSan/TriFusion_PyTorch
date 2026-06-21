#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

jobs=(
  "train_trid_pose_fusion_gate_only.sh"
  "eval_pose_baselines_additional.sh"
)

for job in "${jobs[@]}"; do
  echo "Submitting ${job}"
  qsub "${SCRIPT_DIR}/${job}"
done
