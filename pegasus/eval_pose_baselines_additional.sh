#!/bin/bash
#PBS -A SKIING
#PBS -q gpu
#PBS -l elapstim_req=08:00:00
#PBS -N pose_bases
#PBS -o /work/SKIING/chenkaixu/code/MultiView_DriverAction_PyTorch/logs/pegasus/eval_pose_baselines_additional.out
#PBS -e /work/SKIING/chenkaixu/code/MultiView_DriverAction_PyTorch/logs/pegasus/eval_pose_baselines_additional.err

PROJECT_DIR=/work/SKIING/chenkaixu/code/MultiView_DriverAction_PyTorch
cd "${PROJECT_DIR}"
mkdir -p "${PROJECT_DIR}/logs/pegasus"

set +u
source activate /home/SKIING/chenkaixu/miniconda3/envs/direction
set -u

echo "============================================================"
echo "TriPoseFusion additional pose baseline evaluation"
echo "Project dir: ${PROJECT_DIR}"
echo "Python: $(python --version)"
echo "Python path: $(which python)"
echo "Start time: $(date)"
echo "============================================================"

export PYTHONPATH="${PROJECT_DIR}/TriPoseFusion:${PROJECT_DIR}:${PYTHONPATH:-}"

root_path=/work/SKIING/chenkaixu/data/drive
sam3d_root=${SAM3D_ROOT:-${root_path}/sam3d_body_results_right}
gt_root=${GT_ROOT:-${root_path}/sam3d_body_triangulated_gt}
output_dir=${OUTPUT_DIR:-${PROJECT_DIR}/TriPoseFusion/eval/logs/comparison_additional_baselines_pseudo_gt}
smoothing_window=${SMOOTHING_WINDOW:-5}
num_workers=${NUM_WORKERS:-8}
max_frames_arg=()

if [ -n "${MAX_FRAMES:-}" ]; then
  max_frames_arg=(--max-frames "${MAX_FRAMES}")
fi

python TriPoseFusion/eval/eval_additional_baselines_pseudo_gt.py \
  --sam3d-root "${sam3d_root}" \
  --gt-root "${gt_root}" \
  --output-dir "${output_dir}" \
  --smoothing-window "${smoothing_window}" \
  --num-workers "${num_workers}" \
  "${max_frames_arg[@]}"

echo "============================================================"
echo "Finished additional pose baseline evaluation"
echo "Output dir: ${output_dir}"
echo "End time: $(date)"
echo "============================================================"

# Evaluates:
#   1. Canonicalized single views: front_single, left_single, right_single
#   2. Oracle best_single selected per subject/environment by MPJPE
#   3. Canonicalized fixed fusion: mean, median, confidence
#   4. Canonicalized median + temporal moving-average smoothing
