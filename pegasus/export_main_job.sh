#!/bin/bash
#PBS -A HP260146
#PBS -q gen_S
#PBS -b 1
#PBS -l gpunum_job=1
#PBS -l elapstim_req=01:30:00
#PBS -N tpf_export
#PBS -o /dev/null
#PBS -e /dev/null

# Export the paper's main model (5-fold SetFusion 27.9K, equal-budget training, tag
# eqb_setfusion) into the downstream keypoint files as fused_main / uncertainty_main /
# gate_main / fold_main.  Every sequence is fused by the fold model that never saw its
# subject.  Existing fields are kept; files are rewritten atomically.
#
#   qsub pegasus/export_main_job.sh

REPO="${TPF_REPO:-/work/1/SKIING/chenkaixu/code/TriFusion/.claude/worktrees/hybrid-fusion}"
mkdir -p "${REPO}/logs/pegasus"
exec > "${REPO}/logs/pegasus/export_main.log" 2>&1

set +u
source activate /home/SKIING/chenkaixu/miniconda3/envs/sam_3d_body
set -u
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
cd "${REPO}" || exit 1
echo "export on $(hostname) at $(date)"
python TriPoseFusion/eval/export_fused_keypoints.py \
    --model-root TriPoseFusion/eval/logs/learned_fusion_experiments/eqb_setfusion \
    --suffix main --threads 8 --device auto || echo "!! export failed"
echo "export finished at $(date)"
