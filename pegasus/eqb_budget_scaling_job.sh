#!/bin/bash
#PBS -A HP260146
#PBS -q gen_S
#PBS -b 1
#PBS -l gpunum_job=1
#PBS -l elapstim_req=02:30:00
#PBS -N tpf_scale
#PBS -t 0-4
#PBS -o /dev/null
#PBS -e /dev/null

# Budget-scaling study: does training longer than the common 6000-step budget still help?
# One sub-request = one fold.  The proposed model (set fusion, 27.9K) and the residual-only
# variant (8K) are trained for 24000 steps (4x the common budget, ~92 passes over the
# training frames) with validation every 2000 steps, so result.json["history"] gives the
# error-vs-steps curve.  GPU utilisation is sampled every 5 s per model into
# logs/pegasus/scale_gpu_<model>_fold<k>.csv (compute nodes cannot be reached by ssh).
#
#   qsub pegasus/eqb_budget_scaling_job.sh

REPO="${TPF_REPO:-/work/1/SKIING/chenkaixu/code/TriFusion/.claude/worktrees/hybrid-fusion}"
K="${PBS_SUBREQNO:-0}"
mkdir -p "${REPO}/logs/pegasus"
exec > "${REPO}/logs/pegasus/scale_fold${K}.log" 2>&1

set +u
source activate /home/SKIING/chenkaixu/miniconda3/envs/sam_3d_body
set -u
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
cd "${REPO}" || exit 1
S=TriPoseFusion/eval/learned_fusion_experiments.py
BUDGET="--steps 24000 --batch 32 --window 64 --augment-prob 0.5 --eval-every 2000"
echo "fold ${K} on $(hostname) at $(date)"

run_one() {  # $1 = name, rest = model args
    local name="$1"; shift
    local A="--tag eqb24k_${name} --threads 8 $*"
    nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 5 \
        > "logs/pegasus/scale_gpu_${name}_fold${K}.csv" &
    local mon=$!
    echo "===== eqb24k_${name} fold ${K} start $(date +%H:%M:%S)"
    python ${S} ${A} train --folds "${K}" ${BUDGET} || echo "!! train failed: ${name}"
    kill "${mon}" 2>/dev/null
    python ${S} ${A} driveandact --fold "${K}" | grep -E '^learned +PA' || echo "!! driveandact failed: ${name}"
    echo "===== eqb24k_${name} fold ${K} done $(date +%H:%M:%S)"
}

run_one resid64   --residual-hidden 64 --residual-uncertainty
run_one setfusion --model setfusion --hidden 32
echo "fold ${K} finished at $(date)"
