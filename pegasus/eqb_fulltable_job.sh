#!/bin/bash
#PBS -A HP260146
#PBS -q gen_S
#PBS -b 1
#PBS -l gpunum_job=1
#PBS -l elapstim_req=04:00:00
#PBS -N tpf_full
#PBS -t 0-3
#PBS -o /dev/null
#PBS -e /dev/null

# Equal-budget main table, parallel-fold version: the five folds of a configuration train
# concurrently on ONE GPU (the models are tiny: ~30% utilisation and ~1 GB per process on
# an H100), so a 4x larger step budget costs about the same wall-clock as the sequential
# 6000-step job (eqb_baselines_job.sh).
#
#   qsub -v EQB_STEPS=24000,EQB_PREFIX=eqb24k pegasus/eqb_fulltable_job.sh
#
# EQB_STEPS   common step budget for every learned row (default 6000)
# EQB_PREFIX  tag prefix, results in .../learned_fusion_experiments/<prefix>_<name> (default eqb)
# Folds with an existing result.json are skipped, so re-submitting is safe.
# Training-free baselines do not depend on the budget: reuse eqb_tf_* from eqb_baselines_job.sh.

REPO="${TPF_REPO:-/work/1/SKIING/chenkaixu/code/TriFusion/.claude/worktrees/hybrid-fusion}"
SUB="${PBS_SUBREQNO:-0}"
STEPS="${EQB_STEPS:-6000}"
PREFIX="${EQB_PREFIX:-eqb}"
LOG="${REPO}/logs/pegasus"
mkdir -p "${LOG}"
exec > "${LOG}/${PREFIX}_full_sub${SUB}.log" 2>&1

set +u
source activate /home/SKIING/chenkaixu/miniconda3/envs/sam_3d_body
set -u
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
cd "${REPO}" || exit 1
S=TriPoseFusion/eval/learned_fusion_experiments.py
OUT=TriPoseFusion/eval/logs/learned_fusion_experiments
BUDGET="--steps ${STEPS} --batch 32 --window 64 --augment-prob 0.5 --eval-every $((STEPS / 6))"

echo "sub-request ${SUB} on $(hostname) at $(date); steps=${STEPS} prefix=${PREFIX} cores=$(nproc)"
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader -l 10 \
    > "${LOG}/${PREFIX}_full_gpu_sub${SUB}.csv" &
GPUMON=$!

run_cfg() {  # $1 = name, rest = model arguments
    local name="$1"; shift
    local tag="${PREFIX}_${name}"
    local A="--tag ${tag} --threads 2 $*"
    echo "===== ${tag} ($*) start $(date +%H:%M:%S)"
    for k in 0 1 2 3 4; do
        if [[ -f "${OUT}/${tag}/fold${k}/result.json" ]]; then
            echo "[fold ${k}] result.json exists, skip training"
        else
            python ${S} ${A} train --folds "${k}" ${BUDGET} > "${LOG}/${tag}_train_fold${k}.log" 2>&1 &
        fi
    done
    wait
    for k in 0 1 2 3 4; do
        [[ -f "${OUT}/${tag}/fold${k}/result.json" ]] || echo "!! train failed: ${tag} fold ${k} (see ${tag}_train_fold${k}.log)"
        grep -h 'learned(last)' "${LOG}/${tag}_train_fold${k}.log" 2>/dev/null
    done
    echo "----- ${tag} trained $(date +%H:%M:%S); stress + Drive&Act"
    for k in 0 1 2 3 4; do
        ( python ${S} ${A} stress --folds "${k}" --mode zero  > /dev/null 2>&1 || echo "!! stress zero failed: ${tag} fold ${k}"
          python ${S} ${A} stress --folds "${k}" --mode noise > /dev/null 2>&1 || echo "!! stress noise failed: ${tag} fold ${k}"
          python ${S} ${A} driveandact --fold "${k}" > /dev/null 2>&1 || echo "!! driveandact failed: ${tag} fold ${k}" ) &
    done
    wait
    python ${S} ${A} summarize > /dev/null && echo "===== ${tag} done $(date +%H:%M:%S)"
}

case "${SUB}" in
    0)  run_cfg setfusion  --model setfusion --hidden 32              # proposed model (27.9K)
        run_cfg gating ;;                                             # gating-only (1.9K)
    1)  run_cfg setres     --model setres --hidden 32                 # hybrid (31.2K)
        run_cfg resid64    --residual-hidden 64 --residual-uncertainty   # residual-only (8K)
        run_cfg concat_mlp --model concat_mlp ;;
    2)  run_cfg tripose64  --model tripose --hidden 64 --refiner-dim 128   # v1 backbone (261K)
        run_cfg skel_tr    --model skel_tr ;;
    3)  run_cfg lift_mlp   --model lift_mlp
        run_cfg tcn        --model tcn ;;
    *)  echo "unknown sub-request ${SUB}" ;;
esac
kill "${GPUMON}" 2>/dev/null
echo "sub-request ${SUB} finished at $(date)"
