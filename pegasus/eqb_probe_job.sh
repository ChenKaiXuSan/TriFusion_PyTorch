#!/bin/bash
#PBS -A HP260146
#PBS -q gen_S
#PBS -b 1
#PBS -l gpunum_job=1
#PBS -l elapstim_req=02:30:00
#PBS -N tpf_probe
#PBS -t 0-10
#PBS -o /dev/null
#PBS -e /dev/null

# Follow-up rows of the equal-budget table (same budget: 6000 steps x 32 x 64, aug 0.5).
#
#   sub 0      eqb_gating_nt        gating-only WITHOUT the coordinate temporal conv: a pure convex
#                                   per-view weighting (the default gating row also has a biased
#                                   depthwise temporal conv, which learns a global offset at this budget)
#   sub 1-5    eqb_setfusion_d64    capacity probe of the proposed model, d=64  (fold = sub-1)
#   sub 6-10   eqb_setfusion_d128   capacity probe of the proposed model, d=128 (fold = sub-6)
#
# The set-fusion model saturates one GPU, so its folds run on one GPU each; the tiny gating
# model is CPU-bound and trains its five folds concurrently on one GPU.
#
#   qsub pegasus/eqb_probe_job.sh

REPO="${TPF_REPO:-/work/1/SKIING/chenkaixu/code/TriFusion/.claude/worktrees/hybrid-fusion}"
SUB="${PBS_SUBREQNO:-0}"
LOG="${REPO}/logs/pegasus"
mkdir -p "${LOG}"
exec > "${LOG}/eqb_probe_sub${SUB}.log" 2>&1

set +u
source activate /home/SKIING/chenkaixu/miniconda3/envs/sam_3d_body
set -u
cd "${REPO}" || exit 1
S=TriPoseFusion/eval/learned_fusion_experiments.py
OUT=TriPoseFusion/eval/logs/learned_fusion_experiments
BUDGET="--steps 6000 --batch 32 --window 64 --augment-prob 0.5 --eval-every 1000"
echo "sub-request ${SUB} on $(hostname) at $(date)"

run_fold() {  # $1 = name, $2 = fold, $3 = threads, rest = model arguments
    local name="$1" k="$2" th="$3"; shift 3
    local tag="eqb_${name}"
    local A="--tag ${tag} --threads ${th} $*"
    if [[ -f "${OUT}/${tag}/fold${k}/result.json" ]]; then
        echo "[${tag} fold ${k}] result.json exists, skip training"
    else
        OMP_NUM_THREADS="${th}" python ${S} ${A} train --folds "${k}" ${BUDGET} | grep -E 'device|step |learned\(last\)' \
            || echo "!! train failed: ${tag} fold ${k}"
    fi
    python ${S} ${A} stress --folds "${k}" --mode zero  > /dev/null 2>&1 || echo "!! stress zero failed: ${tag} fold ${k}"
    python ${S} ${A} stress --folds "${k}" --mode noise > /dev/null 2>&1 || echo "!! stress noise failed: ${tag} fold ${k}"
    python ${S} ${A} driveandact --fold "${k}" | grep -E '^learned +PA' || echo "!! driveandact failed: ${tag} fold ${k}"
    echo "===== ${tag} fold ${k} done $(date +%H:%M:%S)"
}

if (( SUB == 0 )); then
    for k in 0 1 2 3 4; do run_fold gating_nt "${k}" 2 --no-temporal & done
    wait
elif (( SUB <= 5 )); then
    run_fold setfusion_d64 $((SUB - 1)) 8 --model setfusion --hidden 64
else
    run_fold setfusion_d128 $((SUB - 6)) 8 --model setfusion --hidden 128
fi
echo "sub-request ${SUB} finished at $(date)"
