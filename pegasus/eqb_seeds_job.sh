#!/bin/bash
#PBS -A HP260146
#PBS -q gen_S
#PBS -b 1
#PBS -l gpunum_job=1
#PBS -l elapstim_req=02:00:00
#PBS -N tpf_seeds
#PBS -t 0-15
#PBS -o /dev/null
#PBS -e /dev/null

# Seed variance of the key rows of the main table (common budget 6000 x 32 x 64, aug 0.5).
# Seed 42 is the existing eqb_<name> run; seeds 43 and 44 are added here as eqb_<name>_s43 / _s44.
#
#   sub 0-9    set fusion (GPU-bound): seed 43 folds 0-4 (sub 0-4), seed 44 folds 0-4 (sub 5-9), one fold per GPU
#   sub 10-15  residual-only, Temporal-CNN, Lifting-MLP (light): one (model, seed) per GPU, five folds concurrently
#
#   qsub pegasus/eqb_seeds_job.sh

REPO="${TPF_REPO:-/work/1/SKIING/chenkaixu/code/TriFusion/.claude/worktrees/hybrid-fusion}"
SUB="${PBS_SUBREQNO:-0}"
LOG="${REPO}/logs/pegasus"
mkdir -p "${LOG}"
exec > "${LOG}/eqb_seeds_sub${SUB}.log" 2>&1

set +u
source activate /home/SKIING/chenkaixu/miniconda3/envs/sam_3d_body
set -u
cd "${REPO}" || exit 1
S=TriPoseFusion/eval/learned_fusion_experiments.py
OUT=TriPoseFusion/eval/logs/learned_fusion_experiments
BUDGET="--steps 6000 --batch 32 --window 64 --augment-prob 0.5 --eval-every 2000"
echo "sub-request ${SUB} on $(hostname) at $(date)"

train() {  # $1 name, $2 seed, $3 fold, $4 threads, rest = model args
    local name="$1" seed="$2" k="$3" th="$4"; shift 4
    local tag="eqb_${name}_s${seed}"
    if [[ -f "${OUT}/${tag}/fold${k}/result.json" ]]; then
        echo "[${tag} fold ${k}] exists, skip"; return
    fi
    OMP_NUM_THREADS="${th}" python ${S} --tag "${tag}" --threads "${th}" --seed "${seed}" "$@" train --folds "${k}" ${BUDGET} \
        | grep -E 'learned\(last\)' | sed "s/^/${tag} /" || echo "!! train failed: ${tag} fold ${k}"
}

if (( SUB <= 9 )); then
    seed=$(( SUB <= 4 ? 43 : 44 )); k=$(( SUB % 5 ))
    train setfusion "${seed}" "${k}" 8 --model setfusion --hidden 32
else
    i=$(( SUB - 10 )); seed=$(( 43 + i % 2 ))
    case $(( i / 2 )) in
        0) name=resid64;  args=(--residual-hidden 64 --residual-uncertainty) ;;
        1) name=tcn;      args=(--model tcn) ;;
        2) name=lift_mlp; args=(--model lift_mlp) ;;
    esac
    for k in 0 1 2 3 4; do train "${name}" "${seed}" "${k}" 2 "${args[@]}" & done
    wait
fi
echo "sub-request ${SUB} finished at $(date)"
