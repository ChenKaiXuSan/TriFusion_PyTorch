#!/bin/bash
#PBS -A HP260146
#PBS -q gen_S
#PBS -b 1
#PBS -l gpunum_job=1
#PBS -l elapstim_req=02:30:00
#PBS -N tpf_paper
#PBS -t 0-17
#PBS -o /dev/null
#PBS -e /dev/null

# Remaining rows of the T-ITS tables, all at the common budget (6000 steps x 32 x 64, seed 42).
#
#   sub 0-4    eqb_setfusion_noaug       proposed model without corruption augmentation   (fold = sub)
#   sub 5-9    eqb_setfusion_nojoint     proposed model without the skeleton-attention layer (fold = sub-5)
#   sub 10-14  eqb_setfusion_notemporal  proposed model without the temporal convolution  (fold = sub-10)
#   sub 15     eqb_tripose128            v1 backbone at its original capacity (870K), 5 folds in sequence
#   sub 16,17  "drop" corruption (one view absent) for the rows of the robustness table
#
# Every trained fold also gets the zero / noise / drop corruption tests.
#
#   qsub pegasus/eqb_paper_job.sh

REPO="${TPF_REPO:-/work/1/SKIING/chenkaixu/code/TriFusion/.claude/worktrees/hybrid-fusion}"
SUB="${PBS_SUBREQNO:-0}"
LOG="${REPO}/logs/pegasus"
mkdir -p "${LOG}"
exec > "${LOG}/eqb_paper_sub${SUB}.log" 2>&1

set +u
source activate /home/SKIING/chenkaixu/miniconda3/envs/sam_3d_body
set -u
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
cd "${REPO}" || exit 1
S=TriPoseFusion/eval/learned_fusion_experiments.py
OUT=TriPoseFusion/eval/logs/learned_fusion_experiments
echo "sub-request ${SUB} on $(hostname) at $(date)"

run_fold() {  # $1 = name, $2 = fold, $3 = augment prob, rest = model arguments
    local name="$1" k="$2" aug="$3"; shift 3
    local tag="eqb_${name}"
    local A="--tag ${tag} --threads 8 $*"
    if [[ -f "${OUT}/${tag}/fold${k}/result.json" ]]; then
        echo "[${tag} fold ${k}] result.json exists, skip training"
    else
        python ${S} ${A} train --folds "${k}" --steps 6000 --batch 32 --window 64 --augment-prob "${aug}" --eval-every 1000 \
            | grep -E 'device|step |learned\(last\)' || echo "!! train failed: ${tag} fold ${k}"
    fi
    for mode in zero noise drop; do
        python ${S} ${A} stress --folds "${k}" --mode "${mode}" > /dev/null 2>&1 || echo "!! stress ${mode} failed: ${tag} fold ${k}"
    done
    echo "===== ${tag} fold ${k} done $(date +%H:%M:%S)"
}

drop_only() {  # $1 = tag suffix, rest = model arguments ; five folds concurrently (CPU-bound metric code)
    local name="$1"; shift
    for k in 0 1 2 3 4; do
        OMP_NUM_THREADS=2 python ${S} --tag "eqb_${name}" --threads 2 "$@" stress --folds "${k}" --mode drop > /dev/null 2>&1 \
            || echo "!! stress drop failed: eqb_${name} fold ${k}" &
    done
    wait
    echo "===== eqb_${name} drop done $(date +%H:%M:%S)"
}

if (( SUB <= 4 )); then
    run_fold setfusion_noaug "${SUB}" 0.0 --model setfusion --hidden 32
elif (( SUB <= 9 )); then
    run_fold setfusion_nojoint $((SUB - 5)) 0.5 --model setfusion --hidden 32 --joint-layers 0
elif (( SUB <= 14 )); then
    run_fold setfusion_notemporal $((SUB - 10)) 0.5 --model setfusion --hidden 32 --no-temporal
elif (( SUB == 15 )); then
    for k in 0 1 2 3 4; do run_fold tripose128 "${k}" 0.5 --model tripose --hidden 128 --refiner-dim 256; done
elif (( SUB == 16 )); then
    drop_only setfusion --model setfusion --hidden 32
    drop_only setres --model setres --hidden 32
    drop_only resid64 --residual-hidden 64 --residual-uncertainty
    drop_only gating_nt --no-temporal
    drop_only tripose64 --model tripose --hidden 64 --refiner-dim 128
else
    for m in tcn lift_mlp concat_mlp skel_tr tf_geomedian tf_consensus; do drop_only "${m}" --model "${m}"; done
fi
echo "sub-request ${SUB} finished at $(date)"
