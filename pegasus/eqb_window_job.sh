#!/bin/bash
#PBS -A HP260146
#PBS -q gen_S
#PBS -b 1
#PBS -l gpunum_job=1
#PBS -l elapstim_req=03:00:00
#PBS -N tpf_win
#PBS -t 0-2
#PBS -o /dev/null
#PBS -e /dev/null

# Training-window ablation.  The window length is varied at a CONSTANT number of frames per
# step (batch x window = 2048) and a constant step budget, so only the temporal context of
# a training sample changes, not the amount of data seen:
#     window 16 x batch 128 | 32 x 64 | (64 x 32 = eqb_<name>, the common setting) | 128 x 16 | 256 x 8
# Models: proposed set fusion (27.9K), residual-only (8K) and the Temporal-CNN baseline
# (receptive field 29 frames, the most window-sensitive row).  Validation always runs on
# whole sequences, so the numbers are directly comparable with eqb_<name>.
# Five folds train concurrently on one GPU.  Results: .../eqbw<W>_<name>/.
#
#   qsub pegasus/eqb_window_job.sh            # EQB_STEPS=6000 by default

REPO="${TPF_REPO:-/work/1/SKIING/chenkaixu/code/TriFusion/.claude/worktrees/hybrid-fusion}"
SUB="${PBS_SUBREQNO:-0}"
STEPS="${EQB_STEPS:-6000}"
LOG="${REPO}/logs/pegasus"
mkdir -p "${LOG}"
exec > "${LOG}/eqbw_sub${SUB}.log" 2>&1

set +u
source activate /home/SKIING/chenkaixu/miniconda3/envs/sam_3d_body
set -u
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
cd "${REPO}" || exit 1
S=TriPoseFusion/eval/learned_fusion_experiments.py
OUT=TriPoseFusion/eval/logs/learned_fusion_experiments
echo "sub-request ${SUB} on $(hostname) at $(date); steps=${STEPS} cores=$(nproc)"

run_win() {  # $1 = window, $2 = name, rest = model arguments
    local win="$1" name="$2"; shift 2
    local batch=$((2048 / win))
    local tag="eqbw${win}_${name}"
    local A="--tag ${tag} --threads 2 $*"
    echo "===== ${tag} (window ${win} x batch ${batch}) start $(date +%H:%M:%S)"
    for k in 0 1 2 3 4; do
        if [[ -f "${OUT}/${tag}/fold${k}/result.json" ]]; then
            echo "[fold ${k}] result.json exists, skip"
        else
            python ${S} ${A} train --folds "${k}" --steps "${STEPS}" --batch "${batch}" --window "${win}" \
                --augment-prob 0.5 --eval-every $((STEPS / 3)) > "${LOG}/${tag}_train_fold${k}.log" 2>&1 &
        fi
    done
    wait
    for k in 0 1 2 3 4; do
        [[ -f "${OUT}/${tag}/fold${k}/result.json" ]] || echo "!! train failed: ${tag} fold ${k}"
        grep -h 'learned(last)' "${LOG}/${tag}_train_fold${k}.log" 2>/dev/null | cut -c1-70
    done
    python ${S} ${A} summarize > /dev/null && echo "===== ${tag} done $(date +%H:%M:%S)"
}

case "${SUB}" in
    0)  for w in 16 32;        do run_win "${w}" setfusion --model setfusion --hidden 32; done ;;
    1)  for w in 128 256;      do run_win "${w}" setfusion --model setfusion --hidden 32; done ;;
    2)  for w in 16 32 128 256; do run_win "${w}" resid64 --residual-hidden 64 --residual-uncertainty; done
        for w in 16 32 128 256; do run_win "${w}" tcn --model tcn; done ;;
    *)  echo "unknown sub-request ${SUB}" ;;
esac
echo "sub-request ${SUB} finished at $(date)"
