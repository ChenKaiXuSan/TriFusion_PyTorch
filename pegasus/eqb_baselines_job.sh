#!/bin/bash
#PBS -A HP260146
#PBS -q gen_S
#PBS -b 1
#PBS -l gpunum_job=1
#PBS -l elapstim_req=04:00:00
#PBS -N tpf_eqb
#PBS -t 0-3
#PBS -o /dev/null
#PBS -e /dev/null

# Equal-budget main table + comparison baselines (T-ITS).
#
# Every learned row is trained with the SAME budget (6000 steps x batch 32 x 64-frame
# windows, corruption augmentation p=0.5, fixed seed, final checkpoint) so that rows are
# comparable; the earlier table mixed 6000x32x64 (residual-only), 3000x32x64 (gating-only)
# and 3000x16x32 (set fusion, v1 backbone).  Training-free baselines are only evaluated.
# Each configuration: train (5 folds, resumable per fold) -> stress zero/noise ->
# Drive&Act zero-shot -> summarize.  Results: TriPoseFusion/eval/logs/learned_fusion_experiments/eqb_*.
#
# Submit (SKIING budget is exhausted; gen_S needs the explicit GPU request):
#   qsub pegasus/eqb_baselines_job.sh
# Re-submitting is safe: folds with an existing result.json are skipped.

REPO="${TPF_REPO:-/work/1/SKIING/chenkaixu/code/TriFusion/.claude/worktrees/hybrid-fusion}"
SUB="${PBS_SUBREQNO:-0}"
mkdir -p "${REPO}/logs/pegasus"
exec > "${REPO}/logs/pegasus/eqb_sub${SUB}.log" 2>&1

set +u
source activate /home/SKIING/chenkaixu/miniconda3/envs/sam_3d_body
set -u
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
cd "${REPO}" || exit 1
S=TriPoseFusion/eval/learned_fusion_experiments.py
OUT=TriPoseFusion/eval/logs/learned_fusion_experiments
BUDGET="--steps 6000 --batch 32 --window 64 --augment-prob 0.5 --eval-every 1000"

echo "sub-request ${SUB} on $(hostname) at $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

run_cfg() {  # $1 = tag suffix, rest = model arguments
    local name="$1"; shift
    local tag="eqb_${name}"
    local A="--tag ${tag} --threads 8 $*"
    echo "===== ${tag} ($*) start $(date +%H:%M:%S)"
    for k in 0 1 2 3 4; do
        if [[ -f "${OUT}/${tag}/fold${k}/result.json" ]]; then
            echo "[fold ${k}] result.json exists, skip"
        else
            python ${S} ${A} train --folds "${k}" ${BUDGET} || echo "!! train failed: ${tag} fold ${k}"
        fi
    done
    python ${S} ${A} stress --folds 0 1 2 3 4 --mode zero  | grep -E 'clean|front' || echo "!! stress zero failed: ${tag}"
    python ${S} ${A} stress --folds 0 1 2 3 4 --mode noise | grep -E 'front' || echo "!! stress noise failed: ${tag}"
    local da_folds="0 1 2 3 4"
    [[ "${name}" == tf_* ]] && da_folds="0"      # training-free: identical for every fold
    for k in ${da_folds}; do
        python ${S} ${A} driveandact --fold "${k}" | grep -E '^learned +PA' || echo "!! driveandact failed: ${tag} fold ${k}"
    done
    python ${S} ${A} summarize > /dev/null && echo "===== ${tag} done $(date +%H:%M:%S)"
}

case "${SUB}" in
    0)  run_cfg setfusion  --model setfusion --hidden 32          # proposed model (27.9K)
        run_cfg setres     --model setres --hidden 32 ;;          # hybrid (31.2K)
    1)  run_cfg resid64    --residual-hidden 64 --residual-uncertainty   # residual-only (8K)
        run_cfg gating                                              # gating-only (1.9K): learned confidence weighting
        run_cfg concat_mlp --model concat_mlp
        for m in tf_geomedian tf_consensus tf_smooth tf_gpa; do run_cfg "${m}" --model "${m}"; done ;;
    2)  run_cfg tripose64  --model tripose --hidden 64 --refiner-dim 128   # v1 backbone (261K)
        run_cfg skel_tr    --model skel_tr ;;
    3)  run_cfg lift_mlp   --model lift_mlp
        run_cfg tcn        --model tcn ;;
    *)  echo "unknown sub-request ${SUB}" ;;
esac
echo "sub-request ${SUB} finished at $(date)"
