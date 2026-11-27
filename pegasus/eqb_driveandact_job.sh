#!/bin/bash
#PBS -A HP260146
#PBS -q gen_S
#PBS -b 1
#PBS -l gpunum_job=1
#PBS -l elapstim_req=02:30:00
#PBS -N tpf_da
#PBS -t 0-9
#PBS -o /dev/null
#PBS -e /dev/null

# Drive&Act: every method of the equal-budget table, zero-shot AND trained in-domain, under
# ONE evaluation protocol (leave-one-subject-out, 3 folds, whole validation sequences,
# compute_metrics against the dataset reference; cache learned_fusion_cache_driveandact).
#
#   zero-shot      `crosseval` of the in-house fold models eqb_<name>  -> eqb_<name>/fold*/crosseval_driveandact.json
#   daeq_<name>    trained on the dataset reference,        1500 steps x 32 x 64, aug 0.5
#   daeqlf_<name>  trained label-free (self-triangulated reference gt_pose_tri9), same budget;
#                  validation still against the dataset reference
#   daeq6k_<name>  trained on the dataset reference with the unchanged in-house recipe
#                  (6000 steps x 32 x 64), validated every 250 steps -> error-vs-steps curve
#
# 1500 steps is the step budget fixed for Drive&Act in August (before any baseline existed);
# batch/window follow the in-house recipe.  Each run is followed by the zero/noise stress test.
# One sub-request per method; the three folds run concurrently on one GPU.
#
#   qsub pegasus/eqb_driveandact_job.sh

REPO="${TPF_REPO:-/work/1/SKIING/chenkaixu/code/TriFusion/.claude/worktrees/hybrid-fusion}"
SUB="${PBS_SUBREQNO:-0}"
LOG="${REPO}/logs/pegasus"
mkdir -p "${LOG}"
exec > "${LOG}/da_sub${SUB}.log" 2>&1

set +u
source activate /home/SKIING/chenkaixu/miniconda3/envs/sam_3d_body
set -u
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
cd "${REPO}" || exit 1
S=TriPoseFusion/eval/learned_fusion_experiments.py
OUT=TriPoseFusion/eval/logs/learned_fusion_experiments
D=/work/1/SKIING/chenkaixu/data/drive
DA_IDX="--index-mapping ${D}/driveandact_index_mapping"
echo "sub-request ${SUB} on $(hostname) at $(date)"

declare -A MODEL=(
    [setfusion]="--model setfusion --hidden 32"
    [setres]="--model setres --hidden 32"
    [resid64]="--residual-hidden 64 --residual-uncertainty"
    [gating_nt]="--no-temporal"
    [gating]=""
    [tripose64]="--model tripose --hidden 64 --refiner-dim 128"
    [concat_mlp]="--model concat_mlp" [lift_mlp]="--model lift_mlp" [tcn]="--model tcn" [skel_tr]="--model skel_tr"
    [tf_geomedian]="--model tf_geomedian" [tf_consensus]="--model tf_consensus"
    [tf_smooth]="--model tf_smooth" [tf_gpa]="--model tf_gpa"
    [setfusion_d64]="--model setfusion --hidden 64" [setfusion_d128]="--model setfusion --hidden 128"
)

run_da() {  # $1 = tag prefix, $2 = name, $3 = steps, $4 = eval-every, $5 = cache dir name, $6 = train gt key
    local prefix="$1" name="$2" steps="$3" every="$4" cache="$5" gtkey="$6"
    local tag="${prefix}_${name}"
    local A="--tag ${tag} --threads 2 --cache-dir ${D}/${cache} ${DA_IDX} ${MODEL[${name}]}"
    echo "===== ${tag} start $(date +%H:%M:%S)"
    for k in 0 1 2; do
        ( if [[ ! -f "${OUT}/${tag}/fold${k}/result.json" ]]; then
              python ${S} ${A} train --folds "${k}" --steps "${steps}" --batch 32 --window 64 --augment-prob 0.5 \
                  --eval-every "${every}" --train-gt-key "${gtkey}" || echo "!! train failed: ${tag} fold ${k}"
          fi
          python ${S} ${A} stress --folds "${k}" --mode zero  > /dev/null 2>&1 || echo "!! stress zero failed: ${tag} fold ${k}"
          python ${S} ${A} stress --folds "${k}" --mode noise > /dev/null 2>&1 || echo "!! stress noise failed: ${tag} fold ${k}"
        ) > "${LOG}/${tag}_fold${k}.log" 2>&1 &
    done
    wait
    grep -h -E 'learned\(last\)|!!' "${LOG}/${tag}"_fold*.log | cut -c1-120
    echo "===== ${tag} done $(date +%H:%M:%S)"
}

run_all() {  # every Drive&Act experiment of one learned method
    run_da daeq   "$1" 1500 250 learned_fusion_cache_driveandact         gt_pose
    run_da daeqlf "$1" 1500 250 learned_fusion_cache_driveandact_selftri gt_pose_tri9
    run_da daeq6k "$1" 6000 250 learned_fusion_cache_driveandact         gt_pose
}

NAMES=(_ concat_mlp lift_mlp tcn skel_tr tripose64 gating_nt resid64 setres setfusion)
if (( SUB == 0 )); then
    # zero-shot of every in-house model under the same protocol, then the training-free baselines
    for name in setfusion setres resid64 gating_nt gating tripose64 concat_mlp lift_mlp tcn skel_tr \
                tf_geomedian tf_consensus tf_smooth tf_gpa setfusion_d64 setfusion_d128; do
        python ${S} --tag "eqb_${name}" --threads 8 ${MODEL[${name}]} crosseval --folds 0 1 2 3 4 \
            | grep -E 'zero-shot' | sed "s/^/eqb_${name} /" || echo "!! crosseval failed: eqb_${name}"
    done
    for name in tf_geomedian tf_consensus tf_smooth tf_gpa; do
        run_da daeq "${name}" 1500 250 learned_fusion_cache_driveandact gt_pose
    done
else
    run_all "${NAMES[${SUB}]}"
fi
echo "sub-request ${SUB} finished at $(date)"
