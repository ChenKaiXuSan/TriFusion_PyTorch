#!/bin/bash
#PBS -A HP260146
#PBS -q gen_S
#PBS -b 1
#PBS -l gpunum_job=1
#PBS -l elapstim_req=01:30:00
#PBS -N tpf_dasc
#PBS -t 0-8
#PBS -o /dev/null
#PBS -e /dev/null

# Drive&Act label-free retraining with a CALIBRATION-FREE reference: identical to daeqlf_<name> in
# eqb_driveandact_job.sh (1500 steps x 32 x 64, aug 0.5, 3 LOSO folds, validation against the dataset
# reference), but the self-triangulated training reference uses self-calibrated extrinsics
# (driveandact_selfcalib.py from official extrinsics perturbed by 30 deg / 15 cm, then
# build_driveandact_cache_selftri.py --extrinsics-json) instead of the official calibration.
#
#   daeqsc_<name>   -> TriPoseFusion/eval/logs/learned_fusion_experiments/daeqsc_<name>/fold*/result.json
#
#   qsub pegasus/da_selfcalib_job.sh

REPO="${TPF_REPO:-/work/1/SKIING/chenkaixu/code/TriFusion/.claude/worktrees/hybrid-fusion}"
SUB="${PBS_SUBREQNO:-0}"
LOG="${REPO}/logs/pegasus"
mkdir -p "${LOG}"
exec > "${LOG}/dasc_sub${SUB}.log" 2>&1

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
    [tripose64]="--model tripose --hidden 64 --refiner-dim 128"
    [concat_mlp]="--model concat_mlp" [lift_mlp]="--model lift_mlp" [tcn]="--model tcn" [skel_tr]="--model skel_tr"
)
NAMES=(concat_mlp lift_mlp tcn skel_tr tripose64 gating_nt resid64 setres setfusion)
name="${NAMES[${SUB}]}"
tag="daeqsc_${name}"
A="--tag ${tag} --threads 2 --cache-dir ${D}/learned_fusion_cache_driveandact_selfcalib ${DA_IDX} ${MODEL[${name}]}"
for k in 0 1 2; do
    ( if [[ ! -f "${OUT}/${tag}/fold${k}/result.json" ]]; then
          python ${S} ${A} train --folds "${k}" --steps 1500 --batch 32 --window 64 --augment-prob 0.5 \
              --eval-every 250 --train-gt-key gt_pose_tri9 || echo "!! train failed: ${tag} fold ${k}"
      fi
    ) > "${LOG}/${tag}_fold${k}.log" 2>&1 &
done
wait
grep -h -E 'learned\(last\)|!!' "${LOG}/${tag}"_fold*.log | cut -c1-120
echo "sub-request ${SUB} finished at $(date)"
