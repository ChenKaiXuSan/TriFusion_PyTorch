#!/bin/bash
#PBS -A HP260146
#PBS -q gen_S
#PBS -b 1
#PBS -l gpunum_job=1
#PBS -l elapstim_req=01:30:00
#PBS -N tpf_tf
#PBS -t 0-3
#PBS -o /dev/null
#PBS -e /dev/null

# Training-free baselines of the equal-budget table, one baseline per sub-request
# (geometric median / leave-one-out consensus / temporally smoothed mean / Procrustes
# re-alignment).  They have no parameters, so "train" only evaluates; the cost is the
# CPU-side metric computation (~4 min per fold), which made them too slow to queue behind
# the learned rows of eqb_baselines_job.sh sub-request 1.  The five folds are evaluated
# concurrently.  Results: .../learned_fusion_experiments/eqb_tf_*.
#
#   qsub pegasus/eqb_trainingfree_job.sh

REPO="${TPF_REPO:-/work/1/SKIING/chenkaixu/code/TriFusion/.claude/worktrees/hybrid-fusion}"
SUB="${PBS_SUBREQNO:-0}"
NAMES=(tf_geomedian tf_consensus tf_smooth tf_gpa)
M="${NAMES[${SUB}]}"
LOG="${REPO}/logs/pegasus"
mkdir -p "${LOG}"
exec > "${LOG}/eqb_${M}.log" 2>&1

set +u
source activate /home/SKIING/chenkaixu/miniconda3/envs/sam_3d_body
set -u
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
cd "${REPO}" || exit 1
S=TriPoseFusion/eval/learned_fusion_experiments.py
OUT=TriPoseFusion/eval/logs/learned_fusion_experiments
A="--tag eqb_${M} --model ${M} --threads 2"
echo "${M} on $(hostname) at $(date)"

for k in 0 1 2 3 4; do
    ( [[ -f "${OUT}/eqb_${M}/fold${k}/result.json" ]] \
          || python ${S} ${A} train --folds "${k}" --steps 6000 --batch 32 --window 64 --augment-prob 0.5 \
          || echo "!! evaluation failed: fold ${k}"
      python ${S} ${A} stress --folds "${k}" --mode zero  > /dev/null 2>&1 || echo "!! stress zero failed: fold ${k}"
      python ${S} ${A} stress --folds "${k}" --mode noise > /dev/null 2>&1 || echo "!! stress noise failed: fold ${k}" ) \
        > "${LOG}/eqb_${M}_fold${k}.log" 2>&1 &
done
wait
grep -h -E 'learned\(last\)|!!' "${LOG}"/eqb_${M}_fold*.log | cut -c1-110
# Drive&Act needs fold0/model.pt, so it runs after the folds; one fold is enough (no parameters).
python ${S} ${A} driveandact --fold 0 | grep -E '^learned +PA' || echo "!! driveandact failed"
python ${S} ${A} summarize > /dev/null && echo "===== eqb_${M} done $(date +%H:%M:%S)"
