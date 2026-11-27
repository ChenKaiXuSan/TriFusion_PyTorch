#!/bin/bash
#PBS -A HP260146
#PBS -q gen_S
#PBS -b 1
#PBS -l gpunum_job=1
#PBS -l elapstim_req=02:00:00
#PBS -N tpf_head
#PBS -o /dev/null
#PBS -e /dev/null

# Head-movement analysis (STEP1) for every keypoint source, with the annotation time base
# corrected (Label Studio frames are 24 fps; videos 29.97 fps).  CPU-only work; it runs on a
# compute node to keep the login node free (gen_S needs the GPU request).  All sources run
# concurrently, then the cross-source comparison and the glance-direction agreement.
#
#   qsub pegasus/head_analysis_job.sh
#   HEAD_OUT=<dir> overrides the output root; HEAD_LABEL_FPS=29.97 reproduces the old alignment.

REPO="${TPF_REPO:-/work/1/SKIING/chenkaixu/code/TriFusion/.claude/worktrees/hybrid-fusion}"
OUT="${HEAD_OUT:-/work/1/SKIING/chenkaixu/data/drive/downstream_analysis/head_movement_analysis}"
LABEL_FPS="${HEAD_LABEL_FPS:-24}"
SOURCES="${HEAD_SOURCES:-fused_main fused_mean fused_median view_front view_left view_right}"
LOG="${REPO}/logs/pegasus"
mkdir -p "${LOG}" "${OUT}"
exec > "${LOG}/head_analysis.log" 2>&1

set +u
source activate /home/SKIING/chenkaixu/miniconda3/envs/asd
set -u
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
cd "${REPO}" || exit 1
A=analysis/head_movement
echo "head analysis on $(hostname) at $(date); out=${OUT} label_fps=${LABEL_FPS}"

for s in ${SOURCES}; do
    python ${A}/run_head_movement_analysis.py --source "${s}" --label-fps "${LABEL_FPS}" --out-dir "${OUT}/${s}" \
        > "${LOG}/head_analysis_${s}.log" 2>&1 &
done
wait
for s in ${SOURCES}; do
    echo "${s}: sequences=$(grep -c ' min, H ' "${LOG}/head_analysis_${s}.log") errors=$(grep -c '^!!' "${LOG}/head_analysis_${s}.log")"
    grep '^!!' "${LOG}/head_analysis_${s}.log" | head -3
done
python ${A}/compare_sources.py --root "${OUT}" > "${LOG}/head_compare.log" 2>&1 || echo "!! compare_sources failed"
python ${A}/glance_direction_agreement.py --root "${OUT}" || echo "!! glance_direction_agreement failed"
echo "head analysis finished at $(date)"
