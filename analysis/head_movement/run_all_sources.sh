#!/bin/bash
# Run the STEP1 head-movement analysis for every keypoint source into per-source subfolders,
# then build the cross-source comparison.  Usage: bash run_all_sources.sh [out_root]
set -u
PY=/home/SKIING/chenkaixu/miniconda3/envs/asd/bin/python
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT=${1:-/work/1/SKIING/chenkaixu/data/drive/downstream_analysis/head_movement_analysis}
LOGDIR=${LOGDIR:-/home/SKIING/chenkaixu/.claude/jobs/028e3236/tmp}
for s in ${SOURCES:-fused_v3 fused_mean fused_median view_front view_left view_right}; do
  nice -n 5 "$PY" "$HERE/run_head_movement_analysis.py" --source "$s" --out-dir "$OUT/$s" > "$LOGDIR/head_analysis_$s.log" 2>&1
  echo "$s exit=$? errors=$(grep -c '^!!' "$LOGDIR/head_analysis_$s.log")"
  grep -A4 "==== SUMMARY" "$LOGDIR/head_analysis_$s.log" | tail -4
done
"$PY" "$HERE/compare_sources.py" --root "$OUT"
echo ALL_SOURCES_DONE
