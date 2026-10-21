#!/bin/bash
# 从 train_template.sh.in 生成修正版重训脚本并（可选）提交。
# 用法：
#   ./generate_and_submit.sh            # 只生成
#   ./generate_and_submit.sh --submit   # 生成并 qsub 提交全部 5 个
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE="${SCRIPT_DIR}/train_template.sh.in"

# experiment          jobname   dilated msvel  lambda robust cva    learned_gate
CONFIGS=(
  "full               c_full    true    true   0.01   true   true   true"
  "uniform_gate       c_unigate true    true   0.0    true   true   false"
  "base_simple        c_base    false   false  0.0    false  true   true"
  "no_cross_view_attention c_nocva true true   0.01   true   false  true"
  "full_gate_lambda0  c_lam0    true    true   0.0    true   true   true"
)

generated=()
for cfg in "${CONFIGS[@]}"; do
  read -r exp job dilated msvel lambda robust cva lgate <<<"${cfg}"
  out="${SCRIPT_DIR}/train_corrected_${exp}.sh"
  sed -e "s|@EXPERIMENT@|${exp}|g" \
      -e "s|@JOBNAME@|${job}|g" \
      -e "s|@DILATED@|${dilated}|g" \
      -e "s|@MSVEL@|${msvel}|g" \
      -e "s|@GATE_LAMBDA@|${lambda}|g" \
      -e "s|@ROBUST@|${robust}|g" \
      -e "s|@CVA@|${cva}|g" \
      -e "s|@LEARNED_GATE@|${lgate}|g" \
      "${TEMPLATE}" > "${out}"
  chmod +x "${out}"
  generated+=("${out}")
  echo "generated: ${out}"
done

if [[ "${1:-}" == "--submit" ]]; then
  for f in "${generated[@]}"; do
    echo "qsub ${f}"
    qsub "${f}"
  done
fi
