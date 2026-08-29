#!/bin/bash
# 留一视角（LOO）自监督 teacher 实验：快速版（关键点缓存 + 12 epoch + 每 1/4 epoch 验证）。
#   train_loo.sh         : loo_teacher=true,  λ_nce=0.01, λ_gate=0, 按 val/loo_mpjpe 选 ckpt
#   train_loo_control.sh : 中位数 teacher,    λ_nce=0.01, λ_gate=0, 按 val/loss 选 ckpt（隔离 LOO 的贡献）
# 用法： ./generate_loo.sh [--submit]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE="${SCRIPT_DIR}/train_template.sh.in"
FAST_DIR=/work/SKIING/chenkaixu/code/TriFusion_fast

gen() {
  local exp=$1 job=$2 loo=$3 monitor=$4 out="${SCRIPT_DIR}/train_${1}.sh"
  sed -e "s|@EXPERIMENT@|${exp}|g" -e "s|@JOBNAME@|${job}|g" \
      -e "s|@DILATED@|true|g" -e "s|@MSVEL@|true|g" -e "s|@GATE_LAMBDA@|0.0|g" \
      -e "s|@ROBUST@|true|g" -e "s|@CVA@|true|g" -e "s|@LEARNED_GATE@|true|g" \
      -e "s|TriFusion_corrected|TriFusion_fast|g" \
      -e "s|elapstim_req=24:00:00|elapstim_req=06:00:00|" \
      -e "s|max_epochs=\${MAX_EPOCHS:-12}|max_epochs=\${MAX_EPOCHS:-8}|" \
      -e "s|early_stop_patience=\${EARLY_STOP_PATIENCE:-3}|early_stop_patience=\${EARLY_STOP_PATIENCE:-6}|" \
      -e "/paths.kpt_cache_root=/a\\
  model.loo_teacher=${loo} \\\\\\
  model.lambda_info_nce=0.01 \\\\\\
  train.val_check_interval=0.25 \\\\\\
  train.monitor_metric=${monitor} \\\\" \
      "${TEMPLATE}" > "${out}"
  chmod +x "${out}"
  echo "generated: ${out}"
}

gen loo c_loo true val/loo_mpjpe
gen loo_control c_looctl false val/loss

if [[ "${1:-}" == "--submit" ]]; then
  mkdir -p "${FAST_DIR}/logs/pegasus"
  for f in "${SCRIPT_DIR}/train_loo.sh" "${SCRIPT_DIR}/train_loo_control.sh"; do
    echo "qsub ${f}"; qsub "${f}"
  done
fi
