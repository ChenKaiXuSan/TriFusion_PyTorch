#!/bin/bash
#PBS -A SKIING
#PBS -q gpu
#PBS -l elapstim_req=24:00:00
#PBS -N run_trid_pose_fusion
#PBS -o ${PROJECT_DIR}/logs/pegasus/run_trid_pose_fusion.log
#PBS -e ${PROJECT_DIR}/logs/pegasus/run_trid_pose_fusion_err.log

# === 切换到项目目录 ===
PROJECT_DIR=/work/SKIING/chenkaixu/code/MultiView_DriverAction_PyTorch
cd "${PROJECT_DIR}"

mkdir -p ${PROJECT_DIR}/logs/pegasus/

# === 加载 Python + 激活 Conda 环境 ===
# Pegasus may preload Intel Python/oneAPI. Its xgboost deactivate hook can
# reference unset variables, so disable nounset only while conda switches envs.
case "$-" in
  *u*) had_nounset=1 ;;
  *) had_nounset=0 ;;
esac
set +u
source activate /home/SKIING/chenkaixu/miniconda3/envs/direction
if [ "${had_nounset}" -eq 1 ]; then
  set -u
fi
conda env list

# === 打印运行环境，方便排查超算日志 ===
nvidia-smi
echo "Current working directory: $(pwd)"
echo "Current Python version: $(python --version)"
echo "Current virtual environment: $(which python)"

# TriPoseFusion/main.py 使用包内的短导入名，这里显式加入 PYTHONPATH。
export PYTHONPATH="${PROJECT_DIR}/TriPoseFusion:${PROJECT_DIR}:${PYTHONPATH:-}"

# === TriPoseFusion 训练参数 ===
root_path=/work/SKIING/chenkaixu/data/drive
index_mapping=${root_path}/index_mapping
sam3d_results_path=/work/SKIING/chenkaixu/data/drive/sam3d_body_results_right

num_workers=32
batch_size=32
uniform_temporal_subsample_num=16
max_epochs=50
devices=1

echo "Training TriPoseFusion with views: front,left,right"
echo "Index mapping: ${index_mapping}"
echo "SAM3D path: ${sam3d_results_path}"

# === 运行 TriPoseFusion 训练脚本 ===
python TriPoseFusion/train.py \
  paths.root_path="${root_path}" \
  paths.sam3d_results_path="${sam3d_results_path}" \
  data.num_workers="${num_workers}" \
  data.batch_size="${batch_size}" \
  data.uniform_temporal_subsample_num="${uniform_temporal_subsample_num}" \
  model.backbone=triple_fusion \
  train.view=multi \
  'train.view_name=["front","left","right"]' \
  train.max_epochs="${max_epochs}" \
  train.devices="${devices}"
