#!/bin/bash
#PBS -A SSR
#PBS -q gpu
#PBS -l elapstim_req=24:00:00
#PBS -N run_trid_pose_fusion
#PBS -o logs/pegasus/run_trid_pose_fusion.log
#PBS -e logs/pegasus/run_trid_pose_fusion_err.log

set -eo pipefail

# === 切换到项目目录 ===
PROJECT_DIR=/work/SSR/share/code/MultiView_DriverAction_PyTorch
cd "${PROJECT_DIR}"

mkdir -p logs/pegasus/

# === 加载 Python + 激活 Conda 环境 ===
source activate /home/SSR/luoxi/miniconda3/envs/multiview-video-cls
conda env list

# === 打印运行环境，方便排查超算日志 ===
nvidia-smi
echo "Current working directory: $(pwd)"
echo "Current Python version: $(python --version)"
echo "Current virtual environment: $(which python)"

# TriPoseFusion/main.py 使用包内的短导入名，这里显式加入 PYTHONPATH。
export PYTHONPATH="${PROJECT_DIR}/TriPoseFusion:${PROJECT_DIR}:${PYTHONPATH:-}"

# === TriPoseFusion 训练参数 ===
root_path=/work/SSR/share/data/drive/multi_view_driver_action
index_mapping=${root_path}/index_mapping
index_file=index.json
sam3d_results_path=/work/SSR/share/data/drive/sam3d_body_results_right_full

num_workers=16
batch_size=22
uniform_temporal_subsample_num=16
max_epochs=50
devices=1

echo "Training TriPoseFusion with views: front,left,right"
echo "Index mapping: ${index_mapping}/${index_file}"
echo "SAM3D path: ${sam3d_results_path}"

# === 运行 TriPoseFusion 训练脚本 ===
python TriPoseFusion/main.py \
  paths.root_path="${root_path}" \
  paths.index_mapping="${index_mapping}" \
  paths.index_file="${index_file}" \
  paths.sam3d_results_path="${sam3d_results_path}" \
  data.num_workers="${num_workers}" \
  data.batch_size="${batch_size}" \
  data.uniform_temporal_subsample_num="${uniform_temporal_subsample_num}" \
  model.backbone=triple_fusion \
  train.view=multi \
  'train.view_name=["front","left","right"]' \
  train.max_epochs="${max_epochs}" \
  train.devices="${devices}"
