#!/bin/bash
#PBS -A SKIING
#PBS -q gpu
#PBS -l elapstim_req=24:00:00
#PBS -N trid_pose_ablation
#PBS -o /work/SKIING/chenkaixu/code/MultiView_DriverAction_PyTorch/logs/pegasus/trid_pose_ablation.out
#PBS -e /work/SKIING/chenkaixu/code/MultiView_DriverAction_PyTorch/logs/pegasus/trid_pose_ablation.err

# =============================================================================
# TriPoseFusion 消融实验通用 runner
# =============================================================================
# 这个脚本现在主要给下面 6 个独立 PBS 脚本复用：
#   train_trid_pose_fusion_base_simple.sh
#   train_trid_pose_fusion_dilated_tcn.sh
#   train_trid_pose_fusion_multiscale_velocity.sh
#   train_trid_pose_fusion_gate_entropy.sh
#   train_trid_pose_fusion_robust_canon.sh
#   train_trid_pose_fusion_full.sh
#
# 推荐提交上面的具体实验脚本，而不是直接提交本文件。
# 每个具体实验脚本都固定 FOLD=0，并设置自己的 EXPERIMENT_NAME。
#
# 日志位置：
#   PBS stdout/stderr 由具体实验脚本决定，例如：
#     logs/pegasus/trid_base_simple.out
#     logs/pegasus/trid_base_simple.err
#   训练日志和 checkpoint:
#     logs/train/trifusion_${EXPERIMENT_NAME}_fold${FOLD}/...
# =============================================================================

set -euo pipefail

# =============================================================================
# 参数覆盖
# =============================================================================
# 这些参数通常由具体实验脚本设置，也可以通过 qsub -v 覆盖：
#   MAX_EPOCHS=5       快速 smoke test；正式实验建议 50
#   NUM_WORKERS=32     DataLoader worker 数量
#   BATCH_SIZE=32      batch size
#   NUM_FRAMES=16      uniform temporal subsample 帧数
#   DEVICES=1          GPU 数量

# =============================================================================
# 消融实验设计
# =============================================================================
# base_simple:
#   最基础的三视角融合 baseline。
#   关闭 dilated TCN、多尺度速度、gate entropy、robust canonicalization。
#
# dilated_tcn:
#   只打开 dilated temporal refiner。
#   用来验证更大 temporal receptive field 是否提升姿态融合稳定性。
#
# multiscale_velocity:
#   只打开 multi-scale velocity。
#   用来验证速度、加速度、jerk 等动态特征是否有贡献。
#
# gate_entropy:
#   只打开 gate entropy regularization。
#   用来观察 view gate 是否更均衡，避免模型完全忽略某个视角。
#
# robust_canon:
#   只打开 robust canonicalization。
#   用来测试肩部关键点异常时，鲁棒坐标规范化是否更稳定。
#
# full:
#   完整 TriPoseFusion。
#   同时打开所有改进模块，作为最终方法。
# =============================================================================

# === 切换到项目目录 ===
PROJECT_DIR=/work/SKIING/chenkaixu/code/MultiView_DriverAction_PyTorch
cd "${PROJECT_DIR}"

mkdir -p "${PROJECT_DIR}/logs/pegasus"

# === 加载 Python + Conda 环境 ===
# Pegasus preloads Intel Python/oneAPI on some nodes. Its xgboost deactivate hook
# reads OCL_ICD_FILENAMES_RESET without guarding for nounset, so relax -u only
# while conda switches environments and restore strict mode immediately after.
set +u
source activate /home/SKIING/chenkaixu/miniconda3/envs/direction
set -u

# === 打印环境信息，便于从 Pegasus 日志排查问题 ===
echo "============================================================"
echo "TriPoseFusion ablation job"
echo "Project dir: ${PROJECT_DIR}"
echo "Python: $(python --version)"
echo "Python path: $(which python)"
echo "Start time: $(date)"
echo "============================================================"
nvidia-smi
conda env list

# TriPoseFusion/train.py 使用包内短导入名，这里显式加入 PYTHONPATH。
export PYTHONPATH="${PROJECT_DIR}/TriPoseFusion:${PROJECT_DIR}:${PYTHONPATH:-}"

# === 数据路径 ===
# root_path:
#   数据集根目录，内部应包含 index_mapping 等子目录。
# index_mapping:
#   fold_{fold}.json 所在目录。train.fold 会选择对应 fold JSON。
# sam3d_results_path:
#   SAM3D 输出的三视角 3D keypoint 目录。
root_path=/work/SKIING/chenkaixu/data/drive
index_mapping=${root_path}/index_mapping
sam3d_results_path=/work/SKIING/chenkaixu/data/drive/sam3d_body_results_right

# === 默认训练参数 ===
# 这些参数可以通过 qsub -v 或提交器脚本覆盖。
num_workers=${NUM_WORKERS:-32}
batch_size=${BATCH_SIZE:-32}
uniform_temporal_subsample_num=${NUM_FRAMES:-16}
max_epochs=${MAX_EPOCHS:-50}
devices=${DEVICES:-1}
fold=${FOLD:-0}
experiment_name=${EXPERIMENT_NAME:-full}

# === 根据 EXPERIMENT_NAME 选择消融开关 ===
# 注意：模型读取的配置名是 geofusion_use_robust_canonicalization。
case "${experiment_name}" in
  base_simple)
    use_dilated_refiner=false
    use_multiscale_velocity=false
    gate_entropy_lambda=0.0
    use_robust_canonicalization=false
    ;;
  dilated_tcn)
    use_dilated_refiner=true
    use_multiscale_velocity=false
    gate_entropy_lambda=0.0
    use_robust_canonicalization=false
    ;;
  multiscale_velocity)
    use_dilated_refiner=false
    use_multiscale_velocity=true
    gate_entropy_lambda=0.0
    use_robust_canonicalization=false
    ;;
  gate_entropy)
    use_dilated_refiner=false
    use_multiscale_velocity=false
    gate_entropy_lambda=0.01
    use_robust_canonicalization=false
    ;;
  robust_canon)
    use_dilated_refiner=false
    use_multiscale_velocity=false
    gate_entropy_lambda=0.0
    use_robust_canonicalization=true
    ;;
  full)
    use_dilated_refiner=true
    use_multiscale_velocity=true
    gate_entropy_lambda=0.01
    use_robust_canonicalization=true
    ;;
  *)
    echo "Unknown EXPERIMENT_NAME: ${experiment_name}" >&2
    echo "Supported: base_simple, dilated_tcn, multiscale_velocity, gate_entropy, robust_canon, full" >&2
    exit 2
    ;;
esac

run_name="trifusion_${experiment_name}_fold${fold}"

# === 打印本次实验配置 ===
echo "Experiment: ${run_name}"
echo "Fold: ${fold}"
echo "Index mapping: ${index_mapping}"
echo "SAM3D path: ${sam3d_results_path}"
echo "Dilated refiner: ${use_dilated_refiner}"
echo "Multiscale velocity: ${use_multiscale_velocity}"
echo "Gate entropy lambda: ${gate_entropy_lambda}"
echo "Robust canonicalization: ${use_robust_canonicalization}"

# === 启动 TriPoseFusion 训练 ===
# 保持数据、fold、epoch、batch size 等条件一致，只改变上面的模块开关，
# 这样 base / 单模块 / full 之间的对比才是公平的。
python TriPoseFusion/train.py \
  paths.root_path="${root_path}" \
  paths.index_mapping="${index_mapping}" \
  paths.sam3d_results_path="${sam3d_results_path}" \
  data.num_workers="${num_workers}" \
  data.batch_size="${batch_size}" \
  data.uniform_temporal_subsample_num="${uniform_temporal_subsample_num}" \
  model.backbone=triple_fusion \
  model.geofusion_use_dilated_refiner="${use_dilated_refiner}" \
  model.geofusion_use_multiscale_velocity="${use_multiscale_velocity}" \
  model.geofusion_gate_entropy_reg_lambda="${gate_entropy_lambda}" \
  model.geofusion_use_robust_canonicalization="${use_robust_canonicalization}" \
  train.view=multi \
  'train.view_name=["front","left","right"]' \
  train.fold="${fold}" \
  train.max_epochs="${max_epochs}" \
  train.devices="${devices}" \
  experiment="${run_name}"

echo "============================================================"
echo "Finished ${run_name}"
echo "End time: $(date)"
echo "============================================================"
