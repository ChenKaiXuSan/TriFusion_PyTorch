#!/usr/bin/env python3
"""
比较 SAM3D 原始三视角 3D keypoints 和三角化/合成 3D keypoints

脚本功能:
1. 计算 SAM3D 3D keypoints 相对三角化/合成 3D keypoints 的常见指标
2. 输出 MPJPE、Root-aligned MPJPE、PCK、AUC、per-joint MPJPE 等
3. 分析不同环境（昼/夜、多/少）和相机视角下的误差差异
4. 生成详细报告和可视化图表

使用方式:
    conda run -n torch113 python scripts/compare_sam3d_triangulation.py \
        --sam3d-root /home/data/xchen/drive/sam3d_body_results_right \
        --tri-gt-root /home/data/xchen/drive/sam3d_body_triangulated_gt

作者：Kaixu Chen
日期：2026-06-06
"""
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import argparse
import json
import numpy as np
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
from map_config import KEEP_KEYPOINT_INDICES


# 环境名称映射
ENV_NAMES = {
    '夜多い': 'Night_High',
    '夜少ない': 'Night_Low',
    '昼多い': 'Day_High',
    '昼少ない': 'Day_Low',
}

CAMERAS = ['front', 'left', 'right']


def frame_id(path: Path) -> str:
    """从 SAM3D 单帧 npz 文件名中提取 frame id。"""
    match = re.search(r"(\d+)_sam3d_body\.npz$", path.name)
    return match.group(1) if match else path.stem


def load_sam3d_3d_kpts(file_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    加载单帧 SAM3D 的 3D 关键点结果

    Returns:
        keypoints_3d: (J, 3) - 3D 坐标
        confidence: (J,) - 2D 置信度（作为质量指标）
    """
    with np.load(file_path, allow_pickle=True) as data:
        if 'output' not in data:
            raise KeyError(f"Missing 'output' in {file_path}")

        output = data['output'].item()
        if not isinstance(output, dict):
            raise ValueError(f"Invalid SAM3D output format in {file_path}")

        keypoints_3d = np.asarray(output.get('pred_keypoints_3d'), dtype=np.float32)
        if keypoints_3d.ndim != 2 or keypoints_3d.shape[1] < 3:
            raise ValueError(f"Invalid pred_keypoints_3d shape {keypoints_3d.shape} in {file_path}")

        confidence = output.get('confidence')
        pred_2d = output.get('pred_keypoints_2d')
        if confidence is None and pred_2d is not None:
            pred_2d = np.asarray(pred_2d, dtype=np.float32)
            if pred_2d.ndim == 2 and pred_2d.shape[1] >= 3:
                confidence = pred_2d[:, 2]

        if confidence is None:
            confidence = np.ones(keypoints_3d.shape[0], dtype=np.float32)

    # 应用关键点筛选
    if KEEP_KEYPOINT_INDICES is not None:
        keypoints_3d = keypoints_3d[KEEP_KEYPOINT_INDICES]
        confidence = confidence[KEEP_KEYPOINT_INDICES]

    return keypoints_3d[:, :3], np.asarray(confidence, dtype=np.float32)


def load_sam3d_frame_sequence(view_dir: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    按 frame 文件加载 SAM3D 结果目录。

    Returns:
        keypoints_3d: (T, J, 3)
        confidence: (T, J)
        frame_ids: 文件名中提取的 frame id 列表
    """
    frame_paths = sorted(view_dir.glob("*_sam3d_body.npz"), key=frame_id)
    if not frame_paths:
        raise FileNotFoundError(f"No SAM3D frame npz found in {view_dir}")

    keypoints_list = []
    confidence_list = []
    frame_ids = []

    for path in frame_paths:
        keypoints_3d, confidence = load_sam3d_3d_kpts(path)
        if keypoints_3d.ndim != 2:
            raise ValueError(f"Expected per-frame keypoints with shape (J, 3), got {keypoints_3d.shape} in {path}")
        if confidence.ndim != 1:
            confidence = np.asarray(confidence).reshape(-1)

        n_keypoints = min(keypoints_3d.shape[0], confidence.shape[0])
        keypoints_list.append(keypoints_3d[:n_keypoints, :3])
        confidence_list.append(confidence[:n_keypoints])
        frame_ids.append(frame_id(path))

    min_keypoints = min(kpts.shape[0] for kpts in keypoints_list)
    keypoints_3d = np.stack([kpts[:min_keypoints] for kpts in keypoints_list], axis=0)
    confidence = np.stack([conf[:min_keypoints] for conf in confidence_list], axis=0)

    return keypoints_3d, confidence, frame_ids


def load_triangulated_gt(file_path: Path) -> np.ndarray:
    """
    加载三角化 GT 的 3D 关键点结果

    Returns:
        keypoints_3d: (T, J, 3) - 三角化后的 3D 坐标
    """
    data = np.load(file_path, allow_pickle=True)
    keys = list(data.keys())

    key_3d_key = None
    for k in ['keypoints_3d', 'KPT_3D', 'coords_3d']:
        if k in keys:
            key_3d_key = k
            break

    if not key_3d_key:
        raise ValueError(f"No 3D keypoints key found in {file_path}. Keys: {keys}")

    keypoints_3d = data[key_3d_key]

    # 应用关键点筛选
    if KEEP_KEYPOINT_INDICES is not None:
        keypoints_3d = keypoints_3d[:, KEEP_KEYPOINT_INDICES]

    return keypoints_3d[:, :, :3].astype(np.float32)


def compute_valid_ratio(confidence: np.ndarray, threshold: float = 0.5) -> float:
    """计算有效帧比例（置信度 > threshold 的帧数占比）"""
    valid_frames = np.mean(confidence > threshold, axis=1)
    return np.mean(valid_frames)


def compute_mean_confidence(confidence: np.ndarray) -> float:
    """计算平均置信度"""
    return float(np.mean(confidence))


def compute_keypoint_metrics(
    pred_kpts: np.ndarray,
    gt_kpts: np.ndarray,
    coord_scale: float = 1000.0,
    root_index: int = 0,
    pck_thresholds: Tuple[float, ...] = (50.0, 100.0, 150.0),
) -> Dict[str, Any]:
    """
    计算 SAM3D 3D keypoints 与 GT 3D keypoints 的常见误差指标。

    默认假设输入坐标单位为 meter，并通过 coord_scale 转成 mm。
    """
    n_frames = min(pred_kpts.shape[0], gt_kpts.shape[0])
    n_keypoints = min(pred_kpts.shape[1], gt_kpts.shape[1])
    pred = np.asarray(pred_kpts[:n_frames, :n_keypoints, :3], dtype=np.float32) * coord_scale
    gt = np.asarray(gt_kpts[:n_frames, :n_keypoints, :3], dtype=np.float32) * coord_scale

    valid_mask = np.isfinite(pred).all(axis=-1) & np.isfinite(gt).all(axis=-1)
    if not np.any(valid_mask):
        return {
            'num_frames': int(n_frames),
            'num_keypoints': int(n_keypoints),
            'num_valid_points': 0,
            'mpjpe_mm': 0.0,
            'median_error_mm': 0.0,
            'root_mpjpe_mm': 0.0,
            'pa_mpjpe_mm': 0.0,
            'pck': {str(int(t)): 0.0 for t in pck_thresholds},
            'auc_150': 0.0,
            'per_axis_mae_mm': {'x': 0.0, 'y': 0.0, 'z': 0.0},
            'per_joint_mpjpe_mm': [],
        }

    diff = pred - gt
    errors = np.linalg.norm(diff, axis=-1)
    valid_errors = errors[valid_mask]

    root_index = min(max(root_index, 0), n_keypoints - 1)
    root_pred = pred[:, root_index:root_index + 1, :]
    root_gt = gt[:, root_index:root_index + 1, :]
    root_aligned_errors = np.linalg.norm((pred - root_pred) - (gt - root_gt), axis=-1)
    root_valid_mask = valid_mask & valid_mask[:, root_index:root_index + 1]
    valid_root_errors = root_aligned_errors[root_valid_mask]
    root_mpjpe = float(np.mean(valid_root_errors)) if valid_root_errors.size else 0.0
    pa_mpjpe = compute_pa_mpjpe(pred, gt, valid_mask)

    per_joint_mpjpe = []
    for joint_idx in range(n_keypoints):
        joint_mask = valid_mask[:, joint_idx]
        if np.any(joint_mask):
            per_joint_mpjpe.append(float(np.mean(errors[:, joint_idx][joint_mask])))
        else:
            per_joint_mpjpe.append(0.0)

    abs_diff = np.abs(diff[valid_mask])
    pck = {
        str(int(threshold)): float(np.mean(valid_errors <= threshold))
        for threshold in pck_thresholds
    }

    auc_thresholds = np.linspace(0.0, 150.0, 31)
    auc = float(np.mean([np.mean(valid_errors <= threshold) for threshold in auc_thresholds]))

    return {
        'num_frames': int(n_frames),
        'num_keypoints': int(n_keypoints),
        'num_valid_points': int(np.sum(valid_mask)),
        'mpjpe_mm': float(np.mean(valid_errors)),
        'median_error_mm': float(np.median(valid_errors)),
        'root_mpjpe_mm': root_mpjpe,
        'pa_mpjpe_mm': pa_mpjpe,
        'pck': pck,
        'auc_150': auc,
        'per_axis_mae_mm': {
            'x': float(np.mean(abs_diff[:, 0])),
            'y': float(np.mean(abs_diff[:, 1])),
            'z': float(np.mean(abs_diff[:, 2])),
        },
        'per_joint_mpjpe_mm': per_joint_mpjpe,
    }


def compute_pa_mpjpe(pred: np.ndarray, gt: np.ndarray, valid_mask: np.ndarray) -> float:
    """计算逐帧刚性相似变换对齐后的 MPJPE。"""
    frame_errors = []
    for frame_idx in range(pred.shape[0]):
        mask = valid_mask[frame_idx]
        if np.sum(mask) < 3:
            continue

        pred_frame = pred[frame_idx, mask]
        gt_frame = gt[frame_idx, mask]
        aligned_pred = procrustes_align(pred_frame, gt_frame)
        frame_errors.extend(np.linalg.norm(aligned_pred - gt_frame, axis=-1).tolist())

    return float(np.mean(frame_errors)) if frame_errors else 0.0


def procrustes_align(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """将 source 通过相似变换对齐到 target。"""
    source_mean = np.mean(source, axis=0, keepdims=True)
    target_mean = np.mean(target, axis=0, keepdims=True)
    source_centered = source - source_mean
    target_centered = target - target_mean

    source_norm = np.linalg.norm(source_centered)
    target_norm = np.linalg.norm(target_centered)
    if source_norm < 1e-8 or target_norm < 1e-8:
        return source.copy()

    source_centered /= source_norm
    target_centered /= target_norm

    h = source_centered.T @ target_centered
    u, _, vt = np.linalg.svd(h)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1
        rotation = vt.T @ u.T

    scale = target_norm / source_norm
    return scale * ((source - source_mean) @ rotation) + target_mean


def analyze_subject(
    subject_id: str,
    sam3d_root: Path,
    gt_root: Path,
    coord_scale: float = 1000.0,
    root_index: int = 0,
) -> Dict[str, Any]:
    """分析单个受试者的 3D keypoint 误差。"""

    results = {
        'person_id': subject_id,
        'environments': {},
        'camera_stats': {},
    }

    for env_name in ENV_NAMES.keys():
        env_stats = {}

        # 加载三视角的 SAM3D 结果，并与三角化/合成 3D keypoints 对比
        camera_data = {}
        total_frames = None

        for cam in CAMERAS:
            sam3d_path = sam3d_root / subject_id / env_name / cam
            gt_path = gt_root / subject_id / env_name / "keypoints_3d.npz"

            if not sam3d_path.exists():
                print(f"Warning: SAM3D directory not found: {sam3d_path}")
                continue

            if not gt_path.exists():
                print(f"Warning: GT file not found: {gt_path}")
                continue

            # 加载数据
            try:
                sam3d_kpts, sam3d_conf, sam3d_frame_ids = load_sam3d_frame_sequence(sam3d_path)
                gt_kpts = load_triangulated_gt(gt_path)

                # 统一帧数（取最小值）
                n_frames = min(sam3d_kpts.shape[0], gt_kpts.shape[0])
                n_keypoints = min(sam3d_kpts.shape[1], gt_kpts.shape[1])

                if total_frames is None:
                    total_frames = n_frames

                metrics = compute_keypoint_metrics(
                    sam3d_kpts[:n_frames, :n_keypoints],
                    gt_kpts[:n_frames, :n_keypoints],
                    coord_scale=coord_scale,
                    root_index=root_index,
                )

                camera_data[cam] = {
                    'num_frames': int(n_frames),
                    'num_keypoints': int(n_keypoints),
                    'mean_sam3d_confidence': compute_mean_confidence(
                        sam3d_conf[:n_frames, :n_keypoints]
                    ),
                    'valid_ratio': compute_valid_ratio(
                        sam3d_conf[:n_frames, :n_keypoints]
                    ),
                    'metrics': metrics,
                    'frame_ids': sam3d_frame_ids[:n_frames],
                }

            except Exception as e:
                print(f"Error loading {sam3d_path}: {e}")
                continue

        # 计算总体统计
        env_stats['total_frames'] = total_frames or 0
        env_stats['cameras'] = camera_data

        if camera_data:
            # 平均置信度和有效帧比例只作为辅助参考，不作为 GT 质量指标
            avg_confidence = np.mean([
                data['mean_sam3d_confidence']
                for data in camera_data.values()
            ])
            env_stats['mean_sam3d_confidence'] = float(avg_confidence)

            valid_ratios = [
                data['valid_ratio']
                for data in camera_data.values()
            ]
            env_stats['mean_valid_ratio'] = float(np.mean(valid_ratios))
            env_stats['valid_ratio_range'] = (
                float(min(valid_ratios)),
                float(max(valid_ratios)),
            )

            metric_names = ['mpjpe_mm', 'median_error_mm', 'root_mpjpe_mm', 'pa_mpjpe_mm', 'auc_150']
            for metric_name in metric_names:
                env_stats[metric_name] = float(np.mean([
                    data['metrics'][metric_name]
                    for data in camera_data.values()
                ]))

            env_stats['pck'] = {}
            for threshold in ['50', '100', '150']:
                env_stats['pck'][threshold] = float(np.mean([
                    data['metrics']['pck'][threshold]
                    for data in camera_data.values()
                ]))

        results['environments'][env_name] = env_stats

    return results


def generate_summary_statistics(all_subjects: List[Dict[str, Any]]) -> Dict[str, Any]:
    """生成汇总统计信息"""

    summary = {
        'total_subjects': len(all_subjects),
        'environment_summary': {},
        'camera_comparison': defaultdict(list),
    }

    for subject in all_subjects:
        envs = subject.get('environments', {})
        for env_name, stats in envs.items():
            if stats.get('total_frames', 0) == 0:
                continue

            # 初始化环境统计
            if env_name not in summary['environment_summary']:
                summary['environment_summary'][env_name] = {
                    'subjects': [],
                    'avg_frames': [],
                    'valid_ratios': [],
                    'mpjpe_mm': [],
                    'median_error_mm': [],
                    'root_mpjpe_mm': [],
                    'pa_mpjpe_mm': [],
                    'pck_50': [],
                    'pck_100': [],
                    'pck_150': [],
                    'auc_150': [],
                }

            summary['environment_summary'][env_name]['subjects'].append(subject['person_id'])
            summary['environment_summary'][env_name]['avg_frames'].append(stats.get('total_frames', 0))

            if 'mean_valid_ratio' in stats:
                summary['environment_summary'][env_name]['valid_ratios'].append(
                    stats['mean_valid_ratio']
                )

            for metric_name in ['mpjpe_mm', 'median_error_mm', 'root_mpjpe_mm', 'pa_mpjpe_mm', 'auc_150']:
                if metric_name in stats:
                    summary['environment_summary'][env_name][metric_name].append(stats[metric_name])

            for threshold in ['50', '100', '150']:
                if threshold in stats.get('pck', {}):
                    summary['environment_summary'][env_name][f'pck_{threshold}'].append(
                        stats['pck'][threshold]
                    )

            # 相机对比
            for cam in CAMERAS:
                if cam in stats.get('cameras', {}):
                    summary['camera_comparison'][cam].append(stats['cameras'][cam])

    # 计算汇总统计
    for env_name, env_data in summary['environment_summary'].items():
        env_data['avg_valid_ratio'] = (
            np.mean(env_data['valid_ratios']) if env_data['valid_ratios'] else 0
        )
        for metric_name in ['mpjpe_mm', 'median_error_mm', 'root_mpjpe_mm', 'pa_mpjpe_mm', 'auc_150']:
            env_data[f'avg_{metric_name}'] = (
                np.mean(env_data[metric_name]) if env_data[metric_name] else 0
            )
        for threshold in ['50', '100', '150']:
            key = f'pck_{threshold}'
            env_data[f'avg_{key}'] = np.mean(env_data[key]) if env_data[key] else 0
        env_data['subjects'] = ','.join(env_data['subjects'])[:50]  # 截断显示

    return summary


def print_report(all_subjects: List[Dict[str, Any]], summary: Dict[str, Any]):
    """打印详细报告"""

    print("\n" + "=" * 80)
    print("SAM3D vs Triangulated GT - 3D Keypoint Comparison Report")
    print("=" * 80)

    # 总览
    print(f"\nTOTAL SUBJECTS: {summary['total_subjects']}")

    # 环境对比
    print("\n" + "=" * 40)
    print("ENVIRONMENT SUMMARY")
    print("=" * 40)
    print(
        f"{'Environment':<20} {'Avg Frames':>12} {'MPJPE(mm)':>12} "
        f"{'Root MPJPE':>12} {'PA-MPJPE':>10} {'PCK@100':>10} {'AUC@150':>10}"
    )
    print("-" * 86)

    for env_name in sorted(summary['environment_summary'].keys()):
        data = summary['environment_summary'][env_name]
        frames = int(np.mean(data.get('avg_frames', [0])))
        mpjpe = data.get('avg_mpjpe_mm', 0)
        root_mpjpe = data.get('avg_root_mpjpe_mm', 0)
        pa_mpjpe = data.get('avg_pa_mpjpe_mm', 0)
        pck_100 = data.get('avg_pck_100', 0)
        auc_150 = data.get('avg_auc_150', 0)

        print(
            f"{ENV_NAMES.get(env_name, env_name):<20} {frames:>12} "
            f"{mpjpe:>12.2f} {root_mpjpe:>12.2f} "
            f"{pa_mpjpe:>10.2f} {pck_100:>10.4f} {auc_150:>10.4f}"
        )

    # 相机对比
    print("\n" + "=" * 40)
    print("CAMERA COMPARISON")
    print("=" * 40)
    print(
        f"{'Camera':<15} {'Samples':>10} {'MPJPE(mm)':>12} "
        f"{'Root MPJPE':>12} {'PA-MPJPE':>10} {'PCK@100':>10}"
    )
    print("-" * 78)

    for cam in CAMERAS:
        samples = len(summary['camera_comparison'].get(cam, []))
        cam_stats = summary['camera_comparison'].get(cam, [])
        mpjpes = [s['metrics']['mpjpe_mm'] for s in cam_stats]
        root_mpjpes = [s['metrics']['root_mpjpe_mm'] for s in cam_stats]
        pa_mpjpes = [s['metrics']['pa_mpjpe_mm'] for s in cam_stats]
        pck_100 = [s['metrics']['pck']['100'] for s in cam_stats]

        avg_mpjpe = np.mean(mpjpes) if mpjpes else 0
        avg_root_mpjpe = np.mean(root_mpjpes) if root_mpjpes else 0
        avg_pa_mpjpe = np.mean(pa_mpjpes) if pa_mpjpes else 0
        avg_pck_100 = np.mean(pck_100) if pck_100 else 0

        print(
            f"{cam:<15} {samples:>10} {avg_mpjpe:>12.2f} "
            f"{avg_root_mpjpe:>12.2f} {avg_pa_mpjpe:>10.2f} {avg_pck_100:>10.4f}"
        )


def create_visualizations(
    all_subjects: List[Dict[str, Any]],
    output_dir: Path,
    summary: Dict[str, Any] = None,
):
    """生成可视化图表"""

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. MPJPE 箱线图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('SAM3D vs Triangulated GT - MPJPE by Environment', fontsize=14)

    env_names = ['夜多い', '夜少ない', '昼多い', '昼少ない']

    for idx, env_name in enumerate(env_names):
        ax = axes.flatten()[idx]

        data_list = []
        for subject in all_subjects:
            env_stats = subject.get('environments', {}).get(env_name, {})
            if 'mpjpe_mm' in env_stats:
                data_list.append(env_stats['mpjpe_mm'])

        if data_list:
            bp = ax.boxplot(data_list, vert=True, patch_artist=True)
            for box in bp['boxes']:
                box.set_facecolor('lightblue')

            ax.set_ylabel('MPJPE (mm)', fontsize=10)
            ax.set_title(ENV_NAMES.get(env_name, env_name), fontsize=12)
            ax.tick_params(axis='both', which='major', labelsize=10)
        else:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=14)

    plt.tight_layout()
    plt.savefig(output_dir / 'mpjpe_boxplot.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 2. 环境对比柱状图
    env_summary = summary['environment_summary']

    fig, ax = plt.subplots(figsize=(10, 6))

    plot_envs = [e for e in env_names if e in env_summary]
    mpjpes = [
        env_summary[e].get('avg_mpjpe_mm', 0) for e in plot_envs
    ]
    root_mpjpes = [
        env_summary[e].get('avg_root_mpjpe_mm', 0) for e in plot_envs
    ]

    x = np.arange(len(plot_envs))
    width = 0.35

    ax.bar(x - width/2, mpjpes, width, label='MPJPE', color='skyblue')
    ax.bar(x + width/2, root_mpjpes, width, label='Root MPJPE', color='coral')

    ax.set_xlabel('Environment', fontsize=12)
    ax.set_ylabel('Error (mm)', fontsize=12)
    ax.set_title('3D Keypoint Error by Environment', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([ENV_NAMES.get(e, e) for e in plot_envs])
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'environment_error_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 3. 相机对比散点图：MPJPE vs PCK@100
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for cam_idx, cam in enumerate(CAMERAS):
        ax = axes[cam_idx]

        mpjpes = [
            s['metrics']['mpjpe_mm'] for s in summary['camera_comparison'].get(cam, [])
        ]
        pck_100 = [
            s['metrics']['pck']['100'] for s in summary['camera_comparison'].get(cam, [])
        ]

        if mpjpes and pck_100:
            ax.scatter(mpjpes, pck_100, alpha=0.6, edgecolors='black')
            ax.set_xlabel('MPJPE (mm)', fontsize=10)
            ax.set_ylabel('PCK@100', fontsize=10)
            ax.set_title(f'{cam} Camera', fontsize=12)
        else:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=14)

    plt.tight_layout()
    plt.savefig(output_dir / 'camera_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nVisualization saved to {output_dir}")


def save_results_by_subject_env(
    all_subjects: List[Dict[str, Any]],
    summary: Dict[str, Any],
    output_dir: Path,
):
    """按 person/env 组织保存结果，同时保留一份全局汇总。"""
    def _mm_to_m(value: Any) -> Any:
        if not isinstance(value, (int, float)):
            return value
        return float(value) / 1000.0

    def _pck_keys_to_decimal(pck_dict: Dict[str, Any]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for k, v in pck_dict.items():
            try:
                key_m = f"{float(k) / 1000.0:.2f}"
                out[key_m] = float(v)
            except (TypeError, ValueError):
                continue
        return out

    def _convert_camera_payload(cameras: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for cam_name, cam_stats in cameras.items():
            cam_metrics = dict(cam_stats.get('metrics', {}))
            cam_pck = _pck_keys_to_decimal(cam_metrics.get('pck', {}))
            out[cam_name] = {
                'num_frames': cam_stats.get('num_frames'),
                'num_keypoints': cam_stats.get('num_keypoints'),
                'mean_sam3d_confidence': cam_stats.get('mean_sam3d_confidence'),
                'valid_ratio': cam_stats.get('valid_ratio'),
                'metrics': {
                    'num_frames': cam_metrics.get('num_frames'),
                    'num_keypoints': cam_metrics.get('num_keypoints'),
                    'num_valid_points': cam_metrics.get('num_valid_points'),
                    'mpjpe_m': _mm_to_m(cam_metrics.get('mpjpe_mm')),
                    'median_error_m': _mm_to_m(cam_metrics.get('median_error_mm')),
                    'root_mpjpe_m': _mm_to_m(cam_metrics.get('root_mpjpe_mm')),
                    'pa_mpjpe_m': _mm_to_m(cam_metrics.get('pa_mpjpe_mm')),
                    'pck': cam_pck,
                    'auc_0.15': cam_metrics.get('auc_150'),
                    'per_axis_mae_m': {
                        'x': _mm_to_m(cam_metrics.get('per_axis_mae_mm', {}).get('x')),
                        'y': _mm_to_m(cam_metrics.get('per_axis_mae_mm', {}).get('y')),
                        'z': _mm_to_m(cam_metrics.get('per_axis_mae_mm', {}).get('z')),
                    },
                    'per_joint_mpjpe_m': [
                        _mm_to_m(v) for v in cam_metrics.get('per_joint_mpjpe_mm', [])
                    ],
                },
                'frame_ids': cam_stats.get('frame_ids', []),
            }
        return out

    output_dir.mkdir(parents=True, exist_ok=True)

    subjects_out: List[Dict[str, Any]] = []

    for subject in all_subjects:
        subject_id = subject['person_id']
        subject_out = {'person_id': subject_id, 'environments': {}}
        for env_name, env_stats in subject.get('environments', {}).items():
            if env_stats.get('total_frames', 0) == 0:
                continue

            env_dir = output_dir / subject_id / ENV_NAMES.get(env_name, env_name)
            env_dir.mkdir(parents=True, exist_ok=True)

            env_payload = {
                'person_id': subject_id,
                'environment': env_name,
                'environment_name': ENV_NAMES.get(env_name, env_name),
                'total_frames': env_stats.get('total_frames', 0),
                'mean_sam3d_confidence': env_stats.get('mean_sam3d_confidence'),
                'mean_valid_ratio': env_stats.get('mean_valid_ratio'),
                'valid_ratio_range': env_stats.get('valid_ratio_range'),
                'metrics': {
                    'mpjpe_m': _mm_to_m(env_stats.get('mpjpe_mm')),
                    'median_error_m': _mm_to_m(env_stats.get('median_error_mm')),
                    'root_mpjpe_m': _mm_to_m(env_stats.get('root_mpjpe_mm')),
                    'pa_mpjpe_m': _mm_to_m(env_stats.get('pa_mpjpe_mm')),
                    'pck': _pck_keys_to_decimal(env_stats.get('pck', {})),
                    'auc_0.15': env_stats.get('auc_150'),
                },
                'cameras': _convert_camera_payload(env_stats.get('cameras', {})),
            }

            with open(env_dir / 'metrics.json', 'w') as f:
                json.dump(env_payload, f, indent=2)

            subject_out['environments'][env_name] = env_payload

        subjects_out.append(subject_out)

    with open(output_dir / 'comparison_data.json', 'w') as f:
        json.dump({
            'subjects': subjects_out,
            'summary': {
                k: {
                    'subjects': v['subjects'],
                    'avg_mpjpe_m': _mm_to_m(v.get('avg_mpjpe_mm', 0)),
                    'avg_median_error_m': _mm_to_m(v.get('avg_median_error_mm', 0)),
                    'avg_root_mpjpe_m': _mm_to_m(v.get('avg_root_mpjpe_mm', 0)),
                    'avg_pa_mpjpe_m': _mm_to_m(v.get('avg_pa_mpjpe_mm', 0)),
                    'avg_pck_0.05': float(v.get('avg_pck_50', 0)),
                    'avg_pck_0.10': float(v.get('avg_pck_100', 0)),
                    'avg_pck_0.15': float(v.get('avg_pck_150', 0)),
                    'avg_auc_0.15': float(v.get('avg_auc_150', 0)),
                }
                for k, v in summary['environment_summary'].items()
            }
        }, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description='Compare SAM3D single-view results with triangulated GT'
    )
    parser.add_argument(
        '--sam3d-root',
        type=str,
        default='/home/data/xchen/drive/sam3d_body_results_right',
        help='Path to SAM3D original results directory'
    )
    parser.add_argument(
        '--gt-root',
        type=str,
        default='/home/data/xchen/drive/sam3d_body_triangulated_gt',
        help='Path to triangulated GT directory'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='logs/comparison_sam3d_3dkpt_with_pesudo_gt',
        help='Output directory for reports and visualizations'
    )
    parser.add_argument(
        '--subject',
        type=str,
        default=None,
        help='Optional: analyze specific subject (e.g., "01")'
    )
    parser.add_argument(
        '--coord-scale',
        type=float,
        default=1000.0,
        help='Scale factor applied before reporting errors in mm. Use 1000 for meters, 1 for millimeters.'
    )
    parser.add_argument(
        '--root-index',
        type=int,
        default=0,
        help='Root joint index used for root-aligned MPJPE.'
    )

    args = parser.parse_args()

    sam3d_root = Path(args.sam3d_root)
    gt_root = Path(args.gt_root)
    output_dir = Path(args.output_dir)

    if not sam3d_root.exists():
        print(f"Error: SAM3D root directory does not exist: {sam3d_root}")
        return
    if not gt_root.exists():
        print(f"Error: GT root directory does not exist: {gt_root}")
        return

    # 获取所有受试者 ID
    subjects = sorted([p.name for p in sam3d_root.iterdir() if p.is_dir()])
    if args.subject:
        subjects = [s for s in subjects if s == args.subject]

    print(f"Analyzing {len(subjects)} subjects...")

    # 分析每个受试者
    all_subjects = []
    for subject_id in subjects:
        print(f"Processing subject {subject_id}...")
        result = analyze_subject(
            subject_id,
            sam3d_root,
            gt_root,
            coord_scale=args.coord_scale,
            root_index=args.root_index,
        )
        all_subjects.append(result)

    # 生成汇总统计
    summary = generate_summary_statistics(all_subjects)

    # 打印报告
    print_report(all_subjects, summary)

    # 生成可视化
    create_visualizations(all_subjects, output_dir, summary)

    # 保存详细数据：根目录保留全局汇总，同时按 person/env 拆分 metrics.json
    save_results_by_subject_env(all_subjects, summary, output_dir)

    print(f"\nDetailed results saved to: {output_dir / 'comparison_data.json'}")
    print(f"Per-subject environment metrics saved under: {output_dir / '<person_id>' / '<env>' / 'metrics.json'}")


if __name__ == '__main__':
    main()
