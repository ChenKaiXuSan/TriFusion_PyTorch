#!/usr/bin/env python3
"""
比较 SAM3D 原始三视角结果和三角化 GT 的数据质量

脚本功能:
1. 统计三个视角的 SAM3D 单目检测结果的质量指标
2. 对比三角化后的 3D 关键点质量
3. 分析不同环境（昼/夜、多/少）下的数据质量差异
4. 生成详细报告，包括重投影误差、有效帧比例等

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


def load_triangulated_gt(file_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    加载三角化 GT 的 3D 关键点结果

    Returns:
        keypoints_3d: (T, J, 3) - 三角化后的 3D 坐标
        quality_score: (T,) - 质量分数（基于重投影误差）
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

    # 质量分数键名：'quality_score', 'reproj_error', 'validity'
    quality_key = None
    for k in ['quality_score', 'reproj_error', 'validity', 'is_valid']:
        if k in keys:
            quality_key = k
            break

    if not quality_key:
        print(f"Warning: No quality score found in {file_path}")
        quality_score = np.ones(keypoints_3d.shape[0])
    else:
        quality_score = data[quality_key]

    # 应用关键点筛选
    if KEEP_KEYPOINT_INDICES is not None:
        keypoints_3d = keypoints_3d[:, KEEP_KEYPOINT_INDICES]
        quality_score = quality_score[:, KEEP_KEYPOINT_INDICES]

    return keypoints_3d, quality_score


def compute_valid_ratio(confidence: np.ndarray, threshold: float = 0.5) -> float:
    """计算有效帧比例（置信度 > threshold 的帧数占比）"""
    valid_frames = np.mean(confidence > threshold, axis=1)
    return np.mean(valid_frames)


def compute_mean_confidence(confidence: np.ndarray) -> float:
    """计算平均置信度"""
    return float(np.mean(confidence))


def compute_pose_std(keypoints_3d: np.ndarray) -> Tuple[float, float, float]:
    """
    计算人体姿态的标准差，作为运动幅度的指标

    Returns:
        x_std, y_std, z_std - 三个坐标轴上的标准差（mm 级别）
    """
    # 计算整体位移
    pose_center = np.mean(keypoints_3d, axis=(0, 1))
    deviations = keypoints_3d - pose_center

    return (
        float(np.std(deviations[..., 0])),
        float(np.std(deviations[..., 1])),
        float(np.std(deviations[..., 2])),
    )


def analyze_subject(
    subject_id: str,
    sam3d_root: Path,
    gt_root: Path,
) -> Dict[str, Any]:
    """分析单个受试者的数据质量"""

    results = {
        'person_id': subject_id,
        'environments': {},
        'camera_stats': {},
    }

    for env_name in ENV_NAMES.keys():
        env_stats = {}

        # 加载三视角的 SAM3D 结果
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
                gt_kpts, gt_quality = load_triangulated_gt(gt_path)

                # 统一帧数（取最小值）
                n_frames = min(sam3d_kpts.shape[0], gt_kpts.shape[0])

                if total_frames is None:
                    total_frames = n_frames

                camera_data[cam] = {
                    'sam3d_kpts': sam3d_kpts[:n_frames],
                    'sam3d_conf': sam3d_conf[:n_frames],
                    'gt_kpts': gt_kpts[:n_frames],
                    'gt_quality': gt_quality[:n_frames],
                    'frame_ids': sam3d_frame_ids[:n_frames],
                }

            except Exception as e:
                print(f"Error loading {sam3d_path}: {e}")
                continue

        # 计算总体统计
        env_stats['total_frames'] = total_frames or 0
        env_stats['cameras'] = camera_data

        if camera_data:
            # 平均置信度
            avg_confidence = np.mean([
                compute_mean_confidence(data['sam3d_conf'])
                for data in camera_data.values()
            ])
            env_stats['mean_sam3d_confidence'] = avg_confidence

            # 有效帧比例
            valid_ratios = [
                compute_valid_ratio(data['sam3d_conf'])
                for data in camera_data.values()
            ]
            env_stats['mean_valid_ratio'] = np.mean(valid_ratios)
            env_stats['valid_ratio_range'] = (min(valid_ratios), max(valid_ratios))

            # GT 重投影误差（如果有）
            if 'gt_quality' in camera_data[list(camera_data.keys())[0]]:
                mean_reproj_error = np.mean(1 - camera_data[list(camera_data.keys())[0]]['gt_quality'])
                env_stats['mean_triangulation_error'] = float(mean_reproj_error)

            # 姿态标准差（运动幅度）
            first_cam = list(camera_data.values())[0]
            x_std, y_std, z_std = compute_pose_std(first_cam['sam3d_kpts'])
            env_stats['pose_std_mm'] = {'x': x_std * 1000, 'y': y_std * 1000, 'z': z_std * 1000}

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
                    'reproj_errors': [],
                }

            summary['environment_summary'][env_name]['subjects'].append(subject['person_id'])
            summary['environment_summary'][env_name]['avg_frames'].append(stats.get('total_frames', 0))

            if 'mean_valid_ratio' in stats:
                summary['environment_summary'][env_name]['valid_ratios'].append(
                    stats['mean_valid_ratio']
                )

            if 'mean_triangulation_error' in stats:
                summary['environment_summary'][env_name]['reproj_errors'].append(
                    stats['mean_triangulation_error']
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
        env_data['avg_reproj_error'] = (
            np.mean(env_data['reproj_errors']) if env_data['reproj_errors'] else 0
        )
        env_data['subjects'] = ','.join(env_data['subjects'])[:50]  # 截断显示

    return summary


def print_report(all_subjects: List[Dict[str, Any]], summary: Dict[str, Any]):
    """打印详细报告"""

    print("\n" + "=" * 80)
    print("SAM3D vs Triangulated GT - Quality Comparison Report")
    print("=" * 80)

    # 总览
    print(f"\n📊 TOTAL SUBJECTS: {summary['total_subjects']}")

    # 环境对比
    print("\n" + "=" * 40)
    print("ENVIRONMENT SUMMARY")
    print("=" * 40)
    print(f"{'Environment':<20} {'Avg Frames':>12} {'Valid Ratio':>12} {'Reproj Error (px)':>18}")
    print("-" * 62)

    for env_name in sorted(summary['environment_summary'].keys()):
        data = summary['environment_summary'][env_name]
        frames = int(np.mean(data.get('avg_frames', [0])))
        valid_ratio = data.get('avg_valid_ratio', 0)
        reproj_error = data.get('avg_reproj_error', 0)

        print(f"{ENV_NAMES.get(env_name, env_name):<20} {frames:>12} {valid_ratio:>12.4f} {reproj_error:>18.4f}")

    # 相机对比
    print("\n" + "=" * 40)
    print("CAMERA COMPARISON")
    print("=" * 40)
    print(f"{'Camera':<15} {'Samples':>10} {'Valid Ratio':>15} {'Confidence':>15}")
    print("-" * 57)

    for cam in CAMERAS:
        samples = len(summary['camera_comparison'].get(cam, []))
        valid_ratios = [s.get('mean_valid_ratio', 0) for s in summary['camera_comparison'].get(cam, [])]
        confidences = [np.mean(s.get('sam3d_conf', np.ones(1))) for s in summary['camera_comparison'].get(cam, [])]

        avg_valid = np.mean(valid_ratios) if valid_ratios else 0
        avg_conf = np.mean(confidences) if confidences else 0

        print(f"{cam:<15} {samples:>10} {avg_valid:>15.4f} {avg_conf:>15.6f}")


def create_visualizations(
    all_subjects: List[Dict[str, Any]],
    output_dir: Path,
    summary: Dict[str, Any] = None,
):
    """生成可视化图表"""

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 重投影误差箱线图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Triangulation Quality Analysis', fontsize=14)

    env_names = ['夜多い', '夜少ない', '昼多い', '昼少ない']

    for idx, env_name in enumerate(env_names):
        ax = axes.flatten()[idx]

        data_list = []
        for subject in all_subjects:
            env_stats = subject.get('environments', {}).get(env_name, {})
            if 'mean_triangulation_error' in env_stats:
                error = env_stats['mean_triangulation_error'] * 1000  # 转换为像素
                data_list.append(error)

        if data_list:
            bp = ax.boxplot(data_list, vert=True, patch_artist=True)
            for box in bp['boxes']:
                box.set_facecolor('lightblue')

            ax.set_ylabel('Reproj Error (scaled)', fontsize=10)
            ax.set_title(ENV_NAMES.get(env_name, env_name), fontsize=12)
            ax.tick_params(axis='both', which='major', labelsize=10)
        else:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=14)

    plt.tight_layout()
    plt.savefig(output_dir / 'reproj_error_boxplot.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 2. 环境对比柱状图
    env_summary = summary['environment_summary']

    fig, ax = plt.subplots(figsize=(10, 6))

    valid_ratios = [
        env_summary[e].get('avg_valid_ratio', 0) for e in env_names if e in env_summary
    ]
    reproj_errors = [
        env_summary[e].get('avg_reproj_error', 0) * 1000 for e in env_names if e in env_summary
    ]

    x = np.arange(len(env_names))
    width = 0.35

    bars1 = ax.bar(x - width/2, valid_ratios, width, label='Valid Ratio', color='skyblue')
    bars2 = ax.bar(x + width/2, reproj_errors, width, label='Reproj Error (x1000)', color='coral')

    ax.set_xlabel('Environment', fontsize=12)
    ax.set_ylabel('Metrics', fontsize=12)
    ax.set_title('Quality Metrics by Environment', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([ENV_NAMES.get(e, e) for e in env_names])
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'environment_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 3. 相机对比散点图
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for cam_idx, cam in enumerate(CAMERAS):
        ax = axes[cam_idx]

        valid_ratios = [
            s.get('mean_valid_ratio', 0) for s in summary['camera_comparison'].get(cam, [])
        ]
        confidences = [
            np.mean(s.get('sam3d_conf', np.ones(1))) for s in summary['camera_comparison'].get(cam, [])
        ]

        if valid_ratios and confidences:
            ax.scatter(valid_ratios, confidences, alpha=0.6, edgecolors='black')
            ax.set_xlabel('Valid Ratio', fontsize=10)
            ax.set_ylabel('Mean Confidence', fontsize=10)
            ax.set_title(f'{cam} Camera', fontsize=12)
        else:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=14)

    plt.tight_layout()
    plt.savefig(output_dir / 'camera_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nVisualization saved to {output_dir}")


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
        result = analyze_subject(subject_id, sam3d_root, gt_root)
        all_subjects.append(result)

    # 生成汇总统计
    summary = generate_summary_statistics(all_subjects)

    # 打印报告
    print_report(all_subjects, summary)

    # 生成可视化
    create_visualizations(all_subjects, output_dir, summary)

    # 保存详细数据
    with open(output_dir / 'comparison_data.json', 'w') as f:
        json.dump({
            'subjects': all_subjects,
            'summary': {
                k: {
                    'subjects': v['subjects'],
                    'avg_valid_ratio': v.get('avg_valid_ratio'),
                    'avg_reproj_error': v.get('avg_reproj_error'),
                }
                for k, v in summary['environment_summary'].items()
            }
        }, f, indent=2)

    print(f"\nDetailed results saved to: {output_dir / 'comparison_data.json'}")


if __name__ == '__main__':
    main()
