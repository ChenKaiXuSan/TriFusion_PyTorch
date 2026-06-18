#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/MultiView_DriverAction_PyTorch/TriPoseFusion/map_config.py
Project: /workspace/MultiView_DriverAction_PyTorch/TriPoseFusion
Created Date: Sunday January 25th 2026
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Sunday January 25th 2026 9:48:24 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2026 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""

from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


environment_mapping_Dict: Dict = {
    0: "夜多い",  # night_high
    1: "夜少ない",  # night_low
    2: "昼多い",  # day_high
    3: "昼少ない",  # day_low
}


# 反向映射：label文件名里的 (day/night, high/low) -> 文件夹名
ENV_KEY_TO_FOLDER = {
    ("night", "high"): "夜多い",
    ("night", "low"): "夜少ない",
    ("day", "high"): "昼多い",
    ("day", "low"): "昼少ない",
}

# 你期望的相机视频文件（按需增减）
CAM_NAMES = ["front", "right", "left"]


@dataclass
class VideoSample:
    person_id: str  # "01"
    env_folder: str  # "夜多"
    env_key: str  # "night_high"
    videos: Dict[str, Path]  # {"front": ..., "right": ..., "left": ...}
    label_path: Path | None = None
    sam3d_kpts: Dict[str, Path] = (
        None  # {"front": ..., "right": ..., "left": ...} SAM 3D body keypoints directory
    )


# 定义需要保留的关键点索引：头部 + 肩部/颈部 + 双手
KEEP_KEYPOINT_INDICES = (
    # 头部: 鼻子、眼睛、耳朵
    list(range(0, 5))  # 0-4: nose, left-eye, right-eye, left-ear, right-ear
    # 肩部和颈部
    + [5, 6]  # left-shoulder, right-shoulder
    # 双手（包括手腕）
    + list(range(21, 63))  # 21-62: 右手(21-41) + 左手(42-62)
    # 肩峰和颈部
    + [67, 68, 69]  # left-acromion, right-acromion, neck
)
