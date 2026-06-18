# -*- coding: utf-8 -*-
"""공용 LiDAR 헬퍼 — PointCloud2 ↔ numpy 변환.

potential·path_planning·ground_division·tank_visual_perception·rviz_visualization 등
여러 패키지가 동일 구현을 각자 복붙해 쓰던 것을 단일 출처로 통합한 모듈이다.
"""
from __future__ import annotations

import numpy as np
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2


def pointcloud2_to_xyz_array(msg: PointCloud2) -> np.ndarray:
    """PointCloud2의 XYZ 필드를 연속(contiguous) float32 (N, 3) 배열로 반환한다.

    ROS2 Humble 이상의 sensor_msgs_py는 read_points_numpy()를 제공해 LiDAR 점마다
    Python dict/list 객체를 만드는 비용을 피한다. fallback은 구버전 sensor_msgs_py에서도
    노드가 동작하도록 유지하기 위한 것이다.
    """
    try:
        arr = point_cloud2.read_points_numpy(
            msg, field_names=("x", "y", "z"), skip_nans=True
        )
    except Exception:
        pts = point_cloud2.read_points(
            msg, field_names=("x", "y", "z"), skip_nans=True
        )
        if isinstance(pts, np.ndarray):
            arr = pts
        else:
            arr = np.asarray(list(pts), dtype=np.float32)
    if arr is None:
        return np.empty((0, 3), dtype=np.float32)
    arr = np.asarray(arr)
    if arr.dtype.fields:
        arr = np.column_stack((arr["x"], arr["y"], arr["z"]))
    arr = np.asarray(arr, dtype=np.float32)
    if arr.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    return np.ascontiguousarray(arr.reshape(-1, 3), dtype=np.float32)
