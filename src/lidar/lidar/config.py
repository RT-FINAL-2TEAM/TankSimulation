# -*- coding: utf-8 -*-
"""Global LiDAR configuration values.

이 파일에는 시뮬레이션 환경이나 launch parameter로 바꿀 수 있는
LiDAR 관련 기본값만 모아둔다. 다른 패키지는 LiDAR raw schema를 직접
해석하지 말고 lidar.payloads / lidar.path_blocking 함수를 사용한다.
"""

from __future__ import annotations

import os


def env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except Exception:
        return default


def env_int(name: str, default: int) -> int:
    try:
        return int(float(os.environ.get(name, str(default))))
    except Exception:
        return default


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


MAP_FRAME = os.environ.get("TANK_MAP_FRAME", "tank_map")
UNITY_FRAME = os.environ.get("TANK_UNITY_FRAME", "tank_unity_raw")

TOPIC_INFO_RAW = os.environ.get("TANK_TOPIC_INFO_RAW", "/tank/api/info/raw")
TOPIC_LIDAR_POINTS = os.environ.get("TANK_TOPIC_LIDAR_POINTS", "/tank/sensor/lidar/points")
TOPIC_LIDAR_POINTS_COUNT = os.environ.get("TANK_TOPIC_LIDAR_POINTS_COUNT", "/tank/sensor/lidar/points_count")
TOPIC_LIDAR_ORIGIN = os.environ.get("TANK_TOPIC_LIDAR_ORIGIN", "/tank/sensor/lidar/origin")
TOPIC_LIDAR_ORIGIN_RAW = os.environ.get("TANK_TOPIC_LIDAR_ORIGIN_RAW", "/tank/sensor/lidar/origin_raw")
TOPIC_LIDAR_ROTATION = os.environ.get("TANK_TOPIC_LIDAR_ROTATION", "/tank/sensor/lidar/rotation")
TOPIC_LIDAR_DETECTED_MAP = os.environ.get("TANK_TOPIC_LIDAR_DETECTED_MAP", "/tank/sensor/lidar/detected_points_map")
TOPIC_LIDAR_ALL_DETECTED_MAP = os.environ.get("TANK_TOPIC_LIDAR_ALL_DETECTED_MAP", "/tank/sensor/lidar/all_detected_points_map")
TOPIC_LIDAR_TERRAIN_MAP = os.environ.get("TANK_TOPIC_LIDAR_TERRAIN_MAP", "/tank/sensor/lidar/terrain_points_map")
TOPIC_TERRAIN_INFO = os.environ.get("TANK_TOPIC_TERRAIN_INFO", "/tank/perception/terrain_info")

GROUND_FILTER_ENABLED = env_bool("TANK_LIDAR_GROUND_FILTER_ENABLED", True)
DEFAULT_LIDAR_ORIGIN_Y = env_float("TANK_LIDAR_DEFAULT_ORIGIN_Y", 8.0)

# Terrain/obstacle separation defaults. These came from the terrain development branch
# and are enabled by default so the existing run command can use the feature immediately.
TERRAIN_GRID_RESOLUTION = env_float("TANK_TERRAIN_GRID_RESOLUTION", 0.5)
TERRAIN_CLIMB_LIMIT = env_float("TANK_TERRAIN_CLIMB_LIMIT", 0.4)
TERRAIN_OBSTACLE_MIN_HEIGHT = env_float("TANK_TERRAIN_OBSTACLE_MIN_HEIGHT", 0.2)

# Dynamic-replanning / path-block check defaults. The planner imports these
# defaults, but the actual ROS2 parameters are still declared inside the planner.
PATH_BLOCK_MIN_DISTANCE = env_float("TANK_LIDAR_BLOCK_MIN_DISTANCE", 4.0)
PATH_BLOCK_MAX_DISTANCE = env_float("TANK_LIDAR_BLOCK_MAX_DISTANCE", 80.0)
CLUSTER_EPS = env_float("TANK_LIDAR_CLUSTER_EPS", 2.0)
CLUSTER_MIN_SAMPLES = env_int("TANK_LIDAR_CLUSTER_MIN_SAMPLES", 3)
HISTORY_RESOLUTION = env_float("TANK_LIDAR_HISTORY_RESOLUTION", 0.5)
MAX_HISTORY_POINTS = env_int("TANK_LIDAR_MAX_HISTORY_POINTS", 1500)
BBOX_MIN_THICKNESS = env_float("TANK_LIDAR_BBOX_MIN_THICKNESS", 1.0)
