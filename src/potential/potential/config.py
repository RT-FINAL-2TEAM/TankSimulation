# -*- coding: utf-8 -*-
"""전역 APF / 포텐셜 필드 설정값."""

from __future__ import annotations

import os

from lidar.config import TOPIC_LIDAR_DETECTED_MAP


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
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


MAP_FRAME = os.environ.get("TANK_MAP_FRAME", "tank_map")
PLAYER_POSE_TOPIC = os.environ.get("TANK_TOPIC_PLAYER_POSE", "/tank/player/pose")
TARGET_POSE_TOPIC = os.environ.get("TANK_GOAL_POSE_TOPIC", "/tank/path/lookahead_pose")
FALLBACK_GOAL_TOPIC = os.environ.get("TANK_FALLBACK_GOAL_TOPIC", "/tank/goal/pose")
LIDAR_POINTS_TOPIC = TOPIC_LIDAR_DETECTED_MAP
DISCOVERED_OBJECTS_TOPIC = os.environ.get("TANK_DISCOVERED_OBJECTS_TOPIC", "/tank/map/discovered/objects")

APF_HZ = env_float("TANK_APF_HZ", 10.0)
K_ATTRACTIVE = env_float("TANK_APF_K_ATT", 3.0)
K_REPULSIVE = env_float("TANK_APF_K_REP", 90.0)
OBSTACLE_INFLUENCE_RADIUS = env_float("TANK_APF_G_STAR", 10.0)
MAX_ATTRACTIVE_FORCE = env_float("TANK_APF_MAX_ATT_FORCE", 20.0)
MAX_REPULSIVE_FORCE = env_float("TANK_APF_MAX_REP_FORCE", 25.0)
MAX_RESULT_FORCE = env_float("TANK_APF_MAX_RESULT_FORCE", 20.0)
LOCAL_TARGET_DISTANCE = env_float("TANK_APF_LOCAL_TARGET_DISTANCE", 8.0)
REPULSIVE_EPS = env_float("TANK_APF_REP_EPS", 0.5)
PASSTHROUGH_WHEN_CLEAR = env_bool("TANK_APF_PASSTHROUGH_WHEN_CLEAR", True)

USE_TANGENTIAL_FORCE = env_bool("TANK_APF_USE_TANGENTIAL", True)
TANGENTIAL_GAIN_SCALE = env_float("TANK_APF_TANGENTIAL_GAIN", 0.8)

MIN_OBSTACLE_DISTANCE = env_float("TANK_APF_MIN_OBS_DIST", 0.3)
MAX_OBSTACLE_DISTANCE = env_float("TANK_APF_MAX_OBS_DIST", 12.0)
FRONT_SECTOR_DEG = env_float("TANK_APF_FRONT_SECTOR_DEG", 120.0)
PATH_CORRIDOR_WIDTH = env_float("TANK_APF_PATH_CORRIDOR_WIDTH", 7.0)
OBSTACLE_VOXEL_RESOLUTION = env_float("TANK_APF_OBS_VOXEL", 1.5)
MAX_OBSTACLE_POINTS = env_int("TANK_APF_MAX_OBS_POINTS", 80)
USE_DISCOVERED_OBJECTS = env_bool("TANK_APF_USE_DISCOVERED", True)

USE_THREAT_AVOIDANCE = env_bool("TANK_APF_USE_THREAT", True)
THREAT_RADIUS = env_float("TANK_APF_THREAT_RADIUS", 25.0)
K_THREAT_REPULSIVE = env_float("TANK_APF_K_THREAT_REP", 2000.0)
THREAT_TYPES = tuple(os.environ.get("TANK_APF_THREAT_TYPES", "House002,Tank001").split(","))

ANGULAR_GAIN_K_THETA = env_float("TANK_APF_K_THETA", 1.2)
ANGLE_EPSILON_DEG = env_float("TANK_APF_ANGLE_EPS_DEG", 10.0)
LINEAR_SPEED_GAIN = env_float("TANK_APF_LINEAR_SPEED_GAIN", 0.5)
MAX_DESIRED_SPEED = env_float("TANK_APF_MAX_DESIRED_SPEED", 1.0)
MAX_DESIRED_ANGULAR_SPEED = env_float("TANK_APF_MAX_DESIRED_OMEGA", 1.5)
MOTION_STRATEGY = os.environ.get("TANK_APF_MOTION_STRATEGY", "second")
MARKER_SCALE = env_float("TANK_APF_MARKER_SCALE", 2.5)

# 시각 인지 클러스터링 + 휴리스틱/RL 대응 가중치 프로파일 설정.
LIDAR_CLUSTERS_TOPIC = os.environ.get("TANK_TOPIC_LIDAR_CLUSTERS", "/tank/visual_perception/lidar_clusters")
USE_LIDAR_CLUSTERS = env_bool("TANK_APF_USE_LIDAR_CLUSTERS", True)
CLUSTER_OBSTACLE_MIN_COUNT = env_int("TANK_APF_CLUSTER_OBSTACLE_MIN_COUNT", 2)
APF_WEIGHT_PROFILE = os.environ.get("TANK_APF_WEIGHT_PROFILE", "default")
APF_WEIGHTS_FILE = os.environ.get("TANK_APF_WEIGHTS_FILE", "")
