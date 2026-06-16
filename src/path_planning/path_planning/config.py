# -*- coding: utf-8 -*-
"""Global path-planning configuration values.

시뮬레이션 상황에 따라 바꿀 수 있는 A* / route / topic 기본값을 이 파일에 모은다.
LiDAR raw parsing이나 clustering 기본값은 lidar.config가 source of truth이다.
"""

from __future__ import annotations

import os
from typing import Dict, Tuple

from lidar.config import (
    CLUSTER_EPS as LIDAR_CLUSTER_EPS,
    CLUSTER_MIN_SAMPLES as LIDAR_CLUSTER_MIN_SAMPLES,
    HISTORY_RESOLUTION as LIDAR_HISTORY_RESOLUTION,
    MAX_HISTORY_POINTS as MAX_LIDAR_HISTORY_POINTS,
    PATH_BLOCK_MAX_DISTANCE as LIDAR_BLOCK_MAX_DISTANCE,
    PATH_BLOCK_MIN_DISTANCE as LIDAR_BLOCK_MIN_DISTANCE,
    TOPIC_LIDAR_DETECTED_MAP,
)


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

TOPIC_PLAYER_POSE = os.environ.get("TANK_TOPIC_PLAYER_POSE", "/tank/player/pose")
TOPIC_GOAL_POSE = os.environ.get("TANK_TOPIC_GOAL_POSE", "/tank/goal/pose")
TOPIC_MAP_OBSTACLES = os.environ.get("TANK_TOPIC_MAP_OBSTACLES", "/tank/map/obstacles")
TOPIC_GLOBAL_PATH = os.environ.get("TANK_TOPIC_GLOBAL_PATH", "/tank/global_path")
TOPIC_LOOKAHEAD_POSE = os.environ.get("TANK_TOPIC_LOOKAHEAD_POSE", "/tank/path/lookahead_pose")
TOPIC_PATH_POINTS = os.environ.get("TANK_TOPIC_PATH_POINTS", "/tank/planner/path_points")
TOPIC_PLANNER_STATUS = os.environ.get("TANK_TOPIC_PLANNER_STATUS", "/tank/planner/status")
TOPIC_LIDAR_BBOXES = os.environ.get("TANK_TOPIC_LIDAR_BBOXES", "/tank/planner/lidar_bboxes")

MAP_WIDTH = env_int("TANK_PLANNER_MAP_WIDTH", 300)
MAP_HEIGHT = env_int("TANK_PLANNER_MAP_HEIGHT", 300)
MAP_RESOLUTION = env_float("TANK_PLANNER_RESOLUTION", 1.0)
OBSTACLE_INFLATE = env_float("TANK_PLANNER_INFLATE", 5.0)
USE_PATH_SMOOTHING = env_bool("TANK_PLANNER_USE_SMOOTHING", True)
USE_GT_OBSTACLES = env_bool("TANK_PLANNER_USE_GT_OBSTACLES", False)
ENABLE_DYNAMIC_REPLAN = env_bool("TANK_PLANNER_ENABLE_DYNAMIC_REPLAN", False)
ENABLE_PERIODIC_REPLAN = env_bool("TANK_PLANNER_ENABLE_PERIODIC_REPLAN", False)
REPLAN_PERIOD_SEC = env_float("TANK_PLANNER_REPLAN_PERIOD_SEC", 0.0)
DYNAMIC_REPLAN_COOLDOWN_SEC = env_float("TANK_PLANNER_DYNAMIC_REPLAN_COOLDOWN_SEC", 8.0)
PLAN_RETRY_PERIOD_SEC = env_float("TANK_PLANNER_PLAN_RETRY_PERIOD_SEC", 3.0)
PATH_BLOCK_MARGIN = env_float("TANK_PLANNER_PATH_BLOCK_MARGIN", 5.0)
PATH_BLOCK_REQUIRED_HITS = env_int("TANK_PLANNER_PATH_BLOCK_REQUIRED_HITS", 5)
LOOKAHEAD_DISTANCE = env_float("TANK_PLANNER_LOOKAHEAD_DISTANCE", 15.0)
PUBLISH_PATH_PERIOD_SEC = env_float("TANK_PLANNER_PUBLISH_PATH_PERIOD_SEC", 1.0)
GOAL_TOLERANCE = env_float("TANK_PLANNER_GOAL_TOLERANCE", 10.0)
DEFAULT_GOAL_ENABLED = env_bool("TANK_PLANNER_DEFAULT_GOAL_ENABLED", True)
DEFAULT_GOAL_X = env_float("TANK_PLANNER_DEFAULT_GOAL_X", 120.0)
DEFAULT_GOAL_Y = env_float("TANK_PLANNER_DEFAULT_GOAL_Y", 250.0)
MAX_EXPANSIONS = env_int("TANK_PLANNER_MAX_EXPANSIONS", 250000)
PLANNER_HZ = env_float("TANK_PLANNER_HZ", 10.0)

# Fallback sizes when /update_obstacle gives prefabName/position rather than bbox.
PREFAB_HALF_SIZES: Dict[str, Tuple[float, float]] = {
    "Human": (0.5, 0.5),
    "Tree": (1.0, 1.0),
    "Rock": (1.5, 1.5),
    "Tank": (2.0, 4.0),
    "House": (4.0, 4.0),
    "wall001x5": (8.0, 1.0),
    "wall002x5": (8.0, 1.0),
    "Wall001": (3.0, 1.0),
    "Wall002": (3.0, 1.0),
}

# Camera + LiDAR calibration / overlay defaults. This calibration belongs to the
# local-path/fusion layer, not to raw LiDAR preprocessing.
CAMERA_LIDAR_PROJECTION_PARAMS = {
    "tx": env_float("TANK_CAM_LIDAR_TX", 0.28),
    "ty": env_float("TANK_CAM_LIDAR_TY", 0.02),
    "tz": env_float("TANK_CAM_LIDAR_TZ", 11.80),
    "yaw_offset": env_float("TANK_CAM_LIDAR_YAW_OFFSET", -0.9),
    "pitch_offset": env_float("TANK_CAM_LIDAR_PITCH_OFFSET", -0.9),
    "roll_offset": env_float("TANK_CAM_LIDAR_ROLL_OFFSET", -0.3),
    "hfov": env_float("TANK_CAM_LIDAR_HFOV", 86.0),
    "vfov": env_float("TANK_CAM_LIDAR_VFOV", 60.2),
}
CAMERA_LIDAR_USE_ONLY_DETECTED = env_bool("TANK_CAM_LIDAR_USE_ONLY_DETECTED", True)
CAMERA_LIDAR_MIN_DISTANCE = env_float("TANK_CAM_LIDAR_MIN_DISTANCE", 1.0)
CAMERA_LIDAR_MAX_DISTANCE = env_float("TANK_CAM_LIDAR_MAX_DISTANCE", 35.0)
CAMERA_LIDAR_POINT_RADIUS = env_int("TANK_CAM_LIDAR_POINT_RADIUS", 2)
CAMERA_LIDAR_DRAW_TEXT = env_bool("TANK_CAM_LIDAR_DRAW_TEXT", True)
TOPIC_CAMERA_IMAGE_COMPRESSED = os.environ.get("TANK_TOPIC_CAMERA_IMAGE_COMPRESSED", "/tank/camera/image_compressed")
TOPIC_INFO_COMPACT = os.environ.get("TANK_TOPIC_INFO_COMPACT", "/tank/api/info/compact")
TOPIC_INFO_RAW = os.environ.get("TANK_TOPIC_INFO_RAW", "/tank/api/info/raw")
TOPIC_CAMERA_LIDAR_PROJECTION_IMAGE = os.environ.get("TANK_TOPIC_CAMERA_LIDAR_PROJECTION_IMAGE", "/tank/camera/lidar_projection/image")
TOPIC_CAMERA_LIDAR_PROJECTION_COMPRESSED = os.environ.get("TANK_TOPIC_CAMERA_LIDAR_PROJECTION_COMPRESSED", "/tank/camera/lidar_projection/compressed")

# Local path / camera-LiDAR fusion topics and default colors.
TOPIC_DETECTIONS = os.environ.get("TANK_TOPIC_DETECTIONS", "/tank/perception/detections")
TOPIC_PLAYER_STATE = os.environ.get("TANK_TOPIC_PLAYER_STATE", "/tank/player/state")
TOPIC_TURRET = os.environ.get("TANK_TOPIC_TURRET", "/tank/api/get_action/turret")
TOPIC_RECON_RAW = os.environ.get("TANK_TOPIC_RECON_RAW", "/tank/map/recon/raw")
TOPIC_FUSED_OBJECTS = os.environ.get("TANK_TOPIC_FUSED_OBJECTS", "/tank/perception/fused_objects")
TOPIC_DISCOVERED_OBJECTS = os.environ.get("TANK_TOPIC_DISCOVERED_OBJECTS", "/tank/map/discovered/objects")
TOPIC_FUSED_OBJECT_MARKERS = os.environ.get("TANK_TOPIC_FUSED_OBJECT_MARKERS", "/tank/rviz/fused_object_markers")
TOPIC_DISCOVERED_OBJECT_MARKERS = os.environ.get("TANK_TOPIC_DISCOVERED_OBJECT_MARKERS", "/tank/rviz/discovered_object_markers")
SERVICE_DISCOVERED_SAVE = os.environ.get("TANK_SERVICE_DISCOVERED_SAVE", "/tank/map/discovered/save")
SERVICE_DISCOVERED_CLEAR = os.environ.get("TANK_SERVICE_DISCOVERED_CLEAR", "/tank/map/discovered/clear")
LOCAL_PATH_TIMER_SEC = env_float("TANK_LOCAL_PATH_TIMER_SEC", 0.2)
CLASS_COLOR_DEFAULTS = {
    "person": "#00FFFF",
    "rock": "#FFA500",
    "tank": "#FF0000",
    "wall": "#00FF00",
    "tent": "#FFFF00",
    "unknown": "#FFFFFF",
}

# Visual perception / clustering integration.
TOPIC_LIDAR_CLUSTERS = os.environ.get("TANK_TOPIC_LIDAR_CLUSTERS", "/tank/visual_perception/lidar_clusters")


# TankSimulation route A/B strategy integration.
USE_ROUTE_WAYPOINTS = env_bool("TANK_PLANNER_USE_ROUTE_WAYPOINTS", False)
ROUTE_MAP_NAME = os.environ.get("TANK_PLANNER_ROUTE_MAP_NAME", "finalmap")
ROUTE_ID = os.environ.get("TANK_PLANNER_ROUTE_ID", "B")
ROUTE_SIDE = os.environ.get("TANK_PLANNER_ROUTE_SIDE", "east")
ROUTE_CLEARANCE_WEIGHT = env_float("TANK_PLANNER_ROUTE_CLEARANCE_WEIGHT", 0.4)
ROUTE_CONFIG_FILE = os.environ.get("TANK_PLANNER_ROUTE_CONFIG_FILE", "")
USE_LIDAR_CLUSTER_BBOXES = env_bool("TANK_PLANNER_USE_LIDAR_CLUSTER_BBOXES", True)
LIDAR_CLUSTER_BBOX_MARGIN = env_float("TANK_PLANNER_LIDAR_CLUSTER_BBOX_MARGIN", 1.0)
