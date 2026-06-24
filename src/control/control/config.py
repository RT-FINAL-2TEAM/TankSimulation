# -*- coding: utf-8 -*-
"""control 패키지 전역 설정값."""

from __future__ import annotations

import os


def env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except Exception:
        return default


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


TOPIC_CONTROL_COMMAND = os.environ.get("TANK_TOPIC_CONTROL_COMMAND", "/tank/control/command")
TOPIC_CONTROL_STATUS = os.environ.get("TANK_TOPIC_CONTROL_STATUS", "/tank/control/status")
TOPIC_PLAYER_POSE = os.environ.get("TANK_TOPIC_PLAYER_POSE", "/tank/player/pose")
TOPIC_PLAYER_STATE = os.environ.get("TANK_TOPIC_PLAYER_STATE", "/tank/player/state")
TOPIC_GOAL_POSE = os.environ.get("TANK_TOPIC_GOAL_POSE", "/tank/goal/pose")
TOPIC_LOOKAHEAD_POSE = os.environ.get("TANK_TOPIC_LOOKAHEAD_POSE", "/tank/path/lookahead_pose")
TOPIC_LOCAL_TARGET_POSE = os.environ.get("TANK_TOPIC_LOCAL_TARGET_POSE", "/tank/local_target/pose")
TOPIC_COLLISION_EVENT = os.environ.get("TANK_TOPIC_COLLISION_EVENT", "/tank/event/collision")

CONTROLLER_HZ = env_float("TANK_CONTROLLER_HZ", 10.0)
ENABLE_LOCAL_TARGET = env_bool("TANK_CONTROLLER_ENABLE_LOCAL_TARGET", True)
TARGET_TTL_SEC = env_float("TANK_CONTROLLER_TARGET_TTL_SEC", 2.0)
GOAL_TOLERANCE = env_float("TANK_CONTROLLER_GOAL_TOLERANCE", 10.0)
HEADING_DEADBAND_DEG = env_float("TANK_CONTROLLER_HEADING_DEADBAND_DEG", 5.0)
STEERING_FULL_ERROR_DEG = env_float("TANK_CONTROLLER_STEERING_FULL_ERROR_DEG", 45.0)
MIN_AD_WEIGHT = env_float("TANK_CONTROLLER_MIN_AD_WEIGHT", 0.0)
MAX_AD_WEIGHT = env_float("TANK_CONTROLLER_MAX_AD_WEIGHT", 1.0)
# weaving(A↔D 토글) 완화용 D 게인(rate feedback, 초 단위 등가). 0이면 기존 순수 P 거동.
STEERING_KD = env_float("TANK_CONTROLLER_STEERING_KD", 0.2)
STRAIGHT_WS_WEIGHT = env_float("TANK_CONTROLLER_STRAIGHT_WS_WEIGHT", 1.0)
TURN_WS_WEIGHT = env_float("TANK_CONTROLLER_TURN_WS_WEIGHT", 0.4)
ROTATE_IN_PLACE_ANGLE_DEG = env_float("TANK_CONTROLLER_ROTATE_IN_PLACE_ANGLE_DEG", 60.0)
SLOWDOWN_ANGLE_DEG = env_float("TANK_CONTROLLER_SLOWDOWN_ANGLE_DEG", 30.0)
STOP_DISTANCE = env_float("TANK_CONTROLLER_STOP_DISTANCE", 10.0)
ENABLE_STUCK_ESCAPE = env_bool("TANK_CONTROLLER_ENABLE_STUCK_ESCAPE", True)
STUCK_CHECK_PERIOD = env_float("TANK_CONTROLLER_STUCK_CHECK_PERIOD", 5.0)
STUCK_MIN_MOVEMENT = env_float("TANK_CONTROLLER_STUCK_MIN_MOVEMENT", 1.5)
ESCAPE_REVERSE_SEC = env_float("TANK_CONTROLLER_ESCAPE_REVERSE_SEC", 1.5)
ESCAPE_TURN_SEC = env_float("TANK_CONTROLLER_ESCAPE_TURN_SEC", 1.5)
