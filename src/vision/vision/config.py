# -*- coding: utf-8 -*-
"""전역 vision / YOLO 설정값."""

from __future__ import annotations

import os


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on", "y"}


def env_int(name: str, default: int) -> int:
    try:
        return int(float(os.getenv(name, str(default))))
    except Exception:
        return default


def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


DEFAULT_CLASS_COLORS = {
    "tank": "#FF0000",
    "rock": "#FFA500",
    "house": "#AA66FF",
    "car": "#FF66CC",
}

DEFAULT_CONFIG_ENV_KEYS = ("TANK_YOLO_CONFIG", "YOLO_CONFIG")
DEFAULT_MODEL_ENV_KEYS = ("TANK_YOLO_MODEL_PATH", "YOLO_MODEL_PATH")
DEFAULT_MODEL_FILENAME = os.getenv("TANK_YOLO_DEFAULT_MODEL", "best_final.pt")
DEFAULT_CONFIG_FILENAME = os.getenv("TANK_YOLO_DEFAULT_CONFIG", "yolo_detection.yaml")
DEFAULT_IMGSZ = env_int("TANK_YOLO_IMGSZ", 416)
DEFAULT_IOU = env_float("TANK_YOLO_IOU", 0.70)
DEFAULT_MAX_DET = env_int("TANK_YOLO_MAX_DET", 20)
DEFAULT_MAX_RETURN = env_int("TANK_YOLO_MAX_RETURN", 5)
DEFAULT_MODEL_CONFIDENCE = env_float("TANK_YOLO_CONF", 0.50)
DEFAULT_FALLBACK_CONFIDENCE = env_float("TANK_YOLO_FALLBACK_CONF", 0.50)
