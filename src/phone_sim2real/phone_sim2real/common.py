# -*- coding: utf-8 -*-
"""Shared helpers for phone_sim2real."""

from __future__ import annotations

import json
import math
from typing import Any, Dict, Iterable, List, Optional, Tuple


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def safe_json_loads(text: str, default: Any = None) -> Any:
    try:
        return json.loads(text)
    except Exception:
        return default


def normalize_angle_rad(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def quaternion_to_yaw_rad(q: Any) -> float:
    x = float(getattr(q, "x", 0.0))
    y = float(getattr(q, "y", 0.0))
    z = float(getattr(q, "z", 0.0))
    w = float(getattr(q, "w", 1.0))
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def yaw_to_quaternion_z(yaw: float) -> Tuple[float, float, float, float]:
    half = yaw * 0.5
    return 0.0, 0.0, math.sin(half), math.cos(half)


def get_nested_float(data: Dict[str, Any], keys: Iterable[str], default: Optional[float] = None) -> Optional[float]:
    cur: Any = data
    try:
        for key in keys:
            if not isinstance(cur, dict):
                return default
            cur = cur.get(key)
        if cur is None:
            return default
        return float(cur)
    except Exception:
        return default


def parse_csv_set(value: str) -> set:
    return {item.strip().lower() for item in str(value or "").split(",") if item.strip()}


def color_for_class(class_name: str, alpha: float = 0.95) -> Tuple[float, float, float, float]:
    name = str(class_name or "unknown").strip().lower()
    table = {
        "car": (0.0, 0.85, 1.0, alpha),
        "tank": (1.0, 0.1, 0.1, alpha),
        "rock": (1.0, 0.55, 0.05, alpha),
        "house": (0.75, 0.25, 1.0, alpha),
        "tent": (1.0, 0.9, 0.15, alpha),
        "person": (0.1, 1.0, 0.1, alpha),
    }
    return table.get(name, (0.35, 0.75, 1.0, alpha))


def estimate_distance_from_bbox(
    bbox: List[float],
    image_height: int,
    vfov_deg: float,
    object_height_m: float,
    scale: float,
    bias_m: float,
    min_distance_m: float,
    max_distance_m: float,
) -> float:
    y1 = float(bbox[1])
    y2 = float(bbox[3])
    box_h = max(1.0, y2 - y1)
    focal_y_px = (float(image_height) * 0.5) / math.tan(math.radians(max(1.0, vfov_deg)) * 0.5)
    distance = (float(object_height_m) * focal_y_px / box_h) * float(scale) + float(bias_m)
    return clamp(distance, float(min_distance_m), float(max_distance_m))


def bbox_bearing_rad(bbox: List[float], image_width: int, hfov_deg: float) -> float:
    x1, _, x2, _ = [float(v) for v in bbox[:4]]
    cx = 0.5 * (x1 + x2)
    half_w = max(1.0, float(image_width) * 0.5)
    normalized = clamp((cx - half_w) / half_w, -1.0, 1.0)
    return normalized * math.radians(float(hfov_deg) * 0.5)


def bbox_center(bbox: List[float]) -> Tuple[float, float]:
    return 0.5 * (float(bbox[0]) + float(bbox[2])), 0.5 * (float(bbox[1]) + float(bbox[3]))
