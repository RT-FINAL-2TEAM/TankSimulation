# -*- coding: utf-8 -*-
"""LiDAR 좌표 변환 유틸리티.

Unity simulator raw 좌표를 ROS/RViz map 좌표로 변환하는 정책을 이곳에만 둔다.
좌표 정책: raw.x -> map.x, raw.z -> map.y, raw.y -> map.z
"""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

from .config import MAP_FRAME, UNITY_FRAME


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def dumps_compact(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def as_xyz(data: Any) -> Optional[Dict[str, float]]:
    if not isinstance(data, dict):
        return None
    return {
        "x": to_float(data.get("x")),
        "y": to_float(data.get("y")),
        "z": to_float(data.get("z")),
    }


def raw_to_map_xyz(position: Dict[str, Any]) -> Dict[str, float]:
    raw_x = to_float(position.get("x"))
    raw_y = to_float(position.get("y"))
    raw_z = to_float(position.get("z"))
    return {"x": raw_x, "y": raw_z, "z": raw_y}


def raw_and_map_point(position: Dict[str, Any], source: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    raw_x = to_float(position.get("x"))
    raw_y = to_float(position.get("y"))
    raw_z = to_float(position.get("z"))
    raw = {
        "source": source,
        "frame_id": UNITY_FRAME,
        "x": raw_x,
        "y": raw_y,
        "z": raw_z,
    }
    mapped = {
        "source": source,
        "frame_id": MAP_FRAME,
        "x": raw_x,
        "y": raw_z,
        "z": raw_y,
    }
    return raw, mapped


def lidar_point_with_map_position(point: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(point, dict):
        return None
    position = point.get("position")
    if not isinstance(position, dict):
        return None
    raw_x = to_float(position.get("x"))
    raw_y = to_float(position.get("y"))
    raw_z = to_float(position.get("z"))
    converted = deepcopy(point)
    converted["position_raw"] = {"x": raw_x, "y": raw_y, "z": raw_z, "frame_id": UNITY_FRAME}
    converted["position_map"] = {"x": raw_x, "y": raw_z, "z": raw_y, "frame_id": MAP_FRAME}
    return converted
