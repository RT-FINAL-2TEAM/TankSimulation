# -*- coding: utf-8 -*-
"""LiDAR 페이로드 파싱과 장애물 전처리.

다른 패키지(path_planning, potential 등)는 LiDAR JSON schema를 직접 파싱하지 않고
여기 함수만 import해서 사용한다. 이렇게 해야 LiDAR schema가 바뀌어도 수정 지점이
lidar 패키지로 제한된다.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.cluster import DBSCAN

from .config import (
    BBOX_MIN_THICKNESS,
    TERRAIN_CLIMB_LIMIT,
    TERRAIN_GRID_RESOLUTION,
    TERRAIN_OBSTACLE_MIN_HEIGHT,
)
from .coordinate_utils import lidar_point_with_map_position, to_float
from .terrain_utils import split_terrain_obstacle_points
from .path_blocking import distance

Point2D = Tuple[float, float]
BBox2D = Dict[str, float]


def extract_payload_list(data: Any, key: str = "points") -> List[Any]:
    if isinstance(data, list):
        return data
    if not isinstance(data, dict):
        return []
    if isinstance(data.get(key), list):
        return data[key]
    inner = data.get("data")
    if isinstance(inner, list):
        return inner
    if isinstance(inner, dict) and isinstance(inner.get(key), list):
        return inner[key]
    return []


def iter_detected_points(raw_points: Any) -> Iterable[Dict[str, Any]]:
    if not isinstance(raw_points, list):
        return []
    return (p for p in raw_points if isinstance(p, dict) and bool(p.get("isDetected", False)))


def _converted_points(source_points: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    points: List[Dict[str, Any]] = []
    for point in source_points:
        converted = lidar_point_with_map_position(point)
        if converted is not None:
            points.append(converted)
    return points


def _base_payload(
    points: Sequence[Dict[str, Any]],
    *,
    timestamp_wall: float,
    map_frame: str,
    source: str,
    lidar_origin_map_for_correction: Optional[Dict[str, Any]] = None,
    lidar_rotation_deg: Optional[Dict[str, Any]] = None,
    player_body_deg: Optional[Dict[str, Any]] = None,
    terrain_filter: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "route": "/info",
        "timestamp_wall": timestamp_wall,
        "source": source,
        "frame_id": map_frame,
        "coordinate_policy": "position_map: x=raw.x, y=raw.z, z=raw.y",
        "count": len(points),
        "points": list(points),
    }
    if lidar_origin_map_for_correction is not None:
        payload["lidar_origin_map_for_correction"] = lidar_origin_map_for_correction
    if lidar_rotation_deg is not None:
        payload["lidar_rotation_deg"] = lidar_rotation_deg
    if player_body_deg is not None:
        payload["player_body_deg"] = player_body_deg
    if terrain_filter is not None:
        payload["terrain_filter"] = terrain_filter
    return payload


def build_classified_lidar_payloads(
    lidar_points: Any,
    *,
    timestamp_wall: float,
    map_frame: str,
    ground_filter_enabled: bool = True,
    lidar_origin_map_for_correction: Optional[Dict[str, Any]] = None,
    lidar_rotation_deg: Optional[Dict[str, Any]] = None,
    player_body_deg: Optional[Dict[str, Any]] = None,
    grid_resolution: float = TERRAIN_GRID_RESOLUTION,
    climb_limit: float = TERRAIN_CLIMB_LIMIT,
    obstacle_min_height: float = TERRAIN_OBSTACLE_MIN_HEIGHT,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """시뮬레이터 raw lidarPoints에서 map 프레임 페이로드 4개를 만든다.

    Returns:
        detected_obstacle_payload, terrain_payload, all_detected_payload, terrain_info_payload
    """
    source_points = list(iter_detected_points(lidar_points))

    if ground_filter_enabled and source_points:
        obstacle_raw, terrain_raw, stats = split_terrain_obstacle_points(
            source_points,
            grid_resolution=grid_resolution,
            climb_limit=climb_limit,
            obstacle_min_height=obstacle_min_height,
        )
        filter_stats = stats.to_dict()
        filter_method = "grid_local_ground_steep_cell"
    else:
        obstacle_raw = source_points
        terrain_raw = []
        filter_stats = {
            "input_points": len(source_points),
            "obstacle_points": len(source_points),
            "terrain_points": 0,
            "grid_resolution": grid_resolution,
            "climb_limit": climb_limit,
            "obstacle_min_height": obstacle_min_height,
        }
        filter_method = "disabled"

    obstacle_points = _converted_points(obstacle_raw)
    terrain_points = _converted_points(terrain_raw)
    all_points = _converted_points(source_points)

    common_meta = {
        "enabled": bool(ground_filter_enabled),
        "method": filter_method,
        **filter_stats,
    }
    detected_payload = _base_payload(
        obstacle_points,
        timestamp_wall=timestamp_wall,
        map_frame=map_frame,
        source="lidarPoints/obstacle_only",
        lidar_origin_map_for_correction=lidar_origin_map_for_correction,
        lidar_rotation_deg=lidar_rotation_deg,
        player_body_deg=player_body_deg,
        terrain_filter=common_meta,
    )
    terrain_payload = _base_payload(
        terrain_points,
        timestamp_wall=timestamp_wall,
        map_frame=map_frame,
        source="lidarPoints/terrain_only",
        lidar_origin_map_for_correction=lidar_origin_map_for_correction,
        lidar_rotation_deg=lidar_rotation_deg,
        player_body_deg=player_body_deg,
        terrain_filter=common_meta,
    )
    all_payload = _base_payload(
        all_points,
        timestamp_wall=timestamp_wall,
        map_frame=map_frame,
        source="lidarPoints/all_detected",
        lidar_origin_map_for_correction=lidar_origin_map_for_correction,
        lidar_rotation_deg=lidar_rotation_deg,
        player_body_deg=player_body_deg,
        terrain_filter=common_meta,
    )
    terrain_info_payload = {
        "route": "/info",
        "timestamp_wall": timestamp_wall,
        "frame_id": map_frame,
        "source": "lidar_terrain_separation",
        "terrain_filter": common_meta,
        "counts": {
            "all_detected": len(all_points),
            "obstacle": len(obstacle_points),
            "terrain": len(terrain_points),
        },
    }
    return detected_payload, terrain_payload, all_payload, terrain_info_payload


def build_detected_map_payload(
    lidar_points: Any,
    timestamp_wall: float,
    map_frame: str,
    ground_filter_enabled: bool = False,
    origin_y: float = 8.0,
) -> Dict[str, Any]:
    """과거 호출부가 쓰는 하위호환 헬퍼.

    이제 ground_filter_enabled=True일 때 obstacle-only 페이로드를 반환한다.
    """
    detected_payload, _, _, _ = build_classified_lidar_payloads(
        lidar_points,
        timestamp_wall=timestamp_wall,
        map_frame=map_frame,
        ground_filter_enabled=ground_filter_enabled,
    )
    return detected_payload


def parse_lidar_points_payload(payload: Any) -> List[Point2D]:
    """/tank/sensor/lidar/detected_points_map를 map 평면 (x, y)로 파싱한다."""
    points: List[Point2D] = []
    for item in extract_payload_list(payload, "points"):
        if not isinstance(item, dict):
            continue
        pos = item.get("position_map") if isinstance(item.get("position_map"), dict) else item.get("position")
        if not isinstance(pos, dict):
            continue
        try:
            if "y" in pos:
                points.append((float(pos.get("x", 0.0)), float(pos.get("y", 0.0))))
            else:
                points.append((float(pos.get("x", 0.0)), float(pos.get("z", 0.0))))
        except Exception:
            continue
    return points



def filter_lidar_points_by_distance(
    current_pos: Point2D,
    lidar_points: Sequence[Point2D],
    min_distance: float,
    max_distance: float,
) -> List[Point2D]:
    filtered: List[Point2D] = []
    for p in lidar_points:
        d = distance(current_pos, p)
        if min_distance <= d <= max_distance:
            filtered.append(p)
    return filtered


def cluster_lidar_points(points: Sequence[Point2D], eps: float = 2.0, min_samples: int = 3) -> List[List[Point2D]]:
    if not points:
        return []
    coords = np.asarray([(p[0], p[1]) for p in points], dtype=np.float32)
    labels = DBSCAN(eps=eps, min_samples=min_samples, algorithm='kd_tree').fit_predict(coords)
    
    clusters: List[List[Point2D]] = []
    unique_labels = set(labels)
    unique_labels.discard(-1)
    
    pts = list(points)
    for label in unique_labels:
        cluster = [pts[idx] for idx, lbl in enumerate(labels) if lbl == label]
        clusters.append(cluster)
    return clusters


def lidar_clusters_to_bboxes(clusters: Sequence[Sequence[Point2D]], min_thickness: float = BBOX_MIN_THICKNESS) -> List[BBox2D]:
    bboxes: List[BBox2D] = []
    for cluster in clusters:
        if not cluster:
            continue
        xs = [p[0] for p in cluster]
        ys = [p[1] for p in cluster]
        x_min, x_max = min(xs), max(xs)
        z_min, z_max = min(ys), max(ys)
        if x_max - x_min < min_thickness:
            pad = 0.5 * (min_thickness - (x_max - x_min))
            x_min -= pad
            x_max += pad
        if z_max - z_min < min_thickness:
            pad = 0.5 * (min_thickness - (z_max - z_min))
            z_min -= pad
            z_max += pad
        bboxes.append({"x_min": x_min, "x_max": x_max, "z_min": z_min, "z_max": z_max})
    return bboxes


def update_lidar_history(
    history: List[Point2D],
    history_set: set,
    points: Sequence[Point2D],
    resolution: float,
    max_points: int,
) -> Tuple[List[Point2D], set]:
    if not points:
        return history, history_set
    q = max(resolution, 0.1)
    pts_arr = np.asarray(points, dtype=np.float64)
    rounded_arr = np.round(pts_arr / q) * q
    unique_arr = np.unique(rounded_arr, axis=0)
    
    for row in unique_arr:
        rounded = (float(row[0]), float(row[1]))
        if rounded not in history_set:
            history_set.add(rounded)
            history.append(rounded)
            
    if len(history) > max_points:
        drop = len(history) - max_points
        for p in history[:drop]:
            history_set.discard(p)
        history = history[drop:]
    return history, history_set
