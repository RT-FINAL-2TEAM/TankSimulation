# -*- coding: utf-8 -*-
"""LiDAR 장애물 메모리용 파싱·클러스터링 유틸리티.

PointCloud2가 실시간 LiDAR 전송의 기본 경로다. 이 모듈은 경로계획·potential 등에서
필요한 경량 JSON 포인트 페이로드 파싱과, LiDAR XY 점의 클러스터링·history 관리만 둔다.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from sklearn.cluster import DBSCAN

from .config import BBOX_MIN_THICKNESS
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
