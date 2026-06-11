# -*- coding: utf-8 -*-
"""LiDAR perception helpers kept for backward compatibility.

The actual terrain/obstacle split implementation lives in lidar.terrain_utils.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from lidar.payloads import cluster_lidar_points, lidar_clusters_to_bboxes
from lidar.terrain_utils import (
    convert_to_2d_coords,
    create_grid_map,
    filter_ground_points,
    get_cell_ground_levels,
    split_terrain_obstacle_points,
)


def extract_valid_points(lidar_points: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return only simulator LiDAR hit points with isDetected=True."""
    return [p for p in lidar_points if isinstance(p, dict) and bool(p.get("isDetected", False))]


get_lidar_bboxes = lidar_clusters_to_bboxes


def process_lidar(lidar_data: Dict[str, Any]) -> List[Tuple[float, float]]:
    """Process raw /info LiDAR payload into 2D obstacle coordinates."""
    if not isinstance(lidar_data, dict):
        return []
    points = lidar_data.get("lidarPoints", [])
    valid_points = extract_valid_points(points if isinstance(points, list) else [])
    obstacle_points = filter_ground_points(valid_points)
    return convert_to_2d_coords(obstacle_points)
