# -*- coding: utf-8 -*-
"""Terrain/obstacle separation utilities for LiDAR hit points.

팀원이 예전 src에서 개발한 지형/장애물 분리 로직을 현재 통합 src의 lidar 패키지로
이식한 모듈이다. 다른 패키지는 raw lidarPoints 스키마를 직접 만지지 않고 이 함수의
출력 topic을 사용한다.

분리 기준:
- x-z 평면을 grid cell로 나눈다.
- 각 cell의 local ground height를 해당 cell 최저 z(raw y)로 추정한다.
- cell 내부 높이차 또는 인접 cell ground height 차이가 climb_limit보다 크면 steep cell로 본다.
- steep cell 안에서 local ground + obstacle_min_height보다 높은 hit point만 장애물로 분류한다.
- 나머지 hit point는 terrain point로 분류한다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple

from .coordinate_utils import to_float

GridKey = Tuple[int, int]


@dataclass
class TerrainSeparationStats:
    input_points: int = 0
    obstacle_points: int = 0
    terrain_points: int = 0
    grid_resolution: float = 0.5
    climb_limit: float = 0.4
    obstacle_min_height: float = 0.2
    grid_cell_count: int = 0
    steep_cell_count: int = 0
    max_cell_height_span: float = 0.0
    mean_cell_height_span: float = 0.0
    max_neighbor_ground_gap: float = 0.0
    roughness_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _position(point: Dict[str, Any]) -> Dict[str, Any]:
    pos = point.get("position")
    return pos if isinstance(pos, dict) else {}


def _raw_xzy(point: Dict[str, Any]) -> Tuple[float, float, float]:
    """Return simulator raw x, y(height), z values."""
    pos = _position(point)
    return (
        to_float(pos.get("x")),
        to_float(pos.get("y")),
        to_float(pos.get("z")),
    )


def grid_key_for_point(point: Dict[str, Any], grid_resolution: float) -> GridKey:
    x, _, z = _raw_xzy(point)
    q = max(float(grid_resolution), 1e-6)
    return (int(math.floor(x / q)), int(math.floor(z / q)))


def create_grid_map(points: Sequence[Dict[str, Any]], grid_resolution: float) -> Dict[GridKey, List[int]]:
    """Map each x-z grid cell to point indices."""
    grid: Dict[GridKey, List[int]] = {}
    for idx, point in enumerate(points):
        key = grid_key_for_point(point, grid_resolution)
        grid.setdefault(key, []).append(idx)
    return grid


def get_cell_ground_levels(points: Sequence[Dict[str, Any]], grid_map: Dict[GridKey, List[int]]) -> Dict[GridKey, float]:
    """Use the lowest point in each cell as local ground height."""
    levels: Dict[GridKey, float] = {}
    for key, indices in grid_map.items():
        levels[key] = min(_raw_xzy(points[i])[1] for i in indices)
    return levels


def find_steep_cells(
    points: Sequence[Dict[str, Any]],
    grid_map: Dict[GridKey, List[int]],
    cell_ground: Dict[GridKey, float],
    climb_limit: float,
) -> Tuple[Set[GridKey], float, float, float]:
    """Detect cells likely to contain vertical obstacles or abrupt terrain discontinuities."""
    steep_cells: Set[GridKey] = set()
    spans: List[float] = []
    max_neighbor_gap = 0.0

    for key, indices in grid_map.items():
        ys = [_raw_xzy(points[i])[1] for i in indices]
        min_y = min(ys)
        max_y = max(ys)
        span = max_y - min_y
        spans.append(span)

        # 1) Cell internal height span: rock/wall often produces tall vertical returns in one cell.
        if span > climb_limit:
            steep_cells.add(key)
            continue

        # 2) Neighbor ground height gradient: abrupt step between adjacent cells.
        gx, gz = key
        for neighbor in ((gx + 1, gz), (gx - 1, gz), (gx, gz + 1), (gx, gz - 1)):
            if neighbor not in cell_ground:
                continue
            gap = abs(cell_ground[key] - cell_ground[neighbor])
            max_neighbor_gap = max(max_neighbor_gap, gap)
            if gap > climb_limit:
                steep_cells.add(key)
                steep_cells.add(neighbor)

    max_span = max(spans) if spans else 0.0
    mean_span = sum(spans) / len(spans) if spans else 0.0
    return steep_cells, max_span, mean_span, max_neighbor_gap


def split_terrain_obstacle_points(
    points: Sequence[Dict[str, Any]],
    grid_resolution: float = 0.5,
    climb_limit: float = 0.4,
    obstacle_min_height: float = 0.2,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], TerrainSeparationStats]:
    """Split detected LiDAR hit points into obstacle and terrain lists.

    Returns:
        obstacle_points, terrain_points, stats
    """
    source = [p for p in points if isinstance(p, dict)]
    stats = TerrainSeparationStats(
        input_points=len(source),
        grid_resolution=float(grid_resolution),
        climb_limit=float(climb_limit),
        obstacle_min_height=float(obstacle_min_height),
    )
    if not source:
        return [], [], stats

    q = max(float(grid_resolution), 1e-6)
    grid_map = create_grid_map(source, q)
    cell_ground = get_cell_ground_levels(source, grid_map)
    steep_cells, max_span, mean_span, max_neighbor_gap = find_steep_cells(
        source, grid_map, cell_ground, float(climb_limit)
    )

    obstacle_indices: Set[int] = set()
    terrain_indices: Set[int] = set()
    min_above_ground = max(float(obstacle_min_height), 0.0)

    for key, indices in grid_map.items():
        if key not in steep_cells:
            terrain_indices.update(indices)
            continue

        gx, gz = key
        neighbors = ((gx, gz), (gx + 1, gz), (gx - 1, gz), (gx, gz + 1), (gx, gz - 1))
        local_ground = min(cell_ground[n] for n in neighbors if n in cell_ground)
        threshold = local_ground + min_above_ground
        for idx in indices:
            _, raw_y, _ = _raw_xzy(source[idx])
            if raw_y > threshold:
                obstacle_indices.add(idx)
            else:
                terrain_indices.add(idx)

    # Every point should be assigned. Conservative fallback: unassigned points are terrain.
    all_indices = set(range(len(source)))
    terrain_indices.update(all_indices - obstacle_indices - terrain_indices)

    obstacle_points = [source[i] for i in sorted(obstacle_indices)]
    terrain_points = [source[i] for i in sorted(terrain_indices - obstacle_indices)]

    stats.obstacle_points = len(obstacle_points)
    stats.terrain_points = len(terrain_points)
    stats.grid_cell_count = len(grid_map)
    stats.steep_cell_count = len(steep_cells)
    stats.max_cell_height_span = float(max_span)
    stats.mean_cell_height_span = float(mean_span)
    stats.max_neighbor_ground_gap = float(max_neighbor_gap)
    # Simple normalized roughness score for monitoring/debug; 1.0 roughly means climb_limit-sized roughness.
    denom = max(float(climb_limit), 1e-6)
    stats.roughness_score = float(max(mean_span, max_neighbor_gap) / denom)
    return obstacle_points, terrain_points, stats


def filter_ground_points(
    points: Sequence[Dict[str, Any]],
    origin_y: float = 8.0,
    grid_resolution: float = 0.5,
    climb_limit: float = 0.4,
    obstacle_min_height: float = 0.2,
) -> List[Dict[str, Any]]:
    """Backward-compatible helper: return only obstacle points.

    origin_y is kept for compatibility with the previous function signature.
    """
    obstacles, _, _ = split_terrain_obstacle_points(
        points,
        grid_resolution=grid_resolution,
        climb_limit=climb_limit,
        obstacle_min_height=obstacle_min_height,
    )
    return obstacles


def convert_to_2d_coords(points: Iterable[Dict[str, Any]]) -> List[Tuple[float, float]]:
    coords: List[Tuple[float, float]] = []
    for point in points:
        pos = _position(point)
        coords.append((to_float(pos.get("x")), to_float(pos.get("z"))))
    return coords
