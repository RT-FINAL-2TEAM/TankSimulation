# -*- coding: utf-8 -*-
"""전역 경로계획에서 쓰는 LiDAR 기반 경로 차단 판정 유틸리티.

A* planner는 전역 경로를 만든다. LiDAR point가 경로를 막는지 판단하는
저수준 계산은 LiDAR 전용 모듈에 둔다.
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

Point2D = Tuple[float, float]


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def distance(a: Point2D, b: Point2D) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def dist_point_to_segment(p: Point2D, a: Point2D, b: Point2D) -> float:
    ax, ay = a
    bx, by = b
    px, py = p
    dx = bx - ax
    dy = by - ay
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return distance(p, a)
    t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
    t = clamp(t, 0.0, 1.0)
    cx = ax + t * dx
    cy = ay + t * dy
    return math.hypot(px - cx, py - cy)


def is_path_blocked(
    current_pos: Point2D,
    route: Sequence[Point2D],
    start_idx: int,
    lidar_points: Sequence[Point2D],
    margin: float,
    lookahead_segments: int = 8,
) -> bool:
    if not route or not lidar_points:
        return False
    segments: List[Tuple[Point2D, Point2D]] = []
    idx = max(0, min(start_idx, len(route) - 1))
    segments.append((current_pos, route[idx]))
    end = min(len(route) - 1, idx + lookahead_segments)
    for i in range(idx, end):
        segments.append((route[i], route[i + 1]))
    for lp in lidar_points:
        for a, b in segments:
            if dist_point_to_segment(lp, a, b) < margin:
                return True
    return False
