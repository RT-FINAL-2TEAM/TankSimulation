# -*- coding: utf-8 -*-
"""경로 planner를 위한 상태 보존형 LiDAR 장애물 메모리.

Global planner가 LiDAR 기반 dynamic replan을 사용할 때 필요한 history,
clustering, path-block check를 이 클래스 안으로 격리한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Tuple

from .path_blocking import is_path_blocked
from .payloads import (
    Point2D,
    cluster_lidar_points,
    filter_lidar_points_by_distance,
    lidar_clusters_to_bboxes,
    parse_lidar_points_payload,
    update_lidar_history,
)


@dataclass
class LidarObstacleMemory:
    current_points: List[Point2D] = field(default_factory=list)
    history: List[Point2D] = field(default_factory=list)
    history_set: set = field(default_factory=set)

    def update_from_payload(
        self,
        payload: Any,
        history_enabled: bool,
        history_resolution: float,
        max_history_points: int,
    ) -> List[Point2D]:
        self.current_points = parse_lidar_points_payload(payload)
        if history_enabled:
            self.history, self.history_set = update_lidar_history(
                self.history,
                self.history_set,
                self.current_points,
                history_resolution,
                max_history_points,
            )
        return self.current_points

    def build_bboxes(self, cluster_eps: float, cluster_min_samples: int) -> List[Dict[str, float]]:
        clusters = cluster_lidar_points(self.history, cluster_eps, cluster_min_samples)
        return lidar_clusters_to_bboxes(clusters)

    def is_current_path_blocked(
        self,
        current_pos: Point2D,
        route: Sequence[Point2D],
        route_index: int,
        min_distance: float,
        max_distance: float,
        margin: float,
    ) -> bool:
        points = filter_lidar_points_by_distance(current_pos, self.current_points, min_distance, max_distance)
        return is_path_blocked(current_pos, route, route_index, points, margin)

    @property
    def current_count(self) -> int:
        return len(self.current_points)

    @property
    def history_count(self) -> int:
        return len(self.history)
