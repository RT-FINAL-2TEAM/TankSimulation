# -*- coding: utf-8 -*-
"""Route configuration loader adapted from TankSimulation/configs/routes.yaml.

The original Team TankSimulation file expected a project-root `configs/routes.yaml`.
This ROS2 package version first tries the installed ament share directory and then
falls back to this source tree's `config/routes.yaml`, so it works both before and
after `colcon build --symlink-install`.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import yaml


def _default_routes_path() -> Path:
    try:
        from ament_index_python.packages import get_package_share_directory
        return Path(get_package_share_directory("path_planning")) / "config" / "routes.yaml"
    except Exception:
        return Path(__file__).resolve().parents[1] / "config" / "routes.yaml"


def load_routes(config_path: str | None = None) -> dict:
    path = Path(config_path).expanduser() if config_path else _default_routes_path()
    if not path.exists():
        raise FileNotFoundError(f"경로 설정 파일을 찾을 수 없습니다: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def get_route_waypoints(map_name: str, route_id: str, config_path: str | None = None) -> List[Tuple[float, float]]:
    data = load_routes(config_path)
    if map_name not in data:
        raise ValueError(f"맵 이름을 찾을 수 없습니다: {map_name}")
    routes = data[map_name].get("routes", {})
    if route_id not in routes:
        raise ValueError(f"해당 맵에 루트 ID가 존재하지 않습니다: {route_id}")
    raw_points = routes[route_id]
    return [(float(p[0]), float(p[1])) for p in raw_points]


def get_route_start_goal(map_name: str, config_path: str | None = None) -> tuple[Tuple[float, float], Tuple[float, float]]:
    data = load_routes(config_path)
    if map_name not in data:
        raise ValueError(f"맵 이름을 찾을 수 없습니다: {map_name}")
    m = data[map_name]
    start = m.get("start", [0.0, 0.0])
    goal = m.get("destination", [0.0, 0.0])
    return (float(start[0]), float(start[1])), (float(goal[0]), float(goal[1]))
