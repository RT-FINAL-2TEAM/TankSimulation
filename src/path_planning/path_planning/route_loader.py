# -*- coding: utf-8 -*-
"""Route configuration loader adapted from TankSimulation/configs/routes.yaml.

The original Team TankSimulation file expected a project-root `configs/routes.yaml`.
This ROS2 package version first tries the installed ament share directory and then
falls back to this source tree's `config/routes.yaml`, so it works both before and
after `colcon build --symlink-install`.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

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


def _xy(raw, default: Tuple[float, float] = (0.0, 0.0)) -> Tuple[float, float]:
    if raw is None:
        return (float(default[0]), float(default[1]))
    return (float(raw[0]), float(raw[1]))


def get_route_waypoints(map_name: str, route_id: str, config_path: str | None = None) -> List[Tuple[float, float]]:
    data = load_routes(config_path)
    if map_name not in data:
        raise ValueError(f"맵 이름을 찾을 수 없습니다: {map_name}")
    routes = data[map_name].get("routes", {})
    if route_id not in routes:
        raise ValueError(f"해당 맵에 루트 ID가 존재하지 않습니다: {route_id}")
    raw_route = routes[route_id]
    # Backward compatible: a route may be either a plain waypoint list or a
    # dict with {waypoints: [...], destination: [...]}.  The current config uses
    # the plain list plus top-level route_destinations, but accepting both keeps
    # older/experimental route files usable.
    raw_points = raw_route.get("waypoints", []) if isinstance(raw_route, dict) else raw_route
    return [(float(p[0]), float(p[1])) for p in raw_points]


def get_route_destination(
    map_name: str, route_id: str, config_path: str | None = None
) -> tuple[Optional[Tuple[float, float]], Tuple[float, float]]:
    """Return (route_specific_destination, map_wide_destination).

    Scenario-1 needs route A and B to terminate at different points, while
    older launch files still pass one map-wide default goal.  If the YAML has
    finalmap.route_destinations.A/B, the planner can replace only that legacy
    shared goal with the route-specific destination.  Missing route-specific
    destinations return None so Scenario-2 checkpoint goals are not overridden.
    """
    data = load_routes(config_path)
    if map_name not in data:
        raise ValueError(f"맵 이름을 찾을 수 없습니다: {map_name}")
    m = data[map_name]
    map_goal = _xy(m.get("destination", [0.0, 0.0]))

    route_key = str(route_id).strip().upper()
    raw_dest = None

    # Preferred format:
    # finalmap:
    #   route_destinations:
    #     A: [50.0, 265.0]
    #     B: [130.0, 255.0]
    for section_name in ("route_destinations", "destinations"):
        section = m.get(section_name)
        if isinstance(section, dict):
            raw_dest = section.get(route_key) or section.get(str(route_id))
            if raw_dest is not None:
                break

    # Optional route-object format:
    # routes:
    #   A:
    #     waypoints: [...]
    #     destination: [...]
    if raw_dest is None:
        routes = m.get("routes", {})
        route_obj = routes.get(route_key) or routes.get(str(route_id))
        if isinstance(route_obj, dict):
            raw_dest = route_obj.get("destination")

    route_goal = _xy(raw_dest) if raw_dest is not None else None
    return route_goal, map_goal


def get_route_start_goal(map_name: str, config_path: str | None = None) -> tuple[Tuple[float, float], Tuple[float, float]]:
    data = load_routes(config_path)
    if map_name not in data:
        raise ValueError(f"맵 이름을 찾을 수 없습니다: {map_name}")
    m = data[map_name]
    start = m.get("start", [0.0, 0.0])
    goal = m.get("destination", [0.0, 0.0])
    return _xy(start), _xy(goal)
