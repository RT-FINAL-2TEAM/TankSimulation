# -*- coding: utf-8 -*-
"""
TankSimulation 최신 경로계획 작업을 ROS2로 이식한 노드.

TankSimulation.zip의 원본 의도:
- src/planning/path_planning.py: 장애물 inflate + LOS 스무딩이 들어간 격자 A*.
- tests/step3_threat_avoidance/run_server.py: 기본적으로 GT 장애물에 의존하지 않는다.
  LiDAR 장애물을 누적하고, 그것이 현재 루트를 막는지 감지한 뒤 재탐색한다.

기존 RViz/컨트롤러 노드가 계속 동작하도록 ROS2 출력은 동일하게 유지한다.

중요한 통합 정책:
- A* 전역 경로를 타이머 틱마다 재계산하지 않는다.
- 기본적으로 start/goal을 알게 되면 1회 계획하고, 이후엔 명시적 goal/GT-장애물 갱신 시에만 재탐색한다.
- LiDAR 기반 동적 재탐색은 opt-in이며 속도 제한이 걸린다. 일반적인 국소 회피는 APF가 처리해야 한다.
"""

import csv
import heapq
import json
import math
import os
import time
import threading
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Vector3Stamped
from nav_msgs.msg import Path as NavPath
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import String

from lidar.obstacle_memory import LidarObstacleMemory
from path_planning.config import (
    DEFAULT_GOAL_ENABLED,
    DEFAULT_GOAL_X,
    DEFAULT_GOAL_Y,
    DYNAMIC_REPLAN_COOLDOWN_SEC,
    ENABLE_DYNAMIC_REPLAN,
    ENABLE_PERIODIC_REPLAN,
    GOAL_TOLERANCE,
    LIDAR_BLOCK_MAX_DISTANCE,
    LIDAR_BLOCK_MIN_DISTANCE,
    LIDAR_CLUSTER_EPS,
    LIDAR_CLUSTER_MIN_SAMPLES,
    LIDAR_HISTORY_RESOLUTION,
    LOOKAHEAD_DISTANCE,
    MAP_FRAME,
    MAP_HEIGHT,
    MAP_RESOLUTION,
    MAP_WIDTH,
    MAX_EXPANSIONS,
    MAX_LIDAR_HISTORY_POINTS,
    OBSTACLE_INFLATE,
    PATH_BLOCK_MARGIN,
    PATH_BLOCK_REQUIRED_HITS,
    PLAN_RETRY_PERIOD_SEC,
    PLANNER_HZ,
    PREFAB_HALF_SIZES,
    prefab_half_size,
    PUBLISH_PATH_PERIOD_SEC,
    REPLAN_PERIOD_SEC,
    TOPIC_GLOBAL_PATH,
    TOPIC_GOAL_POSE,
    TOPIC_LIDAR_BBOXES,
    TOPIC_LIDAR_CLUSTERS,
    TOPIC_LIDAR_DETECTED_MAP,
    TOPIC_LOOKAHEAD_POSE,
    TOPIC_MAP_OBSTACLES,
    TOPIC_DISCOVERED_OBJECTS,
    TOPIC_PATH_POINTS,
    TOPIC_PLANNER_STATUS,
    TOPIC_PLAYER_POSE,
    TOPIC_PLAYER_STATE,
    USE_GT_OBSTACLES,
    USE_LIDAR_CLUSTER_BBOXES,
    USE_PATH_SMOOTHING,
    USE_ROUTE_WAYPOINTS,
    ROUTE_CLEARANCE_WEIGHT,
    ROUTE_CONFIG_FILE,
    ROUTE_ID,
    ROUTE_MAP_NAME,
    ROUTE_SIDE,
    USE_STATIC_MAP,
    STATIC_MAP_FILE,
    TERRAIN_COST_FILE,
    TERRAIN_WEIGHT,
    LIDAR_CLUSTER_BBOX_MARGIN,
)

from ament_index_python.packages import get_package_share_directory
from path_planning.route_loader import get_route_waypoints
from path_planning.team_path_planning import (
    plan_path_through_waypoints as team_plan_path_through_waypoints,
    plan_global_path as team_plan_global_path,
    load_static_obstacles_from_map,
    DEFAULT_STATIC_INFLATE,
)


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def parse_float_map(raw: str, default: Dict[str, float]) -> Dict[str, float]:
    """Parse 'tank:100,house:50' style parameter strings."""
    out = dict(default)
    try:
        for item in str(raw or "").split(","):
            item = item.strip()
            if not item or ":" not in item:
                continue
            key, val = item.split(":", 1)
            key = key.strip().lower()
            if not key:
                continue
            out[key] = float(val.strip())
    except Exception:
        return dict(default)
    return out


from tank_common.pointcloud import pointcloud2_to_xyz_array
from tank_common.vehicle_model import TankVehicleModel


def get_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def create_grid(width: int, height: int, resolution: float) -> List[List[int]]:
    cols = int(width / resolution)
    rows = int(height / resolution)
    return [[0 for _ in range(cols)] for _ in range(rows)]


def add_obstacles(grid: List[List[int]], obstacles: List[Dict[str, float]], res: float, inflate: float) -> None:
    rows, cols = len(grid), len(grid[0])
    for obs in obstacles:
        try:
            obs_inflate = float(obs.get("_inflate_override", inflate)) if isinstance(obs, dict) else float(inflate)
            x_min = max(0, int((float(obs["x_min"]) - obs_inflate) / res))
            x_max = min(cols - 1, int((float(obs["x_max"]) + obs_inflate) / res))
            z_min = max(0, int((float(obs["z_min"]) - obs_inflate) / res))
            z_max = min(rows - 1, int((float(obs["z_max"]) + obs_inflate) / res))
        except Exception:
            continue
        for z in range(z_min, z_max + 1):
            for x in range(x_min, x_max + 1):
                grid[z][x] = 1


def heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def get_neighbors(node: Tuple[int, int], grid: List[List[int]]) -> Iterable[Tuple[int, int]]:
    directions = [
        (0, 1), (1, 0), (0, -1), (-1, 0),
        (1, 1), (1, -1), (-1, 1), (-1, -1),
    ]
    rows, cols = len(grid), len(grid[0])
    x, y = node
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < cols and 0 <= ny < rows and grid[ny][nx] == 0:
            yield nx, ny


def reconstruct_path(came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]], start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
    if goal not in came_from:
        return []
    cur = goal
    path: List[Tuple[int, int]] = []
    while cur != start:
        path.append(cur)
        parent = came_from.get(cur)
        if parent is None:
            return []
        cur = parent
    path.append(start)
    path.reverse()
    return path


def astar_search(grid: List[List[int]], start: Tuple[int, int], goal: Tuple[int, int], max_expansions: int = 250000) -> List[Tuple[int, int]]:
    frontier: List[Tuple[float, int, Tuple[int, int]]] = []
    counter = 0
    heapq.heappush(frontier, (0.0, counter, start))
    came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
    cost_so_far: Dict[Tuple[int, int], float] = {start: 0.0}
    expansions = 0

    while frontier:
        _, _, current = heapq.heappop(frontier)
        if current == goal:
            break
        expansions += 1
        if expansions > max_expansions:
            break
        for nxt in get_neighbors(current, grid):
            move_cost = 1.41421356 if nxt[0] != current[0] and nxt[1] != current[1] else 1.0
            new_cost = cost_so_far[current] + move_cost
            if nxt not in cost_so_far or new_cost < cost_so_far[nxt]:
                cost_so_far[nxt] = new_cost
                counter += 1
                priority = new_cost + heuristic(nxt, goal)
                heapq.heappush(frontier, (priority, counter, nxt))
                came_from[nxt] = current
    return reconstruct_path(came_from, start, goal)


def has_line_of_sight(grid: List[List[int]], p1: Tuple[int, int], p2: Tuple[int, int]) -> bool:
    x0, y0 = p1
    x1, y1 = p2
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x_inc = 1 if x0 < x1 else -1
    y_inc = 1 if y0 < y1 else -1
    err = dx - dy
    rows, cols = len(grid), len(grid[0])
    while True:
        if not (0 <= x0 < cols and 0 <= y0 < rows):
            return False
        if grid[y0][x0] == 1:
            return False
        if x0 == x1 and y0 == y1:
            return True
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += x_inc
        if e2 < dx:
            err += dx
            y0 += y_inc


def smooth_path(grid: List[List[int]], path: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if len(path) <= 2:
        return path
    smoothed = [path[0]]
    current_idx = 0
    while current_idx < len(path) - 1:
        furthest_idx = current_idx + 1
        for i in range(current_idx + 2, len(path)):
            if has_line_of_sight(grid, path[current_idx], path[i]):
                furthest_idx = i
            else:
                break
        smoothed.append(path[furthest_idx])
        current_idx = furthest_idx
    return smoothed


def find_nearest_free_grid(grid: List[List[int]], node: Tuple[int, int], max_radius: int = 30) -> Optional[Tuple[int, int]]:
    rows, cols = len(grid), len(grid[0])
    x0 = int(clamp(node[0], 0, cols - 1))
    y0 = int(clamp(node[1], 0, rows - 1))
    if grid[y0][x0] == 0:
        return x0, y0
    for r in range(1, max_radius + 1):
        best: List[Tuple[float, Tuple[int, int]]] = []
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if abs(dx) != r and abs(dy) != r:
                    continue
                x, y = x0 + dx, y0 + dy
                if 0 <= x < cols and 0 <= y < rows and grid[y][x] == 0:
                    best.append((math.hypot(dx, dy), (x, y)))
        if best:
            best.sort(key=lambda item: item[0])
            return best[0][1]
    return None


def plan_global_path(
    start_pos: Tuple[float, float],
    goal_pos: Tuple[float, float],
    obstacles: List[Dict[str, float]],
    width: int = 300,
    height: int = 300,
    resolution: float = 1.0,
    inflate: float = 5.0,
    use_smoothing: bool = True,
    max_expansions: int = 250000,
    static_obstacles: Optional[List[Dict[str, float]]] = None,
    static_inflate: float = DEFAULT_STATIC_INFLATE,
) -> List[Tuple[float, float]]:
    grid = create_grid(width, height, resolution)
    if static_obstacles:
        # Known static map obstacle은 hard no-go다. 기존 1.0m inflate는 초록점 사이 통과를 허용했다.
        add_obstacles(grid, static_obstacles, resolution, static_inflate)
    add_obstacles(grid, obstacles, resolution, inflate)

    start_grid = (int(start_pos[0] / resolution), int(start_pos[1] / resolution))
    goal_grid = (int(goal_pos[0] / resolution), int(goal_pos[1] / resolution))
    start_grid = find_nearest_free_grid(grid, start_grid) or start_grid
    goal_grid = find_nearest_free_grid(grid, goal_grid) or goal_grid

    grid_path = astar_search(grid, start_grid, goal_grid, max_expansions=max_expansions)
    if not grid_path:
        return []
    if use_smoothing:
        grid_path = smooth_path(grid, grid_path)
    return [(p[0] * resolution, p[1] * resolution) for p in grid_path]


def extract_payload_list(data: Any, key: str = "obstacles") -> List[Any]:
    """브릿지 payload와 직접 payload를 모두 허용한다."""
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



def obstacle_to_bbox(obs: Dict[str, Any]) -> Optional[Dict[str, float]]:
    if all(k in obs for k in ("x_min", "x_max", "z_min", "z_max")):
        try:
            return {"x_min": float(obs["x_min"]), "x_max": float(obs["x_max"]), "z_min": float(obs["z_min"]), "z_max": float(obs["z_max"])}
        except Exception:
            return None
    pos = obs.get("position") if isinstance(obs.get("position"), dict) else None
    if pos is None:
        return None
    try:
        x = float(pos.get("x", 0.0))
        z = float(pos.get("z", 0.0))
    except Exception:
        return None
    hw, hl = prefab_half_size(str(obs.get("prefabName", "")))
    return {"x_min": x - hw, "x_max": x + hw, "z_min": z - hl, "z_max": z + hl}


def parse_obstacles_payload(payload: Any) -> List[Dict[str, float]]:
    bboxes: List[Dict[str, float]] = []
    for item in extract_payload_list(payload, "obstacles"):
        if isinstance(item, dict):
            bbox = obstacle_to_bbox(item)
            if bbox is not None:
                bboxes.append(bbox)
    return bboxes

DISCOVERED_CLASS_RADIUS = {
    # A* persistent obstacle 반경. 값은 전체 폭이 아니라 중심 기준 반경이다.
    # rock/car 실제 폭이 약 8m라면 radius는 4m가 기준이다. 추가 안전여유는
    # discovered_obstacle_inflate에서 별도로 준다.
    "rock": 4.0,
    "car": 4.0,
    "tank": 4.5,
    "house": 5.5,
    "wall": 4.0,
    "tent": 4.0,
    "tree": 4.0,
    "unknown": 3.0,
}


DEFAULT_SEMANTIC_RISK_SCORES = {
    "tank": 100.0,
    "house": 50.0,
    "car": 25.0,
    "tent": 15.0,
    "rock": 10.0,
    "unknown": 5.0,
}
DEFAULT_SEMANTIC_RISK_RADII = {
    "tank": 25.0,
    "house": 18.0,
    "car": 10.0,
    "tent": 8.0,
    "rock": 6.0,
    "unknown": 5.0,
}


def _as_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _as_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "y", "confirmed")
    return False


def _discovered_xy(obj: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    # /tank/map/discovered/objects 규약: DiscoveredObject.map_x/map_y가 A* 평면 x/y이다.
    if "map_x" in obj and "map_y" in obj:
        return _as_float(obj.get("map_x")), _as_float(obj.get("map_y"))
    pos = obj.get("position_map") if isinstance(obj.get("position_map"), dict) else None
    if pos is not None:
        return _as_float(pos.get("x")), _as_float(pos.get("y", pos.get("z", 0.0)))
    # 저장된 discovered .map 규약은 Unity raw: position.x=map.x, position.z=map.y
    pos = obj.get("position") if isinstance(obj.get("position"), dict) else None
    if pos is not None:
        return _as_float(pos.get("x")), _as_float(pos.get("z", pos.get("y", 0.0)))
    return None


def parse_discovered_objects_payload(
    payload: Any,
    *,
    confirmed_only: bool,
    min_observations: int,
    ignored_classes: set[str],
    default_radius: float,
) -> List[Dict[str, float]]:
    if not isinstance(payload, dict):
        return []
    objects = payload.get("objects")
    if not isinstance(objects, list):
        return []
    bboxes: List[Dict[str, float]] = []
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        cls = str(obj.get("class_name", obj.get("className", obj.get("class", "unknown")))).strip().lower()
        if cls in ignored_classes:
            continue
        obs_count = int(_as_float(obj.get("observation_count", obj.get("observations", 0)), 0.0))
        confirmed = _as_bool(obj.get("is_confirmed", obj.get("confirmed", False)))
        if confirmed_only and not confirmed:
            continue
        if obs_count < max(0, min_observations):
            continue
        xy = _discovered_xy(obj)
        if xy is None:
            continue
        x, y = xy
        radius = float(DISCOVERED_CLASS_RADIUS.get(cls, default_radius))
        bboxes.append({
            "x_min": x - radius,
            "x_max": x + radius,
            "z_min": y - radius,
            "z_max": y + radius,
            "source": "discovered_object",
            "class_name": cls,
            "object_id": str(obj.get("object_id", obj.get("id", ""))),
            "observation_count": obs_count,
            "is_confirmed": confirmed,
        })
    return bboxes


def point_in_bbox_with_margin(point: Tuple[float, float], bbox: Dict[str, float], margin: float) -> bool:
    x, y = point
    return (
        float(bbox.get("x_min", 0.0)) - margin <= x <= float(bbox.get("x_max", 0.0)) + margin
        and float(bbox.get("z_min", 0.0)) - margin <= y <= float(bbox.get("z_max", 0.0)) + margin
    )


def bbox_center(bbox: Dict[str, float]) -> Tuple[float, float]:
    return (
        0.5 * (float(bbox.get("x_min", 0.0)) + float(bbox.get("x_max", 0.0))),
        0.5 * (float(bbox.get("z_min", 0.0)) + float(bbox.get("z_max", 0.0))),
    )


def bbox_center_distance(a: Dict[str, float], b: Dict[str, float]) -> float:
    ax, ay = bbox_center(a)
    bx, by = bbox_center(b)
    return math.hypot(ax - bx, ay - by)


def bbox_copy_as_memory(bbox: Dict[str, float], now: float, *, hits: int = 1) -> Dict[str, float]:
    cx, cy = bbox_center(bbox)
    copied = {
        "x_min": float(bbox.get("x_min", cx)),
        "x_max": float(bbox.get("x_max", cx)),
        "z_min": float(bbox.get("z_min", cy)),
        "z_max": float(bbox.get("z_max", cy)),
        "source": "lidar_cluster_memory",
        "cluster_id": int(bbox.get("cluster_id", -1)) if isinstance(bbox.get("cluster_id", -1), (int, float)) else -1,
        "count": int(bbox.get("count", 0)) if isinstance(bbox.get("count", 0), (int, float)) else 0,
        "first_seen_wall": float(bbox.get("first_seen_wall", now)),
        "last_seen_wall": float(now),
        "hits": int(hits),
    }
    return copied


def is_path_blocked_by_bboxes(
    current_pos: Tuple[float, float],
    route: Sequence[Tuple[float, float]],
    route_index: int,
    bboxes: Sequence[Dict[str, float]],
    min_distance: float,
    max_distance: float,
    margin: float,
) -> bool:
    """Persistent discovered/static-like bbox가 현재 진행 경로 corridor를 막는지 검사한다.

    LiDAR obstacle memory와 달리 discovered object는 point history가 아니라 bbox이므로,
    현재 위치부터 lookahead 거리까지 route point/segment 샘플이 bbox+margin 안으로 들어가면 blocked로 본다.
    """
    if not bboxes or not route:
        return False
    start_i = max(0, min(route_index, len(route) - 1))
    prev = current_pos
    dist_from_now = 0.0
    sample_step = 1.0
    for i in range(start_i, len(route)):
        nxt = route[i]
        seg_len = get_distance(prev, nxt)
        steps = max(1, int(math.ceil(seg_len / sample_step)))
        for k in range(1, steps + 1):
            t = k / steps
            p = (prev[0] + (nxt[0] - prev[0]) * t, prev[1] + (nxt[1] - prev[1]) * t)
            d_inc = get_distance(prev, p) if k == 1 else sample_step
            # 누적 거리는 근사로 충분하다. 너무 먼 미래의 discovered 때문에 과민 재계획하지 않게 max_distance로 제한.
            dist_from_now += d_inc
            if dist_from_now < min_distance:
                continue
            if dist_from_now > max_distance:
                return False
            for bbox in bboxes:
                if point_in_bbox_with_margin(p, bbox, margin):
                    return True
        prev = nxt
    return False


def find_lookahead_along_path(pos: Tuple[float, float], route: Sequence[Tuple[float, float]], lookahead_dist: float) -> Tuple[Tuple[float, float], int]:
    """현재 위치를 경로에 투영한 뒤, lookahead 거리만큼 전진한 지점을 구한다."""
    if not route:
        return pos, 0
    if len(route) == 1:
        return route[0], 0

    best_i = 0
    best_t = 0.0
    best_dist = float("inf")
    for i in range(len(route) - 1):
        a = route[i]
        b = route[i + 1]
        ax, ay = a
        bx, by = b
        dx, dy = bx - ax, by - ay
        denom = dx * dx + dy * dy
        t = 0.0 if denom <= 1e-9 else clamp(((pos[0] - ax) * dx + (pos[1] - ay) * dy) / denom, 0.0, 1.0)
        proj = (ax + t * dx, ay + t * dy)
        d = get_distance(pos, proj)
        if d < best_dist:
            best_dist = d
            best_i = i
            best_t = t

    remaining = max(lookahead_dist, 0.1)
    i = best_i
    a = route[i]
    b = route[i + 1]
    seg_len = get_distance(a, b)
    cur = (a[0] + best_t * (b[0] - a[0]), a[1] + best_t * (b[1] - a[1]))
    first_remaining = seg_len * (1.0 - best_t)
    if remaining <= first_remaining and seg_len > 1e-9:
        ratio = remaining / seg_len
        target = (cur[0] + ratio * (b[0] - a[0]), cur[1] + ratio * (b[1] - a[1]))
        return target, i
    remaining -= first_remaining
    for j in range(i + 1, len(route) - 1):
        a = route[j]
        b = route[j + 1]
        seg_len = get_distance(a, b)
        if remaining <= seg_len and seg_len > 1e-9:
            r = remaining / seg_len
            return (a[0] + r * (b[0] - a[0]), a[1] + r * (b[1] - a[1])), j
        remaining -= seg_len
    return route[-1], len(route) - 1


class TeamDynamicAStarPlannerNode(Node):
    def __init__(self) -> None:
        super().__init__("tank_team_dynamic_astar_planner_node")
        self.declare_parameter("map_width", MAP_WIDTH)
        self.declare_parameter("map_height", MAP_HEIGHT)
        self.declare_parameter("resolution", MAP_RESOLUTION)
        self.declare_parameter("inflate", OBSTACLE_INFLATE)
        self.declare_parameter("use_path_smoothing", USE_PATH_SMOOTHING)
        self.declare_parameter("use_gt_obstacles", USE_GT_OBSTACLES)
        self.declare_parameter("enable_dynamic_replan", ENABLE_DYNAMIC_REPLAN)
        # 주기적 재탐색은 전차가 끊임없이 움직이는 전역경로를 쫓게 만들었다.
        # 기본은 비활성으로 둔다. 의도적인 실험에서만 >0으로 설정한다.
        self.declare_parameter("enable_periodic_replan", ENABLE_PERIODIC_REPLAN)
        self.declare_parameter("replan_period_sec", REPLAN_PERIOD_SEC)
        # LiDAR 재탐색에도 속도 제한을 둔다. raw LiDAR 점이 현재 경로를 반복적으로 막힘으로
        # 표시해 계획 루프를 유발할 수 있기 때문이다.
        self.declare_parameter("dynamic_replan_cooldown_sec", DYNAMIC_REPLAN_COOLDOWN_SEC)
        self.declare_parameter("plan_retry_period_sec", PLAN_RETRY_PERIOD_SEC)
        self.declare_parameter("path_block_margin", PATH_BLOCK_MARGIN)
        self.declare_parameter("path_block_required_hits", PATH_BLOCK_REQUIRED_HITS)
        self.declare_parameter("dynamic_replan_max_count", 0)
        self.declare_parameter("dynamic_replan_min_progress_m", 0.0)
        self.declare_parameter("dynamic_replan_progress_guard_sec", 0.0)
        self.declare_parameter("lidar_block_min_distance", LIDAR_BLOCK_MIN_DISTANCE)
        self.declare_parameter("lidar_block_max_distance", LIDAR_BLOCK_MAX_DISTANCE)
        self.declare_parameter("lidar_cluster_eps", LIDAR_CLUSTER_EPS)
        self.declare_parameter("lidar_cluster_min_samples", LIDAR_CLUSTER_MIN_SAMPLES)
        self.declare_parameter("lidar_history_resolution", LIDAR_HISTORY_RESOLUTION)
        self.declare_parameter("max_lidar_history_points", MAX_LIDAR_HISTORY_POINTS)
        self.declare_parameter("lookahead_distance", LOOKAHEAD_DISTANCE)
        self.declare_parameter("publish_path_period_sec", PUBLISH_PATH_PERIOD_SEC)
        self.declare_parameter("goal_tolerance", GOAL_TOLERANCE)
        self.declare_parameter("default_goal_enabled", DEFAULT_GOAL_ENABLED)
        self.declare_parameter("default_goal_x", DEFAULT_GOAL_X)
        self.declare_parameter("default_goal_y", DEFAULT_GOAL_Y)
        self.declare_parameter("max_expansions", MAX_EXPANSIONS)
        self.declare_parameter("use_route_waypoints", USE_ROUTE_WAYPOINTS)
        self.declare_parameter("route_map_name", ROUTE_MAP_NAME)
        self.declare_parameter("route_id", ROUTE_ID)
        self.declare_parameter("route_side", ROUTE_SIDE)
        self.declare_parameter("route_clearance_weight", ROUTE_CLEARANCE_WEIGHT)
        self.declare_parameter("route_config_file", ROUTE_CONFIG_FILE)
        self.declare_parameter("use_static_map", USE_STATIC_MAP)
        self.declare_parameter("static_map_file", STATIC_MAP_FILE)
        self.declare_parameter("terrain_cost_file", TERRAIN_COST_FILE)
        self.declare_parameter("terrain_weight", TERRAIN_WEIGHT)
        self.declare_parameter("tank_param_file", "")
        self.declare_parameter("enable_speed_based_inflation", True)
        self.declare_parameter("enable_speed_based_emergency_replan", True)
        self.declare_parameter("enable_path_feasibility_check", True)
        self.declare_parameter("path_feasibility_topic", "/tank/planner/feasibility_status")
        # Semantic-risk-aware A*: confirmed discovered object 주변에 class별 soft cost를 추가한다.
        self.declare_parameter("enable_semantic_risk_cost", True)
        self.declare_parameter("semantic_risk_weight", 0.06)
        self.declare_parameter("semantic_risk_radius_scale", 1.0)
        self.declare_parameter("semantic_risk_scores", "tank:100,house:50,car:25,tent:15,rock:10,unknown:5")
        self.declare_parameter("semantic_risk_radii", "tank:25,house:18,car:10,tent:8,rock:6,unknown:5")
        # theta-aware-lite: A* 상태에 8방향 heading index를 추가해 급격한 꺾임을 비용화한다.
        self.declare_parameter("enable_theta_aware_astar", True)
        self.declare_parameter("theta_heading_change_weight", 0.25)
        # Curvature smoother: A* polyline corner를 전차 회전반경 기준 곡선으로 후처리한다.
        self.declare_parameter("enable_curvature_path_smoothing", True)
        self.declare_parameter("curvature_smoothing_min_turn_radius_m", 7.0)
        self.declare_parameter("curvature_smoothing_max_corner_angle_deg", 25.0)
        self.declare_parameter("curvature_smoothing_point_spacing_m", 1.0)
        self.declare_parameter("curvature_smoothing_collision_check_margin_m", 1.0)
        # Replan acceptance: dense obstacle에서 새 경로가 조금 보였다고 바로 교체하지 않는다.
        self.declare_parameter("enable_replan_acceptance_filter", True)
        self.declare_parameter("path_commitment_sec", 4.0)
        self.declare_parameter("replan_accept_max_length_ratio", 1.30)
        self.declare_parameter("replan_accept_max_sharp_corner_increase", 1)
        self.declare_parameter("replan_accept_max_heading_change_increase_deg", 120.0)
        self.declare_parameter("enable_avoid_side_lock", True)
        self.declare_parameter("avoid_side_lock_sec", 4.0)
        self.declare_parameter("use_lidar_cluster_bboxes", USE_LIDAR_CLUSTER_BBOXES)
        self.declare_parameter("lidar_cluster_bbox_margin", LIDAR_CLUSTER_BBOX_MARGIN)
        # Known/discovered obstacle costmap policy.
        self.declare_parameter("static_obstacle_inflate", DEFAULT_STATIC_INFLATE)
        self.declare_parameter("use_discovered_objects_for_astar", True)
        self.declare_parameter("discovered_objects_topic", TOPIC_DISCOVERED_OBJECTS)
        self.declare_parameter("discovered_confirmed_only", True)
        self.declare_parameter("discovered_min_observations", 2)
        self.declare_parameter("discovered_default_radius", 3.5)
        self.declare_parameter("discovered_obstacle_inflate", 2.0)
        self.declare_parameter("ignored_discovered_classes_for_astar", "person,human,blue,red")
        # LiDAR cluster persistence: 현재 프레임에서 cluster가 잠깐 사라져도 A* costmap에는 TTL 동안 유지한다.
        self.declare_parameter("enable_lidar_cluster_memory", True)
        self.declare_parameter("lidar_cluster_memory_ttl_sec", 18.0)
        self.declare_parameter("lidar_cluster_memory_merge_distance", 5.0)
        self.declare_parameter("lidar_cluster_memory_inflate", 3.0)
        self.declare_parameter("lidar_cluster_memory_max_count", 80)
        self.declare_parameter("use_lidar_cluster_memory_for_path_block", False)
        # Route/checkpoint commitment: dynamic replan이 체크포인트 진행 상태를 뒤로 되돌려
        # lookahead가 좌우로 튀는 것을 막는다.
        self.declare_parameter("route_index_never_decrease", True)
        self.declare_parameter("dynamic_replan_keep_route_index", True)
        self.declare_parameter("route_commit_lock_sec", 6.0)
        # Checkpoint progress lock: route waypoint를 한 번 지났으면 dynamic/emergency replan 이후에도
        # 이미 지난 checkpoint를 through list에 다시 넣지 않는다. route_index는 path point index라
        # 새 A* 경로마다 의미가 바뀌므로 checkpoint 진행도는 별도로 관리한다.
        self.declare_parameter("route_checkpoint_never_decrease", True)
        self.declare_parameter("route_checkpoint_reached_radius", 8.0)
        self.declare_parameter("route_checkpoint_passed_z_margin", 3.0)
        # Path-block trigger는 현재 보이는 cluster를 주력으로 사용한다.
        # history/discovered는 A* costmap에는 넣되, 반복 replan trigger로 쓰면 경로가 흔들릴 수 있다.
        self.declare_parameter("use_lidar_memory_for_path_block", False)
        self.declare_parameter("use_discovered_objects_for_path_block", False)
        # Emergency fast replan: 현재 보이는 LiDAR cluster가 전차 전방 가까운 A* corridor를 막으면
        # 일반 5초 cooldown/2-hit 조건보다 빠르게 재계획한다. Memory/discovered는 여기서 쓰지 않는다.
        self.declare_parameter("emergency_cluster_replan_enabled", True)
        self.declare_parameter("emergency_replan_cooldown_sec", 1.5)
        self.declare_parameter("emergency_replan_front_distance", 16.0)
        self.declare_parameter("emergency_replan_min_distance", 0.0)
        self.declare_parameter("emergency_replan_margin", 8.0)
        # One-frame LiDAR cluster glitches can make A* jump to the opposite corridor.
        # Require the emergency block to be stable for several planner ticks before accepting a replan.
        self.declare_parameter("emergency_replan_required_hits", 3)
        # Dynamic emergency margin should not grow without bound from the speed-based model;
        # keep trigger detection less conservative than the costmap inflation.
        self.declare_parameter("emergency_replan_margin_max", 11.0)

        # APF를 끈 상태에서도 기존 RViz potential marker 표시를 유지하기 위한 시각화 mirror.
        # 제어에는 사용하지 않고, rviz_visualizer_node가 구독하던 토픽에 A* lookahead 기반
        # target 점과 desired heading vector만 발행한다.
        self.declare_parameter("publish_lookahead_visualization_mirror", True)
        self.declare_parameter("visualization_local_target_topic", "/tank/local_target/pose")
        self.declare_parameter("visualization_result_vector_topic", "/tank/potential/result_vector")
        self.declare_parameter("visualization_attractive_vector_topic", "/tank/potential/attractive_vector")

        self.map_width = int(self.get_parameter("map_width").value)
        self.map_height = int(self.get_parameter("map_height").value)
        self.resolution = float(self.get_parameter("resolution").value)
        self.inflate = float(self.get_parameter("inflate").value)
        self.use_path_smoothing = bool(self.get_parameter("use_path_smoothing").value)
        self.use_gt_obstacles = bool(self.get_parameter("use_gt_obstacles").value)
        self.enable_dynamic_replan = bool(self.get_parameter("enable_dynamic_replan").value)
        self.enable_periodic_replan = bool(self.get_parameter("enable_periodic_replan").value)
        self.replan_period_sec = float(self.get_parameter("replan_period_sec").value)
        self.dynamic_replan_cooldown_sec = float(self.get_parameter("dynamic_replan_cooldown_sec").value)
        self.plan_retry_period_sec = float(self.get_parameter("plan_retry_period_sec").value)
        self.path_block_margin = float(self.get_parameter("path_block_margin").value)
        self.path_block_required_hits = max(1, int(self.get_parameter("path_block_required_hits").value))
        self.dynamic_replan_max_count = int(self.get_parameter("dynamic_replan_max_count").value)
        self.dynamic_replan_min_progress_m = float(self.get_parameter("dynamic_replan_min_progress_m").value)
        self.dynamic_replan_progress_guard_sec = float(self.get_parameter("dynamic_replan_progress_guard_sec").value)
        self.lidar_block_min_distance = float(self.get_parameter("lidar_block_min_distance").value)
        self.lidar_block_max_distance = float(self.get_parameter("lidar_block_max_distance").value)
        self.lidar_cluster_eps = float(self.get_parameter("lidar_cluster_eps").value)
        self.lidar_cluster_min_samples = int(self.get_parameter("lidar_cluster_min_samples").value)
        self.lidar_history_resolution = float(self.get_parameter("lidar_history_resolution").value)
        self.max_lidar_history_points = int(self.get_parameter("max_lidar_history_points").value)
        self.lookahead_distance = float(self.get_parameter("lookahead_distance").value)
        self.publish_path_period_sec = float(self.get_parameter("publish_path_period_sec").value)
        self.goal_tolerance = float(self.get_parameter("goal_tolerance").value)
        self.max_expansions = int(self.get_parameter("max_expansions").value)
        self.use_route_waypoints = bool(self.get_parameter("use_route_waypoints").value)
        self.route_map_name = str(self.get_parameter("route_map_name").value)
        self.route_id = str(self.get_parameter("route_id").value)
        self.route_side = str(self.get_parameter("route_side").value)
        # route_side는 route_id로 결정된다(A=서/B=동). launch에서 route_side를 빠뜨려 기본값(west)이
        # 들어와도 B가 동쪽으로 가도록 route_id에 맞춰 자동 보정한다(side-bias가 웨이포인트와 싸우는 버그 방지).
        _side_by_id = {"A": "west", "B": "east"}
        _expected_side = _side_by_id.get(self.route_id.strip().upper())
        if _expected_side and self.route_side != _expected_side:
            self.get_logger().warn(
                f"route_side='{self.route_side}'가 route_id='{self.route_id}' 기대값('{_expected_side}')과 "
                f"불일치 → '{_expected_side}'로 보정")
            self.route_side = _expected_side
        self.route_clearance_weight = float(self.get_parameter("route_clearance_weight").value)
        self.route_config_file = str(self.get_parameter("route_config_file").value)
        self.use_static_map = bool(self.get_parameter("use_static_map").value)
        self.static_map_file = str(self.get_parameter("static_map_file").value)
        self.terrain_cost_file = str(self.get_parameter("terrain_cost_file").value)
        self.terrain_weight = float(self.get_parameter("terrain_weight").value)
        self.tank_param_file = str(self.get_parameter("tank_param_file").value)
        self.vehicle_model = TankVehicleModel(self.tank_param_file)
        self.enable_speed_based_inflation = bool(self.get_parameter("enable_speed_based_inflation").value)
        self.enable_speed_based_emergency_replan = bool(self.get_parameter("enable_speed_based_emergency_replan").value)
        self.enable_path_feasibility_check = bool(self.get_parameter("enable_path_feasibility_check").value)
        self.path_feasibility_topic = str(self.get_parameter("path_feasibility_topic").value)
        self.enable_semantic_risk_cost = bool(self.get_parameter("enable_semantic_risk_cost").value)
        self.semantic_risk_weight = float(self.get_parameter("semantic_risk_weight").value)
        self.semantic_risk_radius_scale = float(self.get_parameter("semantic_risk_radius_scale").value)
        self.semantic_risk_scores = parse_float_map(
            str(self.get_parameter("semantic_risk_scores").value), DEFAULT_SEMANTIC_RISK_SCORES
        )
        self.semantic_risk_radii = parse_float_map(
            str(self.get_parameter("semantic_risk_radii").value), DEFAULT_SEMANTIC_RISK_RADII
        )
        self.enable_theta_aware_astar = bool(self.get_parameter("enable_theta_aware_astar").value)
        self.theta_heading_change_weight = float(self.get_parameter("theta_heading_change_weight").value)
        self.enable_curvature_path_smoothing = bool(self.get_parameter("enable_curvature_path_smoothing").value)
        self.curvature_smoothing_min_turn_radius_m = float(self.get_parameter("curvature_smoothing_min_turn_radius_m").value)
        self.curvature_smoothing_max_corner_angle_deg = float(self.get_parameter("curvature_smoothing_max_corner_angle_deg").value)
        self.curvature_smoothing_point_spacing_m = float(self.get_parameter("curvature_smoothing_point_spacing_m").value)
        self.curvature_smoothing_collision_check_margin_m = float(self.get_parameter("curvature_smoothing_collision_check_margin_m").value)
        self.enable_replan_acceptance_filter = bool(self.get_parameter("enable_replan_acceptance_filter").value)
        self.path_commitment_sec = float(self.get_parameter("path_commitment_sec").value)
        self.replan_accept_max_length_ratio = float(self.get_parameter("replan_accept_max_length_ratio").value)
        self.replan_accept_max_sharp_corner_increase = int(self.get_parameter("replan_accept_max_sharp_corner_increase").value)
        self.replan_accept_max_heading_change_increase_deg = float(self.get_parameter("replan_accept_max_heading_change_increase_deg").value)
        self.enable_avoid_side_lock = bool(self.get_parameter("enable_avoid_side_lock").value)
        self.avoid_side_lock_sec = float(self.get_parameter("avoid_side_lock_sec").value)
        self.use_lidar_cluster_bboxes = bool(self.get_parameter("use_lidar_cluster_bboxes").value)
        self.lidar_cluster_bbox_margin = float(self.get_parameter("lidar_cluster_bbox_margin").value)
        self.static_obstacle_inflate = float(self.get_parameter("static_obstacle_inflate").value)
        self.use_discovered_objects_for_astar = bool(self.get_parameter("use_discovered_objects_for_astar").value)
        self.discovered_objects_topic = str(self.get_parameter("discovered_objects_topic").value)
        self.discovered_confirmed_only = bool(self.get_parameter("discovered_confirmed_only").value)
        self.discovered_min_observations = int(self.get_parameter("discovered_min_observations").value)
        self.discovered_default_radius = float(self.get_parameter("discovered_default_radius").value)
        self.discovered_obstacle_inflate = float(self.get_parameter("discovered_obstacle_inflate").value)
        ignored_disc_raw = str(self.get_parameter("ignored_discovered_classes_for_astar").value)
        self.ignored_discovered_classes_for_astar = {c.strip().lower() for c in ignored_disc_raw.split(",") if c.strip()}
        self.enable_lidar_cluster_memory = bool(self.get_parameter("enable_lidar_cluster_memory").value)
        self.lidar_cluster_memory_ttl_sec = float(self.get_parameter("lidar_cluster_memory_ttl_sec").value)
        self.lidar_cluster_memory_merge_distance = float(self.get_parameter("lidar_cluster_memory_merge_distance").value)
        self.lidar_cluster_memory_inflate = float(self.get_parameter("lidar_cluster_memory_inflate").value)
        self.lidar_cluster_memory_max_count = int(self.get_parameter("lidar_cluster_memory_max_count").value)
        self.use_lidar_cluster_memory_for_path_block = bool(self.get_parameter("use_lidar_cluster_memory_for_path_block").value)
        self.route_index_never_decrease = bool(self.get_parameter("route_index_never_decrease").value)
        self.dynamic_replan_keep_route_index = bool(self.get_parameter("dynamic_replan_keep_route_index").value)
        self.route_commit_lock_sec = float(self.get_parameter("route_commit_lock_sec").value)
        self.route_checkpoint_never_decrease = bool(self.get_parameter("route_checkpoint_never_decrease").value)
        self.route_checkpoint_reached_radius = float(self.get_parameter("route_checkpoint_reached_radius").value)
        self.route_checkpoint_passed_z_margin = float(self.get_parameter("route_checkpoint_passed_z_margin").value)
        self.use_lidar_memory_for_path_block = bool(self.get_parameter("use_lidar_memory_for_path_block").value)
        self.use_discovered_objects_for_path_block = bool(self.get_parameter("use_discovered_objects_for_path_block").value)
        self.emergency_cluster_replan_enabled = bool(self.get_parameter("emergency_cluster_replan_enabled").value)
        self.emergency_replan_cooldown_sec = float(self.get_parameter("emergency_replan_cooldown_sec").value)
        self.emergency_replan_front_distance = float(self.get_parameter("emergency_replan_front_distance").value)
        self.emergency_replan_min_distance = float(self.get_parameter("emergency_replan_min_distance").value)
        self.emergency_replan_margin = float(self.get_parameter("emergency_replan_margin").value)
        self.emergency_replan_required_hits = max(1, int(self.get_parameter("emergency_replan_required_hits").value))
        self.emergency_replan_margin_max = float(self.get_parameter("emergency_replan_margin_max").value)
        self.enable_lookahead_visualization_mirror = bool(
            self.get_parameter("publish_lookahead_visualization_mirror").value
        )
        self.visualization_local_target_topic = str(self.get_parameter("visualization_local_target_topic").value)
        self.visualization_result_vector_topic = str(self.get_parameter("visualization_result_vector_topic").value)
        self.visualization_attractive_vector_topic = str(self.get_parameter("visualization_attractive_vector_topic").value)
        self.emergency_cluster_blocked = False

        # 정적 맵(나무 등)을 1회 로드해 보관. use_gt_obstacles/obstacles_cb/replan과 독립.
        # 전역 A* 코스트맵에 넣어 나무 회피 + clearance 중앙정렬을 가능하게 한다.
        self.static_obstacles: List[Dict[str, float]] = []
        if self.use_static_map:
            sm_path = self.static_map_file or os.path.join(
                get_package_share_directory("rviz_visualization"), "map", "finalmap.map")
            self.static_obstacles = load_static_obstacles_from_map(sm_path)
            self.get_logger().info(
                f"static map obstacles loaded: {len(self.static_obstacles)} from {sm_path}")

        # 지형 거칠기 비용 격자(게이트형, 시나리오2 전용). 빈 경로면 None → 정찰 동작 불변.
        # {(ix, iy): roughness} — A* 1m 격자 인덱스 기준. scenario2_terrain.json에서 로드.
        self.terrain_grid: Optional[Dict[Tuple[int, int], float]] = self._load_terrain_grid(self.terrain_cost_file)

        self.current_speed: float = 0.0
        self.current_sim_time: float = 0.0
        self.current_pos: Optional[Tuple[float, float]] = None
        self.goal_pos: Optional[Tuple[float, float]] = None
        if bool(self.get_parameter("default_goal_enabled").value):
            self.goal_pos = (float(self.get_parameter("default_goal_x").value), float(self.get_parameter("default_goal_y").value))
        self.gt_obstacles: List[Dict[str, float]] = []
        self.lidar_obstacles = LidarObstacleMemory()
        self.latest_cluster_bboxes: List[Dict[str, float]] = []
        self.latest_cluster_count = 0
        self.lidar_cluster_memory: List[Dict[str, float]] = []
        self.lidar_cluster_memory_last_prune_wall = -1e9
        self.discovered_bboxes: List[Dict[str, float]] = []
        self.discovered_count = 0
        self.discovered_confirmed_count = 0
        self.path_block_source = "none"
        self.route: List[Tuple[float, float]] = []
        self.route_version = 0
        self.route_index = 0
        # route_index는 재계획 중에도 진행 상태로 취급한다. dynamic replan 후 일정 시간 동안
        # 이전보다 뒤쪽 route index를 보지 않게 해 lookahead가 반대방향으로 튀는 것을 막는다.
        self.route_index_floor = 0
        self.route_commit_until_wall = -1e9
        self.path_commit_until_wall = -1e9
        self.avoid_side_lock_sign = 0
        self.avoid_side_lock_until_wall = -1e9
        self.last_candidate_reject_reason = "none"
        self.last_route_quality: Dict[str, float] = {}
        self.last_candidate_quality: Dict[str, float] = {}
        # Index into configured route waypoints, not into the generated A* polyline.
        # This prevents dynamic/emergency replan from targeting a checkpoint already passed.
        self.route_checkpoint_index = 0
        self.route_checkpoint_total = 0
        self.route_remaining_waypoints: List[Tuple[float, float]] = []
        # planner 속도 제한엔 monotonic wall time을 쓴다. /clock이 비활성이면 ROS time이 0에 머물 수 있다.
        self.last_plan_wall = -1e9
        self.last_plan_attempt_wall = -1e9
        self.last_dynamic_replan_wall = -1e9
        self.last_dynamic_replan_pos: Optional[Tuple[float, float]] = None
        self.dynamic_replan_count = 0
        self.dynamic_replan_guard_reason = "none"
        self.last_path_publish_wall = -1e9
        self.path_block_hit_count = 0
        self.emergency_path_block_hit_count = 0
        self.last_replan_reason = "not_planned"
        self.last_feasibility_status: Dict[str, Any] = {"enabled": False}
        self.plan_request_pending = True
        self.plan_request_reason = "initial"

        self.pub_path = self.create_publisher(NavPath, TOPIC_GLOBAL_PATH, 10)
        self.pub_lookahead = self.create_publisher(PoseStamped, TOPIC_LOOKAHEAD_POSE, 10)
        # potential_field_node를 끈 경우에도 기존 RViz 설정을 그대로 살리기 위한 mirror publisher.
        # controller는 enable_local_target=False이면 이 토픽을 무시하므로 제어에는 영향이 없다.
        self.pub_visual_local_target = self.create_publisher(PoseStamped, self.visualization_local_target_topic, 10)
        self.pub_visual_result_vector = self.create_publisher(Vector3Stamped, self.visualization_result_vector_topic, 10)
        self.pub_visual_attractive_vector = self.create_publisher(Vector3Stamped, self.visualization_attractive_vector_topic, 10)
        self.pub_points = self.create_publisher(String, TOPIC_PATH_POINTS, 10)
        self.pub_status = self.create_publisher(String, TOPIC_PLANNER_STATUS, 10)
        self.pub_feasibility = self.create_publisher(String, self.path_feasibility_topic, 10)
        self.pub_lidar_bboxes = self.create_publisher(String, TOPIC_LIDAR_BBOXES, 10)
        # 정찰/자율 시나리오에서 controller·local_path가 "도착"을 판정하려면 goal이 필요하다.
        # planner가 보유한 goal_pos(기본 목적지 또는 sim이 /set_destination으로 준 값)를
        # /tank/goal/pose로 주기 발행해 자율 스택의 단일 goal 소스로 삼는다.
        self.pub_goal = self.create_publisher(PoseStamped, TOPIC_GOAL_POSE, 10)

        self.create_subscription(PoseStamped, TOPIC_PLAYER_POSE, self.player_pose_cb, 10)
        self.create_subscription(String, TOPIC_PLAYER_STATE, self.player_state_cb, 10)
        self.create_subscription(PoseStamped, TOPIC_GOAL_POSE, self.goal_pose_cb, 10)
        self.create_subscription(String, TOPIC_MAP_OBSTACLES, self.obstacles_cb, 10)
        self.create_subscription(PointCloud2, TOPIC_LIDAR_DETECTED_MAP, self.lidar_cb, 10)
        self.create_subscription(String, TOPIC_LIDAR_CLUSTERS, self.lidar_clusters_cb, 10)
        if self.use_discovered_objects_for_astar:
            self.create_subscription(String, self.discovered_objects_topic, self.discovered_cb, 10)
        self.create_timer(1.0 / max(1.0, PLANNER_HZ), self.timer_cb)
        # goal을 2Hz로 주기 발행한다(구독자가 volatile QoS라 latch가 통하지 않으므로 주기 발행).
        self.create_timer(0.5, self.publish_goal)
        self._planning_lock = threading.Lock()
        self._is_planning = False

        self.get_logger().info(
            "Team Dynamic A* planner initialized: "
            f"use_gt_obstacles={self.use_gt_obstacles}, dynamic_replan={self.enable_dynamic_replan}, "
            f"goal={self.goal_pos}, resolution={self.resolution}, inflate={self.inflate}, "
            f"route_waypoints={self.use_route_waypoints}:{self.route_map_name}/{self.route_id}, "
            f"cluster_bboxes={self.use_lidar_cluster_bboxes}, cluster_memory={self.enable_lidar_cluster_memory}, "
            f"static_inflate={self.static_obstacle_inflate}, "
            f"discovered_astar={self.use_discovered_objects_for_astar}, "
            f"route_lock={self.route_commit_lock_sec}s, route_index_never_decrease={self.route_index_never_decrease}, "
            f"lidar_memory_block_trigger={self.use_lidar_memory_for_path_block}, "
            f"speed_inflation={self.enable_speed_based_inflation}, speed_emergency={self.enable_speed_based_emergency_replan}, "
            f"semantic_risk={self.enable_semantic_risk_cost}, theta_astar={self.enable_theta_aware_astar}"
        )

    def wall_time(self) -> float:
        return time.monotonic()

    def player_state_cb(self, msg: String) -> None:
        """현재 속도를 planner에도 공유해 speed-based inflation/replan에 사용한다."""
        try:
            data = json.loads(msg.data)
        except Exception:
            return
        try:
            self.current_speed = float(data.get("speed") or 0.0)
        except Exception:
            self.current_speed = 0.0
        try:
            self.current_sim_time = float(data.get("sim_time") or self.current_sim_time)
        except Exception:
            pass

    def dynamic_inflate(self) -> float:
        if not self.enable_speed_based_inflation:
            return float(self.inflate)
        return float(self.vehicle_model.dynamic_inflation(self.current_speed, base=self.inflate))

    def dynamic_discovered_inflate(self) -> float:
        if not self.enable_speed_based_inflation:
            return float(self.discovered_obstacle_inflate)
        return float(self.vehicle_model.dynamic_inflation(
            self.current_speed, base=self.discovered_obstacle_inflate))

    def dynamic_cluster_memory_inflate(self) -> float:
        if not self.enable_speed_based_inflation:
            return float(self.lidar_cluster_memory_inflate)
        return float(self.vehicle_model.dynamic_inflation(
            self.current_speed, base=self.lidar_cluster_memory_inflate))

    def dynamic_emergency_front_distance(self) -> float:
        if not self.enable_speed_based_emergency_replan:
            return float(self.emergency_replan_front_distance)
        return float(self.vehicle_model.emergency_front_distance(
            self.current_speed, base=self.emergency_replan_front_distance))

    def dynamic_emergency_margin(self) -> float:
        if not self.enable_speed_based_emergency_replan:
            return float(self.emergency_replan_margin)
        extra = min(4.0, max(0.0, self.vehicle_model.speed_margin(self.current_speed) * 0.25))
        raw = float(self.emergency_replan_margin + extra)
        if self.emergency_replan_margin_max > 0.0:
            return min(raw, self.emergency_replan_margin_max)
        return raw

    def semantic_risk_sources_for_plan(self, discovered_bboxes: Sequence[Dict[str, float]]) -> List[Dict[str, float]]:
        if not self.enable_semantic_risk_cost:
            return []
        return [dict(b) for b in discovered_bboxes if str(b.get("class_name", "")).strip().lower() not in self.ignored_discovered_classes_for_astar]

    def theta_heading_weight_for_plan(self) -> float:
        if not self.enable_theta_aware_astar:
            return 0.0
        return max(0.0, float(self.theta_heading_change_weight))

    def curvature_smoothing_kwargs(self) -> Dict[str, Any]:
        return {
            "enable_curvature_smoothing": bool(self.enable_curvature_path_smoothing),
            "curvature_min_turn_radius_m": float(self.curvature_smoothing_min_turn_radius_m),
            "curvature_max_corner_angle_deg": float(self.curvature_smoothing_max_corner_angle_deg),
            "curvature_point_spacing_m": float(self.curvature_smoothing_point_spacing_m),
            "curvature_collision_check_margin_m": float(self.curvature_smoothing_collision_check_margin_m),
        }

    def route_quality_metrics(self, route: Sequence[Tuple[float, float]]) -> Dict[str, float]:
        """후보 경로 채택 여부를 판단하기 위한 간단한 품질 지표."""
        if not route or len(route) < 2:
            return {"length_m": 0.0, "heading_change_deg": 0.0, "sharp_corner_count": 0.0, "points": float(len(route or []))}
        length = 0.0
        heading_change = 0.0
        sharp = 0
        for i in range(len(route) - 1):
            length += get_distance(route[i], route[i + 1])
        for i in range(1, len(route) - 1):
            ax, ay = route[i][0] - route[i - 1][0], route[i][1] - route[i - 1][1]
            bx, by = route[i + 1][0] - route[i][0], route[i + 1][1] - route[i][1]
            la = math.hypot(ax, ay)
            lb = math.hypot(bx, by)
            if la <= 1e-6 or lb <= 1e-6:
                continue
            dot = max(-1.0, min(1.0, (ax * bx + ay * by) / (la * lb)))
            ang = math.degrees(math.acos(dot))
            heading_change += ang
            if ang >= 55.0:
                sharp += 1
        return {
            "length_m": float(length),
            "heading_change_deg": float(heading_change),
            "sharp_corner_count": float(sharp),
            "points": float(len(route)),
        }

    def remaining_current_route(self) -> List[Tuple[float, float]]:
        if not self.route:
            return []
        start_idx = min(max(0, int(self.route_index)), max(0, len(self.route) - 1))
        return list(self.route[start_idx:])

    def route_initial_side_sign(self, route: Sequence[Tuple[float, float]]) -> int:
        """현재 위치→goal 기준으로 후보 경로 첫 구간이 좌/우 어느 쪽으로 빠지는지 부호화한다."""
        if self.current_pos is None or self.goal_pos is None or not route:
            return 0
        ref = (self.goal_pos[0] - self.current_pos[0], self.goal_pos[1] - self.current_pos[1])
        ref_norm = math.hypot(ref[0], ref[1])
        if ref_norm <= 1e-6:
            return 0
        # 8~15m 앞쪽 점을 사용한다. 너무 가까운 점은 noise에 민감하다.
        probe = route[min(len(route) - 1, max(1, min(12, len(route) - 1)))]
        vec = (probe[0] - self.current_pos[0], probe[1] - self.current_pos[1])
        vec_norm = math.hypot(vec[0], vec[1])
        if vec_norm <= 1e-6:
            return 0
        cross = ref[0] * vec[1] - ref[1] * vec[0]
        if abs(cross) < 1.0:
            return 0
        return 1 if cross > 0.0 else -1

    def should_accept_candidate_route(self, route: Sequence[Tuple[float, float]], reason: str) -> tuple[bool, str, Dict[str, float]]:
        """Dense obstacle에서 replan 결과가 좌우로 뒤집히는 것을 줄이기 위한 후보 경로 필터."""
        new_q = self.route_quality_metrics(route)
        if not self.enable_replan_acceptance_filter or not self._is_dynamic_replan_reason(reason) or not self.route:
            return True, "accepted_filter_disabled_or_initial", new_q
        now = self.wall_time()
        is_emergency = str(reason).startswith("emergency_")
        old_remaining = self.remaining_current_route()
        old_q = self.route_quality_metrics(old_remaining)
        self.last_candidate_quality = dict(new_q)
        # 전방 corridor를 실제로 emergency block한 경우는 commitment 예외로 허용한다.
        if not is_emergency and now < self.path_commit_until_wall:
            return False, f"path_commitment_hold {self.path_commit_until_wall - now:.2f}s", new_q
        # Avoid-side lock: 방금 선택한 우회 방향과 반대쪽 후보는 일정 시간 거절한다.
        if self.enable_avoid_side_lock and not is_emergency and now < self.avoid_side_lock_until_wall:
            sign = self.route_initial_side_sign(route)
            if sign != 0 and self.avoid_side_lock_sign != 0 and sign != self.avoid_side_lock_sign:
                return False, f"avoid_side_lock sign={sign} locked={self.avoid_side_lock_sign}", new_q
        if old_q.get("length_m", 0.0) > 1.0:
            length_ratio = new_q.get("length_m", 0.0) / max(1.0, old_q["length_m"])
            sharp_increase = new_q.get("sharp_corner_count", 0.0) - old_q.get("sharp_corner_count", 0.0)
            heading_increase = new_q.get("heading_change_deg", 0.0) - old_q.get("heading_change_deg", 0.0)
            # 새 경로가 너무 길고, 더 부드럽지도 않으면 채택하지 않는다.
            if length_ratio > self.replan_accept_max_length_ratio and sharp_increase >= 0:
                return False, f"candidate_too_long ratio={length_ratio:.2f}", new_q
            if sharp_increase > self.replan_accept_max_sharp_corner_increase:
                return False, f"candidate_more_sharp_corners +{sharp_increase:.0f}", new_q
            if heading_increase > self.replan_accept_max_heading_change_increase_deg:
                return False, f"candidate_heading_change +{heading_increase:.1f}deg", new_q
        return True, "accepted", new_q

    def update_path_commitment_after_accept(self, route: Sequence[Tuple[float, float]], reason: str) -> None:
        if self._is_dynamic_replan_reason(reason):
            now = self.wall_time()
            self.path_commit_until_wall = now + max(0.0, self.path_commitment_sec)
            sign = self.route_initial_side_sign(route)
            if sign != 0:
                self.avoid_side_lock_sign = sign
                self.avoid_side_lock_until_wall = now + max(0.0, self.avoid_side_lock_sec)

    def _apply_runtime_inflate(self, bboxes: Sequence[Dict[str, float]], inflate: float) -> List[Dict[str, float]]:
        out: List[Dict[str, float]] = []
        for b in bboxes or []:
            copied = dict(b)
            copied["_inflate_override"] = float(inflate)
            out.append(copied)
        return out

    def publish_feasibility_status(self, route: Sequence[Tuple[float, float]], reason: str = "") -> Dict[str, Any]:
        status: Dict[str, Any] = {"enabled": self.enable_path_feasibility_check}
        if self.enable_path_feasibility_check:
            status.update(self.vehicle_model.path_curvature_summary(route, self.current_speed))
        status.update({
            "reason": reason,
            "current_speed_mps": float(self.current_speed),
            "stopping_distance_m": float(self.vehicle_model.stopping_distance(self.current_speed)),
            "dynamic_inflate_m": float(self.dynamic_inflate()),
            "dynamic_emergency_front_distance_m": float(self.dynamic_emergency_front_distance()),
            "dynamic_emergency_margin_m": float(self.dynamic_emergency_margin()),
        })
        try:
            msg = String()
            msg.data = json.dumps(status, ensure_ascii=False)
            self.pub_feasibility.publish(msg)
        except Exception:
            pass
        self.last_feasibility_status = status
        return status

    def player_pose_cb(self, msg: PoseStamped) -> None:
        new_pos = (float(msg.pose.position.x), float(msg.pose.position.y))
        if self.current_pos is not None:
            if get_distance(self.current_pos, new_pos) > 10.0:
                self.get_logger().info("Teleport/Restart detected. Triggering replan from new start.")
                self.route = []
                self.plan_request_pending = True
                self.plan_request_reason = "teleport_detected"
                self.dynamic_replan_count = 0
                self.last_dynamic_replan_pos = None
                self.dynamic_replan_guard_reason = "teleport_reset"
                self.route_index = 0
                self.route_index_floor = 0
                self.route_commit_until_wall = -1e9
                self.route_checkpoint_index = 0
                self.route_checkpoint_total = 0
                self.route_remaining_waypoints = []
        self.current_pos = new_pos

    def goal_pose_cb(self, msg: PoseStamped) -> None:
        new_goal = (float(msg.pose.position.x), float(msg.pose.position.y))
        # 목적지가 '의미있게' 바뀔 때만 전역경로 재생성. 0.5m는 과민했다 — /tank/goal/pose에
        # ros_bridge(시뮬 POST)와 planner(2Hz)가 이중 발행해, 좌표변환 부동소수 차로도 매번
        # route를 비워 lookahead가 프레임마다 흔들렸다. goal_tolerance(10m)와 정합되게 10m로 상향.
        if self.goal_pos is None or get_distance(new_goal, self.goal_pos) > 10.0:
            self.goal_pos = new_goal
            self.route = []
            self.plan_request_pending = True
            self.plan_request_reason = "goal_updated"
            self.last_replan_reason = "goal_updated"
            self.path_block_hit_count = 0
            self.dynamic_replan_count = 0
            self.last_dynamic_replan_pos = None
            self.dynamic_replan_guard_reason = "goal_reset"
            self.route_index = 0
            self.route_index_floor = 0
            self.route_commit_until_wall = -1e9
            self.route_checkpoint_index = 0
            self.route_checkpoint_total = 0
            self.route_remaining_waypoints = []

    def obstacles_cb(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
            obs = parse_obstacles_payload(payload)
        except Exception as exc:
            self.get_logger().warn(f"failed to parse /tank/map/obstacles: {exc}")
            return

        # 리스트가 비어 있어도 갱신한다. 안 그러면 stale GT 장애물이 영영 남을 수 있다.
        old_obstacles = self.gt_obstacles
        self.gt_obstacles = obs
        if self.use_gt_obstacles and obs != old_obstacles:
            self.route = []
            self.plan_request_pending = True
            self.plan_request_reason = "gt_obstacles_updated"
            self.last_replan_reason = "gt_obstacles_updated"
            self.path_block_hit_count = 0
        self.get_logger().info(f"GT obstacles received: {len(obs)} bboxes")

    def lidar_cb(self, msg: PointCloud2) -> None:
        try:
            points = pointcloud2_to_xyz_array(msg)
            # 계획 쪽 메모리/클러스터링 로직은 여전히 LidarObstacleMemory가 담당한다.
            # LiDAR JSON String을 파싱하는 대신 최소한의 in-memory payload를 넘겨준다.
            point_items = [
                {
                    "isDetected": True,
                    "position_map": {"x": float(x), "y": float(y), "z": float(z)},
                }
                for x, y, z in points
            ]
            payload = {
                "route": "/info",
                "source": "pointcloud2/detected_points_map",
                "timestamp_ros_sec": msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
                "frame_id": msg.header.frame_id or MAP_FRAME,
                "count": len(point_items),
                "points": point_items,
            }
            self.lidar_obstacles.update_from_payload(
                payload,
                history_enabled=self.enable_dynamic_replan,
                history_resolution=self.lidar_history_resolution,
                max_history_points=self.max_lidar_history_points,
            )
        except Exception as exc:
            self.get_logger().warn(f"failed to update detected lidar obstacle memory from PC2: {exc}")

    def lidar_clusters_cb(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
            clusters = payload.get("clusters", []) if isinstance(payload, dict) else []
            bboxes: List[Dict[str, float]] = []
            margin = max(0.0, self.lidar_cluster_bbox_margin)
            if isinstance(clusters, list):
                for c in clusters:
                    if not isinstance(c, dict):
                        continue
                    bbox = c.get("bbox") if isinstance(c.get("bbox"), dict) else None
                    if bbox is None:
                        continue
                    try:
                        # 클러스터 bbox는 ROS map x/y/z 좌표다. A* bbox는 x/z 키를 쓰며, 여기서 z는 map 평면의 y를 뜻한다.
                        bboxes.append({
                            "x_min": float(bbox["x_min"]) - margin,
                            "x_max": float(bbox["x_max"]) + margin,
                            "z_min": float(bbox["y_min"]) - margin,
                            "z_max": float(bbox["y_max"]) + margin,
                            "source": "dbscan_cluster",
                            "cluster_id": int(c.get("id", -1)),
                            "count": int(c.get("count", 0)),
                        })
                    except Exception:
                        continue
            self.latest_cluster_bboxes = bboxes
            self.latest_cluster_count = len(clusters) if isinstance(clusters, list) else 0
            if self.enable_lidar_cluster_memory and bboxes:
                self.remember_lidar_cluster_bboxes(bboxes)
            else:
                self.prune_lidar_cluster_memory(self.wall_time())
            if self.use_lidar_cluster_bboxes:
                # 동적 재탐색이 꺼져 있어도 클러스터에서 유도한 A* bbox를 발행한다.
                # RViz/디버그 노드가 DBSCAN 출력이 planner bbox 입력에 연결됐는지 확인할 수 있도록.
                self.publish_lidar_bboxes(bboxes)
        except Exception as exc:
            self.get_logger().warn(f"failed to parse lidar clusters: {exc}")

    def discovered_cb(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
            self.discovered_count = int(payload.get("count", 0)) if isinstance(payload, dict) else 0
            self.discovered_confirmed_count = int(payload.get("confirmed_count", 0)) if isinstance(payload, dict) else 0
            self.discovered_bboxes = parse_discovered_objects_payload(
                payload,
                confirmed_only=self.discovered_confirmed_only,
                min_observations=self.discovered_min_observations,
                ignored_classes=self.ignored_discovered_classes_for_astar,
                default_radius=self.discovered_default_radius,
            )
            for bbox in self.discovered_bboxes:
                bbox["_inflate_override"] = self.dynamic_discovered_inflate()
        except Exception as exc:
            self.get_logger().warn(f"failed to parse discovered objects for A*: {exc}")

    def build_discovered_bboxes(self) -> List[Dict[str, float]]:
        if not self.use_discovered_objects_for_astar:
            return []
        if self.discovered_obstacle_inflate <= 0.0:
            return list(self.discovered_bboxes)
        # discovered bbox 자체는 class별 물리 반경이고, A*에는 별도 inflate를 적용한다.
        return list(self.discovered_bboxes)

    def prune_lidar_cluster_memory(self, now: float) -> None:
        if not self.enable_lidar_cluster_memory:
            self.lidar_cluster_memory = []
            return
        ttl = max(0.0, self.lidar_cluster_memory_ttl_sec)
        if ttl <= 0.0:
            self.lidar_cluster_memory = []
            return
        self.lidar_cluster_memory = [
            m for m in self.lidar_cluster_memory
            if (now - float(m.get("last_seen_wall", now))) <= ttl
        ]
        max_count = max(0, self.lidar_cluster_memory_max_count)
        if max_count > 0 and len(self.lidar_cluster_memory) > max_count:
            self.lidar_cluster_memory.sort(key=lambda m: float(m.get("last_seen_wall", 0.0)), reverse=True)
            self.lidar_cluster_memory = self.lidar_cluster_memory[:max_count]
        self.lidar_cluster_memory_last_prune_wall = now

    def remember_lidar_cluster_bboxes(self, bboxes: Sequence[Dict[str, float]]) -> None:
        if not self.enable_lidar_cluster_memory:
            return
        now = self.wall_time()
        self.prune_lidar_cluster_memory(now)
        merge_dist = max(0.0, self.lidar_cluster_memory_merge_distance)
        for bbox in bboxes:
            try:
                # 유효 bbox만 기억한다. 너무 작은/잘못된 bbox는 버린다.
                if float(bbox.get("x_max", 0.0)) < float(bbox.get("x_min", 0.0)):
                    continue
                if float(bbox.get("z_max", 0.0)) < float(bbox.get("z_min", 0.0)):
                    continue
            except Exception:
                continue
            best_i = -1
            best_d = 1e9
            for i, mem in enumerate(self.lidar_cluster_memory):
                d = bbox_center_distance(bbox, mem)
                if d < best_d:
                    best_d = d
                    best_i = i
            if best_i >= 0 and best_d <= merge_dist:
                old = self.lidar_cluster_memory[best_i]
                hits = int(old.get("hits", 1)) + 1
                updated = bbox_copy_as_memory(bbox, now, hits=hits)
                updated["first_seen_wall"] = float(old.get("first_seen_wall", now))
                updated["memory_id"] = old.get("memory_id", f"mem_{int(now * 1000)}_{best_i}")
                updated["_inflate_override"] = self.dynamic_cluster_memory_inflate()
                self.lidar_cluster_memory[best_i] = updated
            else:
                mem = bbox_copy_as_memory(bbox, now, hits=1)
                mem["memory_id"] = f"mem_{int(now * 1000)}_{len(self.lidar_cluster_memory)}"
                mem["_inflate_override"] = self.dynamic_cluster_memory_inflate()
                self.lidar_cluster_memory.append(mem)
        self.prune_lidar_cluster_memory(now)

    def build_lidar_cluster_memory_bboxes(self, current_bboxes: Optional[Sequence[Dict[str, float]]] = None) -> List[Dict[str, float]]:
        if not self.enable_lidar_cluster_memory:
            return []
        now = self.wall_time()
        self.prune_lidar_cluster_memory(now)
        current_bboxes = current_bboxes or []
        out: List[Dict[str, float]] = []
        # 현재 프레임 cluster와 거의 같은 memory는 중복 costmap을 만들지 않도록 제외한다.
        # cluster가 시야에서 사라진 경우에만 memory layer가 A*에 남는다.
        dedupe_dist = max(0.0, self.lidar_cluster_memory_merge_distance)
        for mem in self.lidar_cluster_memory:
            if current_bboxes and any(bbox_center_distance(mem, cur) <= dedupe_dist for cur in current_bboxes):
                continue
            copied = dict(mem)
            copied["source"] = "lidar_cluster_memory"
            copied["age_sec"] = max(0.0, now - float(copied.get("last_seen_wall", now)))
            copied["_inflate_override"] = self.dynamic_cluster_memory_inflate()
            out.append(copied)
        return out

    def build_lidar_bboxes(self) -> List[Dict[str, float]]:
        current: List[Dict[str, float]] = []
        if self.use_lidar_cluster_bboxes and self.latest_cluster_bboxes:
            current = self._apply_runtime_inflate(self.latest_cluster_bboxes, self.dynamic_inflate())
        else:
            current = self._apply_runtime_inflate(
                self.lidar_obstacles.build_bboxes(self.lidar_cluster_eps, self.lidar_cluster_min_samples),
                self.dynamic_inflate(),
            )
        memory = self.build_lidar_cluster_memory_bboxes(current)
        return list(current) + memory

    def maybe_plan(self, reason: str) -> None:
        self._request_plan_async(reason)

    def _request_plan_async(self, reason: str) -> None:
        if self._is_planning:
            return
        if self.current_pos is None or self.goal_pos is None:
            return

        self._is_planning = True
        self.last_plan_attempt_wall = self.wall_time()

        start_pos = self.current_pos
        goal_pos = self.goal_pos

        obstacles: List[Dict[str, float]] = []
        if self.use_gt_obstacles:
            obstacles.extend(deepcopy(self.gt_obstacles))

        # Persistent discovered objects는 한 번 확인되면 현재 LiDAR 시야 밖이어도 A* hard obstacle로 유지한다.
        discovered_bboxes: List[Dict[str, float]] = deepcopy(self.build_discovered_bboxes())
        if discovered_bboxes:
            # add_obstacles는 inflate를 일괄 적용하므로, discovered bbox는 class radius만 들고 있고
            # planning worker에서 discovered_obstacle_inflate를 반영해 확장한다.
            pass

        lidar_bboxes: List[Dict[str, float]] = []
        if self.enable_dynamic_replan and (self.lidar_obstacles.history_count > 0 or self.latest_cluster_bboxes):
            lidar_bboxes = deepcopy(self.build_lidar_bboxes())

        # dynamic obstacles는 source별 inflation이 달라서 worker에서 합성한다.
        obstacles.extend(lidar_bboxes)
        obstacles.extend(discovered_bboxes)

        threading.Thread(
            target=self._plan_worker,
            args=(reason, start_pos, goal_pos, obstacles, lidar_bboxes, discovered_bboxes),
            daemon=True
        ).start()

    def _remaining_route_waypoints(self, start_pos: Tuple[float, float], waypoints: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Return only checkpoints that are still ahead of the tank.

        A* polyline route_index is not stable across replans because a new path is generated
        from the current pose. Checkpoint progress must therefore be tracked against the
        configured route waypoints themselves. A waypoint is considered passed if either:
        - the tank is within route_checkpoint_reached_radius, or
        - in this northbound map, the tank's z/y is ahead of the waypoint by
          route_checkpoint_passed_z_margin.

        This prevents emergency/dynamic replans from rebuilding a path back to a checkpoint
        that the tank already passed while avoiding an obstacle laterally.
        """
        pts = [tuple(map(float, wp)) for wp in waypoints]
        self.route_checkpoint_total = len(pts)
        if not pts:
            self.route_checkpoint_index = 0
            self.route_remaining_waypoints = []
            return []

        idx = int(self.route_checkpoint_index) if self.route_checkpoint_never_decrease else 0
        idx = max(0, min(idx, len(pts)))
        advanced = False
        while idx < len(pts):
            wp = pts[idx]
            reached = get_distance(start_pos, wp) <= max(0.0, self.route_checkpoint_reached_radius)
            passed_by_z = start_pos[1] >= wp[1] + max(0.0, self.route_checkpoint_passed_z_margin)
            if reached or passed_by_z:
                idx += 1
                advanced = True
                continue
            break

        if self.route_checkpoint_never_decrease:
            self.route_checkpoint_index = max(int(self.route_checkpoint_index), idx)
        else:
            self.route_checkpoint_index = idx
        remaining = pts[self.route_checkpoint_index:]
        self.route_remaining_waypoints = list(remaining)
        if advanced:
            self.get_logger().info(
                f"route checkpoint advanced: next={self.route_checkpoint_index}/{len(pts)}, "
                f"remaining={len(remaining)}"
            )
        return remaining

    def _is_dynamic_replan_reason(self, reason: str) -> bool:
        r = str(reason or "")
        return "path_blocked" in r or r.startswith("emergency_") or r.startswith("lidar_")

    def _load_terrain_grid(self, path: str) -> Optional[Dict[Tuple[int, int], float]]:
        """scenario2_terrain.json(셀별 roughness 격자)을 A* 1m 격자 인덱스 dict로 로드한다.

        반환 {(ix, iy): roughness(dz/m)}. 빈 경로/없음/파싱실패면 None(게이트 OFF) → 정찰 동작 불변.
        cells의 cell_size는 1.0(생성기가 A* 격자에 맞춰 생성)이라 ix/iy를 그대로 격자 인덱스로 쓴다.

        부수효과: self.terrain_z_grid({(ix,iy): z_median}, 평지 포함 모든 셀)·self.terrain_cell_size를 채운다 —
        publish_path가 전역경로를 지형 표면 위로 띄울 때 사용. terrain_cost_file이 비면(정찰) 빈 dict라 z=0 유지.
        """
        # 경로 지형-위 표시용 고도 격자(매 호출 초기화). roughness 게이트와 독립.
        self.terrain_z_grid: Dict[Tuple[int, int], float] = {}
        self.terrain_cell_size: float = 1.0
        if not path:
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            self.get_logger().warn(f"terrain cost file load failed (게이트 OFF): {path} ({exc})")
            return None
        cells = data.get("cells", []) if isinstance(data, dict) else []
        cell_size = float(data.get("cell_size", 1.0)) if isinstance(data, dict) else 1.0
        self.terrain_cell_size = cell_size if cell_size > 0 else 1.0
        grid: Dict[Tuple[int, int], float] = {}
        for c in cells:
            try:
                ix, iy = int(c["ix"]), int(c["iy"])
            except Exception:
                continue
            # z_median: 모든 셀(평지 roughness 0 포함) — publish_path가 경로를 지형 표면 위로 lift할 때 사용.
            zmed = c.get("z_median")
            if zmed is not None:
                try:
                    self.terrain_z_grid[(ix, iy)] = float(zmed)
                except (TypeError, ValueError):
                    pass
            # roughness: A* 지형비용용(>0만 보관; 기존과 동일).
            try:
                rough = float(c.get("roughness", 0.0))
            except (TypeError, ValueError):
                continue
            if rough > 0.0:
                grid[(ix, iy)] = rough
        if abs(cell_size - 1.0) > 1e-6:
            self.get_logger().warn(
                f"terrain cell_size={cell_size}≠1.0 — A* 격자와 정렬이 안 맞을 수 있음(생성기 cell_size=1.0 권장)")
        self.get_logger().info(
            f"terrain grid loaded: {len(grid)} rough cells / {len(self.terrain_z_grid)} z cells "
            f"(weight={self.terrain_weight}) from {path}")
        return grid or None

    def _plan_worker(
        self,
        reason: str,
        start_pos: Tuple[float, float],
        goal_pos: Tuple[float, float],
        obstacles: List[Dict[str, float]],
        lidar_bboxes: List[Dict[str, float]],
        discovered_bboxes: List[Dict[str, float]]
    ) -> None:
        try:
            route: List[Tuple[float, float]] = []
            route_mode = "direct_astar"
            if self.use_route_waypoints:
                try:
                    route_config = self.route_config_file or None
                    waypoints = get_route_waypoints(self.route_map_name, self.route_id, route_config)
                    remaining_waypoints = self._remaining_route_waypoints(start_pos, waypoints)
                    # 이미 지난 checkpoint는 through list에서 제외한다. goal은 마지막 목적지로만 유지한다.
                    through = list(remaining_waypoints) + [goal_pos]
                    route = team_plan_path_through_waypoints(
                        start_pos,
                        through,
                        obstacles,
                        static_obstacles=self.static_obstacles,
                        inflate=self.dynamic_inflate(),
                        static_inflate=self.static_obstacle_inflate,
                        clearance_weight=self.route_clearance_weight,
                        side=self.route_side,
                        terrain_grid=self.terrain_grid,
                        terrain_weight=self.terrain_weight,
                        semantic_risk_sources=self.semantic_risk_sources_for_plan(discovered_bboxes),
                        semantic_risk_scores=self.semantic_risk_scores,
                        semantic_risk_radii=self.semantic_risk_radii,
                        semantic_risk_weight=self.semantic_risk_weight if self.enable_semantic_risk_cost else 0.0,
                        semantic_risk_radius_scale=self.semantic_risk_radius_scale,
                        heading_change_weight=self.theta_heading_weight_for_plan(),
                        **self.curvature_smoothing_kwargs(),
                    )
                    route_mode = (
                        f"route_waypoints:{self.route_map_name}/{self.route_id}/{self.route_side}"
                        f":next_checkpoint={self.route_checkpoint_index}/{self.route_checkpoint_total}"
                    )
                except Exception as exc:
                    self.get_logger().warn(f"route waypoint planning failed, fallback direct A*: {exc}")
                    route = []
            if not route:
                # direct fallback도 team_path_planning의 cost_map/semantic/theta-aware A*를 사용한다.
                route = team_plan_global_path(
                    start_pos,
                    goal_pos,
                    obstacles,
                    inflate=self.dynamic_inflate(),
                    static_obstacles=self.static_obstacles,
                    static_inflate=self.static_obstacle_inflate,
                    clearance_weight=self.route_clearance_weight,
                    waypoints_ref=None,
                    side=self.route_side,
                    terrain_grid=self.terrain_grid,
                    terrain_weight=self.terrain_weight,
                    semantic_risk_sources=self.semantic_risk_sources_for_plan(discovered_bboxes),
                    semantic_risk_scores=self.semantic_risk_scores,
                    semantic_risk_radii=self.semantic_risk_radii,
                    semantic_risk_weight=self.semantic_risk_weight if self.enable_semantic_risk_cost else 0.0,
                    semantic_risk_radius_scale=self.semantic_risk_radius_scale,
                    heading_change_weight=self.theta_heading_weight_for_plan(),
                    **self.curvature_smoothing_kwargs(),
                )
                route_mode = "direct_astar"

            if route:
                accepted, reject_reason, candidate_quality = self.should_accept_candidate_route(route, reason)
                if not accepted:
                    with self._planning_lock:
                        self.last_candidate_reject_reason = reject_reason
                        self.last_candidate_quality = candidate_quality
                        self.last_replan_reason = f"rejected_{reason}"
                        self.dynamic_replan_guard_reason = reject_reason
                        if self._is_dynamic_replan_reason(reason):
                            now_wall = self.wall_time()
                            self.last_dynamic_replan_wall = now_wall
                            self.last_dynamic_replan_pos = start_pos
                            self.path_block_hit_count = 0
                            self.emergency_path_block_hit_count = 0
                    self.get_logger().info(
                        f"A* candidate rejected: reason={reason}, guard={reject_reason}, "
                        f"candidate={candidate_quality}"
                    )
                    return
                with self._planning_lock:
                    prev_route_index = int(self.route_index)
                    self.route = route
                    self.last_candidate_reject_reason = "accepted"
                    self.last_route_quality = candidate_quality
                    self.update_path_commitment_after_accept(route, reason)
                    now_wall = self.wall_time()
                    # 일반적으로 새 경로는 새 좌표계(route point list)를 갖지만, dynamic replan에서는
                    # 현재 체크포인트 진행 상태를 뒤로 되돌리면 lookahead가 반대방향으로 튄다.
                    # 따라서 이전 route_index를 floor로 보존하고 publish_lookahead에서 일정 시간 강제한다.
                    if self._is_dynamic_replan_reason(reason) and self.dynamic_replan_keep_route_index:
                        # Dynamic/emergency replan은 checkpoint 진행도는 유지하되, 새 A* polyline은 현재 위치에서
                        # 시작하므로 path-point index를 과거 route_index로 강제하지 않는다.
                        # 이전 polyline index를 보존하면 새 경로의 앞부분을 건너뛰어 오히려 target이 튈 수 있다.
                        self.route_index = 0
                        self.route_index_floor = 0
                        self.route_commit_until_wall = now_wall + max(0.0, self.route_commit_lock_sec)
                    else:
                        self.route_index = 0
                        self.route_index_floor = 0
                        self.route_commit_until_wall = -1e9
                    self.route_version += 1
                    self.last_plan_wall = now_wall
                    if self._is_dynamic_replan_reason(reason):
                        self.last_dynamic_replan_wall = self.last_plan_wall
                        self.last_dynamic_replan_pos = start_pos
                        self.dynamic_replan_count += 1
                        self.dynamic_replan_guard_reason = "accepted"
                    self.plan_request_pending = False
                    self.path_block_hit_count = 0
                    self.last_path_publish_wall = -1e9
                    self.last_replan_reason = reason
                
                self.publish_lidar_bboxes(lidar_bboxes)
                self.publish_path(force=True)
                feasibility = self.publish_feasibility_status(route, reason)
                self.get_logger().info(
                    f"A* path updated: reason={reason}, mode={route_mode}, points={len(route)}, "
                    f"obstacles={len(obstacles)}, lidar_bboxes={len(lidar_bboxes)}, discovered_bboxes={len(discovered_bboxes)}, "
                    f"dyn_inflate={self.dynamic_inflate():.2f}, stop_d={self.vehicle_model.stopping_distance(self.current_speed):.2f}, "
                    f"semantic_risk={self.enable_semantic_risk_cost}, theta_w={self.theta_heading_weight_for_plan():.2f}, "
                    f"min_radius={feasibility.get('min_path_turn_radius_m')}"
                )
            else:
                with self._planning_lock:
                    self.last_replan_reason = f"plan_failed_{reason}"
                self.get_logger().warn(f"A* failed: reason={reason}, obstacles={len(obstacles)}")
        except Exception as exc:
            self.get_logger().error(f"전역 경로 계획 스레드 에러: {exc}")
        finally:
            self._is_planning = False

    def publish_lidar_bboxes(self, bboxes: List[Dict[str, float]]) -> None:
        msg = String()
        msg.data = json.dumps({"count": len(bboxes), "bboxes": bboxes}, ensure_ascii=False)
        self.pub_lidar_bboxes.publish(msg)

    def _terrain_lift_z(self, x: float, y: float) -> float:
        """경로점 z를 지형 표면 위로 띄운다. terrain_z_grid가 있으면 해당 셀 z_median+0.4m(메쉬와 z-fighting 방지),
        없으면(정찰=terrain_cost_file 빈값/지형 없음) 0.0 → 바닥(기존 거동 보존)."""
        grid = getattr(self, "terrain_z_grid", None)
        if not grid:
            return 0.0
        cs = getattr(self, "terrain_cell_size", 1.0) or 1.0
        z = grid.get((int(math.floor(x / cs)), int(math.floor(y / cs))))
        return (float(z) + 0.4) if z is not None else 0.0

    def publish_path(self, force: bool = False) -> None:
        if not self.route:
            return
        wall_now = self.wall_time()
        if (
            not force
            and self.publish_path_period_sec > 0.0
            and (wall_now - self.last_path_publish_wall) < self.publish_path_period_sec
        ):
            return
        self.last_path_publish_wall = wall_now
        now = self.get_clock().now().to_msg()
        path_msg = NavPath()
        path_msg.header.stamp = now
        path_msg.header.frame_id = MAP_FRAME
        for x, y in self.route:
            ps = PoseStamped()
            ps.header.stamp = now
            ps.header.frame_id = MAP_FRAME
            ps.pose.position.x = float(x)
            ps.pose.position.y = float(y)
            ps.pose.position.z = self._terrain_lift_z(float(x), float(y))  # 지형 표면 위로(시나리오2), 정찰은 0.0
            ps.pose.orientation.w = 1.0
            path_msg.poses.append(ps)
        self.pub_path.publish(path_msg)
        payload = {"route_version": self.route_version, "points": [{"x": x, "y": y} for x, y in self.route]}
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self.pub_points.publish(msg)

    def publish_goal(self) -> None:
        # planner의 현재 goal을 /tank/goal/pose로 주기 발행한다.
        # controller(도착 시 정지·종료)와 local_path(도착 로깅)가 동일 goal을 공유하게 한다.
        if self.goal_pos is None:
            return
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = MAP_FRAME
        msg.pose.position.x = float(self.goal_pos[0])
        msg.pose.position.y = float(self.goal_pos[1])
        msg.pose.position.z = 0.0
        msg.pose.orientation.w = 1.0
        self.pub_goal.publish(msg)

    def publish_lookahead_visualization_mirror(self, target: Tuple[float, float]) -> None:
        """APF 없이도 기존 RViz potential marker 표시를 유지한다.

        rviz_visualizer_node는 기존에 /tank/local_target/pose와
        /tank/potential/result_vector를 받아 노란 target 점과 방향 화살표를 그렸다.
        APF를 launch에서 제외하면 해당 토픽이 끊기므로, planner가 현재 A* lookahead를
        시각화용 local target과 desired heading vector로 mirror 발행한다.

        주의: controller가 enable_local_target=False이면 /tank/local_target/pose는 제어 입력으로
        사용되지 않는다. 이 함수는 RViz 표시 유지용이다.
        """
        if not self.enable_lookahead_visualization_mirror or self.current_pos is None:
            return

        stamp = self.get_clock().now().to_msg()

        target_msg = PoseStamped()
        target_msg.header.stamp = stamp
        target_msg.header.frame_id = MAP_FRAME
        target_msg.pose.position.x = float(target[0])
        target_msg.pose.position.y = float(target[1])
        target_msg.pose.position.z = 0.0
        target_msg.pose.orientation.w = 1.0
        self.pub_visual_local_target.publish(target_msg)

        dx = float(target[0] - self.current_pos[0])
        dy = float(target[1] - self.current_pos[1])
        norm = math.hypot(dx, dy)
        if norm > 1.0e-6:
            vx = dx / norm
            vy = dy / norm
        else:
            vx = 0.0
            vy = 0.0

        vec_msg = Vector3Stamped()
        vec_msg.header.stamp = stamp
        vec_msg.header.frame_id = MAP_FRAME
        vec_msg.vector.x = float(vx)
        vec_msg.vector.y = float(vy)
        vec_msg.vector.z = 0.0
        # 기존 RViz에서는 result vector가 보라색/파란색 계열 화살표로 표시된다.
        self.pub_visual_result_vector.publish(vec_msg)
        # attractive vector도 같은 방향으로 발행해 기존 초록색 목표방향 표시를 유지한다.
        self.pub_visual_attractive_vector.publish(vec_msg)


    def publish_lookahead(self) -> Optional[Tuple[float, float]]:
        if self.current_pos is None or not self.route:
            return None
        at_goal = (self.goal_pos is not None
                   and get_distance(self.current_pos, self.goal_pos) < self.goal_tolerance)
        if at_goal:
            target = self.goal_pos
            idx = len(self.route) - 1
        else:
            target, idx = find_lookahead_along_path(self.current_pos, self.route, self.lookahead_distance)
        wall_now = self.wall_time()
        # route_index는 진행 상태다. replan 직후 lock 시간뿐 아니라 일반 주행 중에도
        # lookahead projection이 가까운 이전 segment를 다시 잡으면 target이 뒤로 튀고
        # controller가 제자리 U-turn을 반복한다. 따라서 명시 reset/replan 전까지는 감소를 막는다.
        if self.route_index_never_decrease and self.route:
            prev_floor = min(max(0, int(self.route_index)), len(self.route) - 1)
            commit_floor = min(max(0, int(self.route_index_floor)), len(self.route) - 1)
            floor = max(prev_floor, commit_floor)
            if idx < floor:
                idx = floor
                target = self.route[floor]
        self.route_index = idx
        if self.route_index_never_decrease:
            self.route_index_floor = max(int(self.route_index_floor), int(self.route_index))
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = MAP_FRAME
        msg.pose.position.x = float(target[0])
        msg.pose.position.y = float(target[1])
        msg.pose.position.z = 0.0
        msg.pose.orientation.w = 1.0
        self.pub_lookahead.publish(msg)
        self.publish_lookahead_visualization_mirror(target)
        return target

    def publish_status(self, lookahead: Optional[Tuple[float, float]]) -> None:
        payload = {
            "ok": bool(self.route),
            "reason": self.last_replan_reason,
            "route_version": self.route_version,
            "points": len(self.route),
            "route_index": self.route_index,
            "start": {"x": self.current_pos[0], "y": self.current_pos[1]} if self.current_pos else None,
            "goal": {"x": self.goal_pos[0], "y": self.goal_pos[1]} if self.goal_pos else None,
            "lookahead": {"x": lookahead[0], "y": lookahead[1]} if lookahead else None,
            "use_gt_obstacles": self.use_gt_obstacles,
            "gt_obstacle_count": len(self.gt_obstacles),
            "current_lidar_points": self.lidar_obstacles.current_count,
            "lidar_history_points": self.lidar_obstacles.history_count,
            "lidar_cluster_count": self.latest_cluster_count,
            "lidar_cluster_bbox_count": len(self.latest_cluster_bboxes),
            "lidar_cluster_memory_count": len(self.lidar_cluster_memory),
            "lidar_cluster_memory_astar_bbox_count": len(self.build_lidar_cluster_memory_bboxes(self.latest_cluster_bboxes)),
            "enable_lidar_cluster_memory": self.enable_lidar_cluster_memory,
            "lidar_cluster_memory_ttl_sec": self.lidar_cluster_memory_ttl_sec,
            "lidar_cluster_memory_merge_distance": self.lidar_cluster_memory_merge_distance,
            "lidar_cluster_memory_inflate": self.lidar_cluster_memory_inflate,
            "lidar_cluster_memory_max_count": self.lidar_cluster_memory_max_count,
            "use_lidar_cluster_memory_for_path_block": self.use_lidar_cluster_memory_for_path_block,
            "use_lidar_cluster_bboxes": self.use_lidar_cluster_bboxes,
            "static_obstacle_count": len(self.static_obstacles),
            "static_obstacle_inflate": self.static_obstacle_inflate,
            "discovered_object_count": self.discovered_count,
            "discovered_confirmed_count": self.discovered_confirmed_count,
            "discovered_astar_bbox_count": len(self.discovered_bboxes),
            "use_discovered_objects_for_astar": self.use_discovered_objects_for_astar,
            "discovered_confirmed_only": self.discovered_confirmed_only,
            "discovered_min_observations": self.discovered_min_observations,
            "path_block_source": self.path_block_source,
            "emergency_cluster_blocked": self.emergency_cluster_blocked,
            "emergency_cluster_replan_enabled": self.emergency_cluster_replan_enabled,
            "emergency_replan_cooldown_sec": self.emergency_replan_cooldown_sec,
            "emergency_replan_front_distance": self.emergency_replan_front_distance,
            "dynamic_emergency_replan_front_distance": self.dynamic_emergency_front_distance(),
            "emergency_replan_min_distance": self.emergency_replan_min_distance,
            "emergency_replan_margin": self.emergency_replan_margin,
            "dynamic_emergency_replan_margin": self.dynamic_emergency_margin(),
            "emergency_replan_required_hits": self.emergency_replan_required_hits,
            "emergency_path_block_hit_count": self.emergency_path_block_hit_count,
            "emergency_replan_margin_max": self.emergency_replan_margin_max,
            "current_speed_mps": self.current_speed,
            "stopping_distance_m": self.vehicle_model.stopping_distance(self.current_speed),
            "dynamic_inflate": self.dynamic_inflate(),
            "path_feasibility": getattr(self, "last_feasibility_status", None),
            "path_block_uses_lidar_memory": self.use_lidar_memory_for_path_block,
            "path_block_uses_lidar_cluster_bboxes": self.use_lidar_cluster_bboxes,
            "path_block_uses_discovered_objects": self.use_discovered_objects_for_path_block,
            "route_index_never_decrease": self.route_index_never_decrease,
            "dynamic_replan_keep_route_index": self.dynamic_replan_keep_route_index,
            "route_commit_lock_sec": self.route_commit_lock_sec,
            "route_index_floor": self.route_index_floor,
            "route_checkpoint_never_decrease": self.route_checkpoint_never_decrease,
            "route_checkpoint_index": self.route_checkpoint_index,
            "route_checkpoint_total": self.route_checkpoint_total,
            "route_remaining_waypoints": [{"x": p[0], "y": p[1]} for p in self.route_remaining_waypoints],
            "route_checkpoint_reached_radius": self.route_checkpoint_reached_radius,
            "route_checkpoint_passed_z_margin": self.route_checkpoint_passed_z_margin,
            "route_commit_remaining_sec": max(0.0, self.route_commit_until_wall - self.wall_time()),
            "dynamic_replan": self.enable_dynamic_replan,
            "dynamic_replan_count": self.dynamic_replan_count,
            "dynamic_replan_max_count": self.dynamic_replan_max_count,
            "dynamic_replan_guard_reason": self.dynamic_replan_guard_reason,
            "last_dynamic_replan_pos": {"x": self.last_dynamic_replan_pos[0], "y": self.last_dynamic_replan_pos[1]} if self.last_dynamic_replan_pos else None,
            "path_block_required_hits": self.path_block_required_hits,
            "path_block_hit_count": self.path_block_hit_count,
            "dynamic_replan_min_progress_m": self.dynamic_replan_min_progress_m,
            "dynamic_replan_progress_guard_sec": self.dynamic_replan_progress_guard_sec,
            "dynamic_replan_cooldown_sec": self.dynamic_replan_cooldown_sec,
            "dynamic_replan_cooldown_remaining_sec": max(0.0, self.dynamic_replan_cooldown_sec - (self.wall_time() - self.last_dynamic_replan_wall)),
            "use_route_waypoints": self.use_route_waypoints,
            "route_map_name": self.route_map_name,
            "route_id": self.route_id,
            "route_side": self.route_side,
            "route_clearance_weight": self.route_clearance_weight,
            "semantic_risk_enabled": self.enable_semantic_risk_cost,
            "semantic_risk_weight": self.semantic_risk_weight,
            "semantic_risk_scores": self.semantic_risk_scores,
            "semantic_risk_radii": self.semantic_risk_radii,
            "theta_aware_astar_enabled": self.enable_theta_aware_astar,
            "theta_heading_change_weight": self.theta_heading_weight_for_plan(),
            "curvature_path_smoothing_enabled": self.enable_curvature_path_smoothing,
            "curvature_smoothing_min_turn_radius_m": self.curvature_smoothing_min_turn_radius_m,
            "curvature_smoothing_max_corner_angle_deg": self.curvature_smoothing_max_corner_angle_deg,
            "curvature_smoothing_point_spacing_m": self.curvature_smoothing_point_spacing_m,
            "curvature_smoothing_collision_check_margin_m": self.curvature_smoothing_collision_check_margin_m,
            "enable_replan_acceptance_filter": self.enable_replan_acceptance_filter,
            "path_commitment_sec": self.path_commitment_sec,
            "path_commitment_remaining_sec": max(0.0, self.path_commit_until_wall - self.wall_time()),
            "avoid_side_lock_sec": self.avoid_side_lock_sec,
            "avoid_side_lock_sign": self.avoid_side_lock_sign,
            "avoid_side_lock_remaining_sec": max(0.0, self.avoid_side_lock_until_wall - self.wall_time()),
            "last_candidate_reject_reason": self.last_candidate_reject_reason,
            "last_route_quality": self.last_route_quality,
            "last_candidate_quality": self.last_candidate_quality,
            "path_block_hit_count": self.path_block_hit_count,
            "path_block_required_hits": self.path_block_required_hits,
        }
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self.pub_status.publish(msg)

    def timer_cb(self) -> None:
        if self.current_pos is None or self.goal_pos is None:
            self.publish_status(None)
            return

        wall_now = self.wall_time()
        need_plan = False
        reason = ""

        # 1) 초기 또는 이벤트 기반 계획 요청. 계획이 실패하면 천천히 재시도한다.
        if not self.route and self.plan_request_pending:
            if (wall_now - self.last_plan_attempt_wall) >= self.plan_retry_period_sec:
                need_plan = True
                reason = self.plan_request_reason or "initial"

        # 2) 선택적 LiDAR 동적 재탐색. 국소 회피는 APF가 풀어야 하므로 기본은 비활성이다.
        #    전역경로가 끊임없이 재생성되는 것을 막는다.
        elif self.route and self.enable_dynamic_replan:
            cooldown_ok = (wall_now - self.last_dynamic_replan_wall) >= self.dynamic_replan_cooldown_sec
            # 빠른 재계획 트리거는 "장애물이 보였는가"가 아니라
            # "현재 A* 경로 corridor를 실제로 막는가"를 기준으로 한다.
            # detected_points_map memory뿐 아니라 최신 LiDAR cluster bbox도 직접 검사한다.
            if self.use_lidar_memory_for_path_block:
                lidar_memory_blocked = self.lidar_obstacles.is_current_path_blocked(
                    self.current_pos,
                    self.route,
                    self.route_index,
                    self.lidar_block_min_distance,
                    self.lidar_block_max_distance,
                    self.path_block_margin,
                )
            else:
                lidar_memory_blocked = False
            cluster_blocked = is_path_blocked_by_bboxes(
                self.current_pos,
                self.route,
                self.route_index,
                self.latest_cluster_bboxes if self.use_lidar_cluster_bboxes else [],
                self.lidar_block_min_distance,
                self.lidar_block_max_distance,
                self.path_block_margin,
            )
            # Emergency path block: 현재 보이는 cluster가 가까운 전방 corridor를 막으면
            # 2-hit/5초 일반 replan보다 빠르게 A*를 다시 만든다.
            # 이 검사는 memory/discovered가 아니라 최신 cluster만 사용하므로 경로 흔들림을 크게 늘리지 않는다.
            emergency_cluster_blocked = False
            if self.emergency_cluster_replan_enabled and self.use_lidar_cluster_bboxes:
                emergency_cluster_blocked = is_path_blocked_by_bboxes(
                    self.current_pos,
                    self.route,
                    self.route_index,
                    self.latest_cluster_bboxes,
                    self.emergency_replan_min_distance,
                    self.dynamic_emergency_front_distance(),
                    self.dynamic_emergency_margin(),
                )
            self.emergency_cluster_blocked = bool(emergency_cluster_blocked)
            cluster_memory_bboxes = self.build_lidar_cluster_memory_bboxes(self.latest_cluster_bboxes)
            cluster_memory_blocked = is_path_blocked_by_bboxes(
                self.current_pos,
                self.route,
                self.route_index,
                cluster_memory_bboxes if self.use_lidar_cluster_memory_for_path_block else [],
                self.lidar_block_min_distance,
                self.lidar_block_max_distance,
                self.path_block_margin,
            )
            discovered_blocked = is_path_blocked_by_bboxes(
                self.current_pos,
                self.route,
                self.route_index,
                self.discovered_bboxes if (self.use_discovered_objects_for_astar and self.use_discovered_objects_for_path_block) else [],
                self.lidar_block_min_distance,
                self.lidar_block_max_distance,
                self.path_block_margin,
            )
            lidar_blocked = bool(lidar_memory_blocked or cluster_blocked or cluster_memory_blocked or emergency_cluster_blocked)
            blocked_now = bool(lidar_blocked or discovered_blocked)
            sources = []
            if lidar_memory_blocked:
                sources.append("lidar_memory")
            if emergency_cluster_blocked:
                sources.append("emergency_lidar_cluster")
            elif cluster_blocked:
                sources.append("lidar_cluster")
            if cluster_memory_blocked:
                sources.append("lidar_cluster_memory")
            if discovered_blocked:
                sources.append("discovered")
            self.path_block_source = "+".join(sources) if sources else "none"

            if blocked_now:
                self.path_block_hit_count += 1
            else:
                self.path_block_hit_count = 0
                # 이전 tick의 cooldown/progress 메시지가 status에 계속 남지 않도록 정리한다.
                self.dynamic_replan_guard_reason = "none"

            if emergency_cluster_blocked:
                self.emergency_path_block_hit_count += 1
            else:
                self.emergency_path_block_hit_count = 0

            count_ok = self.dynamic_replan_max_count <= 0 or self.dynamic_replan_count < self.dynamic_replan_max_count
            progress_ok = True
            moved_since_last_replan = None
            if self.last_dynamic_replan_pos is not None and self.current_pos is not None:
                moved_since_last_replan = get_distance(self.current_pos, self.last_dynamic_replan_pos)
                elapsed_since_last = wall_now - self.last_dynamic_replan_wall
                if (
                    elapsed_since_last < self.dynamic_replan_progress_guard_sec
                    and moved_since_last_replan < self.dynamic_replan_min_progress_m
                ):
                    progress_ok = False
                    self.dynamic_replan_guard_reason = (
                        f"progress_guard moved={moved_since_last_replan:.2f}m "
                        f"elapsed={elapsed_since_last:.1f}s"
                    )

            emergency_cooldown_ok = (wall_now - self.last_dynamic_replan_wall) >= max(0.0, self.emergency_replan_cooldown_sec)
            emergency_hits_ok = self.emergency_path_block_hit_count >= self.emergency_replan_required_hits
            if emergency_cluster_blocked and count_ok and progress_ok and emergency_cooldown_ok and emergency_hits_ok:
                need_plan = True
                reason = "emergency_lidar_cluster_path_blocked"
                self.dynamic_replan_guard_reason = "none"
            elif cooldown_ok and count_ok and progress_ok and self.path_block_hit_count >= self.path_block_required_hits:
                need_plan = True
                reason = "lidar_path_blocked"
                self.dynamic_replan_guard_reason = "none"
            elif emergency_cluster_blocked and not emergency_hits_ok:
                self.dynamic_replan_guard_reason = (
                    f"emergency hits {self.emergency_path_block_hit_count}/{self.emergency_replan_required_hits}"
                )
            elif emergency_cluster_blocked and not emergency_cooldown_ok:
                self.dynamic_replan_guard_reason = (
                    f"emergency cooldown remaining={max(0.0, self.emergency_replan_cooldown_sec - (wall_now - self.last_dynamic_replan_wall)):.2f}s"
                )
            elif blocked_now and not cooldown_ok:
                self.dynamic_replan_guard_reason = (
                    f"cooldown remaining={max(0.0, self.dynamic_replan_cooldown_sec - (wall_now - self.last_dynamic_replan_wall)):.2f}s"
                )
            elif not count_ok:
                self.dynamic_replan_guard_reason = "max_dynamic_replans_reached"

        # 3) 선택적 저빈도 루트 갱신. 이것도 기본은 비활성이다.
        if (
            not need_plan
            and self.route
            and self.enable_periodic_replan
            and self.replan_period_sec > 0.0
            and (wall_now - self.last_plan_wall) > self.replan_period_sec
        ):
            need_plan = True
            reason = "periodic_refresh"

        if get_distance(self.current_pos, self.goal_pos) < self.goal_tolerance:
            # 컨트롤러가 멈추는 동안에도 최종 목표점을 계속 발행한다.
            pass
        elif need_plan:
            self.maybe_plan(reason)

        self.publish_path()
        lookahead = self.publish_lookahead()
        self.publish_status(lookahead)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TeamDynamicAStarPlannerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()