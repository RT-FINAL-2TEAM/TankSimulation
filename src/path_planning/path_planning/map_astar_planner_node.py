# -*- coding: utf-8 -*-
"""
ROS2 conversion of the latest TankSimulation path-planning work.

Source intent from TankSimulation.zip:
- src/planning/path_planning.py: grid A* with obstacle inflation and LOS smoothing.
- tests/step3_threat_avoidance/run_server.py: do not rely on GT obstacles by default;
  accumulate LiDAR obstacles, detect when they block the current route, then replan.

This node keeps the same ROS2 outputs so existing RViz/controller nodes continue to work.

Important integration policy:
- A* global path is NOT recomputed every timer tick.
- By default it plans once after start/goal is known, then only replans on explicit goal/GT-obstacle update.
- LiDAR-based dynamic replanning is opt-in and rate-limited; APF should handle normal local avoidance.
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
from geometry_msgs.msg import PoseStamped
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
    PUBLISH_PATH_PERIOD_SEC,
    REPLAN_PERIOD_SEC,
    TOPIC_GLOBAL_PATH,
    TOPIC_GOAL_POSE,
    TOPIC_LIDAR_BBOXES,
    TOPIC_LIDAR_CLUSTERS,
    TOPIC_LIDAR_DETECTED_MAP,
    TOPIC_LOOKAHEAD_POSE,
    TOPIC_MAP_OBSTACLES,
    TOPIC_PATH_POINTS,
    TOPIC_PLANNER_STATUS,
    TOPIC_PLAYER_POSE,
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
    LIDAR_CLUSTER_BBOX_MARGIN,
)

from ament_index_python.packages import get_package_share_directory
from path_planning.route_loader import get_route_waypoints
from path_planning.team_path_planning import (
    plan_path_through_waypoints as team_plan_path_through_waypoints,
    load_static_obstacles_from_map,
)


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def pointcloud2_to_xyz_array(msg: PointCloud2) -> np.ndarray:
    """Return PointCloud2 XYZ fields as a contiguous float32 (N, 3) array.

    ROS2 Humble/newer sensor_msgs_py provides read_points_numpy(), which avoids
    building Python dict/list objects for every LiDAR hit.  The fallback keeps the
    node usable on older sensor_msgs_py versions.
    """
    try:
        arr = point_cloud2.read_points_numpy(
            msg, field_names=("x", "y", "z"), skip_nans=True
        )
    except Exception:
        pts = point_cloud2.read_points(
            msg, field_names=("x", "y", "z"), skip_nans=True
        )
        if isinstance(pts, np.ndarray):
            arr = pts
        else:
            arr = np.asarray(list(pts), dtype=np.float32)
    if arr is None:
        return np.empty((0, 3), dtype=np.float32)
    arr = np.asarray(arr)
    if arr.dtype.fields:
        arr = np.column_stack((arr["x"], arr["y"], arr["z"]))
    arr = np.asarray(arr, dtype=np.float32)
    if arr.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    return np.ascontiguousarray(arr.reshape(-1, 3), dtype=np.float32)


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
            x_min = max(0, int((float(obs["x_min"]) - inflate) / res))
            x_max = min(cols - 1, int((float(obs["x_max"]) + inflate) / res))
            z_min = max(0, int((float(obs["z_min"]) - inflate) / res))
            z_max = min(rows - 1, int((float(obs["z_max"]) + inflate) / res))
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
) -> List[Tuple[float, float]]:
    grid = create_grid(width, height, resolution)
    if static_obstacles:
        # 정적 장애물(나무 등)은 작은 inflate=1.0로(team_path_planning와 동일 정책)
        add_obstacles(grid, static_obstacles, resolution, 1.0)
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
    """Accept bridge payloads and direct payloads."""
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


def prefab_half_size(name: str) -> Tuple[float, float]:
    lname = str(name).lower()
    for key, value in PREFAB_HALF_SIZES.items():
        if key.lower() in lname:
            return value
    return 1.0, 1.0


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


def find_lookahead_along_path(pos: Tuple[float, float], route: Sequence[Tuple[float, float]], lookahead_dist: float) -> Tuple[Tuple[float, float], int]:
    """Project current position to path, then advance by lookahead distance."""
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
        # Periodic replanning caused the vehicle to chase a constantly moving global path.
        # Keep it disabled by default. Set >0 only for deliberate experiments.
        self.declare_parameter("enable_periodic_replan", ENABLE_PERIODIC_REPLAN)
        self.declare_parameter("replan_period_sec", REPLAN_PERIOD_SEC)
        # LiDAR replanning is also throttled because raw LiDAR points can repeatedly mark
        # the current path as blocked and cause planning loops.
        self.declare_parameter("dynamic_replan_cooldown_sec", DYNAMIC_REPLAN_COOLDOWN_SEC)
        self.declare_parameter("plan_retry_period_sec", PLAN_RETRY_PERIOD_SEC)
        self.declare_parameter("path_block_margin", PATH_BLOCK_MARGIN)
        self.declare_parameter("path_block_required_hits", PATH_BLOCK_REQUIRED_HITS)
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
        self.declare_parameter("use_lidar_cluster_bboxes", USE_LIDAR_CLUSTER_BBOXES)
        self.declare_parameter("lidar_cluster_bbox_margin", LIDAR_CLUSTER_BBOX_MARGIN)

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
        self.use_lidar_cluster_bboxes = bool(self.get_parameter("use_lidar_cluster_bboxes").value)
        self.lidar_cluster_bbox_margin = float(self.get_parameter("lidar_cluster_bbox_margin").value)

        # 정적 맵(나무 등)을 1회 로드해 보관. use_gt_obstacles/obstacles_cb/replan과 독립.
        # 전역 A* 코스트맵에 넣어 나무 회피 + clearance 중앙정렬을 가능하게 한다.
        self.static_obstacles: List[Dict[str, float]] = []
        if self.use_static_map:
            sm_path = self.static_map_file or os.path.join(
                get_package_share_directory("rviz_visualization"), "map", "finalmap.map")
            self.static_obstacles = load_static_obstacles_from_map(sm_path)
            self.get_logger().info(
                f"static map obstacles loaded: {len(self.static_obstacles)} from {sm_path}")

        self.current_pos: Optional[Tuple[float, float]] = None
        self.goal_pos: Optional[Tuple[float, float]] = None
        if bool(self.get_parameter("default_goal_enabled").value):
            self.goal_pos = (float(self.get_parameter("default_goal_x").value), float(self.get_parameter("default_goal_y").value))
        self.gt_obstacles: List[Dict[str, float]] = []
        self.lidar_obstacles = LidarObstacleMemory()
        self.latest_cluster_bboxes: List[Dict[str, float]] = []
        self.latest_cluster_count = 0
        self.route: List[Tuple[float, float]] = []
        self.route_version = 0
        self.route_index = 0
        # Use monotonic wall time for planner throttling; ROS time can stay at 0 when /clock is not active.
        self.last_plan_wall = -1e9
        self.last_plan_attempt_wall = -1e9
        self.last_dynamic_replan_wall = -1e9
        self.last_path_publish_wall = -1e9
        self.path_block_hit_count = 0
        self.last_replan_reason = "not_planned"
        self.plan_request_pending = True
        self.plan_request_reason = "initial"

        self.pub_path = self.create_publisher(NavPath, TOPIC_GLOBAL_PATH, 10)
        self.pub_lookahead = self.create_publisher(PoseStamped, TOPIC_LOOKAHEAD_POSE, 10)
        self.pub_points = self.create_publisher(String, TOPIC_PATH_POINTS, 10)
        self.pub_status = self.create_publisher(String, TOPIC_PLANNER_STATUS, 10)
        self.pub_lidar_bboxes = self.create_publisher(String, TOPIC_LIDAR_BBOXES, 10)
        # 정찰/자율 시나리오에서 controller·local_path가 "도착"을 판정하려면 goal이 필요하다.
        # planner가 보유한 goal_pos(기본 목적지 또는 sim이 /set_destination으로 준 값)를
        # /tank/goal/pose로 주기 발행해 자율 스택의 단일 goal 소스로 삼는다.
        self.pub_goal = self.create_publisher(PoseStamped, TOPIC_GOAL_POSE, 10)

        self.create_subscription(PoseStamped, TOPIC_PLAYER_POSE, self.player_pose_cb, 10)
        self.create_subscription(PoseStamped, TOPIC_GOAL_POSE, self.goal_pose_cb, 10)
        self.create_subscription(String, TOPIC_MAP_OBSTACLES, self.obstacles_cb, 10)
        self.create_subscription(PointCloud2, TOPIC_LIDAR_DETECTED_MAP, self.lidar_cb, 10)
        self.create_subscription(String, TOPIC_LIDAR_CLUSTERS, self.lidar_clusters_cb, 10)
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
            f"cluster_bboxes={self.use_lidar_cluster_bboxes}"
        )

    def wall_time(self) -> float:
        return time.monotonic()

    def player_pose_cb(self, msg: PoseStamped) -> None:
        new_pos = (float(msg.pose.position.x), float(msg.pose.position.y))
        if self.current_pos is not None:
            if get_distance(self.current_pos, new_pos) > 10.0:
                self.get_logger().info("Teleport/Restart detected. Triggering replan from new start.")
                self.route = []
                self.plan_request_pending = True
                self.plan_request_reason = "teleport_detected"
        self.current_pos = new_pos

    def goal_pose_cb(self, msg: PoseStamped) -> None:
        new_goal = (float(msg.pose.position.x), float(msg.pose.position.y))
        if self.goal_pos is None or get_distance(new_goal, self.goal_pos) > 0.5:
            self.goal_pos = new_goal
            self.route = []
            self.plan_request_pending = True
            self.plan_request_reason = "goal_updated"
            self.last_replan_reason = "goal_updated"
            self.path_block_hit_count = 0

    def obstacles_cb(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
            obs = parse_obstacles_payload(payload)
        except Exception as exc:
            self.get_logger().warn(f"failed to parse /tank/map/obstacles: {exc}")
            return

        # Update even when the list is empty; otherwise stale GT obstacles can remain forever.
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
            # LidarObstacleMemory still owns the planning-side memory/clustering logic.
            # Feed it a minimal in-memory payload instead of parsing a LiDAR JSON String.
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
                        # Cluster bbox is in ROS map x/y/z. A* bbox uses x/z keys, where z means map-plane y.
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
            if self.use_lidar_cluster_bboxes:
                # Publish cluster-derived A* bboxes even when dynamic replanning is disabled,
                # so RViz/debug nodes can verify that the DBSCAN output is connected to planner bbox input.
                self.publish_lidar_bboxes(bboxes)
        except Exception as exc:
            self.get_logger().warn(f"failed to parse lidar clusters: {exc}")

    def build_lidar_bboxes(self) -> List[Dict[str, float]]:
        if self.use_lidar_cluster_bboxes and self.latest_cluster_bboxes:
            return list(self.latest_cluster_bboxes)
        return self.lidar_obstacles.build_bboxes(self.lidar_cluster_eps, self.lidar_cluster_min_samples)

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
        lidar_bboxes: List[Dict[str, float]] = []
        if self.enable_dynamic_replan and self.lidar_obstacles.history_count > 0:
            lidar_bboxes = deepcopy(self.build_lidar_bboxes())
            obstacles.extend(lidar_bboxes)

        threading.Thread(
            target=self._plan_worker,
            args=(reason, start_pos, goal_pos, obstacles, lidar_bboxes),
            daemon=True
        ).start()

    def _plan_worker(
        self,
        reason: str,
        start_pos: Tuple[float, float],
        goal_pos: Tuple[float, float],
        obstacles: List[Dict[str, float]],
        lidar_bboxes: List[Dict[str, float]]
    ) -> None:
        try:
            route: List[Tuple[float, float]] = []
            route_mode = "direct_astar"
            if self.use_route_waypoints:
                try:
                    route_config = self.route_config_file or None
                    waypoints = get_route_waypoints(self.route_map_name, self.route_id, route_config)
                    through = list(waypoints) + [goal_pos]
                    route = team_plan_path_through_waypoints(
                        start_pos,
                        through,
                        obstacles,
                        static_obstacles=self.static_obstacles,
                        inflate=self.inflate,
                        clearance_weight=self.route_clearance_weight,
                        side=self.route_side,
                    )
                    route_mode = f"route_waypoints:{self.route_map_name}/{self.route_id}/{self.route_side}"
                except Exception as exc:
                    self.get_logger().warn(f"route waypoint planning failed, fallback direct A*: {exc}")
                    route = []
            if not route:
                route = plan_global_path(
                    start_pos,
                    goal_pos,
                    obstacles,
                    width=self.map_width,
                    height=self.map_height,
                    resolution=self.resolution,
                    inflate=self.inflate,
                    use_smoothing=self.use_path_smoothing,
                    max_expansions=self.max_expansions,
                    static_obstacles=self.static_obstacles,
                )
                route_mode = "direct_astar"

            if route:
                with self._planning_lock:
                    self.route = route
                    self.route_index = 0
                    self.route_version += 1
                    self.last_plan_wall = self.wall_time()
                    if reason == "lidar_path_blocked":
                        self.last_dynamic_replan_wall = self.last_plan_wall
                    self.plan_request_pending = False
                    self.path_block_hit_count = 0
                    self.last_path_publish_wall = -1e9
                    self.last_replan_reason = reason
                
                self.publish_lidar_bboxes(lidar_bboxes)
                self.publish_path(force=True)
                self.get_logger().info(
                    f"A* path updated: reason={reason}, mode={route_mode}, points={len(route)}, "
                    f"obstacles={len(obstacles)}, lidar_bboxes={len(lidar_bboxes)}"
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
            ps.pose.position.z = 0.0
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

    def publish_lookahead(self) -> Optional[Tuple[float, float]]:
        if self.current_pos is None or not self.route:
            return None
        if self.goal_pos is not None and get_distance(self.current_pos, self.goal_pos) < self.goal_tolerance:
            target = self.goal_pos
            idx = len(self.route) - 1
        else:
            target, idx = find_lookahead_along_path(self.current_pos, self.route, self.lookahead_distance)
        self.route_index = idx
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = MAP_FRAME
        msg.pose.position.x = float(target[0])
        msg.pose.position.y = float(target[1])
        msg.pose.position.z = 0.0
        msg.pose.orientation.w = 1.0
        self.pub_lookahead.publish(msg)
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
            "use_lidar_cluster_bboxes": self.use_lidar_cluster_bboxes,
            "dynamic_replan": self.enable_dynamic_replan,
            "use_route_waypoints": self.use_route_waypoints,
            "route_map_name": self.route_map_name,
            "route_id": self.route_id,
            "route_side": self.route_side,
            "route_clearance_weight": self.route_clearance_weight,
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

        # 1) Initial or event-driven plan request. Retry slowly if planning fails.
        if not self.route and self.plan_request_pending:
            if (wall_now - self.last_plan_attempt_wall) >= self.plan_retry_period_sec:
                need_plan = True
                reason = self.plan_request_reason or "initial"

        # 2) Optional LiDAR dynamic replanning. Disabled by default because APF should
        #    solve local avoidance; this prevents continuous global-path regeneration.
        elif self.route and self.enable_dynamic_replan:
            cooldown_ok = (wall_now - self.last_dynamic_replan_wall) >= self.dynamic_replan_cooldown_sec
            blocked_now = self.lidar_obstacles.is_current_path_blocked(
                self.current_pos,
                self.route,
                self.route_index,
                self.lidar_block_min_distance,
                self.lidar_block_max_distance,
                self.path_block_margin,
            )
            if blocked_now:
                self.path_block_hit_count += 1
            else:
                self.path_block_hit_count = 0

            if cooldown_ok and self.path_block_hit_count >= self.path_block_required_hits:
                need_plan = True
                reason = "lidar_path_blocked"

        # 3) Optional low-rate route refresh. Also disabled by default.
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
            # Keep publishing the final target while controller stops.
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