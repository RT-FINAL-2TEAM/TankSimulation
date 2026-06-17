# -*- coding: utf-8 -*-
"""
Lecture-style Artificial Potential Field node for Tank Challenge ROS2.

This implementation keeps the existing ROS2 topic contract, but rewrites the
internal APF calculation so that it follows the common potential-field lecture
formulation explicitly:

    U_A = 1/2 * k_A * d^2
    F_A = -grad(U_A) = k_A * (r_D - r_B)

    U_R = 1/2 * k_R * (1/g - 1/g*)^2,  g <= g*
    F_R = -grad(U_R) = k_R * (1/g - 1/g*) / g^3 * (r_B - r_O)

    F = F_A + F_R + F_T + F_threat
    v_S = ||F||
    theta_D = atan2(F_y, F_x)
    theta_dot_S = k_theta * wrap(theta_D - theta)

The node publishes the same local target used by control, plus detailed
status/debug topics for RViz and tuning.

Design policy:
- User-tunable values are placed as global variables near the top of this file.
- The same values are also declared as ROS2 parameters, so launch files can still
  override them later.
- The controller remains responsible for converting the local target into W/A/S/D.
  This APF node computes the desired force direction and local target.
"""

import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Point, PoseStamped, Vector3Stamped
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


# =============================================================================
# 0. User-tunable global variables
# =============================================================================
# Global defaults are centralized in potential.config. ROS2 parameters below keep
# launch-time override compatibility.
from lidar.payloads import parse_lidar_points_payload
from path_planning.config import PREFAB_HALF_SIZES
from potential.config import (
    ANGLE_EPSILON_DEG,
    ANGULAR_GAIN_K_THETA,
    APF_HZ,
    DISCOVERED_OBJECTS_TOPIC,
    FALLBACK_GOAL_TOPIC,
    FRONT_SECTOR_DEG,
    K_ATTRACTIVE,
    K_REPULSIVE,
    K_THREAT_REPULSIVE,
    LIDAR_POINTS_TOPIC,
    LINEAR_SPEED_GAIN,
    LOCAL_TARGET_DISTANCE,
    MAP_FRAME,
    MARKER_SCALE,
    MAX_ATTRACTIVE_FORCE,
    MAX_DESIRED_ANGULAR_SPEED,
    MAX_DESIRED_SPEED,
    MAX_OBSTACLE_DISTANCE,
    MAX_OBSTACLE_POINTS,
    MAX_REPULSIVE_FORCE,
    MAX_RESULT_FORCE,
    MIN_OBSTACLE_DISTANCE,
    MOTION_STRATEGY,
    OBSTACLE_INFLUENCE_RADIUS,
    OBSTACLE_VOXEL_RESOLUTION,
    PASSTHROUGH_WHEN_CLEAR,
    PATH_CORRIDOR_WIDTH,
    PLAYER_POSE_TOPIC,
    REPULSIVE_EPS,
    TANGENTIAL_GAIN_SCALE,
    TARGET_POSE_TOPIC,
    THREAT_RADIUS,
    THREAT_TYPES,
    USE_DISCOVERED_OBJECTS,
    USE_TANGENTIAL_FORCE,
    USE_THREAT_AVOIDANCE,
    LIDAR_CLUSTERS_TOPIC,
    USE_LIDAR_CLUSTERS,
    CLUSTER_OBSTACLE_MIN_COUNT,
    APF_WEIGHT_PROFILE,
    APF_WEIGHTS_FILE,
)


# =============================================================================
# 1. Math utilities
# =============================================================================


@dataclass
class ForceBreakdown:
    attractive: Tuple[float, float]
    repulsive: Tuple[float, float]
    tangential: Tuple[float, float]
    threat: Tuple[float, float]
    result: Tuple[float, float]
    attractive_potential: float
    repulsive_potential: float
    threat_potential: float


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


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


def vector_norm(v: Tuple[float, float]) -> float:
    return math.hypot(v[0], v[1])


def limit_norm(v: Tuple[float, float], max_norm: float) -> Tuple[float, float]:
    n = vector_norm(v)
    if max_norm <= 0.0 or n <= max_norm or n < 1e-9:
        return v
    s = max_norm / n
    return v[0] * s, v[1] * s


def normalize_angle_rad(angle: float) -> float:
    """Wrap angle to [-pi, pi]. This removes atan2 discontinuity at ±pi."""
    return math.atan2(math.sin(angle), math.cos(angle))


def normalize_angle_deg(angle: float) -> float:
    return math.degrees(normalize_angle_rad(math.radians(angle)))


def quaternion_msg_to_yaw_rad(q: Any) -> float:
    """Standard ROS 2D yaw from geometry_msgs Quaternion.

    The APF status uses this only for desired angular velocity debugging. The
    local-target position remains valid even if orientation is not perfect.
    """
    x = float(getattr(q, "x", 0.0))
    y = float(getattr(q, "y", 0.0))
    z = float(getattr(q, "z", 0.0))
    w = float(getattr(q, "w", 1.0))
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def yaw_rad_to_quaternion_z(yaw: float) -> Tuple[float, float, float, float]:
    half = 0.5 * yaw
    return 0.0, 0.0, math.sin(half), math.cos(half)


def simulator_quaternion_to_yaw_deg(rot: Dict[str, Any]) -> float:
    """Yaw helper for .map quaternion written by the simulator.

    Existing map files use Unity-like fields. We keep the previous Y-axis yaw
    extraction because it was used for threat-map parsing.
    """
    qx = float(rot.get("x", 0.0))
    qy = float(rot.get("y", 0.0))
    qz = float(rot.get("z", 0.0))
    qw = float(rot.get("w", 1.0))
    siny_cosp = 2.0 * (qw * qy + qz * qx)
    cosy_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    return math.degrees(math.atan2(siny_cosp, cosy_cosp))


# =============================================================================
# 2. Lecture-style APF equations
# =============================================================================


def calc_attractive_force(
    pos: Tuple[float, float],
    target: Tuple[float, float],
    k_att: float,
    max_force: float,
) -> Tuple[Tuple[float, float], float]:
    """Attractive potential and force.

    d = ||r_B - r_D||
    U_A = 1/2 * k_A * d^2
    F_A = -grad(U_A) = k_A * (r_D - r_B)
    """
    dx = target[0] - pos[0]
    dy = target[1] - pos[1]
    d2 = dx * dx + dy * dy
    potential = 0.5 * k_att * d2
    force = (k_att * dx, k_att * dy)
    force = limit_norm(force, max_force)
    return force, potential


def _repulsive_from_point(
    pos: Tuple[float, float],
    obs: Tuple[float, float],
    k_rep: float,
    g_star: float,
) -> Tuple[Tuple[float, float], float]:
    """Repulsive potential and force from one obstacle point.

    g = ||r_B - r_O||
    U_R = 1/2 * k_R * (1/g - 1/g*)^2, if g <= g*
    F_R = -grad(U_R) = k_R * (1/g - 1/g*) / g^3 * (r_B - r_O)

    The sign is selected so that the force pushes the vehicle away from the
    obstacle. This is the physically useful obstacle-avoidance direction.
    """
    g = get_distance(pos, obs)
    if g <= 1e-6 or g > g_star:
        return (0.0, 0.0), 0.0

    inv_gap = (1.0 / g) - (1.0 / g_star)
    potential = 0.5 * k_rep * inv_gap * inv_gap
    coeff = k_rep * inv_gap / (g ** 3)
    fx = coeff * (pos[0] - obs[0])
    fy = coeff * (pos[1] - obs[1])
    return (fx, fy), potential


def calc_repulsive_force(
    pos: Tuple[float, float],
    obstacles: Iterable[Tuple[float, float]],
    target: Tuple[float, float],
    k_rep: float,
    g_star: float,
    max_force: float,
    use_tangent: bool,
    tangent_gain_scale: float,
) -> Tuple[Tuple[float, float], Tuple[float, float], float]:
    """Sum lecture-style repulsion and optional tangential wrapping force."""
    rep_x, rep_y = 0.0, 0.0
    tan_x, tan_y = 0.0, 0.0
    total_potential = 0.0

    dest_x = target[0] - pos[0]
    dest_y = target[1] - pos[1]

    for obs in obstacles:
        rep, pot = _repulsive_from_point(pos, obs, k_rep, g_star)
        total_potential += pot
        rep_x += rep[0]
        rep_y += rep[1]

        if not use_tangent or pot <= 0.0:
            continue

        # Tangential component is not part of the basic APF derivation. It is a
        # practical extension to reduce local minima and head-on oscillation.
        g = get_distance(pos, obs)
        if g <= 1e-6:
            continue
        away_x = (pos[0] - obs[0]) / g
        away_y = (pos[1] - obs[1]) / g
        ccw = (-away_y, away_x)
        cw = (away_y, -away_x)

        # Pick the tangent direction that is more aligned with the attractive target.
        if ccw[0] * dest_x + ccw[1] * dest_y >= cw[0] * dest_x + cw[1] * dest_y:
            tx, ty = ccw
        else:
            tx, ty = cw

        normal_mag = vector_norm(rep)
        tan_mag = normal_mag * tangent_gain_scale
        tan_x += tan_mag * tx
        tan_y += tan_mag * ty

    rep = limit_norm((rep_x, rep_y), max_force)
    tan = limit_norm((tan_x, tan_y), max_force)
    return rep, tan, total_potential


def segment_intersect_bbox(px: float, pz: float, qx: float, qz: float, bbox: Dict[str, float]) -> bool:
    xmin, xmax = bbox.get("x_min", 0.0), bbox.get("x_max", 0.0)
    zmin, zmax = bbox.get("z_min", 0.0), bbox.get("z_max", 0.0)
    if min(px, qx) > xmax or max(px, qx) < xmin: return False
    if min(pz, qz) > zmax or max(pz, qz) < zmin: return False
    
    t0 = 0.0
    t1 = 1.0
    dx = qx - px
    dz = qz - pz
    
    if abs(dx) > 1e-6:
        tx1 = (xmin - px) / dx
        tx2 = (xmax - px) / dx
        t0 = max(t0, min(tx1, tx2))
        t1 = min(t1, max(tx1, tx2))
    elif px < xmin or px > xmax:
        return False

    if abs(dz) > 1e-6:
        tz1 = (zmin - pz) / dz
        tz2 = (zmax - pz) / dz
        t0 = max(t0, min(tz1, tz2))
        t1 = min(t1, max(tz1, tz2))
    elif pz < zmin or pz > zmax:
        return False

    return t0 <= t1

def check_los(tank_x: float, tank_z: float, threat_x: float, threat_z: float, gt_obstacles: List[Dict[str, float]]) -> bool:
    for obs in gt_obstacles:
        xmin, xmax = obs.get("x_min", 0.0), obs.get("x_max", 0.0)
        zmin, zmax = obs.get("z_min", 0.0), obs.get("z_max", 0.0)
        if xmin <= threat_x <= xmax and zmin <= threat_z <= zmax:
            continue
        if segment_intersect_bbox(tank_x, tank_z, threat_x, threat_z, obs):
            return False
    return True

def is_threat_active(pos: Tuple[float, float], threat: Dict[str, Any], gt_obstacles: List[Dict[str, float]]) -> bool:
    tx, tz = pos
    dx = tx - float(threat.get("x", 0.0))
    dz = tz - float(threat.get("z", 0.0))
    dist = math.hypot(dx, dz)
    
    t_type = str(threat.get("type", "unknown"))
    prefab_name = str(threat.get("prefabName", ""))
    
    if t_type == "House002" or prefab_name.startswith("House002"):
        if dist > 25.0:
            return False
        target_yaw = math.degrees(math.atan2(dx, dz))
        yaw_diff = abs(normalize_angle_deg(target_yaw - float(threat.get("yaw", 0.0))))
        if yaw_diff > 30.0:
            return False
        if check_los(tx, tz, float(threat["x"]), float(threat["z"]), gt_obstacles):
            return True
        return False
    elif t_type == "Tank001" or prefab_name.startswith("Tank001"):
        if dist > 20.0:
            return False
        if check_los(tx, tz, float(threat["x"]), float(threat["z"]), gt_obstacles):
            return True
        return False
    return dist <= 25.0

def calc_resultant_force(
    pos: Tuple[float, float],
    target: Tuple[float, float],
    obstacles: Iterable[Tuple[float, float]],
    threats: Iterable[Dict[str, Any]],
    k_att: float,
    k_rep: float,
    g_star: float,
    max_att: float,
    max_rep: float,
    max_result: float,
    use_tangent: bool,
    tangent_gain_scale: float,
    use_threats: bool,
    threat_radius: float,
    k_threat: float,
    gt_obstacles: List[Dict[str, float]],
) -> ForceBreakdown:
    att, u_att = calc_attractive_force(pos, target, k_att, max_att)
    rep, tan, u_rep = calc_repulsive_force(
        pos, obstacles, target, k_rep, g_star, max_rep, use_tangent, tangent_gain_scale
    )

    threat_force = (0.0, 0.0)
    u_threat = 0.0
    if use_threats:
        tx, ty = 0.0, 0.0
        for threat in threats:
            if is_threat_active(pos, threat, gt_obstacles):
                threat_pos = (float(threat.get("x", 0.0)), float(threat.get("z", 0.0)))
                f, pot = _repulsive_from_point(pos, threat_pos, k_threat, threat_radius)
                tx += f[0]
                ty += f[1]
                u_threat += pot
        threat_force = limit_norm((tx, ty), max_rep)

    result = (
        att[0] + rep[0] + tan[0] + threat_force[0],
        att[1] + rep[1] + tan[1] + threat_force[1],
    )
    if vector_norm(result) < 1e-9:
        result = att
    result = limit_norm(result, max_result)

    return ForceBreakdown(
        attractive=att,
        repulsive=rep,
        tangential=tan,
        threat=threat_force,
        result=result,
        attractive_potential=u_att,
        repulsive_potential=u_rep,
        threat_potential=u_threat,
    )


def compute_desired_motion(
    current_yaw_rad: Optional[float],
    result_force: Tuple[float, float],
    k_theta: float,
    angle_epsilon_deg: float,
    speed_gain: float,
    max_speed: float,
    max_omega: float,
    strategy: str,
) -> Dict[str, float]:
    """Compute lecture-style desired velocity and angular velocity.

    theta_D = atan2(F_y, F_x)
    theta_dot_S = k_theta * wrap(theta_D - theta)

    Strategy:
    - first: move only after heading error is inside epsilon.
    - second: rotate first until epsilon, then translate while continuing rotation.
      In this APF node both strategies expose the same target direction; the
      difference is represented in desired_speed for status/debugging.
    """
    f_norm = vector_norm(result_force)
    if f_norm < 1e-9:
        theta_d = current_yaw_rad if current_yaw_rad is not None else 0.0
    else:
        theta_d = math.atan2(result_force[1], result_force[0])

    if current_yaw_rad is None:
        theta_error = 0.0
    else:
        theta_error = normalize_angle_rad(theta_d - current_yaw_rad)

    omega = clamp(k_theta * theta_error, -max_omega, max_omega)
    aligned = abs(math.degrees(theta_error)) <= angle_epsilon_deg

    if str(strategy).lower() == "first":
        desired_speed = min(max_speed, speed_gain * f_norm) if aligned else 0.0
    else:
        # 강의의 second strategy: epsilon 안에 들어오면 회전과 병진을 동시에 허용.
        desired_speed = min(max_speed, speed_gain * f_norm) if aligned else 0.0

    return {
        "theta_desired_rad": theta_d,
        "theta_desired_deg": math.degrees(theta_d),
        "theta_error_rad": theta_error,
        "theta_error_deg": math.degrees(theta_error),
        "omega_cmd_rad_s": omega,
        "desired_speed": desired_speed,
        "aligned": 1.0 if aligned else 0.0,
    }


# =============================================================================
# 3. Payload parsing utilities
# =============================================================================


def parse_discovered_objects_payload(payload: Any) -> List[Tuple[float, float]]:
    """Parse /tank/map/discovered/objects into map x/y obstacle points."""
    points: List[Tuple[float, float]] = []
    if not isinstance(payload, dict):
        return points

    objects = payload.get("objects")
    if isinstance(objects, list):
        for obj in objects:
            if not isinstance(obj, dict):
                continue
            try:
                if obj.get("map_x") is not None and obj.get("map_y") is not None:
                    points.append((float(obj.get("map_x")), float(obj.get("map_y"))))
                    continue
                pos = obj.get("position_map") if isinstance(obj.get("position_map"), dict) else None
                if pos is not None:
                    points.append((float(pos.get("x", 0.0)), float(pos.get("y", 0.0))))
            except Exception:
                continue

    obstacles = payload.get("obstacles")
    if isinstance(obstacles, list):
        for obj in obstacles:
            if not isinstance(obj, dict):
                continue
            pos = obj.get("position") if isinstance(obj.get("position"), dict) else None
            if pos is None:
                continue
            try:
                # saved discovered map policy: raw.x=map.x, raw.z=map.y
                points.append((float(pos.get("x", 0.0)), float(pos.get("z", 0.0))))
            except Exception:
                continue
    return points




def parse_lidar_clusters_payload(payload: Any, min_count: int = 2) -> List[Tuple[float, float]]:
    """Parse /tank/visual_perception/lidar_clusters into map x/y obstacle points."""
    points: List[Tuple[float, float]] = []
    if not isinstance(payload, dict):
        return points
    clusters = payload.get("clusters")
    if not isinstance(clusters, list):
        return points
    for c in clusters:
        if not isinstance(c, dict):
            continue
        try:
            if int(c.get("count", 0)) < int(min_count):
                continue
        except Exception:
            continue
        centroid = c.get("centroid") if isinstance(c.get("centroid"), dict) else None
        if centroid is None:
            continue
        try:
            points.append((float(centroid.get("x", 0.0)), float(centroid.get("y", 0.0))))
        except Exception:
            continue
    return points


def load_apf_weight_profile(path: str, profile_name: str) -> Dict[str, Any]:
    """Load heuristic/RL-ready APF weight profile YAML.

    The current APF equation still uses ROS parameters as the source of truth.
    This profile is exposed in status and can be used by launch/RL code to choose
    object/situation/terrain multipliers later.
    """
    if yaml is None or not path:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        profiles = data.get("profiles", {}) if isinstance(data, dict) else {}
        profile = profiles.get(profile_name, {}) if isinstance(profiles, dict) else {}
        return profile if isinstance(profile, dict) else {}
    except Exception:
        return {}

def parse_threats_from_map(map_path: str) -> List[Dict[str, Any]]:
    try:
        with open(map_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []

    threats: List[Dict[str, Any]] = []
    for obs in data.get("obstacles", []):
        if not isinstance(obs, dict):
            continue
        prefab_name = str(obs.get("prefabName", ""))
        if not any(prefab_name.startswith(t) for t in THREAT_TYPES):
            continue
        pos = obs.get("position") if isinstance(obs.get("position"), dict) else {}
        rot = obs.get("rotation") if isinstance(obs.get("rotation"), dict) else {}
        threat_type = next((t for t in THREAT_TYPES if prefab_name.startswith(t)), "unknown")
        threats.append(
            {
                "type": threat_type,
                "x": float(pos.get("x", 0.0)),
                "z": float(pos.get("z", 0.0)),
                "yaw": simulator_quaternion_to_yaw_deg(rot),
                "prefabName": prefab_name,
            }
        )
    return threats

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
    obstacles = []
    if isinstance(payload, list):
        obstacles = payload
    elif isinstance(payload, dict):
        obstacles = payload.get("obstacles", payload.get("data", {}).get("obstacles", []))
    if not isinstance(obstacles, list):
        obstacles = []
    for item in obstacles:
        if isinstance(item, dict):
            bbox = obstacle_to_bbox(item)
            if bbox is not None:
                bboxes.append(bbox)
    return bboxes


def filter_obstacles_for_apf(
    obstacles: List[Tuple[float, float]],
    pos: Tuple[float, float],
    target: Tuple[float, float],
    min_distance: float,
    max_distance: float,
    front_sector_deg: float,
    corridor_width: float,
    voxel_resolution: float,
    max_points: int,
) -> List[Tuple[float, float]]:
    """Keep only obstacle points relevant to the next local segment.

    Raw LiDAR may contain thousands of points. Feeding all points into APF makes
    the repulsive force saturate and causes oscillation. This filter keeps points
    in the target-facing sector or near the path corridor, then voxel-downsamples.
    """
    if not obstacles:
        return []

    tx = target[0] - pos[0]
    ty = target[1] - pos[1]
    target_norm = math.hypot(tx, ty)
    if target_norm < 1e-6:
        ux, uy = 1.0, 0.0
    else:
        ux, uy = tx / target_norm, ty / target_norm

    half_sector = max(1.0, front_sector_deg) * 0.5
    min_d = max(0.0, min_distance)
    max_d = max(max_distance, min_d + 0.1)
    corridor = max(0.1, corridor_width)
    voxel = max(0.05, voxel_resolution)

    candidates: List[Tuple[float, Tuple[float, float]]] = []
    for ox, oy in obstacles:
        rx = ox - pos[0]
        ry = oy - pos[1]
        d = math.hypot(rx, ry)
        if d < min_d or d > max_d:
            continue

        forward = rx * ux + ry * uy
        lateral = abs(rx * uy - ry * ux)
        if forward < -1.0:
            continue

        cosang = clamp(forward / max(d, 1e-6), -1.0, 1.0)
        angle = math.degrees(math.acos(cosang))

        if angle > half_sector and lateral > corridor:
            continue
        if lateral > corridor and forward > target_norm + corridor:
            continue

        candidates.append((d, (ox, oy)))

    candidates.sort(key=lambda item: item[0])
    seen = set()
    filtered: List[Tuple[float, float]] = []
    for _, point in candidates:
        key = (round(point[0] / voxel), round(point[1] / voxel))
        if key in seen:
            continue
        seen.add(key)
        filtered.append(point)
        if max_points > 0 and len(filtered) >= max_points:
            break
    return filtered


# =============================================================================
# 4. ROS2 node
# =============================================================================


class TeamPotentialFieldNode(Node):
    def __init__(self) -> None:
        super().__init__("tank_team_potential_field_node")

        default_map = ""
        try:
            default_map = os.path.join(get_package_share_directory("rviz_visualization"), "map", "finalmap.map")
        except Exception:
            pass

        # ROS2 parameter mirror of global variables.
        self.declare_parameter("target_pose_topic", TARGET_POSE_TOPIC)
        self.declare_parameter("fallback_goal_topic", FALLBACK_GOAL_TOPIC)
        self.declare_parameter("lidar_points_topic", "/tank/sensor/lidar/detected_points_map")
        self.declare_parameter("hz", APF_HZ)
        self.declare_parameter("k_att", K_ATTRACTIVE)
        self.declare_parameter("k_rep", K_REPULSIVE)
        self.declare_parameter("influence_radius", OBSTACLE_INFLUENCE_RADIUS)
        self.declare_parameter("max_attractive_norm", MAX_ATTRACTIVE_FORCE)
        self.declare_parameter("max_repulsive_norm", MAX_REPULSIVE_FORCE)
        self.declare_parameter("max_result_norm", MAX_RESULT_FORCE)
        self.declare_parameter("local_target_distance", LOCAL_TARGET_DISTANCE)
        self.declare_parameter("repulsive_eps", REPULSIVE_EPS)
        self.declare_parameter("passthrough_when_clear", PASSTHROUGH_WHEN_CLEAR)
        self.declare_parameter("use_tangential_force", USE_TANGENTIAL_FORCE)
        self.declare_parameter("tangent_gain_scale", TANGENTIAL_GAIN_SCALE)
        self.declare_parameter("min_obstacle_distance", MIN_OBSTACLE_DISTANCE)
        self.declare_parameter("max_obstacle_distance", MAX_OBSTACLE_DISTANCE)
        self.declare_parameter("front_sector_deg", FRONT_SECTOR_DEG)
        self.declare_parameter("path_corridor_width", PATH_CORRIDOR_WIDTH)
        self.declare_parameter("obstacle_voxel_resolution", OBSTACLE_VOXEL_RESOLUTION)
        self.declare_parameter("max_obstacle_points", MAX_OBSTACLE_POINTS)
        self.declare_parameter("use_discovered_objects", USE_DISCOVERED_OBJECTS)
        self.declare_parameter("discovered_objects_topic", DISCOVERED_OBJECTS_TOPIC)
        self.declare_parameter("use_threat_avoidance", USE_THREAT_AVOIDANCE)
        self.declare_parameter("threat_map_file", default_map)
        self.declare_parameter("threat_radius", THREAT_RADIUS)
        self.declare_parameter("k_threat_rep", K_THREAT_REPULSIVE)
        self.declare_parameter("k_theta", ANGULAR_GAIN_K_THETA)
        self.declare_parameter("angle_epsilon_deg", ANGLE_EPSILON_DEG)
        self.declare_parameter("linear_speed_gain", LINEAR_SPEED_GAIN)
        self.declare_parameter("max_desired_speed", MAX_DESIRED_SPEED)
        self.declare_parameter("max_desired_omega", MAX_DESIRED_ANGULAR_SPEED)
        self.declare_parameter("motion_strategy", MOTION_STRATEGY)
        self.declare_parameter("marker_scale", MARKER_SCALE)
        self.declare_parameter("use_lidar_clusters", USE_LIDAR_CLUSTERS)
        self.declare_parameter("lidar_clusters_topic", LIDAR_CLUSTERS_TOPIC)
        self.declare_parameter("cluster_obstacle_min_count", CLUSTER_OBSTACLE_MIN_COUNT)
        self.declare_parameter("apf_weights_file", APF_WEIGHTS_FILE)
        self.declare_parameter("apf_weight_profile", APF_WEIGHT_PROFILE)

        self.target_pose_topic = str(self.get_parameter("target_pose_topic").value)
        self.fallback_goal_topic = str(self.get_parameter("fallback_goal_topic").value)
        self.lidar_points_topic = str(self.get_parameter("lidar_points_topic").value)
        self.hz = float(self.get_parameter("hz").value)
        self.k_att = float(self.get_parameter("k_att").value)
        self.k_rep = float(self.get_parameter("k_rep").value)
        self.influence_radius = float(self.get_parameter("influence_radius").value)
        self.max_attractive_norm = float(self.get_parameter("max_attractive_norm").value)
        self.max_repulsive_norm = float(self.get_parameter("max_repulsive_norm").value)
        self.max_result_norm = float(self.get_parameter("max_result_norm").value)
        self.local_target_distance = float(self.get_parameter("local_target_distance").value)
        self.repulsive_eps = float(self.get_parameter("repulsive_eps").value)
        self.passthrough_when_clear = bool(self.get_parameter("passthrough_when_clear").value)
        self.use_tangential_force = bool(self.get_parameter("use_tangential_force").value)
        self.tangent_gain_scale = float(self.get_parameter("tangent_gain_scale").value)
        self.min_obstacle_distance = float(self.get_parameter("min_obstacle_distance").value)
        self.max_obstacle_distance = float(self.get_parameter("max_obstacle_distance").value)
        self.front_sector_deg = float(self.get_parameter("front_sector_deg").value)
        self.path_corridor_width = float(self.get_parameter("path_corridor_width").value)
        self.obstacle_voxel_resolution = float(self.get_parameter("obstacle_voxel_resolution").value)
        self.max_obstacle_points = int(self.get_parameter("max_obstacle_points").value)
        self.use_discovered_objects = bool(self.get_parameter("use_discovered_objects").value)
        self.discovered_objects_topic = str(self.get_parameter("discovered_objects_topic").value)
        self.use_threat_avoidance = bool(self.get_parameter("use_threat_avoidance").value)
        self.threat_radius = float(self.get_parameter("threat_radius").value)
        self.k_threat_rep = float(self.get_parameter("k_threat_rep").value)
        self.k_theta = float(self.get_parameter("k_theta").value)
        self.angle_epsilon_deg = float(self.get_parameter("angle_epsilon_deg").value)
        self.linear_speed_gain = float(self.get_parameter("linear_speed_gain").value)
        self.max_desired_speed = float(self.get_parameter("max_desired_speed").value)
        self.max_desired_omega = float(self.get_parameter("max_desired_omega").value)
        self.motion_strategy = str(self.get_parameter("motion_strategy").value)
        self.marker_scale = float(self.get_parameter("marker_scale").value)
        self.use_lidar_clusters = bool(self.get_parameter("use_lidar_clusters").value)
        self.lidar_clusters_topic = str(self.get_parameter("lidar_clusters_topic").value)
        self.cluster_obstacle_min_count = int(self.get_parameter("cluster_obstacle_min_count").value)
        self.apf_weights_file = str(self.get_parameter("apf_weights_file").value)
        self.apf_weight_profile_name = str(self.get_parameter("apf_weight_profile").value)
        self.apf_weight_profile = load_apf_weight_profile(self.apf_weights_file, self.apf_weight_profile_name)

        threat_map_file = str(self.get_parameter("threat_map_file").value)
        self.threats = parse_threats_from_map(threat_map_file) if self.use_threat_avoidance and threat_map_file else []

        self.player_pos: Optional[Tuple[float, float]] = None
        self.player_yaw_rad: Optional[float] = None
        self.target_pos: Optional[Tuple[float, float]] = None
        self.fallback_goal: Optional[Tuple[float, float]] = None
        self.raw_obstacles: List[Tuple[float, float]] = []
        self.discovered_obstacles: List[Tuple[float, float]] = []
        self.cluster_obstacles: List[Tuple[float, float]] = []
        self.obstacles: List[Tuple[float, float]] = []
        self.gt_obstacles: List[Dict[str, float]] = []

        self.pub_rep = self.create_publisher(Vector3Stamped, "/tank/potential/repulsive_vector", 10)
        self.pub_att = self.create_publisher(Vector3Stamped, "/tank/potential/attractive_vector", 10)
        self.pub_tan = self.create_publisher(Vector3Stamped, "/tank/potential/tangential_vector", 10)
        self.pub_threat = self.create_publisher(Vector3Stamped, "/tank/potential/threat_vector", 10)
        self.pub_res = self.create_publisher(Vector3Stamped, "/tank/potential/result_vector", 10)
        self.pub_local_target = self.create_publisher(PoseStamped, "/tank/local_target/pose", 10)
        self.pub_desired_motion = self.create_publisher(String, "/tank/potential/desired_motion", 10)
        self.pub_status = self.create_publisher(String, "/tank/potential/status", 10)
        self.pub_markers = self.create_publisher(MarkerArray, "/tank/rviz/potential_field_markers", 10)

        self.create_subscription(PoseStamped, PLAYER_POSE_TOPIC, self.player_cb, 10)
        self.create_subscription(PoseStamped, self.target_pose_topic, self.target_cb, 10)
        if self.fallback_goal_topic != self.target_pose_topic:
            self.create_subscription(PoseStamped, self.fallback_goal_topic, self.fallback_goal_cb, 10)
        self.create_subscription(PointCloud2, self.lidar_points_topic, self.lidar_cb, 10)
        self.create_subscription(String, "/tank/map/obstacles", self.gt_obstacles_cb, 10)
        if self.use_lidar_clusters:
            self.create_subscription(String, self.lidar_clusters_topic, self.lidar_clusters_cb, 10)
        if self.use_discovered_objects:
            self.create_subscription(String, self.discovered_objects_topic, self.discovered_cb, 10)

        self.create_timer(1.0 / max(self.hz, 1.0), self.timer_cb)
        self.get_logger().info(
            "Lecture-style APF initialized: "
            f"target_topic={self.target_pose_topic}, kA={self.k_att}, kR={self.k_rep}, "
            f"g*={self.influence_radius}, tangent={self.use_tangential_force}, "
            f"threats={len(self.threats)}, discovered={self.use_discovered_objects}, "
            f"strategy={self.motion_strategy}, lidar_pc2={self.lidar_points_topic}, "
            f"clusters={self.use_lidar_clusters}, profile={self.apf_weight_profile_name}"
        )

    # -------------------------------------------------------------------------
    # Callbacks
    # -------------------------------------------------------------------------

    def player_cb(self, msg: PoseStamped) -> None:
        self.player_pos = (float(msg.pose.position.x), float(msg.pose.position.y))
        self.player_yaw_rad = quaternion_msg_to_yaw_rad(msg.pose.orientation)

    def target_cb(self, msg: PoseStamped) -> None:
        self.target_pos = (float(msg.pose.position.x), float(msg.pose.position.y))

    def fallback_goal_cb(self, msg: PoseStamped) -> None:
        self.fallback_goal = (float(msg.pose.position.x), float(msg.pose.position.y))

    def lidar_cb(self, msg: PointCloud2) -> None:
        try:
            points = pointcloud2_to_xyz_array(msg)
            if points.size == 0:
                self.raw_obstacles = []
                return
            # APF uses only map-plane x/y.  Filtering/voxel limiting is done in timer_cb.
            self.raw_obstacles = [(float(x), float(y)) for x, y in points[:, :2]]
        except Exception as exc:
            self.get_logger().warn(f"failed to parse lidar APF PointCloud2: {exc}")

    def lidar_clusters_cb(self, msg: String) -> None:
        try:
            self.cluster_obstacles = parse_lidar_clusters_payload(json.loads(msg.data), self.cluster_obstacle_min_count)
        except Exception as exc:
            self.get_logger().warn(f"failed to parse lidar APF clusters: {exc}")

    def discovered_cb(self, msg: String) -> None:
        try:
            self.discovered_obstacles = parse_discovered_objects_payload(json.loads(msg.data))
        except Exception as exc:
            self.get_logger().warn(f"failed to parse discovered APF objects: {exc}")

    def gt_obstacles_cb(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
            self.gt_obstacles = parse_obstacles_payload(payload)
        except Exception as exc:
            self.get_logger().warn(f"failed to parse gt obstacles for APF: {exc}")

    # -------------------------------------------------------------------------
    # Publishing helpers
    # -------------------------------------------------------------------------

    def publish_vec(self, pub: Any, vec: Tuple[float, float]) -> None:
        msg = Vector3Stamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = MAP_FRAME
        msg.vector.x = float(vec[0])
        msg.vector.y = float(vec[1])
        msg.vector.z = 0.0
        pub.publish(msg)

    def arrow_marker(self, marker_id: int, vec: Tuple[float, float], rgba: Tuple[float, float, float, float]) -> Marker:
        m = Marker()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = MAP_FRAME
        m.ns = "potential_field_vectors"
        m.id = marker_id
        m.type = Marker.ARROW
        m.action = Marker.ADD
        m.scale.x = 0.35
        m.scale.y = 0.75
        m.scale.z = 0.75
        m.color.r, m.color.g, m.color.b, m.color.a = rgba
        sx = self.player_pos[0] if self.player_pos else 0.0
        sy = self.player_pos[1] if self.player_pos else 0.0
        start = Point(x=float(sx), y=float(sy), z=1.8)
        end = Point(x=float(sx + vec[0] * self.marker_scale), y=float(sy + vec[1] * self.marker_scale), z=1.8)
        m.points = [start, end]
        return m

    def text_marker(self, marker_id: int, text: str, offset_y: float) -> Marker:
        m = Marker()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = MAP_FRAME
        m.ns = "potential_field_text"
        m.id = marker_id
        m.type = Marker.TEXT_VIEW_FACING
        m.action = Marker.ADD
        m.pose.position.x = float(self.player_pos[0] if self.player_pos else 0.0)
        m.pose.position.y = float((self.player_pos[1] if self.player_pos else 0.0) + offset_y)
        m.pose.position.z = 5.0
        m.pose.orientation.w = 1.0
        m.scale.z = 2.0
        m.color.r = 1.0
        m.color.g = 1.0
        m.color.b = 1.0
        m.color.a = 0.9
        m.text = text
        return m

    def publish_markers(self, forces: ForceBreakdown) -> None:
        arr = MarkerArray()
        arr.markers.append(self.arrow_marker(0, forces.attractive, (0.0, 1.0, 0.0, 0.9)))
        arr.markers.append(self.arrow_marker(1, forces.repulsive, (1.0, 0.0, 0.0, 0.9)))
        arr.markers.append(self.arrow_marker(2, forces.tangential, (1.0, 0.9, 0.0, 0.9)))
        arr.markers.append(self.arrow_marker(3, forces.threat, (1.0, 0.4, 0.0, 0.9)))
        arr.markers.append(self.arrow_marker(4, forces.result, (0.1, 0.4, 1.0, 0.95)))
        arr.markers.append(
            self.text_marker(
                10,
                f"APF | FA {vector_norm(forces.attractive):.2f} | FR {vector_norm(forces.repulsive):.2f} | "
                f"FT {vector_norm(forces.tangential):.2f} | F {vector_norm(forces.result):.2f}",
                0.0,
            )
        )
        self.pub_markers.publish(arr)

    # -------------------------------------------------------------------------
    # Main APF cycle
    # -------------------------------------------------------------------------

    def timer_cb(self) -> None:
        target = self.target_pos or self.fallback_goal
        if self.player_pos is None or target is None:
            return

        pos = self.player_pos
        combined_obstacles = list(self.raw_obstacles)
        if self.use_lidar_clusters:
            combined_obstacles.extend(self.cluster_obstacles)
        if self.use_discovered_objects:
            combined_obstacles.extend(self.discovered_obstacles)

        self.obstacles = filter_obstacles_for_apf(
            combined_obstacles,
            pos,
            target,
            self.min_obstacle_distance,
            min(self.max_obstacle_distance, self.influence_radius),
            self.front_sector_deg,
            self.path_corridor_width,
            self.obstacle_voxel_resolution,
            self.max_obstacle_points,
        )

        forces = calc_resultant_force(
            pos=pos,
            target=target,
            obstacles=self.obstacles,
            threats=self.threats,
            k_att=self.k_att,
            k_rep=self.k_rep,
            g_star=self.influence_radius,
            max_att=self.max_attractive_norm,
            max_rep=self.max_repulsive_norm,
            max_result=self.max_result_norm,
            use_tangent=self.use_tangential_force,
            tangent_gain_scale=self.tangent_gain_scale,
            use_threats=self.use_threat_avoidance,
            threat_radius=self.threat_radius,
            k_threat=self.k_threat_rep,
            gt_obstacles=self.gt_obstacles,
        )

        clear = (
            vector_norm(forces.repulsive) < self.repulsive_eps
            and vector_norm(forces.tangential) < self.repulsive_eps
            and vector_norm(forces.threat) < self.repulsive_eps
        )

        motion = compute_desired_motion(
            self.player_yaw_rad,
            forces.result,
            self.k_theta,
            self.angle_epsilon_deg,
            self.linear_speed_gain,
            self.max_desired_speed,
            self.max_desired_omega,
            self.motion_strategy,
        )

        if self.passthrough_when_clear and clear:
            local_target = target
            source = "passthrough_lookahead"
        else:
            n = vector_norm(forces.result)
            if n < 1e-9:
                local_target = target
            else:
                local_target = (
                    pos[0] + forces.result[0] / n * self.local_target_distance,
                    pos[1] + forces.result[1] / n * self.local_target_distance,
                )
            source = "lecture_apf_result"

        # Publish vector topics.
        self.publish_vec(self.pub_att, forces.attractive)
        self.publish_vec(self.pub_rep, forces.repulsive)
        self.publish_vec(self.pub_tan, forces.tangential)
        self.publish_vec(self.pub_threat, forces.threat)
        self.publish_vec(self.pub_res, forces.result)
        self.publish_markers(forces)

        # Publish local target pose. Orientation encodes desired force direction.
        theta_d = float(motion["theta_desired_rad"])
        qx, qy, qz, qw = yaw_rad_to_quaternion_z(theta_d)
        lt = PoseStamped()
        lt.header.stamp = self.get_clock().now().to_msg()
        lt.header.frame_id = MAP_FRAME
        lt.pose.position.x = float(local_target[0])
        lt.pose.position.y = float(local_target[1])
        lt.pose.position.z = 0.0
        lt.pose.orientation.x = qx
        lt.pose.orientation.y = qy
        lt.pose.orientation.z = qz
        lt.pose.orientation.w = qw
        self.pub_local_target.publish(lt)

        desired_msg = String()
        desired_msg.data = json.dumps(motion, ensure_ascii=False)
        self.pub_desired_motion.publish(desired_msg)

        status = {
            "ok": True,
            "model": "lecture_style_apf",
            "source": source,
            "formula": {
                "attractive": "U_A=0.5*k_A*d^2, F_A=k_A*(r_D-r_B)",
                "repulsive": "U_R=0.5*k_R*(1/g-1/g*)^2, F_R=k_R*(1/g-1/g*)/g^3*(r_B-r_O)",
                "heading": "theta_D=atan2(F_y,F_x), theta_dot=k_theta*wrap(theta_D-theta)",
            },
            "position": {"x": pos[0], "y": pos[1]},
            "target": {"x": target[0], "y": target[1]},
            "local_target": {"x": local_target[0], "y": local_target[1]},
            "obstacle_points": len(self.obstacles),
            "raw_obstacle_points": len(self.raw_obstacles),
            "discovered_obstacle_points": len(self.discovered_obstacles),
            "cluster_obstacle_points": len(self.cluster_obstacles),
            "use_lidar_clusters": self.use_lidar_clusters,
            "threats": len(self.threats),
            "apf_weight_profile": self.apf_weight_profile_name,
            "apf_weight_profile_loaded": bool(self.apf_weight_profile),
            "apf_weight_profile_config": self.apf_weight_profile,
            "clear": clear,
            "parameters": {
                "k_att": self.k_att,
                "k_rep": self.k_rep,
                "g_star": self.influence_radius,
                "k_theta": self.k_theta,
                "angle_epsilon_deg": self.angle_epsilon_deg,
                "use_tangential_force": self.use_tangential_force,
                "tangent_gain_scale": self.tangent_gain_scale,
                "local_target_distance": self.local_target_distance,
                "front_sector_deg": self.front_sector_deg,
                "path_corridor_width": self.path_corridor_width,
            },
            "potential": {
                "u_att": forces.attractive_potential,
                "u_rep": forces.repulsive_potential,
                "u_threat": forces.threat_potential,
            },
            "force": {
                "att": {"x": forces.attractive[0], "y": forces.attractive[1], "norm": vector_norm(forces.attractive)},
                "rep": {"x": forces.repulsive[0], "y": forces.repulsive[1], "norm": vector_norm(forces.repulsive)},
                "tan": {"x": forces.tangential[0], "y": forces.tangential[1], "norm": vector_norm(forces.tangential)},
                "threat": {"x": forces.threat[0], "y": forces.threat[1], "norm": vector_norm(forces.threat)},
                "result": {"x": forces.result[0], "y": forces.result[1], "norm": vector_norm(forces.result)},
            },
            "desired_motion": motion,
        }
        msg = String()
        msg.data = json.dumps(status, ensure_ascii=False)
        self.pub_status.publish(msg)


def main(args: Optional[List[str]] = None) -> None:
    rclpy.init(args=args)
    node = TeamPotentialFieldNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
