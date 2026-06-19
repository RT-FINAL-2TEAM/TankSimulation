# -*- coding: utf-8 -*-
"""LiDAR-카메라 투영 공용 유틸리티.

좌표 기준은 Tank Challenge / Unity raw 월드 좌표다:
  raw.x: 오른쪽, raw.y: 위, raw.z: 앞
프로젝트 나머지에서 쓰는 ROS map 좌표 규약:
  map.x = raw.x, map.y = raw.z, map.z = raw.y

이 모듈은 의도적으로 다음 둘이 함께 공유한다:
- tank_visual_perception/lidar_camera_overlay_node.py  (캘리브레이션 시각화)
- path_planning/local_path_node.py                     (실제 YOLO-LiDAR 융합)
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np


DEFAULT_PROJECTION_PARAMS: Dict[str, float] = {
    # 팀에서 진행한 수동 캘리브레이션 결과.
    # camera_pos = lidarOrigin + R_cam_to_world @ [tx, ty, tz]
    "tx": 0.28,
    "ty": 0.02,
    "tz": 11.80,
    "yaw_offset": -0.9,
    "pitch_offset": -0.9,
    "roll_offset": -0.3,
    "hfov": 86.0,
    "vfov": 60.2,
}


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def deg2rad(deg: float) -> float:
    return float(deg) * math.pi / 180.0


def vec3_from_dict(d: Dict[str, Any]) -> np.ndarray:
    return np.array(
        [to_float(d.get("x")), to_float(d.get("y")), to_float(d.get("z"))],
        dtype=np.float64,
    )


def raw_to_map_xyz(raw: Dict[str, Any]) -> Dict[str, float]:
    return {
        "x": to_float(raw.get("x")),
        "y": to_float(raw.get("z")),
        "z": to_float(raw.get("y")),
    }


def map_to_raw_xyz(map_pos: Dict[str, Any]) -> Dict[str, float]:
    return {
        "x": to_float(map_pos.get("x")),
        "y": to_float(map_pos.get("z")),
        "z": to_float(map_pos.get("y")),
    }


def rotation_matrix_yaw_pitch_roll(yaw_deg: float, pitch_deg: float, roll_deg: float = 0.0) -> np.ndarray:
    """Unity 유사 raw 좌표 규약에서의 R_cam_to_world를 반환한다."""
    yaw = deg2rad(yaw_deg)
    pitch = deg2rad(pitch_deg)
    roll = deg2rad(roll_deg)

    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)

    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]], dtype=np.float64)
    rz = np.array([[cr, -sr, 0.0], [sr, cr, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return ry @ rx @ rz


def get_turret_angle(info: Dict[str, Any]) -> Tuple[float, float]:
    return to_float(info.get("playerTurretX")), to_float(info.get("playerTurretY"))


def extract_info_payload(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """/tank/api/info/raw, /tank/api/info/compact 래퍼, 또는 raw /info dict를 받는다.

    투영에는 lidarOrigin과 포탑 각도 같은 카메라 포즈 입력만 있으면 된다.
    compact /info 토픽은 무거운 lidarPoints 배열을 의도적으로 제거하므로, 여기서
    lidarPoints를 요구하면 TANK_SAVE_FULL_INFO=false일 때 local_path_node/
    lidar_camera_overlay_node가 latest_info를 잃게 된다.
    """
    if not isinstance(payload, dict):
        return None
    if isinstance(payload.get("data"), dict):
        payload = payload["data"]
    if not isinstance(payload, dict):
        return None

    # compute_camera_pose()에는 lidarOrigin만 있으면 충분하다. playerTurretX/Y는
    # get_turret_angle()이 누락값을 이미 0.0으로 기본 처리하므로 선택사항이다.
    if "lidarOrigin" not in payload:
        return None
    return payload


def compute_camera_pose(info: Dict[str, Any], params: Dict[str, float]) -> Tuple[np.ndarray, float, float, float]:
    lidar_origin = vec3_from_dict(info["lidarOrigin"])
    turret_yaw, turret_pitch = get_turret_angle(info)

    camera_yaw = turret_yaw + to_float(params.get("yaw_offset"))
    camera_pitch = turret_pitch + to_float(params.get("pitch_offset"))
    camera_roll = to_float(params.get("roll_offset"))

    r_cam_to_world = rotation_matrix_yaw_pitch_roll(camera_yaw, camera_pitch, camera_roll)
    offset = np.array(
        [to_float(params.get("tx")), to_float(params.get("ty")), to_float(params.get("tz"))],
        dtype=np.float64,
    )
    camera_pos = lidar_origin + r_cam_to_world @ offset
    return camera_pos, camera_yaw, camera_pitch, camera_roll


def project_point(
    point_world_raw: np.ndarray,
    camera_pos_world_raw: np.ndarray,
    camera_yaw_deg: float,
    camera_pitch_deg: float,
    camera_roll_deg: float,
    image_w: int,
    image_h: int,
    params: Dict[str, float],
) -> Optional[Tuple[int, int, float]]:
    hfov = max(1e-3, to_float(params.get("hfov"), DEFAULT_PROJECTION_PARAMS["hfov"]))
    vfov = max(1e-3, to_float(params.get("vfov"), DEFAULT_PROJECTION_PARAMS["vfov"]))
    fx = image_w / (2.0 * math.tan(deg2rad(hfov) / 2.0))
    fy = image_h / (2.0 * math.tan(deg2rad(vfov) / 2.0))
    cx = image_w / 2.0
    cy = image_h / 2.0

    r_cam_to_world = rotation_matrix_yaw_pitch_roll(camera_yaw_deg, camera_pitch_deg, camera_roll_deg)
    point_cam = r_cam_to_world.T @ (point_world_raw - camera_pos_world_raw)
    x_cam, y_cam, z_cam = point_cam
    if z_cam <= 0.01:
        return None

    u = fx * x_cam / z_cam + cx
    v = cy - fy * y_cam / z_cam
    if not np.isfinite(u) or not np.isfinite(v):
        return None
    return int(round(u)), int(round(v)), float(z_cam)


def expand_bbox(bbox: Iterable[float], margin_px: float, image_w: float, image_h: float) -> Optional[Tuple[float, float, float, float]]:
    try:
        x1, y1, x2, y2 = [float(v) for v in list(bbox)[:4]]
    except Exception:
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    m = max(0.0, float(margin_px))
    return (
        max(0.0, x1 - m),
        max(0.0, y1 - m),
        min(float(image_w) - 1.0, x2 + m),
        min(float(image_h) - 1.0, y2 + m),
    )


def point_inside_bbox(u: float, v: float, bbox: Iterable[float], margin_px: float = 0.0, image_w: float = 1e9, image_h: float = 1e9) -> bool:
    b = expand_bbox(bbox, margin_px, image_w, image_h)
    if b is None:
        return False
    x1, y1, x2, y2 = b
    return x1 <= float(u) <= x2 and y1 <= float(v) <= y2


def lidar_point_raw_position(point: Dict[str, Any]) -> Optional[Dict[str, float]]:
    """LiDAR 포인트 payload에서 raw 월드 xyz를 반환한다."""
    if not isinstance(point, dict):
        return None
    pos_raw = point.get("position_raw")
    if isinstance(pos_raw, dict):
        return {"x": to_float(pos_raw.get("x")), "y": to_float(pos_raw.get("y")), "z": to_float(pos_raw.get("z"))}
    pos = point.get("position")
    if isinstance(pos, dict):
        return {"x": to_float(pos.get("x")), "y": to_float(pos.get("y")), "z": to_float(pos.get("z"))}
    pos_map = point.get("position_map")
    if isinstance(pos_map, dict):
        return map_to_raw_xyz(pos_map)
    return None


def lidar_point_map_position(point: Dict[str, Any]) -> Optional[Dict[str, float]]:
    """LiDAR 포인트 payload에서 map xyz를 반환한다."""
    if not isinstance(point, dict):
        return None
    pos_map = point.get("position_map")
    if isinstance(pos_map, dict):
        return {"x": to_float(pos_map.get("x")), "y": to_float(pos_map.get("y")), "z": to_float(pos_map.get("z"))}
    raw = lidar_point_raw_position(point)
    if raw is None:
        return None
    return raw_to_map_xyz(raw)
