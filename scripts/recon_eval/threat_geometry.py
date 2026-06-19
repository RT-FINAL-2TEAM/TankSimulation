# -*- coding: utf-8 -*-
"""정찰 위협 기하 — finalmap.map(GT) 파싱 + FOV/LoS 위협 판정.

이 모듈은 `src/potential/potential/potential_field_node.py`의 검증된 위협 기하 로직
(`is_threat_active`/`check_los`/`segment_intersect_bbox`/`parse_threats_from_map`/
`obstacle_to_bbox`)을 **ROS 의존 없이** 미러링한 것이다. 정찰 보고서 생성기
(`scripts/generate_recon_report.py`)가 시뮬레이터/ROS 런타임 없이 GT 맵과 전차 궤적만으로
노출시간·발각횟수를 사후 계산하기 위해 쓴다.

원본과 동일하게 맞춰야 하는 상수/규칙:
- THREAT_TYPES = ("House002", "Tank001")           # potential/config.py 기본값
- House002: 반경 25m + 시야각(FOV) ±30° + 시선차단(LoS)
- Tank001 : 반경 20m + 시선차단(LoS)
- 그 외(type 미지정): 반경 25m
- PREFAB_HALF_SIZES                                  # path_planning/config.py 사본

좌표계: map (x, z) 2D. 맵 파일의 position.z 가 map의 진행축(=ROS map.y)에 해당한다.
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Optional, Tuple

# 위협으로 취급할 prefab 접두사 (potential/config.py: TANK_APF_THREAT_TYPES 기본값)
THREAT_TYPES: Tuple[str, ...] = ("House002", "Tank001")

# prefab 이름 부분일치 → (half_width, half_length). path_planning/config.py 사본.
PREFAB_HALF_SIZES: Dict[str, Tuple[float, float]] = {
    "Human": (0.5, 0.5),
    "Tree": (1.0, 1.0),
    "Rock": (1.5, 1.5),
    "Tank": (2.0, 4.0),
    "House": (4.0, 4.0),
    "wall001x5": (8.0, 1.0),
    "wall002x5": (8.0, 1.0),
    "Wall001": (3.0, 1.0),
    "Wall002": (3.0, 1.0),
}

# 위협 타입별 탐지 반경/시야각 (potential_field_node.is_threat_active 와 동일)
HOUSE_RADIUS_M = 25.0
HOUSE_FOV_HALF_DEG = 30.0
TANK_RADIUS_M = 20.0
DEFAULT_RADIUS_M = 25.0


# --------------------------------------------------------------------------- #
# 각도 / 쿼터니언 헬퍼
# --------------------------------------------------------------------------- #

def normalize_angle_deg(angle: float) -> float:
    """각도를 [-180, 180]로 래핑."""
    return math.degrees(math.atan2(math.sin(math.radians(angle)), math.cos(math.radians(angle))))


def simulator_quaternion_to_yaw_deg(rot: Dict[str, Any]) -> float:
    """.map 파일의 Unity 식 쿼터니언에서 Y축 yaw(도)를 추출."""
    qx = float(rot.get("x", 0.0))
    qy = float(rot.get("y", 0.0))
    qz = float(rot.get("z", 0.0))
    qw = float(rot.get("w", 1.0))
    siny_cosp = 2.0 * (qw * qy + qz * qx)
    cosy_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    return math.degrees(math.atan2(siny_cosp, cosy_cosp))


# --------------------------------------------------------------------------- #
# 시선 차단(LoS) / 위협 활성 판정 — potential_field_node 미러
# --------------------------------------------------------------------------- #

def segment_intersect_bbox(px: float, pz: float, qx: float, qz: float, bbox: Dict[str, float]) -> bool:
    """선분 (px,pz)-(qx,qz) 가 축정렬 bbox 와 교차하는지(slab 방식)."""
    xmin, xmax = bbox.get("x_min", 0.0), bbox.get("x_max", 0.0)
    zmin, zmax = bbox.get("z_min", 0.0), bbox.get("z_max", 0.0)
    if min(px, qx) > xmax or max(px, qx) < xmin:
        return False
    if min(pz, qz) > zmax or max(pz, qz) < zmin:
        return False

    t0, t1 = 0.0, 1.0
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


def check_los(tank_x: float, tank_z: float, threat_x: float, threat_z: float,
              gt_obstacles: List[Dict[str, float]]) -> bool:
    """전차→위협 직선이 GT 장애물에 가리지 않으면 True(시선 확보)."""
    for obs in gt_obstacles:
        xmin, xmax = obs.get("x_min", 0.0), obs.get("x_max", 0.0)
        zmin, zmax = obs.get("z_min", 0.0), obs.get("z_max", 0.0)
        # 위협 자신을 감싸는 bbox는 무시(자기 차폐 방지)
        if xmin <= threat_x <= xmax and zmin <= threat_z <= zmax:
            continue
        if segment_intersect_bbox(tank_x, tank_z, threat_x, threat_z, obs):
            return False
    return True


def is_threat_active(pos: Tuple[float, float], threat: Dict[str, Any],
                     gt_obstacles: List[Dict[str, float]]) -> bool:
    """전차 위치 pos 가 threat 의 탐지 영역(반경+FOV+LoS) 안에 들었는지.

    potential_field_node.is_threat_active 와 동일 규칙:
      - House002: 25m & |yaw_diff|<=30° & LoS
      - Tank001 : 20m & LoS
      - 기타     : 25m (반경만)
    """
    tx, tz = pos
    threat_x = float(threat.get("x", 0.0))
    threat_z = float(threat.get("z", 0.0))
    dx = tx - threat_x
    dz = tz - threat_z
    dist = math.hypot(dx, dz)

    t_type = str(threat.get("type", "unknown"))
    prefab_name = str(threat.get("prefabName", ""))

    if t_type == "House002" or prefab_name.startswith("House002"):
        if dist > HOUSE_RADIUS_M:
            return False
        target_yaw = math.degrees(math.atan2(dx, dz))
        yaw_diff = abs(normalize_angle_deg(target_yaw - float(threat.get("yaw", 0.0))))
        if yaw_diff > HOUSE_FOV_HALF_DEG:
            return False
        return check_los(tx, tz, threat_x, threat_z, gt_obstacles)
    if t_type == "Tank001" or prefab_name.startswith("Tank001"):
        if dist > TANK_RADIUS_M:
            return False
        return check_los(tx, tz, threat_x, threat_z, gt_obstacles)
    return dist <= DEFAULT_RADIUS_M


def threat_radius(threat: Dict[str, Any]) -> float:
    """위협 타입별 탐지 반경(보고서 표기/근접 dwell 폴백용)."""
    t_type = str(threat.get("type", "unknown"))
    prefab_name = str(threat.get("prefabName", ""))
    if t_type == "House002" or prefab_name.startswith("House002"):
        return HOUSE_RADIUS_M
    if t_type == "Tank001" or prefab_name.startswith("Tank001"):
        return TANK_RADIUS_M
    return DEFAULT_RADIUS_M


# --------------------------------------------------------------------------- #
# 맵(GT) 파싱 — parse_threats_from_map / obstacle_to_bbox 미러
# --------------------------------------------------------------------------- #

def prefab_half_size(name: str) -> Tuple[float, float]:
    lname = str(name).lower()
    for key, value in PREFAB_HALF_SIZES.items():
        if key.lower() in lname:
            return value
    return 1.0, 1.0


def obstacle_to_bbox(obs: Dict[str, Any]) -> Optional[Dict[str, float]]:
    """맵 obstacle 항목 → 축정렬 bbox(x_min/x_max/z_min/z_max)."""
    if all(k in obs for k in ("x_min", "x_max", "z_min", "z_max")):
        try:
            return {
                "x_min": float(obs["x_min"]), "x_max": float(obs["x_max"]),
                "z_min": float(obs["z_min"]), "z_max": float(obs["z_max"]),
            }
        except (TypeError, ValueError):
            return None
    pos = obs.get("position") if isinstance(obs.get("position"), dict) else None
    if pos is None:
        return None
    try:
        x = float(pos.get("x", 0.0))
        z = float(pos.get("z", 0.0))
    except (TypeError, ValueError):
        return None
    hw, hl = prefab_half_size(str(obs.get("prefabName", "")))
    return {"x_min": x - hw, "x_max": x + hw, "z_min": z - hl, "z_max": z + hl}


def parse_threats_from_map(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """맵 dict → 위협 리스트 [{type, x, z, yaw, prefabName}]."""
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
        threats.append({
            "type": threat_type,
            "x": float(pos.get("x", 0.0)),
            "z": float(pos.get("z", 0.0)),
            "yaw": simulator_quaternion_to_yaw_deg(rot),
            "prefabName": prefab_name,
        })
    return threats


def parse_gt_objects_from_map(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """맵 dict → GT 객체 리스트 [{prefabName, x, z, bbox, is_threat}].

    센서 정확도(GT vs 탐지) 비교용. 위협(House002/Tank001) 포함 전체 정적 객체.
    """
    objs: List[Dict[str, Any]] = []
    for obs in data.get("obstacles", []):
        if not isinstance(obs, dict):
            continue
        bbox = obstacle_to_bbox(obs)
        if bbox is None:
            continue
        prefab_name = str(obs.get("prefabName", ""))
        cx = 0.5 * (bbox["x_min"] + bbox["x_max"])
        cz = 0.5 * (bbox["z_min"] + bbox["z_max"])
        objs.append({
            "prefabName": prefab_name,
            "x": cx,
            "z": cz,
            "bbox": bbox,
            "is_threat": any(prefab_name.startswith(t) for t in THREAT_TYPES),
        })
    return objs


def load_map(map_path: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, float]], List[Dict[str, Any]]]:
    """finalmap.map 로드 → (threats, gt_obstacle_bboxes, gt_objects).

    - threats: 위협 판정용 [{type,x,z,yaw,prefabName}]
    - gt_obstacle_bboxes: LoS 차폐 검사용 bbox 리스트
    - gt_objects: 센서 정확도 비교용 전체 객체(prefabName/중심/ bbox/is_threat)
    raise: FileNotFoundError / json.JSONDecodeError (호출자가 처리)
    """
    with open(map_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    threats = parse_threats_from_map(data)
    gt_objects = parse_gt_objects_from_map(data)
    gt_bboxes = [o["bbox"] for o in gt_objects]
    return threats, gt_bboxes, gt_objects
