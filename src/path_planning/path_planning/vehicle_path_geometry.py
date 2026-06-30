# -*- coding: utf-8 -*-
"""전차 운동 제약을 A* polyline에 반영하는 기하 보정 도구.

A*의 LOS smoothing 결과는 직선 구간의 연결이라 코너에서 곡률이 무한대가 된다.
이 모듈은 충분한 공간이 있는 코너만 접선 원호(tangent arc)로 치환한다.
원호가 costmap의 점유 셀을 침범하면 반경을 작게 억지로 줄이지 않고 원래 코너를 유지한다.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Dict, List, Sequence, Tuple

Point2 = Tuple[float, float]


@dataclass
class VehicleGeometryResult:
    points: List[Point2]
    point_kinds: List[str]
    rounded_corner_count: int
    unrounded_corner_count: int
    rejected_by_clearance_count: int

    def as_dict(self, *, minimum_turn_radius_m: float, brake_distance_m: float, arc_sample_step_m: float) -> Dict[str, float | int]:
        return {
            "enabled": True,
            "minimum_turn_radius_m": round(float(minimum_turn_radius_m), 3),
            "brake_distance_m": round(float(brake_distance_m), 3),
            "arc_sample_step_m": round(float(arc_sample_step_m), 3),
            "rounded_corner_count": int(self.rounded_corner_count),
            "unrounded_corner_count": int(self.unrounded_corner_count),
            "rejected_by_clearance_count": int(self.rejected_by_clearance_count),
        }


def _distance(a: Point2, b: Point2) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def _unit(a: Point2, b: Point2) -> Point2 | None:
    d = _distance(a, b)
    if d <= 1.0e-9:
        return None
    return ((b[0] - a[0]) / d, (b[1] - a[1]) / d)


def _cross(a: Point2, b: Point2) -> float:
    return a[0] * b[1] - a[1] * b[0]


def _left(v: Point2) -> Point2:
    return (-v[1], v[0])


def _right(v: Point2) -> Point2:
    return (v[1], -v[0])


def _append_unique(points: List[Point2], kinds: List[str], point: Point2, kind: str) -> None:
    if not points or _distance(points[-1], point) > 1.0e-5:
        points.append((float(point[0]), float(point[1])))
        kinds.append(kind)
    elif kind == "arc":
        # 동일한 점이 겹칠 때도 원호 속성은 유지한다.
        kinds[-1] = "arc"


def _sample_arc(center: Point2, start: Point2, end: Point2, *, turn_sign: float, step_m: float) -> List[Point2]:
    radius = _distance(center, start)
    if radius <= 1.0e-8:
        return [start, end]
    a0 = math.atan2(start[1] - center[1], start[0] - center[0])
    a1 = math.atan2(end[1] - center[1], end[0] - center[0])
    if turn_sign > 0.0:
        delta = (a1 - a0) % (2.0 * math.pi)
    else:
        delta = -((a0 - a1) % (2.0 * math.pi))
    # 접선 원호가 180도를 넘는 것은 이 보정의 대상이 아니다.
    if abs(delta) > math.pi + 1.0e-5:
        return []
    sample_count = max(2, int(math.ceil((radius * abs(delta)) / max(step_m, 0.1))) + 1)
    result: List[Point2] = []
    for index in range(sample_count):
        ratio = index / float(sample_count - 1)
        angle = a0 + delta * ratio
        result.append((center[0] + radius * math.cos(angle), center[1] + radius * math.sin(angle)))
    return result


def round_polyline_with_min_turn_radius(
    polyline: Sequence[Point2],
    *,
    minimum_turn_radius_m: float,
    arc_sample_step_m: float,
    is_free: Callable[[float, float], bool],
) -> VehicleGeometryResult:
    """충돌이 없는 접선 원호만 A* 코너에 삽입한다.

    - 코너 전후 세그먼트가 접선길이를 감당하지 못하면 원래 코너를 유지한다.
    - 원호 샘플 한 점이라도 점유 costmap 안이면 원래 코너를 유지한다.
    - 반경을 축소하지 않기 때문에 결과 경로는 설정한 최소 선회반경을 위반하지 않는다.
    """
    clean: List[Point2] = []
    for raw in polyline:
        p = (float(raw[0]), float(raw[1]))
        if not clean or _distance(clean[-1], p) > 1.0e-5:
            clean.append(p)

    if len(clean) < 3 or minimum_turn_radius_m <= 0.0:
        return VehicleGeometryResult(clean, ["straight"] * len(clean), 0, 0, 0)

    out: List[Point2] = []
    kinds: List[str] = []
    _append_unique(out, kinds, clean[0], "straight")
    rounded = 0
    unrounded = 0
    rejected_clearance = 0
    radius = float(minimum_turn_radius_m)
    step = max(0.15, float(arc_sample_step_m))

    for idx in range(1, len(clean) - 1):
        before, corner, after = clean[idx - 1], clean[idx], clean[idx + 1]
        vin = _unit(before, corner)
        vout = _unit(corner, after)
        if vin is None or vout is None:
            _append_unique(out, kinds, corner, "corner")
            unrounded += 1
            continue

        dot = max(-1.0, min(1.0, vin[0] * vout[0] + vin[1] * vout[1]))
        turn_angle = math.acos(dot)
        turn_sign = _cross(vin, vout)
        # 거의 직진 또는 거의 U-turn은 필렛으로 안전하게 처리하지 않는다.
        if turn_angle < math.radians(2.0) or abs(turn_sign) < 1.0e-7 or turn_angle > math.radians(175.0):
            _append_unique(out, kinds, corner, "corner")
            unrounded += 1
            continue

        tangent_distance = radius * math.tan(0.5 * turn_angle)
        # 전후 구간의 45% 이상을 사용해야 하면 원호가 다음 코너와 간섭할 가능성이 커진다.
        if tangent_distance <= 0.0 or tangent_distance > 0.45 * _distance(before, corner) or tangent_distance > 0.45 * _distance(corner, after):
            _append_unique(out, kinds, corner, "corner")
            unrounded += 1
            continue

        tangent_in = (corner[0] - vin[0] * tangent_distance, corner[1] - vin[1] * tangent_distance)
        tangent_out = (corner[0] + vout[0] * tangent_distance, corner[1] + vout[1] * tangent_distance)
        normal = _left(vin) if turn_sign > 0.0 else _right(vin)
        center = (tangent_in[0] + normal[0] * radius, tangent_in[1] + normal[1] * radius)
        arc = _sample_arc(center, tangent_in, tangent_out, turn_sign=turn_sign, step_m=step)
        if not arc or any(not is_free(x, y) for x, y in arc):
            _append_unique(out, kinds, corner, "corner")
            unrounded += 1
            rejected_clearance += 1
            continue

        _append_unique(out, kinds, tangent_in, "straight")
        for p in arc[1:]:
            _append_unique(out, kinds, p, "arc")
        rounded += 1

    _append_unique(out, kinds, clean[-1], "straight")
    return VehicleGeometryResult(out, kinds, rounded, unrounded, rejected_clearance)


def build_speed_profile(
    points: Sequence[Point2],
    point_kinds: Sequence[str],
    *,
    brake_distance_m: float,
    cruise_ws_weight: float,
    curve_ws_weight: float,
    brake_ws_weight: float,
    speed_per_weight_mps: float,
) -> List[Dict[str, float | str]]:
    """각 path point에 controller가 바로 쓸 속도 프로파일 메타데이터를 부여한다."""
    if not points:
        return []
    remaining = [0.0] * len(points)
    total = 0.0
    for i in range(len(points) - 2, -1, -1):
        total += _distance(points[i], points[i + 1])
        remaining[i] = total

    profile: List[Dict[str, float | str]] = []
    for idx, point in enumerate(points):
        kind = point_kinds[idx] if idx < len(point_kinds) else "straight"
        dist_to_goal = remaining[idx]
        if idx == len(points) - 1:
            phase = "stop"
            weight = 0.0
        elif dist_to_goal <= max(0.0, brake_distance_m):
            phase = "brake"
            weight = max(0.0, brake_ws_weight)
        elif kind == "arc":
            phase = "curve"
            weight = max(0.0, curve_ws_weight)
        else:
            phase = "cruise"
            weight = max(0.0, cruise_ws_weight)
        profile.append({
            "x": float(point[0]),
            "y": float(point[1]),
            "point_type": str(kind),
            "phase": phase,
            "distance_to_goal_m": round(float(dist_to_goal), 3),
            "recommended_ws_weight": round(float(weight), 3),
            "recommended_speed_mps": round(float(weight) * float(speed_per_weight_mps), 3),
        })
    return profile
