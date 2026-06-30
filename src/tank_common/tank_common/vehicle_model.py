# -*- coding: utf-8 -*-
"""전차 동역학/제원 공통 모델.

Tank Challenge Simulator에서 수집한 평지 제어 데이터셋을 control과
path_planning이 같은 방식으로 쓰기 위한 단일 출처 모듈이다.
- W/S weight ↔ 정상상태 속도 보간
- A/D weight ↔ yaw rate 보간
- 속도 기반 정지거리/동적 inflation/emergency 거리 계산
- 경로 곡률/회전반경 feasibility 평가
"""
from __future__ import annotations

import math
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import yaml


def _as_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _load_yaml(path: str) -> Dict[str, Any]:
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            if isinstance(data, dict):
                return data
    return {}


def _table_from_dict(d: Any) -> List[Tuple[float, float]]:
    if not isinstance(d, dict):
        return []
    out: List[Tuple[float, float]] = []
    for k, v in d.items():
        try:
            out.append((float(k), float(v)))
        except Exception:
            continue
    return sorted(out, key=lambda x: x[0])


def _interp_x_to_y(table: Sequence[Tuple[float, float]], x: float) -> float:
    if not table:
        return 0.0
    x = float(x)
    if x <= table[0][0]:
        return table[0][1]
    if x >= table[-1][0]:
        return table[-1][1]
    for (x0, y0), (x1, y1) in zip(table[:-1], table[1:]):
        if x0 <= x <= x1:
            if abs(x1 - x0) < 1e-9:
                return y0
            t = (x - x0) / (x1 - x0)
            return y0 + t * (y1 - y0)
    return table[-1][1]


def _interp_y_to_x(table: Sequence[Tuple[float, float]], y: float) -> float:
    if not table:
        return 0.0
    pairs = sorted(((float(v), float(k)) for k, v in table), key=lambda x: x[0])
    y = float(y)
    if y <= pairs[0][0]:
        return pairs[0][1]
    if y >= pairs[-1][0]:
        return pairs[-1][1]
    for (y0, x0), (y1, x1) in zip(pairs[:-1], pairs[1:]):
        if y0 <= y <= y1:
            if abs(y1 - y0) < 1e-9:
                return x0
            t = (y - y0) / (y1 - y0)
            return x0 + t * (x1 - x0)
    return pairs[-1][1]


def _turn_radius_from_three_points(
    p0: Tuple[float, float], p1: Tuple[float, float], p2: Tuple[float, float]
) -> Optional[float]:
    """세 점으로 정의되는 외접원 반경을 반환한다. 거의 직선이면 None."""
    ax, ay = p0
    bx, by = p1
    cx, cy = p2
    ab = math.hypot(bx - ax, by - ay)
    bc = math.hypot(cx - bx, cy - by)
    ca = math.hypot(ax - cx, ay - cy)
    if ab < 1e-6 or bc < 1e-6 or ca < 1e-6:
        return None
    area2 = abs((bx - ax) * (cy - ay) - (by - ay) * (cx - ax))
    if area2 < 1e-6:
        return None
    return (ab * bc * ca) / (2.0 * area2)


class TankVehicleModel:
    def __init__(self, param_file: str = "") -> None:
        self.param_file = param_file or ""
        try:
            self.params = _load_yaml(self.param_file)
        except Exception:
            self.params = {}

        vehicle = self.params.get("vehicle", {}) if isinstance(self.params.get("vehicle"), dict) else {}
        planning = self.params.get("planning", {}) if isinstance(self.params.get("planning"), dict) else {}
        dynamics = self.params.get("dynamics_model", {}) if isinstance(self.params.get("dynamics_model"), dict) else {}
        stopping = self.params.get("stopping", {}) if isinstance(self.params.get("stopping"), dict) else {}

        self.body_width_m = _as_float(vehicle.get("body_width_m"), 3.667)
        self.body_length_m = _as_float(vehicle.get("body_length_m"), 8.066)
        self.half_width_m = _as_float(planning.get("half_width_m", vehicle.get("half_width_m")), self.body_width_m / 2.0)
        self.half_length_m = _as_float(planning.get("half_length_m", vehicle.get("half_length_m")), self.body_length_m / 2.0)
        self.safety_margin_m = _as_float(planning.get("safety_margin_m"), 1.0)
        self.command_latency_sec = _as_float(planning.get("command_latency_sec", stopping.get("reaction_time_sec")), 0.30)

        self.forward_speed_per_weight = _as_float(dynamics.get("forward_speed_mps_per_weight"), 13.6)
        self.reverse_speed_per_weight = _as_float(dynamics.get("reverse_speed_mps_per_weight"), 8.33)
        self.yaw_rate_degps_per_ad_weight = _as_float(dynamics.get("yaw_rate_degps_per_ad_weight"), 40.0)
        self.yaw_rate_radps_per_ad_weight = math.radians(self.yaw_rate_degps_per_ad_weight)

        self.steady_state_speed_table = _table_from_dict(self.params.get("steady_state_speed", {}))
        self.reverse_speed_table = _table_from_dict(self.params.get("reverse_speed", {}))
        self.yaw_rate_table = _table_from_dict(self.params.get("steady_state_yaw_rate", {}))

        self.reaction_time_sec = _as_float(stopping.get("reaction_time_sec"), self.command_latency_sec)
        self.decel_mps2 = max(0.1, _as_float(stopping.get("decel_mps2"), 3.0))
        self.stop_safety_distance_m = _as_float(stopping.get("safety_distance_m", stopping.get("safety_distance")), 3.0)
        self.min_stop_distance_m = _as_float(stopping.get("min_stop_distance_m"), 1.0)
        self.max_stop_distance_m = _as_float(stopping.get("max_stop_distance_m"), 12.0)

        self.base_static_inflate_m = _as_float(planning.get("base_static_inflate_m"), 2.0)
        self.base_dynamic_inflate_m = _as_float(planning.get("base_dynamic_inflate_m"), 4.5)
        self.base_discovered_inflate_m = _as_float(planning.get("base_discovered_inflate_m"), 2.0)
        self.base_lidar_memory_inflate_m = _as_float(planning.get("base_lidar_memory_inflate_m"), 3.0)
        self.max_speed_inflation_extra_m = _as_float(planning.get("max_speed_inflation_extra_m"), 3.5)
        self.speed_inflation_scale = _as_float(planning.get("speed_inflation_scale"), 0.45)
        self.emergency_front_base_m = _as_float(planning.get("emergency_front_base_m"), 24.0)
        self.emergency_front_min_m = _as_float(planning.get("emergency_front_min_m"), 12.0)
        self.emergency_front_max_m = _as_float(planning.get("emergency_front_max_m"), 45.0)
        self.turn_preparation_distance_m = _as_float(planning.get("turn_preparation_distance_m"), 6.0)

    def speed_for_ws_weight(self, weight: float) -> float:
        w = _clamp(float(weight), 0.0, 1.0)
        if self.steady_state_speed_table:
            return max(0.0, _interp_x_to_y(self.steady_state_speed_table, w))
        return max(0.0, self.forward_speed_per_weight * w)

    def ws_weight_for_speed(self, target_speed_mps: float) -> float:
        v = max(0.0, float(target_speed_mps))
        if self.steady_state_speed_table:
            return _clamp(_interp_y_to_x(self.steady_state_speed_table, v), 0.0, 1.0)
        if self.forward_speed_per_weight <= 1e-6:
            return 0.0
        return _clamp(v / self.forward_speed_per_weight, 0.0, 1.0)

    def yaw_rate_degps_for_ad_weight(self, weight: float) -> float:
        w = _clamp(abs(float(weight)), 0.0, 1.0)
        if self.yaw_rate_table:
            return max(0.0, _interp_x_to_y(self.yaw_rate_table, w))
        return max(0.0, self.yaw_rate_degps_per_ad_weight * w)

    def yaw_rate_radps_for_ad_weight(self, weight: float) -> float:
        return math.radians(self.yaw_rate_degps_for_ad_weight(weight))

    def min_turn_radius(self, speed_mps: float, ad_weight: float = 1.0) -> float:
        omega = max(1e-6, self.yaw_rate_radps_for_ad_weight(ad_weight))
        return max(0.0, abs(float(speed_mps)) / omega)

    def stopping_distance(self, speed_mps: float, include_safety: bool = True) -> float:
        v = max(0.0, abs(float(speed_mps)))
        reaction = v * max(0.0, self.reaction_time_sec)
        braking = (v * v) / (2.0 * max(0.1, self.decel_mps2))
        safety = self.stop_safety_distance_m if include_safety else 0.0
        return _clamp(reaction + braking + safety, self.min_stop_distance_m, self.max_stop_distance_m)

    def speed_margin(self, speed_mps: float) -> float:
        v = max(0.0, abs(float(speed_mps)))
        reaction = v * max(0.0, self.reaction_time_sec)
        braking = (v * v) / (2.0 * max(0.1, self.decel_mps2))
        return max(0.0, reaction + braking)

    def dynamic_inflation(self, speed_mps: float, base: Optional[float] = None) -> float:
        base_val = self.base_dynamic_inflate_m if base is None else float(base)
        extra = _clamp(self.speed_margin(speed_mps) * self.speed_inflation_scale, 0.0, self.max_speed_inflation_extra_m)
        return max(0.0, base_val + extra)

    def emergency_front_distance(self, speed_mps: float, base: Optional[float] = None) -> float:
        base_val = self.emergency_front_base_m if base is None else float(base)
        required = (
            abs(float(speed_mps)) * max(0.0, self.reaction_time_sec)
            + self.stopping_distance(speed_mps, include_safety=True)
            + max(0.0, self.turn_preparation_distance_m)
        )
        return _clamp(max(base_val, required), self.emergency_front_min_m, self.emergency_front_max_m)

    def path_curvature_summary(
        self,
        path: Sequence[Tuple[float, float]],
        current_speed_mps: float = 0.0,
        max_points: int = 120,
        sharp_ratio: float = 0.9,
    ) -> Dict[str, Any]:
        pts = [(float(x), float(y)) for x, y in list(path)[: max(3, max_points)] if x is not None and y is not None]
        radii: List[float] = []
        for i in range(1, len(pts) - 1):
            r = _turn_radius_from_three_points(pts[i - 1], pts[i], pts[i + 1])
            if r is not None and math.isfinite(r):
                radii.append(float(r))
        if not radii:
            min_r = float("inf")
        else:
            min_r = min(radii)
        required_r = self.min_turn_radius(max(0.0, abs(float(current_speed_mps))), ad_weight=1.0)
        # 현재 속도로 해당 곡률을 통과하기 어렵다면 권장속도는 R * omega_max이다.
        if math.isfinite(min_r):
            rec_speed = max(0.5, min_r * self.yaw_rate_radps_for_ad_weight(1.0) * 0.85)
        else:
            rec_speed = self.speed_for_ws_weight(0.35)
        sharp_count = sum(1 for r in radii if r < max(0.1, required_r * sharp_ratio))
        return {
            "min_path_turn_radius_m": None if not math.isfinite(min_r) else round(min_r, 3),
            "required_min_turn_radius_m": round(required_r, 3),
            "sharp_corner_count": int(sharp_count),
            "recommended_speed_limit_mps": round(float(rec_speed), 3),
            "feasible_at_current_speed": bool((not math.isfinite(min_r)) or min_r >= required_r),
        }
