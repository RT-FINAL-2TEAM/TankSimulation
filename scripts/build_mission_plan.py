#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""미션 계획 생성기 — 정찰 산출물 → 시나리오2 사격 위치·루트·교전순서 제안.

파이프라인상 위치 (ROS·시뮬 불필요, 순수 후처리):
    build_scenario2_map.py  ──▶  recon_reports/recon_map/scenario2_map.map (표적=발견 전차 + 장애물)
    generate_recon_report.py ──▶ recon_reports/risk_features.json (루트별 노출 프로파일/위협)
                                              │
                                              ▼
                        build_mission_plan.py  ──▶  recon_reports/mission_plan.json / analysis/mission_plan.md

핵심 아이디어(수식):
  각 표적 T(발견 정지전차 + 최종 적 명목위치)에 대해, 루트를 따라 densify한 점 P 중
    - 사거리:  MIN_RANGE ≤ dist(P,T) ≤ MAX_RANGE   (ballistic min 20 / max 130m)
    - 사선 :   check_los(P→T)이 장애물에 안 막힘  (전차→표적 LoS, threat_geometry 방향무관 재사용)
  를 만족하는 후보 중 점수 최고점 = **사격 체크포인트**.
  점수 = w_range·(가까울수록↑) + w_exp·(1 − 노출)   (노출 = risk_features 위협에 보일 정도)

사격은 정지-조준-발사라 표적은 정지 가정(ballistic_turret_node) — 그래서 '루트 위 한 점에 멈춰 쏜다'가 성립.
routes(scenario2_routes.yaml)·engagements_json은 의도 설계물이라 **자동 덮어쓰지 않고 제안만** 한다(--emit-engagements로 스니펫 출력).

사용:
    python3 scripts/build_mission_plan.py
    python3 scripts/build_mission_plan.py --route A --emit-engagements
    python3 scripts/build_mission_plan.py --no-llm         # ollama 없이 기하 계획만
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from recon_eval import threat_geometry as tg  # noqa: E402

# --------------------------------------------------------------------------- #
# 상수 / 기본 경로
# --------------------------------------------------------------------------- #
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_REPORT_DIR = os.path.join(PROJECT_ROOT, "recon_reports")
DEFAULT_MAP = os.path.join(DEFAULT_REPORT_DIR, "recon_map", "scenario2_map.map")
DEFAULT_ROUTES = os.path.join(PROJECT_ROOT, "src", "control", "config", "scenario2_routes.yaml")
DEFAULT_RISK_FEATURES = os.path.join(DEFAULT_REPORT_DIR, "risk_features.json")
DEFAULT_TERRAIN = os.path.join(DEFAULT_REPORT_DIR, "recon_map", "scenario2_terrain.json")

# 무기 사거리(ballistic_turret_node: min_range_m 20 / max_range_m 130).
MIN_RANGE_M = 20.0
MAX_RANGE_M = 130.0
# 노출 거리감쇠 범위 = 위협반경 × 이 값(반경서 0.5, 2배서 0). generate_recon_report와 동일.
EXPOSURE_DECAY_MULT = 2.0
# 최종 적전차 명목위치(현 ballistic engagements enemy_final) — 발사 시 /tank/enemy/pose로 정밀화됨.
DEFAULT_FINAL_ENEMY_XY = (135.46, 276.87)
# 사격 위치 후보 탐색용 루트 densify 간격(m). 노출 profile(0.75)보다 성글어도 충분.
CANDIDATE_STEP_M = 1.0
# 점수 가중치(합 1.0). range=전술 사거리 대역 선호, exposure=은폐, flatness=평탄 사격 자세.
# 중요: ballistic_turret_node는 평지 사격 데이터 기반이므로 “가까운 사거리”보다
#       “정찰로 위치를 알고 있는 표적을 이전 평탄지에서 안정적으로 타격”하는 것을 우선한다.
DEFAULT_WEIGHTS = {"range": 0.20, "exposure": 0.20, "flatness": 0.45, "advance": 0.15}
FIRE_FLAT_ROUGHNESS_MAX = float(os.getenv("TANK_FIRE_FLAT_ROUGHNESS_MAX", "0.08"))
FIRE_REQUIRE_FLAT = os.getenv("TANK_FIRE_REQUIRE_FLAT", "true").strip().lower() in ("1", "true", "yes", "y", "on")
SCENARIO2_PREFERRED_ROUTE = os.getenv("TANK_SCENARIO2_PREFERRED_ROUTE", "A").strip().upper()

# Tactical standoff policy.  The weapon model can technically shoot from 20m,
# but Scenario-2 should not drive to the enemy tank's nose.  Recon has already
# fixed target coordinates, so camera visibility at the checkpoint is optional.
# Pick an earlier, flat, LoS-capable fire base instead.
FIRE_STATIC_MIN_RANGE_M = float(os.getenv("TANK_FIRE_STATIC_MIN_RANGE_M", "70.0"))
FIRE_STATIC_PREFERRED_RANGE_M = float(os.getenv("TANK_FIRE_STATIC_PREFERRED_RANGE_M", "115.0"))
FIRE_FINAL_MIN_RANGE_M = float(os.getenv("TANK_FIRE_FINAL_MIN_RANGE_M", "70.0"))
FIRE_FINAL_PREFERRED_RANGE_M = float(os.getenv("TANK_FIRE_FINAL_PREFERRED_RANGE_M", "95.0"))
# Prefer a checkpoint that is not too late on the route when scores are similar.
# This prevents the planner from choosing the last possible close-range point.
FIRE_ADVANCE_ARC_SCALE_M = float(os.getenv("TANK_FIRE_ADVANCE_ARC_SCALE_M", "80.0"))
# Local terrain patch stability around the tank footprint.  roughness alone can
# mark a single cell flat while the surrounding hull footprint is tilted.
FIRE_FLAT_PATCH_RADIUS_M = float(os.getenv("TANK_FIRE_FLAT_PATCH_RADIUS_M", "3.0"))
FIRE_FLAT_PATCH_Z_RANGE_MAX = float(os.getenv("TANK_FIRE_FLAT_PATCH_Z_RANGE_MAX", "1.00"))
# Minimum vertical clearance between the estimated ground profile and the straight
# muzzle-to-target sight/fire corridor.  This prevents selecting a very distant
# flat point if an intervening hill is likely to sit in front of the gun.
FIRE_TERRAIN_LOS_CLEARANCE_M = float(os.getenv("TANK_FIRE_TERRAIN_LOS_CLEARANCE_M", "0.50"))
FIRE_MUZZLE_HEIGHT_M = float(os.getenv("TANK_FIRE_MUZZLE_HEIGHT_M", "3.20"))
FIRE_CHECKPOINT_RADIUS_M = float(os.getenv("TANK_FIRE_CHECKPOINT_RADIUS_M", "1.0"))
# Mission planner must be stricter than ballistic_turret_node's runtime 3deg guard.
# Smooth hills can have low roughness but still pitch the hull; reject those
# before Scenario-2 starts instead of driving into fallback positions near target.
FIRE_FLAT_BODY_PITCH_MAX_DEG = float(os.getenv("TANK_FIRE_FLAT_BODY_PITCH_MAX_DEG", "2.2"))
FIRE_FLAT_BODY_ROLL_MAX_DEG = float(os.getenv("TANK_FIRE_FLAT_BODY_ROLL_MAX_DEG", "2.2"))
FIRE_FALLBACK_GOAL_COUNT = int(os.getenv("TANK_FIRE_FALLBACK_GOAL_COUNT", "3"))
# Ballistic planning uses the same low-arc model as ballistic_turret_node.
# The old straight-line terrain corridor could reject far fire bases behind a
# hill even though the actual shell arc clears the terrain.  That made the
# planner drift toward the target-side slope.  Use the ballistic arc instead.
FIRE_BALLISTIC_K = float(os.getenv("TANK_FIRE_BALLISTIC_K", "0.001520"))
FIRE_WORLD_PITCH_MIN_DEG = float(os.getenv("TANK_FIRE_WORLD_PITCH_MIN_DEG", "-4.5"))
FIRE_WORLD_PITCH_MAX_DEG = float(os.getenv("TANK_FIRE_WORLD_PITCH_MAX_DEG", "9.0"))
# Diagnostic stop-footprint sampling.  This is written into mission_plan.json
# so we can see whether a checkpoint is robust to controller stop error.  The
# radius is intentionally smaller than before because the checkpoint radius is
# now 1.5m; if the vehicle is allowed to stop 3~10m away, the plan is not a
# firing checkpoint anymore.
FIRE_RUNTIME_STOP_SIDE_ERROR_M = float(os.getenv("TANK_FIRE_RUNTIME_STOP_SIDE_ERROR_M", "1.5"))
FIRE_RUNTIME_STOP_ALONG_ERROR_M = float(os.getenv("TANK_FIRE_RUNTIME_STOP_ALONG_ERROR_M", "1.5"))
FIRE_RUNTIME_BODY_PITCH_MAX_DEG = float(os.getenv("TANK_FIRE_RUNTIME_BODY_PITCH_MAX_DEG", "999.0"))
FIRE_RUNTIME_BODY_ROLL_MAX_DEG = float(os.getenv("TANK_FIRE_RUNTIME_BODY_ROLL_MAX_DEG", "999.0"))
# Deterministic Scenario-2 stage-1 firebase.  Terrain analysis of the current
# recon map shows the route-A point near (49.4,164.5) is the most robust first
# shot position: long standoff, ballistic pitch inside +10/-5 deg, and a flat
# local tank footprint.  Keep it deterministic by default instead of letting a
# small scoring change move stage 1 back onto the hill near y≈208.
FIRE_FORCE_STATIC_FIREBASE = os.getenv("TANK_FIRE_FORCE_STATIC_FIREBASE", "true").strip().lower() in ("1", "true", "yes", "y", "on")
FIRE_STATIC_FIREBASE_X = float(os.getenv("TANK_FIRE_STATIC_FIREBASE_X", "49.42"))
FIRE_STATIC_FIREBASE_Y = float(os.getenv("TANK_FIRE_STATIC_FIREBASE_Y", "164.5"))
# Stage-1 must shoot the static/object tank in the center lane, not a
# duplicate detected_tank near the final enemy.  The launch file also enforces
# this at runtime, but writing it into mission_plan keeps the MFD and the
# execution sequence consistent.
FIRE_FORCE_STATIC_TARGET = os.getenv("TANK_FIRE_FORCE_STATIC_TARGET", "true").strip().lower() in ("1", "true", "yes", "y", "on")
FIRE_STATIC_TARGET_X = float(os.getenv("TANK_FIRE_STATIC_TARGET_X", "50.0"))
FIRE_STATIC_TARGET_Y = float(os.getenv("TANK_FIRE_STATIC_TARGET_Y", "285.0"))
FIRE_STATIC_TARGET_Z = float(os.getenv("TANK_FIRE_STATIC_TARGET_Z", "8.5"))
FIRE_STATIC_FIREBASE_FALLBACKS = [
    (float(x), float(y))
    for x, y in (
        item.split(":", 1)
        for item in os.getenv("TANK_FIRE_STATIC_FIREBASE_FALLBACKS", "49.42:163.5,49.42:165.0").split(",")
        if ":" in item
    )
]
# 노출 밴드 임계.
EXPOSURE_BANDS = ((0.2, "low"), (0.5, "medium"))  # 그 외 high


# --------------------------------------------------------------------------- #
# .env 로딩 (LLM 모델/URL을 정찰과 동일하게 — ros_bridge.config.load_env_file 경량 미러)
# --------------------------------------------------------------------------- #

def load_env_file() -> None:
    """프로젝트 루트 .env를 os.environ에 반영(이미 지정된 값은 유지, setdefault).

    무거운 ros_bridge 임포트를 피하려 단순 파서만 복제. 이게 있어야 LLMReporter가
    코드 기본값(qwen3:0.6b)이 아니라 .env의 TANK_LLM_MODEL(gemma3:4b)을 쓴다 — 정찰과 동일 모델.
    """
    env_path = os.environ.get("TANK_ENV_FILE") or os.path.join(PROJECT_ROOT, ".env")
    if not os.path.isfile(env_path):
        return
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


# --------------------------------------------------------------------------- #
# 기하 헬퍼
# --------------------------------------------------------------------------- #

def densify_polyline(poly: List[Tuple[float, float]], step: float) -> Tuple[List[Tuple[float, float]], List[float]]:
    """폴리라인을 ~step(m) 간격으로 보간. (점 리스트, 각 점의 누적 호길이) 반환. generate_recon_report 미러."""
    if not poly or len(poly) < 2:
        pts = [tuple(p) for p in (poly or [])]
        return pts, [0.0] * len(pts)
    out: List[Tuple[float, float]] = [tuple(poly[0])]
    arc: List[float] = [0.0]
    acc = 0.0
    for i in range(1, len(poly)):
        x0, z0 = poly[i - 1]
        x1, z1 = poly[i]
        d = math.hypot(x1 - x0, z1 - z0)
        n = max(1, int(d / step))
        for k in range(1, n + 1):
            t = k / n
            out.append((x0 + (x1 - x0) * t, z0 + (z1 - z0) * t))
            acc += d / n
            arc.append(acc)
    return out, arc


def exposure_band(value: float) -> str:
    for thr, name in EXPOSURE_BANDS:
        if value < thr:
            return name
    return "high"


def target_range_policy(target: Dict[str, Any]) -> Tuple[float, float]:
    """Return (minimum tactical range, preferred standoff range) for a target.

    The ballistic model's hard minimum remains MIN_RANGE_M.  This policy adds a
    scenario-level tactical minimum so a known recon target is engaged from a
    safer earlier fire base rather than from immediately in front of it.
    """
    cls = str(target.get("class", ""))
    if cls == "final_enemy":
        return max(MIN_RANGE_M, FIRE_FINAL_MIN_RANGE_M), min(MAX_RANGE_M, FIRE_FINAL_PREFERRED_RANGE_M)
    return max(MIN_RANGE_M, FIRE_STATIC_MIN_RANGE_M), min(MAX_RANGE_M, FIRE_STATIC_PREFERRED_RANGE_M)


def preferred_range_score(distance_m: float, min_allowed_m: float, preferred_m: float) -> float:
    """0..1 score peaked at preferred standoff, not at closest possible range."""
    if distance_m < min_allowed_m or distance_m > MAX_RANGE_M:
        return 0.0
    left = max(1.0, preferred_m - min_allowed_m)
    right = max(1.0, MAX_RANGE_M - preferred_m)
    if distance_m <= preferred_m:
        return max(0.0, 1.0 - (preferred_m - distance_m) / left)
    return max(0.0, 1.0 - (distance_m - preferred_m) / right)


def point_exposure(px: float, py: float, exposure_threats: List[Dict[str, Any]],
                   bboxes: List[Dict[str, float]], exclude_xy: Optional[Tuple[float, float]]) -> float:
    """점 P가 위협들에 얼마나 노출되는가(0=은폐, 1=코앞+시야). 거리감쇠×LoS 중 최대.

    generate_recon_report.compute_centerline_exposure의 세그먼트 판정을 점 단위로 옮긴 것.
    exclude_xy(현재 조준 중인 표적)는 제외 — 표적이 나를 보는 건 교전 그 자체라 노출로 안 셈.
    """
    worst = 0.0
    for th in exposure_threats:
        tx = float(th["x"])
        tz = float(th["z"])
        if exclude_xy is not None and math.hypot(tx - exclude_xy[0], tz - exclude_xy[1]) < 2.0:
            continue
        r = float(th.get("radius") or tg.threat_radius(th))
        if r <= 0:
            continue
        d = math.hypot(px - tx, py - tz)
        decay = max(0.0, 1.0 - d / (EXPOSURE_DECAY_MULT * r))
        if decay > worst and tg.check_los(px, py, tx, tz, bboxes):
            worst = decay
    return worst


def load_terrain_grid(path: str) -> Optional[Dict[str, Any]]:
    """Load scenario2_terrain.json and build a nearest-cell index for flat firing checks."""
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None
    cells = data.get("cells") if isinstance(data, dict) else None
    if not isinstance(cells, list) or not cells:
        return None
    cell_size = float(data.get("cell_size") or 1.0)
    by_key: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for c in cells:
        if not isinstance(c, dict):
            continue
        ix = c.get("ix")
        iy = c.get("iy")
        if ix is None or iy is None:
            x = c.get("x")
            y = c.get("y")
            if x is None or y is None:
                continue
            ix = math.floor(float(x) / cell_size)
            iy = math.floor(float(y) / cell_size)
        try:
            by_key[(int(ix), int(iy))] = c
        except Exception:
            continue
    data["_by_key"] = by_key
    data["_cell_size"] = cell_size
    return data


def terrain_roughness_at(terrain: Optional[Dict[str, Any]], x: float, y: float) -> Optional[float]:
    if not terrain:
        return None
    cell_size = float(terrain.get("_cell_size") or terrain.get("cell_size") or 1.0)
    by_key = terrain.get("_by_key") or {}
    ix = math.floor(float(x) / cell_size)
    iy = math.floor(float(y) / cell_size)
    best = None
    best_d = float("inf")
    # Search a small neighborhood to tolerate route sample / cell-center mismatch.
    for dx in range(-2, 3):
        for dy in range(-2, 3):
            c = by_key.get((ix + dx, iy + dy))
            if not isinstance(c, dict):
                continue
            rough = c.get("roughness")
            if rough is None:
                continue
            cx = float(c.get("x", (ix + dx + 0.5) * cell_size))
            cy = float(c.get("y", (iy + dy + 0.5) * cell_size))
            d = math.hypot(float(x) - cx, float(y) - cy)
            if d < best_d:
                best_d = d
                best = float(rough)
    return best


def terrain_cell_at(terrain: Optional[Dict[str, Any]], x: float, y: float) -> Optional[Dict[str, Any]]:
    if not terrain:
        return None
    cell_size = float(terrain.get("_cell_size") or terrain.get("cell_size") or 1.0)
    by_key = terrain.get("_by_key") or {}
    ix = math.floor(float(x) / cell_size)
    iy = math.floor(float(y) / cell_size)
    best = None
    best_d = float("inf")
    for dx in range(-2, 3):
        for dy in range(-2, 3):
            c = by_key.get((ix + dx, iy + dy))
            if not isinstance(c, dict):
                continue
            cx = float(c.get("x", (ix + dx + 0.5) * cell_size))
            cy = float(c.get("y", (iy + dy + 0.5) * cell_size))
            d = math.hypot(float(x) - cx, float(y) - cy)
            if d < best_d:
                best_d = d
                best = c
    return best


def terrain_z_at(terrain: Optional[Dict[str, Any]], x: float, y: float) -> Optional[float]:
    c = terrain_cell_at(terrain, x, y)
    if not isinstance(c, dict) or c.get("z_median") is None:
        return None
    return float(c["z_median"])


def terrain_fire_corridor_clearance(terrain: Optional[Dict[str, Any]], px: float, py: float,
                                    tx: float, ty: float, target_z: float,
                                    samples: int = 80) -> Optional[Dict[str, float]]:
    """Estimate whether an intervening terrain ridge blocks the shot corridor."""
    if not terrain:
        return None
    start_z = terrain_z_at(terrain, px, py)
    if start_z is None:
        return None
    muzzle_z = start_z + FIRE_MUZZLE_HEIGHT_M
    min_clearance = float("inf")
    worst_x = px
    worst_y = py
    checked = 0
    for i in range(1, max(2, samples)):
        t = i / float(samples)
        x = px + (tx - px) * t
        y = py + (ty - py) * t
        ground_z = terrain_z_at(terrain, x, y)
        if ground_z is None:
            continue
        line_z = muzzle_z + (float(target_z) - muzzle_z) * t
        clearance = line_z - ground_z
        checked += 1
        if clearance < min_clearance:
            min_clearance = clearance
            worst_x = x
            worst_y = y
    if checked <= 0 or min_clearance == float("inf"):
        return None
    return {
        "min_clearance_m": float(min_clearance),
        "worst_x": float(worst_x),
        "worst_y": float(worst_y),
        "muzzle_z_m": float(muzzle_z),
        "samples": float(checked),
    }



def solve_low_arc_pitch_rad(horizontal_range: float, height_delta: float) -> Optional[float]:
    """Same low-arc ballistic solver used by ballistic_turret_node."""
    if horizontal_range <= 1e-6:
        return None
    r2 = horizontal_range * horizontal_range
    discriminant = r2 - 4.0 * FIRE_BALLISTIC_K * r2 * (FIRE_BALLISTIC_K * r2 + height_delta)
    if discriminant < 0.0:
        return None
    tan_theta = (horizontal_range - math.sqrt(discriminant)) / (2.0 * FIRE_BALLISTIC_K * r2)
    return math.atan(tan_theta)


def terrain_ballistic_corridor_clearance(
    terrain: Optional[Dict[str, Any]],
    px: float, py: float, tx: float, ty: float, target_z: float,
    samples: int = 80,
) -> Optional[Dict[str, float]]:
    """Estimate terrain clearance along the shell's low arc, not a straight ray."""
    if not terrain:
        return None
    start_z = terrain_z_at(terrain, px, py)
    if start_z is None:
        return None
    muzzle_z = start_z + FIRE_MUZZLE_HEIGHT_M
    horizontal = math.hypot(tx - px, ty - py)
    theta = solve_low_arc_pitch_rad(horizontal, float(target_z) - muzzle_z)
    if theta is None:
        return None
    pitch_deg = math.degrees(theta)
    if pitch_deg < FIRE_WORLD_PITCH_MIN_DEG or pitch_deg > FIRE_WORLD_PITCH_MAX_DEG:
        return {
            "min_clearance_m": -999.0,
            "worst_x": px,
            "worst_y": py,
            "muzzle_z_m": muzzle_z,
            "samples": 0.0,
            "target_world_pitch_deg": pitch_deg,
            "pitch_limit_reject": 1.0,
        }
    cos2 = max(1e-6, math.cos(theta) ** 2)
    min_clearance = float("inf")
    worst_x = px
    worst_y = py
    checked = 0
    for i in range(1, max(2, samples)):
        t = i / float(samples)
        x = px + (tx - px) * t
        y = py + (ty - py) * t
        ground_z = terrain_z_at(terrain, x, y)
        if ground_z is None:
            continue
        s = horizontal * t
        arc_z = muzzle_z + s * math.tan(theta) - FIRE_BALLISTIC_K * s * s / cos2
        clearance = arc_z - ground_z
        checked += 1
        if clearance < min_clearance:
            min_clearance = clearance
            worst_x = x
            worst_y = y
    if checked <= 0 or min_clearance == float("inf"):
        return None
    return {
        "min_clearance_m": float(min_clearance),
        "worst_x": float(worst_x),
        "worst_y": float(worst_y),
        "muzzle_z_m": float(muzzle_z),
        "samples": float(checked),
        "target_world_pitch_deg": float(pitch_deg),
        "pitch_limit_reject": 0.0,
    }


def route_tangent_at(pts: List[Tuple[float, float]], idx: int, span: int = 3) -> Tuple[float, float]:
    """Approximate local route forward unit vector at densified point idx."""
    if not pts:
        return (0.0, 1.0)
    i0 = max(0, int(idx) - span)
    i1 = min(len(pts) - 1, int(idx) + span)
    if i0 == i1 and len(pts) > 1:
        i0 = max(0, int(idx) - 1)
        i1 = min(len(pts) - 1, int(idx) + 1)
    dx = float(pts[i1][0]) - float(pts[i0][0])
    dy = float(pts[i1][1]) - float(pts[i0][1])
    n = math.hypot(dx, dy)
    if n <= 1e-6:
        return (0.0, 1.0)
    return (dx / n, dy / n)


def _solve_3x3(a: List[List[float]], b: List[float]) -> Optional[Tuple[float, float, float]]:
    """Small Gaussian solver for the normal equation of z=ax+by+c."""
    m = [list(row) + [float(bi)] for row, bi in zip(a, b)]
    for col in range(3):
        pivot = max(range(col, 3), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-9:
            return None
        if pivot != col:
            m[col], m[pivot] = m[pivot], m[col]
        div = m[col][col]
        for j in range(col, 4):
            m[col][j] /= div
        for r in range(3):
            if r == col:
                continue
            factor = m[r][col]
            for j in range(col, 4):
                m[r][j] -= factor * m[col][j]
    return (m[0][3], m[1][3], m[2][3])


def enrich_patch_with_body_tilt(patch: Optional[Dict[str, float]], forward: Tuple[float, float]) -> Optional[Dict[str, float]]:
    """Add expected hull pitch/roll from fitted terrain plane and route heading."""
    if not patch or patch.get("plane_slope_x") is None or patch.get("plane_slope_y") is None:
        return patch
    out = dict(patch)
    gx = float(out["plane_slope_x"])
    gy = float(out["plane_slope_y"])
    fx, fy = forward
    # right vector in map ground plane. Sign is not important for flatness guard.
    rx, ry = fy, -fx
    pitch = math.degrees(math.atan(gx * fx + gy * fy))
    roll = math.degrees(math.atan(gx * rx + gy * ry))
    out["estimated_body_pitch_deg"] = float(pitch)
    out["estimated_body_roll_deg"] = float(roll)
    out["abs_estimated_body_pitch_deg"] = abs(float(pitch))
    out["abs_estimated_body_roll_deg"] = abs(float(roll))
    return out



def terrain_runtime_pose_safety_at(
    terrain: Optional[Dict[str, Any]],
    pts: List[Tuple[float, float]],
    idx: int,
    x: float,
    y: float,
) -> Optional[Dict[str, float]]:
    """Sample the small expected stop footprint around a firing checkpoint."""
    if not terrain:
        return None
    fx, fy = route_tangent_at(pts, idx)
    rx, ry = fy, -fx
    along_values = (-FIRE_RUNTIME_STOP_ALONG_ERROR_M, 0.0, FIRE_RUNTIME_STOP_ALONG_ERROR_M)
    side_values = (-FIRE_RUNTIME_STOP_SIDE_ERROR_M, -0.5 * FIRE_RUNTIME_STOP_SIDE_ERROR_M,
                   0.0, 0.5 * FIRE_RUNTIME_STOP_SIDE_ERROR_M, FIRE_RUNTIME_STOP_SIDE_ERROR_M)
    max_pitch = 0.0
    max_roll = 0.0
    max_z_range = 0.0
    max_rough = 0.0
    valid = 0
    missing = 0
    for along in along_values:
        for side in side_values:
            sx = float(x) + fx * along + rx * side
            sy = float(y) + fy * along + ry * side
            patch = terrain_patch_stats_at(terrain, sx, sy, FIRE_FLAT_PATCH_RADIUS_M)
            patch = enrich_patch_with_body_tilt(patch, (fx, fy))
            rough = terrain_roughness_at(terrain, sx, sy)
            if not patch:
                missing += 1
                continue
            valid += 1
            max_pitch = max(max_pitch, abs(float(patch.get("estimated_body_pitch_deg") or 0.0)))
            max_roll = max(max_roll, abs(float(patch.get("estimated_body_roll_deg") or 0.0)))
            if patch.get("z_range_m") is not None:
                max_z_range = max(max_z_range, abs(float(patch.get("z_range_m") or 0.0)))
            if rough is not None:
                max_rough = max(max_rough, abs(float(rough)))
            if patch.get("roughness_max") is not None:
                max_rough = max(max_rough, abs(float(patch.get("roughness_max") or 0.0)))
    if valid <= 0:
        return None
    return {
        "sample_count": float(valid),
        "missing_count": float(missing),
        "max_abs_body_pitch_deg": float(max_pitch),
        "max_abs_body_roll_deg": float(max_roll),
        "max_z_range_m": float(max_z_range),
        "max_roughness": float(max_rough),
        "side_error_m": FIRE_RUNTIME_STOP_SIDE_ERROR_M,
        "along_error_m": FIRE_RUNTIME_STOP_ALONG_ERROR_M,
    }


def terrain_patch_stats_at(terrain: Optional[Dict[str, Any]], x: float, y: float,
                           radius_m: float = FIRE_FLAT_PATCH_RADIUS_M) -> Optional[Dict[str, float]]:
    """Summarize the local ground patch under/around the hull footprint."""
    if not terrain:
        return None
    cell_size = float(terrain.get("_cell_size") or terrain.get("cell_size") or 1.0)
    by_key = terrain.get("_by_key") or {}
    ix = math.floor(float(x) / cell_size)
    iy = math.floor(float(y) / cell_size)
    r_cells = max(1, int(math.ceil(radius_m / max(1e-6, cell_size))))
    zs: List[float] = []
    roughs: List[float] = []
    pts3: List[Tuple[float, float, float]] = []
    for dx in range(-r_cells, r_cells + 1):
        for dy in range(-r_cells, r_cells + 1):
            c = by_key.get((ix + dx, iy + dy))
            if not isinstance(c, dict):
                continue
            cx = float(c.get("x", (ix + dx + 0.5) * cell_size))
            cy = float(c.get("y", (iy + dy + 0.5) * cell_size))
            if math.hypot(float(x) - cx, float(y) - cy) > radius_m:
                continue
            if c.get("z_median") is not None:
                z = float(c["z_median"])
                zs.append(z)
                pts3.append((cx, cy, z))
            if c.get("roughness") is not None:
                roughs.append(float(c["roughness"]))
    if not zs and not roughs:
        return None
    z_range = (max(zs) - min(zs)) if zs else None
    z_mean = (sum(zs) / len(zs)) if zs else None
    z_std = (sum((z - z_mean) ** 2 for z in zs) / len(zs)) ** 0.5 if zs and z_mean is not None else None
    rough_max = max(roughs) if roughs else None
    rough_mean = sum(roughs) / len(roughs) if roughs else None
    out = {
        "cell_count": float(max(len(zs), len(roughs))),
        "z_range_m": None if z_range is None else float(z_range),
        "z_std_m": None if z_std is None else float(z_std),
        "roughness_max": None if rough_max is None else float(rough_max),
        "roughness_mean": None if rough_mean is None else float(rough_mean),
        "plane_slope_x": None,
        "plane_slope_y": None,
        "plane_slope_deg": None,
    }
    if len(pts3) >= 3:
        sx = sum(x for x, _, _ in pts3)
        sy = sum(y for _, y, _ in pts3)
        sz = sum(z for _, _, z in pts3)
        sxx = sum(x * x for x, _, _ in pts3)
        syy = sum(y * y for _, y, _ in pts3)
        sxy = sum(x * y for x, y, _ in pts3)
        sxz = sum(x * z for x, _, z in pts3)
        syz = sum(y * z for _, y, z in pts3)
        n = float(len(pts3))
        sol = _solve_3x3([[sxx, sxy, sx], [sxy, syy, sy], [sx, sy, n]], [sxz, syz, sz])
        if sol is not None:
            a, b, _ = sol
            out["plane_slope_x"] = float(a)
            out["plane_slope_y"] = float(b)
            out["plane_slope_deg"] = math.degrees(math.atan(math.hypot(a, b)))
    return out


def terrain_flat_score(rough: Optional[float], patch: Optional[Dict[str, float]]) -> Tuple[float, bool]:
    """Return 0..1 flatness score and whether terrain information was available."""
    if rough is None and not patch:
        return 1.0, False
    penalties: List[float] = []
    if rough is not None:
        penalties.append(float(rough) / max(1e-6, FIRE_FLAT_ROUGHNESS_MAX))
    if patch:
        z_range = patch.get("z_range_m")
        if z_range is not None:
            penalties.append(float(z_range) / max(1e-6, FIRE_FLAT_PATCH_Z_RANGE_MAX))
        rough_max = patch.get("roughness_max")
        if rough_max is not None:
            penalties.append(float(rough_max) / max(1e-6, FIRE_FLAT_ROUGHNESS_MAX * 4.0))
        est_pitch = patch.get("abs_estimated_body_pitch_deg")
        if est_pitch is not None:
            penalties.append(float(est_pitch) / max(1e-6, FIRE_FLAT_BODY_PITCH_MAX_DEG))
        est_roll = patch.get("abs_estimated_body_roll_deg")
        if est_roll is not None:
            penalties.append(float(est_roll) / max(1e-6, FIRE_FLAT_BODY_ROLL_MAX_DEG))
    penalty = max(penalties) if penalties else 0.0
    return max(0.0, min(1.0, 1.0 - penalty)), True


def best_firing_position(target: Dict[str, Any], pts: List[Tuple[float, float]], arc: List[float],
                         bboxes: List[Dict[str, float]], exposure_threats: List[Dict[str, Any]],
                         weights: Dict[str, float], terrain: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """표적 T에 대해 루트 점 중 사거리+LoS+평탄도를 만족하는 최고점수 사격 위치."""
    tx, ty = float(target["x"]), float(target["y"])
    raw_target_z = float(target.get("z_height", 0.0) or 0.0)
    terrain_target_z = terrain_z_at(terrain, tx, ty)
    # Discovered tanks often carry a measured object height.  The final enemy
    # uses live /tank/enemy/pose at firing time and may have z=0 in the plan,
    # so for terrain-corridor screening use an estimated turret/body center
    # above local terrain rather than ground level.
    target_z = raw_target_z if raw_target_z > 0.5 else (float(terrain_target_z) + 3.0 if terrain_target_z is not None else 3.0)
    min_allowed, preferred_range = target_range_policy(target)
    total_arc = max(arc) if arc else 0.0

    def scan(require_flat: bool, require_tactical_min: bool = True) -> Optional[Dict[str, Any]]:
        best: Optional[Dict[str, Any]] = None
        candidates: List[Dict[str, Any]] = []
        for idx, (px, py) in enumerate(pts):
            d = math.hypot(px - tx, py - ty)
            hard_min = min_allowed if require_tactical_min else MIN_RANGE_M
            if d < hard_min or d > MAX_RANGE_M:
                continue
            if not tg.check_los(px, py, tx, ty, bboxes):
                continue
            corridor = terrain_ballistic_corridor_clearance(terrain, px, py, tx, ty, target_z)
            if (
                corridor is not None
                and corridor.get("min_clearance_m") is not None
                and float(corridor["min_clearance_m"]) < FIRE_TERRAIN_LOS_CLEARANCE_M
            ):
                continue
            rough = terrain_roughness_at(terrain, px, py)
            patch = terrain_patch_stats_at(terrain, px, py)
            forward = route_tangent_at(pts, idx)
            patch = enrich_patch_with_body_tilt(patch, forward)
            runtime_safety = terrain_runtime_pose_safety_at(terrain, pts, idx, px, py)
            flat_score, flat_available = terrain_flat_score(rough, patch)
            # Keep runtime safety diagnostic in the plan, but do not let a wide
            # assumed stop error reject a point.  The actual stop radius is now
            # clamped to 1.5m in the launch file; the center-cell flatness and
            # ballistic pitch limit are the hard constraints.
            if runtime_safety is not None:
                runtime_pitch_ratio = runtime_safety.get("max_abs_body_pitch_deg", 0.0) / max(1e-6, FIRE_RUNTIME_BODY_PITCH_MAX_DEG)
                runtime_roll_ratio = runtime_safety.get("max_abs_body_roll_deg", 0.0) / max(1e-6, FIRE_RUNTIME_BODY_ROLL_MAX_DEG)
                runtime_rough_ratio = runtime_safety.get("max_roughness", 0.0) / max(1e-6, FIRE_FLAT_ROUGHNESS_MAX * 4.0)
                runtime_penalty = max(runtime_pitch_ratio, runtime_roll_ratio, runtime_rough_ratio)
                flat_score = min(flat_score, max(0.0, 1.0 - runtime_penalty))
            if rough is not None and require_flat and rough > FIRE_FLAT_ROUGHNESS_MAX:
                continue
            if (
                patch is not None
                and require_flat
                and patch.get("z_range_m") is not None
                and float(patch["z_range_m"]) > FIRE_FLAT_PATCH_Z_RANGE_MAX
            ):
                continue
            if (
                patch is not None
                and require_flat
                and patch.get("abs_estimated_body_pitch_deg") is not None
                and float(patch["abs_estimated_body_pitch_deg"]) > FIRE_FLAT_BODY_PITCH_MAX_DEG
            ):
                continue
            if (
                patch is not None
                and require_flat
                and patch.get("abs_estimated_body_roll_deg") is not None
                and float(patch["abs_estimated_body_roll_deg"]) > FIRE_FLAT_BODY_ROLL_MAX_DEG
            ):
                continue
            if (
                corridor is not None
                and corridor.get("pitch_limit_reject") is not None
                and float(corridor.get("pitch_limit_reject") or 0.0) > 0.5
            ):
                continue
            range_score = preferred_range_score(d, min_allowed if require_tactical_min else MIN_RANGE_M, preferred_range)
            exp = point_exposure(px, py, exposure_threats, bboxes, exclude_xy=(tx, ty))
            # Earlier route-arc bonus.  If two candidates are similarly flat and in range,
            # choose the one before the target instead of the last close point.
            advance_score = 1.0 if total_arc <= 1e-6 else max(0.0, min(1.0, 1.0 - arc[idx] / max(total_arc, FIRE_ADVANCE_ARC_SCALE_M)))
            active_weights = dict(weights)
            if not flat_available:
                active_weights["flatness"] = 0.0
            wsum = sum(max(0.0, float(v)) for v in active_weights.values()) or 1.0
            score = (
                active_weights.get("range", 0.0) * range_score
                + active_weights.get("exposure", 0.0) * (1.0 - exp)
                + active_weights.get("flatness", 0.0) * flat_score
                + active_weights.get("advance", 0.0) * advance_score
            ) / wsum
            candidate = {
                "x": round(px, 2), "y": round(py, 2),
                "distance_m": round(d, 2), "los": True,
                "exposure": round(exp, 3), "exposure_band": exposure_band(exp),
                "range_score": round(range_score, 3),
                "preferred_range_m": round(preferred_range, 2),
                "tactical_min_range_m": round(min_allowed, 2),
                "advance_score": round(advance_score, 3),
                "terrain_roughness": None if rough is None else round(float(rough), 4),
                "terrain_patch": None if patch is None else {
                    k: (None if v is None else round(float(v), 4)) for k, v in patch.items()
                },
                "terrain_fire_corridor": None if corridor is None else {
                    k: (None if v is None else round(float(v), 4)) for k, v in corridor.items()
                },
                "runtime_pose_safety": None if runtime_safety is None else {
                    k: (None if v is None else round(float(v), 4)) for k, v in runtime_safety.items()
                },
                "target_world_pitch_deg": None if corridor is None or corridor.get("target_world_pitch_deg") is None else round(float(corridor["target_world_pitch_deg"]), 4),
                "flat_score": round(flat_score, 4),
                "flat_filter_used": bool(require_flat and flat_available),
                "score": round(score, 4),
                "route_arc_m": round(arc[idx], 2),
                "selection_policy": "flat_standoff_recon_target",
            }
            candidates.append(candidate)
            if best is None or score > best["score"]:
                best = candidate
        if best is not None and candidates:
            candidates.sort(key=lambda c: (-float(c.get("score", 0.0)), float(c.get("exposure", 1.0)), float(c.get("route_arc_m", 0.0))))
            alts: List[Dict[str, float]] = []
            for c in candidates:
                if math.hypot(float(c["x"]) - float(best["x"]), float(c["y"]) - float(best["y"])) < 1.0:
                    continue
                # Do not provide fallback goals that move materially closer to
                # the target than the selected tactical standoff.  That was the
                # cause of Scenario-2 driving to y≈255 for the first shot.
                if float(c.get("distance_m", 0.0)) < max(min_allowed, float(best.get("distance_m", 0.0)) - 5.0):
                    continue
                alts.append({
                    "x": float(c["x"]),
                    "y": float(c["y"]),
                    "distance_m": float(c.get("distance_m", 0.0)),
                    "score": float(c.get("score", 0.0)),
                    "estimated_body_pitch_deg": float((c.get("terrain_patch") or {}).get("estimated_body_pitch_deg", 0.0) or 0.0),
                    "estimated_body_roll_deg": float((c.get("terrain_patch") or {}).get("estimated_body_roll_deg", 0.0) or 0.0),
                    "runtime_max_pitch_deg": float((c.get("runtime_pose_safety") or {}).get("max_abs_body_pitch_deg", 0.0) or 0.0),
                    "runtime_max_roll_deg": float((c.get("runtime_pose_safety") or {}).get("max_abs_body_roll_deg", 0.0) or 0.0),
                })
                if len(alts) >= FIRE_FALLBACK_GOAL_COUNT:
                    break
            best["fallback_goals"] = alts
        return best

    best = scan(FIRE_REQUIRE_FLAT and terrain is not None, require_tactical_min=True)
    if best is None and terrain is not None:
        # Relax the patch flatness requirement first, but keep tactical standoff.
        best = scan(False, require_tactical_min=True)
        if best is not None:
            best["flat_filter_fallback"] = True
    if best is None:
        # Last resort: keep the original ballistic range only.  This is marked so
        # the operator knows the checkpoint is a fallback, not the preferred plan.
        best = scan(False, require_tactical_min=False)
        if best is not None:
            best["tactical_range_fallback"] = True
    return best


# --------------------------------------------------------------------------- #
# 입력 로딩
# --------------------------------------------------------------------------- #

def load_targets(map_path: str, final_enemy_xy: Optional[Tuple[float, float]]) -> List[Dict[str, Any]]:
    """scenario2_map.targets(발견 전차) + (선택) 최종 적전차 명목위치 → 표적 리스트."""
    with open(map_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    targets: List[Dict[str, Any]] = []
    for i, t in enumerate(data.get("targets", [])):
        if not isinstance(t, dict):
            continue
        pos = t.get("position") if isinstance(t.get("position"), dict) else {}
        mx = t.get("map_x", pos.get("x"))
        my = t.get("map_y", pos.get("z"))
        if mx is None or my is None:
            continue
        targets.append({
            "id": t.get("prefabName") or f"tank_{i}",
            "class": "discovered_static",
            "x": float(mx), "y": float(my),
            "z_height": float(pos.get("y", 0.0)),
        })
    if final_enemy_xy is not None:
        targets.append({
            "id": "enemy_final", "class": "final_enemy",
            "x": float(final_enemy_xy[0]), "y": float(final_enemy_xy[1]),
            "z_height": 0.0,
        })
    return targets


def load_route_polylines(routes_path: str) -> Dict[str, List[Tuple[float, float]]]:
    """scenario2_routes.yaml → {rid: [start, *waypoints, destination]} 폴리라인."""
    import yaml
    with open(routes_path, "r", encoding="utf-8") as f:
        rd = yaml.safe_load(f)["finalmap"]
    start = tuple(rd["start"])
    dest = tuple(rd["destination"])
    out: Dict[str, List[Tuple[float, float]]] = {}
    for rid, wps in rd.get("routes", {}).items():
        poly = [start] + [tuple(p) for p in wps] + [dest]
        out[str(rid)] = [(float(x), float(y)) for x, y in poly]
    return out


def load_exposure_threats(risk_features_path: str, rid: str) -> List[Dict[str, Any]]:
    """risk_features.json route_<rid>.threat.list → 노출 판정용 위협 [{x,z,radius,type}]. 없으면 []."""
    try:
        with open(risk_features_path, "r", encoding="utf-8") as f:
            rf = json.load(f)
    except (OSError, json.JSONDecodeError):
        return []
    route = rf.get(f"route_{rid}") if isinstance(rf.get(f"route_{rid}"), dict) else {}
    lst = route.get("threat", {}).get("list") if isinstance(route.get("threat"), dict) else None
    if not isinstance(lst, list):
        return []
    threats: List[Dict[str, Any]] = []
    for t in lst:
        if isinstance(t, dict) and t.get("x") is not None and t.get("z") is not None:
            threats.append({"x": float(t["x"]), "z": float(t["z"]),
                            "radius": float(t.get("radius") or 0.0), "type": t.get("type", "unknown")})
    return threats


# --------------------------------------------------------------------------- #
# 루트별 계획 산출 + 추천
# --------------------------------------------------------------------------- #

def plan_route(rid: str, poly: List[Tuple[float, float]], targets: List[Dict[str, Any]],
               bboxes: List[Dict[str, float]], exposure_threats: List[Dict[str, Any]],
               weights: Dict[str, float], terrain: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """한 루트에 대해 표적별 최적 사격 위치 + 커버리지/점수 집계."""
    pts, arc = densify_polyline(poly, CANDIDATE_STEP_M)
    per_target: List[Dict[str, Any]] = []
    for t in targets:
        fp = best_firing_position(t, pts, arc, bboxes, exposure_threats, weights, terrain)
        per_target.append({
            "id": t["id"], "class": t["class"],
            "pos": {"x": round(t["x"], 2), "y": round(t["y"], 2)},
            "z_height": t.get("z_height", 0.0),
            "firing_checkpoint": fp,          # None = 이 루트에서 교전 불가
        })
    covered = [e for e in per_target if e["firing_checkpoint"]]
    coverage = len(covered) / len(targets) if targets else 0.0
    mean_score = round(sum(e["firing_checkpoint"]["score"] for e in covered) / len(covered), 4) if covered else 0.0
    mean_exposure = round(sum(e["firing_checkpoint"]["exposure"] for e in covered) / len(covered), 3) if covered else 0.0
    # 교전 순서: 정적 표적을 루트 진행(호길이) 순으로, 최종 적전차는 **반드시 마지막**
    # (최종 적 사격 = 임무 종료라, 그 전에 정적 표적을 모두 교전해야 함).
    statics = sorted([e for e in covered if e["class"] != "final_enemy"],
                     key=lambda e: e["firing_checkpoint"]["route_arc_m"])
    finals = [e for e in covered if e["class"] == "final_enemy"]
    order = [e["id"] for e in statics] + [e["id"] for e in finals]
    # 순서 실현성: 최종 적 사격 위치보다 루트상 '뒤'에 있는 정적 표적은 최종 적을 먼저
    # 지나쳐 교전 못 함(임무 조기 종료). 그런 표적을 late_statics로 경고.
    late_statics: List[str] = []
    if finals:
        final_arc = min(e["firing_checkpoint"]["route_arc_m"] for e in finals)
        late_statics = [e["id"] for e in statics if e["firing_checkpoint"]["route_arc_m"] > final_arc]
    return {
        "route": rid,
        "covered_count": len(covered), "target_count": len(targets),
        "coverage": round(coverage, 3),
        "mean_score": mean_score, "mean_exposure": mean_exposure,
        "order_feasible": not late_statics, "late_statics": late_statics,
        "engage_order": order,
        "per_target": per_target,
    }


def recommend_route(route_results: Dict[str, Dict[str, Any]]) -> Tuple[str, str]:
    """순서실현 가능 → 커버리지 최대 → 평균점수 최대 → 평균노출 최소 순으로 추천. (rid, 사유)."""
    def key(rid: str) -> Tuple[float, float, float, float]:
        r = route_results[rid]
        return (1.0 if r["order_feasible"] else 0.0, r["coverage"], r["mean_score"], -r["mean_exposure"])
    best = max(route_results, key=key)
    # Scenario-2 launch drives route A. Keep the mission plan on A whenever A
    # can cover the same number of targets with a feasible order. This prevents
    # a slightly flatter B route from producing an inconsistent mission plan.
    pref = SCENARIO2_PREFERRED_ROUTE
    if pref in route_results and pref != best:
        pr = route_results[pref]
        br = route_results[best]
        # Scenario-2 launch normalizes the raw plan into exactly two shots
        # (nearest recon static tank + final enemy).  Therefore route_A should
        # remain the execution route whenever it covers the same number of
        # targets; do not switch to route_B just because duplicate final-area
        # detections make the raw multi-target order look infeasible.
        if pr["covered_count"] >= br["covered_count"] and pr["covered_count"] > 0:
            best = pref
    r = route_results[best]
    reason = (f"route_{best} 추천 — 표적 {r['covered_count']}/{r['target_count']} 교전가능"
              f"(coverage {r['coverage']:.2f}), 평균 사격점수 {r['mean_score']:.3f}, 평균 노출 {r['mean_exposure']:.3f}"
              f", 교전순서 {'실현가능' if r['order_feasible'] else '불가(' + ','.join(r['late_statics']) + ')' }.")
    if best == pref:
        reason += f" Scenario-2 설계축 route_{pref} 우선 정책 적용."
    others = [rid for rid in route_results if rid != best]
    for o in others:
        ro = route_results[o]
        if ro["covered_count"] < r["covered_count"]:
            reason += f" route_{o}는 표적 {ro['covered_count']}/{ro['target_count']}만 교전가능."
        elif not ro["order_feasible"]:
            reason += f" route_{o}는 교전순서 불가(최종 적 뒤 표적 {','.join(ro['late_statics'])})."
    return best, reason


# --------------------------------------------------------------------------- #
# LLM 서술(선택) — 기하 계획을 사람이 읽을 수행 지침으로
# --------------------------------------------------------------------------- #

def llm_narrate(plan: Dict[str, Any]) -> Dict[str, Any]:
    """LLMReporter.call_ollama 재사용해 미션 수행 지침을 JSON으로 생성. 실패시 available=false."""
    try:
        sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "risk_analysis"))
        from risk_analysis.llm_reporter import LLMReporter  # noqa: E402
    except Exception as exc:  # ImportError 등
        return {"available": False, "error": f"LLMReporter import 실패: {exc}"}

    reporter = LLMReporter()

    def _fc(e: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        cp = e.get("firing_checkpoint")
        if not cp:
            return None
        return {"x": cp["x"], "y": cp["y"], "distance_m": cp["distance_m"],
                "exposure_band": cp["exposure_band"],
                "terrain_roughness": cp.get("terrain_roughness"),
                "flat_score": cp.get("flat_score")}

    ctx = {
        "route_recommended": plan["route_recommended"],
        "engage_order": plan["plan"]["engage_order"],
        "targets": [
            {"id": e["id"], "class": e["class"], "pos": e["pos"], "firing_checkpoint": _fc(e)}
            for e in plan["plan"]["targets"]
        ],
    }
    prompt = (
        "너는 전차 미션 계획 참모 AI다. 아래는 각 표적의 사격 위치·거리·노출·교전순서를 기하로 계산한 미션 계획이다.\n"
        "이 계획을 실제 수행 지침으로 한국어로 간결하게 서술하라.\n"
        "규칙:\n"
        "- 반드시 JSON 객체 하나만 출력한다(마크다운/설명 금지).\n"
        "- 숫자를 새로 지어내지 말고 주어진 값만 근거로 서술하라.\n"
        "- 교전 순서는 주어진 engage_order를 그대로 따르라(최종 적전차 enemy_final은 사격 시 임무 종료라 반드시 마지막).\n"
        "출력 JSON 구조:\n"
        '{"summary":"한국어 한 문장","engage_order_reason":"교전 순서를 정한 이유(한국어)",'
        '"per_target":{"<표적id>":"이 표적 접근/사격 시 유의점(한국어)"},'
        '"cautions":["수행상 주의점(한국어)"]}\n'
        "미션 계획 데이터:\n" + json.dumps(ctx, ensure_ascii=False, indent=2)
    )
    try:
        resp = reporter.call_ollama(prompt)
        raw = resp.get("response", "") if isinstance(resp, dict) else ""
        parsed = json.loads(raw)
        parsed["available"] = True
        parsed["model"] = reporter.model_name
        return parsed
    except Exception as exc:  # 연결실패/타임아웃/파싱실패 → 기하 계획만으로 진행
        return {"available": False, "error": f"{type(exc).__name__}: {exc}", "model": reporter.model_name}


# --------------------------------------------------------------------------- #
# 출력
# --------------------------------------------------------------------------- #

def build_engagements_json(plan: Dict[str, Any]) -> str:
    """추천 루트의 사격 계획 → ballistic_turret_node engagements_json 스니펫(제안용, 자동적용 안 함)."""
    engs = []
    id_to_target = {e["id"]: e for e in plan["plan"]["targets"]}
    for order_index, tid in enumerate(plan["plan"]["engage_order"]):
        e = id_to_target[tid]
        cp = dict(e["firing_checkpoint"])
        fallback_goals = [
            {"x": float(g["x"]), "y": float(g["y"])}
            for g in (cp.get("fallback_goals") or [])
            if isinstance(g, dict) and g.get("x") is not None and g.get("y") is not None
        ]
        target_x = float(e["pos"]["x"])
        target_y = float(e["pos"]["y"])
        target_z = float(e.get("z_height", 0.0))
        target_policy = "mission_plan_target"
        # Stage 1 is the recon/static tank.  Use the verified route-A firebase
        # and target by default.  This prevents minor terrain-score changes or a
        # duplicate detected_tank near the final enemy from making stage 1 aim
        # sideways/backward from the firebase.
        if order_index == 0 and e.get("class") != "final_enemy":
            if FIRE_FORCE_STATIC_FIREBASE:
                cp["x"] = FIRE_STATIC_FIREBASE_X
                cp["y"] = FIRE_STATIC_FIREBASE_Y
                cp["selection_policy"] = "forced_static_ballistic_firebase"
                fallback_goals = [
                    {"x": float(x), "y": float(y)}
                    for x, y in FIRE_STATIC_FIREBASE_FALLBACKS
                ]
            if FIRE_FORCE_STATIC_TARGET:
                e["source_pos_before_static_target_force"] = dict(e.get("pos", {}))
                target_x = FIRE_STATIC_TARGET_X
                target_y = FIRE_STATIC_TARGET_Y
                target_z = FIRE_STATIC_TARGET_Z
                target_policy = "forced_static_center_lane_target"
        engs.append({
            "id": tid,
            "checkpoint": {"x": cp["x"], "y": cp["y"], "radius_m": FIRE_CHECKPOINT_RADIUS_M},
            "checkpoint_settle_sec": 1.2,
            "target": {"x": target_x, "y": target_y, "z": target_z},
            "target_selection_policy": target_policy,
            "target_from_enemy_pose": e["class"] == "final_enemy",
            "target_height_offset_m": 0.0,
            "reposition": {
                "enabled": bool(fallback_goals),
                "fallback_goals": fallback_goals,
                "arrival_radius_m": FIRE_CHECKPOINT_RADIUS_M,
                "min_travel_m": 0.0,
                "timeout_sec": 55.0,
                "max_attempts": len(fallback_goals),
            },
        })
    return json.dumps(engs, ensure_ascii=False)


def render_markdown(plan: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# 시나리오2 미션 계획 (자동 제안)\n")
    lines.append(f"- 생성: {plan['created_at']}")
    lines.append(f"- 무기 사거리: {plan['weapon']['min_range_m']}~{plan['weapon']['max_range_m']} m")
    lines.append(f"- **추천 루트: {plan['route_recommended']}** — {plan['route_reason']}\n")

    lines.append("## 루트별 교전 커버리지")
    lines.append("| 루트 | 교전가능 표적 | coverage | 평균 사격점수 | 평균 노출 | 교전순서 |")
    lines.append("|---|---|---|---|---|---|")
    for rid, r in plan["routes"].items():
        star = " ★" if rid == plan["route_recommended"] else ""
        order_ok = "실현가능" if r["order_feasible"] else f"불가({','.join(r['late_statics'])})"
        lines.append(f"| {rid}{star} | {r['covered_count']}/{r['target_count']} | {r['coverage']:.2f} | {r['mean_score']:.3f} | {r['mean_exposure']:.3f} | {order_ok} |")
    lines.append("")

    lines.append(f"## 추천 루트({plan['route_recommended']}) 사격 계획")
    lines.append(f"- 교전 순서: {' → '.join(plan['plan']['engage_order']) or '(없음)'}\n")
    lines.append("| # | 표적 | 종류 | 표적 좌표 | 사격 위치 | 거리(m) | 노출 |")
    lines.append("|---|---|---|---|---|---|---|")
    order_idx = {tid: i + 1 for i, tid in enumerate(plan["plan"]["engage_order"])}
    for e in plan["plan"]["targets"]:
        cp = e["firing_checkpoint"]
        pos = f"({e['pos']['x']}, {e['pos']['y']})"
        if cp:
            fp = f"({cp['x']}, {cp['y']})"
            dist = f"{cp['distance_m']}"
            band = cp["exposure_band"]
            n = order_idx.get(e["id"], "-")
        else:
            fp, dist, band, n = "**교전불가**", "-", "-", "-"
        lines.append(f"| {n} | {e['id']} | {e['class']} | {pos} | {fp} | {dist} | {band} |")
    lines.append("")

    g = plan.get("llm_guidance", {})
    lines.append("## LLM 수행 지침")
    if g.get("available"):
        lines.append(f"- 요약: {g.get('summary', '')}")
        lines.append(f"- 교전 순서 이유: {g.get('engage_order_reason', '')}")
        for tid, note in (g.get("per_target") or {}).items():
            lines.append(f"- {tid}: {note}")
        for c in (g.get("cautions") or []):
            lines.append(f"- ⚠️ {c}")
        lines.append(f"\n(모델: {g.get('model', '?')})")
    else:
        lines.append(f"- (LLM 미사용/실패: {g.get('error', 'N/A')}) — 기하 계획만 유효.")
    lines.append("")
    lines.append("> routes/engagements는 의도 설계물이라 자동 적용하지 않음. 검토 후 수동 반영(--emit-engagements).")
    return "\n".join(lines) + "\n"


def verify_plan(plan: Dict[str, Any], bboxes: List[Dict[str, float]]) -> List[str]:
    """각 사격 체크포인트가 LoS 뚫림 ∧ 사거리 20~130 만족하는지 재검증. 실패 메시지 리스트."""
    problems: List[str] = []
    for e in plan["plan"]["targets"]:
        cp = e["firing_checkpoint"]
        if not cp:
            continue
        tx, ty = e["pos"]["x"], e["pos"]["y"]
        d = math.hypot(cp["x"] - tx, cp["y"] - ty)
        if not (MIN_RANGE_M <= d <= MAX_RANGE_M):
            problems.append(f"{e['id']}: 거리 {d:.1f}m 사거리 밖")
        if not tg.check_los(cp["x"], cp["y"], tx, ty, bboxes):
            problems.append(f"{e['id']}: 사격 위치→표적 LoS 막힘")
    return problems


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(description="정찰 산출물 → 시나리오2 미션 계획(사격 위치·루트·순서) 제안")
    ap.add_argument("--map", default=DEFAULT_MAP, help="scenario2_map.map 경로")
    ap.add_argument("--routes", default=DEFAULT_ROUTES, help="scenario2_routes.yaml 경로")
    ap.add_argument("--risk-features", default=DEFAULT_RISK_FEATURES, help="risk_features.json 경로(노출용, 선택)")
    ap.add_argument("--terrain", default=DEFAULT_TERRAIN, help="scenario2_terrain.json 경로(평탄 사격점 우선)")
    ap.add_argument("--route", default=None, help="추천 대신 이 루트로 강제(A/B)")
    ap.add_argument("--final-enemy", nargs=2, type=float, metavar=("X", "Y"),
                    default=list(DEFAULT_FINAL_ENEMY_XY), help="최종 적전차 명목 좌표")
    ap.add_argument("--no-final-enemy", action="store_true", help="최종 적전차 표적 제외")
    ap.add_argument("--no-llm", action="store_true", help="LLM 서술 생략(기하 계획만)")
    ap.add_argument("--emit-engagements", action="store_true", help="ballistic engagements_json 스니펫 출력")
    ap.add_argument("--out-json", default=os.path.join(DEFAULT_REPORT_DIR, "mission_plan.json"))
    ap.add_argument("--out-md", default=os.path.join(DEFAULT_REPORT_DIR, "analysis", "mission_plan.md"))
    args = ap.parse_args()

    load_env_file()  # LLM 모델/URL을 정찰과 동일(.env의 gemma3:4b)하게

    # 입력 로딩
    final_xy = None if args.no_final_enemy else (args.final_enemy[0], args.final_enemy[1])
    targets = load_targets(args.map, final_xy)
    if not targets:
        print("[mission_plan] 표적 없음 — scenario2_map.targets 확인 필요", file=sys.stderr)
        return 2
    routes = load_route_polylines(args.routes)
    _, bboxes, _ = tg.load_map(args.map)  # LoS 차폐 = scenario2_map 전체 장애물 bbox
    terrain_grid = load_terrain_grid(args.terrain)

    # 루트별 계획
    route_results: Dict[str, Dict[str, Any]] = {}
    for rid, poly in routes.items():
        exp_threats = load_exposure_threats(args.risk_features, rid)
        route_results[rid] = plan_route(rid, poly, targets, bboxes, exp_threats, DEFAULT_WEIGHTS, terrain_grid)

    # 추천(또는 강제)
    if args.route and args.route in route_results:
        rec, reason = args.route, f"route_{args.route} 사용자 지정."
    else:
        rec, reason = recommend_route(route_results)

    rr = route_results[rec]
    plan: Dict[str, Any] = {
        "schema_version": "1.0",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "weapon": {"min_range_m": MIN_RANGE_M, "max_range_m": MAX_RANGE_M},
        "weights": DEFAULT_WEIGHTS,
        "route_recommended": rec,
        "route_reason": reason,
        "routes": route_results,
        "plan": {
            "route": rec,
            "engage_order": rr["engage_order"],
            "targets": rr["per_target"],
        },
    }

    # ballistic_turret_node engagements 형식(자동 연결용) — 사격가능 표적만, 교전순서대로.
    # tank_scenario2.launch.py가 TANK_USE_MISSION_PLAN=true일 때 이 필드를 읽어 사격 시퀀스로 쓴다.
    plan["engagements"] = json.loads(build_engagements_json(plan))
    # Keep the human-readable plan/MFD in sync with the actual engagements
    # consumed by tank_scenario2.launch.py.
    engagement_by_id = {str(e.get("id")): e for e in plan["engagements"] if isinstance(e, dict)}
    for item in plan["plan"].get("targets", []):
        eid = str(item.get("id"))
        eng = engagement_by_id.get(eid)
        if eng and isinstance(item.get("firing_checkpoint"), dict):
            item["firing_checkpoint"]["x"] = float(eng["checkpoint"]["x"])
            item["firing_checkpoint"]["y"] = float(eng["checkpoint"]["y"])
            item["firing_checkpoint"]["selection_policy"] = item["firing_checkpoint"].get("selection_policy", "")
            if eid == plan["plan"].get("engage_order", [None])[0] and FIRE_FORCE_STATIC_FIREBASE:
                item["firing_checkpoint"]["selection_policy"] = "forced_static_ballistic_firebase"
            if eid == plan["plan"].get("engage_order", [None])[0] and FIRE_FORCE_STATIC_TARGET:
                item["source_pos_before_static_target_force"] = dict(item.get("pos", {}))
                item["pos"] = {"x": FIRE_STATIC_TARGET_X, "y": FIRE_STATIC_TARGET_Y}
                item["z_height"] = FIRE_STATIC_TARGET_Z
                item["target_selection_policy"] = "forced_static_center_lane_target"

    # 검증
    problems = verify_plan(plan, bboxes)
    if rr["late_statics"]:
        problems.append(f"교전순서 불가: 정적 표적 {rr['late_statics']}의 사격 위치가 최종 적보다 루트상 뒤 — "
                        "최종 적을 먼저 지나쳐 임무 조기 종료 위험(다른 루트/사격위치 검토)")
    plan["verification"] = {"ok": not problems, "problems": problems}

    # LLM 서술
    plan["llm_guidance"] = {"available": False, "error": "skipped (--no-llm)"} if args.no_llm else llm_narrate(plan)

    # 출력
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_md), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(plan, f, ensure_ascii=False, indent=2)
    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write(render_markdown(plan))

    # 콘솔 요약
    print(f"[mission_plan] 추천 루트: {rec} ({reason})")
    for rid, r in route_results.items():
        order_ok = "순서OK" if r["order_feasible"] else f"순서불가({','.join(r['late_statics'])})"
        print(f"  route {rid}: 교전가능 {r['covered_count']}/{r['target_count']}, 평균점수 {r['mean_score']:.3f}, 평균노출 {r['mean_exposure']:.3f}, {order_ok}")
    print(f"  교전 순서: {' → '.join(rr['engage_order']) or '(없음)'}")
    for e in rr["per_target"]:
        cp = e["firing_checkpoint"]
        if cp:
            print(f"    {e['id']:20s} 표적({e['pos']['x']},{e['pos']['y']}) → 사격({cp['x']},{cp['y']}) d={cp['distance_m']}m 노출={cp['exposure_band']}")
        else:
            print(f"    {e['id']:20s} 표적({e['pos']['x']},{e['pos']['y']}) → 교전불가(사거리·LoS 불충족)")
    if problems:
        print(f"  [검증 경고] {problems}")
    else:
        print("  [검증] 모든 사격 위치 LoS·사거리 OK")
    g = plan["llm_guidance"]
    print(f"  LLM 서술: {'생성됨(' + g.get('model', '?') + ')' if g.get('available') else '미사용(' + str(g.get('error')) + ')'}")
    print(f"  → {args.out_json}\n  → {args.out_md}")

    if args.emit_engagements:
        print("\n--- engagements_json (제안, 수동 반영용) ---")
        print(build_engagements_json(plan))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
