#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""정찰 보고서 생성기 — recon_reports/*.json → A/B 위험도·은밀성 비교 + 루트 추천.

파이프라인상 위치 (ROS·시뮬 불필요, 순수 후처리):
    run_recon_scenario.py  ──(ROS 런타임)──▶  route_A.json / route_B.json / comparison.json
                                                          │
                                                          ▼
                          generate_recon_report.py  ──▶  recon_reports/recon_report.md

명세(Tank_System_Spec.pdf 6.2) 위험도 수식:
    Risk = W1·(적/초소 발견) + W2·(시야 노출시간) + W3·(우회/이탈) + W4·(지형굴곡 σP+σR) + W5·(소요시간)
  - W1: vision_yolo.counts 위협클래스(person/tank/house; rock/car 제외) 가중합 + asset_spotted_gt 보강
  - W2: finalmap.map GT 위협(초소/적전차) + 전차 궤적으로 사후 계산(노출시간·발각횟수)
  - W3: distance_m / 직선거리(start→goal) 우회비(재계획 카운트 미로깅이라 근사)
  - W4: terrain_roughness.pitch_std + roll_std
  - W5: result.sim_time_s
  값이 낮을수록 안전(은밀)한 루트. 추천은 유효 런 중 최저 위험.

사용:
    python3 scripts/generate_recon_report.py
    python3 scripts/generate_recon_report.py --input recon_reports --stdout
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from recon_eval import threat_geometry as tg  # noqa: E402

# --------------------------------------------------------------------------- #
# 상수
# --------------------------------------------------------------------------- #
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_REPORT_DIR = os.path.join(PROJECT_ROOT, "recon_reports")
DEFAULT_MAP = os.path.join(PROJECT_ROOT, "src", "rviz_visualization", "map", "finalmap.map")
DEFAULT_ROUTES = os.path.join(PROJECT_ROOT, "src", "path_planning", "config", "routes.yaml")
# 정찰이 perceive한 위협(센서퓨전 확정 발견객체)의 출처 — 위험도는 이걸로 산정.
DEFAULT_DISCOVERED_DIR = os.path.join(DEFAULT_REPORT_DIR, "recon_map")
# 정답맵(GT) — 위험도엔 안 쓰고, "정찰이 얼마나 정확/충분히 찾았나" 검증에만.
DEFAULT_GT_MAP = os.path.join(PROJECT_ROOT, "src", "rviz_visualization", "map", "final_v3.map")

# 출발/목적 (routes.yaml finalmap 기준). 우회비 직선거리 계산 + 중심선 폴백에 사용.
START_XY = (60.0, 30.0)
GOAL_XY = (110.0, 276.5)

# 위험도 가중치 — perception 기반 3요소(은밀성/위협근접/험지). 합 1.0.
# 시간/거리/우회는 '위험'이 아니라 '효율'이라 위험도에서 분리(별도 표기).
DEFAULT_WEIGHTS = {"stealth": 0.5, "proximity": 0.3, "terrain": 0.2}

# YOLO 위협클래스별 가중(정보 섹션 표기용 — 위험도 점수엔 미반영).
# YOLO 고카운트는 중복 탐지라, 위치 위험도는 센서퓨전 확정 발견객체로만 산정한다.
THREAT_CLASS_WEIGHTS = {"person": 1.0, "tank": 1.0, "house": 1.0}

# 발견객체맵(perception)에서 위협으로 볼 class → 탐지 반경(전방향 가정).
# perception은 객체 heading을 모르므로 House FOV 콘 대신 반경+LoS로 보수적 판정.
# person은 fix/fusion에서 ignored 처리라 발견맵엔 안 들어옴.
PERCEIVED_THREAT_RADII = {"house": tg.HOUSE_RADIUS_M, "tank": tg.TANK_RADIUS_M}

# reference 정규화 기준값. terrain만 위험도에 사용(은밀성/근접은 이미 0~1 길이비).
REFS = {
    "terrain": 8.0,     # 험지 σPitch+σRoll(deg)가 이 값이면 norm=1.0
}

# 데이터 품질 임계
MIN_VALID_DISTANCE = 5.0   # m 미만 이동이면 무효 런(멈춤/미완주)
COLLISION_WARN = 10        # 초과 시 충돌 과다 경고
PERCEPTION_MATCH_TOL = 3.0  # m, LiDAR 탐지↔GT 객체 최근접 매칭 허용오차
THREAT_MATCH_TOL = 10.0     # m, 발견 위협↔GT 위협 매칭 허용오차(초소/전차는 큰 객체라 centroid 오차 큼)

# YOLO 클래스 → GT prefab 접두사 매핑(센서 정확도 비교용)
YOLO_TO_GT_PREFAB = {"person": "Human", "tank": "Tank", "rock": "Rock", "house": "House", "car": "Car"}

EXIT_OK = 0
EXIT_NO_INPUT = 2
EXIT_NO_VALID_RUN = 3
EXIT_SINGLE_ROUTE = 4


# --------------------------------------------------------------------------- #
# 입력 로드
# --------------------------------------------------------------------------- #
def load_inputs(input_path: str, route_a: Optional[str], route_b: Optional[str]) -> Dict[str, dict]:
    """comparison.json 우선, 없으면 route_A/B.json 개별 로드. {'A':..,'B':..}."""
    routes: Dict[str, dict] = {}

    if route_a or route_b:
        for rid, path in (("A", route_a), ("B", route_b)):
            if path and os.path.isfile(path):
                routes[rid] = _read_json(path)
        return routes

    # input_path: 디렉터리 또는 comparison.json 파일
    comparison = input_path if input_path.endswith(".json") else os.path.join(input_path, "comparison.json")
    if os.path.isfile(comparison):
        data = _read_json(comparison)
        for key, rid in (("route_A", "A"), ("route_B", "B")):
            r = data.get(key)
            if isinstance(r, dict) and r:
                routes[rid] = r
        if routes:
            return routes

    # 폴백: route_A.json / route_B.json
    base = input_path if os.path.isdir(input_path) else os.path.dirname(input_path)
    for rid in ("A", "B"):
        p = os.path.join(base, f"route_{rid}.json")
        if os.path.isfile(p):
            routes[rid] = _read_json(p)
    return routes


def _read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# 지표 추출
# --------------------------------------------------------------------------- #
def compute_threat_proxy(yolo_counts: Dict[str, int]) -> Tuple[float, Dict[str, float]]:
    """vision_yolo.counts → 위협클래스 가중합 + 항목별 분해."""
    breakdown: Dict[str, float] = {}
    total = 0.0
    for cls, w in THREAT_CLASS_WEIGHTS.items():
        n = int(yolo_counts.get(cls, 0))
        if n:
            breakdown[cls] = round(n * w, 2)
            total += n * w
    return round(total, 2), breakdown


def _traj_points(report: dict) -> Optional[List[Tuple[float, float, float]]]:
    """report['trajectory'] → [(t,x,z)] 또는 None. dict/list 양식 모두 허용."""
    traj = report.get("trajectory")
    if not isinstance(traj, list) or len(traj) < 2:
        return None
    pts: List[Tuple[float, float, float]] = []
    for item in traj:
        try:
            if isinstance(item, dict):
                pts.append((float(item["t"]), float(item["x"]), float(item["z"])))
            elif isinstance(item, (list, tuple)) and len(item) >= 3:
                pts.append((float(item[0]), float(item[1]), float(item[2])))
        except (KeyError, TypeError, ValueError):
            continue
    return pts if len(pts) >= 2 else None


def extract_metrics(report: dict) -> dict:
    result = report.get("result", {})
    obstacle = report.get("obstacle_summary", {})
    terrain = report.get("terrain_roughness", {})
    yolo_counts = report.get("vision_yolo", {}).get("counts", {}) or {}

    distance = float(result.get("distance_m", 0.0))
    obstacle_count = int(obstacle.get("count", 0))
    # density는 원본의 0-division 잔재가 있어 직접 재계산(거리 유효할 때만).
    density = round(obstacle_count / (distance / 100.0), 2) if distance >= MIN_VALID_DISTANCE else None

    threat_proxy, threat_breakdown = compute_threat_proxy(yolo_counts)
    pitch = float(terrain.get("pitch_std_deg", 0.0))
    roll = float(terrain.get("roll_std_deg", 0.0))

    return {
        "route": str(report.get("route", "?")),
        "map": str(report.get("map", "")),
        "reached": bool(result.get("reached", False)),
        "distance_m": round(distance, 2),
        "sim_time_s": round(float(result.get("sim_time_s", 0.0)), 2),
        "collisions": int(result.get("collisions", 0)),
        "obstacle_count": obstacle_count,
        "obstacle_density": density,
        "terrain_sigma": round(pitch + roll, 4),
        "terrain_pitch": pitch,
        "terrain_roll": roll,
        "yolo_counts": dict(yolo_counts),
        "threat_proxy": threat_proxy,
        "threat_breakdown": threat_breakdown,
        "asset_gt": report.get("asset_spotted_gt", {}) or {},
        "detour_ratio": _detour_ratio(distance),
        "obstacles_detected": report.get("obstacles_detected", []) or [],
        "trajectory": _traj_points(report),
    }


def _detour_ratio(distance: float) -> Optional[float]:
    straight = math.hypot(GOAL_XY[0] - START_XY[0], GOAL_XY[1] - START_XY[1])
    if straight <= 1e-6 or distance < MIN_VALID_DISTANCE:
        return None
    return round(distance / straight, 3)


# --------------------------------------------------------------------------- #
# perception(발견객체맵) 위협 파싱 — 위험도 산정의 위협 출처
# --------------------------------------------------------------------------- #
def parse_perceived_threats(discovered_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """발견객체맵 dict → 위협 리스트 [{type, x, z, radius, prefabName}].

    센서퓨전으로 확정된 house/tank만 위협으로. metadata.class_name 우선,
    없으면 prefab 'detected_<class>_' 에서 class 유도. heading은 모르므로 미포함(전방향).
    """
    threats: List[Dict[str, Any]] = []
    for obs in discovered_data.get("obstacles", []):
        if not isinstance(obs, dict):
            continue
        meta = obs.get("metadata") if isinstance(obs.get("metadata"), dict) else {}
        cls = str(meta.get("class_name", "")).lower()
        prefab = str(obs.get("prefabName", ""))
        if not cls and prefab.startswith("detected_"):
            parts = prefab.split("_")
            cls = parts[1].lower() if len(parts) > 1 else ""
        if cls not in PERCEIVED_THREAT_RADII:
            continue
        pos = obs.get("position") if isinstance(obs.get("position"), dict) else {}
        try:
            x, z = float(pos.get("x", 0.0)), float(pos.get("z", 0.0))
        except (TypeError, ValueError):
            continue
        threats.append({"type": cls, "x": x, "z": z,
                        "radius": PERCEIVED_THREAT_RADII[cls], "prefabName": prefab})
    return threats


# --------------------------------------------------------------------------- #
# 클린 루트 중심선 추출 + densify (planned_paths v0, 폴백 routes.yaml)
# --------------------------------------------------------------------------- #
def extract_centerline(report: dict, rid: str, routes_path: str) -> Optional[List[Tuple[float, float]]]:
    """그 루트를 임무에서 따라갈 '깨끗한 중심선' [(x,z)]. weave 궤적이 아님.

    출처: diagnostics.planned_paths 의 v0(version/t 최소 = start→전 waypoint→goal 전체 루트).
    없으면 routes.yaml 폴백. 좌표계는 map (x,z) = 궤적/위협과 동일.
    """
    diag = report.get("diagnostics") if isinstance(report.get("diagnostics"), dict) else {}
    planned = diag.get("planned_paths") if isinstance(diag.get("planned_paths"), list) else []
    best: Optional[Tuple[float, list]] = None
    for pp in planned:
        if not isinstance(pp, dict):
            continue
        path = pp.get("path")
        if not isinstance(path, list) or len(path) < 2:
            continue
        ver = pp.get("version", pp.get("t", 0.0))
        try:
            ver = float(ver)
        except (TypeError, ValueError):
            ver = 0.0
        if best is None or ver < best[0]:
            best = (ver, path)
    if best is not None:
        pts: List[Tuple[float, float]] = []
        for p in best[1]:
            if isinstance(p, (list, tuple)) and len(p) >= 2:
                try:
                    pts.append((float(p[0]), float(p[1])))
                except (TypeError, ValueError):
                    continue
        if len(pts) >= 2:
            return pts
    return _centerline_from_routes(rid, routes_path)


def _centerline_from_routes(rid: str, routes_path: str) -> Optional[List[Tuple[float, float]]]:
    """폴백: routes.yaml 의 start→waypoints→destination 폴리라인."""
    try:
        import yaml as _yaml
        with open(routes_path, encoding="utf-8") as f:
            rd = _yaml.safe_load(f)["finalmap"]
        poly = [tuple(rd["start"])] + [tuple(p) for p in rd["routes"].get(rid, [])] + [tuple(rd["destination"])]
        pts = [(float(x), float(z)) for x, z in poly]
        return pts if len(pts) >= 2 else None
    except Exception:  # pragma: no cover - 폴백 실패시 노출 N/A
        return None


def densify_polyline(poly: List[Tuple[float, float]], step: float = 0.75) -> List[Tuple[float, float]]:
    """폴리라인을 ~step(m) 간격으로 균등 보간. 끝점 보존."""
    if not poly or len(poly) < 2:
        return [tuple(p) for p in (poly or [])]
    out: List[Tuple[float, float]] = [tuple(poly[0])]
    for i in range(1, len(poly)):
        x0, z0 = poly[i - 1]
        x1, z1 = poly[i]
        d = math.hypot(x1 - x0, z1 - z0)
        n = max(1, int(d / step))
        for k in range(1, n + 1):
            t = k / n
            out.append((x0 + (x1 - x0) * t, z0 + (z1 - z0) * t))
    return out


# --------------------------------------------------------------------------- #
# 길이 기반 노출(은밀성)·근접 — 클린 중심선 × perception 위협
# --------------------------------------------------------------------------- #
def compute_centerline_exposure(centerline: Optional[List[Tuple[float, float]]],
                                threats: List[Dict[str, Any]],
                                los_obstacles: List[Dict[str, float]]) -> Optional[dict]:
    """중심선을 따라 길이 기반 노출/근접 산출(속도 무관, weave 무관).

    - stealth(은밀성): 어느 위협의 반경 내 **AND LoS 트임**(실제로 보임) 구간 길이 비율.
    - proximity(위협근접): 반경 내(LoS 무관, 막혀도 물리적 근접) 구간 길이 비율.
    perception 위협엔 heading이 없어 FOV 콘은 적용 안 함(반경+LoS, 보수적).
    반환 None = 중심선 미수집. 위협 0개면 비율 0으로 정상 반환.
    """
    if not centerline or len(centerline) < 2:
        return None
    pts = densify_polyline(centerline, 0.75)
    total_len = 0.0
    exposed_len = 0.0    # 반경+LoS (any threat)
    proximity_len = 0.0  # 반경 (any threat)
    stats = [{"exposed": 0.0, "within": 0.0, "min_dist": float("inf")} for _ in threats]

    for i in range(1, len(pts)):
        x0, z0 = pts[i - 1]
        x1, z1 = pts[i]
        seg = math.hypot(x1 - x0, z1 - z0)
        if seg <= 0:
            continue
        mx, mz = 0.5 * (x0 + x1), 0.5 * (z0 + z1)
        total_len += seg
        seg_exposed = False
        seg_within = False
        for ti, th in enumerate(threats):
            r = float(th.get("radius") or tg.threat_radius(th))
            d = math.hypot(mx - float(th["x"]), mz - float(th["z"]))
            if d <= r:
                seg_within = True
                stats[ti]["within"] += seg
                if d < stats[ti]["min_dist"]:
                    stats[ti]["min_dist"] = d
                if tg.check_los(mx, mz, float(th["x"]), float(th["z"]), los_obstacles):
                    seg_exposed = True
                    stats[ti]["exposed"] += seg
        if seg_exposed:
            exposed_len += seg
        if seg_within:
            proximity_len += seg

    per_threat: List[dict] = []
    for ti, th in enumerate(threats):
        st = stats[ti]
        if st["within"] > 0:
            per_threat.append({
                "threat": th.get("prefabName") or f"{th.get('type', 'threat')}#{ti}",
                "type": th.get("type", "unknown"),
                "exposed_length_m": round(st["exposed"], 2),
                "within_radius_length_m": round(st["within"], 2),
                "min_dist_m": round(st["min_dist"], 2) if st["min_dist"] != float("inf") else None,
            })
    per_threat.sort(key=lambda e: e["exposed_length_m"], reverse=True)
    return {
        "stealth_ratio": round(exposed_len / total_len, 4) if total_len > 0 else 0.0,
        "proximity_ratio": round(proximity_len / total_len, 4) if total_len > 0 else 0.0,
        "exposed_length_m": round(exposed_len, 2),
        "proximity_length_m": round(proximity_len, 2),
        "total_length_m": round(total_len, 2),
        "threat_count": len(threats),
        "per_threat": per_threat,
    }


# --------------------------------------------------------------------------- #
# 정찰 정확도 검증 — perception 위협 vs GT(정답맵). 위험도엔 미반영, 신뢰도 지표.
# --------------------------------------------------------------------------- #
def _gt_family(g: Dict[str, Any]) -> str:
    t = str(g.get("type", "")) or str(g.get("prefabName", ""))
    if t.startswith("House"):
        return "house"
    if t.startswith("Tank"):
        return "tank"
    return t.lower()


def validate_perception_vs_gt(perceived: List[Dict[str, Any]], gt_threats: List[Dict[str, Any]],
                              tol: float = THREAT_MATCH_TOL) -> dict:
    """발견 위협 ↔ GT 위협 최근접(동일 class family) 매칭 → 발견/누락/오탐/위치오차/신뢰도."""
    from collections import Counter
    matched_gt: set = set()
    pos_errs: List[float] = []
    false_pos = 0
    for pt in perceived:
        fam = str(pt.get("type", ""))
        best_i, best_d = -1, float("inf")
        for gi, g in enumerate(gt_threats):
            if gi in matched_gt or _gt_family(g) != fam:
                continue
            d = math.hypot(float(pt["x"]) - float(g["x"]), float(pt["z"]) - float(g["z"]))
            if d < best_d:
                best_d, best_i = d, gi
        if best_i >= 0 and best_d <= tol:
            matched_gt.add(best_i)
            pos_errs.append(best_d)
        else:
            false_pos += 1
    gt_n = len(gt_threats)
    found = len(matched_gt)
    gt_fam = Counter(_gt_family(g) for g in gt_threats)
    pc_fam = Counter(str(p.get("type", "")) for p in perceived)
    families = sorted(set(gt_fam) | set(pc_fam))
    return {
        "gt_total": gt_n,
        "perceived_total": len(perceived),
        "found": found,
        "missed": gt_n - found,
        "false_pos": false_pos,
        "mean_pos_err_m": round(sum(pos_errs) / len(pos_errs), 2) if pos_errs else None,
        "confidence": round(found / gt_n, 3) if gt_n else None,
        "by_family": [{"family": f, "gt": int(gt_fam.get(f, 0)), "perceived": int(pc_fam.get(f, 0))}
                      for f in families],
    }


# --------------------------------------------------------------------------- #
# (legacy) weave 궤적 기반 노출 — 정보/참고용. 위험도엔 미사용(중심선 노출로 대체).
# --------------------------------------------------------------------------- #
def compute_exposure(trajectory: Optional[List[Tuple[float, float, float]]],
                     threats: List[Dict[str, Any]],
                     gt_bboxes: List[Dict[str, float]]) -> Optional[dict]:
    """궤적 각 샘플에 is_threat_active 적용 → 노출시간/발각횟수/최소거리.

    반환 None = 궤적 미수집(N/A). 위협 0개면 0 노출로 반환.
    """
    if not trajectory:
        return None

    per_threat: List[dict] = []
    total_fov_dwell = 0.0
    total_prox_dwell = 0.0
    total_detections = 0
    global_max_continuous = 0.0

    for ti, threat in enumerate(threats):
        label = threat.get("prefabName") or f"{threat.get('type','threat')}#{ti}"
        radius = tg.threat_radius(threat)
        fov_dwell = 0.0
        prox_dwell = 0.0
        detections = 0
        max_continuous = 0.0
        cur_continuous = 0.0
        min_dist = float("inf")
        prev_active = False

        for i in range(1, len(trajectory)):
            t0, x0, z0 = trajectory[i - 1]
            t1, x1, z1 = trajectory[i]
            dt = max(0.0, t1 - t0)

            active = tg.is_threat_active((x1, z1), threat, gt_bboxes)
            dist = math.hypot(x1 - float(threat["x"]), z1 - float(threat["z"]))
            within_radius = dist <= radius

            if active:
                fov_dwell += dt
                if dist < min_dist:
                    min_dist = dist
                if not prev_active:
                    detections += 1          # 발각 진입 에지
                    cur_continuous = 0.0
                cur_continuous += dt
                max_continuous = max(max_continuous, cur_continuous)
            else:
                cur_continuous = 0.0
            if within_radius:
                prox_dwell += dt
            prev_active = active

        if fov_dwell > 0 or prox_dwell > 0:
            per_threat.append({
                "threat": label,
                "type": threat.get("type", "unknown"),
                "detections": detections,
                "fov_dwell_s": round(fov_dwell, 2),
                "proximity_dwell_s": round(prox_dwell, 2),
                "max_continuous_s": round(max_continuous, 2),
                "min_dist_m": round(min_dist, 2) if min_dist != float("inf") else None,
            })
        total_fov_dwell += fov_dwell
        total_prox_dwell += prox_dwell
        total_detections += detections
        global_max_continuous = max(global_max_continuous, max_continuous)

    per_threat.sort(key=lambda e: e["fov_dwell_s"], reverse=True)
    return {
        "total_fov_dwell_s": round(total_fov_dwell, 2),
        "total_proximity_dwell_s": round(total_prox_dwell, 2),
        "detection_count": total_detections,
        "max_continuous_s": round(global_max_continuous, 2),
        "per_threat": per_threat,
    }


# --------------------------------------------------------------------------- #
# 검증 / 정규화 / 점수
# --------------------------------------------------------------------------- #
def validate_run(m: dict) -> Tuple[bool, List[str]]:
    warnings: List[str] = []
    valid = True
    if m["distance_m"] < MIN_VALID_DISTANCE:
        valid = False
        warnings.append(f"무효 런: 이동거리 {m['distance_m']}m (< {MIN_VALID_DISTANCE}m) — 멈춤/미완주로 추천 제외")
    if not m["reached"] and valid:
        warnings.append("미도착(reached=false) — 유효 데이터지만 완주 실패")
    if m["collisions"] > COLLISION_WARN:
        warnings.append(f"충돌 과다: {m['collisions']}회 — 주행 품질 낮음")
    if not m["yolo_counts"] and m["distance_m"] >= MIN_VALID_DISTANCE:
        warnings.append("비전(YOLO) 미수집 — 정찰 인지 데이터 부족")
    return valid, warnings


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def normalize(metrics_by_route: Dict[str, dict], exposures: Dict[str, Optional[dict]],
              mode: str) -> Dict[str, dict]:
    """루트별 위험 3요소 raw값 추출 후 0~1 정규화. None 항목은 available=False.

    은밀성/위협근접은 이미 0~1 길이비(중심선 노출), 험지만 REF 정규화.
    """
    raw: Dict[str, Dict[str, Optional[float]]] = {}
    for rid, m in metrics_by_route.items():
        exp = exposures.get(rid)
        raw[rid] = {
            "stealth": (exp["stealth_ratio"] if exp else None),
            "proximity": (exp["proximity_ratio"] if exp else None),
            "terrain": m["terrain_sigma"],
        }

    out: Dict[str, dict] = {rid: {} for rid in metrics_by_route}
    keys = ["stealth", "proximity", "terrain"]
    for key in keys:
        vals = {rid: raw[rid][key] for rid in raw}
        avail = {rid: v for rid, v in vals.items() if v is not None}
        for rid in raw:
            v = vals[rid]
            if v is None:
                out[rid][key] = {"raw": None, "norm": None, "available": False}
                continue
            out[rid][key] = {"raw": round(v, 3), "norm": _norm_value(key, v, avail, mode), "available": True}
    return out


def _norm_value(key: str, v: float, avail_vals: Dict[str, float], mode: str) -> float:
    if mode == "minmax":
        lo, hi = min(avail_vals.values()), max(avail_vals.values())
        if hi - lo < 1e-9:
            return 0.5
        return round(_clamp01((v - lo) / (hi - lo)), 3)
    # reference: 은밀성/근접은 이미 0~1 비율, 험지만 σ/REF
    if key in ("stealth", "proximity"):
        return round(_clamp01(v), 3)
    ref = REFS.get(key, 1.0)
    return round(_clamp01(v / ref) if ref > 0 else 0.5, 3)


def risk_score(norm_route: Dict[str, dict], weights: Dict[str, float]) -> Tuple[float, List[dict]]:
    """가중합 위험도 총점 + 항목별 분해 (은밀성/위협근접/험지)."""
    keymap = [("stealth", "stealth", "은밀성(시야 노출 길이비)"),
              ("proximity", "proximity", "위협 근접(반경내 길이비)"),
              ("terrain", "terrain", "험지(지형 굴곡 σ)")]
    total = 0.0
    breakdown: List[dict] = []
    for key, wkey, label in keymap:
        cell = norm_route[key]
        w = weights[wkey]
        if cell["available"] and cell["norm"] is not None:
            contrib = round(w * cell["norm"], 4)
            total += contrib
        else:
            contrib = None
        breakdown.append({"key": wkey, "label": label, "weight": w,
                          "raw": cell["raw"], "norm": cell["norm"],
                          "contrib": contrib, "available": cell["available"]})
    return round(total, 4), breakdown


def recommend(scored: Dict[str, dict], gt_validation: Optional[Dict[str, dict]] = None,
              tie_eps: float = 0.05) -> dict:
    """위험도(perception 기반) 최저 루트 추천. 승자는 위험도로만 정하되, GT 검증상
    인지 신뢰도가 낮으면(위협 누락 多) 추천 신뢰도를 강등하고 재정찰을 경고한다.
    (GT로 승자를 바꾸지 않음 — 위험도는 perception, GT는 신뢰도 캡 용도.)"""
    valid = {rid: s for rid, s in scored.items() if s["valid"]}
    if not valid:
        return {"winner": None, "confidence": "none",
                "reason": "유효 런 없음 — 모든 루트가 무효(멈춤/미완주). 깨끗한 A/B 재주행 필요."}
    ranked = sorted(valid.items(), key=lambda kv: kv[1]["risk_total"])
    winner = ranked[0][0]
    if len(valid) == 1:
        result = {"winner": winner, "confidence": "low",
                  "reason": f"유효 루트가 route_{winner} 하나뿐 — 비교군 부재(상대 루트 무효)로 단독 평가."}
    else:
        second = ranked[1]
        gap = second[1]["risk_total"] - ranked[0][1]["risk_total"]
        if gap < tie_eps:
            result = {"winner": winner, "confidence": "low",
                      "reason": f"위험도 차 {gap:.3f} < {tie_eps} — 사실상 동률. 보조지표(충돌/지형)로 신중 판단 필요."}
        else:
            result = {"winner": winner, "confidence": "medium",
                      "reason": f"route_{winner} 위험도 최저({ranked[0][1]['risk_total']:.3f} vs {second[1]['risk_total']:.3f}, 차 {gap:.3f})."}

    # 인지 신뢰도 캡: 승자의 GT 검증 신뢰도가 낮거나 발견 위협 0이면 강등 + 재정찰 경고.
    # (perception이 위협을 놓쳐 위험도가 과소평가된 '거짓 안전'일 수 있음.)
    if gt_validation and winner in gt_validation:
        gv = gt_validation[winner]
        gconf = gv.get("confidence")
        if gv.get("perceived_total", 0) == 0 or (gconf is not None and gconf < 0.5):
            result["confidence"] = "low"
            result["reason"] += (f" ⚠️ 단, route_{winner} 정찰 인지 신뢰도 낮음"
                                 f"(GT {gv.get('gt_total', 0)}개 중 {gv.get('found', 0)}개 발견) — "
                                 "perception 기반 위험도가 위협 누락으로 과소평가됐을 수 있어 **재정찰 권장**.")
    return result


# --------------------------------------------------------------------------- #
# 센서 정확도(GT vs 탐지)
# --------------------------------------------------------------------------- #
def compute_perception_accuracy(m: dict, gt_objects: List[Dict[str, Any]]) -> dict:
    """LiDAR 탐지 centroid ↔ GT 정적객체 최근접 매칭 + YOLO 카운트 vs GT."""
    detected = m["obstacles_detected"]
    det_pts: List[Tuple[float, float]] = []
    for d in detected:
        try:
            det_pts.append((float(d["x"]), float(d["z"])))
        except (KeyError, TypeError, ValueError):
            continue

    gt_pts = [(o["x"], o["z"]) for o in gt_objects]
    matched_det = 0
    sq_errs: List[float] = []
    covered_gt = set()
    for (dx, dz) in det_pts:
        best_i, best_d = -1, float("inf")
        for gi, (gx, gz) in enumerate(gt_pts):
            dd = math.hypot(dx - gx, dz - gz)
            if dd < best_d:
                best_d, best_i = dd, gi
        if best_i >= 0 and best_d <= PERCEPTION_MATCH_TOL:
            matched_det += 1
            covered_gt.add(best_i)
            sq_errs.append(best_d * best_d)

    n_det = len(det_pts)
    n_gt = len(gt_pts)
    precision = round(matched_det / n_det, 3) if n_det else None
    recall = round(len(covered_gt) / n_gt, 3) if n_gt else None
    rmse = round(math.sqrt(sum(sq_errs) / len(sq_errs)), 3) if sq_errs else None

    # YOLO 카운트 vs GT(정적) — 카운트는 탐지 이벤트라 1:1 아님(참고용)
    from collections import Counter
    gt_prefix = Counter()
    for o in gt_objects:
        name = o["prefabName"]
        for yolo_cls, prefix in YOLO_TO_GT_PREFAB.items():
            if name.startswith(prefix):
                gt_prefix[yolo_cls] += 1
    yolo_vs_gt = []
    for cls in YOLO_TO_GT_PREFAB:
        yolo_vs_gt.append({"class": cls,
                           "yolo_detections": int(m["yolo_counts"].get(cls, 0)),
                           "gt_static": int(gt_prefix.get(cls, 0))})

    return {
        "lidar": {"detections": n_det, "gt_static_objects": n_gt,
                  "matched": matched_det, "covered_gt": len(covered_gt),
                  "precision": precision, "recall": recall, "position_rmse_m": rmse},
        "yolo_vs_gt": yolo_vs_gt,
    }


# --------------------------------------------------------------------------- #
# 렌더링
# --------------------------------------------------------------------------- #
def _setup_korean_font() -> None:
    """matplotlib 한글 라벨이 □로 깨지지 않게 Noto Sans CJK 등록(없으면 폴백)."""
    from matplotlib import font_manager
    import matplotlib.pyplot as plt
    for cjk in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"):
        if os.path.exists(cjk):
            try:
                font_manager.fontManager.addfont(cjk)
                plt.rcParams["font.family"] = font_manager.FontProperties(fname=cjk).get_name()
                break
            except Exception:
                pass
    plt.rcParams["axes.unicode_minus"] = False


def render_exposure_figure(rid: str,
                           centerline: Optional[List[Tuple[float, float]]],
                           trajectory: Optional[List[Tuple[float, float, float]]],
                           threats: List[Dict[str, Any]],
                           los_obstacles: List[Dict[str, float]],
                           gt_threats: List[Dict[str, Any]],
                           exposure: Optional[dict],
                           out_dir: str) -> Optional[str]:
    """채점한 클린 중심선 + 발견 위협(전방향 반경) + 중심선 노출 구간을 PNG로 그린다.

    - 초록 선: 채점한 클린 중심선(planned_paths v0). 옅은 파랑: 실제 weave 궤적(참고).
    - 빨강 원: 발견 위협 반경(perception, heading 없음→전방향). 회색 ◇: GT 위협(검증용 맥락).
    - 빨강 점: 중심선에서 반경+LoS로 노출된 구간(보고서 stealth_ratio와 동일 판정).
    matplotlib이 없거나 중심선 미수집이면 None을 반환(보고서는 그림 없이 계속).
    """
    if not centerline or len(centerline) < 2:
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle
    except Exception as e:  # pragma: no cover - 환경 의존
        print(f"[경고] matplotlib 미설치 — 노출 지도 생략: {e}", file=sys.stderr)
        return None
    _setup_korean_font()

    cl = densify_polyline(centerline, 0.75)
    cxs = [p[0] for p in cl]
    czs = [p[1] for p in cl]

    fig, ax = plt.subplots(figsize=(10, 10), dpi=130)

    # 1) GT 위협(검증 맥락) — 회색 빈 다이아몬드(위험도엔 미반영)
    for g in gt_threats:
        ax.scatter([float(g["x"])], [float(g["z"])], marker="D", s=55, facecolors="none",
                   edgecolors="#888888", linewidths=1.2, zorder=3)

    # 2) 발견 위협(perception) 반경 — heading 없음 → 전방향 원
    for th in threats:
        tx, tz = float(th["x"]), float(th["z"])
        r = float(th.get("radius") or tg.threat_radius(th))
        ax.add_patch(Circle((tx, tz), r, facecolor="#d62728", alpha=0.12,
                            edgecolor="#d62728", lw=0.8, zorder=2))
        ax.scatter([tx], [tz], marker="s", s=70, facecolors="none",
                   edgecolors="#d62728", linewidths=1.6, zorder=5)
        ax.annotate(str(th.get("prefabName", "") or th.get("type", "")), (tx, tz), fontsize=6,
                    color="#a01b1c", xytext=(3, 3), textcoords="offset points", zorder=6)

    # 3) 실제 weave 궤적(옅은 파랑, 참고)
    if trajectory and len(trajectory) >= 2:
        ax.plot([float(p[1]) for p in trajectory], [float(p[2]) for p in trajectory],
                "-", color="#9ec5f0", lw=1.2, alpha=0.8, zorder=3, label="실제 주행(weave, 참고)")

    # 4) 채점한 클린 중심선(초록)
    ax.plot(cxs, czs, "-", color="#1a9850", lw=2.2, alpha=0.95, zorder=4, label="클린 중심선(채점)")

    # 5) 중심선 노출 구간(빨강) — 반경+LoS, 보고서 stealth_ratio와 동일 판정
    ex, ez = [], []
    for (mx, mz) in cl:
        for th in threats:
            r = float(th.get("radius") or tg.threat_radius(th))
            if math.hypot(mx - float(th["x"]), mz - float(th["z"])) <= r and \
               tg.check_los(mx, mz, float(th["x"]), float(th["z"]), los_obstacles):
                ex.append(mx)
                ez.append(mz)
                break
    if ex:
        ax.scatter(ex, ez, s=22, color="#d62728", edgecolors="black", linewidths=0.4,
                   zorder=7, label=f"노출 구간({len(ex)})")

    # 6) 출발/목적
    ax.scatter([cxs[0]], [czs[0]], marker="o", s=130, color="#1a9850",
               edgecolors="black", zorder=8, label="출발")
    ax.scatter([GOAL_XY[0]], [GOAL_XY[1]], marker="^", s=150, color="black",
               edgecolors="white", zorder=8, label="목적지")

    stealth = exposure["stealth_ratio"] if exposure else 0.0
    prox = exposure["proximity_ratio"] if exposure else 0.0
    ax.set_title(f"route_{rid} 중심선·위협 노출 — 은밀성 {stealth:.3f} · 근접 {prox:.3f}", fontsize=11)
    ax.set_xlim(0, 300)
    ax.set_ylim(0, 300)
    ax.set_aspect("equal")
    ax.set_xlabel("map x (동→) [m]")
    ax.set_ylabel("map z (북↑, 위쪽이 목적지) [m]")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    out = os.path.join(out_dir, f"exposure_{rid}.png")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def render_route_overview(map_path: str, routes_path: str, out_dir: str) -> Optional[str]:
    """A/B '계획 경로'를 새 맵 위에 1장으로 그린다(scripts/visualize_routes.py 재사용).

    실주행 노출 지도(5장)와 달리 '설계된' 전역경로를 맥락으로 보여준다(런타임과 동일하게
    웨이포인트+목적지 A*). 모듈/맵 로드 실패 시 None을 반환(보고서는 그림 없이 계속).
    """
    try:
        import yaml as _yaml
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))          # scripts/
        sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "path_planning"))   # team_path_planning
        from path_planning.team_path_planning import load_static_obstacles_from_map
        import visualize_routes as vr
    except Exception as e:  # pragma: no cover - 환경 의존
        print(f"[경고] 계획 경로 지도 생략(모듈 로드 실패): {e}", file=sys.stderr)
        return None
    try:
        planning_obs = load_static_obstacles_from_map(map_path)
        raw_objs = vr.load_raw_objects(map_path)
        with open(routes_path, encoding="utf-8") as f:
            rd = _yaml.safe_load(f)["finalmap"]
        start = tuple(rd["start"])
        goal = tuple(rd["destination"])
        routes_paths: Dict[str, Any] = {}
        wps: Dict[str, Any] = {}
        for rid, side, color in (("A", "west", "#1f77b4"), ("B", "east", "#2ca02c")):
            wp = [tuple(p) for p in rd["routes"][rid]]
            wps[rid] = wp
            routes_paths[rid] = (vr.plan_route(start, wp, goal, side, planning_obs, 5.0, 0.4), color)
        out = os.path.join(out_dir, "route_overlay.png")
        vr.render_overlay(out, planning_obs, raw_objs, routes_paths, start, goal, wps)
        return out
    except Exception as e:  # pragma: no cover
        print(f"[경고] 계획 경로 지도 생성 실패: {e}", file=sys.stderr)
        return None


def _fmt(v: Any, na: str = "—") -> str:
    return na if v is None else (f"{v:g}" if isinstance(v, float) else str(v))


def _pick_better(a: Any, b: Any, lower_better: bool = True) -> Tuple[str, str]:
    """두 값 중 더 나은 쪽에 ✓. None은 비교 제외."""
    if a is None or b is None or a == b:
        return "", ""
    a_better = (a < b) if lower_better else (a > b)
    return ("✓", "") if a_better else ("", "✓")


def render_markdown(routes: List[str], metrics: Dict[str, dict], exposures: Dict[str, Optional[dict]],
                    scored: Dict[str, dict],
                    rec: dict, weights: Dict[str, float], norm_mode: str,
                    gt_validation: Dict[str, dict],
                    figures: Optional[Dict[str, Optional[str]]] = None,
                    overview_fig: Optional[str] = None) -> str:
    L: List[str] = []
    rk = " / ".join(f"route_{r}" for r in routes)
    L.append(f"# 정찰 보고서 — {rk}")
    L.append("")
    L.append("> 정밀 정찰 및 경로 위험도·은밀성 평가 (Scenario 1). 값은 **↓ 낮을수록 안전(은밀)**.")
    L.append("> 위험도는 **정찰이 perceive한 것(센서퓨전 확정 발견객체)** 으로 산정 — 정답(GT)이 아님. "
             "GT는 7장에서 정찰 정확도 검증에만 사용.")
    L.append("")

    # 1. 임무 개요
    L.append("## 1. 임무 개요")
    L.append("")
    mp = next((metrics[r]["map"] for r in routes if metrics[r]["map"]), "finalmap")
    L.append(f"- 맵: `{mp}` · 출발 {START_XY} → 목적지 {GOAL_XY}")
    L.append(f"- 위험도(perception 기반) 가중치: " + ", ".join(f"{k}={v}" for k, v in weights.items()) +
             f" · 정규화: `{norm_mode}`")
    L.append(f"- 입력 루트: {', '.join('route_' + r for r in routes)}")
    L.append("")
    if overview_fig:
        L.append(f"![A/B 계획 경로 (새 맵)]({os.path.basename(overview_fig)})")
        L.append("")
        L.append("> A(서·파랑)/B(동·초록) **계획 경로**를 맵 위에 표시 (▲=목적지, ★=적전차 리스폰). "
                 "실제 주행 궤적·노출은 6장 참고.")
        L.append("")

    # 2. 주행 요약
    L.append("## 2. 주행 요약")
    L.append("")
    L.append("| 항목 | " + " | ".join(f"route_{r}" for r in routes) + " |")
    L.append("|---|" + "---|" * len(routes))
    def row(name, fn):
        return f"| {name} | " + " | ".join(fn(metrics[r]) for r in routes) + " |"
    L.append(row("도착(reached)", lambda m: "✅" if m["reached"] else "❌"))
    L.append(row("유효 런", lambda m: ("✅" if scored[m['route']]['valid'] else "⚠️ 무효")))
    L.append(row("충돌 횟수", lambda m: _fmt(m["collisions"])))
    L.append(row("장애물 밀도 (/100m)", lambda m: _fmt(m["obstacle_density"])))
    L.append("")

    # 3. 효율 (별도 축 — 위험도 점수엔 미포함)
    L.append("## 3. 효율 (별도 축 — 위험도 미포함)")
    L.append("")
    L.append("> 시간·거리·우회는 '위험'이 아니라 '효율'이라 위험도에서 분리해 참고로만 표기.")
    L.append("")
    L.append("| 지표 | " + " | ".join(f"route_{r}" for r in routes) + " |")
    L.append("|---|" + "---|" * len(routes))
    L.append(row("이동거리 (m)", lambda m: _fmt(m["distance_m"])))
    L.append(row("소요시간 (s)", lambda m: _fmt(m["sim_time_s"])))
    L.append(row("우회비 (실제/직선)", lambda m: _fmt(m["detour_ratio"])))
    L.append("")

    # 4. 위협·객체 발견 (정보)
    L.append("## 4. 위협·객체 발견 (정보 — 위험도 미반영)")
    L.append("")
    L.append("> YOLO 카운트는 중복 탐지라 위치 위험도엔 안 씀. 위험도는 센서퓨전 **확정 발견객체**(아래) 기반.")
    L.append("")
    L.append("| 항목 | " + " | ".join(f"route_{r}" for r in routes) + " |")
    L.append("|---|" + "---|" * len(routes))
    L.append(row("확정 위협 수(house/tank)", lambda m: str(gt_validation[m['route']]["perceived_total"])))
    all_cls = sorted({c for r in routes for c in metrics[r]["yolo_counts"]})
    for c in all_cls:
        w = THREAT_CLASS_WEIGHTS.get(c)
        tag = " (위협클래스)" if w else ""
        L.append(row(f"YOLO {c}{tag}", lambda m, c=c: str(m["yolo_counts"].get(c, 0))))
    L.append("")

    # 5. 위험도 비교표 (perception 기반, ↓ 낮을수록 안전)
    L.append("## 5. 위험도 비교표 (↓ 낮을수록 안전)")
    L.append("")
    if len(routes) == 2:
        a, b = routes
        L.append(f"| 지표 | route_{a} | route_{b} |")
        L.append("|---|---|---|")
        def cmp_row(name, va, vb, lower=True):
            ma, mb = _pick_better(va, vb, lower)
            return f"| {name} | {_fmt(va)} {ma} | {_fmt(vb)} {mb} |"
        ea, eb = exposures.get(a), exposures.get(b)
        L.append(cmp_row("위험도 총점", scored[a]["risk_total"], scored[b]["risk_total"]))
        L.append(cmp_row("은밀성 노출(길이비)", ea["stealth_ratio"] if ea else None, eb["stealth_ratio"] if eb else None))
        L.append(cmp_row("위협 근접(길이비)", ea["proximity_ratio"] if ea else None, eb["proximity_ratio"] if eb else None))
        L.append(cmp_row("지형 굴곡 σ", metrics[a]["terrain_sigma"], metrics[b]["terrain_sigma"]))
        L.append(cmp_row("확정 위협 수(참고)", ea["threat_count"] if ea else None, eb["threat_count"] if eb else None))
    else:
        L.append("_단일 루트 — 비교표 생략._")
    L.append("")

    # 6. 위험도 분해 (perception 기반)
    L.append("## 6. 위험도 분해 (perception 기반)")
    L.append("")
    if figures and any(v for v in figures.values()):
        L.append("> 각 루트 지도: **초록 선** = 채점한 클린 중심선, **옅은 파랑** = 실제 주행 궤적(weave, 참고), "
                 "**빨강 원** = 발견 위협 반경(전방향), **빨강 점** = 중심선 노출(반경+LoS) 구간.")
        L.append("")
    for r in routes:
        L.append(f"### route_{r} — 위험도 총점 **{scored[r]['risk_total']:.3f}**" +
                 ("" if scored[r]["valid"] else " ⚠️(무효 런)"))
        L.append("")
        L.append("| 항목 | 가중 | 원시값 | 정규화 | 기여 |")
        L.append("|---|---|---|---|---|")
        for b in scored[r]["breakdown"]:
            raw = _fmt(b["raw"]) if b["available"] else "**N/A**"
            norm = _fmt(b["norm"]) if b["available"] else "—"
            contrib = _fmt(b["contrib"]) if b["available"] else "0"
            L.append(f"| {b['label']} | {b['weight']} | {raw} | {norm} | {contrib} |")
        L.append("")
        exp = exposures.get(r)
        if exp and exp["per_threat"]:
            L.append("발견 위협별 노출(클린 중심선 기준) — 노출 길이 상위:")
            L.append("")
            L.append("| 위협 | 노출 길이 m | 반경내 길이 m | 최소거리 m |")
            L.append("|---|---|---|---|")
            for e in exp["per_threat"][:6]:
                L.append(f"| {e['threat']} | {e['exposed_length_m']} | {e['within_radius_length_m']} | {_fmt(e['min_dist_m'])} |")
            L.append("")
        elif exp is not None and exp.get("threat_count") == 0:
            L.append("_발견된 위협 없음 → 노출/근접 0 (지형 위주). 7장 GT 검증의 신뢰도 확인._")
            L.append("")

        fig = (figures or {}).get(r)
        if fig:
            L.append(f"![route_{r} 중심선·위협 노출 지도]({os.path.basename(fig)})")
            L.append("")

    # 7. 정찰 정확도 (GT 검증 — 위험도엔 미반영)
    L.append("## 7. 정찰 정확도 (GT 검증)")
    L.append("")
    L.append("> 발견 위협 ↔ 정답맵(GT) 위협 비교. 누락이 많으면 위험도가 그만큼 **과소평가**됐다는 신호 → 재정찰 권고.")
    L.append("")
    def gv(r):
        return gt_validation[r]
    L.append("| 항목 | " + " | ".join(f"route_{r}" for r in routes) + " |")
    L.append("|---|" + "---|" * len(routes))
    L.append("| GT 위협 총수 | " + " | ".join(str(gv(r)["gt_total"]) for r in routes) + " |")
    L.append("| 발견(매칭) | " + " | ".join(str(gv(r)["found"]) for r in routes) + " |")
    L.append("| 누락 | " + " | ".join(str(gv(r)["missed"]) for r in routes) + " |")
    L.append("| 오탐(false+) | " + " | ".join(str(gv(r)["false_pos"]) for r in routes) + " |")
    L.append("| 위치오차 평균 m | " + " | ".join(_fmt(gv(r)["mean_pos_err_m"]) for r in routes) + " |")
    L.append("| 탐지 신뢰도(발견/GT) | " + " | ".join(_fmt(gv(r)["confidence"]) for r in routes) + " |")
    fams = sorted({f["family"] for r in routes for f in gv(r)["by_family"]})
    for fam in fams:
        def famcell(r, fam=fam):
            e = next((x for x in gv(r)["by_family"] if x["family"] == fam), None)
            return f"{e['perceived']}/{e['gt']}" if e else "0/0"
        L.append(f"| {fam} 발견/GT | " + " | ".join(famcell(r) for r in routes) + " |")
    L.append("")
    for r in routes:
        conf = gv(r)["confidence"]
        if conf is not None and conf < 0.5:
            L.append(f"- ⚠️ **route_{r}**: 탐지 신뢰도 {conf} (GT {gv(r)['gt_total']}개 중 {gv(r)['found']}개) — "
                     "정찰이 위협을 많이 놓침 → 위험도 과소평가 가능 → 재정찰 권장.")
    L.append("")

    # 8. 최종 추천
    L.append("## 8. 최종 추천")
    L.append("")
    if rec["winner"]:
        L.append(f"### 🏆 권장 루트: **route_{rec['winner']}**  (신뢰도: {rec['confidence']})")
    else:
        L.append("### ⚠️ 추천 불가")
    L.append("")
    L.append(f"- 근거: {rec['reason']}")
    L.append("")

    # 9. 데이터 품질·방법 주석
    L.append("## 9. 데이터 품질 · 방법 주석")
    L.append("")
    any_warn = False
    for r in routes:
        ws = scored[r]["warnings"]
        if ws:
            any_warn = True
            L.append(f"- **route_{r}**")
            for w in ws:
                L.append(f"  - {w}")
    if not any_warn:
        L.append("- (런 품질 경고 없음)")
    L.append("")
    L.append("- **위험도 = perception 기반 3요소**(은밀성·위협근접·험지). 시간/거리/우회는 효율 축으로 분리(3장).")
    L.append("- **노출/근접은 클린 루트 중심선**(planned_paths v0, weave 제외) 길이비. weave 궤적으로 재면 수색행동 때문에 과대평가됨.")
    L.append("- **발견 위협엔 heading이 없어** House FOV 콘 대신 반경+LoS(전방향) 보수적 판정. LoS 차폐는 주행 base 맵 + 발견 obstacle 기준.")
    L.append("- **GT(정답맵)는 7장 검증 전용** — 위험도 점수엔 미반영(실무엔 GT 없음).")
    L.append("- 험지 σ는 실제 주행 body pitch/roll 기반(중심선 아님) — route-inherent 지형 거칠기 근사.")
    L.append("")
    return "\n".join(L)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="정찰 보고서 생성기 (recon_reports/*.json → recon_report.md)")
    p.add_argument("--input", default=DEFAULT_REPORT_DIR,
                   help="comparison.json 또는 route_*.json 이 있는 디렉터리/파일 (기본 recon_reports/)")
    p.add_argument("--route-a", default=None, help="route_A.json 직접 지정(폴백)")
    p.add_argument("--route-b", default=None, help="route_B.json 직접 지정(폴백)")
    p.add_argument("-o", "--output", default=None, help="출력 md 경로 (기본 <input>/recon_report.md)")
    p.add_argument("--discovered-dir", default=DEFAULT_DISCOVERED_DIR,
                   help="발견객체맵 디렉터리 — 위험도 위협 출처 discovered_objects_route_{A,B}.map (기본 recon_reports/recon_map)")
    p.add_argument("--gt-map", default=DEFAULT_GT_MAP,
                   help="정답맵(GT) — 위험도엔 미사용, 정찰 정확도 검증 전용 (기본 final_v3.map)")
    p.add_argument("--map", default=DEFAULT_MAP,
                   help="주행 base 맵 — 계획경로 오버레이 + LoS 차폐 base obstacle (기본 finalmap.map)")
    p.add_argument("--routes", default=DEFAULT_ROUTES,
                   help="중심선 폴백/계획 경로 그림용 routes.yaml (기본 path_planning/config/routes.yaml)")
    p.add_argument("--norm", choices=["reference", "minmax"], default="reference", help="정규화 방식")
    for k in ("stealth", "proximity", "terrain"):
        p.add_argument(f"--w-{k}", type=float, default=None, help=f"가중치 {k} 오버라이드")
    p.add_argument("--stdout", action="store_true", help="파일 대신 표준출력")
    p.add_argument("--no-figures", action="store_true",
                   help="노출 지도 PNG(exposure_*.png) 생성/임베드 생략")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    weights = dict(DEFAULT_WEIGHTS)
    for k in ("stealth", "proximity", "terrain"):
        v = getattr(args, f"w_{k}")
        if v is not None:
            weights[k] = v

    routes_data = load_inputs(args.input, args.route_a, args.route_b)
    if not routes_data:
        print(f"[오류] 입력을 찾지 못함: {args.input} (comparison.json / route_*.json 없음)", file=sys.stderr)
        return EXIT_NO_INPUT

    # GT 위협(정답맵) — 위험도엔 안 쓰고 정찰 정확도 검증에만.
    try:
        gt_threats, _gt_bboxes, gt_objects = tg.load_map(args.gt_map)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"[경고] GT 맵 로드 실패({e}) — 정찰 정확도 검증 생략", file=sys.stderr)
        gt_threats, gt_objects = [], []

    # 주행 base 맵(finalmap) — 계획경로 오버레이 + LoS 차폐 base obstacle(알려진 나무 등).
    try:
        _, base_bboxes, _ = tg.load_map(args.map)
    except (FileNotFoundError, json.JSONDecodeError):
        base_bboxes = []

    routes = sorted(routes_data.keys())
    metrics: Dict[str, dict] = {}
    exposures: Dict[str, Optional[dict]] = {}
    perceived: Dict[str, List[Dict[str, Any]]] = {}
    centerlines: Dict[str, Optional[List[Tuple[float, float]]]] = {}
    los_obstacles: Dict[str, List[Dict[str, float]]] = {}
    gt_validation: Dict[str, dict] = {}
    scored: Dict[str, dict] = {}

    for rid in routes:
        m = extract_metrics(routes_data[rid])
        metrics[rid] = m
        centerlines[rid] = extract_centerline(routes_data[rid], rid, args.routes)
        # perception: 센서퓨전 확정 발견객체맵 → 위협 + LoS 차폐 obstacle
        disc_path = os.path.join(args.discovered_dir, f"discovered_objects_route_{rid}.map")
        disc_data = _read_json(disc_path) if os.path.isfile(disc_path) else {"obstacles": []}
        if not os.path.isfile(disc_path):
            print(f"[경고] 발견객체맵 없음: {disc_path} — route_{rid} 위협 0으로 처리", file=sys.stderr)
        perceived[rid] = parse_perceived_threats(disc_data)
        disc_bboxes = [b for b in (tg.obstacle_to_bbox(o) for o in disc_data.get("obstacles", [])) if b]
        los_obstacles[rid] = base_bboxes + disc_bboxes
        exposures[rid] = compute_centerline_exposure(centerlines[rid], perceived[rid], los_obstacles[rid])
        gt_validation[rid] = validate_perception_vs_gt(perceived[rid], gt_threats)

    norm = normalize(metrics, exposures, args.norm)
    for rid in routes:
        valid, warns = validate_run(metrics[rid])
        total, breakdown = risk_score(norm[rid], weights)
        scored[rid] = {"risk_total": total, "breakdown": breakdown,
                       "valid": valid, "warnings": warns}

    rec = recommend(scored, gt_validation)

    # 노출 지도(PNG)는 md와 같은 폴더에 생성해 basename으로 임베드한다. stdout/--no-figures면 생략.
    figures: Dict[str, Optional[str]] = {}
    overview_fig: Optional[str] = None
    out = None
    if not args.stdout:
        out = args.output or os.path.join(
            args.input if os.path.isdir(args.input) else os.path.dirname(args.input) or ".",
            "analysis", "recon_report.md")   # 파생 리포트/PNG는 analysis/로 분리(입력은 root에서 읽음)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        if not args.no_figures:
            out_dir = os.path.dirname(out) or "."
            overview_fig = render_route_overview(args.map, args.routes, out_dir)
            for rid in routes:
                figures[rid] = render_exposure_figure(
                    rid, centerlines[rid], metrics[rid]["trajectory"], perceived[rid],
                    los_obstacles[rid], gt_threats, exposures[rid], out_dir)

    md = render_markdown(routes, metrics, exposures, scored, rec, weights,
                         args.norm, gt_validation, figures, overview_fig)

    if args.stdout:
        print(md)
    else:
        with open(out, "w", encoding="utf-8") as f:
            f.write(md)
        print(f"[완료] 정찰 보고서 작성: {out}")
        made = ([f"계획경로 1"] if overview_fig else []) + \
               ([f"노출 지도 {sum(1 for v in figures.values() if v)}"] if any(figures.values()) else [])
        if made:
            print("  그림: " + " · ".join(made) + " (recon_reports/*.png)")
        if rec["winner"]:
            print(f"  추천: route_{rec['winner']} (신뢰도 {rec['confidence']})")

    # 종료코드
    if not any(s["valid"] for s in scored.values()):
        return EXIT_NO_VALID_RUN
    if len(routes) < 2:
        return EXIT_SINGLE_ROUTE
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
