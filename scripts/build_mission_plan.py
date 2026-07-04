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
DEFAULT_RISK_RESULT = os.path.join(DEFAULT_REPORT_DIR, "route_risk_result.json")

# 무기 사거리(ballistic_turret_node: min_range_m 20 / max_range_m 130).
MIN_RANGE_M = 20.0
MAX_RANGE_M = 130.0
# 노출 거리감쇠 범위 = 위협반경 × 이 값(반경서 0.5, 2배서 0). generate_recon_report와 동일.
EXPOSURE_DECAY_MULT = 2.0
# 최종 적전차 명목위치(현 ballistic engagements enemy_final) — 발사 시 /tank/enemy/pose로 정밀화됨.
DEFAULT_FINAL_ENEMY_XY = (135.46, 276.87)
# 사격 위치 후보 탐색용 루트 densify 간격(m). 노출 profile(0.75)보다 성글어도 충분.
CANDIDATE_STEP_M = 1.0
# 점수 가중치(합 1.0). range=가까움 선호, exposure=은폐 선호.
DEFAULT_WEIGHTS = {"range": 0.5, "exposure": 0.5}
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


def best_firing_position(target: Dict[str, Any], pts: List[Tuple[float, float]], arc: List[float],
                         bboxes: List[Dict[str, float]], exposure_threats: List[Dict[str, Any]],
                         weights: Dict[str, float]) -> Optional[Dict[str, Any]]:
    """표적 T에 대해 루트 점 중 사거리+LoS 만족하는 최고점수 사격 위치. 없으면 None(교전 불가)."""
    tx, ty = float(target["x"]), float(target["y"])
    span = MAX_RANGE_M - MIN_RANGE_M
    best: Optional[Dict[str, Any]] = None
    for idx, (px, py) in enumerate(pts):
        d = math.hypot(px - tx, py - ty)
        if d < MIN_RANGE_M or d > MAX_RANGE_M:
            continue
        if not tg.check_los(px, py, tx, ty, bboxes):
            continue
        range_score = 1.0 - (d - MIN_RANGE_M) / span if span > 0 else 1.0
        exp = point_exposure(px, py, exposure_threats, bboxes, exclude_xy=(tx, ty))
        score = weights["range"] * range_score + weights["exposure"] * (1.0 - exp)
        if best is None or score > best["score"]:
            best = {
                "x": round(px, 2), "y": round(py, 2),
                "distance_m": round(d, 2), "los": True,
                "exposure": round(exp, 3), "exposure_band": exposure_band(exp),
                "range_score": round(range_score, 3), "score": round(score, 4),
                "route_arc_m": round(arc[idx], 2),
            }
    return best


def get_firing_candidates(
    target: Dict[str, Any],
    pts: List[Tuple[float, float]],
    arc: List[float],
    bboxes: List[Dict[str, float]],
    exposure_threats: List[Dict[str, Any]],
    weights: Dict[str, float],
    sample_step_m: float = 8.0,
    max_candidates: int = 25,
) -> List[Dict[str, Any]]:
    """LLM에 전달할 유효 사격 후보 목록. 8m 간격 샘플링, 점수 상위 max_candidates개."""
    tx, ty = float(target["x"]), float(target["y"])
    span = MAX_RANGE_M - MIN_RANGE_M
    last_arc = -999.0
    candidates = []
    for idx, (px, py) in enumerate(pts):
        if arc[idx] - last_arc < sample_step_m:
            continue
        d = math.hypot(px - tx, py - ty)
        if d < MIN_RANGE_M or d > MAX_RANGE_M:
            continue
        if not tg.check_los(px, py, tx, ty, bboxes):
            continue
        range_score = 1.0 - (d - MIN_RANGE_M) / span if span > 0 else 1.0
        exp = point_exposure(px, py, exposure_threats, bboxes, exclude_xy=(tx, ty))
        score = weights["range"] * range_score + weights["exposure"] * (1.0 - exp)
        candidates.append({
            "x": round(px, 1), "y": round(py, 1),
            "distance_m": round(d, 1),
            "exposure_band": exposure_band(exp),
            "exposure": round(exp, 3),
            "score": round(score, 4),
            "route_arc_m": round(arc[idx], 1),
        })
        last_arc = arc[idx]
    candidates.sort(key=lambda c: -c["score"])
    return candidates[:max_candidates]


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
               weights: Dict[str, float]) -> Dict[str, Any]:
    """한 루트에 대해 표적별 최적 사격 위치 + 커버리지/점수 집계."""
    pts, arc = densify_polyline(poly, CANDIDATE_STEP_M)
    per_target: List[Dict[str, Any]] = []
    for t in targets:
        fp = best_firing_position(t, pts, arc, bboxes, exposure_threats, weights)
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
    r = route_results[best]
    reason = (f"route_{best} 추천 — 표적 {r['covered_count']}/{r['target_count']} 교전가능"
              f"(coverage {r['coverage']:.2f}), 평균 사격점수 {r['mean_score']:.3f}, 평균 노출 {r['mean_exposure']:.3f}"
              f", 교전순서 {'실현가능' if r['order_feasible'] else '불가(' + ','.join(r['late_statics']) + ')'}.")
    others = [rid for rid in route_results if rid != best]
    for o in others:
        ro = route_results[o]
        if ro["covered_count"] < r["covered_count"]:
            reason += f" route_{o}는 표적 {ro['covered_count']}/{ro['target_count']}만 교전가능."
        elif not ro["order_feasible"]:
            reason += f" route_{o}는 교전순서 불가(최종 적 뒤 표적 {','.join(ro['late_statics'])})."
    return best, reason


def _load_risk_selected_route(path: str) -> Optional[str]:
    """route_risk_result.json에서 LLM 위험도 추천 루트(result.selected_route)를 읽는다. 실패 시 None."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        route = data.get("result", {}).get("selected_route", "")
        return route.strip().upper() if route else None
    except Exception:
        return None


def build_enemy_intel(rec: str, plan: Dict[str, Any], risk_features_path: str) -> Dict[str, Any]:
    """추천 루트의 위협 데이터 + 미션 표적을 합쳐 적 정보 요약 생성."""
    _OUTPOST_TYPES = {"house", "checkpoint", "outpost", "bunker", "guard_tower", "sentry"}
    intel: Dict[str, Any] = {"tank_count": 0, "outpost_count": 0, "by_class": {}, "positions": []}

    targets = plan["plan"]["targets"]
    intel["tank_count"] = len(targets)
    intel["positions"] = [
        {"id": t["id"], "x": t["pos"]["x"], "y": t["pos"]["y"], "class": t["class"]}
        for t in targets
    ]

    try:
        with open(risk_features_path, "r", encoding="utf-8") as f:
            rf = json.load(f)
        by_class = rf.get(f"route_{rec}", {}).get("threat", {}).get("by_class", {})
        intel["by_class"] = by_class
        intel["outpost_count"] = sum(v for k, v in by_class.items() if k.lower() in _OUTPOST_TYPES)
    except Exception:
        pass

    return intel


# --------------------------------------------------------------------------- #
# LLM 서술(선택) — 기하 계획을 사람이 읽을 수행 지침으로
# --------------------------------------------------------------------------- #

def _apply_llm_firing_checkpoint(
    plan: Dict[str, Any],
    target_id: str,
    lx: float,
    ly: float,
    bboxes: List[Dict[str, float]],
    exp_threats: List[Dict[str, Any]],
) -> bool:
    """LLM이 선택한 (lx, ly)를 검증(사거리·LoS) 후 plan.plan.targets와 plan.engagements에 반영.
    검증 통과 시 True, 실패 시 False(기하 좌표 유지).
    """
    for t in plan["plan"]["targets"]:
        if t["id"] != target_id:
            continue
        cp = t.get("firing_checkpoint")
        if not cp:
            return False
        tx, ty_t = float(t["pos"]["x"]), float(t["pos"]["y"])
        d = math.hypot(lx - tx, ly - ty_t)
        if not (MIN_RANGE_M <= d <= MAX_RANGE_M):
            return False
        if not tg.check_los(lx, ly, tx, ty_t, bboxes):
            return False
        exp = point_exposure(lx, ly, exp_threats, bboxes, exclude_xy=(tx, ty_t))
        cp["x"] = round(lx, 1)
        cp["y"] = round(ly, 1)
        cp["distance_m"] = round(d, 2)
        cp["exposure"] = round(exp, 3)
        cp["exposure_band"] = exposure_band(exp)
        cp["llm_selected"] = True
        break
    else:
        return False

    for eng in plan.get("engagements", []):
        if eng["id"] == target_id:
            eng["checkpoint"]["x"] = round(lx, 1)
            eng["checkpoint"]["y"] = round(ly, 1)
            break
    return True


def llm_narrate(
    plan: Dict[str, Any],
    route_poly: List[Tuple[float, float]],
    bboxes: List[Dict[str, float]],
    exp_threats: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """LLM에 루트 후보 좌표·적 위치·지형 노출 정보를 넘겨 사격 좌표를 선택하고
    수행 지침(성공확률·실행요약·주의사항)을 생성. 실패 시 available=false."""
    try:
        sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "risk_analysis"))
        from risk_analysis.llm_reporter import LLMReporter  # noqa: E402
    except Exception as exc:
        return {"available": False, "error": f"LLMReporter import 실패: {exc}"}

    reporter = LLMReporter()
    pts, arc = densify_polyline(route_poly, CANDIDATE_STEP_M)

    ctx_targets = []
    for t in plan["plan"]["targets"]:
        target_raw = {"id": t["id"], "x": t["pos"]["x"], "y": t["pos"]["y"]}
        cands = get_firing_candidates(
            {"x": t["pos"]["x"], "y": t["pos"]["y"]},
            pts, arc, bboxes, exp_threats, DEFAULT_WEIGHTS,
        )
        ctx_targets.append({
            "target_id": t["id"],
            "enemy_position": target_raw,
            "firing_range_m": {"min": MIN_RANGE_M, "max": MAX_RANGE_M},
            "firing_candidates": cands,  # 이미 사거리·LoS 검증된 루트 위 후보들
        })

    ctx = {
        "route_recommended": plan["route_recommended"],
        "targets": ctx_targets,
    }
    prompt = (
        "너는 전차 미션 계획 참모 AI다.\n"
        "아래 firing_candidates는 루트 위 좌표 중 적 전차까지 시선(LoS) 확보·사거리(20~130m) 조건을 이미 만족한 후보들이다.\n"
        "exposure_band(low=은폐 우수, medium=보통, high=노출 위험)와 distance_m을 고려해 최적 사격 위치를 선택하라.\n"
        "규칙:\n"
        "- 반드시 JSON 객체 하나만 출력한다(마크다운·설명 금지).\n"
        "- firing_checkpoint.x·y는 반드시 firing_candidates 중 하나의 x·y 값을 그대로 사용할 것(새 숫자 지어내기 금지).\n"
        "- success_probability는 반드시 '높음'·'중간'·'낮음' 중 하나만 출력한다.\n"
        "- execution_brief는 선택한 사격 위치 기준으로 1~2문장(한국어).\n"
        "출력 JSON 구조:\n"
        '{"success_probability":"높음|중간|낮음",'
        '"execution_brief":"실행 방법(한국어)",'
        '"cautions":["주의점(한국어)"],'
        '"firing_checkpoint":{"x":숫자,"y":숫자,"reason":"선택 이유(한국어)"}}\n'
        "미션 데이터:\n" + json.dumps(ctx, ensure_ascii=False, indent=2)
    )
    _eta = {"qwen3:0.6b": "~20-30초", "qwen3:1.7b": "~90-120초", "qwen3:4b": "~130-160초",
            "gemma3:1b": "~20-30초", "gemma3:4b": "~90-120초"}.get(reporter.model_name, "~수 분")
    print(f"  🧠 LLM 사격 좌표 분석 중... ({reporter.model_name} {_eta} · 끊지 말고 기다리세요)", flush=True)
    try:
        resp = reporter.call_ollama(prompt)
        raw = resp.get("response", "") if isinstance(resp, dict) else ""
        parsed = json.loads(raw)

        # LLM 사격 좌표 검증 및 반영
        llm_cp = parsed.get("firing_checkpoint")
        firing_applied = False
        if isinstance(llm_cp, dict):
            try:
                lx = float(llm_cp["x"])
                ly = float(llm_cp["y"])
                if math.isfinite(lx) and math.isfinite(ly):
                    for t in plan["plan"]["targets"]:
                        ok = _apply_llm_firing_checkpoint(plan, t["id"], lx, ly, bboxes, exp_threats)
                        if ok:
                            firing_applied = True
                            print(f"[mission_plan] LLM 사격 좌표 채택: ({lx}, {ly}) — {llm_cp.get('reason', '')}")
                            break
                    if not firing_applied:
                        print(f"[mission_plan] LLM 사격 좌표 검증 실패({lx},{ly}) — 기하 좌표 유지")
            except (KeyError, TypeError, ValueError):
                pass
        parsed["llm_firing_applied"] = firing_applied
        parsed["available"] = True
        parsed["model"] = reporter.model_name
        return parsed
    except Exception as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}", "model": reporter.model_name}


# --------------------------------------------------------------------------- #
# 출력
# --------------------------------------------------------------------------- #

def build_engagements_json(plan: Dict[str, Any]) -> str:
    """추천 루트의 사격 계획 → ballistic_turret_node engagements_json 스니펫(제안용, 자동적용 안 함)."""
    engs = []
    id_to_target = {e["id"]: e for e in plan["plan"]["targets"]}
    for tid in plan["plan"]["engage_order"]:
        e = id_to_target[tid]
        cp = e["firing_checkpoint"]
        engs.append({
            "id": tid,
            "checkpoint": {"x": cp["x"], "y": cp["y"], "radius_m": 10.0},
            "checkpoint_settle_sec": 0.8,
            "target": {"x": e["pos"]["x"], "y": e["pos"]["y"], "z": e.get("z_height", 0.0)},
            "target_from_enemy_pose": e["class"] == "final_enemy",
            "target_height_offset_m": 0.0,
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
    ap.add_argument("--risk-result", default=DEFAULT_RISK_RESULT, help="route_risk_result.json 경로(LLM 위험도 추천 루트)")
    ap.add_argument("--route", default=None, help="추천 대신 이 루트로 강제(A/B); 미지정 시 LLM 위험도 결과 우선")
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

    # 루트별 계획
    route_results: Dict[str, Dict[str, Any]] = {}
    for rid, poly in routes.items():
        exp_threats = load_exposure_threats(args.risk_features, rid)
        route_results[rid] = plan_route(rid, poly, targets, bboxes, exp_threats, DEFAULT_WEIGHTS)

    # 추천(또는 강제): 명시 --route > 정찰 LLM 위험도 > 기하 알고리즘
    if args.route and args.route in route_results:
        rec, reason = args.route, f"route_{args.route} 사용자 지정."
    else:
        risk_route = _load_risk_selected_route(args.risk_result)
        if risk_route and risk_route in route_results:
            rec = risk_route
            reason = f"route_{rec} — 정찰 LLM 위험도 분석 추천(route_risk_result.json)."
            print(f"[mission_plan] LLM 위험도 추천 루트 사용: {rec}")
        else:
            rec, reason = recommend_route(route_results)
            print(f"[mission_plan] 기하 알고리즘 추천 루트 사용: {rec} (위험도 결과 없음)")

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

    plan["friendly_unit"] = "K2"
    plan["enemy_intel"] = build_enemy_intel(rec, plan, args.risk_features)

    # 검증
    problems = verify_plan(plan, bboxes)
    if rr["late_statics"]:
        problems.append(f"교전순서 불가: 정적 표적 {rr['late_statics']}의 사격 위치가 최종 적보다 루트상 뒤 — "
                        "최종 적을 먼저 지나쳐 임무 조기 종료 위험(다른 루트/사격위치 검토)")
    plan["verification"] = {"ok": not problems, "problems": problems}

    # LLM 서술 + 사격 좌표 선택
    if args.no_llm:
        plan["llm_guidance"] = {"available": False, "error": "skipped (--no-llm)"}
    else:
        rec_exp_threats = load_exposure_threats(args.risk_features, rec)
        plan["llm_guidance"] = llm_narrate(plan, routes[rec], bboxes, rec_exp_threats)
        # LLM이 사격 좌표를 바꿨으면 engagements 재빌드
        if plan["llm_guidance"].get("llm_firing_applied"):
            plan["engagements"] = json.loads(build_engagements_json(plan))

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
