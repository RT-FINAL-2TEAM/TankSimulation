#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""돌발(정찰 미지 신규 객체) 대응 결정 코어 — 순수 수식. 라이브 노드가 이걸 호출한다.

시나리오2 미션 주행 중, 정찰에서 몰랐던 신규 객체를 perception이 감지(≤30m)하면
이 코어가 **ENGAGE(정지 사격) / BYPASS(무시·속행) / RETURN(후퇴)** 를 결정한다(Part B).

전제(코드 실측):
- 사격은 정지-조준-발사, 표적 정지 가정. 사거리 min 20 / max 130m → 감지(≤30m)된 신규 전차는 보통 20~30m.
- 탄약·아군체력 없음(시뮬 미제공/불변) → 순수 기하(거리+LoS+수). RETURN은 피해모델 없는 **독트린** 결정.
- LoS/risk 기하는 mission.risk(=scripts/recon_eval/threat_geometry 재사용)로 단일출처.
- 교전 대상은 tank만(초소 등은 RETURN/BYPASS에만 영향, ENGAGE 대상 아님).

이 모듈은 '순간 결정'만 한다. 히스테리시스/쿨다운(매 tick 뒤집힘 방지)은 **호출 노드가 상태로 적용**한다.
오프라인 검증:  python3 src/mission/mission/sudden_decision.py   (합성 시나리오 assert)
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

try:
    from mission import risk  # ROS 패키지/노드 컨텍스트
except ImportError:  # 오프라인 standalone 실행
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import risk  # type: ignore  # noqa: E402

# 기본 파라미터(노드/런치에서 오버라이드 가능).
DEFAULTS: Dict[str, float] = {
    "weapon_min_range_m": 20.0,     # ballistic min_range_m
    "weapon_max_range_m": 130.0,    # ballistic max_range_m
    "threat_radius_m": 20.0,        # risk 반경(Tank001 규칙)
    "known_match_radius_m": 8.0,    # 정찰 known 표적 매칭 반경(=decision_node)
    "return_n_threshold": 2,        # 동시 신규 위협 수 ≥ 이 값 + 정리불가 → RETURN
    "return_risk_threshold": 0.5,   # 근접·노출 위험도 ≥ 이 값 + 교전불가 → RETURN
}

ACTION_ENGAGE = "ENGAGE"
ACTION_BYPASS = "BYPASS"
ACTION_RETURN = "RETURN"
ACTION_NONE = "NONE"


def classify_new_threats(detections: List[Dict[str, Any]], known_xy: List[Tuple[float, float]],
                         match_radius_m: float = 8.0) -> List[Dict[str, Any]]:
    """감지 객체 → 돌발(신규) 위협만 추린다.

    - class가 tank가 아닌 것 제외(교전 대상은 tank만).
    - 정찰 known 표적과 match_radius 안에서 매칭되면 제외(이미 아는 것).
    detections: [{id, xy:(x,y), class}], known_xy: [(x,y), ...]
    """
    new: List[Dict[str, Any]] = []
    for d in detections:
        cls = str(d.get("class", "")).lower()
        if cls and cls != "tank":
            continue
        x, y = d["xy"]
        if any(math.hypot(x - kx, y - ky) <= match_radius_m for kx, ky in known_xy):
            continue
        new.append(d)
    return new


def build_mission_features(player_xy: Tuple[float, float], new_threats: List[Dict[str, Any]],
                           bboxes: List[Dict[str, float]], params: Optional[Dict[str, float]] = None,
                           *, progress_ratio: Optional[float] = None,
                           targets_remaining: Optional[int] = None) -> Dict[str, Any]:
    """돌발 스냅샷 피처 — 수식·LLM 공통 입력(정찰 risk_features 패턴). health/ammo 없음.

    per 위협: dist / los / in_range / too_close / engageable / risk(0~1).
    집계: n_new / n_engageable / any_engageable / max_risk / nearest / nearest_engageable.
    """
    p = {**DEFAULTS, **(params or {})}
    mn, mx, rad = p["weapon_min_range_m"], p["weapon_max_range_m"], p["threat_radius_m"]
    per: List[Dict[str, Any]] = []
    for t in new_threats:
        x, y = t["xy"]
        dist = math.hypot(player_xy[0] - x, player_xy[1] - y)
        los = risk.check_los(player_xy, (x, y), bboxes)
        in_range = mn <= dist <= mx
        engageable = in_range and los
        per.append({
            "id": t.get("id"),
            "xy": [round(float(x), 2), round(float(y), 2)],
            "dist_m": round(dist, 2),
            "los": bool(los),
            "in_range": in_range,
            "too_close": dist < mn,
            "engageable": engageable,
            "risk": round(risk.geometric_risk_score(player_xy, (x, y), bboxes, rad), 3),
        })
    eng = [e for e in per if e["engageable"]]
    return {
        "n_new": len(per),
        "n_engageable": len(eng),
        "any_engageable": bool(eng),
        "max_risk": round(max((e["risk"] for e in per), default=0.0), 3),
        "nearest": min(per, key=lambda e: e["dist_m"]) if per else None,
        "nearest_engageable": min(eng, key=lambda e: e["dist_m"]) if eng else None,
        "progress_ratio": progress_ratio,
        "targets_remaining": targets_remaining,
        "per_threat": per,
    }


def decide(features: Dict[str, Any], params: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """돌발 결정(순간). 우선순위: RETURN(압도/교전불가 위험) > ENGAGE(사격가능) > BYPASS.

    (히스테리시스·쿨다운은 호출 노드가 적용 — 이 함수는 상태 없는 순간 판정.)
    """
    p = {**DEFAULTS, **(params or {})}
    n = features["n_new"]
    n_eng = features["n_engageable"]
    if n == 0:
        return {"action": ACTION_NONE, "reason": "신규 위협 없음", "target": None}
    # 1) 압도: 동시 신규 위협이 많은데 다 정리 못 함 → 후퇴(독트린).
    if n >= p["return_n_threshold"] and n_eng < n:
        return {"action": ACTION_RETURN, "target": None,
                "reason": f"동시 신규 위협 {n}대 중 {n - n_eng}대 교전불가 — 압도적, 후퇴"}
    # 2) 근접·노출인데 교전 불가(너무 가깝거나 차폐) → 후퇴.
    if features["max_risk"] >= p["return_risk_threshold"] and not features["any_engageable"]:
        return {"action": ACTION_RETURN, "target": None,
                "reason": f"근접·노출 위험 {features['max_risk']:.2f}인데 교전 불가 — 후퇴"}
    # 3) 사격 가능 → 교전(최근접 사격가능 표적, 정지 사격 기동).
    if features["any_engageable"]:
        tgt = features["nearest_engageable"]
        return {"action": ACTION_ENGAGE, "target": tgt,
                "reason": f"신규 전차 교전가능(거리 {tgt['dist_m']}m, LoS 트임) — 정지 사격"}
    # 4) 그 외(멀거나 차폐 + 위험 낮음) → 무시하고 임무 속행.
    return {"action": ACTION_BYPASS, "target": None,
            "reason": "신규 위협이 멀거나 차폐 + 위험 낮음 — 무시하고 임무 속행"}


# --------------------------------------------------------------------------- #
# 오프라인 자체 검증 (합성 시나리오) — 시뮬/ROS 불필요
# --------------------------------------------------------------------------- #

def _selftest() -> int:
    P = (0.0, 0.0)
    NB: List[Dict[str, float]] = []  # 장애물 없음

    def T(x: float, y: float, tid: str = "t", cls: str = "tank") -> Dict[str, Any]:
        return {"id": tid, "xy": (x, y), "class": cls}

    def act(threats, bboxes=NB):
        return decide(build_mission_features(P, threats, bboxes))["action"]

    checks = []
    checks.append(("신규 없음 → NONE", act([]), ACTION_NONE))
    checks.append(("단일 25m LoS → ENGAGE", act([T(0, 25)]), ACTION_ENGAGE))
    checks.append(("단일 60m LoS → ENGAGE", act([T(0, 60)]), ACTION_ENGAGE))
    checks.append(("단일 140m(사거리밖) → BYPASS", act([T(0, 140)]), ACTION_BYPASS))
    checks.append(("단일 15m(너무 가깝·노출) → RETURN", act([T(0, 15)]), ACTION_RETURN))
    checks.append(("2대 중 1대 too_close(교전불가) → RETURN", act([T(0, 25, "a"), T(0, 15, "b")]), ACTION_RETURN))
    checks.append(("2대 다 교전가능 → ENGAGE", act([T(0, 25, "a"), T(30, 0, "b")]), ACTION_ENGAGE))
    checks.append(("초소(non-tank) 신규는 위협서 제외", len(classify_new_threats([T(0, 25, "h", "house")], [])), 0))

    if risk.los_available():
        BB = [{"x_min": -3.0, "x_max": 3.0, "z_min": 10.0, "z_max": 12.0}]  # 25m 표적 앞 차폐
        checks.append(("25m LoS 차폐(교전불가·위험0) → BYPASS", act([T(0, 25)], BB), ACTION_BYPASS))
    else:
        print("  (LoS 기하 미가용 — 차폐 케이스 skip, graceful degrade)")

    ok = True
    for name, got, want in checks:
        mark = "OK" if got == want else "FAIL"
        if got != want:
            ok = False
        print(f"  [{mark}] {name}  (got={got}, want={want})")
    print("sudden_decision self-test:", "ALL PASS" if ok else "★FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_selftest())
