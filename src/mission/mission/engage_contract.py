# -*- coding: utf-8 -*-
"""교전(사격) 인터페이스 계약 — 단일 출처.

이 repo는 커스텀 `.msg`를 안 쓰고 `std_msgs/String` + JSON 관행을 따른다(SCENARIO2_DESIGN §12.2 #2).
decision_node(요청측)와 mock_turret_node(=실제 turret 제어의 stand-in, 응답측)가 이 모듈 하나만
공유해 직렬화/토픽명이 어긋나지 않게 한다. 팀원의 실제 turret 제어도 이 계약을 구독/발행하면 된다.

스키마:
  /tank/engage/request → {"target_id": str, "pose": {"x": float, "y": float},
                          "distance_m": float, "los": bool}
  /tank/engage/result  → {"target_id": str, "impact": {"x": float, "y": float},
                          "success": bool, "dist_to_target_m": float}

좌표는 모두 map 평면좌표(x=map.x, y=map.y).
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

TOPIC_ENGAGE_REQUEST = "/tank/engage/request"
TOPIC_ENGAGE_RESULT = "/tank/engage/result"


def make_engage_request(target_id: str, x: float, y: float,
                        distance_m: float, los: bool) -> str:
    """교전 요청 JSON 문자열을 만든다(String.data 에 그대로 싣는다)."""
    return json.dumps({
        "target_id": str(target_id),
        "pose": {"x": float(x), "y": float(y)},
        "distance_m": float(distance_m),
        "los": bool(los),
    }, ensure_ascii=False)


def parse_engage_request(data: str) -> Optional[Dict[str, Any]]:
    """교전 요청 문자열을 dict로 파싱한다. 형식 불량이면 None."""
    try:
        payload = json.loads(data)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    pose = payload.get("pose") if isinstance(payload.get("pose"), dict) else {}
    try:
        return {
            "target_id": str(payload.get("target_id", "")),
            "x": float(pose.get("x", 0.0)),
            "y": float(pose.get("y", 0.0)),
            "distance_m": float(payload.get("distance_m", 0.0)),
            "los": bool(payload.get("los", False)),
        }
    except (TypeError, ValueError):
        return None


def make_engage_result(target_id: str, impact_x: float, impact_y: float,
                       success: bool, dist_to_target_m: float) -> str:
    """교전 결과 JSON 문자열을 만든다."""
    return json.dumps({
        "target_id": str(target_id),
        "impact": {"x": float(impact_x), "y": float(impact_y)},
        "success": bool(success),
        "dist_to_target_m": float(dist_to_target_m),
    }, ensure_ascii=False)


def parse_engage_result(data: str) -> Optional[Dict[str, Any]]:
    """교전 결과 문자열을 dict로 파싱한다. 형식 불량이면 None."""
    try:
        payload = json.loads(data)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    impact = payload.get("impact") if isinstance(payload.get("impact"), dict) else {}
    try:
        return {
            "target_id": str(payload.get("target_id", "")),
            "impact_x": float(impact.get("x", 0.0)),
            "impact_y": float(impact.get("y", 0.0)),
            "success": bool(payload.get("success", False)),
            "dist_to_target_m": float(payload.get("dist_to_target_m", 0.0)),
        }
    except (TypeError, ValueError):
        return None
