#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
comparison.json -> LLM 입력용 route_A / route_B 요약 JSON 생성기

출력 형태:
{
  "route_A": {
    "reached": true,
    "collision_count": 0,
    "enemy_count": 3,
    "closest_enemy_distance_m": 45.1,
    "enemy_visible_time_s": 20.0,
    "max_continuous_visible_time_s": 3.2,
    "obstacle_count": 25,
    "blocked_segment_count": 2,
    "pitch_std_deg": 3.4,
    "roll_std_deg": 4.8
  },
  "route_B": {
    ...
  }
}

사용:
    python3 make_llm_input.py
    python3 make_llm_input.py -i recon_reports/comparison.json -o recon_reports/route_comparison.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional


DEFAULT_INPUT = "recon_reports/comparison.json"
DEFAULT_OUTPUT = "recon_reports/route_comparison.json"


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def round_float(value: Any, ndigits: int = 3, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return round(float(value), ndigits)
    except Exception:
        return default


def get_yolo_enemy_count(route_data: Dict[str, Any]) -> int:
    """
    YOLO counts 기준 적/위협 수.
    person + tank + house만 위협으로 사용.
    rock은 장애물이므로 제외.
    """
    vision_yolo = route_data.get("vision_yolo", {})
    if not isinstance(vision_yolo, dict):
        return 0

    counts = vision_yolo.get("counts", {})
    if not isinstance(counts, dict):
        return 0

    person = safe_int(counts.get("person"), 0)
    tank = safe_int(counts.get("tank"), 0)
    house = safe_int(counts.get("house"), 0)

    return person + tank + house


def get_closest_enemy_distance(route_data: Dict[str, Any]) -> Optional[float]:
    """
    comparison.json에 closest_enemy_distance_m가 있으면 사용.
    없으면 None 대신 0.0으로 둔다.

    현재 네 comparison.json 구조에는 이 값이 없을 가능성이 큼.
    나중에 생성 쪽에서 closest distance를 넣으면 자동 반영됨.
    """
    candidates = [
        route_data.get("closest_enemy_distance_m"),
        route_data.get("closest_threat_distance_m"),
        route_data.get("min_enemy_distance_m"),
    ]

    for value in candidates:
        if value is not None:
            return round_float(value, 3, 0.0)

    # exposure.per_threat 안에 min_dist_m이 있으면 그중 최솟값 사용
    exposure = route_data.get("exposure", {})
    if isinstance(exposure, dict):
        events = exposure.get("per_threat") or exposure.get("events") or []
        if isinstance(events, list):
            dists = []
            for e in events:
                if isinstance(e, dict) and e.get("min_dist_m") is not None:
                    dists.append(safe_float(e.get("min_dist_m")))
            if dists:
                return round(min(dists), 3)

    return 0.0


def summarize_route(route_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    route_A / route_B 하나를 LLM 입력용으로 최소 요약.
    """
    result = route_data.get("result", {})
    if not isinstance(result, dict):
        result = {}

    obstacle = route_data.get("obstacle_summary", {})
    if not isinstance(obstacle, dict):
        obstacle = {}

    exposure = route_data.get("exposure", {})
    if not isinstance(exposure, dict):
        exposure = {}

    terrain = route_data.get("terrain_roughness", {})
    if not isinstance(terrain, dict):
        terrain = {}

    return {
        "reached": bool(result.get("reached", False)),

        # result.collisions -> collision_count로 이름 통일
        "collision_count": safe_int(result.get("collisions"), 0),

        # YOLO 기준 위협 개수: person + tank + house
        "enemy_count": get_yolo_enemy_count(route_data),

        # 현재 comparison.json에 없으면 0.0
        "closest_enemy_distance_m": get_closest_enemy_distance(route_data),

        # exposure 값
        "enemy_visible_time_s": round_float(
            exposure.get("total_dwell_s"),
            3,
            0.0,
        ),
        "max_continuous_visible_time_s": round_float(
            exposure.get("max_continuous_s"),
            3,
            0.0,
        ),

        # 장애물
        "obstacle_count": safe_int(obstacle.get("count"), 0),

        # 현재 comparison.json에 없으면 0
        "blocked_segment_count": safe_int(
            route_data.get("blocked_segment_count"),
            0,
        ),

        # 지형 roughness
        "pitch_std_deg": round_float(
            terrain.get("pitch_std_deg"),
            3,
            0.0,
        ),
        "roll_std_deg": round_float(
            terrain.get("roll_std_deg"),
            3,
            0.0,
        ),
    }


def build_llm_input(comparison_data: Dict[str, Any]) -> Dict[str, Any]:
    route_a = comparison_data.get("route_A")
    route_b = comparison_data.get("route_B")

    if not isinstance(route_a, dict):
        raise ValueError("comparison.json에 route_A가 없습니다.")

    if not isinstance(route_b, dict):
        raise ValueError("comparison.json에 route_B가 없습니다.")

    return {
        "route_A": summarize_route(route_a),
        "route_B": summarize_route(route_b),
    }


def load_json(path: str) -> Dict[str, Any]:
    p = Path(path)

    if not p.exists():
        raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {p}")

    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Dict[str, Any], path: str) -> None:
    p = Path(path)

    if p.parent and str(p.parent) != ".":
        p.parent.mkdir(parents=True, exist_ok=True)

    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(
        description="comparison.json을 LLM 입력용 최소 JSON으로 변환"
    )

    parser.add_argument(
        "--input",
        "-i",
        default=DEFAULT_INPUT,
        help=f"입력 comparison.json 경로. 기본: {DEFAULT_INPUT}",
    )

    parser.add_argument(
        "--output",
        "-o",
        default=DEFAULT_OUTPUT,
        help=f"출력 JSON 경로. 기본: {DEFAULT_OUTPUT}",
    )

    parser.add_argument(
        "--stdout",
        action="store_true",
        help="파일 저장 대신 stdout으로 출력",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    comparison_data = load_json(args.input)
    llm_input = build_llm_input(comparison_data)

    if args.stdout:
        print(json.dumps(llm_input, ensure_ascii=False, indent=2))
    else:
        save_json(llm_input, args.output)
        print(f"[완료] LLM 입력 JSON 생성: {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())