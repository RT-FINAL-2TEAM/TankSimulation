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
import sys
from pathlib import Path
from typing import Any, Dict, Optional


DEFAULT_INPUT = "recon_reports/risk_features.json"
DEFAULT_OUTPUT = "recon_reports/route_comparison.json"
DEFAULT_REPORT_DIR = "recon_reports"


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


def summarize_route(rec: Dict[str, Any]) -> Dict[str, Any]:
    """risk_features.json의 per-route 레코드(그룹별) → LLM 입력용 평면 요약.

    적 수는 센서퓨전 확정 dedup(threat.confirmed_count), 노출은 길이비(exposure),
    효율은 별도 축. yolo_counts_raw는 '누적 프레임'으로 명시해 적 수 오독을 막는다.
    """
    threat = rec.get("threat") if isinstance(rec.get("threat"), dict) else {}
    exp = rec.get("exposure") if isinstance(rec.get("exposure"), dict) else {}
    terr = rec.get("terrain") if isinstance(rec.get("terrain"), dict) else {}
    eff = rec.get("efficiency") if isinstance(rec.get("efficiency"), dict) else {}
    qual = rec.get("quality") if isinstance(rec.get("quality"), dict) else {}
    raw = rec.get("raw_ref") if isinstance(rec.get("raw_ref"), dict) else {}
    exp_ok = bool(exp.get("available"))

    return {
        "route_id": rec.get("route_id"),
        "reached": bool(rec.get("reached", False)),

        # 위협 — 확정 dedup 수(YOLO 누적 아님)
        "enemy_count": safe_int(threat.get("confirmed_count"), 0),
        "enemy_by_class": threat.get("by_class", {}) if isinstance(threat.get("by_class"), dict) else {},
        "closest_enemy_distance_m": threat.get("nearest_dist_m"),  # null=위협 미탐지(판단 제외)

        # 노출(은밀성) — 1차 위험 신호, 길이비 0~1
        "stealth_ratio": round_float(exp.get("stealth_ratio"), 4) if exp_ok else None,
        "proximity_ratio": round_float(exp.get("proximity_ratio"), 4) if exp_ok else None,
        "exposure_available": exp_ok,

        # 효율(별도 축 — 위험 아님)
        "distance_m": round_float(eff.get("distance_m"), 3, 0.0),
        "sim_time_s": round_float(eff.get("sim_time_s"), 3, 0.0),
        "detour_ratio": eff.get("detour_ratio"),
        "collision_count": safe_int(eff.get("collision_count"), 0),
        "obstacle_count": safe_int(eff.get("obstacle_count"), 0),
        "obstacle_density_per_100m": eff.get("obstacle_density_per_100m"),

        # 지형(험지)
        "pitch_std_deg": round_float(terr.get("pitch_std_deg"), 3, 0.0),
        "roll_std_deg": round_float(terr.get("roll_std_deg"), 3, 0.0),
        "terrain_sigma_deg": round_float(terr.get("sigma_deg"), 3, 0.0),

        # 정찰 신뢰도(점수 미반영, 신뢰도 캡용)
        "gt_found": qual.get("found"),
        "gt_total": qual.get("gt_total"),
        "gt_confidence": qual.get("confidence"),

        # 참고 전용
        "yolo_counts_raw": raw.get("yolo_counts", {}) if isinstance(raw.get("yolo_counts"), dict) else {},
        "asset_spotted_gt": raw.get("asset_spotted_gt", {}) if isinstance(raw.get("asset_spotted_gt"), dict) else {},
    }


def build_llm_input(features: Dict[str, Any]) -> Dict[str, Any]:
    """risk_features.json dict → LLM 입력(route_comparison.json) dict."""
    route_a = features.get("route_A")
    route_b = features.get("route_B")

    if not isinstance(route_a, dict) or "threat" not in route_a:
        raise ValueError(
            "risk_features.json 형식이 아닙니다(route_A.threat 없음) — "
            "generate_recon_report.py를 먼저 실행하세요."
        )
    if not isinstance(route_b, dict) or "threat" not in route_b:
        raise ValueError("risk_features.json에 route_B가 없습니다.")

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
        description="risk_features.json(수식·LLM 공통 입력)을 LLM 입력 JSON으로 변환"
    )

    parser.add_argument(
        "--input",
        "-i",
        default=DEFAULT_INPUT,
        help=f"입력 risk_features.json 경로. 기본: {DEFAULT_INPUT} (없으면 자동 생성)",
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

    # risk_features.json이 아직 없으면 generate_recon_report로 즉석 생성(공유 입력 보장).
    if not Path(args.input).exists():
        try:
            import generate_recon_report as grr  # scripts/ 동일 디렉터리
            grr.build_recon_artifacts(report_dir=DEFAULT_REPORT_DIR)
        except Exception as e:
            print(f"[오류] risk_features 생성 실패: {e}", file=sys.stderr)
            return 1

    features = load_json(args.input)
    llm_input = build_llm_input(features)

    if args.stdout:
        print(json.dumps(llm_input, ensure_ascii=False, indent=2))
    else:
        save_json(llm_input, args.output)
        print(f"[완료] LLM 입력 JSON 생성: {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
