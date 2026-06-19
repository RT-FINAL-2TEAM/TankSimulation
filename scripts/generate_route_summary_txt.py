#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create a compact TXT reconnaissance route report.

Inputs:
  recon_reports/route_A.json
  recon_reports/route_B.json
  recon_reports/route_comparison.json
  recon_reports/route_risk_result.json

Output:
  recon_reports/route_analysis_report.txt
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT_DIR = PROJECT_ROOT / "recon_reports"
DEFAULT_OUTPUT = DEFAULT_REPORT_DIR / "route_analysis_report.txt"


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else None


def fmt(value: Any, suffix: str = "", default: str = "-") -> str:
    if value is None:
        return default
    if isinstance(value, float):
        return f"{value:.2f}{suffix}"
    return f"{value}{suffix}"


def route_metrics(report: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(report, dict):
        return {}
    result = report.get("result") if isinstance(report.get("result"), dict) else {}
    obstacle = report.get("obstacle_summary") if isinstance(report.get("obstacle_summary"), dict) else {}
    exposure = report.get("exposure") if isinstance(report.get("exposure"), dict) else {}
    terrain = report.get("terrain_roughness") if isinstance(report.get("terrain_roughness"), dict) else {}
    vision = report.get("vision_yolo") if isinstance(report.get("vision_yolo"), dict) else {}
    return {
        "route": report.get("route"),
        "reached": result.get("reached"),
        "distance_m": result.get("distance_m"),
        "sim_time_s": result.get("sim_time_s"),
        "collisions": result.get("collisions"),
        "obstacle_count": obstacle.get("count"),
        "obstacle_density_per_100m": obstacle.get("density_per_100m"),
        "enemy_visible_time_s": exposure.get("total_dwell_s"),
        "max_continuous_visible_time_s": exposure.get("max_continuous_s"),
        "exposure_event_count": exposure.get("detection_count"),
        "pitch_std_deg": terrain.get("pitch_std_deg"),
        "roll_std_deg": terrain.get("roll_std_deg"),
        "yolo_counts": vision.get("counts") if isinstance(vision.get("counts"), dict) else {},
        "asset_spotted_gt": report.get("asset_spotted_gt") if isinstance(report.get("asset_spotted_gt"), dict) else {},
    }


def risk_result(report: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(report, dict):
        return {}
    result = report.get("result")
    return result if isinstance(result, dict) else report


def route_risk(result: Dict[str, Any], route_id: str) -> Dict[str, Any]:
    risk_level = result.get("risk_level") if isinstance(result.get("risk_level"), dict) else {}
    key_risks = result.get("key_risks") if isinstance(result.get("key_risks"), dict) else {}
    evidence = result.get("used_evidence") if isinstance(result.get("used_evidence"), dict) else {}
    return {
        "level": risk_level.get(route_id),
        "risks": key_risks.get(route_id) if isinstance(key_risks.get(route_id), list) else [],
        "evidence": evidence.get(route_id) if isinstance(evidence.get(route_id), dict) else {},
    }


def render_route_section(route_id: str, metrics: Dict[str, Any], llm: Dict[str, Any]) -> list[str]:
    route_llm = route_risk(llm, route_id)
    yolo_counts = metrics.get("yolo_counts") if isinstance(metrics.get("yolo_counts"), dict) else {}
    gt_counts = metrics.get("asset_spotted_gt") if isinstance(metrics.get("asset_spotted_gt"), dict) else {}
    lines = [
        f"[ROUTE {route_id}]",
        f"- 도착 여부: {metrics.get('reached')}",
        f"- 이동 거리 / 시간: {fmt(metrics.get('distance_m'), 'm')} / {fmt(metrics.get('sim_time_s'), 's')}",
        f"- 충돌 / 장애물: {fmt(metrics.get('collisions'))} / {fmt(metrics.get('obstacle_count'))}",
        f"- 장애물 밀도: {fmt(metrics.get('obstacle_density_per_100m'), '/100m')}",
        f"- 노출 시간 / 최대 연속 노출: {fmt(metrics.get('enemy_visible_time_s'), 's')} / {fmt(metrics.get('max_continuous_visible_time_s'), 's')}",
        f"- 노출 이벤트 수: {fmt(metrics.get('exposure_event_count'))}",
        f"- 지형 안정성 pitch/roll std: {fmt(metrics.get('pitch_std_deg'), 'deg')} / {fmt(metrics.get('roll_std_deg'), 'deg')}",
        f"- YOLO counts: {json.dumps(yolo_counts, ensure_ascii=False)}",
        f"- GT spotted: {json.dumps(gt_counts, ensure_ascii=False)}",
        f"- LLM 위험도: {fmt(route_llm.get('level'))}",
    ]
    risks = route_llm.get("risks") or []
    if risks:
        lines.append("- LLM 요약 위험요인:")
        for item in risks[:4]:
            lines.append(f"  * {item}")
    else:
        lines.append("- LLM 요약 위험요인: -")
    return lines


def build_report(report_dir: Path) -> str:
    route_a = load_json(report_dir / "route_A.json")
    route_b = load_json(report_dir / "route_B.json")
    comparison = load_json(report_dir / "route_comparison.json")
    risk_report = load_json(report_dir / "route_risk_result.json")
    validated = bool(risk_report and risk_report.get("validated_ok"))
    result = risk_result(risk_report) if validated else {}

    lines = [
        "TANK-CV RECON ROUTE ANALYSIS REPORT",
        f"created_at: {datetime.now().isoformat(timespec='seconds')}",
        f"report_dir: {report_dir}",
        "",
        "[MISSION FLOW]",
        "1. Route A autonomous reconnaissance",
        "2. Simulator restart / return to start",
        "3. Route B autonomous reconnaissance",
        "4. route_A.json + route_B.json -> comparison.json",
        "5. comparison.json -> route_comparison.json -> LLM route risk analysis",
        "",
        "[LLM ANALYSIS]",
        f"- validated_ok: {validated}",
        f"- model: {risk_report.get('model') if isinstance(risk_report, dict) else '-'}",
        f"- selected_route: {result.get('selected_route') if isinstance(result, dict) else '-'}",
        f"- confidence: {result.get('confidence') if isinstance(result, dict) else '-'}",
        f"- summary: {result.get('summary') if isinstance(result, dict) else '-'}",
        f"- decision_reason: {result.get('decision_reason') if isinstance(result, dict) else '-'}",
        "",
    ]

    lines.extend(render_route_section("A", route_metrics(route_a), result))
    lines.append("")
    lines.extend(render_route_section("B", route_metrics(route_b), result))

    if comparison:
        lines.extend([
            "",
            "[LLM INPUT JSON]",
            json.dumps(comparison, ensure_ascii=False, indent=2),
        ])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate compact route analysis TXT report.")
    parser.add_argument("--input", default=str(DEFAULT_REPORT_DIR), help="recon_reports directory")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="output txt path")
    args = parser.parse_args()

    report_dir = Path(args.input).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_report(report_dir), encoding="utf-8")
    print(f"[완료] 루트 분석 TXT 보고서 작성: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
