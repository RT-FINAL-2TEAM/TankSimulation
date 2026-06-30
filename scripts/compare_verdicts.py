#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""수식기 verdict ↔ LLM verdict 비교기 — 동일 입력(risk_features) 기반 두 판단을 나란히.

입력:
    recon_reports/formula_verdict.json   (generate_recon_report.py 생성, 수식 점수)
    recon_reports/route_risk_result.json (route_risk_node 생성, LLM 판단)
출력:
    recon_reports/risk_comparison.json   (기계가독: winner/band/rank/발산)
    recon_reports/risk_comparison.md     (사람이 보는 비교표)

정합 3중: ① winner 일치(가장 견고) ② 밴드 매핑(연속 risk_total→범주) ③ 순위 일치.
블라인드: LLM은 수식 점수를 보지 않고 같은 raw feature로 독립 판정 → 여기서 사후 비교.

사용:
    python3 scripts/compare_verdicts.py
    python3 scripts/compare_verdicts.py --input recon_reports
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Optional

DEFAULT_DIR = "recon_reports"

# 연속 risk_total(0~1) → LLM 4밴드 양자화 컷오프(튜닝 가능).
BAND_CUTOFFS = [(0.25, "low"), (0.50, "medium"), (0.75, "high")]
BAND_ORDER = {"low": 0, "medium": 1, "high": 2, "critical": 3}


def _read_json(path: str) -> Optional[dict]:
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def risk_total_to_band(x: Optional[float]) -> Optional[str]:
    """수식 risk_total(0~1)을 low/medium/high/critical 밴드로."""
    if x is None:
        return None
    for cut, label in BAND_CUTOFFS:
        if x < cut:
            return label
    return "critical"


def _sign(v: Optional[float]) -> Optional[int]:
    if v is None:
        return None
    if v > 1e-9:
        return 1
    if v < -1e-9:
        return -1
    return 0


def _formula_top_component(route_node: dict) -> Optional[str]:
    """수식 점수에서 기여(contrib)가 가장 큰 위험 항목."""
    comps = route_node.get("components") if isinstance(route_node.get("components"), dict) else {}
    best, best_c = None, -1.0
    for key, cell in comps.items():
        c = cell.get("contrib")
        if isinstance(c, (int, float)) and c > best_c:
            best_c, best = float(c), key
    return best


def build_comparison(formula: dict, risk_result: dict) -> dict:
    fv_routes = formula.get("routes") if isinstance(formula.get("routes"), dict) else {}
    llm = risk_result.get("result") if isinstance(risk_result.get("result"), dict) else {}
    llm_levels = llm.get("risk_level") if isinstance(llm.get("risk_level"), dict) else {}
    llm_risks = llm.get("key_risks") if isinstance(llm.get("key_risks"), dict) else {}

    f_winner = formula.get("winner")
    l_winner = str(llm.get("selected_route") or "").strip().upper() or None
    if l_winner not in {"A", "B"}:
        l_winner = None

    routes = sorted(set(fv_routes) | set(llm_levels) | {"A", "B"})
    per_route: Dict[str, Any] = {}
    risk_totals: Dict[str, Optional[float]] = {}
    for r in routes:
        fnode = fv_routes.get(r) if isinstance(fv_routes.get(r), dict) else {}
        rt = fnode.get("risk_total")
        risk_totals[r] = rt
        f_band = risk_total_to_band(rt)
        l_band = llm_levels.get(r)
        krisks = llm_risks.get(r) if isinstance(llm_risks.get(r), list) else []
        per_route[r] = {
            "formula": {
                "risk_total": rt,
                "band": f_band,
                "top_component": _formula_top_component(fnode),
            },
            "llm": {
                "risk_level": l_band,
                "top_risk": (str(krisks[0]) if krisks else None),
            },
            "band_match": (f_band is not None and l_band is not None and f_band == l_band),
        }

    # 순위 일치: 수식 sign(rtA-rtB) vs LLM 밴드 순서 sign
    rank_agreement = None
    if risk_totals.get("A") is not None and risk_totals.get("B") is not None:
        f_sign = _sign(risk_totals["A"] - risk_totals["B"])
        la, lb = llm_levels.get("A"), llm_levels.get("B")
        if la in BAND_ORDER and lb in BAND_ORDER:
            l_sign = _sign(BAND_ORDER[la] - BAND_ORDER[lb])
            rank_agreement = (f_sign == l_sign)

    divergence = []
    for r in routes:
        pr = per_route[r]
        if not pr["band_match"] and pr["formula"]["band"] and pr["llm"]["risk_level"]:
            divergence.append({
                "route": r,
                "formula_band": pr["formula"]["band"],
                "llm_band": pr["llm"]["risk_level"],
                "formula_top_component": pr["formula"]["top_component"],
                "llm_top_risk": pr["llm"]["top_risk"],
            })

    return {
        "schema_version": "1.0",
        "blind_mode": True,
        "model": risk_result.get("model"),
        "winner": {"formula": f_winner, "llm": l_winner,
                   "agreement": (f_winner is not None and f_winner == l_winner)},
        "per_route": per_route,
        "rank_agreement": rank_agreement,
        "confidence": {"formula": formula.get("confidence"), "llm": llm.get("confidence"),
                       "agreement": (formula.get("confidence") == llm.get("confidence"))},
        "divergence": divergence,
        "narrative": {
            "formula_reason": formula.get("reason"),
            "llm_summary": llm.get("summary"),
            "llm_decision_reason": llm.get("decision_reason"),
        },
    }


def _yn(v: Optional[bool]) -> str:
    if v is None:
        return "—"
    return "✅" if v else "❌"


def render_markdown(cmp: dict) -> str:
    L = []
    L.append("# 정찰 위험도 — 수식기 vs LLM 비교 (블라인드)")
    L.append("")
    L.append(f"> 동일 입력(`risk_features.json`) 기반 두 판단 비교. LLM 모델: `{cmp.get('model') or '-'}`")
    L.append("")
    w = cmp["winner"]
    L.append(f"## 선택 루트: 수식 **{w.get('formula') or '-'}** / LLM **{w.get('llm') or '-'}** {_yn(w.get('agreement'))}")
    L.append("")
    L.append("| 항목 | 수식기 | LLM | 일치 |")
    L.append("|---|---|---|---|")
    for r in sorted(cmp["per_route"]):
        pr = cmp["per_route"][r]
        f, l = pr["formula"], pr["llm"]
        rt = f.get("risk_total")
        rt_s = f"{rt:.3f}" if isinstance(rt, (int, float)) else "—"
        L.append(f"| route_{r} 위험 | {rt_s} ({f.get('band') or '—'}) | {l.get('risk_level') or '—'} | {_yn(pr['band_match'])} |")
    c = cmp["confidence"]
    L.append(f"| 신뢰도 | {c.get('formula') or '—'} | {c.get('llm') or '—'} | {_yn(c.get('agreement'))} |")
    L.append(f"| 순위(A vs B) 일치 | — | — | {_yn(cmp.get('rank_agreement'))} |")
    L.append("")

    if cmp["divergence"]:
        L.append("## 발산 분석 (밴드 불일치 루트)")
        L.append("")
        L.append("| 루트 | 수식 밴드(최대기여) | LLM 밴드(최우선 위험) |")
        L.append("|---|---|---|")
        for d in cmp["divergence"]:
            L.append(f"| route_{d['route']} | {d['formula_band']} ({d.get('formula_top_component') or '-'}) | "
                     f"{d['llm_band']} ({d.get('llm_top_risk') or '-'}) |")
        L.append("")

    n = cmp["narrative"]
    L.append("## 판단 근거")
    L.append("")
    L.append(f"- **수식 근거**: {n.get('formula_reason') or '-'}")
    L.append(f"- **LLM 요약**: {n.get('llm_summary') or '-'}")
    L.append(f"- **LLM 근거**: {n.get('llm_decision_reason') or '-'}")
    L.append("")
    return "\n".join(L)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="수식기 verdict vs LLM verdict 비교표 생성")
    p.add_argument("--input", default=DEFAULT_DIR, help=f"recon_reports 디렉터리 (기본 {DEFAULT_DIR})")
    p.add_argument("--stdout", action="store_true", help="md를 파일 대신 표준출력")
    return p


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    report_dir = args.input
    formula = _read_json(os.path.join(report_dir, "formula_verdict.json"))
    risk_result = _read_json(os.path.join(report_dir, "route_risk_result.json"))

    if not isinstance(formula, dict):
        print("[오류] formula_verdict.json 없음 — generate_recon_report.py를 먼저 실행하세요.", file=sys.stderr)
        return 2
    if not isinstance(risk_result, dict):
        print("[오류] route_risk_result.json 없음 — route_risk_node(LLM)를 먼저 실행하세요.", file=sys.stderr)
        return 3

    cmp = build_comparison(formula, risk_result)
    json_path = os.path.join(report_dir, "risk_comparison.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(cmp, f, ensure_ascii=False, indent=2)

    md = render_markdown(cmp)
    if args.stdout:
        print(md)
    else:
        md_path = os.path.join(report_dir, "risk_comparison.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md)
        print(f"[완료] 비교 산출: {json_path} · {md_path}")
        w = cmp["winner"]
        print(f"  선택: 수식 {w.get('formula')} / LLM {w.get('llm')} (winner 일치={w.get('agreement')})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
