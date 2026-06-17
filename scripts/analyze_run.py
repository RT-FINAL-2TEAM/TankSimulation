#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""주행 품질 진단 — 정찰 런의 진동/끼임 원인을 경로 / APF / 제어로 수치 귀속.

**시나리오 공식 산출물(recon_report.md)이 아니라 개발/현상태 분석용**이다. route_*.json의
diagnostics 블록(계획경로·lookahead·local_target·route_version·명령)을 읽어 4개 점수로 원인을 가른다:

  - 경로 churn  : route_version 변화 횟수 + 계획경로 교체 횟수      → 크면 '경로(planner 재계획)'
  - 추종 오차   : 실제 위치 ↔ 계획경로 횡거리(평균/최대)            → 크면 '제어/APF가 경로를 못 따라감'
  - APF 불일치  : lookahead 방향 vs APF(local_target) 방향 각도차   → 크면 'APF가 경로와 싸움'
  - 제어 채터   : moveAD A↔D 토글 빈도                             → 크면 '제어 떨림(팀원)'

추가로 계획 vs 실제 궤적 오버레이 PNG와 (참고·캐비엇) 확인탐지 vs GT를 낸다.
→ recon_reports/run_diagnosis.md + diag_overlay_{A,B}.png

사용: python3 scripts/analyze_run.py
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(PROJECT_ROOT, "recon_reports")
DEFAULT_MAP = os.path.join(PROJECT_ROOT, "src", "rviz_visualization", "map", "finalmap.map")

for _cjk in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
             "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"):
    if os.path.exists(_cjk):
        try:
            font_manager.fontManager.addfont(_cjk)
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=_cjk).get_name()
            break
        except Exception:
            pass
plt.rcParams["axes.unicode_minus"] = False


# --------------------------------------------------------------------------- #
# 기하 헬퍼
# --------------------------------------------------------------------------- #
def seg_point_dist(px, pz, ax, az, bx, bz):
    dx, dz = bx - ax, bz - az
    L2 = dx * dx + dz * dz
    if L2 < 1e-9:
        return math.hypot(px - ax, pz - az)
    t = max(0.0, min(1.0, ((px - ax) * dx + (pz - az) * dz) / L2))
    return math.hypot(px - (ax + t * dx), pz - (az + t * dz))


def point_to_path(px, pz, path):
    if not path:
        return None
    if len(path) == 1:
        return math.hypot(px - path[0][0], pz - path[0][1])
    return min(seg_point_dist(px, pz, path[i][0], path[i][1], path[i + 1][0], path[i + 1][1])
               for i in range(len(path) - 1))


def active_path(planned, ts):
    """샘플 시각 ts에서 유효한(가장 최근에 발행된) 계획경로."""
    cur = None
    for pp in planned:
        if pp.get("t", 0.0) <= ts:
            cur = pp.get("path")
    if cur is None and planned:
        cur = planned[0].get("path")
    return cur


def angle_at(p, a, b):
    """p에서 본 p→a 와 p→b 사이 각도(도). 너무 가까우면 None."""
    v1 = (a[0] - p[0], a[1] - p[1])
    v2 = (b[0] - p[0], b[1] - p[1])
    n1, n2 = math.hypot(*v1), math.hypot(*v2)
    if n1 < 0.3 or n2 < 0.3:
        return None
    c = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
    return math.degrees(math.acos(c))


def lateral_cmd(cmd_str):
    try:
        c = json.loads(cmd_str).get("moveAD", {}).get("command", "")
        return c if c in ("A", "D") else ""
    except Exception:
        return ""


# --------------------------------------------------------------------------- #
# 점수 계산
# --------------------------------------------------------------------------- #
def analyze_route(d):
    diag = d.get("diagnostics") or {}
    samples = diag.get("samples") or []
    planned = diag.get("planned_paths") or []
    if not samples:
        return None  # 진단 데이터 없음(옛 런) → 재주행 필요

    follow_errs, apf_angles = [], []
    for s in samples:
        p = s.get("p")
        if not p:
            continue
        path = active_path(planned, s.get("t", 0.0))
        if path:
            fe = point_to_path(p[0], p[1], path)
            if fe is not None:
                follow_errs.append(fe)
        look, ltgt = s.get("look"), s.get("ltgt")
        if look and ltgt:
            a = angle_at(p, look, ltgt)
            if a is not None:
                apf_angles.append(a)

    # 제어 채터: A↔D 토글
    toggles, last = 0, ""
    for s in samples:
        lat = lateral_cmd(s.get("cmd", ""))
        if lat and last and lat != last:
            toggles += 1
        if lat:
            last = lat
    dur = (samples[-1].get("t", 0.0) - samples[0].get("t", 0.0)) or 1.0

    def stat(xs):
        return {"mean": round(sum(xs) / len(xs), 2), "max": round(max(xs), 2), "n": len(xs)} if xs else None

    return {
        "samples": len(samples),
        "duration_s": round(dur, 1),
        "route_churn": {
            "route_version_changes": diag.get("route_version_changes", 0),
            "planned_path_swaps": len(planned),
        },
        "follow_error_m": stat(follow_errs),
        "apf_disagree_deg": stat(apf_angles),
        "control_chatter": {"ad_toggles": toggles, "per_min": round(toggles / dur * 60.0, 1)},
        "_samples": samples,
        "_planned": planned,
    }


def gt_threat_counts(map_path):
    try:
        data = json.load(open(map_path, encoding="utf-8"))
    except Exception:
        return {}
    c = {"outposts": 0, "tanks": 0, "soldiers": 0}
    for o in data.get("obstacles", []):
        n = str(o.get("prefabName", ""))
        if n.startswith("House002"):
            c["outposts"] += 1
        elif n.startswith("Tank"):
            c["tanks"] += 1
        elif n.startswith("Human"):
            c["soldiers"] += 1
    return c


# --------------------------------------------------------------------------- #
# 오버레이 그림 (계획 vs 실제)
# --------------------------------------------------------------------------- #
def render_overlay(rid, d, res, out_dir):
    samples = res["_samples"]
    planned = res["_planned"]
    fig, ax = plt.subplots(figsize=(10, 10), dpi=130)

    # GT 객체(맥락) 옅게
    try:
        from collections import defaultdict
        data = json.load(open(DEFAULT_MAP, encoding="utf-8"))
        pts = defaultdict(list)
        for o in data.get("obstacles", []):
            n = str(o.get("prefabName", ""))
            cls = next((k for k in ("Tree", "Rock", "Car", "House", "Tent", "Human") if n.startswith(k)), None)
            if cls:
                pts[cls].append((o["position"]["x"], o["position"]["z"]))
        col = {"Tree": "#7fae7f", "Rock": "#a9742f", "Car": "#ff8c1a", "House": "#d62728",
               "Tent": "#c2a36b", "Human": "#cfcfcf"}
        for cls, ps in pts.items():
            ax.scatter([p[0] for p in ps], [p[1] for p in ps], s=10, color=col[cls], alpha=0.4, zorder=1)
    except Exception:
        pass

    # 계획경로(들) — 마지막 것 굵게
    for i, pp in enumerate(planned):
        path = pp.get("path") or []
        if len(path) < 2:
            continue
        last = (i == len(planned) - 1)
        ax.plot([x for x, _ in path], [z for _, z in path], "--",
                color="#888888" if not last else "#000000",
                lw=1.0 if not last else 2.0, alpha=0.5 if not last else 0.9,
                label="계획경로(최신)" if last else None, zorder=3)

    # 실제 궤적 + 추종오차 큰 지점 빨강
    xs = [s["p"][0] for s in samples if s.get("p")]
    zs = [s["p"][1] for s in samples if s.get("p")]
    ax.plot(xs, zs, "-", color="#1f6fd6", lw=1.8, alpha=0.9, label="실제 주행", zorder=4)
    bigx, bigz = [], []
    for s in samples:
        p = s.get("p")
        path = active_path(planned, s.get("t", 0.0))
        if p and path:
            fe = point_to_path(p[0], p[1], path)
            if fe is not None and fe > 5.0:
                bigx.append(p[0]); bigz.append(p[1])
    if bigx:
        ax.scatter(bigx, bigz, s=24, color="#d62728", edgecolors="black", linewidths=0.4,
                   zorder=6, label=f"추종오차>5m ({len(bigx)})")
    if xs:
        ax.scatter([xs[0]], [zs[0]], s=130, marker="o", color="#1a9850",
                   edgecolors="black", zorder=7, label="출발")

    fe = res["follow_error_m"]
    ax.set_title(f"route_{rid} 계획 vs 실제 — 추종오차 평균 "
                 f"{fe['mean'] if fe else '?'}m / 최대 {fe['max'] if fe else '?'}m", fontsize=11)
    ax.set_xlim(0, 300); ax.set_ylim(0, 300); ax.set_aspect("equal")
    ax.set_xlabel("map x [m]"); ax.set_ylabel("map z [m]")
    ax.grid(True, alpha=0.2); ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    out = os.path.join(out_dir, f"diag_overlay_{rid}.png")
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# 마크다운
# --------------------------------------------------------------------------- #
def verdict(res):
    """4점수로 1차 원인 추정."""
    fe = (res["follow_error_m"] or {}).get("mean", 0) or 0
    apf = (res["apf_disagree_deg"] or {}).get("mean", 0) or 0
    churn = res["route_churn"]["route_version_changes"]
    chat = res["control_chatter"]["per_min"]
    flags = []
    if churn >= 5:
        flags.append(f"경로 churn 높음(재계획 {churn}회)")
    if fe >= 5:
        flags.append(f"추종오차 큼(평균 {fe}m → 제어/APF가 경로 못 따라감)")
    if apf >= 30:
        flags.append(f"APF 불일치 큼(평균 {apf}° → APF가 경로와 싸움)")
    if chat >= 60:
        flags.append(f"제어 채터 높음({chat}/분 → 제어 떨림, 팀원)")
    return flags or ["뚜렷한 이상치 없음(또는 데이터 부족)"]


def render_md(results, gt, out_dir, figs):
    L = ["# 주행 품질 진단 (개발용 — 시나리오 공식 리포트 아님)", ""]
    L.append("> route_*.json의 diagnostics로 진동/끼임 원인을 **경로 / APF / 제어**로 귀속. 값 ↓ 낮을수록 안정.")
    L.append("")
    if not any(results.values()):
        L.append("## ⚠ 진단 데이터 없음")
        L.append("")
        L.append("route_*.json에 `diagnostics`가 없습니다(로깅 추가 전 런). **스택을 재기동해 A→B를 한 번 재주행**한 뒤 다시 실행하세요.")
        return "\n".join(L)

    L.append("## 1. 원인 귀속 점수")
    L.append("")
    L.append("| 점수 | route_A | route_B | 크면 원인 |")
    L.append("|---|---|---|---|")

    def cell(res, key, sub=None):
        if not res:
            return "—"
        v = res[key]
        if v is None:
            return "N/A"
        if sub:
            return str(v.get(sub))
        return str(v)
    rA, rB = results.get("A"), results.get("B")
    L.append(f"| 경로 churn(재계획 횟수) | {cell(rA,'route_churn','route_version_changes')} | "
             f"{cell(rB,'route_churn','route_version_changes')} | 경로(planner) |")
    L.append(f"| 추종오차 평균(m) | {cell(rA,'follow_error_m','mean')} | {cell(rB,'follow_error_m','mean')} | 제어/APF |")
    L.append(f"| 추종오차 최대(m) | {cell(rA,'follow_error_m','max')} | {cell(rB,'follow_error_m','max')} | 제어/APF |")
    L.append(f"| APF 불일치 평균(°) | {cell(rA,'apf_disagree_deg','mean')} | {cell(rB,'apf_disagree_deg','mean')} | APF |")
    L.append(f"| 제어 채터(A/D 토글/분) | {cell(rA,'control_chatter','per_min')} | {cell(rB,'control_chatter','per_min')} | 제어(팀원) |")
    L.append("")

    for rid in ("A", "B"):
        res = results.get(rid)
        if not res:
            continue
        L.append(f"### route_{rid} 1차 진단")
        for f in verdict(res):
            L.append(f"- {f}")
        if figs.get(rid):
            L.append("")
            L.append(f"![route_{rid} 계획 vs 실제]({os.path.basename(figs[rid])})")
        L.append("")

    L.append("## 2. (참고·캐비엇) 위협 탐지 vs GT")
    L.append("")
    L.append("> ⚠ 인지 검증은 정적맵 GT 한정 + 동적객체 미포함이라 **참고용**. 원시 라이다 클러스터 기반 precision/recall은")
    L.append("> 거친 지형 오탐이 섞여 신뢰도가 낮아 공식 리포트에서 제외했다. 아래는 '확인된(asset_spotted_gt) 위협 vs GT 수'.")
    L.append("")
    L.append("| 클래스 | GT 수 | route_A 확인 | route_B 확인 |")
    L.append("|---|---|---|---|")
    for cls, key in (("초소(House002)", "outposts"), ("적전차(Tank)", "tanks"), ("병사(Human)", "soldiers")):
        a = (results.get("A") or {}).get("_spotted", {}).get(key, "—") if results.get("A") else "—"
        b = (results.get("B") or {}).get("_spotted", {}).get(key, "—") if results.get("B") else "—"
        L.append(f"| {cls} | {gt.get(key,'?')} | {a} | {b} |")
    L.append("")
    L.append("## 3. 다음 판단")
    L.append("- 추종오차가 크면 → route 여유 확보(A의 1m 코리더↑) 또는 APF 게인 재조정.")
    L.append("- APF 불일치가 크면 → APF가 코리더와 싸움(게인/코리더 폭).")
    L.append("- 경로 churn이 크면 → planner 재계획 트리거 점검(이미 goal 임계 10m로 1차 차단).")
    L.append("- 제어 채터만 크면 → 제어 담당 팀원에 PD/히스테리시스 공유.")
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description="주행 품질 진단(개발용)")
    ap.add_argument("--input", default=OUT_DIR)
    ap.add_argument("--map", default=DEFAULT_MAP)
    args = ap.parse_args(argv)

    gt = gt_threat_counts(args.map)
    results, figs = {}, {}
    for rid in ("A", "B"):
        path = os.path.join(args.input, f"route_{rid}.json")
        if not os.path.exists(path):
            results[rid] = None
            continue
        d = json.load(open(path, encoding="utf-8"))
        res = analyze_route(d)
        if res is not None:
            res["_spotted"] = d.get("asset_spotted_gt", {})
            figs[rid] = render_overlay(rid, d, res, args.input)
        results[rid] = res
        if res:
            ch = res["route_churn"]["route_version_changes"]
            fe = res["follow_error_m"]
            print(f"route_{rid}: churn {ch}, 추종오차 {fe['mean'] if fe else '?'}m(평균), "
                  f"APF불일치 {(res['apf_disagree_deg'] or {}).get('mean','?')}°, "
                  f"채터 {res['control_chatter']['per_min']}/분")
        else:
            print(f"route_{rid}: 진단 데이터 없음(diagnostics 미기록) — 재주행 필요")

    md = render_md(results, gt, args.input, figs)
    out = os.path.join(args.input, "run_diagnosis.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"\n[완료] 진단 리포트: {out}")
    if figs:
        print(f"  오버레이: {', '.join(os.path.basename(v) for v in figs.values())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
