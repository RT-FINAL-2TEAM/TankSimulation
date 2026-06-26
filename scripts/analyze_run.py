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
# GT 정답맵(객체 포함 원본; finalmap은 청소돼 객체 없음) — 경로별 '발견했어야 할 객체' 탐지율 산정용.
GT_OBJECTS_MAP = os.path.join(PROJECT_ROOT, "src", "rviz_visualization", "map", "final_v3.map")

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


def nearest_seg_index(px, pz, path):
    """경로 폴리라인에서 가장 가까운 세그먼트 시작 인덱스."""
    best_i, best_d = 0, float("inf")
    for i in range(len(path) - 1):
        d = seg_point_dist(px, pz, path[i][0], path[i][1], path[i + 1][0], path[i + 1][1])
        if d < best_d:
            best_d, best_i = d, i
    return best_i


def curvature_at(path, i):
    """경로 i번째 정점의 방향변화각(도)/길이 — 코너일수록 큼."""
    if i <= 0 or i >= len(path) - 1:
        return 0.0
    a = (path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1])
    b = (path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
    na, nb = math.hypot(*a), math.hypot(*b)
    if na < 1e-6 or nb < 1e-6:
        return 0.0
    c = max(-1.0, min(1.0, (a[0] * b[0] + a[1] * b[1]) / (na * nb)))
    return math.degrees(math.acos(c)) / max(1.0, (na + nb) / 2.0)


# --------------------------------------------------------------------------- #
# 정찰 검증 (weave / stop-to-confirm / 인식 / 충돌)
# --------------------------------------------------------------------------- #
def signed_cross_track(px, pz, path):
    """경로 대비 부호있는 횡거리(좌 +, 우 −). weave(좌우 흔들림) 진동 측정용."""
    if not path or len(path) < 2:
        return None
    i = nearest_seg_index(px, pz, path)
    ax, az = path[i]
    bx, bz = path[i + 1]
    dx, dz = bx - ax, bz - az
    L = math.hypot(dx, dz)
    if L < 1e-6:
        return None
    nx, nz = -dz / L, dx / L            # 좌측 단위 법선
    return (px - ax) * nx + (pz - az) * nz


def load_route(input_dir, rid):
    """route_{rid}.json을 찾는다. 없으면 comparison.json의 route_{rid}로 폴백
    (시나리오2 run이 route_A.json을 지워도 A 분석 가능). analysis/ 하위도 탐색."""
    for sub in (".", "analysis"):
        p = os.path.join(input_dir, sub, f"route_{rid}.json")
        if os.path.exists(p):
            return json.load(open(p, encoding="utf-8")), p
    for sub in (".", "analysis"):
        cp = os.path.join(input_dir, sub, "comparison.json")
        if os.path.exists(cp):
            comp = json.load(open(cp, encoding="utf-8"))
            if isinstance(comp, dict) and comp.get(f"route_{rid}"):
                return comp[f"route_{rid}"], f"{cp}#route_{rid}"
    return None, None


def find_discovered_map(input_dir, rid):
    """발견객체맵 위치(폴더 재구성 전/후 모두 대응)."""
    for sub in ("handoff", "recon_map", "analysis", "."):
        p = os.path.join(input_dir, sub, f"discovered_objects_route_{rid}.map")
        if os.path.exists(p):
            return p
    return None


def recon_recognition(input_dir, rid, yolo_counts):
    """확정 발견객체(센서퓨전 확정) class별 수 + YOLO raw 프레임수."""
    confirmed = {}
    dp = find_discovered_map(input_dir, rid)
    if dp:
        try:
            for o in (json.load(open(dp, encoding="utf-8")).get("obstacles", []) or []):
                meta = o.get("metadata", {}) or {}
                if meta.get("is_confirmed"):
                    c = str(meta.get("class_name", "?"))
                    confirmed[c] = confirmed.get(c, 0) + 1
        except Exception:
            pass
    return {"confirmed_by_class": confirmed, "confirmed_total": sum(confirmed.values()),
            "yolo_raw": yolo_counts or {}}


def gt_detection_rate(route_data, input_dir, rid, gt_map_path, near_m=30.0, match_m=8.0):
    """GT(final_v3.map) 실제 객체 대비 경로별 탐지율 — 'A가 경로 옆 객체를 놓쳤나'에 직답.
    경로 궤적 ≤near_m 이내 GT 객체(=발견했어야 할 것) 중 확정(센서퓨전)으로 잡힌 비율 + 놓친 목록."""
    try:
        gt = json.load(open(gt_map_path, encoding="utf-8"))
    except Exception:
        return None
    objs = []  # (class, map_x, map_y) — map_x=raw.x, map_y=raw.z (좌표규약)
    for o in (gt.get("obstacles", []) or []):
        n = str(o.get("prefabName", ""))
        cls = next((k for k in ("Rock", "Car", "House", "Tank", "Tent") if n.startswith(k)), None)
        if not cls:
            continue
        p = o.get("position", {}) or {}
        objs.append((cls.lower(), float(p.get("x", 0.0)), float(p.get("z", 0.0))))
    samples = ((route_data.get("diagnostics") or {}).get("samples") or [])
    traj = [(s["p"][0], s["p"][1]) for s in samples if s.get("p")]
    if not objs or not traj:
        return None
    confirmed = []  # 확정 객체 위치
    dp = find_discovered_map(input_dir, rid)
    if dp:
        try:
            for o in (json.load(open(dp, encoding="utf-8")).get("obstacles", []) or []):
                m = o.get("metadata", {}) or {}
                if m.get("is_confirmed") and m.get("map_x") is not None:
                    confirmed.append((float(m["map_x"]), float(m["map_y"])))
        except Exception:
            pass
    on_route, detected, missed = 0, 0, []
    for cls, ox, oy in objs:
        md = min(math.hypot(ox - x, oy - y) for x, y in traj)
        if md > near_m:
            continue
        on_route += 1
        if any(math.hypot(ox - cx, oy - cy) <= match_m for cx, cy in confirmed):
            detected += 1
        else:
            missed.append(f"{cls}({round(ox)},{round(oy)})~{round(md)}m")
    return {"on_route": on_route, "detected": detected,
            "rate": (round(detected / on_route, 2) if on_route else None),
            "missed": missed, "near_m": near_m}


def analyze_recon(d):
    """정찰 검증 지표: weave(좌우 흔들림) / stop-to-confirm(정지) / 충돌."""
    diag = d.get("diagnostics") or {}
    samples = diag.get("samples") or []
    planned = diag.get("planned_paths") or []
    if not samples:
        return None
    cts = []
    for s in samples:
        p = s.get("p")
        path = active_path(planned, s.get("t", 0.0))
        if p and path:
            ct = signed_cross_track(p[0], p[1], path)
            if ct is not None:
                cts.append(ct)
    weave_amp = round(sum(abs(c) for c in cts) / len(cts), 2) if cts else None
    weave_max = round(max((abs(c) for c in cts), default=0.0), 2)
    dir_changes, prev = 0, None
    for c in cts:                          # ±0.5m 넘는 좌↔우 전환만 유효 진동으로
        sgn = 1 if c > 0.5 else (-1 if c < -0.5 else 0)
        if sgn != 0:
            if prev is not None and sgn != prev:
                dir_changes += 1
            prev = sgn
    # stop-to-confirm: diag sample의 hold(정지여부) 연속구간 = 의도적 정지 에피소드
    hold_logged = any(("hold" in s) for s in samples)
    episodes, hold_time, in_hold, ep_start = 0, 0.0, False, None
    for s in samples:
        h = bool(s.get("hold"))
        if h and not in_hold:
            episodes += 1
            in_hold = True
            ep_start = s.get("t", 0.0)
        elif not h and in_hold:
            hold_time += s.get("t", 0.0) - (ep_start or 0.0)
            in_hold = False
    if in_hold and ep_start is not None:
        hold_time += samples[-1].get("t", 0.0) - ep_start
    stop_cmds = 0
    for s in samples:                      # 전체 STOP(yaw 떨림 포함) — hold와 비교용
        try:
            if json.loads(s.get("cmd", "{}")).get("moveWS", {}).get("command") == "STOP":
                stop_cmds += 1
        except Exception:
            pass
    # 속도 프로파일(연속 pose 거리/dt, m/s) — 충돌 1순위가 과속이라 진단에 노출.
    spd = []
    for i in range(1, len(samples)):
        p0, p1 = samples[i - 1].get("p"), samples[i].get("p")
        t0, t1 = samples[i - 1].get("t", 0.0), samples[i].get("t", 0.0)
        if p0 and p1 and 0.01 < (t1 - t0) < 2.0:
            spd.append((math.hypot(p1[0] - p0[0], p1[1] - p0[1]) / (t1 - t0), p1[0], p1[1]))
    speed = None
    if spd:
        vs = sorted(s[0] for s in spd)
        n = len(vs)
        # 충돌 군집 진입 속도(≤8m): 과속 진입→ram 확인
        cpts = set((round(c.get("x", 0)), round(c.get("z", 0))) for c in (d.get("collision_events") or []))
        col_vs = [s[0] for s in spd if any(math.hypot(s[1] - cx, s[2] - cy) <= 8 for cx, cy in cpts)]
        speed = {"mean": round(sum(vs) / n, 1), "median": round(vs[n // 2], 1),
                 "p90": round(vs[int(n * 0.9)], 1), "max": round(max(vs), 1),
                 "near_collision_mean": (round(sum(col_vs) / len(col_vs), 1) if col_vs else None)}
    # 정찰 관측 거동(②감속/dwell · ③포탑 step-stare) — diag sample의 obs/cmd에서 집계.
    observe = analyze_observe(samples)
    return {
        "weave": {"amp_mean_m": weave_amp, "amp_max_m": weave_max, "dir_changes": dir_changes, "n": len(cts)},
        "stop_confirm": {"hold_logged": hold_logged,
                         "hold_episodes": episodes, "hold_time_s": round(hold_time, 1),
                         "stop_cmds": stop_cmds, "samples": len(samples)},
        "observe": observe,
        "speed": speed,
        "collisions_count": (d.get("result") or {}).get("collisions"),
        "collision_pts": d.get("collision_events") or [],
        "fusion_rejects": d.get("fusion_rejects") or {},
    }


def analyze_observe(samples):
    """정찰 관측 거동 집계: 미분류 후보 통계 + ②감속/dwell + ③포탑 stare.
    obs(local_path_node 후보요약+mode) 미기록이면 None(구버전 로그 — 재주행 필요)."""
    obs_samples = [s for s in samples if isinstance(s.get("obs"), dict)]
    if not obs_samples:
        return None
    ns = [s["obs"].get("n", 0) for s in obs_samples]
    fov = [s["obs"].get("n_fov", 0) for s in obs_samples]
    side = [s["obs"].get("n_side", 0) for s in obs_samples]
    modes = {}
    cls_max = {}
    for s in obs_samples:
        m = s["obs"].get("mode") or ""
        if m:
            modes[m] = modes.get(m, 0) + 1
        for k, v in (s["obs"].get("by_class") or {}).items():
            cls_max[k] = max(cls_max.get(k, 0), int(v))
    # ③ 포탑 stare 에피소드: cmd의 turretQE!="" 또는 obs.mode==turret 연속구간
    turret_eps, in_t = 0, False
    for s in samples:
        try:
            qe = json.loads(s.get("cmd", "{}")).get("turretQE", {}).get("command", "")
        except Exception:
            qe = ""
        t_on = bool(qe) or (isinstance(s.get("obs"), dict) and s["obs"].get("mode") == "turret")
        if t_on and not in_t:
            turret_eps += 1
            in_t = True
        elif not t_on and in_t:
            in_t = False
    return {
        "samples_with_obs": len(obs_samples),
        "cand_mean": round(sum(ns) / len(ns), 2) if ns else 0.0,
        "cand_fov_mean": round(sum(fov) / len(fov), 2) if fov else 0.0,
        "cand_side_mean": round(sum(side) / len(side), 2) if side else 0.0,
        "by_class_max": cls_max,
        "slow_samples": modes.get("slow", 0),
        "turret_stare_episodes": turret_eps,
    }


def fmt_fusion_rejects(fr):
    """융합 사유 히스토그램 → 'ok N(p%) · 사유 N · …' (성공/실패 한눈에). 왜 확정 안 되나."""
    if not fr:
        return "(없음 — fusion_rejects 미기록, 재주행 필요)"
    total = sum(fr.values()) or 1
    ok = sum(v for k, v in fr.items() if str(k).startswith("ok"))
    items = sorted(fr.items(), key=lambda kv: -kv[1])
    parts = [f"성공 {ok}({100*ok//total}%)"] + [f"{k} {v}" for k, v in items if not str(k).startswith("ok")]
    return " · ".join(parts)


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
    fe_curv = []          # (추종오차, 곡률) — 코너컷 vs 직선이탈 분해용
    track_angles = []     # 이동방향 vs 목표(local_target) 각도 — 제어 추종
    prev_p = None
    for s in samples:
        p = s.get("p")
        if not p:
            continue
        path = active_path(planned, s.get("t", 0.0))
        if path:
            fe = point_to_path(p[0], p[1], path)
            if fe is not None:
                follow_errs.append(fe)
                if len(path) >= 3:
                    fe_curv.append((fe, curvature_at(path, nearest_seg_index(p[0], p[1], path))))
        look, ltgt = s.get("look"), s.get("ltgt")
        if look and ltgt:
            a = angle_at(p, look, ltgt)
            if a is not None:
                apf_angles.append(a)
        # 제어 추종: 직전 대비 실제 이동방향 vs local_target 방향 각도
        if prev_p is not None and ltgt:
            mv = (p[0] - prev_p[0], p[1] - prev_p[1])
            tg = (ltgt[0] - p[0], ltgt[1] - p[1])
            nm, nt = math.hypot(*mv), math.hypot(*tg)
            if nm > 0.05 and nt > 0.3:
                cc = max(-1.0, min(1.0, (mv[0] * tg[0] + mv[1] * tg[1]) / (nm * nt)))
                track_angles.append(math.degrees(math.acos(cc)))
        prev_p = p

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

    # 추종오차 분해: 곡률 하위/상위 33%에서 비교 → 코너에서 크면 lookahead 코너컷, 직선에서도 크면 코리더/제어
    fe_straight = fe_corner = None
    if len(fe_curv) >= 6:
        order = sorted(range(len(fe_curv)), key=lambda j: fe_curv[j][1])
        k = max(1, len(order) // 3)
        fe_straight = round(sum(fe_curv[j][0] for j in order[:k]) / k, 2)
        fe_corner = round(sum(fe_curv[j][0] for j in order[-k:]) / k, 2)
    track = None
    if track_angles:
        track = {"mean": round(sum(track_angles) / len(track_angles), 1),
                 "over90": sum(1 for a in track_angles if a > 90), "n": len(track_angles)}

    return {
        "samples": len(samples),
        "duration_s": round(dur, 1),
        "route_churn": {
            "route_version_changes": diag.get("route_version_changes", 0),
            "planned_path_swaps": len(planned),
        },
        "follow_error_m": stat(follow_errs),
        "follow_decomp": {"straight_m": fe_straight, "corner_m": fe_corner},
        "control_track": track,
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

    # 정찰 검증: 멈춰-확정(hold) 지점(주황 ■) + 충돌 지점(빨강 ✕)
    hx = [s["p"][0] for s in samples if s.get("hold") and s.get("p")]
    hz = [s["p"][1] for s in samples if s.get("hold") and s.get("p")]
    if hx:
        ax.scatter(hx, hz, s=42, marker="s", color="#ff8c1a", edgecolors="black",
                   linewidths=0.3, zorder=5, label=f"멈춰-확정 hold ({len(hx)})")
    coll = d.get("collision_events") or []
    if coll:
        ax.scatter([c.get("x") for c in coll], [c.get("z") for c in coll], s=70, marker="x",
                   color="#d62728", linewidths=1.5, zorder=8, label=f"충돌 ({len(coll)})")

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
    """원인 분해로 1차 진단 — 추종오차를 코너컷(lookahead) / 직선이탈(코리더·제어)로 가른다."""
    fe = (res["follow_error_m"] or {}).get("mean", 0) or 0
    apf = (res["apf_disagree_deg"] or {}).get("mean", 0) or 0
    churn = res["route_churn"]["route_version_changes"]
    chat = res["control_chatter"]["per_min"]
    dec = res.get("follow_decomp") or {}
    st, co = dec.get("straight_m"), dec.get("corner_m")
    trk = res.get("control_track") or {}
    flags = []
    if churn >= 5:
        flags.append(f"경로 churn 높음(재계획 {churn}회) → planner 재계획 점검")
    if fe >= 3:
        if st is not None and co is not None and co - st >= 1.0:
            flags.append(f"추종오차 {fe}m, **코너에서 큼**(직선 {st}/코너 {co}m) → lookahead 코너컷 → **lookahead 단축**")
        elif st is not None and co is not None and (st - co >= 0.5 or trk.get("mean", 0) >= 40):
            flags.append(f"추종오차 {fe}m, **직선에서도 큼**(직선 {st}/코너 {co}m) + 제어추종 {trk.get('mean','?')}°·"
                         f"반대이동 {trk.get('over90','?')}회 → 좁은 코리더 벽충돌·stuck → **코리더 넓히기**")
        else:
            flags.append(f"추종오차 {fe}m (직선 {st}/코너 {co}m) → 경로 여유·lookahead 점검")
    if apf >= 30:
        flags.append(f"APF 불일치 {apf}° → APF가 경로와 싸움")
    if chat >= 60:
        flags.append(f"제어 채터 {chat}/분 → 제어 떨림(팀원)")
    return flags or ["뚜렷한 이상치 없음(또는 데이터 부족)"]


def render_md(results, gt, out_dir, figs, recon=None):
    L = ["# 주행 품질 진단 (개발용 — 시나리오 공식 리포트 아님)", ""]
    L.append("> route_*.json의 diagnostics로 진동/끼임 원인을 **경로 / APF / 제어**로 귀속. 값 ↓ 낮을수록 안정.")
    L.append("")
    if not any(results.values()):
        L.append("## ⚠ 진단 데이터 없음")
        L.append("")
        L.append("route_*.json에 `diagnostics`가 없습니다(로깅 추가 전 런). **스택을 재기동해 A→B를 한 번 재주행**한 뒤 다시 실행하세요.")
        return "\n".join(L)

    # ---- 0. 정찰 검증 (사용자 확인용: weave / 멈춰-확정 / 인식 / 충돌) ----
    recon = recon or {}
    if any(recon.values()):
        def rcg(rid, *path, default="—"):
            cur = recon.get(rid)
            for k in path:
                cur = cur.get(k) if isinstance(cur, dict) else None
            return default if cur is None else cur

        def reco_str(rid):
            r = recon.get(rid) or {}
            reco = r.get("recognition") or {}
            cb = reco.get("confirmed_by_class") or {}
            raw = sum((reco.get("yolo_raw") or {}).values())
            body = ", ".join(f"{k}:{v}" for k, v in sorted(cb.items())) if cb else "0"
            return f"{body} (raw {raw})"

        L.append("## 0. 정찰 검증 (weave / 멈춰-확정 / 인식 / 충돌)")
        L.append("")
        L.append("| 항목 | route_A | route_B | 읽는 법 |")
        L.append("|---|---|---|---|")
        L.append(f"| weave 진폭 평균/최대(m) | {rcg('A','weave','amp_mean_m')}/{rcg('A','weave','amp_max_m')} | "
                 f"{rcg('B','weave','amp_mean_m')}/{rcg('B','weave','amp_max_m')} | 좌우 흔들림 크기(off면 작아야) |")
        L.append(f"| weave 좌우전환 횟수 | {rcg('A','weave','dir_changes')} | {rcg('B','weave','dir_changes')} | 많으면 위빙 多(off면 0~소수) |")
        L.append(f"| 멈춰-확정 정지횟수/시간(s) | {rcg('A','stop_confirm','hold_episodes')}/{rcg('A','stop_confirm','hold_time_s')} | "
                 f"{rcg('B','stop_confirm','hold_episodes')}/{rcg('B','stop_confirm','hold_time_s')} | 후보서 **의도적 정지**(hold) |")
        L.append(f"| 전체 STOP cmd(yaw 포함) | {rcg('A','stop_confirm','stop_cmds')} | {rcg('B','stop_confirm','stop_cmds')} | hold보다 훨씬 크면 떨림성 정지 |")
        # 정찰 관측 거동(②감속/dwell · ③포탑 step-stare) — obs 미기록(구버전)이면 '—'.
        L.append(f"| ②미분류 후보(평균 전방/옆) | {rcg('A','observe','cand_fov_mean')}/{rcg('A','observe','cand_side_mean')} | "
                 f"{rcg('B','observe','cand_fov_mean')}/{rcg('B','observe','cand_side_mean')} | 사거리내 미분류 후보(라이다 prior) |")
        L.append(f"| ③포탑 stare 횟수 | {rcg('A','observe','turret_stare_episodes')} | {rcg('B','observe','turret_stare_episodes')} | 옆 후보 폐루프 응시(0이면 미작동) |")
        L.append(f"| 확정 인식(센서퓨전) | {reco_str('A')} | {reco_str('B')} | class별 확정 수(raw=YOLO 프레임) |")

        def fr_str(rid):
            return fmt_fusion_rejects((recon.get(rid) or {}).get("fusion_rejects") or {})
        L.append(f"| 융합 결과(프레임 사유) | {fr_str('A')} | {fr_str('B')} | 성공%↓·strict_no_cluster多=YOLO↔LiDAR 매칭 실패, stale多=비동기 stale |")

        def gd_str(rid):
            gd = (recon.get(rid) or {}).get("gt_detect")
            if not gd:
                return "—"
            s = f"{gd['detected']}/{gd['on_route']} (rate {gd['rate']})"
            return s + (f" · 놓침: {', '.join(gd['missed'])}" if gd['missed'] else "")
        _gn = next((((recon.get(r) or {}).get("gt_detect") or {}).get("near_m") for r in ("A", "B")
                    if (recon.get(r) or {}).get("gt_detect")), 30)
        L.append(f"| **GT 탐지율**(경로 옆 실객체) | {gd_str('A')} | {gd_str('B')} | final_v3.map 대비 경로 ≤{_gn:.0f}m 객체 중 확정 비율(놓친 객체=버그 후보) |")
        def spd_str(rid):
            sp = (recon.get(rid) or {}).get("speed")
            if not sp:
                return "—"
            nc = f", 충돌진입 {sp['near_collision_mean']}" if sp.get("near_collision_mean") is not None else ""
            return f"mean {sp['mean']} / p90 {sp['p90']} / max {sp['max']}{nc}"
        L.append(f"| 속도(m/s) | {spd_str('A')} | {spd_str('B')} | mean·p90·max + 충돌지점 진입속도(과속 진입=ram 원인) |")
        L.append(f"| 충돌 횟수 | {rcg('A','collisions_count')} | {rcg('B','collisions_count')} | 낮을수록 좋음 |")
        L.append("")
        for rid in ("A", "B"):
            r = recon.get(rid)
            if not r:
                continue
            w, sc = r.get("weave", {}), r.get("stop_confirm", {})
            msgs = []
            dc = w.get("dir_changes") or 0
            msgs.append(f"weave {'활발' if dc >= 8 else '약함/off'}(좌우전환 {dc}회, 진폭 {w.get('amp_mean_m')}m)")
            if not sc.get("hold_logged"):
                msgs.append("⚠ hold 미기록(구버전 로그 — stop-to-confirm 검증하려면 재주행)")
            elif (sc.get("hold_episodes") or 0) == 0:
                msgs.append("**후보서 멈춤 0회** — stop-to-confirm 미작동(후보 미관측/즉시확정/조건 확인 필요)")
            else:
                msgs.append(f"멈춰-확정 {sc.get('hold_episodes')}회/{sc.get('hold_time_s')}s")
            ob = r.get("observe")
            if ob is None:
                msgs.append("⚠ obs 미기록(구버전 — ②③ 검증하려면 재주행)")
            else:
                msgs.append(f"②감속/dwell {sc.get('hold_episodes', 0)}회 · ③포탑 stare {ob.get('turret_stare_episodes', 0)}회"
                            f"(미분류후보 평균 전방{ob.get('cand_fov_mean')}/옆{ob.get('cand_side_mean')})")
            L.append(f"- **route_{rid}**: " + " · ".join(msgs))
        L.append("")

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
    L.append(f"| └ 직선/코너 추종오차(m) | {cell(rA,'follow_decomp','straight_m')}/{cell(rA,'follow_decomp','corner_m')} | "
             f"{cell(rB,'follow_decomp','straight_m')}/{cell(rB,'follow_decomp','corner_m')} | 코너↑=lookahead, 직선↑=코리더/제어 |")
    L.append(f"| └ 제어추종각(°)/반대이동 | {cell(rA,'control_track','mean')}/{cell(rA,'control_track','over90')} | "
             f"{cell(rB,'control_track','mean')}/{cell(rB,'control_track','over90')} | 크면 제어가 목표 못 쫓음 |")
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
    ap.add_argument("--gt-map", default=GT_OBJECTS_MAP, help="GT 정답맵(객체 포함) — 경로별 탐지율 산정")
    ap.add_argument("--gt-near-m", type=float, default=30.0, help="경로에서 이 거리 이내 GT객체를 '발견대상'으로(융합 reach~45m)")
    args = ap.parse_args(argv)

    gt = gt_threat_counts(args.map)
    # 파생 분석물(리포트/오버레이)은 analysis/로 분리. 입력(route_*.json 등)은 root에서 읽는다.
    out_dir = os.path.join(args.input, "analysis")
    os.makedirs(out_dir, exist_ok=True)
    results, figs, recon = {}, {}, {}
    for rid in ("A", "B"):
        # route_{rid}.json → 없으면 comparison.json 폴백(시나리오2 run이 route_A.json 지워도 분석 가능).
        d, src = load_route(args.input, rid)
        if d is None:
            results[rid] = None
            continue
        res = analyze_route(d)
        rc = analyze_recon(d)
        if rc is not None:
            rc["recognition"] = recon_recognition(args.input, rid, (d.get("vision_yolo") or {}).get("counts"))
            rc["gt_detect"] = gt_detection_rate(d, args.input, rid, args.gt_map, near_m=args.gt_near_m)
            rc["_src"] = src
        recon[rid] = rc
        if res is not None:
            res["_spotted"] = d.get("asset_spotted_gt", {})
            figs[rid] = render_overlay(rid, d, res, out_dir)
        results[rid] = res
        if rc:
            w, sc = rc["weave"], rc["stop_confirm"]
            print(f"route_{rid}({os.path.basename(src) if src else '?'}): "
                  f"weave 좌우전환 {w['dir_changes']}회/진폭 {w['amp_mean_m']}m, "
                  f"멈춰-확정 {sc['hold_episodes']}회/{sc['hold_time_s']}s(hold_logged={sc['hold_logged']}), "
                  f"충돌 {rc['collisions_count']}, 확정 {rc['recognition']['confirmed_total']}")
            sp = rc.get("speed")
            if sp:
                nc = f", 충돌진입 {sp['near_collision_mean']}" if sp.get("near_collision_mean") is not None else ""
                print(f"  └ 속도(m/s): mean {sp['mean']} p90 {sp['p90']} max {sp['max']}{nc}")
            print(f"  └ 융합: {fmt_fusion_rejects(rc.get('fusion_rejects') or {})}")
            ob = rc.get("observe")
            if ob is not None:
                print(f"  └ 관측거동(②③): 미분류후보 평균 {ob['cand_mean']}(전방{ob['cand_fov_mean']}/옆{ob['cand_side_mean']}), "
                      f"②dwell {rc['stop_confirm']['hold_episodes']}회, ③포탑 stare {ob['turret_stare_episodes']}회, "
                      f"slow {ob['slow_samples']}샘플, 크기 {ob['by_class_max']}")
            gd = rc.get("gt_detect")
            if gd:
                miss = (" 놓침=" + ", ".join(gd["missed"])) if gd["missed"] else ""
                print(f"  └ GT탐지율(경로 ≤{gd['near_m']:.0f}m): {gd['detected']}/{gd['on_route']}"
                      f" (rate={gd['rate']}){miss}")
        else:
            print(f"route_{rid}: 진단 데이터 없음(diagnostics 미기록) — 재주행 필요")

    md = render_md(results, gt, out_dir, figs, recon)
    out = os.path.join(out_dir, "run_diagnosis.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"\n[완료] 진단 리포트: {out}")
    if figs:
        print(f"  오버레이: {', '.join(os.path.basename(v) for v in figs.values())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
