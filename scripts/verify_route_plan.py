#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""정찰 루트 정적 검증/도출 — 시뮬 없이 전역 경로 품질 확인.

team_path_planning(순수 파이썬)을 직접 호출해 finalmap.map 정적 장애물(나무 등) 위에서
A/B 루트 경로를 떠보고:
  - 나무 관통(경로가 나무 bbox 내부로 들어가는가)
  - 코리더 중앙 이탈(웨이포인트 중심선에서 lateral 거리)
  - 세그먼트 실패/점프(웨이포인트 skip)
  - 끝점 도달/목적지 주변 나무 회피
를 자동 체크한다. routes.yaml 손튜닝을 시뮬 없이 반복 수렴시키기 위한 도구.

모드:
  verify  : routes.yaml의 A/B를 검증(기본)
  derive  : free-band 중앙을 추적해 후보 웨이포인트 자동 도출 → routes.yaml에 붙여넣기용

사용:
  python3 scripts/verify_route_plan.py                 # A/B 검증
  python3 scripts/verify_route_plan.py --derive        # 후보 웨이포인트 도출
  python3 scripts/verify_route_plan.py --clearance 0.8 --inflate 5
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "path_planning"))

from path_planning.team_path_planning import (  # noqa: E402
    load_static_obstacles_from_map,
    plan_path_through_waypoints,
    create_grid,
    add_obstacles,
    _x_ref_at,
)

DEFAULT_MAP = os.path.join(PROJECT_ROOT, "src", "rviz_visualization", "map", "finalmap.map")
DEFAULT_ROUTES = os.path.join(PROJECT_ROOT, "src", "path_planning", "config", "routes.yaml")
STATIC_INFLATE = 1.0   # plan_path_through_waypoints가 정적에 적용하는 값(team_path_planning.py:362)
GOAL_TOLERANCE = 10.0
SAMPLE_STEP = 1.0      # 경로 보간 간격(m)
JUMP_FACTOR = 4.0      # 연속점 거리가 이 배수×샘플스텝 초과면 점프(skip) 의심


# --------------------------------------------------------------------------- #
# 기하 헬퍼
# --------------------------------------------------------------------------- #
def densify(path, step=SAMPLE_STEP):
    """경로 폴리라인을 step 간격으로 보간."""
    if len(path) < 2:
        return list(path)
    out = [path[0]]
    for (x0, z0), (x1, z1) in zip(path, path[1:]):
        d = math.hypot(x1 - x0, z1 - z0)
        n = max(1, int(d / step))
        for i in range(1, n + 1):
            t = i / n
            out.append((x0 + t * (x1 - x0), z0 + t * (z1 - z0)))
    return out


def point_in_bbox(x, z, o, margin=0.0):
    return (o["x_min"] - margin <= x <= o["x_max"] + margin and
            o["z_min"] - margin <= z <= o["z_max"] + margin)


def min_dist_to_obstacles(x, z, obstacles):
    best = float("inf")
    for o in obstacles:
        cx = max(o["x_min"], min(x, o["x_max"]))
        cz = max(o["z_min"], min(z, o["z_max"]))
        best = min(best, math.hypot(x - cx, z - cz))
    return best


def lateral_from_centerline(x, z, waypoints):
    xref = _x_ref_at(z, waypoints)
    return abs(x - xref) if xref is not None else 0.0


# --------------------------------------------------------------------------- #
# 검증
# --------------------------------------------------------------------------- #
def verify_route(route_id, waypoints, start, goal, corridor_w, side, static_obs, clearance, inflate):
    path = plan_path_through_waypoints(
        start, waypoints, dynamic_obstacles=[], static_obstacles=static_obs,
        inflate=inflate, clearance_weight=clearance, side=side,
    )
    res = {"route": route_id, "n_path": len(path), "path": path}
    if len(path) < 2:
        res["fail"] = "경로 생성 실패(빈 경로)"
        return res

    dense = densify(path)
    half = corridor_w / 2.0

    # 1) 나무 관통
    penetrations = sum(1 for (x, z) in dense if any(point_in_bbox(x, z, o) for o in static_obs))
    min_clear = min(min_dist_to_obstacles(x, z, static_obs) for (x, z) in dense)

    # 2) 코리더 중앙 이탈
    laterals = [lateral_from_centerline(x, z, waypoints) for (x, z) in dense]
    max_lat, mean_lat = max(laterals), sum(laterals) / len(laterals)
    out_of_corridor = sum(1 for L in laterals if L > half)

    # 3) 웨이포인트 커버리지(skip 감지): 각 웨이포인트가 경로에서 corridor_half 넘게 벗어나면 missed
    missed = []
    for wp in waypoints:
        if wp[1] < start[1] - 10.0:   # valid_waypoints 후진 필터와 동일
            continue
        dmin = min(math.hypot(x - wp[0], z - wp[1]) for (x, z) in dense)
        if dmin > half:
            missed.append([round(wp[0], 1), round(wp[1], 1), round(dmin, 1)])

    # 4) 끝점
    end = path[-1]
    end_gap = math.hypot(end[0] - goal[0], end[1] - goal[1])
    end_clear = min_dist_to_obstacles(end[0], end[1], static_obs)

    res.update({
        "penetrations": penetrations, "min_clearance_m": round(min_clear, 2),
        "max_lateral_m": round(max_lat, 2), "mean_lateral_m": round(mean_lat, 2),
        "out_of_corridor": out_of_corridor, "missed": missed,
        "end_gap_m": round(end_gap, 2), "end_clearance_m": round(end_clear, 2),
        "corridor_half": half,
    })
    res["ok"] = (penetrations == 0 and not missed and end_gap <= GOAL_TOLERANCE
                 and mean_lat <= half)
    return res


def print_report(res):
    r = res["route"]
    if "fail" in res:
        print(f"  [route_{r}] ❌ {res['fail']}")
        return
    flag = "✅" if res["ok"] else "⚠️"
    print(f"  [route_{r}] {flag}  경로점 {res['n_path']}")
    print(f"     나무관통: {res['penetrations']}  최소이격: {res['min_clearance_m']}m")
    print(f"     코리더이탈(>{res['corridor_half']}m): {res['out_of_corridor']}  "
          f"lateral 평균 {res['mean_lateral_m']}m / 최대 {res['max_lateral_m']}m")
    print(f"     웨이포인트 miss: {res['missed'] if res['missed'] else '없음'}")
    print(f"     끝점: 목적지와 {res['end_gap_m']}m, 나무이격 {res['end_clearance_m']}m")


# --------------------------------------------------------------------------- #
# 도출 (free-band 중앙 추적)
# --------------------------------------------------------------------------- #
def free_intervals(grid, z_idx, res, x_lo, x_hi):
    """주어진 z행에서 [x_lo,x_hi] 범위의 free x구간(미터) 리스트."""
    cols = len(grid[0])
    row = grid[z_idx]
    ivs, run_start = [], None
    xi_lo, xi_hi = int(x_lo / res), min(cols - 1, int(x_hi / res))
    for xi in range(xi_lo, xi_hi + 1):
        if row[xi] == 0:
            if run_start is None:
                run_start = xi
        else:
            if run_start is not None:
                ivs.append((run_start * res, (xi - 1) * res))
                run_start = None
    if run_start is not None:
        ivs.append((run_start * res, xi_hi * res))
    return ivs


def derive_waypoints(static_obs, start, goal, side, z_step=25, bow=35.0, res=1.0):
    """start→(중간 bow)→goal 가이드 곡선에 가장 가까운 free-band 중앙을 추적.

    guide_x(z) = (start→goal 직선) + side_sign·bow·sin(π·진행률)  → 중간에서 좌(A)/우(B)로
    bow만큼 부풀고 양 끝(start/goal)에서는 직선으로 수렴. free구간 중심이 guide에 가장 가까운
    것을 고르되 이전 x에서 급변하지 않게 연속성 페널티를 둔다.
    """
    grid = create_grid(300, 300, res)
    add_obstacles(grid, static_obs, res, inflate=STATIC_INFLATE)
    side_sign = -1.0 if side == "west" else 1.0
    sx, sz = start
    gx, gz = goal

    def guide_x(z):
        t = (z - sz) / (gz - sz)
        base = sx + t * (gx - sx)
        return base + side_sign * bow * math.sin(math.pi * max(0.0, min(1.0, t)))

    pts = []
    prev_x = sx
    z = int(sz) + z_step
    while z < gz - z_step / 2:
        gxz = guide_x(z)
        ivs = [iv for iv in free_intervals(grid, z, res, gxz - 45, gxz + 45)
               if iv[1] - iv[0] >= 5.0]
        if ivs:
            # 가이드 근접(주) + 이전 x 연속성(부)
            iv = min(ivs, key=lambda iv: abs((iv[0] + iv[1]) / 2 - gxz)
                     + 0.4 * abs((iv[0] + iv[1]) / 2 - prev_x))
            lo, hi = iv
            # 구간 내에서 guide에 가장 가까운 점(중심이 아니라 guide 쪽으로)으로 clamp
            cx = round(max(lo + 2.0, min(hi - 2.0, gxz)), 1)
            pts.append([cx, float(z)])
            prev_x = cx
        z += z_step
    pts.append([round(gx, 2), round(gz, 1)])
    return pts


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(description="정찰 루트 정적 검증/도출")
    ap.add_argument("--map", default=DEFAULT_MAP)
    ap.add_argument("--routes", default=DEFAULT_ROUTES)
    ap.add_argument("--clearance", type=float, default=0.4, help="route_clearance_weight")
    ap.add_argument("--inflate", type=float, default=5.0, help="dynamic inflate(정적은 함수내 1.0 고정)")
    ap.add_argument("--derive", action="store_true", help="free-band 중앙 추적 후보 웨이포인트 출력")
    ap.add_argument("--z-step", type=float, default=25.0)
    args = ap.parse_args(argv)

    static_obs = load_static_obstacles_from_map(args.map)
    with open(args.routes, encoding="utf-8") as f:
        rd = yaml.safe_load(f)["finalmap"]
    start = tuple(rd["start"])
    goal = tuple(rd["destination"])
    corridor_w = float(rd.get("corridor_width", 20.0))
    routes = rd["routes"]

    print(f"맵 정적장애물: {len(static_obs)}개 · start {start} → goal {goal} · corridor {corridor_w}m")
    print(f"clearance_weight={args.clearance}, dynamic inflate={args.inflate}, static inflate={STATIC_INFLATE}\n")

    if args.derive:
        for rid, side in (("A", "west"), ("B", "east")):
            pts = derive_waypoints(static_obs, start, goal, side, z_step=int(args.z_step))
            print(f"# {rid}루트({side}) 후보 웨이포인트:")
            print(f"    {rid}:")
            for p in pts:
                print(f"      - [{p[0]}, {p[1]}]")
            print()
        return 0

    for rid, side in (("A", "west"), ("B", "east")):
        wps = [tuple(p) for p in routes[rid]]
        res = verify_route(rid, wps, start, goal, corridor_w, side, static_obs,
                           args.clearance, args.inflate)
        print_report(res)
    return 0


if __name__ == "__main__":
    sys.exit(main())
