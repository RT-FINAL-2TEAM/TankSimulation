#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""정찰 루트 시각화 — 시뮬 없이 새 맵 위 A/B 전역경로를 PNG로 렌더.

team_path_planning(순수 파이썬)으로 finalmap.map 정적 장애물 위에 A(서)/B(동) 루트를
**런타임과 동일하게**(웨이포인트 + 목적지를 이어 A*; map_astar_planner_node.py:714) 떠서
recon_reports/에 PNG를 만든다:

  - route_overlay.png : 장애물 footprint + A/B A* 경로 + 시작/목적지/적전차리스폰 마커 (기본)
  - clearance_map.png : 장애물 이격(클리어런스) 히트맵 + A/B 경로           (--heatmaps)
  - corridor_map.png  : 채널중심+사이드바이어스 비용맵(A/B 2패널) + 경로      (--heatmaps)

기본은 route_overlay.png만 생성한다(보고서가 쓰는 그림). clearance/corridor 히트맵은 경로
재튜닝 진단용이라 --heatmaps로만 만든다. verify_route_plan과 같은 지표(나무관통/최소이격/
코리더이탈/끝점 gap)도 stdout에 출력한다. 경로(routes.yaml)는 건드리지 않는다.

사용:
  python3 scripts/visualize_routes.py              # route_overlay.png만
  python3 scripts/visualize_routes.py --heatmaps   # + clearance/corridor 히트맵
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import yaml
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 디스플레이 없는 환경(헤드리스)에서 파일로만 저장
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle

# 한글 라벨이 두부(□)로 깨지지 않게 Noto Sans CJK를 등록. 없으면 기본 폰트로 폴백.
for _cjk in (
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
):
    if os.path.exists(_cjk):
        try:
            font_manager.fontManager.addfont(_cjk)
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=_cjk).get_name()
            break
        except Exception:
            pass
plt.rcParams["axes.unicode_minus"] = False  # 마이너스 기호 깨짐 방지

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "path_planning"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))

from path_planning.team_path_planning import (  # noqa: E402
    load_static_obstacles_from_map,
    plan_path_through_waypoints,
    create_grid,
    add_obstacles,
    compute_clearance,
    _build_cost_map,
)
# verify_route_plan의 기하 헬퍼/상수를 그대로 재사용(중복 구현 방지)
from verify_route_plan import (  # noqa: E402
    densify,
    point_in_bbox,
    min_dist_to_obstacles,
    lateral_from_centerline,
    STATIC_INFLATE,
)

DEFAULT_MAP = os.path.join(PROJECT_ROOT, "src", "rviz_visualization", "map", "finalmap.map")
DEFAULT_ROUTES = os.path.join(PROJECT_ROOT, "src", "path_planning", "config", "routes.yaml")
OUT_DIR = os.path.join(PROJECT_ROOT, "recon_reports")

# 적전차 리스폰 위치(map 좌표) — ros_bridge RED_START와 동일. 목적지와 별개(정찰 관측 개념).
ENEMY_RESPAWN = (135.46, 276.87)

GRID_M = 300  # 맵/격자 한 변(m)

# 프리팹 클래스별 색/표기 (그리기용)
CLASS_STYLE = {
    "Tree": dict(color="#6f9f6f", label="Tree"),
    "Rock": dict(color="#a9742f", label="Rock"),
    "Wall": dict(color="#7a7a7a", label="Wall"),
    "Car": dict(color="#ff8c1a", label="Car (planner=2.5m 장애물)"),
    "House": dict(color="#d62728", label="House (위협, A* 제외)"),
    "Tent": dict(color="#c2a36b", label="Tent"),
    "Human": dict(color="#b0b0b0", label="Human (동적)"),
}


def prefab_class(name: str) -> str:
    for key in ("Tree", "Rock", "Wall", "Car", "House", "Tent", "Human", "Tank"):
        if name.startswith(key):
            return key
    return "Other"


def load_raw_objects(map_path):
    """그리기용: 원본 맵에서 (클래스, x, z) 센터 목록을 클래스별로 모은다."""
    try:
        with open(map_path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}
    by_class = {}
    for obs in data.get("obstacles", []):
        cls = prefab_class(obs.get("prefabName", ""))
        pos = obs.get("position", {})
        by_class.setdefault(cls, []).append((pos.get("x", 0.0), pos.get("z", 0.0)))
    return by_class


def plan_route(start, waypoints, goal, side, planning_obs, inflate, clearance):
    """런타임(map_astar_planner_node)과 동일하게 through=웨이포인트+목적지로 A* 경로 생성."""
    through = list(waypoints) + [goal]
    return plan_path_through_waypoints(
        start, through, dynamic_obstacles=[], static_obstacles=planning_obs,
        inflate=inflate, clearance_weight=clearance, side=side,
    )


def route_metrics(path, waypoints, goal, planning_obs, corridor_w):
    """verify_route_plan과 동일 지표 + 적전차 리스폰까지의 끝점 거리."""
    if len(path) < 2:
        return {"fail": "경로 생성 실패(빈 경로)"}
    dense = densify(path)
    half = corridor_w / 2.0
    penetrations = sum(1 for (x, z) in dense if any(point_in_bbox(x, z, o) for o in planning_obs))
    min_clear = min(min_dist_to_obstacles(x, z, planning_obs) for (x, z) in dense)
    laterals = [lateral_from_centerline(x, z, waypoints) for (x, z) in dense]
    out_corr = sum(1 for L in laterals if L > half)
    end = path[-1]
    return {
        "n_path": len(path),
        "penetrations": penetrations,
        "min_clearance_m": round(min_clear, 2),
        "mean_lateral_m": round(sum(laterals) / len(laterals), 2),
        "max_lateral_m": round(max(laterals), 2),
        "out_of_corridor": out_corr,
        "end_gap_goal_m": round(math.hypot(end[0] - goal[0], end[1] - goal[1]), 2),
        "end_gap_enemy_m": round(math.hypot(end[0] - ENEMY_RESPAWN[0], end[1] - ENEMY_RESPAWN[1]), 2),
    }


def print_metrics(rid, m):
    if "fail" in m:
        print(f"  [route_{rid}] ❌ {m['fail']}")
        return
    print(f"  [route_{rid}] 경로점 {m['n_path']}")
    print(f"     나무관통 {m['penetrations']} · 최소이격 {m['min_clearance_m']}m · "
          f"코리더이탈 {m['out_of_corridor']} (lateral 평균 {m['mean_lateral_m']}/최대 {m['max_lateral_m']}m)")
    print(f"     끝점→목적지(110.0,276.5) {m['end_gap_goal_m']}m · "
          f"끝점→적리스폰(135.46,276.87) {m['end_gap_enemy_m']}m")


# --------------------------------------------------------------------------- #
# 그리기 헬퍼
# --------------------------------------------------------------------------- #
def _draw_obstacles(ax, planning_obs, raw_objs):
    """A*가 보는 정적 footprint(회색) + 클래스별 센터(색)로 장애물 표시."""
    # 1) planner가 실제로 보는 정적 footprint (bbox)
    for o in planning_obs:
        ax.add_patch(Rectangle(
            (o["x_min"], o["z_min"]), o["x_max"] - o["x_min"], o["z_max"] - o["z_min"],
            facecolor="#c7d2d9", edgecolor="none", alpha=0.55, zorder=1))
    # 2) 클래스별 센터(특히 Car/House는 footprint만으론 구분 안 되므로 별도 표기)
    for cls in ("Tree", "Rock", "Car", "House", "Tent", "Human"):
        pts = raw_objs.get(cls, [])
        if not pts:
            continue
        xs = [p[0] for p in pts]
        zs = [p[1] for p in pts]
        st = CLASS_STYLE[cls]
        if cls == "House":
            ax.scatter(xs, zs, s=120, marker="s", facecolors="none",
                       edgecolors=st["color"], linewidths=2.0, label=st["label"], zorder=4)
        elif cls == "Car":
            ax.scatter(xs, zs, s=42, marker="s", color=st["color"], label=st["label"], zorder=4)
        elif cls == "Human":
            ax.scatter(xs, zs, s=10, marker="x", color=st["color"], alpha=0.5,
                       label=st["label"], zorder=2)
        else:
            ax.scatter(xs, zs, s=10, marker="o", color=st["color"], alpha=0.7,
                       label=st["label"], zorder=2)


def _draw_markers(ax, start, goal):
    ax.scatter([start[0]], [start[1]], s=180, marker="o", color="#1a9850",
               edgecolors="black", linewidths=1.2, label=f"출발 {tuple(start)}", zorder=6)
    ax.scatter([goal[0]], [goal[1]], s=220, marker="^", color="#000000",
               edgecolors="white", linewidths=1.0, label=f"목적지 {tuple(goal)}", zorder=6)
    ax.scatter([ENEMY_RESPAWN[0]], [ENEMY_RESPAWN[1]], s=420, marker="*", color="#d62728",
               edgecolors="black", linewidths=1.2, label=f"적전차 리스폰 {ENEMY_RESPAWN}", zorder=7)


def _draw_path(ax, path, color, label):
    if len(path) < 2:
        return
    xs = [p[0] for p in path]
    zs = [p[1] for p in path]
    ax.plot(xs, zs, "-", color=color, linewidth=2.4, label=label, zorder=5)


def _finish_axes(ax, title):
    ax.set_xlim(0, GRID_M)
    ax.set_ylim(0, GRID_M)
    ax.set_aspect("equal")
    ax.set_xlabel("map x (동→) [m]")
    ax.set_ylabel("map z (북↑, 위쪽이 목적지) [m]")
    ax.set_title(title)
    ax.grid(True, alpha=0.2)


# --------------------------------------------------------------------------- #
# 3종 PNG
# --------------------------------------------------------------------------- #
def render_overlay(path_out, planning_obs, raw_objs, routes_paths, start, goal, wps):
    fig, ax = plt.subplots(figsize=(11, 11), dpi=130)
    _draw_obstacles(ax, planning_obs, raw_objs)
    for rid, (path, color) in routes_paths.items():
        _draw_path(ax, path, color, f"route {rid} A*")
        wp = wps[rid]
        if wp:
            ax.scatter([p[0] for p in wp], [p[1] for p in wp], s=70, marker="D",
                       facecolors="none", edgecolors=color, linewidths=1.6, zorder=5)
    _draw_markers(ax, start, goal)
    _finish_axes(ax, "정찰 A/B 전역경로 over finalmap (런타임: 웨이포인트+목적지 A*)")
    ax.legend(loc="upper left", fontsize=7, framealpha=0.9, ncol=2)
    fig.tight_layout()
    fig.savefig(path_out, bbox_inches="tight")
    plt.close(fig)


def render_clearance(path_out, grid, routes_paths, start, goal, max_r):
    clear = np.array(compute_clearance(grid, max_r=max_r), dtype=float)
    fig, ax = plt.subplots(figsize=(11, 11), dpi=130)
    im = ax.imshow(clear, origin="lower", extent=[0, GRID_M, 0, GRID_M],
                   cmap="viridis", alpha=0.95, zorder=0)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=f"장애물 이격(m, 상한 {max_r})")
    for rid, (path, color) in routes_paths.items():
        _draw_path(ax, path, color, f"route {rid}")
    ax.scatter([start[0], goal[0]], [start[1], goal[1]], c=["#1a9850", "#ffffff"],
               edgecolors="black", s=140, zorder=6)
    ax.scatter([ENEMY_RESPAWN[0]], [ENEMY_RESPAWN[1]], marker="*", s=380,
               color="#d62728", edgecolors="black", zorder=7, label="적전차 리스폰")
    _finish_axes(ax, "클리어런스 히트맵 (밝을수록 장애물에서 멀다 = 코리더 중앙)")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(path_out, bbox_inches="tight")
    plt.close(fig)


def render_corridor(path_out, grid, through_by_side, routes_paths, clearance, start, goal):
    """A(서)/B(동) 각 사이드 바이어스 비용맵을 2패널로 — 각 루트가 어느 '코리더'로 쏠리는지."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 10), dpi=120)
    panels = [("A", "west", "#1f77b4"), ("B", "east", "#2ca02c")]
    for ax, (rid, side, color) in zip(axes, panels):
        cost = np.array(_build_cost_map(grid, 1.0, clearance, through_by_side[rid], side), dtype=float)
        im = ax.imshow(cost, origin="lower", extent=[0, GRID_M, 0, GRID_M],
                       cmap="inferno", zorder=0)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="A* 추가비용(채널중심+사이드)")
        path, _c = routes_paths[rid]
        _draw_path(ax, path, color, f"route {rid} ({side})")
        ax.scatter([start[0], goal[0]], [start[1], goal[1]], c=["#1a9850", "#ffffff"],
                   edgecolors="black", s=130, zorder=6)
        ax.scatter([ENEMY_RESPAWN[0]], [ENEMY_RESPAWN[1]], marker="*", s=340,
                   color="#d62728", edgecolors="black", zorder=7)
        _finish_axes(ax, f"corridor cost — route {rid} ({side} bias)")
        ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(path_out, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description="정찰 A/B 경로 시각화(PNG)")
    ap.add_argument("--map", default=DEFAULT_MAP)
    ap.add_argument("--routes", default=DEFAULT_ROUTES)
    ap.add_argument("--inflate", type=float, default=5.0, help="dynamic inflate(planner와 동일)")
    ap.add_argument("--clearance", type=float, default=0.4, help="route_clearance_weight")
    ap.add_argument("--max-clear", type=float, default=25.0, help="클리어런스 히트맵 상한(m)")
    ap.add_argument("--heatmaps", action="store_true",
                    help="clearance_map/corridor_map 히트맵도 생성(기본 off — 경로 재튜닝 진단용)")
    args = ap.parse_args(argv)

    planning_obs = load_static_obstacles_from_map(args.map)
    raw_objs = load_raw_objects(args.map)
    with open(args.routes, encoding="utf-8") as f:
        rd = yaml.safe_load(f)["finalmap"]
    start = tuple(rd["start"])
    goal = tuple(rd["destination"])
    corridor_w = float(rd.get("corridor_width", 20.0))
    routes = rd["routes"]

    n_by = {k: len(v) for k, v in raw_objs.items()}
    print(f"맵: {args.map}")
    print(f"  정적 planner 장애물 {len(planning_obs)}개 · 원본 클래스 {n_by}")
    print(f"  start {start} → 목적지 {goal} · 적전차 리스폰 {ENEMY_RESPAWN} · corridor {corridor_w}m")
    print(f"  inflate={args.inflate}, clearance_weight={args.clearance}, static inflate={STATIC_INFLATE}\n")

    # 런타임과 동일 격자(정적 inflate=1.0)
    grid = create_grid(GRID_M, GRID_M, 1.0)
    add_obstacles(grid, planning_obs, 1.0, inflate=STATIC_INFLATE)

    routes_paths = {}
    wps = {}
    through_by_side = {}
    side_by = {"A": "west", "B": "east"}
    color_by = {"A": "#1f77b4", "B": "#2ca02c"}
    for rid in ("A", "B"):
        wp = [tuple(p) for p in routes[rid]]
        wps[rid] = wp
        through_by_side[rid] = wp + [goal]
        path = plan_route(start, wp, goal, side_by[rid], planning_obs, args.inflate, args.clearance)
        routes_paths[rid] = (path, color_by[rid])
        print_metrics(rid, route_metrics(path, wp, goal, planning_obs, corridor_w))

    os.makedirs(OUT_DIR, exist_ok=True)
    overlay = os.path.join(OUT_DIR, "route_overlay.png")
    render_overlay(overlay, planning_obs, raw_objs, routes_paths, start, goal, wps)
    print(f"\n저장: {overlay}")

    # clearance/corridor 히트맵은 경로 재튜닝 진단용이라 기본 off. --heatmaps로만 생성.
    if args.heatmaps:
        clearance_png = os.path.join(OUT_DIR, "clearance_map.png")
        corridor_png = os.path.join(OUT_DIR, "corridor_map.png")
        render_clearance(clearance_png, grid, routes_paths, start, goal, int(args.max_clear))
        render_corridor(corridor_png, grid, through_by_side, routes_paths, args.clearance, start, goal)
        print(f"      {clearance_png}")
        print(f"      {corridor_png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
