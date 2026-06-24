#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_scenario2_map.py — 정찰 산출물을 시나리오2 planner 입력으로 합성.

정찰로 발견한 장애물/적전차 + 지형을 깨끗한 base 맵(finalmap) 위에 덮어
시나리오2가 로드할 합본 맵을 만든다. **finalmap 원본은 절대 수정하지 않는다.**

입력(프로젝트 상대 기본값 — route A+B 둘 다 합침; 없는 파일은 건너뜀):
  --base-map         finalmap.map (rviz_visualization share/src, read-only base)
  --discovered-maps  recon_map/discovered_objects_route_{A,B}.map (정찰 발견객체)
  --terrain-npzs     terrain_maps/terrain_map_route_{A,B}.npz      (정찰 지형 점군)

출력:
  --out              recon_reports/recon_map/scenario2_map.map      (obstacles + targets, A+B)
  companion(grid)    recon_reports/recon_map/scenario2_terrain.json  (셀별 roughness 격자, planner용)
  companion(npz)     recon_reports/recon_map/scenario2_terrain.npz   (합본 점군, RViz 메쉬 뷰용)

obstacles : finalmap 나무/바위/벽 + 발견 rock/car/house/tent/tank 전부(A* 회피, A+B 합본).
targets   : 발견 tank 위치(차후 교전/위험도용; 회피는 obstacles에서 이미 처리).
terrain   : A+B ground 점군 concat → 1m 격자 z_median + roughness(이웃 dz/m) → A* 비용 레이어 입력.

사용:
  python3 scripts/build_scenario2_map.py                       # route A+B 자동 합본
  python3 scripts/build_scenario2_map.py --discovered-maps <A.map> --terrain-npzs <A.npz>   # route A만
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

# 기존 스크립트 헬퍼 재사용 (중복 구현 금지)
from merge_recon_terrain_map import (  # noqa: E402
    load_json, write_json, category_from_obstacle, raw_xy, merge_obstacles,
)
from terrain_map_postprocess import load_terrain_npz  # noqa: E402


THREAT_CLASSES = {"tank"}            # 회피 obstacle이면서 교전 targets로도 분리 기록
DROP_CLASSES = {"person", "human"}  # 저신뢰·설계상 제외
DEFAULT_TERRAIN_CELL_SIZE = 1.0     # A* 격자(res=1.0)에 정렬 → ix/iy가 곧 격자 인덱스


def default_path(*parts: str) -> Path:
    return PROJECT_ROOT.joinpath(*parts)


def find_base_map(explicit: Optional[str]) -> Optional[Path]:
    if explicit:
        return Path(explicit).expanduser()
    candidates = [
        PROJECT_ROOT / "install" / "rviz_visualization" / "share" / "rviz_visualization" / "map" / "finalmap.map",
        PROJECT_ROOT / "src" / "rviz_visualization" / "map" / "finalmap.map",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def build_terrain_grid(ground: np.ndarray, cell_size: float) -> Dict[str, Any]:
    """ground 점군 → 셀별 z_median + roughness(이웃 |dz|/cell 최대) 격자.

    cell_size=1.0이면 ix/iy가 A* 1m 격자 인덱스와 그대로 정렬된다.
    """
    empty = {"type": "elevation_roughness_grid", "cell_size": float(cell_size), "cell_count": 0, "cells": []}
    if ground is None or ground.size == 0 or cell_size <= 0:
        return empty
    arr = np.asarray(ground, dtype=np.float32).reshape(-1, 3)
    arr = arr[np.isfinite(arr).all(axis=1)]
    if arr.size == 0:
        return empty
    keys = np.floor(arr[:, :2] / cell_size).astype(np.int64)
    buckets: Dict[Tuple[int, int], List[float]] = {}
    for key, z in zip(map(tuple, keys), arr[:, 2]):
        buckets.setdefault(key, []).append(float(z))
    height: Dict[Tuple[int, int], float] = {
        k: float(np.median(np.asarray(v, dtype=np.float32))) for k, v in buckets.items()
    }
    cells = []
    for (ix, iy), z in sorted(height.items()):
        # roughness = 4-이웃 고도차의 최대 기울기(dz/m). 평탄=0, 급경사=큰 값.
        rough = 0.0
        for nb in ((ix + 1, iy), (ix - 1, iy), (ix, iy + 1), (ix, iy - 1)):
            if nb in height:
                rough = max(rough, abs(height[nb] - z) / cell_size)
        cells.append({
            "ix": int(ix), "iy": int(iy),
            "x": float((ix + 0.5) * cell_size), "y": float((iy + 0.5) * cell_size),
            "z_median": float(z), "roughness": float(rough),
            "point_count": int(len(buckets[(ix, iy)])),
        })
    roughs = np.asarray([c["roughness"] for c in cells], dtype=np.float32) if cells else np.zeros(1, np.float32)
    return {
        "type": "elevation_roughness_grid",
        "frame_id": "tank_map",
        "cell_size": float(cell_size),
        "cell_count": len(cells),
        "roughness_mean": float(np.mean(roughs)),
        "roughness_p90": float(np.percentile(roughs, 90)),
        "roughness_max": float(np.max(roughs)),
        "cells": cells,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="정찰 발견객체+지형을 시나리오2 합본 맵으로 생성 (route A+B 통합)")
    p.add_argument("--base-map", default=None, help="finalmap.map (기본: rviz_visualization share/src)")
    p.add_argument("--discovered-maps", nargs="+", default=[
        str(default_path("recon_reports", "recon_map", "discovered_objects_route_A.map")),
        str(default_path("recon_reports", "recon_map", "discovered_objects_route_B.map")),
    ], help="발견맵 1개 이상 (기본 route A+B). 없는 파일은 건너뜀.")
    p.add_argument("--terrain-npzs", nargs="+", default=[
        str(default_path("recon_reports", "terrain_maps", "terrain_map_route_A.npz")),
        str(default_path("recon_reports", "terrain_maps", "terrain_map_route_B.npz")),
    ], help="지형 NPZ 1개 이상 (기본 route A+B). 없는 파일은 건너뜀.")
    p.add_argument("--out", default=str(default_path("recon_reports", "recon_map", "scenario2_map.map")))
    p.add_argument("--terrain-out", default=None, help="기본: --out 옆 scenario2_terrain.json (planner 격자)")
    p.add_argument("--terrain-npz-out", default=None, help="기본: --out 옆 scenario2_terrain.npz (뷰 메쉬용 합본 점군)")
    p.add_argument("--terrain-cell-size", type=float, default=DEFAULT_TERRAIN_CELL_SIZE)
    args = p.parse_args()

    base_map = find_base_map(args.base_map)
    out_path = Path(args.out).expanduser()
    terrain_out = Path(args.terrain_out).expanduser() if args.terrain_out else out_path.parent / "scenario2_terrain.json"
    terrain_npz_out = Path(args.terrain_npz_out).expanduser() if args.terrain_npz_out else out_path.parent / "scenario2_terrain.npz"

    # --- 발견맵 여러 개(route A+B) 로드·합본 ---
    discovered_obstacles: List[Dict[str, Any]] = []
    used_discovered: List[str] = []
    cand_total = conf_total = 0
    for dm in args.discovered_maps:
        dmp = Path(dm).expanduser()
        if not dmp.exists():
            print(f"[WARN] 발견맵 없음 — 건너뜀: {dmp}", file=sys.stderr)
            continue
        dd = load_json(dmp)
        discovered_obstacles.extend(dd.get("obstacles", []) or [])
        used_discovered.append(str(dmp))
        cand_total += int(dd.get("candidate_count") or 0)
        conf_total += int(dd.get("confirmed_count") or 0)
    if not used_discovered:
        print("[ERR] 발견맵을 하나도 못 찾음. 정찰을 먼저 돌리세요"
              "(run_recon_scenario.py가 종료 시 자동 저장).", file=sys.stderr)
        return 2

    if base_map and base_map.exists():
        base_data = load_json(base_map)
        base_obstacles = base_data.get("obstacles", []) or []
        terrain_index = base_data.get("terrainIndex", 5)
    else:
        base_obstacles = []
        terrain_index = 5
        print("[WARN] base finalmap 없음 — 발견객체만으로 생성", file=sys.stderr)

    # 발견객체 분류: person drop, 나머지 obstacles 후보(tank 포함), tank는 targets에도
    keep_discovered: List[Dict[str, Any]] = []
    targets: List[Dict[str, Any]] = []
    for obs in discovered_obstacles:
        cls = category_from_obstacle(obs)
        if cls in DROP_CLASSES:
            continue
        keep_discovered.append(obs)
        if cls in THREAT_CLASSES:
            tx, ty = raw_xy(obs)
            targets.append({
                "prefabName": obs.get("prefabName", ""),
                "class_name": cls,
                "position": dict(obs.get("position", {})),
                "map_x": tx, "map_y": ty,
                "metadata": dict(obs.get("metadata", {}) or {}),
            })

    # obstacles 합본(base + A+B 발견 전부; dedup) — tank도 obstacle로 포함(회피)
    merged_obstacles = merge_obstacles(base_obstacles, keep_discovered)

    # --- 지형 NPZ 여러 개(route A+B) concat → planner 격자 + 뷰 메쉬용 합본 NPZ ---
    grounds, nongrounds, accums = [], [], []
    used_terrain: List[str] = []
    for tn in args.terrain_npzs:
        tnp = Path(tn).expanduser()
        if not tnp.exists():
            print(f"[WARN] 지형 NPZ 없음 — 건너뜀: {tnp}", file=sys.stderr)
            continue
        try:
            acc, ground, nonground, _ = load_terrain_npz(tnp)
            grounds.append(ground)
            nongrounds.append(nonground)
            accums.append(acc)
            used_terrain.append(str(tnp))
        except Exception as exc:
            print(f"[WARN] 지형 NPZ 처리 실패 — 건너뜀: {tnp} ({exc})", file=sys.stderr)

    terrain_grid = {"type": "elevation_roughness_grid", "cell_count": 0, "cells": []}
    if used_terrain:
        comb_ground = np.vstack(grounds).astype(np.float32) if grounds else np.empty((0, 3), np.float32)
        comb_nonground = np.vstack(nongrounds).astype(np.float32) if nongrounds else np.empty((0, 3), np.float32)
        comb_acc = np.vstack(accums).astype(np.float32) if accums else comb_ground
        terrain_grid = build_terrain_grid(comb_ground, args.terrain_cell_size)  # planner 격자(합친 ground)
        # 뷰 메쉬용 합본 NPZ — terrain_record_finalize_node가 로드하는 키 형식(ground/non_ground/accumulated)
        meta = json.dumps({"source": "build_scenario2_map combined", "routes": used_terrain})
        terrain_npz_out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(terrain_npz_out),
                            accumulated=comb_acc, ground=comb_ground, non_ground=comb_nonground,
                            metadata_json=np.asarray(meta))
    else:
        print("[WARN] 지형 NPZ 하나도 없음(지형 비용/면 비활성)", file=sys.stderr)

    write_json(terrain_out, terrain_grid)

    payload = {
        "terrainIndex": terrain_index,
        "map_role": "scenario2_derived_from_recon",
        "frame_id": "tank_map",
        "coordinate_policy": "obstacles/targets position uses Unity raw convention: x=map.x, y=map.z, z=map.y",
        "created_wall": time.time(),
        "source_files": {
            "base_map": str(base_map) if base_map else None,
            "discovered_maps": used_discovered,
            "terrain_npzs": used_terrain,
        },
        "object_count": len(merged_obstacles),
        "base_object_count": len(base_obstacles),
        "discovered_kept_count": len(keep_discovered),
        "target_count": len(targets),
        "discovered_candidate_count": cand_total,
        "discovered_confirmed_count": conf_total,
        "terrain_companion": terrain_out.name,
        "terrain_npz_companion": terrain_npz_out.name if used_terrain else None,
        "obstacles": merged_obstacles,
        "targets": targets,
    }
    write_json(out_path, payload)

    print("[OK] scenario2 map 생성:", out_path)
    print("     발견맵 입력      :", len(used_discovered), "개", [Path(d).name for d in used_discovered])
    print("     base 객체        :", len(base_obstacles))
    print("     발견 유지(회피)  :", len(keep_discovered), f"(candidate={cand_total}, confirmed={conf_total})")
    print("     합본 obstacles   :", len(merged_obstacles))
    print("     targets(tank)    :", len(targets))
    print("     terrain cells    :", terrain_grid.get("cell_count", 0), "→", terrain_out)
    if used_terrain:
        print("     terrain npz(뷰) :", terrain_npz_out, f"({[Path(t).name for t in used_terrain]})")
    if not keep_discovered:
        print("  [주의] 발견객체 0 — 정찰이 confirmed 객체를 못 만든 것일 수 있음(save_confirmed_only).")
    print("\n시나리오2 실행:  ros2 launch control tank_scenario2.launch.py")
    print("정찰 결과 뷰:    ros2 launch rviz_visualization tank_scenario2_map_view.launch.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
