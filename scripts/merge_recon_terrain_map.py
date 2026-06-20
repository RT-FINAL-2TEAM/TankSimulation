#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge discovered obstacle .map and terrain surface data into one RViz-ready .map.

Default inputs:
  ~/tankcc/tank_discovered_maps/discovered_objects_latest.map
  ~/tankcc/tank_terrain_maps/terrain_surface_latest.map
  fallback: latest *_ground.npy in ~/tankcc/tank_terrain_maps

Default output:
  ~/tankcc/recon_reports/final_recon_map.map

Run:
  cd ~/tankcc
  python3 scripts/merge_recon_terrain_map.py

Then RViz launch automatically uses ~/tankcc/recon_reports/final_recon_map.map if it exists.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


CLASS_MERGE_RADIUS_M = {
    "person": 1.5,
    "rock": 3.0,
    "car": 4.0,
    "tank": 3.0,
    "house": 4.0,
    "tent": 3.0,
    "wall": 3.0,
    "unknown": 3.0,
}


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"JSON root must be object: {path}")
    return data


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def latest_file(directory: Path, patterns: Iterable[str]) -> Optional[Path]:
    candidates: List[Path] = []
    for pattern in patterns:
        candidates.extend(directory.glob(pattern))
    candidates = [p for p in candidates if p.is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def category_from_obstacle(obs: Dict[str, Any]) -> str:
    meta = obs.get("metadata", {}) or {}
    for key in ("canonical_class", "public_class", "class_name", "label", "category"):
        val = meta.get(key)
        if val:
            return str(val).strip().lower()
    prefab = str(obs.get("prefabName", "")).strip().lower()
    if prefab.startswith("detected_"):
        parts = prefab.split("_")
        if len(parts) >= 2:
            return parts[1]
    for cat, prefixes in {
        "tree": ("tree",),
        "rock": ("rock",),
        "person": ("human", "person"),
        "car": ("car",),
        "tank": ("tank",),
        "house": ("house",),
        "tent": ("tent",),
        "wall": ("wall",),
    }.items():
        if prefab.startswith(prefixes):
            return cat
    return "unknown"


def raw_xy(obs: Dict[str, Any]) -> Tuple[float, float]:
    # .map obstacle positions are saved in Unity raw convention: x=map.x, y=map.z, z=map.y.
    pos = obs.get("position", {}) or {}
    return float(pos.get("x", 0.0)), float(pos.get("z", 0.0))


def merge_obstacles(base: List[Dict[str, Any]], discovered: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = [dict(o) for o in base]
    runtime_idx = 0
    for obs in discovered:
        cat = category_from_obstacle(obs)
        ox, oy = raw_xy(obs)
        radius = float(CLASS_MERGE_RADIUS_M.get(cat, CLASS_MERGE_RADIUS_M["unknown"]))
        duplicate = False
        for prev in merged:
            pc = category_from_obstacle(prev)
            if pc != cat:
                continue
            px, py = raw_xy(prev)
            if math.hypot(ox - px, oy - py) <= radius:
                duplicate = True
                break
        if duplicate:
            continue
        item = dict(obs)
        meta = dict(item.get("metadata", {}) or {})
        meta["merged_by"] = "merge_recon_terrain_map.py"
        meta["source_map_role"] = "discovered_runtime_objects"
        item["metadata"] = meta
        if not str(item.get("prefabName", "")):
            item["prefabName"] = f"runtime_{cat}_{runtime_idx:04d}"
        merged.append(item)
        runtime_idx += 1
    return merged


def terrain_cells_from_ground_npy(path: Path, grid_cell_size: float, max_cells: int) -> Dict[str, Any]:
    points = np.load(path)
    arr = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    arr = arr[np.isfinite(arr).all(axis=1)]
    if arr.size == 0:
        cells: List[Dict[str, float]] = []
        z_min = z_max = 0.0
    else:
        keys = np.floor(arr[:, :2] / grid_cell_size).astype(np.int64)
        buckets: Dict[Tuple[int, int], List[float]] = {}
        for key, z in zip(map(tuple, keys), arr[:, 2]):
            buckets.setdefault(key, []).append(float(z))
        items = list(buckets.items())
        if len(items) > max_cells:
            step = int(math.ceil(len(items) / max_cells))
            items = items[::step]
        cells = []
        for (ix, iy), vals in items:
            cells.append({
                "x": float((ix + 0.5) * grid_cell_size),
                "y": float((iy + 0.5) * grid_cell_size),
                "z": float(np.median(np.asarray(vals, dtype=np.float32))),
            })
        zs = np.asarray([c["z"] for c in cells], dtype=np.float32) if cells else np.asarray([0.0], dtype=np.float32)
        z_min = float(np.min(zs))
        z_max = float(np.max(zs))
    return {
        "type": "elevation_grid",
        "frame_id": "tank_map",
        "grid_cell_size": float(grid_cell_size),
        "cell_count": len(cells),
        "z_min": z_min,
        "z_max": z_max,
        "source_file": str(path),
        "cells": cells,
    }


def load_terrain_surface(path: Path, grid_cell_size: float, max_cells: int) -> Dict[str, Any]:
    if path.suffix.lower() == ".npy":
        return terrain_cells_from_ground_npy(path, grid_cell_size, max_cells)
    data = load_json(path)
    surface = data.get("terrain_surface")
    if not isinstance(surface, dict):
        raise ValueError(f"No terrain_surface key found in {path}")
    surface = dict(surface)
    surface.setdefault("source_file", str(path))
    return surface


def resolve_inputs(args: argparse.Namespace) -> Tuple[Path, Path, Optional[Path], Path]:
    workspace = Path(args.workspace).expanduser().resolve()
    discovered_dir = Path(args.discovered_dir).expanduser() if args.discovered_dir else workspace / "tankcc/tank_discovered_maps"
    terrain_dir = Path(args.terrain_dir).expanduser() if args.terrain_dir else workspace / "tankcc/tank_terrain_maps"

    if args.discovered_map:
        discovered_map = Path(args.discovered_map).expanduser()
    else:
        latest = discovered_dir / "discovered_objects_latest.map"
        discovered_map = latest if latest.exists() else latest_file(discovered_dir, ["discovered_objects_*.map", "*.map"])
        if discovered_map is None:
            raise FileNotFoundError(f"No discovered map found in {discovered_dir}")

    if args.terrain:
        terrain = Path(args.terrain).expanduser()
    else:
        latest_map = terrain_dir / "terrain_surface_latest.map"
        terrain = latest_map if latest_map.exists() else latest_file(
            terrain_dir,
            ["terrain_map_*_terrain.map", "terrain_surface_*.map", "*_ground.npy", "terrain_ground_latest.npy"],
        )
        if terrain is None:
            raise FileNotFoundError(f"No terrain map/npy found in {terrain_dir}")

    if args.base_map:
        base_map = Path(args.base_map).expanduser()
    else:
        candidate = workspace / "src" / "rviz_visualization" / "map" / "finalmap.map"
        base_map = candidate if candidate.exists() else None

    output = Path(args.output).expanduser() if args.output else workspace / "recon_reports" / "final_recon_map.map"
    return discovered_map, terrain, base_map, output


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge discovered objects and terrain surface into one RViz-ready .map")
    parser.add_argument("--workspace", default="~/tankcc")
    parser.add_argument("--discovered-dir", default=None)
    parser.add_argument("--terrain-dir", default=None)
    parser.add_argument("--discovered-map", default=None)
    parser.add_argument("--terrain", default=None, help="terrain_surface .map or *_ground.npy")
    parser.add_argument("--base-map", default=None, help="optional base RViz .map. Default: ~/tankcc/src/rviz_visualization/map/finalmap.map if present")
    parser.add_argument("--output", default=None)
    parser.add_argument("--terrain-grid-cell-size", type=float, default=0.8)
    parser.add_argument("--max-terrain-cells", type=int, default=30000)
    args = parser.parse_args()

    discovered_map, terrain_path, base_map, output = resolve_inputs(args)

    discovered_data = load_json(discovered_map)
    discovered_obstacles = discovered_data.get("obstacles", []) or []
    if not isinstance(discovered_obstacles, list):
        raise ValueError(f"discovered obstacles must be a list: {discovered_map}")

    if base_map and base_map.exists():
        base_data = load_json(base_map)
        base_obstacles = base_data.get("obstacles", []) or []
        terrain_index = base_data.get("terrainIndex", discovered_data.get("terrainIndex", 5))
    else:
        base_data = {}
        base_obstacles = []
        terrain_index = discovered_data.get("terrainIndex", 5)

    terrain_surface = load_terrain_surface(terrain_path, args.terrain_grid_cell_size, args.max_terrain_cells)
    merged_obstacles = merge_obstacles(base_obstacles, discovered_obstacles)

    payload = {
        "terrainIndex": terrain_index,
        "map_role": "final_recon_map_with_discovered_objects_and_terrain",
        "frame_id": "tank_map",
        "coordinate_policy": {
            "obstacles": "Unity raw convention: x=map.x, y=map.z, z=map.y",
            "terrain_surface": "RViz tank_map convention: x=map.x, y=map.y, z=map.z",
        },
        "created_wall": time.time(),
        "source_files": {
            "base_map": str(base_map) if base_map else None,
            "discovered_map": str(discovered_map),
            "terrain": str(terrain_path),
        },
        "object_count": len(merged_obstacles),
        "discovered_object_count": len(discovered_obstacles),
        "base_object_count": len(base_obstacles),
        "obstacles": merged_obstacles,
        "terrain_surface": terrain_surface,
    }

    write_json(output, payload)
    print("[OK] merged map saved:", output)
    print("     base objects      :", len(base_obstacles))
    print("     discovered objects:", len(discovered_obstacles))
    print("     merged objects    :", len(merged_obstacles))
    print("     terrain cells     :", terrain_surface.get("cell_count", len(terrain_surface.get("cells", []))))
    print("\nRViz launch will use this file automatically when it exists:")
    print("  ros2 launch rviz_visualization tank_rviz.launch.py")
    print("\nOr explicitly:")
    print(f"  TANK_RECON_MAP_FILE={output} ros2 launch rviz_visualization tank_rviz.launch.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
