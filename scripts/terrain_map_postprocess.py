#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
terrain_map_postprocess.py

단일 terrain map 파일(.npz)을 분석/후처리하는 오프라인 도구.

기본 입력:
    ~/tankcc/tank_terrain_maps/terrain_map_latest.npz

사용 예:
    python3 ~/tankcc/scripts/terrain_map_postprocess.py
    python3 ~/tankcc/scripts/terrain_map_postprocess.py --map ~/tankcc/tank_terrain_maps/terrain_map_latest.npz
    python3 ~/tankcc/scripts/terrain_map_postprocess.py --output-dir ~/tankcc/tank_terrain_maps/report
    python3 ~/tankcc/scripts/terrain_map_postprocess.py --export-csv --output-dir ~/tankcc/tank_terrain_maps/report

기본 동작은 화면에 요약만 출력한다. 파일을 추가로 만들고 싶을 때만 --output-dir,
--export-csv, --height-grid-json을 켠다.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np


def expand_path(path: str) -> Path:
    return Path(os.path.expanduser(path)).resolve()


def load_terrain_npz(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"terrain map 파일이 없습니다: {path}")

    with np.load(str(path), allow_pickle=False) as data:
        if "ground" not in data.files:
            raise ValueError(f"ground 배열이 없는 terrain map입니다: {path}")

        ground = np.asarray(data["ground"], dtype=np.float32).reshape(-1, 3)
        non_ground = (
            np.asarray(data["non_ground"], dtype=np.float32).reshape(-1, 3)
            if "non_ground" in data.files
            else np.empty((0, 3), dtype=np.float32)
        )
        accumulated = (
            np.asarray(data["accumulated"], dtype=np.float32).reshape(-1, 3)
            if "accumulated" in data.files
            else np.vstack([ground, non_ground]).astype(np.float32)
            if non_ground.shape[0] > 0
            else ground.copy()
        )

        metadata: Dict[str, Any] = {}
        if "metadata_json" in data.files:
            raw = data["metadata_json"]
            text = str(raw.item() if raw.shape == () else raw[0])
            try:
                metadata = json.loads(text)
            except Exception:
                metadata = {"metadata_json_parse_error": text[:500]}

    return accumulated, ground, non_ground, metadata


def range_stats(points: np.ndarray) -> Dict[str, Any]:
    if points.size == 0:
        return {"count": 0}
    return {
        "count": int(points.shape[0]),
        "x_min": float(np.min(points[:, 0])),
        "x_max": float(np.max(points[:, 0])),
        "y_min": float(np.min(points[:, 1])),
        "y_max": float(np.max(points[:, 1])),
        "z_min": float(np.min(points[:, 2])),
        "z_max": float(np.max(points[:, 2])),
        "z_mean": float(np.mean(points[:, 2])),
        "z_median": float(np.median(points[:, 2])),
        "z_std": float(np.std(points[:, 2])),
    }


def build_height_grid(ground: np.ndarray, cell_size: float) -> Dict[str, Any]:
    if ground.size == 0:
        return {"cell_size": cell_size, "cells": []}
    if cell_size <= 0.0:
        raise ValueError("cell_size는 0보다 커야 합니다.")

    keys = np.floor(ground[:, :2] / cell_size).astype(np.int64)
    buckets: Dict[Tuple[int, int], list[float]] = {}
    for key, z in zip(map(tuple, keys), ground[:, 2]):
        buckets.setdefault(key, []).append(float(z))

    cells = []
    for (ix, iy), values in sorted(buckets.items()):
        z = float(np.median(np.asarray(values, dtype=np.float32)))
        cells.append(
            {
                "ix": int(ix),
                "iy": int(iy),
                "x": float((ix + 0.5) * cell_size),
                "y": float((iy + 0.5) * cell_size),
                "z_median": z,
                "point_count": int(len(values)),
            }
        )
    return {"cell_size": float(cell_size), "cell_count": len(cells), "cells": cells}


def estimate_roughness(ground: np.ndarray, cell_size: float) -> Dict[str, Any]:
    grid = build_height_grid(ground, cell_size)
    cells = grid.get("cells", [])
    if not cells:
        return {"cell_size": cell_size, "cell_count": 0}

    height_by_key = {(c["ix"], c["iy"]): float(c["z_median"]) for c in cells}
    slopes = []
    for (ix, iy), z1 in height_by_key.items():
        for key2 in ((ix + 1, iy), (ix, iy + 1)):
            if key2 not in height_by_key:
                continue
            dz = abs(height_by_key[key2] - z1)
            slopes.append(dz / cell_size)

    if not slopes:
        return {"cell_size": cell_size, "cell_count": len(cells), "edge_count": 0}
    arr = np.asarray(slopes, dtype=np.float32)
    return {
        "cell_size": float(cell_size),
        "cell_count": len(cells),
        "edge_count": int(arr.shape[0]),
        "roughness_mean_dz_per_m": float(np.mean(arr)),
        "roughness_p90_dz_per_m": float(np.percentile(arr, 90)),
        "roughness_max_dz_per_m": float(np.max(arr)),
    }


def write_csv(path: Path, points: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(str(path), points, delimiter=",", header="x,y,z", comments="")


def main() -> None:
    parser = argparse.ArgumentParser(description="Tank terrain_map_latest.npz 분석/후처리 도구")
    parser.add_argument(
        "--map",
        default="~/tankcc/tank_terrain_maps/terrain_map_latest.npz",
        help="분석할 단일 terrain map .npz 파일",
    )
    parser.add_argument(
        "--cell-size",
        type=float,
        default=1.0,
        help="height grid/roughness 계산용 cell 크기[m]",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="summary.json 등을 저장할 폴더. 생략하면 파일을 만들지 않고 화면 출력만 수행",
    )
    parser.add_argument("--export-csv", action="store_true", help="ground/non_ground/accumulated CSV 내보내기")
    parser.add_argument("--height-grid-json", action="store_true", help="cell별 대표 고도 JSON 내보내기")
    args = parser.parse_args()

    map_path = expand_path(args.map)
    accumulated, ground, non_ground, metadata = load_terrain_npz(map_path)

    summary = {
        "map_file": str(map_path),
        "metadata": metadata,
        "accumulated": range_stats(accumulated),
        "ground": range_stats(ground),
        "non_ground": range_stats(non_ground),
        "roughness": estimate_roughness(ground, args.cell_size),
    }

    print("\n=== Terrain map summary ===")
    print(f"file          : {map_path}")
    print(f"frame         : {metadata.get('map_frame', 'unknown')}")
    print(f"created       : {metadata.get('created_local_time', 'unknown')}")
    print(f"accumulated   : {accumulated.shape[0]} points")
    print(f"ground        : {ground.shape[0]} points")
    print(f"non_ground    : {non_ground.shape[0]} points")
    if ground.size > 0:
        gs = summary["ground"]
        print(f"ground x/y    : x={gs['x_min']:.2f}~{gs['x_max']:.2f}, y={gs['y_min']:.2f}~{gs['y_max']:.2f}")
        print(f"ground height : z={gs['z_min']:.2f}~{gs['z_max']:.2f}, median={gs['z_median']:.2f}, std={gs['z_std']:.2f}")
    rough = summary["roughness"]
    if rough.get("edge_count", 0) > 0:
        print(
            "roughness    : "
            f"mean={rough['roughness_mean_dz_per_m']:.3f}, "
            f"p90={rough['roughness_p90_dz_per_m']:.3f}, "
            f"max={rough['roughness_max_dz_per_m']:.3f} dz/m"
        )

    if args.output_dir:
        out_dir = expand_path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        summary_path = out_dir / "terrain_map_summary.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"summary 저장  : {summary_path}")

        if args.height_grid_json:
            grid_path = out_dir / "terrain_height_grid.json"
            with grid_path.open("w", encoding="utf-8") as f:
                json.dump(build_height_grid(ground, args.cell_size), f, ensure_ascii=False, indent=2)
            print(f"height grid 저장: {grid_path}")

        if args.export_csv:
            write_csv(out_dir / "terrain_accumulated.csv", accumulated)
            write_csv(out_dir / "terrain_ground.csv", ground)
            write_csv(out_dir / "terrain_non_ground.csv", non_ground)
            print(f"CSV 저장      : {out_dir}")


if __name__ == "__main__":
    main()
