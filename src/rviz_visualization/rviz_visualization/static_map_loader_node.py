# -*- coding: utf-8 -*-
"""
Static map loader for Tank Challenge RViz / planning preparation.

This node treats map/*.map files as pre-known drone reconnaissance data and
publishes them as ROS2 static layers.

Design boundary:
- This node does NOT use live LiDAR, camera, YOLO, or tracking-mode data.
- Live sensor topics remain handled by ros_bridge / potential.
- Risk and cost weights are intentionally stored in YAML so they can later be
  replaced by reinforcement-learning derived values without changing code.

Coordinate convention:
- Unity raw: x = left/right, y = height, z = forward/backward
- RViz tank_map: x = raw.x, y = raw.z, z = raw.y
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Point
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray

try:
    import yaml
except Exception:  # pragma: no cover - ROS2 normally ships PyYAML.
    yaml = None


@dataclass(frozen=True)
class MapObject:
    prefab_name: str
    category: str
    raw_x: float
    raw_y: float
    raw_z: float
    map_x: float
    map_y: float
    map_z: float
    rotation: Dict[str, float]


class StaticMapLoaderNode(Node):
    """Publish recon/mission .map files as RViz markers and planning grids."""

    def __init__(self) -> None:
        super().__init__("tank_static_map_loader_node")

        pkg_share = Path(get_package_share_directory("rviz_visualization"))
        default_config = pkg_share / "config" / "static_map_costs.yaml"
        default_recon_map = pkg_share / "map" / "finalmap.map"
        default_mission_map = pkg_share / "map" / "mission_map.map"

        self.declare_parameter("mode", "recon_only")
        self.declare_parameter("config_file", str(default_config))
        self.declare_parameter("recon_map_file", str(default_recon_map))
        self.declare_parameter("mission_map_file", str(default_mission_map))
        self.declare_parameter("publish_period_sec", 1.0)
        self.declare_parameter("publish_mission", False)
        self.declare_parameter("publish_diff", False)
        self.declare_parameter("publish_grids", True)

        self.mode = str(self.get_parameter("mode").value)
        self.config_file = Path(str(self.get_parameter("config_file").value))
        self.recon_map_file = Path(str(self.get_parameter("recon_map_file").value))
        self.mission_map_file = Path(str(self.get_parameter("mission_map_file").value))
        self.publish_period_sec = float(self.get_parameter("publish_period_sec").value)
        self.publish_mission = bool(self.get_parameter("publish_mission").value)
        self.publish_diff = bool(self.get_parameter("publish_diff").value)
        self.publish_grids = bool(self.get_parameter("publish_grids").value)

        if self.mode in ("compare", "recon_mission", "mission_compare"):
            self.publish_mission = True
            self.publish_diff = True
        elif self.mode in ("recon_only", "recon"):
            self.publish_mission = False
            self.publish_diff = False

        self.config = self._load_config(self.config_file)
        self.frame_id = str(self.config["terrain"].get("frame_id", "tank_map"))
        self.resolution = float(self.config["terrain"].get("resolution", 1.0))
        self.min_x = float(self.config["terrain"].get("min_x", 0.0))
        self.max_x = float(self.config["terrain"].get("max_x", 300.0))
        self.min_y = float(self.config["terrain"].get("min_y", 0.0))
        self.max_y = float(self.config["terrain"].get("max_y", 300.0))
        self.width = int(round((self.max_x - self.min_x) / self.resolution))
        self.height = int(round((self.max_y - self.min_y) / self.resolution))

        static_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.recon_raw_pub = self.create_publisher(String, "/tank/map/recon/raw", static_qos)
        self.mission_raw_pub = self.create_publisher(String, "/tank/map/mission/raw", static_qos)
        self.summary_pub = self.create_publisher(String, "/tank/map/static/summary", static_qos)

        self.recon_marker_pub = self.create_publisher(
            MarkerArray, "/tank/rviz/recon_map_markers", static_qos
        )
        self.mission_marker_pub = self.create_publisher(
            MarkerArray, "/tank/rviz/mission_map_markers", static_qos
        )
        self.diff_marker_pub = self.create_publisher(
            MarkerArray, "/tank/rviz/map_diff_markers", static_qos
        )
        self.recon_occupancy_pub = self.create_publisher(
            OccupancyGrid, "/tank/map/recon/occupancy_grid", static_qos
        )
        self.recon_risk_pub = self.create_publisher(
            OccupancyGrid, "/tank/map/recon/risk_grid", static_qos
        )

        self.recon_data = self._read_json(self.recon_map_file)
        self.mission_data = self._read_json(self.mission_map_file) if self.mission_map_file.exists() else {}
        self.recon_objects = self._parse_objects(self.recon_data)
        self.mission_objects = self._parse_objects(self.mission_data) if self.mission_data else []

        self.recon_markers = self._make_object_markers(
            self.recon_objects, namespace_prefix="recon", color_mode="color_recon"
        )
        self.mission_markers = self._make_object_markers(
            self.mission_objects, namespace_prefix="mission", color_mode="color_mission"
        )
        self.diff_markers = self._make_diff_markers(self.recon_objects, self.mission_objects)

        self.recon_grid, self.recon_risk_grid = self._build_grids(self.recon_objects)
        self.recon_occupancy_msg = self._make_occupancy_msg(
            self.recon_grid, "/tank/map/recon/occupancy_grid"
        )
        self.recon_risk_msg = self._make_risk_msg(self.recon_risk_grid, "/tank/map/recon/risk_grid")
        self.recon_raw_msg = self._make_raw_msg("recon", self.recon_data, self.recon_objects)
        self.mission_raw_msg = self._make_raw_msg("mission", self.mission_data, self.mission_objects)
        self.summary_msg = self._make_summary_msg()

        self.get_logger().info(
            f"Loaded recon map: {len(self.recon_objects)} objects from {self.recon_map_file}"
        )
        if self.publish_mission:
            self.get_logger().info(
                f"Loaded mission map: {len(self.mission_objects)} objects from {self.mission_map_file}"
            )
        self.get_logger().info(
            f"Static map mode={self.mode}, publish_mission={self.publish_mission}, "
            f"publish_diff={self.publish_diff}, publish_grids={self.publish_grids}"
        )

        self.publish_all()
        self.timer = self.create_timer(self.publish_period_sec, self.publish_all)

    # ------------------------------------------------------------------
    # Loading / parsing
    # ------------------------------------------------------------------
    def _load_config(self, path: Path) -> Dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f"Static map config not found: {path}")
        if yaml is None:
            raise RuntimeError("PyYAML is required to load static_map_costs.yaml")
        with path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        if not isinstance(cfg, dict):
            raise ValueError(f"Invalid YAML config: {path}")
        return cfg

    def _read_json(self, path: Path) -> Dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f"Map file not found: {path}")
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _parse_objects(self, map_data: Dict[str, Any]) -> List[MapObject]:
        objects: List[MapObject] = []
        for obs in map_data.get("obstacles", []):
            prefab = str(obs.get("prefabName", "unknown"))
            pos = obs.get("position", {}) or {}
            rot = obs.get("rotation", {}) or {}
            raw_x = float(pos.get("x", 0.0))
            raw_y = float(pos.get("y", 0.0))
            raw_z = float(pos.get("z", 0.0))
            map_x, map_y, map_z = self._raw_to_map(raw_x, raw_y, raw_z)
            objects.append(
                MapObject(
                    prefab_name=prefab,
                    category=self._category_for_prefab(prefab),
                    raw_x=raw_x,
                    raw_y=raw_y,
                    raw_z=raw_z,
                    map_x=map_x,
                    map_y=map_y,
                    map_z=map_z,
                    rotation={
                        "x": float(rot.get("x", 0.0)),
                        "y": float(rot.get("y", 0.0)),
                        "z": float(rot.get("z", 0.0)),
                        "w": float(rot.get("w", 1.0)),
                    },
                )
            )
        return objects

    def _raw_to_map(self, raw_x: float, raw_y: float, raw_z: float) -> Tuple[float, float, float]:
        return raw_x, raw_z, raw_y

    def _category_for_prefab(self, prefab: str) -> str:
        for category, spec in self.config.get("categories", {}).items():
            for prefix in spec.get("prefixes", []) or []:
                if prefab.startswith(str(prefix)):
                    return str(category)
        return "unknown"

    def _category_spec(self, category: str) -> Dict[str, Any]:
        categories = self.config.get("categories", {})
        return categories.get(category, categories.get("unknown", {}))

    # ------------------------------------------------------------------
    # Marker generation
    # ------------------------------------------------------------------
    def _make_object_markers(
        self,
        objects: List[MapObject],
        namespace_prefix: str,
        color_mode: str,
    ) -> MarkerArray:
        msg = MarkerArray()
        for marker_id, obj in enumerate(objects):
            spec = self._category_spec(obj.category)
            marker = Marker()
            marker.header.frame_id = self.frame_id
            marker.ns = f"{namespace_prefix}_{obj.category}"
            marker.id = marker_id
            marker.type = self._marker_type(spec.get("marker_type", "CUBE"))
            marker.action = Marker.ADD
            marker.pose.position.x = float(obj.map_x)
            marker.pose.position.y = float(obj.map_y)
            marker.pose.position.z = self._marker_z(obj, spec)
            marker.pose.orientation.w = 1.0

            sx, sy, sz = self._scale_tuple(spec.get("scale", [2.0, 2.0, 2.0]))
            marker.scale.x = sx
            marker.scale.y = sy
            marker.scale.z = sz

            r, g, b, a = self._color_tuple(spec.get(color_mode, [1.0, 1.0, 1.0, 0.5]))
            marker.color.r = r
            marker.color.g = g
            marker.color.b = b
            marker.color.a = a

            # Keep metadata visible in RViz selection panel.
            marker.text = obj.prefab_name
            msg.markers.append(marker)
        return msg

    def _make_diff_markers(self, recon: List[MapObject], mission: List[MapObject]) -> MarkerArray:
        msg = MarkerArray()
        diff_cfg = self.config.get("diff", {})
        round_m = float(diff_cfg.get("position_round_m", 1.0))
        recon_keys = {self._object_signature(o, round_m) for o in recon}
        mission_keys = {self._object_signature(o, round_m) for o in mission}

        mission_only = [o for o in mission if self._object_signature(o, round_m) not in recon_keys]
        recon_only = [o for o in recon if self._object_signature(o, round_m) not in mission_keys]

        marker_id = 0
        for obj in mission_only:
            msg.markers.append(
                self._make_diff_marker(
                    obj,
                    "mission_only_unexpected_object",
                    marker_id,
                    diff_cfg.get("mission_only_color", [1.0, 0.0, 0.0, 1.0]),
                    diff_cfg,
                )
            )
            marker_id += 1

        for obj in recon_only:
            msg.markers.append(
                self._make_diff_marker(
                    obj,
                    "recon_only_not_in_mission",
                    marker_id,
                    diff_cfg.get("recon_only_color", [0.0, 0.7, 1.0, 0.85]),
                    diff_cfg,
                )
            )
            marker_id += 1
        return msg

    def _make_diff_marker(
        self,
        obj: MapObject,
        namespace: str,
        marker_id: int,
        color_values: Iterable[float],
        diff_cfg: Dict[str, Any],
    ) -> Marker:
        spec = self._category_spec(obj.category)
        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.ns = namespace
        marker.id = marker_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = float(obj.map_x)
        marker.pose.position.y = float(obj.map_y)
        marker.pose.position.z = self._marker_z(obj, spec) + 2.0
        marker.pose.orientation.w = 1.0
        scale_mult = float(diff_cfg.get("marker_scale_multiplier", 1.35))
        sx, sy, sz = self._scale_tuple(spec.get("scale", [2.0, 2.0, 2.0]))
        marker.scale.x = max(sx, 2.0) * scale_mult
        marker.scale.y = max(sy, 2.0) * scale_mult
        marker.scale.z = max(sz, 2.0) * scale_mult
        r, g, b, a = self._color_tuple(color_values)
        marker.color.r = r
        marker.color.g = g
        marker.color.b = b
        marker.color.a = a
        marker.text = obj.prefab_name
        return marker

    def _marker_z(self, obj: MapObject, spec: Dict[str, Any]) -> float:
        use_raw_height = bool(self.config.get("terrain", {}).get("use_raw_height", True))
        if use_raw_height:
            return float(obj.map_z)
        return float(self.config.get("terrain", {}).get("default_marker_z", 0.3))

    def _marker_type(self, name: Any) -> int:
        marker_name = str(name).strip().upper()
        return {
            "CUBE": Marker.CUBE,
            "SPHERE": Marker.SPHERE,
            "CYLINDER": Marker.CYLINDER,
            "ARROW": Marker.ARROW,
        }.get(marker_name, Marker.CUBE)

    def _scale_tuple(self, values: Iterable[Any]) -> Tuple[float, float, float]:
        vals = list(values)
        while len(vals) < 3:
            vals.append(vals[-1] if vals else 1.0)
        return float(vals[0]), float(vals[1]), float(vals[2])

    def _color_tuple(self, values: Iterable[Any]) -> Tuple[float, float, float, float]:
        vals = list(values)
        while len(vals) < 4:
            vals.append(1.0)
        return float(vals[0]), float(vals[1]), float(vals[2]), float(vals[3])

    def _object_signature(self, obj: MapObject, round_m: float) -> Tuple[str, int, int]:
        scale = 1.0 / max(round_m, 1e-6)
        return (
            obj.category,
            int(round(obj.raw_x * scale)),
            int(round(obj.raw_z * scale)),
        )

    # ------------------------------------------------------------------
    # Grid / risk generation
    # ------------------------------------------------------------------
    def _build_grids(self, objects: List[MapObject]) -> Tuple[List[int], List[float]]:
        free_value = int(self.config.get("occupancy", {}).get("free_value", 0))
        occupied_value = int(self.config.get("occupancy", {}).get("occupied_value", 100))
        grid = [free_value for _ in range(self.width * self.height)]
        risk = [0.0 for _ in range(self.width * self.height)]

        for obj in objects:
            spec = self._category_spec(obj.category)
            row, col = self._world_to_grid(obj.map_x, obj.map_y)
            if row is None or col is None:
                continue

            if bool(spec.get("impassable", False)):
                radius = float(
                    spec.get(
                        "inflation_radius_m",
                        self.config.get("occupancy", {}).get("default_inflation_radius_m", 4.0),
                    )
                )
                self._mark_inflated_obstacle(grid, row, col, radius, occupied_value)

            self._apply_risk(risk, obj, row, col, spec)

        risk = [max(0.0, min(1.0, v)) for v in risk]
        return grid, risk

    def _world_to_grid(self, map_x: float, map_y: float) -> Tuple[Optional[int], Optional[int]]:
        col = int(math.floor((map_x - self.min_x) / self.resolution))
        row = int(math.floor((map_y - self.min_y) / self.resolution))
        if row < 0 or row >= self.height or col < 0 or col >= self.width:
            return None, None
        return row, col

    def _grid_index(self, row: int, col: int) -> int:
        return row * self.width + col

    def _mark_inflated_obstacle(
        self,
        grid: List[int],
        center_row: int,
        center_col: int,
        radius_m: float,
        occupied_value: int,
    ) -> None:
        radius_cells = int(math.ceil(radius_m / self.resolution))
        for dr in range(-radius_cells, radius_cells + 1):
            for dc in range(-radius_cells, radius_cells + 1):
                dist = math.hypot(dr * self.resolution, dc * self.resolution)
                if dist > radius_m:
                    continue
                row = center_row + dr
                col = center_col + dc
                if 0 <= row < self.height and 0 <= col < self.width:
                    grid[self._grid_index(row, col)] = occupied_value

    def _apply_risk(
        self,
        risk: List[float],
        obj: MapObject,
        center_row: int,
        center_col: int,
        spec: Dict[str, Any],
    ) -> None:
        risk_cfg = spec.get("risk", {}) or {}
        risk_type = str(risk_cfg.get("type", "constant"))

        if risk_type == "gaussian":
            self._add_gaussian_risk(
                risk,
                center_row,
                center_col,
                radius_m=float(risk_cfg.get("radius_m", 10.0)),
                sigma_m=float(risk_cfg.get("sigma_m", 5.0)),
                max_value=float(risk_cfg.get("max_value", 1.0)),
            )
        elif risk_type == "wedge_if_name_contains":
            name_contains = str(risk_cfg.get("name_contains", ""))
            if name_contains and name_contains in obj.prefab_name:
                yaw = self._quat_to_yaw(obj.rotation)
                self._add_wedge_risk(
                    risk,
                    center_row,
                    center_col,
                    yaw_rad=yaw,
                    fov_deg=float(risk_cfg.get("fov_deg", 45.0)),
                    max_dist_m=float(risk_cfg.get("max_dist_m", 50.0)),
                    max_value=float(risk_cfg.get("max_value", 1.0)),
                )
            else:
                value = float(risk_cfg.get("fallback_constant", 0.1))
                risk[self._grid_index(center_row, center_col)] = max(
                    risk[self._grid_index(center_row, center_col)], value
                )
        else:
            value = float(risk_cfg.get("value", self.config.get("risk", {}).get("default_static_obstacle_risk", 0.1)))
            risk[self._grid_index(center_row, center_col)] = max(
                risk[self._grid_index(center_row, center_col)], value
            )

    def _add_gaussian_risk(
        self,
        risk: List[float],
        center_row: int,
        center_col: int,
        radius_m: float,
        sigma_m: float,
        max_value: float,
    ) -> None:
        radius_cells = int(math.ceil(radius_m / self.resolution))
        sigma = max(sigma_m, 1e-6)
        for dr in range(-radius_cells, radius_cells + 1):
            for dc in range(-radius_cells, radius_cells + 1):
                row = center_row + dr
                col = center_col + dc
                if not (0 <= row < self.height and 0 <= col < self.width):
                    continue
                dist_m = math.hypot(dr * self.resolution, dc * self.resolution)
                if dist_m > radius_m:
                    continue
                value = max_value * math.exp(-(dist_m * dist_m) / (2.0 * sigma * sigma))
                idx = self._grid_index(row, col)
                risk[idx] = max(risk[idx], value)

    def _add_wedge_risk(
        self,
        risk: List[float],
        center_row: int,
        center_col: int,
        yaw_rad: float,
        fov_deg: float,
        max_dist_m: float,
        max_value: float,
    ) -> None:
        radius_cells = int(math.ceil(max_dist_m / self.resolution))
        half_fov_rad = math.radians(fov_deg / 2.0)
        for dr in range(-radius_cells, radius_cells + 1):
            for dc in range(-radius_cells, radius_cells + 1):
                row = center_row + dr
                col = center_col + dc
                if not (0 <= row < self.height and 0 <= col < self.width):
                    continue
                dx_m = dc * self.resolution
                dy_m = dr * self.resolution
                dist_m = math.hypot(dx_m, dy_m)
                if dist_m <= 1e-6 or dist_m > max_dist_m:
                    continue
                angle = math.atan2(dy_m, dx_m)
                diff = self._normalize_angle(angle - yaw_rad)
                if abs(diff) <= half_fov_rad:
                    value = max_value * (max_dist_m - dist_m) / max_dist_m
                    idx = self._grid_index(row, col)
                    risk[idx] = max(risk[idx], value)

    def _quat_to_yaw(self, quat: Dict[str, float]) -> float:
        # Same convention as the team map_parser.py: Unity yaw is approximated from x/y/z/w.
        x = float(quat.get("x", 0.0))
        y = float(quat.get("y", 0.0))
        z = float(quat.get("z", 0.0))
        w = float(quat.get("w", 1.0))
        return math.atan2(2.0 * (w * y + x * z), 1.0 - 2.0 * (y * y + z * z))

    def _normalize_angle(self, angle: float) -> float:
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def _make_occupancy_msg(self, data: List[int], topic_name: str) -> OccupancyGrid:
        msg = OccupancyGrid()
        msg.header.frame_id = self.frame_id
        msg.info.resolution = self.resolution
        msg.info.width = self.width
        msg.info.height = self.height
        msg.info.origin.position.x = self.min_x
        msg.info.origin.position.y = self.min_y
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation.w = 1.0
        msg.data = [int(v) for v in data]
        return msg

    def _make_risk_msg(self, risk: List[float], topic_name: str) -> OccupancyGrid:
        msg = OccupancyGrid()
        msg.header.frame_id = self.frame_id
        msg.info.resolution = self.resolution
        msg.info.width = self.width
        msg.info.height = self.height
        msg.info.origin.position.x = self.min_x
        msg.info.origin.position.y = self.min_y
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation.w = 1.0
        msg.data = [int(round(max(0.0, min(1.0, v)) * 100.0)) for v in risk]
        return msg

    # ------------------------------------------------------------------
    # Raw / summary messages
    # ------------------------------------------------------------------
    def _make_raw_msg(self, map_role: str, map_data: Dict[str, Any], objects: List[MapObject]) -> String:
        payload = {
            "map_role": map_role,
            "terrainIndex": map_data.get("terrainIndex"),
            "frame_id": self.frame_id,
            "coordinate_policy": self.config.get("coordinate_policy", {}),
            "object_count": len(objects),
            "category_counts": self._category_counts(objects),
            "map_file": str(self.recon_map_file if map_role == "recon" else self.mission_map_file),
            "obstacles": map_data.get("obstacles", []),
        }
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        return msg

    def _make_summary_msg(self) -> String:
        round_m = float(self.config.get("diff", {}).get("position_round_m", 1.0))
        recon_keys = {self._object_signature(o, round_m) for o in self.recon_objects}
        mission_keys = {self._object_signature(o, round_m) for o in self.mission_objects}
        payload = {
            "frame_id": self.frame_id,
            "mode": self.mode,
            "terrain": {
                "x_range": [self.min_x, self.max_x],
                "y_range": [self.min_y, self.max_y],
                "resolution": self.resolution,
                "width": self.width,
                "height": self.height,
            },
            "assumptions": {
                "recon_map": "drone reconnaissance static map known before driving",
                "mission_map": "actual mission world / ground truth for optional comparison",
                "live_sensor": "not fused in this static-map node",
                "coordinate_transform": "rviz.x=raw.x, rviz.y=raw.z, rviz.z=raw.y",
            },
            "recon": {
                "object_count": len(self.recon_objects),
                "category_counts": self._category_counts(self.recon_objects),
            },
            "mission": {
                "object_count": len(self.mission_objects),
                "category_counts": self._category_counts(self.mission_objects),
            },
            "diff": {
                "mission_only_count": len(mission_keys - recon_keys),
                "recon_only_count": len(recon_keys - mission_keys),
                "position_round_m": round_m,
            },
            "topics": {
                "recon_markers": "/tank/rviz/recon_map_markers",
                "mission_markers": "/tank/rviz/mission_map_markers",
                "diff_markers": "/tank/rviz/map_diff_markers",
                "recon_occupancy_grid": "/tank/map/recon/occupancy_grid",
                "recon_risk_grid": "/tank/map/recon/risk_grid",
            },
        }
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        return msg

    def _category_counts(self, objects: List[MapObject]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for obj in objects:
            counts[obj.category] = counts.get(obj.category, 0) + 1
        return dict(sorted(counts.items()))

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------
    def publish_all(self) -> None:
        now = self.get_clock().now().to_msg()
        self._stamp_marker_array(self.recon_markers, now)
        self.recon_marker_pub.publish(self.recon_markers)
        self.recon_raw_pub.publish(self.recon_raw_msg)
        self.summary_pub.publish(self.summary_msg)

        if self.publish_mission:
            self._stamp_marker_array(self.mission_markers, now)
            self.mission_marker_pub.publish(self.mission_markers)
            self.mission_raw_pub.publish(self.mission_raw_msg)

        if self.publish_diff:
            self._stamp_marker_array(self.diff_markers, now)
            self.diff_marker_pub.publish(self.diff_markers)

        if self.publish_grids:
            self.recon_occupancy_msg.header.stamp = now
            self.recon_risk_msg.header.stamp = now
            self.recon_occupancy_pub.publish(self.recon_occupancy_msg)
            self.recon_risk_pub.publish(self.recon_risk_msg)

    def _stamp_marker_array(self, markers: MarkerArray, stamp: Any) -> None:
        for marker in markers.markers:
            marker.header.stamp = stamp


def main(args: Optional[List[str]] = None) -> None:
    rclpy.init(args=args)
    node = StaticMapLoaderNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
