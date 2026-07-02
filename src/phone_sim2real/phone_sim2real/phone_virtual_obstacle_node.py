# -*- coding: utf-8 -*-
"""Convert smartphone YOLO bboxes into image-only virtual obstacles.

This node intentionally does *not* run APF.  It turns Android-camera YOLO
results into the same kinds of obstacle evidence that the existing stack already
uses during scenario1/scenario2:

1. /tank/phone_sim2real/virtual_obstacles       debug/status payload
2. /tank/map/discovered/objects                 persistent map-style obstacle
3. /tank/phone_sim2real/synthetic_lidar_clusters synthetic image-only cluster
4. RViz MarkerArray topics                      visual confirmation

The synthetic cluster output is the key for APF-free integration.  The current
map_astar_planner_node can treat the muxed LiDAR cluster topic as
path-block evidence and can dynamically replan while scenario scripts are
running.  A phone object is therefore represented as an "image_cluster" with a
map-frame bbox, not as a control command and not as an APF target.
"""

from __future__ import annotations

import json
import math
import time
from typing import Any, Dict, List, Optional, Tuple

import rclpy
from geometry_msgs.msg import Point, PoseStamped, Vector3Stamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray

from phone_sim2real.common import (
    bbox_bearing_rad,
    bbox_center,
    clamp,
    color_for_class,
    estimate_distance_from_bbox,
    get_nested_float,
    normalize_angle_rad,
    parse_csv_set,
    quaternion_to_yaw_rad,
    safe_json_loads,
)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _phone_sim2real_freeze_locked_object(obj):
    """Keep locked map position immutable after lock.

    lock 이후에도 bbox가 흔들리거나 전차 pose가 변하면 새 candidate map_x/y가 조금씩 바뀔 수 있다.
    이 helper는 locked_map_x/y/z가 존재하는 동안 publish 좌표를 locked 좌표로 강제 동기화한다.
    """
    try:
        if not obj.get("is_locked") and not obj.get("locked"):
            return obj
        lx = obj.get("locked_map_x", None)
        ly = obj.get("locked_map_y", None)
        lz = obj.get("locked_map_z", None)
        if lx is None or ly is None:
            return obj
        obj["map_x"] = float(lx)
        obj["map_y"] = float(ly)
        if lz is not None:
            obj["map_z"] = float(lz)
        obj["position_map"] = {
            "x": obj["map_x"],
            "y": obj["map_y"],
            "z": float(obj.get("map_z", 0.0)),
        }
        obj["position_state"] = "virtual_phone_map_locked"
    except Exception:
        pass
    return obj


class PhoneVirtualObstacleNode(Node):
    def __init__(self) -> None:
        super().__init__("phone_virtual_obstacle_node")

        # Input/output topics.
        self.declare_parameter("map_frame", "tank_map")
        self.declare_parameter("player_pose_topic", "/tank/player/pose")
        self.declare_parameter("lookahead_pose_topic", "/tank/path/lookahead_pose")
        self.declare_parameter("detections_topic", "/tank/phone_sim2real/detections")
        self.declare_parameter("imu_json_topic", "/tank/phone_sim2real/imu_json")
        self.declare_parameter("virtual_obstacles_topic", "/tank/phone_sim2real/virtual_obstacles")
        self.declare_parameter("discovered_objects_topic", "/tank/map/discovered/objects")
        self.declare_parameter("fused_objects_topic", "/tank/perception/fused_objects")
        self.declare_parameter("synthetic_lidar_clusters_topic", "/tank/phone_sim2real/synthetic_lidar_clusters")
        self.declare_parameter("marker_topic", "/tank/rviz/phone_sim2real_markers")
        self.declare_parameter("fused_marker_topic", "/tank/rviz/fused_object_markers")
        self.declare_parameter("synthetic_cluster_marker_topic", "/tank/rviz/phone_sim2real_image_cluster_markers")

        # Camera geometry.
        self.declare_parameter("image_width", 416)
        self.declare_parameter("image_height", 416)
        self.declare_parameter("camera_hfov_deg", 62.0)
        self.declare_parameter("camera_vfov_deg", 48.0)

        # Distance estimation.
        self.declare_parameter("distance_mode", "calibrated_table")  # calibrated_table | pinhole
        self.declare_parameter("distance_scale", 0.70)
        self.declare_parameter("distance_bias_m", 0.0)
        self.declare_parameter("min_virtual_distance_m", 2.5)
        self.declare_parameter("max_virtual_distance_m", 8.0)

        # Placement distance is the distance used to place the virtual obstacle in the map.
        # It can intentionally be larger than the bbox-calibrated distance so that a
        # phone detection is treated as an obstacle farther ahead of the tank/turret.
        self.declare_parameter("placement_distance_scale", 1.6)
        self.declare_parameter("placement_distance_bias_m", 4.0)
        self.declare_parameter("min_placement_distance_m", 8.0)
        self.declare_parameter("max_placement_distance_m", 22.0)
        self.declare_parameter("class_height_m_json", '{"car":1.6,"person":1.7,"tank":2.4,"rock":1.2,"house":3.2,"tent":1.4,"unknown":1.6}')
        self.declare_parameter("class_distance_table_json", self._default_distance_table_json())

        # Object filtering/tracking.
        self.declare_parameter("obstacle_ttl_sec", 1.8)
        self.declare_parameter("publish_hz", 10.0)
        self.declare_parameter("min_confidence", 0.20)
        self.declare_parameter("ignored_classes", "person,human,blue,red")
        self.declare_parameter("class_filter_json", self._default_class_filter_json())
        self.declare_parameter("ignore_top_region_ratio", 0.08)
        self.declare_parameter("dedupe_merge_distance_m", 1.5)
        self.declare_parameter("track_merge_distance_m", 2.0)
        self.declare_parameter("locked_bbox_match_px", 120.0)
        self.declare_parameter("min_observations_for_publish", 1)
        self.declare_parameter("min_observations_for_cluster", 2)
        self.declare_parameter("max_active_tracks", 30)

        # Image-only obstacle anchoring.  This is the main improvement for
        # scenario verification: once an object is confirmed, its map position
        # can be fixed so the obstacle does not keep moving with the tank.
        self.declare_parameter("obstacle_anchor_mode", "map_lock_on_confirmed")  # ego_relative_continuous | map_lock_on_confirmed | manual_lock
        self.declare_parameter("lock_after_observations", 3)
        self.declare_parameter("locked_obstacle_ttl_sec", 10.0)
        self.declare_parameter("publish_locked_only_to_clusters", True)
        self.declare_parameter("clear_on_inject_off", False)
        self.declare_parameter("enable_phone_control_commands", True)

        # Pose/bearing.
        self.declare_parameter("use_phone_yaw", False)
        self.declare_parameter("phone_yaw_offset_deg", 0.0)
        # bearing_reference_mode:
        #   path   : current lookahead/path progress direction 기준. 회피 검증용 기본값.
        #            스마트폰 객체를 현재 주행 경로 전방에 주입해 planner 반응을 확실히 만든다.
        #   body   : tank body/player yaw 기준
        #   turret : simulator turret/camera forward 기준
        #   phone  : Android IMU yaw 기준. 스마트폰 방향 보정이 끝났을 때만 사용
        self.declare_parameter("bearing_reference_mode", "path")
        self.declare_parameter("lookahead_stale_sec", 1.0)
        self.declare_parameter("lookahead_min_distance_m", 1.0)
        self.declare_parameter("turret_topic", "/tank/api/get_action/turret")
        self.declare_parameter("turret_subscription_type", "none")  # none | string | vector | both
        self.declare_parameter("turret_vector_topic", "/tank/api/get_action/turret")
        self.declare_parameter("turret_vector_yaw_mode", "z_angle")  # z_angle | x_angle | y_angle | xy_heading | auto
        self.declare_parameter("turret_angle_unit", "auto")  # auto | deg | rad
        self.declare_parameter("turret_yaw_offset_deg", 0.0)
        # sim_heading: simulator/RViz heading convention, h=0 -> +map.y, h=90 -> +map.x.
        # math_yaw: standard map math yaw, 0 -> +map.x, 90 -> +map.y.
        self.declare_parameter("turret_yaw_convention", "sim_heading")
        self.declare_parameter("use_bbox_bearing", False)
        self.declare_parameter("turret_stale_sec", 1.0)
        self.declare_parameter("z_offset_m", 1.0)
        self.declare_parameter("line_to_vehicle", True)

        # APF-free integration with current planner.
        self.declare_parameter("enable_discovered_objects_publish", True)
        self.declare_parameter("enable_fused_objects_mirror", True)
        self.declare_parameter("enable_synthetic_lidar_clusters", True)
        self.declare_parameter("publish_empty_synthetic_clusters", False)
        self.declare_parameter("synthetic_cluster_count_hint", 8)
        self.declare_parameter("synthetic_cluster_z_min", 0.2)
        self.declare_parameter("synthetic_cluster_z_max", 2.2)
        self.declare_parameter("synthetic_cluster_id_offset", 9000)
        self.declare_parameter("synthetic_cluster_radius_scale", 1.0)
        self.declare_parameter("synthetic_cluster_min_radius_m", 1.5)
        self.declare_parameter("synthetic_cluster_max_radius_m", 5.0)
        # Emergency phone detections should block the local path strongly enough for A* to replan.
        # path_wall creates several synthetic clusters across the current path, rather than a single point obstacle.
        self.declare_parameter("emergency_path_wall_enabled", True)
        self.declare_parameter("emergency_wall_cluster_count", 3)
        self.declare_parameter("emergency_wall_half_width_m", 4.5)
        self.declare_parameter("emergency_wall_radius_m", 3.2)
        self.declare_parameter("emergency_wall_first_object_only", True)
        self.declare_parameter("emergency_wall_source_tag", "phone_sim2real_emergency_path_wall")
        # Phone obstacles are emergency input.  By default the wall is snapped to the current
        # lookahead/path point once, then kept there for TTL so the planner can replan around it.
        # Modes: object | lookahead_live | lookahead_snapshot | path_fraction_snapshot
        self.declare_parameter("emergency_wall_center_mode", "lookahead_snapshot")
        self.declare_parameter("emergency_wall_path_fraction", 1.0)
        self.declare_parameter("emergency_wall_min_forward_m", 6.0)
        self.declare_parameter("emergency_wall_max_forward_m", 14.0)
        self.declare_parameter("emergency_wall_disable_discovered_fused", True)
        self.declare_parameter("class_radius_m_json", '{"car":2.0,"tank":3.2,"rock":1.8,"house":4.0,"tent":2.2,"unknown":2.0}')

        self.map_frame = str(self.get_parameter("map_frame").value)
        self.image_width = int(self.get_parameter("image_width").value)
        self.image_height = int(self.get_parameter("image_height").value)
        self.hfov_deg = float(self.get_parameter("camera_hfov_deg").value)
        self.vfov_deg = float(self.get_parameter("camera_vfov_deg").value)
        self.distance_mode = str(self.get_parameter("distance_mode").value).strip().lower()
        self.distance_scale = float(self.get_parameter("distance_scale").value)
        self.distance_bias_m = float(self.get_parameter("distance_bias_m").value)
        self.min_distance_m = float(self.get_parameter("min_virtual_distance_m").value)
        self.max_distance_m = float(self.get_parameter("max_virtual_distance_m").value)
        self.placement_distance_scale = float(self.get_parameter("placement_distance_scale").value)
        self.placement_distance_bias_m = float(self.get_parameter("placement_distance_bias_m").value)
        self.min_placement_distance_m = float(self.get_parameter("min_placement_distance_m").value)
        self.max_placement_distance_m = float(self.get_parameter("max_placement_distance_m").value)
        self.ttl_sec = float(self.get_parameter("obstacle_ttl_sec").value)
        self.publish_hz = float(self.get_parameter("publish_hz").value)
        self.min_confidence = float(self.get_parameter("min_confidence").value)
        self.ignored_classes = parse_csv_set(str(self.get_parameter("ignored_classes").value))
        self.class_filter = self._class_filter_param_json("class_filter_json")
        self.ignore_top_region_ratio = float(self.get_parameter("ignore_top_region_ratio").value)
        self.dedupe_merge_distance_m = float(self.get_parameter("dedupe_merge_distance_m").value)
        self.track_merge_distance_m = float(self.get_parameter("track_merge_distance_m").value)
        self.locked_bbox_match_px = float(self.get_parameter("locked_bbox_match_px").value)
        self.min_observations_for_publish = max(1, int(self.get_parameter("min_observations_for_publish").value))
        self.min_observations_for_cluster = max(1, int(self.get_parameter("min_observations_for_cluster").value))
        self.max_active_tracks = max(1, int(self.get_parameter("max_active_tracks").value))
        self.obstacle_anchor_mode = str(self.get_parameter("obstacle_anchor_mode").value).strip().lower()
        self.lock_after_observations = max(1, int(self.get_parameter("lock_after_observations").value))
        self.locked_obstacle_ttl_sec = float(self.get_parameter("locked_obstacle_ttl_sec").value)
        self.publish_locked_only_to_clusters = bool(self.get_parameter("publish_locked_only_to_clusters").value)
        self.clear_on_inject_off = bool(self.get_parameter("clear_on_inject_off").value)
        self.enable_phone_control_commands = bool(self.get_parameter("enable_phone_control_commands").value)
        self.use_phone_yaw = bool(self.get_parameter("use_phone_yaw").value)
        self.phone_yaw_offset_rad = math.radians(float(self.get_parameter("phone_yaw_offset_deg").value))
        self.bearing_reference_mode = str(self.get_parameter("bearing_reference_mode").value).strip().lower()
        self.lookahead_stale_sec = float(self.get_parameter("lookahead_stale_sec").value)
        self.lookahead_min_distance_m = float(self.get_parameter("lookahead_min_distance_m").value)
        self.turret_topic = str(self.get_parameter("turret_topic").value)
        self.turret_subscription_type = str(self.get_parameter("turret_subscription_type").value).strip().lower()
        self.turret_vector_topic = str(self.get_parameter("turret_vector_topic").value)
        self.turret_vector_yaw_mode = str(self.get_parameter("turret_vector_yaw_mode").value).strip().lower()
        self.turret_angle_unit = str(self.get_parameter("turret_angle_unit").value).strip().lower()
        self.turret_yaw_offset_rad = math.radians(float(self.get_parameter("turret_yaw_offset_deg").value))
        self.turret_yaw_convention = str(self.get_parameter("turret_yaw_convention").value).strip().lower()
        self.use_bbox_bearing = bool(self.get_parameter("use_bbox_bearing").value)
        self.turret_stale_sec = float(self.get_parameter("turret_stale_sec").value)
        self.z_offset_m = float(self.get_parameter("z_offset_m").value)
        self.line_to_vehicle = bool(self.get_parameter("line_to_vehicle").value)
        self.enable_discovered_objects_publish = bool(self.get_parameter("enable_discovered_objects_publish").value)
        self.enable_fused_objects_mirror = bool(self.get_parameter("enable_fused_objects_mirror").value)
        self.enable_synthetic_lidar_clusters = bool(self.get_parameter("enable_synthetic_lidar_clusters").value)
        self.publish_empty_synthetic_clusters = bool(self.get_parameter("publish_empty_synthetic_clusters").value)
        self.synthetic_cluster_count_hint = int(self.get_parameter("synthetic_cluster_count_hint").value)
        self.synthetic_cluster_z_min = float(self.get_parameter("synthetic_cluster_z_min").value)
        self.synthetic_cluster_z_max = float(self.get_parameter("synthetic_cluster_z_max").value)
        self.synthetic_cluster_id_offset = int(self.get_parameter("synthetic_cluster_id_offset").value)
        self.synthetic_cluster_radius_scale = float(self.get_parameter("synthetic_cluster_radius_scale").value)
        self.synthetic_cluster_min_radius_m = float(self.get_parameter("synthetic_cluster_min_radius_m").value)
        self.synthetic_cluster_max_radius_m = float(self.get_parameter("synthetic_cluster_max_radius_m").value)
        self.emergency_path_wall_enabled = bool(self.get_parameter("emergency_path_wall_enabled").value)
        self.emergency_wall_cluster_count = max(1, int(self.get_parameter("emergency_wall_cluster_count").value))
        self.emergency_wall_half_width_m = max(0.0, float(self.get_parameter("emergency_wall_half_width_m").value))
        self.emergency_wall_radius_m = max(0.1, float(self.get_parameter("emergency_wall_radius_m").value))
        self.emergency_wall_first_object_only = bool(self.get_parameter("emergency_wall_first_object_only").value)
        self.emergency_wall_source_tag = str(self.get_parameter("emergency_wall_source_tag").value)
        self.emergency_wall_center_mode = str(self.get_parameter("emergency_wall_center_mode").value).strip().lower()
        self.emergency_wall_path_fraction = float(self.get_parameter("emergency_wall_path_fraction").value)
        self.emergency_wall_min_forward_m = float(self.get_parameter("emergency_wall_min_forward_m").value)
        self.emergency_wall_max_forward_m = float(self.get_parameter("emergency_wall_max_forward_m").value)
        self.emergency_wall_disable_discovered_fused = bool(self.get_parameter("emergency_wall_disable_discovered_fused").value)
        self.class_height_m = self._dict_param_json("class_height_m_json")
        self.class_radius_m = self._dict_param_json("class_radius_m_json")
        self.class_distance_table = self._distance_table_param_json("class_distance_table_json")
        # In phone emergency mode, keep phone evidence out of discovered/fused map streams.
        # Otherwise local_path/fusion/discovered layers can duplicate the same smartphone obstacle
        # while the muxed cluster already feeds the A* planner.  The phone package should own
        # only emergency stop + synthetic clusters + RViz confirmation.
        if self.emergency_wall_disable_discovered_fused and self.emergency_path_wall_enabled:
            self.enable_discovered_objects_publish = False
            self.enable_fused_objects_mirror = False

        transient_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.pub_virtual = self.create_publisher(String, str(self.get_parameter("virtual_obstacles_topic").value), 10)
        self.pub_discovered = self.create_publisher(String, str(self.get_parameter("discovered_objects_topic").value), transient_qos)
        self.pub_fused = self.create_publisher(String, str(self.get_parameter("fused_objects_topic").value), 10)
        self.pub_synthetic_clusters = self.create_publisher(String, str(self.get_parameter("synthetic_lidar_clusters_topic").value), 10)
        self.pub_markers = self.create_publisher(MarkerArray, str(self.get_parameter("marker_topic").value), 10)
        self.pub_fused_markers = self.create_publisher(MarkerArray, str(self.get_parameter("fused_marker_topic").value), 10)
        self.pub_cluster_markers = self.create_publisher(MarkerArray, str(self.get_parameter("synthetic_cluster_marker_topic").value), 10)
        self.pub_status = self.create_publisher(String, "/tank/phone_sim2real/virtual_status", 10)

        self.player_pose: Optional[PoseStamped] = None
        self.player_yaw_rad = 0.0
        self.lookahead_pose: Optional[PoseStamped] = None
        self.latest_lookahead_wall = 0.0
        self.latest_phone_yaw_rad: Optional[float] = None
        self.latest_turret_yaw_rad: Optional[float] = None
        self.latest_turret_wall = 0.0
        self.active: Dict[str, Dict[str, Any]] = {}
        self.inject_enabled_state = True
        self.next_track_id = 1
        self.last_detection_wall = 0.0

        self.create_subscription(PoseStamped, str(self.get_parameter("player_pose_topic").value), self.player_pose_cb, 10)
        self.create_subscription(PoseStamped, str(self.get_parameter("lookahead_pose_topic").value), self.lookahead_pose_cb, 10)
        self.create_subscription(String, str(self.get_parameter("detections_topic").value), self.detections_cb, 10)
        self.create_subscription(String, str(self.get_parameter("imu_json_topic").value), self.imu_json_cb, 10)
        # /tank/api/get_action/turret may have multiple ROS message types.
        # Default path-front mode does not need turret. Enable one type only if you choose turret mode.
        if self.turret_subscription_type in {"string", "both"} and self.turret_topic:
            self.create_subscription(String, self.turret_topic, self.turret_cb, 10)
        if self.turret_subscription_type in {"vector", "both"} and self.turret_vector_topic:
            self.create_subscription(Vector3Stamped, self.turret_vector_topic, self.turret_vector_cb, 10)
        self.create_timer(1.0 / max(1.0, self.publish_hz), self.timer_cb)
        self.get_logger().info(
            "phone_sim2real image-only obstacle node started: "
            f"clusters={self.enable_synthetic_lidar_clusters} -> {self.get_parameter('synthetic_lidar_clusters_topic').value}, "
            f"discovered={self.enable_discovered_objects_publish}, distance_mode={self.distance_mode}, "
            f"anchor_mode={self.obstacle_anchor_mode}, bearing_ref={self.bearing_reference_mode}, locked_clusters_only={self.publish_locked_only_to_clusters}, "
            f"ignored={sorted(self.ignored_classes)}"
        )

    @staticmethod
    def _default_distance_table_json() -> str:
        # These are safe demo starting points, not calibrated truth.  The user is
        # expected to edit them after measuring bbox_height_px at known distances.
        return json.dumps({
            "car": [
                {"bbox_height_px": 45, "distance_m": 8.0},
                {"bbox_height_px": 70, "distance_m": 6.0},
                {"bbox_height_px": 110, "distance_m": 4.0},
                {"bbox_height_px": 170, "distance_m": 3.0},
                {"bbox_height_px": 240, "distance_m": 2.5},
            ],
            "tank": [
                {"bbox_height_px": 45, "distance_m": 8.0},
                {"bbox_height_px": 75, "distance_m": 6.0},
                {"bbox_height_px": 120, "distance_m": 4.2},
                {"bbox_height_px": 185, "distance_m": 3.0},
                {"bbox_height_px": 260, "distance_m": 2.5},
            ],
            "rock": [
                {"bbox_height_px": 40, "distance_m": 7.5},
                {"bbox_height_px": 70, "distance_m": 5.5},
                {"bbox_height_px": 115, "distance_m": 3.8},
                {"bbox_height_px": 180, "distance_m": 2.8},
                {"bbox_height_px": 250, "distance_m": 2.5},
            ],
            "house": [
                {"bbox_height_px": 55, "distance_m": 8.0},
                {"bbox_height_px": 90, "distance_m": 6.0},
                {"bbox_height_px": 145, "distance_m": 4.0},
                {"bbox_height_px": 220, "distance_m": 3.0},
                {"bbox_height_px": 300, "distance_m": 2.5},
            ],
            "tent": [
                {"bbox_height_px": 45, "distance_m": 7.5},
                {"bbox_height_px": 80, "distance_m": 5.2},
                {"bbox_height_px": 130, "distance_m": 3.7},
                {"bbox_height_px": 200, "distance_m": 2.7},
                {"bbox_height_px": 280, "distance_m": 2.5},
            ],
            "unknown": [
                {"bbox_height_px": 50, "distance_m": 8.0},
                {"bbox_height_px": 90, "distance_m": 5.5},
                {"bbox_height_px": 150, "distance_m": 3.5},
                {"bbox_height_px": 240, "distance_m": 2.5},
            ],
        }, ensure_ascii=False)

    @staticmethod
    def _default_class_filter_json() -> str:
        # Class-specific filters remove common image-only false positives.
        # Example from testing: a small top-of-image car false positive with
        # bbox height around 20 px is dropped by the car min_bbox_height_px.
        return json.dumps({
            "rock": {"min_conf": 0.70, "min_bbox_height_px": 40, "min_bbox_area_px": 2500},
            "car": {"min_conf": 0.80, "min_bbox_height_px": 45, "min_bbox_area_px": 3000},
            "tank": {"min_conf": 0.75, "min_bbox_height_px": 45, "min_bbox_area_px": 3000},
            "house": {"min_conf": 0.75, "min_bbox_height_px": 50, "min_bbox_area_px": 3500},
            "tent": {"min_conf": 0.70, "min_bbox_height_px": 45, "min_bbox_area_px": 2500},
            "unknown": {"min_conf": 0.70, "min_bbox_height_px": 45, "min_bbox_area_px": 2500}
        }, ensure_ascii=False)

    def _dict_param_json(self, name: str) -> Dict[str, float]:
        raw = str(self.get_parameter(name).value or "{}")
        data = safe_json_loads(raw, {})
        result: Dict[str, float] = {}
        if isinstance(data, dict):
            for k, v in data.items():
                try:
                    result[str(k).lower()] = float(v)
                except Exception:
                    pass
        return result

    def _distance_table_param_json(self, name: str) -> Dict[str, List[Tuple[float, float]]]:
        raw = str(self.get_parameter(name).value or "{}")
        data = safe_json_loads(raw, {})
        result: Dict[str, List[Tuple[float, float]]] = {}
        if not isinstance(data, dict):
            return result
        for cls, rows in data.items():
            samples: List[Tuple[float, float]] = []
            if isinstance(rows, list):
                for item in rows:
                    if not isinstance(item, dict):
                        continue
                    h = _as_float(item.get("bbox_height_px"), -1.0)
                    d = _as_float(item.get("distance_m"), -1.0)
                    if h > 0.0 and d > 0.0:
                        samples.append((h, d))
            if samples:
                samples.sort(key=lambda x: x[0])
                result[str(cls).lower()] = samples
        return result

    def _class_filter_param_json(self, name: str) -> Dict[str, Dict[str, float]]:
        raw = str(self.get_parameter(name).value or "{}")
        data = safe_json_loads(raw, {})
        result: Dict[str, Dict[str, float]] = {}
        if isinstance(data, dict):
            for cls, cfg in data.items():
                if not isinstance(cfg, dict):
                    continue
                out: Dict[str, float] = {}
                for key in ("min_conf", "min_bbox_height_px", "min_bbox_width_px", "min_bbox_area_px"):
                    if key in cfg:
                        try:
                            out[key] = float(cfg[key])
                        except Exception:
                            pass
                if out:
                    result[str(cls).lower()] = out
        return result

    def _phone_meta_from_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        meta = payload.get("phone") if isinstance(payload.get("phone"), dict) else {}
        return meta if isinstance(meta, dict) else {}

    def _meta_bool(self, meta: Dict[str, Any], keys: Tuple[str, ...], default: bool = False) -> bool:
        for key in keys:
            if key not in meta:
                continue
            value = meta.get(key)
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)):
                return bool(value)
            if isinstance(value, str):
                return value.strip().lower() in {"1", "true", "yes", "y", "on", "enable", "enabled"}
        return default

    def _control_command(self, payload: Dict[str, Any], meta: Dict[str, Any]) -> str:
        if not self.enable_phone_control_commands:
            return ""
        for src in (payload, meta):
            for key in ("command", "control", "action"):
                value = src.get(key) if isinstance(src, dict) else None
                if value is not None:
                    return str(value).strip().lower()
        return ""

    def _is_inject_enabled(self, meta: Dict[str, Any]) -> bool:
        return self._meta_bool(meta, ("inject_enabled", "injection_enabled", "inject", "enable_injection"), True)

    def _lock_all_active(self, now: float, reason: str = "manual") -> None:
        for obj in self.active.values():
            self._apply_lock_to_object(obj, now, reason=reason)

    def _apply_lock_to_object(self, obj: Dict[str, Any], now: float, reason: str = "auto") -> None:
        if not obj.get("is_locked", False):
            obj["locked_map_x"] = float(obj.get("map_x", 0.0))
            obj["locked_map_y"] = float(obj.get("map_y", 0.0))
            obj["locked_map_z"] = float(obj.get("map_z", 1.0))
            obj["locked_wall"] = now
            obj["lock_reason"] = reason
        x = float(obj.get("locked_map_x", obj.get("map_x", 0.0)))
        y = float(obj.get("locked_map_y", obj.get("map_y", 0.0)))
        z = float(obj.get("locked_map_z", obj.get("map_z", 1.0)))
        obj["map_x"] = x
        obj["map_y"] = y
        obj["map_z"] = z
        obj["position_map"] = {"x": x, "y": y, "z": z}
        obj["is_locked"] = True
        obj["locked"] = True
        obj["position_state"] = "virtual_phone_map_locked"
        obj["expires_wall"] = now + max(self.ttl_sec, self.locked_obstacle_ttl_sec)

    def _bbox_center_distance_px(self, a: Dict[str, Any], b: Dict[str, Any]) -> float:
        try:
            ac = a.get("bbox_center", [0.0, 0.0])
            bc = b.get("bbox_center", [0.0, 0.0])
            return math.hypot(float(ac[0]) - float(bc[0]), float(ac[1]) - float(bc[1]))
        except Exception:
            return 1e9

    def player_pose_cb(self, msg: PoseStamped) -> None:
        self.player_pose = msg
        self.player_yaw_rad = quaternion_to_yaw_rad(msg.pose.orientation)

    def lookahead_pose_cb(self, msg: PoseStamped) -> None:
        self.lookahead_pose = msg
        self.latest_lookahead_wall = time.time()

    def imu_json_cb(self, msg: String) -> None:
        payload = safe_json_loads(msg.data, {})
        if not isinstance(payload, dict):
            return
        yaw = get_nested_float(payload, ["imu", "yawRad"], None)
        if yaw is None:
            yaw = get_nested_float(payload, ["imu", "yaw_rad"], None)
        if yaw is not None:
            self.latest_phone_yaw_rad = float(yaw)

    def _angle_to_rad(self, value: Any) -> Optional[float]:
        try:
            angle = float(value)
        except Exception:
            return None
        unit = self.turret_angle_unit
        if unit == "deg":
            return math.radians(angle)
        if unit == "rad":
            return angle
        # auto: Unity/ros_bridge turret 값은 degree 계열인 경우가 많다.
        # 절댓값이 2*pi보다 크면 degree로 보고, 아니면 rad로 본다.
        if abs(angle) > (2.0 * math.pi + 1e-6):
            return math.radians(angle)
        return angle

    def _turret_heading_to_math_yaw(self, yaw_rad: Optional[float]) -> Optional[float]:
        """Convert simulator/RViz turret heading to standard map math yaw.

        RViz uses simulator turret.x as heading where:
          heading 0 deg  -> +map.y
          heading 90 deg -> +map.x
        This node places objects with:
          x = px + cos(yaw) * d
          y = py + sin(yaw) * d
        Therefore the conversion is:
          math_yaw = pi/2 - simulator_heading
        """
        if yaw_rad is None:
            return None
        if self.turret_yaw_convention in {"sim", "sim_heading", "unity", "rviz", "rviz_heading", "tank_heading"}:
            return normalize_angle_rad((math.pi * 0.5) - float(yaw_rad))
        return normalize_angle_rad(float(yaw_rad))

    def _extract_turret_yaw_rad(self, payload: Any) -> Optional[float]:
        if isinstance(payload, (int, float, str)):
            return self._angle_to_rad(payload)
        if not isinstance(payload, dict):
            return None

        # 흔한 평면 yaw key.
        for key in (
            "yaw", "yaw_rad", "yawRad", "yaw_deg", "yawDeg",
            "turret_yaw", "turretYaw", "turret_yaw_rad", "turretYawRad",
            "turret_yaw_deg", "turretYawDeg",
            "angle", "angle_rad", "angleRad", "angle_deg", "angleDeg",
            "azimuth", "azimuth_rad", "azimuth_deg",
        ):
            if key in payload:
                value = payload.get(key)
                if key.lower().endswith("deg"):
                    try:
                        return math.radians(float(value))
                    except Exception:
                        return None
                if key.lower().endswith("rad"):
                    try:
                        return float(value)
                    except Exception:
                        return None
                return self._angle_to_rad(value)

        # nested turret dict.
        for outer in ("turret", "turretRotation", "turret_rotation", "get_action"):
            inner = payload.get(outer)
            value = self._extract_turret_yaw_rad(inner)
            if value is not None:
                return value
        return None

    def turret_cb(self, msg: String) -> None:
        raw = msg.data.strip()
        payload = safe_json_loads(raw, raw)
        yaw = self._extract_turret_yaw_rad(payload)
        if yaw is None:
            return
        self.latest_turret_yaw_rad = self._turret_heading_to_math_yaw(float(yaw))
        self.latest_turret_wall = time.time()

    def turret_vector_cb(self, msg: Vector3Stamped) -> None:
        yaw = self._extract_turret_yaw_from_vector(msg)
        if yaw is None:
            return
        self.latest_turret_yaw_rad = self._turret_heading_to_math_yaw(float(yaw))
        self.latest_turret_wall = time.time()

    def _extract_turret_yaw_from_vector(self, msg: Vector3Stamped) -> Optional[float]:
        mode = self.turret_vector_yaw_mode
        vx = float(msg.vector.x)
        vy = float(msg.vector.y)
        vz = float(msg.vector.z)

        if mode in {"xy", "xy_heading", "heading", "direction"}:
            if abs(vx) < 1e-9 and abs(vy) < 1e-9:
                return None
            return math.atan2(vy, vx)

        if mode in {"x", "x_angle", "roll"}:
            return self._angle_to_rad(vx)
        if mode in {"y", "y_angle", "pitch"}:
            return self._angle_to_rad(vy)
        if mode in {"z", "z_angle", "yaw"}:
            return self._angle_to_rad(vz)

        # auto:
        # - If x/y looks like a direction vector, use atan2(y, x).
        # - Otherwise treat z as the yaw angle.
        if (abs(vx) > 1e-6 or abs(vy) > 1e-6) and abs(vz) < 1e-6:
            return math.atan2(vy, vx)
        return self._angle_to_rad(vz)

    def _bearing_base_yaw(self) -> Tuple[float, str]:
        now = time.time()
        mode = self.bearing_reference_mode

        # Recommended for verification: place the phone obstacle in the current driving/path direction.
        # This avoids injecting the obstacle beside or behind the tank when turret/body yaw is not aligned
        # with the route, or when simulator turret topic type/units are ambiguous.
        if mode in {"path", "lookahead", "route", "progress", "front", "driving"}:
            if self.player_pose is not None and self.lookahead_pose is not None and (now - self.latest_lookahead_wall) <= self.lookahead_stale_sec:
                px = float(self.player_pose.pose.position.x)
                py = float(self.player_pose.pose.position.y)
                lx = float(self.lookahead_pose.pose.position.x)
                ly = float(self.lookahead_pose.pose.position.y)
                dx = lx - px
                dy = ly - py
                dist = math.hypot(dx, dy)
                if dist >= self.lookahead_min_distance_m:
                    return normalize_angle_rad(math.atan2(dy, dx)), "lookahead_path"
            # Fallback order for path mode: body is more stable than ambiguous turret.
            return self.player_yaw_rad, "body_fallback_no_lookahead"

        if mode in {"turret", "sim_turret", "simulator_turret", "turret_forward", "camera", "sim_camera"}:
            if self.latest_turret_yaw_rad is not None and (now - self.latest_turret_wall) <= self.turret_stale_sec:
                return normalize_angle_rad(self.latest_turret_yaw_rad + self.turret_yaw_offset_rad), "turret_rviz_heading"
            return self.player_yaw_rad, "body_fallback_no_turret"

        if mode in {"phone", "android", "phone_imu"} or self.use_phone_yaw:
            if self.latest_phone_yaw_rad is not None:
                return normalize_angle_rad(self.latest_phone_yaw_rad + self.phone_yaw_offset_rad), "phone_imu"
            return self.player_yaw_rad, "body_fallback_no_phone_yaw"

        return self.player_yaw_rad, "body"

    def _placement_distance(self, estimated_distance_m: float) -> float:
        """Distance used for map placement.

        The phone camera estimates distance from bbox size, but for scenario
        validation we often want to inject the obstacle farther ahead of the
        tank/turret than the raw bbox estimate.  This keeps the sensor estimate
        visible while moving the virtual obstacle forward in map frame.
        """
        d = float(estimated_distance_m) * self.placement_distance_scale + self.placement_distance_bias_m
        return clamp(d, self.min_placement_distance_m, self.max_placement_distance_m)



    def _lookahead_anchor_xy(self, fallback_yaw: float, fallback_distance: float) -> Tuple[float, float, str, float]:
        """Return a path-centered emergency wall anchor.

        For smartphone emergency avoidance, the point obstacle estimated from image geometry is less
        important than blocking the route the tank is about to follow.  This helper snaps the phone
        hazard to the current lookahead/path segment and returns (x, y, source, yaw).
        """
        px = float(self.player_pose.pose.position.x) if self.player_pose is not None else 0.0
        py = float(self.player_pose.pose.position.y) if self.player_pose is not None else 0.0
        now = time.time()
        mode = str(getattr(self, "emergency_wall_center_mode", "object")).strip().lower()

        if mode in {"turret", "turret_snapshot", "turret_forward", "turret_forward_snapshot", "camera", "camera_snapshot"}:
            yaw = normalize_angle_rad(float(fallback_yaw))
            min_f = max(0.1, float(getattr(self, "emergency_wall_min_forward_m", 6.0)))
            max_f = max(min_f, float(getattr(self, "emergency_wall_max_forward_m", 14.0)))
            d = clamp(float(fallback_distance), min_f, max_f)
            return px + math.cos(yaw) * d, py + math.sin(yaw) * d, "turret_snapshot", yaw

        if mode in {"lookahead", "lookahead_pose", "lookahead_live", "lookahead_snapshot", "path_fraction", "path_fraction_snapshot"}:
            if self.player_pose is not None and self.lookahead_pose is not None and (now - self.latest_lookahead_wall) <= self.lookahead_stale_sec:
                lx = float(self.lookahead_pose.pose.position.x)
                ly = float(self.lookahead_pose.pose.position.y)
                dx = lx - px
                dy = ly - py
                dist = math.hypot(dx, dy)
                if dist >= max(0.1, self.lookahead_min_distance_m):
                    yaw = normalize_angle_rad(math.atan2(dy, dx))
                    # Clamp wall center to a local range, so it blocks the immediate corridor but does
                    # not become a far-away global obstacle.
                    min_f = max(0.1, float(getattr(self, "emergency_wall_min_forward_m", 6.0)))
                    max_f = max(min_f, float(getattr(self, "emergency_wall_max_forward_m", 14.0)))
                    if mode in {"path_fraction", "path_fraction_snapshot"}:
                        frac = clamp(float(getattr(self, "emergency_wall_path_fraction", 1.0)), 0.1, 1.0)
                        use_dist = clamp(dist * frac, min_f, max_f)
                    else:
                        use_dist = clamp(dist, min_f, max_f)
                    return px + math.cos(yaw) * use_dist, py + math.sin(yaw) * use_dist, "lookahead_snapshot", yaw
        # Fallback: use current body/path bearing and configured placement distance.
        yaw = normalize_angle_rad(float(fallback_yaw))
        d = clamp(float(fallback_distance), self.emergency_wall_min_forward_m, self.emergency_wall_max_forward_m)
        return px + math.cos(yaw) * d, py + math.sin(yaw) * d, "bearing_distance_fallback", yaw

    def _apply_emergency_anchor_to_candidate(self, cand: Dict[str, Any], base_yaw: float, placement_distance: float) -> Dict[str, Any]:
        """Snap the candidate to the emergency path wall anchor when configured."""
        if not bool(getattr(self, "emergency_path_wall_enabled", False)):
            return cand
        mode = str(getattr(self, "emergency_wall_center_mode", "object")).strip().lower()
        if mode in {"object", "object_position", "raw"}:
            cand["emergency_wall_center_x"] = float(cand.get("map_x", 0.0))
            cand["emergency_wall_center_y"] = float(cand.get("map_y", 0.0))
            cand["emergency_wall_yaw_deg"] = float(cand.get("base_yaw_deg", 0.0))
            cand["emergency_wall_center_source"] = "object_position"
            return cand
        ax, ay, source, yaw = self._lookahead_anchor_xy(base_yaw, placement_distance)
        cand["map_x"] = float(ax)
        cand["map_y"] = float(ay)
        cand["position_map"] = {"x": float(ax), "y": float(ay), "z": float(cand.get("map_z", 0.0))}
        cand["emergency_wall_center_x"] = float(ax)
        cand["emergency_wall_center_y"] = float(ay)
        cand["emergency_wall_yaw_deg"] = math.degrees(yaw)
        cand["emergency_wall_center_source"] = source
        cand["position_state"] = "phone_emergency_path_wall_anchor"
        return cand

    def detections_cb(self, msg: String) -> None:
        payload = safe_json_loads(msg.data, {})
        if not isinstance(payload, dict):
            return
        now = time.time()
        self.last_detection_wall = now
        phone_meta = self._phone_meta_from_payload(payload)
        command = self._control_command(payload, phone_meta)

        if command in {"clear", "clear_obstacle", "clear_obstacles", "reset"} or self._meta_bool(phone_meta, ("clear_request", "clear_obstacles"), False):
            self.active.clear()
            self._publish_status("cleared_by_phone_command", payload, [])
            return

        manual_lock_requested = (
            command in {"lock", "lock_obstacle", "lock_obstacles", "freeze"}
            or self._meta_bool(phone_meta, ("manual_lock_request", "lock_request", "lock_obstacle"), False)
        )
        if manual_lock_requested:
            self._lock_all_active(now, reason="manual_command")

        inject_enabled = self._is_inject_enabled(phone_meta)
        if command in {"inject_on", "injection_on", "enable_injection"}:
            inject_enabled = True
        if command in {"inject_off", "injection_off", "disable_injection"}:
            inject_enabled = False
        self.inject_enabled_state = bool(inject_enabled)
        if not self.inject_enabled_state:
            if self.clear_on_inject_off:
                self.active.clear()
            self._publish_status("injection_disabled", payload, [])
            return

        if self.player_pose is None:
            self._publish_status("waiting_player_pose", payload, [])
            return

        frame = payload.get("frame") if isinstance(payload.get("frame"), dict) else {}
        width = int(frame.get("width") or self.image_width)
        height = int(frame.get("height") or self.image_height)
        detections = payload.get("detections") if isinstance(payload.get("detections"), list) else []

        candidates: List[Dict[str, Any]] = []
        for idx, det in enumerate(detections):
            if not isinstance(det, dict):
                continue
            obj = self._convert_detection_candidate(det, idx, width, height, now)
            if obj is not None:
                candidates.append(obj)

        merged_frame = self._dedupe_candidates(candidates)
        converted: List[Dict[str, Any]] = []
        for cand in merged_frame:
            tracked = self._assign_track(cand, now, manual_lock_requested=manual_lock_requested)
            if tracked is not None:
                converted.append(tracked)

        self._prune_active(now)
        self._limit_active_tracks()
        self._publish_status("ok", payload, converted)

    def _estimate_distance(self, cls: str, bbox: List[float], height: int) -> Tuple[float, str]:
        box_h = max(1.0, float(bbox[3]) - float(bbox[1]))
        if self.distance_mode in {"calibrated", "calibrated_table", "table", "piecewise"}:
            samples = self.class_distance_table.get(cls) or self.class_distance_table.get("unknown")
            if samples:
                # Clamp outside measured range, linearly interpolate inside.
                if box_h <= samples[0][0]:
                    d = samples[0][1]
                elif box_h >= samples[-1][0]:
                    d = samples[-1][1]
                else:
                    d = samples[-1][1]
                    for (h0, d0), (h1, d1) in zip(samples[:-1], samples[1:]):
                        if h0 <= box_h <= h1:
                            t = (box_h - h0) / max(1e-6, h1 - h0)
                            d = d0 + (d1 - d0) * t
                            break
                d = d * self.distance_scale + self.distance_bias_m
                return clamp(d, self.min_distance_m, self.max_distance_m), "calibrated_table"

        object_height = self.class_height_m.get(cls, self.class_height_m.get("unknown", 1.6))
        d = estimate_distance_from_bbox(
            bbox=bbox,
            image_height=height,
            vfov_deg=self.vfov_deg,
            object_height_m=object_height,
            scale=self.distance_scale,
            bias_m=self.distance_bias_m,
            min_distance_m=self.min_distance_m,
            max_distance_m=self.max_distance_m,
        )
        return d, "pinhole_height"

    def _passes_class_filter(self, cls: str, conf: float, box_w: float, box_h: float, bbox: List[float], image_h: int) -> bool:
        cfg = self.class_filter.get(cls) or self.class_filter.get("unknown") or {}
        min_conf = float(cfg.get("min_conf", self.min_confidence))
        min_h = float(cfg.get("min_bbox_height_px", 0.0))
        min_w = float(cfg.get("min_bbox_width_px", 0.0))
        min_area = float(cfg.get("min_bbox_area_px", 0.0))
        if conf < min_conf:
            return False
        if box_h < min_h or box_w < min_w:
            return False
        if box_h * box_w < min_area:
            return False
        if self.ignore_top_region_ratio > 0.0:
            _, cy = bbox_center(bbox)
            if cy < float(image_h) * self.ignore_top_region_ratio:
                return False
        return True

    def _convert_detection_candidate(self, det: Dict[str, Any], idx: int, width: int, height: int, now: float) -> Optional[Dict[str, Any]]:
        cls = str(det.get("className", det.get("class_name", det.get("label", det.get("name", "unknown"))))).strip().lower() or "unknown"
        if cls in self.ignored_classes:
            return None
        conf = _as_float(det.get("confidence", det.get("conf", det.get("score", 0.0))), 0.0)
        if conf < self.min_confidence:
            return None
        bbox_raw = det.get("bbox")
        if not isinstance(bbox_raw, list) or len(bbox_raw) < 4:
            return None
        try:
            bbox = [float(v) for v in bbox_raw[:4]]
            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                return None
        except Exception:
            return None

        box_w = max(1.0, bbox[2] - bbox[0])
        box_h = max(1.0, bbox[3] - bbox[1])
        if not self._passes_class_filter(cls, conf, box_w, box_h, bbox, height):
            return None

        estimated_distance, distance_source = self._estimate_distance(cls, bbox, height)
        placement_distance = self._placement_distance(estimated_distance)
        image_bearing = bbox_bearing_rad(bbox, width, self.hfov_deg)
        bearing = image_bearing if self.use_bbox_bearing else 0.0
        base_yaw, bearing_reference_source = self._bearing_base_yaw()
        world_bearing = normalize_angle_rad(base_yaw + bearing)
        px = float(self.player_pose.pose.position.x)  # type: ignore[union-attr]
        py = float(self.player_pose.pose.position.y)  # type: ignore[union-attr]
        pz = float(self.player_pose.pose.position.z) if self.player_pose is not None else 0.0
        mx = px + placement_distance * math.cos(world_bearing)
        my = py + placement_distance * math.sin(world_bearing)
        mz = pz + self.z_offset_m
        cx, cy = bbox_center(bbox)
        base_radius = self.class_radius_m.get(cls, self.class_radius_m.get("unknown", 2.0))
        avoidance_radius = clamp(base_radius, self.synthetic_cluster_min_radius_m, self.synthetic_cluster_max_radius_m)

        cand = {
            "_candidate_idx": idx,
            "class_name": cls,
            "className": cls,
            "confidence": conf,
            "bbox": bbox,
            "bbox_center": [cx, cy],
            "bbox_size": {"width_px": box_w, "height_px": box_h},
            "image_width": width,
            "image_height": height,
            "estimated_distance_m": float(estimated_distance),
            "bbox_distance_m": float(estimated_distance),
            "placement_distance_m": float(placement_distance),
            "distance_m": float(placement_distance),
            "distance_source": distance_source + "+placement_scale",
            "bearing_deg": math.degrees(bearing),
            "image_bearing_deg": math.degrees(image_bearing),
            "base_yaw_deg": math.degrees(base_yaw),
            "world_bearing_deg": math.degrees(world_bearing),
            "bearing_reference_mode": self.bearing_reference_mode,
            "bearing_reference_source": bearing_reference_source,
            "turret_yaw_deg": math.degrees(self.latest_turret_yaw_rad) if self.latest_turret_yaw_rad is not None else None,
            "map_x": mx,
            "map_y": my,
            "map_z": mz,
            "position_map": {"x": mx, "y": my, "z": mz},
            "source": "phone_sim2real_bbox_depth",
            "avoidance_radius_m": float(avoidance_radius),
            "_now": now,
        }
        return self._apply_emergency_anchor_to_candidate(cand, base_yaw, placement_distance)

    def _dedupe_candidates(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        kept: List[Dict[str, Any]] = []
        for cand in sorted(candidates, key=lambda o: float(o.get("confidence", 0.0)), reverse=True):
            duplicate = False
            for old in kept:
                if str(old.get("class_name")) != str(cand.get("class_name")):
                    continue
                dx = float(old.get("map_x", 0.0)) - float(cand.get("map_x", 0.0))
                dy = float(old.get("map_y", 0.0)) - float(cand.get("map_y", 0.0))
                if math.hypot(dx, dy) <= self.dedupe_merge_distance_m:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(cand)
        return kept

    def _assign_track(self, cand: Dict[str, Any], now: float, manual_lock_requested: bool = False) -> Optional[Dict[str, Any]]:
        cls = str(cand.get("class_name", "unknown"))
        best_id = ""
        best_score = 1e9
        for tid, old in self.active.items():
            if str(old.get("class_name")) != cls:
                continue
            if bool(old.get("is_locked", False)) and self.obstacle_anchor_mode in {"map_lock_on_confirmed", "manual_lock"}:
                # Once an object is locked in map coordinates, the tank can move.
                # Match the continued phone observation by image-space bbox center
                # instead of map distance so the locked obstacle does not drift.
                px_dist = self._bbox_center_distance_px(old, cand)
                if px_dist <= self.locked_bbox_match_px and px_dist < best_score:
                    best_score = px_dist
                    best_id = tid
                continue
            dx = float(old.get("map_x", 0.0)) - float(cand.get("map_x", 0.0))
            dy = float(old.get("map_y", 0.0)) - float(cand.get("map_y", 0.0))
            dist = math.hypot(dx, dy)
            if dist < best_score and dist <= self.track_merge_distance_m:
                best_score = dist
                best_id = tid

        old: Optional[Dict[str, Any]] = None
        if best_id:
            old = self.active[best_id]
            obs_count = int(old.get("observation_count", 0)) + 1
            first_seen = float(old.get("first_seen_wall", now))
            object_id = best_id
        else:
            object_id = f"phone_{cls}_track_{self.next_track_id:03d}"
            self.next_track_id += 1
            obs_count = 1
            first_seen = now


        # Preserve the first path-wall snapshot for a continuing phone track.
        # Without this, the wall can chase the newly replanned path and either cause
        # oscillation or fail to represent the original emergency obstacle.
        if old and self.emergency_path_wall_enabled and self.emergency_wall_center_mode in {"lookahead_snapshot", "path_fraction_snapshot", "turret_snapshot", "turret_forward_snapshot", "camera_snapshot"}:
            if "emergency_wall_center_x" in old and "emergency_wall_center_y" in old:
                cand["emergency_wall_center_x"] = float(old.get("emergency_wall_center_x"))
                cand["emergency_wall_center_y"] = float(old.get("emergency_wall_center_y"))
                cand["emergency_wall_yaw_deg"] = float(old.get("emergency_wall_yaw_deg", cand.get("base_yaw_deg", 0.0)))
                cand["emergency_wall_center_source"] = str(old.get("emergency_wall_center_source", "lookahead_snapshot"))
                cand["map_x"] = cand["emergency_wall_center_x"]
                cand["map_y"] = cand["emergency_wall_center_y"]
                cand["position_map"] = {"x": cand["map_x"], "y": cand["map_y"], "z": float(cand.get("map_z", 0.0))}
                cand["position_state"] = "phone_emergency_path_wall_snapshot"

        cand["object_id"] = object_id
        cand["id"] = object_id
        cand["stable_object_id"] = object_id
        cand["observation_count"] = obs_count
        cand["first_seen_wall"] = first_seen
        cand["last_seen_wall"] = now
        cand["is_confirmed"] = obs_count >= self.min_observations_for_publish
        cand["confirmed"] = bool(cand["is_confirmed"])
        cand["trackId"] = object_id
        cand["discovered_eligible"] = bool(cand["is_confirmed"])
        cand["is_locked"] = False
        cand["locked"] = False
        cand["position_state"] = "virtual_phone_image_only"
        cand["expires_wall"] = now + self.ttl_sec

        if old and bool(old.get("is_locked", False)):
            cand["locked_map_x"] = float(old.get("locked_map_x", old.get("map_x", cand.get("map_x", 0.0))))
            cand["locked_map_y"] = float(old.get("locked_map_y", old.get("map_y", cand.get("map_y", 0.0))))
            cand["locked_map_z"] = float(old.get("locked_map_z", old.get("map_z", cand.get("map_z", 1.0))))
            cand["locked_wall"] = float(old.get("locked_wall", now))
            cand["lock_reason"] = old.get("lock_reason", "previous")
            self._apply_lock_to_object(cand, now, reason=str(cand.get("lock_reason", "previous")))
        else:
            should_lock = False
            lock_reason = ""
            if manual_lock_requested:
                should_lock = True
                lock_reason = "manual_command"
            elif self.obstacle_anchor_mode == "map_lock_on_confirmed" and obs_count >= self.lock_after_observations:
                should_lock = True
                lock_reason = "confirmed_observations"
            elif self.obstacle_anchor_mode == "manual_lock":
                should_lock = False
            elif self.obstacle_anchor_mode == "ego_relative_continuous":
                should_lock = False
            if should_lock:
                self._apply_lock_to_object(cand, now, reason=lock_reason)

        self.active[object_id] = cand
        return cand

    def _prune_active(self, now: float) -> None:
        expired = [k for k, v in self.active.items() if float(v.get("expires_wall", 0.0)) < now]
        for k in expired:
            self.active.pop(k, None)

    def _limit_active_tracks(self) -> None:
        if len(self.active) <= self.max_active_tracks:
            return
        items = sorted(self.active.items(), key=lambda kv: float(kv[1].get("last_seen_wall", 0.0)), reverse=True)
        self.active = dict(items[: self.max_active_tracks])

    def _published_objects(self) -> List[Dict[str, Any]]:
        now = time.time()
        self._prune_active(now)
        objects = [
            obj for obj in self.active.values()
            if int(obj.get("observation_count", 0)) >= self.min_observations_for_publish
        ]
        objects.sort(key=lambda o: float(o.get("distance_m", 1e9)))
        return objects

    def _cluster_objects(self, objects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        result: List[Dict[str, Any]] = []
        for obj in [_phone_sim2real_freeze_locked_object(o) for o in objects]:
            if int(obj.get("observation_count", 0)) < self.min_observations_for_cluster:
                continue
            if self.publish_locked_only_to_clusters and not bool(obj.get("is_locked", False)):
                continue
            result.append(obj)
        return result

    def timer_cb(self) -> None:
        objects = self._published_objects() if self.inject_enabled_state else []
        cluster_objects = self._cluster_objects(objects) if self.inject_enabled_state else []
        now = time.time()
        payload = {
            "timestamp_wall": now,
            "frame_id": self.map_frame,
            "source": "phone_sim2real",
            "mode": "image_only_no_apf",
            "count": len(objects),
            "confirmed_count": len(objects),
            "cluster_publish_count": len(cluster_objects),
            "objects": [_phone_sim2real_freeze_locked_object(o) for o in objects],
        }
        self._publish_json(self.pub_virtual, payload)
        if self.enable_discovered_objects_publish:
            self._publish_json(self.pub_discovered, payload)
        if self.enable_fused_objects_mirror:
            self._publish_json(self.pub_fused, payload)
        if self.enable_synthetic_lidar_clusters and (cluster_objects or self.publish_empty_synthetic_clusters):
            self._publish_json(self.pub_synthetic_clusters, self._build_synthetic_cluster_payload(cluster_objects, now))
        self.pub_markers.publish(self._build_markers(objects, now, namespace="phone_virtual_obstacles"))
        self.pub_fused_markers.publish(self._build_markers(objects, now, namespace="phone_fused_mirror"))
        self.pub_cluster_markers.publish(self._build_cluster_markers(cluster_objects, now))

    def _build_synthetic_cluster_payload(self, objects: List[Dict[str, Any]], now: float) -> Dict[str, Any]:
        """Build planner-compatible synthetic cluster payload.

        In emergency path-wall mode, a phone detection creates a short wall of
        clusters across the current path.  This makes the planner treat the
        phone event as a hard local path block instead of a small point obstacle
        that the controller may graze or ignore.
        """
        clusters: List[Dict[str, Any]] = []
        px = float(self.player_pose.pose.position.x) if self.player_pose is not None else 0.0
        py = float(self.player_pose.pose.position.y) if self.player_pose is not None else 0.0

        source_objects = list(objects)
        if self.emergency_path_wall_enabled and self.emergency_wall_first_object_only:
            source_objects = source_objects[:1]

        def append_cluster(obj: Dict[str, Any], cluster_id: int, x: float, y: float, z: float, radius: float, wall_offset_m: float = 0.0) -> None:
            cls = str(obj.get("class_name", "unknown"))
            bbox = {
                "x_min": x - radius,
                "x_max": x + radius,
                "y_min": y - radius,
                "y_max": y + radius,
                "z_min": max(0.0, self.synthetic_cluster_z_min),
                "z_max": max(self.synthetic_cluster_z_max, self.synthetic_cluster_z_min + 0.1),
            }
            points = self._cluster_sample_points(x, y, z, radius)
            clusters.append({
                "id": cluster_id,
                "source": self.emergency_wall_source_tag if self.emergency_path_wall_enabled else "phone_sim2real_image_cluster",
                "phone_object_id": obj.get("object_id", ""),
                "class_name": cls,
                "confidence": float(obj.get("confidence", 0.0)),
                "count": max(1, int(self.synthetic_cluster_count_hint)),
                "centroid": {"x": x, "y": y, "z": z},
                "centroid_raw": {"x": x, "y": z, "z": y},
                "bbox": bbox,
                "bbox_raw": {
                    "x_min": bbox["x_min"], "x_max": bbox["x_max"],
                    "y_min": bbox["z_min"], "y_max": bbox["z_max"],
                    "z_min": bbox["y_min"], "z_max": bbox["y_max"],
                },
                "nearest_tank_distance_m": math.hypot(x - px, y - py),
                "distance_m": float(obj.get("distance_m", math.hypot(x - px, y - py))),
                "bearing_deg": float(obj.get("bearing_deg", 0.0)),
                "is_phone_synthetic": True,
                "is_phone_emergency_wall": bool(self.emergency_path_wall_enabled),
                "wall_offset_m": float(wall_offset_m),
                "points_map": points,
            })

        next_id = self.synthetic_cluster_id_offset
        for obj in source_objects:
            x0 = float(obj.get("emergency_wall_center_x", obj.get("map_x", 0.0)))
            y0 = float(obj.get("emergency_wall_center_y", obj.get("map_y", 0.0)))
            z0 = float(obj.get("map_z", 1.0))
            cls = str(obj.get("class_name", "unknown"))
            base_radius = clamp(
                float(obj.get("avoidance_radius_m", self.class_radius_m.get(cls, 2.0))) * self.synthetic_cluster_radius_scale,
                self.synthetic_cluster_min_radius_m,
                self.synthetic_cluster_max_radius_m,
            )

            if self.emergency_path_wall_enabled:
                # Build a wall perpendicular to the current path/base yaw.  If base_yaw is unavailable,
                # fall back to the vector from tank to obstacle.
                try:
                    yaw = math.radians(float(obj.get("emergency_wall_yaw_deg", obj.get("base_yaw_deg"))))
                except Exception:
                    yaw = math.atan2(y0 - py, x0 - px)
                nx = -math.sin(yaw)
                ny = math.cos(yaw)
                n = max(1, int(self.emergency_wall_cluster_count))
                radius = clamp(
                    float(self.emergency_wall_radius_m),
                    self.synthetic_cluster_min_radius_m,
                    self.synthetic_cluster_max_radius_m,
                )
                if n == 1:
                    offsets = [0.0]
                else:
                    half = float(self.emergency_wall_half_width_m)
                    offsets = [-half + 2.0 * half * i / float(n - 1) for i in range(n)]
                for off in offsets:
                    append_cluster(obj, next_id, x0 + nx * off, y0 + ny * off, z0, radius, wall_offset_m=off)
                    next_id += 1
            else:
                append_cluster(obj, next_id, x0, y0, z0, base_radius, wall_offset_m=0.0)
                next_id += 1

        return {
            "timestamp_ros_sec": self.get_clock().now().nanoseconds * 1e-9,
            "timestamp_wall": now,
            "frame_id": self.map_frame,
            "algorithm": "phone_sim2real_emergency_path_wall" if self.emergency_path_wall_enabled else "phone_sim2real_image_only_bbox_depth",
            "source": "phone_sim2real",
            "input_point_count": sum(int(c.get("count", 0)) for c in clusters),
            "noise_count": 0,
            "cluster_count": len(clusters),
            "phone_object_count": len(objects),
            "emergency_path_wall_enabled": bool(self.emergency_path_wall_enabled),
            "emergency_wall_cluster_count": int(self.emergency_wall_cluster_count),
            "emergency_wall_center_mode": str(self.emergency_wall_center_mode),
            "cluster_point_sample_limit": 8,
            "clusters": clusters,
        }

    def _cluster_sample_points(self, x: float, y: float, z: float, radius: float) -> List[Dict[str, float]]:
        r = max(0.1, radius)
        return [
            {"x": x, "y": y, "z": z},
            {"x": x + r, "y": y, "z": z},
            {"x": x - r, "y": y, "z": z},
            {"x": x, "y": y + r, "z": z},
            {"x": x, "y": y - r, "z": z},
            {"x": x + 0.7 * r, "y": y + 0.7 * r, "z": z},
            {"x": x - 0.7 * r, "y": y + 0.7 * r, "z": z},
            {"x": x + 0.7 * r, "y": y - 0.7 * r, "z": z},
        ]

    def _publish_json(self, publisher: Any, payload: Dict[str, Any]) -> None:
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        publisher.publish(msg)

    def _publish_status(self, state: str, detection_payload: Dict[str, Any], converted: List[Dict[str, Any]]) -> None:
        payload = {
            "timestamp_wall": time.time(),
            "state": state,
            "mode": "image_only_no_apf",
            "source": "phone_sim2real_virtual_obstacle_node",
            "inject_enabled": self.inject_enabled_state,
            "active_count": len(self.active),
            "converted_count": len(converted),
            "published_count": len(self._published_objects()),
            "incoming_detection_count": detection_payload.get("count", None),
            "last_detection_wall": self.last_detection_wall,
            "distance_mode": self.distance_mode,
            "placement_distance_scale": self.placement_distance_scale,
            "placement_distance_bias_m": self.placement_distance_bias_m,
            "min_placement_distance_m": self.min_placement_distance_m,
            "max_placement_distance_m": self.max_placement_distance_m,
            "bearing_reference_mode": self.bearing_reference_mode,
            "latest_lookahead_age_sec": (time.time() - self.latest_lookahead_wall) if self.lookahead_pose is not None else None,
            "latest_turret_age_sec": (time.time() - self.latest_turret_wall) if self.latest_turret_yaw_rad is not None else None,
            "latest_turret_yaw_deg": math.degrees(self.latest_turret_yaw_rad) if self.latest_turret_yaw_rad is not None else None,
            "turret_vector_yaw_mode": self.turret_vector_yaw_mode,
            "obstacle_anchor_mode": self.obstacle_anchor_mode,
            "lock_after_observations": self.lock_after_observations,
            "locked_count": sum(1 for o in self.active.values() if bool(o.get("is_locked", False))),
            "cluster_publish_count": len(self._cluster_objects(self._published_objects())),
            "synthetic_lidar_clusters_enabled": self.enable_synthetic_lidar_clusters,
            "emergency_path_wall_enabled": bool(self.emergency_path_wall_enabled),
            "emergency_wall_cluster_count": int(self.emergency_wall_cluster_count),
            "emergency_wall_half_width_m": float(self.emergency_wall_half_width_m),
            "emergency_wall_radius_m": float(self.emergency_wall_radius_m),
            "synthetic_lidar_clusters_topic": str(self.get_parameter("synthetic_lidar_clusters_topic").value),
            "discovered_objects_topic": str(self.get_parameter("discovered_objects_topic").value),
        }
        self._publish_json(self.pub_status, payload)

    def _build_markers(self, objects: List[Dict[str, Any]], now: float, namespace: str) -> MarkerArray:
        arr = MarkerArray()
        clear = Marker()
        clear.header.frame_id = self.map_frame
        clear.header.stamp = self.get_clock().now().to_msg()
        clear.ns = namespace
        clear.id = 0
        clear.action = Marker.DELETEALL
        arr.markers.append(clear)

        marker_id = 1
        for obj in [_phone_sim2real_freeze_locked_object(o) for o in objects]:
            x = float(obj.get("map_x", 0.0))
            y = float(obj.get("map_y", 0.0))
            z = float(obj.get("map_z", 1.0))
            cls = str(obj.get("class_name", "unknown"))
            radius = float(obj.get("avoidance_radius_m", 2.0))
            r, g, b, a = color_for_class(cls, 0.45)

            sphere = Marker()
            sphere.header.frame_id = self.map_frame
            sphere.header.stamp = self.get_clock().now().to_msg()
            sphere.ns = namespace + "_sphere"
            sphere.id = marker_id
            marker_id += 1
            sphere.type = Marker.SPHERE
            sphere.action = Marker.ADD
            sphere.pose.position = Point(x=x, y=y, z=max(0.8, z))
            sphere.pose.orientation.w = 1.0
            sphere.scale.x = radius * 2.0
            sphere.scale.y = radius * 2.0
            sphere.scale.z = 1.2
            sphere.color.r = r
            sphere.color.g = g
            sphere.color.b = b
            sphere.color.a = a
            arr.markers.append(sphere)

            text = Marker()
            text.header.frame_id = self.map_frame
            text.header.stamp = self.get_clock().now().to_msg()
            text.ns = namespace + "_label"
            text.id = marker_id
            marker_id += 1
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position = Point(x=x, y=y, z=max(2.0, z + 1.2))
            text.pose.orientation.w = 1.0
            text.scale.z = 0.9
            text.color.r = 1.0
            text.color.g = 1.0
            text.color.b = 1.0
            text.color.a = 1.0
            lock_tag = "LOCKED" if bool(obj.get("is_locked", False)) else "LIVE"
            text.text = (
                f"PHONE {cls} [{lock_tag}]\n"
                f"D={float(obj.get('distance_m', 0.0)):.1f}m "
                f"bboxH={float(obj.get('bbox_size', {}).get('height_px', 0.0)):.0f}px\n"
                f"obs={int(obj.get('observation_count', 0))} {obj.get('lock_reason', '')}"
            )
            arr.markers.append(text)

            if self.line_to_vehicle and self.player_pose is not None:
                line = Marker()
                line.header.frame_id = self.map_frame
                line.header.stamp = self.get_clock().now().to_msg()
                line.ns = namespace + "_line"
                line.id = marker_id
                marker_id += 1
                line.type = Marker.LINE_STRIP
                line.action = Marker.ADD
                line.scale.x = 0.12
                line.color.r = r
                line.color.g = g
                line.color.b = b
                line.color.a = 0.85
                px = float(self.player_pose.pose.position.x)
                py = float(self.player_pose.pose.position.y)
                pz = float(self.player_pose.pose.position.z)
                line.points = [Point(x=px, y=py, z=max(0.4, pz + 0.3)), Point(x=x, y=y, z=max(0.8, z))]
                arr.markers.append(line)
        return arr

    def _build_cluster_markers(self, objects: List[Dict[str, Any]], now: float) -> MarkerArray:
        arr = MarkerArray()
        clear = Marker()
        clear.header.frame_id = self.map_frame
        clear.header.stamp = self.get_clock().now().to_msg()
        clear.ns = "phone_image_cluster"
        clear.id = 0
        clear.action = Marker.DELETEALL
        arr.markers.append(clear)
        marker_id = 1
        for obj in [_phone_sim2real_freeze_locked_object(o) for o in objects]:
            x = float(obj.get("map_x", 0.0))
            y = float(obj.get("map_y", 0.0))
            cls = str(obj.get("class_name", "unknown"))
            radius = clamp(
                float(obj.get("avoidance_radius_m", self.class_radius_m.get(cls, 2.0))) * self.synthetic_cluster_radius_scale,
                self.synthetic_cluster_min_radius_m,
                self.synthetic_cluster_max_radius_m,
            )
            r, g, b, a = color_for_class(cls, 0.25)
            cube = Marker()
            cube.header.frame_id = self.map_frame
            cube.header.stamp = self.get_clock().now().to_msg()
            cube.ns = "phone_image_cluster_bbox"
            cube.id = marker_id
            marker_id += 1
            cube.type = Marker.CUBE
            cube.action = Marker.ADD
            cube.pose.position = Point(x=x, y=y, z=(self.synthetic_cluster_z_min + self.synthetic_cluster_z_max) * 0.5)
            cube.pose.orientation.w = 1.0
            cube.scale.x = radius * 2.0
            cube.scale.y = radius * 2.0
            cube.scale.z = max(0.2, self.synthetic_cluster_z_max - self.synthetic_cluster_z_min)
            cube.color.r = r
            cube.color.g = g
            cube.color.b = b
            cube.color.a = a
            arr.markers.append(cube)
        return arr


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PhoneVirtualObstacleNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
