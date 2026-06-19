# -*- coding: utf-8 -*-
"""
YOLO + LiDAR 캘리브레이션 융합을 위한 로컬 경로 / 로컬 매핑 노드.

패키지 역할 원칙:
- vision: 객체 class + bbox 제공
- lidar: LiDAR raw schema 해석 및 map 좌표 변환
- tank_visual_perception: camera-LiDAR projection 캘리브레이션 수식 + overlay/cluster 노드
- path_planning/local_path_node: 객체의 map 위치 추정과 discovered map 갱신

수정 사항 (Robust Version):
1. 스레드 안전성(Thread Safety) 확보를 위한 Lock 적용
2. 카메라-라이다 Time Synchronization 강제 동기화 (고스트 현상 방지)
3. Discovered Map 메모리 누수 방지 (Decay 로직 추가)
4. 카메라 투영용 /info는 compact topic 사용, LiDAR hit는 PointCloud2 사용
"""

from __future__ import annotations

import json
import math
import time
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ament_index_python.packages import get_package_share_directory
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Vector3Stamped, Point
from nav_msgs.msg import Path as NavPath
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import ColorRGBA, String
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker, MarkerArray

# [추가] PointCloud2 및 변환 라이브러리
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2

from path_planning.recon_logger import ReconLogger

try:
    import yaml
except Exception:
    yaml = None


from tank_common.pointcloud import pointcloud2_to_xyz_array

SERVICE_TERRAIN_FINALIZE = "/tank/terrain/finalize_map"
TOPIC_FUSION_DEBUG_STATUS = "/tank/debug/fusion/status"

from lidar.config import TOPIC_LIDAR_DETECTED_MAP
from path_planning.config import (
    CAMERA_LIDAR_PROJECTION_PARAMS,
    CLASS_COLOR_DEFAULTS,
    LOCAL_PATH_TIMER_SEC,
    MAP_FRAME,
    SERVICE_DISCOVERED_CLEAR,
    SERVICE_DISCOVERED_SAVE,
    TOPIC_DETECTIONS,
    TOPIC_DISCOVERED_OBJECT_MARKERS,
    TOPIC_DISCOVERED_OBJECTS,
    TOPIC_FUSED_OBJECT_MARKERS,
    TOPIC_FUSED_OBJECTS,
    TOPIC_INFO_COMPACT,
    TOPIC_LIDAR_CLUSTERS,
    TOPIC_PLAYER_POSE,
    TOPIC_PLAYER_STATE,
    TOPIC_RECON_RAW,
    TOPIC_TURRET,
)
from tank_visual_perception.projection import (
    compute_camera_pose,
    extract_info_payload,
    map_to_raw_xyz,
    point_inside_bbox,
    project_point,
    to_float,
    vec3_from_dict,
)


@dataclass
class ParsedDetection:
    class_name: str
    confidence: float
    bbox: List[float]
    track_id: Optional[int] = None
    class_fixed_id: Optional[int] = None
    center: Optional[List[float]] = None

    def metadata(self) -> Dict[str, Any]:
        return {
            "trackId": self.track_id,
            "classFixedId": self.class_fixed_id,
            "center": self.center,
            "bbox_center": self.center,
        }


@dataclass
class StaticObject:
    prefab_name: str
    category: str
    map_x: float
    map_y: float
    map_z: float


@dataclass
class DiscoveredObject:
    object_id: str
    class_name: str
    map_x: float
    map_y: float
    map_z: float
    distance_m: float
    confidence: float
    observation_count: int
    first_seen_wall: float
    last_seen_wall: float
    source: str = "projection_yolo_lidar_fusion"
    track_id: Optional[int] = None
    class_fixed_id: Optional[int] = None
    class_votes: Dict[str, float] = field(default_factory=dict)
    is_confirmed: bool = False
    confirmed_wall: Optional[float] = None


class LocalPathNode(Node):
    def __init__(self) -> None:
        super().__init__("tank_local_path_node")

        # 1. Thread Lock 선언 (안전한 데이터 공유)
        self._lock = threading.Lock()

        pkg_share = Path(get_package_share_directory("path_planning"))
        default_config = pkg_share / "config" / "fusion_mapping.yaml"

        self.declare_parameter("config_file", str(default_config))
        self.config_file = Path(str(self.get_parameter("config_file").value)).expanduser()
        self.cfg = self._load_config(self.config_file)
        self.latest_lidar_points = np.empty((0, 3), dtype=np.float32)
        self.latest_lidar_ts = 0.0
        self.map_frame = str(self._cfg(["frame", "map_frame"], MAP_FRAME))
        self.hfov_deg = float(self._cfg(["camera", "horizontal_fov_deg"], 47.81061))
        self.default_image_width = int(self._cfg(["camera", "default_image_width"], 1920))
        self.default_image_height = int(self._cfg(["camera", "default_image_height"], 1057))
        self.heading_source = str(self._cfg(["camera", "heading_source"], "body"))

        self.fusion_method = str(self._cfg(["fusion", "method"], "projection_then_cluster_then_angle"))
        self.angle_gate_extra_deg = float(self._cfg(["fusion", "angle_gate_extra_deg"], 4.0))
        self.max_fusion_range_m = float(self._cfg(["fusion", "max_fusion_range_m"], 45.0))
        self.min_fusion_range_m = float(self._cfg(["fusion", "min_fusion_range_m"], 0.5))
        self.min_lidar_points = int(self._cfg(["fusion", "min_lidar_points_per_object"], 3))
        self.use_nearest_points = int(self._cfg(["fusion", "use_nearest_points"], 20))
        self.min_detection_conf = float(self._cfg(["fusion", "min_detection_confidence"], 0.20))
        self.allow_angle_fallback = bool(self._cfg(["fusion", "allow_angle_fallback"], True))

        # Strict semantic 융합 정책. semantic 객체는 YOLO detection과 LiDAR DBSCAN cluster가
        # 매칭될 때만 생성해야 한다.
        self.semantic_requires_cluster = bool(self._cfg(["fusion", "semantic_requires_cluster"], True))
        self.cluster_match_max_center_norm = float(self._cfg(["fusion", "cluster_match_max_center_norm"], 1.75))
        self.cluster_match_max_score = float(self._cfg(["fusion", "cluster_match_max_score"], 2.10))
        self.cluster_match_ambiguity_delta = float(self._cfg(["fusion", "cluster_match_ambiguity_delta"], 0.10))
        self.cluster_match_distance_weight = float(self._cfg(["fusion", "cluster_match_distance_weight"], 0.0015))
        self.cluster_match_bbox_area_weight = float(self._cfg(["fusion", "cluster_match_bbox_area_weight"], 0.45))
        self.cluster_match_person_anchor_y = float(self._cfg(["fusion", "cluster_match_person_anchor_y"], 0.90))
        self.cluster_match_default_anchor_y = float(self._cfg(["fusion", "cluster_match_default_anchor_y"], 0.50))
        self.cluster_match_person_x_limit = float(self._cfg(["fusion", "cluster_match_person_x_limit"], 1.80))
        self.cluster_match_person_y_limit = float(self._cfg(["fusion", "cluster_match_person_y_limit"], 2.80))

        self.use_projection_fusion = bool(self._cfg(["projection", "enabled"], True))
        self.projection_params = dict(CAMERA_LIDAR_PROJECTION_PARAMS)
        self.projection_params.update(dict(self._cfg(["projection", "params"], {}) or {}))
        self.projection_bbox_margin_px = float(self._cfg(["projection", "bbox_margin_px"], 8.0))
        self.min_projected_points = int(self._cfg(["projection", "min_projected_lidar_points"], 3))
        self.prefer_clusters = bool(self._cfg(["projection", "prefer_dbscan_clusters"], True))
        self.cluster_bbox_margin_px = float(self._cfg(["projection", "cluster_bbox_margin_px"], 18.0))
        self.min_cluster_points = int(self._cfg(["projection", "min_cluster_points"], 2))
        self.use_only_detected_projection_points = bool(self._cfg(["projection", "use_only_detected_points"], True))

        self.add_only_unmatched = bool(self._cfg(["static_matching", "add_only_unmatched_to_recon"], True))
        self.static_match_radius_m = float(self._cfg(["static_matching", "static_match_radius_m"], 4.0))
        self.same_category_only = bool(self._cfg(["static_matching", "same_category_only"], True))

        self.mapping_enabled = bool(self._cfg(["mapping", "enabled"], True))
        self.merge_radius_m = float(self._cfg(["mapping", "merge_radius_m"], 5.0))
        self.ema_alpha = float(self._cfg(["mapping", "position_ema_alpha"], 0.35))
        self.add_classes = set(str(x).lower() for x in self._cfg(["mapping", "add_classes"], ["person", "rock", "tank", "car", "house", "tent"]))
        self.merge_radius_by_class = dict(self._cfg(["mapping", "merge_radius_by_class"], {}) or {})
        self.save_directory = Path(str(self._cfg(["mapping", "save_directory"], "~/tank_discovered_maps"))).expanduser()
        self.save_latest_filename = str(self._cfg(["mapping", "save_latest_filename"], "discovered_objects_latest.map"))
        self.save_timestamped_copy = bool(self._cfg(["mapping", "save_timestamped_copy"], True))
        self.save_confirmed_only = bool(self._cfg(["mapping", "save_confirmed_only"], True))
        self.min_confirm_observations = int(self._cfg(["mapping", "min_confirm_observations"], 5))
        self.min_confirm_age_sec = float(self._cfg(["mapping", "min_confirm_age_sec"], 1.0))
        self.merge_across_classes = bool(self._cfg(["mapping", "merge_across_classes"], True))
        self.track_id_merge_enabled = bool(self._cfg(["mapping", "track_id_merge_enabled"], True))
        self.track_id_merge_radius_m = float(self._cfg(["mapping", "track_id_merge_radius_m"], 10.0))
        self.class_vote_by_confidence = bool(self._cfg(["mapping", "class_vote_by_confidence"], True))

        # [핵심 변경 1] 고스트 방지 및 메모리 누수 방지 파라미터
        self.drop_stale_async_detection = bool(self._cfg(["async_detection", "drop_stale"], True))
        self.max_async_result_age_ms = float(self._cfg(["async_detection", "max_result_age_ms"], 300.0))
        self.max_sync_diff_sec = float(self._cfg(["fusion", "max_sync_diff_sec"], 0.50))
        # 기본값은 drop이 아니라 warning이다. /detect와 /info/LiDAR는 서로 다른 HTTP route이므로
        # 0.15초 strict drop을 걸면 YOLO와 LiDAR가 모두 정상이어도 fused_objects가 0개가 될 수 있다.
        self.drop_on_sync_mismatch = bool(self._cfg(["fusion", "drop_on_sync_mismatch"], False))
        self.debug_fusion_enabled = bool(self._cfg(["debug", "fusion_status"], True))
        self.memory_decay_sec = float(self._cfg(["mapping", "memory_decay_sec"], 10.0))  # 10초간 안 보이면 메모리에서 삭제

        self.current_lifetime_sec = float(self._cfg(["rviz", "current_object_lifetime_sec"], 1.2))
        self.discovered_z_offset = float(self._cfg(["rviz", "discovered_marker_z_offset"], 1.5))
        self.current_z_offset = float(self._cfg(["rviz", "current_marker_z_offset"], 2.0))
        self.line_width = float(self._cfg(["rviz", "line_width"], 0.12))
        self.text_height = float(self._cfg(["rviz", "text_height"], 1.2))
        self.sphere_scale = float(self._cfg(["rviz", "sphere_scale"], 1.5))
        self.discovered_cube_scale = float(self._cfg(["rviz", "discovered_cube_scale"], 2.0))
        self.class_colors = dict(CLASS_COLOR_DEFAULTS)
        self.class_colors.update(dict(self._cfg(["rviz", "colors"], {}) or {}))

        self.latest_detections_payload: Optional[Dict[str, Any]] = None
        self.latest_lidar_payload: Optional[Dict[str, Any]] = None
        self.latest_info: Optional[Dict[str, Any]] = None
        self.latest_clusters_payload: Optional[Dict[str, Any]] = None
        self.player_pose: Optional[PoseStamped] = None
        self.player_heading_deg: float = 0.0
        self.turret_heading_deg: Optional[float] = None
        self.static_objects: List[StaticObject] = []
        self.discovered: List[DiscoveredObject] = []
        self.fused_current: List[Dict[str, Any]] = []
        self._next_id = 1
        self._last_fusion_debug: Optional[Dict[str, Any]] = None
        self._last_cluster_assignment_stats: Dict[str, Any] = {}

        transient_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.create_subscription(String, TOPIC_DETECTIONS, self.detections_cb, 10)
        self.create_subscription(PointCloud2, TOPIC_LIDAR_DETECTED_MAP, self.lidar_cb, 10)
        self.create_subscription(String, TOPIC_INFO_COMPACT, self.info_raw_cb, 10)
        self.create_subscription(String, TOPIC_LIDAR_CLUSTERS, self.lidar_clusters_cb, 10)
        self.create_subscription(PoseStamped, TOPIC_PLAYER_POSE, self.player_pose_cb, 10)
        self.create_subscription(String, TOPIC_PLAYER_STATE, self.player_state_cb, 10)
        self.create_subscription(Vector3Stamped, TOPIC_TURRET, self.turret_cb, 10)
        self.create_subscription(String, TOPIC_RECON_RAW, self.recon_raw_cb, transient_qos)

        self.fused_pub = self.create_publisher(String, TOPIC_FUSED_OBJECTS, 10)
        self.discovered_pub = self.create_publisher(String, TOPIC_DISCOVERED_OBJECTS, transient_qos)
        self.current_marker_pub = self.create_publisher(MarkerArray, TOPIC_FUSED_OBJECT_MARKERS, 10)
        self.discovered_marker_pub = self.create_publisher(MarkerArray, TOPIC_DISCOVERED_OBJECT_MARKERS, transient_qos)
        self.fusion_debug_pub = self.create_publisher(String, TOPIC_FUSION_DEBUG_STATUS, 10)

        self.declare_parameter("route_id", "A")
        self.declare_parameter("route_map_name", "finalmap")
        self.declare_parameter("recon_report_dir", "./recon_reports")
        self.declare_parameter("goal_pose_topic", "/tank/goal/pose")
        self.declare_parameter("goal_tolerance", 5.0)

        self.route_id = str(self.get_parameter("route_id").value)
        self.route_map_name = str(self.get_parameter("route_map_name").value)
        self.recon_report_dir = str(self.get_parameter("recon_report_dir").value)
        self.goal_pose_topic = str(self.get_parameter("goal_pose_topic").value)
        self.goal_tolerance = float(self.get_parameter("goal_tolerance").value)

        self.recon_logger = ReconLogger(self.route_id, self.route_map_name, self.recon_report_dir)
        self.sim_time = 0.0
        self._last_sim_time = 0.0
        self._report_saved = False
        self.goal_pos = None

        self.create_subscription(PoseStamped, self.goal_pose_topic, self.goal_pose_cb, 10)
        self.create_subscription(String, "/tank/event/collision", self.collision_cb, 10)

        # 주행 품질 진단용 구독 (읽기만 — 제어/계획 거동은 안 건드림). recon_logger에 per-step 기록해
        # 진동/끼임 원인이 경로 churn / APF 불일치 / 제어 채터 중 무엇인지 사후에 수치로 가린다.
        self.create_subscription(String, "/tank/planner/status", self._diag_planner_status_cb, 10)
        self.create_subscription(NavPath, "/tank/global_path", self._diag_global_path_cb, 10)
        self.create_subscription(PoseStamped, "/tank/path/lookahead_pose", self._diag_lookahead_cb, 10)
        self.create_subscription(PoseStamped, "/tank/local_target/pose", self._diag_local_target_cb, 10)
        self.create_subscription(String, "/tank/control/command", self._diag_command_cb, 10)

        self.create_service(Trigger, SERVICE_DISCOVERED_SAVE, self.save_service_cb)
        self.create_service(Trigger, SERVICE_DISCOVERED_CLEAR, self.clear_service_cb)
        self.terrain_finalize_client = self.create_client(Trigger, SERVICE_TERRAIN_FINALIZE)
        self.create_timer(LOCAL_PATH_TIMER_SEC, self.timer_cb)
        self.get_logger().info("local_path_node started (PC2 Optimized Version)")

    # ------------------------------------------------------------------
    # 설정(Configuration) 헬퍼
    # ------------------------------------------------------------------
    def _load_config(self, path: Path) -> Dict[str, Any]:
        if yaml is None:
            self.get_logger().warn("PyYAML unavailable; using built-in fusion defaults")
            return {}
        if not path.exists():
            self.get_logger().warn(f"fusion config not found: {path}; using defaults")
            return {}
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data if isinstance(data, dict) else {}

    def _cfg(self, keys: List[str], default: Any) -> Any:
        cur: Any = self.cfg
        for key in keys:
            if not isinstance(cur, dict) or key not in cur:
                return default
            cur = cur[key]
        return cur

    # ------------------------------------------------------------------
    # 콜백(Callbacks) — 스레드 안전(Thread Safe)
    # ------------------------------------------------------------------
    def detections_cb(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
            if isinstance(data, dict):
                with self._lock:
                    self.latest_detections_payload = data
        except Exception as exc:
            self.get_logger().debug(f"detections parse failed: {exc}")

    def lidar_cb(self, msg: PointCloud2) -> None:
        try:
            # PointCloud2를 NumPy 배열로 바로 변환
            points = pointcloud2_to_xyz_array(msg)
            stamp_sec = msg.header.stamp.sec + (msg.header.stamp.nanosec * 1e-9)
            with self._lock:
                self.latest_lidar_points = points
                self.latest_lidar_ts = stamp_sec
                self.latest_lidar_payload = {
                    "timestamp_wall": stamp_sec,
                    "timestamp_ros_sec": stamp_sec,
                    "frame_id": msg.header.frame_id or MAP_FRAME,
                    "count": int(points.shape[0]),
                }
        except Exception as exc:
            self.get_logger().debug(f"lidar pc2 parse failed: {exc}")

    def info_raw_cb(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
            data_part = payload.get("data", {})
            with self._lock:
                self.sim_time = float(data_part.get("time", self.sim_time))
            info = extract_info_payload(payload)
            if info is not None:
                with self._lock:
                    self.latest_info = info
        except Exception as exc:
            self.get_logger().debug(f"info compact parse failed: {exc}")

    def lidar_clusters_cb(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
            if isinstance(payload, dict):
                with self._lock:
                    self.latest_clusters_payload = payload
        except Exception as exc:
            self.get_logger().debug(f"cluster parse failed: {exc}")

    def player_pose_cb(self, msg: PoseStamped) -> None:
        with self._lock:
            if self.player_pose is not None:
                dx = msg.pose.position.x - self.player_pose.pose.position.x
                dy = msg.pose.position.y - self.player_pose.pose.position.y
                dist = math.hypot(dx, dy)
                if dist < 10.0:
                    self.recon_logger.total_distance += dist
            self.player_pose = msg
            # Recon 보고서 궤적 로깅: map x=position.x, map y/z=position.y, yaw=body heading.
            self.recon_logger.log_pose(
                self.sim_time,
                float(msg.pose.position.x),
                float(msg.pose.position.y),
                float(self.player_heading_deg),
            )
            # 진단 스냅샷(0.2s 간격, recon_logger 내부에서 시간 게이트). 끼임/제자리 진동도 포착.
            self.recon_logger.log_diag_sample(
                self.sim_time, float(msg.pose.position.x), float(msg.pose.position.y)
            )

    def goal_pose_cb(self, msg: PoseStamped) -> None:
        with self._lock:
            self.goal_pos = (float(msg.pose.position.x), float(msg.pose.position.y))

    def collision_cb(self, msg: String) -> None:
        with self._lock:
            self.recon_logger.collisions += 1

    # -- 주행 품질 진단 구독 콜백 (읽기만) ----------------------------------
    def _diag_planner_status_cb(self, msg: String) -> None:
        try:
            v = int(json.loads(msg.data).get("route_version", 0))
        except Exception:
            return
        with self._lock:
            self.recon_logger.set_route_version(v)

    def _diag_global_path_cb(self, msg: NavPath) -> None:
        try:
            path_xz = [(float(p.pose.position.x), float(p.pose.position.y)) for p in msg.poses]
        except Exception:
            return
        with self._lock:
            self.recon_logger.log_planned_path(self.sim_time, path_xz)

    def _diag_lookahead_cb(self, msg: PoseStamped) -> None:
        with self._lock:
            self.recon_logger.set_lookahead(float(msg.pose.position.x), float(msg.pose.position.y))

    def _diag_local_target_cb(self, msg: PoseStamped) -> None:
        with self._lock:
            self.recon_logger.set_local_target(float(msg.pose.position.x), float(msg.pose.position.y))

    def _diag_command_cb(self, msg: String) -> None:
        with self._lock:
            self.recon_logger.set_command(str(msg.data))

    def player_state_cb(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
            if not isinstance(data, dict):
                return
            body = data.get("body")
            with self._lock:
                if isinstance(body, dict):
                    if body.get("x") is not None:
                        self.player_heading_deg = float(body.get("x"))
                    pitch = float(body.get("y", 0.0))
                    roll = float(body.get("z", 0.0))
                    self.recon_logger.log_body_angles(pitch, roll)
                elif data.get("playerBodyX") is not None:
                    self.player_heading_deg = float(data.get("playerBodyX"))
        except Exception:
            pass

    def turret_cb(self, msg: Vector3Stamped) -> None:
        try:
            with self._lock:
                self.turret_heading_deg = float(msg.vector.x)
        except Exception:
            with self._lock:
                self.turret_heading_deg = None

    def recon_raw_cb(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
            obstacles = payload.get("obstacles", []) if isinstance(payload, dict) else []
            parsed: List[StaticObject] = []
            for obs in obstacles:
                if not isinstance(obs, dict):
                    continue
                prefab = str(obs.get("prefabName", "unknown"))
                pos = obs.get("position", {}) if isinstance(obs.get("position"), dict) else {}
                raw_x = self._as_float(pos.get("x"), 0.0)
                raw_y = self._as_float(pos.get("y"), 0.0)
                raw_z = self._as_float(pos.get("z"), 0.0)
                parsed.append(
                    StaticObject(
                        prefab_name=prefab,
                        category=self._category_for_prefab(prefab),
                        map_x=raw_x,
                        map_y=raw_z,
                        map_z=raw_y,
                    )
                )
            with self._lock:
                self.static_objects = parsed
            self.get_logger().info(f"Loaded recon static obstacles for matching: {len(parsed)}")
        except Exception as exc:
            self.get_logger().warn(f"Failed to parse /tank/map/recon/raw: {exc}")

    # ------------------------------------------------------------------
    # 융합(Fusion) — Time Sync & Decay 로직 추가
    # ------------------------------------------------------------------
    def timer_cb(self) -> None:
        with self._lock:
            fused = self.compute_fused_objects_locked()
            self.fused_current = fused
            if fused:
                self.update_discovered_map_locked(fused)

            now = time.time()
            self.discovered = [
                obj for obj in self.discovered
                if obj.is_confirmed or (now - obj.last_seen_wall) < self.memory_decay_sec
            ]

            # YOLO detections 로깅
            if self.latest_detections_payload:
                detections = self.latest_detections_payload.get("detections", [])
                if isinstance(detections, list):
                    for det in detections:
                        class_name = det.get("className", "unknown")
                        conf = det.get("confidence", 0.0)
                        bbox = det.get("bbox", [])
                        turret_x = self.turret_heading_deg if self.turret_heading_deg is not None else 0.0
                        self.recon_logger.log_vision(self.sim_time, class_name, conf, bbox, turret_x)

            # Lidar clusters 로깅
            if self.latest_clusters_payload:
                clusters = self.latest_clusters_payload.get("clusters", [])
                if isinstance(clusters, list):
                    for c in clusters:
                        centroid = c.get("centroid", {})
                        bbox = c.get("bbox", {})
                        bbox_list = [bbox.get("x_min", 0.0), bbox.get("x_max", 0.0), bbox.get("y_min", 0.0), bbox.get("y_max", 0.0)]
                        self.recon_logger.log_obstacle(
                            self.sim_time,
                            centroid.get("x", 0.0),
                            centroid.get("y", 0.0),
                            bbox_list
                        )

            # Spotted assets 로깅
            for obj in self.discovered:
                if obj.is_confirmed:
                    if obj.class_name == "person":
                        self.recon_logger.log_spotted_asset("soldiers", obj.object_id)
                    elif obj.class_name == "tank":
                        self.recon_logger.log_spotted_asset("tanks", obj.object_id)
                    elif obj.class_name in ("tent", "house", "outpost"):
                        self.recon_logger.log_spotted_asset("outposts", obj.object_id)

            self.recon_logger.total_sim_time = self.sim_time

            # 1) Restart 감지 — 진짜 시뮬 재시작(클럭이 0 근처로 리셋)일 때만 로거를 폐기한다.
            #    주행 중 클럭 지터(sim_time이 여전히 높은데 ±수초 되감김; /tank/info 혼선/네트워크
            #    재정렬에서 발생)에 통째 폐기하면 루트 첫 레그가 통째로 날아간다. 그래서 'backward>2s'에
            #    더해 'sim_time이 리셋 임계 미만(=0 근처)'을 AND 조건으로 둔다.
            _RESTART_RESET_ABS_S = 5.0
            if self.sim_time < self._last_sim_time - 2.0 and self.sim_time < _RESTART_RESET_ABS_S:
                if not self._report_saved:
                    self.recon_logger.save_report()
                self.recon_logger = ReconLogger(self.route_id, self.route_map_name, self.recon_report_dir)
                self._report_saved = False
            self._last_sim_time = self.sim_time

            # 2) 도달 감지
            if self.player_pose and self.goal_pos:
                px = self.player_pose.pose.position.x
                py = self.player_pose.pose.position.y
                dist = math.hypot(px - self.goal_pos[0], py - self.goal_pos[1])
                if dist < self.goal_tolerance:
                    if not self._report_saved:
                        self.recon_logger.reached = True
                        self.recon_logger.save_report()
                        self._report_saved = True

            # 퍼블리싱 전 안전하게 복사
            fused_to_pub = list(fused)
            discovered_to_pub = list(self.discovered)
            fusion_debug_to_pub = dict(self._last_fusion_debug) if self._last_fusion_debug else None

        self.publish_fused(fused_to_pub)
        self.publish_current_markers(fused_to_pub)
        self.publish_discovered(discovered_to_pub)
        self.publish_fusion_debug(fusion_debug_to_pub)

    def compute_fused_objects_locked(self) -> List[Dict[str, Any]]:
        debug = self._make_fusion_debug_base_locked()

        def finish(reason: str, fused: Optional[List[Dict[str, Any]]] = None, **extra: Any) -> List[Dict[str, Any]]:
            result = fused or []
            if self.debug_fusion_enabled:
                debug.update(extra)
                debug["reject_reason"] = reason
                debug["fused_count"] = len(result)
                debug["success"] = len(result) > 0
                self._last_fusion_debug = debug
            return result

        if self.player_pose is None:
            return finish("no_player_pose")
        if self.latest_detections_payload is None:
            return finish("no_detection_payload")
        if self.latest_lidar_points.shape[0] == 0:
            return finish("no_lidar_points")

        raw_detections = self.latest_detections_payload.get("detections", [])
        if not isinstance(raw_detections, list):
            return finish("detections_not_list")

        if self._is_stale_async_detection_payload(self.latest_detections_payload):
            return finish(
                "stale_async_detection",
                async_result_age_ms=self._as_float(self.latest_detections_payload.get("resultAgeMs"), -1.0),
                max_async_result_age_ms=self.max_async_result_age_ms,
            )

        # 카메라/LiDAR time synchronization.
        # Fusion이 DBSCAN cluster를 우선 사용하므로, 가능하면 detection timestamp와 cluster timestamp를 비교한다.
        # 단, 기본값은 hard drop이 아니라 warning이다. strict drop은 config에서 drop_on_sync_mismatch=true일 때만 수행한다.
        det_ts = self._as_float(self.latest_detections_payload.get("timestamp_wall", 0.0))
        lidar_ts = self._as_float(self.latest_lidar_ts, 0.0)
        cluster_ts = self._extract_cluster_timestamp(self.latest_clusters_payload)
        sync_ref_ts = cluster_ts if self.prefer_clusters and cluster_ts > 0.0 else lidar_ts
        sync_ref_name = "cluster" if self.prefer_clusters and cluster_ts > 0.0 else "lidar_pc2"
        sync_diff_sec = abs(det_ts - sync_ref_ts) if det_ts > 0 and sync_ref_ts > 0 else -1.0
        debug.update({
            "det_ts": det_ts,
            "lidar_ts": lidar_ts,
            "cluster_ts": cluster_ts,
            "sync_ref": sync_ref_name,
            "sync_diff_sec": sync_diff_sec,
            "max_sync_diff_sec": self.max_sync_diff_sec,
            "sync_warning": bool(sync_diff_sec >= 0.0 and sync_diff_sec > self.max_sync_diff_sec),
            "drop_on_sync_mismatch": self.drop_on_sync_mismatch,
        })
        if sync_diff_sec >= 0.0 and sync_diff_sec > self.max_sync_diff_sec and self.drop_on_sync_mismatch:
            return finish("sync_mismatch_drop")

        parsed_detections: List[ParsedDetection] = []
        dropped_low_conf = 0
        dropped_bad_bbox = 0
        for det in raw_detections:
            parsed = self._parse_detection(det)
            if parsed is not None:
                parsed_detections.append(parsed)
            elif isinstance(det, dict):
                conf = self._as_float(det.get("confidence"), 0.0)
                if conf < self.min_detection_conf:
                    dropped_low_conf += 1
                else:
                    dropped_bad_bbox += 1
        debug.update({
            "raw_detection_count": len(raw_detections),
            "parsed_detection_count": len(parsed_detections),
            "dropped_low_conf": dropped_low_conf,
            "dropped_bad_bbox": dropped_bad_bbox,
            "min_detection_conf": self.min_detection_conf,
        })
        if not parsed_detections:
            return finish("no_parsed_detection")

        image_w, image_h = self._extract_image_size(self.latest_detections_payload)
        px = float(self.player_pose.pose.position.x)
        py = float(self.player_pose.pose.position.y)
        pz = float(self.player_pose.pose.position.z)
        camera_heading = self._camera_heading_deg()

        projection_context = self._build_projection_context(int(image_w), int(image_h)) if self.use_projection_fusion else None
        projected_clusters = self._project_clusters(projection_context, px, py, int(image_w), int(image_h)) if projection_context else []
        debug.update({
            "image_w": float(image_w),
            "image_h": float(image_h),
            "projection_enabled": bool(self.use_projection_fusion),
            "projection_context_ok": projection_context is not None,
            "projected_cluster_count": len(projected_clusters),
            "semantic_requires_cluster": self.semantic_requires_cluster,
            "prefer_clusters": self.prefer_clusters,
            "allow_angle_fallback": self.allow_angle_fallback,
        })

        # 우선 경로: 전역 일대일(global one-to-one) YOLO bbox <-> DBSCAN cluster 할당.
        if self.prefer_clusters and projected_clusters:
            fused = self._fuse_with_global_cluster_assignment(
                parsed_detections,
                projected_clusters,
                px,
                py,
                pz,
                camera_heading,
                float(image_w),
                float(image_h),
            )
            debug.update(self._last_cluster_assignment_stats)
            if fused:
                return finish("ok_projection_cluster", fused)

        # Strict 모드: YOLO와 DBSCAN cluster가 일치하지 않으면 semantic 객체를 내보내지 않는다.
        if self.semantic_requires_cluster:
            if projection_context is None:
                return finish("strict_no_projection_context")
            if not projected_clusters:
                return finish("strict_no_projected_cluster")
            return finish("strict_no_cluster_assignment")

        # Legacy fallback 경로 — semantic_requires_cluster=false일 때 긴급 디버깅용으로만 유지한다.
        angle_points = self._build_angle_lidar_points(px, py) if self.allow_angle_fallback else []
        projected_points = self._project_raw_lidar_points(projection_context, px, py, int(image_w), int(image_h)) if projection_context else []
        debug.update({
            "projected_point_count": len(projected_points),
            "angle_point_count": len(angle_points),
        })

        fused: List[Dict[str, Any]] = []
        assigned_cluster_ids: set[int] = set()
        for parsed in parsed_detections:
            class_name, confidence, bbox = parsed.class_name, parsed.confidence, parsed.bbox
            obj = None
            if self.prefer_clusters and projected_clusters:
                obj = self._fuse_from_projected_clusters(
                    class_name, confidence, bbox, projected_clusters, assigned_cluster_ids,
                    px, py, pz, camera_heading, image_w, image_h
                )
            if obj is None and projected_points:
                obj = self._fuse_from_projected_points(class_name, confidence, bbox, projected_points, px, py, pz, camera_heading, image_w, image_h)
            if obj is None and self.allow_angle_fallback and angle_points:
                obj = self._fuse_from_angle(class_name, confidence, bbox, angle_points, px, py, pz, camera_heading, image_w)
            if obj is not None:
                obj.update(parsed.metadata())
                fused.append(obj)
        if fused:
            return finish("ok_fallback_path", fused)
        return finish("fallback_no_match")

    def update_discovered_map_locked(self, fused: List[Dict[str, Any]]) -> None:
        if not self.mapping_enabled:
            return
        now = time.time()
        for obj in fused:
            class_name = str(obj.get("className", "unknown")).lower()
            if class_name not in self.add_classes:
                continue
            if self.semantic_requires_cluster and not bool(obj.get("discovered_eligible", False)):
                continue
            if self.semantic_requires_cluster and str(obj.get("lidar_match_type", "")) != "dbscan_cluster":
                continue
            if self.add_only_unmatched and bool(obj.get("known_static")):
                continue
            pos = obj.get("position_map") if isinstance(obj.get("position_map"), dict) else {}
            x = self._as_float(pos.get("x"), 0.0)
            y = self._as_float(pos.get("y"), 0.0)
            z = self._as_float(pos.get("z"), 0.0)
            conf = self._as_float(obj.get("confidence"), 0.0)
            distance_m = self._as_float(obj.get("distance_m"), 0.0)
            track_id = self._as_optional_int(obj.get("trackId", obj.get("track_id")))
            class_fixed_id = self._as_optional_int(obj.get("classFixedId", obj.get("class_fixed_id")))
            
            existing = self._find_existing_discovered(class_name, x, y, track_id=track_id)
            vote_weight = max(conf, 1e-6) if self.class_vote_by_confidence else 1.0
            
            if existing is None:
                object_id = f"detected_{class_name}_{self._next_id:04d}"
                self._next_id += 1
                candidate = DiscoveredObject(
                    object_id=object_id,
                    class_name=class_name,
                    map_x=x,
                    map_y=y,
                    map_z=z,
                    distance_m=distance_m,
                    confidence=conf,
                    observation_count=1,
                    first_seen_wall=now,
                    last_seen_wall=now,
                    source=str(obj.get("source", "projection_yolo_lidar_fusion")),
                    track_id=track_id,
                    class_fixed_id=class_fixed_id,
                    class_votes={class_name: vote_weight},
                )
                self._refresh_confirmation(candidate, now)
                self.discovered.append(candidate)
            else:
                a = max(0.0, min(1.0, self.ema_alpha))
                existing.map_x = (1.0 - a) * existing.map_x + a * x
                existing.map_y = (1.0 - a) * existing.map_y + a * y
                existing.map_z = (1.0 - a) * existing.map_z + a * z
                existing.distance_m = distance_m
                existing.confidence = max(existing.confidence, conf)
                existing.observation_count += 1
                existing.last_seen_wall = now
                existing.source = str(obj.get("source", existing.source))
                if existing.track_id is None and track_id is not None:
                    existing.track_id = track_id
                if class_fixed_id is not None:
                    existing.class_fixed_id = class_fixed_id
                existing.class_votes[class_name] = existing.class_votes.get(class_name, 0.0) + vote_weight
                existing.class_name = max(existing.class_votes.items(), key=lambda kv: kv[1])[0]
                self._refresh_confirmation(existing, now)

    def _parse_detection(self, det: Dict[str, Any]) -> Optional[ParsedDetection]:
        if not isinstance(det, dict):
            return None
        class_name = str(det.get("className", det.get("class_name", "unknown"))).strip().lower()
        confidence = self._as_float(det.get("confidence"), 0.0)
        if confidence < self.min_detection_conf:
            return None
        bbox = det.get("bbox")
        if not isinstance(bbox, list) or len(bbox) < 4:
            return None
        try:
            x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
        except Exception:
            return None
        if x2 <= x1 or y2 <= y1:
            return None
        track_id = self._as_optional_int(det.get("trackId", det.get("track_id")))
        class_fixed_id = self._as_optional_int(det.get("classFixedId", det.get("class_fixed_id", det.get("id"))))
        center = det.get("center")
        if isinstance(center, list) and len(center) >= 2:
            parsed_center = [self._as_float(center[0], 0.5 * (x1 + x2)), self._as_float(center[1], 0.5 * (y1 + y2))]
        else:
            parsed_center = [0.5 * (x1 + x2), 0.5 * (y1 + y2)]
        return ParsedDetection(
            class_name=class_name,
            confidence=confidence,
            bbox=[x1, y1, x2, y2],
            track_id=track_id,
            class_fixed_id=class_fixed_id,
            center=parsed_center,
        )

    def _build_projection_context(self, image_w: int, image_h: int) -> Optional[Dict[str, Any]]:
        if self.latest_info is None:
            return None
        try:
            cam_pos, cam_yaw, cam_pitch, cam_roll = compute_camera_pose(self.latest_info, self.projection_params)
            return {
                "camera_pos": cam_pos,
                "camera_yaw": cam_yaw,
                "camera_pitch": cam_pitch,
                "camera_roll": cam_roll,
                "image_w": image_w,
                "image_h": image_h,
            }
        except Exception as exc:
            self.get_logger().debug(f"projection context failed: {exc}")
            return None

    def _project_raw_lidar_points(self, ctx: Dict[str, Any], px: float, py: float, image_w: int, image_h: int) -> List[Dict[str, Any]]:
        points = self.latest_lidar_points
        if points.shape[0] == 0:
            return []
        out: List[Dict[str, Any]] = []
        for x, y, z in points:
            x = float(x)
            y = float(y)
            z = float(z)
            distance = math.hypot(x - px, y - py)
            if distance < self.min_fusion_range_m or distance > self.max_fusion_range_m:
                continue
            # projection 유틸리티는 Unity raw xyz를 쓴다. PC2는 map xyz이므로 다시 변환한다.
            map_pos = {"x": x, "y": y, "z": z}
            raw_pos = map_to_raw_xyz(map_pos)
            projected = project_point(
                vec3_from_dict(raw_pos),
                ctx["camera_pos"],
                ctx["camera_yaw"],
                ctx["camera_pitch"],
                ctx["camera_roll"],
                image_w,
                image_h,
                self.projection_params,
            )
            if projected is None:
                continue
            u, v, depth = projected
            if not (0 <= u < image_w and 0 <= v < image_h):
                continue
            out.append({"x": x, "y": y, "z": z, "u": u, "v": v, "depth": depth, "distance_m": distance, "raw": map_pos})
        return out

    def _project_clusters(self, ctx: Dict[str, Any], px: float, py: float, image_w: int, image_h: int) -> List[Dict[str, Any]]:
        payload = self.latest_clusters_payload
        if not isinstance(payload, dict):
            return []
        clusters = payload.get("clusters", [])
        if not isinstance(clusters, list):
            return []
        out: List[Dict[str, Any]] = []
        for c in clusters:
            if not isinstance(c, dict):
                continue
            count = int(self._as_float(c.get("count"), 0.0))
            if count < self.min_cluster_points:
                continue
            centroid = c.get("centroid") if isinstance(c.get("centroid"), dict) else None
            if centroid is None:
                continue
            map_pos = {"x": self._as_float(centroid.get("x")), "y": self._as_float(centroid.get("y")), "z": self._as_float(centroid.get("z"))}
            raw_pos = c.get("centroid_raw") if isinstance(c.get("centroid_raw"), dict) else map_to_raw_xyz(map_pos)
            projected = project_point(
                vec3_from_dict(raw_pos),
                ctx["camera_pos"],
                ctx["camera_yaw"],
                ctx["camera_pitch"],
                ctx["camera_roll"],
                image_w,
                image_h,
                self.projection_params,
            )
            if projected is None:
                continue
            u, v, depth = projected
            if not (0 <= u < image_w and 0 <= v < image_h):
                continue
            distance = math.hypot(map_pos["x"] - px, map_pos["y"] - py)
            if distance < self.min_fusion_range_m or distance > self.max_fusion_range_m:
                continue
            out.append({"id": int(self._as_float(c.get("id"), -1)), "count": count, "centroid": map_pos, "bbox": c.get("bbox"), "u": u, "v": v, "depth": depth, "distance_m": distance, "raw": c})
        return out

    def _bbox_anchor_uv(self, bbox: List[float], class_name: str) -> Tuple[float, float, float, float]:
        x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)
        anchor_y_ratio = self.cluster_match_person_anchor_y if class_name == "person" else self.cluster_match_default_anchor_y
        anchor_y_ratio = max(0.0, min(1.15, float(anchor_y_ratio)))
        return x1 + 0.5 * bw, y1 + anchor_y_ratio * bh, bw, bh

    def _cluster_detection_candidate(
        self,
        det_index: int,
        parsed: ParsedDetection,
        cluster: Dict[str, Any],
        image_w: float,
        image_h: float,
    ) -> Optional[Dict[str, Any]]:
        bbox = parsed.bbox
        class_name = parsed.class_name
        margin = self.cluster_bbox_margin_px
        if class_name == "person":
            # person bbox는 좁고 보통 카메라 이미지 하단에 닿는다.
            # x 정렬은 의미 있게 유지하면서 수직 여유를 추가로 허용한다.
            margin = max(margin, self.cluster_bbox_margin_px)
        if not point_inside_bbox(cluster["u"], cluster["v"], bbox, margin, image_w, image_h):
            return None

        ax, ay, bw, bh = self._bbox_anchor_uv(bbox, class_name)
        half_w = max(1.0, 0.5 * bw)
        half_h = max(1.0, 0.5 * bh)
        dx_norm = abs(float(cluster["u"]) - ax) / half_w
        dy_norm = abs(float(cluster["v"]) - ay) / half_h

        if class_name == "person":
            # person의 경우 LiDAR가 다리/발에 자주 맞으므로 y가 x보다 더 많이 어긋날 수 있다.
            if dx_norm > self.cluster_match_person_x_limit or dy_norm > self.cluster_match_person_y_limit:
                return None
            center_norm = math.sqrt(dx_norm * dx_norm + 0.35 * dy_norm * dy_norm)
        else:
            center_norm = math.sqrt(dx_norm * dx_norm + dy_norm * dy_norm)
            if center_norm > self.cluster_match_max_center_norm:
                return None

        bbox_area_norm = max(0.0, min(1.0, (bw * bh) / max(1.0, image_w * image_h)))
        distance_term = self.cluster_match_distance_weight * float(cluster.get("distance_m", 0.0))
        size_penalty = self.cluster_match_bbox_area_weight * bbox_area_norm
        score = center_norm + distance_term + size_penalty
        if score > self.cluster_match_max_score:
            return None

        return {
            "det_index": det_index,
            "cluster_id": int(cluster.get("id", -1)),
            "score": float(score),
            "center_norm": float(center_norm),
            "dx_norm": float(dx_norm),
            "dy_norm": float(dy_norm),
            "bbox_area_norm": float(bbox_area_norm),
            "anchor": {"u": float(ax), "v": float(ay)},
            "cluster": cluster,
            "parsed": parsed,
        }

    def _fuse_with_global_cluster_assignment(
        self,
        parsed_detections: List[ParsedDetection],
        projected_clusters: List[Dict[str, Any]],
        px: float,
        py: float,
        pz: float,
        camera_heading: float,
        image_w: float,
        image_h: float,
    ) -> List[Dict[str, Any]]:
        candidates: List[Dict[str, Any]] = []
        per_class_candidates: Dict[str, int] = {}
        for det_index, parsed in enumerate(parsed_detections):
            for cluster in projected_clusters:
                cand = self._cluster_detection_candidate(det_index, parsed, cluster, image_w, image_h)
                if cand is not None:
                    candidates.append(cand)
                    per_class_candidates[parsed.class_name] = per_class_candidates.get(parsed.class_name, 0) + 1
        if not candidates:
            self._last_cluster_assignment_stats = {
                "cluster_candidate_count": 0,
                "cluster_selected_count": 0,
                "cluster_candidate_classes": {},
            }
            return []

        # 여러 detection에 들어맞을 법한 cluster는 먼저 처리되는 detection이 아니라
        # 가장 잘 정렬된 bbox에 할당되어야 한다.
        candidates.sort(key=lambda c: (c["score"], c["bbox_area_norm"], -int(c["cluster"].get("count", 0))))
        used_detections: set[int] = set()
        used_clusters: set[int] = set()
        selected: List[Dict[str, Any]] = []

        for cand in candidates:
            det_index = int(cand["det_index"])
            cluster_id = int(cand["cluster_id"])
            if det_index in used_detections or cluster_id in used_clusters:
                continue

            # 같은 detection이 거의 동일한 대체 cluster를 가지면 안전을 위해 이 detection을
            # 기각한다. 앞/뒤(front/back)가 애매한 매칭을 막는다.
            same_det = [
                c for c in candidates
                if int(c["det_index"]) == det_index
                and int(c["cluster_id"]) != cluster_id
                and int(c["cluster_id"]) not in used_clusters
            ]
            if same_det:
                delta = float(same_det[0]["score"]) - float(cand["score"])
                if delta < self.cluster_match_ambiguity_delta:
                    used_detections.add(det_index)
                    continue

            selected.append(cand)
            used_detections.add(det_index)
            if cluster_id >= 0:
                used_clusters.add(cluster_id)

        self._last_cluster_assignment_stats = {
            "cluster_candidate_count": len(candidates),
            "cluster_selected_count": len(selected),
            "cluster_candidate_classes": per_class_candidates,
            "cluster_best_score": float(candidates[0]["score"]) if candidates else None,
            "cluster_best_center_norm": float(candidates[0]["center_norm"]) if candidates else None,
        }

        fused: List[Dict[str, Any]] = []
        for cand in selected:
            parsed: ParsedDetection = cand["parsed"]
            cluster = cand["cluster"]
            pos = cluster["centroid"]
            obj = self._make_fused_object(
                parsed.class_name,
                parsed.confidence,
                parsed.bbox,
                pos["x"],
                pos["y"],
                pos["z"],
                px,
                py,
                pz,
                camera_heading,
                source="yolo_lidar_projection_cluster_fusion",
                matched_lidar_points=int(cluster.get("count", 0)),
                used_lidar_points=int(cluster.get("count", 0)),
                extra={
                    "cluster_id": int(cluster.get("id", -1)),
                    "projection_uv": {"u": float(cluster.get("u", 0.0)), "v": float(cluster.get("v", 0.0))},
                    "cluster_bbox": cluster.get("bbox"),
                    "lidar_match_type": "dbscan_cluster",
                    "semantic_confirmed": True,
                    "discovered_eligible": True,
                    "cluster_match_policy": "global_one_to_one_bbox_alignment_small_bbox_priority_person_relaxed",
                    "cluster_assignment_score": float(cand["score"]),
                    "cluster_center_norm": float(cand["center_norm"]),
                    "cluster_dx_norm": float(cand["dx_norm"]),
                    "cluster_dy_norm": float(cand["dy_norm"]),
                    "bbox_area_norm": float(cand["bbox_area_norm"]),
                    "bbox_anchor": cand["anchor"],
                },
            )
            obj.update(parsed.metadata())
            fused.append(obj)
        return fused

    def _fuse_from_projected_clusters(
        self, class_name: str, confidence: float, bbox: List[float], clusters: List[Dict[str, Any]], assigned_ids: set[int],
        px: float, py: float, pz: float, camera_heading: float, image_w: float, image_h: float,
    ) -> Optional[Dict[str, Any]]:
        candidates = [
            c for c in clusters
            if c["id"] not in assigned_ids
            and point_inside_bbox(c["u"], c["v"], bbox, self.cluster_bbox_margin_px, image_w, image_h)
        ]
        if not candidates:
            return None
        candidates.sort(key=lambda c: (c["distance_m"], -c["count"]))
        best = candidates[0]
        if best["id"] >= 0:
            assigned_ids.add(best["id"])
        pos = best["centroid"]
        return self._make_fused_object(
            class_name, confidence, bbox, pos["x"], pos["y"], pos["z"], px, py, pz, camera_heading,
            source="yolo_lidar_projection_cluster_fusion", matched_lidar_points=int(best["count"]),
            used_lidar_points=int(best["count"]), extra={
                "cluster_id": best["id"],
                "projection_uv": {"u": best["u"], "v": best["v"]},
                "cluster_bbox": best.get("bbox"),
                "lidar_match_type": "dbscan_cluster",
                "semantic_confirmed": True,
                "discovered_eligible": True,
                "cluster_match_policy": "sequential_cluster_fallback",
            },
        )

    def _fuse_from_projected_points(
        self, class_name: str, confidence: float, bbox: List[float], points: List[Dict[str, Any]],
        px: float, py: float, pz: float, camera_heading: float, image_w: float, image_h: float,
    ) -> Optional[Dict[str, Any]]:
        candidates = [p for p in points if point_inside_bbox(p["u"], p["v"], bbox, self.projection_bbox_margin_px, image_w, image_h)]
        if len(candidates) < self.min_projected_points:
            return None
        candidates.sort(key=lambda p: p["distance_m"])
        selected = candidates[: max(1, self.use_nearest_points)]
        est_x, est_y, est_z = self._median_xyz(selected)
        return self._make_fused_object(
            class_name, confidence, bbox, est_x, est_y, est_z, px, py, pz, camera_heading,
            source="yolo_lidar_projection_point_fusion", matched_lidar_points=len(candidates), used_lidar_points=len(selected),
            extra={"projection_bbox_margin_px": self.projection_bbox_margin_px},
        )

    def _fuse_from_angle(
        self, class_name: str, confidence: float, bbox: List[float], usable_points: List[Dict[str, Any]],
        px: float, py: float, pz: float, camera_heading: float, image_w: float,
    ) -> Optional[Dict[str, Any]]:
        x1, _y1, x2, _y2 = bbox
        center_x = 0.5 * (x1 + x2)
        bbox_w = max(1.0, x2 - x1)
        rel_angle = ((center_x - image_w * 0.5) / max(1.0, image_w * 0.5)) * (self.hfov_deg * 0.5)
        bbox_half_angle = max(1.0, (bbox_w / max(1.0, image_w)) * self.hfov_deg * 0.5)
        gate = bbox_half_angle + self.angle_gate_extra_deg
        candidates = [p for p in usable_points if abs(self._normalize_angle_deg(p["relative_bearing_deg"] - rel_angle)) <= gate]
        if len(candidates) < self.min_lidar_points:
            return None
        candidates.sort(key=lambda x: x["distance_m"])
        selected = candidates[: max(1, self.use_nearest_points)]
        est_x, est_y, est_z = self._median_xyz(selected)
        return self._make_fused_object(
            class_name, confidence, bbox, est_x, est_y, est_z, px, py, pz, camera_heading,
            source="yolo_lidar_angle_fusion_fallback", matched_lidar_points=len(candidates), used_lidar_points=len(selected),
            extra={"bbox_center_angle_deg": rel_angle, "angle_gate_deg": gate},
        )

    def _make_fused_object(
        self, class_name: str, confidence: float, bbox: List[float], x: float, y: float, z: float, px: float, py: float, pz: float,
        camera_heading: float, source: str, matched_lidar_points: int, used_lidar_points: int, extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        distance_m = math.sqrt((x - px) ** 2 + (y - py) ** 2 + (z - pz) ** 2)
        known_static = self._match_static_object(class_name, x, y)
        obj = {
            "className": class_name, "confidence": confidence, "bbox": bbox, "camera_heading_deg": camera_heading,
            "distance_m": distance_m, "nearest_distance_m": distance_m, "position_map": {"x": x, "y": y, "z": z, "frame_id": self.map_frame},
            "matched_lidar_points": matched_lidar_points, "used_lidar_points": used_lidar_points,
            "known_static": known_static is not None, "matched_static": known_static, "source": source,
        }
        if extra:
            obj.update(extra)
        return obj

    def _build_angle_lidar_points(self, px: float, py: float) -> List[Dict[str, Any]]:
        points = self.latest_lidar_points
        if points.shape[0] == 0:
            return []
            
        usable_points = []
        cam_heading = self._camera_heading_deg()
        
        # NumPy 벡터화 연산으로 거리 및 각도 계산
        dx = points[:, 0] - px
        dy = points[:, 1] - py
        dist = np.hypot(dx, dy)
        
        mask = (dist >= self.min_fusion_range_m) & (dist <= self.max_fusion_range_m)
        valid_points = points[mask]
        valid_dist = dist[mask]
        
        for i, pt in enumerate(valid_points):
            global_bearing = math.degrees(math.atan2(pt[0] - px, pt[1] - py))
            rel_bearing = self._normalize_angle_deg(global_bearing - cam_heading)
            
            usable_points.append({
                "x": float(pt[0]), "y": float(pt[1]), "z": float(pt[2]),
                "distance_m": float(valid_dist[i]),
                "global_bearing_deg": global_bearing,
                "relative_bearing_deg": rel_bearing
            })
        return usable_points

    def _refresh_confirmation(self, obj: DiscoveredObject, now: float) -> None:
        if obj.is_confirmed:
            return
        age_sec = max(0.0, now - obj.first_seen_wall)
        if obj.observation_count >= self.min_confirm_observations and age_sec >= self.min_confirm_age_sec:
            obj.is_confirmed = True
            obj.confirmed_wall = now

    # ------------------------------------------------------------------
    # 퍼블리싱(Publishing) / 서비스
    # ------------------------------------------------------------------
    def publish_fused(self, fused: List[Dict[str, Any]]) -> None:
        payload = {
            "timestamp_wall": time.time(), "frame_id": self.map_frame, "count": len(fused), "objects": fused,
            "notes": {
                "fusion_method": self.fusion_method,
                "active_priority": "projection_cluster -> projection_points -> angle_fallback",
                "projection_params": self.projection_params,
                "save_service": SERVICE_DISCOVERED_SAVE,
                "clear_service": SERVICE_DISCOVERED_CLEAR,
            },
        }
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self.fused_pub.publish(msg)

    def publish_fusion_debug(self, payload: Optional[Dict[str, Any]]) -> None:
        if not self.debug_fusion_enabled or payload is None:
            return
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self.fusion_debug_pub.publish(msg)

    def publish_discovered(self, discovered_list: List[DiscoveredObject]) -> None:
        confirmed_count = sum(1 for obj in discovered_list if obj.is_confirmed)
        payload = {
            "timestamp_wall": time.time(), "frame_id": self.map_frame, "count": len(discovered_list),
            "confirmed_count": confirmed_count, "candidate_count": len(discovered_list) - confirmed_count,
            "objects": [asdict(o) for o in discovered_list],
            "save_service": SERVICE_DISCOVERED_SAVE, "save_confirmed_only": self.save_confirmed_only,
        }
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self.discovered_pub.publish(msg)
        self.discovered_marker_pub.publish(self.make_discovered_markers(discovered_list))

    def publish_current_markers(self, fused: List[Dict[str, Any]]) -> None:
        self.current_marker_pub.publish(self.make_current_markers(fused))

    def request_terrain_finalize(self) -> str:
        try:
            if not self.terrain_finalize_client.service_is_ready():
                self.terrain_finalize_client.wait_for_service(timeout_sec=0.2)
            if not self.terrain_finalize_client.service_is_ready():
                return "terrain finalize service not ready"
            self.terrain_finalize_client.call_async(Trigger.Request())
            return "terrain finalize requested"
        except Exception as exc:
            return f"terrain finalize request failed: {exc}"

    def save_service_cb(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        try:
            with self._lock:
                saved_paths = self.save_discovered_map_locked()
            terrain_status = self.request_terrain_finalize()
            response.success = True
            response.message = "Saved discovered map: " + ", ".join(str(p) for p in saved_paths) + f"; {terrain_status}"
        except Exception as exc:
            response.success = False
            response.message = f"Failed to save discovered map: {exc}"
        return response

    def clear_service_cb(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        with self._lock:
            count = len(self.discovered)
            self.discovered.clear()
        response.success = True
        response.message = f"Cleared {count} discovered object(s)"
        return response

    def save_discovered_map_locked(self) -> List[Path]:
        self.save_directory.mkdir(parents=True, exist_ok=True)
        payload = self.make_map_payload_locked()
        latest_path = self.save_directory / self.save_latest_filename
        latest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        paths = [latest_path]
        if self.save_timestamped_copy:
            stamp = time.strftime("%Y%m%d_%H%M%S")
            archive_path = self.save_directory / f"discovered_objects_{stamp}.map"
            archive_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            paths.append(archive_path)
        return paths

    def make_map_payload_locked(self) -> Dict[str, Any]:
        obstacles = []
        objects_to_save = [obj for obj in self.discovered if obj.is_confirmed] if self.save_confirmed_only else list(self.discovered)
        for obj in objects_to_save:
            obstacles.append({
                "prefabName": obj.object_id,
                "position": {"x": obj.map_x, "y": obj.map_z, "z": obj.map_y},
                "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                "metadata": asdict(obj),
            })
        return {
            "terrainIndex": 5, "map_role": "discovered_runtime_objects", "frame_id": self.map_frame,
            "coordinate_policy": "saved position uses Unity raw convention: x=map.x, y=map.z, z=map.y",
            "created_wall": time.time(), "object_count": len(obstacles), "candidate_count": len(self.discovered),
            "confirmed_count": sum(1 for obj in self.discovered if obj.is_confirmed),
            "save_confirmed_only": self.save_confirmed_only, "obstacles": obstacles,
        }

    # ------------------------------------------------------------------
    # 마커(Marker) 생성
    # ------------------------------------------------------------------
    def make_current_markers(self, fused: List[Dict[str, Any]]) -> MarkerArray:
        markers = MarkerArray()
        if self.player_pose is None:
            return markers
        px = float(self.player_pose.pose.position.x)
        py = float(self.player_pose.pose.position.y)
        pz = float(self.player_pose.pose.position.z)
        mid = 0
        for obj in fused:
            pos = obj.get("position_map", {}) if isinstance(obj.get("position_map"), dict) else {}
            x = self._as_float(pos.get("x"), 0.0)
            y = self._as_float(pos.get("y"), 0.0)
            z = self._as_float(pos.get("z"), 0.0) + self.current_z_offset
            cls = str(obj.get("className", "unknown")).lower()
            color = self._color_for_class(cls, 0.95)
            dist = self._as_float(obj.get("distance_m"), 0.0)
            conf = self._as_float(obj.get("confidence"), 0.0)
            known = bool(obj.get("known_static"))
            source = str(obj.get("source", ""))
            markers.markers.append(self._sphere_marker("fused_object", mid, x, y, z, self.sphere_scale, color, self.current_lifetime_sec)); mid += 1
            markers.markers.append(self._line_marker("fused_object_distance", mid, [(px, py, pz + 2.0), (x, y, z)], self._color_for_class(cls, 0.75), self.current_lifetime_sec)); mid += 1
            label = f"LIVE {cls} {dist:.1f}m conf={conf:.2f}\n{source.replace('yolo_lidar_', '')}"
            if known:
                label += " static"
            markers.markers.append(self._text_marker("fused_object_label", mid, x, y, z + 1.8, label, self._color_for_class(cls, 1.0), self.current_lifetime_sec)); mid += 1
        return markers

    def make_discovered_markers(self, discovered_list: List[DiscoveredObject]) -> MarkerArray:
        markers = MarkerArray()

        # stale candidate/SAVED 마커를 먼저 지운다. 이렇게 하면 memory decay로 제거된 옛 candidate를
        # RViz가 여전히 활성인 것처럼 표시하는 것을 막는다.
        clear_marker = Marker()
        clear_marker.header.frame_id = self.map_frame
        clear_marker.header.stamp = self.get_clock().now().to_msg()
        clear_marker.action = Marker.DELETEALL
        markers.markers.append(clear_marker)

        for idx, obj in enumerate(discovered_list):
            alpha = 0.95 if obj.is_confirmed else 0.55
            color = self._color_for_class(obj.class_name, alpha)
            x = obj.map_x
            y = obj.map_y
            z = obj.map_z + self.discovered_z_offset
            base_id = idx * 3
            markers.markers.append(self._cube_marker("discovered_object", base_id, x, y, z, self.discovered_cube_scale, color))
            status = "SAVED" if obj.is_confirmed else "CANDIDATE"
            label = f"{status} {obj.class_name}\nobs={obj.observation_count} conf={obj.confidence:.2f}"
            markers.markers.append(self._text_marker("discovered_object_label", base_id + 1, x, y, z + 2.0, label, self._color_for_class(obj.class_name, 1.0), 0.0))

            if obj.is_confirmed:
                saved_color = ColorRGBA()
                saved_color.r = 1.0
                saved_color.g = 1.0
                saved_color.b = 1.0
                saved_color.a = 1.0
                markers.markers.append(
                    self._sphere_marker(
                        "discovered_saved_badge",
                        base_id + 2,
                        x,
                        y,
                        z + 3.2,
                        max(0.45, self.sphere_scale * 0.45),
                        saved_color,
                        0.0,
                    )
                )
        return markers

    def _base_marker(self, ns: str, marker_id: int, lifetime_sec: float = 0.0) -> Marker:
        m = Marker()
        m.header.frame_id = self.map_frame
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = ns
        m.id = int(marker_id)
        m.action = Marker.ADD
        if lifetime_sec > 0.0:
            m.lifetime.sec = int(lifetime_sec)
            m.lifetime.nanosec = int((lifetime_sec - int(lifetime_sec)) * 1e9)
        return m

    def _sphere_marker(self, ns: str, marker_id: int, x: float, y: float, z: float, scale: float, color: ColorRGBA, lifetime: float) -> Marker:
        m = self._base_marker(ns, marker_id, lifetime)
        m.type = Marker.SPHERE
        m.pose.position.x = float(x); m.pose.position.y = float(y); m.pose.position.z = float(z)
        m.pose.orientation.w = 1.0
        m.scale.x = scale; m.scale.y = scale; m.scale.z = scale
        m.color = color
        return m

    def _cube_marker(self, ns: str, marker_id: int, x: float, y: float, z: float, scale: float, color: ColorRGBA) -> Marker:
        m = self._base_marker(ns, marker_id, 0.0)
        m.type = Marker.CUBE
        m.pose.position.x = float(x); m.pose.position.y = float(y); m.pose.position.z = float(z)
        m.pose.orientation.w = 1.0
        m.scale.x = scale; m.scale.y = scale; m.scale.z = scale
        m.color = color
        return m

    def _line_marker(self, ns: str, marker_id: int, points: List[Tuple[float, float, float]], color: ColorRGBA, lifetime: float) -> Marker:
        m = self._base_marker(ns, marker_id, lifetime)
        m.type = Marker.LINE_STRIP
        m.scale.x = self.line_width
        m.color = color
        for x, y, z in points:
            p = Point(); p.x = float(x); p.y = float(y); p.z = float(z)
            m.points.append(p)
        return m

    def _text_marker(self, ns: str, marker_id: int, x: float, y: float, z: float, text: str, color: ColorRGBA, lifetime: float) -> Marker:
        m = self._base_marker(ns, marker_id, lifetime)
        m.type = Marker.TEXT_VIEW_FACING
        m.pose.position.x = float(x); m.pose.position.y = float(y); m.pose.position.z = float(z)
        m.pose.orientation.w = 1.0
        m.scale.z = self.text_height
        m.color = color
        m.text = text
        return m

    # ------------------------------------------------------------------
    # 수식(Math) / 파싱 헬퍼
    # ------------------------------------------------------------------
    def _make_fusion_debug_base_locked(self) -> Dict[str, Any]:
        det_payload = self.latest_detections_payload if isinstance(self.latest_detections_payload, dict) else {}
        cluster_payload = self.latest_clusters_payload if isinstance(self.latest_clusters_payload, dict) else {}
        raw_dets = det_payload.get("detections", []) if isinstance(det_payload.get("detections", []), list) else []
        clusters = cluster_payload.get("clusters", []) if isinstance(cluster_payload.get("clusters", []), list) else []
        return {
            "timestamp_wall": time.time(),
            "frame_id": self.map_frame,
            "has_player_pose": self.player_pose is not None,
            "has_detection_payload": isinstance(self.latest_detections_payload, dict),
            "has_info": self.latest_info is not None,
            "has_cluster_payload": isinstance(self.latest_clusters_payload, dict),
            "lidar_point_count": int(self.latest_lidar_points.shape[0]) if isinstance(self.latest_lidar_points, np.ndarray) else 0,
            "raw_detection_count": len(raw_dets),
            "cluster_count": len(clusters),
        }

    def _extract_cluster_timestamp(self, payload: Optional[Dict[str, Any]]) -> float:
        if not isinstance(payload, dict):
            return 0.0
        for key in ("timestamp_ros_sec", "timestamp_wall", "stamp_sec"):
            if payload.get(key) is not None:
                return self._as_float(payload.get(key), 0.0)
        return 0.0

    def _merge_radius_for_class(self, class_name: str) -> float:
        try:
            value = self.merge_radius_by_class.get(str(class_name).lower())
            if value is not None:
                return float(value)
        except Exception:
            pass
        return float(self.merge_radius_m)

    def _is_stale_async_detection_payload(self, payload: Dict[str, Any]) -> bool:
        if not self.drop_stale_async_detection or not isinstance(payload, dict):
            return False
        if not bool(payload.get("asyncYolo", False)):
            return False
        age_ms = self._as_float(payload.get("resultAgeMs"), -1.0)
        stale_flag = bool(payload.get("staleAsyncResult", False))
        if age_ms < 0.0:
            return stale_flag
        # 줄어든 max_result_age_ms (100ms) 기준으로 판별
        return stale_flag or age_ms > self.max_async_result_age_ms

    def _extract_image_size(self, payload: Dict[str, Any]) -> Tuple[float, float]:
        for key in ("image_shape", "frame_shape", "latestFrameShape"):
            shape = payload.get(key)
            if isinstance(shape, list) and len(shape) >= 2:
                h = self._as_float(shape[0], self.default_image_height)
                w = self._as_float(shape[1], self.default_image_width)
                return max(1.0, w), max(1.0, h)
        image = payload.get("image")
        if isinstance(image, dict):
            w = self._as_float(image.get("width"), self.default_image_width)
            h = self._as_float(image.get("height"), self.default_image_height)
            return max(1.0, w), max(1.0, h)
        return float(self.default_image_width), float(self.default_image_height)

    def _extract_lidar_point(self, point: Dict[str, Any], px: float, py: float) -> Optional[Dict[str, Any]]:
        if not isinstance(point, dict):
            return None
        pos = point.get("position_map") if isinstance(point.get("position_map"), dict) else point.get("position")
        if not isinstance(pos, dict):
            return None
        x = self._as_float(pos.get("x"), 0.0)
        y = self._as_float(pos.get("y"), 0.0)
        z = self._as_float(pos.get("z"), 0.0)
        dx = x - px
        dy = y - py
        distance = math.sqrt(dx * dx + dy * dy)
        if distance < self.min_fusion_range_m or distance > self.max_fusion_range_m:
            return None
        global_bearing = math.degrees(math.atan2(dx, dy))
        rel_bearing = self._normalize_angle_deg(global_bearing - self._camera_heading_deg())
        return {"x": x, "y": y, "z": z, "distance_m": distance, "global_bearing_deg": global_bearing, "relative_bearing_deg": rel_bearing, "raw": point}

    def _camera_heading_deg(self) -> float:
        if self.heading_source == "turret" and self.turret_heading_deg is not None:
            return float(self.turret_heading_deg)
        if self.heading_source == "body_plus_turret" and self.turret_heading_deg is not None:
            return self._normalize_angle_360(float(self.player_heading_deg) + float(self.turret_heading_deg))
        return float(self.player_heading_deg)

    def _median_xyz(self, items: List[Dict[str, Any]]) -> Tuple[float, float, float]:
        xs = sorted(float(i["x"]) for i in items)
        ys = sorted(float(i["y"]) for i in items)
        zs = sorted(float(i["z"]) for i in items)
        mid = len(items) // 2
        if len(items) % 2 == 1:
            return xs[mid], ys[mid], zs[mid]
        return 0.5 * (xs[mid - 1] + xs[mid]), 0.5 * (ys[mid - 1] + ys[mid]), 0.5 * (zs[mid - 1] + zs[mid])

    def _match_static_object(self, class_name: str, x: float, y: float) -> Optional[Dict[str, Any]]:
        best = None
        best_d = 1e9
        for obj in self.static_objects:
            if self.same_category_only and not self._category_matches_detection(obj.category, class_name):
                continue
            d = math.hypot(obj.map_x - x, obj.map_y - y)
            if d < best_d:
                best_d = d
                best = obj
        if best is not None and best_d <= self.static_match_radius_m:
            return {"prefabName": best.prefab_name, "category": best.category, "distance_m": best_d}
        return None

    def _find_existing_discovered(self, class_name: str, x: float, y: float, track_id: Optional[int] = None) -> Optional[DiscoveredObject]:
        # 중복 저장 방지는 YOLO trackId보다 tank_map 좌표 기반이 우선이다.
        # trackId는 시야 이탈/재진입 시 바뀔 수 있으므로 보조 merge 조건으로만 사용한다.
        merge_radius = self._merge_radius_for_class(class_name)

        best = None
        best_d = 1e9
        for obj in self.discovered:
            if not self.merge_across_classes and obj.class_name != class_name:
                continue
            d = math.hypot(obj.map_x - x, obj.map_y - y)
            if d < best_d:
                best_d = d
                best = obj
        if best is not None and best_d <= merge_radius:
            return best

        if self.track_id_merge_enabled and track_id is not None:
            best_track = None
            best_track_d = 1e9
            for obj in self.discovered:
                if obj.track_id != track_id:
                    continue
                if not self.merge_across_classes and obj.class_name != class_name:
                    continue
                d = math.hypot(obj.map_x - x, obj.map_y - y)
                if d < best_track_d:
                    best_track_d = d
                    best_track = obj
            if best_track is not None and best_track_d <= self.track_id_merge_radius_m:
                return best_track
        return None

    @staticmethod
    def _as_optional_int(value: Any) -> Optional[int]:
        if value is None:
            return None
        try:
            return int(float(value))
        except Exception:
            return None

    @staticmethod
    def _category_for_prefab(prefab: str) -> str:
        name = prefab.lower()
        if name.startswith("human"): return "person"
        for prefix, category in (("rock", "rock"), ("wall", "wall"), ("tank", "tank"), ("tent", "tent"), ("tree", "tree"), ("house", "house")):
            if name.startswith(prefix): return category
        return "unknown"

    @staticmethod
    def _category_matches_detection(category: str, class_name: str) -> bool:
        return str(category).lower() == str(class_name).lower()

    @staticmethod
    def _normalize_angle_deg(angle: float) -> float:
        return (float(angle) + 180.0) % 360.0 - 180.0

    @staticmethod
    def _normalize_angle_360(angle: float) -> float:
        return float(angle) % 360.0

    @staticmethod
    def _as_float(value: Any, default: float = 0.0) -> float:
        try:
            if value is None: return float(default)
            return float(value)
        except Exception:
            return float(default)

    def _color_for_class(self, class_name: str, alpha: float) -> ColorRGBA:
        hex_color = self.class_colors.get(str(class_name).lower(), self.class_colors.get("unknown", "#FFFFFF"))
        r, g, b = self._hex_to_rgb(hex_color)
        c = ColorRGBA(); c.r = r; c.g = g; c.b = b; c.a = float(alpha)
        return c

    @staticmethod
    def _hex_to_rgb(value: str) -> Tuple[float, float, float]:
        s = str(value).strip()
        if s.startswith("#"): s = s[1:]
        if len(s) != 6: return 1.0, 1.0, 1.0
        try:
            return int(s[0:2], 16) / 255.0, int(s[2:4], 16) / 255.0, int(s[4:6], 16) / 255.0
        except Exception:
            return 1.0, 1.0, 1.0


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LocalPathNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()