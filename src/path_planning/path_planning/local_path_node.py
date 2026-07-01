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
from geometry_msgs.msg import PoseStamped, Point
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
# 정찰 전용: 미분류 후보(라이다엔 잡혔으나 카메라 분류 안 된 클러스터)를 controller에 알려
# 감속/dwell(전방 후보) + 포탑 step-stare(옆 후보) 큐로 쓰게 한다. mission_type==recon에서만 발행.
TOPIC_OBSERVE_REQUEST = "/tank/recon/observe_request"
# Scenario-2 mission gate. When enabled, route_A.json must not report
# ``reached`` merely because the vehicle arrived at the firing checkpoint.
# The ballistic node opens this gate only after it has fired and returned.
TOPIC_TURRET_STATUS = "/tank/turret/status"

from lidar.config import TOPIC_LIDAR_DETECTED_MAP
from path_planning.config import (
    CAMERA_LIDAR_PROJECTION_PARAMS,
    CLASS_COLOR_DEFAULTS,
    DISCOVERED_CLASS_RADIUS,
    DISCOVERED_OBSTACLE_INFLATE,
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
)
from tank_visual_perception.projection import (
    camera_pose_source,
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
    # raw_*는 이번 센서 프레임 측정, map_*은 planner/RViz가 쓰는 안정화 좌표다.
    raw_map_x: float = 0.0
    raw_map_y: float = 0.0
    raw_map_z: float = 0.0
    position_state: str = "candidate"
    avoidance_radius_m: float = 0.0


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

        # Cluster centroid 하나만 bbox와 비교하면, 넓거나 비대칭인 물체에서는
        # centroid가 화면 밖으로 나가도 실제 LiDAR 표면점은 YOLO bbox에 남는다.
        # DBSCAN 노드가 보내는 points_map 샘플을 함께 투영하여 다수 점 근거로 매칭한다.
        self.cluster_projected_point_matching_enabled = bool(
            self._cfg(["projection", "cluster_projected_point_matching_enabled"], True)
        )
        self.cluster_projected_min_points_in_detection = max(
            1,
            int(self._cfg(["projection", "cluster_projected_min_points_in_detection"], 1)),
        )
        self.cluster_projected_bbox_fallback_enabled = bool(
            self._cfg(["projection", "cluster_projected_bbox_fallback_enabled"], True)
        )
        self.cluster_projected_bbox_min_overlap_px2 = max(
            0.0,
            float(self._cfg(["projection", "cluster_projected_bbox_min_overlap_px2"], 16.0)),
        )

        self.add_only_unmatched = bool(self._cfg(["static_matching", "add_only_unmatched_to_recon"], True))
        self.static_match_radius_m = float(self._cfg(["static_matching", "static_match_radius_m"], 4.0))
        self.same_category_only = bool(self._cfg(["static_matching", "same_category_only"], True))

        self.mapping_enabled = bool(self._cfg(["mapping", "enabled"], True))
        self.merge_radius_m = float(self._cfg(["mapping", "merge_radius_m"], 5.0))
        self.ema_alpha = float(self._cfg(["mapping", "position_ema_alpha"], 0.35))
        # 후보 상태에서는 EMA + 1회 이동 상한을 쓰고, 확정 후에는 지도 좌표를 동결한다.
        self.freeze_position_on_confirm = bool(self._cfg(["mapping", "freeze_position_on_confirm"], True))
        self.candidate_max_step_m = max(0.05, float(self._cfg(["mapping", "candidate_max_step_m"], 1.0)))
        self.candidate_outlier_reject_m = max(self.candidate_max_step_m, float(self._cfg(["mapping", "candidate_outlier_reject_m"], 6.0)))
        self.avoidance_inflate_m = max(0.0, float(self._cfg(["mapping", "avoidance_inflate_m"], DISCOVERED_OBSTACLE_INFLATE)))
        self.add_classes = set(str(x).lower() for x in self._cfg(["mapping", "add_classes"], ["person", "rock", "tank", "car", "house", "tent"]))
        self.merge_radius_by_class = dict(self._cfg(["mapping", "merge_radius_by_class"], {}) or {})
        self.save_directory = Path(str(self._cfg(["mapping", "save_directory"], "~/tankcc/tank_discovered_maps"))).expanduser()
        self.save_latest_filename = str(self._cfg(["mapping", "save_latest_filename"], "discovered_objects_latest.map"))
        self.save_timestamped_copy = bool(self._cfg(["mapping", "save_timestamped_copy"], True))
        self.save_confirmed_only = bool(self._cfg(["mapping", "save_confirmed_only"], True))
        self.min_confirm_observations = int(self._cfg(["mapping", "min_confirm_observations"], 5))
        self.min_confirm_age_sec = float(self._cfg(["mapping", "min_confirm_age_sec"], 1.0))
        # 정찰 stop-to-confirm: 확정 기준을 정찰에서만 상향(ROS 파라미터 override; <0이면 yaml값 유지).
        # 시나리오2 새 적탱크 감지는 빠르게 유지해야 하므로 launch에서 recon만 올린다.
        self.declare_parameter("min_confirm_observations_override", -1)
        self.declare_parameter("min_confirm_age_sec_override", -1.0)
        _mco = int(self.get_parameter("min_confirm_observations_override").value)
        if _mco >= 0:
            self.min_confirm_observations = _mco
        _mca = float(self.get_parameter("min_confirm_age_sec_override").value)
        if _mca >= 0.0:
            self.min_confirm_age_sec = _mca
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

        # ── 정찰 전용 "지각 주도 관측" 설정 ────────────────────────────────
        # mission_type==recon에서만 미분류 후보를 집계해 observe_request로 발행한다.
        # mission/return은 발행 안 함(현행 주행 거동 불변).
        self.declare_parameter("mission_type", "mission")
        self.mission_type = str(self.get_parameter("mission_type").value).lower()
        # 이미 확정 분류된 객체에서 이 반경 안의 클러스터는 후보 아님(이미 분류됨).
        self.declare_parameter("observe_classified_radius_m", 8.0)
        self.observe_classified_radius_m = float(self.get_parameter("observe_classified_radius_m").value)
        # 맵에 이미 있는 정적 장애물(나무 등)에서 이 반경 안의 클러스터는 후보 아님 — '맵에 없는 것'만 관측.
        # (안 그러면 숲 나무가 후보를 도배해 포탑이 나무를 응시 → YOLO에 tree 클래스 없어 no_parsed_detection 폭증.)
        self.declare_parameter("observe_static_exclude_radius_m", 4.0)
        self.observe_static_exclude_radius_m = float(self.get_parameter("observe_static_exclude_radius_m").value)
        # 포탑 응시 대상은 전방-대각 arc(FOV밖~이 각도)까지만 — 후방(지나친) 후보는 안 쫓음(포탑이 뒤를 안 보게).
        self.declare_parameter("observe_turret_max_bearing_deg", 100.0)
        self.observe_turret_max_bearing_deg = float(self.get_parameter("observe_turret_max_bearing_deg").value)
        # 라이다 기하 prior: map-frame bbox 치수로 거친 크기 추정(분류 아님, 우선순위용).
        self.declare_parameter("observe_vehicle_min_size_m", 2.0)   # 전차/차-크기 수평 하한
        self.declare_parameter("observe_large_min_size_m", 6.0)     # 집/초소-크기 수평 하한
        self.declare_parameter("observe_min_height_m", 1.0)         # 차량-크기 높이 하한
        self.declare_parameter("observe_large_min_height_m", 3.0)   # 대형 높이 하한
        self.observe_vehicle_min_size_m = float(self.get_parameter("observe_vehicle_min_size_m").value)
        self.observe_large_min_size_m = float(self.get_parameter("observe_large_min_size_m").value)
        self.observe_min_height_m = float(self.get_parameter("observe_min_height_m").value)
        self.observe_large_min_height_m = float(self.get_parameter("observe_large_min_height_m").value)
        self.declare_parameter("observe_max_publish", 12)           # observe_request에 실을 후보 상한
        self.observe_max_publish = int(self.get_parameter("observe_max_publish").value)

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
        self._fusion_reject_counts: Dict[str, int] = {}   # 융합 결과 사유 누적(프레임당 1건) — 왜 확정 안 되나 진단
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
        self.create_subscription(String, TOPIC_RECON_RAW, self.recon_raw_cb, transient_qos)

        self.fused_pub = self.create_publisher(String, TOPIC_FUSED_OBJECTS, 10)
        self.discovered_pub = self.create_publisher(String, TOPIC_DISCOVERED_OBJECTS, transient_qos)
        self.current_marker_pub = self.create_publisher(MarkerArray, TOPIC_FUSED_OBJECT_MARKERS, 10)
        self.discovered_marker_pub = self.create_publisher(MarkerArray, TOPIC_DISCOVERED_OBJECT_MARKERS, transient_qos)
        self.fusion_debug_pub = self.create_publisher(String, TOPIC_FUSION_DEBUG_STATUS, 10)
        self.observe_request_pub = self.create_publisher(String, TOPIC_OBSERVE_REQUEST, 10)

        self.declare_parameter("route_id", "A")
        self.declare_parameter("route_map_name", "finalmap")
        self.declare_parameter("recon_report_dir", "./recon_reports")
        self.declare_parameter("goal_pose_topic", "/tank/goal/pose")
        self.declare_parameter("goal_tolerance", 5.0)
        # Scenario-2 sets this true. Until ballistic_turret_node reaches its
        # terminal ``returned`` phase, arriving at the checkpoint is only an
        # intermediate stop — it must not create route_A.json(reached=true),
        # because the scenario runner interprets that file as mission success
        # and terminates the whole ROS launch.
        self.declare_parameter("require_turret_completion_for_reached", False)
        self.declare_parameter("turret_status_topic", TOPIC_TURRET_STATUS)

        self.route_id = str(self.get_parameter("route_id").value)
        self.route_map_name = str(self.get_parameter("route_map_name").value)
        self.recon_report_dir = str(self.get_parameter("recon_report_dir").value)
        self.goal_pose_topic = str(self.get_parameter("goal_pose_topic").value)
        self.goal_tolerance = float(self.get_parameter("goal_tolerance").value)
        self.require_turret_completion_for_reached = bool(
            self.get_parameter("require_turret_completion_for_reached").value
        )
        self.turret_status_topic = str(self.get_parameter("turret_status_topic").value)

        self.recon_logger = ReconLogger(self.route_id, self.route_map_name, self.recon_report_dir)
        self.sim_time = 0.0
        self._last_sim_time = 0.0
        self._report_saved = False
        self.goal_pos = None
        self._turret_phase = "unknown"
        # False by default preserves existing recon/one-way route behavior.
        self._turret_completion_seen = not self.require_turret_completion_for_reached
        self._reach_gate_wait_logged = False

        self.create_subscription(PoseStamped, self.goal_pose_topic, self.goal_pose_cb, 10)
        self.create_subscription(String, self.turret_status_topic, self.turret_status_cb, 10)
        self.create_subscription(String, "/tank/event/collision", self.collision_cb, 10)

        # 주행 품질 진단용 구독 (읽기만 — 제어/계획 거동은 안 건드림). recon_logger에 per-step 기록해
        # 진동/끼임 원인이 경로 churn / APF 불일치 / 제어 채터 중 무엇인지 사후에 수치로 가린다.
        self.create_subscription(String, "/tank/planner/status", self._diag_planner_status_cb, 10)
        self.create_subscription(NavPath, "/tank/global_path", self._diag_global_path_cb, 10)
        self.create_subscription(PoseStamped, "/tank/path/lookahead_pose", self._diag_lookahead_cb, 10)
        self.create_subscription(PoseStamped, "/tank/local_target/pose", self._diag_local_target_cb, 10)
        self.create_subscription(String, "/tank/control/command", self._diag_command_cb, 10)
        # 정찰 관측 거동(②dwell·③포탑) 진단: 컨트롤러 status의 speed_mode/recon_observation을 recon_logger로.
        self.create_subscription(String, "/tank/control/status", self._diag_status_cb, 10)

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

    def turret_status_cb(self, msg: String) -> None:
        """Open Scenario-2 completion gate only after return is physically done."""
        try:
            payload = json.loads(msg.data)
        except Exception:
            return
        if not isinstance(payload, dict):
            return
        phase = str(payload.get("phase", "")).strip().lower()
        if not phase:
            return
        with self._lock:
            self._turret_phase = phase
            if phase in {"returned", "complete"}:
                self._turret_completion_seen = True
            elif phase in {
                "approach", "settling", "aim", "firing", "wait_impact",
                "hit", "miss", "impact_timeout", "returning",
            }:
                self._turret_completion_seen = False

    def collision_cb(self, msg: String) -> None:
        with self._lock:
            self.recon_logger.collisions += 1
            # 충돌 위치(현재 전차 pose) 기록 → analyze_run이 궤적에 충돌 지점 오버레이.
            if self.player_pose is not None:
                self.recon_logger.collision_events.append({
                    "t": round(float(self.sim_time), 2),
                    "x": round(float(self.player_pose.pose.position.x), 2),
                    "z": round(float(self.player_pose.pose.position.y), 2),
                })

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

    def _diag_status_cb(self, msg: String) -> None:
        """컨트롤러 status에서 정찰 관측 거동(dwell/slow/turret)을 추출해 recon_logger로 흘린다."""
        try:
            data = json.loads(msg.data)
        except Exception:
            return
        if not isinstance(data, dict):
            return
        speed_mode = str(data.get("speed_mode", ""))
        ro = data.get("recon_observation") if isinstance(data.get("recon_observation"), dict) else {}
        dwell = ("recon_dwell" in speed_mode) or (ro.get("mode") == "dwell")
        if ro.get("mode"):
            mode = str(ro.get("mode"))            # dwell|slow
        elif ro.get("turret"):
            mode = "turret"
        elif "recon_observe_slow" in speed_mode:
            mode = "slow"
        else:
            mode = ""
        with self._lock:
            self.recon_logger.set_observe_mode(dwell, mode)

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
                # 현재 프레임 객체도 map 좌표를 새 raw 측정치가 아닌 안정화/동결 좌표로 다시 기입한다.
                self._attach_stable_pose_to_fused_locked(fused)

            now = time.time()
            self.discovered = [
                obj for obj in self.discovered
                if obj.is_confirmed or (now - obj.last_seen_wall) < self.memory_decay_sec
            ]

            self.recon_logger.set_fusion_rejects(self._fusion_reject_counts)  # 융합 드롭 사유 누적(왜 확정 안 되나)

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
                    gate_open = (
                        not self.require_turret_completion_for_reached
                        or self._turret_completion_seen
                    )
                    if not gate_open:
                        # Do not write route_A.json(reached=true) at (50,260).
                        # The external Scenario-2 harness treats that file as an
                        # immediate success signal and would kill the aim/fire node.
                        if not self._reach_gate_wait_logged:
                            self.get_logger().info(
                                "Goal radius reached; withholding route report until "
                                f"turret completion (phase={self._turret_phase!r})"
                            )
                            self._reach_gate_wait_logged = True
                    elif not self._report_saved:
                        self.recon_logger.reached = True
                        self.recon_logger.save_report()
                        self._report_saved = True

            # 퍼블리싱 전 안전하게 복사
            fused_to_pub = list(fused)
            discovered_to_pub = list(self.discovered)
            fusion_debug_to_pub = dict(self._last_fusion_debug) if self._last_fusion_debug else None
            # 정찰 전용: 미분류 후보 집계(읽기만; 주행/경로 거동은 안 건드림). recon에서만.
            observe_candidates = (
                self.compute_observe_candidates_locked() if self.mission_type == "recon" else []
            )
            if self.mission_type == "recon":
                by_class: Dict[str, int] = {}
                for c in observe_candidates:
                    by_class[c["size_class"]] = by_class.get(c["size_class"], 0) + 1
                self.recon_logger.set_observe_candidates({
                    "n": len(observe_candidates),
                    "n_fov": sum(1 for c in observe_candidates if c["in_forward_fov"]),
                    "n_side": sum(1 for c in observe_candidates if not c["in_forward_fov"]),
                    "by_class": by_class,
                })

        self.publish_fused(fused_to_pub)
        self.publish_current_markers(fused_to_pub)
        self.publish_discovered(discovered_to_pub)
        self.publish_fusion_debug(fusion_debug_to_pub)
        if self.mission_type == "recon":
            self.publish_observe_request(observe_candidates)

    def compute_fused_objects_locked(self) -> List[Dict[str, Any]]:
        debug = self._make_fusion_debug_base_locked()

        def finish(reason: str, fused: Optional[List[Dict[str, Any]]] = None, **extra: Any) -> List[Dict[str, Any]]:
            result = fused or []
            # 사유별 누적(항상) — recon_logger→route_*.json. ok_* 성공 vs strict_no_cluster_assignment/stale 등 실패.
            self._fusion_reject_counts[reason] = self._fusion_reject_counts.get(reason, 0) + 1
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
            "projection_pose_source": projection_context.get("pose_source") if projection_context else None,
            "camera_pose_deg": ({
                "yaw": float(projection_context["camera_yaw"]),
                "pitch": float(projection_context["camera_pitch"]),
                "roll": float(projection_context["camera_roll"]),
            } if projection_context else None),
            "projected_cluster_count": len(projected_clusters),
            "semantic_requires_cluster": self.semantic_requires_cluster,
            "prefer_clusters": self.prefer_clusters,
            "allow_angle_fallback": self.allow_angle_fallback,
        })
        if self.debug_fusion_enabled and projection_context is not None:
            debug.update(self._make_projection_tuning_debug(parsed_detections, int(image_w), int(image_h), px, py))

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

    def _avoidance_radius_for_class(self, class_name: str) -> float:
        base = float(DISCOVERED_CLASS_RADIUS.get(str(class_name).lower(), DISCOVERED_CLASS_RADIUS["unknown"]))
        return base + self.avoidance_inflate_m

    def _candidate_filtered_position(
        self,
        existing: DiscoveredObject,
        raw_x: float,
        raw_y: float,
        raw_z: float,
    ) -> tuple[float, float, float, bool]:
        """후보 단계의 급격한 프레임 이동을 거부하고 EMA 이동량도 제한한다."""
        dx, dy, dz = raw_x - existing.map_x, raw_y - existing.map_y, raw_z - existing.map_z
        distance = math.sqrt(dx * dx + dy * dy + dz * dz)
        if distance > self.candidate_outlier_reject_m:
            return existing.map_x, existing.map_y, existing.map_z, False
        alpha = max(0.0, min(1.0, self.ema_alpha))
        step_x, step_y, step_z = alpha * dx, alpha * dy, alpha * dz
        step_norm = math.sqrt(step_x * step_x + step_y * step_y + step_z * step_z)
        if step_norm > self.candidate_max_step_m:
            scale = self.candidate_max_step_m / step_norm
            step_x, step_y, step_z = step_x * scale, step_y * scale, step_z * scale
        return existing.map_x + step_x, existing.map_y + step_y, existing.map_z + step_z, True

    def _copy_stable_pose_to_fused(self, fused_obj: Dict[str, Any], tracked: DiscoveredObject) -> None:
        raw = fused_obj.get("position_map") if isinstance(fused_obj.get("position_map"), dict) else {}
        fused_obj["raw_position_map"] = {
            "x": self._as_float(raw.get("x"), tracked.raw_map_x),
            "y": self._as_float(raw.get("y"), tracked.raw_map_y),
            "z": self._as_float(raw.get("z"), tracked.raw_map_z),
            "frame_id": self.map_frame,
        }
        fused_obj["position_map"] = {
            "x": float(tracked.map_x), "y": float(tracked.map_y), "z": float(tracked.map_z), "frame_id": self.map_frame,
        }
        fused_obj["stable_object_id"] = tracked.object_id
        fused_obj["position_state"] = tracked.position_state
        fused_obj["avoidance_radius_m"] = float(tracked.avoidance_radius_m)
        fused_obj["avoidance_radius_includes_planner_inflate"] = True
        fused_obj["is_confirmed"] = bool(tracked.is_confirmed)

    def _attach_stable_pose_to_fused_locked(self, fused: List[Dict[str, Any]]) -> None:
        for item in fused:
            pos = item.get("position_map") if isinstance(item.get("position_map"), dict) else {}
            cls = str(item.get("className", "unknown")).lower()
            tracked = self._find_existing_discovered(
                cls, self._as_float(pos.get("x")), self._as_float(pos.get("y")),
                self._as_optional_int(item.get("trackId", item.get("track_id"))),
            )
            if tracked is not None:
                self._copy_stable_pose_to_fused(item, tracked)
            else:
                item.setdefault("raw_position_map", dict(pos))
                item.setdefault("position_state", "raw")
                item.setdefault("avoidance_radius_m", self._avoidance_radius_for_class(cls))
                item.setdefault("avoidance_radius_includes_planner_inflate", True)

    def update_discovered_map_locked(self, fused: List[Dict[str, Any]]) -> None:
        if not self.mapping_enabled:
            return
        now = time.time()
        for obj in fused:
            class_name = str(obj.get("className", "unknown")).lower()
            if class_name not in self.add_classes:
                obj.setdefault("avoidance_radius_m", self._avoidance_radius_for_class(class_name))
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
                    raw_map_x=x,
                    raw_map_y=y,
                    raw_map_z=z,
                    position_state="candidate",
                    avoidance_radius_m=self._avoidance_radius_for_class(class_name),
                )
                self._refresh_confirmation(candidate, now)
                self.discovered.append(candidate)
                self._copy_stable_pose_to_fused(obj, candidate)
                continue

            # 모든 프레임의 raw 측정치는 저장하지만, 확정 후 map_x/y/z는 절대 덮어쓰지 않는다.
            existing.raw_map_x, existing.raw_map_y, existing.raw_map_z = x, y, z
            existing.distance_m = distance_m
            existing.confidence = max(existing.confidence, conf)
            existing.last_seen_wall = now
            existing.source = str(obj.get("source", existing.source))
            if existing.track_id is None and track_id is not None:
                existing.track_id = track_id
            if class_fixed_id is not None:
                existing.class_fixed_id = class_fixed_id
            existing.class_votes[class_name] = existing.class_votes.get(class_name, 0.0) + vote_weight
            existing.class_name = max(existing.class_votes.items(), key=lambda kv: kv[1])[0]
            existing.avoidance_radius_m = self._avoidance_radius_for_class(existing.class_name)

            if existing.is_confirmed and self.freeze_position_on_confirm:
                existing.position_state = "frozen"
            else:
                nx, ny, nz, accepted = self._candidate_filtered_position(existing, x, y, z)
                if accepted:
                    existing.map_x, existing.map_y, existing.map_z = nx, ny, nz
                    existing.observation_count += 1
                    existing.position_state = "candidate_tracking"
                else:
                    existing.position_state = "candidate_outlier_rejected"
                self._refresh_confirmation(existing, now)
                if existing.is_confirmed and self.freeze_position_on_confirm:
                    existing.position_state = "frozen"
            self._copy_stable_pose_to_fused(obj, existing)

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

    def _build_projection_context(
        self,
        image_w: int,
        image_h: int,
        projection_params: Optional[Dict[str, float]] = None,
    ) -> Optional[Dict[str, Any]]:
        if self.latest_info is None:
            return None
        params = dict(self.projection_params)
        if isinstance(projection_params, dict):
            params.update(projection_params)
        try:
            cam_pos, cam_yaw, cam_pitch, cam_roll = compute_camera_pose(self.latest_info, params)
            return {
                "camera_pos": cam_pos,
                "camera_yaw": cam_yaw,
                "camera_pitch": cam_pitch,
                "camera_roll": cam_roll,
                "pose_source": camera_pose_source(self.latest_info),
                "image_w": image_w,
                "image_h": image_h,
                "projection_params": params,
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
                ctx.get("projection_params", self.projection_params),
            )
            if projected is None:
                continue
            u, v, depth = projected
            if not (0 <= u < image_w and 0 <= v < image_h):
                continue
            out.append({"x": x, "y": y, "z": z, "u": u, "v": v, "depth": depth, "distance_m": distance, "raw": map_pos})
        return out

    @staticmethod
    def _bbox_intersection_area(
        a: Dict[str, float],
        b: Tuple[float, float, float, float],
    ) -> float:
        """Return intersection area for two image-space xyxy boxes."""
        ax1, ay1 = float(a.get("x1", 0.0)), float(a.get("y1", 0.0))
        ax2, ay2 = float(a.get("x2", 0.0)), float(a.get("y2", 0.0))
        bx1, by1, bx2, by2 = [float(v) for v in b]
        w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
        h = max(0.0, min(ay2, by2) - max(ay1, by1))
        return w * h

    @staticmethod
    def _expanded_image_bbox(
        bbox: List[float],
        margin: float,
        image_w: float,
        image_h: float,
    ) -> Tuple[float, float, float, float]:
        x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
        m = max(0.0, float(margin))
        return (
            max(0.0, x1 - m),
            max(0.0, y1 - m),
            min(float(image_w) - 1.0, x2 + m),
            min(float(image_h) - 1.0, y2 + m),
        )

    @staticmethod
    def _median_uv(points: List[Dict[str, float]]) -> Tuple[float, float]:
        if not points:
            return 0.0, 0.0
        us = np.asarray([float(p["u"]) for p in points], dtype=np.float64)
        vs = np.asarray([float(p["v"]) for p in points], dtype=np.float64)
        return float(np.median(us)), float(np.median(vs))

    def _project_clusters(self, ctx: Dict[str, Any], px: float, py: float, image_w: int, image_h: int) -> List[Dict[str, Any]]:
        """Project DBSCAN clusters using centroid + sampled member points.

        The DBSCAN publisher now sends a capped ``points_map`` sample for each
        cluster.  A cluster remains usable when its centroid is outside the
        image but one or more actual member points are inside it.  This avoids
        rejecting a wide/partial rock merely because its 3-D centroid is not
        visible to the camera.
        """
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
            map_pos = {
                "x": self._as_float(centroid.get("x")),
                "y": self._as_float(centroid.get("y")),
                "z": self._as_float(centroid.get("z")),
            }
            distance = math.hypot(map_pos["x"] - px, map_pos["y"] - py)
            if distance < self.min_fusion_range_m or distance > self.max_fusion_range_m:
                continue

            # Centroid projection is retained for compatibility and for final
            # object-position estimation, but no longer decides visibility alone.
            raw_pos = c.get("centroid_raw") if isinstance(c.get("centroid_raw"), dict) else map_to_raw_xyz(map_pos)
            centroid_projected = project_point(
                vec3_from_dict(raw_pos),
                ctx["camera_pos"],
                ctx["camera_yaw"],
                ctx["camera_pitch"],
                ctx["camera_roll"],
                image_w,
                image_h,
                ctx.get("projection_params", self.projection_params),
            )
            centroid_uv: Optional[Dict[str, float]] = None
            if centroid_projected is not None:
                cu, cv, cdepth = centroid_projected
                centroid_uv = {"u": float(cu), "v": float(cv), "depth": float(cdepth)}

            # Actual DBSCAN member-point samples are preferred over synthetic
            # 3-D bbox corners.  They describe the observed LiDAR surface.
            sample_map = c.get("points_map", [])
            projected_samples: List[Dict[str, float]] = []
            if self.cluster_projected_point_matching_enabled and isinstance(sample_map, list):
                for point in sample_map:
                    if not isinstance(point, dict):
                        continue
                    raw_point = map_to_raw_xyz(point)
                    projected = project_point(
                        vec3_from_dict(raw_point),
                        ctx["camera_pos"],
                        ctx["camera_yaw"],
                        ctx["camera_pitch"],
                        ctx["camera_roll"],
                        image_w,
                        image_h,
                        ctx.get("projection_params", self.projection_params),
                    )
                    if projected is None:
                        continue
                    u, v, depth = projected
                    projected_samples.append({"u": float(u), "v": float(v), "depth": float(depth)})

            # Legacy fallback: old cluster payloads do not have points_map.  In
            # that case project the 3-D cluster box corners to obtain a coarse
            # image-space footprint instead of dropping back to centroid only.
            if not projected_samples:
                bbox3d = c.get("bbox") if isinstance(c.get("bbox"), dict) else None
                if isinstance(bbox3d, dict):
                    try:
                        xs = [float(bbox3d["x_min"]), float(bbox3d["x_max"])]
                        ys = [float(bbox3d["y_min"]), float(bbox3d["y_max"])]
                        zs = [float(bbox3d["z_min"]), float(bbox3d["z_max"])]
                    except Exception:
                        xs = ys = zs = []
                    for x in xs:
                        for y in ys:
                            for z in zs:
                                projected = project_point(
                                    vec3_from_dict(map_to_raw_xyz({"x": x, "y": y, "z": z})),
                                    ctx["camera_pos"],
                                    ctx["camera_yaw"],
                                    ctx["camera_pitch"],
                                    ctx["camera_roll"],
                                    image_w,
                                    image_h,
                                    ctx.get("projection_params", self.projection_params),
                                )
                                if projected is None:
                                    continue
                                u, v, depth = projected
                                projected_samples.append({"u": float(u), "v": float(v), "depth": float(depth)})

            visible_samples = [
                p for p in projected_samples
                if 0.0 <= p["u"] < float(image_w) and 0.0 <= p["v"] < float(image_h)
            ]

            projected_bbox_2d: Optional[Dict[str, float]] = None
            if projected_samples:
                us = [p["u"] for p in projected_samples]
                vs = [p["v"] for p in projected_samples]
                projected_bbox_2d = {
                    "x1": float(min(us)),
                    "y1": float(min(vs)),
                    "x2": float(max(us)),
                    "y2": float(max(vs)),
                }

            centroid_visible = bool(
                centroid_uv is not None
                and 0.0 <= centroid_uv["u"] < float(image_w)
                and 0.0 <= centroid_uv["v"] < float(image_h)
            )
            if not centroid_visible and not visible_samples:
                continue

            # ``u/v`` are a stable display/debug point: centroid when visible,
            # otherwise median of visible actual member points.
            if centroid_visible and centroid_uv is not None:
                u, v, depth = centroid_uv["u"], centroid_uv["v"], centroid_uv["depth"]
            else:
                u, v = self._median_uv(visible_samples)
                depth = float(np.median([p["depth"] for p in visible_samples]))

            out.append({
                "id": int(self._as_float(c.get("id"), -1)),
                "count": count,
                "centroid": map_pos,
                "bbox": c.get("bbox"),
                "u": float(u),
                "v": float(v),
                "depth": float(depth),
                "distance_m": distance,
                "raw": c,
                "centroid_uv": centroid_uv,
                "centroid_visible": centroid_visible,
                "projected_samples": projected_samples,
                "visible_sample_count": len(visible_samples),
                "projected_bbox_2d": projected_bbox_2d,
            })
        return out

    def _normalize_deg_180(self, deg: float) -> float:
        return (float(deg) + 180.0) % 360.0 - 180.0

    def _body_angle_from_info(self, key: str) -> float:
        if not isinstance(self.latest_info, dict):
            return 0.0
        return self._normalize_deg_180(self._as_float(self.latest_info.get(key), 0.0))

    def _make_projection_tuning_debug(
        self,
        parsed_detections: List[ParsedDetection],
        image_w: int,
        image_h: int,
        px: float,
        py: float,
    ) -> Dict[str, Any]:
        """Return compact debug values for pitch/roll gain sign tuning.

        This does not affect fusion results. It re-projects the current DBSCAN
        clusters with body_pitch_gain/body_roll_gain set to +1 and -1, then
        compares the nearest projected cluster to the YOLO bbox anchor.
        Smaller center_norm/dist_px means better projection alignment.
        """
        if not parsed_detections or self.latest_info is None:
            return {}

        pose_source = camera_pose_source(self.latest_info)
        if pose_source == "lidarRotation":
            # lidarRotation is the active sensor pose, so changing legacy body
            # gains cannot affect projection. Keep the pose source explicit, but
            # still emit the active-pose alignment result so calibration can be
            # diagnosed from /tank/debug/fusion/status.
            ctx = self._build_projection_context(image_w, image_h)
            active_clusters = (
                self._project_clusters(ctx, px, py, image_w, image_h)
                if ctx is not None else []
            )
            active_best = self._best_projection_alignment(
                parsed_detections,
                active_clusters,
                image_w,
                image_h,
            )
            return {
                "projection_tuning_debug": {
                    "pose_source": pose_source,
                    "body_gains_active": False,
                    "note": "lidarRotation pose is active; body gain +/- tuning is skipped",
                    "camera_pose_deg": ({
                        "yaw": float(ctx["camera_yaw"]),
                        "pitch": float(ctx["camera_pitch"]),
                        "roll": float(ctx["camera_roll"]),
                    } if ctx else None),
                    # The nearest centroid-to-bbox pairing under the exact active
                    # lidarRotation pose. It does not alter fusion selection.
                    "current_best": active_best,
                    "active_projected_cluster_count": len(active_clusters),
                    "cluster_bbox_margin_px": float(self.cluster_bbox_margin_px),
                    "cluster_match_max_center_norm": float(self.cluster_match_max_center_norm),
                    "cluster_match_max_score": float(self.cluster_match_max_score),
                }
            }

        current_params = dict(self.projection_params)
        current_pitch_gain = self._as_float(current_params.get("body_pitch_gain"), 0.0)
        current_roll_gain = self._as_float(current_params.get("body_roll_gain"), 0.0)
        body_pitch_deg = self._body_angle_from_info("playerBodyY")
        body_roll_deg = self._body_angle_from_info("playerBodyZ")

        def project_with(overrides: Dict[str, float]) -> List[Dict[str, Any]]:
            params = dict(current_params)
            params.update(overrides)
            ctx = self._build_projection_context(image_w, image_h, params)
            if ctx is None:
                return []
            return self._project_clusters(ctx, px, py, image_w, image_h)

        current_clusters = project_with({})
        pitch_plus_clusters = project_with({"body_pitch_gain": 1.0})
        pitch_minus_clusters = project_with({"body_pitch_gain": -1.0})
        roll_plus_clusters = project_with({"body_roll_gain": 1.0})
        roll_minus_clusters = project_with({"body_roll_gain": -1.0})

        current_best = self._best_projection_alignment(parsed_detections, current_clusters, image_w, image_h)
        pitch_plus_best = self._best_projection_alignment(parsed_detections, pitch_plus_clusters, image_w, image_h)
        pitch_minus_best = self._best_projection_alignment(parsed_detections, pitch_minus_clusters, image_w, image_h)
        roll_plus_best = self._best_projection_alignment(parsed_detections, roll_plus_clusters, image_w, image_h)
        roll_minus_best = self._best_projection_alignment(parsed_detections, roll_minus_clusters, image_w, image_h)

        def score(best: Optional[Dict[str, Any]]) -> float:
            if not isinstance(best, dict):
                return float("inf")
            return self._as_float(best.get("center_norm"), float("inf"))

        pitch_score_plus = score(pitch_plus_best)
        pitch_score_minus = score(pitch_minus_best)
        roll_score_plus = score(roll_plus_best)
        roll_score_minus = score(roll_minus_best)

        pitch_recommended: Optional[float]
        roll_recommended: Optional[float]
        pitch_note = "ok"
        roll_note = "ok"
        angle_threshold_deg = 0.20
        if abs(body_pitch_deg) < angle_threshold_deg:
            pitch_recommended = None
            pitch_note = "body_pitch_deg_too_small_for_reliable_sign_decision"
        elif not math.isfinite(pitch_score_plus) and not math.isfinite(pitch_score_minus):
            pitch_recommended = None
            pitch_note = "no_projected_cluster_for_pitch_comparison"
        else:
            pitch_recommended = 1.0 if pitch_score_plus <= pitch_score_minus else -1.0

        if abs(body_roll_deg) < angle_threshold_deg:
            roll_recommended = None
            roll_note = "body_roll_deg_too_small_for_reliable_sign_decision"
        elif not math.isfinite(roll_score_plus) and not math.isfinite(roll_score_minus):
            roll_recommended = None
            roll_note = "no_projected_cluster_for_roll_comparison"
        else:
            roll_recommended = 1.0 if roll_score_plus <= roll_score_minus else -1.0

        return {
            "projection_tuning_debug": {
                "body_pitch_deg": float(body_pitch_deg),
                "body_roll_deg": float(body_roll_deg),
                "current_body_pitch_gain": float(current_pitch_gain),
                "current_body_roll_gain": float(current_roll_gain),
                "cluster_bbox_margin_px": float(self.cluster_bbox_margin_px),
                "current_best": current_best,
                "pitch_debug": {
                    "best_plus": pitch_plus_best,
                    "best_minus": pitch_minus_best,
                    "dist_px_plus": None if pitch_plus_best is None else pitch_plus_best.get("dist_px"),
                    "dist_px_minus": None if pitch_minus_best is None else pitch_minus_best.get("dist_px"),
                    "center_norm_plus": None if pitch_plus_best is None else pitch_plus_best.get("center_norm"),
                    "center_norm_minus": None if pitch_minus_best is None else pitch_minus_best.get("center_norm"),
                    "recommended_body_pitch_gain": pitch_recommended,
                    "note": pitch_note,
                },
                "roll_debug": {
                    "best_plus": roll_plus_best,
                    "best_minus": roll_minus_best,
                    "dist_px_plus": None if roll_plus_best is None else roll_plus_best.get("dist_px"),
                    "dist_px_minus": None if roll_minus_best is None else roll_minus_best.get("dist_px"),
                    "center_norm_plus": None if roll_plus_best is None else roll_plus_best.get("center_norm"),
                    "center_norm_minus": None if roll_minus_best is None else roll_minus_best.get("center_norm"),
                    "recommended_body_roll_gain": roll_recommended,
                    "note": roll_note,
                },
            }
        }

    def _cluster_detection_geometry(
        self,
        parsed: ParsedDetection,
        cluster: Dict[str, Any],
        image_w: float,
        image_h: float,
    ) -> Optional[Dict[str, Any]]:
        """Build detection-specific 2-D evidence for a projected cluster.

        Priority is actual projected member-point hits.  Centroid-only remains
        a compatibility path, while 2-D box overlap is a last fallback for old
        cluster messages that lack sampled points.
        """
        bbox = parsed.bbox
        margin = float(self.cluster_bbox_margin_px)
        expanded = self._expanded_image_bbox(bbox, margin, image_w, image_h)
        samples = cluster.get("projected_samples", [])
        samples = samples if isinstance(samples, list) else []
        hit_samples = [
            p for p in samples
            if isinstance(p, dict)
            and point_inside_bbox(float(p.get("u", 0.0)), float(p.get("v", 0.0)), bbox, margin, image_w, image_h)
        ]
        required_hits = min(
            self.cluster_projected_min_points_in_detection,
            max(1, len(samples)),
        )

        centroid_uv = cluster.get("centroid_uv")
        centroid_inside = bool(
            isinstance(centroid_uv, dict)
            and point_inside_bbox(
                float(centroid_uv.get("u", 0.0)),
                float(centroid_uv.get("v", 0.0)),
                bbox,
                margin,
                image_w,
                image_h,
            )
        )

        evidence = None
        if hit_samples and len(hit_samples) >= required_hits:
            ru, rv = self._median_uv(hit_samples)
            evidence = "sample_points"
        elif centroid_inside and isinstance(centroid_uv, dict):
            ru = float(centroid_uv["u"])
            rv = float(centroid_uv["v"])
            evidence = "centroid"
        else:
            # Do not allow projected-bbox overlap to override real sampled
            # points that say the cluster misses the detection.  This fallback
            # exists only for older publishers without points_map.
            bbox2d = cluster.get("projected_bbox_2d")
            if (
                self.cluster_projected_bbox_fallback_enabled
                and not samples
                and isinstance(bbox2d, dict)
            ):
                overlap = self._bbox_intersection_area(bbox2d, expanded)
                if overlap >= self.cluster_projected_bbox_min_overlap_px2:
                    ix1 = max(float(bbox2d["x1"]), expanded[0])
                    iy1 = max(float(bbox2d["y1"]), expanded[1])
                    ix2 = min(float(bbox2d["x2"]), expanded[2])
                    iy2 = min(float(bbox2d["y2"]), expanded[3])
                    ru, rv = 0.5 * (ix1 + ix2), 0.5 * (iy1 + iy2)
                    evidence = "projected_bbox_fallback"
                else:
                    return None
            else:
                return None

        return {
            "u": float(ru),
            "v": float(rv),
            "evidence": evidence,
            "sample_hit_count": int(len(hit_samples)),
            "sample_point_count": int(len(samples)),
            "required_sample_hits": int(required_hits),
            "centroid_inside_bbox": bool(centroid_inside),
            "projected_bbox_2d": cluster.get("projected_bbox_2d"),
        }

    def _best_projection_alignment(
        self,
        parsed_detections: List[ParsedDetection],
        projected_clusters: List[Dict[str, Any]],
        image_w: int,
        image_h: int,
    ) -> Optional[Dict[str, Any]]:
        if not parsed_detections or not projected_clusters:
            return None
        best: Optional[Dict[str, Any]] = None
        best_score = float("inf")
        for det_index, parsed in enumerate(parsed_detections):
            ax, ay, bw, bh = self._bbox_anchor_uv(parsed.bbox, parsed.class_name)
            half_w = max(1.0, 0.5 * bw)
            half_h = max(1.0, 0.5 * bh)
            for cluster in projected_clusters:
                geometry = self._cluster_detection_geometry(parsed, cluster, float(image_w), float(image_h))
                if geometry is None:
                    # Debug still needs a nearest cluster when no candidate is
                    # valid.  Use its display representative, not centroid only.
                    u = self._as_float(cluster.get("u"), 0.0)
                    v = self._as_float(cluster.get("v"), 0.0)
                    evidence = "no_overlap"
                    sample_hit_count = 0
                    sample_point_count = len(cluster.get("projected_samples", []) or [])
                    inside_margin = False
                    projected_bbox_2d = cluster.get("projected_bbox_2d")
                else:
                    u = float(geometry["u"])
                    v = float(geometry["v"])
                    evidence = str(geometry["evidence"])
                    sample_hit_count = int(geometry["sample_hit_count"])
                    sample_point_count = int(geometry["sample_point_count"])
                    inside_margin = True
                    projected_bbox_2d = geometry.get("projected_bbox_2d")
                du = u - ax
                dv = v - ay
                dx_norm = abs(du) / half_w
                dy_norm = abs(dv) / half_h
                center_norm = math.sqrt(dx_norm * dx_norm + dy_norm * dy_norm)
                dist_px = math.hypot(du, dv)
                score = center_norm + 1e-6 * dist_px
                if score < best_score:
                    x1, y1, x2, y2 = [float(x) for x in parsed.bbox[:4]]
                    best_score = score
                    best = {
                        "det_index": int(det_index),
                        "class_name": parsed.class_name,
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                        "bbox_center": [float(0.5 * (x1 + x2)), float(0.5 * (y1 + y2))],
                        "bbox_anchor": [float(ax), float(ay)],
                        "cluster_id": int(cluster.get("id", -1)),
                        "cluster_count": int(cluster.get("count", 0)),
                        "projected_uv": [float(u), float(v)],
                        "delta_uv_from_anchor": [float(du), float(dv)],
                        "dist_px": float(dist_px),
                        "center_norm": float(center_norm),
                        "dx_norm": float(dx_norm),
                        "dy_norm": float(dy_norm),
                        "inside_bbox_with_margin": bool(inside_margin),
                        "match_evidence": evidence,
                        "sample_hit_count": int(sample_hit_count),
                        "sample_point_count": int(sample_point_count),
                        "projected_bbox_2d": projected_bbox_2d,
                        "distance_m": float(cluster.get("distance_m", 0.0)),
                    }
        return best

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
        geometry = self._cluster_detection_geometry(parsed, cluster, image_w, image_h)
        if geometry is None:
            return None

        u = float(geometry["u"])
        v = float(geometry["v"])
        ax, ay, bw, bh = self._bbox_anchor_uv(bbox, class_name)
        half_w = max(1.0, 0.5 * bw)
        half_h = max(1.0, 0.5 * bh)
        dx_norm = abs(u - ax) / half_w
        dy_norm = abs(v - ay) / half_h

        if class_name == "person":
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
            "match_evidence": str(geometry["evidence"]),
            "sample_hit_count": int(geometry["sample_hit_count"]),
            "sample_point_count": int(geometry["sample_point_count"]),
            "required_sample_hits": int(geometry["required_sample_hits"]),
            "projected_bbox_2d": geometry.get("projected_bbox_2d"),
            "projection_uv": {"u": float(u), "v": float(v)},
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
            "cluster_best_match_evidence": candidates[0].get("match_evidence") if candidates else None,
            "cluster_best_sample_hit_count": int(candidates[0].get("sample_hit_count", 0)) if candidates else 0,
            "cluster_best_sample_point_count": int(candidates[0].get("sample_point_count", 0)) if candidates else 0,
            "cluster_best_projected_bbox_2d": candidates[0].get("projected_bbox_2d") if candidates else None,
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
                    "projection_uv": cand.get("projection_uv", {"u": float(cluster.get("u", 0.0)), "v": float(cluster.get("v", 0.0))}),
                    "cluster_bbox": cluster.get("bbox"),
                    "cluster_projected_bbox_2d": cand.get("projected_bbox_2d"),
                    "cluster_match_evidence": cand.get("match_evidence", "centroid"),
                    "cluster_sample_hit_count": int(cand.get("sample_hit_count", 0)),
                    "cluster_sample_point_count": int(cand.get("sample_point_count", 0)),
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
        candidates = []
        parsed = ParsedDetection(class_name=class_name, confidence=confidence, bbox=bbox)
        for c in clusters:
            if c["id"] in assigned_ids:
                continue
            geometry = self._cluster_detection_geometry(parsed, c, image_w, image_h)
            if geometry is not None:
                candidates.append((c, geometry))
        if not candidates:
            return None
        candidates.sort(key=lambda item: (item[0]["distance_m"], -item[0]["count"]))
        best, best_geometry = candidates[0]
        if best["id"] >= 0:
            assigned_ids.add(best["id"])
        pos = best["centroid"]
        return self._make_fused_object(
            class_name, confidence, bbox, pos["x"], pos["y"], pos["z"], px, py, pz, camera_heading,
            source="yolo_lidar_projection_cluster_fusion", matched_lidar_points=int(best["count"]),
            used_lidar_points=int(best["count"]), extra={
                "cluster_id": best["id"],
                "projection_uv": {"u": best_geometry["u"], "v": best_geometry["v"]},
                "cluster_bbox": best.get("bbox"),
                "cluster_projected_bbox_2d": best_geometry.get("projected_bbox_2d"),
                "cluster_match_evidence": best_geometry.get("evidence", "centroid"),
                "cluster_sample_hit_count": int(best_geometry.get("sample_hit_count", 0)),
                "cluster_sample_point_count": int(best_geometry.get("sample_point_count", 0)),
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
            "distance_m": distance_m, "nearest_distance_m": distance_m,
            "position_map": {"x": x, "y": y, "z": z, "frame_id": self.map_frame},
            "raw_position_map": {"x": x, "y": y, "z": z, "frame_id": self.map_frame},
            "position_state": "raw",
            "avoidance_radius_m": self._avoidance_radius_for_class(class_name),
            "avoidance_radius_includes_planner_inflate": True,
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
            if self.freeze_position_on_confirm:
                obj.position_state = "frozen"

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
            radius = self._as_float(obj.get("avoidance_radius_m"), self._avoidance_radius_for_class(cls))
            state = str(obj.get("position_state", "raw"))
            label = f"LIVE {cls} {dist:.1f}m r={radius:.1f}m [{state}]\n{source.replace('yolo_lidar_', '')}"
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
            label = f"{status} {obj.class_name} r={obj.avoidance_radius_m:.1f}m\nobs={obj.observation_count} conf={obj.confidence:.2f} [{obj.position_state}]"
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

    # ------------------------------------------------------------------
    # 정찰 전용: 미분류 후보 집계 → controller가 감속/dwell·포탑 step-stare로 쓸 관측요청 발행
    # ------------------------------------------------------------------
    def compute_observe_candidates_locked(self) -> List[Dict[str, Any]]:
        """사거리 내 LiDAR 클러스터 중 '아직 확정 분류 안 된 것'(=미분류 후보)을 집계한다.
        각 후보에 bearing/in_forward_fov/라이다 크기 prior/우선순위를 매겨, controller가
        전방 후보엔 감속·dwell, 옆 후보엔 포탑 step-stare 큐로 쓰게 한다. 위치는 안 바꾼다(읽기만)."""
        if self.player_pose is None or not isinstance(self.latest_clusters_payload, dict):
            return []
        clusters = self.latest_clusters_payload.get("clusters", [])
        if not isinstance(clusters, list):
            return []
        px = float(self.player_pose.pose.position.x)
        py = float(self.player_pose.pose.position.y)
        cam_heading = self._camera_heading_deg()
        half_fov = max(1.0, self.hfov_deg * 0.5)
        out: List[Dict[str, Any]] = []
        for c in clusters:
            if not isinstance(c, dict):
                continue
            centroid = c.get("centroid")
            if not isinstance(centroid, dict):
                continue
            x = self._as_float(centroid.get("x"))
            y = self._as_float(centroid.get("y"))
            z = self._as_float(centroid.get("z"))
            dx, dy = x - px, y - py
            dist = math.hypot(dx, dy)
            if dist < self.min_fusion_range_m or dist > self.max_fusion_range_m:
                continue
            if self._near_confirmed_classified(x, y):   # 이미 분류·확정됨 → 후보 아님
                continue
            if self._near_static_obstacle(x, y):        # 맵에 이미 있는 장애물(나무 등) → 후보 아님
                continue
            size_class, width_m, height_m = self._cluster_size_prior(c.get("bbox"))
            global_bearing = math.degrees(math.atan2(dx, dy))
            rel_bearing = self._normalize_angle_deg(global_bearing - cam_heading)
            in_fov = abs(rel_bearing) <= half_fov
            # 포탑 응시 대상 = 전방-대각만(FOV 밖 ~ max_bearing). 후방(지나친) 후보는 제외해 포탑이 뒤를 안 봄.
            turret_eligible = (not in_fov) and (abs(rel_bearing) <= self.observe_turret_max_bearing_deg)
            out.append({
                "x": x, "y": y, "z": z,
                "distance_m": round(dist, 2),
                "bearing_global_deg": round(global_bearing, 1),
                "bearing_relative_deg": round(rel_bearing, 1),
                "in_forward_fov": bool(in_fov),
                "turret_eligible": bool(turret_eligible),
                "size_class": size_class,
                "width_m": round(width_m, 2),
                "height_m": round(height_m, 2),
                "priority": round(self._observe_priority(size_class, dist), 3),
            })
        out.sort(key=lambda d: d["priority"], reverse=True)
        return out

    def _near_confirmed_classified(self, x: float, y: float) -> bool:
        for obj in self.discovered:
            if not obj.is_confirmed:
                continue
            if math.hypot(obj.map_x - x, obj.map_y - y) <= self.observe_classified_radius_m:
                return True
        return False

    def _near_static_obstacle(self, x: float, y: float) -> bool:
        """맵에 이미 있는 정적 장애물(나무 등)에 가까운가 — recon 관측 후보에서 제외('맵에 없는 것'만 관측)."""
        r = self.observe_static_exclude_radius_m
        for obj in self.static_objects:
            if math.hypot(obj.map_x - x, obj.map_y - y) <= r:
                return True
        return False

    def _cluster_size_prior(self, bbox: Any) -> Tuple[str, float, float]:
        """라이다 map-frame bbox(z=높이)로 거친 크기 추정 → 위협-크기 prior(분류 아님, 우선순위용)."""
        if not isinstance(bbox, dict):
            return "unknown", 0.0, 0.0
        wx = abs(self._as_float(bbox.get("x_max")) - self._as_float(bbox.get("x_min")))
        wy = abs(self._as_float(bbox.get("y_max")) - self._as_float(bbox.get("y_min")))
        width = max(wx, wy)
        height = abs(self._as_float(bbox.get("z_max")) - self._as_float(bbox.get("z_min")))
        if width >= self.observe_large_min_size_m or height >= self.observe_large_min_height_m:
            return "large", width, height      # 집/초소-크기
        if width >= self.observe_vehicle_min_size_m and height >= self.observe_min_height_m:
            return "vehicle", width, height    # 전차/차-크기
        return "small", width, height          # 바위-크기

    def _observe_priority(self, size_class: str, dist: float) -> float:
        threat_w = {"large": 1.0, "vehicle": 0.9, "unknown": 0.5, "small": 0.3}.get(size_class, 0.5)
        prox = 1.0 - min(1.0, dist / max(1.0, self.max_fusion_range_m))   # 가까울수록 ↑
        return threat_w * (0.5 + 0.5 * prox)

    def publish_observe_request(self, candidates: List[Dict[str, Any]]) -> None:
        if self.observe_request_pub is None:
            return
        pending_fov = [c for c in candidates if c["in_forward_fov"]]
        # 포탑은 전방-대각(turret_eligible)만 응시 — 후방(지나친) 후보 제외. candidates는 priority 내림차순.
        pending_turret = [c for c in candidates if c.get("turret_eligible")]
        side_total = sum(1 for c in candidates if not c["in_forward_fov"])
        payload = {
            "timestamp_wall": time.time(),
            "frame_id": self.map_frame,
            "mission_type": self.mission_type,
            "count": len(candidates),
            "has_pending_fov": bool(pending_fov),       # controller: 감속/dwell
            "has_pending_side": bool(pending_turret),   # controller: 포탑 step-stare (전방-대각만)
            "side_total": side_total,                   # 진단: 전체 옆 후보(후방 포함)
            "best_side_bearing_rel_deg": pending_turret[0]["bearing_relative_deg"] if pending_turret else None,
            "best_side_bearing_global_deg": pending_turret[0]["bearing_global_deg"] if pending_turret else None,
            "candidates": candidates[: self.observe_max_publish],
        }
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        self.observe_request_pub.publish(msg)

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