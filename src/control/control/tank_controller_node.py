# -*- coding: utf-8 -*-
"""
최신 TankSimulation Flask 컨트롤러를 ROS2로 옮긴 컨트롤러.

제어 정책은 의도적으로 팀 서버 로직을 따른다:
- target yaw = atan2(dx, dy). 여기서 map x/y는 Unity x/z에 대응한다.
- 조향 명령은 yaw error에 비례하는 weight의 A/D다.
- 속도 명령은 yaw error가 크면 동역학 데이터 기반으로 감속하고 최종 goal에서 정지한다.
- 큰 yaw error에서는 STOP 제자리 회전 대신 W 저속 crawl turn을 사용한다.
- 지역최소/끼임(stuck) 탈출: 후진 후 저속 선회.

공식 Tank Challenge /get_action JSON을 /tank/control/command로 발행한다.
"""

import json
import math
import os
import time
from typing import Any, Dict, Optional, Tuple, List

import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped, Vector3Stamped
from nav_msgs.msg import Path as NavPath
from rclpy.node import Node
from std_msgs.msg import String

from control.config import (
    CONTROLLER_HZ,
    ENABLE_LOCAL_TARGET,
    ENABLE_STUCK_ESCAPE,
    ESCAPE_REVERSE_SEC,
    ESCAPE_TURN_SEC,
    GOAL_TOLERANCE,
    HEADING_DEADBAND_DEG,
    MAX_AD_WEIGHT,
    MIN_AD_WEIGHT,
    ROTATE_IN_PLACE_ANGLE_DEG,
    CRAWL_PIVOT_ANGLE_DEG,
    CRAWL_PIVOT_WS_WEIGHT,
    CRAWL_TURN_WS_WEIGHT,
    SLOWDOWN_ANGLE_DEG,
    STOP_DISTANCE,
    STRAIGHT_WS_WEIGHT,
    STEERING_FULL_ERROR_DEG,
    STEERING_KD,
    STUCK_CHECK_PERIOD,
    STUCK_MIN_MOVEMENT,
    TARGET_TTL_SEC,
    TOPIC_COLLISION_EVENT,
    TOPIC_CONTROL_COMMAND,
    TOPIC_CONTROL_STATUS,
    TOPIC_GOAL_POSE,
    TOPIC_LOCAL_TARGET_POSE,
    TOPIC_LOOKAHEAD_POSE,
    TOPIC_PLAYER_POSE,
    TOPIC_PLAYER_STATE,
    TURN_WS_WEIGHT,
    TOPIC_PATH_POINTS,
    ENABLE_PLANNER_SPEED_PROFILE,
    PLANNER_SPEED_PROFILE_TTL_SEC,
    PLANNER_GOAL_STOP_DISTANCE_M,
)


def normalize_angle(angle: float) -> float:
    while angle > 180.0:
        angle -= 360.0
    while angle < -180.0:
        angle += 360.0
    return angle


def get_distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def empty_action() -> Dict[str, Any]:
    return {
        "moveWS": {"command": "STOP", "weight": 1.0},
        "moveAD": {"command": "", "weight": 0.0},
        "turretQE": {"command": "", "weight": 0.0},
        "turretRF": {"command": "", "weight": 0.0},
        "fire": False,
    }


class TeamPathControllerNode(Node):
    def __init__(self) -> None:
        super().__init__("tank_team_path_controller_node")
        default_param_file = ""
        try:
            default_param_file = os.path.join(get_package_share_directory("control"), "config", "tank_parameters.yaml")
        except Exception:
            pass

        self.declare_parameter("tank_param_file", default_param_file)
        self.declare_parameter("controller_hz", 10.0)
        self.declare_parameter("max_speed", 25.0)
        self.declare_parameter("max_yaw_rate", 2.0)
        self.declare_parameter("goal_tolerance", 10.0)
        # Goal 도착 후 simulator pause는 일반 정찰/주행에는 유용하지만,
        # scenario2의 정지-조준-발사 단계에서는 /get_action을 계속 받아야 한다.
        self.declare_parameter("pause_on_goal_reached", True)
        # Keep the controller alive at a stop-aim-fire checkpoint.  The legacy
        # default preserves the previous behavior for ordinary one-way runs.
        self.declare_parameter("exit_on_goal_reached", True)
        self.declare_parameter("enable_local_target", ENABLE_LOCAL_TARGET)
        self.declare_parameter("target_ttl_sec", TARGET_TTL_SEC)
        self.declare_parameter("mission_type", "mission")  # 'recon', 'mission', 'return'
        # 단일 /tank/control/command 발행자를 유지하기 위한 포탑 override 입력.
        # ballistic_turret_node 같은 교전 노드는 이 topic만 발행하고, 이 컨트롤러가
        # 주행 명령과 합성해 simulator로 보낸다.
        self.declare_parameter("turret_override_topic", "/tank/turret/override")
        self.declare_parameter("turret_override_ttl_sec", 0.35)
        self.declare_parameter("heading_deadband_deg", HEADING_DEADBAND_DEG)
        self.declare_parameter("steering_full_error_deg", STEERING_FULL_ERROR_DEG)
        self.declare_parameter("min_ad_weight", MIN_AD_WEIGHT)
        self.declare_parameter("max_ad_weight", MAX_AD_WEIGHT)
        self.declare_parameter("steering_kd", STEERING_KD)
        self.declare_parameter("straight_ws_weight", STRAIGHT_WS_WEIGHT)
        self.declare_parameter("turn_ws_weight", TURN_WS_WEIGHT)
        self.declare_parameter("rotate_in_place_angle_deg", ROTATE_IN_PLACE_ANGLE_DEG)
        self.declare_parameter("crawl_pivot_angle_deg", CRAWL_PIVOT_ANGLE_DEG)
        self.declare_parameter("crawl_pivot_ws_weight", CRAWL_PIVOT_WS_WEIGHT)
        self.declare_parameter("crawl_turn_ws_weight", CRAWL_TURN_WS_WEIGHT)
        self.declare_parameter("slowdown_angle_deg", SLOWDOWN_ANGLE_DEG)
        self.declare_parameter("stop_distance", STOP_DISTANCE)
        self.declare_parameter("enable_stuck_escape", ENABLE_STUCK_ESCAPE)
        self.declare_parameter("stuck_check_period", STUCK_CHECK_PERIOD)
        self.declare_parameter("stuck_min_movement", STUCK_MIN_MOVEMENT)
        self.declare_parameter("escape_reverse_sec", ESCAPE_REVERSE_SEC)
        self.declare_parameter("escape_turn_sec", ESCAPE_TURN_SEC)
        self.declare_parameter("enable_safety_speed_limit", True)
        self.declare_parameter("safety_status_topic", "/tank/potential/safety_status")
        self.declare_parameter("safety_status_ttl_sec", 1.0)
        # APF 합벡터 기반 전차식 정지 선회 제어.
        # APF가 원하는 이동방향과 현재 차체 heading 차이가 크면 W를 떼고 A/D만 수행한다.
        self.declare_parameter("enable_apf_vector_stop_pivot", True)
        self.declare_parameter("apf_stop_angle_deg", 35.0)
        self.declare_parameter("apf_slow_angle_deg", 20.0)
        self.declare_parameter("apf_stop_distance", 9.0)
        self.declare_parameter("apf_ttc_stop_sec", 2.0)
        self.declare_parameter("apf_ttc_slow_sec", 3.0)
        self.declare_parameter("apf_min_speed_for_ttc", 0.8)
        self.declare_parameter("apf_slow_ws_weight", 0.12)
        # APF STOP pivot이 오래 지속되면 제자리에서 빙빙 도는 루프가 생긴다.
        # 짧게 차체 방향만 바꾸고, 이후 cooldown 동안은 저속 전진 탈출을 허용한다.
        self.declare_parameter("apf_stop_pivot_max_sec", 1.15)
        self.declare_parameter("apf_stop_pivot_release_angle_deg", 24.0)
        self.declare_parameter("apf_stop_pivot_cooldown_sec", 1.0)
        # 장애물 좌/우 위치 기반 회피 방향 강제.
        # 장애물이 오른쪽이면 A, 왼쪽이면 D를 우선 사용해 장애물 쪽으로 감기는 루프를 막는다.
        self.declare_parameter("prefer_turn_away_from_nearest_obstacle", True)
        self.declare_parameter("away_turn_lock_sec", 1.2)
        # A/D 반복 토글 방지: 짧은 시간 안에 조향 방향이 바뀌면 기존 방향을 잠시 유지한다.
        self.declare_parameter("enable_steering_direction_lock", True)
        self.declare_parameter("steering_direction_lock_sec", 0.9)
        # 장애물이 이미 옆으로 지나가는 상황에서는 STOP pivot을 하지 않고 저속 W로 통과한다.
        # nearest obstacle bearing이 큰 경우(좌/우 측면)는 정면 충돌 위험보다 side-pass 상황으로 본다.
        self.declare_parameter("enable_side_pass_forward", True)
        self.declare_parameter("side_pass_bearing_deg", 58.0)
        self.declare_parameter("side_pass_min_distance", 4.2)
        self.declare_parameter("side_pass_ws_weight", 0.22)
        self.declare_parameter("front_stop_bearing_deg", 42.0)
        self.declare_parameter("hard_stop_distance", 3.8)
        self.declare_parameter("turn_away_hard_distance", 4.8)
        # 지그재그 장애물 사이에서 nearest side가 좌/우로 바뀌며 A/D가 반복되는 현상 억제.
        self.declare_parameter("enable_ad_oscillation_guard", True)
        self.declare_parameter("ad_flip_window_sec", 3.0)
        self.declare_parameter("ad_flip_threshold", 3)
        self.declare_parameter("ad_oscillation_hold_sec", 2.0)
        self.declare_parameter("ad_oscillation_slow_ws_weight", 0.18)
        # 급격한 경로 꺾임 대응: yaw error가 큰데 W를 계속 누르면 큰 원을 그리며 벗어난다.
        # 일정 각도 이상이면 W를 떼고 A/D만 눌러 차체 방향을 먼저 맞춘다.
        self.declare_parameter("enable_sharp_turn_stop_pivot", True)
        self.declare_parameter("sharp_turn_stop_angle_deg", 70.0)
        self.declare_parameter("sharp_turn_release_angle_deg", 30.0)
        self.declare_parameter("sharp_turn_min_target_distance", 4.0)
        self.declare_parameter("sharp_turn_max_sec", 1.25)
        self.declare_parameter("sharp_turn_cooldown_sec", 0.6)
        self.declare_parameter("sharp_turn_block_when_apf_side_pass", True)
        # 실제 속도 기반 오버슛 방지. 입력 weight를 줄여도 관성으로 코너/장애물에 밀고 들어가면
        # yaw error와 현재 속도를 보고 STOP/S braking을 짧게 건다.
        self.declare_parameter("enable_turn_overspeed_guard", True)
        self.declare_parameter("turn_overspeed_angle_deg", 35.0)
        self.declare_parameter("turn_overspeed_speed_mps", 2.8)
        self.declare_parameter("turn_overspeed_hard_angle_deg", 60.0)
        self.declare_parameter("turn_overspeed_hard_speed_mps", 4.5)
        self.declare_parameter("turn_overspeed_reverse_weight", 0.38)
        self.declare_parameter("turn_overspeed_slow_ws_weight", 0.10)
        self.declare_parameter("danger_obstacle_brake_speed_mps", 1.0)
        self.declare_parameter("danger_obstacle_reverse_weight", 0.50)
        # APF/local target이 현재 위치 너무 가까이 있거나 뒤쪽으로 튀면 차체가 제자리에서 좌우로 돈다.
        # 이런 경우에는 현재 global path에서 더 앞쪽 point를 다시 선택해 target을 안정화한다.
        self.declare_parameter("enable_forward_target_guard", True)
        self.declare_parameter("forward_guard_min_target_distance", 6.0)
        self.declare_parameter("forward_guard_yaw_error_deg", 85.0)
        self.declare_parameter("forward_guard_target_distance", 12.0)
        self.declare_parameter("forward_guard_max_search_points", 120)
        self.declare_parameter("forward_guard_allow_in_danger", False)
        # A*가 발행하는 곡선/감속 profile. APF의 desired_motion은 여기서 사용하지 않는다.
        self.declare_parameter("enable_planner_speed_profile", ENABLE_PLANNER_SPEED_PROFILE)
        self.declare_parameter("planner_path_points_topic", TOPIC_PATH_POINTS)
        self.declare_parameter("planner_speed_profile_ttl_sec", PLANNER_SPEED_PROFILE_TTL_SEC)
        self.declare_parameter("planner_goal_stop_distance_m", PLANNER_GOAL_STOP_DISTANCE_M)

        self.enable_local_target = bool(self.get_parameter("enable_local_target").value)
        self.target_ttl_sec = float(self.get_parameter("target_ttl_sec").value)
        self.goal_tolerance = float(self.get_parameter("goal_tolerance").value)
        self.heading_deadband_deg = float(self.get_parameter("heading_deadband_deg").value)
        self.steering_full_error_deg = max(1.0, float(self.get_parameter("steering_full_error_deg").value))
        self.min_ad_weight = float(self.get_parameter("min_ad_weight").value)
        self.max_ad_weight = float(self.get_parameter("max_ad_weight").value)
        self.controller_hz = max(1.0, float(self.get_parameter("controller_hz").value))
        self.steering_kd = float(self.get_parameter("steering_kd").value)
        # PD(rate feedback)용 상태: 직전 헤딩값(yaw_rate 산출). None이면 첫 틱 rate=0.
        self._last_current_yaw: Optional[float] = None
        self.straight_ws_weight = float(self.get_parameter("straight_ws_weight").value)
        self.turn_ws_weight = float(self.get_parameter("turn_ws_weight").value)
        self.rotate_in_place_angle_deg = float(self.get_parameter("rotate_in_place_angle_deg").value)
        self.crawl_pivot_angle_deg = float(self.get_parameter("crawl_pivot_angle_deg").value)
        self.crawl_pivot_ws_weight = float(self.get_parameter("crawl_pivot_ws_weight").value)
        self.crawl_turn_ws_weight = float(self.get_parameter("crawl_turn_ws_weight").value)
        self.max_yaw_rate = float(self.get_parameter("max_yaw_rate").value)
        self.goal_tolerance = float(self.get_parameter("goal_tolerance").value)
        self.pause_on_goal_reached = bool(self.get_parameter("pause_on_goal_reached").value)
        self.exit_on_goal_reached = bool(self.get_parameter("exit_on_goal_reached").value)
        self.enable_local_target = bool(self.get_parameter("enable_local_target").value)
        self.target_ttl_sec = float(self.get_parameter("target_ttl_sec").value)
        self.mission_type = str(self.get_parameter("mission_type").value).lower()
        self.turret_override_topic = str(self.get_parameter("turret_override_topic").value)
        self.turret_override_ttl_sec = max(0.05, float(self.get_parameter("turret_override_ttl_sec").value))
        self._turret_override: Optional[Dict[str, Any]] = None
        self._turret_override_stamp: float = -1e9

        # ── 정찰 전용 "지각 주도 관측" 소비 (mission_type==recon에서만) ──────────
        # local_path_node의 /tank/recon/observe_request(미분류 후보)를 받아
        #   ② 전방 후보: 감속/dwell(깨끗한 분류 프레임)  ③ 옆 후보: 포탑 step-stare.
        # 경로/조향은 안 건드리고 속도(W)와 turret만 만진다.
        self.declare_parameter("recon_observe_enabled", True)
        self.declare_parameter("recon_observe_stale_sec", 1.0)
        self.declare_parameter("recon_observe_ws_weight", 0.25)     # 전방 후보 있을 때 저속(0이면 멈춤 — 진행은 유지)
        self.declare_parameter("recon_dwell_sec", 1.2)              # 정지관측 길이
        self.declare_parameter("recon_dwell_cooldown_sec", 2.5)     # dwell 후 재dwell 금지 구간
        self.declare_parameter("recon_dwell_priority", 0.7)         # dwell 트리거 최소 우선순위
        self.declare_parameter("recon_dwell_distance_m", 30.0)      # dwell 트리거 최대 거리
        self.declare_parameter("recon_turret_enable", True)
        self.declare_parameter("recon_turret_tol_deg", 6.0)         # |오차|<=tol 이면 on-target(응시)
        self.declare_parameter("recon_turret_max_weight", 0.9)
        self.declare_parameter("recon_turret_qe_sign", 1)           # Q/E 부호 — 라이브 검증 후 뒤집기 가능
        self.recon_observe_enabled = bool(self.get_parameter("recon_observe_enabled").value)
        self.recon_observe_stale_sec = float(self.get_parameter("recon_observe_stale_sec").value)
        self.recon_observe_ws_weight = float(self.get_parameter("recon_observe_ws_weight").value)
        self.recon_dwell_sec = float(self.get_parameter("recon_dwell_sec").value)
        self.recon_dwell_cooldown_sec = float(self.get_parameter("recon_dwell_cooldown_sec").value)
        self.recon_dwell_priority = float(self.get_parameter("recon_dwell_priority").value)
        self.recon_dwell_distance_m = float(self.get_parameter("recon_dwell_distance_m").value)
        # 같은 지점(이 반경 내)에서 이미 dwell했으면 재dwell 금지 — 장애물 앞 무한 dwell 데드락 방지.
        self.declare_parameter("recon_dwell_spot_radius_m", 8.0)
        self.recon_dwell_spot_radius_m = float(self.get_parameter("recon_dwell_spot_radius_m").value)
        self.recon_turret_enable = bool(self.get_parameter("recon_turret_enable").value)
        self.recon_turret_tol_deg = float(self.get_parameter("recon_turret_tol_deg").value)
        self.recon_turret_max_weight = float(self.get_parameter("recon_turret_max_weight").value)
        self.recon_turret_qe_sign = 1 if int(self.get_parameter("recon_turret_qe_sign").value) >= 0 else -1
        self._observe_payload: Optional[Dict[str, Any]] = None
        self._observe_wall: float = 0.0
        self._turret_world_deg: Optional[float] = None
        self._recon_dwell_until: float = 0.0
        self._recon_dwell_cooldown_until: float = 0.0
        self._dwelled_spots: List[Tuple[float, float]] = []   # 이미 dwell(관측)한 지점들 — 재dwell 금지용
        self.slowdown_angle_deg = float(self.get_parameter("slowdown_angle_deg").value)
        self.stop_distance = float(self.get_parameter("stop_distance").value)
        self.enable_stuck_escape = bool(self.get_parameter("enable_stuck_escape").value)
        self.stuck_check_period = float(self.get_parameter("stuck_check_period").value)
        self.stuck_min_movement = float(self.get_parameter("stuck_min_movement").value)
        self.escape_reverse_sec = float(self.get_parameter("escape_reverse_sec").value)
        self.escape_turn_sec = float(self.get_parameter("escape_turn_sec").value)
        self.enable_safety_speed_limit = bool(self.get_parameter("enable_safety_speed_limit").value)
        self.safety_status_topic = str(self.get_parameter("safety_status_topic").value)
        self.safety_status_ttl_sec = float(self.get_parameter("safety_status_ttl_sec").value)
        self.safety_speed_limit_ws: Optional[float] = None
        self.safety_mode: str = "unknown"
        self.safety_nearest_obstacle_distance: Optional[float] = None
        self.safety_status_stamp: float = 0.0
        self.safety_apf_active: bool = False
        self.safety_apf_heading_error_deg: Optional[float] = None
        self.safety_apf_heading_abs_error_deg: Optional[float] = None
        self.safety_apf_desired_heading_deg: Optional[float] = None
        self.safety_apf_result_vector: Dict[str, float] = {"x": 0.0, "y": 0.0, "norm": 0.0}
        self.safety_nearest_obstacle_side: str = ""
        self.safety_nearest_obstacle_bearing_error_deg: Optional[float] = None
        self.safety_nearest_obstacle_turn_away_cmd: str = ""
        self.safety_recommended_turn_cmd: str = ""
        self._away_turn_lock_cmd: str = ""
        self._away_turn_lock_until: float = 0.0
        self._apf_stop_pivot_start: float = 0.0
        self._apf_stop_pivot_cooldown_until: float = 0.0
        self._apf_stop_pivot_last_reason: str = "none"
        self.safety_emergency_pivot_recommended: bool = False
        self.safety_apf_heading_stop_recommended: bool = False
        self.safety_apf_heading_slow_recommended: bool = False
        self.enable_apf_vector_stop_pivot = bool(self.get_parameter("enable_apf_vector_stop_pivot").value)
        self.apf_stop_angle_deg = float(self.get_parameter("apf_stop_angle_deg").value)
        self.apf_slow_angle_deg = float(self.get_parameter("apf_slow_angle_deg").value)
        self.apf_stop_distance = float(self.get_parameter("apf_stop_distance").value)
        self.apf_ttc_stop_sec = float(self.get_parameter("apf_ttc_stop_sec").value)
        self.apf_ttc_slow_sec = float(self.get_parameter("apf_ttc_slow_sec").value)
        self.apf_min_speed_for_ttc = float(self.get_parameter("apf_min_speed_for_ttc").value)
        self.apf_slow_ws_weight = float(self.get_parameter("apf_slow_ws_weight").value)
        self.apf_stop_pivot_max_sec = float(self.get_parameter("apf_stop_pivot_max_sec").value)
        self.apf_stop_pivot_release_angle_deg = float(self.get_parameter("apf_stop_pivot_release_angle_deg").value)
        self.apf_stop_pivot_cooldown_sec = float(self.get_parameter("apf_stop_pivot_cooldown_sec").value)
        self.prefer_turn_away_from_nearest_obstacle = bool(self.get_parameter("prefer_turn_away_from_nearest_obstacle").value)
        self.away_turn_lock_sec = float(self.get_parameter("away_turn_lock_sec").value)
        self.enable_steering_direction_lock = bool(self.get_parameter("enable_steering_direction_lock").value)
        self.steering_direction_lock_sec = float(self.get_parameter("steering_direction_lock_sec").value)
        self.enable_side_pass_forward = bool(self.get_parameter("enable_side_pass_forward").value)
        self.side_pass_bearing_deg = float(self.get_parameter("side_pass_bearing_deg").value)
        self.side_pass_min_distance = float(self.get_parameter("side_pass_min_distance").value)
        self.side_pass_ws_weight = float(self.get_parameter("side_pass_ws_weight").value)
        self.front_stop_bearing_deg = float(self.get_parameter("front_stop_bearing_deg").value)
        self.hard_stop_distance = float(self.get_parameter("hard_stop_distance").value)
        self.turn_away_hard_distance = float(self.get_parameter("turn_away_hard_distance").value)
        self.enable_ad_oscillation_guard = bool(self.get_parameter("enable_ad_oscillation_guard").value)
        self.ad_flip_window_sec = float(self.get_parameter("ad_flip_window_sec").value)
        self.ad_flip_threshold = int(self.get_parameter("ad_flip_threshold").value)
        self.ad_oscillation_hold_sec = float(self.get_parameter("ad_oscillation_hold_sec").value)
        self.ad_oscillation_slow_ws_weight = float(self.get_parameter("ad_oscillation_slow_ws_weight").value)
        self.enable_sharp_turn_stop_pivot = bool(self.get_parameter("enable_sharp_turn_stop_pivot").value)
        self.sharp_turn_stop_angle_deg = float(self.get_parameter("sharp_turn_stop_angle_deg").value)
        self.sharp_turn_release_angle_deg = float(self.get_parameter("sharp_turn_release_angle_deg").value)
        self.sharp_turn_min_target_distance = float(self.get_parameter("sharp_turn_min_target_distance").value)
        self.sharp_turn_max_sec = float(self.get_parameter("sharp_turn_max_sec").value)
        self.sharp_turn_cooldown_sec = float(self.get_parameter("sharp_turn_cooldown_sec").value)
        self.sharp_turn_block_when_apf_side_pass = bool(self.get_parameter("sharp_turn_block_when_apf_side_pass").value)
        self.enable_turn_overspeed_guard = bool(self.get_parameter("enable_turn_overspeed_guard").value)
        self.turn_overspeed_angle_deg = float(self.get_parameter("turn_overspeed_angle_deg").value)
        self.turn_overspeed_speed_mps = float(self.get_parameter("turn_overspeed_speed_mps").value)
        self.turn_overspeed_hard_angle_deg = float(self.get_parameter("turn_overspeed_hard_angle_deg").value)
        self.turn_overspeed_hard_speed_mps = float(self.get_parameter("turn_overspeed_hard_speed_mps").value)
        self.turn_overspeed_reverse_weight = float(self.get_parameter("turn_overspeed_reverse_weight").value)
        self.turn_overspeed_slow_ws_weight = float(self.get_parameter("turn_overspeed_slow_ws_weight").value)
        self.danger_obstacle_brake_speed_mps = float(self.get_parameter("danger_obstacle_brake_speed_mps").value)
        self.danger_obstacle_reverse_weight = float(self.get_parameter("danger_obstacle_reverse_weight").value)
        self.enable_forward_target_guard = bool(self.get_parameter("enable_forward_target_guard").value)
        self.forward_guard_min_target_distance = float(self.get_parameter("forward_guard_min_target_distance").value)
        self.forward_guard_yaw_error_deg = float(self.get_parameter("forward_guard_yaw_error_deg").value)
        self.forward_guard_target_distance = float(self.get_parameter("forward_guard_target_distance").value)
        self.forward_guard_max_search_points = int(self.get_parameter("forward_guard_max_search_points").value)
        self.forward_guard_allow_in_danger = bool(self.get_parameter("forward_guard_allow_in_danger").value)
        self.enable_planner_speed_profile = bool(self.get_parameter("enable_planner_speed_profile").value)
        self.planner_path_points_topic = str(self.get_parameter("planner_path_points_topic").value)
        self.planner_speed_profile_ttl_sec = max(0.1, float(self.get_parameter("planner_speed_profile_ttl_sec").value))
        self.planner_goal_stop_distance_m = max(0.1, float(self.get_parameter("planner_goal_stop_distance_m").value))
        self._steering_direction_lock_cmd: str = ""
        self._steering_direction_lock_until: float = 0.0
        self._steering_direction_lock_reason: str = "none"
        self._last_effective_ad_cmd: str = ""
        self._ad_flip_times: List[float] = []
        self._ad_oscillation_hold_cmd: str = ""
        self._ad_oscillation_hold_until: float = 0.0
        self._ad_oscillation_last_reason: str = "none"
        self._sharp_turn_pivot_start: float = 0.0
        self._sharp_turn_pivot_cooldown_until: float = 0.0
        self._sharp_turn_pivot_last_reason: str = "none"

        self.tank_params = self.load_tank_params(str(self.get_parameter("tank_param_file").value))
        self.max_speed = self.extract_max_from_dict(self.tank_params.get("steady_state_speed", {}), 19.45)
        self.max_yaw_rate = self.extract_max_from_dict(self.tank_params.get("steady_state_yaw_rate", {}), 38.84)

        self.current_pos: Optional[Tuple[float, float]] = None
        self.current_yaw: float = 0.0
        self.current_speed: float = 0.0
        self.current_sim_time: float = 0.0
        self.goal_pos: Optional[Tuple[float, float]] = None
        self.lookahead_target: Optional[Tuple[float, float]] = None
        self.lookahead_stamp: float = 0.0
        self.local_target: Optional[Tuple[float, float]] = None
        self.local_target_stamp: float = 0.0
        self.collision_count = 0
        self.mission_complete = False
        self.target_guard_status: Dict[str, Any] = {"enabled": False, "active": False, "reason": "init"}

        self.last_stuck_check_time = 0.0
        self.last_stuck_check_pos: Optional[Tuple[float, float]] = None
        self.last_stuck_check_yaw: float = 0.0
        self.is_escaping = False
        self.escape_start_time = 0.0
        self._pivot_turn_count = 0
        self._last_pose_wall_time = 0.0

        self.global_path: list[tuple[float, float]] = []
        self.trajectory_history: list[tuple[float, float]] = []
        self.planner_speed_points: List[Dict[str, Any]] = []
        self.planner_vehicle_geometry: Dict[str, Any] = {}
        self.planner_route_version: Optional[int] = None
        self._planner_profile_stamp_mono: float = -1e9
        self._planner_profile_status: Dict[str, Any] = {"active": False, "reason": "no_profile"}

        self.pub_cmd = self.create_publisher(String, TOPIC_CONTROL_COMMAND, 10)
        self.pub_status = self.create_publisher(String, TOPIC_CONTROL_STATUS, 10)
        self.create_subscription(PoseStamped, TOPIC_PLAYER_POSE, self.player_pose_cb, 10)
        self.create_subscription(String, TOPIC_PLAYER_STATE, self.player_state_cb, 10)
        self.create_subscription(PoseStamped, TOPIC_GOAL_POSE, self.goal_pose_cb, 10)
        self.create_subscription(PoseStamped, TOPIC_LOOKAHEAD_POSE, self.lookahead_cb, 10)
        self.create_subscription(PoseStamped, TOPIC_LOCAL_TARGET_POSE, self.local_target_cb, 10)
        self.create_subscription(NavPath, "/tank/global_path", self.global_path_cb, 10)
        self.create_subscription(String, self.planner_path_points_topic, self.planner_path_points_cb, 10)
        self.create_subscription(String, TOPIC_COLLISION_EVENT, self.collision_cb, 10)
        self.create_subscription(String, self.turret_override_topic, self.turret_override_cb, 10)
        self.create_subscription(String, self.safety_status_topic, self.safety_status_cb, 10)
        # 정찰 관측요청(미분류 후보) + 포탑 현재각 피드백(step-stare 폐루프용).
        self.create_subscription(String, "/tank/recon/observe_request", self.observe_request_cb, 10)
        self.create_subscription(Vector3Stamped, "/tank/api/get_action/turret", self.turret_feedback_cb, 10)
        hz = float(self.get_parameter("controller_hz").value)
        self.create_timer(1.0 / max(1.0, hz), self.timer_cb)
        self.get_logger().info(
            f"Team path controller initialized: max_speed={self.max_speed:.2f}, max_yaw_rate={self.max_yaw_rate:.2f}, "
            f"local_target={self.enable_local_target}"
        )

    def load_tank_params(self, path: str) -> Dict[str, Any]:
        if path and os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return yaml.safe_load(f) or {}
            except Exception as exc:
                self.get_logger().warn(f"failed to load tank params: {exc}")
        return {}

    @staticmethod
    def extract_max_from_dict(d: Dict[Any, Any], default: float) -> float:
        vals = []
        for v in d.values():
            try:
                vals.append(float(v))
            except Exception:
                pass
        return max(vals) if vals else default

    def player_pose_cb(self, msg: PoseStamped) -> None:
        new_pos = (float(msg.pose.position.x), float(msg.pose.position.y))
        if getattr(self, 'current_pos', None) is not None:
            if get_distance(self.current_pos, new_pos) > 10.0:
                self.get_logger().info("Teleport/Restart detected. Resetting mission complete flag.")
                self.mission_complete = False
                self._stop_published_count = 0
                self.is_escaping = False
                self.global_path = []
        self.current_pos = new_pos
        self._last_pose_wall_time = time.time()
        if self.last_stuck_check_pos is None:
            self.last_stuck_check_pos = self.current_pos
        self.trajectory_history.append(self.current_pos)
        if len(self.trajectory_history) > 5000:
            self.trajectory_history.pop(0)

    def global_path_cb(self, msg: NavPath) -> None:
        path = []
        for pose in msg.poses:
            path.append((float(pose.pose.position.x), float(pose.pose.position.y)))
        self.global_path = path

    def player_state_cb(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
        except Exception:
            return
        try:
            self.current_speed = float(data.get("speed") or 0.0)
        except Exception:
            self.current_speed = 0.0
        body = data.get("body") if isinstance(data.get("body"), dict) else {}
        try:
            self.current_yaw = float(body.get("x") or 0.0)
        except Exception:
            pass
        try:
            self.current_sim_time = float(data.get("sim_time") or self.current_sim_time)
        except Exception:
            pass

    def goal_pose_cb(self, msg: PoseStamped) -> None:
        self.goal_pos = (float(msg.pose.position.x), float(msg.pose.position.y))
        self.mission_complete = False

    def lookahead_cb(self, msg: PoseStamped) -> None:
        self.lookahead_target = (float(msg.pose.position.x), float(msg.pose.position.y))
        self.lookahead_stamp = self.get_clock().now().nanoseconds / 1e9

    def local_target_cb(self, msg: PoseStamped) -> None:
        self.local_target = (float(msg.pose.position.x), float(msg.pose.position.y))
        self.local_target_stamp = self.get_clock().now().nanoseconds / 1e9

    def collision_cb(self, msg: String) -> None:
        self.collision_count += 1

    def turret_override_cb(self, msg: String) -> None:
        """Receive a short-lived turret-only override from the engagement layer.

        The controller remains the *only* publisher of /tank/control/command,
        so movement and turret actions cannot race at the bridge.
        """
        try:
            payload = json.loads(msg.data)
        except Exception:
            return
        if not isinstance(payload, dict):
            return
        self._turret_override = payload
        self._turret_override_stamp = time.monotonic()

    def safety_status_cb(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
        except Exception:
            return
        try:
            self.safety_speed_limit_ws = float(data.get("speed_limit_ws"))
        except Exception:
            self.safety_speed_limit_ws = None
        self.safety_mode = str(data.get("mode", "unknown"))
        try:
            nearest = data.get("nearest_obstacle_distance")
            self.safety_nearest_obstacle_distance = None if nearest is None else float(nearest)
        except Exception:
            self.safety_nearest_obstacle_distance = None

        self.safety_nearest_obstacle_side = str(data.get("nearest_obstacle_side") or "")
        try:
            v = data.get("nearest_obstacle_bearing_error_deg")
            self.safety_nearest_obstacle_bearing_error_deg = None if v is None else float(v)
        except Exception:
            self.safety_nearest_obstacle_bearing_error_deg = None
        self.safety_nearest_obstacle_turn_away_cmd = str(data.get("nearest_obstacle_turn_away_cmd") or "")
        self.safety_recommended_turn_cmd = str(data.get("recommended_turn_cmd") or "")

        self.safety_apf_active = bool(data.get("apf_active", False))
        self.safety_emergency_pivot_recommended = bool(data.get("emergency_pivot_recommended", False))
        self.safety_apf_heading_stop_recommended = bool(data.get("apf_heading_stop_recommended", False))
        self.safety_apf_heading_slow_recommended = bool(data.get("apf_heading_slow_recommended", False))
        try:
            v = data.get("apf_heading_error_deg")
            self.safety_apf_heading_error_deg = None if v is None else float(v)
        except Exception:
            self.safety_apf_heading_error_deg = None
        try:
            v = data.get("apf_heading_abs_error_deg")
            self.safety_apf_heading_abs_error_deg = None if v is None else float(v)
        except Exception:
            self.safety_apf_heading_abs_error_deg = None
        try:
            v = data.get("apf_desired_heading_deg")
            self.safety_apf_desired_heading_deg = None if v is None else float(v)
        except Exception:
            self.safety_apf_desired_heading_deg = None
        vec = data.get("apf_result_vector") if isinstance(data.get("apf_result_vector"), dict) else {}
        try:
            self.safety_apf_result_vector = {
                "x": float(vec.get("x", 0.0)),
                "y": float(vec.get("y", 0.0)),
                "norm": float(vec.get("norm", 0.0)),
            }
        except Exception:
            self.safety_apf_result_vector = {"x": 0.0, "y": 0.0, "norm": 0.0}

        self.safety_status_stamp = self.get_clock().now().nanoseconds / 1e9

    def current_safety_limit(self) -> Tuple[Optional[float], str]:
        if not self.enable_safety_speed_limit:
            return None, "disabled"
        if self.safety_speed_limit_ws is None:
            return None, "none"
        now = self.get_clock().now().nanoseconds / 1e9
        if now - self.safety_status_stamp > self.safety_status_ttl_sec:
            return None, "stale"
        return clamp(float(self.safety_speed_limit_ws), 0.0, 1.0), self.safety_mode

    def apply_turn_overspeed_guard(
        self,
        cmd_ws: str,
        w_ws: float,
        yaw_error: float,
        speed_mode: str,
        safety_mode: str,
        apf_vector_control: Dict[str, Any],
    ) -> Tuple[str, float, str, Dict[str, Any]]:
        """코너/장애물 앞 오버슛 방지용 실제 속도 기반 braking layer.

        기존 제어는 W weight를 줄이는 방식이라 이미 5~10m/s로 붙은 관성은 바로 줄이지 못한다.
        그래서 큰 yaw error 또는 danger obstacle 상태에서 속도가 높으면 짧게 S/STOP을 우선한다.
        """
        status: Dict[str, Any] = {
            "enabled": self.enable_turn_overspeed_guard,
            "active": False,
            "reason": "disabled",
            "current_speed_mps": self.current_speed,
            "yaw_abs_error_deg": abs(float(yaw_error)),
            "cmd_before": cmd_ws,
            "w_before": float(w_ws),
        }
        if not self.enable_turn_overspeed_guard or cmd_ws == "S":
            return cmd_ws, w_ws, speed_mode, status

        speed = max(0.0, float(self.current_speed))
        yaw_abs = abs(float(yaw_error))
        front_blocking = bool(apf_vector_control.get("front_blocking", False))
        hard_close = bool(apf_vector_control.get("hard_close", False))
        danger = str(safety_mode).lower() == "danger" or (front_blocking and hard_close)

        # 장애물이 가까운데 속도가 남아 있으면 W/STOP만으로는 밀려 들어가므로 S로 먼저 감속한다.
        if danger and speed >= self.danger_obstacle_brake_speed_mps:
            status.update({
                "active": True,
                "reason": "danger_obstacle_brake",
                "front_blocking": front_blocking,
                "hard_close": hard_close,
                "cmd_after": "S",
                "w_after": self.danger_obstacle_reverse_weight,
            })
            return "S", clamp(self.danger_obstacle_reverse_weight, 0.0, 1.0), f"{speed_mode}|danger_obstacle_brake", status

        # 큰 코너 진입 고속: 역입력으로 짧게 감속.
        if yaw_abs >= self.turn_overspeed_hard_angle_deg and speed >= self.turn_overspeed_hard_speed_mps:
            status.update({
                "active": True,
                "reason": "hard_turn_overspeed_brake",
                "cmd_after": "S",
                "w_after": self.turn_overspeed_reverse_weight,
            })
            return "S", clamp(self.turn_overspeed_reverse_weight, 0.0, 1.0), f"{speed_mode}|hard_turn_overspeed_brake", status

        # 중간 코너 고속: 가속을 끊고 선회 안정화.
        if yaw_abs >= self.turn_overspeed_angle_deg and speed >= self.turn_overspeed_speed_mps:
            status.update({
                "active": True,
                "reason": "turn_overspeed_stop",
                "cmd_after": "STOP",
                "w_after": 1.0,
            })
            return "STOP", 1.0, f"{speed_mode}|turn_overspeed_stop", status

        # 코너에서는 아직 위험 threshold 아래라도 W를 더 낮춘다.
        if cmd_ws == "W" and yaw_abs >= self.turn_overspeed_angle_deg and w_ws > self.turn_overspeed_slow_ws_weight:
            status.update({
                "active": True,
                "reason": "turn_overspeed_slow",
                "cmd_after": "W",
                "w_after": self.turn_overspeed_slow_ws_weight,
            })
            return "W", clamp(self.turn_overspeed_slow_ws_weight, 0.0, 1.0), f"{speed_mode}|turn_overspeed_slow", status

        status.update({"reason": "clear", "cmd_after": cmd_ws, "w_after": float(w_ws)})
        return cmd_ws, w_ws, speed_mode, status

    def is_safety_status_fresh(self) -> bool:
        if self.safety_status_stamp <= 0.0:
            return False
        now = self.get_clock().now().nanoseconds / 1e9
        return (now - self.safety_status_stamp) <= self.safety_status_ttl_sec

    def obstacle_bearing_abs_deg(self) -> Optional[float]:
        if self.safety_nearest_obstacle_bearing_error_deg is None:
            return None
        try:
            return abs(float(self.safety_nearest_obstacle_bearing_error_deg))
        except Exception:
            return None

    def is_side_pass_context(self) -> bool:
        """장애물이 전방을 막는 것이 아니라 차체 옆에 있는 통과 상황인지 판단한다."""
        if not self.enable_side_pass_forward:
            return False
        nearest = self.safety_nearest_obstacle_distance
        bearing_abs = self.obstacle_bearing_abs_deg()
        if nearest is None or bearing_abs is None:
            return False
        return nearest >= self.side_pass_min_distance and bearing_abs >= self.side_pass_bearing_deg

    def is_front_blocking_context(self) -> bool:
        """장애물이 전방 corridor에 있어 STOP pivot까지 고려해야 하는 상황인지 판단한다."""
        nearest = self.safety_nearest_obstacle_distance
        bearing_abs = self.obstacle_bearing_abs_deg()
        if nearest is None:
            return False
        if nearest <= self.hard_stop_distance:
            return True
        if bearing_abs is None:
            return True
        return bearing_abs <= self.front_stop_bearing_deg

    def apply_steering_direction_lock(self, cmd_ad: str, context: str = "normal") -> Tuple[str, Dict[str, Any]]:
        """짧은 주기의 A↔D 토글을 억제한다.

        APF 경계나 경로 코너에서 target이 좌우로 흔들리면 매 tick A/D가 뒤집힌다.
        전차는 차동구동이라 이런 토글이 곧 제자리 진동/큰 원운동으로 이어진다.
        그래서 한 번 정한 조향 방향을 짧게 유지한다.
        """
        now = self.get_clock().now().nanoseconds / 1e9
        original_cmd = cmd_ad if cmd_ad in ("A", "D") else ""
        locked = False
        reason = "none"

        if not self.enable_steering_direction_lock or original_cmd == "":
            if original_cmd == "":
                self._steering_direction_lock_cmd = ""
                self._steering_direction_lock_until = 0.0
                self._steering_direction_lock_reason = "no_ad_cmd"
            return cmd_ad, {
                "enabled": self.enable_steering_direction_lock,
                "original_cmd": original_cmd,
                "cmd": cmd_ad,
                "locked": False,
                "lock_cmd": self._steering_direction_lock_cmd,
                "lock_until": self._steering_direction_lock_until,
                "reason": self._steering_direction_lock_reason,
                "context": context,
            }

        if (
            self._steering_direction_lock_cmd in ("A", "D")
            and now < self._steering_direction_lock_until
            and original_cmd != self._steering_direction_lock_cmd
        ):
            cmd_ad = self._steering_direction_lock_cmd
            locked = True
            reason = "direction_lock_hold"
        else:
            if original_cmd != self._steering_direction_lock_cmd or now >= self._steering_direction_lock_until:
                self._steering_direction_lock_cmd = original_cmd
                self._steering_direction_lock_until = now + max(0.0, self.steering_direction_lock_sec)
                reason = "direction_lock_set"
            else:
                reason = "direction_lock_continue"

        self._steering_direction_lock_reason = reason
        return cmd_ad, {
            "enabled": self.enable_steering_direction_lock,
            "original_cmd": original_cmd,
            "cmd": cmd_ad,
            "locked": locked,
            "lock_cmd": self._steering_direction_lock_cmd,
            "lock_until": self._steering_direction_lock_until,
            "reason": reason,
            "context": context,
            "lock_sec": self.steering_direction_lock_sec,
        }

    def apply_ad_oscillation_guard(self, cmd_ad: str, context: str = "normal") -> Tuple[str, Dict[str, Any]]:
        """지그재그 장애물 사이에서 A↔D가 반복될 때 한쪽 방향을 잠시 고정한다."""
        now = self.get_clock().now().nanoseconds / 1e9
        original_cmd = cmd_ad if cmd_ad in ("A", "D") else ""
        active = False
        reason = "none"

        if not self.enable_ad_oscillation_guard or original_cmd == "":
            if original_cmd == "":
                self._last_effective_ad_cmd = ""
                self._ad_flip_times = [t for t in self._ad_flip_times if now - t <= self.ad_flip_window_sec]
            return cmd_ad, {
                "enabled": self.enable_ad_oscillation_guard,
                "active": False,
                "original_cmd": original_cmd,
                "cmd": cmd_ad,
                "hold_cmd": self._ad_oscillation_hold_cmd,
                "hold_until": self._ad_oscillation_hold_until,
                "flip_count": len(self._ad_flip_times),
                "reason": "disabled_or_no_ad",
                "context": context,
            }

        # 이미 oscillation hold 중이면 현재 판단이 바뀌어도 유지한다.
        if self._ad_oscillation_hold_cmd in ("A", "D") and now < self._ad_oscillation_hold_until:
            cmd_ad = self._ad_oscillation_hold_cmd
            active = True
            reason = "oscillation_hold"
        else:
            # window 내 flip count 계산.
            self._ad_oscillation_hold_cmd = ""
            self._ad_oscillation_hold_until = 0.0
            self._ad_flip_times = [t for t in self._ad_flip_times if now - t <= self.ad_flip_window_sec]
            if (
                self._last_effective_ad_cmd in ("A", "D")
                and original_cmd in ("A", "D")
                and original_cmd != self._last_effective_ad_cmd
            ):
                self._ad_flip_times.append(now)

            if len(self._ad_flip_times) >= max(1, self.ad_flip_threshold):
                # 새 명령으로 뒤집지 말고 직전 유효 방향을 잠시 유지해 좌우 진동을 끊는다.
                hold_cmd = self._last_effective_ad_cmd if self._last_effective_ad_cmd in ("A", "D") else original_cmd
                self._ad_oscillation_hold_cmd = hold_cmd
                self._ad_oscillation_hold_until = now + max(0.0, self.ad_oscillation_hold_sec)
                self._ad_flip_times.clear()
                cmd_ad = hold_cmd
                active = True
                reason = "oscillation_detected_hold_previous"
            else:
                cmd_ad = original_cmd
                reason = "tracking"

        if cmd_ad in ("A", "D"):
            self._last_effective_ad_cmd = cmd_ad
        self._ad_oscillation_last_reason = reason
        return cmd_ad, {
            "enabled": self.enable_ad_oscillation_guard,
            "active": active,
            "original_cmd": original_cmd,
            "cmd": cmd_ad,
            "hold_cmd": self._ad_oscillation_hold_cmd,
            "hold_until": self._ad_oscillation_hold_until,
            "flip_count": len(self._ad_flip_times),
            "reason": reason,
            "context": context,
            "policy": {
                "window_sec": self.ad_flip_window_sec,
                "threshold": self.ad_flip_threshold,
                "hold_sec": self.ad_oscillation_hold_sec,
                "slow_ws_weight": self.ad_oscillation_slow_ws_weight,
            },
        }

    def sharp_turn_pivot_decision(
        self,
        target: Tuple[float, float],
        source: str,
        yaw_error: float,
        cmd_ad: str,
    ) -> Dict[str, Any]:
        """경로가 급격히 꺾일 때 W를 끊고 차체 방향을 먼저 맞춘다."""
        now = self.get_clock().now().nanoseconds / 1e9
        target_distance = get_distance(self.current_pos, target) if self.current_pos is not None else 0.0
        abs_err = abs(yaw_error)
        cooldown_active = now < self._sharp_turn_pivot_cooldown_until
        side_pass = self.is_side_pass_context()
        front_blocking = self.is_front_blocking_context()
        # APF가 장애물 옆 통과 상황을 만들고 있을 때 sharp-turn STOP까지 겹치면
        # 지나가도 되는 장애물 옆에서 멈췄다 출발하는 현상이 생긴다.
        block_by_side_pass = bool(self.sharp_turn_block_when_apf_side_pass and self.safety_apf_active and side_pass and not front_blocking)
        trigger = (
            self.enable_sharp_turn_stop_pivot
            and not cooldown_active
            and not block_by_side_pass
            and target_distance >= self.sharp_turn_min_target_distance
            and abs_err >= self.sharp_turn_stop_angle_deg
        )

        active = False
        elapsed = 0.0
        guard_reason = "none"
        if trigger:
            if self._sharp_turn_pivot_start <= 0.0:
                self._sharp_turn_pivot_start = now
            elapsed = now - self._sharp_turn_pivot_start
            aligned_enough = abs_err <= self.sharp_turn_release_angle_deg
            timed_out = elapsed >= max(0.1, self.sharp_turn_max_sec)
            if aligned_enough or timed_out:
                active = False
                guard_reason = "aligned_release" if aligned_enough else "max_sec_release_allow_slow_turn"
                self._sharp_turn_pivot_start = 0.0
                self._sharp_turn_pivot_cooldown_until = now + max(0.0, self.sharp_turn_cooldown_sec)
            else:
                active = True
                guard_reason = "active"
        else:
            self._sharp_turn_pivot_start = 0.0
            if cooldown_active:
                guard_reason = "cooldown_allow_slow_turn"

        turn_cmd = cmd_ad if cmd_ad in ("A", "D") else ("D" if yaw_error > 0.0 else "A")
        self._sharp_turn_pivot_last_reason = guard_reason
        return {
            "enabled": self.enable_sharp_turn_stop_pivot,
            "active": active,
            "turn_cmd": turn_cmd,
            "target_source": source,
            "target_distance": target_distance,
            "yaw_error_deg": yaw_error,
            "yaw_abs_error_deg": abs_err,
            "side_pass": side_pass,
            "front_blocking": front_blocking,
            "block_by_side_pass": block_by_side_pass,
            "start": self._sharp_turn_pivot_start,
            "elapsed": elapsed,
            "cooldown_until": self._sharp_turn_pivot_cooldown_until,
            "guard_reason": guard_reason,
            "policy": {
                "stop_angle_deg": self.sharp_turn_stop_angle_deg,
                "release_angle_deg": self.sharp_turn_release_angle_deg,
                "min_target_distance": self.sharp_turn_min_target_distance,
                "max_sec": self.sharp_turn_max_sec,
                "cooldown_sec": self.sharp_turn_cooldown_sec,
            },
        }

    def apf_vector_control_decision(self) -> Dict[str, Any]:
        """APF 합벡터와 차체 heading 차이를 이용해 W 차단 여부를 결정한다."""
        fresh = self.is_safety_status_fresh()
        nearest = self.safety_nearest_obstacle_distance
        apf_err = self.safety_apf_heading_error_deg
        apf_abs = self.safety_apf_heading_abs_error_deg
        if apf_abs is None and apf_err is not None:
            apf_abs = abs(apf_err)

        speed = max(0.0, float(self.current_speed))
        ttc = None
        if nearest is not None and speed >= self.apf_min_speed_for_ttc:
            # 보수적 TTC: 장애물 bearing을 모르는 상태에서는 현재 속도로 닫힌다고 가정한다.
            ttc = nearest / max(speed, 0.1)

        bearing_abs = self.obstacle_bearing_abs_deg()
        close = nearest is not None and nearest <= self.apf_stop_distance
        hard_close = nearest is not None and nearest <= self.hard_stop_distance
        side_pass = self.is_side_pass_context()
        front_blocking = self.is_front_blocking_context()
        stop_allowed = bool(hard_close or front_blocking)

        # 장애물이 옆에 있는 side-pass 상황이면 APF 합벡터가 크게 꺾여도 STOP하지 않는다.
        # STOP은 전방 corridor를 막거나 매우 가까운 hard-close 상황에서만 허용한다.
        heading_stop = (
            fresh
            and self.safety_apf_active
            and close
            and stop_allowed
            and not (side_pass and not hard_close)
            and apf_abs is not None
            and apf_abs >= self.apf_stop_angle_deg
        )
        ttc_stop = (
            fresh
            and self.safety_apf_active
            and stop_allowed
            and ttc is not None
            and ttc <= self.apf_ttc_stop_sec
            and apf_abs is not None
            and apf_abs >= self.apf_slow_angle_deg
        )
        recommended_stop = (
            fresh
            and self.safety_emergency_pivot_recommended
            and stop_allowed
            and not (side_pass and not hard_close)
            and apf_abs is not None
            and apf_abs >= self.apf_slow_angle_deg
        )
        stop_pivot = bool(self.enable_apf_vector_stop_pivot and (heading_stop or ttc_stop or recommended_stop))

        slow_turn = bool(
            self.enable_apf_vector_stop_pivot
            and fresh
            and self.safety_apf_active
            and close
            and apf_abs is not None
            and apf_abs >= self.apf_slow_angle_deg
        )
        if stop_pivot:
            slow_turn = True

        turn_cmd = ""
        turn_reason = "none"
        now = self.get_clock().now().nanoseconds / 1e9

        # APF STOP pivot 루프 방지.
        # APF 합벡터가 계속 옆을 가리키면 stop_pivot이 영구 유지되어 제자리 회전만 한다.
        # 그래서 STOP pivot은 짧게만 허용하고, 이후 cooldown 동안은 slow_turn(W 저속)을 허용한다.
        pivot_elapsed = 0.0
        pivot_guard_reason = "none"
        if stop_pivot and now < self._apf_stop_pivot_cooldown_until:
            stop_pivot = False
            pivot_guard_reason = "cooldown_allow_slow_escape"
        elif stop_pivot:
            if self._apf_stop_pivot_start <= 0.0:
                self._apf_stop_pivot_start = now
            pivot_elapsed = now - self._apf_stop_pivot_start
            aligned_enough = (apf_abs is not None and apf_abs <= self.apf_stop_pivot_release_angle_deg)
            timed_out = pivot_elapsed >= max(0.1, self.apf_stop_pivot_max_sec)
            if aligned_enough or timed_out:
                stop_pivot = False
                pivot_guard_reason = "aligned_release" if aligned_enough else "max_sec_release_allow_slow_escape"
                self._apf_stop_pivot_start = 0.0
                self._apf_stop_pivot_cooldown_until = now + max(0.0, self.apf_stop_pivot_cooldown_sec)
        else:
            self._apf_stop_pivot_start = 0.0

        self._apf_stop_pivot_last_reason = pivot_guard_reason

        # 1순위: 이미 잠근 회피 방향 유지. 너무 자주 A/D가 바뀌면 제자리 루프가 생긴다.
        if self._away_turn_lock_cmd and now < self._away_turn_lock_until:
            turn_cmd = self._away_turn_lock_cmd
            turn_reason = "locked_away_from_nearest_obstacle"

        # 2순위: 가장 가까운 장애물의 반대 방향으로 선회.
        # 장애물 right -> A, 장애물 left -> D.
        use_nearest_away = bool(
            self.prefer_turn_away_from_nearest_obstacle
            and self.safety_nearest_obstacle_turn_away_cmd in ("A", "D")
            and (front_blocking or hard_close or not side_pass or (nearest is not None and nearest <= self.turn_away_hard_distance))
        )
        if not turn_cmd and use_nearest_away:
            turn_cmd = self.safety_nearest_obstacle_turn_away_cmd
            turn_reason = f"away_from_nearest_obstacle_{self.safety_nearest_obstacle_side}"
            if stop_pivot or slow_turn:
                self._away_turn_lock_cmd = turn_cmd
                self._away_turn_lock_until = now + max(0.0, self.away_turn_lock_sec)

        # 3순위: potential node가 권고한 방향. 단, side-pass 상황에서는 nearest 기준 권고가
        # 좌우 장애물 사이에서 계속 뒤집히므로 APF heading 부호/기본 조향을 우선한다.
        if not turn_cmd and (not side_pass) and self.safety_recommended_turn_cmd in ("A", "D"):
            turn_cmd = self.safety_recommended_turn_cmd
            turn_reason = "potential_recommended_turn"

        # 최후 fallback: APF heading error 부호.
        if not turn_cmd and apf_err is not None and abs(apf_err) >= 1.0:
            # 기존 controller 관례와 동일하게 yaw_error > 0이면 D, < 0이면 A.
            turn_cmd = "D" if apf_err > 0.0 else "A"
            turn_reason = "apf_heading_error_sign"

        if not (stop_pivot or slow_turn):
            self._away_turn_lock_cmd = ""
            self._away_turn_lock_until = 0.0

        return {
            "fresh": fresh,
            "enabled": self.enable_apf_vector_stop_pivot,
            "stop_pivot": stop_pivot,
            "slow_turn": slow_turn,
            "turn_cmd": turn_cmd,
            "turn_reason": turn_reason,
            "nearest_obstacle_distance": nearest,
            "nearest_obstacle_side": self.safety_nearest_obstacle_side,
            "nearest_obstacle_bearing_abs_deg": bearing_abs,
            "front_blocking": front_blocking,
            "side_pass": side_pass,
            "hard_close": hard_close,
            "stop_allowed": stop_allowed,
            "use_nearest_away": use_nearest_away,
            "side_pass_ws_weight": self.side_pass_ws_weight,
            "nearest_obstacle_side": self.safety_nearest_obstacle_side,
            "nearest_obstacle_bearing_error_deg": self.safety_nearest_obstacle_bearing_error_deg,
            "nearest_obstacle_turn_away_cmd": self.safety_nearest_obstacle_turn_away_cmd,
            "recommended_turn_cmd": self.safety_recommended_turn_cmd,
            "away_turn_lock_cmd": self._away_turn_lock_cmd,
            "away_turn_lock_until": self._away_turn_lock_until,
            "apf_stop_pivot_start": self._apf_stop_pivot_start,
            "apf_stop_pivot_elapsed": pivot_elapsed,
            "apf_stop_pivot_cooldown_until": self._apf_stop_pivot_cooldown_until,
            "apf_stop_pivot_guard_reason": pivot_guard_reason,
            "ttc_sec": ttc,
            "apf_heading_error_deg": apf_err,
            "apf_heading_abs_error_deg": apf_abs,
            "apf_desired_heading_deg": self.safety_apf_desired_heading_deg,
            "apf_result_vector": self.safety_apf_result_vector,
            "apf_active": self.safety_apf_active,
            "emergency_pivot_recommended": self.safety_emergency_pivot_recommended,
            "heading_stop": heading_stop,
            "ttc_stop": ttc_stop,
            "recommended_stop": recommended_stop,
            "policy": {
                "stop_angle_deg": self.apf_stop_angle_deg,
                "slow_angle_deg": self.apf_slow_angle_deg,
                "stop_distance_m": self.apf_stop_distance,
                "ttc_stop_sec": self.apf_ttc_stop_sec,
                "ttc_slow_sec": self.apf_ttc_slow_sec,
                "slow_ws_weight": self.apf_slow_ws_weight,
                "stop_pivot_max_sec": self.apf_stop_pivot_max_sec,
                "stop_pivot_release_angle_deg": self.apf_stop_pivot_release_angle_deg,
                "stop_pivot_cooldown_sec": self.apf_stop_pivot_cooldown_sec,
                "front_stop_bearing_deg": self.front_stop_bearing_deg,
                "side_pass_bearing_deg": self.side_pass_bearing_deg,
                "side_pass_min_distance": self.side_pass_min_distance,
                "hard_stop_distance": self.hard_stop_distance,
                "turn_away_hard_distance": self.turn_away_hard_distance,
            },
        }

    def _yaw_error_to_target(self, target: Tuple[float, float]) -> float:
        if self.current_pos is None:
            return 0.0
        dx = target[0] - self.current_pos[0]
        dy = target[1] - self.current_pos[1]
        desired_yaw = math.degrees(math.atan2(dx, dy))
        return normalize_angle(desired_yaw - self.current_yaw)

    def _forward_target_from_global_path(self, lookahead_distance: Optional[float] = None) -> Optional[Tuple[float, float]]:
        """현재 위치를 global_path에 투영한 뒤 일정 거리 앞의 point를 반환한다.

        local_target/APF target이 현재 위치 가까이에서 좌우로 바뀌면 controller가
        target을 향해 제자리 회전만 반복한다. 이때 path의 "진행 방향" 기준 target을
        다시 잡아 회전 루프를 끊는다.
        """
        if self.current_pos is None or len(self.global_path) < 2:
            return None
        lookahead = max(float(lookahead_distance if lookahead_distance is not None else self.forward_guard_target_distance), 1.0)
        route = self.global_path
        max_points = max(2, int(self.forward_guard_max_search_points))

        best_i = 0
        best_t = 0.0
        best_dist = float("inf")
        # 너무 긴 path 전체를 매 tick 다 보지 않도록 제한하되, 기본 120 segment면 충분하다.
        end_i = min(len(route) - 1, max_points)
        for i in range(0, end_i):
            ax, ay = route[i]
            bx, by = route[i + 1]
            dx = bx - ax
            dy = by - ay
            denom = dx * dx + dy * dy
            t = 0.0 if denom <= 1e-9 else clamp(((self.current_pos[0] - ax) * dx + (self.current_pos[1] - ay) * dy) / denom, 0.0, 1.0)
            px = ax + t * dx
            py = ay + t * dy
            d = get_distance(self.current_pos, (px, py))
            if d < best_dist:
                best_dist = d
                best_i = i
                best_t = t

        remaining = lookahead
        ax, ay = route[best_i]
        bx, by = route[best_i + 1]
        seg_len = get_distance((ax, ay), (bx, by))
        cur = (ax + best_t * (bx - ax), ay + best_t * (by - ay))
        first_remaining = seg_len * (1.0 - best_t)
        if remaining <= first_remaining and seg_len > 1e-9:
            r = remaining / seg_len
            return (cur[0] + r * (bx - ax), cur[1] + r * (by - ay))
        remaining -= first_remaining
        for j in range(best_i + 1, len(route) - 1):
            a = route[j]
            b = route[j + 1]
            seg_len = get_distance(a, b)
            if remaining <= seg_len and seg_len > 1e-9:
                r = remaining / seg_len
                return (a[0] + r * (b[0] - a[0]), a[1] + r * (b[1] - a[1]))
            remaining -= seg_len
        return route[-1]

    def apply_forward_target_guard(self, target: Tuple[float, float], source: str) -> Tuple[Tuple[float, float], str, Dict[str, Any]]:
        status: Dict[str, Any] = {
            "enabled": self.enable_forward_target_guard,
            "active": False,
            "reason": "disabled",
            "source_before": source,
            "source_after": source,
        }
        if not self.enable_forward_target_guard or self.current_pos is None:
            return target, source, status

        target_distance = get_distance(self.current_pos, target)
        yaw_error = self._yaw_error_to_target(target)
        yaw_abs = abs(yaw_error)
        safety_mode = str(getattr(self, "safety_mode", "unknown")).lower()
        apf_active = bool(getattr(self, "safety_apf_active", False))
        in_danger = safety_mode == "danger"
        status.update({
            "reason": "clear",
            "target_distance_before": target_distance,
            "yaw_error_before_deg": yaw_error,
            "safety_mode": safety_mode,
            "safety_apf_active": apf_active,
        })

        # 위험 장애물에 대한 APF 회피 중이면 APF target을 존중한다. 문제는 회피가 끝난 뒤
        # local target이 너무 가까운 지점/뒤쪽 지점으로 남아 제자리 회전을 만드는 경우다.
        if in_danger and not self.forward_guard_allow_in_danger:
            status["reason"] = "blocked_by_danger_safety"
            return target, source, status
        if apf_active and target_distance >= self.forward_guard_min_target_distance:
            status["reason"] = "blocked_by_active_apf"
            return target, source, status

        too_close = target_distance < self.forward_guard_min_target_distance
        bad_bearing = yaw_abs >= self.forward_guard_yaw_error_deg
        # local_target_apf가 passthrough 상태에서도 사용되므로 source에만 의존하지 말고,
        # close+bearing 조건이면 전방 path target으로 교체한다.
        if not (too_close or bad_bearing):
            return target, source, status

        forward = self._forward_target_from_global_path(self.forward_guard_target_distance)
        if forward is None:
            status["reason"] = "no_global_path_forward_target"
            return target, source, status
        forward_dist = get_distance(self.current_pos, forward)
        forward_yaw_error = self._yaw_error_to_target(forward)
        # 교체 target이 기존보다 더 가깝거나 더 심한 후방 target이면 교체하지 않는다.
        if forward_dist <= max(1.0, target_distance + 1.0):
            status["reason"] = "forward_target_not_far_enough"
            status["forward_distance"] = forward_dist
            status["forward_yaw_error_deg"] = forward_yaw_error
            return target, source, status

        status.update({
            "active": True,
            "reason": "near_or_bad_bearing_local_target_replaced",
            "source_after": "global_path_forward_guard",
            "target_before": {"x": target[0], "y": target[1]},
            "target_after": {"x": forward[0], "y": forward[1]},
            "forward_distance": forward_dist,
            "forward_yaw_error_deg": forward_yaw_error,
            "too_close": too_close,
            "bad_bearing": bad_bearing,
        })
        return forward, "global_path_forward_guard", status

    def choose_target(self) -> Tuple[Optional[Tuple[float, float]], str]:
        now = self.get_clock().now().nanoseconds / 1e9
        if self.enable_local_target and self.local_target is not None and now - self.local_target_stamp <= self.target_ttl_sec:
            return self.local_target, "local_target_apf"
        if self.lookahead_target is not None and now - self.lookahead_stamp <= self.target_ttl_sec:
            return self.lookahead_target, "astar_lookahead"
        if self.goal_pos is not None:
            return self.goal_pos, "goal_pose_fallback"
        return None, "no_target"

    def calculate_steering(self, target: Tuple[float, float]) -> Tuple[str, float, float, float]:
        assert self.current_pos is not None
        dx = target[0] - self.current_pos[0]
        dy = target[1] - self.current_pos[1]
        desired_yaw = math.degrees(math.atan2(dx, dy))
        yaw_error = normalize_angle(desired_yaw - self.current_yaw)
        # yaw_rate: 헤딩 자체의 회전속도(deg/s). 타겟(setpoint) 변화엔 안 반응 → derivative kick 방지.
        if self._last_current_yaw is None:
            yaw_rate = 0.0
        else:
            yaw_rate = normalize_angle(self.current_yaw - self._last_current_yaw) * self.controller_hz
        self._last_current_yaw = self.current_yaw
        if abs(yaw_error) < self.heading_deadband_deg:
            return "", 0.0, yaw_error, desired_yaw
        # PD: P가 타겟으로 끌고, D(rate feedback)가 빠른 회전을 눌러 오버슈팅(=A↔D weaving) 억제.
        # 회전명령을 0으로 죽이지 않아(coast 없음) 좁은 코리더에서 언더스티어로 벽 긁는 부작용 없음.
        u = yaw_error - self.steering_kd * yaw_rate
        cmd = "D" if u > 0 else "A"
        weight = abs(u) / self.steering_full_error_deg
        if self.min_ad_weight > 0:
            weight = max(self.min_ad_weight, weight)
        weight = clamp(weight, 0.0, self.max_ad_weight)
        return cmd, weight, yaw_error, desired_yaw

    def planner_path_points_cb(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
            points = payload.get("points", []) if isinstance(payload, dict) else []
            if not isinstance(points, list):
                return
            cleaned: List[Dict[str, Any]] = []
            for point in points:
                if not isinstance(point, dict):
                    continue
                try:
                    cleaned.append({
                        "x": float(point["x"]), "y": float(point["y"]),
                        "phase": str(point.get("phase", "cruise")),
                        "point_type": str(point.get("point_type", "straight")),
                        "recommended_ws_weight": float(point.get("recommended_ws_weight", self.straight_ws_weight)),
                        "recommended_speed_mps": float(point.get("recommended_speed_mps", 0.0)),
                        "distance_to_goal_m": float(point.get("distance_to_goal_m", 0.0)),
                    })
                except (KeyError, TypeError, ValueError):
                    continue
            self.planner_speed_points = cleaned
            self.planner_vehicle_geometry = payload.get("vehicle_geometry", {}) if isinstance(payload, dict) else {}
            self.planner_route_version = payload.get("route_version") if isinstance(payload, dict) else None
            self._planner_profile_stamp_mono = time.monotonic()
        except Exception:
            return

    def _planner_profile_at_current_pose(self) -> Optional[Dict[str, Any]]:
        if not self.enable_planner_speed_profile or self.current_pos is None:
            self._planner_profile_status = {"active": False, "reason": "disabled_or_no_pose"}
            return None
        age = time.monotonic() - self._planner_profile_stamp_mono
        if not self.planner_speed_points or age > self.planner_speed_profile_ttl_sec:
            self._planner_profile_status = {"active": False, "reason": "stale_or_empty", "age_sec": round(age, 3)}
            return None
        nearest = min(self.planner_speed_points, key=lambda p: get_distance(self.current_pos, (p["x"], p["y"])))
        profile = dict(nearest)
        profile["age_sec"] = round(age, 3)
        profile["active"] = True
        self._planner_profile_status = profile
        return profile

    def _effective_goal_stop_distance(self) -> float:
        profile = self._planner_profile_at_current_pose()
        if profile is not None:
            return self.planner_goal_stop_distance_m
        return self.goal_tolerance

    def apply_planner_speed_profile(
        self, cmd_ws: str, ws_weight: float, speed_mode: str
    ) -> Tuple[str, float, str, Dict[str, Any]]:
        profile = self._planner_profile_at_current_pose()
        if profile is None:
            return cmd_ws, ws_weight, speed_mode, dict(self._planner_profile_status)
        phase = str(profile.get("phase", "cruise"))
        suggested = max(0.0, min(1.0, float(profile.get("recommended_ws_weight", ws_weight))))
        distance_to_goal = float(profile.get("distance_to_goal_m", 0.0))
        # 실제 goal 근처에서만 stop phase를 강제한다. 오래된 마지막 point가 멀리 있는 문제를 막는다.
        if phase == "stop" and self.goal_pos is not None and self.current_pos is not None and get_distance(self.current_pos, self.goal_pos) <= self.planner_goal_stop_distance_m:
            return "STOP", 1.0, "planner_stop", profile
        if cmd_ws == "W" and suggested < ws_weight:
            return "W", suggested, f"{speed_mode}|planner_{phase}", profile
        if cmd_ws == "W" and phase == "curve":
            return "W", min(ws_weight, suggested), f"{speed_mode}|planner_curve", profile
        return cmd_ws, ws_weight, speed_mode, profile

    def calculate_speed(self, target: Tuple[float, float], yaw_error: float) -> Tuple[str, float, str]:
        assert self.current_pos is not None
        abs_err = abs(yaw_error)
        if self.goal_pos is not None and get_distance(self.current_pos, self.goal_pos) < self._effective_goal_stop_distance():
            self.mission_complete = True

            # The ballistic checkpoint needs this node to keep publishing
            # /tank/control/command; otherwise the turret override cannot reach
            # the simulator after the first second at the checkpoint.
            if self.exit_on_goal_reached:
                if not hasattr(self, '_stop_published_count'):
                    self._stop_published_count = 0
                self._stop_published_count += 1
                if self._stop_published_count > 10:
                    self.get_logger().info(
                        f"Mission Complete [{self.mission_type}]: Destination Reached. "
                        "Terminating Control Node."
                    )
                    import sys
                    sys.exit(0)
            return "STOP", 1.0, "goal_reached"
        if get_distance(self.current_pos, target) < 1.0 and self.goal_pos and get_distance(target, self.goal_pos) < self.stop_distance:
            return "STOP", 1.0, "target_stop"
        # 새 전차 제어 데이터셋 기준:
        # - STOP + A/D 제자리 회전은 검증 데이터가 부족함.
        # - W=0.1~0.2 + A/D는 2~4m급 회전반경으로 검증되어 있어 stuck 위험이 낮다.
        if abs_err > self.crawl_pivot_angle_deg:
            w_ws = clamp(self.crawl_pivot_ws_weight, 0.05, 1.0)
            return "W", w_ws, "crawl_pivot"
        if abs_err > self.rotate_in_place_angle_deg:
            w_ws = clamp(self.crawl_turn_ws_weight, 0.05, 1.0)
            return "W", w_ws, "crawl_turn"

        # W/S weight는 시뮬레이터 입력 그 자체다.
        # tank_parameters.yaml의 max_speed가 갱신되어도 명령 weight가 임의로 축소되지 않도록
        # 기존 speed_factor = max_speed / 19.45 보정은 제거한다.
        if abs_err > self.slowdown_angle_deg:
            w_ws = clamp(self.turn_ws_weight, 0.1, 1.0)
            return "W", w_ws, "slow_turn"

        error_scale = 1.0 - (abs_err / self.slowdown_angle_deg) * 0.3
        w_ws = clamp(self.straight_ws_weight * error_scale, 0.1, 1.0)
        return "W", w_ws, "cruise"

    def escape_command_if_needed(self, target: Optional[Tuple[float, float]] = None) -> Optional[Tuple[Dict[str, Any], str]]:
        if not self.enable_stuck_escape or self.current_pos is None:
            return None
        # Goal/checkpoint에서의 STOP은 의도된 대기다. 이를 stuck으로 오인해
        # 후진하면 포탑 조준 중 차체가 흔들리고, 사격 뒤 복귀 goal도 망가진다.
        if self.goal_pos is not None and get_distance(self.current_pos, self.goal_pos) < self._effective_goal_stop_distance():
            t = self.current_sim_time if self.current_sim_time > 0.0 else time.time()
            self.last_stuck_check_pos = self.current_pos
            self.last_stuck_check_yaw = self.current_yaw
            self.last_stuck_check_time = t
            return None
        # 정찰 의도적 dwell(정지관측) 중에는 stuck-escape를 끈다 — 의도 정지를 끼임으로 오인 방지.
        # 베이스라인을 현재 위치/시각으로 리셋해 dwell 종료 후 stuck 윈도가 새로 시작하게 한다.
        if self.mission_type == "recon" and not self.is_escaping and time.time() < self._recon_dwell_until:
            self.last_stuck_check_pos = self.current_pos
            self.last_stuck_check_time = self.current_sim_time if self.current_sim_time > 0.0 else time.time()
            return None
        t = self.current_sim_time if self.current_sim_time > 0.0 else time.time()
        if self.is_escaping:
            elapsed = t - self.escape_start_time
            if elapsed < self.escape_reverse_sec:
                return self.make_action("S", 1.0, "", 0.0), "escape_reverse"
            if elapsed < self.escape_reverse_sec + self.escape_turn_sec:
                pivot_dir = "D"
                if target is not None and self.current_pos is not None:
                    dx = target[0] - self.current_pos[0]
                    dy = target[1] - self.current_pos[1]
                    desired_yaw = math.degrees(math.atan2(dx, dy))
                    yaw_error = normalize_angle(desired_yaw - self.current_yaw)
                    pivot_dir = "D" if yaw_error > 0 else "A"
                return self.make_action("W", 0.4, pivot_dir, 1.0), "escape_pivot"
            self.is_escaping = False
            self.last_stuck_check_time = t
            self.last_stuck_check_pos = self.current_pos
            return None
            
        # 첫 호출(또는 baseline 미초기화) 시 stuck 판정 기준점을 현재 상태/시각으로 잡고
        # 한 주기(stuck_check_period)만큼 실제 주행할 시간을 준다.
        # ※ last_stuck_check_time은 0.0으로 초기화되는데 sim_time은 이미 크기 때문에,
        #   이 시각 가드가 없으면 출발 첫 사이클에 (t-0)>period 가 즉시 참이 되고
        #   moved≈0 으로 "끼임" 오판 → 출발 직후 계속 후진하는 버그가 난다.
        if self.last_stuck_check_pos is None or self.last_stuck_check_time <= 0.0:
            self.last_stuck_check_pos = self.current_pos
            self.last_stuck_check_yaw = self.current_yaw
            self.last_stuck_check_time = t
            return None

        if (t - self.last_stuck_check_time) > self.stuck_check_period:
            moved = get_distance(self.current_pos, self.last_stuck_check_pos)
            yaw_moved = abs(normalize_angle(self.current_yaw - self.last_stuck_check_yaw))
            if moved < self.stuck_min_movement and yaw_moved < 15.0:
                self.is_escaping = True
                self.escape_start_time = t
                self.get_logger().warn(f"stuck detected: moved={moved:.2f}m, yaw_moved={yaw_moved:.2f}deg, starting escape")
                return self.make_action("S", 1.0, "", 0.0), "escape_start_reverse"
            self.last_stuck_check_time = t
            self.last_stuck_check_pos = self.current_pos
            self.last_stuck_check_yaw = self.current_yaw
        return None

    def observe_request_cb(self, msg: String) -> None:
        try:
            self._observe_payload = json.loads(msg.data)
            self._observe_wall = time.time()
        except Exception:
            pass

    def turret_feedback_cb(self, msg: Vector3Stamped) -> None:
        self._turret_world_deg = float(msg.vector.x)

    @staticmethod
    def _normalize_180(angle: float) -> float:
        return (float(angle) + 180.0) % 360.0 - 180.0

    def _near_dwelled_spot(self) -> bool:
        """현재 위치가 이미 dwell(관측)한 지점 반경 내인가 — 같은 곳 무한 재dwell 방지."""
        if self.current_pos is None:
            return False
        cx, cy = self.current_pos[0], self.current_pos[1]
        r2 = self.recon_dwell_spot_radius_m ** 2
        return any((cx - sx) ** 2 + (cy - sy) ** 2 <= r2 for sx, sy in self._dwelled_spots)

    def apply_recon_observation(
        self, cmd_ws: str, w_ws: float, speed_mode: str
    ) -> Tuple[str, float, str, Dict[str, Any], Dict[str, Any]]:
        """정찰 전용: observe_request(미분류 후보) 기반 ②감속/dwell + ③포탑 step-stare.
        경로/조향은 안 건드리고 속도(W)와 turret만 만진다. 반환:
        (cmd_ws, w_ws, speed_mode, turret_qe, status)."""
        turret_qe = {"command": "", "weight": 0.0}
        status: Dict[str, Any] = {"active": False}
        if not self.recon_observe_enabled:
            return cmd_ws, w_ws, speed_mode, turret_qe, status
        payload = self._observe_payload
        now = time.time()
        if not isinstance(payload, dict) or (now - self._observe_wall) > self.recon_observe_stale_sec:
            return cmd_ws, w_ws, speed_mode, turret_qe, status
        has_fov = bool(payload.get("has_pending_fov"))
        has_side = bool(payload.get("has_pending_side"))
        cands = payload.get("candidates") or []
        status = {"active": True, "has_pending_fov": has_fov, "has_pending_side": has_side}

        # ② 전방 미분류 후보 → 감속(깨끗한 프레임), 고우선·근접이면 잠깐 정지(dwell).
        if has_fov:
            fov_cands = [c for c in cands if c.get("in_forward_fov")]
            best = max(fov_cands, key=lambda c: c.get("priority", 0.0)) if fov_cands else None
            dwelling = now < self._recon_dwell_until
            if (best is not None and not dwelling and now >= self._recon_dwell_cooldown_until
                    and not self._near_dwelled_spot()           # 같은 지점 재dwell 금지 — 장애물 앞 무한 dwell 데드락 방지
                    and float(best.get("priority", 0.0)) >= self.recon_dwell_priority
                    and float(best.get("distance_m", 1e9)) <= self.recon_dwell_distance_m):
                self._recon_dwell_until = now + self.recon_dwell_sec
                if self.current_pos is not None:
                    self._dwelled_spots.append((self.current_pos[0], self.current_pos[1]))
                dwelling = True
            if dwelling:
                cmd_ws, w_ws, speed_mode = "STOP", 1.0, "recon_dwell"
                status["mode"] = "dwell"
            else:
                if self._recon_dwell_until and now >= self._recon_dwell_until:
                    self._recon_dwell_cooldown_until = now + self.recon_dwell_cooldown_sec
                    self._recon_dwell_until = 0.0
                if cmd_ws == "W" and w_ws > self.recon_observe_ws_weight:
                    w_ws = self.recon_observe_ws_weight
                    speed_mode = f"{speed_mode}|recon_observe_slow"
                    status["mode"] = "slow"

        # ③ 전방이 비었을 때만(전방 우선) 옆(전방-대각) 미분류 후보로 포탑을 폐루프로 돌려 응시.
        if self.recon_turret_enable and has_side and not has_fov:
            g = payload.get("best_side_bearing_global_deg")
            if g is not None and self._turret_world_deg is not None:
                err = self._normalize_180(float(g) - float(self._turret_world_deg))
                tw = round(float(self._turret_world_deg), 1)
                if abs(err) > self.recon_turret_tol_deg:
                    direction = "E" if (err * self.recon_turret_qe_sign) > 0 else "Q"
                    w = clamp(abs(err) / 30.0, 0.2, 1.0) * self.recon_turret_max_weight
                    turret_qe = {"command": direction, "weight": float(w)}
                    # 포탑이 슬루 중이면 차체 감속 — 포탑이 따라잡기 전에 후보가 뒤로 빠지지 않게(능동지각엔 시간 필요).
                    if cmd_ws == "W" and w_ws > self.recon_observe_ws_weight:
                        w_ws = self.recon_observe_ws_weight
                        speed_mode = f"{speed_mode}|recon_turret_pursue_slow"
                        status["mode"] = status.get("mode") or "turret_slow"
                    status["turret"] = {"err_deg": round(err, 1), "cmd": direction, "w": round(w, 2), "world_deg": tw}
                else:
                    status["turret"] = {"on_target": True, "err_deg": round(err, 1), "world_deg": tw}
        return cmd_ws, w_ws, speed_mode, turret_qe, status

    def make_action(self, cmd_ws: str, w_ws: float, cmd_ad: str, w_ad: float) -> Dict[str, Any]:
        return {
            "moveWS": {"command": cmd_ws, "weight": float(clamp(w_ws, 0.0, 1.0))},
            "moveAD": {"command": cmd_ad, "weight": float(clamp(w_ad, 0.0, 1.0))},
            "turretQE": {"command": "", "weight": 0.0},
            "turretRF": {"command": "", "weight": 0.0},
            "fire": False,
        }

    @staticmethod
    def _sanitize_turret_axis(part: Any, allowed: set[str]) -> Dict[str, Any]:
        """Validate a turret axis from the override boundary without throwing."""
        if not isinstance(part, dict):
            return {"command": "", "weight": 0.0}
        command = str(part.get("command", ""))
        if command not in allowed:
            command = ""
        try:
            weight = float(part.get("weight", 0.0))
        except (TypeError, ValueError):
            weight = 0.0
        weight = clamp(weight, 0.0, 1.0)
        if not command:
            weight = 0.0
        return {"command": command, "weight": weight}

    def apply_turret_override(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Merge a fresh active override into one simulator command.

        ``hold_motion`` intentionally cancels W/S and A/D during the final
        ballistic aiming phase.  The override expires quickly if its producer
        stops, restoring the normal driving controller automatically.
        """
        override = self._turret_override
        age = time.monotonic() - self._turret_override_stamp
        if not isinstance(override, dict) or age > self.turret_override_ttl_sec:
            return action, {"active": False, "reason": "missing_or_stale", "age_sec": round(age, 3)}
        if not bool(override.get("active", False)):
            return action, {"active": False, "reason": "inactive", "age_sec": round(age, 3)}

        action["turretQE"] = self._sanitize_turret_axis(override.get("turretQE"), {"Q", "E"})
        action["turretRF"] = self._sanitize_turret_axis(override.get("turretRF"), {"R", "F"})
        action["fire"] = bool(override.get("fire", False))
        hold_motion = bool(override.get("hold_motion", False))
        if hold_motion:
            action["moveWS"] = {"command": "STOP", "weight": 1.0}
            action["moveAD"] = {"command": "", "weight": 0.0}
        status = override.get("status") if isinstance(override.get("status"), dict) else {}
        return action, {
            "active": True,
            "age_sec": round(age, 3),
            "hold_motion": hold_motion,
            "fire": action["fire"],
            "turretQE": action["turretQE"],
            "turretRF": action["turretRF"],
            "producer": status,
        }

    def publish_command(self, action: Dict[str, Any]) -> None:
        msg = String()
        msg.data = json.dumps(action, ensure_ascii=False, separators=(",", ":"))
        self.pub_cmd.publish(msg)

    def publish_status(self, payload: Dict[str, Any]) -> None:
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self.pub_status.publish(msg)

    def timer_cb(self) -> None:
        if self._last_pose_wall_time > 0.0 and time.time() - self._last_pose_wall_time > 2.0:
            self.publish_command(empty_action())
            self.publish_status({"ok": False, "reason": "player_pose_stale_fallback"})
            return

        if self.current_pos is None:
            self.publish_command(empty_action())
            self.publish_status({"ok": False, "reason": "no_player_pose"})
            return
        target, source = self.choose_target()
        if target is None:
            self.publish_command(empty_action())
            self.publish_status({"ok": False, "reason": source, "current": {"x": self.current_pos[0], "y": self.current_pos[1]}})
            return

        target, source, self.target_guard_status = self.apply_forward_target_guard(target, source)

        escape = self.escape_command_if_needed(target)
        if escape is not None:
            action, mode = escape
            action, turret_override_status = self.apply_turret_override(action)
            self.publish_command(action)
            self.publish_status({
                "ok": True, "mode": mode, "target_source": source,
                "turret_override": turret_override_status, "command": action,
            })
            return

        cmd_ad, w_ad, yaw_error, desired_yaw = self.calculate_steering(target)
        cmd_ws, w_ws, speed_mode = self.calculate_speed(target, yaw_error)

        apf_vector_control = self.apf_vector_control_decision()
        sharp_turn_control = self.sharp_turn_pivot_decision(target, source, yaw_error, cmd_ad)
        apf_vector_override = False
        sharp_turn_override = False
        steering_lock_status: Dict[str, Any] = {"enabled": self.enable_steering_direction_lock, "reason": "not_applied"}
        ad_oscillation_status: Dict[str, Any] = {"enabled": self.enable_ad_oscillation_guard, "reason": "not_applied"}
        if apf_vector_control.get("stop_pivot", False):
            # APF 합벡터 방향과 차체 heading 차이가 크고 장애물이 가까우면 W를 완전히 끊는다.
            cmd_ws = "STOP"
            w_ws = 1.0
            cmd_ad = apf_vector_control.get("turn_cmd") or cmd_ad
            cmd_ad, steering_lock_status = self.apply_steering_direction_lock(cmd_ad, "apf_stop_pivot")
            w_ad = 1.0 if cmd_ad else 0.0
            speed_mode = "apf_vector_stop_pivot"
            apf_vector_override = True
        elif sharp_turn_control.get("active", False):
            # 경로가 확 꺾이는 구간에서는 W를 누르지 않고 A/D로 차체 방향을 먼저 맞춘다.
            cmd_ws = "STOP"
            w_ws = 1.0
            cmd_ad = sharp_turn_control.get("turn_cmd") or cmd_ad
            cmd_ad, steering_lock_status = self.apply_steering_direction_lock(cmd_ad, "sharp_path_turn_stop_pivot")
            w_ad = 1.0 if cmd_ad else 0.0
            speed_mode = "sharp_path_turn_stop_pivot"
            sharp_turn_override = True
        else:
            if apf_vector_control.get("slow_turn", False) and cmd_ws == "W":
                # APF 방향 차이가 중간 수준이면 W는 허용하되 매우 낮은 속도로만 진행한다.
                if w_ws > self.apf_slow_ws_weight:
                    w_ws = self.apf_slow_ws_weight
                    speed_mode = f"{speed_mode}|apf_vector_slow"
                    apf_vector_override = True
            # STOP pivot이 아니더라도 A/D 토글은 짧게 억제한다.
            cmd_ad, steering_lock_status = self.apply_steering_direction_lock(cmd_ad, "normal_or_slow_turn")

        # 최종 A/D 명령에 대해 oscillation guard를 한 번 더 적용한다.
        cmd_ad, ad_oscillation_status = self.apply_ad_oscillation_guard(cmd_ad, speed_mode)
        if ad_oscillation_status.get("active", False):
            if cmd_ad in ("A", "D"):
                w_ad = max(w_ad, 0.85)
            if cmd_ws == "W" and w_ws > self.ad_oscillation_slow_ws_weight:
                w_ws = self.ad_oscillation_slow_ws_weight
                speed_mode = f"{speed_mode}|ad_oscillation_guard"

        # 장애물이 옆에 있고 전방을 막지 않는 통과 상황은 STOP 대신 저속 W를 허용한다.
        if (
            cmd_ws == "W"
            and apf_vector_control.get("side_pass", False)
            and not apf_vector_control.get("front_blocking", False)
            and w_ws < self.side_pass_ws_weight
        ):
            w_ws = self.side_pass_ws_weight
            speed_mode = f"{speed_mode}|side_pass_forward"

        safety_limit, safety_mode = self.current_safety_limit()
        safety_clamped = False
        if cmd_ws == "W" and safety_limit is not None and w_ws > safety_limit:
            w_ws = safety_limit
            speed_mode = f"{speed_mode}|safety_{safety_mode}"
            safety_clamped = True

        cmd_ws, w_ws, speed_mode, turn_overspeed_guard_status = self.apply_turn_overspeed_guard(
            cmd_ws, w_ws, yaw_error, speed_mode, safety_mode, apf_vector_control
        )
        # planner 곡선/감속 profile은 최종 W 상한으로만 적용한다. 안전/APF STOP은 항상 우선한다.
        cmd_ws, w_ws, speed_mode, planner_profile_status = self.apply_planner_speed_profile(cmd_ws, w_ws, speed_mode)
        # 정찰 전용: 미분류 후보 관측을 위한 ②감속/dwell + ③포탑 step-stare (recon에서만, 마지막에 적용).
        recon_turret_qe: Dict[str, Any] = {"command": "", "weight": 0.0}
        recon_obs_status: Dict[str, Any] = {"active": False}
        if self.mission_type == "recon":
            cmd_ws, w_ws, speed_mode, recon_turret_qe, recon_obs_status = self.apply_recon_observation(
                cmd_ws, w_ws, speed_mode
            )
        action = self.make_action(cmd_ws, w_ws, cmd_ad, w_ad)
        if recon_turret_qe.get("command"):
            action["turretQE"] = recon_turret_qe
        # 교전 노드의 override를 마지막에 합성한다. 특히 checkpoint에서 active이면
        # goal_reached의 control:pause를 억제해 simulator가 /get_action을 계속 호출하고
        # Q/E/R/F 및 fire pulse를 실제로 받을 수 있게 한다.
        action, turret_override_status = self.apply_turret_override(action)
        if (
            speed_mode == "goal_reached"
            and self.pause_on_goal_reached
            and not turret_override_status.get("active", False)
        ):
            # 목적지 도달 시 시뮬레이터 일시정지 요청.
            # 브릿지 select_action_command가 latest_command를 그대로 반환하므로
            # 이 control 필드가 /get_action 응답에 실려 시뮬로 전달된다.
            # (시뮬이 get_action 응답의 control:pause를 지원하면 정지, 미지원이면 STOP만 적용 — 무해)
            action["control"] = "pause"
        self.publish_command(action)

        mean_cte = 0.0
        max_cte = 0.0
        if self.global_path and self.trajectory_history:
            from .trajectory_metrics import calculate_cross_track_error
            try:
                mean_cte, max_cte = calculate_cross_track_error(self.trajectory_history, self.global_path)
            except Exception as exc:
                self.get_logger().error(f"Failed to calculate CTE: {exc}")

        self.publish_status({
            "ok": True,
            "target_source": source,
            "target": {"x": target[0], "y": target[1]},
            "current": {"x": self.current_pos[0], "y": self.current_pos[1]},
            "distance_to_target": get_distance(self.current_pos, target),
            "distance_to_goal": get_distance(self.current_pos, self.goal_pos) if self.goal_pos else None,
            "current_yaw_deg": self.current_yaw,
            "desired_yaw_deg": desired_yaw,
            "yaw_error_deg": yaw_error,
            "current_speed_mps": self.current_speed,
            "speed_mode": speed_mode,
            "recon_observation": recon_obs_status,
            "planner_speed_profile": planner_profile_status,
            "planner_vehicle_geometry": self.planner_vehicle_geometry,
            "safety": {
                "enabled": self.enable_safety_speed_limit,
                "mode": self.safety_mode,
                "limit_ws": safety_limit,
                "nearest_obstacle_distance": self.safety_nearest_obstacle_distance,
                "clamped": safety_clamped,
                "apf_vector_override": apf_vector_override,
                "sharp_turn_override": sharp_turn_override,
                "apf_vector_control": apf_vector_control,
                "sharp_turn_control": sharp_turn_control,
                "steering_direction_lock": steering_lock_status,
                "ad_oscillation_guard": ad_oscillation_status,
                "turn_overspeed_guard": turn_overspeed_guard_status,
                "target_guard": self.target_guard_status,
            },
            "turret_override": turret_override_status,
            "controller_profile": {
                "straight_ws_weight": self.straight_ws_weight,
                "turn_ws_weight": self.turn_ws_weight,
                "enable_planner_speed_profile": self.enable_planner_speed_profile,
                "planner_speed_profile_ttl_sec": self.planner_speed_profile_ttl_sec,
                "planner_goal_stop_distance_m": self.planner_goal_stop_distance_m,
                "crawl_pivot_ws_weight": self.crawl_pivot_ws_weight,
                "crawl_turn_ws_weight": self.crawl_turn_ws_weight,
                "crawl_pivot_angle_deg": self.crawl_pivot_angle_deg,
                "rotate_in_place_angle_deg": self.rotate_in_place_angle_deg,
                "slowdown_angle_deg": self.slowdown_angle_deg,
                "enable_apf_vector_stop_pivot": self.enable_apf_vector_stop_pivot,
                "prefer_turn_away_from_nearest_obstacle": self.prefer_turn_away_from_nearest_obstacle,
                "away_turn_lock_sec": self.away_turn_lock_sec,
                "apf_stop_pivot_max_sec": self.apf_stop_pivot_max_sec,
                "apf_stop_pivot_release_angle_deg": self.apf_stop_pivot_release_angle_deg,
                "apf_stop_pivot_cooldown_sec": self.apf_stop_pivot_cooldown_sec,
                "apf_stop_angle_deg": self.apf_stop_angle_deg,
                "apf_slow_angle_deg": self.apf_slow_angle_deg,
                "apf_stop_distance": self.apf_stop_distance,
                "apf_ttc_stop_sec": self.apf_ttc_stop_sec,
                "apf_slow_ws_weight": self.apf_slow_ws_weight,
                "enable_steering_direction_lock": self.enable_steering_direction_lock,
                "steering_direction_lock_sec": self.steering_direction_lock_sec,
                "enable_side_pass_forward": self.enable_side_pass_forward,
                "side_pass_bearing_deg": self.side_pass_bearing_deg,
                "side_pass_min_distance": self.side_pass_min_distance,
                "side_pass_ws_weight": self.side_pass_ws_weight,
                "front_stop_bearing_deg": self.front_stop_bearing_deg,
                "hard_stop_distance": self.hard_stop_distance,
                "enable_ad_oscillation_guard": self.enable_ad_oscillation_guard,
                "ad_flip_window_sec": self.ad_flip_window_sec,
                "ad_flip_threshold": self.ad_flip_threshold,
                "ad_oscillation_hold_sec": self.ad_oscillation_hold_sec,
                "enable_sharp_turn_stop_pivot": self.enable_sharp_turn_stop_pivot,
                "sharp_turn_stop_angle_deg": self.sharp_turn_stop_angle_deg,
                "sharp_turn_release_angle_deg": self.sharp_turn_release_angle_deg,
                "sharp_turn_min_target_distance": self.sharp_turn_min_target_distance,
                "sharp_turn_max_sec": self.sharp_turn_max_sec,
                "sharp_turn_cooldown_sec": self.sharp_turn_cooldown_sec,
                "sharp_turn_block_when_apf_side_pass": self.sharp_turn_block_when_apf_side_pass,
                "enable_turn_overspeed_guard": self.enable_turn_overspeed_guard,
                "turn_overspeed_angle_deg": self.turn_overspeed_angle_deg,
                "turn_overspeed_speed_mps": self.turn_overspeed_speed_mps,
                "turn_overspeed_hard_angle_deg": self.turn_overspeed_hard_angle_deg,
                "turn_overspeed_hard_speed_mps": self.turn_overspeed_hard_speed_mps,
                "turn_overspeed_reverse_weight": self.turn_overspeed_reverse_weight,
                "turn_overspeed_slow_ws_weight": self.turn_overspeed_slow_ws_weight,
                "danger_obstacle_brake_speed_mps": self.danger_obstacle_brake_speed_mps,
                "danger_obstacle_reverse_weight": self.danger_obstacle_reverse_weight,
                "enable_forward_target_guard": self.enable_forward_target_guard,
                "forward_guard_min_target_distance": self.forward_guard_min_target_distance,
                "forward_guard_yaw_error_deg": self.forward_guard_yaw_error_deg,
                "forward_guard_target_distance": self.forward_guard_target_distance,
            },
            "mission_complete": self.mission_complete,
            "collision_count": self.collision_count,
            "mean_cte": mean_cte,
            "max_cte": max_cte,
            "command": action,
        })


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TeamPathControllerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
