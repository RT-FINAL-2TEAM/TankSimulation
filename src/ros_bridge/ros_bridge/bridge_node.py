# -*- coding: utf-8 -*-
"""
Tank Challenge ROS2 Bridge Node

공식 문서 기준 역할:
- 3.2 API Docs의 /init, /start, /info, /get_action, /detect, /stereo_image,
  /update_bullet, /set_destination, /update_obstacle, /collision endpoint 데이터를
  Flask route에서 넘겨받아 ROS2 topic으로 publish한다.
- 2.3 메뉴 구성 및 기능 기준으로 monitor 모드는 logMode 중심 관측,
  auto 모드는 trackingMode 중심 제어로 해석한다.
- 3.2 API Docs의 /get_action 응답 형식(moveWS, moveAD, turretQE, turretRF, fire)을
  ROS2 알고리즘 노드가 publish한 /tank/control/command로부터 선택해 시뮬레이터에 반환한다.

********************************************************************************
- Publisher  : 이 노드가 ROS2 topic으로 데이터를 보낸다.
- Subscriber : 이 노드가 ROS2 topic에서 제어 명령을 받는다.
- Timer      : 일정 주기로 최신 상태를 다시 publish한다.
- Lock       : Flask thread와 ROS2 thread가 같은 변수를 동시에 수정하지 못하게 막는다.
********************************************************************************
"""


############################################################
# 1. Import
############################################################
# JSON 문자열로 들어오는 ROS2 String 명령을 dict로 파싱하기 위해 사용한다.
import json
# heading degree를 RViz/APF용 2D 방향 벡터로 변환하기 위해 사용한다.
import math
import os
# Flask thread와 ROS2 executor thread가 공유 상태를 동시에 만지므로 Lock을 쓰기 위해 import한다.
import threading
# 최신 상태 dict를 저장/publish할 때 원본 변경 부작용을 막기 위해 깊은 복사를 사용한다.
from copy import deepcopy
# 타입 힌트를 통해 팀원이 함수 입력/출력 구조를 쉽게 이해하도록 한다.
from typing import Any, Dict, Optional, Tuple

# RViz/경로계획에서 바로 쓸 수 있는 ROS2 geometry 메시지 타입을 가져온다.
from geometry_msgs.msg import PointStamped, PoseStamped, Vector3Stamped
from sensor_msgs.msg import CompressedImage
# ROS2 Python 클라이언트 라이브러리 rclpy를 사용한다.
import rclpy
# RosBridge가 ROS2 Node를 상속받기 위해 Node 클래스를 가져온다.
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
# JSON 문자열, 정수 카운트, 단순 이벤트를 publish하기 위한 표준 메시지를 가져온다.
from std_msgs.msg import Empty, Int32, String

# /get_action 제어 명령 생성/검증 함수들을 가져온다.
from .commands import fallback_command, neutral_command, validate_action_command
# 실행 모드, 명령 유효시간, 좌표 frame 이름 같은 전역 설정값을 가져온다.
from .config import AUTO_FALLBACK, COMMAND_TTL_SEC, MAP_FRAME, SAVE_FULL_INFO, TANK_MODE, UNITY_FRAME
# 시간, JSON 변환, 좌표 변환, 로그 저장 유틸리티를 가져온다.
from .utils import (
    # SAVE_JSONL 옵션이 켜져 있으면 이 endpoint 데이터를 JSONL 파일에 저장한다.
    append_jsonl,
    as_xyz,
    compact_info,
    dumps,
    now_wall,
    raw_and_map_pose,
    to_float,
    to_int,
)


def _forced_route_id() -> Optional[str]:
    raw = os.environ.get("TANK_FORCE_ROUTE", "A").strip().upper()
    if raw in {"", "0", "FALSE", "NO", "NONE", "OFF", "AUTO"}:
        return None
    if raw in {"A", "B"}:
        return raw
    return None


def _apply_forced_route_policy_to_result(result: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(result, dict):
        result = {}
    updated = deepcopy(result)
    forced = _forced_route_id()
    if forced not in {"A", "B"}:
        return updated

    label = "LEFT A" if forced == "A" else "RIGHT B"
    previous = str(updated.get("selected_route") or "").strip().upper()
    policy_note = f"임무 정책상 {label} 루트를 반드시 선택합니다. 위험 수치는 참고용으로만 표시합니다."
    updated["selected_route"] = forced
    updated["confidence"] = "high"
    updated["summary"] = policy_note
    existing_reason = str(updated.get("decision_reason") or "").strip()
    if previous in {"A", "B"} and previous != forced:
        updated["decision_reason"] = (
            f"{policy_note} 기존 LLM 판단은 {previous}였지만 강제 정책을 우선했습니다."
            + (f" 기존 판단: {existing_reason}" if existing_reason else "")
        )
    elif existing_reason and policy_note not in existing_reason:
        updated["decision_reason"] = f"{policy_note} 기존 판단: {existing_reason}"
    else:
        updated["decision_reason"] = policy_note
    return updated



############################################################
# 2. RosBridge 노드 클래스
############################################################
# RosBridge는 Flask 서버와 ROS2 topic 사이를 연결하는 중심 노드다.
class RosBridge(Node):
    """Tank Challenge Flask endpoint와 ROS2 topic 사이를 중계하는 핵심 node."""

    # 노드 이름, publisher, subscriber, timer, 내부 상태 저장소를 한 번에 초기화한다.
    def __init__(self) -> None:
        # ROS2 graph에 표시될 node 이름을 지정하고 Node 부모 클래스를 초기화한다.
        super().__init__("tank_ros_bridge_node")

        # 제어 명령 subscriber와 진단 timer가 무거운 상태 publish callback에 막히면
        # 안 되므로, 서로 다른 callback group을 쓴다.
        self.control_callback_group = ReentrantCallbackGroup()
        self.timer_callback_group = ReentrantCallbackGroup()

        # ----------------------------------------------------
        # 공식 endpoint 기반 topic들
        # ----------------------------------------------------
        # Publisher 생성: '/tank/api/init/config' topic으로 String 메시지를 publish한다.
        self.pub_init_config = self.create_publisher(String, "/tank/api/init/config", 10)
        # Publisher 생성: '/tank/api/start/event' topic으로 Empty 메시지를 publish한다.
        self.pub_start_event = self.create_publisher(Empty, "/tank/api/start/event", 10)

        # Publisher 생성: '/tank/api/info/raw' topic으로 String 메시지를 publish한다.
        self.pub_info_raw = self.create_publisher(String, "/tank/api/info/raw", 10)
        # Publisher 생성: '/tank/api/info/compact' topic으로 String 메시지를 publish한다.
        self.pub_info_compact = self.create_publisher(String, "/tank/api/info/compact", 10)
        # Publisher 생성: '/tank/api/info/player/pose_raw' topic으로 PoseStamped 메시지를 publish한다.
        self.pub_info_player_pose_raw = self.create_publisher(PoseStamped, "/tank/api/info/player/pose_raw", 10)
        # Publisher 생성: '/tank/api/info/player/pose_map' topic으로 PoseStamped 메시지를 publish한다.
        self.pub_info_player_pose_map = self.create_publisher(PoseStamped, "/tank/api/info/player/pose_map", 10)
        # Publisher 생성: '/tank/api/info/player/state' topic으로 String 메시지를 publish한다.
        self.pub_info_player_state = self.create_publisher(String, "/tank/api/info/player/state", 10)
        # Publisher 생성: '/tank/api/info/enemy/pose_raw' topic으로 PoseStamped 메시지를 publish한다.
        self.pub_info_enemy_pose_raw = self.create_publisher(PoseStamped, "/tank/api/info/enemy/pose_raw", 10)
        # Publisher 생성: '/tank/api/info/enemy/pose_map' topic으로 PoseStamped 메시지를 publish한다.
        self.pub_info_enemy_pose_map = self.create_publisher(PoseStamped, "/tank/api/info/enemy/pose_map", 10)
        # Publisher 생성: '/tank/api/info/enemy/state' topic으로 String 메시지를 publish한다.
        self.pub_info_enemy_state = self.create_publisher(String, "/tank/api/info/enemy/state", 10)
        # LiDAR 전용 parsing/publishing은 lidar 패키지가 담당한다.

        # Publisher 생성: '/tank/api/get_action/raw' topic으로 String 메시지를 publish한다.
        self.pub_get_action_raw = self.create_publisher(String, "/tank/api/get_action/raw", 10)
        # Publisher 생성: '/tank/api/get_action/pose_raw' topic으로 PoseStamped 메시지를 publish한다.
        self.pub_get_action_pose_raw = self.create_publisher(PoseStamped, "/tank/api/get_action/pose_raw", 10)
        # Publisher 생성: '/tank/api/get_action/pose_map' topic으로 PoseStamped 메시지를 publish한다.
        self.pub_get_action_pose_map = self.create_publisher(PoseStamped, "/tank/api/get_action/pose_map", 10)
        # Publisher 생성: '/tank/api/get_action/turret' topic으로 Vector3Stamped 메시지를 publish한다.
        self.pub_get_action_turret = self.create_publisher(Vector3Stamped, "/tank/api/get_action/turret", 10)
        # Publisher 생성: '/tank/api/get_action/response' topic으로 String 메시지를 publish한다.
        self.pub_get_action_response = self.create_publisher(String, "/tank/api/get_action/response", 10)

        # Publisher 생성: '/tank/api/detect/result' topic으로 String 메시지를 publish한다.
        self.pub_detect_result = self.create_publisher(String, "/tank/api/detect/result", 10)
        # LiDAR-camera overlay / visual perception용 turret camera compressed image topic.
        # Tank simulator는 /detect endpoint로 이미지를 보내므로, bridge가 이 bytes를 ROS2 sensor topic으로 중계한다.
        self.pub_camera_image_compressed = self.create_publisher(CompressedImage, "/tank/camera/image_compressed", 10)
        self.pub_api_detect_image_compressed = self.create_publisher(CompressedImage, "/tank/api/detect/image_compressed", 10)
        # Publisher 생성: '/tank/api/stereo_image/status' topic으로 String 메시지를 publish한다.
        self.pub_stereo_status = self.create_publisher(String, "/tank/api/stereo_image/status", 10)

        # Publisher 생성: '/tank/api/update_bullet/raw' topic으로 String 메시지를 publish한다.
        self.pub_bullet_raw = self.create_publisher(String, "/tank/api/update_bullet/raw", 10)
        # Publisher 생성: '/tank/api/update_bullet/impact_raw' topic으로 PointStamped 메시지를 publish한다.
        self.pub_bullet_impact_raw = self.create_publisher(PointStamped, "/tank/api/update_bullet/impact_raw", 10)
        # Publisher 생성: '/tank/api/update_bullet/impact_map' topic으로 PointStamped 메시지를 publish한다.
        self.pub_bullet_impact_map = self.create_publisher(PointStamped, "/tank/api/update_bullet/impact_map", 10)
        # Publisher 생성: '/tank/api/update_bullet/target' topic으로 String 메시지를 publish한다.
        self.pub_bullet_target = self.create_publisher(String, "/tank/api/update_bullet/target", 10)

        # Publisher 생성: '/tank/api/set_destination/raw' topic으로 String 메시지를 publish한다.
        self.pub_destination_raw = self.create_publisher(String, "/tank/api/set_destination/raw", 10)
        # Publisher 생성: '/tank/api/set_destination/pose_raw' topic으로 PoseStamped 메시지를 publish한다.
        self.pub_destination_pose_raw = self.create_publisher(PoseStamped, "/tank/api/set_destination/pose_raw", 10)
        # Publisher 생성: '/tank/api/set_destination/pose_map' topic으로 PoseStamped 메시지를 publish한다.
        self.pub_destination_pose_map = self.create_publisher(PoseStamped, "/tank/api/set_destination/pose_map", 10)

        # Publisher 생성: '/tank/api/update_obstacle/raw' topic으로 String 메시지를 publish한다.
        self.pub_obstacle_raw = self.create_publisher(String, "/tank/api/update_obstacle/raw", 10)
        # Publisher 생성: '/tank/api/update_obstacle/list' topic으로 String 메시지를 publish한다.
        self.pub_obstacle_list = self.create_publisher(String, "/tank/api/update_obstacle/list", 10)

        # Publisher 생성: '/tank/api/collision/raw' topic으로 String 메시지를 publish한다.
        self.pub_collision_raw = self.create_publisher(String, "/tank/api/collision/raw", 10)
        # Publisher 생성: '/tank/api/collision/point_raw' topic으로 PointStamped 메시지를 publish한다.
        self.pub_collision_point_raw = self.create_publisher(PointStamped, "/tank/api/collision/point_raw", 10)
        # Publisher 생성: '/tank/api/collision/point_map' topic으로 PointStamped 메시지를 publish한다.
        self.pub_collision_point_map = self.create_publisher(PointStamped, "/tank/api/collision/point_map", 10)

        # ----------------------------------------------------
        # 알고리즘/RViz용 안정 high-level topic들
        # ----------------------------------------------------
        # Publisher 생성: '/tank/state/latest' topic으로 String 메시지를 publish한다.
        self.pub_state_latest = self.create_publisher(String, "/tank/state/latest", 10)
        # Publisher 생성: '/tank/player/pose' topic으로 PoseStamped 메시지를 publish한다.
        self.pub_player_pose = self.create_publisher(PoseStamped, "/tank/player/pose", 10)
        # Publisher 생성: '/tank/enemy/pose' topic으로 PoseStamped 메시지를 publish한다.
        self.pub_enemy_pose = self.create_publisher(PoseStamped, "/tank/enemy/pose", 10)
        # Publisher 생성: '/tank/player/state' topic으로 String 메시지를 publish한다.
        self.pub_player_state = self.create_publisher(String, "/tank/player/state", 10)
        # Publisher 생성: '/tank/enemy/state' topic으로 String 메시지를 publish한다.
        self.pub_enemy_state = self.create_publisher(String, "/tank/enemy/state", 10)
        # Publisher 생성: '/tank/map/obstacles' topic으로 String 메시지를 publish한다.
        self.pub_obstacles = self.create_publisher(String, "/tank/map/obstacles", 10)
        # Publisher 생성: '/tank/goal/pose' topic으로 PoseStamped 메시지를 publish한다.
        self.pub_destination = self.create_publisher(PoseStamped, "/tank/goal/pose", 10)
        # ----------------------------------------------------
        # RViz / APF / planner 노드용 안정 파생 topic들
        # ----------------------------------------------------
        # LiDAR 전용 high-level topic(/tank/sensor/lidar/*)은 lidar 패키지가 담당한다.
        # ros_bridge는 /tank/api/info/raw와 pose/state처럼 HTTP 원본과 기본 상태만 publish한다.
        self.pub_sim_status = self.create_publisher(String, "/tank/sim/status", 10)
        self.pub_player_heading = self.create_publisher(Vector3Stamped, "/tank/player/heading", 10)
        self.pub_enemy_heading = self.create_publisher(Vector3Stamped, "/tank/enemy/heading", 10)
        self.pub_event_detection = self.create_publisher(String, "/tank/event/detection", 10)
        self.pub_perception_detections = self.create_publisher(String, "/tank/perception/detections", 10)
        self.pub_event_bullet = self.create_publisher(String, "/tank/event/bullet", 10)
        self.pub_event_collision = self.create_publisher(String, "/tank/event/collision", 10)

        # ----------------------------------------------------
        # 이전 스크립트와의 하위호환 alias들
        # ----------------------------------------------------
        # Publisher 생성: '/tank/latest_state' topic으로 String 메시지를 publish한다.
        self.pub_latest_state_alias = self.create_publisher(String, "/tank/latest_state", 10)
        # Publisher 생성: '/tank/latest_pose' topic으로 PoseStamped 메시지를 publish한다.
        self.pub_latest_pose_alias = self.create_publisher(PoseStamped, "/tank/latest_pose", 10)
        # Publisher 생성: '/tank/pose' topic으로 PoseStamped 메시지를 publish한다.
        self.pub_pose_alias = self.create_publisher(PoseStamped, "/tank/pose", 10)
        # Publisher 생성: '/tank/action_raw' topic으로 String 메시지를 publish한다.
        self.pub_action_raw_alias = self.create_publisher(String, "/tank/action_raw", 10)
        # Publisher 생성: '/tank/sent_action' topic으로 String 메시지를 publish한다.
        self.pub_sent_action_alias = self.create_publisher(String, "/tank/sent_action", 10)
        # Publisher 생성: '/tank/obstacles' topic으로 String 메시지를 publish한다.
        self.pub_obstacles_alias = self.create_publisher(String, "/tank/obstacles", 10)
        # Publisher 생성: '/tank/bullet_impact' topic으로 PointStamped 메시지를 publish한다.
        self.pub_bullet_alias = self.create_publisher(PointStamped, "/tank/bullet_impact", 10)

        # ----------------------------------------------------
        # ROS2 -> 시뮬레이터 명령 subscription들
        # ----------------------------------------------------
        # Subscriber 생성: '/tank/control/command' topic의 String 메시지를 받아 self.on_control_command callback으로 처리한다.
        self.sub_control_command = self.create_subscription(
            String,
            "/tank/control/command",
            self.on_control_command,
            10,
            callback_group=self.control_callback_group,
        )
        # Subscriber 생성: '/tank/api/get_action/override' topic의 String 메시지를 받아 self.on_one_shot_override callback으로 처리한다.
        self.sub_action_override = self.create_subscription(
            String,
            "/tank/api/get_action/override",
            self.on_one_shot_override,
            10,
            callback_group=self.control_callback_group,
        )
        # Subscriber 생성: '/tank/action_override' topic의 String 메시지를 받아 self.on_one_shot_override callback으로 처리한다.
        self.sub_action_override_legacy = self.create_subscription(
            String,
            "/tank/action_override",
            self.on_one_shot_override,
            10,
            callback_group=self.control_callback_group,
        )
        self.sub_route_risk_report = self.create_subscription(
            String,
            "/tank/risk/route_report",
            self.on_route_risk_report,
            10,
            callback_group=self.control_callback_group,
        )

        # ----------------------------------------------------
        # 센서 융합 데이터 로깅 subscription
        # ----------------------------------------------------
        self.sub_fused_objects = self.create_subscription(
            String,
            "/tank/perception/fused_objects",
            self.on_fused_objects,
            10,
            callback_group=self.control_callback_group,
        )

        # ----------------------------------------------------
        # LLM 위험도/전술 결정 구독 → MFD(aiLog)에 노출
        # route_risk_node가 발행한 최신 결정을 latest["decision"]에 저장하면
        # /api/dashboard/state가 aiLog로 노출하고 MFD 프론트가 렌더한다.
        # ----------------------------------------------------
        self.sub_risk_report = self.create_subscription(
            String,
            "/tank/risk/route_report",
            self.on_risk_report,
            10,
            callback_group=self.control_callback_group,
        )

        # ----------------------------------------------------
        # 내부 공유 상태
        # ----------------------------------------------------
        # 공유 상태 보호용 Lock이다. Flask thread와 ROS2 callback thread가 동시에 접근하는 것을 막는다.
        self._lock = threading.Lock()
        # /tank/control/command에서 받은 최신 지속 제어 명령을 저장한다.
        self._latest_command: Optional[Dict[str, Any]] = None
        # 최신 지속 제어 명령이 들어온 wall-clock 시간을 저장한다.
        self._latest_command_stamp: Optional[float] = None
        # 다음 /get_action 응답 1회에만 사용할 override 명령을 저장한다.
        self._one_shot_override: Optional[Dict[str, Any]] = None
        # 명령 수신 경로의 진단 로그를 throttle(빈도 제한)하기 위한 값이다.
        self._last_command_log_wall: float = 0.0

        # /init, /info, /get_action 등 endpoint별 수신 횟수를 기록한다.
        self._route_counts: Dict[str, int] = {}
        # /tank/state/latest로 주기 publish할 최신 상태 저장소를 초기화한다.
        self._latest: Dict[str, Any] = {
            # latest['init_config'] 초기값. 해당 endpoint 데이터가 들어오기 전까지는 None이다.
            "init_config": None,
            # latest['start_event'] 초기값. 해당 endpoint 데이터가 들어오기 전까지는 None이다.
            "start_event": None,
            # latest['info_raw'] 초기값. 해당 endpoint 데이터가 들어오기 전까지는 None이다.
            "info_raw": None,
            # latest['info_compact'] 초기값. 해당 endpoint 데이터가 들어오기 전까지는 None이다.
            "info_compact": None,
            # latest['player_state'] 초기값. 해당 endpoint 데이터가 들어오기 전까지는 None이다.
            "player_state": None,
            # latest['enemy_state'] 초기값. 해당 endpoint 데이터가 들어오기 전까지는 None이다.
            "enemy_state": None,
            # latest['player_pose_map'] 초기값. 해당 endpoint 데이터가 들어오기 전까지는 None이다.
            "player_pose_map": None,
            # latest['enemy_pose_map'] 초기값. 해당 endpoint 데이터가 들어오기 전까지는 None이다.
            "enemy_pose_map": None,
            # latest['sim_status'] 초기값. /info 기반 경량 주행 상태 요약이다.
            "sim_status": None,
            # latest['player_heading'] / latest['enemy_heading'] 초기값. 차체 yaw를 2D vector로 변환한 값이다.
            "player_heading": None,
            "enemy_heading": None,
            # latest['get_action_raw'] 초기값. 해당 endpoint 데이터가 들어오기 전까지는 None이다.
            "get_action_raw": None,
            # latest['get_action_pose_map'] 초기값. 해당 endpoint 데이터가 들어오기 전까지는 None이다.
            "get_action_pose_map": None,
            # latest['get_action_response'] 초기값. 해당 endpoint 데이터가 들어오기 전까지는 None이다.
            "get_action_response": None,
            # latest['detect_result'] 초기값. 해당 endpoint 데이터가 들어오기 전까지는 None이다.
            "detect_result": None,
            # latest['stereo_status'] 초기값. 해당 endpoint 데이터가 들어오기 전까지는 None이다.
            "stereo_status": None,
            # latest['bullet'] 초기값. 해당 endpoint 데이터가 들어오기 전까지는 None이다.
            "bullet": None,
            # latest['destination'] 초기값. 해당 endpoint 데이터가 들어오기 전까지는 None이다.
            "destination": None,
            # latest['obstacles'] 초기값. 해당 endpoint 데이터가 들어오기 전까지는 None이다.
            "obstacles": None,
            # latest['collision'] 초기값. 해당 endpoint 데이터가 들어오기 전까지는 None이다.
            "collision": None,
            # latest['route_risk_report'] 초기값. risk_analysis LLM 결과가 들어오기 전까지는 None이다.
            "route_risk_report": None,
            # Web dashboard AI tab compatibility fields.
            "ai_log": None,
            "llm_log": None,
            "decision": None,
        }

        # ROS2 timer를 등록한다. 현재는 0.1초마다 latest state를 publish하므로 10Hz 주기다.
        self.timer = self.create_timer(
            float(os.environ.get("TANK_LATEST_STATE_PERIOD_SEC", "1.0")),
            self.publish_latest_state,
            callback_group=self.timer_callback_group,
        )

        # 실행 터미널에 bridge 초기화 상태를 ROS2 logger로 출력한다.
        self.get_logger().info("Tank Challenge ROS2 final bridge initialized")
        # 실행 터미널에 bridge 초기화 상태를 ROS2 logger로 출력한다.
        self.get_logger().info(f"TANK_MODE={TANK_MODE}, COMMAND_TTL_SEC={COMMAND_TTL_SEC}, AUTO_FALLBACK={AUTO_FALLBACK}")
        # 실행 터미널에 bridge 초기화 상태를 ROS2 logger로 출력한다.
        self.get_logger().info("ROS2 command input: /tank/control/command")

    # --------------------------------------------------------
    # ROS 메시지 helper들
    # --------------------------------------------------------

    ########################################################
    # 4. ROS 메시지 publish helper 함수들
    ########################################################
    # dict/list 데이터를 JSON 문자열로 바꾸어 std_msgs/String topic에 publish하는 helper다.
    def publish_json(self, publisher, data: Any) -> None:
        # ROS2 String 메시지 객체를 새로 만든다.
        msg = String()
        # Python dict/list를 compact JSON 문자열로 직렬화해 msg.data에 넣는다.
        msg.data = dumps(data)
        # 완성된 ROS2 메시지를 실제 topic으로 publish한다.
        publisher.publish(msg)

    # 숫자 값을 std_msgs/Int32 topic에 publish하는 helper다.
    def publish_int(self, publisher, value: Any) -> None:
        # ROS2 Int32 메시지 객체를 새로 만든다.
        msg = Int32()
        # 안전하게 int로 변환한 값을 msg.data에 넣는다.
        msg.data = to_int(value)
        # 완성된 ROS2 메시지를 실제 topic으로 publish한다.
        publisher.publish(msg)

    # 좌표 dict를 geometry_msgs/PoseStamped로 변환해 publish하는 helper다.
    def publish_pose(self, publisher, pose: Dict[str, Any]) -> None:
        # ROS2 PoseStamped 메시지 객체를 새로 만든다.
        msg = PoseStamped()
        # 메시지 생성 시각을 ROS clock 기준 timestamp로 기록한다.
        msg.header.stamp = self.get_clock().now().to_msg()
        # 이 좌표가 어떤 좌표계(raw/map)에 속하는지 frame_id에 기록한다.
        msg.header.frame_id = pose.get("frame_id", MAP_FRAME)
        # pose의 x 위치를 안전하게 float로 변환해 채운다.
        msg.pose.position.x = to_float(pose.get("x"))
        # pose의 y 위치를 안전하게 float로 변환해 채운다.
        msg.pose.position.y = to_float(pose.get("y"))
        # pose의 z 위치를 안전하게 float로 변환해 채운다.
        msg.pose.position.z = to_float(pose.get("z"))
        # 회전 정보는 아직 계산하지 않으므로 단위 quaternion의 w=1.0만 넣는다.
        msg.pose.orientation.w = 1.0
        # 완성된 ROS2 메시지를 실제 topic으로 publish한다.
        publisher.publish(msg)

    # 좌표 dict를 geometry_msgs/PointStamped로 변환해 publish하는 helper다.
    def publish_point(self, publisher, point: Dict[str, Any]) -> None:
        # ROS2 PointStamped 메시지 객체를 새로 만든다.
        msg = PointStamped()
        # 메시지 생성 시각을 ROS clock 기준 timestamp로 기록한다.
        msg.header.stamp = self.get_clock().now().to_msg()
        # 이 좌표가 어떤 좌표계(raw/map)에 속하는지 frame_id에 기록한다.
        msg.header.frame_id = point.get("frame_id", MAP_FRAME)
        # point의 x 좌표를 안전하게 float로 변환해 채운다.
        msg.point.x = to_float(point.get("x"))
        # point의 y 좌표를 안전하게 float로 변환해 채운다.
        msg.point.y = to_float(point.get("y"))
        # point의 z 좌표를 안전하게 float로 변환해 채운다.
        msg.point.z = to_float(point.get("z"))
        # 완성된 ROS2 메시지를 실제 topic으로 publish한다.
        publisher.publish(msg)

    # x/y/z 벡터 dict를 geometry_msgs/Vector3Stamped로 변환해 publish하는 helper다.
    def publish_vector3(self, publisher, vector: Dict[str, Any], frame_id: str = UNITY_FRAME) -> None:
        # ROS2 Vector3Stamped 메시지 객체를 새로 만든다.
        msg = Vector3Stamped()
        # 메시지 생성 시각을 ROS clock 기준 timestamp로 기록한다.
        msg.header.stamp = self.get_clock().now().to_msg()
        # 이 좌표가 어떤 좌표계(raw/map)에 속하는지 frame_id에 기록한다.
        msg.header.frame_id = frame_id
        # vector의 x 성분을 안전하게 float로 변환해 채운다.
        msg.vector.x = to_float(vector.get("x"))
        # vector의 y 성분을 안전하게 float로 변환해 채운다.
        msg.vector.y = to_float(vector.get("y"))
        # vector의 z 성분을 안전하게 float로 변환해 채운다.
        msg.vector.z = to_float(vector.get("z"))
        # 완성된 ROS2 메시지를 실제 topic으로 publish한다.
        publisher.publish(msg)

    # heading degree를 RViz/APF에서 쓰기 쉬운 2D map-frame 단위 벡터로 변환한다.
    def heading_vector_from_degree(self, degree: Any) -> Dict[str, float]:
        # Tank Challenge 좌표 기준: body.x=0이면 +raw.z, 즉 RViz map의 +y 방향이다.
        rad = math.radians(to_float(degree))
        return {"x": math.sin(rad), "y": math.cos(rad), "z": 0.0}

    # endpoint별 수신 횟수를 1씩 증가시키는 내부 helper다.
    def _count(self, route: str) -> None:
        # /init, /info, /get_action 등 endpoint별 수신 횟수를 기록한다.
        self._route_counts[route] = self._route_counts.get(route, 0) + 1

    # latest 상태 dict의 특정 key를 안전하게 갱신하는 helper다.
    def update_latest(self, key: str, value: Any) -> None:
        # 공유 상태를 읽거나 쓸 때 Lock을 잡아 thread race condition을 방지한다.
        with self._lock:
            # 외부에서 받은 값을 참조로 최신 상태 저장소에 반영한다. (불필요한 deepcopy 제거)
            self._latest[key] = value

    def get_latest_snapshot(self) -> Dict[str, Any]:
        """Return a JSON-friendly copy of the bridge state for dashboard views."""
        now = now_wall()
        with self._lock:
            latest_command_age = None
            if self._latest_command_stamp is not None:
                latest_command_age = now - self._latest_command_stamp
            return {
                "available": True,
                "timestampWall": now,
                "mode": TANK_MODE,
                "routeCounts": deepcopy(self._route_counts),
                "route_counts": deepcopy(self._route_counts),
                "latest": deepcopy(self._latest),
                "latestCommand": deepcopy(self._latest_command),
                "latestCommandAgeSec": latest_command_age,
                "hasOneShotOverride": self._one_shot_override is not None,
                "coordinatePolicy": {
                    "raw": "Unity API x,y,z",
                    "map": "x=raw.x, y=raw.z, z=raw.y",
                },
            }

    # --------------------------------------------------------
    # 명령 callback들
    # --------------------------------------------------------

    ########################################################
    # 5. ROS2 -> 시뮬레이터 명령 callback들
    ########################################################
    def on_route_risk_report(self, msg: String) -> None:
        """Store risk_analysis LLM output for the browser dashboard."""
        timestamp_wall = now_wall()
        try:
            report = json.loads(msg.data)
            if not isinstance(report, dict):
                raise ValueError("route risk report is not a JSON object")
        except Exception as exc:
            self.get_logger().error(f"Invalid /tank/risk/route_report: {exc}")
            return

        result = report.get("result") if isinstance(report.get("result"), dict) else {}
        result = _apply_forced_route_policy_to_result(result)
        report = deepcopy(report)
        report["result"] = deepcopy(result)
        decision = {
            "route": "/tank/risk/route_report",
            "timestamp_wall": timestamp_wall,
            "model": report.get("model"),
            "ok": report.get("ok"),
            "parsed_ok": report.get("parsed_ok"),
            "validated_ok": report.get("validated_ok"),
            "selected_route": result.get("selected_route"),
            "risk_level": deepcopy(result.get("risk_level")),
            "confidence": result.get("confidence"),
            "summary": result.get("summary"),
            "decision_reason": result.get("decision_reason"),
            "key_risks": deepcopy(result.get("key_risks")),
            "recommended_behavior": deepcopy(result.get("recommended_behavior")),
            "used_evidence": deepcopy(result.get("used_evidence")),
        }
        ai_entry = {
            "timestamp_wall": timestamp_wall,
            "source": "/tank/risk/route_report",
            "model": report.get("model"),
            "selected_route": result.get("selected_route"),
            "confidence": result.get("confidence"),
            "summary": result.get("summary"),
            "decision_reason": result.get("decision_reason"),
            "validated_ok": report.get("validated_ok"),
        }

        with self._lock:
            self._count("/tank/risk/route_report")
            self._latest["route_risk_report"] = deepcopy(report)
            self._latest["decision"] = deepcopy(decision)
            log_entries = self._latest.get("llm_log")
            if not isinstance(log_entries, list):
                log_entries = []
            log_entries.append(ai_entry)
            log_entries = log_entries[-20:]
            self._latest["llm_log"] = deepcopy(log_entries)
            self._latest["ai_log"] = deepcopy(log_entries)

        selected_route = result.get("selected_route") or "?"
        summary = result.get("summary") or "LLM route risk report received"
        self.get_logger().info(f"route risk report updated: selected={selected_route}, summary={summary}")

    # 지속 제어 명령 topic(/tank/control/command)을 수신했을 때 실행되는 callback이다.
    def on_control_command(self, msg: String) -> None:
        # JSON 파싱/명령 검증 중 오류가 날 수 있으므로 예외 처리를 시작한다.
        try:
            # ROS2 String.data에 담긴 JSON 문자열을 Python dict로 변환한다.
            command = json.loads(msg.data)
            # 공식 /get_action 응답 형식(moveWS/moveAD/turretQE/turretRF/fire)에 맞는지 검사한다.
            validate_action_command(command)
        # 파싱 또는 검증 실패 시 bridge를 죽이지 않고 로그만 남긴다.
        except Exception as exc:
            # 잘못된 명령이 들어왔음을 ROS2 error 로그로 출력한다.
            self.get_logger().error(f"Invalid /tank/control/command: {exc}")
            # 잘못된 입력은 무시하고 callback 처리를 종료한다.
            return
        # 공유 상태를 읽거나 쓸 때 Lock을 잡아 thread race condition을 방지한다.
        with self._lock:
            # 검증된 최신 지속 제어 명령을 저장한다.
            self._latest_command = command
            # 최신 지속 제어 명령이 들어온 wall-clock 시간을 저장한다.
            self._latest_command_stamp = now_wall()
        # 정상 수신된 제어 명령을 2초에 한 번 INFO로 남긴다.
        # 이 로그가 보이면 /tank/control/command -> bridge callback 경로가 정상이다.
        log_now = now_wall()
        if log_now - self._last_command_log_wall >= 2.0:
            self._last_command_log_wall = log_now
            self.get_logger().info(f"control command updated: {command}")

    # 다음 /get_action 응답 한 번만 덮어쓸 override 명령을 수신하는 callback이다.
    def on_one_shot_override(self, msg: String) -> None:
        # JSON 파싱/명령 검증 중 오류가 날 수 있으므로 예외 처리를 시작한다.
        try:
            # ROS2 String.data에 담긴 JSON 문자열을 Python dict로 변환한다.
            command = json.loads(msg.data)
            # 공식 /get_action 응답 형식(moveWS/moveAD/turretQE/turretRF/fire)에 맞는지 검사한다.
            validate_action_command(command)
        # 파싱 또는 검증 실패 시 bridge를 죽이지 않고 로그만 남긴다.
        except Exception as exc:
            # 잘못된 명령이 들어왔음을 ROS2 error 로그로 출력한다.
            self.get_logger().error(f"Invalid one-shot /get_action override: {exc}")
            # 잘못된 입력은 무시하고 callback 처리를 종료한다.
            return
        # 공유 상태를 읽거나 쓸 때 Lock을 잡아 thread race condition을 방지한다.
        with self._lock:
            # 다음 /get_action 응답 1회에만 사용할 override 명령을 저장한다.
            self._one_shot_override = command
        # 실행 터미널에 bridge 초기화 상태를 ROS2 logger로 출력한다.
        self.get_logger().info("one-shot /get_action override received")

    # 퓨전된 객체 데이터를 수신하여 로그로 저장한다.
    def on_fused_objects(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
            append_jsonl("fused.jsonl", {
                "timestamp_wall": now_wall(),
                "route": "/tank/perception/fused_objects",
                "data": payload
            })
        except Exception as exc:
            self.get_logger().error(f"Failed to log fused objects: {exc}")

    def on_risk_report(self, msg: String) -> None:
        """route_risk_node의 LLM 전술 결정을 받아 latest["decision"]에 저장(MFD aiLog 노출)."""
        try:
            data = json.loads(msg.data)
        except Exception as exc:
            self.get_logger().warn(f"risk report parse 실패: {exc}")
            return
        res = data.get("result") if isinstance(data, dict) else None
        res = res if isinstance(res, dict) else {}
        rl = res.get("risk_level") if isinstance(res.get("risk_level"), dict) else {}
        summary = (
            f"추천 route_{res.get('selected_route', '?')} · "
            f"risk A:{rl.get('A', '?')}/B:{rl.get('B', '?')} · {res.get('summary', '')}"
        )
        self.update_latest("decision", {
            "summary": summary,
            "result": res,
            "timestamp_wall": now_wall(),
        })

    # /get_action 응답에 실제로 사용할 명령과 명령 출처 문자열을 선택한다.
    def select_action_command(self) -> Tuple[Dict[str, Any], str]:
        # monitor 모드에서는 자율제어를 하지 않으므로 항상 중립 명령을 반환한다.
        if TANK_MODE == "monitor":
            # 중립 명령과 출처 문자열을 함께 반환한다.
            return neutral_command(), "monitor_mode_neutral"

        # 현재 시간을 저장해 최신 명령의 age를 계산한다.
        now = now_wall()
        # 공유 상태를 읽거나 쓸 때 Lock을 잡아 thread race condition을 방지한다.
        with self._lock:
            # one-shot override가 있으면 지속 명령보다 우선 사용한다.
            if self._one_shot_override is not None:
                # override 명령을 임시 변수에 복사한다.
                cmd = self._one_shot_override
                # 다음 /get_action 응답 1회에만 사용할 override 명령을 저장한다.
                self._one_shot_override = None
                # override 명령과 출처 문자열을 반환한다.
                return cmd.copy(), "one_shot_override"

            # 지속 제어 명령이 있고 수신 시각도 기록되어 있는지 확인한다.
            if self._latest_command is not None and self._latest_command_stamp is not None:
                # 최신 명령이 들어온 지 얼마나 지났는지 계산한다.
                age = now - self._latest_command_stamp
                # 명령이 TTL 안에 있으면 아직 안전하게 사용할 수 있다고 본다.
                if age <= COMMAND_TTL_SEC:
                    # 최신 지속 제어 명령과 명령 age 정보를 출처 문자열로 반환한다.
                    return self._latest_command.copy(), f"ros2_control_command_age_{age:.3f}s"
                self.get_logger().warn(
                    f"latest control command is stale: age={age:.3f}s > ttl={COMMAND_TTL_SEC:.3f}s"
                )
            else:
                self.get_logger().warn("no control command has been received by bridge yet")

        # auto 모드에서 신선한 명령이 없으면 안전 fallback 명령을 반환한다.
        return fallback_command(), "auto_fallback_no_fresh_command"

    # --------------------------------------------------------
    # 파싱 및 publish handler들
    # --------------------------------------------------------

    ########################################################
    # 6. Flask endpoint 데이터 -> ROS2 topic handler들
    ########################################################
    # Flask route가 handle_init를 호출하면, 여기서 데이터를 ROS2 topic으로 변환/publish한다.
    def handle_init(self, config: Dict[str, Any]) -> None:
        # 이 endpoint의 ROS2 publish 및 latest 저장용 payload를 만든다.
        payload = {"route": "/init", "timestamp_wall": now_wall(), "mode": TANK_MODE, "config": config}
        # 공유 상태를 읽거나 쓸 때 Lock을 잡아 thread race condition을 방지한다.
        with self._lock:
            # 현재 endpoint의 수신 횟수를 증가시킨다.
            self._count("/init")
            # latest state에 저장
            self._latest["init_config"] = payload
        # JSON payload를 String topic으로 publish한다.
        self.publish_json(self.pub_init_config, payload)

    # Flask route가 handle_start를 호출하면, 여기서 데이터를 ROS2 topic으로 변환/publish한다.
    def handle_start(self) -> None:
        # 이 endpoint의 ROS2 publish 및 latest 저장용 payload를 만든다.
        payload = {"route": "/start", "timestamp_wall": now_wall()}
        # 공유 상태를 읽거나 쓸 때 Lock을 잡아 thread race condition을 방지한다.
        with self._lock:
            # 현재 endpoint의 수신 횟수를 증가시킨다.
            self._count("/start")
            # latest state에 저장
            self._latest["start_event"] = payload
        self.pub_start_event.publish(Empty())

    # Flask route가 handle_info를 호출하면, 여기서 데이터를 ROS2 topic으로 변환/publish한다.
    def handle_info(self, data: Dict[str, Any]) -> Dict[str, Any]:
        # 이 이벤트가 bridge에 들어온 wall-clock timestamp를 기록한다.
        ts = now_wall()
        # /info 원본에서 핵심 필드만 추린 compact JSON을 만든다.
        compact = compact_info(data)

        # playerPos가 없을 수도 있으므로 raw/map pose 변수를 None으로 초기화한다.
        player_raw = player_map = None
        # enemyPos가 없을 수도 있으므로 raw/map pose 변수를 None으로 초기화한다.
        enemy_raw = enemy_map = None
        # /info에 playerPos dict가 있으면 Unity raw 좌표와 ROS map 좌표를 둘 다 만든다.
        if isinstance(data.get("playerPos"), dict):
            # 아군 전차 위치를 raw pose와 map pose로 변환한다.
            player_raw, player_map = raw_and_map_pose(data.get("playerPos"), "/info/playerPos")
        # /info에 enemyPos dict가 있으면 Unity raw 좌표와 ROS map 좌표를 둘 다 만든다.
        if isinstance(data.get("enemyPos"), dict):
            # 적 전차 위치를 raw pose와 map pose로 변환한다.
            enemy_raw, enemy_map = raw_and_map_pose(data.get("enemyPos"), "/info/enemyPos")
        # 아군 전차 상태를 알고리즘이 쓰기 쉬운 JSON dict로 재구성한다.
        player_state = {
            "timestamp_wall": ts,
            "source": "/info",
            "pose_raw": player_raw,
            "pose_map": player_map,
            "speed": data.get("playerSpeed"),
            "health": data.get("playerHealth"),
            "turret": {"x": data.get("playerTurretX"), "y": data.get("playerTurretY")},
            "body": {"x": data.get("playerBodyX"), "y": data.get("playerBodyY"), "z": data.get("playerBodyZ")},
            "sim_time": data.get("time"),
            "distance": data.get("distance"),
        }
        # 적 전차 상태를 알고리즘이 쓰기 쉬운 JSON dict로 재구성한다.
        enemy_state = {
            "timestamp_wall": ts,
            "source": "/info",
            "pose_raw": enemy_raw,
            "pose_map": enemy_map,
            "speed": data.get("enemySpeed"),
            "health": data.get("enemyHealth"),
            "turret": {"x": data.get("enemyTurretX"), "y": data.get("enemyTurretY")},
            "body": {"x": data.get("enemyBodyX"), "y": data.get("enemyBodyY"), "z": data.get("enemyBodyZ")},
            "sim_time": data.get("time"),
            "distance": data.get("distance"),
        }

        # 원본 endpoint 데이터를 그대로 보존하는 payload를 만든다. (불필요한 deepcopy 제거)
        raw_payload = {"route": "/info", "timestamp_wall": ts, "data": data}
        # /info compact topic과 Flask 반환 로그에 사용할 payload를 만든다.
        compact_payload = {"route": "/info", "timestamp_wall": ts, "data": compact}
        # /info에서 자주 쓰는 경량 상태를 별도 stable topic으로 만든다.
        sim_status = {
            "route": "/info",
            "timestamp_wall": ts,
            "sim_time": data.get("time"),
            "distance": data.get("distance"),
            "player_speed": data.get("playerSpeed"),
            "player_health": data.get("playerHealth"),
            "enemy_speed": data.get("enemySpeed"),
            "enemy_health": data.get("enemyHealth"),
            "terrain_size_unity": {"x": 300.0, "z": 300.0},
            "turret_camera_fov_deg": {"vertical": 28.0, "horizontal": 47.81061},
        }
        # 차체 yaw를 map-frame heading vector로 변환한다.
        player_heading = self.heading_vector_from_degree(data.get("playerBodyX"))
        enemy_heading = self.heading_vector_from_degree(data.get("enemyBodyX"))

        # 공유 상태를 읽거나 쓸 때 Lock을 잡아 thread race condition을 방지한다.
        with self._lock:
            # 현재 endpoint의 수신 횟수를 증가시킨다.
            self._count("/info")
            self._latest["info_raw"] = raw_payload
            self._latest["info_compact"] = compact_payload
            self._latest["player_state"] = player_state
            self._latest["enemy_state"] = enemy_state
            # player map 좌표가 생성된 경우에만 latest pose를 갱신한다.
            if player_map:
                self._latest["player_pose_map"] = player_map
            # enemy map 좌표가 생성된 경우에만 latest enemy pose를 갱신한다.
            if enemy_map:
                self._latest["enemy_pose_map"] = enemy_map
            self._latest["sim_status"] = sim_status
            self._latest["player_heading"] = player_heading
            self._latest["enemy_heading"] = enemy_heading

        # JSON payload를 String topic으로 publish한다.
        self.publish_json(self.pub_info_raw, raw_payload)
        # JSON payload를 String topic으로 publish한다.
        self.publish_json(self.pub_info_compact, compact_payload)
        # JSON payload를 String topic으로 publish한다.
        self.publish_json(self.pub_info_player_state, player_state)
        # JSON payload를 String topic으로 publish한다.
        self.publish_json(self.pub_info_enemy_state, enemy_state)
        # JSON payload를 String topic으로 publish한다.
        self.publish_json(self.pub_player_state, player_state)
        # JSON payload를 String topic으로 publish한다.
        self.publish_json(self.pub_enemy_state, enemy_state)
        # 주행 판단용 경량 상태 topic을 publish한다.
        self.publish_json(self.pub_sim_status, sim_status)
        # 차체 heading vector를 publish한다.
        self.publish_vector3(self.pub_player_heading, player_heading, MAP_FRAME)
        self.publish_vector3(self.pub_enemy_heading, enemy_heading, MAP_FRAME)

        # 아군 위치 raw/map 좌표가 모두 있을 때 관련 pose topic들을 publish한다.
        if player_raw and player_map:
            # pose payload를 PoseStamped topic으로 publish한다.
            self.publish_pose(self.pub_info_player_pose_raw, player_raw)
            # pose payload를 PoseStamped topic으로 publish한다.
            self.publish_pose(self.pub_info_player_pose_map, player_map)
            # 같은 pose를 안정 topic과 호환 alias topic에 반복 publish한다.
            for pub in (self.pub_player_pose, self.pub_latest_pose_alias, self.pub_pose_alias):
                # pose payload를 PoseStamped topic으로 publish한다.
                self.publish_pose(pub, player_map)
        # 적 위치 raw/map 좌표가 모두 있을 때 관련 pose topic들을 publish한다.
        if enemy_raw and enemy_map:
            # pose payload를 PoseStamped topic으로 publish한다.
            self.publish_pose(self.pub_info_enemy_pose_raw, enemy_raw)
            # pose payload를 PoseStamped topic으로 publish한다.
            self.publish_pose(self.pub_info_enemy_pose_map, enemy_map)
            # pose payload를 PoseStamped topic으로 publish한다.
            self.publish_pose(self.pub_enemy_pose, enemy_map)
        # SAVE_JSONL 옵션이 켜져 있으면 이 endpoint 데이터를 JSONL 파일에 저장한다.
        append_jsonl("info.jsonl", {
            "timestamp_wall": ts,
            "route": "/info",
            "data": data if SAVE_FULL_INFO else compact,
            "player_state": player_state,
            "enemy_state": enemy_state,
        })
        # Flask route가 터미널 출력 등에 사용할 compact payload를 반환한다.
        return compact_payload

    # Flask route가 handle_get_action를 호출하면, 여기서 데이터를 ROS2 topic으로 변환/publish한다.
    def handle_get_action(self, data: Dict[str, Any]) -> Dict[str, Any]:
        # 이 이벤트가 bridge에 들어온 wall-clock timestamp를 기록한다.
        ts = now_wall()
        # /get_action 요청 body에서 현재 전차 position dict를 꺼낸다.
        position = data.get("position", {}) if isinstance(data, dict) else {}
        # /get_action 요청 body에서 현재 포탑 turret dict를 꺼낸다.
        turret = data.get("turret", {}) if isinstance(data.get("turret"), dict) else {}

        # 현재 위치를 Unity raw 좌표와 ROS map 좌표로 동시에 변환한다.
        pose_raw, pose_map = raw_and_map_pose(position, "/get_action/position")
        # 포탑 방향/각도 정보를 Vector3Stamped로 보낼 수 있는 dict로 정리한다.
        turret_vec = {
            "x": to_float(turret.get("x")),
            "y": to_float(turret.get("y")),
            "z": to_float(turret.get("z")),
        }

        # monitor/auto/override/TTL 규칙에 따라 시뮬레이터에 보낼 명령을 고른다.
        command, source = self.select_action_command()
        # 실제 /get_action 응답으로 반환한 명령과 출처를 기록하는 payload다.
        response_payload = {
            "route": "/get_action",
            "timestamp_wall": ts,
            "mode": TANK_MODE,
            "source": source,
            "command": command,
        }
        # 원본 endpoint 데이터를 그대로 보존하는 payload를 만든다.
        raw_payload = {
            "route": "/get_action",
            "timestamp_wall": ts,
            "request": data,
            "pose_raw": pose_raw,
            "pose_map": pose_map,
            "turret": turret_vec,
        }

        # 공유 상태를 읽거나 쓸 때 Lock을 잡아 thread race condition을 방지한다.
        with self._lock:
            # 현재 endpoint의 수신 횟수를 증가시킨다.
            self._count("/get_action")
            self._latest["get_action_raw"] = raw_payload
            self._latest["get_action_pose_map"] = pose_map
            self._latest["get_action_response"] = response_payload
            self._latest["player_pose_map"] = pose_map

        # JSON payload를 String topic으로 publish한다.
        self.publish_json(self.pub_get_action_raw, raw_payload)
        # pose payload를 PoseStamped topic으로 publish한다.
        self.publish_pose(self.pub_get_action_pose_raw, pose_raw)
        # pose payload를 PoseStamped topic으로 publish한다.
        self.publish_pose(self.pub_get_action_pose_map, pose_map)
        # vector payload를 Vector3Stamped topic으로 publish한다.
        self.publish_vector3(self.pub_get_action_turret, turret_vec, UNITY_FRAME)
        # JSON payload를 String topic으로 publish한다.
        self.publish_json(self.pub_get_action_response, response_payload)

        # JSON payload를 String topic으로 publish한다.
        self.publish_json(self.pub_action_raw_alias, raw_payload)
        # JSON payload를 String topic으로 publish한다.
        self.publish_json(self.pub_sent_action_alias, response_payload)
        # 같은 pose를 안정 topic과 호환 alias topic에 반복 publish한다.
        for pub in (self.pub_player_pose, self.pub_latest_pose_alias, self.pub_pose_alias):
            # pose payload를 PoseStamped topic으로 publish한다.
            self.publish_pose(pub, pose_map)

        # SAVE_JSONL 옵션이 켜져 있으면 이 endpoint 데이터를 JSONL 파일에 저장한다.
        append_jsonl("get_action.jsonl", {"request": raw_payload, "response": response_payload})
        return command

    # Flask route가 /detect image bytes를 수신하면, 여기서 ROS2 CompressedImage topic으로 중계한다.
    def handle_detect_image(self, image_bytes: bytes, metadata: Optional[Dict[str, Any]] = None) -> None:
        msg = CompressedImage()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "tank_camera"
        msg.format = "jpeg"
        msg.data = image_bytes

        with self._lock:
            self._latest["camera_image"] = {
                "route": "/detect",
                "timestamp_wall": now_wall(),
                "bytes": len(image_bytes),
                "metadata": metadata if isinstance(metadata, dict) else {},
            }

        self.pub_camera_image_compressed.publish(msg)
        self.pub_api_detect_image_compressed.publish(msg)

    # Flask route가 handle_detect_result를 호출하면, 여기서 데이터를 ROS2 topic으로 변환/publish한다.
    def handle_detect_result(self, detections: Any, metadata: Optional[Dict[str, Any]] = None) -> None:
        # 이 endpoint의 ROS2 publish 및 latest 저장용 payload를 만든다.
        payload = {"route": "/detect", "timestamp_wall": now_wall(), "count": len(detections) if isinstance(detections, list) else None, "detections": detections}
        if isinstance(metadata, dict):
            payload.update(metadata)
        # 공유 상태를 읽거나 쓸 때 Lock을 잡아 thread race condition을 방지한다.
        with self._lock:
            # 현재 endpoint의 수신 횟수를 증가시킨다.
            self._count("/detect")
            # latest state에 저장
            self._latest["detect_result"] = payload
        # JSON payload를 String topic으로 publish한다.
        self.publish_json(self.pub_detect_result, payload)
        # 시나리오 매니저용 high-level detection event alias를 publish한다.
        self.publish_json(self.pub_event_detection, payload)
        # 비전/센서융합 노드가 직접 subscribe하기 좋은 perception alias를 publish한다.
        self.publish_json(self.pub_perception_detections, payload)

    # Flask route가 handle_stereo_status를 호출하면, 여기서 데이터를 ROS2 topic으로 변환/publish한다.
    def handle_stereo_status(self, status: Dict[str, Any]) -> None:
        # 이 endpoint의 ROS2 publish 및 latest 저장용 payload를 만든다.
        payload = {"route": "/stereo_image", "timestamp_wall": now_wall(), **status}
        # 공유 상태를 읽거나 쓸 때 Lock을 잡아 thread race condition을 방지한다.
        with self._lock:
            # 현재 endpoint의 수신 횟수를 증가시킨다.
            self._count("/stereo_image")
            # latest state에 저장
            self._latest["stereo_status"] = payload
        # JSON payload를 String topic으로 publish한다.
        self.publish_json(self.pub_stereo_status, payload)

    # Flask route가 handle_bullet를 호출하면, 여기서 데이터를 ROS2 topic으로 변환/publish한다.
    def handle_bullet(self, data: Dict[str, Any]) -> None:
        # 이 이벤트가 bridge에 들어온 wall-clock timestamp를 기록한다.
        ts = now_wall()
        # 탄착 위치를 raw/map 좌표로 변환한다.
        impact_raw, impact_map = raw_and_map_pose(data, "/update_bullet")
        # 포탄이 맞은 대상(hit 필드)을 추출한다.
        target = data.get("hit") if isinstance(data, dict) else None
        # 이 endpoint의 ROS2 publish 및 latest 저장용 payload를 만든다.
        payload = {"route": "/update_bullet", "timestamp_wall": ts, "data": data, "impact_raw": impact_raw, "impact_map": impact_map, "target": target}
        # 공유 상태를 읽거나 쓸 때 Lock을 잡아 thread race condition을 방지한다.
        with self._lock:
            # 현재 endpoint의 수신 횟수를 증가시킨다.
            self._count("/update_bullet")
            # latest state에 저장
            self._latest["bullet"] = payload
        # JSON payload를 String topic으로 publish한다.
        self.publish_json(self.pub_bullet_raw, payload)
        # point payload를 PointStamped topic으로 publish한다.
        self.publish_point(self.pub_bullet_impact_raw, impact_raw)
        # point payload를 PointStamped topic으로 publish한다.
        self.publish_point(self.pub_bullet_impact_map, impact_map)
        # JSON payload를 String topic으로 publish한다.
        self.publish_json(self.pub_bullet_target, {"timestamp_wall": ts, "target": target})
        # point payload를 PointStamped topic으로 publish한다.
        self.publish_point(self.pub_bullet_alias, impact_map)
        # 시나리오 매니저용 high-level bullet event alias를 publish한다.
        self.publish_json(self.pub_event_bullet, payload)
        # SAVE_JSONL 옵션이 켜져 있으면 이 endpoint 데이터를 JSONL 파일에 저장한다.
        append_jsonl("bullet.jsonl", payload)

    # Flask route가 handle_destination를 호출하면, 여기서 데이터를 ROS2 topic으로 변환/publish한다.
    def handle_destination(self, x: float, y: float, z: float) -> Dict[str, Any]:
        # 이 이벤트가 bridge에 들어온 wall-clock timestamp를 기록한다.
        ts = now_wall()
        # 현재 위치를 Unity raw 좌표와 ROS map 좌표로 동시에 변환한다.
        pose_raw, pose_map = raw_and_map_pose({"x": x, "y": y, "z": z}, "/set_destination")
        # 이 endpoint의 ROS2 publish 및 latest 저장용 payload를 만든다.
        payload = {"route": "/set_destination", "timestamp_wall": ts, "pose_raw": pose_raw, "pose_map": pose_map}
        # 공유 상태를 읽거나 쓸 때 Lock을 잡아 thread race condition을 방지한다.
        with self._lock:
            # 현재 endpoint의 수신 횟수를 증가시킨다.
            self._count("/set_destination")
            # latest state에 저장
            self._latest["destination"] = payload
        # JSON payload를 String topic으로 publish한다.
        self.publish_json(self.pub_destination_raw, payload)
        # pose payload를 PoseStamped topic으로 publish한다.
        self.publish_pose(self.pub_destination_pose_raw, pose_raw)
        # pose payload를 PoseStamped topic으로 publish한다.
        self.publish_pose(self.pub_destination_pose_map, pose_map)
        # pose payload를 PoseStamped topic으로 publish한다.
        self.publish_pose(self.pub_destination, pose_map)
        # SAVE_JSONL 옵션이 켜져 있으면 이 endpoint 데이터를 JSONL 파일에 저장한다.
        append_jsonl("destination.jsonl", payload)
        return pose_raw

    # Flask route가 handle_obstacles를 호출하면, 여기서 데이터를 ROS2 topic으로 변환/publish한다.
    def handle_obstacles(self, data: Any) -> None:
        # 이 이벤트가 bridge에 들어온 wall-clock timestamp를 기록한다.
        ts = now_wall()
        # 이 endpoint의 ROS2 publish 및 latest 저장용 payload를 만든다.
        payload = {"route": "/update_obstacle", "timestamp_wall": ts, "data": data}
        # 공유 상태를 읽거나 쓸 때 Lock을 잡아 thread race condition을 방지한다.
        with self._lock:
            # 현재 endpoint의 수신 횟수를 증가시킨다.
            self._count("/update_obstacle")
            self._latest["obstacles"] = payload
        # JSON payload를 String topic으로 publish한다.
        self.publish_json(self.pub_obstacle_raw, payload)
        # JSON payload를 String topic으로 publish한다.
        self.publish_json(self.pub_obstacle_list, payload)
        # JSON payload를 String topic으로 publish한다.
        self.publish_json(self.pub_obstacles, payload)
        # JSON payload를 String topic으로 publish한다.
        self.publish_json(self.pub_obstacles_alias, payload)
        # SAVE_JSONL 옵션이 켜져 있으면 이 endpoint 데이터를 JSONL 파일에 저장한다.
        append_jsonl("obstacles.jsonl", payload)

    # Flask route가 handle_collision를 호출하면, 여기서 데이터를 ROS2 topic으로 변환/publish한다.
    def handle_collision(self, data: Dict[str, Any]) -> None:
        # 이 이벤트가 bridge에 들어온 wall-clock timestamp를 기록한다.
        ts = now_wall()
        # /get_action 요청 body에서 현재 전차 position dict를 꺼낸다.
        position = data.get("position", {}) if isinstance(data, dict) else {}
        # 충돌 위치를 raw/map point로 변환한다.
        point_raw, point_map = raw_and_map_pose(position, "/collision/position")
        # 이 endpoint의 ROS2 publish 및 latest 저장용 payload를 만든다.
        payload = {"route": "/collision", "timestamp_wall": ts, "data": data, "point_raw": point_raw, "point_map": point_map, "objectName": data.get("objectName")}
        # 공유 상태를 읽거나 쓸 때 Lock을 잡아 thread race condition을 방지한다.
        with self._lock:
            # 현재 endpoint의 수신 횟수를 증가시킨다.
            self._count("/collision")
            self._latest["collision"] = payload
        # JSON payload를 String topic으로 publish한다.
        self.publish_json(self.pub_collision_raw, payload)
        # point payload를 PointStamped topic으로 publish한다.
        self.publish_point(self.pub_collision_point_raw, point_raw)
        # point payload를 PointStamped topic으로 publish한다.
        self.publish_point(self.pub_collision_point_map, point_map)
        # 시나리오 매니저용 high-level collision event alias를 publish한다.
        self.publish_json(self.pub_event_collision, payload)
        # SAVE_JSONL 옵션이 켜져 있으면 이 endpoint 데이터를 JSONL 파일에 저장한다.
        append_jsonl("collision.jsonl", payload)


    ########################################################
    # 7. 주기적 최신 상태 publisher
    ########################################################
    # timer callback: 최신 상태 전체를 10Hz로 주기 publish한다.
    def publish_latest_state(self) -> None:
        # 공유 상태를 읽거나 쓸 때 Lock을 잡아 thread race condition을 방지한다.
        with self._lock:
            # /tank/state/latest에 담을 통합 상태 payload를 만든다.
            state = {
                "timestamp_wall": now_wall(),
                "mode": TANK_MODE,
                "coordinate_policy": {
                    "raw": "Unity API x,y,z 그대로",
                    "map": "x=raw.x, y=raw.z, z=raw.y",
                },
                "route_counts": self._route_counts.copy(),
                "latest": self._latest.copy(),
            }
            # 호환 alias topic에 주기적으로 다시 publish할 최신 player pose를 복사한다.
            latest_pose = self._latest.get("player_pose_map")
            # 최신 player state를 복사해 Lock 밖에서 publish한다.
            latest_player_state = self._latest.get("player_state")
            # 최신 enemy state를 복사해 Lock 밖에서 publish한다.
            latest_enemy_state = self._latest.get("enemy_state")
            # 최신 obstacle payload를 복사해 Lock 밖에서 publish한다.
            latest_obstacles = self._latest.get("obstacles")

        # JSON payload를 String topic으로 publish한다.
        self.publish_json(self.pub_state_latest, state)
        # JSON payload를 String topic으로 publish한다.
        self.publish_json(self.pub_latest_state_alias, state)
        # 최신 pose가 있을 때만 pose alias topic을 publish한다.
        if latest_pose:
            # pose payload를 PoseStamped topic으로 publish한다.
            self.publish_pose(self.pub_latest_pose_alias, latest_pose)
            # pose payload를 PoseStamped topic으로 publish한다.
            self.publish_pose(self.pub_pose_alias, latest_pose)
        # 최신 player state가 있을 때만 player state topic을 publish한다.
        if latest_player_state:
            # JSON payload를 String topic으로 publish한다.
            self.publish_json(self.pub_player_state, latest_player_state)
        # 최신 enemy state가 있을 때만 enemy state topic을 publish한다.
        if latest_enemy_state:
            # JSON payload를 String topic으로 publish한다.
            self.publish_json(self.pub_enemy_state, latest_enemy_state)
        # 최신 obstacle 정보가 있을 때만 obstacle alias topic을 publish한다.
        if latest_obstacles:
            # JSON payload를 String topic으로 publish한다.
            self.publish_json(self.pub_obstacles_alias, latest_obstacles)
