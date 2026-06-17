# -*- coding: utf-8 -*-
"""
############################################################
# rviz_visualizer_node.py
############################################################

패키지:
    rviz_visualization

역할:
    Tank Challenge 시뮬레이터에서 ROS2 topic으로 들어온 데이터를
    RViz2에서 보기 좋은 MarkerArray 형태로 변환한다.

핵심 입력:
    - /tank/player/pose
    - /tank/enemy/pose
    - /tank/goal/pose
    - /tank/map/obstacles
    - /tank/sensor/lidar/points

핵심 출력:
    - /tank/rviz/object_markers
    - /tank/rviz/terrain_markers
    - /tank/rviz/obstacle_markers
    - /tank/rviz/lidar_markers
    - /tank/rviz/risk_markers

중요 설계 원칙:
    이 node는 "판단 알고리즘"을 수행하지 않는다.

    즉, 아래 작업은 하지 않는다.
        - YOLO 객체 판단
        - LiDAR clustering
        - 장애물 분류
        - 위험도 계산
        - 지형 복잡도 계산
        - A* 경로계획
        - Potential Field 계산
        - 전차 제어 명령 생성

    이 node는 이미 들어온 정보를 RViz2에서 보기 좋게 바꾸는
    "시각화 어댑터" 역할만 담당한다.

전체 데이터 흐름:
    Tank Challenge Simulator
        ↓ HTTP API
    ros_bridge
        ↓ ROS2 topic
    rviz_visualizer_node
        ↓ MarkerArray
    RViz2

향후 확장 방향:
    perception/planning node에서 아래와 같은 정보를 publish하면,
    이 node는 해당 값을 색상/크기/투명도 marker로 표시할 수 있다.

    예:
        {
            "x": 120.0,
            "y": 80.0,
            "type": "rock",
            "risk": 0.7,
            "complexity": 0.5,
            "size": 4.0
        }
"""


############################################################
# 1. Python 기본 모듈 import
############################################################
import math

# ros_bridge에서 들어오는 장애물/LiDAR 정보는 std_msgs/String 안에
# JSON 문자열로 들어오므로 json 파싱이 필요하다.
import json

import numpy as np

# 타입 힌트용 import.
#
# Any:
#   JSON 내부 값처럼 타입이 고정되지 않은 값을 표현할 때 사용.
#
# Dict:
#   dict[str, Any] 형태의 JSON object를 표현할 때 사용.
#
# List:
#   장애물 목록, LiDAR point 목록처럼 list 자료구조를 표현할 때 사용.
#
# Optional:
#   아직 값이 들어오지 않았을 수 있는 변수에 사용.
#   예: Optional[PoseStamped] = None
from typing import Any, Dict, List, Optional, Tuple


############################################################
# 2. ROS2 기본 모듈 import
############################################################

# rclpy:
#   ROS2 Python client library.
#   Python으로 ROS2 node를 만들 때 사용하는 핵심 모듈.
import rclpy

# PoseStamped:
#   위치 + 자세 + frame_id + timestamp를 함께 담는 메시지.
#   ros_bridge에서 아군/적/목표 위치를 이 타입으로 publish한다.
from geometry_msgs.msg import PointStamped, PoseStamped, Vector3Stamped
from nav_msgs.msg import Path as NavPath
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2

# Node:
#   rclpy 기반 ROS2 node의 기본 클래스.
#   이 클래스를 상속해서 subscriber, publisher, timer를 만든다.
from rclpy.node import Node

# String:
#   JSON 문자열을 topic으로 주고받을 때 사용한다.
#   장애물 정보와 LiDAR point 정보는 현재 std_msgs/String(JSON) 형태로 받는다.
from std_msgs.msg import String

# MarkerArray:
#   RViz2에 여러 marker를 한 번에 표시할 때 사용하는 메시지.
#   예: 전차 marker + 목표 marker + 장애물 marker 목록
from visualization_msgs.msg import MarkerArray


############################################################
# 3. 프로젝트 설정값 import
############################################################

# config.py에는 topic 이름, frame 이름, marker 크기, 시각화 주기만 둔다.
# 판단/계산 로직은 config.py에 넣지 않는다.
from .config import (
    ############################################################
    # Terrain / simulator map settings
    ############################################################
    TERRAIN_MIN_X,
    TERRAIN_MAX_X,
    TERRAIN_MIN_Y,
    TERRAIN_MAX_Y,
    TERRAIN_GRID_STEP,
    TERRAIN_MARKER_Z,
    SHOW_TERRAIN_BOUNDARY,
    SHOW_TERRAIN_GRID_MARKER,
    SHOW_TERRAIN_REFERENCE_POINTS,
    TOPIC_RVIZ_TERRAIN_MARKERS,

    SIM_PLAYER_START_RAW_X,
    SIM_PLAYER_START_RAW_Y,
    SIM_PLAYER_START_RAW_Z,
    SIM_LEFT_FRONT_CORNER_RAW_X,
    SIM_LEFT_FRONT_CORNER_RAW_Y,
    SIM_LEFT_FRONT_CORNER_RAW_Z,
    TERRAIN_CORNER_MARKER_SIZE,
    SPAWN_MARKER_SIZE,

    ############################################################
    # Tank / object marker scale
    ############################################################
    ENEMY_TANK_SCALE_X,
    ENEMY_TANK_SCALE_Y,
    ENEMY_TANK_SCALE_Z,
    GOAL_MARKER_RADIUS,
    LIDAR_POINT_SIZE,
    LIDAR_VISUALIZATION_SAMPLE_STEP,
    LIDAR_VISUALIZE_DETECTED_ONLY,
    LIDAR_POSITION_IS_UNITY_RAW,
    LIDAR_DETECTED_ALPHA,
    LIDAR_FREE_SPACE_ALPHA,
    SIM_LIDAR_CHANNELS,
    SIM_LIDAR_MAX_DISTANCE,
    MAP_FRAME,
    OBSTACLE_DEFAULT_SCALE,
    PLAYER_TANK_SCALE_X,
    PLAYER_TANK_SCALE_Y,
    PLAYER_TANK_SCALE_Z,
    RISK_MARKER_Z,

    # 전차 heading을 위한 import
    TOPIC_PLAYER_STATE,
    TOPIC_ENEMY_STATE,
    SHOW_TANK_HEADING_ARROW,
    PLAYER_HEADING_ARROW_LENGTH,
    ENEMY_HEADING_ARROW_LENGTH,
    HEADING_ARROW_Z_OFFSET,
    HEADING_DEGREE_KEY_PLAYER,
    HEADING_DEGREE_KEY_ENEMY,

    ############################################################
    # Input topics
    ############################################################
    TOPIC_ENEMY_POSE,
    TOPIC_GOAL_POSE,
    TOPIC_LIDAR_POINTS,
    TOPIC_OBSTACLES,
    TOPIC_PLAYER_POSE,

    ############################################################
    # RViz output topics
    ############################################################
    TOPIC_RVIZ_LIDAR_MARKERS,
    TOPIC_RVIZ_OBJECT_MARKERS,
    TOPIC_RVIZ_OBSTACLE_MARKERS,
    TOPIC_RVIZ_RISK_MARKERS,
    TOPIC_RVIZ_POTENTIAL_MARKERS,
    TOPIC_POTENTIAL_REPULSIVE_VECTOR,
    TOPIC_POTENTIAL_ATTRACTIVE_VECTOR,
    TOPIC_POTENTIAL_RESULT_VECTOR,
    TOPIC_LOCAL_TARGET_POSE,
    TOPIC_GLOBAL_PATH,
    SHOW_POTENTIAL_MARKERS,
    POTENTIAL_VECTOR_SCALE,
    POTENTIAL_ARROW_Z_OFFSET,
    POTENTIAL_ARROW_SHAFT_DIAMETER,
    POTENTIAL_ARROW_HEAD_DIAMETER,
    POTENTIAL_ARROW_HEAD_LENGTH,
    LOCAL_TARGET_MARKER_RADIUS,

    ############################################################
    # Timer
    ############################################################
    VISUALIZATION_HZ,
)


############################################################
# 4. Marker 생성 utility import
############################################################

# marker_utils.py는 Marker 메시지를 쉽게 만드는 helper 함수만 제공한다.
# 이 파일에서 직접 Marker 필드를 매번 채우지 않기 위해 사용한다.
from .marker_utils import (
    # ColorRGBA 생성
    make_color,

    # CUBE marker 생성
    make_cube_marker,

    # POINTS marker 생성
    make_points_marker,

    # SPHERE marker 생성
    make_sphere_marker,

    # LINE_LIST marker 생성
    make_line_list_marker,

    # risk 0.0~1.0 값을 색상으로 변환
    risk_to_color,

    # 전차 heading을 위한 import
    make_arrow_marker,
)




def pointcloud2_to_xyz_array(msg: PointCloud2) -> np.ndarray:
    """Return PointCloud2 XYZ fields as a contiguous float32 (N, 3) array.

    ROS2 Humble/newer sensor_msgs_py provides read_points_numpy(), which avoids
    building Python dict/list objects for every LiDAR hit.  The fallback keeps the
    node usable on older sensor_msgs_py versions.
    """
    try:
        arr = point_cloud2.read_points_numpy(
            msg, field_names=("x", "y", "z"), skip_nans=True
        )
    except Exception:
        pts = point_cloud2.read_points(
            msg, field_names=("x", "y", "z"), skip_nans=True
        )
        if isinstance(pts, np.ndarray):
            arr = pts
        else:
            arr = np.asarray(list(pts), dtype=np.float32)
    if arr is None:
        return np.empty((0, 3), dtype=np.float32)
    arr = np.asarray(arr)
    if arr.dtype.fields:
        arr = np.column_stack((arr["x"], arr["y"], arr["z"]))
    arr = np.asarray(arr, dtype=np.float32)
    if arr.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    return np.ascontiguousarray(arr.reshape(-1, 3), dtype=np.float32)

############################################################
# 5. RViz Visualizer Node 정의
############################################################

class RvizVisualizerNode(Node):
    """
    Tank Challenge 데이터를 RViz2 marker로 변환하는 ROS2 node.

    역할:
        - ros_bridge에서 publish한 상태 topic을 subscribe
        - 최신 상태를 내부 변수에 저장
        - timer 주기마다 RViz2 MarkerArray를 publish

    이 node의 핵심 특징:
        - subscribe callback에서는 "값 저장"만 한다.
        - 실제 marker publish는 timer_callback에서 주기적으로 수행한다.

    이렇게 나누는 이유:
        1. 입력 topic마다 수신 주기가 다를 수 있음
        2. RViz marker publish 주기를 하나로 통제할 수 있음
        3. LiDAR point가 많이 들어와도 RViz 렌더링 부하를 제어할 수 있음
    """

    ############################################################
    # 5.1 생성자
    ############################################################

    def __init__(self) -> None:
        """
        node 초기화 함수.

        수행 내용:
            1. ROS2 node 이름 설정
            2. 최신 상태 저장 변수 초기화
            3. subscriber 생성
            4. publisher 생성
            5. timer 생성
        """

        ########################################################
        # A. ROS2 node 이름 설정
        ########################################################

        # super().__init__("node_name")으로 ROS2 node 이름을 지정한다.
        #
        # ros2 node list에서 보이는 이름:
        #   /tank_rviz_visualizer_node
        super().__init__("tank_rviz_visualizer_node")
        self.declare_parameter("lidar_pc2_topic", "/tank/sensor/lidar/detected_points_map")
        self.declare_parameter("lidar_ray_pc2_topic", "/tank/sensor/lidar/all_detected_points_map")
        self.declare_parameter("lidar_origin_topic", "/tank/sensor/lidar/origin")
        self.lidar_pc2_topic = str(self.get_parameter("lidar_pc2_topic").value)
        self.lidar_ray_pc2_topic = str(self.get_parameter("lidar_ray_pc2_topic").value)
        self.lidar_origin_topic = str(self.get_parameter("lidar_origin_topic").value)


        ########################################################
        # B. 최신 입력 상태 저장 변수
        ########################################################

        # 아군 전차 최신 위치.
        #
        # 처음에는 아직 topic을 받지 않았으므로 None.
        # /tank/player/pose가 들어오면 PoseStamped로 갱신된다.
        self.player_pose: Optional[PoseStamped] = None

        # 적 전차 최신 위치.
        self.enemy_pose: Optional[PoseStamped] = None

        # 목표 지점 최신 위치.
        self.goal_pose: Optional[PoseStamped] = None

        # 장애물 목록.
        #
        # /tank/map/obstacles에서 JSON 문자열을 받아 list[dict]로 저장한다.
        # 아직 schema가 확정되지 않았으므로 dict 기반으로 유연하게 처리한다.
        self.obstacles: List[Dict[str, Any]] = []

        # LiDAR point 목록.
        #
        # /tank/sensor/lidar/detected_points_map PointCloud2에서 받은 map-frame XYZ tuple 목록.
        self.lidar_points: List[Tuple[float, float, float]] = []

        # LiDAR ray 표시용 endpoint 목록.
        # detected_points_map은 지형 분리 후 obstacle-only일 수 있으므로,
        # 실시간 스캔 ray는 all_detected_points_map을 별도로 사용한다.
        self.lidar_ray_points: List[Tuple[float, float, float]] = []

        # LiDAR ray 시작점. lidar_processor_node가 publish하는 map-frame PointStamped.
        self.lidar_origin: Optional[PointStamped] = None

        # 아군/적 전차 상태 dict.
        #
        # /tank/player/state, /tank/enemy/state는 JSON 문자열 형태로 들어오며,
        # 이 안의 playerBodyX / enemyBodyX 값을 heading degree로 사용한다.
        self.player_state: Dict[str, Any] = {}
        self.enemy_state: Dict[str, Any] = {}

        # RViz heading 시각화에 사용할 최신 heading degree.
        #
        # 시뮬레이터 기준:
        #   X Degree = 0    → 전차 전방이 +Pos 두 번째 방향
        #   RViz tank_map   → +map.y 방향
        #   오른쪽 회전     → degree 증가
        self.player_heading_deg: float = 0.0
        self.enemy_heading_deg: float = 0.0

        # Potential Field/APF vector topics. 값이 들어오면 RViz arrow로 표시한다.
        self.potential_repulsive_vector: Optional[Vector3Stamped] = None
        self.potential_attractive_vector: Optional[Vector3Stamped] = None
        self.potential_result_vector: Optional[Vector3Stamped] = None
        self.local_target_pose: Optional[PoseStamped] = None
        self.global_path: Optional[NavPath] = None


        ########################################################
        # C. Subscriber 생성
        ########################################################

        # 아군 전차 위치 subscribe.
        #
        # 입력 타입:
        #   geometry_msgs/msg/PoseStamped
        #
        # callback:
        #   player_pose_callback()
        self.create_subscription(
            PoseStamped,
            TOPIC_PLAYER_POSE,
            self.player_pose_callback,
            10,
        )

        # 적 전차 위치 subscribe.
        self.create_subscription(
            PoseStamped,
            TOPIC_ENEMY_POSE,
            self.enemy_pose_callback,
            10,
        )

        # 목표 지점 위치 subscribe.
        self.create_subscription(
            PoseStamped,
            TOPIC_GOAL_POSE,
            self.goal_pose_callback,
            10,
        )

        # 장애물 정보 subscribe.
        #
        # 입력 타입:
        #   std_msgs/msg/String
        #
        # 실제 내용:
        #   JSON 문자열
        self.create_subscription(
            String,
            TOPIC_OBSTACLES,
            self.obstacles_callback,
            10,
        )

        # LiDAR point 정보 subscribe.
        # 입력 타입: sensor_msgs/msg/PointCloud2
        # lidar_processor_node가 map 좌표계로 변환한 obstacle hit point만 받는다.
        self.create_subscription(
            PointCloud2,
            self.lidar_pc2_topic,
            self.lidar_points_callback,
            10,
        )

        # LiDAR ray 표시용 endpoint.
        # detected_points_map은 obstacle-only일 수 있으므로 ray는 all_detected_points_map을 사용한다.
        self.create_subscription(
            PointCloud2,
            self.lidar_ray_pc2_topic,
            self.lidar_ray_points_callback,
            10,
        )

        # LiDAR ray 시작점.
        self.create_subscription(
            PointStamped,
            self.lidar_origin_topic,
            self.lidar_origin_callback,
            10,
        )

        # 아군 전차 상태 subscribe.
        #
        # 입력 타입:
        #   std_msgs/msg/String
        #
        # 실제 내용:
        #   JSON 문자열
        #
        # 사용 목적:
        #   playerBodyX 값을 읽어서 heading arrow와 전차 marker 회전에 반영한다.
        self.create_subscription(
            String,
            TOPIC_PLAYER_STATE,
            self.player_state_callback,
            10,
        )

        # 적 전차 상태 subscribe.
        #
        # 사용 목적:
        #   enemyBodyX 값을 읽어서 적 전차 heading arrow와 marker 회전에 반영한다.
        self.create_subscription(
            String,
            TOPIC_ENEMY_STATE,
            self.enemy_state_callback,
            10,
        )

        # Potential Field/APF vector subscribe.
        self.create_subscription(
            Vector3Stamped,
            TOPIC_POTENTIAL_REPULSIVE_VECTOR,
            self.potential_repulsive_callback,
            10,
        )
        self.create_subscription(
            Vector3Stamped,
            TOPIC_POTENTIAL_ATTRACTIVE_VECTOR,
            self.potential_attractive_callback,
            10,
        )
        self.create_subscription(
            Vector3Stamped,
            TOPIC_POTENTIAL_RESULT_VECTOR,
            self.potential_result_callback,
            10,
        )
        self.create_subscription(
            PoseStamped,
            TOPIC_LOCAL_TARGET_POSE,
            self.local_target_callback,
            10,
        )

        # A* global path subscribe. path_planning가 publish하는 nav_msgs/Path를 선으로 표시한다.
        self.create_subscription(
            NavPath,
            TOPIC_GLOBAL_PATH,
            self.global_path_callback,
            10,
        )


        ########################################################
        # D. Publisher 생성
        ########################################################

        # 아군 전차, 적 전차, 목표 지점 marker publish.
        self.object_marker_pub = self.create_publisher(
            MarkerArray,
            TOPIC_RVIZ_OBJECT_MARKERS,
            10,
        )

        # 시뮬레이터 Terrain 외곽선/grid/reference point marker publish.
        self.terrain_marker_pub = self.create_publisher(
            MarkerArray,
            TOPIC_RVIZ_TERRAIN_MARKERS,
            10,
        )

        # 장애물 marker publish.
        self.obstacle_marker_pub = self.create_publisher(
            MarkerArray,
            TOPIC_RVIZ_OBSTACLE_MARKERS,
            10,
        )

        # LiDAR POINTS marker publish.
        self.lidar_marker_pub = self.create_publisher(
            MarkerArray,
            TOPIC_RVIZ_LIDAR_MARKERS,
            10,
        )

        # 위험도/복잡도 marker publish.
        self.risk_marker_pub = self.create_publisher(
            MarkerArray,
            TOPIC_RVIZ_RISK_MARKERS,
            10,
        )

        # Potential Field/APF vector marker publish.
        self.potential_marker_pub = self.create_publisher(
            MarkerArray,
            TOPIC_RVIZ_POTENTIAL_MARKERS,
            10,
        )


        ########################################################
        # E. Timer 생성
        ########################################################

        # VISUALIZATION_HZ 주기로 timer_callback을 실행한다.
        #
        # 예:
        #   VISUALIZATION_HZ = 5.0
        #   → 1.0 / 5.0 = 0.2초마다 실행
        self.timer = self.create_timer(
            1.0 / VISUALIZATION_HZ,
            self.timer_callback,
        )

        # node 시작 로그.
        self.get_logger().info("tank_rviz_visualizer_node started")


    ############################################################
    # 6. Subscribe callback
    ############################################################

    def player_pose_callback(self, msg: PoseStamped) -> None:
        """
        /tank/player/pose callback.

        아군 전차 최신 위치를 저장한다.
        marker는 여기서 바로 publish하지 않고 timer_callback에서 주기적으로 publish한다.
        """
        self.player_pose = msg

    def enemy_pose_callback(self, msg: PoseStamped) -> None:
        """
        /tank/enemy/pose callback.

        적 전차 최신 위치를 저장한다.
        """
        self.enemy_pose = msg

    def goal_pose_callback(self, msg: PoseStamped) -> None:
        """
        /tank/goal/pose callback.

        목표 지점 최신 위치를 저장한다.
        """
        self.goal_pose = msg

    def player_state_callback(self, msg: String) -> None:
        """
        /tank/player/state callback.

        ros_bridge의 /tank/player/state는 playerBodyX를 top-level에 그대로 두지 않고,
        다음처럼 body dict 안에 넣어 publish한다.

            "body": {"x": playerBodyX, "y": playerBodyY, "z": playerBodyZ}

        따라서 heading은 data["body"]["x"]를 1순위로 읽어야 한다.
        """
        try:
            data = json.loads(msg.data)

            if not isinstance(data, dict):
                return

            self.player_state = data
            self.player_heading_deg = self._extract_body_heading_deg(
                data,
                top_level_keys=[
                    HEADING_DEGREE_KEY_PLAYER,
                    "playerBodyX",
                    "bodyX",
                    "xDegree",
                    "heading",
                ],
                default=self.player_heading_deg,
            )

        except Exception:
            # state 파싱 실패가 RViz node 전체를 죽이면 안 되므로 무시한다.
            pass

    def enemy_state_callback(self, msg: String) -> None:
        """
        /tank/enemy/state callback.

        enemy heading도 data["body"]["x"]를 1순위로 읽는다.
        """
        try:
            data = json.loads(msg.data)

            if not isinstance(data, dict):
                return

            self.enemy_state = data
            self.enemy_heading_deg = self._extract_body_heading_deg(
                data,
                top_level_keys=[
                    HEADING_DEGREE_KEY_ENEMY,
                    "enemyBodyX",
                    "bodyX",
                    "xDegree",
                    "heading",
                ],
                default=self.enemy_heading_deg,
            )

        except Exception:
            pass

    def obstacles_callback(self, msg: String) -> None:
        """
        /tank/map/obstacles callback.

        역할:
            JSON 문자열 형태의 장애물 정보를 list[dict]로 변환해서 저장한다.

        기대 입력 형식 1:
            [
                {
                    "x": 120.0,
                    "y": 80.0,
                    "z": 0.0,
                    "type": "rock",
                    "risk": 0.7,
                    "complexity": 0.5,
                    "size": 4.0
                }
            ]

        기대 입력 형식 2:
            {
                "obstacles": [
                    {
                        "x": 120.0,
                        "y": 80.0,
                        "risk": 0.7
                    }
                ]
            }

        방어적 파싱을 하는 이유:
            - ros_bridge의 obstacle schema가 바뀔 수 있음
            - perception/planning node에서 다른 key를 추가할 수 있음
            - JSON 파싱 실패로 node 전체가 죽으면 안 됨
        """
        try:
            # std_msgs/String의 실제 문자열은 msg.data에 들어있다.
            data = json.loads(msg.data)

            # 형식 1: 바로 list가 들어온 경우
            if isinstance(data, list):
                self.obstacles = data

            # 형식 2: dict 안에 obstacles key가 있는 경우
            elif isinstance(data, dict):
                self.obstacles = data.get("obstacles", [])

            # 예상하지 못한 형식이면 빈 목록으로 처리
            else:
                self.obstacles = []

        except Exception:
            # JSON 파싱 실패 시 node를 죽이지 않고 빈 목록으로 처리
            self.obstacles = []

    def lidar_points_callback(self, msg: PointCloud2) -> None:
        """Store obstacle-only LiDAR PointCloud2 as lightweight map-frame XYZ tuples."""
        try:
            points = pointcloud2_to_xyz_array(msg)
            if LIDAR_VISUALIZATION_SAMPLE_STEP > 1 and points.shape[0] > 0:
                points = points[:: max(1, int(LIDAR_VISUALIZATION_SAMPLE_STEP))]
            self.lidar_points = [(float(x), float(y), float(z)) for x, y, z in points]
        except Exception as exc:
            self.get_logger().debug(f"failed to parse lidar PointCloud2 for RViz: {exc}")
            self.lidar_points = []

    def lidar_ray_points_callback(self, msg: PointCloud2) -> None:
        """Store all detected LiDAR endpoints for live ray visualization."""
        try:
            points = pointcloud2_to_xyz_array(msg)
            if LIDAR_VISUALIZATION_SAMPLE_STEP > 1 and points.shape[0] > 0:
                points = points[:: max(1, int(LIDAR_VISUALIZATION_SAMPLE_STEP))]
            self.lidar_ray_points = [(float(x), float(y), float(z)) for x, y, z in points]
        except Exception as exc:
            self.get_logger().debug(f"failed to parse lidar ray PointCloud2 for RViz: {exc}")
            self.lidar_ray_points = []

    def lidar_origin_callback(self, msg: PointStamped) -> None:
        """Store latest map-frame LiDAR origin for ray start point."""
        self.lidar_origin = msg

    def _current_lidar_origin_xyz(self) -> Optional[Tuple[float, float, float]]:
        """Return LiDAR origin, falling back to player pose if origin topic is not available yet."""
        if self.lidar_origin is not None:
            p = self.lidar_origin.point
            return (float(p.x), float(p.y), float(p.z))
        if self.player_pose is not None:
            p = self.player_pose.pose.position
            return (float(p.x), float(p.y), float(p.z) + 1.2)
        return None

    def potential_repulsive_callback(self, msg: Vector3Stamped) -> None:
        self.potential_repulsive_vector = msg

    def potential_attractive_callback(self, msg: Vector3Stamped) -> None:
        self.potential_attractive_vector = msg

    def potential_result_callback(self, msg: Vector3Stamped) -> None:
        self.potential_result_vector = msg

    def local_target_callback(self, msg: PoseStamped) -> None:
        self.local_target_pose = msg

    def global_path_callback(self, msg: NavPath) -> None:
        self.global_path = msg


    ############################################################
    # 7. Timer callback
    ############################################################

    def timer_callback(self) -> None:
        """
        RViz2 marker 주기 publish 함수.

        이 함수는 VISUALIZATION_HZ 주기로 반복 실행된다.

        실행 순서:
            1. 전차/목표 marker publish
            2. 장애물 marker publish
            3. LiDAR point marker publish
            4. 위험도/복잡도 marker publish
        """

        self.publish_terrain_markers()
        self.publish_object_markers()
        self.publish_obstacle_markers()
        self.publish_lidar_markers()
        self.publish_risk_markers()
        self.publish_potential_markers()



    ############################################################
    # 7-1. Terrain markers
    ############################################################

    def _raw_to_map_xyz(self, raw_x: float, raw_y: float, raw_z: float):
        """
        Unity raw 좌표를 RViz tank_map 좌표로 변환한다.

        시뮬레이터 UI 기준:
            Pos A/B = raw.x / raw.z
            Alt     = raw.y

        RViz tank_map 기준:
            map.x = raw.x
            map.y = raw.z
            map.z = raw.y
        """
        return float(raw_x), float(raw_z), float(raw_y)


    @staticmethod
    def _extract_body_heading_deg(
        data: Dict[str, Any],
        top_level_keys,
        default: float,
    ) -> float:
        """
        /tank/player/state 또는 /tank/enemy/state JSON에서 차체 heading degree를 추출한다.

        실제 ros_bridge state 구조:
            {
                "body": {"x": playerBodyX, "y": playerBodyY, "z": playerBodyZ},
                "turret": {"x": playerTurretX, "y": playerTurretY},
                ...
            }

        Simple Flat 검증 결과:
            - 최초 리스폰: body.x ≈ 0
            - 왼쪽 회전 : body.x ≈ 326
            - 오른쪽 회전: body.x ≈ 30

        따라서 heading은 body.x를 1순위로 사용한다.
        top-level key는 schema 변경을 대비한 fallback이다.
        """

        # 1순위: 현재 bridge가 publish하는 실제 구조 data["body"]["x"]
        body = data.get("body")
        if isinstance(body, dict):
            for key in ("x", "bodyX", "playerBodyX", "enemyBodyX"):
                if key in body:
                    try:
                        return float(body[key])
                    except (TypeError, ValueError):
                        pass

        # 2순위: 혹시 body가 없고 top-level에 회전값이 들어오는 경우
        for key in top_level_keys:
            if key in data:
                try:
                    return float(data[key])
                except (TypeError, ValueError):
                    pass

        # 3순위: 읽을 수 없으면 이전 heading 유지
        return float(default)

    @staticmethod
    def _heading_deg_to_map_vector(heading_deg: float):
        """
        시뮬레이터 X Degree를 RViz tank_map 평면 방향 벡터로 변환한다.

        기준:
            X Degree = 0:
                시뮬레이터 전방은 +raw.z 방향
                RViz에서는 +map.y 방향

            오른쪽 회전:
                X Degree 증가

        따라서 heading h에 대해:
            map dx = sin(h)
            map dy = cos(h)

        예:
            h = 0도:
                dx = 0, dy = 1
                RViz +y 방향

            h = 90도:
                dx = 1, dy = 0
                RViz +x 방향
        """
        rad = math.radians(float(heading_deg))

        dx = math.sin(rad)
        dy = math.cos(rad)

        return dx, dy

    @staticmethod
    def _heading_deg_to_marker_yaw_quaternion(heading_deg: float):
        """
        전차 CUBE marker를 heading 방향에 맞춰 회전시키기 위한 quaternion을 반환한다.

        RViz CUBE marker는 기본적으로 local x/y 축을 가진다.
        우리는 전차의 길이 방향을 local +y로 보고 있다.

        local +y를 heading vector와 맞추려면 yaw = -heading_deg를 사용한다.

        반환:
            (qz, qw)

        z축 회전만 사용하므로 qx=qy=0이다.
        """
        yaw = math.radians(-float(heading_deg))
        qz = math.sin(yaw / 2.0)
        qw = math.cos(yaw / 2.0)
        return qz, qw

    @staticmethod
    def _apply_yaw_to_marker(marker, heading_deg: float) -> None:
        """
        CUBE marker에 heading yaw 회전을 적용한다.

        이 함수는 marker 객체를 직접 수정한다.
        """
        qz, qw = RvizVisualizerNode._heading_deg_to_marker_yaw_quaternion(heading_deg)
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = qz
        marker.pose.orientation.w = qw

    def publish_terrain_markers(self) -> None:
        """
        시뮬레이터 Simple Flat/Terrain의 실제 0~300 좌표계를 RViz2에 표시한다.

        왜 필요한가?
            RViz 기본 Grid는 원점 중심으로 보일 수 있다.
            그러나 Tank Challenge Terrain은 사용자가 확인한 기준으로
            Pos x/z가 0~300 범위에서 움직인다.

        표시 내용:
            - 흰색 외곽선: 실제 시뮬레이터 terrain boundary
            - 회색 선: 10m 간격 grid
            - 파란 sphere: 시작 위치 Pos=20/20, Alt=8
            - 노란 sphere: 사용자가 확인한 left-front corner Pos=0/300, Alt=8
        """

        markers = MarkerArray()
        marker_id = 0
        z = TERRAIN_MARKER_Z

        ########################################################
        # 1. Terrain boundary: (0,0) ~ (300,300)
        ########################################################

        if SHOW_TERRAIN_BOUNDARY:
            boundary_segments = [
                ((TERRAIN_MIN_X, TERRAIN_MIN_Y, z), (TERRAIN_MAX_X, TERRAIN_MIN_Y, z)),
                ((TERRAIN_MAX_X, TERRAIN_MIN_Y, z), (TERRAIN_MAX_X, TERRAIN_MAX_Y, z)),
                ((TERRAIN_MAX_X, TERRAIN_MAX_Y, z), (TERRAIN_MIN_X, TERRAIN_MAX_Y, z)),
                ((TERRAIN_MIN_X, TERRAIN_MAX_Y, z), (TERRAIN_MIN_X, TERRAIN_MIN_Y, z)),
            ]

            markers.markers.append(
                make_line_list_marker(
                    MAP_FRAME,
                    "simulator_terrain_boundary",
                    marker_id,
                    boundary_segments,
                    0.25,
                    make_color(1.0, 1.0, 1.0, 0.95),
                )
            )
            marker_id += 1

        ########################################################
        # 2. Terrain grid
        ########################################################

        if SHOW_TERRAIN_GRID_MARKER:
            grid_segments = []

            x = TERRAIN_MIN_X
            while x <= TERRAIN_MAX_X + 1e-6:
                grid_segments.append(((x, TERRAIN_MIN_Y, z), (x, TERRAIN_MAX_Y, z)))
                x += TERRAIN_GRID_STEP

            y = TERRAIN_MIN_Y
            while y <= TERRAIN_MAX_Y + 1e-6:
                grid_segments.append(((TERRAIN_MIN_X, y, z), (TERRAIN_MAX_X, y, z)))
                y += TERRAIN_GRID_STEP

            markers.markers.append(
                make_line_list_marker(
                    MAP_FRAME,
                    "simulator_terrain_grid",
                    marker_id,
                    grid_segments,
                    0.05,
                    make_color(0.45, 0.45, 0.45, 0.45),
                )
            )
            marker_id += 1

        ########################################################
        # 3. Reference points
        ########################################################

        if SHOW_TERRAIN_REFERENCE_POINTS:
            # 시작 위치:
            #   raw=(20,8,20) → map=(20,20,8)
            sx, sy, sz = self._raw_to_map_xyz(
                SIM_PLAYER_START_RAW_X,
                SIM_PLAYER_START_RAW_Y,
                SIM_PLAYER_START_RAW_Z,
            )

            markers.markers.append(
                make_sphere_marker(
                    MAP_FRAME,
                    "simulator_spawn_point",
                    marker_id,
                    sx,
                    sy,
                    0.5,
                    SPAWN_MARKER_SIZE,
                    make_color(0.0, 0.3, 1.0, 0.9),
                )
            )
            marker_id += 1

            # 사용자가 확인한 left-front corner:
            #   raw=(0,8,300) → map=(0,300,8)
            cx, cy, cz = self._raw_to_map_xyz(
                SIM_LEFT_FRONT_CORNER_RAW_X,
                SIM_LEFT_FRONT_CORNER_RAW_Y,
                SIM_LEFT_FRONT_CORNER_RAW_Z,
            )

            markers.markers.append(
                make_sphere_marker(
                    MAP_FRAME,
                    "simulator_left_front_corner",
                    marker_id,
                    cx,
                    cy,
                    0.5,
                    TERRAIN_CORNER_MARKER_SIZE,
                    make_color(1.0, 0.85, 0.0, 0.9),
                )
            )

        self.terrain_marker_pub.publish(markers)



    ############################################################
    # 8. Object markers
    ############################################################

    def publish_object_markers(self) -> None:
        """
        아군 전차, 적 전차, 목표 지점을 RViz2에 표시한다.

        출력 topic:
            /tank/rviz/object_markers

        표시 방식:
            - 아군 전차: 파란색 CUBE
            - 아군 heading: 청록색 ARROW
            - 적 전차  : 빨간색 CUBE
            - 적 heading: 주황색 ARROW
            - 목표 지점: 초록색 SPHERE

        heading 기준:
            /tank/player/state의 playerBodyX 값을 사용한다.

            playerBodyX = 0:
                전차 전방이 시뮬레이터 +Pos 두 번째 방향
                RViz에서는 +map.y 방향

            오른쪽 회전:
                degree 증가
        """

        # MarkerArray는 여러 Marker를 담는 컨테이너 메시지.
        markers = MarkerArray()

        # MarkerArray 내부 marker ID.
        marker_id = 0


        ########################################################
        # 8.1 아군 전차 marker
        ########################################################

        if self.player_pose is not None:
            # PoseStamped 안의 실제 위치는 pose.position에 있다.
            p = self.player_pose.pose.position

            # 전차를 CUBE marker로 표시한다.
            # z 위치는 실제 pose.z를 기준으로 전차 높이의 절반만큼 올린다.
            player_marker = make_cube_marker(
                MAP_FRAME,
                "player_tank",
                marker_id,
                p.x,
                p.y,
                max(p.z, 0.0) + PLAYER_TANK_SCALE_Z / 2.0,
                PLAYER_TANK_SCALE_X,
                PLAYER_TANK_SCALE_Y,
                PLAYER_TANK_SCALE_Z,
                make_color(0.0, 0.25, 1.0, 0.85),
            )

            # 차체 방향을 CUBE marker 회전에 반영한다.
            self._apply_yaw_to_marker(player_marker, self.player_heading_deg)

            markers.markers.append(player_marker)
            marker_id += 1

            ####################################################
            # 8.1.1 아군 전차 heading arrow
            ####################################################

            if SHOW_TANK_HEADING_ARROW:
                dx, dy = self._heading_deg_to_map_vector(self.player_heading_deg)

                start_x = p.x
                start_y = p.y
                start_z = max(p.z, 0.0) + HEADING_ARROW_Z_OFFSET

                end_x = start_x + dx * PLAYER_HEADING_ARROW_LENGTH
                end_y = start_y + dy * PLAYER_HEADING_ARROW_LENGTH
                end_z = start_z

                markers.markers.append(
                    make_arrow_marker(
                        MAP_FRAME,
                        "player_heading_arrow",
                        marker_id,
                        start_x,
                        start_y,
                        start_z,
                        end_x,
                        end_y,
                        end_z,
                        0.6,
                        1.5,
                        2.0,
                        make_color(0.0, 0.8, 1.0, 0.95),
                    )
                )

                marker_id += 1


        ########################################################
        # 8.2 적 전차 marker
        ########################################################

        if self.enemy_pose is not None:
            p = self.enemy_pose.pose.position

            enemy_marker = make_cube_marker(
                MAP_FRAME,
                "enemy_tank",
                marker_id,
                p.x,
                p.y,
                max(p.z, 0.0) + ENEMY_TANK_SCALE_Z / 2.0,
                ENEMY_TANK_SCALE_X,
                ENEMY_TANK_SCALE_Y,
                ENEMY_TANK_SCALE_Z,
                make_color(1.0, 0.05, 0.05, 0.85),
            )

            # 적 전차 차체 방향도 CUBE marker 회전에 반영한다.
            self._apply_yaw_to_marker(enemy_marker, self.enemy_heading_deg)

            markers.markers.append(enemy_marker)
            marker_id += 1

            ####################################################
            # 8.2.1 적 전차 heading arrow
            ####################################################

            if SHOW_TANK_HEADING_ARROW:
                dx, dy = self._heading_deg_to_map_vector(self.enemy_heading_deg)

                start_x = p.x
                start_y = p.y
                start_z = max(p.z, 0.0) + HEADING_ARROW_Z_OFFSET

                end_x = start_x + dx * ENEMY_HEADING_ARROW_LENGTH
                end_y = start_y + dy * ENEMY_HEADING_ARROW_LENGTH
                end_z = start_z

                markers.markers.append(
                    make_arrow_marker(
                        MAP_FRAME,
                        "enemy_heading_arrow",
                        marker_id,
                        start_x,
                        start_y,
                        start_z,
                        end_x,
                        end_y,
                        end_z,
                        0.5,
                        1.3,
                        1.8,
                        make_color(1.0, 0.35, 0.0, 0.95),
                    )
                )

                marker_id += 1


        ########################################################
        # 8.3 목표 지점 marker
        ########################################################

        if self.goal_pose is not None:
            p = self.goal_pose.pose.position

            markers.markers.append(
                make_sphere_marker(
                    MAP_FRAME,
                    "goal",
                    marker_id,
                    p.x,
                    p.y,
                    1.0,
                    GOAL_MARKER_RADIUS,
                    make_color(0.0, 1.0, 0.0, 0.9),
                )
            )

        # MarkerArray publish.
        self.object_marker_pub.publish(markers)


    ############################################################
    # 9. Obstacle markers
    ############################################################

    def publish_obstacle_markers(self) -> None:
        """
        장애물 marker를 RViz2에 표시한다.

        출력 topic:
            /tank/rviz/obstacle_markers

        입력 데이터:
            self.obstacles

        obstacle dict 예시:
            {
                "x": 120.0,
                "y": 80.0,
                "z": 0.0,
                "type": "rock",
                "risk": 0.7,
                "complexity": 0.5,
                "size": 4.0
            }

        표시 방식:
            - CUBE marker
            - size/radius/scale 값이 있으면 marker 크기로 사용
            - risk/risk_score/danger 값이 있으면 색상에 반영
            - risk 값이 없으면 기본 risk=0.5로 표시
        """

        markers = MarkerArray()

        for idx, obs in enumerate(self.obstacles):
            # 장애물 하나는 dict 형태여야 한다.
            if not isinstance(obs, dict):
                continue

            # obstacle 좌표 파싱.
            #
            # 여러 후보 key를 허용하는 이유:
            #   - ros_bridge schema가 바뀔 수 있음
            #   - raw/map 좌표 key가 다를 수 있음
            #   - perception node에서 다른 key 이름을 쓸 수 있음
            x = self._get_float(obs, ["x", "map_x", "raw_x"], 0.0)
            y = self._get_float(obs, ["y", "map_y", "raw_z"], 0.0)
            z = self._get_float(obs, ["z", "map_z", "raw_y"], 0.3)

            # 장애물 크기.
            size = self._get_float(
                obs,
                ["size", "radius", "scale"],
                OBSTACLE_DEFAULT_SCALE,
            )

            # 위험도.
            # 없으면 중간값 0.5로 표시한다.
            risk = self._get_float(
                obs,
                ["risk", "risk_score", "danger"],
                0.5,
            )

            markers.markers.append(
                make_cube_marker(
                    MAP_FRAME,
                    "obstacles",
                    idx,
                    x,
                    y,
                    max(z, 0.3),
                    size,
                    size,
                    size,
                    risk_to_color(risk),
                )
            )

        self.obstacle_marker_pub.publish(markers)


    ############################################################
    # 10. LiDAR markers
    ############################################################

    def publish_lidar_markers(self) -> None:
        """LiDAR PC2 point와 실시간 ray를 RViz2 MarkerArray로 표시한다."""

        markers = MarkerArray()

        # 0) Live LiDAR rays: origin -> all detected endpoints.
        # all_detected_points_map을 사용해야 지형 분리로 빠진 point까지 스캔 ray로 볼 수 있다.
        origin = self._current_lidar_origin_xyz()
        if origin is not None and self.lidar_ray_points:
            line_segments = [(origin, end) for end in self.lidar_ray_points]
            markers.markers.append(
                make_line_list_marker(
                    MAP_FRAME,
                    "lidar_live_rays",
                    10,
                    line_segments,
                    0.035,
                    make_color(0.1, 0.9, 1.0, 0.22),
                )
            )
            markers.markers.append(
                make_sphere_marker(
                    MAP_FRAME,
                    "lidar_origin",
                    11,
                    origin[0],
                    origin[1],
                    origin[2],
                    0.35,
                    make_color(0.1, 0.9, 1.0, 0.85),
                )
            )

        # 1) Obstacle-only hit points.
        if self.lidar_points:
            markers.markers.append(
                make_points_marker(
                    MAP_FRAME,
                    "lidar_detected_points",
                    0,
                    self.lidar_points,
                    LIDAR_POINT_SIZE,
                    make_color(0.0, 1.0, 1.0, LIDAR_DETECTED_ALPHA),
                )
            )

        self.lidar_marker_pub.publish(markers)


    ############################################################
    # 11. Risk / complexity markers
    ############################################################

    def publish_risk_markers(self) -> None:
        """
        위험도/복잡도 marker를 RViz2에 표시한다.

        출력 topic:
            /tank/rviz/risk_markers

        목적:
            카메라/라이다 기반 인식 결과가 나중에 들어왔을 때,
            위험도와 지형 복잡도를 시각적으로 확인할 수 있게 한다.

        이 함수에서 하지 않는 것:
            - risk 계산
            - complexity 계산
            - 객체 분류
            - 경로 비용 계산

        이 함수에서 하는 것:
            - 이미 들어온 risk/complexity 값을 읽음
            - combined score로 단순 결합
            - 반투명 CUBE marker로 표시

        combined 계산:
            combined = 0.7 * risk + 0.3 * complexity

        해석:
            - risk가 더 중요하므로 70%
            - 지형 복잡도는 보조 판단으로 30%

        추후 확장:
            - 수풀 속 초소 부채꼴 위험 구역
            - 적 전차 포탑 방향 기반 사격 위험 구역
            - 지뢰 추정 영역
            - 벽/바위 회피 영역
            - LiDAR point 밀집도 기반 rough terrain 영역
        """

        markers = MarkerArray()

        for idx, obs in enumerate(self.obstacles):
            if not isinstance(obs, dict):
                continue

            x = self._get_float(obs, ["x", "map_x", "raw_x"], 0.0)
            y = self._get_float(obs, ["y", "map_y", "raw_z"], 0.0)

            # 위험도 후보 값.
            risk = self._get_float(
                obs,
                ["risk", "risk_score", "danger"],
                0.0,
            )

            # 지형 복잡도 후보 값.
            complexity = self._get_float(
                obs,
                ["complexity", "roughness", "difficulty"],
                0.0,
            )

            # risk와 complexity를 단순 결합.
            # 이 값은 시각화용 임시 점수이며, 최종 경로계획 cost와는 별도로 다룬다.
            combined = max(0.0, min(1.0, 0.7 * risk + 0.3 * complexity))

            # risk/complexity가 모두 0이면 표시하지 않는다.
            if combined <= 0.0:
                continue

            # 위험도가 높을수록 marker 영역을 크게 표시.
            scale = 4.0 + 12.0 * combined

            markers.markers.append(
                make_cube_marker(
                    MAP_FRAME,
                    "risk_area",
                    idx,
                    x,
                    y,
                    RISK_MARKER_Z,
                    scale,
                    scale,
                    0.05,
                    risk_to_color(combined),
                )
            )

        self.risk_marker_pub.publish(markers)


    ############################################################
    # 12. Potential Field / APF vector markers
    ############################################################

    def publish_potential_markers(self) -> None:
        """Potential Field 벡터를 RViz2 ARROW marker로 표시한다."""
        markers = MarkerArray()

        if not SHOW_POTENTIAL_MARKERS:
            self.potential_marker_pub.publish(markers)
            return

        if self.player_pose is None:
            self.potential_marker_pub.publish(markers)
            return

        start_x = float(self.player_pose.pose.position.x)
        start_y = float(self.player_pose.pose.position.y)
        start_z = float(self.player_pose.pose.position.z) + POTENTIAL_ARROW_Z_OFFSET

        def add_vector_arrow(vector_msg, marker_id: int, namespace: str, color):
            if vector_msg is None:
                return
            vx = float(vector_msg.vector.x)
            vy = float(vector_msg.vector.y)
            vz = float(vector_msg.vector.z)
            mag = (vx * vx + vy * vy + vz * vz) ** 0.5
            if mag < 1.0e-6:
                return
            # RViz 가독성을 위해 단위 방향으로 정규화 후 표시 배율을 곱한다.
            end_x = start_x + (vx / mag) * POTENTIAL_VECTOR_SCALE
            end_y = start_y + (vy / mag) * POTENTIAL_VECTOR_SCALE
            end_z = start_z + (vz / mag) * POTENTIAL_VECTOR_SCALE
            markers.markers.append(
                make_arrow_marker(
                    MAP_FRAME,
                    namespace,
                    marker_id,
                    start_x,
                    start_y,
                    start_z,
                    end_x,
                    end_y,
                    end_z,
                    POTENTIAL_ARROW_SHAFT_DIAMETER,
                    POTENTIAL_ARROW_HEAD_DIAMETER,
                    POTENTIAL_ARROW_HEAD_LENGTH,
                    color,
                )
            )

        add_vector_arrow(
            self.potential_repulsive_vector,
            0,
            "potential_repulsive_vector",
            make_color(1.0, 0.2, 0.0, 0.95),
        )
        add_vector_arrow(
            self.potential_attractive_vector,
            1,
            "potential_attractive_vector",
            make_color(0.0, 0.9, 0.2, 0.95),
        )
        add_vector_arrow(
            self.potential_result_vector,
            2,
            "potential_result_vector",
            make_color(0.7, 0.2, 1.0, 0.95),
        )

        if self.local_target_pose is not None:
            markers.markers.append(
                make_sphere_marker(
                    MAP_FRAME,
                    "local_target_pose",
                    10,
                    float(self.local_target_pose.pose.position.x),
                    float(self.local_target_pose.pose.position.y),
                    float(self.local_target_pose.pose.position.z) + 1.0,
                    LOCAL_TARGET_MARKER_RADIUS,
                    make_color(1.0, 1.0, 0.0, 0.9),
                )
            )

        if self.global_path is not None and len(self.global_path.poses) >= 2:
            segments = []
            poses = self.global_path.poses
            for i in range(len(poses) - 1):
                p0 = poses[i].pose.position
                p1 = poses[i + 1].pose.position
                segments.append(((p0.x, p0.y, p0.z + 0.35), (p1.x, p1.y, p1.z + 0.35)))
            markers.markers.append(
                make_line_list_marker(
                    MAP_FRAME,
                    "astar_global_path",
                    20,
                    segments,
                    0.45,
                    make_color(1.0, 0.0, 0.0, 0.95),
                )
            )

        self.potential_marker_pub.publish(markers)


    ############################################################
    # 12. Utility
    ############################################################

    @staticmethod
    def _get_float(data: Dict[str, Any], keys, default: float) -> float:
        """
        dict에서 여러 후보 key 중 첫 번째로 찾은 값을 float로 변환한다.

        Parameters:
            data:
                값을 읽을 dict.

            keys:
                후보 key 목록.
                앞에서부터 순서대로 검사한다.

            default:
                key가 없거나 float 변환에 실패했을 때 사용할 기본값.

        사용 예:
            x = self._get_float(obs, ["x", "map_x", "raw_x"], 0.0)

        이렇게 하는 이유:
            obstacle/LiDAR/perception schema가 아직 완전히 고정되지 않았기 때문에
            key 이름이 조금 달라도 visualizer가 최대한 동작하게 하기 위함이다.
        """

        for key in keys:
            if key in data:
                try:
                    return float(data[key])
                except Exception:
                    return float(default)

        return float(default)


############################################################
# 13. main
############################################################

def main(args=None) -> None:
    """
    ROS2 node 실행 진입점.

    ros2 run으로 실행될 때 호출된다.

    실행 예:
        ros2 run rviz_visualization rviz_visualizer_node
    """

    # rclpy 초기화.
    rclpy.init(args=args)

    # node 객체 생성.
    node = RvizVisualizerNode()

    try:
        # rclpy.spin:
        #   node가 subscriber callback과 timer callback을 계속 처리하도록 한다.
        rclpy.spin(node)

    finally:
        # 종료 시 node 자원 해제.
        node.destroy_node()

        # rclpy 종료.
        rclpy.shutdown()


############################################################
# 14. 직접 실행 대응
############################################################

if __name__ == "__main__":
    main()
