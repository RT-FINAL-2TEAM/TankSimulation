# -*- coding: utf-8 -*-

import os
"""

############################################################
# config.py
############################################################

패키지:
    rviz_visualization

역할:
    RViz2 시각화 노드에서 사용할 전역 설정값을 관리한다.

이 파일에서 관리하는 것:
    1. RViz2 Fixed Frame 기준
    2. 시각화 publish 주기
    3. ros_bridge에서 subscribe할 입력 topic 이름
    4. RViz2에서 표시할 출력 MarkerArray topic 이름
    5. 전차, 목표, 장애물, LiDAR point, 위험도 marker 크기

이 파일에서 하지 않는 것:
    - A* 경로계획
    - Potential Field 계산
    - Risk Map 계산
    - YOLO 객체 판단
    - LiDAR clustering
    - 장애물 위험도 산정
    - 주행 제어 명령 생성

설계 원칙:
    rviz_visualization은 판단 알고리즘 패키지가 아니다.
    이 패키지는 다른 노드가 publish한 정보를 RViz2에서 보기 좋게 표시하는 역할만 한다.

데이터 흐름:
    Tank Challenge Simulator
        ↓ HTTP API
    ros_bridge
        ↓ ROS2 topic
    rviz_visualization
        ↓ MarkerArray
    RViz2
"""


############################################################
# 1. RViz2 Fixed Frame 기준
############################################################

# ros_bridge에서 Unity 좌표를 ROS/RViz용 map 좌표로 변환할 때 사용하는 frame.
#
# ros_bridge의 좌표 변환 기준:
#   Unity raw 좌표: x=좌우, y=높이, z=전후
#   RViz map 좌표 : x=좌우, y=전후, z=높이
#
# 즉:
#   map.x = raw.x
#   map.y = raw.z
#   map.z = raw.y
#
# RViz2의 Fixed Frame도 반드시 이 값으로 맞춘다.
# RViz2 Global Options → Fixed Frame = tank_map
MAP_FRAME = "tank_map"

############################################################
# 2-1. 시뮬레이터 지형 / 좌표 설정
############################################################

# 사용자가 Simple Flat map에서 직접 확인한 시뮬레이터 좌표 기준:
#   시작 위치:
#       Pos : 20 / 20
#       Alt : 8
#       X Degree : 0
#       Y Degree : 0
#
# 해석:
#   시뮬레이터 UI Pos A/B = Unity raw x / raw z
#   시뮬레이터 UI Alt     = Unity raw y
#
# 따라서 RViz tank_map 좌표는 다음 기준으로 고정한다.
#   map.x = raw.x
#   map.y = raw.z
#   map.z = raw.y
#
# Simple Flat map에서 확인한 전방 방향:
#   X Degree = 0일 때 전차 전방은 +raw.z 방향
#   RViz에서는 +map.y 방향
#
# Terrain 크기:
#   Q&A 기준 Unity 좌표 300 x 300
#   RViz에서는 x=0~300, y=0~300 평면으로 표시한다.
TERRAIN_MIN_X = float(os.environ.get("TANK_TERRAIN_MIN_X", "0.0"))
TERRAIN_MAX_X = float(os.environ.get("TANK_TERRAIN_MAX_X", "300.0"))
TERRAIN_MIN_Y = float(os.environ.get("TANK_TERRAIN_MIN_Y", "0.0"))
TERRAIN_MAX_Y = float(os.environ.get("TANK_TERRAIN_MAX_Y", "300.0"))

# RViz에 직접 그릴 시뮬레이터 실제 grid 간격.
# RViz 기본 Grid는 원점 중심으로 보일 수 있으므로,
# 실제 시뮬레이터 terrain 확인은 /tank/rviz/terrain_markers를 기준으로 한다.
TERRAIN_GRID_STEP = float(os.environ.get("TANK_TERRAIN_GRID_STEP", "10.0"))

# Terrain marker를 표시할지 여부.
SHOW_TERRAIN_BOUNDARY = os.environ.get(
    "TANK_SHOW_TERRAIN_BOUNDARY",
    "true",
).strip().lower() in ("1", "true", "yes", "y")

SHOW_TERRAIN_GRID_MARKER = os.environ.get(
    "TANK_SHOW_TERRAIN_GRID_MARKER",
    "true",
).strip().lower() in ("1", "true", "yes", "y")

SHOW_TERRAIN_REFERENCE_POINTS = os.environ.get(
    "TANK_SHOW_TERRAIN_REFERENCE_POINTS",
    "true",
).strip().lower() in ("1", "true", "yes", "y")

# line marker가 지면과 겹쳐 깜빡이지 않도록 살짝 위에 표시한다.
TERRAIN_MARKER_Z = float(os.environ.get("TANK_TERRAIN_MARKER_Z", "0.02"))

# RViz에 표시할 terrain boundary/grid marker topic.
TOPIC_RVIZ_TERRAIN_MARKERS = "/tank/rviz/terrain_markers"


############################################################
# 2-2. 시뮬레이터 스폰 / 기준점
############################################################

# 시뮬레이터 UI 기준 시작 좌표 mirror.
#
# 주의:
#   ros_bridge의 /init BL/RED 시작 좌표와도 반드시 맞춰야 한다.
#   사용자가 확인한 시작 위치가 Pos=20/20, Alt=8이면
#   Unity raw = (x=20, y=8, z=20)
#   RViz map  = (x=20, y=20, z=8)
SIM_PLAYER_START_RAW_X = float(os.environ.get("TANK_PLAYER_START_RAW_X", "20.0"))
SIM_PLAYER_START_RAW_Y = float(os.environ.get("TANK_PLAYER_START_RAW_Y", "8.0"))
SIM_PLAYER_START_RAW_Z = float(os.environ.get("TANK_PLAYER_START_RAW_Z", "20.0"))

# X Degree=0일 때 전차가 바라보는 방향.
# 사용자가 확인한 기준:
#   0 deg = +raw.z 방향 = RViz +map.y 방향
SIM_PLAYER_START_X_DEGREE = float(os.environ.get("TANK_PLAYER_START_X_DEGREE", "0.0"))
SIM_PLAYER_START_Y_DEGREE = float(os.environ.get("TANK_PLAYER_START_Y_DEGREE", "0.0"))

# 기준점 표시 크기.
TERRAIN_CORNER_MARKER_SIZE = float(os.environ.get("TANK_TERRAIN_CORNER_MARKER_SIZE", "3.0"))
SPAWN_MARKER_SIZE = float(os.environ.get("TANK_SPAWN_MARKER_SIZE", "5.0"))

# Simple Flat map에서 확인한 전방/좌측 앞쪽 꼭짓점.
# 사용자가 확인한 좌표:
#   Pos : 0 / 300
#   Alt : 8
SIM_LEFT_FRONT_CORNER_RAW_X = float(os.environ.get("TANK_LEFT_FRONT_RAW_X", "0.0"))
SIM_LEFT_FRONT_CORNER_RAW_Y = float(os.environ.get("TANK_LEFT_FRONT_RAW_Y", "8.0"))
SIM_LEFT_FRONT_CORNER_RAW_Z = float(os.environ.get("TANK_LEFT_FRONT_RAW_Z", "300.0"))


############################################################
# 2. RViz Marker publish 주기
############################################################

# RViz2 marker를 publish하는 주기.
#
# 5Hz 의미:
#   1초에 5번 marker 갱신
#
# 너무 높이면:
#   - RViz2가 무거워질 수 있음
#   - LiDAR point가 많을 때 렌더링 부하 증가
#
# 너무 낮으면:
#   - 전차 위치/장애물 표시가 늦게 반영됨
#
# 추천:
#   초기 디버깅: 5Hz
#   LiDAR point가 많으면: 2~3Hz
#   전차 위치만 볼 때: 10Hz까지 가능
VISUALIZATION_HZ = 5.0


############################################################
# 3. ros_bridge 입력 topic
############################################################

# 아래 topic들은 ros_bridge 패키지에서 publish한다.
# rviz_visualization은 이 topic들을 subscribe해서 RViz2용 marker로 변환한다.


##############################
# 3.1 전차 / 목표 위치
##############################

# 아군 전차 위치.
# 메시지 타입:
#   geometry_msgs/msg/PoseStamped
#
# 좌표 기준:
#   frame_id = tank_map
TOPIC_PLAYER_POSE = "/tank/player/pose"

# 적 전차 위치.
# 메시지 타입:
#   geometry_msgs/msg/PoseStamped
#
# 좌표 기준:
#   frame_id = tank_map
TOPIC_ENEMY_POSE = "/tank/enemy/pose"

# 목표 지점 위치.
# 메시지 타입:
#   geometry_msgs/msg/PoseStamped
#
# 좌표 기준:
#   frame_id = tank_map
#
# 예:
#   최종 지휘소, 목표 접근 지점, route endpoint
TOPIC_GOAL_POSE = "/tank/goal/pose"


##############################
# 3.2 장애물 정보
##############################

# 장애물 정보.
# 메시지 타입:
#   std_msgs/msg/String
#
# 내부 데이터 형식:
#   JSON 문자열
#
# 현재 기대 형식 예시:
#   [
#     {
#       "x": 120.0,
#       "y": 80.0,
#       "z": 0.0,
#       "type": "rock",
#       "risk": 0.7,
#       "complexity": 0.5,
#       "size": 4.0
#     }
#   ]
#
# 주의:
#   이 패키지는 risk/complexity를 계산하지 않는다.
#   값이 들어오면 색상/크기로 표시만 한다.
TOPIC_OBSTACLES = "/tank/map/obstacles"


##############################
# 3.3 LiDAR 정보
##############################

# 3D LiDAR point 배열.
# 메시지 타입:
#   std_msgs/msg/String
#
# 내부 데이터 형식:
#   JSON 문자열
#
# 기대 형식 예시:
#   [
#     {"x": 10.0, "y": 20.0, "z": 0.3},
#     {"x": 10.5, "y": 20.2, "z": 0.4}
#   ]
#
# 좌표 기준:
#   가능하면 ros_bridge에서 이미 tank_map 기준으로 변환해서 publish하는 것이 좋다.
#
# 이 topic은 RViz2에서 POINTS marker로 표시한다.
TOPIC_LIDAR_POINTS = "/tank/sensor/lidar/points"

# LiDAR point 개수.
# 메시지 타입:
#   std_msgs/msg/Int32
#
# 현재 visualizer node에서는 필수로 사용하지 않는다.
# 추후 디버깅 text marker 또는 상태 패널에 표시할 수 있다.
TOPIC_LIDAR_POINTS_COUNT = "/tank/sensor/lidar/points_count"


############################################################
# 4. RViz2 표시용 출력 topic
############################################################

# 아래 topic들은 rviz_visualization 패키지가 publish한다.
# RViz2에서는 MarkerArray Display를 추가하고 이 topic들을 선택한다.


##############################
# 4.1 전차 / 목표 marker
##############################

# 아군 전차, 적 전차, 목표 지점을 하나의 MarkerArray로 표시한다.
#
# 포함 marker:
#   - player_tank: 파란색 박스
#   - enemy_tank : 빨간색 박스
#   - goal       : 초록색 구
#
# 메시지 타입:
#   visualization_msgs/msg/MarkerArray
TOPIC_RVIZ_OBJECT_MARKERS = "/tank/rviz/object_markers"

############################################################
# 전차 heading / 방향 시각화
############################################################

# ros_bridge가 publish하는 player state topic.
# 여기서 playerBodyX 값을 읽어 전차 heading으로 사용한다.
TOPIC_PLAYER_STATE = "/tank/player/state"
TOPIC_ENEMY_STATE = "/tank/enemy/state"

# heading arrow 표시 여부.
SHOW_TANK_HEADING_ARROW = True

# 전차 heading arrow 길이.
PLAYER_HEADING_ARROW_LENGTH = 15.0
ENEMY_HEADING_ARROW_LENGTH = 12.0

# arrow가 지면과 겹치지 않도록 올리는 높이.
HEADING_ARROW_Z_OFFSET = 2.5

# 포신/포탑 방향 arrow 표시 여부.
# /tank/player/state의 turret.x/y를 사용한다.
# turret.x: 포탑 좌우(yaw, world heading)
# turret.y: 포신 상하(pitch)
SHOW_TURRET_GUN_ARROW = os.environ.get(
    "TANK_SHOW_TURRET_GUN_ARROW",
    "true",
).strip().lower() in ("1", "true", "yes", "y")

# 포신 방향 arrow 길이. 차체 heading(청록색)보다 살짝 길게 표시한다.
PLAYER_GUN_ARROW_LENGTH = float(os.environ.get("TANK_PLAYER_GUN_ARROW_LENGTH", "17.0"))
ENEMY_GUN_ARROW_LENGTH = float(os.environ.get("TANK_ENEMY_GUN_ARROW_LENGTH", "14.0"))

# 포신 arrow 시작 높이. 차체 marker/heading arrow와 겹치지 않도록 조금 더 위에 띄운다.
GUN_ARROW_Z_OFFSET = float(os.environ.get("TANK_GUN_ARROW_Z_OFFSET", "3.2"))

# 시뮬레이터 기준:
#   X Degree = 0  → +raw.z 방향
#   RViz map     → +map.y 방향
#
# 오른쪽 회전 시 X Degree가 증가하므로,
# heading vector는:
#   dx = sin(rad)
#   dy = cos(rad)
HEADING_DEGREE_KEY_PLAYER = "playerBodyX"
HEADING_DEGREE_KEY_ENEMY = "enemyBodyX"


##############################
# 4.2 장애물 marker
##############################

# 장애물을 MarkerArray로 표시한다.
#
# 표시 방식:
#   - 기본 CUBE marker
#   - obstacle type/risk/size 값이 있으면 색상과 크기에 반영
#
# 메시지 타입:
#   visualization_msgs/msg/MarkerArray
TOPIC_RVIZ_OBSTACLE_MARKERS = "/tank/rviz/obstacle_markers"


##############################
# 4.3 LiDAR point marker
##############################

# LiDAR point를 RViz POINTS marker로 표시한다.
#
# 표시 방식:
#   - Marker.POINTS
#   - 많은 point를 모두 표시하면 무거우므로 visualizer node에서 샘플링 가능
#
# 메시지 타입:
#   visualization_msgs/msg/MarkerArray
TOPIC_RVIZ_LIDAR_MARKERS = "/tank/rviz/lidar_markers"


##############################
# 4.4 위험도 / 복잡도 marker
##############################

# 위험도 또는 지형 복잡도 정보를 반투명 영역으로 표시한다.
#
# 입력 데이터에 아래 값이 들어오면 사용:
#   - risk
#   - risk_score
#   - danger
#   - complexity
#   - roughness
#   - difficulty
#
# 현재 역할:
#   계산 X
#   표시 O
#
# 추후 사용 예:
#   카메라로 적 초소 탐지 → 부채꼴 위험 영역 표시
#   LiDAR 밀집도 높음 → 지형 복잡도 영역 표시
#   벽/바위 탐지 → 장애물 회피 영역 표시
#
# 메시지 타입:
#   visualization_msgs/msg/MarkerArray
TOPIC_RVIZ_RISK_MARKERS = "/tank/rviz/risk_markers"


############################################################
# 5. 전차 marker 크기
############################################################

# Tank Challenge Q&A 기준 Player 전차 크기:
#   하단 몸체 기준 최대 박스: (3.667, 1.582, 8.066)
#
# Unity 원본 기준을 RViz 평면 표시 기준으로 해석:
#   width  ≈ 3.7
#   length ≈ 8.1
#   height ≈ 1.6
#
# 실제 전차 mesh가 아니라 RViz 디버깅용 bounding box이므로
# 정확한 mesh 형상보다 위치와 방향 파악이 목적이다.
PLAYER_TANK_SCALE_X = 3.7
PLAYER_TANK_SCALE_Y = 8.1
PLAYER_TANK_SCALE_Z = 1.6


# Tank Challenge Q&A 기준 Enemy 전차 크기:
#   하단 몸체 기준 최대 박스: (3.303, 1.131, 6.339)
#
# RViz 디버깅용 bounding box:
#   width  ≈ 3.3
#   length ≈ 6.3
#   height ≈ 1.2
ENEMY_TANK_SCALE_X = 3.3
ENEMY_TANK_SCALE_Y = 6.3
ENEMY_TANK_SCALE_Z = 1.2


############################################################
# 6. 목표 marker 크기
############################################################

# 목표 지점 표시용 sphere marker 지름.
#
# RViz Marker.SPHERE의 scale.x/y/z는 반지름이 아니라 diameter처럼 동작한다.
# 여기서는 목표 지점이 잘 보이도록 5m 크기로 표시한다.
GOAL_MARKER_RADIUS = 5.0


############################################################
# 7. 장애물 marker 기본 크기
############################################################

# 장애물 정보에 size/radius/scale 값이 없을 때 사용할 기본 marker 크기.
#
# 예:
#   바위, 벽 일부, 미확인 장애물
OBSTACLE_DEFAULT_SCALE = 3.0


############################################################
# 8. LiDAR point marker 크기
############################################################

# RViz Marker.POINTS에서 각 point가 화면에 표시되는 크기.
#
# 너무 작으면:
#   LiDAR point가 잘 보이지 않음
#
# 너무 크면:
#   점들이 뭉쳐서 장애물 형상이 흐려짐
#
# 추천 시작값:
#   0.2 ~ 0.3
LIDAR_POINT_SIZE = 0.25

############################################################
# 8-1. LiDAR RViz 표시 정책
############################################################

# RViz에 LiDAR point를 몇 개마다 하나씩 표시할지 결정한다.
#
# 현재 시뮬레이터 설정:
#   Channel = 8
#   수평 angle = 0~359
#   총 point = 360 * 8 = 2880
#
# 예:
#   1 → 2880개 전부 표시
#   2 → 1440개 표시
#   3 → 960개 표시
#   4 → 720개 표시
LIDAR_VISUALIZATION_SAMPLE_STEP = int(os.environ.get("TANK_LIDAR_SAMPLE_STEP", "3"))

# True이면 isDetected=True인 LiDAR hit point만 RViz에 표시한다.
#
# isDetected=True:
#   실제 지형/장애물/물체에 맞은 point
#
# isDetected=False:
#   아무것도 감지하지 않고 Max Distance까지 간 endpoint
#
# 장애물/지형 인지 디버깅 목적이면 True가 맞다.
LIDAR_VISUALIZE_DETECTED_ONLY = os.environ.get(
    "TANK_LIDAR_VISUALIZE_DETECTED_ONLY",
    "true",
).strip().lower() in ("1", "true", "yes", "y")

# LiDAR point["position"]이 시뮬레이터 raw 좌표라고 보고,
# RViz tank_map 좌표로 변환할지 여부.
#
# 변환:
#   map.x = raw.x
#   map.y = raw.z
#   map.z = raw.y
LIDAR_POSITION_IS_UNITY_RAW = os.environ.get(
    "TANK_LIDAR_POSITION_IS_UNITY_RAW",
    "true",
).strip().lower() in ("1", "true", "yes", "y")

# isDetected=True point 색상 투명도.
LIDAR_DETECTED_ALPHA = float(os.environ.get("TANK_LIDAR_DETECTED_ALPHA", "0.75"))

# isDetected=False point를 표시할 때 색상 투명도.
# 기본 설정에서는 detected only라서 사용되지 않는다.
LIDAR_FREE_SPACE_ALPHA = float(os.environ.get("TANK_LIDAR_FREE_SPACE_ALPHA", "0.12"))

# 시뮬레이터 LiDAR channel 수.
# sanity check 용도.
SIM_LIDAR_CHANNELS = int(os.environ.get("TANK_LIDAR_CHANNELS", "8"))

# 시뮬레이터 LiDAR 최대 거리.
# sanity check / free-space 해석용.
SIM_LIDAR_MAX_DISTANCE = float(os.environ.get("TANK_LIDAR_MAX_DISTANCE", "30.0"))


############################################################
# 9. 위험도 / 복잡도 marker 높이
############################################################

# 위험도 영역을 지면보다 살짝 위에 표시하기 위한 z 값.
#
# z=0에 완전히 붙이면 grid나 지면 marker와 겹쳐 보일 수 있으므로
# 0.05~0.1 정도로 띄운다.
RISK_MARKER_Z = 0.08
############################################################
# 12. Potential Field / APF 벡터 시각화
############################################################
# 이 topic들은 potential 또는 경로계획 노드가 publish한다.
# rviz_visualization은 계산하지 않고 RViz 화살표로 표시만 한다.
TOPIC_POTENTIAL_REPULSIVE_VECTOR = "/tank/potential/repulsive_vector"
TOPIC_POTENTIAL_ATTRACTIVE_VECTOR = "/tank/potential/attractive_vector"
TOPIC_POTENTIAL_RESULT_VECTOR = "/tank/potential/result_vector"
TOPIC_LOCAL_TARGET_POSE = "/tank/local_target/pose"

# RViz MarkerArray 출력 topic.
TOPIC_RVIZ_POTENTIAL_MARKERS = "/tank/rviz/potential_markers"

SHOW_POTENTIAL_MARKERS = os.environ.get(
    "TANK_SHOW_POTENTIAL_MARKERS",
    "true",
).strip().lower() in ("1", "true", "yes", "y")

# potential vector는 계산값의 단위가 작을 수 있으므로 RViz 표시용 배율을 둔다.
POTENTIAL_VECTOR_SCALE = float(os.environ.get("TANK_POTENTIAL_VECTOR_SCALE", "12.0"))
POTENTIAL_ARROW_Z_OFFSET = float(os.environ.get("TANK_POTENTIAL_ARROW_Z_OFFSET", "4.0"))
POTENTIAL_ARROW_SHAFT_DIAMETER = float(os.environ.get("TANK_POTENTIAL_ARROW_SHAFT_DIAMETER", "0.25"))
POTENTIAL_ARROW_HEAD_DIAMETER = float(os.environ.get("TANK_POTENTIAL_ARROW_HEAD_DIAMETER", "0.8"))
POTENTIAL_ARROW_HEAD_LENGTH = float(os.environ.get("TANK_POTENTIAL_ARROW_HEAD_LENGTH", "1.2"))
LOCAL_TARGET_MARKER_RADIUS = float(os.environ.get("TANK_LOCAL_TARGET_MARKER_RADIUS", "1.2"))

# path_planning에서 발행하는 A* 전역 경로.
TOPIC_GLOBAL_PATH = "/tank/global_path"

# ---------------------------------------------------------------------------
# Fused/discovered 및 정적 장애물 회피반경 시각화
# ---------------------------------------------------------------------------
# rviz_visualization은 위치를 다시 필터링하지 않는다. path_planning이 발행한
# 안정화된 position_map / avoidance_radius_m을 그대로 그린다.
TOPIC_FUSED_OBJECTS = "/tank/perception/fused_objects"
TOPIC_DISCOVERED_OBJECTS = "/tank/map/discovered/objects"
TOPIC_RVIZ_DYNAMIC_AVOIDANCE_MARKERS = "/tank/rviz/dynamic_avoidance_markers"
TOPIC_RVIZ_STATIC_AVOIDANCE_MARKERS = "/tank/rviz/static_avoidance_markers"
DYNAMIC_AVOIDANCE_DISK_Z_OFFSET = float(os.environ.get("TANK_DYNAMIC_AVOIDANCE_DISK_Z_OFFSET", "0.10"))
DYNAMIC_AVOIDANCE_DISK_HEIGHT = float(os.environ.get("TANK_DYNAMIC_AVOIDANCE_DISK_HEIGHT", "0.08"))
DYNAMIC_AVOIDANCE_TEXT_HEIGHT = float(os.environ.get("TANK_DYNAMIC_AVOIDANCE_TEXT_HEIGHT", "0.80"))
DYNAMIC_AVOIDANCE_TEXT_Z_OFFSET = float(os.environ.get("TANK_DYNAMIC_AVOIDANCE_TEXT_Z_OFFSET", "1.00"))
DYNAMIC_AVOIDANCE_RADIUS_FALLBACK_M = float(os.environ.get("TANK_DYNAMIC_AVOIDANCE_RADIUS_FALLBACK_M", "5.0"))
