# -*- coding: utf-8 -*-
"""
############################################################
# marker_utils.py
############################################################

패키지:
    rviz_visualization

역할:
    RViz2에서 사용할 visualization_msgs/msg/Marker 메시지를 생성하는
    공통 utility 함수 모음.

이 파일에서 하는 것:
    - ColorRGBA 색상 객체 생성
    - CUBE marker 생성
    - SPHERE marker 생성
    - POINTS marker 생성
    - 위험도 값을 RViz 색상으로 변환

이 파일에서 하지 않는 것:
    - 장애물 판단
    - 위험도 계산
    - LiDAR clustering
    - YOLO 객체 인식
    - A* 경로계획
    - 전차 제어 명령 생성

설계 원칙:
    marker_utils.py는 "표시 도구"만 제공한다.
    판단 로직은 rviz_visualizer_node.py 또는 추후 perception/planning 패키지에서 수행한다.

RViz Marker 기본 개념:
    Marker는 RViz2 화면에 도형을 띄우기 위한 ROS 메시지이다.

    주요 필드:
        header.frame_id:
            어떤 좌표계 기준으로 marker를 표시할지 지정한다.
            이 프로젝트에서는 기본적으로 "tank_map"을 사용한다.

        ns:
            marker namespace.
            RViz에서 marker를 그룹처럼 구분하는 이름이다.
            예: "player_tank", "enemy_tank", "obstacles", "lidar_points"

        id:
            같은 namespace 안에서 marker를 구분하는 정수 ID.
            ns와 id 조합이 같으면 기존 marker가 업데이트된다.

        type:
            marker 모양.
            예: CUBE, SPHERE, POINTS, LINE_STRIP, ARROW 등

        action:
            ADD, DELETE 등.
            보통 표시/갱신에는 ADD를 사용한다.

        pose:
            marker의 위치와 자세.

        scale:
            marker 크기.

        color:
            marker 색상과 투명도.
"""


############################################################
# 1. 타입 힌트 import
############################################################

# Iterable:
#   for 문으로 순회할 수 있는 객체 타입.
#   예: list, tuple, generator
#
# Tuple:
#   고정 길이 tuple 타입 표기.
#   여기서는 (x, y, z) 형태의 3차원 좌표를 표현하는 데 사용.
from typing import Iterable, Tuple


############################################################
# 2. ROS2 메시지 import
############################################################

# Point:
#   geometry_msgs/msg/Point
#   x, y, z 좌표 하나를 담는 메시지.
#   POINTS marker나 LINE_STRIP marker의 점 목록에 사용된다.
from geometry_msgs.msg import Point

# ColorRGBA:
#   std_msgs/msg/ColorRGBA
#   RViz marker의 색상과 투명도를 표현한다.
#
#   r: red   0.0 ~ 1.0
#   g: green 0.0 ~ 1.0
#   b: blue  0.0 ~ 1.0
#   a: alpha 0.0 ~ 1.0
#
#   alpha가 0이면 완전 투명, 1이면 완전 불투명.
from std_msgs.msg import ColorRGBA

# Marker:
#   visualization_msgs/msg/Marker
#   RViz2에 도형을 표시하기 위한 핵심 메시지.
from visualization_msgs.msg import Marker


############################################################
# 3. 색상 유틸리티
############################################################

def make_color(r: float, g: float, b: float, a: float) -> ColorRGBA:
    """
    RViz Marker 색상 객체를 생성한다.

    Parameters:
        r:
            빨간색 성분. 0.0 ~ 1.0

        g:
            초록색 성분. 0.0 ~ 1.0

        b:
            파란색 성분. 0.0 ~ 1.0

        a:
            투명도. 0.0 ~ 1.0
            0.0 = 완전 투명
            1.0 = 완전 불투명

    Returns:
        std_msgs/msg/ColorRGBA

    사용 예:
        파란색 전차:
            make_color(0.0, 0.25, 1.0, 0.85)

        빨간색 적 전차:
            make_color(1.0, 0.05, 0.05, 0.85)

        반투명 위험 영역:
            make_color(1.0, 0.0, 0.0, 0.45)
    """

    # ColorRGBA 메시지 객체 생성
    color = ColorRGBA()

    # 각 색상 성분을 float로 변환해서 저장
    color.r = float(r)
    color.g = float(g)
    color.b = float(b)
    color.a = float(a)

    return color


############################################################
# 4. 큐브 marker
############################################################

def make_cube_marker(
    frame_id: str,
    namespace: str,
    marker_id: int,
    x: float,
    y: float,
    z: float,
    scale_x: float,
    scale_y: float,
    scale_z: float,
    color: ColorRGBA,
) -> Marker:
    """
    RViz2 CUBE marker를 생성한다.

    사용 목적:
        - 아군 전차 bounding box
        - 적 전차 bounding box
        - 장애물 영역
        - 위험도/복잡도 영역
        - 지형 cell 표시

    Parameters:
        frame_id:
            marker를 표시할 좌표계 이름.
            이 프로젝트에서는 보통 "tank_map".

        namespace:
            marker 그룹 이름.
            예: "player_tank", "enemy_tank", "obstacles"

        marker_id:
            namespace 안에서 marker를 구분하는 ID.
            같은 namespace와 id를 다시 publish하면 기존 marker가 갱신된다.

        x, y, z:
            marker 중심 위치.
            RViz map 좌표 기준.

        scale_x, scale_y, scale_z:
            marker 크기.
            CUBE의 가로/세로/높이.

        color:
            marker 색상.

    Returns:
        visualization_msgs/msg/Marker
    """

    # Marker 메시지 객체 생성
    marker = Marker()

    ########################################################
    # 4.1 좌표계 / 식별 정보
    ########################################################

    # RViz가 이 marker를 어떤 frame 기준으로 그릴지 결정한다
    marker.header.frame_id = frame_id

    # marker namespace
    marker.ns = namespace

    # marker ID
    marker.id = int(marker_id)

    ########################################################
    # 4.2 marker 형태 / 동작
    ########################################################

    # CUBE 형태로 표시
    marker.type = Marker.CUBE

    # ADD:
    #   marker를 새로 추가하거나,
    #   같은 ns/id가 이미 있으면 업데이트한다.
    marker.action = Marker.ADD

    ########################################################
    # 4.3 위치 / 자세
    ########################################################

    # marker 중심 위치
    marker.pose.position.x = float(x)
    marker.pose.position.y = float(y)
    marker.pose.position.z = float(z)

    # 회전 없음.
    # quaternion에서 w=1.0, x=y=z=0.0이면 identity rotation.
    marker.pose.orientation.w = 1.0

    ########################################################
    # 4.4 크기
    ########################################################

    marker.scale.x = float(scale_x)
    marker.scale.y = float(scale_y)
    marker.scale.z = float(scale_z)

    ########################################################
    # 4.5 색상
    ########################################################

    marker.color = color

    return marker


############################################################
# 5. 구(sphere) marker
############################################################

def make_sphere_marker(
    frame_id: str,
    namespace: str,
    marker_id: int,
    x: float,
    y: float,
    z: float,
    radius: float,
    color: ColorRGBA,
) -> Marker:
    """
    RViz2 SPHERE marker를 생성한다.

    사용 목적:
        - 목표 지점 표시
        - 관심 지점 표시
        - 탐지된 객체 중심점 표시
        - waypoint 표시

    주의:
        RViz Marker.SPHERE에서 scale.x/y/z는 실제 의미상
        지름처럼 동작한다.
        여기서는 코드 가독성을 위해 인자 이름을 radius로 두었지만,
        내부적으로는 scale.x/y/z에 같은 값을 넣는다.

    Parameters:
        frame_id:
            marker 좌표계 이름.

        namespace:
            marker 그룹 이름.

        marker_id:
            marker ID.

        x, y, z:
            sphere 중심 위치.

        radius:
            sphere 표시 크기.
            실제 RViz에서는 scale.x/y/z 값으로 사용된다.

        color:
            marker 색상.

    Returns:
        visualization_msgs/msg/Marker
    """

    marker = Marker()

    ########################################################
    # 5.1 좌표계 / 식별 정보
    ########################################################

    marker.header.frame_id = frame_id
    marker.ns = namespace
    marker.id = int(marker_id)

    ########################################################
    # 5.2 marker 형태 / 동작
    ########################################################

    marker.type = Marker.SPHERE
    marker.action = Marker.ADD

    ########################################################
    # 5.3 위치 / 자세
    ########################################################

    marker.pose.position.x = float(x)
    marker.pose.position.y = float(y)
    marker.pose.position.z = float(z)
    marker.pose.orientation.w = 1.0

    ########################################################
    # 5.4 크기
    ########################################################

    marker.scale.x = float(radius)
    marker.scale.y = float(radius)
    marker.scale.z = float(radius)

    ########################################################
    # 5.5 색상
    ########################################################

    marker.color = color

    return marker


############################################################
# 6. 점(points) marker
############################################################

def make_points_marker(
    frame_id: str,
    namespace: str,
    marker_id: int,
    points: Iterable[Tuple[float, float, float]],
    point_size: float,
    color: ColorRGBA,
) -> Marker:
    """
    RViz2 POINTS marker를 생성한다.

    사용 목적:
        - Tank Challenge 3D LiDAR point 표시
        - 탐색 중 관측된 지형/장애물 point 표시
        - LiDAR 기반 obstacle candidate 확인

    POINTS marker 특징:
        - 하나의 Marker 메시지 안에 여러 개의 Point를 담을 수 있다.
        - point마다 개별 marker를 만드는 것보다 훨씬 효율적이다.
        - scale.x, scale.y가 각 point의 표시 크기를 결정한다.
        - scale.z는 POINTS marker에서는 보통 사용하지 않는다.

    Parameters:
        frame_id:
            marker 좌표계 이름.
            보통 "tank_map".

        namespace:
            marker 그룹 이름.
            예: "lidar_points"

        marker_id:
            marker ID.

        points:
            (x, y, z) tuple들의 iterable.
            예:
                [
                    (10.0, 20.0, 0.5),
                    (10.5, 20.1, 0.7)
                ]

        point_size:
            RViz에서 각 point가 표시되는 크기.

        color:
            모든 point에 적용할 색상.

    Returns:
        visualization_msgs/msg/Marker
    """

    marker = Marker()

    ########################################################
    # 6.1 좌표계 / 식별 정보
    ########################################################

    marker.header.frame_id = frame_id
    marker.ns = namespace
    marker.id = int(marker_id)

    ########################################################
    # 6.2 marker 형태 / 동작
    ########################################################

    marker.type = Marker.POINTS
    marker.action = Marker.ADD

    ########################################################
    # 6.3 기본 자세
    ########################################################

    marker.pose.orientation.w = 1.0

    ########################################################
    # 6.4 point 표시 크기
    ########################################################

    # POINTS marker에서는 scale.x, scale.y가 각 점의 크기를 의미한다.
    marker.scale.x = float(point_size)
    marker.scale.y = float(point_size)

    ########################################################
    # 6.5 색상
    ########################################################

    marker.color = color

    ########################################################
    # 6.6 point 목록 채우기
    ########################################################

    for x, y, z in points:
        p = Point()
        p.x = float(x)
        p.y = float(y)
        p.z = float(z)
        marker.points.append(p)

    return marker


############################################################
# 7. 위험도 → 색상 매핑
############################################################

def risk_to_color(risk: float) -> ColorRGBA:
    """
    위험도 값을 RViz 색상으로 변환한다.

    입력:
        risk:
            0.0 ~ 1.0 범위의 위험도 값.

    색상 의미:
        risk = 0.0:
            낮은 위험.
            파란색 계열.

        risk = 0.5:
            중간 위험.
            보라색 계열.

        risk = 1.0:
            높은 위험.
            빨간색 계열.

    현재 변환 방식:
        red   = risk
        green = 0.15
        blue  = 1.0 - risk
        alpha = 0.45

    즉:
        risk가 커질수록 빨간색 성분이 커지고,
        파란색 성분은 줄어든다.

    사용 예:
        - 장애물 위험도 표시
        - 지형 복잡도 표시
        - LiDAR 기반 roughness 후보 표시
        - 카메라 기반 적/초소 탐지 위험 영역 표시

    주의:
        이 함수는 위험도를 계산하지 않는다.
        이미 계산된 risk 값을 색상으로 바꾸기만 한다.
    """

    ########################################################
    # 7.1 risk 값 범위 제한
    ########################################################

    # risk 값이 0보다 작거나 1보다 큰 경우가 들어와도
    # RViz 색상 계산이 깨지지 않도록 0.0~1.0으로 clamp한다.
    r = max(0.0, min(1.0, float(risk)))

    ########################################################
    # 7.2 낮음=파랑, 높음=빨강 색상 생성
    ########################################################

    return make_color(r, 0.15, 1.0 - r, 0.45)

############################################################
# 8. 선분 목록(line list) marker
############################################################

def make_line_list_marker(
    frame_id: str,
    namespace: str,
    marker_id: int,
    line_segments: Iterable[Tuple[Tuple[float, float, float], Tuple[float, float, float]]],
    line_width: float,
    color: ColorRGBA,
) -> Marker:
    """
    RViz2 LINE_LIST marker를 생성한다.

    사용 목적:
        - 시뮬레이터 Terrain 외곽선
        - 시뮬레이터 Terrain grid
        - 기준선 / 위험구역 외곽선 / 탐색 영역 표시

    LINE_LIST 특징:
        - marker.points에 두 점씩 넣으면 하나의 선분이 된다.
        - 예: [p0, p1, p2, p3]이면 p0-p1, p2-p3 선분이 표시된다.
    """

    marker = Marker()
    marker.header.frame_id = frame_id
    marker.ns = namespace
    marker.id = int(marker_id)
    marker.type = Marker.LINE_LIST
    marker.action = Marker.ADD

    marker.pose.orientation.w = 1.0
    marker.scale.x = float(line_width)
    marker.color = color

    for start, end in line_segments:
        p0 = Point()
        p0.x = float(start[0])
        p0.y = float(start[1])
        p0.z = float(start[2])

        p1 = Point()
        p1.x = float(end[0])
        p1.y = float(end[1])
        p1.z = float(end[2])

        marker.points.append(p0)
        marker.points.append(p1)

    return marker

############################################################
# 9. 시작점/끝점 기준 화살표(arrow) marker
############################################################

def make_arrow_marker(
    frame_id: str,
    namespace: str,
    marker_id: int,
    start_x: float,
    start_y: float,
    start_z: float,
    end_x: float,
    end_y: float,
    end_z: float,
    shaft_diameter: float,
    head_diameter: float,
    head_length: float,
    color: ColorRGBA,
) -> Marker:
    """
    RViz2 ARROW marker를 시작점/끝점 기준으로 생성한다.

    이 방식은 quaternion orientation을 직접 계산하지 않아도 되므로,
    heading 방향 시각화에 가장 직관적이다.

    start:
        arrow 시작점, 보통 전차 중심

    end:
        arrow 끝점, 전차 heading 방향으로 일정 거리 떨어진 점
    """

    marker = Marker()
    marker.header.frame_id = frame_id
    marker.ns = namespace
    marker.id = int(marker_id)
    marker.type = Marker.ARROW
    marker.action = Marker.ADD

    # ARROW marker에서 points를 2개 넣으면
    # 첫 번째 점 → 두 번째 점 방향으로 화살표가 그려진다.
    p0 = Point()
    p0.x = float(start_x)
    p0.y = float(start_y)
    p0.z = float(start_z)

    p1 = Point()
    p1.x = float(end_x)
    p1.y = float(end_y)
    p1.z = float(end_z)

    marker.points.append(p0)
    marker.points.append(p1)

    # ARROW marker scale 의미:
    #   scale.x = 화살대 지름(shaft diameter)
    #   scale.y = 화살촉 지름(head diameter)
    #   scale.z = 화살촉 길이(head length)
    marker.scale.x = float(shaft_diameter)
    marker.scale.y = float(head_diameter)
    marker.scale.z = float(head_length)

    marker.color = color

    return marker