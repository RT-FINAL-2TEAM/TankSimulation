# -*- coding: utf-8 -*-
"""
############################################################
# tank_rviz.launch.py
############################################################

역할:
- rviz_visualization 패키지의 RViz2 시각화 노드를 실행한다.
- 동시에 RViz2 프로그램을 실행하고, 미리 저장해둔 설정 파일을 불러온다.

실행:
    ros2 launch rviz_visualization tank_rviz.launch.py

전체 실행 흐름:
    1. ros_bridge / autonomous stack이 Tank Challenge 데이터를 ROS2 topic으로 publish
    2. rviz_visualizer_node가 해당 topic들을 subscribe
    3. rviz_visualizer_node가 RViz2용 MarkerArray topic을 publish
    4. RViz2가 MarkerArray/Image topic을 화면에 표시

주의:
- 이 launch 파일은 Gazebo를 실행하지 않는다.
- 물리 시뮬레이션은 Tank Challenge 시뮬레이터가 담당한다.
- RViz2는 ROS2 topic을 시각화하는 viewer 역할만 한다.
"""

############################################################
# 1. Python 기본 모듈 import
############################################################

# os.path.join()을 사용해 패키지 내부 rviz 설정 파일 경로를 만들기 위해 사용
import os


############################################################
# 2. ROS2 launch 관련 import
############################################################

# 설치된 ROS2 패키지의 share 디렉터리 경로를 찾기 위한 함수
# 예:
#   get_package_share_directory("rviz_visualization")
#   → ~/.../install/rviz_visualization/share/rviz_visualization
from ament_index_python.packages import get_package_share_directory

# LaunchDescription:
# - ros2 launch가 실행할 action 목록을 담는 객체
from launch import LaunchDescription

# ExecuteProcess:
# - rviz2 같은 일반 프로세스를 실행할 때 사용
from launch.actions import ExecuteProcess

# Node:
# - ROS2 node를 launch 파일에서 실행할 때 사용
from launch_ros.actions import Node


############################################################
# 3. launch 진입 함수
############################################################

def generate_launch_description():
    """
    ros2 launch가 호출하는 표준 함수.

    ROS2 launch 시스템은 이 함수가 반환하는 LaunchDescription을 읽어서
    어떤 node와 process를 실행할지 결정한다.
    """

    ############################################################
    # 3.1 rviz_visualization 패키지 share 경로 찾기
    ############################################################

    # colcon build 후 설치된 패키지의 share 경로를 가져온다.
    #
    # 예시 경로:
    #   ~/HyundaiRotem_Bootcamp/project/final_project/ros2_ws/install/
    #     rviz_visualization/share/rviz_visualization
    #
    # 이 경로 안에 launch/, rviz/, package.xml 등이 설치된다.
    pkg_share = get_package_share_directory("rviz_visualization")

    ############################################################
    # 3.2 RViz2 설정 파일 경로 생성
    ############################################################

    # setup.py의 data_files 설정에 의해
    # rviz/tank_debug.rviz 파일은 install/share/rviz_visualization/rviz/ 아래에 설치된다.
    #
    # RViz2 실행 시 -d 옵션으로 이 파일을 넘기면,
    # Fixed Frame, MarkerArray Display, Grid Display 등이 미리 설정된 상태로 열린다.
    rviz_config = os.path.join(pkg_share, "rviz", "tank_debug.rviz")

    ############################################################
    # 3.3 실행할 node/process 목록 반환
    ############################################################

    return LaunchDescription(
        [
            ####################################################
            # A. RViz Marker 변환 노드 실행
            ####################################################
            #
            # 이 node는 ros_bridge에서 나온 topic을 직접 RViz2에 표시하지 않고,
            # RViz2가 보기 좋은 MarkerArray 형태로 변환한다.
            #
            # 입력 예:
            #   /tank/player/pose
            #   /tank/enemy/pose
            #   /tank/goal/pose
            #   /tank/map/obstacles
            #   /tank/sensor/lidar/points
            #
            # 출력 예:
            #   /tank/rviz/object_markers
            #   /tank/rviz/obstacle_markers
            #   /tank/rviz/lidar_markers
            #   /tank/rviz/risk_markers
            Node(
                package="rviz_visualization",
                executable="rviz_visualizer_node",
                name="tank_rviz_visualizer_node",
                output="screen",
            ),

            ####################################################
            # B. RViz2 실행
            ####################################################
            #
            # rviz2:
            #   ROS2 topic을 시각적으로 확인하는 GUI 프로그램
            #
            # -d 옵션:
            #   지정한 .rviz 설정 파일을 불러온다.
            #
            # 이 설정 파일에는 다음과 같은 Display가 포함된다.
            #   - Grid
            #   - Tank / Goal MarkerArray
            #   - Obstacle MarkerArray
            #   - LiDAR MarkerArray
            #   - Risk / Complexity MarkerArray
            Node(
                package="rviz_visualization",
                executable="terrain_record_finalize_node",
                name="terrain_record_finalize_node",
                output="screen",
                parameters=[
                    {
                        "input_topic": "/tank/sensor/lidar/all_detected_points_map",
                        "map_frame": "tank_map",
                        "auto_finalize_after_idle_sec": 0.0,
                    }
                ],
            ),

            ExecuteProcess(
                cmd=["rviz2", "-d", rviz_config],
                output="screen",
            ),
        ]
    )