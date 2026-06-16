# -*- coding: utf-8 -*-
"""
############################################################
# tank_rviz.launch.py  (정찰/자율 표준 RViz)
############################################################

역할:
- 정찰/자율주행을 모니터링하는 RViz2 시각화를 한 번에 띄운다.
- finalmap 정적맵(static_map_loader_node) + 마커 변환(rviz_visualizer_node)
  + 누적 지형 마커(terrain_record_finalize_node) + RViz2(tank_finalmap.rviz)를 실행한다.

실행:
    ros2 launch rviz_visualization tank_rviz.launch.py

주의:
- 이 launch는 시뮬레이션 물리를 실행하지 않는다(Tank Challenge 시뮬레이터가 담당).
- RViz2는 ROS2 topic을 시각화하는 viewer 역할만 한다.
- finalmap.map을 정적맵으로 로드해 표시하므로, 자율 스택 없이 bridge만 있어도 맵이 보인다.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory("rviz_visualization")
    rviz_config = os.path.join(pkg_share, "rviz", "tank_finalmap.rviz")
    config_file = os.path.join(pkg_share, "config", "static_map_costs.yaml")
    finalmap_file = os.path.join(pkg_share, "map", "finalmap.map")

    return LaunchDescription(
        [
            # A. finalmap 정적맵 로더 — /tank/rviz/recon_map_markers + occupancy/risk grid 발행
            Node(
                package="rviz_visualization",
                executable="static_map_loader_node",
                name="tank_static_map_loader_node",
                output="screen",
                parameters=[
                    {
                        "mode": "recon_only",
                        "config_file": config_file,
                        "recon_map_file": finalmap_file,
                        "publish_mission": False,
                        "publish_diff": False,
                        "publish_grids": True,
                        "publish_period_sec": 1.0,
                    }
                ],
            ),
            # B. ros_bridge topic → RViz MarkerArray 변환 노드
            Node(
                package="rviz_visualization",
                executable="rviz_visualizer_node",
                name="tank_rviz_visualizer_node",
                output="screen",
            ),
            # C. 누적 지형(Final Terrain) 마커 발행 노드
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
            # D. RViz2 실행 (finalmap 정적맵 + 라이브 인지 표시)
            ExecuteProcess(
                cmd=["rviz2", "-d", rviz_config],
                output="screen",
            ),
        ]
    )
