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
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory("rviz_visualization")
    rviz_config = os.path.join(pkg_share, "rviz", "tank_finalmap.rviz")
    config_file = os.path.join(pkg_share, "config", "static_map_costs.yaml")
    finalmap_file = os.path.join(pkg_share, "map", "finalmap.map")

    return LaunchDescription(
        [
            # use_rviz:=false 면 데스크톱 RViz2 창은 생략하고 마커 발행 노드만 실행한다.
            # → 웹 RViz 3D(/rviz3d)가 동일한 마커(정적맵/객체/지형)를 그대로 받는다.
            #   GPU-less에서 무거운 소프트웨어 렌더링 RViz2 창을 끄고도 웹 시각화는 유지된다.
            DeclareLaunchArgument("use_rviz", default_value="true"),
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
            #    지형 노드 단일출처화: rviz copy 삭제 → ground_division 노드로 통합(2026-06-18).
            Node(
                package="ground_division",
                executable="terrain_record_finalize_node",
                name="terrain_record_finalize_node",
                output="screen",
                parameters=[
                    {
                        # lidar_processor_node가 이미 분리한 지면/장애물 결과를 그대로 사용한다.
                        # 언덕 지면을 all_detected_points_map에서 다시 z-filter로 잘라내지 않도록 한다.
                        "use_preclassified_lidar": True,
                        "terrain_input_topic": "/tank/sensor/lidar/terrain_points_map",
                        "obstacle_input_topic": "/tank/sensor/lidar/detected_points_map",
                        "input_topic": "/tank/sensor/lidar/all_detected_points_map",
                        "map_frame": "tank_map",
                        "auto_finalize_after_idle_sec": 0.0,
                        "wireframe_max_height_gap": 3.0,
                        "save_dir": "~/tankcc/tank_terrain_maps",
                        "save_filename": "terrain_map_latest.npz",
                        "save_csv": False,
                        "save_legacy_split_files": False,
                        # 정찰 뷰는 깨끗하게: 시작 시 이전 정찰 지형을 불러오지 않고 주행하며 build한다
                        # ("정찰=모르는 정보" 원칙 일치). 저장된 지형 면을 보려면 tank_recon_apply_terrain.launch.py 사용.
                        "load_saved_map_on_start": False,
                        "recording_enabled_on_start": True,
                    }
                ],
            ),
            # D. RViz2 실행 (finalmap 정적맵 + 라이브 인지 표시) — use_rviz:=false 면 생략(마커 노드만).
            ExecuteProcess(
                cmd=["rviz2", "-d", rviz_config],
                output="screen",
                condition=IfCondition(LaunchConfiguration("use_rviz")),
            ),
        ]
    )
