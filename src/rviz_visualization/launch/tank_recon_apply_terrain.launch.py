# -*- coding: utf-8 -*-
"""
수집 완료 terrain 적용/시각화용 launch.

역할:
- 기존 finalmap.map 정적 맵을 표시한다.
- ~/tankcc/tank_terrain_maps/terrain_map_latest.npz를 로드한다.
- 로드한 ground/non_ground/elevation/wireframe을 RViz topic으로 계속 발행한다.
- 이 launch에서는 기본적으로 새 LiDAR 기록을 하지 않는다.

실행 전 수집 파일이 있어야 한다:
    ~/tankcc/tank_terrain_maps/terrain_map_latest.npz
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

    terrain_save_dir = "~/tankcc/tank_terrain_maps"
    terrain_save_filename = "terrain_map_latest.npz"

    return LaunchDescription([
        # 1) 기본/기존 정적 map layer.
        Node(
            package="rviz_visualization",
            executable="static_map_loader_node",
            name="tank_static_map_loader_node",
            output="screen",
            parameters=[{
                "mode": "recon_only",
                "config_file": config_file,
                "recon_map_file": finalmap_file,
                "publish_mission": False,
                "publish_diff": False,
                "publish_grids": True,
                "publish_period_sec": 1.0,
            }],
        ),
        # 2) 현재 전차 pose, bridge topic marker도 같이 보고 싶을 때 사용된다.
        Node(
            package="rviz_visualization",
            executable="rviz_visualizer_node",
            name="tank_rviz_visualizer_node",
            output="screen",
        ),
        # 3) 저장된 단일 terrain map 파일을 로드해 RViz에 계속 publish한다.
        Node(
            package="ground_division",
            executable="terrain_record_finalize_node",
            name="terrain_saved_map_visualizer_node",
            output="screen",
            parameters=[{
                "use_preclassified_lidar": True,
                "map_frame": "tank_map",
                "publish_period_sec": 0.5,
                "save_dir": terrain_save_dir,
                "save_filename": terrain_save_filename,
                "save_csv": False,
                "save_legacy_split_files": False,
                "load_saved_map_on_start": True,
                "recording_enabled_on_start": False,
                "grid_cell_size": 0.8,
                "max_elevation_cells": 30000,
                "marker_alpha": 0.75,
                "marker_z_thickness": 0.08,
                "wireframe_enabled": True,
                "wireframe_line_width": 0.04,
                "wireframe_max_height_gap": 3.0,
                "wireframe_connect_diagonal": False,
            }],
        ),
        ExecuteProcess(
            cmd=["rviz2", "-d", rviz_config],
            output="screen",
        ),
    ])
