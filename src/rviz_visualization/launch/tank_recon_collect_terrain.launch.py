# -*- coding: utf-8 -*-
"""
정찰/수집용 launch.

역할:
- 기존 finalmap.map 기준 RViz를 띄운다.
- LiDAR processor가 /terrain_points_map, /detected_points_map을 만든다.
- terrain_record_finalize_node가 주행 중 지형/장애물 point를 누적한다.
- 주행 종료 후 아래 서비스를 호출하면 단일 파일 하나로 저장된다.

    ros2 service call /tank/terrain/finalize_map std_srvs/srv/Trigger "{}"

저장 파일:
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
        # 1) 기존 정적 맵. 아직 저장된 terrain_map_latest.npz를 적용하지 않는 수집 화면이다.
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
        # 2) simulator/bridge topic을 RViz marker로 변환한다.
        Node(
            package="rviz_visualization",
            executable="rviz_visualizer_node",
            name="tank_rviz_visualizer_node",
            output="screen",
        ),
        # 3) raw LiDAR를 map 좌표계 terrain/obstacle PointCloud2로 분리한다.
        Node(
            package="lidar",
            executable="lidar_processor_node",
            name="tank_lidar_processor_node",
            output="screen",
            parameters=[{
                # String JSON legacy 토픽은 대역폭/CPU 낭비라 수집 launch에서는 끈다.
                "publish_legacy_lidar_json": False,
            }],
        ),
        # 4) 장애물 cluster marker 확인용. 저장 파일 자체는 terrain_record_finalize_node가 담당한다.
        Node(
            package="tank_visual_perception",
            executable="lidar_dbscan_cluster_node",
            name="lidar_dbscan_cluster_node",
            output="screen",
            parameters=[{
                "eps": 1.5,
                "min_samples": 2,
                "min_cluster_size": 2,
            }],
        ),
        # 5) 주행 중 누적하고 finalize 시 단일 NPZ 파일 하나로 저장한다.
        Node(
            package="ground_division",
            executable="terrain_record_finalize_node",
            name="terrain_record_finalize_node",
            output="screen",
            parameters=[{
                "use_preclassified_lidar": True,
                "terrain_input_topic": "/tank/sensor/lidar/terrain_points_map",
                "obstacle_input_topic": "/tank/sensor/lidar/detected_points_map",
                "input_topic": "/tank/sensor/lidar/all_detected_points_map",
                "map_frame": "tank_map",
                "voxel_size": 0.35,
                "use_csf": False,
                "publish_period_sec": 0.5,
                "save_dir": terrain_save_dir,
                "save_filename": terrain_save_filename,
                "saved_map_file": "",
                "save_csv": False,
                "save_legacy_split_files": False,
                "load_saved_map_on_start": False,
                "recording_enabled_on_start": True,
                "auto_finalize_after_idle_sec": 0.0,
                "grid_cell_size": 0.8,
                "max_elevation_cells": 30000,
                "terrain_prefilter_enabled": True,
                "terrain_cell_size": 0.7,
                "terrain_low_percentile": 70.0,
                "terrain_height_margin": 10.0,
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
