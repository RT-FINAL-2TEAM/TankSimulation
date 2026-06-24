# -*- coding: utf-8 -*-
"""주행 중 LiDAR를 기록하고, 주행 종료 후 service 호출로 최종 지형 맵을 생성하는 launch."""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            # 지형 노드 단일출처화: rviz copy 삭제 → ground_division 노드로 통합(2026-06-18).
            package="ground_division",
            executable="terrain_record_finalize_node",
            name="terrain_record_finalize_node",
            output="screen",
            parameters=[{
                # lidar_processor_node가 이미 분리한 지면/장애물 결과를 그대로 사용한다.
                # legacy all_detected 재분리 모드는 언덕 지면을 non_ground로 잘라낼 수 있어서 끈다.
                "use_preclassified_lidar": True,
                "terrain_input_topic": "/tank/sensor/lidar/terrain_points_map",
                "obstacle_input_topic": "/tank/sensor/lidar/detected_points_map",
                "input_topic": "/tank/sensor/lidar/all_detected_points_map",
                "map_frame": "tank_map",
                "voxel_size": 0.35,
                "use_csf": False,
                "fallback_ground_percentile": 70.0,
                "fallback_ground_margin": 0.70,
                "publish_period_sec": 0.5,
                "save_dir": "~/tankcc/tank_terrain_maps",
                "save_filename": "terrain_map_latest.npz",
                "save_csv": False,
                "save_legacy_split_files": False,
                "load_saved_map_on_start": False,
                "recording_enabled_on_start": True,
                "auto_finalize_after_idle_sec": 0.0,
                "grid_cell_size": 0.8,
                "max_elevation_cells": 30000,
                "attitude_correction_enabled": True,
                "attitude_correction_source": "lidar_then_body",
                "attitude_reference_mode": "first_frame",
                "attitude_use_origin_z_delta": False,
                "attitude_max_abs_delta_deg": 30.0,
                "min_map_z": -10.0,
                "max_map_z": 100.0,
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
        Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2_terrain_final_map",
            output="screen",
        ),
    ])
