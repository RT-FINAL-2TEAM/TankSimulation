# -*- coding: utf-8 -*-
"""주행 중 LiDAR를 기록하고, 주행 종료 후 service 호출로 최종 지형 맵을 생성하는 launch."""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package="rviz_visualization",
            executable="terrain_record_finalize_node",
            name="terrain_record_finalize_node",
            output="screen",
            parameters=[{
                "input_topic": "/tank/sensor/lidar/all_detected_points_map",
                "map_frame": "tank_map",
                "voxel_size": 0.35,
                "use_csf": True,
                "fallback_ground_percentile": 70.0,
                "fallback_ground_margin": 0.70,
                "publish_period_sec": 0.5,
                "save_dir": "~/tank_terrain_maps",
                "save_csv": False,
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
                "wireframe_max_height_gap": 1.5,
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
