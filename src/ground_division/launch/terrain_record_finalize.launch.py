# -*- coding: utf-8 -*-
"""Launch terrain-map recording/finalization node from ground_division."""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
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
                "auto_finalize_after_idle_sec": 0.0,
                "wireframe_max_height_gap": 3.0,
                "load_saved_map_on_start": False,
                "recording_enabled_on_start": True,
                "save_dir": "~/tankcc/tank_terrain_maps",
                "save_filename": "terrain_map_latest.npz",
                "save_csv": False,
                "save_legacy_split_files": False,
            }],
        ),
    ])
