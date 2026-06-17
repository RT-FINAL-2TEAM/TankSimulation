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
                "input_topic": "/tank/sensor/lidar/all_detected_points_map",
                "map_frame": "tank_map",
                "auto_finalize_after_idle_sec": 0.0,
            }],
        ),
    ])
