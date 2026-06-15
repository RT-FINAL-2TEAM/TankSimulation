# -*- coding: utf-8 -*-
"""
Launch RViz with only the drone reconnaissance static map layer.

Use this when the presentation/story should show only what the tank knows before
mission execution:
- recon_map.map -> /tank/rviz/recon_map_markers
- recon_map.map -> /tank/map/recon/occupancy_grid
- recon_map.map -> /tank/map/recon/risk_grid

Live sensor fusion is not required for this launch.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory("rviz_visualization")
    rviz_config = os.path.join(pkg_share, "rviz", "tank_recon_map.rviz")
    config_file = os.path.join(pkg_share, "config", "static_map_costs.yaml")
    recon_map_file = os.path.join(pkg_share, "map", "finalmap.map")

    return LaunchDescription(
        [
            Node(
                package="rviz_visualization",
                executable="static_map_loader_node",
                name="tank_static_recon_map_loader_node",
                output="screen",
                parameters=[
                    {
                        "mode": "recon_only",
                        "config_file": config_file,
                        "recon_map_file": recon_map_file,
                        "publish_mission": False,
                        "publish_diff": False,
                        "publish_grids": True,
                        "publish_period_sec": 1.0,
                    }
                ],
            ),
            Node(
                package="rviz_visualization",
                executable="rviz_visualizer_node",
                name="tank_rviz_visualizer_node",
                output="screen",
            ),
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

            ExecuteProcess(cmd=["rviz2", "-d", rviz_config], output="screen"),
        ]
    )
