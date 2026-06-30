# -*- coding: utf-8 -*-
"""RViz 없이 web에서 시나리오2 정찰 맵 + 저장 지형 메쉬를 보는 launch.

RViz의 tank_scenario2_map_view.launch.py와 같은 데이터 소스를 사용한다.
- scenario2_map.map -> /tank/rviz/recon_map_markers
- scenario2_terrain.npz -> /tank/terrain/final_elevation_markers, /tank/terrain/final_wireframe_markers
- rviz_web_server -> http://127.0.0.1:5055/rviz3d?frame=tank_map&cloud=off&rays=0&vectors=0
"""
import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _project_root() -> str:
    env = os.environ.get("TANK_PROJECT_ROOT")
    if env:
        return env
    # source tree: src/rviz_web/launch/file.py -> project root parents[3]
    return str(Path(__file__).resolve().parents[3])


def generate_launch_description():
    root = _project_root()
    rviz_pkg_share = get_package_share_directory("rviz_visualization")
    config_file = os.path.join(rviz_pkg_share, "config", "static_map_costs.yaml")
    default_map = os.path.join(root, "recon_reports", "recon_map", "scenario2_map.map")
    default_terrain = os.path.join(root, "recon_reports", "recon_map", "scenario2_terrain.npz")

    return LaunchDescription([
        DeclareLaunchArgument("web_port", default_value="5055"),
        DeclareLaunchArgument("rosbridge_port", default_value="9090"),
        DeclareLaunchArgument("start_rosbridge", default_value="true"),
        DeclareLaunchArgument("scenario2_map", default_value=default_map),
        DeclareLaunchArgument("terrain_npz", default_value=default_terrain),
        DeclareLaunchArgument("mesh_grid_cell", default_value="0.8"),
        DeclareLaunchArgument("mesh_max_cells", default_value="60000"),

        # rosbridge websocket: browser -> ROS2 topic bridge
        ExecuteProcess(
            condition=IfCondition(LaunchConfiguration("start_rosbridge")),
            cmd=[
                "ros2", "launch", "rosbridge_server", "rosbridge_websocket_launch.xml",
                ["port:=", LaunchConfiguration("rosbridge_port")],
            ],
            output="screen",
        ),

        # scenario2_map.map: finalmap + discovered objects
        Node(
            package="rviz_visualization",
            executable="static_map_loader_node",
            name="tank_static_map_loader_node_web",
            output="screen",
            parameters=[{
                "mode": "recon_only",
                "config_file": config_file,
                "recon_map_file": LaunchConfiguration("scenario2_map"),
                "publish_mission": False,
                "publish_diff": False,
                "publish_grids": True,
                "publish_period_sec": 1.0,
            }],
        ),

        # pose / dynamic RViz marker publisher
        Node(
            package="rviz_visualization",
            executable="rviz_visualizer_node",
            name="tank_rviz_visualizer_node_web",
            output="screen",
        ),

        # saved terrain npz -> final_elevation_markers TRIANGLE_LIST mesh
        Node(
            package="ground_division",
            executable="terrain_record_finalize_node",
            name="terrain_saved_map_visualizer_node_web",
            output="screen",
            parameters=[{
                "use_preclassified_lidar": True,
                "map_frame": "tank_map",
                "publish_period_sec": 0.5,
                "saved_map_file": LaunchConfiguration("terrain_npz"),
                "load_saved_map_on_start": True,
                "recording_enabled_on_start": False,
                "surface_mesh_enabled": True,
                "grid_cell_size": LaunchConfiguration("mesh_grid_cell"),
                "max_elevation_cells": LaunchConfiguration("mesh_max_cells"),
                "marker_alpha": 0.75,
                "wireframe_enabled": False,
                "wireframe_max_height_gap": 3.0,
            }],
        ),

        # standalone web server. Run through `ros2 run` so the console-script executable
        # is resolved from the sourced ROS2 install space. Do not use launch_ros Node
        # because this is a Flask process, not an rclpy node.
        ExecuteProcess(
            cmd=[
                "ros2", "run", "rviz_web", "rviz_web_server",
                "--host", "0.0.0.0",
                "--port", LaunchConfiguration("web_port"),
                "--rosbridge-port", LaunchConfiguration("rosbridge_port"),
            ],
            output="screen",
        ),
    ])
