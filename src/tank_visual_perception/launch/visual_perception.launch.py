# -*- coding: utf-8 -*-
"""Launch team LiDAR-camera overlay and LiDAR DBSCAN clustering nodes."""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package="tank_visual_perception",
            executable="lidar_camera_overlay_node",
            name="lidar_camera_overlay_node",
            output="screen",
        ),
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
    ])
