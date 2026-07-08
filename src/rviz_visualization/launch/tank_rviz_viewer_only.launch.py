# -*- coding: utf-8 -*-
"""RViz2 viewer only for scenario 1/recon.

Use this on a second PC or second terminal when the scenario runner is already
running the visualization backend. This launch intentionally does not start
static_map_loader_node, rviz_visualizer_node, or terrain_record_finalize_node,
so marker publishers are not duplicated.
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess


def generate_launch_description():
    pkg_share = get_package_share_directory("rviz_visualization")
    rviz_config = os.path.join(pkg_share, "rviz", "tank_finalmap.rviz")
    return LaunchDescription([
        ExecuteProcess(cmd=["rviz2", "-d", rviz_config], output="screen"),
    ])
