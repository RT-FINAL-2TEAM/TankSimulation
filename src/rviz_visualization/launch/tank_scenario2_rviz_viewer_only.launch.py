# -*- coding: utf-8 -*-
"""RViz2 viewer only for scenario 2 map view.

Use this on a second PC or second terminal when the scenario2 runner is already
running tank_scenario2_map_view.launch.py with use_rviz:=false/true. This avoids
starting duplicate static/terrain visualization publishers.
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
