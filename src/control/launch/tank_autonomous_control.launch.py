# -*- coding: utf-8 -*-
"""
Latest TankSimulation -> ROS2 autonomous stack.

Runs:
- path_planning/map_astar_planner_node      : TankSimulation bbox A* + LiDAR dynamic replanning
- potential/potential_field_node: TankSimulation APF with tangential repulsion + threat repulsion
- control/tank_controller_node      : TankSimulation W/A/S/D rule controller + stuck escape

Prerequisite:
- TANK_MODE=auto ros2 run ros_bridge ros_bridge
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import SetEnvironmentVariable
from launch_ros.actions import Node


def generate_launch_description():
    gui_share = get_package_share_directory("rviz_visualization")
    controller_share = get_package_share_directory("control")
    recon_map_file = os.path.join(gui_share, "map", "recon_map.map")
    tank_param_file = os.path.join(controller_share, "config", "tank_parameters.yaml")
    path_planning_share = get_package_share_directory("path_planning")
    potential_share = get_package_share_directory("potential")
    route_config_file = os.path.join(path_planning_share, "config", "routes.yaml")
    apf_weights_file = os.path.join(potential_share, "config", "apf_weight_profiles.yaml")

    return LaunchDescription([
        SetEnvironmentVariable("TANK_START_CONTROL", "start"),
        SetEnvironmentVariable("TANK_APF_PASSTHROUGH_WHEN_CLEAR", "true"),

        Node(
            package="lidar",
            executable="lidar_processor_node",
            name="tank_lidar_processor_node",
            output="screen",
        ),
        # Team visual perception integration:
        # - /detect image from ros_bridge + /info LiDAR raw -> camera LiDAR projection overlay
        # - /tank/sensor/lidar/detected_points_map -> DBSCAN cluster markers/status
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
        Node(
            package="path_planning",
            executable="map_astar_planner_node",
            name="tank_team_dynamic_astar_planner_node",
            output="screen",
            parameters=[{
                "map_width": 300,
                "map_height": 300,
                "resolution": 1.0,
                "inflate": 5.0,
                "use_path_smoothing": True,
                # False = latest TankSimulation step3 policy: do not cheat with GT obstacles.
                # True  = use /update_obstacle bbox list for static A* validation.
                "use_gt_obstacles": False,
                "enable_dynamic_replan": False,
                "enable_periodic_replan": False,
                # TankSimulation route A/B strategy is now active in the ROS2 planner.
                "use_route_waypoints": True,
                "route_config_file": route_config_file,
                "route_map_name": "recon_map",
                "route_id": "B",
                "route_side": "east",
                "route_clearance_weight": 0.4,
                # DBSCAN cluster bboxes are used when dynamic replanning is enabled.
                "use_lidar_cluster_bboxes": True,
                "lidar_cluster_bbox_margin": 1.0,
                "replan_period_sec": 0.0,
                "dynamic_replan_cooldown_sec": 8.0,
                "plan_retry_period_sec": 3.0,
                "path_block_margin": 5.0,
                "lidar_cluster_eps": 2.0,
                "lidar_cluster_min_samples": 3,
                "lidar_history_resolution": 0.5,
                "max_lidar_history_points": 1500,
                "lookahead_distance": 8.0,
                "publish_path_period_sec": 5.0,
                "goal_tolerance": 10.0,
                "default_goal_enabled": True,
                # Latest TankSimulation DESTINATION=(120,250)
                "default_goal_x": 120.0,
                "default_goal_y": 250.0,
            }],
        ),

        Node(
            package="path_planning",
            executable="local_path_node",
            name="tank_local_path_node",
            output="screen",
            parameters=[{
                "config_file": os.path.join(path_planning_share, "config", "fusion_mapping.yaml"),
            }],
        ),
        Node(
            package="potential",
            executable="potential_field_node",
            name="tank_team_potential_field_node",
            output="screen",
            parameters=[{
                "target_pose_topic": "/tank/path/lookahead_pose",
                "fallback_goal_topic": "/tank/goal/pose",
                "hz": 10.0,
                "influence_radius": 12.0,
                "k_att": 1.0,
                "k_rep": 160.0,
                "tangent_gain_scale": 1.8,
                "local_target_distance": 6.0,
                "max_repulsive_norm": 20.0,
                "max_result_norm": 20.0,
                "min_obstacle_distance": 1.5,
                "max_obstacle_distance": 12.0,
                "front_sector_deg": 140.0,
                "path_corridor_width": 7.0,
                "obstacle_voxel_resolution": 1.0,
                "max_obstacle_points": 300,
                "use_discovered_objects": True,
                "passthrough_when_clear": True,
                "repulsive_eps": 0.05,
                "use_threat_avoidance": True,
                "threat_map_file": recon_map_file,
                "threat_radius": 25.0,
                "k_threat_rep": 2000.0,
                "use_lidar_clusters": True,
                "cluster_obstacle_min_count": 2,
                "apf_weights_file": apf_weights_file,
                "apf_weight_profile": "default",
                "marker_scale": 2.5,
            }],
        ),
        Node(
            package="control",
            executable="tank_controller_node",
            name="tank_team_path_controller_node",
            output="screen",
            parameters=[{
                "tank_param_file": tank_param_file,
                "controller_hz": 10.0,
                "enable_local_target": True,
                "target_ttl_sec": 2.0,
                "goal_tolerance": 10.0,
                "heading_deadband_deg": 5.0,
                "steering_full_error_deg": 45.0,
                "min_ad_weight": 0.0,
                "max_ad_weight": 1.0,
                "straight_ws_weight": 1.0,
                "turn_ws_weight": 0.4,
                "rotate_in_place_angle_deg": 60.0,
                "slowdown_angle_deg": 30.0,
                "stop_distance": 10.0,
                "enable_stuck_escape": True,
                "stuck_check_period": 5.0,
                "stuck_min_movement": 1.5,
                "escape_reverse_sec": 1.5,
                "escape_turn_sec": 1.5,
            }],
        ),
    ])
