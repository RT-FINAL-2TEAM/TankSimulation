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
from launch.actions import SetEnvironmentVariable, DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    gui_share = get_package_share_directory("rviz_visualization")
    controller_share = get_package_share_directory("control")
    recon_map_file = os.path.join(gui_share, "map", "final_v4.map")
    tank_param_file = os.path.join(controller_share, "config", "tank_parameters.yaml")
    path_planning_share = get_package_share_directory("path_planning")
    potential_share = get_package_share_directory("potential")
    route_config_file = os.path.join(path_planning_share, "config", "routes.yaml")
    apf_weights_file = os.path.join(potential_share, "config", "apf_weight_profiles.yaml")

    mission_type_arg = DeclareLaunchArgument(
        'mission_type',
        default_value='mission',
        description='Mission scenario: recon, mission, return'
    )
    
    route_id_arg = DeclareLaunchArgument(
        'route_id',
        default_value='A',
        description='Route to use: A, B'
    )
    
    route_side_arg = DeclareLaunchArgument(
        'route_side',
        default_value='west',
        description='Side bias for A*: west (for A), east (for B)'
    )

    return LaunchDescription([
        mission_type_arg,
        route_id_arg,
        route_side_arg,
        SetEnvironmentVariable("TANK_START_CONTROL", "start"),
        SetEnvironmentVariable("TANK_APF_PASSTHROUGH_WHEN_CLEAR", "true"),

        Node(
            package="lidar",
            executable="lidar_processor_node",
            name="tank_lidar_processor_node",
            output="screen",
            parameters=[{"publish_legacy_lidar_json": True}],
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
                "route_map_name": "finalmap",
                "route_id": LaunchConfiguration("route_id"),
                "route_side": LaunchConfiguration("route_side"),
                "route_clearance_weight": 0.4,
                # 정적 맵(finalmap.map) 나무/바위를 전역 A* 코스트맵에 반영 → 루트가 나무 관통 안 하고
                # 코리더 중앙으로. use_gt_obstacles와 독립(빈 static_map_file이면 finalmap.map 자동 해석).
                "use_static_map": True,
                "static_map_file": recon_map_file,
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
                # 주행 목적지 = routes.yaml destination 단일 출처로 통일 (110.0, 276.5).
                # 적전차 리스폰(135.46, 276.87)과는 별개 — 전차는 목적지에서 멈추는 정찰 관측 개념.
                "default_goal_x": 110.0,
                "default_goal_y": 276.5,
            }],
        ),

        Node(
            package="path_planning",
            executable="local_path_node",
            name="tank_local_path_node",
            output="screen",
            parameters=[{
                "config_file": os.path.join(path_planning_share, "config", "fusion_mapping.yaml"),
                # 도착 로깅(route_*.json의 reached) 기준을 컨트롤러 정지 기준(10m)과 일치시킨다.
                # (불일치 시 컨트롤러는 ~8m에서 종료해도 local_path는 5m 기준이라 reached가 안 찍힘)
                "goal_tolerance": 10.0,
                # ★ route_id를 반드시 전달해야 정찰 리포트가 route_{A|B}.json으로 올바로 저장된다.
                #   (미전달 시 local_path가 기본값 'A'로 고정 → B 주행도 route_A.json에 덮어써져
                #    route_B.json이 영영 안 생긴다)
                "route_id": LaunchConfiguration("route_id"),
                "route_map_name": "finalmap",
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
                # APF 게인 재균형(진동 완화): 척력이 인력을 160배 압도해 코리더 밖으로 튕기던 문제.
                # k_att↑/k_rep↓/tangent↓로 경로 추종 vs 회피 균형, 보는 거리(influence/max_obstacle)도 좁힘.
                # min_obstacle_distance·위협회피는 유지 → 근접 안전 보장. 실주행 보며 미세조정.
                "influence_radius": 9.0,
                "k_att": 3.0,
                "k_rep": 60.0,
                "tangent_gain_scale": 1.0,
                "local_target_distance": 8.0,   # planner lookahead_distance(8m)와 정합
                "max_repulsive_norm": 20.0,
                "max_result_norm": 20.0,
                "min_obstacle_distance": 1.5,
                "max_obstacle_distance": 9.0,
                "front_sector_deg": 140.0,
                "path_corridor_width": 7.0,
                "obstacle_voxel_resolution": 1.0,
                "max_obstacle_points": 300,
                "use_discovered_objects": True,
                "passthrough_when_clear": True,
                # clear 판정 임계 상향(0.05→0.5): 노이즈 1~2점으로 직진↔회피 모드가 깜빡이던 채터링 방지.
                "repulsive_eps": 0.5,
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
                "mission_type": LaunchConfiguration("mission_type"),
                "goal_tolerance": 10.0,
                "heading_deadband_deg": 5.0,
                "steering_full_error_deg": 45.0,
                "min_ad_weight": 0.0,
                "max_ad_weight": 1.0,
                # weaving(A↔D 토글) 완화: PD(rate feedback) D 게인. 0이면 기존 순수 P 거동.
                "steering_kd": 0.2,
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
