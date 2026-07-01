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
from launch.actions import SetEnvironmentVariable, DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def _maybe_rosbridge(context, *args, **kwargs):
    """rosbridge_server가 설치돼 있으면 websocket 노드를 띄운다(웹 3D 뷰어 데이터 소켓 :9090).

    미설치면 graceful skip — 코어 스택은 영향 없음. start_rosbridge:=false로 끌 수 있음.
    웹 RViz는 ros_bridge(:5000)가 서빙하고, 토픽 데이터는 이 rosbridge가 ws로 중계한다.
    """
    flag = LaunchConfiguration("start_rosbridge").perform(context).strip().lower()
    if flag not in ("1", "true", "yes", "y"):
        return []
    try:
        get_package_share_directory("rosbridge_server")
    except Exception:
        print("[launch] rosbridge_server 미설치 — 웹 3D RViz 데이터 소켓 생략 "
              "(sudo apt install ros-humble-rosbridge-suite)")
        return []
    try:
        port = int(os.environ.get("TANK_RVIZ_WEB_ROSBRIDGE_PORT", "9090"))
    except (TypeError, ValueError):
        port = 9090
    return [Node(
        package="rosbridge_server",
        executable="rosbridge_websocket",
        name="rosbridge_websocket",
        output="screen",
        parameters=[{"port": port}],
    )]


def generate_launch_description():
    gui_share = get_package_share_directory("rviz_visualization")
    controller_share = get_package_share_directory("control")
    recon_map_file = os.path.join(gui_share, "map", "finalmap.map")
    tank_param_file = os.path.join(controller_share, "config", "tank_parameters.yaml")
    path_planning_share = get_package_share_directory("path_planning")
    potential_share = get_package_share_directory("potential")
    route_config_file = os.path.join(path_planning_share, "config", "routes.yaml")
    fusion_mapping_file = os.path.join(path_planning_share, "config", "fusion_mapping.yaml")
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
    route_config_file_arg = DeclareLaunchArgument(
        'route_config_file',
        default_value=route_config_file,
        description='Route YAML (scenario2 can provide a checkpoint-specific route).',
    )
    default_goal_x_arg = DeclareLaunchArgument(
        'default_goal_x', default_value='110.0',
        description='Initial map-frame x goal used by the global planner.',
    )
    default_goal_y_arg = DeclareLaunchArgument(
        'default_goal_y', default_value='276.5',
        description='Initial map-frame y goal used by the global planner.',
    )
    pause_on_goal_reached_arg = DeclareLaunchArgument(
        'pause_on_goal_reached', default_value='true',
        description='Request simulator pause at goal (false for stop-aim-fire checkpoints).',
    )
    exit_on_goal_reached_arg = DeclareLaunchArgument(
        'exit_on_goal_reached', default_value='true',
        description='Keep controller alive at a stop-aim-fire checkpoint when false.',
    )
    accept_external_goal_updates_arg = DeclareLaunchArgument(
        'accept_external_goal_updates', default_value='true',
        description='Accept simulator /set_destination updates on /tank/goal/pose.',
    )

    # 시나리오2 인계: 정찰로 만든 합본맵/지형격자를 planner에 주입하기 위한 오버라이드.
    # 기본값 = finalmap / 빈 지형 → 정찰(recon) 동작 불변(behavior-preserving).
    static_map_file_arg = DeclareLaunchArgument(
        'static_map_file',
        default_value=recon_map_file,
        description='A* static obstacle map (default finalmap; scenario2 overrides with scenario2_map.map)'
    )
    terrain_cost_file_arg = DeclareLaunchArgument(
        'terrain_cost_file',
        default_value='',
        description='A* terrain roughness cost grid (empty=off; scenario2 sets scenario2_terrain.json)'
    )
    recon_min_confirm_obs_arg = DeclareLaunchArgument(
        'recon_min_confirm_observations', default_value='-1',
        description='Override min fused observations to confirm (recon ~3; -1=use yaml).'
    )
    recon_min_confirm_age_arg = DeclareLaunchArgument(
        'recon_min_confirm_age_sec', default_value='-1.0',
        description='Override min age(sec) to confirm (recon ~0.5; -1=use yaml).'
    )
    # A* 지형 비용 가중치. recon은 terrain_cost_file 비어서 무영향; 시나리오2가 상향해 험지 회피.
    terrain_weight_arg = DeclareLaunchArgument(
        'terrain_weight', default_value='0.6',
        description='A* terrain roughness cost weight (scenario2 raises so path detours steep cells).'
    )
    # ReconLogger 출력 폴더(route_*.json). 기본=정찰 현행. 시나리오2는 recon_reports/scenario2로
    # 오버라이드해 정찰 route_A.json을 덮어쓰지 않게 격리.
    recon_report_dir_arg = DeclareLaunchArgument(
        'recon_report_dir', default_value='./recon_reports',
        description='ReconLogger output dir for route_*.json (scenario2 overrides to isolate from recon).'
    )
    # 웹 3D RViz 뷰어(:5000)용 데이터 소켓(rosbridge :9090) 자동 실행. 미설치면 graceful skip.
    start_rosbridge_arg = DeclareLaunchArgument(
        'start_rosbridge', default_value='true',
        description='Auto-start rosbridge_websocket(:9090) for the web 3D viewer if installed.'
    )
    require_turret_completion_for_reached_arg = DeclareLaunchArgument(
        'require_turret_completion_for_reached', default_value='false',
        description='Gate route report completion on /tank/turret/status terminal phase.'
    )

    return LaunchDescription([
        mission_type_arg,
        route_id_arg,
        route_side_arg,
        route_config_file_arg,
        default_goal_x_arg,
        default_goal_y_arg,
        pause_on_goal_reached_arg,
        exit_on_goal_reached_arg,
        accept_external_goal_updates_arg,
        static_map_file_arg,
        terrain_cost_file_arg,
        recon_min_confirm_obs_arg,
        recon_min_confirm_age_arg,
        terrain_weight_arg,
        recon_report_dir_arg,
        start_rosbridge_arg,
        OpaqueFunction(function=_maybe_rosbridge),
        require_turret_completion_for_reached_arg,
        SetEnvironmentVariable("TANK_START_CONTROL", "start"),
        SetEnvironmentVariable("TANK_APF_PASSTHROUGH_WHEN_CLEAR", "true"),

        Node(
            package="lidar",
            executable="lidar_processor_node",
            name="tank_lidar_processor_node",
            output="screen",
            parameters=[{"publish_legacy_lidar_json": False}],
        ),
        # Team visual perception integration:
        # - /detect image from ros_bridge + /info LiDAR raw -> camera LiDAR projection overlay
        # - /tank/sensor/lidar/detected_points_map -> DBSCAN cluster markers/status
        Node(
            package="tank_visual_perception",
            executable="lidar_camera_overlay_node",
            name="lidar_camera_overlay_node",
            output="screen",
            # Overlay and local_path_node must use the same calibration YAML.
            parameters=[{"config_file": fusion_mapping_file}],
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
                "inflate": 4.5,
                "use_path_smoothing": True,
                # False = latest TankSimulation step3 policy: do not cheat with GT obstacles.
                # True  = use /update_obstacle bbox list for static A* validation.
                "use_gt_obstacles": False,
                "enable_dynamic_replan": True,
                "enable_periodic_replan": False,
                # TankSimulation route A/B strategy is now active in the ROS2 planner.
                "use_route_waypoints": True,
                "route_config_file": LaunchConfiguration("route_config_file"),
                "route_map_name": "finalmap",
                "route_id": LaunchConfiguration("route_id"),
                "route_side": LaunchConfiguration("route_side"),
                "route_clearance_weight": 1.0,
                # 정적 맵(finalmap.map) 나무/바위를 전역 A* 코스트맵에 반영 → 루트가 나무 관통 안 하고
                # 코리더 중앙으로. use_gt_obstacles와 독립(빈 static_map_file이면 finalmap.map 자동 해석).
                # 시나리오2는 static_map_file을 scenario2_map.map으로, terrain_cost_file을
                # scenario2_terrain.json으로 오버라이드(정찰은 기본값 finalmap / 빈 지형).
                "use_static_map": True,
                # 시나리오2 인계: 정찰=finalmap 기본, 시나리오2=scenario2_map.map/지형 오버라이드.
                "static_map_file": LaunchConfiguration("static_map_file"),
                "terrain_cost_file": LaunchConfiguration("terrain_cost_file"),
                "terrain_weight": ParameterValue(
                    LaunchConfiguration("terrain_weight"), value_type=float),
                "tank_param_file": tank_param_file,
                "enable_speed_based_inflation": True,
                "enable_speed_based_emergency_replan": True,
                "enable_path_feasibility_check": True,
                "enable_semantic_risk_cost": True,
                "semantic_risk_weight": 0.06,
                "semantic_risk_radius_scale": 1.0,
                "semantic_risk_scores": "tank:100,house:50,car:25,tent:15,rock:10,unknown:5",
                "semantic_risk_radii": "tank:25,house:18,car:10,tent:8,rock:6,unknown:5",
                "enable_theta_aware_astar": True,
                "theta_heading_change_weight": 0.25,
                # A* polyline corner를 곡선으로 후처리해 꼭짓점 정지/yaw pivot을 줄인다.
                "enable_curvature_path_smoothing": True,
                "curvature_smoothing_min_turn_radius_m": 7.0,
                "curvature_smoothing_max_corner_angle_deg": 25.0,
                "curvature_smoothing_point_spacing_m": 1.0,
                "curvature_smoothing_collision_check_margin_m": 2.0,
                # Dense obstacle에서 새 path가 순간적으로 좌우로 뒤집히는 것을 막는 채택 필터.
                "enable_replan_acceptance_filter": False,
                "path_commitment_sec": 0.0,
                "replan_accept_max_length_ratio": 1.30,
                "replan_accept_max_sharp_corner_increase": 1,
                "replan_accept_max_heading_change_increase_deg": 120.0,
                "enable_avoid_side_lock": False,
                "avoid_side_lock_sec": 0.0,
                # Known map 초록점은 자율주행 전 이미 아는 hard no-go다(팀원 fix/control).
                "static_obstacle_inflate": 2.0,
                # DBSCAN cluster bboxes are used when dynamic replanning is enabled.
                "use_lidar_cluster_bboxes": True,
                "lidar_cluster_bbox_margin": 1.0,
                # 한 번 잡힌 LiDAR cluster는 TTL 동안 A* costmap에는 유지하되, 재계획 trigger로는 쓰지 않는다.
                "enable_lidar_cluster_memory": True,
                "lidar_cluster_memory_ttl_sec": 18.0,
                "lidar_cluster_memory_merge_distance": 5.0,
                "lidar_cluster_memory_inflate": 3.0,
                "lidar_cluster_memory_max_count": 80,
                "use_lidar_cluster_memory_for_path_block": False,
                # detected_points_map history/discovered는 A* costmap에는 반영하지만, 반복 replan trigger에서는 제외한다.
                # 현재 보이는 LiDAR cluster만 경로 차단 트리거로 쓰는 쪽이 route 흔들림이 적다.
                "use_lidar_memory_for_path_block": False,
                "use_discovered_objects_for_path_block": False,
                # Dynamic replan이 체크포인트 진행 상태를 뒤로 되돌려 lookahead가 반대편으로 튀는 것을 막는다.
                "route_index_never_decrease": True,
                "dynamic_replan_keep_route_index": True,
                "route_commit_lock_sec": 6.0,
                # Checkpoint progress is tracked separately from A* polyline route_index.
                # Once a route waypoint has been reached/passed in z, dynamic/emergency replan must not target it again.
                "route_checkpoint_never_decrease": True,
                "route_checkpoint_reached_radius": 8.0,
                "route_checkpoint_passed_z_margin": 3.0,
                # RViz discovered object도 confirmed 이후 A* persistent hard obstacle로 반영한다.
                "use_discovered_objects_for_astar": True,
                "discovered_confirmed_only": True,
                "discovered_min_observations": 2,
                "discovered_default_radius": 3.5,
                "discovered_obstacle_inflate": 2.0,
                "ignored_discovered_classes_for_astar": "person,human,blue,red",
                "replan_period_sec": 0.0,
                "dynamic_replan_cooldown_sec": 4.0,
                "plan_retry_period_sec": 3.0,
                "path_block_margin": 7.0,
                "path_block_required_hits": 2,
                # 현재 보이는 cluster가 전방 가까운 A* 경로 corridor를 막으면 일반 쿨타임보다 빠르게 재계획한다.
                "lidar_block_min_distance": 0.5,
                "lidar_block_max_distance": 80.0,
                "emergency_cluster_replan_enabled": True,
                # Emergency replan은 one-frame LiDAR/fusion 흔들림에 과민하면 경로가 좌우로 튄다.
                # 2초 cooldown + 3-hit 안정화로 실제로 지속되는 corridor block만 재계획한다.
                "emergency_replan_cooldown_sec": 2.0,
                "emergency_replan_front_distance": 24.0,
                "emergency_replan_min_distance": 0.0,
                "emergency_replan_margin": 9.0,
                "emergency_replan_required_hits": 3,
                "emergency_replan_margin_max": 11.0,
                # Conditional dynamic A*: 현재 보이는 LiDAR cluster가 실제 경로를 막을 때만 재계획한다.
                "dynamic_replan_max_count": 0,
                # 같은 위치에서 경로가 번갈아 뒤집히는 것을 막기 위한 progress guard.
                "dynamic_replan_min_progress_m": 2.0,
                "dynamic_replan_progress_guard_sec": 4.0,
                "lidar_cluster_eps": 2.0,
                "lidar_cluster_min_samples": 3,
                "lidar_history_resolution": 0.5,
                "max_lidar_history_points": 1500,
                "lookahead_distance": 13.0,
                "publish_path_period_sec": 2.0,
                "goal_tolerance": 10.0,
                "default_goal_enabled": True,
                # 기본은 기존 finalmap 목적지. 시나리오2는 checkpoint (50,260)로 override한다.
                "default_goal_x": ParameterValue(LaunchConfiguration("default_goal_x"), value_type=float),
                "default_goal_y": ParameterValue(LaunchConfiguration("default_goal_y"), value_type=float),
                # Scenario2 locks the firing checkpoint against legacy simulator destination posts.
                "accept_external_goal_updates": ParameterValue(
                    LaunchConfiguration("accept_external_goal_updates"), value_type=bool),
            }],
        ),

        Node(
            package="path_planning",
            executable="local_path_node",
            name="tank_local_path_node",
            output="screen",
            parameters=[{
                "config_file": fusion_mapping_file,
                # 도착 로깅(route_*.json의 reached) 기준을 컨트롤러 정지 기준(10m)과 일치시킨다.
                # (불일치 시 컨트롤러는 ~8m에서 종료해도 local_path는 5m 기준이라 reached가 안 찍힘)
                "goal_tolerance": 10.0,
                # ★ route_id를 반드시 전달해야 정찰 리포트가 route_{A|B}.json으로 올바로 저장된다.
                #   (미전달 시 local_path가 기본값 'A'로 고정 → B 주행도 route_A.json에 덮어써져
                #    route_B.json이 영영 안 생긴다)
                "route_id": LaunchConfiguration("route_id"),
                "route_map_name": "finalmap",
                # 정찰 확정 기준 override(-1=yaml). 발견객체 확정 누적 임계.
                "min_confirm_observations_override": ParameterValue(
                    LaunchConfiguration("recon_min_confirm_observations"), value_type=int),
                "min_confirm_age_sec_override": ParameterValue(
                    LaunchConfiguration("recon_min_confirm_age_sec"), value_type=float),
                # route_*.json 출력 폴더(시나리오2는 recon_reports/scenario2로 격리).
                "recon_report_dir": LaunchConfiguration("recon_report_dir"),
                # 정찰 전용 미분류-후보 관측요청(observe_request) 발행은 mission_type==recon에서만.
                "mission_type": LaunchConfiguration("mission_type"),
                "require_turret_completion_for_reached": ParameterValue(
                    LaunchConfiguration("require_turret_completion_for_reached"), value_type=bool),
                "turret_status_topic": "/tank/turret/status",
            }],
        ),
        # Node(
        #     package="potential",
        #     executable="potential_field_node",
        #     name="tank_team_potential_field_node",
        #     output="screen",
        #     parameters=[{
        #         # APF가 너무 강하면 제자리에서 빙빙 도는 루프가 생긴다.
        #         # A*는 기본 경로만 제공하고, APF는 부드러운 국소 회피 방향만 제공한다.
                
        #         "target_pose_topic": "/tank/path/lookahead_pose",
        #         "fallback_goal_topic": "/tank/goal/pose",
        #         "hz": 10.0,

        #         # APF 활성 조건
        #         # force가 작아도 장애물이 8m 이내면 회피 모드 진입
        #         "apf_activate_distance": 7.0,
        #         "safety_watch_distance": 9.0,
        #         "safety_caution_distance": 5.5,
        #         "safety_danger_distance": 3.5,
        #         "safety_watch_speed_limit_ws": 0.34,
        #         "safety_caution_speed_limit_ws": 0.22,
        #         "safety_danger_speed_limit_ws": 0.06,
        #         # APF 합벡터 방향과 현재 heading 차이가 크면 controller가 W를 끊는다.
        #         "apf_heading_stop_angle_deg": 78.0,
        #         "apf_heading_slow_angle_deg": 38.0,
        #         "apf_heading_stop_distance": 4.2,
        #         "apf_heading_min_result_norm": 0.5,
        #         "repulsive_eps": 0.05,
        #         "blocked_to_apf_ticks": 1,
        #         "clear_to_passthrough_ticks": 3,

        #         # APF 힘 비율
        #         # global path 추종력보다 실시간 장애물 회피력이 우선권을 갖도록 조정
        #         "influence_radius": 8.0,
        #         "k_att": 4.2,
        #         "k_rep": 80.0,
        #         "tangent_gain_scale": 0.22,

        #         # local target
        #         "local_target_distance": 5.5,
        #         "max_attractive_norm": 16.0,
        #         "max_repulsive_norm": 12.0,
        #         "max_result_norm": 8.0,

        #         # 장애물 필터
        #         "min_obstacle_distance": 0.3,
        #         "max_obstacle_distance": 8.0,
        #         "front_sector_deg": 100.0,
        #         "path_corridor_width": 4.5,
        #         "obstacle_voxel_resolution": 1.5,
        #         "max_obstacle_points": 50,

        #         # LiDAR cluster 기반 회피
        #         "use_lidar_clusters": True,
        #         "cluster_obstacle_min_count": 2,
        #         "cluster_first": True,
        #         "use_raw_lidar_fallback": True,

        #         # 발견 객체 반영
        #         "use_discovered_objects": True,
        #         "ignored_discovered_classes": "person,human,blue,red",

        #         # 장애물 없을 때는 A* lookahead 그대로 추종
        #         "passthrough_when_clear": True,

        #         # 위협 회피
        #         # 지금은 네가 finalmap을 계속 쓴다고 했으니 유지
        #         "use_threat_avoidance": True,
        #         "threat_map_file": recon_map_file,
        #         "threat_radius": 25.0,
        #         "k_threat_rep": 2000.0,

        #         # APF weight profile
        #         "apf_weights_file": apf_weights_file,
        #         "apf_weight_profile": "default",

        #         # RViz marker
        #         "marker_scale": 2.5,
        #     }],
        # ),
        Node(
            package="control",
            executable="tank_controller_node",
            name="tank_team_path_controller_node",
            output="screen",
            parameters=[{
                "tank_param_file": tank_param_file,
                "enable_dynamic_speed_policy": True,
                "enable_stopping_distance_model": True,
                "enable_curvature_speed_limit": True,
                "controller_hz": 10.0,
                "enable_local_target": False,
                "target_ttl_sec": 2.0,
                "enable_safety_speed_limit": False,
                "safety_status_topic": "/tank/potential/safety_status",
                "safety_status_ttl_sec": 1.2,
                # APF 합벡터 기반 W 차단. heading 차이가 크고 장애물이 가까우면 STOP + A/D만 수행.
                "enable_apf_vector_stop_pivot": False,
                "apf_stop_angle_deg": 90.0,
                "apf_slow_angle_deg": 42.0,
                "apf_stop_distance": 4.2,
                "apf_ttc_stop_sec": 1.45,
                "apf_ttc_slow_sec": 2.4,
                "apf_min_speed_for_ttc": 0.8,
                "apf_slow_ws_weight": 0.16,
                # STOP pivot 무한회전 방지: 짧게 차체만 돌린 뒤 저속 W로 빠져나가게 한다.
                "apf_stop_pivot_max_sec": 0.45,
                "apf_stop_pivot_release_angle_deg": 24.0,
                "apf_stop_pivot_cooldown_sec": 1.8,
                # 장애물 옆 통과 시 A↔D 반복 방지. 장애물 반대 방향을 조금 더 오래 유지한다.
                "prefer_turn_away_from_nearest_obstacle": True,
                "away_turn_lock_sec": 1.4,
                "enable_steering_direction_lock": True,
                "steering_direction_lock_sec": 1.25,
                # 장애물 옆 통과: bearing이 큰 side obstacle은 STOP하지 말고 저속 W로 통과
                "enable_side_pass_forward": True,
                "side_pass_bearing_deg": 58.0,
                "side_pass_min_distance": 4.2,
                "side_pass_ws_weight": 0.18,
                "front_stop_bearing_deg": 42.0,
                "hard_stop_distance": 3.8,
                "turn_away_hard_distance": 4.8,
                # 지그재그 장애물 중간에서 A/D가 반복되면 한쪽 방향을 잠시 유지
                "enable_ad_oscillation_guard": True,
                "ad_flip_window_sec": 3.0,
                "ad_flip_threshold": 3,
                "ad_oscillation_hold_sec": 2.0,
                "ad_oscillation_slow_ws_weight": 0.15,
                # 경로가 급격히 꺾일 때는 W를 끊고 차체 yaw를 먼저 맞춘다.
                # 큰 원을 그리며 경로 밖으로 밀려나는 현상을 줄인다.
                "enable_sharp_turn_stop_pivot": True,
                "sharp_turn_stop_angle_deg": 130.0,
                "sharp_turn_release_angle_deg": 70.0,
                "sharp_turn_min_target_distance": 4.0,
                "sharp_turn_max_sec": 0.25,
                "sharp_turn_cooldown_sec": 1.5,
                "sharp_turn_block_when_apf_side_pass": True,
                # 실제 속도가 높은 상태에서 급커브/장애물 앞을 W로 밀고 들어가면 관성으로 오버슛한다.
                # yaw error/장애물 위험도에 따라 STOP/S braking을 걸어 코너 진입 속도를 강제로 낮춘다.
                "enable_turn_overspeed_guard": True,
                "turn_overspeed_angle_deg": 35.0,
                "turn_overspeed_speed_mps": 2.8,
                "turn_overspeed_hard_angle_deg": 60.0,
                "turn_overspeed_hard_speed_mps": 4.5,
                "turn_overspeed_reverse_weight": 0.38,
                "turn_overspeed_slow_ws_weight": 0.10,
                "danger_obstacle_brake_speed_mps": 1.0,
                "danger_obstacle_reverse_weight": 0.50,
                # local/APF target이 너무 가까운 점에서 좌우로 바뀌며 제자리 회전하는 것을 방지
                "enable_forward_target_guard": True,
                "forward_guard_min_target_distance": 6.0,
                "forward_guard_yaw_error_deg": 85.0,
                "forward_guard_target_distance": 12.0,
                "forward_guard_max_search_points": 120,
                "forward_guard_allow_in_danger": False,
                "mission_type": LaunchConfiguration("mission_type"),
                "goal_tolerance": 10.0,
                "pause_on_goal_reached": ParameterValue(
                    LaunchConfiguration("pause_on_goal_reached"), value_type=bool),
                "exit_on_goal_reached": ParameterValue(
                    LaunchConfiguration("exit_on_goal_reached"), value_type=bool),
                "heading_deadband_deg": 5.0,
                "steering_full_error_deg": 45.0,
                "min_ad_weight": 0.0,
                "max_ad_weight": 1.0,
                # weaving(A↔D 토글) 완화: PD(rate feedback) D 게인. 주행 파라미터 = 팀원 fix/control 기준.
                "steering_kd": 0.18,
                "straight_ws_weight": 0.34,
                "turn_ws_weight": 0.18,
                "crawl_pivot_angle_deg": 90.0,
                "crawl_pivot_ws_weight": 0.06,
                "crawl_turn_ws_weight": 0.12,
                "rotate_in_place_angle_deg": 75.0,
                "slowdown_angle_deg": 30.0,
                "stop_distance": 12.0,
                "enable_stuck_escape": True,
                "stuck_check_period": 5.0,
                "stuck_min_movement": 1.5,
                "escape_reverse_sec": 1.5,
                "escape_turn_sec": 1.5,
            }],
        ),
    ])
