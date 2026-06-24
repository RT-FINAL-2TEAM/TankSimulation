# -*- coding: utf-8 -*-
"""
tank_scenario2.launch.py — 시나리오2 진입점.

정찰(시나리오1)로 만든 합본맵(scenario2_map.map) + 지형격자(scenario2_terrain.json)를
planner에 주입해 자율 스택을 띄운다. tank_autonomous_control.launch.py를 그대로 include하되
static_map_file / terrain_cost_file 만 오버라이드(나머지 노드/파라미터는 동일).

전제: build_scenario2_map.py로 recon_reports/recon_map/scenario2_map.map 을 미리 생성.
  python3 scripts/build_scenario2_map.py

실행:
  ros2 launch control tank_scenario2.launch.py
  ros2 launch control tank_scenario2.launch.py scenario2_map:=/abs/scenario2_map.map
"""
import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _project_root() -> str:
    """프로젝트 루트. TANK_PROJECT_ROOT env 우선, 없으면 이 파일 위치에서 유추.

    src/control/launch/tank_scenario2.launch.py → parents[3] = 프로젝트 루트
    (colcon --symlink-install이면 resolve()가 심링크를 따라 src 실경로로 돌아간다).
    """
    env = os.environ.get("TANK_PROJECT_ROOT")
    if env:
        return env
    return str(Path(__file__).resolve().parents[3])


def generate_launch_description():
    control_share = get_package_share_directory("control")
    autonomous_launch = os.path.join(control_share, "launch", "tank_autonomous_control.launch.py")

    recon_map_dir = os.path.join(_project_root(), "recon_reports", "recon_map")
    default_map = os.path.join(recon_map_dir, "scenario2_map.map")
    default_terrain = os.path.join(recon_map_dir, "scenario2_terrain.json")

    scenario2_map_arg = DeclareLaunchArgument(
        "scenario2_map", default_value=default_map,
        description="정찰 합본맵(scenario2_map.map) 절대경로",
    )
    scenario2_terrain_arg = DeclareLaunchArgument(
        "scenario2_terrain", default_value=default_terrain,
        description="정찰 지형격자(scenario2_terrain.json) 절대경로",
    )

    return LaunchDescription([
        scenario2_map_arg,
        scenario2_terrain_arg,
        # decision/risk 노드가 scripts/recon_eval(threat_geometry)을 찾도록 프로젝트 루트를 노출.
        SetEnvironmentVariable("TANK_PROJECT_ROOT", _project_root()),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(autonomous_launch),
            launch_arguments={
                "mission_type": "mission",
                "route_id": "A",        # 시나리오2 임무 루트 = A 고정(설계)
                "route_side": "west",
                "static_map_file": LaunchConfiguration("scenario2_map"),
                "terrain_cost_file": LaunchConfiguration("scenario2_terrain"),
                # 험지 회피 가시화: 기본 0.6은 너무 낮아 경로가 직선이 됨(거칠기 mean~0.22). 4.0으로
                # 올려 급경사(고-거칠기) 셀을 강하게 회피. 평지(거칠기≈0)는 거의 무영향.
                "terrain_weight": "4.0",
                # 시나리오2 local_path 리포트(route_A.json)를 정찰과 격리 → 정찰 recon_reports/route_A.json 미접촉.
                "recon_report_dir": os.path.join(_project_root(), "recon_reports", "scenario2"),
            }.items(),
        ),
        # --- 시나리오2 전술층: decision FSM(돌파/교전/복귀) + mock turret(교전 폐루프 stand-in) ---
        Node(
            package="mission", executable="decision_node", name="tank_decision_node",
            output="screen",
            parameters=[{
                "scenario2_map_file": LaunchConfiguration("scenario2_map"),
                "start_x": 59.0, "start_y": 27.0,    # routes.yaml finalmap.start
                "goal_tolerance": 10.0,              # ★ planner/control/local_path 와 일치
                "known_match_radius_m": 8.0,
                "engage_range_m": 20.0,              # Tank001 반경과 정합
                "risk_radius_m": 20.0,
                "risk_threshold": 0.5,
            }],
        ),
        Node(
            package="mission", executable="mock_turret_node", name="tank_mock_turret_node",
            output="screen",
            parameters=[{"aim_sec": 1.5, "always_hit": True}],
        ),
    ])
