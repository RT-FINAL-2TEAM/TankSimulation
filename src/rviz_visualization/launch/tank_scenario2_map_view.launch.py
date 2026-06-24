# -*- coding: utf-8 -*-
"""
tank_scenario2_map_view.launch.py — 정찰 결과(시나리오2 입력) 시각화 전용.

apply_terrain이 finalmap(나무만) + latest(=B) 지형만 보여주는 것과 달리, 이 뷰는
"정찰로 만든 시나리오2 맵"을 그대로 보여준다:
- static_map_loader가 scenario2_map.map을 로드 → finalmap 나무 + **발견 장애물(rock/car/house 종류별 색)**.
- terrain 노드가 **A+B 합본 지형(scenario2_terrain.npz)** 을 면(메쉬)으로 표시(recording OFF, 주행해도 유지).

전제: build_scenario2_map.py로 recon_reports/recon_map/scenario2_map.map 생성 + 정찰로 route A 지형 저장.

실행:
  ros2 launch rviz_visualization tank_scenario2_map_view.launch.py
  ros2 launch rviz_visualization tank_scenario2_map_view.launch.py route:=B   # B 지형/맵 보기
"""
import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _project_root() -> str:
    """프로젝트 루트. TANK_PROJECT_ROOT env 우선, 없으면 이 파일 위치에서 유추.
    src/rviz_visualization/launch/X.launch.py → parents[3] = 루트(symlink-install이면 src로 resolve)."""
    env = os.environ.get("TANK_PROJECT_ROOT")
    if env:
        return env
    return str(Path(__file__).resolve().parents[3])


def generate_launch_description():
    pkg_share = get_package_share_directory("rviz_visualization")
    rviz_config = os.path.join(pkg_share, "rviz", "tank_finalmap.rviz")
    config_file = os.path.join(pkg_share, "config", "static_map_costs.yaml")

    root = _project_root()
    default_map = os.path.join(root, "recon_reports", "recon_map", "scenario2_map.map")
    # build_scenario2_map이 만든 A+B 합본 지형 NPZ(뷰 메쉬용). 없으면 인자로 route별 NPZ 지정 가능.
    default_terrain = os.path.join(root, "recon_reports", "recon_map", "scenario2_terrain.npz")

    scenario2_map_arg = DeclareLaunchArgument(
        "scenario2_map", default_value=default_map,
        description="정찰 합본맵(scenario2_map.map) 절대경로 — 발견 장애물 A+B 포함",
    )
    terrain_npz_arg = DeclareLaunchArgument(
        "terrain_npz", default_value=default_terrain,
        description="표시할 지형 NPZ (기본 scenario2_terrain.npz = A+B 합본)",
    )

    return LaunchDescription([
        scenario2_map_arg,
        terrain_npz_arg,
        # 1) 정적 맵 레이어 = scenario2_map.map (finalmap 나무 + 발견 장애물).
        #    발견객체는 metadata.class_name으로 범주 매칭되어 종류별 색/모양으로 렌더된다.
        Node(
            package="rviz_visualization",
            executable="static_map_loader_node",
            name="tank_static_map_loader_node",
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
        # 2) 현재 전차 pose / bridge marker 표시.
        Node(
            package="rviz_visualization",
            executable="rviz_visualizer_node",
            name="tank_rviz_visualizer_node",
            output="screen",
        ),
        # 3) 저장된 route 지형 NPZ를 면(메쉬)으로 표시. recording OFF라 주행해도 유지.
        Node(
            package="ground_division",
            executable="terrain_record_finalize_node",
            name="terrain_saved_map_visualizer_node",
            output="screen",
            parameters=[{
                "use_preclassified_lidar": True,
                "map_frame": "tank_map",
                "publish_period_sec": 0.5,
                "saved_map_file": LaunchConfiguration("terrain_npz"),
                "load_saved_map_on_start": True,
                "recording_enabled_on_start": False,
                "surface_mesh_enabled": True,
                "grid_cell_size": 0.8,
                # 메쉬는 grid_height_map의 stride 샘플링(items[::step])을 거치는데, 이게 셀 캡을 넘으면
                # 2D 인접(quad)을 깨 면이 ~94% 사라진다(A+B 0.8m=51,280셀 > 옛 캡 30,000 → B 면이 특히 소멸).
                # 캡을 셀 수 이상으로 올려 stride 샘플링 자체를 막는다 → A+B 전체 면 렌더(마커 ~10.6MB).
                # 마커가 무거워 안 뜨면 grid_cell_size를 1.0~1.1로 올려 셀 수를 줄일 것(여전히 full mesh).
                "max_elevation_cells": 60000,
                "marker_alpha": 0.75,
                "wireframe_enabled": False,
                "wireframe_max_height_gap": 3.0,
            }],
        ),
        ExecuteProcess(
            cmd=["rviz2", "-d", rviz_config],
            output="screen",
        ),
    ])
