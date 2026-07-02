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


def _clear_stale_scenario2_completion_files(project_root: str) -> None:
    """Remove stale completion sentinels before launching a new mission.

    The simulator-side Scenario-2 harness watches ``route_A.json`` and
    ``scenario2_result.json``. Leaving a previous run's files in place can
    make a new run finish before the vehicle reaches the firing checkpoint.
    """
    report_dir = Path(project_root) / "recon_reports" / "scenario2"
    for filename in ("route_A.json", "scenario2_result.json"):
        try:
            (report_dir / filename).unlink()
        except FileNotFoundError:
            pass
        except OSError as exc:
            print(f"[scenario2] stale report cleanup failed for {filename}: {exc}")


# 기본 사격 시퀀스 — cheol가 라이브 튜닝·검증한 값(안전 기본).
# mission_plan opt-in이 아니면 이걸 쓴다. reposition은 경사 pitch 한계 시 A* 재배치 fallback.
_DEFAULT_ENGAGEMENTS_JSON = (
    '[{'
    '"id":"enemy_mid",'
    '"checkpoint":{"x":48.0,"y":224.0,"radius_m":10.0},'
    '"checkpoint_settle_sec":0.8,'
    '"target":{"x":50.0,"y":285.0,"z":8.5},'
    '"target_from_enemy_pose":false,'
    '"target_height_offset_m":0.0,'
    '"reposition":{"enabled":true,"fallback_goals":[{"x":55.0,"y":230.0}],"arrival_radius_m":3.0,"min_travel_m":2.0,"timeout_sec":45.0,"max_attempts":1}'
    '},{'
    '"id":"enemy_final",'
    '"checkpoint":{"x":50.0,"y":260.0,"radius_m":10.0},'
    '"checkpoint_settle_sec":0.8,'
    '"target":{"x":135.46,"y":276.87,"z":0.0},'
    '"target_from_enemy_pose":true,'
    '"target_height_offset_m":0.0,'
    '"reposition":{"enabled":true,"heading_deg":0.0,"goal_offset_m":16.0,"min_travel_m":3.0,"arrival_radius_m":10.5,"max_attempts":2}'
    '}]'
)


def _scenario2_engagements(project_root: str) -> str:
    """사격 시퀀스(engagements_json) 결정.

    TANK_USE_MISSION_PLAN=true 이고 mission_plan.json에 engagements가 있으면 **정찰→자동 도출** 사격
    시퀀스를 쓴다(build_mission_plan.py 산출). 아니면 cheol 검증 하드코딩값(_DEFAULT)을 쓴다(안전 기본).
    실패(파일 없음/파싱 오류)해도 항상 기본값으로 폴백해 시나리오2가 깨지지 않게 한다.
    """
    use_mp = os.environ.get("TANK_USE_MISSION_PLAN", "false").strip().lower() in ("1", "true", "yes", "y")
    if not use_mp:
        return _DEFAULT_ENGAGEMENTS_JSON
    import json
    mp_file = os.environ.get(
        "TANK_MISSION_PLAN_FILE", os.path.join(project_root, "recon_reports", "mission_plan.json")
    )
    try:
        with open(mp_file, "r", encoding="utf-8") as f:
            engs = json.load(f).get("engagements")
        if isinstance(engs, list) and engs:
            print(f"[scenario2] mission_plan 사격 시퀀스 사용: {mp_file} (표적 {len(engs)}개)")
            return json.dumps(engs, ensure_ascii=False)
        print(f"[scenario2] mission_plan에 engagements 없음 → 기본 사격 시퀀스 사용: {mp_file}")
    except FileNotFoundError:
        print(f"[scenario2] mission_plan 파일 없음 → 기본 사격 시퀀스 사용: {mp_file}")
    except Exception as exc:  # noqa: BLE001 - 어떤 오류든 기본값 폴백
        print(f"[scenario2] mission_plan 로드 실패({exc}) → 기본 사격 시퀀스 사용")
    return _DEFAULT_ENGAGEMENTS_JSON


def generate_launch_description():
    project_root = _project_root()
    _clear_stale_scenario2_completion_files(project_root)
    engagements_json = _scenario2_engagements(project_root)

    control_share = get_package_share_directory("control")
    autonomous_launch = os.path.join(control_share, "launch", "tank_autonomous_control.launch.py")

    recon_map_dir = os.path.join(project_root, "recon_reports", "recon_map")
    default_map = os.path.join(recon_map_dir, "scenario2_map.map")
    default_terrain = os.path.join(recon_map_dir, "scenario2_terrain.json")
    scenario2_route_file = os.path.join(control_share, "config", "scenario2_routes.yaml")

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
        SetEnvironmentVariable("TANK_PROJECT_ROOT", project_root),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(autonomous_launch),
            launch_arguments={
                "mission_type": "mission",
                "route_id": "A",        # 시나리오2 임무 루트 = A 고정(설계)
                "route_side": "west",
                # 정찰 공용 route는 유지하고, 시나리오2에서는 50,260에서 멈추는 별도 route를 사용한다.
                "route_config_file": scenario2_route_file,
                "default_goal_x": "50.0",
                "default_goal_y": "260.0",
                # 도착 후 pause/exit하지 않고 controller가 STOP을 유지해야 포탑 노드가 실제 발사한다.
                "pause_on_goal_reached": "false",
                "exit_on_goal_reached": "false",
                # The external Scenario-2 harness declares success when it sees
                # route_A.json(reached=true). Hold that report at (50,260)
                # until ballistic_turret_node has fired and physically returned.
                "require_turret_completion_for_reached": "true",
                # Simulator's original /set_destination points at the enemy.
                # Lock it out: scenario2 must always visit the firing checkpoint first.
                "accept_external_goal_updates": "false",
                "static_map_file": LaunchConfiguration("scenario2_map"),
                "terrain_cost_file": LaunchConfiguration("scenario2_terrain"),
                # 지형 비용(험지 회피)을 정찰과 동일하게 OFF. 정찰은 terrain_cost_file이 비어 무영향이라,
                # 시나리오2도 weight=0.0으로 지형 비용을 꺼 주행 거동을 정찰과 일치시킨다(decision_node·맵은 유지).
                # 재활성화하려면 값만 올리면 됨(예전엔 4.0으로 급경사 셀 강회피).
                "terrain_weight": "0.0",
                # 시나리오2 local_path 리포트(route_A.json)를 정찰과 격리 → 정찰 recon_reports/route_A.json 미접촉.
                "recon_report_dir": os.path.join(project_root, "recon_reports", "scenario2"),
            }.items(),
        ),
        # Strict Scenario-2 sequence is owned by ballistic_turret_node:
        # checkpoint (50,260) -> full stop -> aim/fire -> internal return goal.
        # Do NOT launch decision_node here: its independent RETURN FSM can
        # overwrite the checkpoint while the ballistic node is waiting to fire.
        # The ballistic node owns the strict two-target sequence:
        # (48,224) -> static target (50,285,8.5) -> barrel down ->
        # (50,260) -> final enemy -> barrel down -> return home.
        Node(
            package="control", executable="ballistic_turret_node", name="tank_ballistic_turret_node",
            output="screen",
            parameters=[{
                # 기본=cheol 검증 하드코딩값. TANK_USE_MISSION_PLAN=true면 정찰→자동도출(mission_plan.json)로 교체.
                "engagements_json": engagements_json,
                # SCENARIO2_FIXED_FALLBACK_55_230: enemy_mid fixed fallback is (55, 230); no north-offset fallback.
        # Dataset-based ballistic and turret-feedback convention.
                "ballistic_k": 0.001520,
                "muzzle_height_m": 3.199,
                # Convert the world ballistic arc into hull-relative turret
                # yaw/pitch using playerBodyX/Y/Z (yaw/pitch/roll).  This is
                # essential when the hull is side-tilted on Scenario-2 terrain.
                "use_body_attitude_compensation": True,
                "body_pitch_sign": 1.0,
                "body_roll_sign": 1.0,
                "turret_yaw_feedback_is_world": True,
                "turret_pitch_feedback_is_world": True,
                "muzzle_offset_right_m": 0.0,
                "muzzle_offset_forward_m": 0.0,
                "body_attitude_ttl_sec": 1.0,
                "min_pitch_deg": -5.0,
                "max_pitch_deg": 10.0,
                "pitch_feedback_sign": 1.0,
                "max_range_m": 130.0,
                "fire_pulse_sec": 0.35,
                "impact_timeout_sec": 8.0,
                # Q/E를 기존보다 더 촘촘히 맞춘 뒤 사격한다.  1.0도 이하는
                # Q/E 입력을 멈추고, 해당 범위를 0.60초 유지해야 발사한다.
                "yaw_tolerance_deg": 1.0,
                "pitch_tolerance_deg": 0.75,
                "yaw_control_deadband_deg": 1.0,
                "pitch_control_deadband_deg": 0.75,
                "yaw_weight_max": 0.42,
                "aim_stable_sec": 0.60,
                "turret_feedback_ttl_sec": 0.75,
                "on_target_cycles": 1,
                # F is held down after *each* target so the next drive leg has
                # a clear forward camera view.
                "lower_barrel_after_engagement": True,
                "lower_barrel_target_deg": -5.0,
                "lower_barrel_weight": 1.0,
                "center_turret_tolerance_deg": 1.0,
                # At a slope-induced pitch limit, request a short direct A*
                # reposition instead of repeatedly pressing F/R at the stop.
                "reposition_on_unreachable_pitch": True,
                "reposition_goal_offset_m": 16.0,
                "reposition_min_travel_m": 3.0,
                "reposition_arrival_radius_m": 10.5,
                "reposition_timeout_sec": 35.0,
                "reposition_max_attempts": 2,
                # Only after engagement 2 is complete does the node issue home.
                "return_enabled": True,
                "return_x": 59.0,
                "return_y": 27.0,
                "return_radius_m": 10.0,
                "return_goal_topic": "/tank/mission/goal_pose",
            }],
        ),
    ])
