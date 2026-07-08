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
    '"checkpoint":{"x":50.0,"y":260.0,"radius_m":10.0},'
    '"checkpoint_settle_sec":0.8,'
    '"target":{"x":50.0,"y":285.0,"z":8.5},'
    '"target_from_enemy_pose":false,'
    '"target_height_offset_m":0.0,'
    '"reposition":{"enabled":true,"fallback_goals":[{"x":48.0,"y":208.0}],"arrival_radius_m":10.0,"min_travel_m":0.0,"timeout_sec":20.0,"max_attempts":1}'
    '},{'
    '"id":"enemy_final",'
    '"checkpoint":{"x":50.0,"y":260.0,"radius_m":10.0},'
    '"checkpoint_settle_sec":0.8,'
    '"target":{"x":135.46,"y":276.87,"z":0.0},'
    '"target_from_enemy_pose":true,'
    '"target_height_offset_m":0.0,'
    '"reposition":{"enabled":false,"heading_deg":0.0,"goal_offset_m":16.0,"min_travel_m":3.0,"arrival_radius_m":10.5,"max_attempts":2}'
    '}]'
)


def _env_bool(name: str, default: bool = False) -> bool:
    """Parse a user-facing boolean env var while preserving explicit false."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "y", "on")


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return float(default)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(default)


def _standoff_point() -> tuple[float, float]:
    # This is the empirically flatter Scenario-2 fire base.  The target is not
    # here; the tank stops here, then rotates the turret for both shots.
    return (
        _env_float("TANK_S2_STANDOFF_X", 48.0),
        _env_float("TANK_S2_STANDOFF_Y", 208.0),
    )


def _standoff_radius() -> float:
    return max(1.0, _env_float("TANK_S2_STANDOFF_RADIUS_M", 1.5))


def _static_firebase_point() -> tuple[float, float]:
    # Verified from the current recon terrain: far enough for stage-1 standoff,
    # flat enough for the level-ground ballistic dataset, and on the route-A
    # centerline so the hull approaches facing the target.
    return (
        _env_float("TANK_S2_STATIC_FIREBASE_X", _env_float("TANK_FIRE_STATIC_FIREBASE_X", 49.42)),
        _env_float("TANK_S2_STATIC_FIREBASE_Y", _env_float("TANK_FIRE_STATIC_FIREBASE_Y", 164.5)),
    )


def _mid_target_anchor() -> tuple[float, float]:
    # Static/object tank1 in final_v5.map is near (50, 286).  Use this anchor
    # only to choose the correct detected tank when multiple detections exist.
    return (
        _env_float("TANK_S2_MID_TARGET_X", 50.0),
        _env_float("TANK_S2_MID_TARGET_Y", 285.0),
    )


def _eng_point_xy(eng: dict, key: str = "target") -> tuple[float, float] | None:
    """Return x/y from an engagement target/checkpoint dict.

    Mission-plan files can use Unity x/z or map x/y naming.  Scenario-2
    ballistic_turret_node expects x/y, but older plan files sometimes preserve z.
    """
    if not isinstance(eng, dict):
        return None
    obj = eng.get(key)
    if not isinstance(obj, dict):
        return None
    try:
        x = float(obj.get("x"))
        y = float(obj.get("y", obj.get("z")))
    except (TypeError, ValueError):
        return None
    if not (x == x and y == y):
        return None
    return (x, y)


def _xy_dist(a: tuple[float, float] | None, b: tuple[float, float] | None) -> float | None:
    if a is None or b is None:
        return None
    dx = float(a[0]) - float(b[0])
    dy = float(a[1]) - float(b[1])
    return (dx * dx + dy * dy) ** 0.5


def _append_unique_goal(goals: list[dict], x: float, y: float, min_sep_m: float = 1.0) -> None:
    for g in goals:
        try:
            if _xy_dist((float(g.get("x")), float(g.get("y"))), (x, y)) < min_sep_m:
                return
        except Exception:
            continue
    goals.append({"x": float(x), "y": float(y)})


def _eng_range_m(eng: dict) -> float:
    d = _xy_dist(_eng_point_xy(eng, "checkpoint"), _eng_point_xy(eng, "target"))
    return float("inf") if d is None else d


def _default_engagements_list() -> list[dict]:
    import json
    return json.loads(_DEFAULT_ENGAGEMENTS_JSON)


def _normalize_scenario2_engagements(raw_engs: list[dict]) -> list[dict]:
    """Force Scenario-2 into the intended two-shot story.

    Intended sequence:
      1) static/object tank discovered during recon, e.g. Tank001_3 near (50, 286)
      2) simulator health-bearing final enemy tank, enemy_final near (135.46, 276.87)

    build_mission_plan.py can also include a detected tank that is spatially the
    same as enemy_final.  If that duplicate is passed directly to the ballistic
    FSM, the vehicle fires final enemy first, then static tank, then final enemy
    again.  This normalizer removes final-enemy duplicates and returns exactly
    two engagements whenever possible.
    """
    if not isinstance(raw_engs, list) or not raw_engs:
        return []
    engs = [e for e in raw_engs if isinstance(e, dict)]
    if not engs:
        return []

    fallback = _default_engagements_list()

    # Prefer explicit enemy_final as the true health-bearing simulator tank.
    final = next((e for e in engs if str(e.get("id", "")).lower() == "enemy_final"), None)
    if final is None:
        final = next((e for e in engs if e.get("target_from_enemy_pose") is True), None)
    if final is None:
        final = fallback[1]
    final_xy = _eng_point_xy(final, "target") or (135.46, 276.87)

    # Candidate for first shot: a discovered/static tank, not the final enemy.
    static_candidates: list[dict] = []
    for e in engs:
        eid = str(e.get("id", "")).lower()
        if eid == "enemy_final" or e.get("target_from_enemy_pose") is True:
            continue
        target_xy = _eng_point_xy(e, "target")
        # Drop duplicate detections of the real final enemy.
        d_final = _xy_dist(target_xy, final_xy)
        if d_final is not None and d_final < 25.0:
            continue
        static_candidates.append(e)

    # If there are multiple static tanks, choose the one nearest the known
    # object-tank anchor around (50, 285), not the one with the shortest range
    # from a possibly hilly mission-plan checkpoint.
    anchor = _mid_target_anchor()
    static = min(static_candidates, key=lambda e: (_xy_dist(_eng_point_xy(e, "target"), anchor) or float("inf"))) if static_candidates else fallback[0]

    # Stabilize IDs shown in logs/MFD while keeping target coordinates and
    # firing checkpoints from mission_plan.  Previous versions forced both
    # shots to a single standoff point near (50,260), which could put stage 1
    # directly in front of the recon tank.  The mission planner now selects
    # per-target flat standoff checkpoints; preserve them by default.  Operators
    # can still force a fixed fallback with TANK_S2_FORCE_STANDOFF=true.
    force_standoff = _env_bool("TANK_S2_FORCE_STANDOFF", default=False)
    sx, sy = _standoff_point()
    static_fx, static_fy = _static_firebase_point()
    force_static_firebase = _env_bool("TANK_S2_FORCE_STATIC_FIREBASE", default=True)

    def _checkpoint_for(e: dict, fallback_xy: tuple[float, float]) -> dict:
        src_xy = _eng_point_xy(e, "checkpoint")
        if force_standoff or src_xy is None:
            src_xy = fallback_xy
        raw = e.get("checkpoint") if isinstance(e.get("checkpoint"), dict) else {}
        radius = raw.get("radius_m", _standoff_radius()) if isinstance(raw, dict) else _standoff_radius()
        try:
            radius = float(radius)
        except (TypeError, ValueError):
            radius = _standoff_radius()
        fire_radius = _env_float("TANK_S2_FIRE_CHECKPOINT_RADIUS_M", _env_float("TANK_FIRE_CHECKPOINT_RADIUS_M", 1.0))
        radius = min(max(1.0, radius), max(1.0, fire_radius))
        return {"x": float(src_xy[0]), "y": float(src_xy[1]), "radius_m": radius}

    def _reposition_for(e: dict, cp: dict) -> dict:
        # Use mission-plan fallback fire bases only after sanitizing them.
        # Never allow a slope fallback to move the first shot toward the enemy
        # nose; a fallback must preserve roughly the same tactical standoff as
        # the selected checkpoint.
        target_xy = _eng_point_xy(e, "target")
        cp_xy = (float(cp["x"]), float(cp["y"]))
        cp_d = _xy_dist(cp_xy, target_xy)
        raw_rep = e.get("reposition") if isinstance(e.get("reposition"), dict) else {}
        raw_goals = raw_rep.get("fallback_goals", []) if isinstance(raw_rep, dict) else []
        goals: list[dict] = []
        if force_standoff:
            raw_goals = [{"x": sx, "y": sy}]
        if isinstance(raw_goals, list):
            for g in raw_goals:
                if not isinstance(g, dict):
                    continue
                try:
                    gx = float(g.get("x")); gy = float(g.get("y"))
                except (TypeError, ValueError):
                    continue
                if target_xy is not None and cp_d is not None:
                    gd = _xy_dist((gx, gy), target_xy)
                    if gd is not None and gd < max(20.0, cp_d - 5.0):
                        print(
                            f"[scenario2] reject target-side fallback for {e.get('id')}: "
                            f"({gx:.1f},{gy:.1f}) dist={gd:.1f}m < checkpoint_dist={cp_d:.1f}m"
                        )
                        continue
                if (gx, gy) == cp_xy:
                    continue
                goals.append({"x": gx, "y": gy})

        # Final safety fallback for stage-1 only: if the far, flat standoff still
        # fails the runtime hull-pitch guard, move to the empirically stable
        # final fire base instead of stalling forever.  This is deliberately the
        # last fallback, not the primary tactical plan.
        if bool(e.get("_allow_stage1_close_emergency_fallback", False)):
            ex = _env_float("TANK_S2_STATIC_EMERGENCY_FALLBACK_X", 50.0)
            ey = _env_float("TANK_S2_STATIC_EMERGENCY_FALLBACK_Y", 260.0)
            if target_xy is not None:
                ed = _xy_dist((ex, ey), target_xy)
            else:
                ed = None
            if ed is None or ed >= _env_float("TANK_S2_STATIC_EMERGENCY_MIN_RANGE_M", 20.0):
                _append_unique_goal(goals, ex, ey)
        enabled = bool(goals)
        return {
            "enabled": enabled,
            "fallback_goals": goals,
            "arrival_radius_m": float(cp.get("radius_m", _standoff_radius())),
            "min_travel_m": 0.0,
            "timeout_sec": _env_float("TANK_S2_REPOSITION_TIMEOUT_SEC", 55.0),
            "max_attempts": len(goals),
        }

    static = dict(static)
    static.setdefault("id", "enemy_mid")
    if str(static.get("id", "")).startswith("detected_"):
        static["source_id"] = static.get("id")
        static["id"] = "enemy_mid"
    static_cp = _checkpoint_for(static, (static_fx, static_fy) if force_static_firebase else (sx, sy))
    if force_static_firebase:
        fire_radius = _env_float("TANK_S2_FIRE_CHECKPOINT_RADIUS_M", _env_float("TANK_FIRE_CHECKPOINT_RADIUS_M", 1.0))
        static_cp = {"x": float(static_fx), "y": float(static_fy), "radius_m": max(1.0, fire_radius)}
    static["checkpoint"] = static_cp
    static["target_from_enemy_pose"] = False
    # Do not let stage 1 fall forward to the enemy nose. If the verified
    # firebase somehow fails, ballistic_turret_node's aim-error timeout advances
    # the mission instead of blocking the whole scenario.
    static["_allow_stage1_close_emergency_fallback"] = _env_bool("TANK_S2_ENABLE_STATIC_EMERGENCY_FALLBACK", default=False)
    static["checkpoint_settle_sec"] = float(static.get("checkpoint_settle_sec", 0.8) or 0.8)
    static["reposition"] = _reposition_for(static, static_cp)

    final = dict(final)
    final["id"] = "enemy_final"
    if _env_bool("TANK_S2_FORCE_FINAL_STABLE_FIREBASE", default=True):
        fire_radius = _env_float("TANK_S2_FIRE_CHECKPOINT_RADIUS_M", _env_float("TANK_FIRE_CHECKPOINT_RADIUS_M", 1.0))
        final_cp = {"x": 50.0, "y": 260.0, "radius_m": max(1.0, fire_radius)}
    else:
        final_cp = _checkpoint_for(final, (50.0, 260.0))
    final["checkpoint"] = final_cp
    final["target_from_enemy_pose"] = True
    final["checkpoint_settle_sec"] = float(final.get("checkpoint_settle_sec", 0.8) or 0.8)
    final["reposition"] = _reposition_for(final, final_cp)

    return [static, final]


def _scenario2_engagements(project_root: str) -> str:
    """사격 시퀀스(engagements_json) 결정.

    기본 정책을 "mission_plan.json이 있으면 자동 사용"으로 바꾼다.
    - TANK_USE_MISSION_PLAN=true  : mission_plan 강제 사용
    - TANK_USE_MISSION_PLAN=false : 검증 하드코딩값 강제 사용
    - env 미지정                  : recon_reports/mission_plan.json이 있으면 자동 사용

    실패(파일 없음/파싱 오류/engagements 없음)하면 항상 안전 기본값으로 폴백한다.
    """
    import json

    mp_file = os.environ.get(
        "TANK_MISSION_PLAN_FILE", os.path.join(project_root, "recon_reports", "mission_plan.json")
    )
    mission_plan_exists = Path(mp_file).is_file()
    use_mp = _env_bool("TANK_USE_MISSION_PLAN", default=mission_plan_exists)

    if not use_mp:
        reason = "env=false" if os.environ.get("TANK_USE_MISSION_PLAN") is not None else "mission_plan 없음"
        print(f"[scenario2] mission_plan 미사용({reason}) → 기본 사격 시퀀스 사용")
        return _DEFAULT_ENGAGEMENTS_JSON

    try:
        with open(mp_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        engs = data.get("engagements")
        if isinstance(engs, list) and engs:
            normalized = _normalize_scenario2_engagements(engs)
            if len(normalized) == 2:
                print(
                    f"[scenario2] mission_plan 사격 시퀀스 사용: {mp_file} "
                    f"(원본 {len(engs)}개 → 실행 2개: "
                    f"{normalized[0].get('id')} -> {normalized[1].get('id')})"
                )
                print(f"[scenario2] route_recommended={data.get('route_recommended', '-')} order={data.get('plan', {}).get('engage_order', '-')}")
                return json.dumps(normalized, ensure_ascii=False)
            print(f"[scenario2] mission_plan 정규화 실패 → 기본 사격 시퀀스 사용: {mp_file}")
        else:
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
                # Intermediate firing checkpoints must remain alive; final home
                # arrival becomes an all-axis terminal STOP only after the
                # ballistic FSM reports returned/complete.
                # Final STOP is owned by ballistic_turret_node, not by the
                # controller's generic /tank/goal/pose check.  Home == spawn,
                # so controller-owned terminal logic can falsely stop at launch
                # when another stale planner/status publisher exists.
                "terminal_stop_on_turret_complete": "false",
                "terminal_stop_require_goal_departure": "false",
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
                # SCENARIO2_FLAT_FIRE_FALLBACK: enemy_mid/final fallback is the flat standoff point (default 50,260).
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
                "yaw_weight_max": 0.38,
                # Delayed-feedback hybrid yaw control: coarse closed-loop
                # tracking, then neutral/brake + bounded time pulse +
                # fresh-feedback observation around the target.
                "hybrid_yaw_enabled": True,
                "yaw_overshoot_brake_sec": 0.18,
                "yaw_pulse_weight": 0.14,
                "yaw_pulse_rate_q_deg_s": 4.3,
                "yaw_pulse_rate_e_deg_s": 5.1,
                "yaw_pulse_gain": 0.55,
                # tank_controller_node is normally 10 Hz, so use a
                # pulse no shorter than one actual command cycle.
                "yaw_pulse_min_sec": 0.12,
                "yaw_pulse_max_sec": 0.30,
                "yaw_observe_sec": 0.16,
                "yaw_settle_rate_deg_s": 0.65,
                "yaw_overshoot_min_prev_error_deg": 1.20,
                "yaw_overshoot_min_current_error_deg": 0.35,
                "aim_stable_sec": 0.22,
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
                # Flat-fire guard: ballistic data is reliable on level ground.
                # If the hull is tilted at a firing stop, the turret node will
                # request the stage fallback standoff point instead of aiming forever.
                "require_flat_fire_pose": True,
                "max_fire_body_pitch_deg": 3.0,
                "max_fire_body_roll_deg": 3.0,
                "align_hull_before_fire": True,
                "max_fire_body_yaw_error_deg": 35.0,
                "hull_alignment_backoff_m": 8.0,
                "hull_alignment_arrival_radius_m": 1.5,
                "hull_alignment_timeout_sec": 35.0,
                "hull_alignment_max_attempts": 1,
                "aim_error_skip_after_sec": 12.0,
                "reposition_goal_offset_m": 16.0,
                "reposition_min_travel_m": 3.0,
                "reposition_arrival_radius_m": 1.5,
                "reposition_timeout_sec": 55.0,
                "reposition_max_attempts": 2,
                # Only after engagement 2 is complete does the node issue home.
                "return_enabled": True,
                "return_x": 59.0,
                "return_y": 27.0,
                "return_radius_m": 10.0,
                "return_goal_topic": "/tank/mission/goal_pose",
                # sudden_advisor RETURN is a safety preemption of the fixed
                # two-target firing FSM, not merely an MFD recommendation.
                "mission_abort_enabled": True,
                "mission_abort_topic": "/tank/mission/abort",
            }],
        ),
        # 돌발 대응: perception→sudden_decision→/tank/decision/status(+MFD 패널).
        # RETURN만 히스테리시스 통과 후 실제 실행한다. abort 토픽이 ballistic FSM을
        # 중단하고 planner를 home direct-A*로 고정하므로, 기존 북진 checkpoint를 다시 밟지 않는다.
        # use_llm=false로 LLM 자문만 끄고 수식 판단+복귀 실행은 유지할 수 있다.
        Node(
            package="mission", executable="sudden_advisor_node", name="tank_sudden_advisor_node",
            output="screen",
            parameters=[{
                "scenario2_map": LaunchConfiguration("scenario2_map"),
                "tick_hz": 2.0,
                "hysteresis_ticks": 2,
                "use_llm": True,
                "execute_return": True,
                # SCENARIO2_RETURN_ARM_GUARD_V1: prevent spawn-point self-return
                # if start-area detections are briefly classified as new threats.
                "return_arm_min_distance_from_home_m": 35.0,
                "return_x": 59.0,
                "return_y": 27.0,
                "return_goal_topic": "/tank/mission/goal_pose",
                "mission_abort_topic": "/tank/mission/abort",
                "return_republish_sec": 0.5,
            }],
        ),
    ])
