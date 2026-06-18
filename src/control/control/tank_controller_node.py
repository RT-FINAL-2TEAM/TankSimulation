# -*- coding: utf-8 -*-
"""
ROS2 controller converted from the latest TankSimulation Flask controllers.

The control policy intentionally follows the team server logic:
- target yaw = atan2(dx, dy), where map x/y corresponds Unity x/z.
- steering command is A/D with weight proportional to yaw error.
- speed command slows/stops for large yaw error and stops at the final goal.
- local-minimum/stuck escape: reverse, then pivot turn.

It publishes the official Tank Challenge /get_action JSON to /tank/control/command.
"""

import json
import math
import os
from typing import Any, Dict, Optional, Tuple

import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path as NavPath
from rclpy.node import Node
from std_msgs.msg import String

from control.config import (
    CONTROLLER_HZ,
    ENABLE_LOCAL_TARGET,
    ENABLE_STUCK_ESCAPE,
    ESCAPE_REVERSE_SEC,
    ESCAPE_TURN_SEC,
    GOAL_TOLERANCE,
    HEADING_DEADBAND_DEG,
    MAX_AD_WEIGHT,
    MIN_AD_WEIGHT,
    ROTATE_IN_PLACE_ANGLE_DEG,
    SLOWDOWN_ANGLE_DEG,
    STOP_DISTANCE,
    STRAIGHT_WS_WEIGHT,
    STEERING_FULL_ERROR_DEG,
    STEERING_KD,
    STUCK_CHECK_PERIOD,
    STUCK_MIN_MOVEMENT,
    TARGET_TTL_SEC,
    TOPIC_COLLISION_EVENT,
    TOPIC_CONTROL_COMMAND,
    TOPIC_CONTROL_STATUS,
    TOPIC_GOAL_POSE,
    TOPIC_LOCAL_TARGET_POSE,
    TOPIC_LOOKAHEAD_POSE,
    TOPIC_PLAYER_POSE,
    TOPIC_PLAYER_STATE,
    TURN_WS_WEIGHT,
)


def normalize_angle(angle: float) -> float:
    while angle > 180.0:
        angle -= 360.0
    while angle < -180.0:
        angle += 360.0
    return angle


def get_distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def empty_action() -> Dict[str, Any]:
    return {
        "moveWS": {"command": "STOP", "weight": 1.0},
        "moveAD": {"command": "", "weight": 0.0},
        "turretQE": {"command": "", "weight": 0.0},
        "turretRF": {"command": "", "weight": 0.0},
        "fire": False,
    }


class TeamPathControllerNode(Node):
    def __init__(self) -> None:
        super().__init__("tank_team_path_controller_node")
        default_param_file = ""
        try:
            default_param_file = os.path.join(get_package_share_directory("control"), "config", "tank_parameters.yaml")
        except Exception:
            pass

        self.declare_parameter("tank_param_file", default_param_file)
        self.declare_parameter("controller_hz", 10.0)
        self.declare_parameter("max_speed", 25.0)
        self.declare_parameter("max_yaw_rate", 2.0)
        self.declare_parameter("goal_tolerance", 10.0)
        self.declare_parameter("enable_local_target", ENABLE_LOCAL_TARGET)
        self.declare_parameter("target_ttl_sec", TARGET_TTL_SEC)
        self.declare_parameter("mission_type", "mission")  # 'recon', 'mission', 'return'
        self.declare_parameter("heading_deadband_deg", HEADING_DEADBAND_DEG)
        self.declare_parameter("steering_full_error_deg", STEERING_FULL_ERROR_DEG)
        self.declare_parameter("min_ad_weight", MIN_AD_WEIGHT)
        self.declare_parameter("max_ad_weight", MAX_AD_WEIGHT)
        self.declare_parameter("steering_kd", STEERING_KD)
        self.declare_parameter("straight_ws_weight", STRAIGHT_WS_WEIGHT)
        self.declare_parameter("turn_ws_weight", TURN_WS_WEIGHT)
        self.declare_parameter("rotate_in_place_angle_deg", ROTATE_IN_PLACE_ANGLE_DEG)
        self.declare_parameter("slowdown_angle_deg", SLOWDOWN_ANGLE_DEG)
        self.declare_parameter("stop_distance", STOP_DISTANCE)
        self.declare_parameter("enable_stuck_escape", ENABLE_STUCK_ESCAPE)
        self.declare_parameter("stuck_check_period", STUCK_CHECK_PERIOD)
        self.declare_parameter("stuck_min_movement", STUCK_MIN_MOVEMENT)
        self.declare_parameter("escape_reverse_sec", ESCAPE_REVERSE_SEC)
        self.declare_parameter("escape_turn_sec", ESCAPE_TURN_SEC)

        self.enable_local_target = bool(self.get_parameter("enable_local_target").value)
        self.target_ttl_sec = float(self.get_parameter("target_ttl_sec").value)
        self.goal_tolerance = float(self.get_parameter("goal_tolerance").value)
        self.heading_deadband_deg = float(self.get_parameter("heading_deadband_deg").value)
        self.steering_full_error_deg = max(1.0, float(self.get_parameter("steering_full_error_deg").value))
        self.min_ad_weight = float(self.get_parameter("min_ad_weight").value)
        self.max_ad_weight = float(self.get_parameter("max_ad_weight").value)
        self.controller_hz = max(1.0, float(self.get_parameter("controller_hz").value))
        self.steering_kd = float(self.get_parameter("steering_kd").value)
        # PD(rate feedback)용 상태: 직전 헤딩값(yaw_rate 산출). None이면 첫 틱 rate=0.
        self._last_current_yaw: Optional[float] = None
        self.straight_ws_weight = float(self.get_parameter("straight_ws_weight").value)
        self.turn_ws_weight = float(self.get_parameter("turn_ws_weight").value)
        self.rotate_in_place_angle_deg = float(self.get_parameter("rotate_in_place_angle_deg").value)
        self.max_yaw_rate = float(self.get_parameter("max_yaw_rate").value)
        self.goal_tolerance = float(self.get_parameter("goal_tolerance").value)
        self.enable_local_target = bool(self.get_parameter("enable_local_target").value)
        self.target_ttl_sec = float(self.get_parameter("target_ttl_sec").value)
        self.mission_type = str(self.get_parameter("mission_type").value).lower()
        self.slowdown_angle_deg = float(self.get_parameter("slowdown_angle_deg").value)
        self.stop_distance = float(self.get_parameter("stop_distance").value)
        self.enable_stuck_escape = bool(self.get_parameter("enable_stuck_escape").value)
        self.stuck_check_period = float(self.get_parameter("stuck_check_period").value)
        self.stuck_min_movement = float(self.get_parameter("stuck_min_movement").value)
        self.escape_reverse_sec = float(self.get_parameter("escape_reverse_sec").value)
        self.escape_turn_sec = float(self.get_parameter("escape_turn_sec").value)

        self.tank_params = self.load_tank_params(str(self.get_parameter("tank_param_file").value))
        self.max_speed = self.extract_max_from_dict(self.tank_params.get("steady_state_speed", {}), 19.45)
        self.max_yaw_rate = self.extract_max_from_dict(self.tank_params.get("steady_state_yaw_rate", {}), 38.84)

        self.current_pos: Optional[Tuple[float, float]] = None
        self.current_yaw: float = 0.0
        self.current_speed: float = 0.0
        self.current_sim_time: float = 0.0
        self.goal_pos: Optional[Tuple[float, float]] = None
        self.lookahead_target: Optional[Tuple[float, float]] = None
        self.lookahead_stamp: float = 0.0
        self.local_target: Optional[Tuple[float, float]] = None
        self.local_target_stamp: float = 0.0
        self.collision_count = 0
        self.mission_complete = False

        self.last_stuck_check_time = 0.0
        self.last_stuck_check_pos: Optional[Tuple[float, float]] = None
        self.last_stuck_check_yaw: float = 0.0
        self.is_escaping = False
        self.escape_start_time = 0.0
        self._pivot_turn_count = 0
        self._last_pose_wall_time = 0.0

        self.global_path: list[tuple[float, float]] = []
        self.trajectory_history: list[tuple[float, float]] = []

        self.pub_cmd = self.create_publisher(String, TOPIC_CONTROL_COMMAND, 10)
        self.pub_status = self.create_publisher(String, TOPIC_CONTROL_STATUS, 10)
        self.create_subscription(PoseStamped, TOPIC_PLAYER_POSE, self.player_pose_cb, 10)
        self.create_subscription(String, TOPIC_PLAYER_STATE, self.player_state_cb, 10)
        self.create_subscription(PoseStamped, TOPIC_GOAL_POSE, self.goal_pose_cb, 10)
        self.create_subscription(PoseStamped, TOPIC_LOOKAHEAD_POSE, self.lookahead_cb, 10)
        self.create_subscription(PoseStamped, TOPIC_LOCAL_TARGET_POSE, self.local_target_cb, 10)
        self.create_subscription(NavPath, "/tank/global_path", self.global_path_cb, 10)
        self.create_subscription(String, TOPIC_COLLISION_EVENT, self.collision_cb, 10)
        hz = float(self.get_parameter("controller_hz").value)
        self.create_timer(1.0 / max(1.0, hz), self.timer_cb)
        self.get_logger().info(
            f"Team path controller initialized: max_speed={self.max_speed:.2f}, max_yaw_rate={self.max_yaw_rate:.2f}, "
            f"local_target={self.enable_local_target}"
        )

    def load_tank_params(self, path: str) -> Dict[str, Any]:
        if path and os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return yaml.safe_load(f) or {}
            except Exception as exc:
                self.get_logger().warn(f"failed to load tank params: {exc}")
        return {}

    @staticmethod
    def extract_max_from_dict(d: Dict[Any, Any], default: float) -> float:
        vals = []
        for v in d.values():
            try:
                vals.append(float(v))
            except Exception:
                pass
        return max(vals) if vals else default

    def player_pose_cb(self, msg: PoseStamped) -> None:
        import time
        new_pos = (float(msg.pose.position.x), float(msg.pose.position.y))
        if getattr(self, 'current_pos', None) is not None:
            if get_distance(self.current_pos, new_pos) > 10.0:
                self.get_logger().info("Teleport/Restart detected. Resetting mission complete flag.")
                self.mission_complete = False
                self._stop_published_count = 0
                self.is_escaping = False
                self.global_path = []
        self.current_pos = new_pos
        self._last_pose_wall_time = time.time()
        if self.last_stuck_check_pos is None:
            self.last_stuck_check_pos = self.current_pos
        self.trajectory_history.append(self.current_pos)
        if len(self.trajectory_history) > 5000:
            self.trajectory_history.pop(0)

    def global_path_cb(self, msg: NavPath) -> None:
        path = []
        for pose in msg.poses:
            path.append((float(pose.pose.position.x), float(pose.pose.position.y)))
        self.global_path = path

    def player_state_cb(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
        except Exception:
            return
        try:
            self.current_speed = float(data.get("speed") or 0.0)
        except Exception:
            self.current_speed = 0.0
        body = data.get("body") if isinstance(data.get("body"), dict) else {}
        try:
            self.current_yaw = float(body.get("x") or 0.0)
        except Exception:
            pass
        try:
            self.current_sim_time = float(data.get("sim_time") or self.current_sim_time)
        except Exception:
            pass

    def goal_pose_cb(self, msg: PoseStamped) -> None:
        self.goal_pos = (float(msg.pose.position.x), float(msg.pose.position.y))
        self.mission_complete = False

    def lookahead_cb(self, msg: PoseStamped) -> None:
        self.lookahead_target = (float(msg.pose.position.x), float(msg.pose.position.y))
        self.lookahead_stamp = self.get_clock().now().nanoseconds / 1e9

    def local_target_cb(self, msg: PoseStamped) -> None:
        self.local_target = (float(msg.pose.position.x), float(msg.pose.position.y))
        self.local_target_stamp = self.get_clock().now().nanoseconds / 1e9

    def collision_cb(self, msg: String) -> None:
        self.collision_count += 1

    def choose_target(self) -> Tuple[Optional[Tuple[float, float]], str]:
        now = self.get_clock().now().nanoseconds / 1e9
        if self.enable_local_target and self.local_target is not None and now - self.local_target_stamp <= self.target_ttl_sec:
            return self.local_target, "local_target_apf"
        if self.lookahead_target is not None and now - self.lookahead_stamp <= self.target_ttl_sec:
            return self.lookahead_target, "astar_lookahead"
        if self.goal_pos is not None:
            return self.goal_pos, "goal_pose_fallback"
        return None, "no_target"

    def calculate_steering(self, target: Tuple[float, float]) -> Tuple[str, float, float, float]:
        assert self.current_pos is not None
        dx = target[0] - self.current_pos[0]
        dy = target[1] - self.current_pos[1]
        desired_yaw = math.degrees(math.atan2(dx, dy))
        yaw_error = normalize_angle(desired_yaw - self.current_yaw)
        # yaw_rate: 헤딩 자체의 회전속도(deg/s). 타겟(setpoint) 변화엔 안 반응 → derivative kick 방지.
        if self._last_current_yaw is None:
            yaw_rate = 0.0
        else:
            yaw_rate = normalize_angle(self.current_yaw - self._last_current_yaw) * self.controller_hz
        self._last_current_yaw = self.current_yaw
        if abs(yaw_error) < self.heading_deadband_deg:
            return "", 0.0, yaw_error, desired_yaw
        # PD: P가 타겟으로 끌고, D(rate feedback)가 빠른 회전을 눌러 오버슈팅(=A↔D weaving) 억제.
        # 회전명령을 0으로 죽이지 않아(coast 없음) 좁은 코리더에서 언더스티어로 벽 긁는 부작용 없음.
        u = yaw_error - self.steering_kd * yaw_rate
        cmd = "D" if u > 0 else "A"
        weight = abs(u) / self.steering_full_error_deg
        if self.min_ad_weight > 0:
            weight = max(self.min_ad_weight, weight)
        weight = clamp(weight, 0.0, self.max_ad_weight)
        return cmd, weight, yaw_error, desired_yaw

    def calculate_speed(self, target: Tuple[float, float], yaw_error: float) -> Tuple[str, float, str]:
        assert self.current_pos is not None
        abs_err = abs(yaw_error)
        if self.goal_pos is not None and get_distance(self.current_pos, self.goal_pos) < self.goal_tolerance:
            self.mission_complete = True
            
            # 시나리오(mission_type)에 따른 도착 후 행동 분기
            if self.mission_type == "mission":
                self.fire_cmd = True  # 목표 도달 시 사격 (파괴 임무)
            else:
                self.fire_cmd = False # 정찰(recon)이나 복귀(return) 시 사격 금지

            if not hasattr(self, '_stop_published_count'):
                self._stop_published_count = 0
            self._stop_published_count += 1
            if self._stop_published_count > 10:
                self.get_logger().info(f"Mission Complete [{self.mission_type}]: Destination Reached. Terminating Control Node.")
                import sys
                sys.exit(0)
            return "STOP", 1.0, "goal_reached"
        if get_distance(self.current_pos, target) < 1.0 and self.goal_pos and get_distance(target, self.goal_pos) < self.stop_distance:
            return "STOP", 1.0, "target_stop"
        if abs_err > self.rotate_in_place_angle_deg:
            return "STOP", 1.0, "rotate_before_drive"
        
        speed_factor = self.max_speed / 19.45 if self.max_speed > 0.0 else 1.0
        if abs_err > self.slowdown_angle_deg:
            w_ws = clamp(self.turn_ws_weight * speed_factor, 0.1, 1.0)
            return "W", w_ws, "slow_turn"
            
        error_scale = 1.0 - (abs_err / self.slowdown_angle_deg) * 0.3
        w_ws = clamp(self.straight_ws_weight * speed_factor * error_scale, 0.1, 1.0)
        return "W", w_ws, "cruise"

    def escape_command_if_needed(self, target: Optional[Tuple[float, float]] = None) -> Optional[Tuple[Dict[str, Any], str]]:
        import time
        if not self.enable_stuck_escape or self.current_pos is None:
            return None
        t = self.current_sim_time if self.current_sim_time > 0.0 else time.time()
        if self.is_escaping:
            elapsed = t - self.escape_start_time
            if elapsed < self.escape_reverse_sec:
                return self.make_action("S", 1.0, "", 0.0), "escape_reverse"
            if elapsed < self.escape_reverse_sec + self.escape_turn_sec:
                pivot_dir = "D"
                if target is not None and self.current_pos is not None:
                    dx = target[0] - self.current_pos[0]
                    dy = target[1] - self.current_pos[1]
                    desired_yaw = math.degrees(math.atan2(dx, dy))
                    yaw_error = normalize_angle(desired_yaw - self.current_yaw)
                    pivot_dir = "D" if yaw_error > 0 else "A"
                return self.make_action("W", 0.4, pivot_dir, 1.0), "escape_pivot"
            self.is_escaping = False
            self.last_stuck_check_time = t
            self.last_stuck_check_pos = self.current_pos
            return None
            
        # 첫 호출(또는 baseline 미초기화) 시 stuck 판정 기준점을 현재 상태/시각으로 잡고
        # 한 주기(stuck_check_period)만큼 실제 주행할 시간을 준다.
        # ※ last_stuck_check_time은 0.0으로 초기화되는데 sim_time은 이미 크기 때문에,
        #   이 시각 가드가 없으면 출발 첫 사이클에 (t-0)>period 가 즉시 참이 되고
        #   moved≈0 으로 "끼임" 오판 → 출발 직후 계속 후진하는 버그가 난다.
        if self.last_stuck_check_pos is None or self.last_stuck_check_time <= 0.0:
            self.last_stuck_check_pos = self.current_pos
            self.last_stuck_check_yaw = self.current_yaw
            self.last_stuck_check_time = t
            return None

        if (t - self.last_stuck_check_time) > self.stuck_check_period:
            moved = get_distance(self.current_pos, self.last_stuck_check_pos)
            yaw_moved = abs(normalize_angle(self.current_yaw - self.last_stuck_check_yaw))
            if moved < self.stuck_min_movement and yaw_moved < 15.0:
                self.is_escaping = True
                self.escape_start_time = t
                self.get_logger().warn(f"stuck detected: moved={moved:.2f}m, yaw_moved={yaw_moved:.2f}deg, starting escape")
                return self.make_action("S", 1.0, "", 0.0), "escape_start_reverse"
            self.last_stuck_check_time = t
            self.last_stuck_check_pos = self.current_pos
            self.last_stuck_check_yaw = self.current_yaw
        return None

    def make_action(self, cmd_ws: str, w_ws: float, cmd_ad: str, w_ad: float) -> Dict[str, Any]:
        return {
            "moveWS": {"command": cmd_ws, "weight": float(clamp(w_ws, 0.0, 1.0))},
            "moveAD": {"command": cmd_ad, "weight": float(clamp(w_ad, 0.0, 1.0))},
            "turretQE": {"command": "", "weight": 0.0},
            "turretRF": {"command": "", "weight": 0.0},
            "fire": False,
        }

    def publish_command(self, action: Dict[str, Any]) -> None:
        msg = String()
        msg.data = json.dumps(action, ensure_ascii=False, separators=(",", ":"))
        self.pub_cmd.publish(msg)

    def publish_status(self, payload: Dict[str, Any]) -> None:
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self.pub_status.publish(msg)

    def timer_cb(self) -> None:
        import time
        if self._last_pose_wall_time > 0.0 and time.time() - self._last_pose_wall_time > 2.0:
            self.publish_command(empty_action())
            self.publish_status({"ok": False, "reason": "player_pose_stale_fallback"})
            return

        if self.current_pos is None:
            self.publish_command(empty_action())
            self.publish_status({"ok": False, "reason": "no_player_pose"})
            return
        target, source = self.choose_target()
        if target is None:
            self.publish_command(empty_action())
            self.publish_status({"ok": False, "reason": source, "current": {"x": self.current_pos[0], "y": self.current_pos[1]}})
            return

        escape = self.escape_command_if_needed(target)
        if escape is not None:
            action, mode = escape
            self.publish_command(action)
            self.publish_status({"ok": True, "mode": mode, "target_source": source, "command": action})
            return

        cmd_ad, w_ad, yaw_error, desired_yaw = self.calculate_steering(target)
        cmd_ws, w_ws, speed_mode = self.calculate_speed(target, yaw_error)
        action = self.make_action(cmd_ws, w_ws, cmd_ad, w_ad)
        if speed_mode == "goal_reached":
            # 목적지 도달 시 시뮬레이터 일시정지 요청.
            # 브릿지 select_action_command가 latest_command를 그대로 반환하므로
            # 이 control 필드가 /get_action 응답에 실려 시뮬로 전달된다.
            # (시뮬이 get_action 응답의 control:pause를 지원하면 정지, 미지원이면 STOP만 적용 — 무해)
            action["control"] = "pause"
        self.publish_command(action)

        mean_cte = 0.0
        max_cte = 0.0
        if self.global_path and self.trajectory_history:
            from .trajectory_metrics import calculate_cross_track_error
            try:
                mean_cte, max_cte = calculate_cross_track_error(self.trajectory_history, self.global_path)
            except Exception as exc:
                self.get_logger().error(f"Failed to calculate CTE: {exc}")

        self.publish_status({
            "ok": True,
            "target_source": source,
            "target": {"x": target[0], "y": target[1]},
            "current": {"x": self.current_pos[0], "y": self.current_pos[1]},
            "distance_to_target": get_distance(self.current_pos, target),
            "distance_to_goal": get_distance(self.current_pos, self.goal_pos) if self.goal_pos else None,
            "current_yaw_deg": self.current_yaw,
            "desired_yaw_deg": desired_yaw,
            "yaw_error_deg": yaw_error,
            "current_speed_mps": self.current_speed,
            "speed_mode": speed_mode,
            "mission_complete": self.mission_complete,
            "collision_count": self.collision_count,
            "mean_cte": mean_cte,
            "max_cte": max_cte,
            "command": action,
        })


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TeamPathControllerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
