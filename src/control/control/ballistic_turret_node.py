#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Scenario-2 multi-checkpoint ballistic auto-aim controller.

The node owns turret intent only.  The driving controller remains the sole
publisher to ``/tank/control/command`` and merges this node's short-lived
``/tank/turret/override`` command.

Scenario-2 mission plan
-----------------------
``engagements_json`` defines a strict ordered sequence of engagements.  For
an intermediate engagement the vehicle stops, aims, fires, lowers the barrel
with F, and releases the hull so the planner can continue to the next route
checkpoint.  Only after the final engagement is the return goal published.

Coordinate convention: map x/y are ground-plane coordinates and map z is
height.  At the instant of each shot the node transforms the muzzle mount
from the hull/body frame into the map frame using playerBodyX/Y/Z
(yaw/pitch/roll).  The ballistic world direction is then inverse-rotated back
into a turret-relative yaw/pitch command.  This compensates for side-slope
roll and nose-up/nose-down hull attitude instead of treating the tank as level.
"""

from __future__ import annotations

import json
import math
import time
from typing import Any, Dict, List, Optional, Tuple

import rclpy
from geometry_msgs.msg import PoseStamped, Vector3Stamped
from rclpy.node import Node
from std_msgs.msg import String


TOPIC_PLAYER_POSE = "/tank/player/pose"
TOPIC_PLAYER_STATE = "/tank/player/state"
TOPIC_ENEMY_POSE = "/tank/enemy/pose"
TOPIC_TURRET_FEEDBACK = "/tank/api/get_action/turret"
TOPIC_BULLET_RAW = "/tank/api/update_bullet/raw"
TOPIC_TURRET_OVERRIDE = "/tank/turret/override"
TOPIC_TURRET_STATUS = "/tank/turret/status"
TOPIC_ENGAGE_RESULT = "/tank/engage/result"
TOPIC_MISSION_GOAL = "/tank/mission/goal_pose"


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def normalize_180(angle_deg: float) -> float:
    return (float(angle_deg) + 180.0) % 360.0 - 180.0


def as_finite_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


class BallisticTurretNode(Node):
    """Ordered stop → aim/fire → lower-barrel → continue mission controller."""

    def __init__(self) -> None:
        super().__init__("ballistic_turret_node")

        # Legacy single-stage parameters remain available as a safe fallback.
        self.declare_parameter("checkpoint_x", 50.0)
        self.declare_parameter("checkpoint_y", 260.0)
        self.declare_parameter("checkpoint_radius_m", 10.0)
        self.declare_parameter("checkpoint_settle_sec", 0.8)
        self.declare_parameter("target_from_enemy_pose", True)
        self.declare_parameter("target_pose_ttl_sec", 1.0)
        self.declare_parameter("target_x", 135.46)
        self.declare_parameter("target_y", 276.87)
        self.declare_parameter("target_z", 0.0)
        self.declare_parameter("target_height_offset_m", 0.0)
        self.declare_parameter("target_id", "enemy_main")
        # JSON list of ordered engagement objects.  See tank_scenario2.launch.py.
        self.declare_parameter("engagements_json", "")

        # Dataset-calibrated ballistic model and measured mechanical limits.
        self.declare_parameter("ballistic_k", 0.001520)
        self.declare_parameter("muzzle_height_m", 3.199)
        # Hull-attitude compensation.  The local body axes are
        # +x=right, +y=forward, +z=up.  playerBodyX/Y/Z are yaw/pitch/roll.
        self.declare_parameter("use_body_attitude_compensation", False)
        self.declare_parameter("body_pitch_sign", 1.0)
        self.declare_parameter("body_roll_sign", 1.0)
        self.declare_parameter("turret_yaw_feedback_is_world", True)
        self.declare_parameter("muzzle_offset_right_m", 0.0)
        self.declare_parameter("muzzle_offset_forward_m", 0.0)
        self.declare_parameter("body_attitude_ttl_sec", 1.0)
        self.declare_parameter("min_range_m", 20.0)
        self.declare_parameter("max_range_m", 130.0)
        self.declare_parameter("min_pitch_deg", -5.0)
        self.declare_parameter("max_pitch_deg", 10.0)
        self.declare_parameter("pitch_feedback_sign", 1.0)

        # Closed-loop aim controls.
        self.declare_parameter("control_hz", 20.0)
        self.declare_parameter("yaw_tolerance_deg", 1.6)
        self.declare_parameter("pitch_tolerance_deg", 0.75)
        self.declare_parameter("yaw_control_deadband_deg", 1.6)
        self.declare_parameter("pitch_control_deadband_deg", 0.75)
        self.declare_parameter("aim_stable_sec", 0.45)
        self.declare_parameter("turret_feedback_ttl_sec", 0.75)
        self.declare_parameter("on_target_cycles", 1)
        self.declare_parameter("yaw_weight_max", 0.55)
        self.declare_parameter("pitch_weight_max", 0.45)
        self.declare_parameter("fire_pulse_sec", 0.35)
        self.declare_parameter("post_fire_hold_sec", 1.5)
        self.declare_parameter("impact_timeout_sec", 8.0)
        self.declare_parameter("hit_radius_m", 4.0)
        self.declare_parameter("max_shots", 2)

        # Full barrel lowering after every engagement, including the middle one.
        self.declare_parameter("lower_barrel_after_engagement", True)
        self.declare_parameter("lower_barrel_target_deg", -5.0)
        self.declare_parameter("lower_barrel_tolerance_deg", 0.25)
        self.declare_parameter("lower_barrel_weight", 1.0)
        self.declare_parameter("lower_barrel_settle_sec", 0.20)
        self.declare_parameter("lower_barrel_timeout_sec", 8.0)

        # Return only after the final engagement.
        self.declare_parameter("return_enabled", True)
        self.declare_parameter("return_x", 59.0)
        self.declare_parameter("return_y", 27.0)
        self.declare_parameter("return_radius_m", 10.0)
        self.declare_parameter("return_goal_topic", TOPIC_MISSION_GOAL)

        self.target_pose_ttl_sec = max(0.05, float(self.get_parameter("target_pose_ttl_sec").value))
        self.default_checkpoint_settle_sec = max(
            0.0, float(self.get_parameter("checkpoint_settle_sec").value)
        )
        self.default_target_height_offset_m = float(
            self.get_parameter("target_height_offset_m").value
        )
        self.engagements = self._load_engagements()
        self.stage_index = 0

        self.k = max(1e-9, float(self.get_parameter("ballistic_k").value))
        self.muzzle_height_m = float(self.get_parameter("muzzle_height_m").value)
        self.use_body_attitude_compensation = bool(
            self.get_parameter("use_body_attitude_compensation").value
        )
        self.body_pitch_sign = 1.0 if float(
            self.get_parameter("body_pitch_sign").value
        ) >= 0.0 else -1.0
        self.body_roll_sign = 1.0 if float(
            self.get_parameter("body_roll_sign").value
        ) >= 0.0 else -1.0
        self.turret_yaw_feedback_is_world = bool(
            self.get_parameter("turret_yaw_feedback_is_world").value
        )
        self.muzzle_offset_right_m = float(
            self.get_parameter("muzzle_offset_right_m").value
        )
        self.muzzle_offset_forward_m = float(
            self.get_parameter("muzzle_offset_forward_m").value
        )
        self.body_attitude_ttl_sec = max(
            0.05, float(self.get_parameter("body_attitude_ttl_sec").value)
        )
        self.min_range_m = max(0.0, float(self.get_parameter("min_range_m").value))
        self.max_range_m = max(self.min_range_m, float(self.get_parameter("max_range_m").value))
        self.min_pitch_deg = float(self.get_parameter("min_pitch_deg").value)
        self.max_pitch_deg = float(self.get_parameter("max_pitch_deg").value)
        self.pitch_feedback_sign = 1.0 if float(
            self.get_parameter("pitch_feedback_sign").value
        ) >= 0.0 else -1.0

        self.yaw_tolerance_deg = max(0.05, float(self.get_parameter("yaw_tolerance_deg").value))
        self.pitch_tolerance_deg = max(0.05, float(self.get_parameter("pitch_tolerance_deg").value))
        self.yaw_control_deadband_deg = max(
            self.yaw_tolerance_deg,
            float(self.get_parameter("yaw_control_deadband_deg").value),
        )
        self.pitch_control_deadband_deg = max(
            self.pitch_tolerance_deg,
            float(self.get_parameter("pitch_control_deadband_deg").value),
        )
        self.aim_stable_sec = max(0.05, float(self.get_parameter("aim_stable_sec").value))
        self.turret_feedback_ttl_sec = max(
            0.05, float(self.get_parameter("turret_feedback_ttl_sec").value)
        )
        self.on_target_cycles_required = max(1, int(self.get_parameter("on_target_cycles").value))
        self.yaw_weight_max = clamp(float(self.get_parameter("yaw_weight_max").value), 0.10, 1.0)
        self.pitch_weight_max = clamp(float(self.get_parameter("pitch_weight_max").value), 0.20, 1.0)
        self.fire_pulse_sec = max(0.05, float(self.get_parameter("fire_pulse_sec").value))
        self.post_fire_hold_sec = max(0.0, float(self.get_parameter("post_fire_hold_sec").value))
        self.impact_timeout_sec = max(0.5, float(self.get_parameter("impact_timeout_sec").value))
        self.hit_radius_m = max(0.1, float(self.get_parameter("hit_radius_m").value))
        self.max_shots = max(1, int(self.get_parameter("max_shots").value))

        self.lower_barrel_after_engagement = bool(
            self.get_parameter("lower_barrel_after_engagement").value
        )
        self.lower_barrel_target_deg = clamp(
            float(self.get_parameter("lower_barrel_target_deg").value),
            self.min_pitch_deg,
            self.max_pitch_deg,
        )
        self.lower_barrel_tolerance_deg = max(
            0.05, float(self.get_parameter("lower_barrel_tolerance_deg").value)
        )
        self.lower_barrel_weight = clamp(
            float(self.get_parameter("lower_barrel_weight").value), 0.10, 1.0
        )
        self.lower_barrel_settle_sec = max(
            0.0, float(self.get_parameter("lower_barrel_settle_sec").value)
        )
        self.lower_barrel_timeout_sec = max(
            0.5, float(self.get_parameter("lower_barrel_timeout_sec").value)
        )

        self.return_enabled = bool(self.get_parameter("return_enabled").value)
        self.return_point = (
            float(self.get_parameter("return_x").value),
            float(self.get_parameter("return_y").value),
        )
        self.return_radius_m = max(0.5, float(self.get_parameter("return_radius_m").value))
        self.return_goal_topic = str(self.get_parameter("return_goal_topic").value)

        # Latest simulator state.
        self.player_pose: Optional[Tuple[float, float, float]] = None
        self.enemy_pose: Optional[Tuple[float, float, float]] = None
        self.enemy_pose_wall = -1e9
        self.turret_yaw_deg: Optional[float] = None
        self.turret_elevation_deg: Optional[float] = None
        self.turret_feedback_wall = -1e9
        self.dedicated_turret_feedback_wall = -1e9
        self.turret_feedback_source = "none"

        # Latest hull attitude from /tank/player/state.  These values use the
        # map/body convention established by the bridge: X=yaw, Y=pitch,
        # Z=roll.  A level fallback remains available until the first sample.
        self.body_yaw_deg = 0.0
        self.body_pitch_deg = 0.0
        self.body_roll_deg = 0.0
        self.body_attitude_wall = -1e9
        self.body_attitude_source = "level_fallback"

        # Per-stage controller state.
        self.phase = "approach"
        self.checkpoint_enter_wall: Optional[float] = None
        self.engagement_target: Optional[Tuple[float, float, float]] = None
        self.shot_target: Optional[Tuple[float, float, float]] = None
        self.shot_count = 0
        self.total_shot_count = 0
        self.on_target_cycles = 0
        self.on_target_since_wall: Optional[float] = None
        self.fire_started_wall: Optional[float] = None
        self.fire_until_wall: Optional[float] = None
        self.post_fire_until_wall: Optional[float] = None
        self.lowering_started_wall: Optional[float] = None
        self.lowered_since_wall: Optional[float] = None
        self.lowering_reason: Optional[str] = None
        self.last_result: Optional[Dict[str, Any]] = None
        self.stage_results: List[Dict[str, Any]] = []
        self.return_goal_sent = False
        self.return_started_wall: Optional[float] = None
        self._last_status_wall = -1e9

        self.override_pub = self.create_publisher(String, TOPIC_TURRET_OVERRIDE, 10)
        self.status_pub = self.create_publisher(String, TOPIC_TURRET_STATUS, 10)
        self.engage_result_pub = self.create_publisher(String, TOPIC_ENGAGE_RESULT, 10)
        self.return_goal_pub = self.create_publisher(PoseStamped, self.return_goal_topic, 10)

        self.create_subscription(PoseStamped, TOPIC_PLAYER_POSE, self._player_pose_cb, 10)
        self.create_subscription(PoseStamped, TOPIC_ENEMY_POSE, self._enemy_pose_cb, 10)
        self.create_subscription(String, TOPIC_PLAYER_STATE, self._player_state_cb, 10)
        self.create_subscription(Vector3Stamped, TOPIC_TURRET_FEEDBACK, self._turret_feedback_cb, 10)
        self.create_subscription(String, TOPIC_BULLET_RAW, self._bullet_cb, 10)

        hz = max(1.0, float(self.get_parameter("control_hz").value))
        self.create_timer(1.0 / hz, self._tick)

        plan_text = ", ".join(
            f"{i + 1}:{stage['id']}@({stage['checkpoint'][0]:.1f},{stage['checkpoint'][1]:.1f})"
            for i, stage in enumerate(self.engagements)
        )
        self.get_logger().info(
            "BallisticTurretNode ready: "
            f"engagements=[{plan_text}], return=({self.return_point[0]:.1f},{self.return_point[1]:.1f}), "
            f"k={self.k:.6f}, muzzle_h={self.muzzle_height_m:.3f}m, "
            f"attitude_comp={self.use_body_attitude_compensation}, "
            f"pitch_sign={self.body_pitch_sign:+.0f}, roll_sign={self.body_roll_sign:+.0f}"
        )

    # ------------------------------------------------------------------
    # Mission-plan parsing
    # ------------------------------------------------------------------
    def _load_engagements(self) -> List[Dict[str, Any]]:
        """Parse ``engagements_json`` and fall back to legacy one-stage params."""
        fallback = [{
            "id": str(self.get_parameter("target_id").value),
            "checkpoint": (
                float(self.get_parameter("checkpoint_x").value),
                float(self.get_parameter("checkpoint_y").value),
            ),
            "checkpoint_radius_m": max(
                0.5, float(self.get_parameter("checkpoint_radius_m").value)
            ),
            "checkpoint_settle_sec": self.default_checkpoint_settle_sec,
            "target": (
                float(self.get_parameter("target_x").value),
                float(self.get_parameter("target_y").value),
                float(self.get_parameter("target_z").value),
            ),
            "target_from_enemy_pose": bool(self.get_parameter("target_from_enemy_pose").value),
            "target_height_offset_m": self.default_target_height_offset_m,
        }]
        raw = str(self.get_parameter("engagements_json").value).strip()
        if not raw:
            return fallback
        try:
            payload = json.loads(raw)
            if not isinstance(payload, list) or not payload:
                raise ValueError("engagements_json must be a non-empty JSON list")
            stages: List[Dict[str, Any]] = []
            for index, item in enumerate(payload):
                if not isinstance(item, dict):
                    raise ValueError(f"stage {index} is not an object")
                checkpoint = item.get("checkpoint")
                target = item.get("target")
                if not isinstance(checkpoint, dict) or not isinstance(target, dict):
                    raise ValueError(f"stage {index} needs checkpoint and target objects")
                cx = as_finite_float(checkpoint.get("x"))
                cy = as_finite_float(checkpoint.get("y"))
                tx = as_finite_float(target.get("x"))
                ty = as_finite_float(target.get("y"))
                tz = as_finite_float(target.get("z"))
                if None in (cx, cy, tx, ty, tz):
                    raise ValueError(f"stage {index} contains a non-finite coordinate")
                radius = as_finite_float(checkpoint.get("radius_m"))
                settle = as_finite_float(item.get("checkpoint_settle_sec"))
                offset = as_finite_float(item.get("target_height_offset_m"))
                stages.append({
                    "id": str(item.get("id") or f"target_{index + 1}"),
                    "checkpoint": (float(cx), float(cy)),
                    "checkpoint_radius_m": max(0.5, float(radius if radius is not None else 10.0)),
                    "checkpoint_settle_sec": max(
                        0.0, float(settle if settle is not None else self.default_checkpoint_settle_sec)
                    ),
                    "target": (float(tx), float(ty), float(tz)),
                    "target_from_enemy_pose": bool(item.get("target_from_enemy_pose", False)),
                    "target_height_offset_m": float(
                        offset if offset is not None else self.default_target_height_offset_m
                    ),
                })
            return stages
        except Exception as exc:
            self.get_logger().error(
                f"Invalid engagements_json ({exc}); using legacy single-stage parameters"
            )
            return fallback

    @property
    def stage(self) -> Dict[str, Any]:
        return self.engagements[min(self.stage_index, len(self.engagements) - 1)]

    # ------------------------------------------------------------------
    # Input callbacks
    # ------------------------------------------------------------------
    def _player_pose_cb(self, msg: PoseStamped) -> None:
        self.player_pose = (
            float(msg.pose.position.x),
            float(msg.pose.position.y),
            float(msg.pose.position.z),
        )

    def _enemy_pose_cb(self, msg: PoseStamped) -> None:
        self.enemy_pose = (
            float(msg.pose.position.x),
            float(msg.pose.position.y),
            float(msg.pose.position.z),
        )
        self.enemy_pose_wall = time.monotonic()

    def _player_state_cb(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except Exception:
            return
        now = time.monotonic()
        body = payload.get("body") if isinstance(payload, dict) else None
        if isinstance(body, dict):
            raw_body_yaw = as_finite_float(body.get("x"))
            raw_body_pitch = as_finite_float(body.get("y"))
            raw_body_roll = as_finite_float(body.get("z"))
            if raw_body_yaw is not None:
                self.body_yaw_deg = normalize_180(raw_body_yaw)
            if raw_body_pitch is not None:
                self.body_pitch_deg = self.body_pitch_sign * raw_body_pitch
            if raw_body_roll is not None:
                self.body_roll_deg = self.body_roll_sign * raw_body_roll
            if any(value is not None for value in (raw_body_yaw, raw_body_pitch, raw_body_roll)):
                self.body_attitude_wall = now
                self.body_attitude_source = "player_state.body"

        turret = payload.get("turret") if isinstance(payload, dict) else None
        if not isinstance(turret, dict):
            return
        yaw = as_finite_float(turret.get("x"))
        raw_pitch = as_finite_float(turret.get("y"))
        # Dedicated feedback wins while fresh; do not alternate old/new samples.
        if now - self.dedicated_turret_feedback_wall <= 0.45:
            return
        if yaw is not None:
            self.turret_yaw_deg = normalize_180(yaw)
        if raw_pitch is not None:
            self.turret_elevation_deg = self.pitch_feedback_sign * raw_pitch
        if yaw is not None or raw_pitch is not None:
            self.turret_feedback_wall = now
            self.turret_feedback_source = "player_state"

    def _turret_feedback_cb(self, msg: Vector3Stamped) -> None:
        self.turret_yaw_deg = normalize_180(float(msg.vector.x))
        self.turret_elevation_deg = self.pitch_feedback_sign * float(msg.vector.y)
        now = time.monotonic()
        self.turret_feedback_wall = now
        self.dedicated_turret_feedback_wall = now
        self.turret_feedback_source = "get_action_turret"

    def _bullet_cb(self, msg: String) -> None:
        if self.fire_started_wall is None or self.phase not in {"firing", "wait_impact"}:
            return
        try:
            payload = json.loads(msg.data)
        except Exception:
            return
        impact = payload.get("impact_map") if isinstance(payload, dict) else None
        if not isinstance(impact, dict):
            return
        ix = as_finite_float(impact.get("x"))
        iy = as_finite_float(impact.get("y"))
        if ix is None or iy is None:
            return
        target = self.shot_target if self.shot_target is not None else self._selected_target()
        distance = math.hypot(ix - target[0], iy - target[1])
        hit_field = payload.get("target") if isinstance(payload, dict) else None
        success = bool(hit_field) or distance <= self.hit_radius_m
        result = {
            "engagement_index": self.stage_index + 1,
            "engagement_count": len(self.engagements),
            "target_id": self.stage["id"],
            "impact": {"x": ix, "y": iy},
            "success": success,
            "dist_to_target_m": distance,
            "source": "/tank/api/update_bullet",
            "shot_count": self.shot_count,
            "total_shot_count": self.total_shot_count,
        }
        self.last_result = result
        self.engage_result_pub.publish(String(data=json.dumps(result, ensure_ascii=False)))
        if success:
            self.phase = "hit"
            self.post_fire_until_wall = time.monotonic() + self.post_fire_hold_sec
            self.get_logger().info(
                f"Impact confirmed: stage={self.stage_index + 1}/{len(self.engagements)} "
                f"id={self.stage['id']}, error={distance:.2f}m"
            )
        elif self.shot_count < self.max_shots:
            self.phase = "aim"
            self._reset_aim_dwell()
            self.fire_started_wall = None
            self.fire_until_wall = None
            self.post_fire_until_wall = None
            self.get_logger().warn(
                f"Impact miss at stage={self.stage_index + 1} ({distance:.2f}m) — "
                f"re-aiming for shot {self.shot_count + 1}/{self.max_shots}"
            )
        else:
            self.phase = "miss"
            self.post_fire_until_wall = time.monotonic() + self.post_fire_hold_sec
            self.get_logger().warn(
                f"Impact miss at stage={self.stage_index + 1} ({distance:.2f}m), max shots reached"
            )

    # ------------------------------------------------------------------
    # Geometry and mission helpers
    # ------------------------------------------------------------------
    def _selected_target(self) -> Tuple[float, float, float]:
        if self.engagement_target is not None:
            return self.engagement_target
        stage = self.stage
        if (
            stage["target_from_enemy_pose"]
            and self.enemy_pose is not None
            and (time.monotonic() - self.enemy_pose_wall) <= self.target_pose_ttl_sec
        ):
            return self.enemy_pose
        return stage["target"]

    def _lock_engagement_target(self) -> None:
        if self.engagement_target is None:
            self.engagement_target = self._selected_target()
            self.get_logger().info(
                f"Locked target stage={self.stage_index + 1}/{len(self.engagements)} "
                f"id={self.stage['id']}: "
                f"({self.engagement_target[0]:.2f}, {self.engagement_target[1]:.2f}, "
                f"{self.engagement_target[2]:.2f})"
            )

    def _at_checkpoint(self) -> bool:
        if self.player_pose is None:
            return False
        checkpoint = self.stage["checkpoint"]
        return math.hypot(
            self.player_pose[0] - checkpoint[0], self.player_pose[1] - checkpoint[1]
        ) <= self.stage["checkpoint_radius_m"]

    def _at_return_point(self) -> bool:
        if self.player_pose is None:
            return False
        return math.hypot(
            self.player_pose[0] - self.return_point[0],
            self.player_pose[1] - self.return_point[1],
        ) <= self.return_radius_m

    def _publish_return_goal(self, reason: str) -> None:
        if not self.return_enabled or self.return_goal_sent:
            return
        msg = PoseStamped()
        msg.header.frame_id = "tank_map"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x = self.return_point[0]
        msg.pose.position.y = self.return_point[1]
        msg.pose.orientation.w = 1.0
        self.return_goal_pub.publish(msg)
        self.return_goal_sent = True
        self.return_started_wall = time.monotonic()
        self.get_logger().info(
            f"FINAL return goal after {reason}: ({self.return_point[0]:.1f},{self.return_point[1]:.1f})"
        )

    def _begin_final_return(self, reason: str) -> None:
        if self.return_enabled:
            self._publish_return_goal(reason)
            self.phase = "returning"
        else:
            self.phase = "complete"

    def _reset_aim_dwell(self) -> None:
        self.on_target_cycles = 0
        self.on_target_since_wall = None

    def _reset_stage_runtime(self) -> None:
        self.phase = "approach"
        self.checkpoint_enter_wall = None
        self.engagement_target = None
        self.shot_target = None
        self.shot_count = 0
        self._reset_aim_dwell()
        self.fire_started_wall = None
        self.fire_until_wall = None
        self.post_fire_until_wall = None
        self.lowering_started_wall = None
        self.lowered_since_wall = None
        self.lowering_reason = None
        self.last_result = None

    def _advance_or_return(self, reason: str) -> None:
        completed = {
            "engagement_index": self.stage_index + 1,
            "engagement_count": len(self.engagements),
            "target_id": self.stage["id"],
            "result": self.last_result,
            "completion_reason": reason,
        }
        self.stage_results.append(completed)
        if self.stage_index + 1 < len(self.engagements):
            finished_id = self.stage["id"]
            self.stage_index += 1
            self._reset_stage_runtime()
            next_stage = self.stage
            self.get_logger().info(
                f"Engagement {finished_id} complete; continuing to stage "
                f"{self.stage_index + 1}/{len(self.engagements)} "
                f"{next_stage['id']} at ({next_stage['checkpoint'][0]:.1f},{next_stage['checkpoint'][1]:.1f})"
            )
        else:
            self._begin_final_return(reason)

    def _begin_lowering_barrel(self, reason: str) -> None:
        if not self.lower_barrel_after_engagement:
            self._advance_or_return(reason)
            return
        self.phase = "lowering_barrel"
        self.lowering_reason = reason
        self.lowering_started_wall = time.monotonic()
        self.lowered_since_wall = None
        self.get_logger().info(
            f"Lowering barrel after stage={self.stage_index + 1} {reason}: "
            f"holding F to {self.lower_barrel_target_deg:.2f} deg"
        )

    def _solve_low_arc_pitch_deg(self, horizontal_range: float, height_delta: float) -> Optional[float]:
        if horizontal_range <= 1e-6:
            return None
        r2 = horizontal_range * horizontal_range
        discriminant = r2 - 4.0 * self.k * r2 * (self.k * r2 + height_delta)
        if discriminant < 0.0:
            return None
        tan_theta = (horizontal_range - math.sqrt(discriminant)) / (2.0 * self.k * r2)
        return math.degrees(math.atan(tan_theta))

    @staticmethod
    def _mat_mul(a: Tuple[Tuple[float, float, float], ...], b: Tuple[Tuple[float, float, float], ...]) -> Tuple[Tuple[float, float, float], ...]:
        return tuple(
            tuple(sum(a[row][k] * b[k][col] for k in range(3)) for col in range(3))
            for row in range(3)
        )

    @staticmethod
    def _mat_vec(a: Tuple[Tuple[float, float, float], ...], v: Tuple[float, float, float]) -> Tuple[float, float, float]:
        return (
            a[0][0] * v[0] + a[0][1] * v[1] + a[0][2] * v[2],
            a[1][0] * v[0] + a[1][1] * v[1] + a[1][2] * v[2],
            a[2][0] * v[0] + a[2][1] * v[1] + a[2][2] * v[2],
        )

    @staticmethod
    def _mat_transpose(a: Tuple[Tuple[float, float, float], ...]) -> Tuple[Tuple[float, float, float], ...]:
        return tuple(tuple(a[col][row] for col in range(3)) for row in range(3))

    def _world_from_body_rotation(self) -> Tuple[Tuple[float, float, float], ...]:
        """Return body-local -> map-world rotation for X=yaw,Y=pitch,Z=roll.

        Local body coordinates are right(+x), forward(+y), up(+z).  Thus a
        level hull at yaw=0 has forward aligned with map +y.  Positive pitch
        lifts the forward axis, while positive roll lowers the right side.
        The sign parameters preserve a one-line calibration path if a simulator
        reports pitch/roll with the opposite sign.
        """
        if not self.use_body_attitude_compensation:
            yaw_deg = self.body_yaw_deg
            pitch_deg = 0.0
            roll_deg = 0.0
        else:
            yaw_deg = self.body_yaw_deg
            pitch_deg = self.body_pitch_deg
            roll_deg = self.body_roll_deg

        yaw = math.radians(yaw_deg)
        pitch = math.radians(pitch_deg)
        roll = math.radians(roll_deg)
        cy, sy = math.cos(yaw), math.sin(yaw)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cr, sr = math.cos(roll), math.sin(roll)

        # yaw: local forward(+y) -> map (sin(yaw), cos(yaw), 0)
        r_yaw = (
            (cy, sy, 0.0),
            (-sy, cy, 0.0),
            (0.0, 0.0, 1.0),
        )
        # positive pitch: local forward(+y) gains world/local +z
        r_pitch = (
            (1.0, 0.0, 0.0),
            (0.0, cp, -sp),
            (0.0, sp, cp),
        )
        # positive roll: local right(+x) rotates down toward -z
        r_roll = (
            (cr, 0.0, sr),
            (0.0, 1.0, 0.0),
            (-sr, 0.0, cr),
        )
        return self._mat_mul(r_yaw, self._mat_mul(r_pitch, r_roll))

    def _body_attitude_debug(self, now: Optional[float] = None) -> Dict[str, Any]:
        if now is None:
            now = time.monotonic()
        age = now - self.body_attitude_wall
        return {
            "enabled": self.use_body_attitude_compensation,
            "yaw_deg": self.body_yaw_deg,
            "pitch_deg": self.body_pitch_deg,
            "roll_deg": self.body_roll_deg,
            "age_sec": round(max(0.0, age), 3),
            "fresh": age <= self.body_attitude_ttl_sec,
            "source": self.body_attitude_source,
            "pitch_sign": self.body_pitch_sign,
            "roll_sign": self.body_roll_sign,
            "turret_yaw_feedback_is_world": self.turret_yaw_feedback_is_world,
        }

    def _desired_solution(self) -> Tuple[Optional[Dict[str, float]], Optional[str]]:
        """Solve a world ballistic arc, then convert it to hull-relative gimbal angles."""
        if self.player_pose is None:
            return None, "no_player_pose"

        tx, ty, tz = self._selected_target()
        px, py, pz = self.player_pose
        world_from_body = self._world_from_body_rotation()
        muzzle_offset_world = self._mat_vec(
            world_from_body,
            (
                self.muzzle_offset_right_m,
                self.muzzle_offset_forward_m,
                self.muzzle_height_m,
            ),
        )
        muzzle_x = px + muzzle_offset_world[0]
        muzzle_y = py + muzzle_offset_world[1]
        muzzle_z = pz + muzzle_offset_world[2]

        dx = tx - muzzle_x
        dy = ty - muzzle_y
        distance = math.hypot(dx, dy)
        if distance < self.min_range_m or distance > self.max_range_m:
            return None, f"range_out_of_model:{distance:.2f}m"
        target_z = tz + self.stage["target_height_offset_m"]
        world_pitch_deg = self._solve_low_arc_pitch_deg(distance, target_z - muzzle_z)
        if world_pitch_deg is None:
            return None, "ballistic_discriminant_negative"

        world_yaw_deg = normalize_180(math.degrees(math.atan2(dx, dy)))
        yaw_rad = math.radians(world_yaw_deg)
        pitch_rad = math.radians(world_pitch_deg)
        desired_world_dir = (
            math.sin(yaw_rad) * math.cos(pitch_rad),
            math.cos(yaw_rad) * math.cos(pitch_rad),
            math.sin(pitch_rad),
        )
        desired_body_dir = self._mat_vec(
            self._mat_transpose(world_from_body), desired_world_dir
        )
        horizontal_body = math.hypot(desired_body_dir[0], desired_body_dir[1])
        if horizontal_body <= 1e-9:
            return None, "relative_direction_vertical"
        target_relative_yaw_deg = normalize_180(math.degrees(math.atan2(
            desired_body_dir[0], desired_body_dir[1]
        )))
        target_relative_pitch_deg = math.degrees(math.atan2(
            desired_body_dir[2], horizontal_body
        ))
        if target_relative_pitch_deg < self.min_pitch_deg or target_relative_pitch_deg > self.max_pitch_deg:
            return None, f"pitch_limit:{target_relative_pitch_deg:.2f}deg"

        # playerTurretX is published as an absolute map heading by this simulator.
        # Keep the local alternative parameterized so a future simulator variant
        # can expose relative turret yaw without changing the geometry.
        target_feedback_yaw_deg = (
            normalize_180(self.body_yaw_deg + target_relative_yaw_deg)
            if self.turret_yaw_feedback_is_world else target_relative_yaw_deg
        )
        return {
            "distance_m": distance,
            "target_yaw_deg": target_feedback_yaw_deg,
            "target_pitch_deg": target_relative_pitch_deg,
            "target_world_yaw_deg": world_yaw_deg,
            "target_world_pitch_deg": world_pitch_deg,
            "target_relative_yaw_deg": target_relative_yaw_deg,
            "target_relative_pitch_deg": target_relative_pitch_deg,
            "target_z_m": target_z,
            "muzzle_x_m": muzzle_x,
            "muzzle_y_m": muzzle_y,
            "muzzle_z_m": muzzle_z,
            "muzzle_offset_world_x_m": muzzle_offset_world[0],
            "muzzle_offset_world_y_m": muzzle_offset_world[1],
            "muzzle_offset_world_z_m": muzzle_offset_world[2],
        }, None

    @staticmethod
    def _yaw_weight(error_deg: float, max_weight: float) -> float:
        magnitude = abs(error_deg)
        if magnitude <= 1.0:
            return 0.0
        target_rate = clamp(0.75 * magnitude, 2.2, 14.0 if magnitude < 30.0 else 20.0)
        if error_deg >= 0.0:  # E
            weight = (target_rate - 0.660) / 31.848
        else:  # Q
            weight = (target_rate + 0.107) / 31.454
        return clamp(weight, 0.10, max_weight)

    @staticmethod
    def _pitch_weight(error_deg: float, max_weight: float) -> float:
        magnitude = abs(error_deg)
        if magnitude <= 0.20:
            return 0.0
        target_rate = clamp(0.85 * magnitude, 0.85, 2.0)
        if error_deg >= 0.0:  # R
            weight = (target_rate - 0.198) / 2.544
        else:  # F
            weight = (target_rate - 0.294) / 3.397
        return clamp(weight, 0.20, max_weight)

    @staticmethod
    def _axis(command: str, weight: float) -> Dict[str, Any]:
        return {"command": command if weight > 0.0 else "", "weight": float(weight if weight > 0.0 else 0.0)}

    def _stage_status_base(self, *, active: bool) -> Dict[str, Any]:
        checkpoint = self.stage["checkpoint"]
        target = self._selected_target()
        return {
            "phase": self.phase,
            "active": active,
            "engagement_index": self.stage_index + 1,
            "engagement_count": len(self.engagements),
            "engagement_id": self.stage["id"],
            "checkpoint": {"x": checkpoint[0], "y": checkpoint[1]},
            "player": (
                {"x": self.player_pose[0], "y": self.player_pose[1], "z": self.player_pose[2]}
                if self.player_pose else None
            ),
            "target": {
                "x": target[0],
                "y": target[1],
                "z": target[2] + self.stage["target_height_offset_m"],
            },
            "shot_count": self.shot_count,
            "total_shot_count": self.total_shot_count,
            "last_result": self.last_result,
            "completed_engagements": self.stage_results,
            "target_locked": self.engagement_target is not None,
            "body_attitude": self._body_attitude_debug(),
        }

    def _publish_override(
        self,
        *,
        active: bool,
        hold_motion: bool,
        turret_qe: Dict[str, Any],
        turret_rf: Dict[str, Any],
        fire: bool,
        status: Dict[str, Any],
    ) -> None:
        payload = {
            "active": bool(active),
            "hold_motion": bool(hold_motion),
            "turretQE": turret_qe,
            "turretRF": turret_rf,
            "fire": bool(fire),
            "status": status,
            "timestamp_monotonic": time.monotonic(),
        }
        self.override_pub.publish(String(data=json.dumps(payload, ensure_ascii=False)))
        now = time.monotonic()
        if now - self._last_status_wall >= 0.10:
            self._last_status_wall = now
            self.status_pub.publish(String(data=json.dumps(status, ensure_ascii=False)))

    # ------------------------------------------------------------------
    # State machine
    # ------------------------------------------------------------------
    def _tick(self) -> None:
        now = time.monotonic()

        # Terminal leg: never re-arm after the final stage.
        if self.phase in {"returning", "returned", "complete"}:
            if self.phase == "returning" and self._at_return_point():
                self.phase = "returned"
            status = self._stage_status_base(active=False)
            status.update({
                "phase": self.phase,
                "return": {"x": self.return_point[0], "y": self.return_point[1]},
                "return_goal_sent": self.return_goal_sent,
            })
            self._publish_override(
                active=False, hold_motion=False,
                turret_qe=self._axis("", 0.0), turret_rf=self._axis("", 0.0),
                fire=False, status=status,
            )
            return

        # Drive freely until the current engagement checkpoint is reached.
        if self.phase == "approach" and not self._at_checkpoint():
            status = self._stage_status_base(active=False)
            status.update({"phase": "approach", "reason": "outside_checkpoint"})
            self._publish_override(
                active=False, hold_motion=False,
                turret_qe=self._axis("", 0.0), turret_rf=self._axis("", 0.0),
                fire=False, status=status,
            )
            return

        # First arrival at each stage claims the hull and lets it settle.
        if self.phase == "approach":
            self.phase = "settling"
            self.checkpoint_enter_wall = now
        if self.phase == "settling":
            settle_sec = self.stage["checkpoint_settle_sec"]
            elapsed = now - (self.checkpoint_enter_wall or now)
            if elapsed < settle_sec:
                status = self._stage_status_base(active=True)
                status.update({
                    "phase": "settling",
                    "settle_remaining_sec": round(max(0.0, settle_sec - elapsed), 3),
                })
                self._publish_override(
                    active=True, hold_motion=True,
                    turret_qe=self._axis("", 0.0), turret_rf=self._axis("", 0.0),
                    fire=False, status=status,
                )
                return
            self.phase = "aim"
            self._lock_engagement_target()

        # Re-acquire only when entering aim for the first time in this stage.
        if self.phase in {"aim", "aim_error"}:
            self._lock_engagement_target()

        # Fire pulse and impact waiting are intentionally held before any aim
        # calculation so that no new control command interrupts a shot.
        if self.phase == "firing":
            fire = self.fire_until_wall is not None and now < self.fire_until_wall
            if not fire:
                self.phase = "wait_impact"
            status = self._stage_status_base(active=True)
            status.update({"phase": self.phase, "fire": fire})
            self._publish_override(
                active=True, hold_motion=True,
                turret_qe=self._axis("", 0.0), turret_rf=self._axis("", 0.0),
                fire=fire, status=status,
            )
            return

        if self.phase == "wait_impact":
            if self.fire_started_wall is not None and now - self.fire_started_wall > self.impact_timeout_sec:
                self.phase = "impact_timeout"
                self.post_fire_until_wall = now + self.post_fire_hold_sec
            status = self._stage_status_base(active=True)
            status.update({"phase": self.phase})
            self._publish_override(
                active=True, hold_motion=True,
                turret_qe=self._axis("", 0.0), turret_rf=self._axis("", 0.0),
                fire=False, status=status,
            )
            return

        if self.phase in {"hit", "miss", "impact_timeout"}:
            if self.post_fire_until_wall is not None and now < self.post_fire_until_wall:
                status = self._stage_status_base(active=True)
                status.update({"phase": self.phase})
                self._publish_override(
                    active=True, hold_motion=True,
                    turret_qe=self._axis("", 0.0), turret_rf=self._axis("", 0.0),
                    fire=False, status=status,
                )
                return
            self._begin_lowering_barrel(self.phase)

        if self.phase == "lowering_barrel":
            feedback_age = now - self.turret_feedback_wall
            feedback_fresh = (
                self.turret_elevation_deg is not None
                and feedback_age <= self.turret_feedback_ttl_sec
            )
            current_pitch = self.turret_elevation_deg if feedback_fresh else None
            reached = (
                current_pitch is not None
                and current_pitch <= self.lower_barrel_target_deg + self.lower_barrel_tolerance_deg
            )
            timed_out = (
                self.lowering_started_wall is not None
                and now - self.lowering_started_wall >= self.lower_barrel_timeout_sec
            )
            if reached:
                if self.lowered_since_wall is None:
                    self.lowered_since_wall = now
                stable_sec = now - self.lowered_since_wall
            else:
                self.lowered_since_wall = None
                stable_sec = 0.0

            status = self._stage_status_base(active=True)
            status.update({
                "phase": "lowering_barrel",
                "lowering_reason": self.lowering_reason,
                "lower_barrel_target_deg": self.lower_barrel_target_deg,
                "current_pitch_deg": current_pitch,
                "lower_barrel_reached": reached,
                "lower_barrel_stable_sec": round(stable_sec, 3),
                "turret_feedback_age_sec": round(max(0.0, feedback_age), 3),
                "turret_feedback_source": self.turret_feedback_source,
                "command": {"turretQE": {"command": "", "weight": 0.0},
                            "turretRF": {"command": "F", "weight": self.lower_barrel_weight}},
            })
            if reached and stable_sec >= self.lower_barrel_settle_sec:
                reason = f"{self.lowering_reason or 'engagement'}:barrel_lowered"
                self._advance_or_return(reason)
                next_active = self.phase not in {"approach", "returning", "complete"}
                next_status = self._stage_status_base(active=next_active)
                next_status.update({"phase": self.phase, "active": next_active,
                                    "return_goal_sent": self.return_goal_sent})
                self._publish_override(
                    active=next_active, hold_motion=next_active,
                    turret_qe=self._axis("", 0.0), turret_rf=self._axis("", 0.0),
                    fire=False, status=next_status,
                )
                return
            if timed_out:
                self.get_logger().warn(
                    f"Barrel-lowering timeout at stage={self.stage_index + 1}; continuing mission"
                )
                reason = f"{self.lowering_reason or 'engagement'}:barrel_lower_timeout"
                self._advance_or_return(reason)
                next_active = self.phase not in {"approach", "returning", "complete"}
                next_status = self._stage_status_base(active=next_active)
                next_status.update({"phase": self.phase, "active": next_active,
                                    "return_goal_sent": self.return_goal_sent})
                self._publish_override(
                    active=next_active, hold_motion=next_active,
                    turret_qe=self._axis("", 0.0), turret_rf=self._axis("", 0.0),
                    fire=False, status=next_status,
                )
                return
            self._publish_override(
                active=True, hold_motion=True,
                turret_qe=self._axis("", 0.0),
                turret_rf=self._axis("F", self.lower_barrel_weight),
                fire=False, status=status,
            )
            return

        # Closed-loop aiming stage.
        solution, reason = self._desired_solution()
        if solution is None:
            self.phase = "aim_error"
            status = self._stage_status_base(active=True)
            status.update({"phase": self.phase, "reason": reason})
            self._publish_override(
                active=True, hold_motion=True,
                turret_qe=self._axis("", 0.0), turret_rf=self._axis("", 0.0),
                fire=False, status=status,
            )
            return

        feedback_age = now - self.turret_feedback_wall
        if (
            self.turret_yaw_deg is None
            or self.turret_elevation_deg is None
            or feedback_age > self.turret_feedback_ttl_sec
        ):
            status = self._stage_status_base(active=True)
            status.update({
                "phase": "aim",
                "reason": "no_turret_feedback" if (
                    self.turret_yaw_deg is None or self.turret_elevation_deg is None
                ) else "turret_feedback_stale",
                "turret_feedback_age_sec": round(max(0.0, feedback_age), 3),
            })
            self._reset_aim_dwell()
            self._publish_override(
                active=True, hold_motion=True,
                turret_qe=self._axis("", 0.0), turret_rf=self._axis("", 0.0),
                fire=False, status=status,
            )
            return

        yaw_error = normalize_180(solution["target_yaw_deg"] - self.turret_yaw_deg)
        pitch_error = solution["target_pitch_deg"] - self.turret_elevation_deg
        on_target = (
            abs(yaw_error) <= self.yaw_tolerance_deg
            and abs(pitch_error) <= self.pitch_tolerance_deg
        )
        yaw_weight = 0.0 if abs(yaw_error) <= self.yaw_control_deadband_deg else self._yaw_weight(
            yaw_error, self.yaw_weight_max
        )
        pitch_weight = 0.0 if abs(pitch_error) <= self.pitch_control_deadband_deg else self._pitch_weight(
            pitch_error, self.pitch_weight_max
        )
        yaw_cmd = "E" if yaw_error > 0.0 else "Q"
        pitch_cmd = "R" if pitch_error > 0.0 else "F"

        if on_target:
            if self.on_target_since_wall is None:
                self.on_target_since_wall = now
            stable_sec = now - self.on_target_since_wall
            self.on_target_cycles += 1
        else:
            self._reset_aim_dwell()
            stable_sec = 0.0

        self.phase = "aim"
        status = self._stage_status_base(active=True)
        status.update(solution)
        status.update({
            "phase": "aim",
            "current_yaw_deg": self.turret_yaw_deg,
            "current_pitch_deg": self.turret_elevation_deg,
            "pitch_feedback_sign": self.pitch_feedback_sign,
            "body_attitude": self._body_attitude_debug(now),
            "yaw_error_deg": yaw_error,
            "pitch_error_deg": pitch_error,
            "on_target": on_target,
            "on_target_cycles": self.on_target_cycles,
            "on_target_cycles_required": self.on_target_cycles_required,
            "on_target_stable_sec": round(stable_sec, 3),
            "aim_stable_sec_required": self.aim_stable_sec,
            "turret_feedback_age_sec": round(max(0.0, feedback_age), 3),
            "turret_feedback_source": self.turret_feedback_source,
            "tolerances_deg": {
                "yaw_fire": self.yaw_tolerance_deg,
                "pitch_fire": self.pitch_tolerance_deg,
                "yaw_deadband": self.yaw_control_deadband_deg,
                "pitch_deadband": self.pitch_control_deadband_deg,
            },
            "command": {
                "turretQE": {"command": yaw_cmd if yaw_weight else "", "weight": yaw_weight},
                "turretRF": {"command": pitch_cmd if pitch_weight else "", "weight": pitch_weight},
            },
        })

        if on_target and stable_sec >= self.aim_stable_sec:
            self.shot_count += 1
            self.total_shot_count += 1
            self.phase = "firing"
            self.shot_target = self._selected_target()
            self.fire_started_wall = now
            self.fire_until_wall = now + self.fire_pulse_sec
            self._reset_aim_dwell()
            status.update({"phase": "firing", "fire": True})
            self.get_logger().info(
                f"FIRE stage={self.stage_index + 1}/{len(self.engagements)} "
                f"id={self.stage['id']} shot={self.shot_count}/{self.max_shots}: "
                f"range={solution['distance_m']:.2f}m yaw={solution['target_yaw_deg']:.2f}deg "
                f"pitch={solution['target_pitch_deg']:.2f}deg stable={stable_sec:.2f}s"
            )
            self._publish_override(
                active=True, hold_motion=True,
                turret_qe=self._axis("", 0.0), turret_rf=self._axis("", 0.0),
                fire=True, status=status,
            )
            return

        self._publish_override(
            active=True, hold_motion=True,
            turret_qe=self._axis(yaw_cmd, yaw_weight),
            turret_rf=self._axis(pitch_cmd, pitch_weight),
            fire=False, status=status,
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = BallisticTurretNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
