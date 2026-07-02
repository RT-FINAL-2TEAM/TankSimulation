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
TOPIC_REPOSITION_GOAL = "/tank/mission/reposition_goal"


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
        # playerTurretY/get_action_turret.y is exposed as a map/world gun
        # elevation in the current simulator.  Keep the switch explicit for
        # older simulator builds that report a hull-relative elevation.
        self.declare_parameter("turret_pitch_feedback_is_world", True)
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
        # HYBRID_YAW_DELAY_COMPENSATION_V1
        # Coarse yaw acquisition remains closed-loop.  When delayed
        # feedback shows a target crossing, release Q/E first and then
        # use a bounded time-based correction pulse plus observation.
        self.declare_parameter("hybrid_yaw_enabled", True)
        self.declare_parameter("yaw_overshoot_brake_sec", 0.18)
        self.declare_parameter("yaw_pulse_weight", 0.14)
        self.declare_parameter("yaw_pulse_rate_q_deg_s", 4.3)
        self.declare_parameter("yaw_pulse_rate_e_deg_s", 5.1)
        self.declare_parameter("yaw_pulse_gain", 0.55)
        self.declare_parameter("yaw_pulse_min_sec", 0.12)
        self.declare_parameter("yaw_pulse_max_sec", 0.30)
        self.declare_parameter("yaw_observe_sec", 0.16)
        self.declare_parameter("yaw_settle_rate_deg_s", 0.65)
        self.declare_parameter("yaw_overshoot_min_prev_error_deg", 1.20)
        self.declare_parameter("yaw_overshoot_min_current_error_deg", 0.35)
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

        # After each engagement, rotate the turret back to the hull-forward
        # heading before releasing the vehicle to drive again.  The simulator
        # publishes turret yaw as an absolute map heading by default.
        self.declare_parameter("center_turret_after_engagement", True)
        self.declare_parameter("center_turret_heading_offset_deg", 0.0)
        self.declare_parameter("center_turret_tolerance_deg", 1.5)
        self.declare_parameter("center_turret_stable_sec", 0.35)
        self.declare_parameter("center_turret_timeout_sec", 8.0)
        self.declare_parameter("center_turret_weight_max", 0.40)

        # Return only after the final engagement.
        self.declare_parameter("return_enabled", True)
        self.declare_parameter("return_x", 59.0)
        self.declare_parameter("return_y", 27.0)
        self.declare_parameter("return_radius_m", 10.0)
        self.declare_parameter("return_goal_topic", TOPIC_MISSION_GOAL)

        # When the target solution is below/above the mechanically reachable
        # gun pitch because the hull is tilted, do not hold F/R forever.
        # Instead release the hull, request a short direct A* reposition along
        # the current route heading, then solve/aim again from the new attitude.
        self.declare_parameter("reposition_on_unreachable_pitch", True)
        # Planner goal-stop distance is 10m by default, so the target must be
        # farther than that for the controller to make physical progress.
        self.declare_parameter("reposition_goal_offset_m", 16.0)
        self.declare_parameter("reposition_min_travel_m", 3.0)
        self.declare_parameter("reposition_arrival_radius_m", 10.5)
        self.declare_parameter("reposition_timeout_sec", 35.0)
        self.declare_parameter("reposition_max_attempts", 2)
        self.declare_parameter("reposition_goal_topic", TOPIC_REPOSITION_GOAL)

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
        self.turret_pitch_feedback_is_world = bool(
            self.get_parameter("turret_pitch_feedback_is_world").value
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
        self.hybrid_yaw_enabled = bool(self.get_parameter("hybrid_yaw_enabled").value)
        self.yaw_overshoot_brake_sec = max(0.0, float(
            self.get_parameter("yaw_overshoot_brake_sec").value
        ))
        self.yaw_pulse_weight = clamp(float(
            self.get_parameter("yaw_pulse_weight").value
        ), 0.10, self.yaw_weight_max)
        self.yaw_pulse_rate_q_deg_s = max(0.10, float(
            self.get_parameter("yaw_pulse_rate_q_deg_s").value
        ))
        self.yaw_pulse_rate_e_deg_s = max(0.10, float(
            self.get_parameter("yaw_pulse_rate_e_deg_s").value
        ))
        self.yaw_pulse_gain = clamp(float(
            self.get_parameter("yaw_pulse_gain").value
        ), 0.05, 1.0)
        self.yaw_pulse_min_sec = max(0.01, float(
            self.get_parameter("yaw_pulse_min_sec").value
        ))
        self.yaw_pulse_max_sec = max(self.yaw_pulse_min_sec, float(
            self.get_parameter("yaw_pulse_max_sec").value
        ))
        self.yaw_observe_sec = max(0.0, float(
            self.get_parameter("yaw_observe_sec").value
        ))
        self.yaw_settle_rate_deg_s = max(0.01, float(
            self.get_parameter("yaw_settle_rate_deg_s").value
        ))
        self.yaw_overshoot_min_prev_error_deg = max(0.0, float(
            self.get_parameter("yaw_overshoot_min_prev_error_deg").value
        ))
        self.yaw_overshoot_min_current_error_deg = max(0.0, float(
            self.get_parameter("yaw_overshoot_min_current_error_deg").value
        ))
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
        self.center_turret_after_engagement = bool(
            self.get_parameter("center_turret_after_engagement").value
        )
        self.center_turret_heading_offset_deg = float(
            self.get_parameter("center_turret_heading_offset_deg").value
        )
        self.center_turret_tolerance_deg = max(
            0.10, float(self.get_parameter("center_turret_tolerance_deg").value)
        )
        self.center_turret_stable_sec = max(
            0.05, float(self.get_parameter("center_turret_stable_sec").value)
        )
        self.center_turret_timeout_sec = max(
            0.5, float(self.get_parameter("center_turret_timeout_sec").value)
        )
        self.center_turret_weight_max = clamp(
            float(self.get_parameter("center_turret_weight_max").value), 0.10, 1.0
        )

        self.return_enabled = bool(self.get_parameter("return_enabled").value)
        self.return_point = (
            float(self.get_parameter("return_x").value),
            float(self.get_parameter("return_y").value),
        )
        self.return_radius_m = max(0.5, float(self.get_parameter("return_radius_m").value))
        self.return_goal_topic = str(self.get_parameter("return_goal_topic").value)
        self.reposition_on_unreachable_pitch = bool(
            self.get_parameter("reposition_on_unreachable_pitch").value
        )
        self.reposition_goal_offset_m = max(0.5, float(
            self.get_parameter("reposition_goal_offset_m").value
        ))
        self.reposition_min_travel_m = max(0.0, float(
            self.get_parameter("reposition_min_travel_m").value
        ))
        self.reposition_arrival_radius_m = max(0.5, float(
            self.get_parameter("reposition_arrival_radius_m").value
        ))
        self.reposition_timeout_sec = max(1.0, float(
            self.get_parameter("reposition_timeout_sec").value
        ))
        self.reposition_max_attempts = max(0, int(
            self.get_parameter("reposition_max_attempts").value
        ))
        self.reposition_goal_topic = str(self.get_parameter("reposition_goal_topic").value)

        # Latest simulator state.
        self.player_pose: Optional[Tuple[float, float, float]] = None
        self.enemy_pose: Optional[Tuple[float, float, float]] = None
        self.enemy_pose_wall = -1e9
        self.turret_yaw_deg: Optional[float] = None
        self.turret_elevation_deg: Optional[float] = None
        self.turret_feedback_wall = -1e9
        self.dedicated_turret_feedback_wall = -1e9
        self.turret_feedback_source = "none"
        # Feedback sequence/rate are based on receipt time.  They are used
        # only to require an observation after Q/E is released; fire still
        # relies on the received measured angle, never a prediction.
        self._turret_feedback_seq = 0
        self._turret_last_yaw_sample_wall: Optional[float] = None
        self.turret_yaw_rate_deg_s = 0.0

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
        self.centering_started_wall: Optional[float] = None
        self.centered_since_wall: Optional[float] = None
        self.centering_reason: Optional[str] = None
        self.last_result: Optional[Dict[str, Any]] = None
        self.reposition_attempts = 0
        self.reposition_goal: Optional[Tuple[float, float]] = None
        self.reposition_start_pose: Optional[Tuple[float, float]] = None
        self.reposition_started_wall: Optional[float] = None
        self.reposition_reason: Optional[str] = None
        self.last_reposition_solution: Optional[Dict[str, Any]] = None
        self.stage_results: List[Dict[str, Any]] = []
        self.return_goal_sent = False
        self.return_started_wall: Optional[float] = None
        self._last_status_wall = -1e9
        self._reset_hybrid_yaw_control()

        self.override_pub = self.create_publisher(String, TOPIC_TURRET_OVERRIDE, 10)
        self.status_pub = self.create_publisher(String, TOPIC_TURRET_STATUS, 10)
        self.engage_result_pub = self.create_publisher(String, TOPIC_ENGAGE_RESULT, 10)
        self.return_goal_pub = self.create_publisher(PoseStamped, self.return_goal_topic, 10)
        self.reposition_goal_pub = self.create_publisher(PoseStamped, self.reposition_goal_topic, 10)

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
            f"pitch_feedback_world={self.turret_pitch_feedback_is_world}, "
            f"pitch_sign={self.body_pitch_sign:+.0f}, roll_sign={self.body_roll_sign:+.0f}, "
            f"reposition_on_pitch_limit={self.reposition_on_unreachable_pitch}, "
            f"center_after_fire={self.center_turret_after_engagement}"
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
            "reposition": {},
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
                    "reposition": (
                        dict(item.get("reposition"))
                        if isinstance(item.get("reposition"), dict) else {}
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

    def _record_turret_feedback(
        self,
        *,
        yaw: Optional[float],
        raw_pitch: Optional[float],
        source: str,
        now: float,
    ) -> None:
        """Record a received turret sample and estimate measured yaw rate.

        The rate is deliberately derived from *received* samples only.  It is
        not used to predict a fire angle; it only prevents a shot while the
        delayed stream still reports appreciable turret motion.
        """
        received = False
        if yaw is not None:
            normalized_yaw = normalize_180(yaw)
            previous_yaw = self.turret_yaw_deg
            previous_wall = self._turret_last_yaw_sample_wall
            if previous_yaw is not None and previous_wall is not None:
                dt = now - previous_wall
                if dt > 1e-3:
                    self.turret_yaw_rate_deg_s = normalize_180(
                        normalized_yaw - previous_yaw
                    ) / dt
            else:
                self.turret_yaw_rate_deg_s = 0.0
            self.turret_yaw_deg = normalized_yaw
            self._turret_last_yaw_sample_wall = now
            received = True
        if raw_pitch is not None:
            self.turret_elevation_deg = self.pitch_feedback_sign * raw_pitch
            received = True
        if received:
            self.turret_feedback_wall = now
            self.turret_feedback_source = source
            self._turret_feedback_seq += 1

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
        self._record_turret_feedback(
            yaw=yaw,
            raw_pitch=raw_pitch,
            source="player_state",
            now=now,
        )

    def _turret_feedback_cb(self, msg: Vector3Stamped) -> None:
        now = time.monotonic()
        self._record_turret_feedback(
            yaw=float(msg.vector.x),
            raw_pitch=float(msg.vector.y),
            source="get_action_turret",
            now=now,
        )
        self.dedicated_turret_feedback_wall = now

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
            self._reset_hybrid_yaw_control()
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
            self._reset_hybrid_yaw_control()
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

    def _make_goal_pose(self, point: Tuple[float, float]) -> PoseStamped:
        msg = PoseStamped()
        msg.header.frame_id = "tank_map"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x = float(point[0])
        msg.pose.position.y = float(point[1])
        msg.pose.orientation.w = 1.0
        return msg

    def _publish_mission_goal(self, point: Tuple[float, float], reason: str) -> None:
        self.return_goal_pub.publish(self._make_goal_pose(point))
        self.get_logger().info(
            f"Mission goal ({reason}): ({point[0]:.1f},{point[1]:.1f})"
        )

    def _publish_return_goal(self, reason: str) -> None:
        if not self.return_enabled or self.return_goal_sent:
            return
        self._publish_mission_goal(self.return_point, f"final_return:{reason}")
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
        self._reset_hybrid_yaw_control()
        self.fire_started_wall = None
        self.fire_until_wall = None
        self.post_fire_until_wall = None
        self.lowering_started_wall = None
        self.lowered_since_wall = None
        self.lowering_reason = None
        self.centering_started_wall = None
        self.centered_since_wall = None
        self.centering_reason = None
        self.last_result = None
        self.reposition_attempts = 0
        self.reposition_goal = None
        self.reposition_start_pose = None
        self.reposition_started_wall = None
        self.reposition_reason = None
        self.last_reposition_solution = None

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
            # A temporary direct reposition goal must never become the next
            # engagement goal.  Restore the normal checkpoint route here.
            self._publish_mission_goal(next_stage["checkpoint"], "next_engagement_checkpoint")
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

    def _begin_post_fire_recovery(self, reason: str) -> None:
        """Start the safe post-shot sequence: lower barrel, then center yaw."""
        if self.lower_barrel_after_engagement:
            self._begin_lowering_barrel(reason)
        else:
            self._begin_turret_centering(reason)

    def _center_target_yaw_deg(self) -> float:
        """Return the turret-yaw feedback target for the hull-forward direction.

        ``playerTurretX`` / ``get_action_turret.x`` are absolute map headings in
        the current simulator.  If a future simulator reports relative yaw, the
        existing ``turret_yaw_feedback_is_world`` parameter keeps this target in
        the corresponding convention.
        """
        if self.turret_yaw_feedback_is_world:
            return normalize_180(self.body_yaw_deg + self.center_turret_heading_offset_deg)
        return normalize_180(self.center_turret_heading_offset_deg)

    def _begin_turret_centering(self, reason: str) -> None:
        if not self.center_turret_after_engagement:
            self._advance_or_return(reason)
            return
        self.phase = "centering_turret"
        self.centering_reason = reason
        self.centering_started_wall = time.monotonic()
        self.centered_since_wall = None
        self.get_logger().info(
            f"Centering turret after stage={self.stage_index + 1}: "
            f"target world-yaw={self._center_target_yaw_deg():.2f} deg"
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

    @staticmethod
    def _local_gun_direction(
        relative_yaw_deg: float,
        relative_pitch_deg: float,
    ) -> Tuple[float, float, float]:
        """Return a unit muzzle direction in the hull-local frame.

        Hull-local axes are right(+x), forward(+y), up(+z); this is the same
        convention used by ``_world_from_body_rotation``.
        """
        yaw = math.radians(relative_yaw_deg)
        pitch = math.radians(relative_pitch_deg)
        horizontal = math.cos(pitch)
        return (
            math.sin(yaw) * horizontal,
            math.cos(yaw) * horizontal,
            math.sin(pitch),
        )

    def _world_pitch_from_local_gun(
        self,
        relative_yaw_deg: float,
        relative_pitch_deg: float,
    ) -> float:
        world_dir = self._mat_vec(
            self._world_from_body_rotation(),
            self._local_gun_direction(relative_yaw_deg, relative_pitch_deg),
        )
        return math.degrees(math.atan2(world_dir[2], math.hypot(world_dir[0], world_dir[1])))

    def _effective_pitch_limits(self, relative_yaw_deg: float) -> Dict[str, float]:
        """Mechanical local limits expressed in the active feedback frame.

        The physical F/R limits remain hull-relative.  On a slope, however,
        their equivalent map/world elevation depends on hull pitch, roll, and
        turret yaw.  Publishing both frames prevents the controller from
        treating a world-frame telemetry value as a fixed local limit.
        """
        world_at_min = self._world_pitch_from_local_gun(
            relative_yaw_deg, self.min_pitch_deg
        )
        world_at_max = self._world_pitch_from_local_gun(
            relative_yaw_deg, self.max_pitch_deg
        )
        return {
            "mechanical_min_pitch_deg": self.min_pitch_deg,
            "mechanical_max_pitch_deg": self.max_pitch_deg,
            "world_min_pitch_deg": min(world_at_min, world_at_max),
            "world_max_pitch_deg": max(world_at_min, world_at_max),
            "world_pitch_at_mechanical_min_deg": world_at_min,
            "world_pitch_at_mechanical_max_deg": world_at_max,
        }

    def _feedback_pitch_for_local_command(
        self,
        relative_yaw_deg: float,
        local_pitch_deg: float,
    ) -> float:
        if self.turret_pitch_feedback_is_world:
            return self._world_pitch_from_local_gun(relative_yaw_deg, local_pitch_deg)
        return local_pitch_deg

    def _turret_relative_yaw_from_feedback(self) -> float:
        if self.turret_yaw_deg is None:
            return 0.0
        if self.turret_yaw_feedback_is_world:
            return normalize_180(self.turret_yaw_deg - self.body_yaw_deg)
        return normalize_180(self.turret_yaw_deg)

    def _stage_reposition_config(self) -> Dict[str, Any]:
        raw = self.stage.get("reposition", {})
        raw = raw if isinstance(raw, dict) else {}
        heading = as_finite_float(raw.get("heading_deg"))
        goal_offset = as_finite_float(raw.get("goal_offset_m"))
        min_travel = as_finite_float(raw.get("min_travel_m"))
        arrival_radius = as_finite_float(raw.get("arrival_radius_m"))
        timeout = as_finite_float(raw.get("timeout_sec"))
        max_attempts = as_finite_float(raw.get("max_attempts"))
        return {
            "enabled": bool(raw.get("enabled", self.reposition_on_unreachable_pitch)),
            "heading_deg": heading,
            "goal_offset_m": max(
                0.5,
                float(self.reposition_goal_offset_m if goal_offset is None else goal_offset),
            ),
            "min_travel_m": max(
                0.0,
                float(self.reposition_min_travel_m if min_travel is None else min_travel),
            ),
            "arrival_radius_m": max(
                0.5,
                float(self.reposition_arrival_radius_m if arrival_radius is None else arrival_radius),
            ),
            "timeout_sec": max(
                1.0,
                float(self.reposition_timeout_sec if timeout is None else timeout),
            ),
            "max_attempts": max(
                0,
                int(self.reposition_max_attempts if max_attempts is None else max_attempts),
            ),
        }

    # SCENARIO2_FIXED_FALLBACK_55_230
    def _stage_fixed_fallback_goals(self) -> List[Tuple[float, float]]:
        """Return explicit stage fallback firing positions in declared order.

        A stage that declares ``fallback_goals`` intentionally does *not* fall
        back to the generic heading-offset reposition.  This prevents a failed
        fixed firing point from silently producing the old northbound candidate.
        """
        raw = self.stage.get("reposition", {})
        raw = raw if isinstance(raw, dict) else {}
        values = raw.get("fallback_goals", [])
        if not isinstance(values, list):
            return []

        goals: List[Tuple[float, float]] = []
        for value in values:
            if not isinstance(value, dict):
                continue
            x = as_finite_float(value.get("x"))
            y = as_finite_float(value.get("y"))
            if x is None or y is None:
                continue
            goals.append((float(x), float(y)))
        return goals

    def _start_reposition_for_pitch_limit(
        self,
        solution: Dict[str, Any],
        reason: str,
        now: float,
    ) -> bool:
        """Release the hull and request a short direct route-local reposition.

        This is intentionally a *planner* request rather than a raw W command:
        the A* stack retains collision checks and the normal controller retains
        its steering/safety logic.
        """
        if self.player_pose is None:
            return False
        cfg = self._stage_reposition_config()
        if not cfg["enabled"] or self.reposition_attempts >= cfg["max_attempts"]:
            return False

        start = (self.player_pose[0], self.player_pose[1])
        fixed_fallback_goals = self._stage_fixed_fallback_goals()
        if fixed_fallback_goals:
            # Explicit fallback positions are authoritative for this stage.
            # Once they are exhausted we report the pitch-limit failure rather
            # than recreating the previous generic northbound offset goal.
            if self.reposition_attempts >= len(fixed_fallback_goals):
                return False
            goal = fixed_fallback_goals[self.reposition_attempts]
            goal_source = f"fixed_fallback[{self.reposition_attempts + 1}/{len(fixed_fallback_goals)}]"
        else:
            heading_deg = cfg["heading_deg"]
            if heading_deg is None:
                # The incoming hull heading is the best route-tangent fallback
                # at a checkpoint; scenario files may override it explicitly.
                heading_deg = self.body_yaw_deg
            heading = math.radians(float(heading_deg))
            goal = (
                start[0] + math.sin(heading) * cfg["goal_offset_m"],
                start[1] + math.cos(heading) * cfg["goal_offset_m"],
            )
            goal_source = f"heading_offset:{float(heading_deg):.1f}deg"
        self.reposition_attempts += 1
        self.reposition_start_pose = start
        self.reposition_goal = goal
        self.reposition_started_wall = now
        self.reposition_reason = reason
        self.last_reposition_solution = dict(solution)
        self.phase = "reposition_for_shot"
        self._reset_aim_dwell()
        self._reset_hybrid_yaw_control()
        self.reposition_goal_pub.publish(self._make_goal_pose(goal))
        self.get_logger().warn(
            f"Pitch limit at stage={self.stage_index + 1}/{len(self.engagements)} "
            f"({reason}); direct route-local reposition {self.reposition_attempts}/{cfg['max_attempts']} "
            f"mode={goal_source} target=({goal[0]:.1f},{goal[1]:.1f})"
        )
        return True

    def _reposition_status(self, now: float) -> Dict[str, Any]:
        cfg = self._stage_reposition_config()
        start = self.reposition_start_pose
        goal = self.reposition_goal
        current = (self.player_pose[0], self.player_pose[1]) if self.player_pose else None
        travelled = (
            math.hypot(current[0] - start[0], current[1] - start[1])
            if current is not None and start is not None else 0.0
        )
        distance_to_goal = (
            math.hypot(current[0] - goal[0], current[1] - goal[1])
            if current is not None and goal is not None else None
        )
        return {
            "attempt": self.reposition_attempts,
            "max_attempts": cfg["max_attempts"],
            "reason": self.reposition_reason,
            "goal_source": "fixed_fallback" if self._stage_fixed_fallback_goals() else "heading_offset",
            "goal": {"x": goal[0], "y": goal[1]} if goal else None,
            "start": {"x": start[0], "y": start[1]} if start else None,
            "travelled_m": round(travelled, 3),
            "min_travel_m": cfg["min_travel_m"],
            "distance_to_goal_m": round(distance_to_goal, 3) if distance_to_goal is not None else None,
            "arrival_radius_m": cfg["arrival_radius_m"],
            "elapsed_sec": round(max(0.0, now - (self.reposition_started_wall or now)), 3),
            "timeout_sec": cfg["timeout_sec"],
        }

    def _desired_solution(self) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """Solve a world ballistic arc and express it in the feedback frame.

        Gun travel limits are mechanical hull-relative limits.  The simulator
        reports pitch as a world elevation by default, so a slope changes the
        *world* minimum/maximum even though the local F/R limits do not move.
        The returned solution therefore carries both frames and reports a
        structured pitch-limit reason instead of blindly driving F/R.
        """
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

        # playerTurretX is absolute map yaw in the current simulator.
        target_feedback_yaw_deg = (
            normalize_180(self.body_yaw_deg + target_relative_yaw_deg)
            if self.turret_yaw_feedback_is_world else target_relative_yaw_deg
        )
        target_feedback_pitch_deg = self._feedback_pitch_for_local_command(
            target_relative_yaw_deg, target_relative_pitch_deg
        )
        limits = self._effective_pitch_limits(target_relative_yaw_deg)
        solution: Dict[str, Any] = {
            "distance_m": distance,
            "target_yaw_deg": target_feedback_yaw_deg,
            "target_pitch_deg": target_feedback_pitch_deg,
            "target_control_pitch_deg": target_feedback_pitch_deg,
            "target_world_yaw_deg": world_yaw_deg,
            "target_world_pitch_deg": world_pitch_deg,
            "target_relative_yaw_deg": target_relative_yaw_deg,
            "target_relative_pitch_deg": target_relative_pitch_deg,
            "pitch_feedback_frame": (
                "world" if self.turret_pitch_feedback_is_world else "hull_relative"
            ),
            "target_z_m": target_z,
            "muzzle_x_m": muzzle_x,
            "muzzle_y_m": muzzle_y,
            "muzzle_z_m": muzzle_z,
            "muzzle_offset_world_x_m": muzzle_offset_world[0],
            "muzzle_offset_world_y_m": muzzle_offset_world[1],
            "muzzle_offset_world_z_m": muzzle_offset_world[2],
            **limits,
        }
        # IMPORTANT: these local mechanical numbers are diagnostic only here.
        # Do not reposition based on body attitude / transformed local angle.
        # The simulator's *actual* turret elevation feedback decides whether an
        # R/F hard stop has been reached in the closed-loop aiming state.
        if target_relative_pitch_deg < self.min_pitch_deg:
            solution["theoretical_relative_pitch_limit"] = "below_min"
        elif target_relative_pitch_deg > self.max_pitch_deg:
            solution["theoretical_relative_pitch_limit"] = "above_max"
        return solution, None



    # HYBRID_YAW_DELAY_COMPENSATION_V1
    def _reset_hybrid_yaw_control(self) -> None:
        """Clear per-engagement delayed-feedback yaw-control state."""
        self._aim_yaw_mode = "track"
        self._aim_yaw_reason = "reset"
        self._aim_yaw_prev_error_deg: Optional[float] = None
        self._aim_yaw_last_processed_feedback_seq = -1
        self._aim_yaw_brake_until = 0.0
        self._aim_yaw_wait_feedback_seq = -1
        self._aim_yaw_pulse_command = ""
        self._aim_yaw_pulse_until = 0.0
        self._aim_yaw_pulse_sec = 0.0
        self._aim_yaw_observe_until = 0.0

    def _start_yaw_brake(self, now: float, reason: str) -> None:
        """Release Q/E and wait for a post-release feedback sample."""
        self._aim_yaw_mode = "brake"
        self._aim_yaw_reason = reason
        self._aim_yaw_brake_until = now + self.yaw_overshoot_brake_sec
        self._aim_yaw_wait_feedback_seq = self._turret_feedback_seq
        self._aim_yaw_pulse_command = ""
        self._aim_yaw_pulse_until = 0.0

    def _enter_yaw_observe(self, now: float, reason: str) -> None:
        """Keep Q/E neutral until a new sample confirms the result."""
        self._aim_yaw_mode = "observe"
        self._aim_yaw_reason = reason
        self._aim_yaw_observe_until = now + self.yaw_observe_sec
        self._aim_yaw_wait_feedback_seq = self._turret_feedback_seq
        self._aim_yaw_pulse_command = ""
        self._aim_yaw_pulse_until = 0.0

    def _start_yaw_pulse(self, yaw_error_deg: float, now: float, reason: str) -> None:
        """Schedule a bounded low-speed correction pulse from measured error."""
        command = "E" if yaw_error_deg > 0.0 else "Q"
        calibrated_rate = (
            self.yaw_pulse_rate_e_deg_s if command == "E"
            else self.yaw_pulse_rate_q_deg_s
        )
        pulse_sec = self.yaw_pulse_gain * abs(yaw_error_deg) / calibrated_rate
        pulse_sec = clamp(pulse_sec, self.yaw_pulse_min_sec, self.yaw_pulse_max_sec)
        self._aim_yaw_mode = "pulse"
        self._aim_yaw_reason = reason
        self._aim_yaw_pulse_command = command
        self._aim_yaw_pulse_sec = pulse_sec
        self._aim_yaw_pulse_until = now + pulse_sec

    def _hybrid_aim_yaw_control(
        self,
        yaw_error_deg: float,
        now: float,
    ) -> Tuple[str, float, Dict[str, Any], bool]:
        """Return Q/E command and whether yaw is measured stable for firing.

        ``track`` is ordinary real-time closed-loop control for a large yaw
        error.  A target crossing, or even merely entering the fire tolerance,
        switches to neutral/brake.  Any remaining error is corrected by a
        short Q/E pulse computed from calibrated degrees-per-second, then the
        controller waits for fresh feedback before another decision.
        """
        feedback_seq = self._turret_feedback_seq
        new_feedback = feedback_seq != self._aim_yaw_last_processed_feedback_seq
        previous_error = self._aim_yaw_prev_error_deg
        if new_feedback:
            self._aim_yaw_prev_error_deg = yaw_error_deg
            self._aim_yaw_last_processed_feedback_seq = feedback_seq

        overshoot = (
            new_feedback
            and previous_error is not None
            and previous_error * yaw_error_deg < 0.0
            and abs(previous_error) >= self.yaw_overshoot_min_prev_error_deg
            and abs(yaw_error_deg) >= self.yaw_overshoot_min_current_error_deg
        )
        in_tolerance = abs(yaw_error_deg) <= self.yaw_tolerance_deg

        def debug(extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
            payload: Dict[str, Any] = {
                "mode": self._aim_yaw_mode,
                "reason": self._aim_yaw_reason,
                "ready": self._aim_yaw_mode == "settled",
                "new_feedback": new_feedback,
                "feedback_seq": feedback_seq,
                "yaw_rate_deg_s": round(self.turret_yaw_rate_deg_s, 3),
                "in_tolerance": in_tolerance,
                "overshoot_detected": overshoot,
                "brake_remaining_sec": round(max(0.0, self._aim_yaw_brake_until - now), 3),
                "pulse_remaining_sec": round(max(0.0, self._aim_yaw_pulse_until - now), 3),
                "pulse_sec": round(self._aim_yaw_pulse_sec, 3),
                "observe_remaining_sec": round(max(0.0, self._aim_yaw_observe_until - now), 3),
                "wait_feedback_after_seq": self._aim_yaw_wait_feedback_seq,
                "settle_rate_limit_deg_s": self.yaw_settle_rate_deg_s,
            }
            if extra:
                payload.update(extra)
            return payload

        if self._aim_yaw_mode == "track":
            if overshoot:
                self._start_yaw_brake(now, "overshoot_sign_change")
                return "", 0.0, debug(), False
            if in_tolerance:
                # Do not advance to pitch while the last Q/E command may still
                # be acting in the simulator.  Release and observe first.
                self._start_yaw_brake(now, "entered_yaw_tolerance")
                return "", 0.0, debug(), False
            weight = (
                0.0 if abs(yaw_error_deg) <= self.yaw_control_deadband_deg
                else self._yaw_weight(yaw_error_deg, self.yaw_weight_max)
            )
            command = "" if weight <= 0.0 else ("E" if yaw_error_deg > 0.0 else "Q")
            return command, weight, debug({"command_source": "closed_loop_track"}), False

        if self._aim_yaw_mode == "brake":
            post_release_feedback = feedback_seq > self._aim_yaw_wait_feedback_seq
            if now < self._aim_yaw_brake_until or not post_release_feedback:
                return "", 0.0, debug({"command_source": "neutral_brake"}), False
            if in_tolerance:
                self._enter_yaw_observe(now, "brake_complete_in_tolerance")
                return "", 0.0, debug({"command_source": "neutral_observe"}), False
            self._start_yaw_pulse(yaw_error_deg, now, "post_brake_correction")
            return (
                self._aim_yaw_pulse_command,
                self.yaw_pulse_weight,
                debug({"command_source": "time_based_pulse"}),
                False,
            )

        if self._aim_yaw_mode == "pulse":
            if now < self._aim_yaw_pulse_until:
                return (
                    self._aim_yaw_pulse_command,
                    self.yaw_pulse_weight,
                    debug({"command_source": "time_based_pulse"}),
                    False,
                )
            self._enter_yaw_observe(now, "pulse_complete")
            return "", 0.0, debug({"command_source": "neutral_observe"}), False

        if self._aim_yaw_mode == "observe":
            post_pulse_feedback = feedback_seq > self._aim_yaw_wait_feedback_seq
            if now < self._aim_yaw_observe_until or not post_pulse_feedback:
                return "", 0.0, debug({"command_source": "neutral_observe"}), False
            if in_tolerance:
                if abs(self.turret_yaw_rate_deg_s) <= self.yaw_settle_rate_deg_s:
                    self._aim_yaw_mode = "settled"
                    self._aim_yaw_reason = "measured_angle_and_rate_settled"
                    return "", 0.0, debug({"command_source": "neutral_settled"}), True
                # The received angle is close but still moving.  Keep Q/E
                # released and require one more fresh sample instead of firing.
                self._enter_yaw_observe(now, "in_tolerance_waiting_for_low_rate")
                return "", 0.0, debug({"command_source": "neutral_observe"}), False
            self._start_yaw_pulse(yaw_error_deg, now, "observe_error_correction")
            return (
                self._aim_yaw_pulse_command,
                self.yaw_pulse_weight,
                debug({"command_source": "time_based_pulse"}),
                False,
            )

        # settled
        if in_tolerance and abs(self.turret_yaw_rate_deg_s) <= self.yaw_settle_rate_deg_s:
            return "", 0.0, debug({"command_source": "neutral_settled"}), True
        if in_tolerance:
            self._enter_yaw_observe(now, "settled_but_rate_increased")
            return "", 0.0, debug({"command_source": "neutral_observe"}), False
        # A late sample can reveal residual error after settling.  Correct that
        # residual with a bounded pulse, never by returning to high-speed track.
        self._start_yaw_pulse(yaw_error_deg, now, "late_residual_correction")
        return (
            self._aim_yaw_pulse_command,
            self.yaw_pulse_weight,
            debug({"command_source": "time_based_pulse"}),
            False,
        )

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

    # SCENARIO2_PHYSICAL_PITCH_LIMIT_ONLY
    def _reset_physical_pitch_watch(self) -> None:
        """Clear the R/F movement watchdog used only while pitch is commanded."""
        self._physical_pitch_watch = None

    def _observe_physical_pitch_watch(
        self,
        *,
        now: float,
        command: str,
        current_pitch_deg: Optional[float],
        target_pitch_deg: float,
    ) -> Dict[str, Any]:
        """Detect a *real* R/F hard stop from turret feedback.

        This deliberately does not use body pitch/roll or nominal local gun
        limits as a decision.  A hard stop exists only when the same R/F
        direction has been commanded and turret-elevation feedback has not
        changed by ``motion_epsilon_deg`` for ``stall_sec``.
        """
        stall_sec = 0.80
        motion_epsilon_deg = 0.08

        if command not in {"R", "F"} or current_pitch_deg is None:
            self._reset_physical_pitch_watch()
            return {
                "active": False,
                "stalled": False,
                "reason": "no_pitch_command",
                "stall_sec": stall_sec,
                "motion_epsilon_deg": motion_epsilon_deg,
            }

        watch = getattr(self, "_physical_pitch_watch", None)
        if not isinstance(watch, dict) or watch.get("command") != command:
            watch = {
                "command": command,
                "baseline_pitch_deg": float(current_pitch_deg),
                "started_wall": now,
                "last_motion_wall": now,
            }
            self._physical_pitch_watch = watch

        baseline = float(watch["baseline_pitch_deg"])
        change_deg = float(current_pitch_deg) - baseline
        if abs(change_deg) >= motion_epsilon_deg:
            # The gun is still physically moving.  Restart only the no-motion
            # timer, preserving the original command direction for debugging.
            watch["baseline_pitch_deg"] = float(current_pitch_deg)
            watch["last_motion_wall"] = now
            change_deg = 0.0

        no_motion_sec = max(0.0, now - float(watch["last_motion_wall"]))
        stalled = no_motion_sec >= stall_sec
        return {
            "active": True,
            "stalled": stalled,
            "command": command,
            "current_pitch_deg": float(current_pitch_deg),
            "target_pitch_deg": float(target_pitch_deg),
            "baseline_pitch_deg": float(watch["baseline_pitch_deg"]),
            "change_since_baseline_deg": round(change_deg, 4),
            "no_motion_sec": round(no_motion_sec, 3),
            "stall_sec": stall_sec,
            "motion_epsilon_deg": motion_epsilon_deg,
            "reason": "feedback_stalled_at_physical_endstop" if stalled else "feedback_moving_or_waiting",
        }

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
            "pitch_feedback_frame": (
                "world" if self.turret_pitch_feedback_is_world else "hull_relative"
            ),
            "reposition": self._reposition_status(time.monotonic()),
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

        # A pitch-limit reposition publishes a temporary direct planner goal.
        # Do not use the normal route-waypoint mode here: otherwise the planner
        # can jump ahead to the next engagement checkpoint before the gun has
        # re-acquired a feasible attitude.
        if self.phase == "reposition_for_shot":
            cfg = self._stage_reposition_config()
            rstatus = self._reposition_status(now)
            travelled = float(rstatus.get("travelled_m") or 0.0)
            distance_to_goal = rstatus.get("distance_to_goal_m")
            arrived = (
                distance_to_goal is not None
                and float(distance_to_goal) <= cfg["arrival_radius_m"]
                and travelled >= cfg["min_travel_m"]
            )
            timed_out = (
                self.reposition_started_wall is not None
                and now - self.reposition_started_wall >= cfg["timeout_sec"]
            )
            if arrived:
                self.phase = "settling"
                self.checkpoint_enter_wall = now
                status = self._stage_status_base(active=True)
                status.update({
                    "phase": "settling_after_reposition",
                    "reposition": rstatus,
                    "reason": "reposition_arrived_reacquire_aim",
                })
                self._publish_override(
                    active=True, hold_motion=True,
                    turret_qe=self._axis("", 0.0), turret_rf=self._axis("", 0.0),
                    fire=False, status=status,
                )
                return
            if timed_out:
                self.get_logger().warn(
                    f"Reposition timeout at stage={self.stage_index + 1}; rechecking pitch solution"
                )
                self.phase = "aim"
                self._reset_aim_dwell()
                status = self._stage_status_base(active=True)
                status.update({
                    "phase": "aim",
                    "reposition": rstatus,
                    "reason": "reposition_timeout_recheck",
                })
                self._publish_override(
                    active=True, hold_motion=True,
                    turret_qe=self._axis("", 0.0), turret_rf=self._axis("", 0.0),
                    fire=False, status=status,
                )
                return
            status = self._stage_status_base(active=False)
            status.update({
                "phase": "reposition_for_shot",
                "reposition": rstatus,
                "reason": self.reposition_reason,
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
            self._reset_hybrid_yaw_control()
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
            self._begin_post_fire_recovery(self.phase)

        if self.phase == "lowering_barrel":
            feedback_age = now - self.turret_feedback_wall
            feedback_fresh = (
                self.turret_elevation_deg is not None
                and feedback_age <= self.turret_feedback_ttl_sec
            )
            current_pitch = self.turret_elevation_deg if feedback_fresh else None
            lower_relative_yaw_deg = self._turret_relative_yaw_from_feedback()
            lower_target_feedback_pitch_deg = self._feedback_pitch_for_local_command(
                lower_relative_yaw_deg, self.lower_barrel_target_deg
            )
            reached = (
                current_pitch is not None
                and current_pitch <= lower_target_feedback_pitch_deg + self.lower_barrel_tolerance_deg
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
                "lower_barrel_target_relative_deg": self.lower_barrel_target_deg,
                "lower_barrel_target_feedback_pitch_deg": lower_target_feedback_pitch_deg,
                "lower_barrel_relative_yaw_deg": lower_relative_yaw_deg,
                "pitch_feedback_frame": (
                    "world" if self.turret_pitch_feedback_is_world else "hull_relative"
                ),
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
                self._begin_turret_centering(reason)
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
                self._begin_turret_centering(reason)
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

        # Post-shot turret home: keep the hull stopped until the gun yaw is
        # aligned with the current hull-forward heading.  Pitch remains at the
        # previously lowered value; no R/F command is issued here.
        if self.phase == "centering_turret":
            feedback_age = now - self.turret_feedback_wall
            yaw_feedback_fresh = (
                self.turret_yaw_deg is not None
                and feedback_age <= self.turret_feedback_ttl_sec
            )
            body_age = now - self.body_attitude_wall
            body_heading_fresh = body_age <= self.body_attitude_ttl_sec
            target_yaw = self._center_target_yaw_deg()
            current_yaw = self.turret_yaw_deg if yaw_feedback_fresh else None
            yaw_error = (
                normalize_180(target_yaw - current_yaw)
                if current_yaw is not None else None
            )
            reached = (
                yaw_error is not None
                and abs(yaw_error) <= self.center_turret_tolerance_deg
            )
            timed_out = (
                self.centering_started_wall is not None
                and now - self.centering_started_wall >= self.center_turret_timeout_sec
            )
            if reached:
                if self.centered_since_wall is None:
                    self.centered_since_wall = now
                stable_sec = now - self.centered_since_wall
            else:
                self.centered_since_wall = None
                stable_sec = 0.0

            yaw_weight = (
                0.0 if yaw_error is None or reached
                else self._yaw_weight(yaw_error, self.center_turret_weight_max)
            )
            yaw_cmd = "" if yaw_weight <= 0.0 or yaw_error is None else (
                "E" if yaw_error > 0.0 else "Q"
            )
            status = self._stage_status_base(active=True)
            status.update({
                "phase": "centering_turret",
                "centering_reason": self.centering_reason,
                "center_target_yaw_deg": target_yaw,
                "current_yaw_deg": current_yaw,
                "center_yaw_error_deg": yaw_error,
                "center_tolerance_deg": self.center_turret_tolerance_deg,
                "center_reached": reached,
                "center_stable_sec": round(stable_sec, 3),
                "center_stable_sec_required": self.center_turret_stable_sec,
                "turret_feedback_age_sec": round(max(0.0, feedback_age), 3),
                "turret_feedback_source": self.turret_feedback_source,
                "body_heading_age_sec": round(max(0.0, body_age), 3),
                "body_heading_fresh": body_heading_fresh,
                "command": {
                    "turretQE": {"command": yaw_cmd, "weight": yaw_weight},
                    "turretRF": {"command": "", "weight": 0.0},
                },
            })
            if reached and stable_sec >= self.center_turret_stable_sec:
                reason = f"{self.centering_reason or 'engagement'}:turret_centered"
                self._advance_or_return(reason)
                next_active = self.phase not in {"approach", "returning", "complete"}
                next_status = self._stage_status_base(active=next_active)
                next_status.update({
                    "phase": self.phase,
                    "active": next_active,
                    "return_goal_sent": self.return_goal_sent,
                })
                self._publish_override(
                    active=next_active, hold_motion=next_active,
                    turret_qe=self._axis("", 0.0), turret_rf=self._axis("", 0.0),
                    fire=False, status=next_status,
                )
                return
            if timed_out:
                self.get_logger().warn(
                    f"Turret-centering timeout at stage={self.stage_index + 1}; continuing mission"
                )
                reason = f"{self.centering_reason or 'engagement'}:turret_center_timeout"
                self._advance_or_return(reason)
                next_active = self.phase not in {"approach", "returning", "complete"}
                next_status = self._stage_status_base(active=next_active)
                next_status.update({
                    "phase": self.phase,
                    "active": next_active,
                    "return_goal_sent": self.return_goal_sent,
                })
                self._publish_override(
                    active=next_active, hold_motion=next_active,
                    turret_qe=self._axis("", 0.0), turret_rf=self._axis("", 0.0),
                    fire=False, status=next_status,
                )
                return
            self._publish_override(
                active=True, hold_motion=True,
                turret_qe=self._axis(yaw_cmd, yaw_weight),
                turret_rf=self._axis("", 0.0),
                fire=False, status=status,
            )
            return

        # Closed-loop aiming stage.
        #
        # Axis order is intentional:
        #   1) Q/E must align yaw first.
        #   2) Only then is R/F allowed to move.
        #   3) A physical limit is confirmed exclusively by *unchanged turret
        #      elevation feedback* while R or F is being commanded.
        #
        # Chassis pitch/roll remains part of ballistic target geometry, but it
        # never independently triggers a reposition.
        solution, reason = self._desired_solution()
        if reason is not None or solution is None:
            self.phase = "aim_error"
            status = self._stage_status_base(active=True)
            if solution is not None:
                status.update(solution)
            status.update({
                "phase": "aim_error",
                "reason": reason or "empty_solution",
                "physical_pitch_limit": {
                    "active": False,
                    "stalled": False,
                    "reason": "no_ballistic_solution",
                },
            })
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
            self._reset_physical_pitch_watch()
            self._reset_hybrid_yaw_control()
            status = self._stage_status_base(active=True)
            status.update(solution)
            status.update({
                "phase": "aim",
                "reason": "no_turret_feedback" if (
                    self.turret_yaw_deg is None or self.turret_elevation_deg is None
                ) else "turret_feedback_stale",
                "turret_feedback_age_sec": round(max(0.0, feedback_age), 3),
                "physical_pitch_limit": {
                    "active": False,
                    "stalled": False,
                    "reason": "feedback_not_fresh",
                },
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
        yaw_in_tolerance = abs(yaw_error) <= self.yaw_tolerance_deg
        pitch_aligned = abs(pitch_error) <= self.pitch_tolerance_deg

        yaw_cmd = ""
        yaw_weight = 0.0
        pitch_cmd = ""
        pitch_weight = 0.0
        yaw_control_debug: Dict[str, Any]
        if self.hybrid_yaw_enabled:
            yaw_cmd, yaw_weight, yaw_control_debug, yaw_aligned = self._hybrid_aim_yaw_control(
                yaw_error, now
            )
        else:
            yaw_aligned = yaw_in_tolerance
            yaw_control_debug = {
                "mode": "standard_track" if not yaw_aligned else "yaw_aligned",
                "reason": "hybrid_yaw_disabled",
                "ready": yaw_aligned,
                "yaw_rate_deg_s": round(self.turret_yaw_rate_deg_s, 3),
            }

        physical_pitch_limit: Dict[str, Any]
        aim_axis = "on_target"

        if not yaw_aligned:
            # Never mix Q/E and R/F during acquisition or while a delayed
            # feedback correction is braking/observing.  This prevents pitch
            # movement while yaw is still passing through the target.
            self._reset_physical_pitch_watch()
            if not self.hybrid_yaw_enabled:
                yaw_weight = (
                    0.0 if abs(yaw_error) <= self.yaw_control_deadband_deg
                    else self._yaw_weight(yaw_error, self.yaw_weight_max)
                )
                yaw_cmd = ("E" if yaw_error > 0.0 else "Q") if yaw_weight > 0.0 else ""
            physical_pitch_limit = {
                "active": False,
                "stalled": False,
                "reason": "waiting_for_yaw_stable_alignment",
            }
            aim_axis = "yaw"
        elif not pitch_aligned:
            # Actual R/F travel test: stay on the yaw target and watch the
            # simulator feedback.  Only a proven no-motion endstop can move
            # the vehicle to a fallback firing position.
            pitch_cmd = "R" if pitch_error > 0.0 else "F"
            pitch_weight = self._pitch_weight(pitch_error, self.pitch_weight_max)
            physical_pitch_limit = self._observe_physical_pitch_watch(
                now=now,
                command=pitch_cmd if pitch_weight > 0.0 else "",
                current_pitch_deg=self.turret_elevation_deg,
                target_pitch_deg=solution["target_pitch_deg"],
            )
            aim_axis = "pitch"

            if physical_pitch_limit.get("stalled", False):
                stall_reason = (
                    "physical_pitch_endstop:"
                    f"cmd={pitch_cmd},current={self.turret_elevation_deg:.3f},"
                    f"target={solution['target_pitch_deg']:.3f},"
                    f"error={pitch_error:.3f}"
                )
                if self._start_reposition_for_pitch_limit(solution, stall_reason, now):
                    status = self._stage_status_base(active=False)
                    status.update(solution)
                    status.update({
                        "phase": "reposition_for_shot",
                        "reason": stall_reason,
                        "aim_axis": "physical_pitch_endstop",
                        "physical_pitch_limit": physical_pitch_limit,
                        "reposition": self._reposition_status(now),
                    })
                    self._publish_override(
                        active=False, hold_motion=False,
                        turret_qe=self._axis("", 0.0), turret_rf=self._axis("", 0.0),
                        fire=False, status=status,
                    )
                    return

                self.phase = "aim_error"
                status = self._stage_status_base(active=True)
                status.update(solution)
                status.update({
                    "phase": "aim_error",
                    "reason": f"no_fallback_after_{stall_reason}",
                    "aim_axis": "physical_pitch_endstop",
                    "physical_pitch_limit": physical_pitch_limit,
                    "reposition": self._reposition_status(now),
                })
                self._publish_override(
                    active=True, hold_motion=True,
                    turret_qe=self._axis("", 0.0), turret_rf=self._axis("", 0.0),
                    fire=False, status=status,
                )
                return
        else:
            self._reset_physical_pitch_watch()
            physical_pitch_limit = {
                "active": False,
                "stalled": False,
                "reason": "target_pitch_reached",
            }

        # ``yaw_aligned`` here means measured angle *and* post-release
        # settling observation, not merely instantaneous angle error.
        on_target = yaw_aligned and pitch_aligned
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
            "aim_axis": aim_axis,
            "current_yaw_deg": self.turret_yaw_deg,
            "current_pitch_deg": self.turret_elevation_deg,
            "pitch_feedback_sign": self.pitch_feedback_sign,
            "body_attitude": self._body_attitude_debug(now),
            "yaw_error_deg": yaw_error,
            "yaw_in_tolerance": yaw_in_tolerance,
            "yaw_ready": yaw_aligned,
            "yaw_rate_deg_s": round(self.turret_yaw_rate_deg_s, 3),
            "pitch_error_deg": pitch_error,
            "on_target": on_target,
            "on_target_cycles": self.on_target_cycles,
            "on_target_cycles_required": self.on_target_cycles_required,
            "on_target_stable_sec": round(stable_sec, 3),
            "aim_stable_sec_required": self.aim_stable_sec,
            "turret_feedback_age_sec": round(max(0.0, feedback_age), 3),
            "turret_feedback_source": self.turret_feedback_source,
            "yaw_control": yaw_control_debug,
            "physical_pitch_limit": physical_pitch_limit,
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
