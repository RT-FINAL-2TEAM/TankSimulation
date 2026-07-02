# -*- coding: utf-8 -*-
"""Emergency brake and replan guard for smartphone detections.

Smartphone detections are treated as emergency evidence, not as ordinary map
annotations.  When a valid phone object/virtual obstacle/synthetic cluster is
present, this node repeatedly publishes a STOP action to ros_bridge's one-shot
/get_action override topic.  It keeps the stop active long enough for the muxed
phone cluster to reach the planner and for dynamic replan to update the route.

The node is deliberately independent from /tank/control/command.  The normal
controller keeps publishing persistent commands; this node only issues short
one-shot overrides through ros_bridge.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, Iterable, Optional, Tuple

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


def _safe_json_loads(data: str, default: Any = None) -> Any:
    try:
        return json.loads(data)
    except Exception:
        return default


def _bool_from_meta(meta: Dict[str, Any], key: str, default: bool = True) -> bool:
    if key not in meta:
        return default
    value = meta.get(key)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    try:
        return bool(value)
    except Exception:
        return default


class PhoneEmergencyBrakeNode(Node):
    def __init__(self) -> None:
        super().__init__("phone_emergency_brake_node")

        self.declare_parameter("detections_topic", "/tank/phone_sim2real/detections")
        self.declare_parameter("virtual_obstacles_topic", "/tank/phone_sim2real/virtual_obstacles")
        self.declare_parameter("synthetic_lidar_clusters_topic", "/tank/phone_sim2real/synthetic_lidar_clusters")
        self.declare_parameter("planner_status_topic", "/tank/planner/status")
        self.declare_parameter("override_topic", "/tank/api/get_action/override")
        self.declare_parameter("status_topic", "/tank/phone_sim2real/emergency_status")

        self.declare_parameter("enable_emergency_brake", True)
        self.declare_parameter("trigger_on_detections", True)
        self.declare_parameter("trigger_on_virtual_obstacles", True)
        self.declare_parameter("trigger_on_synthetic_clusters", True)
        self.declare_parameter("require_inject_enabled", True)
        self.declare_parameter("ignored_classes", "person,human,blue,red")
        self.declare_parameter("min_detection_confidence", 0.50)
        self.declare_parameter("min_virtual_confidence", 0.35)

        # Emergency stop policy.
        # min: always hold this long after a new phone emergency.
        # max: safety upper bound so the phone module cannot freeze the tank forever.
        # While planner reports plan_failed or emergency block, stop is held until max.
        self.declare_parameter("emergency_stop_min_sec", 3.0)
        self.declare_parameter("emergency_stop_max_sec", 6.0)
        self.declare_parameter("phone_signal_ttl_sec", 1.2)
        self.declare_parameter("retrigger_cooldown_sec", 0.5)
        self.declare_parameter("stop_until_route_version_change", True)
        self.declare_parameter("stop_while_plan_failed", True)
        self.declare_parameter("stop_while_emergency_blocked", True)
        self.declare_parameter("stop_while_phone_signal_active", False)
        self.declare_parameter("release_requires_route_change_or_clear", True)
        self.declare_parameter("publish_hz", 30.0)
        self.declare_parameter("status_hz", 5.0)

        self.enable_emergency_brake = bool(self.get_parameter("enable_emergency_brake").value)
        self.trigger_on_detections = bool(self.get_parameter("trigger_on_detections").value)
        self.trigger_on_virtual_obstacles = bool(self.get_parameter("trigger_on_virtual_obstacles").value)
        self.trigger_on_synthetic_clusters = bool(self.get_parameter("trigger_on_synthetic_clusters").value)
        self.require_inject_enabled = bool(self.get_parameter("require_inject_enabled").value)
        self.ignored_classes = {s.strip().lower() for s in str(self.get_parameter("ignored_classes").value).split(",") if s.strip()}
        self.min_detection_confidence = float(self.get_parameter("min_detection_confidence").value)
        self.min_virtual_confidence = float(self.get_parameter("min_virtual_confidence").value)
        self.emergency_stop_min_sec = float(self.get_parameter("emergency_stop_min_sec").value)
        self.emergency_stop_max_sec = float(self.get_parameter("emergency_stop_max_sec").value)
        self.phone_signal_ttl_sec = float(self.get_parameter("phone_signal_ttl_sec").value)
        self.retrigger_cooldown_sec = float(self.get_parameter("retrigger_cooldown_sec").value)
        self.stop_until_route_version_change = bool(self.get_parameter("stop_until_route_version_change").value)
        self.stop_while_plan_failed = bool(self.get_parameter("stop_while_plan_failed").value)
        self.stop_while_emergency_blocked = bool(self.get_parameter("stop_while_emergency_blocked").value)
        self.stop_while_phone_signal_active = bool(self.get_parameter("stop_while_phone_signal_active").value)
        self.release_requires_route_change_or_clear = bool(self.get_parameter("release_requires_route_change_or_clear").value)

        self.override_topic = str(self.get_parameter("override_topic").value)
        self.status_topic = str(self.get_parameter("status_topic").value)
        self.override_pub = self.create_publisher(String, self.override_topic, 10)
        self.status_pub = self.create_publisher(String, self.status_topic, 10)

        self.create_subscription(String, str(self.get_parameter("detections_topic").value), self.detections_cb, 10)
        self.create_subscription(String, str(self.get_parameter("virtual_obstacles_topic").value), self.virtual_cb, 10)
        self.create_subscription(String, str(self.get_parameter("synthetic_lidar_clusters_topic").value), self.cluster_cb, 10)
        self.create_subscription(String, str(self.get_parameter("planner_status_topic").value), self.planner_status_cb, 10)

        pub_hz = max(1.0, float(self.get_parameter("publish_hz").value))
        status_hz = max(1.0, float(self.get_parameter("status_hz").value))
        self.stop_timer = self.create_timer(1.0 / pub_hz, self.timer_cb)
        self.status_timer = self.create_timer(1.0 / status_hz, self.status_timer_cb)

        self.phone_active_until = 0.0
        self.emergency_active = False
        self.emergency_started_wall = 0.0
        self.emergency_min_until = 0.0
        self.emergency_max_until = 0.0
        self.last_release_wall = 0.0
        self.stop_publish_count = 0

        self.start_route_version: Optional[int] = None
        self.last_route_version: Optional[int] = None
        self.last_planner_reason = ""
        self.last_emergency_cluster_blocked = False
        self.last_dynamic_replan_count: Optional[int] = None
        self.route_changed_since_start = False

        self.last_reason = "idle"
        self.last_trigger_source = ""
        self.last_class_name = ""
        self.last_confidence = 0.0
        self.last_object_count = 0
        self.last_cluster_count = 0
        self.last_phone_event_wall = 0.0

        self.get_logger().info(
            "phone emergency brake started: override=%s min=%.1fs max=%.1fs planner=%s"
            % (self.override_topic, self.emergency_stop_min_sec, self.emergency_stop_max_sec, self.get_parameter("planner_status_topic").value)
        )

    @staticmethod
    def _stop_action() -> Dict[str, Any]:
        return {
            "moveWS": {"command": "STOP", "weight": 1.0},
            "moveAD": {"command": "", "weight": 0.0},
            "turretQE": {"command": "", "weight": 0.0},
            "turretRF": {"command": "", "weight": 0.0},
            "fire": False,
        }

    def _publish_stop_override(self) -> None:
        self.override_pub.publish(String(data=json.dumps(self._stop_action(), separators=(",", ":"), ensure_ascii=False)))
        self.stop_publish_count += 1

    def _iter_detection_items(self, payload: Dict[str, Any]) -> Iterable[Tuple[str, float]]:
        for det in payload.get("detections", []) if isinstance(payload.get("detections"), list) else []:
            if not isinstance(det, dict):
                continue
            cls = str(det.get("className") or det.get("class_name") or det.get("label") or det.get("name") or "unknown").lower()
            try:
                conf = float(det.get("confidence", det.get("conf", 0.0)))
            except Exception:
                conf = 0.0
            yield cls, conf

    def _iter_virtual_items(self, payload: Dict[str, Any]) -> Iterable[Tuple[str, float]]:
        for obj in payload.get("objects", []) if isinstance(payload.get("objects"), list) else []:
            if not isinstance(obj, dict):
                continue
            cls = str(obj.get("class_name") or obj.get("className") or "unknown").lower()
            try:
                conf = float(obj.get("confidence", 0.0))
            except Exception:
                conf = 0.0
            yield cls, conf

    def _valid_class_conf(self, items: Iterable[Tuple[str, float]], min_conf: float) -> Tuple[bool, str, float, int]:
        best_cls, best_conf, count = "", 0.0, 0
        for cls, conf in items:
            if cls in self.ignored_classes or conf < min_conf:
                continue
            count += 1
            if conf >= best_conf:
                best_cls, best_conf = cls, conf
        return count > 0, best_cls, best_conf, count

    def _activate_emergency(self, source: str, reason: str, cls: str = "", conf: float = 0.0, count: int = 0) -> None:
        if not self.enable_emergency_brake:
            return
        now = time.time()
        self.phone_active_until = now + self.phone_signal_ttl_sec
        self.last_phone_event_wall = now
        self.last_trigger_source = source
        self.last_reason = reason
        self.last_class_name = cls
        self.last_confidence = float(conf)
        self.last_object_count = int(count)

        if self.emergency_active:
            return
        if now - self.last_release_wall < self.retrigger_cooldown_sec:
            return

        self.emergency_active = True
        self.emergency_started_wall = now
        self.emergency_min_until = now + self.emergency_stop_min_sec
        self.emergency_max_until = now + self.emergency_stop_max_sec
        self.start_route_version = self.last_route_version
        self.route_changed_since_start = False
        self.get_logger().warn(
            "PHONE EMERGENCY STOP: source=%s reason=%s class=%s conf=%.2f count=%d route=%s"
            % (source, reason, cls, conf, count, str(self.start_route_version))
        )

    def detections_cb(self, msg: String) -> None:
        if not self.trigger_on_detections:
            return
        payload = _safe_json_loads(msg.data, {})
        if not isinstance(payload, dict):
            return
        meta = payload.get("phone") if isinstance(payload.get("phone"), dict) else {}
        command = str(payload.get("command", meta.get("command", meta.get("action", meta.get("control", ""))))).strip().lower()
        if command in {"clear", "clear_obstacle", "clear_obstacles", "reset", "inject_off", "injection_off", "disable_injection"}:
            self.emergency_active = False
            self.phone_active_until = 0.0
            self.last_release_wall = time.time()
            self.last_reason = "phone_clear_command"
            return
        if self.require_inject_enabled and not _bool_from_meta(meta, "inject_enabled", True):
            return
        ok, cls, conf, count = self._valid_class_conf(self._iter_detection_items(payload), self.min_detection_confidence)
        if ok:
            self._activate_emergency("detections", "phone_yolo_object_detected", cls, conf, count)

    def virtual_cb(self, msg: String) -> None:
        if not self.trigger_on_virtual_obstacles:
            return
        payload = _safe_json_loads(msg.data, {})
        if not isinstance(payload, dict):
            return
        obj_count = int(payload.get("count", 0) or 0)
        ok, cls, conf, count = self._valid_class_conf(self._iter_virtual_items(payload), self.min_virtual_confidence)
        if ok or obj_count > 0:
            self._activate_emergency("virtual_obstacles", "phone_virtual_obstacle_active", cls, conf, max(count, obj_count))

    def cluster_cb(self, msg: String) -> None:
        if not self.trigger_on_synthetic_clusters:
            return
        payload = _safe_json_loads(msg.data, {})
        if not isinstance(payload, dict):
            return
        cluster_count = int(payload.get("cluster_count", 0) or 0)
        self.last_cluster_count = cluster_count
        if cluster_count > 0:
            self._activate_emergency("synthetic_lidar_clusters", "phone_cluster_published", count=cluster_count)

    def planner_status_cb(self, msg: String) -> None:
        payload = _safe_json_loads(msg.data, {})
        if not isinstance(payload, dict):
            return
        try:
            rv = int(payload.get("route_version"))
        except Exception:
            rv = None
        if rv is not None:
            self.last_route_version = rv
            if self.emergency_active and self.start_route_version is not None and rv != self.start_route_version:
                self.route_changed_since_start = True
        self.last_planner_reason = str(payload.get("reason", ""))
        self.last_emergency_cluster_blocked = bool(payload.get("emergency_cluster_blocked", False))
        try:
            self.last_dynamic_replan_count = int(payload.get("dynamic_replan_count"))
        except Exception:
            pass

    def _should_stop(self) -> bool:
        if not self.enable_emergency_brake or not self.emergency_active:
            return False
        now = time.time()
        if now >= self.emergency_max_until:
            return False
        if now < self.emergency_min_until:
            return True

        reason = self.last_planner_reason.lower()
        plan_failed = "plan_failed" in reason or "failed" in reason
        if self.stop_while_plan_failed and plan_failed:
            return True
        if self.stop_while_emergency_blocked and self.last_emergency_cluster_blocked:
            return True
        if self.stop_while_phone_signal_active and now < self.phone_active_until:
            return True
        if self.stop_until_route_version_change and not self.route_changed_since_start:
            return True
        if self.release_requires_route_change_or_clear and self.start_route_version is not None and not self.route_changed_since_start:
            return True
        # If route changed and minimum hold elapsed, release even if the phone is still seeing the same object.
        return False

    def timer_cb(self) -> None:
        now = time.time()
        if self._should_stop():
            self._publish_stop_override()
            return
        if self.emergency_active:
            self.emergency_active = False
            self.last_release_wall = now
            self.emergency_min_until = 0.0
            self.emergency_max_until = 0.0

    def _status_payload(self) -> Dict[str, Any]:
        now = time.time()
        return {
            "ok": True,
            "source": "phone_sim2real_emergency_brake",
            "timestamp_wall": now,
            "enabled": self.enable_emergency_brake,
            "stop_active": self._should_stop(),
            "emergency_active": self.emergency_active,
            "stop_remaining_sec": max(0.0, self.emergency_max_until - now) if self.emergency_active else 0.0,
            "min_hold_remaining_sec": max(0.0, self.emergency_min_until - now) if self.emergency_active else 0.0,
            "phone_signal_active": now < self.phone_active_until,
            "last_phone_event_age_sec": None if self.last_phone_event_wall <= 0.0 else now - self.last_phone_event_wall,
            "last_reason": self.last_reason,
            "last_trigger_source": self.last_trigger_source,
            "last_class_name": self.last_class_name,
            "last_confidence": self.last_confidence,
            "last_object_count": self.last_object_count,
            "last_cluster_count": self.last_cluster_count,
            "start_route_version": self.start_route_version,
            "last_route_version": self.last_route_version,
            "route_changed_since_start": self.route_changed_since_start,
            "last_planner_reason": self.last_planner_reason,
            "last_emergency_cluster_blocked": self.last_emergency_cluster_blocked,
            "stop_while_emergency_blocked": self.stop_while_emergency_blocked,
            "stop_while_phone_signal_active": self.stop_while_phone_signal_active,
            "release_requires_route_change_or_clear": self.release_requires_route_change_or_clear,
            "last_dynamic_replan_count": self.last_dynamic_replan_count,
            "stop_publish_count": self.stop_publish_count,
            "override_topic": self.override_topic,
        }

    def status_timer_cb(self) -> None:
        self.status_pub.publish(String(data=json.dumps(self._status_payload(), separators=(",", ":"), ensure_ascii=False)))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PhoneEmergencyBrakeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
