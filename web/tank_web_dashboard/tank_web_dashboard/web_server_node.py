# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import os
import time
from copy import deepcopy
from pathlib import Path
from threading import Lock, Thread
from typing import Any, Dict, Optional

from flask import Flask, jsonify, send_file

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from std_msgs.msg import String
from sensor_msgs.msg import CompressedImage

from . import live_view


WEB_HOST = os.environ.get("TANK_WEB_HOST", "0.0.0.0")
WEB_PORT = int(os.environ.get("TANK_WEB_PORT", "5001"))

DASHBOARD_POLL_MS = int(os.environ.get("TANK_DASHBOARD_POLL_MS", "1000"))
LIVE_VIEW_FPS = float(os.environ.get("TANK_LIVE_VIEW_FPS", "8"))
LIVE_VIEW_JPEG_QUALITY = int(os.environ.get("TANK_LIVE_VIEW_JPEG_QUALITY", "65"))

STATIC_MAP_PATH = os.environ.get("TANK_STATIC_MAP_PATH", "").strip()
STATIC_MAP_OVERVIEW_PATH = os.environ.get("TANK_STATIC_MAP_OVERVIEW_PATH", "").strip()


class TankWebDashboardNode(Node):
    def __init__(self) -> None:
        super().__init__("tank_web_dashboard_node")

        self._lock = Lock()

        self._state_latest: Dict[str, Any] = {}
        self._last_state_wall = 0.0

        self._latest_detect_result: Dict[str, Any] = {}
        self._last_detect_wall = 0.0

        self.create_subscription(
            String,
            "/tank/state/latest",
            self.on_state_latest,
            10,
        )

        self.create_subscription(
            CompressedImage,
            "/tank/api/detect/image_compressed",
            self.on_detect_image,
            10,
        )

        self.create_subscription(
            String,
            "/tank/api/detect/result",
            self.on_detect_result,
            10,
        )

        self.get_logger().info("tank_web_dashboard_node started")
        self.get_logger().info("sub: /tank/state/latest")
        self.get_logger().info("sub: /tank/api/detect/image_compressed")
        self.get_logger().info("sub: /tank/api/detect/result")

    def on_state_latest(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
        except Exception as exc:
            self.get_logger().warn(f"failed to parse /tank/state/latest: {exc}")
            return

        with self._lock:
            self._state_latest = data
            self._last_state_wall = time.time()

    def on_detect_image(self, msg: CompressedImage) -> None:
        try:
            # 여기서 웹 컴퓨터의 live_view.py 전역 변수에 최신 프레임 저장
            live_view.update_frame(bytes(msg.data))
        except Exception as exc:
            self.get_logger().warn(f"failed to update live frame: {exc}")

    def on_detect_result(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
        except Exception as exc:
            self.get_logger().warn(f"failed to parse /tank/api/detect/result: {exc}")
            return

        detections = data.get("detections", [])
        if not isinstance(detections, list):
            detections = []

        metadata = deepcopy(data)

        # ROS 컴퓨터에서 TANK_LIVE_VIEW=false이면 image_shape가 없을 수 있어서
        # 웹 컴퓨터에서 디코딩한 최신 프레임 크기를 metadata에 보강한다.
        live_state = live_view.debug_state()
        if "image_shape" not in metadata:
            shape = live_state.get("latestSourceFrameShape") or live_state.get("latestFrameShape")
            if isinstance(shape, list) and len(shape) >= 2:
                metadata["image_shape"] = shape
                metadata["image"] = {
                    "height": shape[0],
                    "width": shape[1],
                }

        with self._lock:
            self._latest_detect_result = data
            self._last_detect_wall = time.time()

        # 여기서 웹 컴퓨터의 live_view.py 전역 변수에 최신 detection 저장
        live_view.update_detections(detections, metadata)

    def dashboard_payload(self) -> Dict[str, Any]:
        with self._lock:
            state = deepcopy(self._state_latest)
            last_state_wall = self._last_state_wall
            detect_result = deepcopy(self._latest_detect_result)

        latest = state.get("latest", {}) if isinstance(state, dict) else {}
        route_counts = state.get("route_counts", {}) if isinstance(state, dict) else {}

        if not isinstance(latest, dict):
            latest = {}

        if not detect_result:
            maybe_detect = latest.get("detect_result")
            if isinstance(maybe_detect, dict):
                detect_result = maybe_detect

        detections = detect_result.get("detections", [])
        if not isinstance(detections, list):
            detections = []

        now = time.time()
        ros_connected = bool(state) and (now - last_state_wall < 3.0)

        return {
            "serverTime": now,
            "mode": state.get("mode", "monitor") if isinstance(state, dict) else "monitor",

            "liveView": live_view.debug_state(),

            "yolo": {
                "loaded": True,
                "ready": True,
                "status": "from_ros_topic",
                "latestReturnedDetections": detections,
                "latestReturnedDetectionCount": len(detections),
                "latestDetectMs": detect_result.get("yolo_detect_ms"),
                "error": detect_result.get("yolo_error"),
            },

            "bridge": {
                "available": ros_connected,
                "error": None if ros_connected else "no recent /tank/state/latest",
                "latest": latest,
                "routeCounts": route_counts,
                "route_counts": route_counts,
            },

            "sensor": {
                "rosConnected": ros_connected,
                "liveViewEnabled": True,
                "playerPose": (
                    latest.get("player_pose_map")
                    or latest.get("get_action_pose_map")
                    or latest.get("info_compact", {}).get("player_pose_map")
                    if isinstance(latest.get("info_compact"), dict)
                    else latest.get("player_pose_map") or latest.get("get_action_pose_map")
                ),
                "enemyPose": (
                    latest.get("enemy_pose_map")
                    or latest.get("info_compact", {}).get("enemy_pose_map")
                    if isinstance(latest.get("info_compact"), dict)
                    else latest.get("enemy_pose_map")
                ),
                "routeCounts": route_counts,
            },

            "aiLog": latest.get("ai_log") or latest.get("llm_log") or [],
            "reconLog": [
                {
                    "className": det.get("className") or det.get("class_name") or "object",
                    "confidence": det.get("confidence"),
                    "timestamp": detect_result.get("timestamp_wall"),
                }
                for det in detections
                if isinstance(det, dict)
            ],

            "staticMap": {},
            "routeCandidates": {},
            "windowsRecon": {},
            "riskComparison": None,
            "riskFeatures": None,
            "missionPlan": None,
            "suddenDecision": latest.get("sudden_decision"),
            "rosGraph": None,
        }


_ros_node: Optional[TankWebDashboardNode] = None


def _resolve_static_map_path() -> Optional[Path]:
    if STATIC_MAP_PATH:
        p = Path(STATIC_MAP_PATH).expanduser()
        return p if p.exists() else None

    try:
        from ament_index_python.packages import get_package_share_directory

        p = Path(get_package_share_directory("rviz_visualization")) / "map" / "finalmap.map"
        if p.exists():
            return p
    except Exception:
        pass

    return None


def _resolve_static_map_overview_path() -> Optional[Path]:
    if STATIC_MAP_OVERVIEW_PATH:
        p = Path(STATIC_MAP_OVERVIEW_PATH).expanduser()
        return p if p.exists() else None

    try:
        from ament_index_python.packages import get_package_share_directory

        base = Path(get_package_share_directory("rviz_visualization")) / "map"
        for name in ("finalmap_overview.png", "finalmap_overview.jpg"):
            p = base / name
            if p.exists():
                return p
    except Exception:
        pass

    return None


def make_app() -> Flask:
    app = Flask(__name__)

    @app.route("/", methods=["GET"])
    @app.route("/view", methods=["GET"])
    def route_view():
        return live_view.render_view_page(poll_ms=DASHBOARD_POLL_MS)

    @app.route("/video_feed", methods=["GET"])
    def route_video_feed():
        return live_view.video_response(
            web_fps=LIVE_VIEW_FPS,
            jpeg_quality=LIVE_VIEW_JPEG_QUALITY,
        )

    @app.route("/api/dashboard/state", methods=["GET"])
    def route_dashboard_state():
        if _ros_node is None:
            return jsonify({"error": "ROS node is not ready"}), 503
        return jsonify(_ros_node.dashboard_payload())

    @app.route("/api/static-map", methods=["GET"])
    def route_static_map():
        p = _resolve_static_map_path()
        if p is None:
            return jsonify({
                "error": "static map not found",
                "hint": "set TANK_STATIC_MAP_PATH or copy rviz_visualization package",
            }), 404

        with p.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        obstacles = payload.get("obstacles")
        if not isinstance(obstacles, list):
            payload["obstacles"] = []

        payload["mapFile"] = str(p)
        payload["objectCount"] = len(payload["obstacles"])

        overview = _resolve_static_map_overview_path()
        payload["overviewImage"] = {
            "available": overview is not None,
            "path": str(overview) if overview else "",
            "url": "/api/static-map/overview",
        }

        payload.setdefault(
            "bounds",
            {
                "min_x": 0.0,
                "max_x": 300.0,
                "min_y": 0.0,
                "max_y": 300.0,
                "min_z": 0.0,
                "max_z": 300.0,
            },
        )

        return jsonify(payload)

    @app.route("/api/static-map/overview", methods=["GET"])
    def route_static_map_overview():
        p = _resolve_static_map_overview_path()
        if p is None:
            return jsonify({"available": False, "error": "overview image not found"}), 404
        return send_file(p)

    @app.route("/rviz3d", methods=["GET"])
    def route_rviz3d():
        try:
            from rviz_web.web_server_node import HTML_PAGE
            return HTML_PAGE
        except Exception as exc:
            return (
                "<!doctype html><meta charset='utf-8'>"
                "<body style='background:#050806;color:#9ec5f0;font-family:monospace;padding:16px'>"
                f"rviz_web import failed: {exc}<br>"
                "build rviz_web or disable RVIZ 3D tab."
                "</body>",
                503,
            )

    @app.route("/api/config", methods=["GET"])
    def route_config():
        try:
            from rviz_web.web_server_node import DEFAULT_CONFIG
            payload = deepcopy(DEFAULT_CONFIG)
        except Exception:
            payload = {}

        payload["rosbridgePort"] = int(os.environ.get("TANK_RVIZ_WEB_ROSBRIDGE_PORT", "9090"))
        return jsonify(payload)

    @app.route("/api/rviz3d/config", methods=["GET"])
    def route_rviz3d_config():
        return jsonify({
            "enabled": True,
            "servedBy": "tank_web_dashboard",
            "url": "/rviz3d?frame=tank_map&cloud=detected&rays=1",
            "rosbridgePort": int(os.environ.get("TANK_RVIZ_WEB_ROSBRIDGE_PORT", "9090")),
        })

    @app.route("/api/ros/params", methods=["GET"])
    def route_ros_params():
        return jsonify({
            "ok": False,
            "error": "ROS parameter editing is disabled on separated web server",
            "params": [],
        })

    @app.route("/api/ros/params/set", methods=["POST"])
    def route_ros_params_set():
        return jsonify({
            "ok": False,
            "error": "ROS parameter editing is disabled on separated web server",
        }), 403

    @app.route("/health", methods=["GET"])
    def route_health():
        return jsonify({
            "ok": True,
            "server": "tank_web_dashboard",
            "rosNodeReady": _ros_node is not None,
        })

    return app


def main() -> None:
    global _ros_node

    rclpy.init()

    _ros_node = TankWebDashboardNode()

    executor = MultiThreadedExecutor()
    executor.add_node(_ros_node)

    spin_thread = Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    app = make_app()

    try:
        print("============================================================")
        print("Tank Web Dashboard")
        print(f"Listening: http://{WEB_HOST}:{WEB_PORT}/view")
        print("Subscribing:")
        print("  /tank/state/latest")
        print("  /tank/api/detect/image_compressed")
        print("  /tank/api/detect/result")
        print("============================================================")

        app.run(host=WEB_HOST, port=WEB_PORT, threaded=True)
    finally:
        executor.shutdown()
        if _ros_node is not None:
            _ros_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()