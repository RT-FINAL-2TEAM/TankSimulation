# -*- coding: utf-8 -*-
"""Android HTTP frame/IMU receiver + YOLO gateway.

This node intentionally owns every smartphone-facing concern so ros_bridge,
path_planning, potential, and control stay untouched.
"""

from __future__ import annotations

import json
import threading
import time
from typing import Any, Dict, Optional

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from std_msgs.msg import String

try:
    from flask import Flask, jsonify, request
except Exception:  # pragma: no cover
    Flask = None
    jsonify = None
    request = None

from phone_sim2real.common import get_nested_float, parse_csv_set, safe_json_loads
from vision.yolo_detector import get_detector


class PhoneYoloGatewayNode(Node):
    def __init__(self) -> None:
        super().__init__("phone_yolo_gateway_node")

        self.declare_parameter("phone_host", "0.0.0.0")
        self.declare_parameter("phone_port", 5002)
        self.declare_parameter("phone_endpoint", "/phone/detect")
        self.declare_parameter("control_endpoint", "/phone/control")
        self.declare_parameter("status_topic", "/tank/phone_sim2real/status")
        self.declare_parameter("detections_topic", "/tank/phone_sim2real/detections")
        self.declare_parameter("imu_topic", "/tank/phone_sim2real/imu_raw")
        self.declare_parameter("imu_json_topic", "/tank/phone_sim2real/imu_json")
        self.declare_parameter("allow_any_client", True)
        self.declare_parameter("allowed_clients", "127.0.0.1,::1")
        self.declare_parameter("publish_empty_detections", True)
        self.declare_parameter("max_image_bytes", 2 * 1024 * 1024)

        self.host = str(self.get_parameter("phone_host").value)
        self.port = int(self.get_parameter("phone_port").value)
        self.endpoint = str(self.get_parameter("phone_endpoint").value)
        if not self.endpoint.startswith("/"):
            self.endpoint = "/" + self.endpoint
        self.control_endpoint = str(self.get_parameter("control_endpoint").value)
        if not self.control_endpoint.startswith("/"):
            self.control_endpoint = "/" + self.control_endpoint
        self.allow_any_client = bool(self.get_parameter("allow_any_client").value)
        self.allowed_clients = parse_csv_set(str(self.get_parameter("allowed_clients").value))
        self.publish_empty = bool(self.get_parameter("publish_empty_detections").value)
        self.max_image_bytes = int(self.get_parameter("max_image_bytes").value)

        self.pub_status = self.create_publisher(String, str(self.get_parameter("status_topic").value), 10)
        self.pub_detections = self.create_publisher(String, str(self.get_parameter("detections_topic").value), 10)
        self.pub_imu = self.create_publisher(Imu, str(self.get_parameter("imu_topic").value), 10)
        self.pub_imu_json = self.create_publisher(String, str(self.get_parameter("imu_json_topic").value), 10)

        self.detector = get_detector()
        self._started = False
        self._request_count = 0
        self._last_status: Dict[str, Any] = {}
        self._app = self._make_flask_app()
        self._server_thread = threading.Thread(target=self._run_flask, name="phone_sim2real_flask", daemon=True)
        self._server_thread.start()
        self.create_timer(1.0, self._publish_status)
        self.get_logger().info(
            f"phone_sim2real gateway listening on http://{self.host}:{self.port}{self.endpoint} "
            f"control=http://{self.host}:{self.port}{self.control_endpoint} "
            f"detector_loaded={self.detector.loaded}"
        )

    def _make_flask_app(self):
        if Flask is None:
            raise RuntimeError("Flask is not installed. Install python3-flask or run in the same environment as ros_bridge.")
        app = Flask("phone_sim2real_gateway")
        app.config["MAX_CONTENT_LENGTH"] = self.max_image_bytes

        @app.route("/phone/health", methods=["GET"])
        def health():  # type: ignore[no-untyped-def]
            state = self.detector.debug_state()
            return jsonify({
                "ok": True,
                "service": "phone_sim2real",
                "endpoint": self.endpoint,
                "controlEndpoint": self.control_endpoint,
                "detectorLoaded": bool(state.get("loaded")),
                "modelPath": state.get("modelPath"),
                "imgsz": state.get("imgsz"),
                "time": time.time(),
            })

        @app.route(self.control_endpoint, methods=["POST"])
        def control():  # type: ignore[no-untyped-def]
            remote_addr = str(request.headers.get("X-Forwarded-For", request.remote_addr or "")).split(",")[0].strip()
            if not self._client_allowed(remote_addr):
                return jsonify({"ok": False, "error": f"client_not_allowed:{remote_addr}"}), 403
            metadata = self._extract_metadata()
            command = str(metadata.get("command", metadata.get("action", metadata.get("control", "")))).strip().lower()
            now_wall = time.time()
            payload = {
                "ok": True,
                "source": "phone_sim2real",
                "timestamp_wall": now_wall,
                "server_receive_time": now_wall,
                "remote_addr": remote_addr,
                "command": command,
                "phone": metadata,
                "count": 0,
                "detections": [],
                "control_only": True,
            }
            self._publish_json(self.pub_detections, payload)
            self._publish_imu(metadata)
            self._last_status = {
                "last_request_wall": now_wall,
                "request_count": self._request_count,
                "remote_addr": remote_addr,
                "last_detection_count": 0,
                "last_detect_ms": 0.0,
                "frame_width": 0,
                "frame_height": 0,
                "detector_loaded": self.detector.loaded,
                "last_command": command,
            }
            return jsonify(payload)

        @app.route(self.endpoint, methods=["POST"])
        def detect():  # type: ignore[no-untyped-def]
            remote_addr = str(request.headers.get("X-Forwarded-For", request.remote_addr or "")).split(",")[0].strip()
            if not self._client_allowed(remote_addr):
                return jsonify({"ok": False, "error": f"client_not_allowed:{remote_addr}"}), 403

            started = time.perf_counter()
            image_bytes = self._extract_image_bytes()
            if not image_bytes:
                return jsonify({"ok": False, "error": "missing_image"}), 400
            if len(image_bytes) > self.max_image_bytes:
                return jsonify({"ok": False, "error": "image_too_large", "bytes": len(image_bytes)}), 413

            metadata = self._extract_metadata()
            frame_w, frame_h = self._decode_frame_size(image_bytes)
            detections = self.detector.detect_bytes(image_bytes)
            self._request_count += 1

            now_wall = time.time()
            payload = {
                "ok": True,
                "source": "phone_sim2real",
                "timestamp_wall": now_wall,
                "server_receive_time": now_wall,
                "remote_addr": remote_addr,
                "frame": {
                    "width": frame_w,
                    "height": frame_h,
                    "expected_width": 416,
                    "expected_height": 416,
                },
                "phone": metadata,
                "count": len(detections),
                "detections": detections,
                "detect_ms": (time.perf_counter() - started) * 1000.0,
            }
            if detections or self.publish_empty:
                self._publish_json(self.pub_detections, payload)
            self._publish_imu(metadata)
            self._last_status = {
                "last_request_wall": now_wall,
                "request_count": self._request_count,
                "remote_addr": remote_addr,
                "last_detection_count": len(detections),
                "last_detect_ms": payload["detect_ms"],
                "frame_width": frame_w,
                "frame_height": frame_h,
                "detector_loaded": self.detector.loaded,
            }
            return jsonify(payload)

        return app

    def _run_flask(self) -> None:
        try:
            self._started = True
            self._app.run(host=self.host, port=self.port, debug=False, use_reloader=False, threaded=True)
        except Exception as exc:
            self._started = False
            self.get_logger().error(f"phone HTTP server failed: {exc}")

    def _client_allowed(self, remote_addr: str) -> bool:
        if self.allow_any_client:
            return True
        if not remote_addr:
            return False
        return remote_addr.lower() in self.allowed_clients

    def _extract_image_bytes(self) -> bytes:
        if request.files:
            for key in ("image", "frame", "file"):
                item = request.files.get(key)
                if item is not None:
                    return item.read()
        raw = request.get_data(cache=False) or b""
        return bytes(raw)

    def _extract_metadata(self) -> Dict[str, Any]:
        candidates = []
        try:
            json_body = request.get_json(silent=True)
            if isinstance(json_body, dict):
                candidates.append(json.dumps(json_body, ensure_ascii=False))
            # Android versions in this project have used both names.  Keep all
            # accepted so the app can be moved inside phone_sim2real without
            # breaking IMU transfer.
            candidates.append(request.form.get("meta"))
            candidates.append(request.form.get("metadata"))
            candidates.append(request.form.get("imu"))
            candidates.append(request.headers.get("X-Phone-Metadata"))
        except Exception:
            pass
        for candidate in candidates:
            if not candidate:
                continue
            data = safe_json_loads(str(candidate), None)
            if isinstance(data, dict):
                return data
        return {}

    def _decode_frame_size(self, image_bytes: bytes) -> tuple[int, int]:
        try:
            arr = np.frombuffer(image_bytes, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                return 416, 416
            h, w = frame.shape[:2]
            return int(w), int(h)
        except Exception:
            return 416, 416

    def _publish_json(self, publisher: Any, payload: Dict[str, Any]) -> None:
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        publisher.publish(msg)

    def _publish_imu(self, metadata: Dict[str, Any]) -> None:
        imu = metadata.get("imu") if isinstance(metadata.get("imu"), dict) else metadata
        if not isinstance(imu, dict):
            return
        self._publish_json(self.pub_imu_json, {"timestamp_wall": time.time(), "source": "phone_sim2real", "imu": imu})

        msg = Imu()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "phone_imu"
        q = imu.get("quaternion") if isinstance(imu.get("quaternion"), dict) else {}
        q_list = imu.get("quat_xyzw") if isinstance(imu.get("quat_xyzw"), list) else []
        try:
            if q:
                msg.orientation.x = float(q.get("x", 0.0))
                msg.orientation.y = float(q.get("y", 0.0))
                msg.orientation.z = float(q.get("z", 0.0))
                msg.orientation.w = float(q.get("w", 1.0))
            elif len(q_list) >= 4:
                msg.orientation.x = float(q_list[0])
                msg.orientation.y = float(q_list[1])
                msg.orientation.z = float(q_list[2])
                msg.orientation.w = float(q_list[3])
            else:
                msg.orientation_covariance[0] = -1.0
        except Exception:
            msg.orientation_covariance[0] = -1.0
        gyro = imu.get("gyro") if isinstance(imu.get("gyro"), list) else []
        accel = imu.get("accel") if isinstance(imu.get("accel"), list) else []
        if len(gyro) >= 3:
            msg.angular_velocity.x = float(gyro[0])
            msg.angular_velocity.y = float(gyro[1])
            msg.angular_velocity.z = float(gyro[2])
        else:
            msg.angular_velocity_covariance[0] = -1.0
        if len(accel) >= 3:
            msg.linear_acceleration.x = float(accel[0])
            msg.linear_acceleration.y = float(accel[1])
            msg.linear_acceleration.z = float(accel[2])
        else:
            msg.linear_acceleration_covariance[0] = -1.0
        self.pub_imu.publish(msg)

    def _publish_status(self) -> None:
        payload = {
            "ok": True,
            "source": "phone_sim2real_gateway",
            "server_started": bool(self._started),
            "host": self.host,
            "port": self.port,
            "endpoint": self.endpoint,
            "request_count": self._request_count,
            "detector_loaded": bool(self.detector.loaded),
            **self._last_status,
            "timestamp_wall": time.time(),
        }
        self._publish_json(self.pub_status, payload)


def main(args: Optional[list[str]] = None) -> None:
    rclpy.init(args=args)
    node = PhoneYoloGatewayNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
