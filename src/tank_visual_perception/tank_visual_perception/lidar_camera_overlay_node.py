# -*- coding: utf-8 -*-
"""
ROS2 노드: LiDAR 포인트 -> 포탑 카메라 이미지 투영 오버레이.

이 노드는 캘리브레이션 시각화용이며, path_planning/local_path_node.py와 동일한 투영
수식을 쓴다. ``config_file``로 전달된 path_planning/config/fusion_mapping.yaml의
``projection.params``를 읽기 때문에, 오버레이와 실제 YOLO-LiDAR 융합은 같은
캘리브레이션 값을 사용한다.

구독(Subscribe):
  /tank/camera/image_compressed     sensor_msgs/CompressedImage
  /tank/api/info/raw                std_msgs/String

발행(Publish):
  /tank/camera/lidar_projection/image       sensor_msgs/Image
  /tank/camera/lidar_projection/compressed  sensor_msgs/CompressedImage
  /tank/camera/lidar_projection/status      std_msgs/String
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import math
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image, PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import String

try:
    import yaml
except Exception:  # pragma: no cover - deployment fallback
    yaml = None

from tank_visual_perception.projection import (
    DEFAULT_PROJECTION_PARAMS,
    compute_camera_pose,
    extract_info_payload,
    map_to_raw_xyz,
    project_point,
    to_float,
    vec3_from_dict,
)


def compressed_msg_to_cv2(msg: CompressedImage) -> Optional[np.ndarray]:
    np_arr = np.frombuffer(msg.data, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)


from tank_common.pointcloud import pointcloud2_to_xyz_array


def cv2_to_image_msg(image_bgr: np.ndarray, stamp, frame_id: str) -> Image:
    msg = Image()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height = int(image_bgr.shape[0])
    msg.width = int(image_bgr.shape[1])
    msg.encoding = "bgr8"
    msg.is_bigendian = False
    msg.step = int(image_bgr.shape[1] * 3)
    msg.data = image_bgr.tobytes()
    return msg


def cv2_to_compressed_msg(image_bgr: np.ndarray, stamp, frame_id: str, quality: int = 85) -> Optional[CompressedImage]:
    ok, buffer = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return None
    msg = CompressedImage()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.format = "jpeg"
    msg.data = buffer.tobytes()
    return msg


class LidarCameraOverlayNode(Node):
    def __init__(self) -> None:
        super().__init__("lidar_camera_overlay_node")

        self.declare_parameter("image_topic", "/tank/camera/image_compressed")
        self.declare_parameter("info_topic", "/tank/api/info/compact")
        self.declare_parameter("lidar_pc2_topic", "/tank/sensor/lidar/detected_points_map")
        self.declare_parameter("out_image_topic", "/tank/camera/lidar_projection/image")
        self.declare_parameter("out_compressed_topic", "/tank/camera/lidar_projection/compressed")
        self.declare_parameter("out_status_topic", "/tank/camera/lidar_projection/status")

        # Both launch files pass the exact same custom YAML path to this node
        # and local_path_node. Direct ROS parameter values still override YAML.
        self.declare_parameter("config_file", "")
        config_raw = str(self.get_parameter("config_file").value).strip()
        self.config_file = Path(config_raw).expanduser() if config_raw else None
        yaml_projection_params = self._load_projection_params(self.config_file)
        self.projection_config_loaded = bool(yaml_projection_params)

        # Declare after loading YAML so the YAML values become defaults while
        # explicit launch/CLI parameter overrides retain higher priority.
        for name, default_value in DEFAULT_PROJECTION_PARAMS.items():
            self.declare_parameter(name, yaml_projection_params.get(name, default_value))
        self.declare_parameter("use_only_detected", True)
        self.declare_parameter("min_distance", 1.0)
        self.declare_parameter("max_distance", 35.0)
        self.declare_parameter("point_radius", 2)
        self.declare_parameter("draw_text", True)
        self.declare_parameter("jpeg_quality", 85)

        self.image_topic = str(self.get_parameter("image_topic").value)
        self.info_topic = str(self.get_parameter("info_topic").value)
        self.lidar_pc2_topic = str(self.get_parameter("lidar_pc2_topic").value)
        self.out_image_topic = str(self.get_parameter("out_image_topic").value)
        self.out_compressed_topic = str(self.get_parameter("out_compressed_topic").value)
        self.out_status_topic = str(self.get_parameter("out_status_topic").value)
        self.params = {key: float(self.get_parameter(key).value) for key in DEFAULT_PROJECTION_PARAMS.keys()}
        self.use_only_detected = bool(self.get_parameter("use_only_detected").value)
        self.min_distance = float(self.get_parameter("min_distance").value)
        self.max_distance = float(self.get_parameter("max_distance").value)
        self.point_radius = int(self.get_parameter("point_radius").value)
        self.draw_text = bool(self.get_parameter("draw_text").value)
        self.jpeg_quality = int(self.get_parameter("jpeg_quality").value)

        self._lock = threading.Lock()
        self._latest_info: Optional[Dict[str, Any]] = None
        self._latest_info_stamp = None
        self._latest_lidar_points = np.empty((0, 3), dtype=np.float32)
        self._latest_lidar_stamp = None

        self.create_subscription(String, self.info_topic, self.on_info, 10)
        self.create_subscription(PointCloud2, self.lidar_pc2_topic, self.on_lidar_pc2, 10)
        self.create_subscription(CompressedImage, self.image_topic, self.on_image, 10)
        self.pub_overlay_image = self.create_publisher(Image, self.out_image_topic, 10)
        self.pub_overlay_compressed = self.create_publisher(CompressedImage, self.out_compressed_topic, 10)
        self.pub_status = self.create_publisher(String, self.out_status_topic, 10)

        self.get_logger().info("LiDAR-camera overlay node started")
        self.get_logger().info(f"subscribe image: {self.image_topic}")
        self.get_logger().info(f"subscribe info : {self.info_topic}")
        self.get_logger().info(f"subscribe lidar: {self.lidar_pc2_topic}")
        self.get_logger().info(f"publish image  : {self.out_image_topic}")
        if self.config_file is not None and self.projection_config_loaded:
            self.get_logger().info(f"projection config: {self.config_file}")
        elif self.config_file is not None:
            self.get_logger().warn(f"projection config unavailable; using defaults: {self.config_file}")
        else:
            self.get_logger().warn("projection config_file not set; using built-in defaults")
        self.get_logger().info(f"effective projection params: {self.params}")

    def _load_projection_params(self, config_file: Optional[Path]) -> Dict[str, float]:
        """Read projection.params from the shared custom fusion YAML.

        This is intentionally not a ROS parameter YAML parser. The project
        already uses a custom nested YAML schema in local_path_node, so the
        overlay reads the same section directly.
        """
        if config_file is None:
            return {}
        if yaml is None:
            self.get_logger().warn("PyYAML unavailable; overlay cannot read config_file")
            return {}
        if not config_file.exists():
            self.get_logger().warn(f"projection config not found: {config_file}")
            return {}
        try:
            with config_file.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except Exception as exc:
            self.get_logger().warn(f"failed to load projection config {config_file}: {exc}")
            return {}

        if not isinstance(data, dict):
            self.get_logger().warn(f"projection config is not a mapping: {config_file}")
            return {}
        projection = data.get("projection")
        params = projection.get("params") if isinstance(projection, dict) else None
        if not isinstance(params, dict):
            self.get_logger().warn(f"projection.params not found in: {config_file}")
            return {}
        return {
            name: to_float(params.get(name), default_value)
            for name, default_value in DEFAULT_PROJECTION_PARAMS.items()
        }

    def on_info(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
            info = extract_info_payload(payload)
            if info is None:
                return
            with self._lock:
                self._latest_info = info
                self._latest_info_stamp = self.get_clock().now()
        except Exception as exc:
            self.get_logger().warn(f"failed to parse info raw: {exc}")

    def on_lidar_pc2(self, msg: PointCloud2) -> None:
        try:
            points = pointcloud2_to_xyz_array(msg)
            with self._lock:
                self._latest_lidar_points = points
                self._latest_lidar_stamp = self.get_clock().now()
        except Exception as exc:
            self.get_logger().warn(f"failed to parse lidar PC2: {exc}")

    def on_image(self, msg: CompressedImage) -> None:
        image = compressed_msg_to_cv2(msg)
        if image is None:
            self.get_logger().warn("failed to decode compressed image")
            return

        with self._lock:
            info = self._latest_info
            lidar_points = self._latest_lidar_points.copy()

        if info is None:
            overlay = image.copy()
            cv2.putText(overlay, f"Waiting for {self.info_topic}...", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
            self.publish_overlay(overlay, msg.header.stamp, msg.header.frame_id or "tank_camera")
            self.publish_status(0, 0, 0, "waiting_info")
            return

        overlay, projected_count, used_count, total_count = self.draw_lidar_overlay(image, info, lidar_points)
        if self.draw_text:
            text = f"LiDAR projection: {projected_count}/{used_count} used, total={total_count}"
            cv2.putText(overlay, text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(overlay, text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        self.publish_overlay(overlay, msg.header.stamp, msg.header.frame_id or "tank_camera")
        self.publish_status(projected_count, used_count, total_count, "ok")

    def draw_lidar_overlay(self, image: np.ndarray, info: Dict[str, Any], lidar_points: np.ndarray) -> Tuple[np.ndarray, int, int, int]:
        h, w = image.shape[:2]
        overlay = image.copy()
        try:
            camera_pos, camera_yaw, camera_pitch, camera_roll = compute_camera_pose(info, self.params)
        except Exception as exc:
            cv2.putText(overlay, f"Invalid info: {exc}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
            return overlay, 0, 0, 0

        if lidar_points.size == 0:
            return overlay, 0, 0, 0

        # compute_camera_pose() returns an np.ndarray in Unity raw order
        # [x, y, z]. Do not use getattr(.x/.z): NumPy arrays have no such
        # fields and that silently turns the distance-filter origin into (0, 0),
        # causing every real tank-local LiDAR point to be discarded.
        try:
            camera_pos_raw = np.asarray(camera_pos, dtype=np.float64).reshape(-1)
            if camera_pos_raw.size < 3:
                raise ValueError(f"camera_pos has {camera_pos_raw.size} values")
            cam_raw_x = float(camera_pos_raw[0])
            cam_raw_z = float(camera_pos_raw[2])
        except Exception as exc:
            cv2.putText(overlay, f"Invalid camera pose: {exc}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            return overlay, 0, 0, int(lidar_points.shape[0])

        projected_count = 0
        used_count = 0
        total_count = int(lidar_points.shape[0])
        for x, y, z in lidar_points:
            map_pos = {"x": float(x), "y": float(y), "z": float(z)}
            # map.x == raw.x, map.y == raw.z. 거리 필터는 오버레이 부하 제한용일 뿐이다.
            distance = math.hypot(map_pos["x"] - cam_raw_x, map_pos["y"] - cam_raw_z)
            if distance < self.min_distance or distance > self.max_distance:
                continue
            pos_raw = map_to_raw_xyz(map_pos)
            used_count += 1
            projected = project_point(
                point_world_raw=vec3_from_dict(pos_raw),
                camera_pos_world_raw=camera_pos,
                camera_yaw_deg=camera_yaw,
                camera_pitch_deg=camera_pitch,
                camera_roll_deg=camera_roll,
                image_w=w,
                image_h=h,
                params=self.params,
            )
            if projected is None:
                continue
            u, v, _depth = projected
            if 0 <= u < w and 0 <= v < h:
                ratio = max(0.0, min(1.0, distance / max(0.001, self.max_distance)))
                b = int(255 * ratio)
                r = int(255 * (1.0 - ratio))
                cv2.circle(overlay, (u, v), self.point_radius, (b, 255, r), -1, cv2.LINE_AA)
                projected_count += 1
        return overlay, projected_count, used_count, total_count

    def publish_overlay(self, image_bgr: np.ndarray, stamp, frame_id: str) -> None:
        self.pub_overlay_image.publish(cv2_to_image_msg(image_bgr, stamp, frame_id))
        comp_msg = cv2_to_compressed_msg(image_bgr, stamp, frame_id, self.jpeg_quality)
        if comp_msg is not None:
            self.pub_overlay_compressed.publish(comp_msg)

    def publish_status(self, projected_count: int, used_count: int, total_count: int, state: str) -> None:
        msg = String()
        msg.data = json.dumps(
            {
                "state": state,
                "projected_count": projected_count,
                "used_count": used_count,
                "total_count": total_count,
                "params": self.params,
                "config_file": str(self.config_file) if self.config_file is not None else None,
                "projection_config_loaded": self.projection_config_loaded,
                "method": "shared_projection_math_and_fusion_yaml",
            },
            ensure_ascii=False,
        )
        self.pub_status.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LidarCameraOverlayNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
