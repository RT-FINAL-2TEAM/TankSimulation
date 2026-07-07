# -*- coding: utf-8 -*-
"""Tank Challenge용 ROS2 LiDAR 후처리 노드.

입력은 ros_bridge가 /info 수신 직후 만든 Unity raw-frame PointCloud2다.
이 노드는 JSON /info 또는 lidarPoints 원본 배열을 다시 파싱하지 않는다.

흐름:
    ros_bridge
      /tank/sensor/lidar/raw_detected_points (PointCloud2, tank_unity_raw)
        -> lidar_processor_node
      /tank/sensor/lidar/{detected,terrain,all_detected}_points_map (PointCloud2, tank_map)

책임 범위:
- Unity raw XYZ -> tank_map XYZ 좌표 변환
- grid local-ground 기준 지형/장애물 분리
- 분류된 PointCloud2와 가벼운 분류 상태만 publish
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header, String

from lidar.config import (
    GROUND_FILTER_ENABLED,
    MAP_FRAME,
    TERRAIN_CLIMB_LIMIT,
    TERRAIN_GRID_RESOLUTION,
    TERRAIN_OBSTACLE_MIN_HEIGHT,
    TOPIC_LIDAR_ALL_DETECTED_MAP,
    TOPIC_LIDAR_DETECTED_MAP,
    TOPIC_LIDAR_RAW_DETECTED,
    TOPIC_LIDAR_TERRAIN_MAP,
    TOPIC_TERRAIN_INFO,
    UNITY_FRAME,
)
from lidar.coordinate_utils import dumps_compact
from lidar.terrain_utils import split_terrain_obstacle_xyz
from tank_common.pointcloud import pointcloud2_to_xyz_array


class LidarProcessorNode(Node):
    """Raw LiDAR PointCloud2를 map-frame 분류 cloud로 변환한다."""

    def __init__(self) -> None:
        super().__init__("lidar_processor_node")

        self.declare_parameter("input_pc2_topic", TOPIC_LIDAR_RAW_DETECTED)
        self.declare_parameter("ground_filter_enabled", GROUND_FILTER_ENABLED)
        self.declare_parameter("terrain_grid_resolution", TERRAIN_GRID_RESOLUTION)
        self.declare_parameter("terrain_climb_limit", TERRAIN_CLIMB_LIMIT)
        self.declare_parameter("terrain_obstacle_min_height", TERRAIN_OBSTACLE_MIN_HEIGHT)

        self.input_pc2_topic = str(self.get_parameter("input_pc2_topic").value)
        self.ground_filter_enabled = bool(self.get_parameter("ground_filter_enabled").value)
        self.terrain_grid_resolution = float(self.get_parameter("terrain_grid_resolution").value)
        self.terrain_climb_limit = float(self.get_parameter("terrain_climb_limit").value)
        self.terrain_obstacle_min_height = float(
            self.get_parameter("terrain_obstacle_min_height").value
        )
        self._last_unexpected_frame_warn_wall = 0.0

        self.pub_detected_map = self.create_publisher(
            PointCloud2, TOPIC_LIDAR_DETECTED_MAP, 10
        )
        self.pub_all_detected_map = self.create_publisher(
            PointCloud2, TOPIC_LIDAR_ALL_DETECTED_MAP, 10
        )
        self.pub_terrain_map = self.create_publisher(
            PointCloud2, TOPIC_LIDAR_TERRAIN_MAP, 10
        )
        self.pub_terrain_info = self.create_publisher(String, TOPIC_TERRAIN_INFO, 10)

        self.create_subscription(
            PointCloud2, self.input_pc2_topic, self.raw_pc2_cb, 10
        )
        self.get_logger().info(
            "LiDAR processor started: "
            f"sub={self.input_pc2_topic} ({UNITY_FRAME}) -> {MAP_FRAME}; "
            f"obstacle={TOPIC_LIDAR_DETECTED_MAP}, "
            f"terrain={TOPIC_LIDAR_TERRAIN_MAP}, "
            f"all={TOPIC_LIDAR_ALL_DETECTED_MAP}, "
            f"ground_filter={self.ground_filter_enabled}, "
            f"grid={self.terrain_grid_resolution}, "
            f"climb_limit={self.terrain_climb_limit}"
        )

    @staticmethod
    def _map_from_raw_xyz(raw_xyz: np.ndarray) -> np.ndarray:
        """Unity raw (x, y-height, z) -> map (x, y-plane, z-height)."""
        raw_xyz = np.asarray(raw_xyz, dtype=np.float32).reshape(-1, 3)
        if raw_xyz.size == 0:
            return np.empty((0, 3), dtype=np.float32)
        mapped = np.empty_like(raw_xyz, dtype=np.float32)
        mapped[:, 0] = raw_xyz[:, 0]
        mapped[:, 1] = raw_xyz[:, 2]
        mapped[:, 2] = raw_xyz[:, 1]
        return mapped

    @staticmethod
    def _xyz32_cloud(xyz: np.ndarray, stamp: Any, frame_id: str) -> PointCloud2:
        """float32 Nx3 배열을 Python point-list 없이 PointCloud2로 만든다."""
        xyz = np.ascontiguousarray(np.asarray(xyz, dtype="<f4").reshape(-1, 3))
        header = Header()
        header.stamp = stamp
        header.frame_id = frame_id

        msg = PointCloud2()
        msg.header = header
        msg.height = 1
        msg.width = int(xyz.shape[0])
        msg.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        msg.is_bigendian = False
        msg.point_step = 12
        msg.row_step = 12 * int(xyz.shape[0])
        msg.is_dense = True
        msg.data = xyz.tobytes()
        return msg

    def _publish_terrain_info(
        self,
        source_msg: PointCloud2,
        input_count: int,
        obstacle_count: int,
        terrain_count: int,
        terrain_filter: dict,
    ) -> None:
        payload = {
            "route": "/info",
            "timestamp_wall": time.time(),
            "frame_id": MAP_FRAME,
            "source": "lidar_pc2_terrain_separation",
            "input_topic": self.input_pc2_topic,
            "input_frame_id": source_msg.header.frame_id or UNITY_FRAME,
            "terrain_filter": terrain_filter,
            "counts": {
                "all_detected": int(input_count),
                "obstacle": int(obstacle_count),
                "terrain": int(terrain_count),
            },
        }
        msg = String()
        msg.data = dumps_compact(payload)
        self.pub_terrain_info.publish(msg)

    def raw_pc2_cb(self, msg: PointCloud2) -> None:
        """bridge가 만든 raw detected cloud를 분류해 map-frame cloud로 발행한다."""
        raw_frame = msg.header.frame_id or UNITY_FRAME
        if raw_frame != UNITY_FRAME:
            now = time.monotonic()
            if now - self._last_unexpected_frame_warn_wall >= 2.0:
                self._last_unexpected_frame_warn_wall = now
                self.get_logger().warn(
                    f"expected raw LiDAR frame '{UNITY_FRAME}', got '{raw_frame}'; "
                    "applying raw.x/raw.z/raw.y conversion anyway"
                )

        try:
            raw_xyz = pointcloud2_to_xyz_array(msg)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"failed to read raw LiDAR PointCloud2: {exc}")
            return

        all_map_xyz = self._map_from_raw_xyz(raw_xyz)
        stamp = msg.header.stamp

        if not self.ground_filter_enabled:
            # Without terrain separation every detected point is an obstacle.
            # Reuse the same map cloud for both outputs instead of mapping and
            # serializing the identical array twice.
            terrain_filter = {
                "enabled": False,
                "method": "disabled",
                "input_points": int(raw_xyz.shape[0]),
                "obstacle_points": int(raw_xyz.shape[0]),
                "terrain_points": 0,
                "grid_resolution": self.terrain_grid_resolution,
                "climb_limit": self.terrain_climb_limit,
                "obstacle_min_height": self.terrain_obstacle_min_height,
            }
            all_map_msg = self._xyz32_cloud(all_map_xyz, stamp, MAP_FRAME)
            terrain_map_msg = self._xyz32_cloud(
                np.empty((0, 3), dtype=np.float32),
                stamp,
                MAP_FRAME,
            )
            self.pub_all_detected_map.publish(all_map_msg)
            self.pub_detected_map.publish(all_map_msg)
            self.pub_terrain_map.publish(terrain_map_msg)
            self._publish_terrain_info(
                msg,
                input_count=raw_xyz.shape[0],
                obstacle_count=all_map_xyz.shape[0],
                terrain_count=0,
                terrain_filter=terrain_filter,
            )
            return

        if raw_xyz.size:
            obstacle_raw_xyz, terrain_raw_xyz, stats = split_terrain_obstacle_xyz(
                raw_xyz,
                grid_resolution=self.terrain_grid_resolution,
                climb_limit=self.terrain_climb_limit,
                obstacle_min_height=self.terrain_obstacle_min_height,
            )
            terrain_filter = {
                "enabled": True,
                "method": "grid_local_ground_steep_cell",
                **stats.to_dict(),
            }
        else:
            # Preserve the existing empty-cloud status payload behavior.
            obstacle_raw_xyz = raw_xyz
            terrain_raw_xyz = np.empty((0, 3), dtype=np.float32)
            terrain_filter = {
                "enabled": False,
                "method": "disabled",
                "input_points": int(raw_xyz.shape[0]),
                "obstacle_points": int(raw_xyz.shape[0]),
                "terrain_points": 0,
                "grid_resolution": self.terrain_grid_resolution,
                "climb_limit": self.terrain_climb_limit,
                "obstacle_min_height": self.terrain_obstacle_min_height,
            }

        obstacle_map_xyz = self._map_from_raw_xyz(obstacle_raw_xyz)
        terrain_map_xyz = self._map_from_raw_xyz(terrain_raw_xyz)

        self.pub_all_detected_map.publish(
            self._xyz32_cloud(all_map_xyz, stamp, MAP_FRAME)
        )
        self.pub_detected_map.publish(
            self._xyz32_cloud(obstacle_map_xyz, stamp, MAP_FRAME)
        )
        self.pub_terrain_map.publish(
            self._xyz32_cloud(terrain_map_xyz, stamp, MAP_FRAME)
        )
        self._publish_terrain_info(
            msg,
            input_count=raw_xyz.shape[0],
            obstacle_count=obstacle_map_xyz.shape[0],
            terrain_count=terrain_map_xyz.shape[0],
            terrain_filter=terrain_filter,
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LidarProcessorNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
