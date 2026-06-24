# -*- coding: utf-8 -*-
"""Tank Challenge용 ROS2 LiDAR 전처리 노드.

책임 범위:
- /tank/api/info/raw 안의 lidarOrigin, lidarRotation, lidarPoints 분리
- Unity raw 좌표를 tank_map 좌표로 변환
- 팀원 지형 개발본의 grid local-ground 로직으로 지형/장애물을 분리
- 장애물 후보만 /tank/sensor/lidar/detected_points_map으로 publish
- 지형 후보는 /tank/sensor/lidar/terrain_points_map으로 publish
- 전체 hit point는 /tank/sensor/lidar/all_detected_points_map으로 publish

다른 패키지는 LiDAR raw schema를 직접 해석하지 않고 이 노드의 출력 topic을 사용한다.
"""

from __future__ import annotations
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
import json
from typing import Any, Dict, Optional

import rclpy
from geometry_msgs.msg import PointStamped, Vector3Stamped
from rclpy.node import Node
from std_msgs.msg import Int32, String

from lidar.config import (
    DEFAULT_LIDAR_ORIGIN_Y,
    GROUND_FILTER_ENABLED,
    MAP_FRAME,
    TERRAIN_CLIMB_LIMIT,
    TERRAIN_GRID_RESOLUTION,
    TERRAIN_OBSTACLE_MIN_HEIGHT,
    TOPIC_INFO_RAW,
    TOPIC_LIDAR_ALL_DETECTED_MAP,
    TOPIC_LIDAR_DETECTED_MAP,
    TOPIC_LIDAR_ORIGIN,
    TOPIC_LIDAR_ORIGIN_RAW,
    TOPIC_LIDAR_POINTS,
    TOPIC_LIDAR_POINTS_COUNT,
    TOPIC_LIDAR_ROTATION,
    TOPIC_LIDAR_TERRAIN_MAP,
    TOPIC_TERRAIN_INFO,
    UNITY_FRAME,
)
from lidar.coordinate_utils import as_xyz, dumps_compact, raw_and_map_point, to_float
from lidar.payloads import build_classified_lidar_payloads


class LidarProcessorNode(Node):
    def __init__(self) -> None:
        super().__init__("lidar_processor_node")

        self.declare_parameter("info_raw_topic", TOPIC_INFO_RAW)
        self.declare_parameter("ground_filter_enabled", GROUND_FILTER_ENABLED)
        self.declare_parameter("default_lidar_origin_y", DEFAULT_LIDAR_ORIGIN_Y)
        self.declare_parameter("terrain_grid_resolution", TERRAIN_GRID_RESOLUTION)
        self.declare_parameter("terrain_climb_limit", TERRAIN_CLIMB_LIMIT)
        self.declare_parameter("terrain_obstacle_min_height", TERRAIN_OBSTACLE_MIN_HEIGHT)
        # 용량 큰 레거시 JSON LiDAR 토픽. 하위 노드는 PC2 토픽을 소비해야 한다.
        self.declare_parameter("publish_legacy_lidar_json", False)

        self.info_raw_topic = str(self.get_parameter("info_raw_topic").value)
        self.ground_filter_enabled = bool(self.get_parameter("ground_filter_enabled").value)
        self.default_lidar_origin_y = float(self.get_parameter("default_lidar_origin_y").value)
        self.terrain_grid_resolution = float(self.get_parameter("terrain_grid_resolution").value)
        self.terrain_climb_limit = float(self.get_parameter("terrain_climb_limit").value)
        self.terrain_obstacle_min_height = float(self.get_parameter("terrain_obstacle_min_height").value)
        self.publish_legacy_lidar_json = bool(self.get_parameter("publish_legacy_lidar_json").value)

        self.pub_points = self.create_publisher(String, TOPIC_LIDAR_POINTS, 10)
        self.pub_points_count = self.create_publisher(Int32, TOPIC_LIDAR_POINTS_COUNT, 10)
        self.pub_origin = self.create_publisher(PointStamped, TOPIC_LIDAR_ORIGIN, 10)
        self.pub_origin_raw = self.create_publisher(PointStamped, TOPIC_LIDAR_ORIGIN_RAW, 10)
        self.pub_rotation = self.create_publisher(Vector3Stamped, TOPIC_LIDAR_ROTATION, 10)
        self.pub_detected_map = self.create_publisher(PointCloud2, TOPIC_LIDAR_DETECTED_MAP, 10)
        self.pub_all_detected_map = self.create_publisher(PointCloud2, TOPIC_LIDAR_ALL_DETECTED_MAP, 10)
        self.pub_terrain_map = self.create_publisher(PointCloud2, TOPIC_LIDAR_TERRAIN_MAP, 10)
        self.pub_terrain_info = self.create_publisher(String, TOPIC_TERRAIN_INFO, 10)

        self.create_subscription(String, self.info_raw_topic, self.info_raw_cb, 10)
        self.get_logger().info(
            f"LiDAR processor started: sub={self.info_raw_topic}, "
            f"pub={TOPIC_LIDAR_DETECTED_MAP} obstacle-only, "
            f"pub={TOPIC_LIDAR_TERRAIN_MAP} terrain-only, "
            f"pub={TOPIC_LIDAR_ALL_DETECTED_MAP} all hits, "
            f"legacy_json={self.publish_legacy_lidar_json}, "
            f"ground_filter={self.ground_filter_enabled}, "
            f"grid={self.terrain_grid_resolution}, climb_limit={self.terrain_climb_limit}"
        )

    def publish_json(self, publisher: Any, data: Any) -> None:
        msg = String()
        msg.data = dumps_compact(data)
        publisher.publish(msg)

    def publish_int(self, publisher: Any, value: int) -> None:
        msg = Int32()
        msg.data = int(value)
        publisher.publish(msg)

    def publish_point(self, publisher: Any, point: Dict[str, Any]) -> None:
        msg = PointStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = str(point.get("frame_id", MAP_FRAME))
        msg.point.x = to_float(point.get("x"))
        msg.point.y = to_float(point.get("y"))
        msg.point.z = to_float(point.get("z"))
        publisher.publish(msg)

    def publish_vector3(self, publisher: Any, vector: Dict[str, Any], frame_id: str = UNITY_FRAME) -> None:
        msg = Vector3Stamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = frame_id
        msg.vector.x = to_float(vector.get("x"))
        msg.vector.y = to_float(vector.get("y"))
        msg.vector.z = to_float(vector.get("z"))
        publisher.publish(msg)

    @staticmethod
    def _extract_body_angles(data: Dict[str, Any]) -> Optional[Dict[str, float]]:
        """시뮬레이터에서 흔히 쓰는 body-heading 필드명을 받아들인다."""
        for key in ("playerBody", "body", "player_body", "playerBodyDeg"):
            body = data.get(key)
            parsed = as_xyz(body)
            if parsed is not None:
                return parsed
        # 일부 브릿지 payload는 body x/y/z를 playerBodyX/Y/Z로 펼쳐서 보낸다.
        if any(k in data for k in ("playerBodyX", "playerBodyY", "playerBodyZ")):
            return {
                "x": to_float(data.get("playerBodyX")),
                "y": to_float(data.get("playerBodyY")),
                "z": to_float(data.get("playerBodyZ")),
            }
        return None
        
    def create_pc2(self, payload_dict: Dict[str, Any], frame_id: str) -> PointCloud2:
        """JSON 포인트 리스트를 바이너리 PointCloud2로 변환 (속도 최적화 버전)"""
        raw_points = payload_dict.get("points", [])
        
        # 리스트 컴프리헨션으로 빠르게 추출
        points = [
            [
                to_float(p.get("position_map", p.get("position", {})).get("x", 0.0)),
                to_float(p.get("position_map", p.get("position", {})).get("y", 0.0)),
                to_float(p.get("position_map", p.get("position", {})).get("z", 0.0))
            ]
            for p in raw_points if isinstance(p, dict)
        ]
            
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = frame_id
        return point_cloud2.create_cloud_xyz32(header, points)

    def info_raw_cb(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
            data = payload.get("data", payload) if isinstance(payload, dict) else {}
            if not isinstance(data, dict):
                return
        except Exception as exc:
            self.get_logger().debug(f"/tank/api/info/raw parse failed: {exc}")
            return

        ts = to_float(payload.get("timestamp_wall")) if isinstance(payload, dict) else 0.0
        lidar_points = data.get("lidarPoints") if isinstance(data.get("lidarPoints"), list) else []
        lidar_count = len(lidar_points)

        points_payload = {
            "route": "/info",
            "timestamp_wall": ts,
            "source": "lidarPoints/raw",
            "count": lidar_count,
            "points": lidar_points,
        }

        origin = data.get("lidarOrigin")
        origin_y = to_float(origin.get("y"), self.default_lidar_origin_y) if isinstance(origin, dict) else self.default_lidar_origin_y
        origin_map: Optional[Dict[str, Any]] = None
        if isinstance(origin, dict):
            origin_raw, origin_map = raw_and_map_point(origin, "/info/lidarOrigin")
            self.publish_point(self.pub_origin_raw, origin_raw)
            self.publish_point(self.pub_origin, origin_map)

        rotation = as_xyz(data.get("lidarRotation"))
        player_body = self._extract_body_angles(data)

        detected_payload, terrain_payload, all_payload, terrain_info_payload = build_classified_lidar_payloads(
            lidar_points,
            timestamp_wall=ts,
            map_frame=MAP_FRAME,
            ground_filter_enabled=self.ground_filter_enabled,
            lidar_origin_map_for_correction=origin_map,
            lidar_rotation_deg=rotation,
            player_body_deg=player_body,
            grid_resolution=self.terrain_grid_resolution,
            climb_limit=self.terrain_climb_limit,
            obstacle_min_height=self.terrain_obstacle_min_height,
        )
        # 구버전 디버깅 스크립트를 위해 origin_y를 payload 메타데이터에 유지한다.
        detected_payload["origin_y"] = origin_y
        terrain_payload["origin_y"] = origin_y
        all_payload["origin_y"] = origin_y

        self.publish_int(self.pub_points_count, lidar_count)
        # 수천 개 LiDAR 포인트의 JSON 직렬화는 비용이 크다. 구형 디버그 도구를
        # 위해 파라미터 뒤에 둔다. 정상 경로(fast path)는 PC2다.
        if self.publish_legacy_lidar_json:
            self.publish_json(self.pub_points, points_payload)

        self.pub_all_detected_map.publish(self.create_pc2(all_payload, MAP_FRAME))
        self.pub_detected_map.publish(self.create_pc2(detected_payload, MAP_FRAME))
        self.pub_terrain_map.publish(self.create_pc2(terrain_payload, MAP_FRAME))
        self.publish_json(self.pub_terrain_info, terrain_info_payload)
        if rotation is not None:
            self.publish_vector3(self.pub_rotation, rotation, UNITY_FRAME)

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
