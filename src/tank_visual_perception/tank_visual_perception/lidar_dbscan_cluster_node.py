# -*- coding: utf-8 -*-
"""
ROS2 노드: LiDAR로 탐지한 맵 포인트 -> 경량 DBSCAN 클러스터.
(PointCloud2, NumPy, Scikit-Learn으로 최적화)

팀 명령 호환:
  ros2 run tank_visual_perception lidar_dbscan_cluster_node \
    --ros-args \
    -p eps:=1.5 \
    -p min_samples:=2 \
    -p min_cluster_size:=2

Subscribe:
  /tank/sensor/lidar/detected_points_map      sensor_msgs/PointCloud2 <-- 바이너리 최적화됨
  /tank/player/pose                           geometry_msgs/PoseStamped
Publish:
  /tank/visual_perception/lidar_clusters      std_msgs/String         <-- 가벼운 BBox 메타데이터 유지
  /tank/rviz/lidar_cluster_markers            visualization_msgs/MarkerArray
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
from sklearn.cluster import DBSCAN

import rclpy
from geometry_msgs.msg import Point, PoseStamped
from rclpy.node import Node
from std_msgs.msg import ColorRGBA, String
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from visualization_msgs.msg import Marker, MarkerArray


@dataclass
class Cluster:
    cluster_id: int
    points: np.ndarray  # (N, 3) numpy 배열

    @property
    def count(self) -> int:
        return len(self.points)

    @property
    def centroid(self) -> np.ndarray:
        return np.mean(self.points, axis=0)

    @property
    def bbox(self) -> Dict[str, float]:
        mins = np.min(self.points, axis=0)
        maxs = np.max(self.points, axis=0)
        return {
            "x_min": float(mins[0]), "x_max": float(maxs[0]),
            "y_min": float(mins[1]), "y_max": float(maxs[1]),
            "z_min": float(mins[2]), "z_max": float(maxs[2]),
        }


def make_color(r: float, g: float, b: float, a: float = 1.0) -> ColorRGBA:
    c = ColorRGBA()
    c.r = float(r)
    c.g = float(g)
    c.b = float(b)
    c.a = float(a)
    return c


def point_msg(x: float, y: float, z: float = 0.0) -> Point:
    p = Point()
    p.x = float(x)
    p.y = float(y)
    p.z = float(z)
    return p


from tank_common.pointcloud import pointcloud2_to_xyz_array


class LidarDbscanClusterNode(Node):
    def __init__(self) -> None:
        super().__init__("lidar_dbscan_cluster_node")

        self.declare_parameter("input_topic", "/tank/sensor/lidar/detected_points_map")
        self.declare_parameter("pose_topic", "/tank/player/pose")
        self.declare_parameter("clusters_topic", "/tank/visual_perception/lidar_clusters")
        self.declare_parameter("markers_topic", "/tank/rviz/lidar_cluster_markers")
        self.declare_parameter("frame_id", "tank_map")
        self.declare_parameter("eps", 1.5)
        self.declare_parameter("min_samples", 2)
        self.declare_parameter("min_cluster_size", 2)
        self.declare_parameter("max_points", 2500)
        self.declare_parameter("bbox_min_thickness", 0.8)
        self.declare_parameter("text_height", 1.0)

        self.input_topic = str(self.get_parameter("input_topic").value)
        self.pose_topic = str(self.get_parameter("pose_topic").value)
        self.clusters_topic = str(self.get_parameter("clusters_topic").value)
        self.markers_topic = str(self.get_parameter("markers_topic").value)
        self.frame_id = str(self.get_parameter("frame_id").value)
        self.eps = float(self.get_parameter("eps").value)
        self.min_samples = int(self.get_parameter("min_samples").value)
        self.min_cluster_size = int(self.get_parameter("min_cluster_size").value)
        self.max_points = int(self.get_parameter("max_points").value)
        self.bbox_min_thickness = float(self.get_parameter("bbox_min_thickness").value)
        self.text_height = float(self.get_parameter("text_height").value)

        self.tank_x = 0.0
        self.tank_y = 0.0

        # [수정됨] JSON(String) 구독을 PointCloud2 구독으로 변경
        self.sub = self.create_subscription(PointCloud2, self.input_topic, self.on_lidar, 10)
        self.sub_pose = self.create_subscription(PoseStamped, self.pose_topic, self.on_pose, 10)
        
        self.pub_clusters = self.create_publisher(String, self.clusters_topic, 10)
        self.pub_markers = self.create_publisher(MarkerArray, self.markers_topic, 10)
        
        # Scikit-learn DBSCAN 인스턴스 초기화 (KD-Tree 알고리즘 사용)
        self.dbscan_algo = DBSCAN(
            eps=max(self.eps, 0.001),
            min_samples=max(self.min_samples, 1),
            algorithm='kd_tree'
        )

        self.get_logger().info(
            f"lidar_dbscan_cluster_node started: input={self.input_topic}(PC2), eps={self.eps}, "
            f"min_samples={self.min_samples}, min_cluster_size={self.min_cluster_size}"
        )

    def on_pose(self, msg: PoseStamped) -> None:
        self.tank_x = float(msg.pose.position.x)
        self.tank_y = float(msg.pose.position.y)

    def on_lidar(self, msg: PointCloud2) -> None:
        try:
            # JSON 파싱 없이 바이너리 PointCloud2를 즉시 NumPy 배열로 변환
            points = pointcloud2_to_xyz_array(msg)
        except Exception as exc:
            self.get_logger().warn(f"failed to read PointCloud2: {exc}")
            return

        original_point_count = len(points)
        
        if original_point_count == 0:
            return

        # 다운샘플링: 랜덤 샘플링을 사용하여 공간 정보를 고르게 유지
        if self.max_points > 0 and original_point_count > self.max_points:
            indices = np.random.choice(original_point_count, self.max_points, replace=False)
            points = points[indices]

        # DBSCAN 클러스터링 (2D x,y 기준)
        points_2d = points[:, :2]
        labels = self.dbscan_algo.fit_predict(points_2d)

        # 결과 그룹화
        clusters: List[Cluster] = []
        noise_count = np.sum(labels == -1)
        
        unique_labels = set(labels)
        unique_labels.discard(-1) # 노이즈 라벨 제거

        new_id = 0
        for label in unique_labels:
            mask = (labels == label)
            group_points = points[mask]
            
            if len(group_points) < self.min_cluster_size:
                noise_count += len(group_points)
                continue
                
            clusters.append(Cluster(new_id, group_points))
            new_id += 1

        self.publish_clusters(clusters, len(points), int(noise_count))
        self.publish_markers(clusters)

    def publish_clusters(self, clusters: List[Cluster], point_count: int, noise_count: int) -> None:
        data = {
            "timestamp_ros_sec": self.get_clock().now().nanoseconds * 1e-9,
            "frame_id": self.frame_id,
            "algorithm": "sklearn_dbscan_2d_map_xy",
            "eps": self.eps,
            "min_samples": self.min_samples,
            "min_cluster_size": self.min_cluster_size,
            "input_point_count": point_count,
            "noise_count": noise_count,
            "cluster_count": len(clusters),
            "clusters": [],
        }
        
        for c in clusters:
            cx, cy, cz = float(c.centroid[0]), float(c.centroid[1]), float(c.centroid[2])
            
            # [최적화] NumPy 벡터 연산으로 가장 가까운 거리 계산
            if c.count > 0:
                distances = np.hypot(c.points[:, 0] - self.tank_x, c.points[:, 1] - self.tank_y)
                nearest = float(np.min(distances))
            else:
                nearest = None
                
            bbox = c.bbox
            data["clusters"].append(
                {
                    "id": c.cluster_id,
                    "count": c.count,
                    "centroid": {"x": cx, "y": cy, "z": cz},
                    "centroid_raw": {"x": cx, "y": cz, "z": cy},
                    "bbox": bbox,
                    "bbox_raw": {
                        "x_min": bbox["x_min"], "x_max": bbox["x_max"],
                        "y_min": bbox["z_min"], "y_max": bbox["z_max"],
                        "z_min": bbox["y_min"], "z_max": bbox["y_max"],
                    },
                    "nearest_tank_distance_m": nearest,
                }
            )
            
        out = String()
        out.data = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
        self.pub_clusters.publish(out)

    def publish_markers(self, clusters: List[Cluster]) -> None:
        arr = MarkerArray()

        clear = Marker()
        clear.header.frame_id = self.frame_id
        clear.header.stamp = self.get_clock().now().to_msg()
        clear.ns = "lidar_clusters"
        clear.id = 0
        clear.action = Marker.DELETEALL
        arr.markers.append(clear)

        marker_id = 1
        for c in clusters:
            cx, cy, cz = float(c.centroid[0]), float(c.centroid[1]), float(c.centroid[2])
            bbox = c.bbox
            sx = max(self.bbox_min_thickness, bbox["x_max"] - bbox["x_min"])
            sy = max(self.bbox_min_thickness, bbox["y_max"] - bbox["y_min"])
            sz = max(self.bbox_min_thickness, bbox["z_max"] - bbox["z_min"])
            center_z = max(0.4, (bbox["z_min"] + bbox["z_max"]) * 0.5)

            cube = Marker()
            cube.header.frame_id = self.frame_id
            cube.header.stamp = self.get_clock().now().to_msg()
            cube.ns = "lidar_cluster_bbox"
            cube.id = marker_id
            marker_id += 1
            cube.type = Marker.CUBE
            cube.action = Marker.ADD
            cube.pose.position = point_msg((bbox["x_min"] + bbox["x_max"]) * 0.5, (bbox["y_min"] + bbox["y_max"]) * 0.5, center_z)
            cube.pose.orientation.w = 1.0
            cube.scale.x = sx
            cube.scale.y = sy
            cube.scale.z = sz
            cube.color = make_color(0.0, 0.8, 1.0, 0.22)
            arr.markers.append(cube)

            sphere = Marker()
            sphere.header.frame_id = self.frame_id
            sphere.header.stamp = self.get_clock().now().to_msg()
            sphere.ns = "lidar_cluster_centroid"
            sphere.id = marker_id
            marker_id += 1
            sphere.type = Marker.SPHERE
            sphere.action = Marker.ADD
            sphere.pose.position = point_msg(cx, cy, max(0.5, cz))
            sphere.pose.orientation.w = 1.0
            sphere.scale.x = 1.0
            sphere.scale.y = 1.0
            sphere.scale.z = 1.0
            sphere.color = make_color(0.0, 0.4, 1.0, 0.75)
            arr.markers.append(sphere)

            text = Marker()
            text.header.frame_id = self.frame_id
            text.header.stamp = self.get_clock().now().to_msg()
            text.ns = "lidar_cluster_label"
            text.id = marker_id
            marker_id += 1
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position = point_msg(cx, cy, max(1.5, center_z + sz * 0.5 + 0.8))
            text.pose.orientation.w = 1.0
            text.scale.z = self.text_height
            text.color = make_color(1.0, 1.0, 1.0, 1.0)
            
            if c.count > 0:
                distances = np.hypot(c.points[:, 0] - self.tank_x, c.points[:, 1] - self.tank_y)
                nearest = float(np.min(distances))
            else:
                nearest = 0.0
                
            text.text = f"cluster {c.cluster_id}\nN={c.count}\nD={nearest:.1f}m"
            arr.markers.append(text)

        self.pub_markers.publish(arr)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LidarDbscanClusterNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
