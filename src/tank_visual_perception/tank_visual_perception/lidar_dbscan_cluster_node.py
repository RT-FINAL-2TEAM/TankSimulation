# -*- coding: utf-8 -*-
"""
ROS2 node: LiDAR detected map points -> lightweight DBSCAN clusters.

Team command compatibility:
  ros2 run tank_visual_perception lidar_dbscan_cluster_node \
    --ros-args \
    -p eps:=1.5 \
    -p min_samples:=2 \
    -p min_cluster_size:=2

Subscribe:
  /tank/sensor/lidar/detected_points_map      std_msgs/String
  /tank/player/pose                           geometry_msgs/PoseStamped  <-- [추가됨] 전차 위치

Publish:
  /tank/visual_perception/lidar_clusters      std_msgs/String
  /tank/rviz/lidar_cluster_markers            visualization_msgs/MarkerArray
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import rclpy
from geometry_msgs.msg import Point, PoseStamped
from rclpy.node import Node
from std_msgs.msg import ColorRGBA, String
from visualization_msgs.msg import Marker, MarkerArray

Point2D = Tuple[float, float]
Point3D = Tuple[float, float, float]


@dataclass
class Cluster:
    cluster_id: int
    points: List[Point3D]

    @property
    def count(self) -> int:
        return len(self.points)

    @property
    def centroid(self) -> Point3D:
        n = max(1, len(self.points))
        return (
            sum(p[0] for p in self.points) / n,
            sum(p[1] for p in self.points) / n,
            sum(p[2] for p in self.points) / n,
        )

    @property
    def bbox(self) -> Dict[str, float]:
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        zs = [p[2] for p in self.points]
        return {
            "x_min": min(xs), "x_max": max(xs),
            "y_min": min(ys), "y_max": max(ys),
            "z_min": min(zs), "z_max": max(zs),
        }


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _extract_position_map(point: Dict[str, Any]) -> Optional[Point3D]:
    pos = point.get("position_map")
    if isinstance(pos, dict):
        return (_to_float(pos.get("x")), _to_float(pos.get("y")), _to_float(pos.get("z")))

    # Conservative fallback for older payloads.
    pos = point.get("position")
    if isinstance(pos, dict):
        if "z" in pos:
            # lidar raw convention: map.x=raw.x, map.y=raw.z, map.z=raw.y
            return (_to_float(pos.get("x")), _to_float(pos.get("z")), _to_float(pos.get("y")))
        return (_to_float(pos.get("x")), _to_float(pos.get("y")), 0.0)

    if point.get("map_x") is not None and point.get("map_y") is not None:
        return (_to_float(point.get("map_x")), _to_float(point.get("map_y")), _to_float(point.get("map_z")))
    return None


def parse_lidar_payload(payload: Dict[str, Any]) -> List[Point3D]:
    raw_points = payload.get("points") if isinstance(payload, dict) else []
    if isinstance(raw_points, dict):
        raw_points = raw_points.get("points", [])
    if not isinstance(raw_points, list):
        return []

    points: List[Point3D] = []
    for p in raw_points:
        if not isinstance(p, dict):
            continue
        pos = _extract_position_map(p)
        if pos is None:
            continue
        if not all(math.isfinite(v) for v in pos):
            continue
        points.append(pos)
    return points


def _dist2(a: Point3D, b: Point3D) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx * dx + dy * dy


def dbscan(points: Sequence[Point3D], eps: float, min_samples: int) -> List[int]:
    """Small dependency-free DBSCAN over map-plane x/y."""
    n = len(points)
    if n == 0:
        return []

    eps2 = eps * eps
    labels = [-99] * n  # -99: unvisited, -1: noise, >=0: cluster id

    neighbors_cache: List[Optional[List[int]]] = [None] * n

    def neighbors(i: int) -> List[int]:
        cached = neighbors_cache[i]
        if cached is not None:
            return cached
        res = [j for j in range(n) if _dist2(points[i], points[j]) <= eps2]
        neighbors_cache[i] = res
        return res

    cluster_id = 0
    for i in range(n):
        if labels[i] != -99:
            continue
        neigh = neighbors(i)
        if len(neigh) < min_samples:
            labels[i] = -1
            continue

        labels[i] = cluster_id
        seeds = list(neigh)
        k = 0
        while k < len(seeds):
            j = seeds[k]
            if labels[j] == -1:
                labels[j] = cluster_id
            if labels[j] != -99:
                k += 1
                continue
            labels[j] = cluster_id
            neigh_j = neighbors(j)
            if len(neigh_j) >= min_samples:
                for candidate in neigh_j:
                    if candidate not in seeds:
                        seeds.append(candidate)
            k += 1
        cluster_id += 1
    return labels


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


class LidarDbscanClusterNode(Node):
    def __init__(self) -> None:
        super().__init__("lidar_dbscan_cluster_node")

        self.declare_parameter("input_topic", "/tank/sensor/lidar/detected_points_map")
        self.declare_parameter("pose_topic", "/tank/player/pose") # [추가됨] 전차 위치 토픽
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

        # [추가됨] 전차 위치 저장용 변수
        self.tank_x = 0.0
        self.tank_y = 0.0

        self.sub = self.create_subscription(String, self.input_topic, self.on_lidar, 10)
        self.sub_pose = self.create_subscription(PoseStamped, self.pose_topic, self.on_pose, 10) # [추가됨] 구독
        
        self.pub_clusters = self.create_publisher(String, self.clusters_topic, 10)
        self.pub_markers = self.create_publisher(MarkerArray, self.markers_topic, 10)

        self.get_logger().info(
            f"lidar_dbscan_cluster_node started: input={self.input_topic}, eps={self.eps}, "
            f"min_samples={self.min_samples}, min_cluster_size={self.min_cluster_size}"
        )

    # [추가됨] 전차 위치 업데이트 콜백
    def on_pose(self, msg: PoseStamped) -> None:
        self.tank_x = float(msg.pose.position.x)
        self.tank_y = float(msg.pose.position.y)

    def on_lidar(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
            if not isinstance(payload, dict):
                return
        except Exception as exc:
            self.get_logger().warn(f"failed to parse lidar payload: {exc}")
            return

        points = parse_lidar_payload(payload)
        if self.max_points > 0 and len(points) > self.max_points:
            step = max(1, len(points) // self.max_points)
            points = points[::step][: self.max_points]

        labels = dbscan(points, max(self.eps, 0.001), max(self.min_samples, 1))
        grouped: Dict[int, List[Point3D]] = {}
        noise_count = 0
        for p, label in zip(points, labels):
            if label < 0:
                noise_count += 1
                continue
            grouped.setdefault(label, []).append(p)

        clusters: List[Cluster] = []
        new_id = 0
        for _label, group in sorted(grouped.items()):
            if len(group) < self.min_cluster_size:
                noise_count += len(group)
                continue
            clusters.append(Cluster(new_id, group))
            new_id += 1

        self.publish_clusters(clusters, len(points), noise_count)
        self.publish_markers(clusters)

    def publish_clusters(self, clusters: List[Cluster], point_count: int, noise_count: int) -> None:
        data = {
            "timestamp_ros_sec": self.get_clock().now().nanoseconds * 1e-9,
            "frame_id": self.frame_id,
            "algorithm": "dbscan_2d_map_xy",
            "eps": self.eps,
            "min_samples": self.min_samples,
            "min_cluster_size": self.min_cluster_size,
            "input_point_count": point_count,
            "noise_count": noise_count,
            "cluster_count": len(clusters),
            "clusters": [],
        }
        for c in clusters:
            cx, cy, cz = c.centroid
            # [수정됨] 0,0 원점이 아니라 현재 전차 위치(tank_x, tank_y)를 기준으로 거리 계산!
            nearest = min(math.hypot(p[0] - self.tank_x, p[1] - self.tank_y) for p in c.points) if c.points else None
            bbox = c.bbox
            data["clusters"].append(
                {
                    "id": c.cluster_id,
                    "count": c.count,
                    "centroid": {"x": cx, "y": cy, "z": cz},
                    # raw coordinate mirrors the project policy: raw.x=map.x, raw.y=map.z, raw.z=map.y
                    "centroid_raw": {"x": cx, "y": cz, "z": cy},
                    "bbox": bbox,
                    "bbox_raw": {
                        "x_min": bbox["x_min"], "x_max": bbox["x_max"],
                        "y_min": bbox["z_min"], "y_max": bbox["z_max"],
                        "z_min": bbox["y_min"], "z_max": bbox["y_max"],
                    },
                    "nearest_origin_distance_m": nearest,
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
            cx, cy, cz = c.centroid
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
            
            # [수정됨] RViz 텍스트 마커에도 원점이 아닌 '현재 전차 위치 기준' 거리가 뜹니다!
            nearest = min(math.hypot(p[0] - self.tank_x, p[1] - self.tank_y) for p in c.points) if c.points else 0.0
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