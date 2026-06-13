#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Polar LiDAR debug viewer.

PC2 optimization policy:
- LiDAR point data is received from lidar_processor_node as PointCloud2.
- DBSCAN itself is still performed only by lidar_dbscan_cluster_node.
- This viewer subscribes to the cluster result JSON and overlays the clustered
  points/centroids in polar coordinates for debugging.

The original raw JSON LiDAR fields(channelIndex, angle, distance) are not used
here because they are intentionally removed from the high-rate downstream path.
"""

from __future__ import annotations

import json
import math
import threading
import time
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2


TOPIC_LIDAR_PC2 = "/tank/sensor/lidar/detected_points_map"
TOPIC_PLAYER_POSE = "/tank/player/pose"
TOPIC_PLAYER_STATE = "/tank/player/state"
TOPIC_CLUSTERS = "/tank/visual_perception/lidar_clusters"

# Polar viewer 전용 색상이다. ROS 메시지/알고리즘에는 영향을 주지 않는다.
CLUSTER_COLORS = [
    "tab:red", "tab:blue", "tab:green", "tab:orange", "tab:purple",
    "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan",
]


def pointcloud2_to_xyz_array(msg: PointCloud2) -> np.ndarray:
    """Convert PointCloud2 XYZ fields to a contiguous float32 (N, 3) array."""
    try:
        arr = point_cloud2.read_points_numpy(
            msg, field_names=("x", "y", "z"), skip_nans=True
        )
    except Exception:
        pts = point_cloud2.read_points(
            msg, field_names=("x", "y", "z"), skip_nans=True
        )
        if isinstance(pts, np.ndarray):
            arr = pts
        else:
            arr = np.asarray(list(pts), dtype=np.float32)

    if arr is None:
        return np.empty((0, 3), dtype=np.float32)

    arr = np.asarray(arr)
    if arr.dtype.fields:
        arr = np.column_stack((arr["x"], arr["y"], arr["z"]))

    arr = np.asarray(arr, dtype=np.float32)
    if arr.size == 0:
        return np.empty((0, 3), dtype=np.float32)

    return np.ascontiguousarray(arr.reshape(-1, 3), dtype=np.float32)


class PolarViewer(Node):
    def __init__(self):
        super().__init__("polar_scan_viewer")

        self.declare_parameter("lidar_topic", TOPIC_LIDAR_PC2)
        self.declare_parameter("pose_topic", TOPIC_PLAYER_POSE)
        self.declare_parameter("state_topic", TOPIC_PLAYER_STATE)
        self.declare_parameter("clusters_topic", TOPIC_CLUSTERS)
        self.declare_parameter("max_render_points", 5000)
        self.declare_parameter("show_clustered_points", True)
        self.declare_parameter("show_unclustered_points", True)

        self.lidar_topic = str(self.get_parameter("lidar_topic").value)
        self.pose_topic = str(self.get_parameter("pose_topic").value)
        self.state_topic = str(self.get_parameter("state_topic").value)
        self.clusters_topic = str(self.get_parameter("clusters_topic").value)
        self.max_render_points = int(self.get_parameter("max_render_points").value)
        self.show_clustered_points = bool(self.get_parameter("show_clustered_points").value)
        self.show_unclustered_points = bool(self.get_parameter("show_unclustered_points").value)

        self.subscription_lidar = self.create_subscription(
            PointCloud2, self.lidar_topic, self.callback_lidar, 10
        )
        self.subscription_pose = self.create_subscription(
            PoseStamped, self.pose_topic, self.callback_pose, 10
        )
        self.subscription_state = self.create_subscription(
            String, self.state_topic, self.callback_state, 10
        )
        self.subscription_clusters = self.create_subscription(
            String, self.clusters_topic, self.callback_clusters, 10
        )

        self.lock = threading.Lock()

        self.tank_x = 0.0
        self.tank_y = 0.0
        self.tank_heading = 0.0

        # columns: theta_rad, radius_m, map_x, map_y, map_z
        self.polar_points = np.empty((0, 5), dtype=np.float32)
        self.clusters: List[Dict] = []

        # 기존 UI 호환용 CH 토글. PC2에는 raw channelIndex가 없으므로 CH1은 전체 PC2 point를 의미한다.
        self.visible = {ch: True for ch in range(1, 17)}
        self.channel_colors = {
            1: "lightgray", 2: "blue", 3: "green", 4: "orange",
            5: "purple", 6: "brown", 7: "pink", 8: "black",
            9: "cyan", 10: "magenta", 11: "yellow", 12: "lime",
            13: "teal", 14: "navy", 15: "maroon", 16: "olive",
        }

        plt.ion()
        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_axes([0.05, 0.05, 0.70, 0.90], projection="polar")
        check_ax = self.fig.add_axes([0.80, 0.10, 0.15, 0.80])

        labels = [f"CH{i}" for i in range(1, 17)]
        states = [True] + [False] * 15
        self.check = CheckButtons(check_ax, labels, states)
        self.check.on_clicked(self.on_click)
        self.fig.canvas.mpl_connect("close_event", self.on_close)

        self.running = True

        self.raw_point_scatter = self.ax.scatter(
            [], [], s=5, c=self.channel_colors[1], alpha=0.35, label="PC2 points"
        )
        self.cluster_point_scatter = self.ax.scatter(
            [], [], s=18, alpha=0.9, zorder=4, label="clustered points"
        )
        self.cluster_center_scatter = self.ax.scatter(
            [], [], s=220, marker="o", facecolors="none", edgecolors="magenta",
            linewidth=2, zorder=5, label="cluster centroid"
        )
        self.cluster_texts = []

        self.ax.set_theta_zero_location("N")
        self.ax.set_theta_direction(-1)
        self.ax.grid(True)
        self.ax.legend(loc="upper right", bbox_to_anchor=(1.18, 1.1), fontsize=8)
        self.title_text = self.ax.set_title("LiDAR PC2 Polar Scan & DBSCAN Clusters")

        self.get_logger().info(
            f"polar_scan_viewer started: lidar={self.lidar_topic}(PC2), clusters={self.clusters_topic}"
        )

    def on_close(self, _event):
        self.running = False

    def on_click(self, label):
        ch = int(label.replace("CH", ""))
        self.visible[ch] = not self.visible[ch]

    def callback_pose(self, msg: PoseStamped):
        self.tank_x = float(msg.pose.position.x)
        self.tank_y = float(msg.pose.position.y)

    def callback_state(self, msg: String):
        try:
            data = json.loads(msg.data)
            body = data.get("body", {})
            if "x" in body:
                self.tank_heading = float(body["x"])
            elif "playerBodyX" in data:
                self.tank_heading = float(data["playerBodyX"])
        except Exception:
            pass

    def callback_lidar(self, msg: PointCloud2):
        try:
            xyz = pointcloud2_to_xyz_array(msg)
        except Exception as exc:
            self.get_logger().debug(f"PointCloud2 read failed: {exc}")
            return

        if len(xyz) == 0:
            with self.lock:
                self.polar_points = np.empty((0, 5), dtype=np.float32)
            return

        if self.max_render_points > 0 and len(xyz) > self.max_render_points:
            step = max(1, len(xyz) // self.max_render_points)
            xyz = xyz[::step]

        dx = xyz[:, 0] - self.tank_x
        dy = xyz[:, 1] - self.tank_y
        radius = np.hypot(dx, dy)
        mask = radius > 0.05
        if not np.any(mask):
            with self.lock:
                self.polar_points = np.empty((0, 5), dtype=np.float32)
            return

        xyz = xyz[mask]
        dx = dx[mask]
        dy = dy[mask]
        radius = radius[mask]

        global_bearing = np.degrees(np.arctan2(dx, dy))
        rel_bearing = (global_bearing - self.tank_heading + 180.0) % 360.0 - 180.0
        theta = np.radians(rel_bearing)

        polar_points = np.column_stack(
            [theta, radius, xyz[:, 0], xyz[:, 1], xyz[:, 2]]
        ).astype(np.float32, copy=False)

        with self.lock:
            self.polar_points = polar_points

    def callback_clusters(self, msg: String):
        try:
            payload = json.loads(msg.data)
            clusters_data = payload.get("clusters", [])
            parsed_clusters = []

            for c in clusters_data:
                centroid = c.get("centroid", {}) or {}
                bbox = c.get("bbox", {}) or {}

                cx = float(centroid.get("x", 0.0))
                cy = float(centroid.get("y", 0.0))
                cz = float(centroid.get("z", 0.0))

                dx = cx - self.tank_x
                dy = cy - self.tank_y
                distance_2d = math.hypot(dx, dy)
                global_bearing = math.degrees(math.atan2(dx, dy))
                rel_bearing_deg = (global_bearing - self.tank_heading + 180.0) % 360.0 - 180.0

                parsed_clusters.append(
                    {
                        "id": int(c.get("id", 0)),
                        "count": int(c.get("count", 0)),
                        "theta": math.radians(rel_bearing_deg),
                        "radius": distance_2d,
                        "centroid": (cx, cy, cz),
                        "bbox": {
                            "x_min": float(bbox.get("x_min", cx)),
                            "x_max": float(bbox.get("x_max", cx)),
                            "y_min": float(bbox.get("y_min", cy)),
                            "y_max": float(bbox.get("y_max", cy)),
                            "z_min": float(bbox.get("z_min", cz)),
                            "z_max": float(bbox.get("z_max", cz)),
                        },
                    }
                )

            with self.lock:
                self.clusters = parsed_clusters
        except Exception as exc:
            self.get_logger().debug(f"Clusters parse failed: {exc}")

    @staticmethod
    def _points_in_bbox(points: np.ndarray, bbox: Dict[str, float], pad: float = 0.05) -> np.ndarray:
        if points.size == 0:
            return np.zeros((0,), dtype=bool)
        x = points[:, 2]
        y = points[:, 3]
        z = points[:, 4]
        return (
            (x >= bbox["x_min"] - pad) & (x <= bbox["x_max"] + pad) &
            (y >= bbox["y_min"] - pad) & (y <= bbox["y_max"] + pad) &
            (z >= bbox["z_min"] - pad) & (z <= bbox["z_max"] + pad)
        )

    def _build_cluster_point_overlay(
        self, points: np.ndarray, clusters: List[Dict]
    ) -> Tuple[np.ndarray, List[str], np.ndarray]:
        """Return clustered point offsets/colors and an unclustered mask.

        The DBSCAN node does not publish every member point in JSON because that
        would reintroduce the original serialization bottleneck.  For visualization
        only, this viewer maps PC2 points into each cluster bbox from the lightweight
        cluster JSON result.
        """
        if points.size == 0 or not clusters:
            return np.empty((0, 2)), [], np.ones((len(points),), dtype=bool)

        assigned = np.zeros((len(points),), dtype=bool)
        overlay_offsets = []
        overlay_colors: List[str] = []

        for idx, cluster in enumerate(clusters):
            bbox = cluster.get("bbox") or {}
            mask = self._points_in_bbox(points, bbox) & (~assigned)
            if not np.any(mask):
                continue

            cluster_points = points[mask]
            overlay_offsets.append(cluster_points[:, :2])
            overlay_colors.extend([CLUSTER_COLORS[idx % len(CLUSTER_COLORS)]] * len(cluster_points))
            assigned[mask] = True

        if overlay_offsets:
            offsets = np.vstack(overlay_offsets)
        else:
            offsets = np.empty((0, 2))

        return offsets, overlay_colors, ~assigned

    def update(self):
        with self.lock:
            points = self.polar_points.copy()
            clusters = list(self.clusters)

        total_points = len(points)
        max_r = 1.0

        for txt in self.cluster_texts:
            txt.remove()
        self.cluster_texts.clear()

        cluster_offsets, cluster_colors, unclustered_mask = self._build_cluster_point_overlay(points, clusters)

        # CH1 토글은 PC2 전체/비클러스터 point 배경 표시를 의미한다.
        if self.visible.get(1, True) and self.show_unclustered_points and total_points > 0:
            raw_points = points[unclustered_mask] if len(unclustered_mask) == total_points else points
            self.raw_point_scatter.set_offsets(raw_points[:, :2] if len(raw_points) else np.empty((0, 2)))
            if len(raw_points):
                max_r = max(max_r, float(np.max(raw_points[:, 1])))
        else:
            self.raw_point_scatter.set_offsets(np.empty((0, 2)))

        if self.show_clustered_points and len(cluster_offsets) > 0:
            self.cluster_point_scatter.set_offsets(cluster_offsets)
            self.cluster_point_scatter.set_color(cluster_colors)
            max_r = max(max_r, float(np.max(cluster_offsets[:, 1])))
        else:
            self.cluster_point_scatter.set_offsets(np.empty((0, 2)))

        if clusters:
            c_thetas = []
            c_radii = []
            for cluster in clusters:
                theta = float(cluster["theta"])
                radius = float(cluster["radius"])
                c_thetas.append(theta)
                c_radii.append(radius)
                max_r = max(max_r, radius)

                txt = self.ax.text(
                    theta,
                    radius + 1.5,
                    f"C{cluster['id']} N={cluster['count']} D={radius:.1f}m",
                    fontsize=9,
                    fontweight="bold",
                    color="magenta",
                    ha="center",
                    va="bottom",
                    zorder=6,
                )
                self.cluster_texts.append(txt)

            self.cluster_center_scatter.set_offsets(np.column_stack([c_thetas, c_radii]))
        else:
            self.cluster_center_scatter.set_offsets(np.empty((0, 2)))

        self.ax.set_rmax(max_r + 5.0)
        self.title_text.set_text(
            f"LiDAR PC2 Polar Scan ({total_points} pts) | DBSCAN clusters: {len(clusters)}"
        )

        plt.draw()
        plt.pause(0.001)


def main(args=None):
    rclpy.init(args=args)
    node = PolarViewer()

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    try:
        while rclpy.ok() and node.running:
            node.update()
            time.sleep(0.03)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
