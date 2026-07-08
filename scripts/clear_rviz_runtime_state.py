#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clear RViz runtime-only visualization state before a new scenario starts.

This script intentionally clears visualization topics only. It does not delete
static map source files, scenario reports, mission plans, or terrain NPZ files.
It is meant to run for a few seconds while RViz/visualization publishers are
starting, so a newly opened RViz does not inherit stale markers from the
previous run.
"""

from __future__ import annotations

import argparse
import time
from typing import Iterable, List

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray

MAP_FRAME = "tank_map"

# Runtime/dynamic markers that must not survive a new scenario.
RUNTIME_MARKER_TOPICS: List[str] = [
    "/tank/rviz/object_markers",
    "/tank/rviz/obstacle_markers",
    "/tank/rviz/lidar_markers",
    "/tank/rviz/risk_markers",
    "/tank/rviz/potential_markers",
    "/tank/rviz/dynamic_avoidance_markers",
    "/tank/rviz/fused_object_markers",
    "/tank/rviz/discovered_object_markers",
    "/tank/rviz/lidar_cluster_markers",
    "/tank/rviz/phone_sim2real_markers",
    "/tank/rviz/phone_sim2real_image_cluster_markers",
    "/tank/terrain/final_elevation_markers",
    "/tank/terrain/final_wireframe_markers",
]

# Static/map-like marker topics. These are also cleared during startup, then the
# current static_map_loader republishes the correct map again. This removes
# stale transient/local RViz contents from older launches.
STATIC_MARKER_TOPICS: List[str] = [
    "/tank/rviz/terrain_markers",
    "/tank/rviz/recon_map_markers",
    "/tank/rviz/mission_map_markers",
    "/tank/rviz/map_diff_markers",
    "/tank/rviz/static_avoidance_markers",
]

TERRAIN_CLOUD_TOPICS: List[str] = [
    "/tank/terrain/final_accumulated_cloud",
    "/tank/terrain/final_ground_points",
    "/tank/terrain/final_non_ground_points",
]


class RvizRuntimeClearNode(Node):
    def __init__(self, marker_topics: Iterable[str], cloud_topics: Iterable[str], frame_id: str) -> None:
        super().__init__("tank_rviz_runtime_clear_node")
        self.frame_id = frame_id
        self.marker_pubs = [self.create_publisher(MarkerArray, topic, 10) for topic in marker_topics]
        self.cloud_pubs = [self.create_publisher(PointCloud2, topic, 10) for topic in cloud_topics]

    def _delete_all_marker_array(self) -> MarkerArray:
        stamp = self.get_clock().now().to_msg()
        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = stamp
        marker.action = Marker.DELETEALL
        arr = MarkerArray()
        arr.markers.append(marker)
        return arr

    def _empty_cloud(self) -> PointCloud2:
        header = Header()
        header.frame_id = self.frame_id
        header.stamp = self.get_clock().now().to_msg()
        return point_cloud2.create_cloud_xyz32(header, [])

    def publish_clear_once(self) -> None:
        marker_msg = self._delete_all_marker_array()
        for pub in self.marker_pubs:
            pub.publish(marker_msg)

        cloud_msg = self._empty_cloud()
        for pub in self.cloud_pubs:
            pub.publish(cloud_msg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clear RViz runtime marker/cloud topics.")
    parser.add_argument("--duration", type=float, default=3.0, help="seconds to keep publishing clear messages")
    parser.add_argument("--rate", type=float, default=8.0, help="clear publish rate in Hz")
    parser.add_argument("--frame", default=MAP_FRAME, help="RViz frame_id")
    parser.add_argument("--include-static", action="store_true", help="also clear static map marker topics")
    parser.add_argument("--no-empty-clouds", action="store_true", help="do not publish empty PointCloud2 messages")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    marker_topics = list(RUNTIME_MARKER_TOPICS)
    if args.include_static:
        marker_topics.extend(STATIC_MARKER_TOPICS)
    cloud_topics = [] if args.no_empty_clouds else list(TERRAIN_CLOUD_TOPICS)

    rclpy.init()
    node = RvizRuntimeClearNode(marker_topics, cloud_topics, args.frame)
    try:
        # Let discovery connect to RViz subscribers/publishers.
        deadline = time.time() + max(0.1, args.duration)
        period = 1.0 / max(0.5, args.rate)
        first = True
        while first or time.time() < deadline:
            first = False
            node.publish_clear_once()
            rclpy.spin_once(node, timeout_sec=0.02)
            time.sleep(period)
        node.get_logger().info(
            f"cleared {len(marker_topics)} marker topics and {len(cloud_topics)} cloud topics"
        )
        return 0
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
