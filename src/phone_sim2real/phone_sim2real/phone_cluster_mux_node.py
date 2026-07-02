# -*- coding: utf-8 -*-
"""Merge real LiDAR clusters and phone_sim2real synthetic clusters.

목적:
  - phone_sim2real이 /tank/visual_perception/lidar_clusters에 직접 publish하지 않게 한다.
  - 실제 LiDAR cluster와 스마트폰 synthetic cluster를 하나의 JSON payload로 합쳐 planner에 공급한다.

기본 구조:
  real lidar cluster:
    /tank/visual_perception/lidar_clusters
  phone synthetic cluster:
    /tank/phone_sim2real/synthetic_lidar_clusters
  mux output:
    /tank/phone_sim2real/muxed_lidar_clusters

planner는 TANK_TOPIC_LIDAR_CLUSTERS=/tank/phone_sim2real/muxed_lidar_clusters 로 실행한다.
"""

from __future__ import annotations

import json
import time
from copy import deepcopy
from typing import Any, Dict, List, Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


def _safe_json_loads(data: str, default: Any = None) -> Any:
    try:
        return json.loads(data)
    except Exception:
        return default


def _as_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


class PhoneClusterMuxNode(Node):
    def __init__(self) -> None:
        super().__init__("phone_cluster_mux_node")

        self.declare_parameter("real_cluster_topic", "/tank/visual_perception/lidar_clusters")
        self.declare_parameter("phone_cluster_topic", "/tank/phone_sim2real/synthetic_lidar_clusters")
        self.declare_parameter("muxed_cluster_topic", "/tank/phone_sim2real/muxed_lidar_clusters")
        self.declare_parameter("status_topic", "/tank/phone_sim2real/cluster_mux_status")

        self.declare_parameter("publish_rate_hz", 10.0)
        self.declare_parameter("real_ttl_sec", 0.7)
        self.declare_parameter("phone_ttl_sec", 1.2)
        self.declare_parameter("dedupe_distance_m", 2.0)
        self.declare_parameter("phone_id_offset", 9000)
        self.declare_parameter("prefer_phone_on_overlap", True)

        self.real_topic = str(self.get_parameter("real_cluster_topic").value)
        self.phone_topic = str(self.get_parameter("phone_cluster_topic").value)
        self.muxed_topic = str(self.get_parameter("muxed_cluster_topic").value)
        self.status_topic = str(self.get_parameter("status_topic").value)

        self.real_ttl_sec = float(self.get_parameter("real_ttl_sec").value)
        self.phone_ttl_sec = float(self.get_parameter("phone_ttl_sec").value)
        self.dedupe_distance_m = float(self.get_parameter("dedupe_distance_m").value)
        self.phone_id_offset = int(self.get_parameter("phone_id_offset").value)
        self.prefer_phone_on_overlap = bool(self.get_parameter("prefer_phone_on_overlap").value)

        self.last_real_msg: Optional[Dict[str, Any]] = None
        self.last_real_wall = 0.0
        self.last_phone_msg: Optional[Dict[str, Any]] = None
        self.last_phone_wall = 0.0

        self.real_sub = self.create_subscription(String, self.real_topic, self._on_real, 10)
        self.phone_sub = self.create_subscription(String, self.phone_topic, self._on_phone, 10)
        self.pub = self.create_publisher(String, self.muxed_topic, 10)
        self.status_pub = self.create_publisher(String, self.status_topic, 10)

        hz = max(1.0, float(self.get_parameter("publish_rate_hz").value))
        self.timer = self.create_timer(1.0 / hz, self._on_timer)

        self.get_logger().info(
            "phone cluster mux started: real=%s, phone=%s, out=%s"
            % (self.real_topic, self.phone_topic, self.muxed_topic)
        )

    def _on_real(self, msg: String) -> None:
        payload = _safe_json_loads(msg.data, None)
        if not isinstance(payload, dict):
            return

        # mux output이 다시 real input으로 들어오는 loop 방지.
        if payload.get("algorithm") == "phone_sim2real_cluster_mux":
            return
        if payload.get("source") == "phone_sim2real_cluster_mux":
            return

        self.last_real_msg = payload
        self.last_real_wall = time.time()

    def _on_phone(self, msg: String) -> None:
        payload = _safe_json_loads(msg.data, None)
        if not isinstance(payload, dict):
            return
        self.last_phone_msg = payload
        self.last_phone_wall = time.time()

    @staticmethod
    def _centroid_xy(cluster: Dict[str, Any]):
        c = cluster.get("centroid", {})
        try:
            return float(c.get("x", 0.0)), float(c.get("y", 0.0))
        except Exception:
            return 0.0, 0.0

    def _overlaps(self, a: Dict[str, Any], b: Dict[str, Any]) -> bool:
        ax, ay = self._centroid_xy(a)
        bx, by = self._centroid_xy(b)
        return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5 <= self.dedupe_distance_m

    def _renumber_phone_clusters(self, clusters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out = []
        for i, c in enumerate(clusters):
            cc = deepcopy(c)
            cc["id"] = self.phone_id_offset + i
            cc["source"] = cc.get("source", "phone_sim2real_image_cluster")
            cc["is_phone_synthetic"] = True
            out.append(cc)
        return out

    def _merge(self) -> Dict[str, Any]:
        now = time.time()

        real_valid = self.last_real_msg is not None and (now - self.last_real_wall) <= self.real_ttl_sec
        phone_valid = self.last_phone_msg is not None and (now - self.last_phone_wall) <= self.phone_ttl_sec

        real_clusters = _as_list((self.last_real_msg or {}).get("clusters")) if real_valid else []
        phone_clusters = _as_list((self.last_phone_msg or {}).get("clusters")) if phone_valid else []
        phone_clusters = self._renumber_phone_clusters(phone_clusters)

        merged_real = []
        removed_real = 0

        for rc in real_clusters:
            overlap = any(self._overlaps(rc, pc) for pc in phone_clusters)
            if overlap and self.prefer_phone_on_overlap:
                removed_real += 1
                continue
            merged_real.append(deepcopy(rc))

        clusters = merged_real + phone_clusters

        base = deepcopy(self.last_real_msg) if real_valid and isinstance(self.last_real_msg, dict) else {}
        payload = {
            "timestamp_ros_sec": now,
            "timestamp_wall": now,
            "frame_id": base.get("frame_id", (self.last_phone_msg or {}).get("frame_id", "tank_map")),
            "algorithm": "phone_sim2real_cluster_mux",
            "source": "phone_sim2real_cluster_mux",
            "real_cluster_topic": self.real_topic,
            "phone_cluster_topic": self.phone_topic,
            "real_valid": real_valid,
            "phone_valid": phone_valid,
            "real_cluster_count": len(real_clusters),
            "phone_cluster_count": len(phone_clusters),
            "removed_real_overlap_count": removed_real,
            "cluster_count": len(clusters),
            "input_point_count": int(base.get("input_point_count", 0)) + int((self.last_phone_msg or {}).get("input_point_count", 0) if phone_valid else 0),
            "noise_count": int(base.get("noise_count", 0)),
            "clusters": clusters,
        }
        return payload

    def _on_timer(self) -> None:
        payload = self._merge()
        self.pub.publish(String(data=json.dumps(payload, ensure_ascii=False, separators=(",", ":"))))

        status = {
            "ok": True,
            "source": "phone_sim2real_cluster_mux",
            "timestamp_wall": time.time(),
            "real_topic": self.real_topic,
            "phone_topic": self.phone_topic,
            "muxed_topic": self.muxed_topic,
            "real_valid": payload["real_valid"],
            "phone_valid": payload["phone_valid"],
            "real_cluster_count": payload["real_cluster_count"],
            "phone_cluster_count": payload["phone_cluster_count"],
            "cluster_count": payload["cluster_count"],
            "removed_real_overlap_count": payload["removed_real_overlap_count"],
        }
        self.status_pub.publish(String(data=json.dumps(status, ensure_ascii=False, separators=(",", ":"))))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PhoneClusterMuxNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
