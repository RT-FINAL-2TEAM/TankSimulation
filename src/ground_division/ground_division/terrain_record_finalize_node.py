# -*- coding: utf-8 -*-
"""
ground_division/terrain_record_finalize_node.py

주행 중 LiDAR hit point를 계속 기록하고, 주행 종료 후 service 호출로
누적 point cloud -> voxel downsampling -> RViz topic publish -> 파일 저장을 수행한다.

기본 모드에서는 lidar_processor_node가 이미 분리한 terrain_points_map/detected_points_map을
그대로 사용한다. 즉, RViz에서 파란색/주황색으로 확인하는 실시간 분리 결과와
최종 지형 생성 결과가 같은 기준을 쓰게 한다.

핵심 의도:
- 주행 중에는 무거운 CSF를 반복 실행하지 않는다.
- 주행이 끝났다는 기준은 사용자가 /tank/terrain/finalize_map service를 호출하는 시점이다.
- JSON 파싱과 자세 보정(Attitude Correction)은 앞단(LidarProcessorNode)에서 이미
  처리한 PointCloud2 메시지를 수신하여 CPU 점유율을 극도로 최적화함.

입력:
- /tank/sensor/lidar/terrain_points_map       (sensor_msgs/PointCloud2, 지면 전용)
- /tank/sensor/lidar/detected_points_map      (sensor_msgs/PointCloud2, 장애물 전용, 선택 기록)
- legacy mode: /tank/sensor/lidar/all_detected_points_map

출력:
- /tank/terrain/final_accumulated_cloud   (sensor_msgs/PointCloud2)
- /tank/terrain/final_ground_points       (sensor_msgs/PointCloud2)
- /tank/terrain/final_non_ground_points   (sensor_msgs/PointCloud2)
- /tank/terrain/final_elevation_markers   (visualization_msgs/MarkerArray)
- /tank/terrain/final_wireframe_markers   (visualization_msgs/MarkerArray)

서비스:
- /tank/terrain/finalize_map  (std_srvs/Trigger)
- /tank/terrain/reset_map     (std_srvs/Trigger)
"""

import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA

try:
    import CSF  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    CSF = None


def pointcloud2_to_xyz_array(msg: PointCloud2) -> np.ndarray:
    """Return PointCloud2 XYZ fields as a contiguous float32 (N, 3) array.

    ROS2 Humble/newer sensor_msgs_py provides read_points_numpy(), which avoids
    building Python dict/list objects for every LiDAR hit.  The fallback keeps the
    node usable on older sensor_msgs_py versions.
    """
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


class TerrainRecordFinalizeNode(Node):
    """LiDAR 누적 기록 후 주행 종료 시점에 최종 지형 지도를 생성하는 노드."""

    def __init__(self) -> None:
        super().__init__("terrain_record_finalize_node")

        # -----------------------------
        # Parameters
        # -----------------------------
        # 기본은 lidar_processor_node가 이미 분리한 결과를 그대로 사용한다.
        # use_preclassified_lidar=False로 두면 예전처럼 all_detected_points_map을 다시 분리한다.
        self.declare_parameter("use_preclassified_lidar", True)
        self.declare_parameter("terrain_input_topic", "/tank/sensor/lidar/terrain_points_map")
        self.declare_parameter("obstacle_input_topic", "/tank/sensor/lidar/detected_points_map")
        self.declare_parameter("input_topic", "/tank/sensor/lidar/all_detected_points_map")
        self.declare_parameter("map_frame", "tank_map")
        self.declare_parameter("voxel_size", 0.25)
        self.declare_parameter("use_csf", True)
        self.declare_parameter("fallback_ground_percentile", 35.0)
        self.declare_parameter("fallback_ground_margin", 0.35)
        self.declare_parameter("min_points_to_finalize", 30)
        self.declare_parameter("max_points_before_random_crop", 300000)
        self.declare_parameter("publish_period_sec", 1.0)
        self.declare_parameter("save_dir", "~/tankcc/tank_terrain_maps")
        # finalize 시 기본 저장물은 아래 단일 NPZ 파일 하나뿐이다.
        # NPZ 내부에 accumulated/ground/non_ground/metadata_json을 모두 넣어
        # timestamp별 NPY/CSV/metadata 파일이 계속 쌓이지 않게 한다.
        self.declare_parameter("save_filename", "terrain_map_latest.npz")
        self.declare_parameter("saved_map_file", "")
        self.declare_parameter("save_csv", False)
        self.declare_parameter("save_legacy_split_files", False)
        self.declare_parameter("load_saved_map_on_start", False)
        self.declare_parameter("recording_enabled_on_start", True)
        self.declare_parameter("auto_finalize_after_idle_sec", 0.0)  # 0이면 자동 finalize 끔
        self.declare_parameter("grid_cell_size", 0.5)
        self.declare_parameter("max_elevation_cells", 20000)
        self.declare_parameter("marker_alpha", 0.75)
        self.declare_parameter("marker_z_thickness", 0.08)
        self.declare_parameter("wireframe_enabled", True)
        self.declare_parameter("wireframe_line_width", 0.04)
        self.declare_parameter("wireframe_max_height_gap", 1.5)
        self.declare_parameter("wireframe_connect_diagonal", False)
        # 바닥을 점/타일이 아니라 연속 면(TRIANGLE_LIST 메쉬)으로 표시. 색은 기존 height_color 유지.
        self.declare_parameter("surface_mesh_enabled", True)
        self.declare_parameter("surface_fill_gaps", True)       # 관측 안 된 셀을 이웃 보간으로 메움
        self.declare_parameter("surface_max_gap_cells", 2)      # 보간 확장 한계(과한 외삽 방지)

        # 장애물 제거용 local low-surface prefilter.
        self.declare_parameter("terrain_prefilter_enabled", True)
        self.declare_parameter("terrain_cell_size", 0.5)
        self.declare_parameter("terrain_low_percentile", 45.0)
        self.declare_parameter("terrain_height_margin", 0.65)

        self.use_preclassified_lidar = bool(self.get_parameter("use_preclassified_lidar").value)
        self.terrain_input_topic = str(self.get_parameter("terrain_input_topic").value)
        self.obstacle_input_topic = str(self.get_parameter("obstacle_input_topic").value)
        self.input_topic = str(self.get_parameter("input_topic").value)
        self.map_frame = str(self.get_parameter("map_frame").value)
        self.voxel_size = float(self.get_parameter("voxel_size").value)
        self.use_csf = bool(self.get_parameter("use_csf").value)
        self.fallback_ground_percentile = float(self.get_parameter("fallback_ground_percentile").value)
        self.fallback_ground_margin = float(self.get_parameter("fallback_ground_margin").value)
        self.min_points_to_finalize = int(self.get_parameter("min_points_to_finalize").value)
        self.max_points_before_random_crop = int(self.get_parameter("max_points_before_random_crop").value)
        self.publish_period_sec = float(self.get_parameter("publish_period_sec").value)
        self.save_dir = Path(os.path.expanduser(str(self.get_parameter("save_dir").value)))
        self.save_filename = str(self.get_parameter("save_filename").value).strip() or "terrain_map_latest.npz"
        if not self.save_filename.endswith(".npz"):
            self.save_filename += ".npz"
        self.saved_map_file = str(self.get_parameter("saved_map_file").value).strip()
        self.save_csv = bool(self.get_parameter("save_csv").value)
        self.save_legacy_split_files = bool(self.get_parameter("save_legacy_split_files").value)
        self.load_saved_map_on_start = bool(self.get_parameter("load_saved_map_on_start").value)
        self.recording_enabled_on_start = bool(self.get_parameter("recording_enabled_on_start").value)
        self.auto_finalize_after_idle_sec = float(self.get_parameter("auto_finalize_after_idle_sec").value)
        self.grid_cell_size = float(self.get_parameter("grid_cell_size").value)
        self.max_elevation_cells = int(self.get_parameter("max_elevation_cells").value)
        self.marker_alpha = float(self.get_parameter("marker_alpha").value)
        self.marker_z_thickness = float(self.get_parameter("marker_z_thickness").value)
        self.wireframe_enabled = bool(self.get_parameter("wireframe_enabled").value)
        self.wireframe_line_width = float(self.get_parameter("wireframe_line_width").value)
        self.wireframe_max_height_gap = float(self.get_parameter("wireframe_max_height_gap").value)
        self.wireframe_connect_diagonal = bool(self.get_parameter("wireframe_connect_diagonal").value)
        self.surface_mesh_enabled = bool(self.get_parameter("surface_mesh_enabled").value)
        self.surface_fill_gaps = bool(self.get_parameter("surface_fill_gaps").value)
        self.surface_max_gap_cells = int(self.get_parameter("surface_max_gap_cells").value)
        self.terrain_prefilter_enabled = bool(self.get_parameter("terrain_prefilter_enabled").value)
        self.terrain_cell_size = float(self.get_parameter("terrain_cell_size").value)
        self.terrain_low_percentile = float(self.get_parameter("terrain_low_percentile").value)
        self.terrain_height_margin = float(self.get_parameter("terrain_height_margin").value)

        self.save_dir.mkdir(parents=True, exist_ok=True)

        # -----------------------------
        # Internal state
        # -----------------------------
        # _recording_points는 최종 ground 후보, 즉 terrain_points_map 누적분이다.
        # _recording_obstacle_points는 detected_points_map 누적분이며 final_non_ground_points로 저장한다.
        self._recording_points: List[List[float]] = []
        self._recording_obstacle_points: List[List[float]] = []
        self._received_frames = 0
        self._received_points = 0
        self._received_obstacle_frames = 0
        self._received_obstacle_points = 0
        self._last_lidar_wall: Optional[float] = None
        self._finalized = False
        self._final_accumulated: Optional[np.ndarray] = None
        self._final_ground: Optional[np.ndarray] = None
        self._final_non_ground: Optional[np.ndarray] = None
        self._final_markers: Optional[MarkerArray] = None
        self._final_wireframe_markers: Optional[MarkerArray] = None
        self._recording_enable = self.recording_enabled_on_start
        self._last_summary = "아직 finalize되지 않았습니다."

        # -----------------------------
        # ROS interfaces
        # -----------------------------
        # JSON(String) 대신 PointCloud2로 직접 구독.
        # 기본 모드: lidar_processor_node의 분리 결과를 그대로 기록한다.
        if self.use_preclassified_lidar:
            self.create_subscription(PointCloud2, self.terrain_input_topic, self.on_terrain_pc2, 30)
            self.create_subscription(PointCloud2, self.obstacle_input_topic, self.on_obstacle_pc2, 30)
        else:
            self.create_subscription(PointCloud2, self.input_topic, self.on_lidar_pc2, 30)

        self.pub_final_accumulated = self.create_publisher(
            PointCloud2, "/tank/terrain/final_accumulated_cloud", 10
        )
        self.pub_final_ground = self.create_publisher(
            PointCloud2, "/tank/terrain/final_ground_points", 10
        )
        self.pub_final_non_ground = self.create_publisher(
            PointCloud2, "/tank/terrain/final_non_ground_points", 10
        )
        self.pub_final_markers = self.create_publisher(
            MarkerArray, "/tank/terrain/final_elevation_markers", 10
        )
        self.pub_final_wireframe_markers = self.create_publisher(
            MarkerArray, "/tank/terrain/final_wireframe_markers", 10
        )

        self.create_service(Trigger, "/tank/terrain/finalize_map", self.on_finalize_service)
        self.create_service(Trigger, "/tank/terrain/reset_map", self.on_reset_service)

        self.create_timer(self.publish_period_sec, self.on_publish_timer)
        self.create_timer(1.0, self.on_watchdog_timer)

        if self.load_saved_map_on_start:
            ok, msg = self.load_saved_outputs(self.current_map_file())
            if ok:
                self.get_logger().info(msg)
                self.publish_final_outputs()
            else:
                self.get_logger().warn(msg)

        mode_text = "recording enabled" if self._recording_enable else "recording disabled"
        self.get_logger().info(
            "terrain_record_finalize_node started. "
            + (
                f"Recording preclassified terrain={self.terrain_input_topic}, obstacle={self.obstacle_input_topic}. "
                if self.use_preclassified_lidar
                else f"Recording legacy all-hit input={self.input_topic}. "
            )
            + f"Mode={mode_text}, save_file={self.current_map_file()}. "
            + "Call /tank/terrain/finalize_map when driving is finished."
        )

    # ------------------------------------------------------------------
    # Input parsing (Optimized for PointCloud2)
    # ------------------------------------------------------------------
    def _record_pc2_points(self, msg: PointCloud2, target: str) -> None:
        """바이너리 PointCloud2를 NumPy로 변환해 terrain 또는 obstacle 누적 버퍼에 기록한다."""
        if not self._recording_enable:
            return

        points = pointcloud2_to_xyz_array(msg)
        if points.size == 0:
            return

        if target == "obstacle":
            self._recording_obstacle_points.extend(points.tolist())
            self._received_obstacle_frames += 1
            self._received_obstacle_points += int(points.shape[0])
        else:
            self._recording_points.extend(points.tolist())
            self._received_frames += 1
            self._received_points += int(points.shape[0])

        self._last_lidar_wall = time.time()
        self._finalized = False

        total_frames = self._received_frames + self._received_obstacle_frames
        if total_frames > 0 and total_frames % 50 == 0:
            self.get_logger().info(
                f"Recording LiDAR terrain_frames={self._received_frames}, "
                f"terrain_points={self._received_points}, "
                f"obstacle_frames={self._received_obstacle_frames}, "
                f"obstacle_points={self._received_obstacle_points}, "
                f"stored_terrain={len(self._recording_points)}, "
                f"stored_obstacle={len(self._recording_obstacle_points)}"
            )

    def on_terrain_pc2(self, msg: PointCloud2) -> None:
        """lidar_processor_node가 지면으로 분류한 /terrain_points_map을 그대로 기록한다."""
        self._record_pc2_points(msg, "terrain")

    def on_obstacle_pc2(self, msg: PointCloud2) -> None:
        """lidar_processor_node가 장애물로 분류한 /detected_points_map을 final non-ground로 기록한다."""
        self._record_pc2_points(msg, "obstacle")

    def on_lidar_pc2(self, msg: PointCloud2) -> None:
        """Legacy mode: all_detected_points_map을 기록한 뒤 finalize 시 다시 분리한다."""
        self._record_pc2_points(msg, "terrain")

    # ------------------------------------------------------------------
    # Services
    # ------------------------------------------------------------------
    def on_finalize_service(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        ok, summary = self.finalize_map()
        response.success = ok
        response.message = summary
        return response

    def on_reset_service(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        self._recording_points.clear()
        self._recording_obstacle_points.clear()
        self._received_frames = 0
        self._received_points = 0
        self._received_obstacle_frames = 0
        self._received_obstacle_points = 0
        self._last_lidar_wall = None
        self._finalized = False
        self._recording_enable = True
        self._final_accumulated = None
        self._final_ground = None
        self._final_non_ground = None
        self._final_markers = self.make_delete_all_markers("final_elevation_grid")
        self._final_wireframe_markers = self.make_delete_all_markers("final_terrain_wireframe")
        self._last_summary = "reset 완료"

        self.publish_final_outputs()
        response.success = True
        response.message = "기록된 LiDAR와 최종 지도 결과를 초기화했습니다."
        self.get_logger().info(response.message)
        return response

    def on_watchdog_timer(self) -> None:
        """옵션: 일정 시간 LiDAR가 안 들어오면 자동 finalize."""
        if self.auto_finalize_after_idle_sec <= 0.0:
            return
        if self._finalized or self._last_lidar_wall is None:
            return
        idle = time.time() - self._last_lidar_wall
        if idle >= self.auto_finalize_after_idle_sec and len(self._recording_points) >= self.min_points_to_finalize:
            self.get_logger().info(
                f"No LiDAR for {idle:.1f}s. Auto-finalizing terrain map."
            )
            self.finalize_map()

    # ------------------------------------------------------------------
    # Final map generation
    # ------------------------------------------------------------------
    def finalize_map(self) -> Tuple[bool, str]:
        if len(self._recording_points) < self.min_points_to_finalize:
            summary = (
                f"point가 너무 적어서 finalize할 수 없습니다. "
                f"stored={len(self._recording_points)}, required={self.min_points_to_finalize}"
            )
            self.get_logger().warn(summary)
            return False, summary

        raw = np.asarray(self._recording_points, dtype=np.float32)
        raw = raw[np.isfinite(raw).all(axis=1)]
        if raw.size == 0:
            return False, "유효한 terrain point가 없습니다."

        obstacle_raw = np.asarray(self._recording_obstacle_points, dtype=np.float32)
        if obstacle_raw.size == 0:
            obstacle_raw = np.empty((0, 3), dtype=np.float32)
        else:
            obstacle_raw = obstacle_raw[np.isfinite(obstacle_raw).all(axis=1)]

        if raw.shape[0] > self.max_points_before_random_crop:
            idx = np.random.choice(raw.shape[0], self.max_points_before_random_crop, replace=False)
            raw = raw[idx]
        if obstacle_raw.shape[0] > self.max_points_before_random_crop:
            idx = np.random.choice(obstacle_raw.shape[0], self.max_points_before_random_crop, replace=False)
            obstacle_raw = obstacle_raw[idx]

        if self.use_preclassified_lidar:
            # 핵심 변경점:
            # RViz 실시간 표시와 같은 lidar_processor_node의 분리 결과를 최종 지형에도 그대로 사용한다.
            # terrain_points_map은 ground, detected_points_map은 non_ground로 저장한다.
            ground = self.voxel_downsample(raw, self.voxel_size)
            non_ground = self.voxel_downsample(obstacle_raw, self.voxel_size)
            if non_ground.shape[0] > 0:
                accumulated_src = np.vstack([ground, non_ground]).astype(np.float32)
            else:
                accumulated_src = ground.astype(np.float32)
            accumulated = self.voxel_downsample(accumulated_src, self.voxel_size)
            method = "preclassified_lidar_processor"
        else:
            accumulated = self.voxel_downsample(raw, self.voxel_size)
            terrain_candidates, obstacle_candidates = self.prefilter_low_surface(accumulated)
            ground, non_ground_from_ground_filter, method = self.split_ground(terrain_candidates)

            if obstacle_candidates.shape[0] > 0 and non_ground_from_ground_filter.shape[0] > 0:
                non_ground = np.vstack([obstacle_candidates, non_ground_from_ground_filter]).astype(np.float32)
            elif obstacle_candidates.shape[0] > 0:
                non_ground = obstacle_candidates.astype(np.float32)
            else:
                non_ground = non_ground_from_ground_filter.astype(np.float32)
            method = f"local_low_surface+{method}"

        markers = (self.make_surface_mesh_markers(ground)
                   if self.surface_mesh_enabled else self.make_elevation_markers(ground))
        wireframe_markers = self.make_wireframe_markers(ground)

        self._final_accumulated = accumulated
        self._final_ground = ground
        self._final_non_ground = non_ground
        self._final_markers = markers
        self._final_wireframe_markers = wireframe_markers
        self._finalized = True
        self._recording_enable = False

        out_file = self.current_map_file()
        self.save_outputs(out_file, accumulated, ground, non_ground, method)

        summary = (
            f"finalize 완료: frames={self._received_frames}, "
            f"terrain_points={self._received_points}, "
            f"obstacle_points={self._received_obstacle_points}, "
            f"stored_terrain={len(self._recording_points)}, "
            f"stored_obstacle={len(self._recording_obstacle_points)}, "
            f"voxel_points={accumulated.shape[0]}, "
            f"ground={ground.shape[0]}, non_ground={non_ground.shape[0]}, "
            f"method={method}, save_file={out_file}"
        )
        self._last_summary = summary
        self.get_logger().info(summary)
        self.publish_final_outputs()
        return True, summary

    def voxel_downsample(self, points: np.ndarray, voxel_size: float) -> np.ndarray:
        """같은 voxel 안의 점들을 평균 대표점 하나로 줄인다."""
        if points.size == 0 or voxel_size <= 0:
            return points.astype(np.float32)

        keys = np.floor(points / voxel_size).astype(np.int64)
        voxel_dict: Dict[Tuple[int, int, int], Tuple[np.ndarray, int]] = {}
        for key, point in zip(map(tuple, keys), points):
            if key in voxel_dict:
                s, c = voxel_dict[key]
                voxel_dict[key] = (s + point, c + 1)
            else:
                voxel_dict[key] = (point.astype(np.float64), 1)

        down = np.asarray([s / c for s, c in voxel_dict.values()], dtype=np.float32)
        return down

    def prefilter_low_surface(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """같은 x-y cell 안에서 낮은 연속 표면만 지형 후보로 남기고 높은 점은 장애물 후보로 분리."""
        if (not self.terrain_prefilter_enabled) or points.shape[0] == 0 or self.terrain_cell_size <= 0:
            return points.astype(np.float32), np.empty((0, 3), dtype=np.float32)

        keys = np.floor(points[:, :2] / self.terrain_cell_size).astype(np.int64)
        cell_z: Dict[Tuple[int, int], List[float]] = {}
        for key, z in zip(map(tuple, keys), points[:, 2]):
            cell_z.setdefault(key, []).append(float(z))

        thresholds: Dict[Tuple[int, int], float] = {}
        pct = float(np.clip(self.terrain_low_percentile, 0.0, 100.0))
        for key, values in cell_z.items():
            base = float(np.percentile(np.asarray(values, dtype=np.float32), pct))
            thresholds[key] = base + self.terrain_height_margin

        terrain_mask = np.zeros(points.shape[0], dtype=bool)
        for i, key in enumerate(map(tuple, keys)):
            terrain_mask[i] = float(points[i, 2]) <= thresholds[key]

        return points[terrain_mask].astype(np.float32), points[~terrain_mask].astype(np.float32)

    def split_ground(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, str]:
        """CSF가 가능하면 CSF, 실패하면 z-percentile fallback으로 ground/non-ground 분리."""
        if points.shape[0] == 0:
            return points, points, "empty"

        if self.use_csf and CSF is not None and points.shape[0] >= 50:
            try:
                csf = CSF.CSF()
                csf.params.cloth_resolution = 0.5
                csf.params.rigidness = 3
                csf.params.class_threshold = 0.25
                # Python wrapper에 따라 interations 오타 이름을 사용한다.
                if hasattr(csf.params, "interations"):
                    csf.params.interations = 500
                elif hasattr(csf.params, "iterations"):
                    csf.params.iterations = 500

                csf.setPointCloud(points.astype(np.float64))
                ground_idx = CSF.VecInt()
                non_ground_idx = CSF.VecInt()
                csf.do_filtering(ground_idx, non_ground_idx)

                gi = np.asarray(list(ground_idx), dtype=np.int64)
                ni = np.asarray(list(non_ground_idx), dtype=np.int64)
                if gi.size > 0:
                    return points[gi], points[ni], "CSF"
            except Exception as exc:
                self.get_logger().warn(f"CSF failed, fallback z-filter will be used: {exc}")

        z = points[:, 2]
        base = np.percentile(z, self.fallback_ground_percentile)
        threshold = base + self.fallback_ground_margin
        mask = z <= threshold
        return points[mask], points[~mask], "fallback_z_filter"

    # ------------------------------------------------------------------
    # Publishing / visualization
    # ------------------------------------------------------------------
    def on_publish_timer(self) -> None:
        if self._finalized:
            self.publish_final_outputs()

    def publish_final_outputs(self) -> None:
        now = self.get_clock().now().to_msg()
        if self._final_accumulated is not None:
            self.pub_final_accumulated.publish(self.to_cloud_msg(self._final_accumulated, now))
        if self._final_ground is not None:
            self.pub_final_ground.publish(self.to_cloud_msg(self._final_ground, now))
        if self._final_non_ground is not None:
            self.pub_final_non_ground.publish(self.to_cloud_msg(self._final_non_ground, now))
        if self._final_markers is not None:
            # marker stamp 갱신
            for m in self._final_markers.markers:
                m.header.stamp = now
            self.pub_final_markers.publish(self._final_markers)
        if self._final_wireframe_markers is not None:
            for m in self._final_wireframe_markers.markers:
                m.header.stamp = now
            self.pub_final_wireframe_markers.publish(self._final_wireframe_markers)

    def to_cloud_msg(self, points: np.ndarray, stamp: Any) -> PointCloud2:
        from std_msgs.msg import Header
        h = Header()
        h.stamp = stamp
        h.frame_id = self.map_frame
        return point_cloud2.create_cloud_xyz32(h, points.astype(np.float32).tolist())

    def make_elevation_markers(self, ground: np.ndarray) -> MarkerArray:
        """ground point를 x-y grid cell별 대표 높이 타일 MarkerArray로 변환."""
        arr = MarkerArray()
        arr.markers.append(self.make_delete_all_marker())
        if ground.shape[0] == 0 or self.grid_cell_size <= 0:
            return arr

        keys = np.floor(ground[:, :2] / self.grid_cell_size).astype(np.int64)
        cell_values: Dict[Tuple[int, int], List[float]] = {}
        for key, z in zip(map(tuple, keys), ground[:, 2]):
            cell_values.setdefault(key, []).append(float(z))

        items = list(cell_values.items())
        if len(items) > self.max_elevation_cells:
            step = math.ceil(len(items) / self.max_elevation_cells)
            items = items[::step]

        zs = np.asarray([np.median(v) for _, v in items], dtype=np.float32)
        z_min = float(np.min(zs)) if zs.size else 0.0
        z_max = float(np.max(zs)) if zs.size else 1.0
        z_range = max(z_max - z_min, 1e-6)

        stamp = self.get_clock().now().to_msg()
        for marker_id, ((ix, iy), zlist) in enumerate(items, start=1):
            z = float(np.median(zlist))
            t = (z - z_min) / z_range
            marker = Marker()
            marker.header.frame_id = self.map_frame
            marker.header.stamp = stamp
            marker.ns = "final_elevation_grid"
            marker.id = marker_id
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.pose.position.x = (ix + 0.5) * self.grid_cell_size
            marker.pose.position.y = (iy + 0.5) * self.grid_cell_size
            marker.pose.position.z = z
            marker.pose.orientation.w = 1.0
            marker.scale.x = self.grid_cell_size
            marker.scale.y = self.grid_cell_size
            marker.scale.z = self.marker_z_thickness
            # 낮은 곳: 초록 성분 높게, 높은 곳: 빨강 성분 높게.
            marker.color.r = float(t)
            marker.color.g = float(1.0 - 0.6 * t)
            marker.color.b = 0.1
            marker.color.a = self.marker_alpha
            arr.markers.append(marker)
        return arr

    def grid_height_map(self, ground: np.ndarray) -> Dict[Tuple[int, int], float]:
        """ground point를 x-y grid cell별 대표 높이 dict로 변환한다."""
        if ground.shape[0] == 0 or self.grid_cell_size <= 0:
            return {}

        keys = np.floor(ground[:, :2] / self.grid_cell_size).astype(np.int64)
        cell_values: Dict[Tuple[int, int], List[float]] = {}
        for key, z in zip(map(tuple, keys), ground[:, 2]):
            cell_values.setdefault(key, []).append(float(z))

        items = list(cell_values.items())
        if len(items) > self.max_elevation_cells:
            step = math.ceil(len(items) / self.max_elevation_cells)
            items = items[::step]

        return {key: float(np.median(zlist)) for key, zlist in items}

    def height_color(self, z: float, z_min: float, z_max: float, alpha: float) -> ColorRGBA:
        """높이 z를 초록(낮음) -> 노랑 -> 빨강(높음)으로 변환."""
        z_range = max(z_max - z_min, 1e-6)
        t = max(0.0, min(1.0, (z - z_min) / z_range))
        c = ColorRGBA()
        c.r = float(t)
        c.g = float(1.0 - 0.6 * t)
        c.b = 0.1
        c.a = float(alpha)
        return c

    def _fill_grid_gaps(self, height: Dict[Tuple[int, int], float], max_gap: int) -> Dict[Tuple[int, int], float]:
        """관측 안 된 셀을 이웃(상하좌우) 평균으로 메운다. max_gap회 반복(과한 외삽 방지).
        고립점 외삽을 막기 위해 이웃이 2개 이상일 때만 채운다."""
        filled = dict(height)
        for _ in range(max(0, int(max_gap))):
            added: Dict[Tuple[int, int], float] = {}
            for (ix, iy) in list(filled.keys()):
                for nb in ((ix + 1, iy), (ix - 1, iy), (ix, iy + 1), (ix, iy - 1)):
                    if nb in filled or nb in added:
                        continue
                    vals = [filled[m] for m in
                            ((nb[0] + 1, nb[1]), (nb[0] - 1, nb[1]), (nb[0], nb[1] + 1), (nb[0], nb[1] - 1))
                            if m in filled]
                    if len(vals) >= 2:
                        added[nb] = float(sum(vals) / len(vals))
            if not added:
                break
            filled.update(added)
        return filled

    def make_surface_mesh_markers(self, ground: np.ndarray) -> MarkerArray:
        """ground grid를 연속 표면(TRIANGLE_LIST 메쉬)으로 렌더. 색은 기존 height_color(초록→빨강) 재사용.
        인접 4셀(quad)이 모두 있으면 두 삼각형으로 채운다. 구멍은 surface_fill_gaps로 이웃 보간.
        점/타일(make_elevation_markers) 대비 '진짜 바닥처럼' 보이고 마커 1개라 렉도 가볍다."""
        arr = MarkerArray()
        arr.markers.append(self.make_delete_all_marker("final_elevation_grid"))
        height = self.grid_height_map(ground)
        if not height:
            return arr
        if self.surface_fill_gaps:
            height = self._fill_grid_gaps(height, self.surface_max_gap_cells)
        zs = list(height.values())
        z_min, z_max = min(zs), max(zs)
        cs = self.grid_cell_size

        marker = Marker()
        marker.header.frame_id = self.map_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "final_elevation_grid"
        marker.id = 1
        marker.type = Marker.TRIANGLE_LIST
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = marker.scale.y = marker.scale.z = 1.0
        # 전체 material 색/알파. 정점색(marker.colors)이 실제 면 색을 정하지만,
        # color.a가 0(기본값)이면 RViz2가 TRIANGLE_LIST를 투명 처리해 안 보인다 → 반드시 불투명 지정.
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.color.a = 1.0

        def vtx(ix: int, iy: int):
            z = height[(ix, iy)]
            return Point(x=float((ix + 0.5) * cs), y=float((iy + 0.5) * cs), z=float(z)), z

        for (ix, iy) in height.keys():
            quad = [(ix, iy), (ix + 1, iy), (ix + 1, iy + 1), (ix, iy + 1)]
            if not all(k in height for k in quad):
                continue
            pts = [vtx(*k) for k in quad]
            for tri in ((0, 1, 2), (0, 2, 3)):
                for j in tri:
                    p, z = pts[j]
                    marker.points.append(p)
                    marker.colors.append(self.height_color(z, z_min, z_max, self.marker_alpha))
        if not marker.points:
            return arr
        arr.markers.append(marker)
        return arr

    def make_wireframe_markers(self, ground: np.ndarray) -> MarkerArray:
        """ground grid cell 중심을 인접 cell끼리 선으로 연결해 지형 wireframe을 만든다."""
        arr = MarkerArray()
        arr.markers.append(self.make_delete_all_marker("final_terrain_wireframe"))
        if not self.wireframe_enabled or ground.shape[0] == 0 or self.grid_cell_size <= 0:
            return arr

        grid = self.grid_height_map(ground)
        if not grid:
            return arr

        z_values = np.asarray(list(grid.values()), dtype=np.float32)
        z_min = float(np.min(z_values))
        z_max = float(np.max(z_values))

        marker = Marker()
        marker.header.frame_id = self.map_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "final_terrain_wireframe"
        marker.id = 1
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = max(self.wireframe_line_width, 0.001)
        marker.color.a = 1.0

        neighbor_offsets = [(1, 0), (0, 1)]
        if self.wireframe_connect_diagonal:
            neighbor_offsets.extend([(1, 1), (1, -1)])

        max_gap = self.wireframe_max_height_gap
        segment_count = 0
        for (ix, iy), z1 in grid.items():
            x1 = (ix + 0.5) * self.grid_cell_size
            y1 = (iy + 0.5) * self.grid_cell_size
            for dx, dy in neighbor_offsets:
                key2 = (ix + dx, iy + dy)
                if key2 not in grid:
                    continue
                z2 = grid[key2]
                if max_gap > 0.0 and abs(z2 - z1) > max_gap:
                    continue

                x2 = (key2[0] + 0.5) * self.grid_cell_size
                y2 = (key2[1] + 0.5) * self.grid_cell_size

                p1 = Point(x=float(x1), y=float(y1), z=float(z1))
                p2 = Point(x=float(x2), y=float(y2), z=float(z2))
                marker.points.append(p1)
                marker.points.append(p2)

                c = self.height_color((z1 + z2) * 0.5, z_min, z_max, self.marker_alpha)
                marker.colors.append(c)
                marker.colors.append(c)
                segment_count += 1

        if segment_count > 0:
            arr.markers.append(marker)
        return arr

    def make_delete_all_markers(self, namespace: str = "final_elevation_grid") -> MarkerArray:
        arr = MarkerArray()
        arr.markers.append(self.make_delete_all_marker(namespace))
        return arr

    def make_delete_all_marker(self, namespace: str = "final_elevation_grid") -> Marker:
        marker = Marker()
        marker.header.frame_id = self.map_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = namespace
        marker.id = 0
        marker.action = Marker.DELETEALL
        return marker

    # ------------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------------
    def current_map_file(self) -> Path:
        """저장/로드에 사용할 단일 terrain map 파일 경로를 반환한다."""
        if self.saved_map_file:
            path = Path(os.path.expanduser(self.saved_map_file))
        else:
            path = self.save_dir / self.save_filename
        if path.suffix != ".npz":
            path = path.with_suffix(".npz")
        return path

    def save_outputs(
        self,
        out_file: Path,
        accumulated: np.ndarray,
        ground: np.ndarray,
        non_ground: np.ndarray,
        method: str,
    ) -> None:
        """최종 지형 결과를 단일 NPZ 파일 하나로 저장한다.

        기본 설정에서는 finalize를 여러 번 호출해도
        ~/tankcc/tank_terrain_maps/terrain_map_latest.npz 하나만 계속 덮어쓴다.
        """
        out_file.parent.mkdir(parents=True, exist_ok=True)

        meta = {
            "created_wall_time": time.time(),
            "created_local_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "map_frame": self.map_frame,
            "use_preclassified_lidar": self.use_preclassified_lidar,
            "input_topic": self.input_topic,
            "terrain_input_topic": self.terrain_input_topic,
            "obstacle_input_topic": self.obstacle_input_topic,
            "received_frames": self._received_frames,
            "received_points": self._received_points,
            "received_obstacle_frames": self._received_obstacle_frames,
            "received_obstacle_points": self._received_obstacle_points,
            "stored_points": len(self._recording_points),
            "stored_obstacle_points": len(self._recording_obstacle_points),
            "voxel_size": self.voxel_size,
            "accumulated_points": int(accumulated.shape[0]),
            "ground_points": int(ground.shape[0]),
            "non_ground_points": int(non_ground.shape[0]),
            "ground_filter_method": method,
            "grid_cell_size": self.grid_cell_size,
            "file_format": "tank_terrain_map_npz_v1",
        }
        metadata_json = json.dumps(meta, ensure_ascii=False)

        # 단일 핵심 파일. arrays + metadata_json을 모두 내부에 저장한다.
        np.savez_compressed(
            str(out_file),
            accumulated=accumulated.astype(np.float32),
            ground=ground.astype(np.float32),
            non_ground=non_ground.astype(np.float32),
            metadata_json=np.asarray(metadata_json),
        )

        # 디버그/외부 분석이 필요할 때만 켜는 선택 기능. 기본 False라 파일이 늘어나지 않는다.
        if self.save_legacy_split_files:
            out_prefix = out_file.with_suffix("")
            np.save(str(out_prefix) + "_accumulated.npy", accumulated)
            np.save(str(out_prefix) + "_ground.npy", ground)
            np.save(str(out_prefix) + "_non_ground.npy", non_ground)
            with open(str(out_prefix) + "_metadata.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

        if self.save_csv:
            out_prefix = out_file.with_suffix("")
            np.savetxt(str(out_prefix) + "_accumulated.csv", accumulated, delimiter=",", header="x,y,z", comments="")
            np.savetxt(str(out_prefix) + "_ground.csv", ground, delimiter=",", header="x,y,z", comments="")
            np.savetxt(str(out_prefix) + "_non_ground.csv", non_ground, delimiter=",", header="x,y,z", comments="")

    def load_saved_outputs(self, in_file: Path) -> Tuple[bool, str]:
        """단일 NPZ terrain map을 읽어 RViz publish용 내부 상태로 복원한다."""
        if not in_file.exists():
            return False, f"저장된 terrain map 파일이 없습니다: {in_file}"

        try:
            with np.load(str(in_file), allow_pickle=False) as data:
                if "ground" not in data.files:
                    return False, f"terrain map에 ground 배열이 없습니다: {in_file}"
                ground = np.asarray(data["ground"], dtype=np.float32).reshape(-1, 3)
                non_ground = (
                    np.asarray(data["non_ground"], dtype=np.float32).reshape(-1, 3)
                    if "non_ground" in data.files
                    else np.empty((0, 3), dtype=np.float32)
                )
                if "accumulated" in data.files:
                    accumulated = np.asarray(data["accumulated"], dtype=np.float32).reshape(-1, 3)
                elif non_ground.shape[0] > 0:
                    accumulated = np.vstack([ground, non_ground]).astype(np.float32)
                else:
                    accumulated = ground.astype(np.float32)

                meta: Dict[str, Any] = {}
                if "metadata_json" in data.files:
                    meta_value = data["metadata_json"]
                    meta_text = str(meta_value.item() if meta_value.shape == () else meta_value[0])
                    try:
                        meta = json.loads(meta_text)
                    except Exception:
                        meta = {"metadata_json_parse_error": meta_text[:200]}
        except Exception as exc:
            return False, f"terrain map 로드 실패: {in_file} ({exc})"

        saved_frame = str(meta.get("map_frame", self.map_frame)) if meta else self.map_frame
        if saved_frame:
            self.map_frame = saved_frame

        self._final_accumulated = accumulated
        self._final_ground = ground
        self._final_non_ground = non_ground
        self._final_markers = (self.make_surface_mesh_markers(ground)
                               if self.surface_mesh_enabled else self.make_elevation_markers(ground))
        self._final_wireframe_markers = self.make_wireframe_markers(ground)
        self._finalized = True
        self._last_summary = (
            f"저장 terrain map 로드 완료: file={in_file}, "
            f"accumulated={accumulated.shape[0]}, ground={ground.shape[0]}, "
            f"non_ground={non_ground.shape[0]}, frame={self.map_frame}"
        )
        return True, self._last_summary


def main(args: Optional[List[str]] = None) -> None:
    rclpy.init(args=args)
    node = TerrainRecordFinalizeNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
