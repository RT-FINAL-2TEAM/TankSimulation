import json
import math
import os
from typing import Dict, List, Optional, Tuple


class ReconLogger:
    """정찰 루트별 장애물/발각/비전/GT 포착 로그를 수집하고 리포트를 산출합니다."""

    def __init__(self, route_id: str, map_name: str, output_dir: str):
        self.route_id: str = route_id
        self.map_name: str = map_name
        self.output_dir: str = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # 장애물 로그
        self.obstacles_detected: list[dict] = []
        self._seen_obstacle_keys: set = set()

        # 발각(노출) 로그
        self.exposure_events: list[dict] = []
        self._active_exposures: dict[str, dict] = {}

        # 비전(YOLO) 로그
        self.vision_detections: list[dict] = []

        # GT 포착 카운트
        self.spotted_assets: dict[str, set] = {
            "outposts": set(),
            "tanks": set(),
            "soldiers": set(),
        }

        # 주행 결과
        self.reached: bool = False
        self.collisions: int = 0
        self.total_sim_time: float = 0.0
        self.total_distance: float = 0.0

        # 전차 궤적(노출/발각 사후계산용). [t, x, z, yaw] map 좌표(x=map.x, z=map.y).
        # 0.5m 이상 이동 시에만 적재해 파일 크기를 억제한다.
        self.trajectory: list[list] = []
        self._last_traj_xz: Optional[Tuple[float, float]] = None
        self._traj_min_step_m: float = 0.5

        # terrain roughness 수집
        self._pitch_samples: list[float] = []
        self._roll_samples: list[float] = []

    # -- 장애물 로깅 --------------------------------------------------------

    def log_obstacle(self, sim_time: float, x: float, z: float, bbox: list) -> None:
        """검출된 장애물의 위치를 중복 제거 후 기록합니다."""
        key = (round(x, 1), round(z, 1))
        if key in self._seen_obstacle_keys:
            return
        self._seen_obstacle_keys.add(key)
        self.obstacles_detected.append({
            "t": round(sim_time, 2),
            "x": round(x, 2),
            "z": round(z, 2),
            "bbox": bbox,
        })

    # -- 전차 궤적 로깅 -----------------------------------------------------

    def log_pose(self, sim_time: float, x: float, z: float, yaw: float = 0.0) -> None:
        """전차 map 좌표(x=map.x, z=map.y)를 0.5m 간격으로 적재한다."""
        if self._last_traj_xz is not None:
            lx, lz = self._last_traj_xz
            if math.hypot(x - lx, z - lz) < self._traj_min_step_m:
                return
        self._last_traj_xz = (x, z)
        self.trajectory.append([round(sim_time, 2), round(x, 2), round(z, 2), round(yaw, 1)])

    # -- 발각(노출) 로깅 ----------------------------------------------------

    def update_exposure(self, sim_time: float, asset_id: str, detected: bool, dist: float) -> None:
        """매 프레임 호출하여 발각 진입/이탈 이벤트를 기록합니다."""
        if detected and asset_id not in self._active_exposures:
            self._active_exposures[asset_id] = {
                "asset": asset_id,
                "enter_t": round(sim_time, 2),
                "min_dist_m": round(dist, 2),
            }
        elif detected and asset_id in self._active_exposures:
            entry = self._active_exposures[asset_id]
            entry["min_dist_m"] = round(min(entry["min_dist_m"], dist), 2)
        elif not detected and asset_id in self._active_exposures:
            entry = self._active_exposures.pop(asset_id)
            entry["exit_t"] = round(sim_time, 2)
            entry["dwell_s"] = round(entry["exit_t"] - entry["enter_t"], 2)
            self.exposure_events.append(entry)

    def flush_active_exposures(self, sim_time: float) -> None:
        """에피소드 종료 시 아직 열린 노출 이벤트를 닫습니다."""
        for asset_id in list(self._active_exposures.keys()):
            entry = self._active_exposures.pop(asset_id)
            entry["exit_t"] = round(sim_time, 2)
            entry["dwell_s"] = round(entry["exit_t"] - entry["enter_t"], 2)
            self.exposure_events.append(entry)

    # -- 비전(YOLO) 로깅 ---------------------------------------------------

    def log_vision(self, sim_time: float, class_name: str, confidence: float, bbox: list, turret_x: float) -> None:
        """YOLO 감지 결과를 기록합니다."""
        self.vision_detections.append({
            "t": round(sim_time, 2),
            "class": class_name,
            "conf": round(confidence, 3),
            "bbox": bbox,
            "turret_x": round(turret_x, 2),
        })

    # -- GT 포착 로깅 -------------------------------------------------------

    def log_spotted_asset(self, asset_type: str, asset_id: str) -> None:
        """심판 레이어에서 포착된 적자산을 기록합니다."""
        if asset_type in self.spotted_assets:
            self.spotted_assets[asset_type].add(asset_id)

    # -- terrain roughness 수집 --------------------------------------------

    def log_body_angles(self, pitch_deg: float, roll_deg: float) -> None:
        """playerBodyY(pitch), playerBodyZ(roll) 값을 수집합니다."""
        def norm(a):
            while a > 180.0: a -= 360.0
            while a < -180.0: a += 360.0
            return a
        self._pitch_samples.append(norm(pitch_deg))
        self._roll_samples.append(norm(roll_deg))

    # -- 리포트 산출 --------------------------------------------------------

    def _calc_std(self, samples: list[float]) -> float:
        if len(samples) < 2:
            return 0.0
        mean = sum(samples) / len(samples)
        variance = sum((s - mean) ** 2 for s in samples) / len(samples)
        return round(math.sqrt(variance), 4)

    def _build_exposure_summary(self) -> dict:
        count = len(self.exposure_events)
        total_dwell = round(sum(e.get("dwell_s", 0.0) for e in self.exposure_events), 2)
        max_continuous = round(max((e.get("dwell_s", 0.0) for e in self.exposure_events), default=0.0), 2)
        return {
            "detection_count": count,
            "total_dwell_s": total_dwell,
            "max_continuous_s": max_continuous,
            "events": self.exposure_events,
        }

    def _build_vision_summary(self) -> dict:
        counts: dict[str, int] = {}
        for d in self.vision_detections:
            cls = d["class"]
            counts[cls] = counts.get(cls, 0) + 1
        return {
            "counts": counts,
            "detections": self.vision_detections,
        }

    def _build_asset_spotted_gt(self) -> dict:
        return {
            "outposts": len(self.spotted_assets["outposts"]),
            "tanks": len(self.spotted_assets["tanks"]),
            "soldiers": len(self.spotted_assets["soldiers"]),
        }

    def build_report(self) -> dict:
        """6장 (5)의 JSON 스키마에 따라 루트 리포트를 구성합니다."""
        density = 0.0
        if self.total_distance > 0:
            density = round(len(self.obstacles_detected) / (self.total_distance / 100.0), 2)

        return {
            "route": self.route_id,
            "map": self.map_name,
            "result": {
                "reached": self.reached,
                "collisions": self.collisions,
                "sim_time_s": round(self.total_sim_time, 2),
                "distance_m": round(self.total_distance, 2),
            },
            "obstacle_summary": {
                "count": len(self.obstacles_detected),
                "density_per_100m": density,
            },
            "trajectory": self.trajectory,
            "obstacles_detected": self.obstacles_detected,
            "exposure": self._build_exposure_summary(),
            "vision_yolo": self._build_vision_summary(),
            "asset_spotted_gt": self._build_asset_spotted_gt(),
            "terrain_roughness": {
                "pitch_std_deg": self._calc_std(self._pitch_samples),
                "roll_std_deg": self._calc_std(self._roll_samples),
            },
        }

    def save_report(self) -> str:
        """리포트를 JSON 파일로 저장하고 경로를 반환합니다."""
        report = self.build_report()
        path = os.path.join(self.output_dir, f"route_{self.route_id}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        return path


def save_comparison(report_a: dict, report_b: dict, output_dir: str) -> str:
    """A/B 루트 리포트를 나란히 담은 비교 파일을 저장합니다."""
    comparison = {"route_A": report_a, "route_B": report_b}
    path = os.path.join(output_dir, "comparison.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)
    return path
