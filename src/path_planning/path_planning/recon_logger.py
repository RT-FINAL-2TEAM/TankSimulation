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
        self.collision_events: list[dict] = []   # [{t,x,z}] 충돌 발생 위치(analyze_run 궤적 오버레이용)
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

        # 주행 품질 진단(원인 귀속: 경로/APF/제어)용 — 공식 리포트 본문엔 안 쓰고 route_*.json 원자료만 확장.
        # 끼임/제자리 진동도 잡으려 '이동량'이 아니라 '시간(0.2s)'으로 적재한다.
        self.route_version: int = 0
        self.route_version_changes: int = 0
        self.planned_paths: list[dict] = []           # [{t, version, path:[[x,z],...]}] (경로 바뀔 때만)
        self.diag_samples: list[dict] = []            # per-step {t, p, rv, look, ltgt, cmd}
        self._latest_lookahead: Optional[Tuple[float, float]] = None
        self._latest_local_target: Optional[Tuple[float, float]] = None
        self._latest_cmd: str = ""
        self._fusion_rejects: dict = {}               # 융합 드롭 사유 누적 {reason: count} — 왜 확정 안 되나(analyze_run)
        self._last_path_sig = None
        self._last_diag_t: Optional[float] = None
        self._diag_min_dt: float = 0.2

        # 정찰 관측 거동(②감속/dwell·③포탑 stare) 진단 — diag_sample에 흘려보내 analyze_run이 소비.
        # dwell/mode는 컨트롤러 status(speed_mode/recon_observation)에서, 후보요약은 local_path_node에서.
        self._obs_dwell: bool = False
        self._obs_mode: str = ""
        self._obs_candidates: Optional[dict] = None

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

    # -- 주행 품질 진단 로깅 (원인 귀속: 경로 churn / APF 불일치 / 제어 채터) ---

    def set_route_version(self, version: int) -> None:
        """planner status의 route_version. 증가할 때마다 churn 카운트."""
        if version > self.route_version:
            self.route_version = version
            self.route_version_changes += 1

    def set_lookahead(self, x: float, z: float) -> None:
        self._latest_lookahead = (x, z)

    def set_local_target(self, x: float, z: float) -> None:
        self._latest_local_target = (x, z)

    def set_command(self, cmd: str) -> None:
        self._latest_cmd = cmd

    def set_fusion_rejects(self, counts: dict) -> None:
        """융합 드롭 사유 누적 {reason: count}. YOLO를 봐도 왜 확정 안 되는지(no_cluster/stale 등) 진단용."""
        self._fusion_rejects = dict(counts)

    def set_observe_mode(self, dwell: bool, mode: str) -> None:
        """컨트롤러 status에서 정찰 관측 거동 상태(dwell 여부 / mode=dwell|slow|turret|"")."""
        self._obs_dwell = bool(dwell)
        self._obs_mode = str(mode or "")

    def set_observe_candidates(self, summary: Optional[dict]) -> None:
        """local_path_node의 미분류 후보 요약(n/n_fov/n_side/by_class)."""
        self._obs_candidates = summary

    def log_planned_path(self, sim_time: float, path_xz: list) -> None:
        """전역 계획경로를 경로가 '실제로 바뀔 때만' 저장(끝점/길이 시그니처로 중복 제거, 다운샘플)."""
        if not path_xz:
            return
        sig = (len(path_xz),
               round(path_xz[0][0], 1), round(path_xz[0][1], 1),
               round(path_xz[-1][0], 1), round(path_xz[-1][1], 1))
        if sig == self._last_path_sig:
            return
        self._last_path_sig = sig
        ds = path_xz[::3] if len(path_xz) > 60 else path_xz
        self.planned_paths.append({
            "t": round(sim_time, 2),
            "version": self.route_version,
            "path": [[round(x, 2), round(z, 2)] for (x, z) in ds],
        })

    def log_diag_sample(self, sim_time: float, px: float, pz: float) -> None:
        """0.2s 시간 간격으로 (위치+최신 lookahead/local_target/명령+route_version) 스냅샷.
        이동량이 아니라 시간 기준이라 끼임/제자리 진동도 포착한다."""
        if self._last_diag_t is not None and (sim_time - self._last_diag_t) < self._diag_min_dt:
            return
        self._last_diag_t = sim_time
        look = self._latest_lookahead
        ltgt = self._latest_local_target
        sample = {
            "t": round(sim_time, 2),
            "p": [round(px, 2), round(pz, 2)],
            "rv": self.route_version,
            "look": [round(look[0], 2), round(look[1], 2)] if look else None,
            "ltgt": [round(ltgt[0], 2), round(ltgt[1], 2)] if ltgt else None,
            "cmd": self._latest_cmd,
        }
        # 정찰 관측 거동: hold(=의도적 dwell, analyze_run이 이미 읽음) + obs(후보요약+mode).
        if self._obs_mode or self._obs_candidates:
            sample["hold"] = self._obs_dwell
            sample["obs"] = {"mode": self._obs_mode, **(self._obs_candidates or {})}
        self.diag_samples.append(sample)

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
            "collision_events": self.collision_events,
            "exposure": self._build_exposure_summary(),
            "vision_yolo": self._build_vision_summary(),
            "asset_spotted_gt": self._build_asset_spotted_gt(),
            "terrain_roughness": {
                "pitch_std_deg": self._calc_std(self._pitch_samples),
                "roll_std_deg": self._calc_std(self._roll_samples),
            },
            # 주행 품질 진단 원자료(공식 리포트 본문엔 미사용 — scripts/analyze_run.py가 소비).
            # 융합 드롭 사유 히스토그램(프레임당 1건). ok_* = 성공, strict_no_cluster_assignment =
            # YOLO bbox가 DBSCAN cluster와 매칭 안 됨, stale_async_detection = 비동기 stale 등.
            "fusion_rejects": self._fusion_rejects,
            # 주행 품질 진단 원자료(공식 리포트 본문엔 미사용 — scripts/analyze_run.py가 소비).
            "diagnostics": {
                "route_version_final": self.route_version,
                "route_version_changes": self.route_version_changes,
                "planned_paths": self.planned_paths,
                "samples": self.diag_samples,
            },
        }

    def save_report(self) -> str:
        """리포트를 JSON 파일로 저장하고 경로를 반환합니다."""
        report = self.build_report()
        os.makedirs(self.output_dir, exist_ok=True)   # 새 출력폴더(예: 시나리오2 격리 dir) 자동 생성
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
