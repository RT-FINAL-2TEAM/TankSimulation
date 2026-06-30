#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
정찰(recon) 시나리오 관리자 — A 루트 → B 루트 자동 시퀀스.

전제(직접 띄워두어야 함):
  1) ros_bridge 를 auto + 에피소드 제어 모드로(루트 사이 자동 리셋):
       TANK_MODE=auto TANK_EPISODE_CONTROL=true ros2 run ros_bridge ros_bridge
     (TANK_EPISODE_CONTROL을 안 켜면 자동 리셋이 무시되어 수동 RESTART 폴백으로 동작)
  2) 시뮬레이터 실행 + tracking on (auto 브릿지로 /init 되도록 시뮬을 그 다음에 시작)
  3) (선택) RViz:  ros2 launch rviz_visualization tank_rviz.launch.py

동작:
  - Route A 자율 스택 실행 → recon_reports/route_A.json 의 reached=true 감지 → A 스택 종료
  - /tank/episode/control 로 reset 발행 → 브릿지가 /info 응답 control:reset 으로 시뮬을 출발지로 되돌림
    → 전차가 출발지(START)로 복귀하는 것을 자동 감지 (reset 미honor 시 수동 RESTART 폴백)
  - Route B 자율 스택 실행 → route_B.json reached 감지 → B 스택 종료
  - comparison.json 생성

실행:
  source ~/tank_project/install/setup.bash
  python3 ~/tank_project/scripts/run_recon_scenario.py
"""

import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.request

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String

# 프로젝트 루트 = 이 스크립트의 상위(scripts/)의 상위
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT_DIR = os.path.join(PROJECT_ROOT, "recon_reports")
# 시나리오2 인계 산출물 경로(프로젝트 안). local_path_node/terrain 노드는 ~/tankcc에 저장하므로
# 정찰 종료 시 여기로 루트별 복사해 보존한다(다음 루트가 latest를 덮어써도 유지).
RECON_MAP_DIR = os.path.join(REPORT_DIR, "recon_map")
TERRAIN_DIR = os.path.join(REPORT_DIR, "terrain_maps")
DISCOVERED_LATEST = os.path.expanduser("~/tankcc/tank_discovered_maps/discovered_objects_latest.map")
TERRAIN_LATEST = os.path.expanduser("~/tankcc/tank_terrain_maps/terrain_map_latest.npz")

# 출발지(map 좌표). BLUE_START raw (60, 8, 30) → map (x=60, y=z=30)
START = (60.0, 30.0)
# 목적지(map 좌표) = routes.yaml destination로 통일. 도착 판정은 route_*.json의 reached가 1순위, 포즈 기반이 폴백.
GOAL = (110.0, 276.5)
ARRIVE_TOL = 8.0     # 포즈 기반 도착 인정 반경(m)
STABLE_SEC = 20.0    # 목적지 근처에서 이 시간만큼 정지해 있으면 도착으로 간주(폴백)
START_TOL = 12.0          # 출발지 복귀 인정 반경(m)
ARRIVE_TIMEOUT = 900.0    # 한 루트 도착 대기 한계(s)
RESTART_TIMEOUT = 900.0   # 시뮬 재시작(출발지 복귀) 대기 한계(s)
RESET_REPUBLISH_SEC = 7.0 # 자동 reset 재발행 주기(s) — one-shot drain·/info 타이밍 누락 대비
HEALTH_URL = "http://localhost:5000/health"

# (route_id, route_side)
ROUTES = [("A", "west"), ("B", "east")]


class PoseWatcher(Node):
    def __init__(self):
        super().__init__("recon_scenario_pose_watcher")
        self.pos = None
        self.create_subscription(PoseStamped, "/tank/player/pose", self._cb, 10)
        # 루트 사이 시뮬 자동 리셋용 퍼블리셔. 브릿지가 TANK_EPISODE_CONTROL=true일 때만 실제 동작한다.
        self.reset_pub = self.create_publisher(String, "/tank/episode/control", 10)

    def _cb(self, msg: PoseStamped) -> None:
        self.pos = (float(msg.pose.position.x), float(msg.pose.position.y))

    def request_reset(self) -> None:
        """다음 /info 응답으로 시뮬에 reset을 하달하도록 1회 요청한다(one-shot)."""
        self.reset_pub.publish(String(data="reset"))


def dist(a, b) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def report_path(route_id: str) -> str:
    return os.path.join(REPORT_DIR, f"route_{route_id}.json")


def load_report(route_id: str):
    try:
        with open(report_path(route_id), "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def is_reached(route_id: str) -> bool:
    rep = load_report(route_id)
    return bool(rep and rep.get("result", {}).get("reached"))


def _validate_npz(path: str, min_size: int = 1024) -> tuple:
    """terrain NPZ가 정상인지 검증. 반환 (ok, n_points, reason)."""
    if not os.path.exists(path):
        return False, 0, "파일 없음"
    size = os.path.getsize(path)
    if size < min_size:
        return False, 0, f"{size}바이트(임계 {min_size}B 미만 — 0바이트/미완성)"
    try:
        with np.load(path, allow_pickle=True) as d:
            if "accumulated" not in d.files:
                return False, 0, "accumulated 키 없음"
            n = int(getattr(d["accumulated"], "shape", (0,))[0])
    except Exception as e:  # zip 깨짐/EOFError 등
        return False, 0, f"np.load 실패: {type(e).__name__} {e}"
    if n <= 0:
        return False, 0, "누적 점 0개"
    return True, n, "ok"


def _atomic_copy(src: str, dst: str) -> None:
    """src를 dst로 원자적 복사(tmp→os.replace) — 중간 0바이트/부분파일 노출 방지."""
    tmp = dst + ".tmp"
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)


def _safe_remove(path: str) -> None:
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


def _service_available(name: str, timeout: float = 10.0) -> bool:
    """서비스가 그래프에 있는지 빠르게 확인 — 없는 서비스에 call 걸어 길게 멈추는 것 방지."""
    try:
        r = subprocess.run(
            ["ros2", "service", "list"], cwd=PROJECT_ROOT, check=False, timeout=timeout,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        )
        return name in (r.stdout or "")
    except Exception:
        return False


def finalize_recon_artifacts(route_id: str) -> None:
    """정찰 발견객체+지형을 디스크에 저장하고 프로젝트 경로(recon_reports/)로 루트별 복사한다.

    ★ 반드시 스택 생존 중(kill_stack 전)에 호출 — local_path_node가 살아있어야 서비스가 응답한다.
    저장은 두 서비스로 명시 트리거한다:
      1) /tank/map/discovered/save — 발견객체 맵(local_path_node, 동기)
      2) /tank/terrain/finalize_map — 지형 NPZ. 콜백이 save_outputs까지 콜백 내에서 동기 실행 후
         응답하므로 `ros2 service call`(동기)이 **저장 완료까지 블록**한다(과거 call_async+4초 sleep
         레이스로 첫 레그 A가 0바이트로 복사되던 버그 정석 해결).
    복사 전 NPZ를 검증(크기+np.load+누적점>0)하고 원자적으로 복사하며, 실패 시 깨진 복사본을 남기지
    않고 시끄럽게 경고한다(다음 build_scenario2_map.py가 0바이트 파일을 만나지 않게).
    실패해도 정찰 시퀀스엔 영향 없음(graceful).
    """
    os.makedirs(RECON_MAP_DIR, exist_ok=True)
    os.makedirs(TERRAIN_DIR, exist_ok=True)
    t_start = time.time()  # 원본 mtime 신선도 판정(이번 finalize 결과인지 vs 이전 루트 잔재)

    # 1) 발견객체 맵 저장(동기)
    print(f"  [Route {route_id}] 발견객체 저장 트리거 (/tank/map/discovered/save)...")
    try:
        subprocess.run(
            ["ros2", "service", "call", "/tank/map/discovered/save", "std_srvs/srv/Trigger", "{}"],
            cwd=PROJECT_ROOT, check=False, timeout=30,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except Exception as e:
        print(f"  [Route {route_id}] [경고] 발견맵 save 트리거 실패: {e}")

    # 2) 지형 finalize — 노드가 떠 있으면 동기 호출(저장 완료까지 블록). 없으면 graceful skip(멈춤 방지).
    if _service_available("/tank/terrain/finalize_map"):
        print(f"  [Route {route_id}] 지형 finalize 트리거 (/tank/terrain/finalize_map, 동기 대기)...")
        try:
            r = subprocess.run(
                ["ros2", "service", "call", "/tank/terrain/finalize_map", "std_srvs/srv/Trigger", "{}"],
                cwd=PROJECT_ROOT, check=False, timeout=90,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            )
            out = (r.stdout or "").strip()
            if "success=True" not in out and "success=true" not in out:
                print(f"  [Route {route_id}] [참고] 지형 finalize 미확정(점 부족 가능): {out[:160]}")
        except Exception as e:
            print(f"  [Route {route_id}] [참고] 지형 finalize 호출 실패: {e}")
    else:
        print(f"  [Route {route_id}] [참고] /tank/terrain/finalize_map 서비스 없음 — 지형 노드 미기동(지형 없이 진행)")

    # 3) 발견맵 복사(존재+비어있지 않음 검증)
    disc_dst = os.path.join(RECON_MAP_DIR, f"discovered_objects_route_{route_id}.map")
    if os.path.exists(DISCOVERED_LATEST) and os.path.getsize(DISCOVERED_LATEST) > 0:
        _atomic_copy(DISCOVERED_LATEST, disc_dst)
        print(f"  [Route {route_id}] 발견맵 → {disc_dst}")
    else:
        print(f"  [Route {route_id}] [경고] 발견맵 미생성/0바이트: {DISCOVERED_LATEST}")

    # 4) 지형 NPZ 검증 → 원자적 복사 (0바이트/미완성/이전루트 잔재 차단)
    terr_dst = os.path.join(TERRAIN_DIR, f"terrain_map_route_{route_id}.npz")
    ok, npts, reason = _validate_npz(TERRAIN_LATEST)
    fresh = os.path.exists(TERRAIN_LATEST) and os.path.getmtime(TERRAIN_LATEST) >= t_start - 1.0
    if ok and fresh:
        _atomic_copy(TERRAIN_LATEST, terr_dst)
        ok2, npts2, reason2 = _validate_npz(terr_dst)
        if ok2:
            print(f"  [Route {route_id}] 지형({npts2:,}점) → {terr_dst}")
        else:
            print(f"  [Route {route_id}] [경고] 지형 복사본 검증 실패({reason2}) — 제거")
            _safe_remove(terr_dst)
    elif not os.path.exists(TERRAIN_LATEST):
        print(f"  [Route {route_id}] [참고] 지형 NPZ 미생성: {TERRAIN_LATEST} "
              f"(terrain finalize 노드가 안 떠 있으면 정상 — 지형 없이 진행)")
        _safe_remove(terr_dst)
    else:
        why = reason if not ok else f"이번 finalize 결과 아님(stale, mtime<시작 {t_start:.0f})"
        print("  " + "!" * 64)
        print(f"  [Route {route_id}] [경고] 지형 NPZ 저장 실패 — 복사 생략: {why}")
        print(f"           원본: {TERRAIN_LATEST}")
        print(f"           ⇒ 시나리오2 지형에서 {route_id} 구역이 누락됩니다. 이 루트를 재정찰하세요.")
        print("  " + "!" * 64)
        _safe_remove(terr_dst)  # 과거 0바이트 잔재 제거(build가 깨진 파일 안 만나게)


def reset_terrain_recording(route_id: str) -> None:
    """각 루트 시작 시 지형 노드를 reset(점 clear + 녹화 재개)한다.

    이전 루트의 finalize가 녹화를 꺼두므로(terrain node _recording_enable=False), reset 없이는
    다음 루트(B) 지형이 안 쌓인다. 또 이전 정찰/루트 점이 그 위로 누적되지 않게 clear한다.
    → 루트별 fresh 지형(route_A=A, route_B=B) + 정찰 간 누적 방지. graceful(노드 없으면 무시)."""
    try:
        subprocess.run(
            ["ros2", "service", "call", "/tank/terrain/reset_map", "std_srvs/srv/Trigger", "{}"],
            cwd=PROJECT_ROOT, check=False, timeout=15,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        print(f"  [Route {route_id}] 지형 녹화 reset(clear+재개)")
    except Exception as e:
        print(f"  [Route {route_id}] [참고] 지형 reset 생략(노드 없을 수 있음): {e}")


def launch_route(route_id: str, side: str) -> subprocess.Popen:
    # 정찰 = 평범 주행 + 발견객체 기록. 포탑 stop-and-aim·차체 weave는 그냥 주행보다 확정을
    # 깎아먹어 제거함(2026-06-23). 커버리지는 routes.yaml 웨이포인트(사용자 설계)로.
    cmd = [
        "ros2", "launch", "control", "tank_autonomous_control.launch.py",
        "mission_type:=recon", f"route_id:={route_id}", f"route_side:={side}",
    ]
    # 자체 프로세스 그룹으로 띄워, 종료 시 launch가 띄운 7개 노드를 한 번에 정리한다.
    return subprocess.Popen(cmd, cwd=PROJECT_ROOT, preexec_fn=os.setsid)


def kill_stack(proc: subprocess.Popen) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGINT)
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except ProcessLookupError:
        pass


# 자율 스택 노드 실행파일 키워드. (브릿지/RViz 노드는 제외 — 함께 죽이지 않는다)
# pkill -f tank_autonomous_control 은 launch 래퍼만 잡고 노드 프로세스(명령줄에
# launch 파일명이 없음)는 못 잡으므로, 노드 실행파일 이름으로 직접 정리한다.
AUTONOMY_NODE_KEYS = [
    "lidar_processor_node",
    "lidar_camera_overlay",
    "lidar_dbscan_cluster_node",
    "map_astar_planner_node",
    "local_path_node",
    "potential_field_node",
    "tank_controller_node",
]

# 노드 이름 pkill이 안 먹는 환경 대비, 자율 패키지 install 경로로도 강제 종료한다.
# 매니저가 띄운 노드 명령줄에는 이 경로가 확실히 들어있어 반드시 잡힌다.
# (ros_bridge / rviz_visualization 경로는 제외 — 브릿지·RViz는 죽이지 않음)
AUTONOMY_PKG_PATHS = [
    "install/lidar/lib",
    "install/tank_visual_perception/lib",
    "install/path_planning/lib",
    "install/potential/lib",
    "install/control/lib",
]


def cleanup_stack() -> None:
    """이전 실행에서 남은(orphan) 자율 스택 노드를 강제 정리한다. 중복 실행 충돌 방지."""
    for key in AUTONOMY_NODE_KEYS:
        subprocess.run(["pkill", "-9", "-f", key], check=False)
    for path in AUTONOMY_PKG_PATHS:
        subprocess.run(["pkill", "-9", "-f", path], check=False)
    time.sleep(2.0)


def check_bridge_auto() -> None:
    try:
        with urllib.request.urlopen(HEALTH_URL, timeout=2.0) as r:
            data = json.loads(r.read().decode("utf-8"))
        mode = data.get("mode")
        if mode != "auto":
            print(f"[경고] 브릿지 mode='{mode}' (auto 아님) — 전차가 안 움직일 수 있습니다. "
                  f"TANK_MODE=auto 로 브릿지를 다시 띄우세요.")
        else:
            print("[확인] 브릿지 auto 모드 OK")
        # 루트 사이 자동 리셋은 브릿지의 에피소드 제어가 켜져 있어야 동작한다.
        if not data.get("episode_control"):
            print("[경고] 브릿지 episode_control=off — 루트 사이 자동 리셋이 무시됩니다. "
                  "TANK_EPISODE_CONTROL=true 로 브릿지를 띄우거나, 안내대로 시뮬을 수동 RESTART 하세요.")
        else:
            print("[확인] 브릿지 episode_control ON — 자동 리셋 사용 가능")
    except Exception:
        print("[경고] /health 확인 실패 — 브릿지가 떠 있는지(auto), 포트(5000)를 확인하세요.")


def main() -> int:
    rclpy.init()
    watcher = PoseWatcher()
    spin = threading.Thread(target=rclpy.spin, args=(watcher,), daemon=True)
    spin.start()

    current_proc = None

    print(f"recon_reports 경로: {REPORT_DIR}")
    check_bridge_auto()
    print("기존 자율 스택(orphan) 노드 정리 중...")
    cleanup_stack()

    try:
        for idx, (route_id, side) in enumerate(ROUTES):
            # 이전 결과 제거(이번 시퀀스 산출물만 남도록)
            try:
                os.remove(report_path(route_id))
            except FileNotFoundError:
                pass

            # 지형 노드를 루트별로 reset(점 clear + 녹화 재개). 이전 루트 finalize가 녹화를 꺼두므로
            # 안 하면 B 지형이 안 쌓이고, 이전 점 위에 누적됨. (지형 노드 없으면 graceful 무시)
            reset_terrain_recording(route_id)

            print(f"\n=== [Route {route_id}] 자율주행 시작 (side={side}) ===")
            current_proc = launch_route(route_id, side)

            # 도착(route_*.json의 reached=true)까지 대기. 10초마다 목적지까지 남은 거리를 표시한다.
            t0 = time.time()
            arrived = False
            arrived_by = ""
            last_beat = 0.0
            near_since = None
            while time.time() - t0 < ARRIVE_TIMEOUT:
                if is_reached(route_id):
                    arrived = True
                    arrived_by = "route_json"
                    break
                now = time.time()
                # 포즈 기반 폴백: 목적지 근처(ARRIVE_TOL)에서 STABLE_SEC 이상 정지 → 도착 간주
                if watcher.pos is not None and dist(watcher.pos, GOAL) < ARRIVE_TOL:
                    if near_since is None:
                        near_since = now
                    elif now - near_since >= STABLE_SEC:
                        arrived = True
                        arrived_by = "pose_fallback"
                        print(f"  [Route {route_id}] 포즈 기반 도착 감지 "
                              f"(목적지 {dist(watcher.pos, GOAL):.1f}m, {STABLE_SEC:.0f}s 정지). "
                              f"⚠ route_{route_id}.json은 local_path 미기록일 수 있음.")
                        break
                else:
                    near_since = None
                if now - last_beat >= 10.0:
                    last_beat = now
                    if watcher.pos is not None:
                        print(f"  [Route {route_id}] 주행 중... 목적지까지 {dist(watcher.pos, GOAL):.1f}m "
                              f"(현재 {watcher.pos[0]:.1f}, {watcher.pos[1]:.1f})")
                    else:
                        print(f"  [Route {route_id}] 주행 중... (player pose 수신 대기)")
                time.sleep(1.0)
            print(f"  [Route {route_id}] 도착감지={arrived} ({arrived_by})")
            # 스택 죽이기 전에 발견객체·지형을 저장·복사(시나리오2 인계용). 도착 시에만.
            if arrived:
                finalize_recon_artifacts(route_id)
            kill_stack(current_proc)
            current_proc = None
            cleanup_stack()
            if not arrived:
                print(f"[Route {route_id}] 도착 실패/타임아웃 — 시퀀스 중단")
                return 1
            print(f"=== [Route {route_id}] 도착 완료 → {report_path(route_id)} ===")

            # 마지막 루트가 아니면 시뮬을 출발지로 되돌리고 복귀를 기다린다.
            if idx < len(ROUTES) - 1:
                print("\n★ 시뮬레이터 자동 리셋 전송 → 전차를 출발지(START)로 되돌립니다.")
                print("   (자동 리셋이 안 먹으면: 브릿지를 TANK_EPISODE_CONTROL=true로 띄웠는지 확인,")
                print("    또는 시뮬을 수동 RESTART 하세요 — 출발지 복귀는 자동 감지됩니다. Ctrl+C로 중단)")
                # reset은 1회성(다음 /info 응답에 1번 실리고 비워짐)이라, 복귀 감지 전까지 주기적으로 재발행한다.
                watcher.request_reset()
                t_reset = time.time()
                t0 = time.time()
                back = False
                while time.time() - t0 < RESTART_TIMEOUT:
                    if watcher.pos is not None and dist(watcher.pos, START) < START_TOL:
                        back = True
                        break
                    if time.time() - t_reset >= RESET_REPUBLISH_SEC:
                        watcher.request_reset()
                        t_reset = time.time()
                    time.sleep(0.5)
                if not back:
                    print("출발지 복귀 미감지 — 시퀀스 중단")
                    return 1
                print("출발지 복귀 감지 — 다음 루트 진행")
                time.sleep(2.0)  # 노드/토픽 안정화 여유

        # 비교 리포트 생성
        rep_a, rep_b = load_report("A"), load_report("B")
        if rep_a and rep_b:
            comp = {"route_A": rep_a, "route_B": rep_b}
            comp_path = os.path.join(REPORT_DIR, "comparison.json")
            with open(comp_path, "w", encoding="utf-8") as f:
                json.dump(comp, f, ensure_ascii=False, indent=2)
            print("\n" + "=" * 56)
            print("✅ 정찰 시퀀스 완료 (A·B) — 시뮬레이터는 수동 종료하셔도 됩니다.")
            print("=" * 56)
            print(f"  리포트: {report_path('A')}")
            print(f"         {report_path('B')}")
            print(f"         {comp_path}")
            print("\n--- 정찰 비교 (A vs B) ---")
            for rep in (rep_a, rep_b):
                r = rep.get("result", {})
                o = rep.get("obstacle_summary", {})
                print(
                    f"  Route {rep.get('route')}: "
                    f"reached={r.get('reached')}, "
                    f"distance={r.get('distance_m')}m, "
                    f"sim_time={r.get('sim_time_s')}s, "
                    f"collisions={r.get('collisions')}, "
                    f"obstacles={o.get('count')}"
                )

            # LLM 전술 분석 자동 실행(graceful) → /tank/risk/route_report 발행 → 브릿지 MFD 표시.
            # comparison.json → make_llm_input(route_comparison.json) → route_risk_node(ollama)
            # → generate_route_summary_txt(route_analysis_report.txt).
            # ollama 미가동/타임아웃이어도 정찰 완료엔 영향 없음(예외 흡수).
            scripts_dir = os.path.dirname(os.path.abspath(__file__))
            print("\n--- LLM 전술 분석 자동 실행 ---")
            try:
                # 새 comparison.json 반영: 수식·LLM 공통 입력(risk_features)+수식 verdict+보고서 갱신.
                # (stale risk_features 재사용 방지 — make_llm_input은 이 산출물을 읽는다.)
                subprocess.run(
                    ["python3", os.path.join(scripts_dir, "generate_recon_report.py")],
                    cwd=PROJECT_ROOT, check=False, timeout=120,
                )
                subprocess.run(
                    ["python3", os.path.join(scripts_dir, "make_llm_input.py")],
                    cwd=PROJECT_ROOT, check=True, timeout=60,
                )
                print("  🧠 ollama 추론 중... (qwen3:0.6b ~15-30초 · 끊지 말고 기다리세요)", flush=True)
                subprocess.run(
                    ["ros2", "run", "risk_analysis", "route_risk_node"],
                    cwd=PROJECT_ROOT, timeout=180,
                )
                print(f"  ✅ LLM 결과: {os.path.join(REPORT_DIR, 'route_risk_result.json')} (+ MFD AI LOG 표시)")
            except Exception as e:
                print(f"  [LLM] 자동 분석 생략: {e} (ollama 미가동/타임아웃 — 수동 실행 가능)")
            finally:
                try:
                    subprocess.run(
                        ["python3", os.path.join(scripts_dir, "generate_route_summary_txt.py")],
                        cwd=PROJECT_ROOT, check=True, timeout=60,
                    )
                    print(f"  📝 TXT 보고서: {os.path.join(REPORT_DIR, 'route_analysis_report.txt')}")
                except Exception as e:
                    print(f"  [TXT] 루트 분석 보고서 생성 실패: {e}")
                try:
                    # 수식 verdict vs LLM 판단 비교 → risk_comparison.{json,md} (+ MFD RECON RISK 패널).
                    subprocess.run(
                        ["python3", os.path.join(scripts_dir, "compare_verdicts.py")],
                        cwd=PROJECT_ROOT, check=True, timeout=60,
                    )
                    print(f"  ⚖️  수식 vs LLM 비교: {os.path.join(REPORT_DIR, 'risk_comparison.md')}")
                except Exception as e:
                    print(f"  [CMP] 수식·LLM 비교 생성 실패: {e}")
        else:
            print("\n[경고] route_A.json / route_B.json 일부 누락 — comparison 생략")
        return 0

    except KeyboardInterrupt:
        print("\n사용자 중단(Ctrl+C)")
        return 130
    finally:
        kill_stack(current_proc)
        try:
            watcher.destroy_node()
        except Exception:
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
