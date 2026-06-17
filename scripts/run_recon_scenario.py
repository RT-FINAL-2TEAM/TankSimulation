#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
정찰(recon) 시나리오 관리자 — A 루트 → B 루트 자동 시퀀스.

전제(직접 띄워두어야 함):
  1) ros_bridge 를 auto 모드로:   TANK_MODE=auto ros2 run ros_bridge ros_bridge
  2) 시뮬레이터 실행 + tracking on (auto 브릿지로 /init 되도록 시뮬을 그 다음에 시작)
  3) (선택) RViz:  ros2 launch rviz_visualization tank_rviz.launch.py

동작:
  - Route A 자율 스택 실행 → recon_reports/route_A.json 의 reached=true 감지 → A 스택 종료
  - "시뮬 RESTART" 안내 → 전차가 출발지(START)로 복귀하는 것을 자동 감지
    (ROS는 시뮬을 리셋할 수 없으므로 루트 사이 시뮬 재시작은 사용자가 수행)
  - Route B 자율 스택 실행 → route_B.json reached 감지 → B 스택 종료
  - comparison.json 생성

실행:
  source ~/tank_project/install/setup.bash
  python3 ~/tank_project/scripts/run_recon_scenario.py
"""

import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.request

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped

# 프로젝트 루트 = 이 스크립트의 상위(scripts/)의 상위
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT_DIR = os.path.join(PROJECT_ROOT, "recon_reports")

# 출발지(map 좌표). BLUE_START raw (60, 8, 30) → map (x=60, y=z=30)
START = (60.0, 30.0)
# 목적지(map 좌표) = routes.yaml destination로 통일. 도착 판정은 route_*.json의 reached가 1순위, 포즈 기반이 폴백.
GOAL = (110.0, 276.5)
ARRIVE_TOL = 8.0     # 포즈 기반 도착 인정 반경(m)
STABLE_SEC = 20.0    # 목적지 근처에서 이 시간만큼 정지해 있으면 도착으로 간주(폴백)
START_TOL = 12.0          # 출발지 복귀 인정 반경(m)
ARRIVE_TIMEOUT = 900.0    # 한 루트 도착 대기 한계(s)
RESTART_TIMEOUT = 900.0   # 시뮬 재시작(출발지 복귀) 대기 한계(s)
HEALTH_URL = "http://localhost:5000/health"

# (route_id, route_side)
ROUTES = [("A", "west"), ("B", "east")]


class PoseWatcher(Node):
    def __init__(self):
        super().__init__("recon_scenario_pose_watcher")
        self.pos = None
        self.create_subscription(PoseStamped, "/tank/player/pose", self._cb, 10)

    def _cb(self, msg: PoseStamped) -> None:
        self.pos = (float(msg.pose.position.x), float(msg.pose.position.y))


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


def launch_route(route_id: str, side: str) -> subprocess.Popen:
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
    except Exception:
        print("[경고] /health 확인 실패 — 브릿지가 떠 있는지(auto), 포트(5000)를 확인하세요.")


def main() -> int:
    rclpy.init()
    watcher = PoseWatcher()
    spin = threading.Thread(target=rclpy.spin, args=(watcher,), daemon=True)
    spin.start()

    current_proc = None

    def wait_until(predicate, timeout: float, poll: float = 1.0) -> bool:
        t0 = time.time()
        while time.time() - t0 < timeout:
            if predicate():
                return True
            time.sleep(poll)
        return False

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
            kill_stack(current_proc)
            current_proc = None
            cleanup_stack()
            if not arrived:
                print(f"[Route {route_id}] 도착 실패/타임아웃 — 시퀀스 중단")
                return 1
            print(f"=== [Route {route_id}] 도착 완료 → {report_path(route_id)} ===")

            # 마지막 루트가 아니면 시뮬 재시작 대기
            if idx < len(ROUTES) - 1:
                print("\n★ 시뮬레이터를 RESTART 해주세요(전차를 출발지로 되돌림).")
                print("   전차가 출발지로 복귀하면 자동으로 다음 루트를 시작합니다... (Ctrl+C로 중단)")
                back = wait_until(
                    lambda: watcher.pos is not None and dist(watcher.pos, START) < START_TOL,
                    RESTART_TIMEOUT, poll=0.5,
                )
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
