#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""시나리오2 임무 관리자 — 단일 루트 A 돌파/복귀 1회 실행.

정찰(시1)로 만든 합본맵(scenario2_map.map) 위에서 임무를 1회 주행한다. 정찰의 A→B 2레그
오케스트레이터(run_recon_scenario.py)를 미러하되, 시나리오2는 **한 번의 임무 run**이고
돌파↔복귀는 decision_node(전술 FSM)가 내부에서 처리한다(별도 레그 아님).

전제(직접 띄워두어야 함):
  1) ros_bridge auto: TANK_MODE=auto ros2 run ros_bridge ros_bridge
  2) 시뮬레이터 실행 + tracking on
  3) scenario2_map.map 사전 생성: python3 scripts/build_scenario2_map.py
  (선택) RViz: ros2 launch rviz_visualization tank_rviz.launch.py

동작:
  - tank_scenario2.launch.py 로 자율 스택 + decision_node + mock_turret_node 실행.
  - 종료 판정(셋 중 하나):
      성공(reached)   = 목적지 도달(route_A.json reached 또는 pose가 GOAL 근처 안정).
      중단(aborted)   = decision가 RETURN 진입 후 출발지(START) 복귀.
      타임아웃(timeout)= ARRIVE_TIMEOUT 초과.
  - recon_reports/scenario2_result.json 기록.

실행:
  source ~/tank_project/install/setup.bash
  python3 ~/tank_project/scripts/run_scenario2_scenario.py
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
from std_msgs.msg import String

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT_DIR = os.path.join(PROJECT_ROOT, "recon_reports")
# 시나리오2 자체 산출물(자기 route_A.json 텔레메트리 + scenario2_result.json)은 별도 폴더로 격리.
# 정찰의 recon_reports/route_A.json을 덮어쓰지 않기 위함(파일명 충돌 근본 수정).
SCENARIO2_DIR = os.path.join(REPORT_DIR, "scenario2")

# routes.yaml finalmap 과 정합. 출발지/목적지(map 좌표).
START = (59.0, 27.0)
GOAL = (110.0, 276.5)
ARRIVE_TOL = 8.0          # 포즈 기반 도착 인정 반경(m)
STABLE_SEC = 20.0         # 목적지 근처 정지 유지 시 도착 간주(폴백)
START_TOL = 12.0          # 출발지 복귀 인정 반경(m)
ARRIVE_TIMEOUT = 900.0    # 임무 종료 대기 한계(s)
HEALTH_URL = "http://localhost:5000/health"

ROUTE_ID = "A"  # 시나리오2 임무 루트 = A 고정(설계)


class MissionWatcher(Node):
    """player pose + decision 상태 + 교전 결과를 모아 종료 판정·리포트에 쓴다."""

    def __init__(self):
        super().__init__("scenario2_mission_watcher")
        self.pos = None
        self.fsm_state = None
        self.last_status = None
        self.entered_return = False
        self.engagements = []   # 교전 결과 누적
        self.risk_events = []   # RETURN 트리거 등 위험 이벤트
        self.create_subscription(PoseStamped, "/tank/player/pose", self._pose_cb, 10)
        self.create_subscription(String, "/tank/decision/status", self._status_cb, 10)
        self.create_subscription(String, "/tank/engage/result", self._engage_cb, 10)

    def _pose_cb(self, msg: PoseStamped) -> None:
        self.pos = (float(msg.pose.position.x), float(msg.pose.position.y))

    def _status_cb(self, msg: String) -> None:
        try:
            st = json.loads(msg.data)
        except Exception:
            return
        self.last_status = st
        new_state = st.get("state")
        if new_state != self.fsm_state:
            print(f"  [FSM] {self.fsm_state} → {new_state} ({st.get('reason')})")
            self.fsm_state = new_state
            if new_state == "RETURN":
                self.entered_return = True
                self.risk_events.append({"wall": st.get("wall"), "reason": st.get("reason"),
                                         "risk": st.get("risk")})

    def _engage_cb(self, msg: String) -> None:
        try:
            self.engagements.append(json.loads(msg.data))
        except Exception:
            pass


def dist(a, b) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def report_path(route_id: str) -> str:
    # 시나리오2 local_path가 recon_report_dir=SCENARIO2_DIR로 쓰는 자기 텔레메트리.
    return os.path.join(SCENARIO2_DIR, f"route_{route_id}.json")


def load_route_report(route_id: str):
    try:
        with open(report_path(route_id), "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def is_reached(route_id: str) -> bool:
    rep = load_route_report(route_id)
    return bool(rep and rep.get("result", {}).get("reached"))


def launch_stack() -> subprocess.Popen:
    cmd = ["ros2", "launch", "control", "tank_scenario2.launch.py"]
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


# 자율 스택 + 시나리오2 노드 실행파일 키워드(브릿지/RViz 제외).
AUTONOMY_NODE_KEYS = [
    "lidar_processor_node", "lidar_camera_overlay", "lidar_dbscan_cluster_node",
    "map_astar_planner_node", "local_path_node", "potential_field_node",
    "tank_controller_node", "decision_node", "mock_turret_node",
]
AUTONOMY_PKG_PATHS = [
    "install/lidar/lib", "install/tank_visual_perception/lib",
    "install/path_planning/lib", "install/potential/lib",
    "install/control/lib", "install/mission/lib",
]


def cleanup_stack() -> None:
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


def check_scenario2_map() -> None:
    path = os.path.join(REPORT_DIR, "recon_map", "scenario2_map.map")
    if not os.path.exists(path):
        print(f"[경고] {path} 없음 — 먼저 build_scenario2_map.py 로 생성하세요 "
              f"(없으면 launch 기본경로 로드 실패).")
    else:
        try:
            d = json.load(open(path))
            print(f"[확인] scenario2_map.map OK (obstacles={d.get('object_count')}, "
                  f"targets/known-tank={d.get('target_count')})")
        except Exception as e:
            print(f"[경고] scenario2_map.map 파싱 실패: {e}")


def write_result(outcome: str, watcher: MissionWatcher) -> str:
    rep = load_route_report(ROUTE_ID) or {}
    result = rep.get("result", {})
    out = {
        "outcome": outcome,                                   # success | aborted | timeout
        "route": ROUTE_ID,
        "reached": bool(result.get("reached")) or outcome == "success",
        "returned": watcher.entered_return,
        "final_fsm_state": watcher.fsm_state,
        "engagements": watcher.engagements,
        "engagement_success_count": sum(1 for e in watcher.engagements if e.get("success")),
        "risk_events": watcher.risk_events,
        "distance_m": result.get("distance_m"),
        "sim_time_s": result.get("sim_time_s"),
        "collisions": result.get("collisions"),
        "wall": time.time(),
    }
    os.makedirs(SCENARIO2_DIR, exist_ok=True)
    path = os.path.join(SCENARIO2_DIR, "scenario2_result.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    return path


def main() -> int:
    rclpy.init()
    watcher = MissionWatcher()
    spin = threading.Thread(target=rclpy.spin, args=(watcher,), daemon=True)
    spin.start()

    current_proc = None
    os.makedirs(SCENARIO2_DIR, exist_ok=True)   # 시나리오2 격리 폴더(자기 route_A.json·결과)
    print(f"시나리오2 산출물 경로: {SCENARIO2_DIR} (정찰 recon_reports/route_A.json 미접촉)")
    check_bridge_auto()
    check_scenario2_map()
    print("기존 자율 스택(orphan) 노드 정리 중...")
    cleanup_stack()

    # 이전 결과 제거(이번 run 산출물만 남도록)
    for p in (report_path(ROUTE_ID), os.path.join(SCENARIO2_DIR, "scenario2_result.json")):
        try:
            os.remove(p)
        except FileNotFoundError:
            pass

    outcome = "timeout"
    try:
        print(f"\n=== 시나리오2 임무 시작 (route={ROUTE_ID}) ===")
        current_proc = launch_stack()

        t0 = time.time()
        last_beat = 0.0
        near_since = None
        while time.time() - t0 < ARRIVE_TIMEOUT:
            now = time.time()
            # 1) 성공: 목적지 도달.
            if is_reached(ROUTE_ID):
                outcome = "success"
                print("  [종료] 목적지 도달(route_json reached)")
                break
            if watcher.pos is not None and dist(watcher.pos, GOAL) < ARRIVE_TOL:
                if near_since is None:
                    near_since = now
                elif now - near_since >= STABLE_SEC:
                    outcome = "success"
                    print(f"  [종료] 포즈 기반 목적지 도달({dist(watcher.pos, GOAL):.1f}m, {STABLE_SEC:.0f}s 정지)")
                    break
            else:
                near_since = None
            # 2) 중단: RETURN 진입 후 출발지 복귀.
            if watcher.entered_return and watcher.pos is not None and dist(watcher.pos, START) < START_TOL:
                outcome = "aborted"
                print(f"  [종료] 복귀 완료 — 출발지 도달({dist(watcher.pos, START):.1f}m)")
                break
            # 하트비트
            if now - last_beat >= 10.0:
                last_beat = now
                st = watcher.fsm_state or "?"
                if watcher.pos is not None:
                    tgt = START if watcher.entered_return else GOAL
                    label = "출발지" if watcher.entered_return else "목적지"
                    print(f"  [{st}] 주행 중... {label}까지 {dist(watcher.pos, tgt):.1f}m "
                          f"(현재 {watcher.pos[0]:.1f}, {watcher.pos[1]:.1f})")
                else:
                    print(f"  [{st}] player pose 수신 대기...")
            time.sleep(1.0)

        path = write_result(outcome, watcher)
        print("\n" + "=" * 56)
        print(f"시나리오2 종료: outcome={outcome}")
        print("=" * 56)
        print(f"  결과: {path}")
        print(f"  교전 {len(watcher.engagements)}건(성공 "
              f"{sum(1 for e in watcher.engagements if e.get('success'))}건), "
              f"위험 이벤트 {len(watcher.risk_events)}건")
        return 0 if outcome in ("success", "aborted") else 1

    except KeyboardInterrupt:
        print("\n사용자 중단(Ctrl+C)")
        try:
            write_result(outcome, watcher)
        except Exception:
            pass
        return 130
    finally:
        kill_stack(current_proc)
        cleanup_stack()
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
