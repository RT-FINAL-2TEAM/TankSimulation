#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""포탑·사격 자동 실험 — 정지 상태에서 포탑 동역학 + 사격 탄도를 자동 sweep·기록.

**용도**: 포탑 제어 게인(정찰 step-stare/사격 조준)과 사격 탄도(거리별 pitch·지형 보정)를 *데이터로* 잡는다.
이 PC는 GPU 없음 → perception 무관한 포탑/사격 실험만 여기서. (퓨전·GPU는 GPU PC에서 별도.)

**실행 전제**:
  - 브릿지를 auto로 띄우고(`TANK_MODE=auto ros2 run ros_bridge ros_bridge`) 시뮬을 (재)시작.
  - **recon 컨트롤러(tank_autonomous_control)는 띄우지 말 것** — 명령 충돌(brige가 latest_command를 씀).
    이 스크립트가 유일한 /tank/control/command 발행자가 되어 포탑/사격을 직접 제어한다.
  - 새 터미널: `source install/setup.bash` 후 실행.

**사용**:
  python3 scripts/run_turret_experiment.py --phase dynamics      # 포탑 yaw/pitch 각속도·데드밴드
  python3 scripts/run_turret_experiment.py --phase fire          # 정지 사격 탄도(차체 자세도 함께 기록=지형실험 겸용)
  python3 scripts/run_turret_experiment.py --phase both
  # 사격은 dynamics에서 측정된 부호를 쓰는 게 정확: --qe-sign / --rf-sign (기본 +1)

**산출**: recon_reports/experiments/turret_dynamics.csv , fire_ballistics.csv
  지형 실험 = 전차를 경사지에 세워두고 `--phase fire` 재실행하면 body_pitch/roll가 달라져 같은 표에 쌓인다.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import threading
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import PointStamped, PoseStamped, Vector3Stamped

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(PROJECT_ROOT, "recon_reports", "experiments")


def norm180(a: float) -> float:
    return (float(a) + 180.0) % 360.0 - 180.0


class TurretExperiment(Node):
    def __init__(self, qe_sign: int, rf_sign: int):
        super().__init__("turret_fire_experiment")
        self.qe_sign = 1 if qe_sign >= 0 else -1
        self.rf_sign = 1 if rf_sign >= 0 else -1
        self.keep_active_w = 0.0   # >0이면 STOP 대신 저속 W(시뮬을 '움직이는 상태'로 유지 → /get_action 폴링 빨라짐)
        self.fixed_yaw = None      # 고정 yaw(개활지 특정 방향). None이면 전방(차체) 또는 적
        self.fire_attempts = 3     # 탄착 안 오면(재장전) 재발사 시도 횟수
        self.reload_sec = 6.0      # 재발사 전 재장전 대기
        self.impact_timeout = 10.0 # 탄착 대기(포탄 비행시간 포함)
        self.aim_enemy = False     # 기본=전방(차체) 조준; True면 적전차 방위(적이 뒤면 포탑이 뒤로 돎 — 탄도엔 부적합)
        self.body_yaw = None       # playerBodyX = 전방(차체 yaw)
        self._last_fire_wall = 0.0 # 직전 발사 시각(재장전 대기용)
        self._fire_t = 0.0         # 직전 fire 명령 시작 시각(탄착 판정 기준)
        self.pub_cmd = self.create_publisher(String, "/tank/control/command", 10)
        # 피드백 상태
        self.turret_yaw = None        # playerTurretX (deg, world)
        self.turret_pitch = None      # playerTurretY (deg)
        self.body_pitch = 0.0         # playerBodyY
        self.body_roll = 0.0          # playerBodyZ
        self.player_xy = None
        self.enemy_xy = None
        self._impact = None           # 최근 탄착 (x,y,z, t)
        self._impact_target = ""      # 최근 탄착 hit 종류
        self.create_subscription(Vector3Stamped, "/tank/api/get_action/turret", self._turret_cb, 10)
        self.create_subscription(String, "/tank/api/info/compact", self._info_cb, 10)
        self.create_subscription(PointStamped, "/tank/api/update_bullet/impact_map", self._impact_cb, 10)
        self.create_subscription(String, "/tank/api/update_bullet/target", self._target_cb, 10)
        self.create_subscription(PoseStamped, "/tank/player/pose", self._player_cb, 10)
        self.create_subscription(PoseStamped, "/tank/enemy/pose", self._enemy_cb, 10)

    # ---- 콜백 ----
    def _turret_cb(self, m: Vector3Stamped):
        self.turret_yaw = float(m.vector.x)
        self.turret_pitch = float(m.vector.y)

    def _info_cb(self, m: String):
        try:
            d = json.loads(m.data)
            data = d.get("data", d) if isinstance(d, dict) else {}
            if "playerBodyX" in data:
                self.body_yaw = float(data.get("playerBodyX", 0.0))
            if "playerBodyY" in data:
                self.body_pitch = norm180(float(data.get("playerBodyY", 0.0)))
            if "playerBodyZ" in data:
                self.body_roll = norm180(float(data.get("playerBodyZ", 0.0)))
        except Exception:
            pass

    def _impact_cb(self, m: PointStamped):
        self._impact = (float(m.point.x), float(m.point.y), float(m.point.z), time.time())

    def _target_cb(self, m: String):
        self._impact_target = str(m.data)

    def _player_cb(self, m: PoseStamped):
        self.player_xy = (float(m.pose.position.x), float(m.pose.position.y))

    def _enemy_cb(self, m: PoseStamped):
        self.enemy_xy = (float(m.pose.position.x), float(m.pose.position.y))

    # ---- 명령 ----
    def send(self, ws=None, ws_w=None, ad="", ad_w=0.0, qe="", qe_w=0.0, rf="", rf_w=0.0, fire=False):
        if ws is None:
            # keep_active>0이면 STOP 대신 저속 W — 시뮬이 정지 시 /get_action을 throttle하는 걸 회피.
            ws, ws_w = ("W", self.keep_active_w) if self.keep_active_w > 0.0 else ("STOP", 1.0)
        cmd = {
            "moveWS": {"command": ws, "weight": float(ws_w if ws_w is not None else 1.0)},
            "moveAD": {"command": ad, "weight": float(ad_w)},
            "turretQE": {"command": qe, "weight": float(qe_w)},
            "turretRF": {"command": rf, "weight": float(rf_w)},
            "fire": bool(fire),
        }
        msg = String()
        msg.data = json.dumps(cmd, ensure_ascii=False, separators=(",", ":"))
        self.pub_cmd.publish(msg)

    def hold(self):
        self.send()  # STOP, 포탑 정지

    def wait_feedback(self, timeout=15.0) -> bool:
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.turret_yaw is not None:
                return True
            time.sleep(0.1)
        return False

    # ---- 진단: 단일 명령 poke ----
    def poke(self, cmd, w, sec):
        """한 명령(Q/E/R/F)을 sec초 sustained로 주고 각 변화를 출력 — 어느 명령이 실제로 움직이나 진단."""
        if not self.wait_feedback():
            self.get_logger().error("포탑 피드백 없음"); return
        is_yaw = cmd in ("Q", "E")
        getter = (lambda: self.turret_yaw) if is_yaw else (lambda: self.turret_pitch)
        self.hold(); time.sleep(0.5)
        a0 = getter()
        t0 = time.time()
        while time.time() - t0 < sec:
            if is_yaw:
                self.send(qe=cmd, qe_w=w)
            else:
                self.send(rf=cmd, rf_w=w)
            time.sleep(0.05)
        self.hold(); time.sleep(0.4)
        a1 = getter()
        self.get_logger().info(
            f"[poke] {cmd} w={w} {sec}s: {a0:.1f} -> {a1:.1f}  (Δ{norm180(a1 - a0):+.1f}deg)  "
            f"{'움직임 OK' if abs(norm180(a1 - a0)) > 1.0 else '★안 움직임'}")

    # ---- 실험 1: 포탑 동역학 ----
    def run_dynamics(self, weights, dur=2.0):
        rows = []
        for axis, cmds, getter in (
            ("yaw", ("Q", "E"), lambda: self.turret_yaw),
            ("pitch", ("R", "F"), lambda: self.turret_pitch),
        ):
            for cmd in cmds:
                for w in weights:
                    self.hold()
                    time.sleep(0.6)
                    a0 = getter()
                    samples = []
                    t0 = time.time()
                    while time.time() - t0 < dur:
                        if axis == "yaw":
                            self.send(qe=cmd, qe_w=w)
                        else:
                            self.send(rf=cmd, rf_w=w)
                        time.sleep(0.05)
                        samples.append((time.time() - t0, getter()))
                    self.hold()
                    rate, moved = self._rate_from_samples(samples)
                    rows.append({
                        "axis": axis, "cmd": cmd, "weight": round(w, 2),
                        "deg_per_s": round(rate, 2), "moved_deg": round(moved, 1),
                        "start_deg": round(a0 or 0.0, 1), "samples": len(samples),
                    })
                    self.get_logger().info(f"[dyn] {axis} {cmd} w={w}: {rate:.1f} deg/s (moved {moved:.1f})")
        self._write_csv("turret_dynamics.csv", rows,
                        ["axis", "cmd", "weight", "deg_per_s", "moved_deg", "start_deg", "samples"])

    @staticmethod
    def _rate_from_samples(samples):
        """언랩한 각으로 최소제곱 기울기(deg/s) + 총 이동량. 언랩=연속 raw차의 norm180 누적."""
        pts = [(t, a) for t, a in samples if a is not None]
        if len(pts) < 3:
            return 0.0, 0.0
        ts = [t for t, _ in pts]
        raws = [a for _, a in pts]
        unwrapped = [raws[0]]
        for i in range(1, len(raws)):
            unwrapped.append(unwrapped[-1] + norm180(raws[i] - raws[i - 1]))
        n = len(ts)
        mt = sum(ts) / n
        ma = sum(unwrapped) / n
        den = sum((t - mt) ** 2 for t in ts) or 1e-9
        slope = sum((t - mt) * (a - ma) for t, a in zip(ts, unwrapped)) / den
        return slope, unwrapped[-1] - unwrapped[0]

    # ---- 포탑 폐루프 조준 ----
    def aim_to(self, yaw, pitch, tol=1.2, max_w=0.5, timeout=10.0) -> bool:
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.turret_yaw is None:
                time.sleep(0.05); continue
            ey = norm180(yaw - self.turret_yaw)
            ep = norm180(pitch - self.turret_pitch)
            if abs(ey) < tol and abs(ep) < tol:
                self.hold(); return True
            qe = "E" if (ey * self.qe_sign) > 0 else "Q"
            rf = "R" if (ep * self.rf_sign) > 0 else "F"
            qew = min(max_w, abs(ey) / 30.0 * max_w + 0.12) if abs(ey) >= tol else 0.0
            rfw = min(max_w, abs(ep) / 30.0 * max_w + 0.12) if abs(ep) >= tol else 0.0
            self.send(qe=(qe if qew > 0 else ""), qe_w=qew, rf=(rf if rfw > 0 else ""), rf_w=rfw)
            time.sleep(0.05)
        self.hold(); return False

    def fire_once(self):
        # fire=true를 충분히(~1초) 유지해야 시뮬 /get_action 폴링(~0.5초 간격)이 확실히 잡는다.
        # 0.15초면 폴링 사이에 끼어 자주 놓쳐 발사가 들쭉날쭉. 쿨다운이 중복발사를 막아준다.
        self._impact = None
        self._fire_t = time.time()
        self.send(fire=True)
        time.sleep(1.0)
        self.send(fire=False)

    def wait_impact(self, timeout=5.0):
        # 발사 시작(self._fire_t) 이후의 탄착이면 채택 — fire 유지(1초) 중 착탄도 포착.
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self._impact is not None and self._impact[3] >= self._fire_t - 0.2:
                return self._impact, self._impact_target
            time.sleep(0.05)
        return None, ""

    # ---- 재장전(쿨다운) 측정 ----
    def reload_probe(self, duration=180.0, fire_every=3.0):
        """fire를 주기적으로 쏘고 명중 시각 간격을 본다 = 재장전 쿨다운(추측 말고 측정)."""
        if not self.wait_feedback():
            self.get_logger().error("피드백 없음"); return
        impacts = []
        t0 = time.time()
        last_fire = 0.0
        self.get_logger().info(f"재장전 측정 ({duration:.0f}s, {fire_every:.0f}s마다 발사 시도) — 전방 개활지 향하게")
        while time.time() - t0 < duration:
            now = time.time()
            if now - last_fire >= fire_every:
                self.fire_once(); last_fire = now
            imp = self._impact
            if imp is not None and (not impacts or imp[3] > impacts[-1] + 0.5):
                impacts.append(imp[3])
                gap = (impacts[-1] - impacts[-2]) if len(impacts) >= 2 else None
                self.get_logger().info(f"  명중 {len(impacts)}: +{imp[3]-t0:.1f}s" + (f" (직전과 {gap:.1f}s)" if gap else ""))
            time.sleep(0.2)
        self.hold()
        if len(impacts) >= 2:
            gaps = sorted(impacts[i] - impacts[i-1] for i in range(1, len(impacts)))
            self.get_logger().info(f"=== 재장전 간격: {[round(g,1) for g in gaps]}s → 최대 {gaps[-1]:.0f}s ===")
            self.get_logger().info(f"  → 사격 실험: --reload-sec={int(gaps[-1])+2}")
        else:
            self.get_logger().warn(f"명중 {len(impacts)}건뿐 — 전방 막혔거나 duration 짧음")

    # ---- 지형 tour: 다음 spot으로 짧게 블라인드 이동(자세 변화 유도) ----
    def reposition(self, leg: int, move_sec=2.5, ws_w=0.25):
        """장애물 회피 없는 짧은 이동 — 개활지에서만, 다리 짧게. 같은 자리 반복 방지로 좌/우 교대 선회."""
        turn = "D" if (leg % 2) else "A"
        t0 = time.time()
        while time.time() - t0 < move_sec:
            self.send(ws="W", ws_w=ws_w, ad=turn, ad_w=0.4)
            time.sleep(0.05)
        self.hold()
        time.sleep(0.8)  # 정착(차체 자세 안정)

    # ---- 실험 2/3: 정지 사격 탄도(차체 자세 함께 기록 = 지형 겸용) ----
    def run_fire(self, pitch_grid, yaw_offsets, settle=0.6, leg=0):
        """현재 spot에서 사격 sweep. rows 반환(여러 지형 tour를 누적 후 한 번에 기록)."""
        rows = []
        if not self.wait_feedback():
            self.get_logger().error("포탑 피드백 없음 — 브릿지/시뮬 확인")
            return rows
        # 조준 기준 yaw: --fixed-yaw > --aim-enemy(적 방위) > 전방(차체 yaw, 기본). 탄도는 전방이 안전.
        base_yaw = self.turret_yaw or 0.0
        dist = None
        if self.fixed_yaw is not None:
            base_yaw = float(self.fixed_yaw)
            self.get_logger().info(f"[fire] fixed yaw={base_yaw:.1f}deg")
        elif self.aim_enemy and self.player_xy and self.enemy_xy:
            dx = self.enemy_xy[0] - self.player_xy[0]
            dy = self.enemy_xy[1] - self.player_xy[1]
            base_yaw = math.degrees(math.atan2(dx, dy))
            dist = math.hypot(dx, dy)
            self.get_logger().info(f"[fire] 적 방위={base_yaw:.1f}deg dist={dist:.1f}m (적이 뒤면 포탑 뒤로 돎)")
        elif self.body_yaw is not None:
            base_yaw = self.body_yaw
            self.get_logger().info(f"[fire] 전방(차체) yaw={base_yaw:.1f}deg — 탄도 측정용(개활지 전방으로 사격)")
        for dyaw in yaw_offsets:
            for p in pitch_grid:
                tgt_yaw = norm180(base_yaw + dyaw)
                ok = self.aim_to(tgt_yaw, p)
                time.sleep(settle)
                # 재장전 대기: 직전 발사 후 reload_sec 경과까지 기다린 뒤 1발(쿨다운 중 헛발사 방지).
                wait = self.reload_sec - (time.time() - self._last_fire_wall)
                if wait > 0:
                    self.get_logger().info(f"  (재장전 대기 {wait:.0f}s)")
                    time.sleep(wait)
                self.fire_once()
                self._last_fire_wall = time.time()
                impact, hit = self.wait_impact(self.impact_timeout)
                attempts = 1
                d2e = None
                if impact and self.enemy_xy:
                    d2e = math.hypot(impact[0] - self.enemy_xy[0], impact[2] - self.enemy_xy[1])
                rows.append({
                    "leg": leg,
                    "tgt_yaw": round(tgt_yaw, 1), "tgt_pitch": round(p, 1),
                    "turret_yaw": round(self.turret_yaw or 0.0, 1), "turret_pitch": round(self.turret_pitch or 0.0, 1),
                    "aimed_ok": ok, "attempts": attempts,
                    "body_pitch": round(self.body_pitch, 2), "body_roll": round(self.body_roll, 2),
                    "dist_to_enemy_m": round(dist, 1) if dist else "",
                    "impact_x": round(impact[0], 2) if impact else "",
                    "impact_y": round(impact[1], 2) if impact else "",
                    "impact_z": round(impact[2], 2) if impact else "",
                    "hit": hit,
                    "impact_to_enemy_m": round(d2e, 2) if d2e is not None else "",
                })
                self.get_logger().info(
                    f"[fire] yaw={tgt_yaw:.0f} pitch={p:.0f} -> hit={hit} miss={d2e if d2e is not None else '?'}")
        self.hold()
        return rows

    FIRE_FIELDS = [
        "leg", "tgt_yaw", "tgt_pitch", "turret_yaw", "turret_pitch", "aimed_ok", "attempts",
        "body_pitch", "body_roll", "dist_to_enemy_m",
        "impact_x", "impact_y", "impact_z", "hit", "impact_to_enemy_m"]

    def _write_csv(self, name, rows, fields):
        os.makedirs(OUT_DIR, exist_ok=True)
        path = os.path.join(OUT_DIR, name)
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        self.get_logger().info(f"[완료] 기록: {path} ({len(rows)} rows)")


def main():
    ap = argparse.ArgumentParser(description="포탑·사격 자동 실험")
    ap.add_argument("--phase", choices=["dynamics", "fire", "both"], default="dynamics")
    ap.add_argument("--weights", default="0.2,0.4,0.6,0.9")
    ap.add_argument("--pitch-grid", default="0,2,4,6,8,10")   # 포각 가동범위(~-5~+10°) 내. 음수는 =로: --pitch-grid=-3,0,..
    ap.add_argument("--yaw-offsets", default="0")             # 탄도(pitch→사거리)엔 한 방향이면 충분
    ap.add_argument("--qe-sign", type=int, default=1, help="E가 yaw 증가면 +1(동역학 결과로 확정)")
    ap.add_argument("--rf-sign", type=int, default=1, help="R이 pitch 증가면 +1(동역학 결과로 확정)")
    ap.add_argument("--terrain-tour", type=int, default=1,
                    help="지형 자동 tour 횟수(>1이면 사격 sweep 사이에 짧게 블라인드 이동해 자세를 바꿈; 개활지 권장)")
    ap.add_argument("--tour-move-sec", type=float, default=2.5, help="tour 1회 이동 시간(초)")
    ap.add_argument("--poke", default="", help="진단: 단일 명령 'E:0.9:3' = E를 weight0.9로 3초 (시뮬 화면 보며 움직이나 확인)")
    ap.add_argument("--reload-probe", type=float, default=0.0, help="재장전 측정 모드: 이 초만큼 주기 발사해 쿨다운 측정(예 180)")
    ap.add_argument("--keep-active", type=float, default=0.0,
                    help="STOP 대신 저속 W(예 0.15)로 시뮬을 깨워둠 — 정지 시 /get_action throttle 회피. ★개활지에서만(전차가 천천히 전진)")
    ap.add_argument("--fixed-yaw", type=float, default=None,
                    help="사격 조준 yaw 고정(특정 방향). 없으면 전방(차체)")
    ap.add_argument("--aim-enemy", action="store_true",
                    help="적전차 방위로 조준(기본은 전방). 적이 뒤에 있으면 포탑이 뒤로 돌아 맵 밖 사격 — 탄도엔 부적합")
    ap.add_argument("--reload-sec", type=float, default=6.0, help="재장전 대기(탄착 안 오면 이만큼 쉬고 재발사)")
    ap.add_argument("--fire-attempts", type=int, default=3, help="발당 최대 재발사 시도(재장전 처리)")
    ap.add_argument("--impact-timeout", type=float, default=10.0, help="탄착 대기(초; 원거리 비행시간 포함)")
    args = ap.parse_args()

    rclpy.init()
    node = TurretExperiment(args.qe_sign, args.rf_sign)
    node.keep_active_w = max(0.0, float(args.keep_active))
    node.fixed_yaw = args.fixed_yaw
    node.aim_enemy = bool(args.aim_enemy)
    node.fire_attempts = max(1, int(args.fire_attempts))
    node.reload_sec = max(0.0, float(args.reload_sec))
    node.impact_timeout = max(1.0, float(args.impact_timeout))
    _stop = threading.Event()

    def _spin():
        while rclpy.ok() and not _stop.is_set():
            rclpy.spin_once(node, timeout_sec=0.1)

    spin = threading.Thread(target=_spin, daemon=True)
    spin.start()
    try:
        if not node.wait_feedback():
            node.get_logger().error("포탑 피드백(/tank/api/get_action/turret) 없음 — 브릿지 auto + 시뮬 + (recon 컨트롤러 끔) 확인")
            return
        if args.poke:
            p = args.poke.split(":")
            node.poke(p[0].strip(), float(p[1]), float(p[2]) if len(p) > 2 else 3.0)
            return
        if args.reload_probe > 0:
            node.reload_probe(duration=args.reload_probe)
            return
        weights = [float(x) for x in args.weights.split(",") if x.strip()]
        if args.phase in ("dynamics", "both"):
            node.get_logger().info("=== 실험1: 포탑 동역학 ===")
            node.run_dynamics(weights)
        if args.phase in ("fire", "both"):
            node.get_logger().info("=== 실험2/3: 정지 사격 탄도(차체 자세=지형 겸용) ===")
            pitch_grid = [float(x) for x in args.pitch_grid.split(",")]
            yaw_offsets = [float(x) for x in args.yaw_offsets.split(",")]
            all_rows = []
            for leg in range(max(1, args.terrain_tour)):
                if leg > 0:
                    node.get_logger().info(f"=== 지형 tour: 재배치 leg {leg} (블라인드 짧은 이동) ===")
                    node.reposition(leg, args.tour_move_sec)
                all_rows.extend(node.run_fire(pitch_grid, yaw_offsets, leg=leg))
            node._write_csv("fire_ballistics.csv", all_rows, TurretExperiment.FIRE_FIELDS)
    finally:
        try:
            node.hold()
            time.sleep(0.2)
        except Exception:
            pass
        _stop.set()                 # spin 스레드 먼저 멈춤(종료 크래시·좀비 방지)
        spin.join(timeout=1.5)
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
