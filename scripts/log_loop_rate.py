#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""시뮬 제어 루프 속도 기록·요약 — 정찰(또는 임의) 주행 중 켜두고 끝에 요약을 본다.

`/tank/api/get_action/turret` 도착 간격으로 **/get_action 폴링 Hz**(=시뮬 제어 루프 속도=포탑/주행 응답성)를
재고, `/tank/api/info/compact`(센서 스트림)와 비교한다. 포탑 각도 궤적도 기록해 **정찰 중 포탑이 실제로
움직였는지(step-stare 작동)** 까지 사후 확인.

용도: "detect 켠 채로 포탑이 빠릿하게 도나?"를 라이브로 안 보고 기록으로 판정.

실행 (정찰 풀스택 + 시뮬이 도는 동안, 새 터미널):
  source install/setup.bash
  python3 scripts/log_loop_rate.py                 # Ctrl+C로 종료 시 요약 + CSV
  python3 scripts/log_loop_rate.py --duration 120  # 120초 자동 종료

산출: recon_reports/experiments/loop_rate.csv + 콘솔 요약(평균 Hz, 간격 median/p90/max, 포탑 가동범위)
"""
from __future__ import annotations

import argparse
import csv
import os
import statistics
import threading
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Vector3Stamped

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "recon_reports", "experiments")


class LoopRateLogger(Node):
    def __init__(self):
        super().__init__("loop_rate_logger")
        self.events = []          # (wall_t, topic, yaw, pitch)
        self._t0 = time.time()
        self._last_report = self._t0
        self.create_subscription(Vector3Stamped, "/tank/api/get_action/turret", self._turret_cb, 50)
        self.create_subscription(String, "/tank/api/info/compact", self._info_cb, 50)

    def _turret_cb(self, m: Vector3Stamped):
        self.events.append((time.time(), "get_action", float(m.vector.x), float(m.vector.y)))
        self._maybe_report()

    def _info_cb(self, m: String):
        self.events.append((time.time(), "info", None, None))
        self._maybe_report()

    def _maybe_report(self):
        now = time.time()
        if now - self._last_report >= 5.0:
            self._last_report = now
            ga = [e for e in self.events if e[1] == "get_action"]
            inf = [e for e in self.events if e[1] == "info"]
            self.get_logger().info(
                f"[{now - self._t0:5.0f}s] get_action {len(ga)}건(~{self._hz(ga):.2f}Hz) · "
                f"info {len(inf)}건(~{self._hz(inf):.2f}Hz)")

    @staticmethod
    def _hz(evts):
        if len(evts) < 2:
            return 0.0
        dur = evts[-1][0] - evts[0][0]
        return (len(evts) - 1) / dur if dur > 0 else 0.0

    def summarize_and_write(self):
        os.makedirs(OUT_DIR, exist_ok=True)
        path = os.path.join(OUT_DIR, "loop_rate.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["t_rel", "topic", "turret_yaw", "turret_pitch"])
            for t, top, y, p in self.events:
                w.writerow([round(t - self._t0, 3), top, "" if y is None else round(y, 2),
                            "" if p is None else round(p, 2)])
        for top in ("get_action", "info"):
            evts = [e for e in self.events if e[1] == top]
            if len(evts) < 2:
                print(f"  {top}: {len(evts)}건 (부족)")
                continue
            gaps = sorted(evts[i][0] - evts[i - 1][0] for i in range(1, len(evts)))
            print(f"  {top:11s}: {len(evts):5d}건 · 평균 {self._hz(evts):6.2f}Hz · "
                  f"간격 median {statistics.median(gaps):5.2f}s p90 {gaps[int(len(gaps) * 0.9)]:5.2f}s "
                  f"max {gaps[-1]:6.2f}s")
        # 포탑이 실제로 움직였나(가동 범위) — step-stare 작동 확인
        yaws = [y for _, top, y, _ in self.events if top == "get_action" and y is not None]
        if yaws:
            print(f"  포탑 yaw 범위: {min(yaws):.0f}~{max(yaws):.0f}° (가동폭 {max(yaws) - min(yaws):.0f}°) "
                  f"— 폭이 크면 정찰 중 포탑이 실제로 움직인 것")
        print(f"  CSV: {path}")


def main():
    ap = argparse.ArgumentParser(description="시뮬 제어 루프 속도 기록")
    ap.add_argument("--duration", type=float, default=0.0, help="초; 0=Ctrl+C까지")
    args = ap.parse_args()

    rclpy.init()
    node = LoopRateLogger()
    stop = threading.Event()

    def spin():
        while rclpy.ok() and not stop.is_set():
            rclpy.spin_once(node, timeout_sec=0.1)

    th = threading.Thread(target=spin, daemon=True)
    th.start()
    print("기록 시작 — 정찰/시뮬 돌리는 동안 두세요. (Ctrl+C 종료 시 요약+CSV)")
    try:
        if args.duration > 0:
            time.sleep(args.duration)
        else:
            while True:
                time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        th.join(timeout=1.5)
        print("\n=== 요약 ===")
        try:
            node.summarize_and_write()
        except Exception as exc:
            print(f"요약 실패: {exc}")
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
