# -*- coding: utf-8 -*-
"""mock turret 노드 — 실제 포탑 조준/발사(팀원·control)의 stand-in.

SCENARIO2_DESIGN §5/§12.3 ④: 실제 turret 제어 전에도 교전 폐루프(decision→engage→result)를
오프라인 검증할 수 있게, `/tank/engage/request`를 구독해 '조준 지연' 후 `/tank/engage/result`를
발행한다. 실제 구현이 들어오면 같은 계약(engage_contract)을 구독/발행하도록 이 노드를 교체한다.

성공 판정(시나리오 책임): 정지표적 + 근접성공이라 정밀 탄도 불요 → always_hit이면 표적 위치를
임팩트로, 거리 0으로 성공 보고한다.
"""

from __future__ import annotations

import time
from typing import List, Tuple

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from mission.engage_contract import (
    TOPIC_ENGAGE_REQUEST,
    TOPIC_ENGAGE_RESULT,
    make_engage_result,
    parse_engage_request,
)


class MockTurretNode(Node):
    def __init__(self) -> None:
        super().__init__("tank_mock_turret_node")
        self.declare_parameter("aim_sec", 1.5)        # 조준 지연(발사까지)
        self.declare_parameter("always_hit", True)    # 골격 검증용 결정론적 성공
        self.declare_parameter("hit_radius_m", 3.0)   # 성공 인정 임팩트 반경(§5)
        self.aim_sec = max(0.0, float(self.get_parameter("aim_sec").value))
        self.always_hit = bool(self.get_parameter("always_hit").value)
        self.hit_radius_m = float(self.get_parameter("hit_radius_m").value)

        # 발사 예약 큐: (fire_at_monotonic, target_id, x, y)
        self._pending: List[Tuple[float, str, float, float]] = []

        self.result_pub = self.create_publisher(String, TOPIC_ENGAGE_RESULT, 10)
        self.create_subscription(String, TOPIC_ENGAGE_REQUEST, self._request_cb, 10)
        self.create_timer(0.1, self._tick)  # 10Hz로 발사 예약 점검

        self.get_logger().info(
            f"MockTurret 시작: aim={self.aim_sec}s, always_hit={self.always_hit}, "
            f"hit_radius={self.hit_radius_m}m")

    def _request_cb(self, msg: String) -> None:
        req = parse_engage_request(msg.data)
        if req is None:
            self.get_logger().warn("engage_request 파싱 실패 — 무시")
            return
        fire_at = time.monotonic() + self.aim_sec
        self._pending.append((fire_at, req["target_id"], req["x"], req["y"]))
        self.get_logger().info(
            f"교전 요청 수신: {req['target_id']} @({req['x']:.0f},{req['y']:.0f}) "
            f"d={req['distance_m']:.1f} los={req['los']} → {self.aim_sec:.1f}s 후 발사")

    def _tick(self) -> None:
        if not self._pending:
            return
        now = time.monotonic()
        still: List[Tuple[float, str, float, float]] = []
        for (fire_at, tid, x, y) in self._pending:
            if now < fire_at:
                still.append((fire_at, tid, x, y))
                continue
            # 발사: 정지표적 근접성공. always_hit이면 표적 위치=임팩트, 거리 0.
            jitter = 0.0
            success = self.always_hit or jitter <= self.hit_radius_m
            self.result_pub.publish(String(data=make_engage_result(
                tid, x + jitter, y, success, jitter)))
            self.get_logger().info(
                f"발사 → 임팩트 보고: {tid} success={success} dist={jitter:.1f}m")
        self._pending = still


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MockTurretNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
