# -*- coding: utf-8 -*-
"""시나리오2 전술 decision FSM — 돌파(FORWARD/ENGAGE) ↔ 복귀(RETURN).

SCENARIO2_DESIGN §7 의 임무 FSM을 구현한다. **전술층**(이벤트 단위, ~수초)만 담당하고
조향·충돌회피(APF/A*/제어)는 반사층(10Hz)이 그대로 지킨다(§4). 이 노드는 100ms 제어 루프에
절대 끼어들지 않는다(틱 2Hz).

상태:
  FORWARD — A 루트 주행(planner/APF가 운전). 표적 분류·위험도 평가를 돌린다.
  ENGAGE  — known 적전차(또는 목적지 적전차)가 사거리+LoS 안 → 교전 요청 발행, 결과 대기.
  RETURN  — 정찰에 없던 '새 적전차'의 위험도가 임계 초과 → goal=출발지 발행(임무 중단).
            planner가 /tank/goal/pose 를 구독→재계획하므로 출발지로 돌아간다(종단 상태).

표적 분류(§3): scenario2_map.map 의 targets[](정찰 known-tank) 와 주행 중 탐지
(/tank/map/discovered/objects 의 class=tank, + /tank/enemy 목적지 적전차)를 **map 좌표로 매칭**.
  - known(매칭됨) + 목적지 적전차 → 교전 후보.
  - 매칭 실패(새 적전차) → 위험도 평가 후보.
known-tank가 비어 있어도(§12.2 #1) 정상 동작한다 — 모든 탐지가 'new'로 분류돼 위험도 경로만 돈다.

좌표: targets[].map_x/map_y, discovered objects 의 map_x/map_y, pose.position.x/y 는 모두
map 평면좌표(x=map.x, y=map.y). position.y(=Unity raw map.z)는 평면 수식에 쓰지 않는다.
"""

from __future__ import annotations

import json
import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String

from mission import risk
from mission.engage_contract import (
    TOPIC_ENGAGE_REQUEST,
    TOPIC_ENGAGE_RESULT,
    make_engage_request,
    parse_engage_result,
)

# 소비/발행 토픽 (이 repo 표준명)
TOPIC_PLAYER_POSE = "/tank/player/pose"
TOPIC_ENEMY_POSE = "/tank/enemy/pose"
TOPIC_ENEMY_STATE = "/tank/enemy/state"
TOPIC_DISCOVERED = "/tank/map/discovered/objects"
TOPIC_GOAL_POSE = "/tank/goal/pose"
TOPIC_DECISION_STATUS = "/tank/decision/status"

STATE_FORWARD = "FORWARD"
STATE_ENGAGE = "ENGAGE"
STATE_RETURN = "RETURN"


class DecisionNode(Node):
    def __init__(self) -> None:
        super().__init__("tank_decision_node")

        # --- 파라미터 ---
        self.declare_parameter("scenario2_map_file", "")
        self.declare_parameter("start_x", 59.0)   # routes.yaml finalmap.start
        self.declare_parameter("start_y", 27.0)
        self.declare_parameter("goal_tolerance", 10.0)        # ★ 전 노드 일치 필수
        self.declare_parameter("known_match_radius_m", 8.0)   # known/new 매칭 반경(map좌표)
        self.declare_parameter("engage_range_m", 20.0)        # Tank001 반경과 정합
        self.declare_parameter("risk_radius_m", 20.0)
        self.declare_parameter("risk_threshold", 0.5)         # 0~1; 초과 시 복귀
        self.declare_parameter("engage_cooldown_sec", 5.0)
        self.declare_parameter("engage_timeout_sec", 8.0)     # 결과 미수신 시 ENGAGE 해제
        self.declare_parameter("tick_hz", 2.0)
        self.declare_parameter("min_confidence", 0.0)         # 탐지 신뢰도 하한(0=무시)

        self.map_file = str(self.get_parameter("scenario2_map_file").value)
        self.start_x = float(self.get_parameter("start_x").value)
        self.start_y = float(self.get_parameter("start_y").value)
        self.goal_tolerance = float(self.get_parameter("goal_tolerance").value)
        self.known_match_radius_m = float(self.get_parameter("known_match_radius_m").value)
        self.engage_range_m = float(self.get_parameter("engage_range_m").value)
        self.risk_radius_m = float(self.get_parameter("risk_radius_m").value)
        self.risk_threshold = float(self.get_parameter("risk_threshold").value)
        self.engage_cooldown_sec = float(self.get_parameter("engage_cooldown_sec").value)
        self.engage_timeout_sec = float(self.get_parameter("engage_timeout_sec").value)
        self.tick_hz = max(0.5, float(self.get_parameter("tick_hz").value))
        self.min_confidence = float(self.get_parameter("min_confidence").value)

        # --- 맵 로드: known-tank 목록 + LoS bbox ---
        self.known_tanks: List[Tuple[float, float, str]] = []   # (map_x, map_y, prefab)
        self.gt_bboxes: List[Dict[str, float]] = []
        self._load_scenario2_map()
        if not risk.los_available():
            self.get_logger().warn(
                "threat_geometry(LoS 기하)를 못 찾음 — 위험도/교전을 거리만으로 평가합니다 "
                "(LoS 보수적 True). TANK_PROJECT_ROOT 또는 scripts/recon_eval 경로를 확인하세요.")

        # --- 상태 ---
        self.player_xy: Optional[Tuple[float, float]] = None
        self.enemy_xy: Optional[Tuple[float, float]] = None
        self.enemy_health: Optional[float] = None
        self.detected_tanks: List[Dict[str, Any]] = []   # discovered objects의 class=tank
        self.killed: set = set()                          # 격파 처리된 target_id
        self.state = STATE_FORWARD
        self.last_reason = "init"
        self.last_engage_request_wall = 0.0
        self.engage_target_id: Optional[str] = None
        self.engage_started_wall = 0.0
        self.last_risk = 0.0

        # --- pub/sub ---
        self.engage_pub = self.create_publisher(String, TOPIC_ENGAGE_REQUEST, 10)
        self.goal_pub = self.create_publisher(PoseStamped, TOPIC_GOAL_POSE, 10)
        self.status_pub = self.create_publisher(String, TOPIC_DECISION_STATUS, 10)

        self.create_subscription(PoseStamped, TOPIC_PLAYER_POSE, self._player_cb, 10)
        self.create_subscription(PoseStamped, TOPIC_ENEMY_POSE, self._enemy_pose_cb, 10)
        self.create_subscription(String, TOPIC_ENEMY_STATE, self._enemy_state_cb, 10)
        self.create_subscription(String, TOPIC_DISCOVERED, self._discovered_cb, 10)
        self.create_subscription(String, TOPIC_ENGAGE_RESULT, self._engage_result_cb, 10)

        self.create_timer(1.0 / self.tick_hz, self._tick)

        self.get_logger().info(
            f"DecisionNode 시작: known_tanks={len(self.known_tanks)}, gt_bboxes={len(self.gt_bboxes)}, "
            f"start=({self.start_x},{self.start_y}), engage_range={self.engage_range_m}m, "
            f"risk_threshold={self.risk_threshold}, los={risk.los_available()}")

    # ------------------------------------------------------------------ #
    # 맵 로드
    # ------------------------------------------------------------------ #
    def _load_scenario2_map(self) -> None:
        if not self.map_file or not os.path.isfile(self.map_file):
            self.get_logger().warn(
                f"scenario2_map 파일 없음/미지정: '{self.map_file}' — known=0, LoS bbox=0로 진행")
            return
        try:
            with open(self.map_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            self.get_logger().warn(f"scenario2_map 로드 실패({exc}) — known=0로 진행")
            return
        for t in data.get("targets", []) or []:
            try:
                mx = float(t.get("map_x"))
                my = float(t.get("map_y"))
            except (TypeError, ValueError):
                continue
            self.known_tanks.append((mx, my, str(t.get("prefabName", ""))))
        self.gt_bboxes = risk.bboxes_from_map(data)
        if not self.known_tanks:
            # §12.2 #1: 제대로 된 정찰 run 전엔 known이 빈약/0일 수 있음 — 정상 동작.
            self.get_logger().warn(
                "known-tank 0개 — 모든 탐지를 'new'로 분류(위험도 경로만). "
                "정찰 시뮬 run 후 targets[]가 채워지면 교전(ENGAGE)이 활성화됩니다.")

    # ------------------------------------------------------------------ #
    # 콜백
    # ------------------------------------------------------------------ #
    def _player_cb(self, msg: PoseStamped) -> None:
        self.player_xy = (float(msg.pose.position.x), float(msg.pose.position.y))

    def _enemy_pose_cb(self, msg: PoseStamped) -> None:
        self.enemy_xy = (float(msg.pose.position.x), float(msg.pose.position.y))

    def _enemy_state_cb(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
            if isinstance(payload, dict) and payload.get("health") is not None:
                self.enemy_health = float(payload.get("health"))
        except Exception:
            pass

    def _discovered_cb(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except Exception:
            return
        objs = payload.get("objects") if isinstance(payload, dict) else None
        if not isinstance(objs, list):
            return
        tanks: List[Dict[str, Any]] = []
        for o in objs:
            if not isinstance(o, dict):
                continue
            if str(o.get("class_name", "")).lower() != "tank":
                continue
            if o.get("map_x") is None or o.get("map_y") is None:
                continue
            try:
                conf = float(o.get("confidence", 0.0))
            except (TypeError, ValueError):
                conf = 0.0
            if conf < self.min_confidence:
                continue
            try:
                tanks.append({
                    "object_id": str(o.get("object_id", "")),
                    "xy": (float(o.get("map_x")), float(o.get("map_y"))),
                    "confidence": conf,
                })
            except (TypeError, ValueError):
                continue
        self.detected_tanks = tanks

    def _engage_result_cb(self, msg: String) -> None:
        res = parse_engage_result(msg.data)
        if res is None:
            return
        tid = res["target_id"]
        if res["success"]:
            self.killed.add(tid)
            self.get_logger().info(
                f"교전 성공 처리: {tid} (impact {res['dist_to_target_m']:.1f}m) → 격파 등록")
            if self.state == STATE_ENGAGE and self.engage_target_id == tid:
                self._set_state(STATE_FORWARD, f"engaged:{tid}")
                self.engage_target_id = None

    # ------------------------------------------------------------------ #
    # 분류 헬퍼
    # ------------------------------------------------------------------ #
    def _match_known(self, x: float, y: float) -> Optional[str]:
        """탐지 좌표를 known-tank 목록에 최근접+반경으로 매칭(local_path _find_existing_discovered 패턴)."""
        best = None
        best_d = 1e9
        for (mx, my, prefab) in self.known_tanks:
            d = math.hypot(mx - x, my - y)
            if d < best_d:
                best_d = d
                best = prefab
        if best is not None and best_d <= self.known_match_radius_m:
            return best
        return None

    def _engage_candidates(self) -> List[Dict[str, Any]]:
        """교전 후보 = known으로 매칭된 탐지전차 + 목적지 적전차(/tank/enemy)."""
        cands: List[Dict[str, Any]] = []
        for t in self.detected_tanks:
            prefab = self._match_known(t["xy"][0], t["xy"][1])
            if prefab is not None:
                cands.append({"target_id": prefab, "xy": t["xy"]})
        if self.enemy_xy is not None:
            cands.append({"target_id": "enemy_main", "xy": self.enemy_xy})
        return cands

    def _new_tanks(self) -> List[Tuple[float, float]]:
        """새 적전차(=known 미매칭, 목적지 적전차도 아님) 좌표 목록 → 위험도 평가 대상."""
        out: List[Tuple[float, float]] = []
        for t in self.detected_tanks:
            if self._match_known(t["xy"][0], t["xy"][1]) is not None:
                continue
            if self.enemy_xy is not None and math.hypot(
                    t["xy"][0] - self.enemy_xy[0], t["xy"][1] - self.enemy_xy[1]) <= self.known_match_radius_m:
                continue  # 목적지 적전차와 같은 객체로 보고 제외
            out.append(t["xy"])
        return out

    # ------------------------------------------------------------------ #
    # FSM 틱
    # ------------------------------------------------------------------ #
    def _tick(self) -> None:
        if self.player_xy is None:
            self._publish_status("no_player_pose")
            return

        # RETURN은 종단 — 출발지 도달까지 매 틱 goal 재발행(planner 2Hz 자기 republish 덮어쓰기).
        if self.state == STATE_RETURN:
            self._publish_return_goal()
            self._publish_status(self.last_reason)
            return

        # 1) 새 적전차 위험도 — 복귀가 교전보다 우선(임무 중단).
        new_tanks = self._new_tanks()
        score, worst_xy = risk.worst_risk(
            self.player_xy, new_tanks, self.gt_bboxes,
            radius_m=self.risk_radius_m)
        self.last_risk = score
        if score > self.risk_threshold:
            reason = (f"new_tank_risk={score:.2f}>{self.risk_threshold:.2f}"
                      f"@({worst_xy[0]:.0f},{worst_xy[1]:.0f})" if worst_xy else f"risk={score:.2f}")
            self._set_state(STATE_RETURN, reason)
            self._publish_return_goal()
            self._publish_status(reason)
            return

        # 2) 교전: known/목적지 적전차가 사거리+LoS 안이면 engage 요청.
        target = self._pick_engage_target()
        now = time.monotonic()
        if target is not None:
            tid = target["target_id"]
            dist = math.hypot(self.player_xy[0] - target["xy"][0], self.player_xy[1] - target["xy"][1])
            los = risk.check_los(self.player_xy, target["xy"], self.gt_bboxes)
            if now - self.last_engage_request_wall >= self.engage_cooldown_sec:
                self.engage_pub.publish(String(data=make_engage_request(
                    tid, target["xy"][0], target["xy"][1], dist, los)))
                self.last_engage_request_wall = now
                self.engage_target_id = tid
                self.engage_started_wall = now
                self._set_state(STATE_ENGAGE, f"engage:{tid} d={dist:.1f} los={los}")
            self._publish_status(self.last_reason)
            return

        # 3) ENGAGE 중인데 후보가 사라졌거나 타임아웃 → FORWARD 복귀.
        if self.state == STATE_ENGAGE:
            if now - self.engage_started_wall >= self.engage_timeout_sec:
                self.engage_target_id = None
                self._set_state(STATE_FORWARD, "engage_timeout")
            self._publish_status(self.last_reason)
            return

        # 4) 평시 전진.
        if self.state != STATE_FORWARD:
            self._set_state(STATE_FORWARD, "clear")
        self._publish_status(f"forward risk={score:.2f} new={len(new_tanks)}")

    def _pick_engage_target(self) -> Optional[Dict[str, Any]]:
        """사거리+LoS 안의 미격파 교전 후보 중 가장 가까운 것."""
        best = None
        best_d = 1e9
        for c in self._engage_candidates():
            if c["target_id"] in self.killed:
                continue
            d = math.hypot(self.player_xy[0] - c["xy"][0], self.player_xy[1] - c["xy"][1])
            if d > self.engage_range_m:
                continue
            if not risk.check_los(self.player_xy, c["xy"], self.gt_bboxes):
                continue
            if d < best_d:
                best_d = d
                best = c
        return best

    # ------------------------------------------------------------------ #
    # 발행 헬퍼
    # ------------------------------------------------------------------ #
    def _publish_return_goal(self) -> None:
        msg = PoseStamped()
        msg.header.frame_id = "tank_map"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x = self.start_x
        msg.pose.position.y = self.start_y
        msg.pose.orientation.w = 1.0
        self.goal_pub.publish(msg)

    def _set_state(self, state: str, reason: str) -> None:
        if state != self.state:
            self.get_logger().info(f"FSM {self.state} → {state} ({reason})")
        self.state = state
        self.last_reason = reason

    def _publish_status(self, reason: str) -> None:
        payload = {
            "state": self.state,
            "reason": reason,
            "risk": round(self.last_risk, 3),
            "engage_target": self.engage_target_id,
            "known_count": len(self.known_tanks),
            "killed_count": len(self.killed),
            "detected_tanks": len(self.detected_tanks),
            "wall": time.time(),
        }
        self.status_pub.publish(String(data=json.dumps(payload, ensure_ascii=False)))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DecisionNode()
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
