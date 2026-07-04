#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""시나리오2 돌발 대응 노드 — 신규 위협 판단 + 안전 복귀 실행.

미션 주행 중 perception이 **정찰에서 몰랐던 신규 전차 위협**(known 미매칭 tank)을 감지하면
``sudden_decision`` 코어가 ENGAGE/BYPASS/RETURN을 판단하고 `/tank/decision/status`로 발행한다
(→ 브릿지 MFD 패널). LLM은 돌발 때만 비동기 **자문**을 제공하며, 실제 주행 결정을 덮어쓰지 않는다.

RETURN이 히스테리시스를 통과해 확정되면 이 노드는 다음을 함께 수행한다.

* `/tank/mission/abort`에 RETURN + 출발지 goal을 heartbeat로 발행한다.
  - ballistic_turret_node는 이를 받아 사격/체크포인트 FSM을 즉시 중단하고 차체 hold를 해제한다.
  - map_astar_planner_node는 이를 받아 기존 시나리오2 북진 waypoint를 거치지 않는 **직접 귀환 A***로 전환한다.
* `/tank/mission/goal_pose`에도 같은 출발지 goal을 주기 발행한다. 이는 planner 재시작/토픽 유실에 대한 보조 경로다.

복귀는 한 번 시작되면 탐지가 한 프레임 사라져도 취소하지 않는다. 교전(ENGAGE)은 기존처럼
표시만 한다. 즉, 의도적으로 실제 행동이 연결된 것은 안전 우선의 RETURN뿐이다.
"""
import json
import math
import time
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String

from mission import risk
from mission import sudden_decision as sd

TOPIC_PLAYER = "/tank/player/pose"
TOPIC_DISCOVERED = "/tank/map/discovered/objects"
TOPIC_STATUS = "/tank/decision/status"
TOPIC_MISSION_GOAL = "/tank/mission/goal_pose"
TOPIC_MISSION_ABORT = "/tank/mission/abort"


class SuddenAdvisorNode(Node):
    def __init__(self) -> None:
        super().__init__("sudden_advisor_node")
        self.declare_parameter("tick_hz", 2.0)
        self.declare_parameter("hysteresis_ticks", 2)   # 최근 N tick 동일해야 commit(churn 방지)
        self.declare_parameter("llm_cooldown_sec", 20.0)
        self.declare_parameter("known_match_radius_m", 8.0)
        self.declare_parameter("min_confidence", 0.0)
        self.declare_parameter("use_llm", True)
        # RETURN만 실제 실행한다. ENGAGE/BYPASS는 체크포인트 사격 시퀀스와
        # 충돌할 수 있으므로 여전히 상태 패널용 결정으로만 유지한다.
        self.declare_parameter("execute_return", True)
        # SCENARIO2_RETURN_ARM_GUARD_V1
        # Do not let detections already visible at spawn immediately turn the
        # home coordinate into both the active goal and the terminal STOP.
        # RETURN becomes executable only after the tank has departed home.
        self.declare_parameter("return_arm_min_distance_from_home_m", 35.0)
        self.declare_parameter("return_x", 59.0)
        self.declare_parameter("return_y", 27.0)
        self.declare_parameter("return_goal_topic", TOPIC_MISSION_GOAL)
        self.declare_parameter("mission_abort_topic", TOPIC_MISSION_ABORT)
        self.declare_parameter("return_republish_sec", 0.5)
        import os
        default_map = os.path.join(
            os.environ.get("TANK_PROJECT_ROOT", os.getcwd()),
            "recon_reports", "recon_map", "scenario2_map.map")
        self.declare_parameter("scenario2_map", default_map)

        self.hyst = max(1, int(self.get_parameter("hysteresis_ticks").value))
        self.llm_cooldown = float(self.get_parameter("llm_cooldown_sec").value)
        self.match_r = float(self.get_parameter("known_match_radius_m").value)
        self.min_conf = float(self.get_parameter("min_confidence").value)
        self.use_llm = bool(self.get_parameter("use_llm").value)
        self.execute_return = bool(self.get_parameter("execute_return").value)
        self.return_arm_min_distance_from_home_m = max(
            0.0,
            float(self.get_parameter("return_arm_min_distance_from_home_m").value),
        )
        self.return_point = (
            float(self.get_parameter("return_x").value),
            float(self.get_parameter("return_y").value),
        )
        self.return_goal_topic = str(self.get_parameter("return_goal_topic").value)
        self.mission_abort_topic = str(self.get_parameter("mission_abort_topic").value)
        self.return_republish_sec = max(
            0.1, float(self.get_parameter("return_republish_sec").value)
        )
        map_path = str(self.get_parameter("scenario2_map").value)

        self.known_xy, self.bboxes = self._load_map(map_path)
        execution_text = (
            f"RETURN 실행=on → home=({self.return_point[0]:.1f},{self.return_point[1]:.1f})"
            if self.execute_return else "RETURN 실행=off(상태 표시만)"
        )
        self.get_logger().info(
            f"scenario2_map: known={len(self.known_xy)} bbox={len(self.bboxes)} · {execution_text}")

        self.player_xy: Optional[Tuple[float, float]] = None
        self.detected: List[Dict[str, Any]] = []
        self._recent = deque(maxlen=self.hyst)
        self._committed = sd.ACTION_NONE
        self._llm: Dict[str, Any] = {"available": False}
        self._llm_wall = 0.0
        self._llm_running = False
        # RETURN은 terminal safety transition이다. 한 번 확정되면 detector가
        # 잠깐 비어도 waypoint/사격 FSM으로 되돌리지 않는다.
        self.return_active = False
        self.return_reason = ""
        self.return_started_wall: Optional[float] = None
        # SCENARIO2_RETURN_ARM_GUARD_V1: one-way latch.  Once the tank has
        # truly departed, ordinary sudden RETURN behavior is restored.
        self.return_armed = self.return_arm_min_distance_from_home_m <= 0.0
        self.return_home_distance_m = 0.0
        self._last_return_command_wall = -1e9

        self.create_subscription(PoseStamped, TOPIC_PLAYER, self._player_cb, 10)
        self.create_subscription(String, TOPIC_DISCOVERED, self._discovered_cb, 10)
        self.status_pub = self.create_publisher(String, TOPIC_STATUS, 10)
        self.return_goal_pub = self.create_publisher(PoseStamped, self.return_goal_topic, 10)
        self.abort_pub = self.create_publisher(String, self.mission_abort_topic, 10)
        hz = max(0.2, float(self.get_parameter("tick_hz").value))
        self.create_timer(1.0 / hz, self._tick)

    def _load_map(self, path: str) -> Tuple[List[Tuple[float, float]], List[Dict[str, float]]]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"scenario2_map 로드 실패({exc}) — known 0, LoS 없음(보수적 노출)")
            return [], []
        known: List[Tuple[float, float]] = []
        for t in data.get("targets", []) or []:
            if not isinstance(t, dict):
                continue
            mx, my = t.get("map_x"), t.get("map_y")
            if mx is None or my is None:
                pos = t.get("position") or {}
                mx, my = pos.get("x"), pos.get("z")
            if mx is not None and my is not None:
                known.append((float(mx), float(my)))
        return known, risk.bboxes_from_map(data)

    def _player_cb(self, msg: PoseStamped) -> None:
        self.player_xy = (float(msg.pose.position.x), float(msg.pose.position.y))

    def _discovered_cb(self, msg: String) -> None:
        try:
            objs = json.loads(msg.data).get("objects")
        except Exception:  # noqa: BLE001
            return
        if not isinstance(objs, list):
            return
        out: List[Dict[str, Any]] = []
        for o in objs:
            if not isinstance(o, dict) or str(o.get("class_name", "")).lower() != "tank":
                continue
            if o.get("map_x") is None or o.get("map_y") is None:
                continue
            try:
                if float(o.get("confidence", 0.0)) < self.min_conf:
                    continue
                out.append({"id": str(o.get("object_id", "")),
                            "xy": (float(o["map_x"]), float(o["map_y"])), "class": "tank"})
            except (TypeError, ValueError):
                continue
        self.detected = out


    # SCENARIO2_RETURN_ARM_GUARD_V1
    def _update_return_arm_state(self) -> None:
        """Arm emergency RETURN only after a real departure from home.

        At launch, the route begins at the same coordinate used as the RETURN
        goal.  Without this guard, two pre-existing or briefly unclassified
        tank detections can cause a self-return at spawn, which looks like the
        controller has frozen.  Arming is intentionally one-way for the run.
        """
        if self.player_xy is None:
            return
        self.return_home_distance_m = math.hypot(
            self.player_xy[0] - self.return_point[0],
            self.player_xy[1] - self.return_point[1],
        )
        if (
            not self.return_armed
            and self.return_home_distance_m >= self.return_arm_min_distance_from_home_m
        ):
            self.return_armed = True
            self.get_logger().info(
                "Scenario-2 emergency RETURN armed after departure: "
                f"home_distance={self.return_home_distance_m:.1f}m "
                f"(threshold={self.return_arm_min_distance_from_home_m:.1f}m)"
            )

    def _tick(self) -> None:
        if self.player_xy is None:
            return
        self._update_return_arm_state()
        new = sd.classify_new_threats(self.detected, self.known_xy, self.match_r)
        feat = sd.build_mission_features(self.player_xy, new, self.bboxes)
        d = sd.decide(feat)

        # Before leaving spawn, report the instantaneous judgment but do not
        # accumulate/execute RETURN.  This prevents a start-point self-return
        # from becoming a latched ballistic abort.  Once armed, normal
        # hysteresis starts fresh and still requires its configured N ticks.
        if not self.return_active and not self.return_armed:
            self._recent.clear()
            self._committed = sd.ACTION_NONE
            self._publish(feat, d)
            return

        # 히스테리시스: 최근 hyst tick이 모두 같은 action일 때만 commit(매 tick 뒤집힘 방지).
        self._recent.append(d["action"])
        if len(self._recent) == self._recent.maxlen and len(set(self._recent)) == 1:
            committed = self._recent[0]
        else:
            committed = self._committed
        changed = committed != self._committed
        self._committed = committed
        # 돌발(committed != NONE)로 새로 전이할 때만 LLM 자문 async(쿨다운).
        if self.use_llm and changed and committed != sd.ACTION_NONE:
            self._maybe_llm(feat, d)

        # 수식 판단이 RETURN으로 안정화된 때에만 실제 귀환을 시작한다.
        # LLM 응답은 자문 표시용이며 이 제어 경로를 변경하지 않는다.
        if (
            not self.return_active
            and self.execute_return
            and self._committed == sd.ACTION_RETURN
        ):
            self._start_return(feat, d)
        if self.return_active:
            # Keep status/state terminal as well as the physical return.  A
            # detector dropout must not make the MFD say NONE mid-retreat.
            self._committed = sd.ACTION_RETURN
            self._recent.clear()
            self._publish_return_commands(feat)
        self._publish(feat, d)

    def _make_goal_pose(self) -> PoseStamped:
        msg = PoseStamped()
        msg.header.frame_id = "tank_map"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x = self.return_point[0]
        msg.pose.position.y = self.return_point[1]
        msg.pose.orientation.w = 1.0
        return msg

    def _start_return(self, feat: Dict[str, Any], d: Dict[str, Any]) -> None:
        """Latch a safety return after hysteresis and publish it immediately."""
        self.return_active = True
        self.return_started_wall = time.monotonic()
        self.return_reason = str(d.get("reason") or "sudden_return")
        self._committed = sd.ACTION_RETURN
        self._recent.clear()
        self._last_return_command_wall = -1e9
        self.get_logger().warn(
            "Sudden RETURN committed: "
            f"new={feat.get('n_new')} engageable={feat.get('n_engageable')} "
            f"risk={feat.get('max_risk')} → home=({self.return_point[0]:.1f},{self.return_point[1]:.1f}); "
            "aborting ballistic checkpoint sequence"
        )
        self._publish_return_commands(feat, force=True)

    def _publish_return_commands(self, feat: Dict[str, Any], *, force: bool = False) -> None:
        """Publish abort+goal heartbeat so a volatile subscriber cannot miss RETURN."""
        now = time.monotonic()
        if not force and now - self._last_return_command_wall < self.return_republish_sec:
            return
        self._last_return_command_wall = now
        self.return_goal_pub.publish(self._make_goal_pose())
        payload = {
            "action": sd.ACTION_RETURN,
            "source": "sudden_advisor_node",
            "reason": self.return_reason,
            "goal": {"x": self.return_point[0], "y": self.return_point[1]},
            "n_new": feat.get("n_new"),
            "n_engageable": feat.get("n_engageable"),
            "max_risk": feat.get("max_risk"),
            "wall": time.time(),
        }
        self.abort_pub.publish(String(data=json.dumps(payload, ensure_ascii=False)))

    def _maybe_llm(self, feat: Dict[str, Any], d: Dict[str, Any]) -> None:
        now = time.monotonic()
        if self._llm_running or (now - self._llm_wall) < self.llm_cooldown:
            return
        self._llm_running = True
        self._llm_wall = now
        import threading
        threading.Thread(target=self._llm_worker, args=(feat, d), daemon=True).start()

    def _llm_worker(self, feat: Dict[str, Any], d: Dict[str, Any]) -> None:
        try:
            from risk_analysis.llm_reporter import LLMReporter  # 크로스패키지(설치시 가용), graceful
            rep = LLMReporter()
            prompt = (
                "너는 전차 미션 중 돌발 대응 참모다. 정찰에서 몰랐던 신규 위협이 나타났다.\n"
                "상황을 보고 ENGAGE(정지 사격)/BYPASS(무시 속행)/RETURN(후퇴) 중 권고와 이유를 한국어로.\n"
                '반드시 JSON 하나만 출력: {"action":"ENGAGE|BYPASS|RETURN","reason":"한국어 한두 문장"}\n'
                f"수식 판단: {d.get('action')} ({d.get('reason')})\n"
                f"신규 위협 {feat.get('n_new')}개, 교전가능 {feat.get('n_engageable')}, 최대위험 {feat.get('max_risk')}\n"
                f"상세: {json.dumps(feat.get('per_threat', []), ensure_ascii=False)}"
            )
            resp = rep.call_ollama(prompt)
            parsed = json.loads(resp.get("response", "")) if isinstance(resp, dict) else {}
            self._llm = {"available": True, "action": parsed.get("action"),
                         "reason": parsed.get("reason"), "model": rep.model_name}
        except Exception as exc:  # noqa: BLE001 - ollama 미가동/파싱실패 → 수식만
            self._llm = {"available": False, "error": type(exc).__name__}
        finally:
            self._llm_running = False

    def _publish(self, feat: Dict[str, Any], d: Dict[str, Any]) -> None:
        effective_action = sd.ACTION_RETURN if self.return_active else self._committed
        effective_reason = self.return_reason if self.return_active else d["reason"]
        payload = {
            "action": effective_action,          # 히스테리시스 반영 확정 결정
            "committed_action": self._committed,
            "instant_action": d["action"],      # 순간 판정(참고)
            "reason": effective_reason,
            "n_new": feat["n_new"],
            "n_engageable": feat["n_engageable"],
            "max_risk": feat["max_risk"],
            "target": d.get("target"),
            "per_threat": feat.get("per_threat", []),
            "llm": self._llm,
            "advise_only": not self.execute_return,
            "return_execution": {
                "enabled": self.execute_return,
                "active": self.return_active,
                "armed": self.return_armed,
                "home_distance_m": round(self.return_home_distance_m, 3),
                "arm_min_distance_m": self.return_arm_min_distance_from_home_m,
                "guard_reason": (
                    None if self.return_armed or self.return_active
                    else "waiting_to_leave_home_before_return_can_execute"
                ),
                "goal": {"x": self.return_point[0], "y": self.return_point[1]},
                "goal_topic": self.return_goal_topic,
                "abort_topic": self.mission_abort_topic,
                "started_wall_monotonic": self.return_started_wall,
            },
            "wall": time.time(),
        }
        self.status_pub.publish(String(data=json.dumps(payload, ensure_ascii=False)))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SuddenAdvisorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
