#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""돌발 대응 자문 노드 (advise-only) — Part B 라이브.

미션 주행 중 perception이 **정찰에서 몰랐던 신규 위협**(known 미매칭 tank)을 감지하면
sudden_decision 코어로 ENGAGE/BYPASS/RETURN을 판단해 `/tank/decision/status`로 발행한다
(→ 브릿지 MFD 패널). LLM 자문은 **돌발 때만** async(쿨다운).

**advise-only**: goal/engage 명령을 **발행하지 않는다** — ballistic_turret_node의 체크포인트
시퀀스를 덮어쓰지 않게(과거 decision_node의 RETURN FSM이 체크포인트를 덮어써 비활성됐던 문제 회피).
실제 행동(복귀/교전) 연결은 ballistic 공존 설계·라이브 검증 후 별도 단계.

수식/히스테리시스는 여기(노드), 순간 판정은 sudden_decision(순수 코어)이 담당한다.
"""
import json
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


class SuddenAdvisorNode(Node):
    def __init__(self) -> None:
        super().__init__("sudden_advisor_node")
        self.declare_parameter("tick_hz", 2.0)
        self.declare_parameter("hysteresis_ticks", 2)   # 최근 N tick 동일해야 commit(churn 방지)
        self.declare_parameter("llm_cooldown_sec", 20.0)
        self.declare_parameter("known_match_radius_m", 8.0)
        self.declare_parameter("min_confidence", 0.0)
        self.declare_parameter("use_llm", True)
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
        map_path = str(self.get_parameter("scenario2_map").value)

        self.known_xy, self.bboxes = self._load_map(map_path)
        self.get_logger().info(
            f"scenario2_map: known={len(self.known_xy)} bbox={len(self.bboxes)} · advise-only(goal/engage 미발행)")

        self.player_xy: Optional[Tuple[float, float]] = None
        self.detected: List[Dict[str, Any]] = []
        self._recent = deque(maxlen=self.hyst)
        self._committed = sd.ACTION_NONE
        self._llm: Dict[str, Any] = {"available": False}
        self._llm_wall = 0.0
        self._llm_running = False

        self.create_subscription(PoseStamped, TOPIC_PLAYER, self._player_cb, 10)
        self.create_subscription(String, TOPIC_DISCOVERED, self._discovered_cb, 10)
        self.status_pub = self.create_publisher(String, TOPIC_STATUS, 10)
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

    def _tick(self) -> None:
        if self.player_xy is None:
            return
        new = sd.classify_new_threats(self.detected, self.known_xy, self.match_r)
        feat = sd.build_mission_features(self.player_xy, new, self.bboxes)
        d = sd.decide(feat)
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
        self._publish(feat, d)

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
        payload = {
            "action": self._committed,          # 히스테리시스 반영 확정 결정
            "instant_action": d["action"],      # 순간 판정(참고)
            "reason": d["reason"],
            "n_new": feat["n_new"],
            "n_engageable": feat["n_engageable"],
            "max_risk": feat["max_risk"],
            "target": d.get("target"),
            "per_threat": feat.get("per_threat", []),
            "llm": self._llm,
            "advise_only": True,
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
