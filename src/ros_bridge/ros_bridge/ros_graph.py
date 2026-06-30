"""ROS 계산 그래프(노드↔토픽) 실시간 introspection + 토픽 레이트(Hz) 모니터.

rqt_graph를 웹으로 대체 — 브릿지(rclpy 노드)가 그래프를 직접 introspect해서 JSON으로
대시보드 payload에 실어보내고, 프론트(Cytoscape.js)가 흐름 그래프로 그린다.

안전:
- 그래프 introspection·레이트 갱신은 **별도 ReentrantCallbackGroup** 타이머에서 → 제어 핫패스 비차단.
- 레이트용 generic 구독은 무거운 타입(PointCloud2/Image)·노이즈 토픽 제외 + 개수 cap.
- env로 끌 수 있음: TANK_ROS_GRAPH=false(전체), TANK_ROS_GRAPH_RATES=false(레이트만).
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any, Dict

from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

# 레이트 구독에서 제외할 무거운 메시지 타입(대역폭/CPU 보호).
_HEAVY_TYPES = ("PointCloud2", "Image", "CompressedImage", "LaserScan")
# 그래프에서 숨길 ROS 내부 노이즈 토픽.
_SKIP_TOPICS = ("/rosout", "/parameter_events")
_SKIP_NODE_SUBSTR = ("rosbridge", "_ros2cli", "rqt")
# 노드마다 자동 생기는 파라미터/lifecycle 서비스(노이즈) — 서비스 목록에서 숨김.
_NOISE_SERVICES = {
    "describe_parameters", "get_parameter_types", "get_parameters",
    "list_parameters", "set_parameters", "set_parameters_atomically",
}


def _flag(name: str, default: bool) -> bool:
    return os.environ.get(name, "true" if default else "false").strip().lower() in ("1", "true", "yes", "y")


def _full_name(name: str, ns: str) -> str:
    ns = ns or ""
    if ns in ("", "/"):
        return "/" + name
    return ns.rstrip("/") + "/" + name


class RosGraphMonitor:
    """브릿지 노드에 붙어 그래프+레이트를 주기적으로 갱신, get_payload()로 최신 스냅샷 제공."""

    def __init__(self, node):
        self.node = node
        self.enabled = _flag("TANK_ROS_GRAPH", True)
        self.rates_enabled = _flag("TANK_ROS_GRAPH_RATES", True)
        try:
            self.max_subs = int(os.environ.get("TANK_ROS_GRAPH_MAX_SUBS", "60"))
        except (TypeError, ValueError):
            self.max_subs = 60
        self.window = 2.0

        self._cbg = ReentrantCallbackGroup()  # 제어 그룹과 분리
        self._subs: Dict[str, Any] = {}        # topic -> subscription
        self._counts: Dict[str, int] = {}      # topic -> 윈도우 내 수신 수
        self._hz: Dict[str, float] = {}        # topic -> Hz
        self._win_start = time.monotonic()
        self._lock = threading.Lock()
        self._payload: Dict[str, Any] = {"nodes": [], "topics": [], "edges": [], "available": False}

        self._tf: Dict[str, Dict[str, Any]] = {}  # child_frame -> {parent, static}

        if self.enabled:
            try:
                self._timer = node.create_timer(1.5, self._tick, callback_group=self._cbg)
            except Exception:
                self.enabled = False
            # TF 트리: /tf(동적) + /tf_static(래치) 구독 → child→parent 누적
            try:
                from tf2_msgs.msg import TFMessage
                from rclpy.qos import DurabilityPolicy
                node.create_subscription(TFMessage, "/tf", lambda m: self._on_tf(m, False),
                                         self._qos(), callback_group=self._cbg)
                static_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                                        durability=DurabilityPolicy.TRANSIENT_LOCAL,
                                        history=HistoryPolicy.KEEP_LAST, depth=1)
                node.create_subscription(TFMessage, "/tf_static", lambda m: self._on_tf(m, True),
                                         static_qos, callback_group=self._cbg)
            except Exception:
                pass

    def _on_tf(self, msg, static: bool) -> None:
        try:
            for tr in msg.transforms:
                self._tf[tr.child_frame_id] = {"parent": tr.header.frame_id, "static": static}
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    def get_payload(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._payload)

    # ------------------------------------------------------------------ #
    def _qos(self) -> QoSProfile:
        return QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                          history=HistoryPolicy.KEEP_LAST, depth=1)

    def _is_heavy(self, types) -> bool:
        return any(any(h in t for h in _HEAVY_TYPES) for t in types)

    def _on_msg(self, topic: str) -> None:
        self._counts[topic] = self._counts.get(topic, 0) + 1

    def _tick(self) -> None:
        try:
            self._rebuild()
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    def _rebuild(self) -> None:
        node = self.node
        topics = node.get_topic_names_and_types()  # [(name, [types])]

        # 1) Hz 윈도우 마감
        now = time.monotonic()
        dt = now - self._win_start
        if dt >= self.window:
            self._hz = {t: round(c / dt, 1) for t, c in self._counts.items()}
            for k in self._counts:
                self._counts[k] = 0
            self._win_start = now

        # 2) 레이트 구독 동기화(추가/삭제)
        if self.rates_enabled:
            self._sync_subs(topics)

        # 3) 구조 빌드
        node_ids = set()
        for name, ns in node.get_node_names_and_namespaces():
            full = _full_name(name, ns)
            if any(s in full for s in _SKIP_NODE_SUBSTR):
                continue
            node_ids.add(full)

        topic_list = []
        edges = []
        for tname, ttypes in topics:
            if tname in _SKIP_TOPICS:
                continue
            hz = self._hz.get(tname)
            topic_list.append({"id": "t:" + tname, "name": tname,
                               "type": (ttypes[0] if ttypes else ""), "hz": hz})
            try:
                pubs = node.get_publishers_info_by_topic(tname)
                subs = node.get_subscriptions_info_by_topic(tname)
            except Exception:
                pubs, subs = [], []
            for p in pubs:
                nn = _full_name(getattr(p, "node_name", ""), getattr(p, "node_namespace", ""))
                if nn in node_ids:
                    edges.append({"id": "p|" + nn + "|" + tname, "source": nn,
                                  "target": "t:" + tname, "hz": hz})
            for s in subs:
                nn = _full_name(getattr(s, "node_name", ""), getattr(s, "node_namespace", ""))
                if nn in node_ids:
                    edges.append({"id": "s|" + tname + "|" + nn, "source": "t:" + tname,
                                  "target": nn, "hz": hz})

        # 고립 토픽(발행·구독 둘 다 숨긴 노드뿐)은 제외해 그래프 깔끔하게
        connected_topics = {e["source"] for e in edges} | {e["target"] for e in edges}
        topic_list = [t for t in topic_list if t["id"] in connected_topics]

        # 서비스: 노드별 서버 서비스 → service→[nodes] (노이즈 파라미터 서비스 제외)
        services: Dict[str, Dict[str, Any]] = {}
        for name, ns in node.get_node_names_and_namespaces():
            full = _full_name(name, ns)
            if any(s in full for s in _SKIP_NODE_SUBSTR):
                continue
            try:
                for sname, stypes in node.get_service_names_and_types_by_node(name, ns):
                    if sname.rsplit("/", 1)[-1] in _NOISE_SERVICES:
                        continue
                    d = services.setdefault(sname, {"type": stypes[0] if stypes else "", "nodes": []})
                    if full not in d["nodes"]:
                        d["nodes"].append(full)
            except Exception:
                pass
        service_list = [{"name": k, "type": v["type"], "nodes": v["nodes"]}
                        for k, v in sorted(services.items())]

        # TF: child→parent 누적본 → 리스트
        tf_list = [{"child": c, "parent": v.get("parent", ""), "static": bool(v.get("static"))}
                   for c, v in sorted(self._tf.items())]

        payload = {
            "nodes": [{"id": nid} for nid in sorted(node_ids)],
            "topics": topic_list,
            "edges": edges,
            "services": service_list,
            "tf": tf_list,
            "available": True,
            "ratesEnabled": self.rates_enabled,
        }
        with self._lock:
            self._payload = payload

    def _sync_subs(self, topics) -> None:
        want: Dict[str, str] = {}
        for tname, ttypes in topics:
            if tname in _SKIP_TOPICS or not ttypes:
                continue
            if self._is_heavy(ttypes):
                continue
            want[tname] = ttypes[0]

        # 신규 추가(cap까지)
        for tname, ttype in want.items():
            if tname in self._subs or len(self._subs) >= self.max_subs:
                continue
            try:
                from rosidl_runtime_py.utilities import get_message
                cls = get_message(ttype)
                self._counts.setdefault(tname, 0)
                self._subs[tname] = self.node.create_subscription(
                    cls, tname, (lambda m, t=tname: self._on_msg(t)),
                    self._qos(), callback_group=self._cbg)
            except Exception:
                pass

        # 사라진 토픽 정리
        for tname in list(self._subs):
            if tname not in want:
                try:
                    self.node.destroy_subscription(self._subs[tname])
                except Exception:
                    pass
                self._subs.pop(tname, None)
                self._counts.pop(tname, None)
                self._hz.pop(tname, None)


# --------------------------------------------------------------------------- #
# 파라미터 view/edit (on-demand) — 노드별 파라미터 서비스 호출. app_routes REST가 사용.
# 브릿지 노드는 executor(별도 thread)가 spin 중이라, Flask thread에서 call_async 후
# future.done()을 폴링하면 응답이 처리된다.
# --------------------------------------------------------------------------- #
def _call_service(node, cli, req, timeout=3.0):
    if not cli.wait_for_service(timeout_sec=1.0):
        return None
    fut = cli.call_async(req)
    t0 = time.monotonic()
    while not fut.done() and (time.monotonic() - t0) < timeout:
        time.sleep(0.02)
    return fut.result() if fut.done() else None


def _pv_to_json(v) -> Dict[str, Any]:
    if v is None:
        return {"type": "unknown", "value": None}
    t = v.type
    if t == 1:
        return {"type": "bool", "value": bool(v.bool_value)}
    if t == 2:
        return {"type": "int", "value": int(v.integer_value)}
    if t == 3:
        return {"type": "double", "value": round(float(v.double_value), 6)}
    if t == 4:
        return {"type": "string", "value": str(v.string_value)}
    if t == 6:
        return {"type": "bool[]", "value": list(v.bool_array_value)}
    if t == 7:
        return {"type": "int[]", "value": list(v.integer_array_value)}
    if t == 8:
        return {"type": "double[]", "value": [round(float(x), 6) for x in v.double_array_value]}
    if t == 9:
        return {"type": "string[]", "value": list(v.string_array_value)}
    return {"type": "other", "value": None}


def list_node_params(node, target: str):
    """노드의 파라미터 이름+값 목록. 실패 시 None."""
    try:
        from rcl_interfaces.srv import GetParameters, ListParameters
    except Exception:
        return None
    target = (target or "").rstrip("/")
    if not target:
        return None
    lc = node.create_client(ListParameters, f"{target}/list_parameters")
    try:
        lr = _call_service(node, lc, ListParameters.Request())
    finally:
        node.destroy_client(lc)
    if lr is None:
        return None
    names = list(lr.result.names)
    if not names:
        return []
    gc = node.create_client(GetParameters, f"{target}/get_parameters")
    try:
        greq = GetParameters.Request()
        greq.names = names
        gr = _call_service(node, gc, greq)
    finally:
        node.destroy_client(gc)
    vals = list(gr.values) if gr else []
    out = []
    for i, nm in enumerate(names):
        v = vals[i] if i < len(vals) else None
        out.append({"name": nm, **_pv_to_json(v)})
    out.sort(key=lambda p: p["name"])
    return out


def set_node_param(node, target: str, name: str, raw):
    """파라미터 설정(문자열에서 타입 추론). {ok, reason}."""
    try:
        from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
        from rcl_interfaces.srv import SetParameters
    except Exception:
        return {"ok": False, "reason": "rcl_interfaces 불가"}
    target = (target or "").rstrip("/")
    pv = ParameterValue()
    s = str(raw).strip()
    if s.lower() in ("true", "false"):
        pv.type = ParameterType.PARAMETER_BOOL
        pv.bool_value = (s.lower() == "true")
    else:
        try:
            iv = int(s)
            pv.type = ParameterType.PARAMETER_INTEGER
            pv.integer_value = iv
        except ValueError:
            try:
                dv = float(s)
                pv.type = ParameterType.PARAMETER_DOUBLE
                pv.double_value = dv
            except ValueError:
                pv.type = ParameterType.PARAMETER_STRING
                pv.string_value = s
    p = Parameter()
    p.name = name
    p.value = pv
    cli = node.create_client(SetParameters, f"{target}/set_parameters")
    try:
        req = SetParameters.Request()
        req.parameters = [p]
        res = _call_service(node, cli, req)
    finally:
        node.destroy_client(cli)
    if res is None:
        return {"ok": False, "reason": "응답 없음(노드 미응답)"}
    r0 = res.results[0] if res.results else None
    return {"ok": bool(r0 and r0.successful), "reason": (getattr(r0, "reason", "") if r0 else "")}
