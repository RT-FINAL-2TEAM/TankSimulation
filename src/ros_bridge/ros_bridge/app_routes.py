# -*- coding: utf-8 -*-
"""
############################################################
# app_routes.py
# Tank Challenge Flask Routes
############################################################

이 파일의 역할
------------------------------------------------------------
- Tank Challenge 시뮬레이터가 호출하는 공식 HTTP API endpoint를 Flask route로 제공한다.
- 각 route에서 시뮬레이터 요청을 검증한다.
- 검증된 데이터는 ROS2 bridge node로 전달한다.
- ROS2 bridge node는 이 데이터를 ROS2 topic으로 publish한다.
- 시뮬레이터가 요구하는 JSON 응답 형식에 맞춰 응답을 반환한다.
"""

############################################################
# 1. Python type hint import
############################################################

# Any:
# - image, JSON, detection 결과처럼 타입이 고정되지 않은 값을 표현할 때 사용한다.
# Dict:
# - Flask 요청 body, ROS2로 넘길 payload, status 응답처럼 dict 구조를 명시할 때 사용한다.
from typing import Any, Dict

import json
import os
from copy import deepcopy
from pathlib import Path
from threading import Lock


############################################################
# 2. Flask import
############################################################

# Flask:
# - HTTP 서버 애플리케이션 객체를 만든다.
# - @app.route(...) decorator로 Tank Challenge 공식 endpoint를 등록한다.
#
# jsonify:
# - Python dict/list를 Flask HTTP JSON response로 변환한다.
# - 시뮬레이터는 각 endpoint 응답을 JSON으로 기대하므로 반드시 필요하다.
#
# request:
# - 시뮬레이터가 보낸 JSON body 또는 image multipart file을 읽는 객체이다.
from flask import Flask, jsonify, request, abort, send_file


############################################################
# 3. Project module import
############################################################

# fallback_command:
# - ROS2 bridge가 아직 준비되지 않았거나 auto 모드에서 최신 명령이 없을 때
#   안전하게 반환할 /get_action 명령을 생성한다.
#
# init_config:
# - 공식 /init endpoint가 반환해야 하는 초기 설정 JSON을 생성한다.
# - startMode, 아군/적군 시작 위치, trackingMode, logMode 등의 값이 포함된다.
from .commands import fallback_command, init_config

# IMAGE_DIR:
# - /detect, /stereo_image에서 이미지 저장 옵션이 켜진 경우 저장할 폴더이다.
#
# PORT:
# - Flask endpoint server port이다.
# - 시뮬레이터 Properties의 Request Port와 맞춰야 한다.
#
# SAVE_IMAGES:
# - True이면 /detect, /stereo_image로 들어온 이미지를 파일로 저장한다.
#
# TANK_MODE:
# - "monitor" 또는 "auto"이다.
# - monitor: trackingMode=False, logMode=True 중심 관측.
# - auto   : trackingMode=True, logMode=True 중심 자율제어.
from .config import (
    EPISODE_CONTROL_ENABLED,
    IMAGE_DIR,
    LIVE_VIEW_ENABLED,
    LIVE_VIEW_FPS,
    LIVE_VIEW_JPEG_QUALITY,
    PORT,
    SAVE_IMAGES,
    TANK_MODE,
    YOLO_ASYNC_ENABLED,
    YOLO_ASYNC_LOG_INTERVAL_SEC,
    YOLO_ASYNC_MAX_RESULT_AGE_MS,
    YOLO_ASYNC_MIN_INTERVAL_SEC,
)

# get_bridge:
# - 현재 실행 중인 ROS2 RosBridge node 인스턴스를 가져온다.
# - Flask route는 직접 ROS2 topic을 publish하지 않고,
#   bridge handler에 데이터를 넘기는 구조이다.
from .ros_runtime import get_bridge, ros_status
from . import live_view
from .async_yolo import AsyncYoloService

# compact_info:
# - /info 원본 JSON에서 핵심 필드만 추려 터미널 출력과 fallback 처리에 사용한다.
#
# now_wall:
# - 현재 wall-clock time을 timestamp로 만든다.
# - 이미지 파일명에 timestamp를 붙일 때 사용한다.
#
# pretty:
# - dict/list를 사람이 보기 좋은 JSON 문자열로 변환하여 터미널에 출력한다.
from .utils import compact_info, now_wall, pretty, raw_and_map_pose


############################################################
# 3-1. Optional embedded YOLO detector import
############################################################
# vision은 별도 ROS2 package로 구성되어 있다.
# ultralytics/torch가 설치되어 있지 않아도 bridge 자체가 죽지 않도록
# 실제 detector 생성은 /detect 요청 시 lazy-loading으로 수행한다.
try:
    from vision.yolo_detector import get_detector
except Exception as exc:  # pragma: no cover - runtime dependency fallback
    get_detector = None
    _YOLO_IMPORT_ERROR = exc
else:
    _YOLO_IMPORT_ERROR = None


_ASYNC_YOLO_SERVICE = None


def _get_async_yolo_service():
    """선택적 async YOLO worker를 lazy 방식으로 생성한다."""
    global _ASYNC_YOLO_SERVICE
    if _ASYNC_YOLO_SERVICE is None:
        if get_detector is None:
            raise RuntimeError(f"YOLO unavailable: {_YOLO_IMPORT_ERROR}")
        _ASYNC_YOLO_SERVICE = AsyncYoloService(
            get_detector,
            min_interval_sec=YOLO_ASYNC_MIN_INTERVAL_SEC,
            max_result_age_ms=YOLO_ASYNC_MAX_RESULT_AGE_MS,
            log_interval_sec=YOLO_ASYNC_LOG_INTERVAL_SEC,
        )
    return _ASYNC_YOLO_SERVICE


############################################################
# 4. Flask application object
############################################################

# Flask app 객체.
# 이 객체에 @app.route(...)로 Tank Challenge 공식 endpoint들을 등록한다.
#
# 시뮬레이터 PC는 이 Flask 서버의 IP/Port로 요청을 보낸다.
# Ubuntu 작업 PC에서 이 서버를 실행하면 Windows 시뮬레이터 PC가
# http://<Ubuntu_IP>:5000/init 같은 주소로 접근하게 된다.
app = Flask(__name__)

_FALLBACK_STATE_LOCK = Lock()
_FALLBACK_STATE: Dict[str, Any] = {
    "latest": {},
    "routeCounts": {},
}


def _fallback_count(route: str) -> None:
    with _FALLBACK_STATE_LOCK:
        counts = _FALLBACK_STATE.setdefault("routeCounts", {})
        counts[route] = int(counts.get(route, 0)) + 1


def _fallback_snapshot() -> Dict[str, Any]:
    with _FALLBACK_STATE_LOCK:
        return deepcopy(_FALLBACK_STATE)


def _store_fallback_info(data: Dict[str, Any]) -> Dict[str, Any]:
    ts = now_wall()
    compact = compact_info(data)
    player_raw = player_map = None
    enemy_raw = enemy_map = None
    if isinstance(data.get("playerPos"), dict):
        player_raw, player_map = raw_and_map_pose(data.get("playerPos"), "/info/playerPos")
    if isinstance(data.get("enemyPos"), dict):
        enemy_raw, enemy_map = raw_and_map_pose(data.get("enemyPos"), "/info/enemyPos")

    compact_payload = {"route": "/info", "timestamp_wall": ts, "data": compact}
    player_state = {
        "timestamp_wall": ts,
        "source": "/info",
        "pose_raw": player_raw,
        "pose_map": player_map,
        "speed": data.get("playerSpeed"),
        "health": data.get("playerHealth"),
        "turret": {"x": data.get("playerTurretX"), "y": data.get("playerTurretY")},
        "body": {"x": data.get("playerBodyX"), "y": data.get("playerBodyY"), "z": data.get("playerBodyZ")},
        "sim_time": data.get("time"),
        "distance": data.get("distance"),
    }
    enemy_state = {
        "timestamp_wall": ts,
        "source": "/info",
        "pose_raw": enemy_raw,
        "pose_map": enemy_map,
        "speed": data.get("enemySpeed"),
        "health": data.get("enemyHealth"),
    }

    with _FALLBACK_STATE_LOCK:
        latest = _FALLBACK_STATE.setdefault("latest", {})
        latest["info_compact"] = deepcopy(compact_payload)
        latest["player_state"] = deepcopy(player_state)
        latest["enemy_state"] = deepcopy(enemy_state)
        if player_map:
            latest["player_pose_map"] = deepcopy(player_map)
        if enemy_map:
            latest["enemy_pose_map"] = deepcopy(enemy_map)
        latest["sim_status"] = {
            "route": "/info",
            "timestamp_wall": ts,
            "sim_time": data.get("time"),
            "distance": data.get("distance"),
            "player_speed": data.get("playerSpeed"),
            "player_health": data.get("playerHealth"),
            "enemy_speed": data.get("enemySpeed"),
            "enemy_health": data.get("enemyHealth"),
            "terrain_size_unity": {"x": 300.0, "z": 300.0},
        }
    return compact_payload


def _store_fallback_get_action(data: Dict[str, Any], command: Dict[str, Any]) -> None:
    position = data.get("position") if isinstance(data, dict) else None
    if not isinstance(position, dict):
        return
    ts = now_wall()
    pose_raw, pose_map = raw_and_map_pose(position, "/get_action/position")
    turret = data.get("turret") if isinstance(data.get("turret"), dict) else {}
    raw_payload = {
        "route": "/get_action",
        "timestamp_wall": ts,
        "request": deepcopy(data),
        "pose_raw": pose_raw,
        "pose_map": pose_map,
        "turret": deepcopy(turret),
    }
    response_payload = {
        "route": "/get_action",
        "timestamp_wall": ts,
        "mode": TANK_MODE,
        "source": "fallback_without_ros",
        "command": deepcopy(command),
    }
    with _FALLBACK_STATE_LOCK:
        latest = _FALLBACK_STATE.setdefault("latest", {})
        latest["get_action_raw"] = deepcopy(raw_payload)
        latest["get_action_pose_map"] = deepcopy(pose_map)
        latest["get_action_response"] = deepcopy(response_payload)
        latest["player_pose_map"] = deepcopy(pose_map)


def _resolve_static_map_path() -> Path:
    env_path = os.environ.get("TANK_STATIC_MAP_PATH", "").strip()
    if env_path:
        return Path(env_path).expanduser().resolve()

    try:
        from ament_index_python.packages import get_package_share_directory

        share_path = Path(get_package_share_directory("rviz_visualization")) / "map" / "finalmap.map"
        if share_path.exists():
            return share_path
    except Exception:
        pass

    # source tree fallback: .../src/ros_bridge/ros_bridge/app_routes.py -> .../src
    return Path(__file__).resolve().parents[2] / "rviz_visualization" / "map" / "finalmap.map"


def _resolve_static_map_overview_path() -> Path:
    env_path = os.environ.get("TANK_STATIC_MAP_OVERVIEW_PATH", "").strip()
    if env_path:
        return Path(env_path).expanduser().resolve()

    candidates = [
        Path(__file__).resolve().parents[2] / "rviz_visualization" / "map" / "finalmap_overview.png",
        Path(__file__).resolve().parents[2] / "rviz_visualization" / "map" / "finalmap_overview.jpg",
        Path(r"C:\Users\green\OneDrive\Desktop\teamproject\map\finalmap_overview.png"),
        Path(r"C:\Users\green\OneDrive\Desktop\teamproject\map\finalmap_overview.jpg"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _map_object_category(name: Any) -> str:
    text = str(name or "").lower()
    if text.startswith("tree"):
        return "tree"
    if text.startswith("rock"):
        return "rock"
    if text.startswith("house"):
        return "house"
    if text.startswith("human"):
        return "human"
    if text.startswith("car"):
        return "car"
    if text.startswith("tank"):
        return "tank"
    return "unknown"


def _overview_terrain_zones() -> Dict[str, Any]:
    """제공된 top-down map overview에서 수작업으로 따낸 terrain zone들."""

    return {
        "source": "user_overview_image",
        "coordinate": "map_x_rawx_y_rawz",
        "waterDataAvailable": True,
        "zones": [
            {
                "name": "main_waterway",
                "type": "water",
                "points": [
                    {"x": 163.8, "y": 300.0},
                    {"x": 197.3, "y": 300.0},
                    {"x": 204.6, "y": 288.0},
                    {"x": 207.3, "y": 269.4},
                    {"x": 220.3, "y": 252.8},
                    {"x": 234.6, "y": 250.8},
                    {"x": 245.8, "y": 255.8},
                    {"x": 257.1, "y": 254.5},
                    {"x": 262.1, "y": 241.5},
                    {"x": 252.5, "y": 231.6},
                    {"x": 235.9, "y": 226.6},
                    {"x": 232.9, "y": 211.0},
                    {"x": 234.2, "y": 195.7},
                    {"x": 238.8, "y": 181.1},
                    {"x": 235.2, "y": 161.2},
                    {"x": 228.9, "y": 144.5},
                    {"x": 226.6, "y": 123.9},
                    {"x": 232.9, "y": 105.9},
                    {"x": 242.5, "y": 87.1},
                    {"x": 248.2, "y": 66.2},
                    {"x": 249.2, "y": 45.6},
                    {"x": 256.1, "y": 22.6},
                    {"x": 256.5, "y": 0.0},
                    {"x": 223.6, "y": 0.0},
                    {"x": 219.3, "y": 20.3},
                    {"x": 214.9, "y": 42.5},
                    {"x": 206.9, "y": 61.8},
                    {"x": 197.3, "y": 81.4},
                    {"x": 205.0, "y": 100.0},
                    {"x": 199.5, "y": 116.5},
                    {"x": 198.0, "y": 130.5},
                    {"x": 202.5, "y": 144.0},
                    {"x": 203.5, "y": 156.5},
                    {"x": 198.2, "y": 169.0},
                    {"x": 187.0, "y": 181.0},
                    {"x": 181.1, "y": 195.0},
                    {"x": 180.1, "y": 210.3},
                    {"x": 171.4, "y": 226.9},
                    {"x": 171.1, "y": 240.9},
                    {"x": 179.1, "y": 253.5},
                    {"x": 188.0, "y": 262.5},
                    {"x": 184.7, "y": 275.4},
                    {"x": 177.1, "y": 289.0},
                ],
            },
            {
                "name": "north_east_stream",
                "type": "water",
                "points": [
                    {"x": 276.4, "y": 300.0},
                    {"x": 295.3, "y": 300.0},
                    {"x": 293.7, "y": 287.4},
                    {"x": 285.7, "y": 276.1},
                    {"x": 283.1, "y": 262.8},
                    {"x": 272.1, "y": 255.2},
                    {"x": 257.1, "y": 254.5},
                    {"x": 245.8, "y": 255.8},
                    {"x": 260.1, "y": 266.4},
                    {"x": 271.4, "y": 277.4},
                ],
            },
            {
                "name": "center_pond",
                "type": "water",
                "points": [
                    {"x": 159.8, "y": 145.2},
                    {"x": 174.4, "y": 145.2},
                    {"x": 184.1, "y": 136.5},
                    {"x": 187.0, "y": 122.3},
                    {"x": 181.1, "y": 113.3},
                    {"x": 168.1, "y": 114.3},
                    {"x": 158.5, "y": 122.9},
                    {"x": 156.1, "y": 136.2},
                ],
            },
            {
                "name": "water_passage",
                "type": "passage",
                "role": "tank_route_corridor",
                "points": [
                    {"x": 184.5, "y": 162.5},
                    {"x": 197.5, "y": 158.5},
                    {"x": 205.5, "y": 145.5},
                    {"x": 203.0, "y": 130.0},
                    {"x": 194.5, "y": 116.0},
                    {"x": 183.0, "y": 112.0},
                    {"x": 177.5, "y": 119.5},
                    {"x": 186.8, "y": 130.2},
                    {"x": 188.5, "y": 144.8},
                    {"x": 181.0, "y": 155.5},
                ],
            },
            {
                "name": "east_rocky_ridge",
                "type": "rocky",
                "points": [
                    {"x": 211.6, "y": 239.2},
                    {"x": 262.1, "y": 238.5},
                    {"x": 274.8, "y": 213.6},
                    {"x": 267.8, "y": 176.4},
                    {"x": 260.1, "y": 136.9},
                    {"x": 253.8, "y": 100.0},
                    {"x": 243.2, "y": 71.8},
                    {"x": 230.2, "y": 87.0},
                    {"x": 226.2, "y": 118.9},
                    {"x": 232.2, "y": 153.5},
                    {"x": 237.5, "y": 185.7},
                    {"x": 234.9, "y": 213.6},
                ],
            },
        ],
    }


def _load_static_map_payload() -> Dict[str, Any]:
    map_path = _resolve_static_map_path()
    with map_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    obstacles = payload.get("obstacles")
    if not isinstance(obstacles, list):
        obstacles = []
        payload["obstacles"] = obstacles
    payload["mapFile"] = str(map_path)
    payload["objectCount"] = len(obstacles)
    overview_path = _resolve_static_map_overview_path()
    payload["overviewImage"] = {
        "available": overview_path.exists(),
        "path": str(overview_path),
        "url": "/api/static-map/overview",
    }
    overview_zones = _overview_terrain_zones()
    zone_counts: Dict[str, int] = {}
    for zone in overview_zones.get("zones", []):
        if isinstance(zone, dict):
            zone_type = str(zone.get("type") or "unknown")
            zone_counts[zone_type] = zone_counts.get(zone_type, 0) + 1
    overview_zones["zoneCounts"] = zone_counts
    payload["terrainZones"] = overview_zones
    payload.setdefault(
        "bounds",
        {
            "min_x": 0.0,
            "max_x": 300.0,
            "min_y": 0.0,
            "max_y": 300.0,
            "min_z": 0.0,
            "max_z": 300.0,
        },
    )

    category_counts: Dict[str, int] = {}
    xs = []
    zs = []
    heights = []
    for obj in obstacles:
        if not isinstance(obj, dict):
            continue
        category = _map_object_category(obj.get("prefabName"))
        category_counts[category] = category_counts.get(category, 0) + 1
        pos = obj.get("position")
        if not isinstance(pos, dict):
            continue
        try:
            x = float(pos.get("x"))
            z = float(pos.get("z"))
            height = float(pos.get("y"))
        except (TypeError, ValueError):
            continue
        xs.append(x)
        zs.append(z)
        heights.append(height)

    payload["categoryCounts"] = category_counts
    if xs and zs:
        payload["objectBounds"] = {
            "min_x": min(xs),
            "max_x": max(xs),
            "min_y": min(zs),
            "max_y": max(zs),
            "min_z": min(zs),
            "max_z": max(zs),
        }

    if heights:
        min_h = min(heights)
        max_h = max(heights)
        avg_h = sum(heights) / len(heights)
        span = max(max_h - min_h, 0.0)
        low_threshold = min_h + span * 0.18
        high_threshold = min_h + span * 0.78
        payload["heightSummary"] = {
            "source": "obstacle.position.y",
            "mode": "inferred_from_static_objects",
            "sampleCount": len(heights),
            "min": min_h,
            "max": max_h,
            "avg": avg_h,
        }
        payload["surfaceSummary"] = {
            "source": "user overview image + height heuristic",
            "mode": "overview_water_polygons",
            "lowThreshold": low_threshold,
            "highThreshold": high_threshold,
            "lowlandCount": sum(1 for h in heights if h <= low_threshold),
            "highlandCount": sum(1 for h in heights if h >= high_threshold),
            "waterDataAvailable": True,
            "waterZoneCount": zone_counts.get("water", 0),
            "rockyZoneCount": zone_counts.get("rocky", 0),
            "passageZoneCount": zone_counts.get("passage", 0),
        }
        payload["terrainLayers"] = {
            "elevation": "inferred_from_static_object_heights",
            "surface": "manual_water_passage_and_rocky_zones_from_overview",
        }
    else:
        payload["heightSummary"] = {
            "source": "obstacle.position.y",
            "mode": "unavailable",
            "sampleCount": 0,
        }
        payload["surfaceSummary"] = {
            "source": "height heuristic",
            "mode": "unavailable",
            "waterDataAvailable": False,
        }
    return payload

############################################################
# 4-1. Client IP allowlist
############################################################
# 시뮬레이터가 실행되는 Windows PC IP만 Flask endpoint 접근을 허용한다.
# 기본 허용 IP:
# - 127.0.0.1 / ::1     : Ubuntu 로컬 테스트용
# - 192.168.0.82        : Windows Tank Simulator PC IP
#
# Windows PC IP가 바뀌면 아래 DEFAULT_ALLOWED_CLIENTS의 IP를 바꾸거나,
# 실행 시 환경변수로 지정한다.
#
# 예:
# TANK_ALLOWED_CLIENTS=127.0.0.1,192.168.0.82 \
# TANK_MODE=auto ros2 run ros_bridge ros_bridge
############################################################

DEFAULT_ALLOWED_CLIENTS = {
    "127.0.0.1",
    "::1",
    # "192.168.0.44",  # Windows / Tank Simulator PC IP
    "192.168.0.18",  # Windows / Tank Simulator PC IP
}

_allowed_clients_env = os.environ.get("TANK_ALLOWED_CLIENTS", "").strip()
if _allowed_clients_env:
    ALLOWED_CLIENTS = {
        ip.strip()
        for ip in _allowed_clients_env.split(",")
        if ip.strip()
    }
else:
    ALLOWED_CLIENTS = DEFAULT_ALLOWED_CLIENTS


@app.before_request
def block_other_clients():
    """등록된 Windows 시뮬레이터 PC 또는 localhost 요청만 허용한다."""

    client_ip = request.headers.get("X-Forwarded-For", request.remote_addr)
    if client_ip and "," in client_ip:
        client_ip = client_ip.split(",", 1)[0].strip()

    print(f"[REQ] {client_ip} {request.method} {request.path}")
    _fallback_count(request.path)

    if client_ip not in ALLOWED_CLIENTS:
        print(f"[BLOCKED OTHER CLIENT] {client_ip}; allowed={sorted(ALLOWED_CLIENTS)}")
        abort(403)


############################################################
# 5. /init
############################################################
# 공식 문서 기준
# ----------------------------------------------------------
# Method : GET
# 역할   : Unity scene이 시작되거나 episode가 초기화될 때,
#          시뮬레이터 초기화 정보를 End Point에서 가져간다.
#
# 메뉴 연결
# ----------------------------------------------------------
# - 2.3 Run > Restart:
#   episode 환경을 초기화할 때 /init 요청이 실행된다.
#
# 반환하는 주요 값
# ----------------------------------------------------------
# - startMode:
#   "start" 또는 "pause"
# - blStartX/Y/Z:
#   Blue, 즉 아군 전차 시작 좌표
# - rdStartX/Y/Z:
#   Red, 즉 적 전차 시작 좌표
# - trackingMode:
#   True이면 키보드 기동 대신 /get_action 응답으로 전차를 운용한다.
# - logMode:
#   True이면 시뮬레이터가 /info로 로그 데이터를 전송한다.
# - detectMode:
#   True이면 터렛 이미지가 /detect로 전송된다.
# - stereoCameraMode:
#   True이면 stereo image가 /stereo_image로 전송된다.
############################################################

@app.route("/init", methods=["GET"])
def route_init():
    """Tank Challenge 공식 GET /init endpoint."""

    # commands.py의 init_config()에서 현재 TANK_MODE와 config.py 설정을 읽어
    # 시뮬레이터에 반환할 초기 설정 JSON을 만든다.
    config = init_config()

    # 터미널 로그:
    # 시뮬레이터가 실제로 /init을 호출했는지,
    # 어떤 초기 설정값을 받아가는지 확인하기 위한 출력이다.
    print("[init] config")
    print(pretty(config))

    # 현재 실행 중인 ROS2 bridge node를 가져온다.
    # bridge가 None이면 ROS2 node가 아직 준비되지 않은 상태이다.
    bridge = get_bridge()

    # bridge가 존재하면 /init 설정값을 ROS2 topic으로 publish하고,
    # 내부 latest_state에도 저장한다.
    if bridge:
        bridge.handle_init(config)

    # 시뮬레이터에는 반드시 JSON 형태의 초기 설정값을 반환한다.
    return jsonify(config)


############################################################
# 6. /start
############################################################
# 공식 문서 기준
# ----------------------------------------------------------
# Method : GET
# 역할   : episode가 일시정지 상태일 때 시뮬레이터가 End Point로
#          주기적으로 /start 요청을 보내며 재시작/제어 신호를 확인한다.
#
# 메뉴 연결
# ----------------------------------------------------------
# - 2.3 Run > Start:
#   Start/Pause 상태와 연결된다.
#
# 현재 구현
# ----------------------------------------------------------
# - ROS2에는 start event만 publish한다.
# - 시뮬레이터에는 {"control": ""}를 반환하여 추가 제어 없이 진행한다.
############################################################

@app.route("/start", methods=["GET"])
def route_start():
    """Tank Challenge 공식 GET /start endpoint."""

    # 터미널에서 episode start 요청이 들어왔음을 확인한다.
    print("[start] requested")

    # ROS2 bridge node를 가져온다.
    bridge = get_bridge()

    # bridge가 준비되어 있으면 /tank/api/start/event로 Empty event를 publish한다.
    if bridge:
        bridge.handle_start()

    # 공식 API 응답 형식을 맞추기 위해 control key를 반환한다.
    # 빈 문자열은 별도 pause/reset/start 명령을 내리지 않는다는 의미로 사용한다.
    return jsonify({"control": os.environ.get("TANK_START_CONTROL", "start")})


############################################################
# 7. /info
############################################################
# 공식 문서 기준
# ----------------------------------------------------------
# Method : POST
# 역할   : Log Mode가 활성화되면 시뮬레이터가 전차 로그 데이터를
#          /info URI로 End Point에 전송한다.
#
# 메뉴 연결
# ----------------------------------------------------------
# - 2.3 Run > Log Mode:
#   활성화되면 전차 정보를 /info로 전송한다.
#
# 활용
# ----------------------------------------------------------
# - playerPos, enemyPos, speed, health, turret, body 등 시뮬레이터 상태를 받는다.
# - LiDAR raw payload는 /tank/api/info/raw에 포함되어 lidar 패키지가 후처리한다.
# - A*, 위험도 맵, 상태 모니터링, 데이터 로깅의 기본 입력이다.
# - 이 route는 직접 알고리즘을 수행하지 않고 bridge.handle_info(data)로 넘긴다.
#
# 응답
# ----------------------------------------------------------
# - status/message/control을 반환한다.
# - 공식 문서상 /info 응답의 control 값으로 pause/reset 같은 episode 제어를
#   설계할 수 있다.
############################################################

@app.route("/info", methods=["POST"])
def route_info():
    """Tank Challenge 공식 POST /info endpoint."""

    # 시뮬레이터가 보낸 JSON body를 읽는다.
    # force=True:
    # - Content-Type이 완벽하지 않아도 JSON 파싱을 시도한다.
    #
    # silent=True:
    # - 파싱 실패 시 예외를 던지지 않고 None을 반환한다.
    data = request.get_json(force=True, silent=True)

    # /info는 JSON object(dict)를 기대한다.
    # 잘못된 요청이면 400 Bad Request를 반환한다.
    if not isinstance(data, dict):
        return jsonify({"error": "Invalid request"}), 400

    # 현재 ROS2 bridge node를 가져온다.
    bridge = get_bridge()

    # bridge가 존재하면:
    # - bridge.handle_info(data)가 원본/compact/player/enemy 상태 정보를 정리한다.
    # - LiDAR 세부 처리는 lidar 패키지에서 /tank/api/info/raw를 subscribe하여 수행한다.
    # - 관련 ROS2 topic으로 publish한다.
    #
    # bridge가 없으면:
    # - 최소한 compact_info(data)만 만들어 터미널 출력이 가능하게 한다.
    compact_payload = bridge.handle_info(data) if bridge else _store_fallback_info(data)

    # 터미널에는 /info 전체 원본이 아니라 compact 형태만 출력한다.
    # LiDAR points가 많으면 터미널이 과도하게 길어지기 때문이다.
    print("[info] compact")
    print(pretty(compact_payload.get("data", {})))

    # 시뮬레이터에 성공 응답을 반환한다.
    # control 필드: 대기 중인 에피소드 제어값(reset/pause/start)을 1회 실어 보낸다.
    # TANK_EPISODE_CONTROL이 꺼져 있거나 대기값이 없으면 ""(기존 동작 그대로, 아무 제어도 안 보냄).
    control = bridge.take_episode_control() if bridge else ""
    if control:
        print(f"[info] sending episode control to sim: {control}")
    return jsonify({"status": "success", "message": "Data received", "control": control})


############################################################
# 8. /get_action
############################################################
# 공식 문서 기준
# ----------------------------------------------------------
# Method : POST
# 역할   : Tracking Mode가 활성화되면 시뮬레이터가 현재 전차 상태를 보내고,
#          End Point는 다음 조작 명령을 JSON으로 반환한다.
#
# 키보드 동작과의 관계
# ----------------------------------------------------------
# - 2.2 키보드 동작 기준:
#   W/S : 전진/후진
#   A/D : 좌/우 회전
#   Q/E : 포탑 좌/우 회전
#   R/F : 포탑 상/하 각도
#   SPACE : 발사
#
# - Tracking Mode가 활성화되면 키보드 기동이 비활성화되고,
#   이 /get_action 응답이 전차 조작 입력 역할을 한다.
#
# 반환 명령 형식
# ----------------------------------------------------------
# {
#   "moveWS":   {"command": "W" 또는 "S" 또는 "STOP" 또는 "", "weight": 0.0~1.0},
#   "moveAD":   {"command": "A" 또는 "D" 또는 "",        "weight": 0.0~1.0},
#   "turretQE": {"command": "Q" 또는 "E" 또는 "",        "weight": 0.0~1.0},
#   "turretRF": {"command": "R" 또는 "F" 또는 "",        "weight": 0.0~1.0},
#   "fire": false 또는 true
# }
#
# 현재 구현
# ----------------------------------------------------------
# - bridge.handle_get_action(data)가 ROS2에서 받은 최신 제어 명령을 선택한다.
# - auto 모드에서 최신 명령이 없으면 fallback_command()를 반환한다.
# - monitor 모드에서는 중립 명령을 반환하도록 bridge 쪽에서 처리한다.
############################################################

@app.route("/get_action", methods=["POST"])
def route_get_action():
    """Tank Challenge 공식 POST /get_action endpoint."""

    # 시뮬레이터가 보낸 현재 상태 JSON을 읽는다.
    # 여기에는 position, turret 등 현재 전차 상태가 포함된다.
    data = request.get_json(force=True, silent=True)

    # /get_action도 JSON object(dict)를 기대한다.
    # 잘못된 요청이면 공식 API 스타일로 ERROR 응답을 반환한다.
    if not isinstance(data, dict):
        return jsonify({"status": "ERROR", "message": "Invalid request"}), 400

    # ROS2 bridge node를 가져온다.
    bridge = get_bridge()

    # bridge가 있으면:
    # - 현재 position/turret을 ROS2 topic으로 publish한다.
    # - /tank/control/command에서 받은 최신 명령을 선택한다.
    # - 선택된 명령을 /get_action 응답으로 반환한다.
    #
    # bridge가 없으면:
    # - 안전 fallback 명령을 반환한다.
    if bridge:
        command = bridge.handle_get_action(data)
    else:
        command = fallback_command()
        _store_fallback_get_action(data, command)

    # 실제로 시뮬레이터에 반환하는 명령을 터미널에 출력한다.
    print("🎮 /get_action response")
    print(pretty(command))

    # 시뮬레이터는 이 JSON을 읽어서 전차 이동/포탑/발사를 수행한다.
    return jsonify(command)


############################################################
# 9. /detect
############################################################
# 공식 문서 기준
# ----------------------------------------------------------
# Method : POST
# 역할   : Detect Mode가 활성화되면 시뮬레이터가 터렛 뷰 이미지를
#          image file로 End Point에 전송한다.
#
# 메뉴 연결
# ----------------------------------------------------------
# - 2.3 Run > Detect Mode:
#   활성화되면 터렛 시점 이미지가 /detect URI로 전송된다.
# - 객체탐지는 터렛 시점에서만 활성화된다.
#
# 공식 응답 형식
# ----------------------------------------------------------
# [
#   {
#     "className": "person",
#     "bbox": [10, 10, 50, 50],
#     "confidence": 0.85,
#     "color": "#00FF00",
#     "filled": false,
#     "updateBoxWhileMoving": false
#   }
# ]
#
# 현재 구현
# ----------------------------------------------------------
# - 현재 이 route에서는 YOLO 추론을 직접 수행하지 않는다.
# - image 저장 옵션이 켜져 있으면 파일로 저장한다.
# - detections=[]를 반환한다.
# - 추후 YOLO는 별도 ROS2 node 또는 별도 inference server로 분리하는 것을 권장한다.
############################################################

@app.route("/detect", methods=["POST"])
def route_detect():
    """Tank Challenge 공식 POST /detect endpoint. 선택적 async YOLO와 live view를 지원한다."""

    image = request.files.get("image")
    if image is None:
        return jsonify({"error": "No image received"}), 400

    image_bytes = image.read()
    if not image_bytes:
        return jsonify({"error": "Empty image received"}), 400

    if SAVE_IMAGES:
        IMAGE_DIR.mkdir(parents=True, exist_ok=True)
        path = IMAGE_DIR / f"detect_{int(now_wall() * 1000)}.jpg"
        path.write_bytes(image_bytes)

    # 선택적 web live view를 위해 frame을 저장한다. 여기서는 YOLO를 실행하지 않는다.
    frame_shape = None
    if LIVE_VIEW_ENABLED:
        try:
            frame_shape = live_view.update_frame(image_bytes)
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] live view frame update failed: {exc}")
            frame_shape = None

    bridge = get_bridge()
    if bridge:
        bridge.handle_detect_image(image_bytes, metadata={"route": "/detect"})

    detections = []
    metadata = {}
    if isinstance(frame_shape, list) and len(frame_shape) >= 2:
        metadata["image_shape"] = frame_shape
        metadata["image"] = {"height": frame_shape[0], "width": frame_shape[1]}

    if get_detector is None:
        print(f"[warn] /detect YOLO unavailable: {_YOLO_IMPORT_ERROR}")
    else:
        try:
            if YOLO_ASYNC_ENABLED:
                detections, async_meta = _get_async_yolo_service().enqueue(image_bytes)
                metadata.update(async_meta)
            else:
                detections = get_detector().detect_bytes(image_bytes)
        except Exception as exc:
            print(f"[warn] /detect YOLO inference failed: {exc}")
            detections = []
            metadata["yolo_error"] = str(exc)

    # detector debug metadata가 있으면 추가한다. async 모드에서는 위에서 구한 frame shape를 우선한다.
    if get_detector is not None:
        try:
            debug = get_detector().debug_state()
            debug_shape = debug.get("latestFrameShape")
            if "image_shape" not in metadata and isinstance(debug_shape, list) and len(debug_shape) >= 2:
                metadata["image_shape"] = debug_shape
                metadata["image"] = {"height": debug_shape[0], "width": debug_shape[1]}
            metadata["yolo_detect_ms"] = debug.get("latestDetectMs")
            metadata["yolo_cached"] = debug.get("latestDetectCached")
            metadata["yolo_tracking_enabled"] = debug.get("trackingEnabled")
        except Exception:
            pass

    if LIVE_VIEW_ENABLED:
        try:
            live_view.update_detections(detections, metadata)
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] live view detection update failed: {exc}")

    bridge = get_bridge()
    if bridge:
        bridge.handle_detect_result(detections, metadata=metadata)

    return jsonify(detections)


@app.route("/debug/yolo", methods=["GET"])
def route_debug_yolo():
    """현재 내장 YOLO runtime/debug 상태를 반환한다."""
    if get_detector is None:
        return jsonify({"loaded": False, "importError": str(_YOLO_IMPORT_ERROR)})
    try:
        return jsonify(get_detector().debug_state())
    except Exception as exc:
        return jsonify({"loaded": False, "error": str(exc)}), 500


@app.route("/api/static-map", methods=["GET"])
def route_static_map():
    """browser MFD가 사용하는 static terrain/object map을 반환한다."""
    try:
        return jsonify(_load_static_map_payload())
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 500


@app.route("/api/static-map/overview", methods=["GET"])
def route_static_map_overview():
    """사용 가능한 경우 top-down map overview texture를 반환한다."""
    try:
        overview_path = _resolve_static_map_overview_path()
        if not overview_path.exists():
            return jsonify({"available": False, "error": f"overview image not found: {overview_path}"}), 404
        return send_file(overview_path)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"available": False, "error": str(exc)}), 500


@app.route("/api/dashboard/state", methods=["GET"])
def route_dashboard_state():
    """browser MFD dashboard용으로 합쳐진 state payload."""
    payload: Dict[str, Any] = {
        "serverTime": now_wall(),
        "mode": TANK_MODE,
        "liveView": {},
        "yolo": {},
        "bridge": {},
        "aiLog": [],
        "reconLog": [],
        "sensor": {},
        "staticMap": {},
    }

    try:
        payload["liveView"] = live_view.debug_state() if LIVE_VIEW_ENABLED else {"enabled": False}
        payload["liveView"]["asyncYoloEnabled"] = YOLO_ASYNC_ENABLED
        if YOLO_ASYNC_ENABLED and _ASYNC_YOLO_SERVICE is not None:
            payload["liveView"]["asyncYolo"] = _ASYNC_YOLO_SERVICE.debug_state()
    except Exception as exc:  # noqa: BLE001
        payload["liveView"] = {"error": str(exc)}

    try:
        if get_detector is None:
            payload["yolo"] = {"loaded": False, "importError": str(_YOLO_IMPORT_ERROR)}
        else:
            payload["yolo"] = get_detector().debug_state()
    except Exception as exc:  # noqa: BLE001
        payload["yolo"] = {"loaded": False, "error": str(exc)}

    bridge = None
    try:
        bridge = get_bridge()
        if bridge is None:
            status = ros_status()
            fallback = _fallback_snapshot()
            payload["bridge"] = {
                "available": False,
                "error": status.get("importError") or "ROS bridge is not running",
                "runtime": status,
                "latest": fallback.get("latest", {}),
                "routeCounts": fallback.get("routeCounts", {}),
            }
        elif hasattr(bridge, "get_latest_snapshot"):
            payload["bridge"] = bridge.get_latest_snapshot()
        else:
            payload["bridge"] = {"available": True, "error": "get_latest_snapshot unavailable"}
    except Exception as exc:  # noqa: BLE001
        payload["bridge"] = {"available": bridge is not None, "error": str(exc)}

    latest = payload.get("bridge", {}).get("latest", {})
    if not isinstance(latest, dict):
        latest = {}

    detect_result = latest.get("detect_result") if isinstance(latest.get("detect_result"), dict) else {}
    detections = detect_result.get("detections") if isinstance(detect_result, dict) else []
    if not isinstance(detections, list):
        detections = []
    if not detections and isinstance(payload.get("liveView"), dict):
        live_detections = payload["liveView"].get("latestDetections")
        if isinstance(live_detections, list):
            detections = live_detections

    if isinstance(payload.get("yolo"), dict) and (
        "latestReturnedDetections" not in payload["yolo"] or not payload["yolo"].get("latestReturnedDetections")
    ):
        payload["yolo"]["latestReturnedDetections"] = detections

    for key in ("ai_log", "llm_log", "decision"):
        value = latest.get(key)
        if isinstance(value, list):
            payload["aiLog"] = value
            break
        if value:
            payload["aiLog"] = [value]
            break

    timestamp = detect_result.get("timestamp_wall") if isinstance(detect_result, dict) else None
    payload["reconLog"] = [
        {
            "className": det.get("className") or det.get("class_name") or det.get("modelClassName") or "object",
            "confidence": det.get("confidence"),
            "timestamp": timestamp,
        }
        for det in detections
        if isinstance(det, dict)
    ]

    payload["sensor"] = {
        "rosConnected": bridge is not None and not payload.get("bridge", {}).get("error"),
        "liveViewEnabled": LIVE_VIEW_ENABLED,
        "latestYoloMs": payload.get("yolo", {}).get("latestYoloMs"),
        "latestReturnedDetectionCount": payload.get("yolo", {}).get("latestReturnedDetectionCount"),
        "playerPose": latest.get("player_pose_map") or latest.get("get_action_pose_map"),
        "enemyPose": latest.get("enemy_pose_map"),
        "routeCounts": payload.get("bridge", {}).get("routeCounts") or payload.get("bridge", {}).get("route_counts") or {},
    }

    try:
        static_map = _load_static_map_payload()
        payload["staticMap"] = {
            "loaded": True,
            "terrainIndex": static_map.get("terrainIndex"),
            "objectCount": static_map.get("objectCount"),
            "mapFile": static_map.get("mapFile"),
            "heightSummary": static_map.get("heightSummary"),
            "surfaceSummary": static_map.get("surfaceSummary"),
            "categoryCounts": static_map.get("categoryCounts"),
            "terrainZones": static_map.get("terrainZones"),
        }
    except Exception as exc:  # noqa: BLE001
        payload["staticMap"] = {"loaded": False, "error": str(exc)}

    return jsonify(payload)


@app.route("/view", methods=["GET"])
def route_live_view():
    """최신 /detect image와 detection overlay를 보여주는 browser live view."""
    if not LIVE_VIEW_ENABLED:
        return jsonify({"enabled": False, "error": "TANK_LIVE_VIEW is false"}), 404
    return live_view.render_view_page()


@app.route("/video_feed", methods=["GET"])
def route_video_feed():
    """/view가 사용하는 MJPEG stream."""
    if not LIVE_VIEW_ENABLED:
        return jsonify({"enabled": False, "error": "TANK_LIVE_VIEW is false"}), 404
    return live_view.video_response(web_fps=LIVE_VIEW_FPS, jpeg_quality=LIVE_VIEW_JPEG_QUALITY)


@app.route("/debug/live_view", methods=["GET"])
def route_debug_live_view():
    """현재 live-view와 async YOLO 상태를 반환한다."""
    state = live_view.debug_state() if LIVE_VIEW_ENABLED else {"enabled": False}
    state["asyncYoloEnabled"] = YOLO_ASYNC_ENABLED
    if YOLO_ASYNC_ENABLED and _ASYNC_YOLO_SERVICE is not None:
        try:
            state["asyncYolo"] = _ASYNC_YOLO_SERVICE.debug_state()
        except Exception as exc:  # noqa: BLE001
            state["asyncYolo"] = {"error": str(exc)}
    return jsonify(state)


@app.route("/debug_state", methods=["GET"])
def route_debug_state_alias():
    """팀 live-view debug endpoint에 대한 호환용 alias."""
    return route_debug_live_view()


############################################################
# 10. /stereo_image
############################################################
# 공식 문서 기준
# ----------------------------------------------------------
# Method : POST
# 역할   : Stereo Camera Mode가 활성화되면 시뮬레이터가
#          left_image, right_image를 End Point에 전송한다.
#
# 메뉴 연결
# ----------------------------------------------------------
# - 2.3 Run > Stereo Camera Mode:
#   활성화되면 전차의 stereo camera 이미지가 /stereo_image로 전송된다.
#
# 공식 설명 기준
# ----------------------------------------------------------
# - left_image : 왼쪽 stereo camera 화면 이미지
# - right_image: 오른쪽 stereo camera 화면 이미지
# - stereo camera는 turret view 기준 좌우에 설치되어 있다.
# - 두 카메라 사이 거리는 1.115
# - FoV는 Vertical 28, Horizontal 47.81061
#
# 현재 구현
# ----------------------------------------------------------
# - left/right image 존재 여부를 검증한다.
# - SAVE_IMAGES=True이면 두 이미지를 파일로 저장한다.
# - ROS2 bridge로 status를 전달한다.
############################################################

@app.route("/stereo_image", methods=["POST"])
def route_stereo_image():
    """Tank Challenge 공식 POST /stereo_image endpoint."""

    # 왼쪽 stereo image를 multipart/form-data에서 읽는다.
    left_image = request.files.get("left_image")

    # 오른쪽 stereo image를 multipart/form-data에서 읽는다.
    right_image = request.files.get("right_image")

    # ROS2 bridge node를 미리 가져온다.
    # 오류 status도 bridge로 넘길 수 있기 때문이다.
    bridge = get_bridge()

    # 둘 중 하나라도 없으면 400 error를 반환한다.
    if left_image is None or right_image is None:
        status = {"result": "error", "message": "Left or Right image missing"}

        # 오류 status도 ROS2 topic으로 publish하여 디버깅 가능하게 한다.
        if bridge:
            bridge.handle_stereo_status(status)

        return jsonify(status), 400

    # status는 아래 분기에서 success/error 형태로 채워진다.
    status: Dict[str, Any]

    # SAVE_IMAGES=True이면 stereo image pair를 파일로 저장한다.
    if SAVE_IMAGES:
        # 이미지 저장 디렉터리가 없으면 생성한다.
        IMAGE_DIR.mkdir(parents=True, exist_ok=True)

        # 같은 timestamp를 L/R 파일명에 적용하여 pair 관계를 보존한다.
        stamp = int(now_wall() * 1000)

        # 왼쪽 이미지 저장 경로.
        left_path = IMAGE_DIR / f"stereo_left_{stamp}.jpg"

        # 오른쪽 이미지 저장 경로.
        right_path = IMAGE_DIR / f"stereo_right_{stamp}.jpg"

        # 파일 저장 중 디스크/권한 문제가 생길 수 있으므로 try/except로 감싼다.
        try:
            # 왼쪽 이미지 저장.
            left_image.save(str(left_path))

            # 오른쪽 이미지 저장.
            right_image.save(str(right_path))

            # 저장 성공 status.
            status = {
                "result": "success",
                "left_path": str(left_path),
                "right_path": str(right_path),
            }

        except Exception as exc:
            # 저장 실패 status.
            status = {"result": "error", "message": str(exc)}

            # 실패도 ROS2로 publish한다.
            if bridge:
                bridge.handle_stereo_status(status)

            # 파일 저장 실패는 서버 내부 처리 실패이므로 500을 반환한다.
            return jsonify(status), 500

    # SAVE_IMAGES=False이면 파일 저장 없이 수신 성공만 기록한다.
    else:
        status = {"result": "success", "saved": False}

    # 최종 stereo status를 ROS2 topic으로 publish한다.
    if bridge:
        bridge.handle_stereo_status(status)

    # 시뮬레이터에는 성공 응답을 반환한다.
    return jsonify({"result": "success"})


############################################################
# 11. /update_bullet
############################################################
# 공식 문서 기준
# ----------------------------------------------------------
# Method : POST
# 역할   : 포탄이 충돌한 위치 및 대상 정보를 End Point에 전달한다.
#
# 키보드 동작과의 관계
# ----------------------------------------------------------
# - 2.2 기준 SPACE가 포탄 발사이다.
# - /get_action에서는 fire=true가 SPACE 입력에 해당한다.
# - 발사 후 포탄이 충돌하면 /update_bullet로 충돌 정보가 들어온다.
#
# 활용
# ----------------------------------------------------------
# - 명중 여부 판단
# - 탄착 위치 RViz 표시
# - 보상 함수 설계
# - 적/장애물 타격 이벤트 기록
############################################################

@app.route("/update_bullet", methods=["POST"])
def route_update_bullet():
    """Tank Challenge 공식 POST /update_bullet endpoint."""

    # 포탄 충돌 JSON body를 읽는다.
    data = request.get_json(force=True, silent=True)

    # 포탄 충돌 정보는 JSON object여야 한다.
    if not isinstance(data, dict):
        return jsonify({"status": "ERROR", "message": "Invalid request data"}), 400

    # 터미널에 탄착 좌표와 hit 대상을 출력한다.
    print(
        f"💥 /update_bullet "
        f"x={data.get('x')} "
        f"y={data.get('y')} "
        f"z={data.get('z')} "
        f"hit={data.get('hit')}"
    )

    # bridge가 준비되어 있으면 탄착 정보를 ROS2 topic으로 publish한다.
    bridge = get_bridge()
    if bridge:
        bridge.handle_bullet(data)

    # 시뮬레이터에 수신 성공 응답을 반환한다.
    return jsonify({"status": "OK", "message": "Bullet impact data received"})


############################################################
# 12. /set_destination
############################################################
# 공식 문서 기준
# ----------------------------------------------------------
# Method : POST
# 역할   : Tracking Edit Mode에서 설정한 목적지를 End Point에 전달한다.
#
# 메뉴 연결
# ----------------------------------------------------------
# - 2.3 Setting > Tracking Edit Mode:
#   목적지 설정을 위해 custom view에서 목표 지점을 클릭하면
#   destination 좌표가 설정되고 /set_destination이 호출된다.
#
# 현재 구현
# ----------------------------------------------------------
# - data["destination"]을 "x,y,z" 문자열로 받고 float로 파싱한다.
# - bridge.handle_destination(x, y, z)로 raw/map pose를 생성하고 publish한다.
# - 경로계획 node는 /tank/goal/pose 등을 subscribe해서 목표점으로 사용한다.
############################################################

@app.route("/set_destination", methods=["POST"])
def route_set_destination():
    """Tank Challenge 공식 POST /set_destination endpoint."""

    # destination JSON body를 읽는다.
    data = request.get_json(force=True, silent=True)

    # destination key가 없으면 400 error를 반환한다.
    if not isinstance(data, dict) or "destination" not in data:
        return jsonify({"status": "ERROR", "message": "Missing destination data"}), 400

    # 공식 샘플/시뮬레이터 구현에서 destination은 "x,y,z" 문자열 형태로 들어온다.
    # 이를 float 3개로 파싱한다.
    try:
        x, y, z = [float(v.strip()) for v in str(data["destination"]).split(",")]

    # 파싱 실패 시 format error를 반환한다.
    except Exception as exc:
        return jsonify({"status": "ERROR", "message": f"Invalid format: {exc}"}), 400

    # bridge가 있으면 목적지를 ROS2 topic으로 publish한다.
    bridge = get_bridge()
    pose_raw = bridge.handle_destination(x, y, z) if bridge else {"x": x, "y": y, "z": z}

    # 터미널에 원본 목적지 좌표를 출력한다.
    print(f"🎯 /set_destination raw=({x}, {y}, {z})")

    # 시뮬레이터에 목적지 수신 성공 응답을 반환한다.
    return jsonify({
        "status": "OK",
        "destination": {
            "x": pose_raw["x"],
            "y": pose_raw["y"],
            "z": pose_raw["z"],
        },
    })


############################################################
# 13. /update_obstacle
############################################################
# 공식 문서 기준
# ----------------------------------------------------------
# Method : POST
# 역할   : 시뮬레이터 환경에 추가된 Obstacle 정보를 End Point에 전달한다.
#
# 메뉴 연결
# ----------------------------------------------------------
# - 2.3 Setting > Object Edit Mode:
#   Object를 추가/삭제하면 전체 Obstacle 정보가 /update_obstacle로 전송된다.
#
# 활용
# ----------------------------------------------------------
# - A* 장애물 grid 생성
# - costmap 구성
# - 위험도 맵 구성
# - RViz obstacle marker 표시
############################################################

@app.route("/update_obstacle", methods=["POST"])
def route_update_obstacle():
    """Tank Challenge 공식 POST /update_obstacle endpoint."""

    # obstacle JSON body를 읽는다.
    # obstacle 데이터 구조가 list일 수도 있으므로 dict만 강제하지 않고 None만 검사한다.
    data = request.get_json(force=True, silent=True)

    # body가 없으면 400 error를 반환한다.
    if data is None:
        return jsonify({"status": "error", "message": "No data received"}), 400

    # 터미널에는 장애물 데이터가 들어왔다는 이벤트만 출력한다.
    # 장애물 전체를 출력하면 로그가 길어질 수 있다.
    print("🪨 /update_obstacle received")

    # bridge가 있으면 obstacle raw/list topic으로 publish한다.
    bridge = get_bridge()
    if bridge:
        bridge.handle_obstacles(data)

    # 시뮬레이터에 수신 성공 응답을 반환한다.
    return jsonify({"status": "success", "message": "Obstacle data received"})


############################################################
# 14. /collision
############################################################
# 공식 문서 기준
# ----------------------------------------------------------
# Method : POST
# 역할   : 전차가 obstacle 등과 충돌했을 때 충돌 정보를 End Point로 전달한다.
#
# 공식 예시 구조
# ----------------------------------------------------------
# {
#   "objectName": "Obstacle001(Clone)",
#   "position": {
#     "x": 123.45,
#     "y": 7.89,
#     "z": 98.76
#   }
# }
#
# 활용
# ----------------------------------------------------------
# - 충돌 위치 RViz 표시
# - 장애물 회피 실패 로그
# - 강화학습 penalty 설계
# - 경로계획 알고리즘 성능 평가
############################################################

@app.route("/collision", methods=["POST"])
def route_collision():
    """Tank Challenge 공식 POST /collision endpoint."""

    # 충돌 JSON body를 읽는다.
    data = request.get_json(force=True, silent=True)

    # 충돌 정보는 JSON object여야 한다.
    if not isinstance(data, dict):
        return jsonify({"status": "error", "message": "No collision data received"}), 400

    # 터미널에 충돌 객체 이름과 위치를 출력한다.
    print(f"💥 /collision object={data.get('objectName')} position={data.get('position')}")

    # bridge가 있으면 collision raw/point topic으로 publish한다.
    bridge = get_bridge()
    if bridge:
        bridge.handle_collision(data)

    # 시뮬레이터에 수신 성공 응답을 반환한다.
    return jsonify({"status": "success", "message": "Collision data received"})


############################################################
# 15. /health
############################################################
# 프로젝트 내부 확인용 endpoint
# ----------------------------------------------------------
# 공식 Tank Challenge endpoint는 아니지만, 개발 중 서버 상태 확인에 유용하다.
#
# 사용 예
# ----------------------------------------------------------
# curl http://localhost:5000/health
#
# 반환 내용
# ----------------------------------------------------------
# - Flask 서버가 살아있는지
# - 현재 TANK_MODE가 monitor인지 auto인지
# - ROS2 bridge가 준비되었는지
# - 사용 중인 port
# - ROS2 제어 명령 topic
############################################################

@app.route("/health", methods=["GET"])
def route_health():
    """개발 편의를 위한 GET /health endpoint."""

    # 현재 bridge node 상태를 확인한다.
    bridge = get_bridge()

    # 사람이 curl로 확인하기 쉬운 상태 JSON을 반환한다.
    return jsonify({
        "status": "ok",
        "mode": TANK_MODE,
        "ros_bridge": bridge is not None,
        "port": PORT,
        "command_topic": "/tank/control/command",
        # 에피소드 제어(reset/pause/start) 활성 여부 — 정찰 자동 리셋이 동작하려면 true여야 한다.
        "episode_control": EPISODE_CONTROL_ENABLED,
    })
