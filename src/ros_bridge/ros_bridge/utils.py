# -*- coding: utf-8 -*-
"""
############################################################
# utils.py
# Tank Challenge Flask + ROS2 Bridge 공통 유틸리티 모듈
############################################################

역할:
- 여러 모듈에서 반복해서 쓰는 공통 함수를 한 곳에 모아 관리한다.
- Flask route, ROS2 bridge node, command 처리 코드가 모두 공통으로 쓰는
  시간 처리, JSON 직렬화, 타입 변환, 좌표 변환, 로그 저장 함수를 제공한다.

공식 문서 기준:
- Tank Challenge API는 /info, /get_action, /update_bullet, /set_destination,
  /collision 등에서 JSON 형태의 위치/상태 데이터를 주고받는다.
- /info는 시뮬레이터가 전송한 로그 데이터를 endpoint로 전달한다.
- /get_action은 전차의 현재 위치와 포탑 정보를 endpoint에 전달하고,
  endpoint는 이동/포탑/발사 명령을 JSON으로 반환한다.
- /detect와 /stereo_image는 이미지 파일을 다루지만,
  본 파일은 이미지 저장이 아니라 공통 JSON/좌표/로그 처리만 담당한다.

좌표 정책:
- raw 좌표:
  시뮬레이터가 보내는 Unity API 원본 x, y, z를 그대로 보존한다.
- map 좌표:
  ROS2/RViz/2D 경로계획에서 쓰기 쉽도록 지상 평면을 x-z로 해석한다.
  변환식은 x=raw.x, y=raw.z, z=raw.y 이다.

주의:
- 이 파일은 알고리즘을 수행하는 파일이 아니다.
- A*, 위험도 맵, YOLO, 제어기 등은 별도 ROS2 node에서 구현하고,
  이 파일은 데이터 정규화/변환/저장 보조 기능만 담당한다.
"""

############################################################
# 1. Python 기본 모듈 import
############################################################

# JSON 데이터를 문자열로 변환하거나 문자열을 JSON으로 다룰 때 사용하는 표준 모듈이다.
import json

# wall-clock time, 즉 실제 PC 기준 현재 시간을 얻기 위해 사용하는 표준 모듈이다.
import time

# 원본 dict/list를 안전하게 복사하여, 이후 값 변경이 원본 데이터에 영향을 주지 않도록 한다.
from copy import deepcopy

# 타입 힌트용 모듈이다.
# Any  : 어떤 타입이든 들어올 수 있다는 의미
# Dict : dictionary 타입
# Tuple: 여러 값을 묶어 반환하는 tuple 타입
from typing import Any, Dict, Tuple

############################################################
# 2. 프로젝트 설정값 import
############################################################

# config.py는 프로젝트 전체 설정을 모아둔 파일이다.
# utils.py는 직접 환경변수를 읽지 않고, config.py에서 정리된 값을 가져와 사용한다.
from .config import (
    JSONL_DIR,       # JSONL 로그 파일을 저장할 디렉터리
    MAP_FRAME,       # ROS/RViz map 좌표계 frame_id 이름
    SAVE_FULL_INFO,  # /info 전체 원본을 compact하지 않고 저장할지 여부
    SAVE_JSONL,      # JSONL 로그 저장 기능 활성화 여부
    UNITY_FRAME,     # Unity 원본 좌표계 frame_id 이름
)


############################################################
# 3. 시간 처리 함수
############################################################

def now_wall() -> float:
    """
    현재 PC 기준 wall-clock 시간을 초 단위 timestamp로 반환한다.

    사용 위치:
    - /info, /get_action, /collision 등 이벤트 payload에 timestamp_wall 추가
    - JSONL 로그 저장 시 기록 시각 저장
    - ROS2 제어 명령의 age 계산

    wall-clock time 의미:
    - 시뮬레이터 내부 time이 아니라, 이 Python 서버가 실행 중인 PC의 실제 시간이다.
    - 공식 FAQ에 따르면 시뮬레이터 시간은 frame 기준으로 계산될 수 있으므로,
      서버 수신 시각과 시뮬레이터 내부 시간은 분리해서 보는 것이 안전하다.
    """
    # time.time()은 1970-01-01 UTC 이후 경과 시간을 float 초 단위로 반환한다.
    return time.time()


############################################################
# 4. 안전한 숫자 변환 함수
############################################################

def to_float(value: Any, default: float = 0.0) -> float:
    """
    입력값을 안전하게 float로 변환한다.

    필요한 이유:
    - 시뮬레이터 JSON에서 x, y, z, speed, health 등이 숫자로 들어오지만,
      None, 문자열, 비정상 값이 들어오는 경우에도 서버가 죽으면 안 된다.
    - ROS2 PoseStamped/PointStamped/Vector3Stamped 메시지는 float 값을 요구한다.
    - 잘못된 값은 default로 대체하여 bridge가 계속 동작하도록 한다.

    Parameters:
    - value:
      float로 변환할 입력값
    - default:
      변환 실패 시 사용할 기본값

    Returns:
    - 변환 성공 시 float(value)
    - value가 None이거나 변환 실패 시 default
    """
    try:
        # None은 float(None)이 불가능하므로 즉시 기본값을 반환한다.
        if value is None:
            return default

        # int, float, 숫자 문자열 등을 float로 변환한다.
        return float(value)

    # TypeError: list/dict 등 float 변환 불가 타입
    # ValueError: "abc" 같은 숫자가 아닌 문자열
    except (TypeError, ValueError):
        # 변환 실패 시 서버를 중단하지 않고 기본값으로 대체한다.
        return default


def to_int(value: Any, default: int = 0) -> int:
    """
    입력값을 안전하게 int로 변환한다.

    사용 위치:
    - LiDAR point 개수처럼 정수 메시지(std_msgs/Int32)로 publish할 때 사용한다.

    Parameters:
    - value:
      int로 변환할 입력값
    - default:
      변환 실패 시 사용할 기본값

    Returns:
    - 변환 성공 시 int(value)
    - value가 None이거나 변환 실패 시 default
    """
    try:
        # None은 int(None)이 불가능하므로 즉시 기본값을 반환한다.
        if value is None:
            return default

        # int, float, 숫자 문자열 등을 int로 변환한다.
        return int(value)

    # TypeError 또는 ValueError가 발생해도 bridge가 죽지 않도록 기본값을 반환한다.
    except (TypeError, ValueError):
        return default


############################################################
# 5. JSON 문자열 변환 함수
############################################################

def dumps(data: Any) -> str:
    """
    Python 객체를 ROS2 String topic에 넣기 좋은 compact JSON 문자열로 변환한다.

    사용 위치:
    - bridge_node.py의 publish_json()
    - /tank/state/latest
    - /tank/api/info/raw
    - /tank/api/get_action/response
    - JSONL 로그 저장

    옵션 설명:
    - ensure_ascii=False:
      한글을 \\uXXXX 형태로 깨지게 보이지 않도록 그대로 저장한다.
    - default=str:
      JSON 직렬화가 어려운 객체가 들어와도 str()로 변환해 에러를 줄인다.
    - separators=(",", ":"):
      불필요한 공백을 줄여 ROS2 String 메시지 크기를 작게 만든다.
    """
    return json.dumps(data, ensure_ascii=False, default=str, separators=(",", ":"))


def pretty(data: Any) -> str:
    """
    터미널 출력용 readable JSON 문자열로 변환한다.

    사용 위치:
    - app_routes.py에서 /init, /info, /get_action 결과를 보기 좋게 출력할 때 사용한다.

    dumps()와 차이:
    - dumps()는 ROS2 topic/로그 저장용 compact JSON
    - pretty()는 사람이 터미널에서 읽기 쉬운 indent JSON
    """
    return json.dumps(data, ensure_ascii=False, default=str, indent=2)


############################################################
# 6. 선택적 JSONL 로그 저장 함수
############################################################

def append_jsonl(filename: str, record: Dict[str, Any]) -> None:
    """
    SAVE_JSONL=true일 때만 record를 JSONL 파일에 append한다.

    JSONL 의미:
    - JSON Lines 형식
    - 한 줄에 JSON 객체 하나씩 저장
    - 나중에 pandas, Python, grep 등으로 분석하기 쉽다.

    사용 예:
    - info.jsonl:
      /info에서 받은 상태 로그 저장
    - get_action.jsonl:
      /get_action request/response 저장
    - bullet.jsonl:
      탄착 정보 저장
    - obstacles.jsonl:
      장애물 정보 저장

    Parameters:
    - filename:
      JSONL_DIR 아래에 생성할 파일명
    - record:
      저장할 dictionary 데이터

    주의:
    - LiDAR points를 전부 저장하면 파일이 매우 커질 수 있다.
    - SAVE_FULL_INFO=false이면 compact_info()로 요약한 정보만 저장하는 구조가 안전하다.
    """
    # 설정에서 JSONL 저장을 꺼둔 경우 아무 작업도 하지 않는다.
    if not SAVE_JSONL:
        return

    # 로그 저장 디렉터리가 없으면 생성한다.
    # parents=True : 중간 경로가 없어도 함께 생성
    # exist_ok=True: 이미 있어도 에러를 내지 않음
    JSONL_DIR.mkdir(parents=True, exist_ok=True)

    # 지정된 파일을 append 모드로 열어 record를 한 줄 JSON으로 저장한다.
    with (JSONL_DIR / filename).open("a", encoding="utf-8") as f:
        # dumps(record)는 compact JSON 문자열을 만든다.
        # 끝에 "\n"을 붙여 JSONL 형식, 즉 한 줄에 하나의 JSON record가 되게 한다.
        f.write(dumps(record) + "\n")


############################################################
# 7. x/y/z 좌표 정규화 함수
############################################################

def as_xyz(obj: Any) -> Dict[str, float]:
    """
    dict 형태의 x/y/z 값을 안전하게 float dict로 정규화한다.

    필요한 이유:
    - 공식 API Docs의 예시처럼 /info에는 playerPos, enemyPos 등이
      {"x": ..., "y": ..., "z": ...} 형태로 들어온다.
    - /get_action, /update_bullet, /collision 등도 위치 정보에 x/y/z를 사용한다.
    - 하지만 누락/None/문자열 값이 들어와도 ROS2 메시지 변환 단계에서 죽지 않도록
      항상 float x/y/z dict로 정리한다.

    Parameters:
    - obj:
      x/y/z key를 가진 dict일 것으로 기대되는 입력값

    Returns:
    - {"x": float, "y": float, "z": float}
    """
    # dict가 아니면 위치 정보가 없다고 보고 빈 dict로 대체한다.
    if not isinstance(obj, dict):
        obj = {}

    # 각 좌표값을 안전하게 float로 변환한다.
    return {
        "x": to_float(obj.get("x", 0.0)),
        "y": to_float(obj.get("y", 0.0)),
        "z": to_float(obj.get("z", 0.0)),
    }


############################################################
# 8. Unity raw 좌표 생성 함수
############################################################

def raw_pose(obj: Any, source: str) -> Dict[str, Any]:
    """
    Unity API 원본 좌표계를 pose dict로 만든다.

    raw 좌표 의미:
    - 시뮬레이터가 endpoint로 보내준 x/y/z를 그대로 보존한 좌표
    - 좌표 검증, 디버깅, 공식 API 데이터 확인용으로 사용한다.

    Parameters:
    - obj:
      시뮬레이터가 보낸 위치 dict
    - source:
      이 좌표가 어느 endpoint/field에서 왔는지 기록하는 문자열
      예: "/info/playerPos", "/get_action/position", "/collision/position"

    Returns:
    - x, y, z:
      Unity 원본 좌표
    - frame_id:
      UNITY_FRAME, 기본값은 "tank_unity_raw"
    - source:
      좌표 출처
    - coordinate:
      좌표계 설명 문자열
    """
    # 입력값을 x/y/z float dict로 정규화한다.
    p = as_xyz(obj)

    # 원본 좌표에 frame/source/coordinate 메타데이터를 붙여 반환한다.
    return {
        **p,
        "frame_id": UNITY_FRAME,
        "source": source,
        "coordinate": "unity_raw_xyz",
    }


############################################################
# 9. Unity raw 좌표 -> ROS/RViz map 좌표 변환 함수
############################################################

def map_pose_from_raw(raw: Dict[str, Any], source: str) -> Dict[str, Any]:
    """
    Unity raw 좌표를 RViz/2D 지도 좌표로 변환한다.

    공식 문서와 Q&A 기준:
    - Terrain은 Unity 좌표 기준으로 300 x 300 크기의 환경으로 구성되어 있다.
    - 시뮬레이터의 position 예시는 x/y/z 형태로 전달된다.
    - Unity에서는 일반적으로 y가 높이축으로 쓰이고, 지상 평면은 x-z로 해석한다.

    우리 프로젝트 좌표 정책:
    - Unity raw:
      raw.x = 좌우 위치
      raw.y = 높이
      raw.z = 전후 위치

    - ROS/RViz map:
      map.x = 좌우 위치 = raw.x
      map.y = 전후 위치 = raw.z
      map.z = 높이     = raw.y

    변환식:
    - x = raw.x
    - y = raw.z
    - z = raw.y

    Parameters:
    - raw:
      raw_pose()가 만든 Unity 원본 좌표 dict
    - source:
      좌표 출처 문자열

    Returns:
    - ROS/RViz에서 사용하기 쉬운 map 좌표 dict
    """
    return {
        # ROS map x축은 Unity raw x축을 그대로 사용한다.
        "x": to_float(raw.get("x", 0.0)),

        # ROS map y축은 지상 평면의 전후 방향으로 사용하기 위해 Unity raw z를 넣는다.
        "y": to_float(raw.get("z", 0.0)),

        # ROS map z축은 높이로 사용하기 위해 Unity raw y를 넣는다.
        "z": to_float(raw.get("y", 0.0)),

        # raw_x/raw_y/raw_z는 변환 후에도 원본 좌표를 추적하기 위한 디버깅용 필드다.
        "raw_x": to_float(raw.get("x", 0.0)),
        "raw_y": to_float(raw.get("y", 0.0)),
        "raw_z": to_float(raw.get("z", 0.0)),

        # ROS2 메시지 header.frame_id에 들어갈 map 좌표계 이름이다.
        "frame_id": MAP_FRAME,

        # 이 좌표가 어떤 endpoint/field에서 만들어졌는지 기록한다.
        "source": source,

        # 사람이 JSON을 볼 때 좌표 변환 방식을 바로 알 수 있게 남기는 설명 필드다.
        "coordinate": "map_x_rawx_y_rawz_z_rawy",
    }


def raw_and_map_pose(obj: Any, source: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    하나의 좌표 입력에서 raw pose와 map pose를 동시에 만든다.

    사용 위치:
    - /info playerPos
    - /info enemyPos
    - /info lidarOrigin
    - /get_action position
    - /update_bullet impact
    - /set_destination destination
    - /collision position

    Returns:
    - rp:
      Unity 원본 좌표 dict
    - mp:
      ROS/RViz map 변환 좌표 dict
    """
    # 먼저 Unity 원본 좌표를 만든다.
    rp = raw_pose(obj, source)

    # 원본 좌표를 기준으로 ROS/RViz map 좌표를 만든다.
    mp = map_pose_from_raw(rp, source)

    # 두 좌표를 함께 반환하여 handler 쪽에서 raw topic과 map topic을 모두 publish할 수 있게 한다.
    return rp, mp


############################################################
# 10. /info compact 변환 함수
############################################################

def compact_info(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    라이다 포인트 전체를 제외하고 /info 핵심 필드만 추린다.

    공식 API 기준:
    - /info는 시뮬레이터에서 전송된 로그 데이터를 수신하는 endpoint이다.
    - /info 데이터에는 playerPos, playerSpeed, playerHealth,
      enemyPos, enemySpeed, enemyHealth, lidarOrigin, lidarRotation,
      lidarPoints 등이 포함될 수 있다.

    compact가 필요한 이유:
    - LiDAR point 배열은 매우 클 수 있다.
    - 터미널에 매번 전체 LiDAR point를 출력하면 가독성이 떨어진다.
    - /tank/state/latest 같은 통합 상태 topic이 너무 커질 수 있다.
    - 그래서 기본적으로 핵심 상태만 남기고, lidarPoints는 개수만 기록한다.

    SAVE_FULL_INFO:
    - true:
      data 전체를 deepcopy해서 반환한다.
    - false:
      핵심 key만 추리고, lidarPoints는 lidarPoints_count로 요약한다.

    Parameters:
    - data:
      /info route로 들어온 원본 JSON dict

    Returns:
    - compact된 /info dict
    """
    # /info body가 dict가 아니면 처리할 수 없으므로 빈 dict를 반환한다.
    if not isinstance(data, dict):
        return {}

    # 설정상 전체 /info 원본을 보존해야 한다면 그대로 깊은 복사해서 반환한다.
    if SAVE_FULL_INFO:
        return deepcopy(data)

    # /info에서 자주 쓰는 핵심 필드 목록이다.
    # 시뮬레이터 버전이나 설정에 따라 일부 key는 없을 수 있으므로,
    # 아래에서 실제 존재하는 key만 골라 복사한다.
    keys = [
        # 시뮬레이터 시간 및 거리 정보
        "time", "distance",

        # 아군 전차 상태
        "playerPos", "playerSpeed", "playerHealth",
        "playerTurretX", "playerTurretY",
        "playerBodyX", "playerBodyY", "playerBodyZ",

        # 적 전차 상태
        "enemyPos", "enemySpeed", "enemyHealth",
        "enemyTurretX", "enemyTurretY",
        "enemyBodyX", "enemyBodyY", "enemyBodyZ",

        # LiDAR 원점 및 회전 정보
        "lidarOrigin", "lidarRotation",
    ]

    # data에 실제 존재하는 key만 깊은 복사해서 compact dict를 만든다.
    out = {k: deepcopy(data[k]) for k in keys if k in data}

    # lidarPoints가 있으면 전체 배열은 제외하고 개수만 저장한다.
    # 장애물/지형 분석에서 point 원본이 필요하면 bridge_node.py에서 별도 topic으로 publish한다.
    if "lidarPoints" in data:
        out["lidarPoints_count"] = len(data.get("lidarPoints") or [])

    # compact된 /info 데이터를 반환한다.
    return out
