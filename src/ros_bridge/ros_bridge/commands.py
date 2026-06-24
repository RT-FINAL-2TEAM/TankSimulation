# -*- coding: utf-8 -*-
"""
############################################################
# commands.py
############################################################

Tank Challenge /get_action 명령과 /init 초기 설정을 생성/검증하는 모듈입니다.

이 파일은 Flask route나 ROS2 publisher를 직접 만들지 않습니다.
대신 아래 두 가지 "데이터 규격"만 책임집니다.

1) /get_action 응답 명령 JSON
   - 시뮬레이터가 Tracking Mode에서 전차를 움직일 때 사용하는 명령입니다.
   - 공식 문서 기준 key:
     moveWS, moveAD, turretQE, turretRF, fire

2) /init 응답 config JSON
   - Unity scene 시작 또는 Restart 시 시뮬레이터가 요청하는 초기 설정입니다.
   - 시작 위치, Tracking Mode, Detect Mode, Log Mode, Stereo Camera Mode 등을 설정합니다.

공식 문서 기준:
- 2.2 키보드 동작
  W/S       : 전진 / 후진
  A/D       : 좌 / 우 회전
  Q/E       : 포탑 좌 / 우 회전
  R/F       : 포탑 상 / 하 각도 조절
  SPACE     : 포탄 발사
  Tracking Mode 활성화 시 키보드 기동은 동작하지 않고 API 응답으로 전차를 운용합니다.

- 2.3 메뉴 구성 및 기능
  Tracking Mode     : API End Point 응답을 통해 전차 운용
  Detect Mode       : /detect로 터렛 시점 이미지 전송
  Log Mode          : /info로 전차 정보 전송
  Stereo Camera Mode: /stereo_image로 좌/우 이미지 전송

- 3.2 API Docs
  POST /get_action  : 현재 위치/포탑 정보를 받고 다음 액션 명령을 반환
  GET  /init        : Unity scene 시작 시 초기 설정 정보 반환
"""

############################################################
# 1. 타입 힌트 import
############################################################

# Any:
# - command/config 안에 문자열, 숫자, bool, dict 등 여러 타입이 섞이므로 사용합니다.
# Dict:
# - Tank Challenge API는 대부분 JSON object(dict) 형태를 사용하므로 사용합니다.
from typing import Any, Dict


############################################################
# 2. 프로젝트 설정(config) import
############################################################

# config.py는 프로젝트에서 자주 바꿀 값을 모아둔 파일입니다.
# 이 파일에서는 /get_action fallback 정책과 /init 초기 설정값을 가져옵니다.
from .config import (
    # AUTO_FALLBACK:
    # - auto 모드에서 최신 ROS2 제어 명령이 없을 때 어떤 안전 명령을 반환할지 결정합니다.
    # - "neutral"이면 아무 키도 누르지 않는 명령을 반환합니다.
    # - "stop"이면 moveWS를 STOP으로 반환합니다.
    AUTO_FALLBACK,

    # BLUE_START:
    # - /init 응답에 들어갈 아군 전차 시작 위치입니다.
    # - 공식 API key: blStartX, blStartY, blStartZ
    BLUE_START,

    # DESTROY_OBSTACLES_ON_HIT:
    # - 포탄이 장애물에 맞았을 때 장애물을 제거할지 여부입니다.
    # - 공식 문서/샘플의 오타 호환을 위해 destoryObstaclesOnHit도 같이 사용합니다.
    DESTROY_OBSTACLES_ON_HIT,

    # ENABLE_DETECT:
    # - /init의 detectMode/detactMode에 들어갈 값입니다.
    # - True이면 Detect Mode를 켜서 시뮬레이터가 /detect로 이미지를 보냅니다.
    ENABLE_DETECT,

    # ENABLE_ENEMY_TRACKING:
    # - /init의 enemyTracking에 들어갈 값입니다.
    # - True이면 적 전차가 아군 전차를 추적하는 설정입니다.
    ENABLE_ENEMY_TRACKING,

    # ENABLE_SAVE_LIDAR:
    # - /init의 saveLidarData에 들어갈 값입니다.
    # - True이면 시뮬레이터가 LiDAR 데이터를 파일로 저장하도록 설정합니다.
    ENABLE_SAVE_LIDAR,

    # ENABLE_SAVE_LOG:
    # - /init의 saveLog에 들어갈 값입니다.
    # - True이면 시뮬레이터 자체 로그 저장 기능을 활성화합니다.
    ENABLE_SAVE_LOG,

    # ENABLE_SAVE_SNAPSHOT:
    # - /init의 saveSnapshot에 들어갈 값입니다.
    # - True이면 터렛 뷰 snapshot 저장 기능을 활성화합니다.
    ENABLE_SAVE_SNAPSHOT,

    # ENABLE_STEREO:
    # - /init의 stereoCameraMode/saveStereoCamera에 들어갈 값입니다.
    # - True이면 스테레오 카메라 이미지 관련 기능을 활성화합니다.
    ENABLE_STEREO,

    # LUX:
    # - /init의 lux에 들어갈 조명값입니다.
    # - 시뮬레이터 환경 밝기 설정에 해당합니다.
    LUX,

    # RED_START:
    # - /init 응답에 들어갈 적 전차 시작 위치입니다.
    # - 공식 API key: rdStartX, rdStartY, rdStartZ
    RED_START,

    # TANK_MODE:
    # - "monitor": 수동/관측 중심. trackingMode=False.
    # - "auto"   : 자율제어 중심. trackingMode=True.
    TANK_MODE,
)

# to_float:
# - 외부 ROS2 topic에서 들어온 weight가 문자열/None 등으로 들어와도 안전하게 float로 변환합니다.
from .utils import to_float


############################################################
# 3. /get_action 기본 명령 생성 함수
############################################################

def neutral_command() -> Dict[str, Any]:
    """
    아무 키도 누르지 않는 중립 명령을 생성합니다.

    공식 /get_action 응답 형식은 유지하되,
    모든 command를 빈 문자열로 두고 weight를 0.0으로 둡니다.

    사용 상황:
    - monitor 모드에서 시뮬레이터를 관측만 할 때
    - 전차를 움직이지 않고 현재 상태를 유지하고 싶을 때

    공식 키보드 대응:
    - moveWS.command == ""      : W/S/STOP 중 아무 입력 없음
    - moveAD.command == ""      : A/D 중 아무 입력 없음
    - turretQE.command == ""    : Q/E 중 아무 입력 없음
    - turretRF.command == ""    : R/F 중 아무 입력 없음
    - fire == False             : SPACE 발사 입력 없음
    """

    return {
        # moveWS:
        # - W: 전진
        # - S: 후진
        # - STOP: 정지
        # - "": 이 코드에서는 입력 없음으로 사용
        "moveWS": {"command": "", "weight": 0.0},

        # moveAD:
        # - A: 전차 좌회전
        # - D: 전차 우회전
        # - "": 조향 입력 없음
        "moveAD": {"command": "", "weight": 0.0},

        # turretQE:
        # - Q: 포탑 좌회전
        # - E: 포탑 우회전
        # - "": 포탑 좌우 입력 없음
        "turretQE": {"command": "", "weight": 0.0},

        # turretRF:
        # - R: 포각 상승
        # - F: 포각 하강
        # - "": 포각 상하 입력 없음
        "turretRF": {"command": "", "weight": 0.0},

        # fire:
        # - True : 포탄 발사
        # - False: 발사하지 않음
        "fire": False,
    }


def stop_command() -> Dict[str, Any]:
    """
    전차 이동을 STOP으로 고정하는 안전 명령을 생성합니다.

    사용 상황:
    - auto 모드에서 최신 ROS2 제어 명령이 끊겼을 때
    - 네트워크 지연, 알고리즘 오류, planner 미동작 상황에서 runaway를 막고 싶을 때

    neutral_command와의 차이:
    - neutral_command는 moveWS.command가 빈 문자열입니다.
    - stop_command는 moveWS.command가 "STOP"입니다.
    - 공식 /get_action 문서에서 moveWS의 STOP은 정지 명령으로 정의됩니다.
    """

    return {
        # moveWS를 STOP으로 두어 전진/후진 입력을 명시적으로 정지시킵니다.
        "moveWS": {"command": "STOP", "weight": 1.0},

        # 조향 입력 없음
        "moveAD": {"command": "", "weight": 0.0},

        # 포탑 좌우 회전 입력 없음
        "turretQE": {"command": "", "weight": 0.0},

        # 포각 상하 입력 없음
        "turretRF": {"command": "", "weight": 0.0},

        # 안전상 fallback에서는 발사하지 않습니다.
        "fire": False,
    }


def fallback_command() -> Dict[str, Any]:
    """
    auto 모드에서 최신 ROS2 명령이 없을 때 사용할 fallback 명령을 선택합니다.

    AUTO_FALLBACK 값은 config.py에서 관리합니다.

    AUTO_FALLBACK == "neutral":
    - 모든 입력을 비웁니다.
    - 시뮬레이터 물리 상태에 따라 관성/기존 상태가 남을 수 있습니다.

    AUTO_FALLBACK == "stop":
    - moveWS를 STOP으로 명시합니다.
    - 프로젝트 안전 기준에서는 stop을 기본값으로 추천합니다.
    """

    # Python 삼항 연산자:
    # - 조건이 참이면 neutral_command()
    # - 조건이 거짓이면 stop_command()
    return neutral_command() if AUTO_FALLBACK == "neutral" else stop_command()


############################################################
# 4. /get_action 명령 유효성 검사 함수
############################################################

def validate_action_command(command: Dict[str, Any]) -> None:
    """
    ROS2 알고리즘 노드가 보낸 제어 명령이 공식 /get_action 응답 형식에 맞는지 검사합니다.

    이 함수는 bridge_node.py에서 다음 topic을 받을 때 사용됩니다.
    - /tank/control/command
    - /tank/api/get_action/override
    - /tank/action_override

    검사 목적:
    - 잘못된 JSON이 시뮬레이터로 그대로 전달되는 것을 방지
    - moveWS/moveAD/turretQE/turretRF/fire 누락 방지
    - 허용되지 않은 command 문자열 방지
    - weight 범위 오류 방지
    - fire가 bool이 아닌 경우 방지

    정상일 때:
    - 아무 값도 return하지 않습니다.
    - Python 관례상 return None입니다.

    비정상일 때:
    - ValueError를 발생시킵니다.
    - bridge_node.py의 callback에서 이 예외를 잡아 로그로 출력합니다.
    """

    # command는 JSON object여야 하므로 Python dict인지 먼저 확인합니다.
    if not isinstance(command, dict):
        raise ValueError("command must be a JSON object")

    # 공식 /get_action 응답에서 반드시 포함되어야 하는 최상위 key 목록입니다.
    required = ["moveWS", "moveAD", "turretQE", "turretRF", "fire"]

    # 필수 key가 하나라도 빠지면 시뮬레이터가 명령을 해석할 수 없으므로 오류 처리합니다.
    for key in required:
        if key not in command:
            raise ValueError(f"missing key: {key}")

    # 각 제어축별로 허용되는 command 문자열을 정의합니다.
    # 빈 문자열("")은 이 프로젝트에서 "해당 축 입력 없음"으로 허용합니다.
    allowed = {
        # moveWS:
        # - W    : 전진
        # - S    : 후진
        # - STOP : 정지
        # - ""   : 입력 없음
        "moveWS": {"", "W", "S", "STOP"},

        # moveAD:
        # - A  : 전차 좌회전
        # - D  : 전차 우회전
        # - "" : 조향 입력 없음
        "moveAD": {"", "A", "D"},

        # turretQE:
        # - Q  : 포탑 좌회전
        # - E  : 포탑 우회전
        # - "" : 포탑 좌우 입력 없음
        "turretQE": {"", "Q", "E"},

        # turretRF:
        # - R  : 포각 상승
        # - F  : 포각 하강
        # - "" : 포각 상하 입력 없음
        "turretRF": {"", "R", "F"},
    }

    # allowed에 정의된 4개 제어축을 하나씩 검사합니다.
    for key, allowed_values in allowed.items():
        # 예: command["moveWS"] = {"command": "W", "weight": 0.5}
        part = command[key]

        # 각 제어축은 command/weight를 가진 dict여야 합니다.
        if not isinstance(part, dict):
            raise ValueError(f"{key} must be an object")

        # command 문자열과 weight 가중치가 모두 있어야 합니다.
        if "command" not in part or "weight" not in part:
            raise ValueError(f"{key} must include command and weight")

        # command 값을 문자열로 변환합니다.
        # 예: None이나 숫자가 들어와도 str로 바뀌지만, allowed 검사에서 걸러집니다.
        cmd = str(part.get("command", ""))

        # 공식 문서/프로젝트 기준 허용 command인지 검사합니다.
        if cmd not in allowed_values:
            raise ValueError(f"invalid {key}.command: {cmd}")

        # weight는 기존 물리량에 곱해지는 가중치 개념입니다.
        # 공식 문서에는 0.1~1.0 범위가 언급되지만,
        # 이 프로젝트에서는 "입력 없음"을 표현하기 위해 0.0도 허용합니다.
        weight = to_float(part.get("weight"), -1.0)

        # 0.0보다 작거나 1.0보다 크면 비정상 명령으로 처리합니다.
        if weight < 0.0 or weight > 1.0:
            raise ValueError(f"invalid {key}.weight: {weight}; use 0.0~1.0")

    # fire는 포탄 발사 여부이므로 반드시 bool이어야 합니다.
    # 문자열 "true"나 숫자 1은 허용하지 않습니다.
    if not isinstance(command["fire"], bool):
        raise ValueError("fire must be boolean")


############################################################
# 5. /init 초기 설정 JSON 생성 함수
############################################################

def init_config() -> Dict[str, Any]:
    """
    Tank Challenge 시뮬레이터가 GET /init에서 요구하는 초기 설정 JSON을 생성합니다.

    공식 API Docs 기준 /init 역할:
    - Unity scene이 시작될 때 시뮬레이션 초기화 정보를 설정합니다.
    - 아군/적 전차 시작 위치
    - 에피소드 시작/중지 상태
    - Tracking Mode, Detect Mode, Log Mode, Stereo Camera Mode 등 기능 플래그
    - 조명값, 장애물 파괴 여부 등을 포함합니다.

    이 프로젝트 기준:
    - TANK_MODE == "monitor"이면 trackingMode=False
      → 키보드/수동 조작 또는 관측 중심
      → Log Mode를 통해 /info를 받는 구조

    - TANK_MODE == "auto"이면 trackingMode=True
      → 시뮬레이터가 /get_action 응답을 이용해 전차를 운용
      → ROS2 알고리즘 노드가 /tank/control/command로 명령을 공급
    """

    # Tracking Mode 여부는 TANK_MODE 하나로 통일해서 관리합니다.
    # auto 모드이면 API 응답으로 전차를 움직이므로 trackingMode=True입니다.
    tracking = TANK_MODE == "auto"

    return {
        ####################################################
        # 5.1 에피소드 시작 모드
        ####################################################

        # startMode:
        # - "start": 에피소드 시작 시 진행 상태
        # - "pause": 에피소드 시작 시 중지 상태
        # 현재 프로젝트는 시뮬레이터를 바로 진행시키기 위해 "start"를 사용합니다.
        "startMode": "start",

        ####################################################
        # 5.2 아군(Blue) 전차 시작 위치
        ####################################################

        # blStartX:
        # - Blue, 즉 아군 전차 시작 위치 X 좌표
        "blStartX": BLUE_START[0],

        # blStartY:
        # - 아군 전차 시작 위치 Y 좌표
        # - Unity 기준으로 높이/고도 성격의 축으로 사용됩니다.
        "blStartY": BLUE_START[1],

        # blStartZ:
        # - 아군 전차 시작 위치 Z 좌표
        "blStartZ": BLUE_START[2],

        ####################################################
        # 5.3 적군(Red) 전차 시작 위치
        ####################################################

        # rdStartX:
        # - Red, 즉 적 전차 시작 위치 X 좌표
        "rdStartX": RED_START[0],

        # rdStartY:
        # - 적 전차 시작 위치 Y 좌표
        "rdStartY": RED_START[1],

        # rdStartZ:
        # - 적 전차 시작 위치 Z 좌표
        "rdStartZ": RED_START[2],

        ####################################################
        # 5.4 실행 모드 플래그
        ####################################################

        # trackingMode:
        # - True이면 키보드 기동이 비활성화되고 API End Point 응답으로 전차를 운용합니다.
        # - False이면 수동 조작/관측 중심으로 운용합니다.
        "trackingMode": tracking,

        # detectMode:
        # - True이면 시뮬레이터가 터렛 시점 이미지를 /detect URI로 보냅니다.
        # - endpoint는 bbox/className/confidence 등의 탐지 결과를 반환할 수 있습니다.
        "detectMode": ENABLE_DETECT,

        # detactMode:
        # - 공식 문서 일부/기존 샘플에서 보이는 오타 호환용 key입니다.
        # - detectMode와 같은 값을 함께 보내서 버전 차이로 인한 문제를 줄입니다.
        "detactMode": ENABLE_DETECT,

        # logMode:
        # - True이면 시뮬레이터가 전차 상태 정보를 /info URI로 보냅니다.
        # - 자율주행, 위험도 맵, 디버깅 로그 수집의 핵심 데이터 소스입니다.
        # - 이 프로젝트에서는 monitor/auto 모두 True로 둡니다.
        "logMode": True,

        # stereoCameraMode:
        # - True이면 스테레오 카메라 이미지가 /stereo_image URI로 전송됩니다.
        # - left_image/right_image를 이용해 거리 추정이나 인식 알고리즘을 붙일 수 있습니다.
        "stereoCameraMode": ENABLE_STEREO,

        # enemyTracking:
        # - True이면 적 전차가 아군 전차를 따라 이동하는 설정입니다.
        "enemyTracking": ENABLE_ENEMY_TRACKING,

        ####################################################
        # 5.5 저장 옵션
        ####################################################

        # saveSnapshot:
        # - True이면 터렛 뷰 이미지를 시뮬레이터 PC에 저장합니다.
        # - 객체탐지 학습/검증 데이터 생성에 사용할 수 있습니다.
        "saveSnapshot": ENABLE_SAVE_SNAPSHOT,

        # saveStereoCamera:
        # - True이면 스테레오 좌/우 이미지를 시뮬레이터 PC에 저장합니다.
        # - ENABLE_STEREO와 같은 값으로 맞춰 관리합니다.
        "saveStereoCamera": ENABLE_STEREO,

        # saveLog:
        # - True이면 시뮬레이터 자체 로그 저장 기능을 활성화합니다.
        # - 이 bridge의 JSONL 저장 옵션과는 별도입니다.
        "saveLog": ENABLE_SAVE_LOG,

        # saveLidarData:
        # - True이면 시뮬레이터가 LiDAR raw 데이터를 파일로 저장합니다.
        # - ROS2 topic으로 받는 LiDAR 데이터 publish와는 별도 기능입니다.
        "saveLidarData": ENABLE_SAVE_LIDAR,

        ####################################################
        # 5.6 환경 및 장애물 옵션
        ####################################################

        # lux:
        # - 시뮬레이터 조명값입니다.
        # - 시각 기반 탐지 성능 실험 시 조도 조건으로 활용할 수 있습니다.
        "lux": LUX,

        # destoryObstaclesOnHit:
        # - 공식 샘플/문서에서 사용된 오타 key입니다.
        # - 장애물 피격 시 제거 여부를 의미합니다.
        "destoryObstaclesOnHit": DESTROY_OBSTACLES_ON_HIT,

        # destroyObstaclesOnHit:
        # - 올바른 철자의 호환용 key입니다.
        # - 시뮬레이터 버전 차이를 고려해 두 key를 모두 보냅니다.
        "destroyObstaclesOnHit": DESTROY_OBSTACLES_ON_HIT,
    }
