# -*- coding: utf-8 -*-
"""
############################################################
# Tank Challenge Flask + ROS2 Bridge Runtime Configuration
############################################################

이 파일의 역할
- Tank Challenge 시뮬레이터와 Flask/ROS2 bridge가 사용할 실행 설정을 한 곳에서 관리한다.
- 다른 모듈(app_routes.py, bridge_node.py, commands.py, utils.py)은 이 파일의 전역 설정값을 import해서 사용한다.
- 팀 프로젝트 진행 중 자주 바꿀 값은 가능하면 이 파일 위쪽에서 수정한다.

공식 문서 기준 연결
- /init endpoint는 Unity 씬 시작 시 시뮬레이션 초기화 정보를 반환한다.
- /init 응답에는 아군/적 전차 시작 위치, trackingMode, detectMode/detactMode,
  logMode, stereoCameraMode, enemyTracking, saveLog, saveLidarData, lux 등이 포함된다.
- Tracking Mode가 켜지면 시뮬레이터 전차는 키보드가 아니라 /get_action 응답 명령으로 제어된다.
- Log Mode가 켜지면 시뮬레이터 상태 로그가 /info endpoint로 전달된다.

주의
- 이 파일은 설정값만 담당한다.
- Flask route 처리 로직은 app_routes.py에서 담당한다.
- ROS2 publisher/subscriber 생성은 bridge_node.py에서 담당한다.
- /get_action 명령 형식과 /init JSON 생성은 commands.py에서 담당한다.
"""

############################################################
# 0. Python standard library imports
############################################################

# os: 환경변수 읽기/설정에 사용한다.
# 예: TANK_MODE=auto, TANK_BRIDGE_PORT=5000 같은 값을 읽는다.
import os

# Path: 로그/이미지 저장 디렉터리를 문자열보다 안전하게 다루기 위해 사용한다.
from pathlib import Path

def load_env_file() -> None:
    """
    프로젝트 루트의 .env 파일을 읽어서 os.environ에 반영한다.
    이미 shell에서 지정한 환경변수는 덮어쓰지 않는다.
    """
    env_path = os.environ.get("TANK_ENV_FILE")

    if env_path:
        candidates = [Path(env_path)]
    else:
        candidates = [
            Path.cwd() / ".env",
            Path(__file__).resolve().parents[3] / ".env",
        ]

    for path in candidates:
        if not path.exists():
            continue

        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                if not line or line.startswith("#"):
                    continue

                if "=" not in line:
                    continue

                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")

                os.environ.setdefault(key, value)

        print(f"[ENV] loaded: {path}")
        break


load_env_file()

############################################################
# 1. OpenMP duplicate runtime workaround
############################################################

# YOLO, torch, numpy, OpenCV 등을 같은 프로세스에서 사용할 때
# libiomp5/libgomp 같은 OpenMP runtime이 중복 로드되며 경고나 오류가 날 수 있다.
# 이 bridge 서버 자체는 YOLO 추론을 직접 수행하지 않는 구조를 권장하지만,
# 나중에 perception 기능을 붙이거나 같은 환경에서 실행할 때를 대비해 기본값을 설정한다.
#
# setdefault를 쓰는 이유:
# - 사용자가 이미 shell에서 KMP_DUPLICATE_LIB_OK 값을 지정했다면 그 값을 존중한다.
# - 지정하지 않았을 때만 "TRUE"를 넣는다.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


############################################################
# 2. Server mode configuration
############################################################

# TANK_MODE는 bridge의 전체 운용 모드를 결정한다.
#
# monitor:
# - 수동/관측 중심 모드.
# - /init 응답에서 trackingMode=False로 설정한다.
# - 키보드 조작 또는 시뮬레이터 자체 조작을 방해하지 않고 /info 데이터를 ROS2 topic으로 중계한다.
# - /get_action 요청이 오더라도 neutral 명령을 반환한다.
#
# auto:
# - 자율제어 중심 모드.
# - /init 응답에서 trackingMode=True로 설정한다.
# - 시뮬레이터가 /get_action으로 현재 위치/포탑 상태를 보내면,
#   ROS2 알고리즘 노드가 /tank/control/command로 보낸 명령을 시뮬레이터에 반환한다.
TANK_MODE = os.environ.get("TANK_MODE", "monitor").strip().lower()

# 지원하지 않는 문자열이 들어오면 안전하게 monitor 모드로 강제한다.
# 예: TANK_MODE=abc처럼 잘못 입력해도 전차가 임의로 움직이지 않게 한다.
if TANK_MODE not in ("monitor", "auto"):
    TANK_MODE = "monitor"


############################################################
# 3. Flask server network configuration
############################################################

# HOST는 Flask 서버가 어느 네트워크 인터페이스에서 요청을 받을지 결정한다.
#
# "0.0.0.0":
# - 모든 네트워크 인터페이스에서 요청을 받는다.
# - Windows 시뮬레이터 PC가 Ubuntu 작업 PC로 접속해야 하므로 일반적으로 이 값을 사용한다.
#
# "127.0.0.1":
# - 같은 PC 내부 요청만 받는다.
# - 시뮬레이터와 서버가 같은 PC에서만 돌 때 사용 가능하다.
HOST = os.environ.get("TANK_BRIDGE_HOST", "0.0.0.0")

# PORT는 Tank Challenge 시뮬레이터가 접속할 Flask endpoint 포트다.
# 시뮬레이터 메뉴의 Request Port 또는 Endpoint Port와 맞춰야 한다.
PORT = int(os.environ.get("TANK_BRIDGE_PORT", "5000"))


############################################################
# 4. ROS2 control command safety configuration
############################################################

# COMMAND_TTL_SEC는 ROS2 제어 명령의 유효 시간이다.
#
# auto 모드에서 알고리즘 노드가 /tank/control/command로 명령을 publish하면
# bridge는 가장 최근 명령을 저장한다.
# 그런데 그 명령이 너무 오래된 상태에서 계속 재사용되면 전차가 의도치 않게 계속 움직일 수 있다.
# 그래서 마지막 명령 수신 시각으로부터 COMMAND_TTL_SEC를 초과하면 fallback 명령으로 대체한다.
#
# 예:
# - 0.5초: 알고리즘 노드가 최소 2Hz 이상 새 명령을 내야 제어 유지.
# - 1.0초: 조금 느슨한 안전 기준.
COMMAND_TTL_SEC = float(os.environ.get("TANK_COMMAND_TTL_SEC", "5.0"))

# AUTO_FALLBACK은 auto 모드에서 최신 ROS2 명령이 없을 때 반환할 안전 명령 정책이다.
#
# neutral:
# - moveWS/moveAD/turretQE/turretRF 모두 빈 문자열.
# - fire=False.
# - 아무 입력도 누르지 않는 상태에 가깝다.
#
# stop:
# - moveWS.command="STOP".
# - 나머지 조향/포탑/발사는 비활성.
# - 자율주행 실험 중 가장 안전한 기본값으로 추천한다.
AUTO_FALLBACK = os.environ.get("TANK_AUTO_FALLBACK", "stop").strip().lower()

# 잘못된 fallback 문자열이 들어오면 안전하게 stop 정책으로 보정한다.
if AUTO_FALLBACK not in ("neutral", "stop"):
    AUTO_FALLBACK = "stop"


############################################################
# 5. Optional local JSON/Image logging
############################################################

# SAVE_JSONL은 bridge가 수신한 주요 이벤트를 로컬 JSONL 파일로 저장할지 결정한다.
#
# false:
# - 기본값.
# - 실시간 실험 중 디스크 사용량을 줄인다.
#
# true:
# - /info, /get_action, /update_bullet, /collision 같은 데이터를 로그로 남긴다.
# - 사후 분석, 디버깅, 강화학습 데이터셋 저장에 유용하다.
SAVE_JSONL = os.environ.get("TANK_SAVE_JSONL", "true").strip().lower() in ("1", "true", "yes", "y")

# JSONL_DIR은 JSONL 로그를 저장할 디렉터리다.
from datetime import datetime
_session_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
JSONL_DIR = Path(os.environ.get("TANK_JSONL_DIR", f"./tank_logs/session_{_session_ts}"))

# SAVE_FULL_INFO는 /info 원본 전체를 저장할지 결정한다.
#
# false:
# - compact_info만 저장한다.
# - LiDAR points 같은 대용량 필드는 개수만 저장해 파일 크기를 줄인다.
#
# true:
# - /info 원본 JSON 전체를 저장한다.
# - LiDAR point까지 모두 남길 수 있어 파일이 매우 커질 수 있다.
SAVE_FULL_INFO = os.environ.get("TANK_SAVE_FULL_INFO", "true").strip().lower() in ("1", "true", "yes", "y")

# SAVE_IMAGES는 /detect 또는 /stereo_image로 들어온 이미지 파일을 로컬에 저장할지 결정한다.
#
# false:
# - 기본값.
# - bridge는 통신 허브 역할만 하고 이미지 저장을 하지 않는다.
#
# true:
# - 터렛 이미지 또는 스테레오 이미지를 IMAGE_DIR에 저장한다.
# - YOLO 학습/디버깅 데이터 확보에 유용하다.
SAVE_IMAGES = os.environ.get("TANK_SAVE_IMAGES", "false").strip().lower() in ("1", "true", "yes", "y")

# IMAGE_DIR은 detect/stereo 이미지 저장 디렉터리다.
IMAGE_DIR = Path(os.environ.get("TANK_IMAGE_DIR", "./tank_images"))


############################################################
# 5-1. YOLO/live-view runtime options
############################################################

# TANK_LIVE_VIEW=true이면 ros_bridge 안에서 /view, /video_feed를 제공한다.
# 이 기능은 YOLO를 다시 실행하지 않고, /detect로 들어온 최신 프레임과
# 이미 계산된 detection 결과를 화면에 표시만 한다.
LIVE_VIEW_ENABLED = os.environ.get("TANK_LIVE_VIEW", "true").strip().lower() in ("1", "true", "yes", "y")
LIVE_VIEW_FPS = float(os.environ.get("TANK_LIVE_VIEW_FPS", "20"))
LIVE_VIEW_JPEG_QUALITY = int(os.environ.get("TANK_LIVE_VIEW_JPEG_QUALITY", "80"))

# TANK_YOLO_ASYNC=true이면 /detect 요청에서 YOLO 완료를 기다리지 않고,
# 백그라운드 worker가 최신 프레임만 처리한다. 기본값은 기존 동기식 안전 동작이다.
YOLO_ASYNC_ENABLED = os.environ.get("TANK_YOLO_ASYNC", "false").strip().lower() in ("1", "true", "yes", "y")
YOLO_ASYNC_MIN_INTERVAL_SEC = float(os.environ.get("TANK_YOLO_ASYNC_MIN_INTERVAL_SEC", "0.0"))
YOLO_ASYNC_MAX_RESULT_AGE_MS = float(os.environ.get("TANK_YOLO_ASYNC_MAX_RESULT_AGE_MS", "300"))


############################################################
# 6. /init default simulator start positions
############################################################

# BLUE_START는 아군 전차 시작 좌표다.
# 공식 /init 응답에서는 blStartX, blStartY, blStartZ로 전달된다.
#
# 좌표는 Unity 시뮬레이터 기준 raw 좌표다.
# - X: 좌우 방향으로 해석
# - Y: 높이 방향으로 해석
# - Z: 전후/진행 평면 방향으로 해석
BLUE_START = (
    float(os.environ.get("TANK_BLUE_START_X", "60.0")),  # blStartX: 아군 전차 시작 X 좌표
    float(os.environ.get("TANK_BLUE_START_Y", "8")),       # blStartY: 아군 전차 시작 Y 좌표(Alt)
    float(os.environ.get("TANK_BLUE_START_Z", "30.0")),    # blStartZ: 아군 전차 시작 Z 좌표(Pos 두 번째 값)
)

# RED_START는 적 전차 시작 좌표다.
RED_START = (
    float(os.environ.get("TANK_RED_START_X", "105.23")),   # rdStartX: 적 전차 시작 X 좌표
    float(os.environ.get("TANK_RED_START_Y", "10")),       # rdStartY: 적 전차 시작 Y 좌표
    float(os.environ.get("TANK_RED_START_Z", "295.0")),    # rdStartZ: 적 전차 시작 Z 좌표
)


############################################################
# 7. /init mode flags from official API
############################################################

# ENABLE_DETECT는 /init의 detectMode/detactMode에 대응한다.
# Detect Mode를 켜면 시뮬레이터가 터렛 뷰 이미지를 /detect endpoint로 보낸다.
# 문서/샘플 간 오타 호환을 위해 commands.py에서 detectMode와 detactMode를 둘 다 보낸다.
ENABLE_DETECT = os.environ.get("TANK_ENABLE_DETECT", "true").strip().lower() in ("1", "true", "yes", "y")

# ENABLE_STEREO는 /init의 stereoCameraMode와 saveStereoCamera에 대응한다.
# Stereo Camera Mode를 켜면 시뮬레이터가 /stereo_image endpoint로 left/right 이미지를 보낼 수 있다.
ENABLE_STEREO = os.environ.get("TANK_ENABLE_STEREO", "false").strip().lower() in ("1", "true", "yes", "y")

# ENABLE_ENEMY_TRACKING은 /init의 enemyTracking에 대응한다.
# 적 전차 tracking 관련 정보를 사용할지 결정하는 플래그다.
ENABLE_ENEMY_TRACKING = os.environ.get("TANK_ENABLE_ENEMY_TRACKING", "false").strip().lower() in ("1", "true", "yes", "y")

# ENABLE_SAVE_SNAPSHOT은 /init의 saveSnapshot에 대응한다.
# 시뮬레이터 측 snapshot 저장 기능을 사용할지 결정한다.
ENABLE_SAVE_SNAPSHOT = os.environ.get("TANK_ENABLE_SAVE_SNAPSHOT", "false").strip().lower() in ("1", "true", "yes", "y")

# ENABLE_SAVE_LOG는 /init의 saveLog에 대응한다.
# 시뮬레이터 자체 로그 저장 기능을 사용할지 결정한다.
# 이 파일의 SAVE_JSONL은 bridge 로컬 저장 옵션이고, ENABLE_SAVE_LOG는 시뮬레이터 측 저장 옵션이다.
ENABLE_SAVE_LOG = os.environ.get("TANK_ENABLE_SAVE_LOG", "false").strip().lower() in ("1", "true", "yes", "y")

# ENABLE_SAVE_LIDAR는 /init의 saveLidarData에 대응한다.
# 시뮬레이터 측 LiDAR 데이터 저장 기능을 사용할지 결정한다.
ENABLE_SAVE_LIDAR = os.environ.get("TANK_ENABLE_SAVE_LIDAR", "false").strip().lower() in ("1", "true", "yes", "y")

# DESTROY_OBSTACLES_ON_HIT는 /init의 destoryObstaclesOnHit에 대응한다.
# 공식 문서에는 destory로 표기된 오타 key가 있으므로 commands.py에서 호환 key를 같이 보낸다.
# True이면 포탄이 장애물에 맞았을 때 장애물 제거 동작을 활성화하는 의미로 사용한다.
DESTROY_OBSTACLES_ON_HIT = os.environ.get("TANK_DESTROY_OBSTACLES_ON_HIT", "true").strip().lower() in ("1", "true", "yes", "y")

# LUX는 /init의 lux에 대응한다.
# 시뮬레이터 조명값 설정으로 사용된다.
LUX = int(os.environ.get("TANK_LUX", "30000"))


############################################################
# 8. Coordinate frame names
############################################################

# UNITY_FRAME은 시뮬레이터 원본 좌표계를 표시하기 위한 ROS frame_id다.
# raw 좌표는 Unity API가 준 x, y, z를 그대로 보존한다.
UNITY_FRAME = "tank_unity_raw"

# MAP_FRAME은 ROS/RViz/2D 경로계획용 좌표계를 표시하기 위한 ROS frame_id다.
# utils.py에서는 다음 기준으로 변환한다.
# - map.x = raw.x
# - map.y = raw.z
# - map.z = raw.y
MAP_FRAME = "tank_map"
