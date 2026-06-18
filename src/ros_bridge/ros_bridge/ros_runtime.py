# -*- coding: utf-8 -*-
"""
############################################################
# ROS2 lifecycle management
############################################################

이 파일의 역할
- Flask 서버와 ROS2 node를 같은 Python 프로세스 안에서 함께 실행한다.
- rclpy를 초기화하고 종료한다.
- RosBridge node를 생성한다.
- MultiThreadedExecutor를 별도 daemon thread에서 spin한다.
- Flask route(app_routes.py)가 현재 실행 중인 RosBridge 인스턴스에 접근할 수 있게 한다.

왜 이 파일이 필요한가?
- Flask의 app.run()은 HTTP 요청을 계속 기다리는 blocking 함수이다.
- ROS2의 executor.spin()도 ROS2 callback을 계속 기다리는 blocking 함수이다.
- 둘 다 같은 main thread에서 실행하면 하나만 동작하고 다른 하나는 멈춘다.
- 그래서 ROS2 executor는 별도 background thread에서 돌리고,
  Flask는 main thread에서 실행하는 구조를 사용한다.

전체 실행 흐름
1. main.py에서 start_ros() 호출
2. start_ros()가 rclpy.init() 실행
3. RosBridge node 생성
4. MultiThreadedExecutor 생성
5. executor에 RosBridge node 등록
6. executor.spin()을 daemon thread에서 실행
7. main.py에서 Flask app.run() 실행
8. 프로그램 종료 시 stop_ros()가 executor/node/rclpy를 정리

관련 파일
- main.py        : start_ros()와 stop_ros()를 호출하는 실행 진입점
- app_routes.py  : get_bridge()로 RosBridge 인스턴스를 받아 route 처리 중 ROS2 publish 호출
- bridge_node.py : 실제 ROS2 publisher/subscriber/timer를 가진 RosBridge node 정의
"""

############################################################
# 1. Python standard library imports
############################################################

# threading:
# - Flask 서버와 ROS2 executor를 동시에 돌리기 위해 사용한다.
# - 여기서는 ROS2 executor.spin()을 별도 thread에서 실행한다.
import threading

# Optional:
# - bridge/executor/spin_thread가 아직 생성되지 않았을 때 None일 수 있음을 type hint로 표현한다.
from typing import Any, Optional


############################################################
# 2. ROS2 Python client library imports
############################################################

# rclpy:
# - ROS2 Python client library이다.
# - rclpy.init(), rclpy.ok(), rclpy.shutdown()으로 ROS2 lifecycle을 관리한다.
try:
    import rclpy
except Exception as exc:  # pragma: no cover - runtime environment guard
    rclpy = None
    _ROS_IMPORT_ERROR = exc
else:
    _ROS_IMPORT_ERROR = None

# MultiThreadedExecutor:
# - ROS2 callback을 처리하는 executor이다.
# - 여러 callback을 병렬 처리할 수 있다.
# - 이 프로젝트에서는 Flask 요청 처리와 ROS2 subscriber/timer callback이 동시에 들어올 수 있으므로
#   SingleThreadedExecutor보다 MultiThreadedExecutor가 안전하다.
if rclpy is not None:
    try:
        from rclpy.executors import MultiThreadedExecutor
    except Exception as exc:  # pragma: no cover - runtime environment guard
        MultiThreadedExecutor = None
        _ROS_IMPORT_ERROR = exc
else:
    MultiThreadedExecutor = None


############################################################
# 3. Project imports
############################################################

# RosBridge:
# - bridge_node.py에 정의된 ROS2 Node 클래스이다.
# - 시뮬레이터 데이터를 ROS2 topic으로 publish하고,
#   ROS2 제어 명령 topic을 subscribe하는 핵심 node이다.
if rclpy is not None and MultiThreadedExecutor is not None:
    try:
        from .bridge_node import RosBridge
    except Exception as exc:  # pragma: no cover - runtime environment guard
        RosBridge = None
        _ROS_IMPORT_ERROR = exc
else:
    RosBridge = None

# ROS_EXECUTOR_THREADS:
# - MultiThreadedExecutor에서 사용할 thread 개수이다.
# - 팀원이 config.py에서 쉽게 바꿀 수 있도록 전역 설정으로 분리하는 것을 권장한다.
# - 만약 현재 config.py에 없다면, 아래 import 대신 num_threads=4를 그대로 써도 된다.
try:
    from .config import ROS_EXECUTOR_THREADS
except ImportError:
    ROS_EXECUTOR_THREADS = 4


############################################################
# 4. Global ROS2 runtime objects
############################################################

# bridge:
# - 현재 실행 중인 RosBridge node 인스턴스를 저장한다.
# - app_routes.py에서 get_bridge()를 통해 이 객체에 접근한다.
# - 아직 start_ros()가 호출되지 않았으면 None이다.
bridge: Optional[Any] = None

# executor:
# - ROS2 callback을 처리하는 executor 객체이다.
# - RosBridge node의 subscriber callback, timer callback 등을 실행한다.
# - 아직 start_ros()가 호출되지 않았으면 None이다.
executor: Optional[Any] = None

# spin_thread:
# - executor.spin()을 실행하는 background thread이다.
# - daemon=True로 생성하므로 main process가 종료될 때 함께 정리된다.
# - 아직 start_ros()가 호출되지 않았으면 None이다.
spin_thread: Optional[threading.Thread] = None


############################################################
# 5. Bridge accessor
############################################################

def get_bridge() -> Optional[Any]:
    """
    현재 실행 중인 RosBridge 인스턴스를 반환한다.

    사용 위치
    - app_routes.py의 각 Flask route에서 호출한다.
    - 예: /info route가 시뮬레이터 JSON을 받으면,
      bridge = get_bridge()로 node를 얻고 bridge.handle_info(data)를 호출한다.

    반환값
    - RosBridge 객체: ROS2 node가 정상 실행 중인 경우
    - None: 아직 start_ros()가 호출되지 않았거나 종료된 경우

    왜 전역 bridge를 직접 import하지 않고 함수로 가져오나?
    - 전역 변수는 start_ros()/stop_ros()에서 바뀐다.
    - 함수로 감싸면 다른 모듈이 runtime 상태를 더 명확하게 조회할 수 있다.
    """

    # 현재 module 전역 변수 bridge를 그대로 반환한다.
    return bridge


def ros_status() -> dict:
    """Return the current ROS2 runtime status for dashboard/debug routes."""
    return {
        "available": bridge is not None,
        "importError": None if _ROS_IMPORT_ERROR is None else str(_ROS_IMPORT_ERROR),
        "executorRunning": spin_thread is not None and spin_thread.is_alive(),
    }


############################################################
# 6. ROS2 startup
############################################################

def start_ros() -> None:
    """
    ROS2 node와 executor thread를 시작한다.

    호출 위치
    - main.py의 main() 함수 시작 부분에서 호출한다.

    실행 순서
    1. 이미 bridge가 있으면 중복 실행을 막고 return
    2. rclpy가 아직 초기화되지 않았으면 rclpy.init()
    3. RosBridge node 생성
    4. MultiThreadedExecutor 생성
    5. executor에 RosBridge node 등록
    6. executor.spin()을 daemon thread로 실행

    왜 중복 실행을 막나?
    - rclpy.init()과 Node 생성이 중복되면 lifecycle 오류가 발생할 수 있다.
    - Flask debug reload나 재호출 상황에서도 안전하게 동작하도록 방어한다.
    """

    # 이 함수 안에서 module 전역 변수 bridge/executor/spin_thread를 갱신하겠다는 선언이다.
    global bridge, executor, spin_thread

    if _ROS_IMPORT_ERROR is not None or rclpy is None or MultiThreadedExecutor is None or RosBridge is None:
        print(f"[ROS] disabled: {_ROS_IMPORT_ERROR}")
        print("[ROS] Flask routes remain available; /api/dashboard/state will report bridge unavailable.")
        return

    # 이미 RosBridge node가 생성되어 있다면 중복으로 ROS2를 시작하지 않는다.
    if bridge is not None:
        return

    # rclpy.ok()가 False이면 아직 ROS2 Python client가 초기화되지 않은 상태이다.
    if not rclpy.ok():
        # ROS2 Python client library를 초기화한다.
        # 이 작업 이후에 Node 생성, publisher/subscriber 생성이 가능하다.
        rclpy.init()

    # RosBridge node를 생성한다.
    # 이 시점에 bridge_node.py의 __init__이 실행되며,
    # publisher/subscriber/timer가 모두 등록된다.
    bridge = RosBridge()

    # MultiThreadedExecutor를 생성한다.
    # 여러 ROS2 callback을 병렬 처리할 수 있다.
    # 예: /tank/control/command subscriber callback과 latest_state timer callback.
    executor = MultiThreadedExecutor(num_threads=ROS_EXECUTOR_THREADS)

    # executor가 RosBridge node의 callback들을 관리하도록 node를 등록한다.
    executor.add_node(bridge)

    # executor.spin()은 blocking 함수이므로 main thread에서 직접 실행하면 Flask 서버가 시작되지 못한다.
    # 따라서 별도 background thread를 만든다.
    spin_thread = threading.Thread(
        target=executor.spin,  # thread가 시작되면 executor.spin()을 실행한다.
        daemon=True,           # main process 종료 시 이 thread도 함께 종료될 수 있게 한다.
    )

    # background thread를 시작한다.
    # 이 이후부터 ROS2 subscriber/timer callback이 처리되기 시작한다.
    spin_thread.start()


############################################################
# 7. ROS2 shutdown
############################################################

def stop_ros() -> None:
    """
    ROS2 executor, node, rclpy를 안전하게 종료한다.

    호출 위치
    - main.py에서 Flask app.run()이 끝난 뒤 finally 블록에서 호출한다.
    - Ctrl+C, 서버 종료, 예외 종료 상황에서도 자원 정리를 수행한다.

    종료 순서
    1. executor.shutdown()
    2. bridge.destroy_node()
    3. rclpy.shutdown()
    4. 전역 변수 None 초기화

    왜 순서가 중요한가?
    - 먼저 executor를 멈춰 callback 실행을 중단한다.
    - 그 다음 node를 destroy해서 publisher/subscriber/timer 자원을 정리한다.
    - 마지막으로 rclpy.shutdown()으로 ROS2 client library를 종료한다.
    """

    # 이 함수 안에서 module 전역 변수 bridge/executor/spin_thread를 갱신하겠다는 선언이다.
    global bridge, executor, spin_thread

    # executor가 존재하면 callback 처리를 중단한다.
    if executor is not None:
        executor.shutdown()

    # RosBridge node가 존재하면 ROS2 node 자원을 해제한다.
    if bridge is not None:
        bridge.destroy_node()

    # rclpy가 아직 살아 있으면 ROS2 Python client library를 종료한다.
    if rclpy is not None and rclpy.ok():
        rclpy.shutdown()

    # 종료 후 재시작 가능하도록 전역 참조를 None으로 초기화한다.
    bridge = None

    # executor 참조도 제거한다.
    executor = None

    # spin thread 참조도 제거한다.
    # daemon thread이므로 보통 별도 join 없이 process 종료와 함께 정리된다.
    spin_thread = None
