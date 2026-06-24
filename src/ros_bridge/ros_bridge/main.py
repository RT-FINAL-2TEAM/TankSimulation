# -*- coding: utf-8 -*-
"""
############################################################
# main.py
############################################################

Tank Challenge Flask Endpoint + ROS2 Bridge의 "실행 진입점" 파일입니다.

이 파일의 역할:
1. ROS2 bridge node를 먼저 시작한다.
2. Flask 서버를 실행해서 Tank Challenge 시뮬레이터의 HTTP 요청을 받는다.
3. 서버가 종료될 때 ROS2 node/executor/rclpy 자원을 안전하게 정리한다.

공식 문서 기준 실행 흐름:
- Tank Challenge 시뮬레이터는 Endpoint 서버에 HTTP 요청을 보낸다.
- /init       : 시뮬레이션 초기 설정을 요청한다.
- /start      : 시뮬레이션 시작 이벤트를 알린다.
- /info       : Log Mode에서 전차/센서 상태를 전송한다.
- /get_action : Tracking Mode에서 전차 제어 명령을 요청한다.
- /detect, /stereo_image, /update_bullet, /collision 등도 상황에 따라 호출된다.

이 main.py는 위 endpoint 자체를 직접 구현하지 않는다.
endpoint 구현은 app_routes.py에 있고,
main.py는 app_routes.py의 Flask app을 실제로 실행하는 역할만 한다.
"""

############################################################
# 1. 모듈 import
############################################################

# app_routes.py에서 Flask app 객체를 가져온다.
# app_routes.py 안에는 /init, /start, /info, /get_action 등
# Tank Challenge 공식 API endpoint route 함수들이 등록되어 있다.
from .app_routes import app

# config.py에서 실행에 필요한 전역 설정값을 가져온다.
# HOST      : Flask 서버가 바인딩할 IP 주소
# PORT      : Flask 서버 포트
# TANK_MODE : monitor 또는 auto 실행 모드
from .config import HOST, PORT, TANK_MODE

# ros_runtime.py에서 ROS2 lifecycle 관리 함수를 가져온다.
# start_ros(): rclpy 초기화, RosBridge node 생성, executor thread 시작
# stop_ros() : executor 종료, node destroy, rclpy shutdown
from .ros_runtime import start_ros, stop_ros


############################################################
# 2. main 함수
############################################################

def main() -> None:
    """
    Tank Bridge 전체 시스템을 실행하는 main 함수.

    실행 순서:
    1. start_ros()로 ROS2 bridge node를 시작한다.
    2. Flask app.run()으로 HTTP endpoint 서버를 연다.
    3. 종료 시 finally 블록에서 stop_ros()를 호출해 ROS2 자원을 정리한다.

    왜 ROS2를 먼저 시작하는가?
    - Flask route가 시뮬레이터 요청을 받으면 get_bridge()를 통해
      현재 실행 중인 RosBridge node에 접근한다.
    - 따라서 Flask 서버가 요청을 받기 전에 ROS2 bridge node가 준비되어 있어야 한다.

    왜 finally에서 stop_ros()를 호출하는가?
    - Ctrl+C, Flask 서버 종료, 예외 발생 등 어떤 상황에서도
      ROS2 executor와 node를 안전하게 정리하기 위해서이다.
    """

    ########################################################
    # 2-1. ROS2 bridge node 시작
    ########################################################

    # ROS2를 먼저 시작한다.
    # 내부적으로 ros_runtime.py의 start_ros()는 다음 일을 수행한다.
    # - rclpy.init()
    # - RosBridge node 생성
    # - MultiThreadedExecutor 생성
    # - executor.add_node(bridge)
    # - 별도 daemon thread에서 executor.spin() 실행
    start_ros()

    ########################################################
    # 2-2. Flask 서버 실행
    ########################################################

    # try/finally 구조를 사용한다.
    # 이유:
    # - app.run()은 서버가 실행되는 동안 blocking 상태로 동작한다.
    # - 서버가 종료될 때 finally가 실행되어 stop_ros()로 ROS2 자원을 정리한다.
    try:
        ####################################################
        # 2-2-1. 실행 정보 터미널 출력
        ####################################################

        # 터미널에서 bridge 서버가 정상 실행되었는지 보기 쉽게 구분선을 출력한다.
        print("============================================================")

        # 현재 프로그램의 역할을 표시한다.
        print("Tank Challenge Flask Endpoint + ROS2 Bridge")

        # 현재 실행 모드를 출력한다.
        # monitor:
        #   - trackingMode=False
        #   - logMode=True
        #   - /info 중심으로 관측 데이터를 수집한다.
        #
        # auto:
        #   - trackingMode=True
        #   - logMode=True
        #   - /get_action 중심으로 ROS2 제어 명령을 시뮬레이터에 반환한다.
        print(f"Mode: {TANK_MODE}")

        # Flask 서버가 어느 주소와 포트에서 대기하는지 출력한다.
        # Windows 시뮬레이터 PC는 이 Ubuntu 작업 PC의 IP와 PORT로 접속해야 한다.
        #
        # 예:
        #   HOST=0.0.0.0, PORT=5000이면
        #   외부 PC는 Ubuntu 작업 PC의 실제 IP:5000으로 접속한다.
        print(f"Listening: http://{HOST}:{PORT}")

        # ROS2 알고리즘 노드가 제어 명령을 보내야 하는 topic을 표시한다.
        # 이 topic의 JSON 형식은 공식 /get_action 응답 형식과 동일하다.
        #
        # 예:
        # {
        #   "moveWS": {"command": "W", "weight": 0.5},
        #   "moveAD": {"command": "", "weight": 0.0},
        #   "turretQE": {"command": "", "weight": 0.0},
        #   "turretRF": {"command": "", "weight": 0.0},
        #   "fire": false
        # }
        print("ROS2 command input: /tank/control/command")

        # 좌표 변환 기준을 출력한다.
        # raw:
        #   - Unity 시뮬레이터가 보낸 x, y, z 원본 좌표
        #
        # map:
        #   - ROS/RViz/2D 경로계획에서 쓰기 위해 변환한 좌표
        #   - map.x = raw.x
        #   - map.y = raw.z
        #   - map.z = raw.y
        #
        # 즉 Unity의 y축은 높이로 보고,
        # 지상 평면은 Unity x-z 평면으로 해석한다.
        print("Coordinate: raw=(x,y,z), map=(x,z,y)")

        # 실행 정보 출력 끝 구분선이다.
        print("============================================================")

        ####################################################
        # 2-2-2. Flask HTTP 서버 시작
        ####################################################

        # Flask 서버를 실행한다.
        #
        # host=HOST:
        #   - "0.0.0.0"이면 외부 PC에서도 접속 가능하다.
        #   - Windows 시뮬레이터 PC와 Ubuntu 작업 PC를 분리해서 사용할 때 필요하다.
        #
        # port=PORT:
        #   - Tank Challenge 메뉴에서 Endpoint Port로 지정할 포트와 일치해야 한다.
        #
        # threaded=True:
        #   - Flask가 여러 HTTP 요청을 thread 기반으로 처리할 수 있게 한다.
        #   - /info, /get_action, /detect 등 요청이 빠르게 들어올 수 있으므로 켜두는 편이 안전하다.
        #
        # 주의:
        #   - app.run()은 blocking 함수다.
        #   - 이 줄 아래 코드는 서버 종료 전까지 실행되지 않는다.
        #   - ROS2 executor는 start_ros()에서 이미 별도 thread로 돌고 있으므로,
        #     Flask가 blocking되어도 ROS2 callback은 계속 처리된다.
        app.run(host=HOST, port=PORT, threaded=True)

    ########################################################
    # 2-3. 종료 처리
    ########################################################

    # finally는 try 안에서 예외가 발생하거나 Ctrl+C로 종료해도 실행된다.
    finally:
        # ROS2 자원을 안전하게 정리한다.
        #
        # 내부적으로 stop_ros()는 다음 일을 수행한다.
        # - executor.shutdown()
        # - bridge.destroy_node()
        # - rclpy.shutdown()
        #
        # 이 정리를 하지 않으면 다음 실행 때 ROS2 node/executor가 꼬이거나
        # 프로세스 종료가 깔끔하게 되지 않을 수 있다.
        stop_ros()


############################################################
# 3. 직접 실행 보호 구문
############################################################

# 이 파일이 직접 실행될 때만 main()을 호출한다.
#
# 예:
#   python3 -m ros_bridge.main
#   또는 setup.py entry point를 통해 ros2 run으로 실행
#
# 다른 파일에서 import될 때는 main()이 자동 실행되지 않는다.
# Python 패키지에서 일반적으로 사용하는 안전한 진입점 패턴이다.
if __name__ == "__main__":
    # 전체 bridge 시스템을 실행한다.
    main()
