# Tank Bridge

Tank Challenge 시뮬레이터와 ROS2를 연결하기 위한 Flask Endpoint + ROS2 Bridge 패키지입니다.

기존 단일 파일 `tank_server_ros2_bridge.py`를 기능별 모듈로 분리하여, 앞으로 A*, 위험도 맵, YOLO, RViz 시각화, 제어기 등을 확장하기 쉽게 만든 구조입니다.

---

## 1. 목적

이 패키지는 다음 역할을 담당합니다.

1. Tank Challenge 공식 Flask API endpoint 제공
2. 시뮬레이터가 보내는 상태/이벤트 데이터를 ROS2 topic으로 publish
3. ROS2 알고리즘 노드가 만든 제어 명령을 `/get_action` 응답으로 다시 시뮬레이터에 반환
4. Unity 좌표계와 ROS/RViz용 map 좌표계를 분리하여 관리
5. 수동 관측 모드와 자율제어 모드를 환경변수로 전환

---

## 2. 권장 실행 구조

```text
Windows 시뮬레이터 PC
└── Tank Challenge 실행
    └── Endpoint IP/Port를 Ubuntu 작업 PC로 지정

Ubuntu 작업 PC
└── ROS2 workspace
    ├── ros_bridge        # 시뮬레이터 ↔ ROS2 통신 허브
    ├── path_planning       # A*, 위험도 맵, 경로계획
    ├── control    # 주행/포탑 제어 명령 생성
    ├── tank_perception    # YOLO, LiDAR 처리
    └── tank_visualization # RViz marker, debug visualization
    
