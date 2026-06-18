# Tank Challenge ROS2 Workspace

이 저장소는 전차 시뮬레이터(Tank Simulator) 자율주행 알고리즘을 위한 ROS2 작업 공간입니다.
자세한 아키텍처 및 내부 구조에 대한 설명은 [docs/README_REORGANIZED_STRUCTURE.md](docs/README_REORGANIZED_STRUCTURE.md) 문서를 참고해 주십시오.

## 구성 요소 (src 패키지)

*   `ros_bridge`: Tank Simulator HTTP endpoint (`/info`, `/detect`, `/get_action` 등)를 수신하여 ROS2 topic으로 변환 및 퍼블리시
*   `vision`: YOLO 모델 추론을 수행하여 객체 탐지
*   `lidar`: 원본 LiDAR 데이터 파싱·좌표 변환, 지면/장애물 분리
*   `tank_visual_perception`: LiDAR DBSCAN 클러스터링 + LiDAR↔카메라 융합
*   `path_planning`: A* 알고리즘 기반 Global Path 생성, 카메라/LiDAR 융합(local_path), 정찰 로깅·루트 A/B
*   `potential`: A* 경로 및 LiDAR 기반 Artificial Potential Field(APF) 회피 타겟 생성
*   `control`: 조향 명령(`command`) 생성
*   `ground_division`: 주행 중 지면/장애물 점 누적 → 지형(고도·거칠기) 맵 생성·저장
*   `risk_analysis`: 로컬 LLM(ollama) 기반 정찰 루트 위험도 분석
*   `tank_common`: 패키지 공용 헬퍼(PointCloud2→numpy 변환 등) 단일 출처
*   `rviz_visualization`: 시스템 통합 모니터링 환경 구성

## 실행 가이드

### 1. 수동 조작 (mode=monitor) + 센서 모니터링 + 정찰 기록 테스트

**목적**
- Windows 시뮬레이터 PC에서 탱크를 직접 조작
- ROS2에서 LiDAR, YOLO detection, clustering, fusion, RViz 시각화 확인
- 주행 중 새로 포착한 객체를 discovered map에 누적
- 주행 후 map 파일로 저장

> [!WARNING]
> `192.168.0.??` 에는 Windows 시뮬레이터 PC의 실제 IP를 입력하십시오.

*   **Terminal 1: ros_bridge 실행**
    ```bash
    TANK_ALLOWED_CLIENTS=127.0.0.1,::1,192.168.0.?? TANK_MODE=monitor ros2 run ros_bridge ros_bridge
    ```
*   **Terminal 2: LiDAR processor 실행** (raw LiDAR 정보를 map 좌표계의 detected_points로 변환)
    ```bash
    ros2 run lidar lidar_processor_node
    ```
*   **Terminal 3: LiDAR clustering 실행**
    ```bash
    ros2 run tank_visual_perception lidar_dbscan_cluster_node
    ```
*   **Terminal 4: RViz 실행**
    1) 기본 RViz:
    ```bash
    ros2 launch rviz_visualization tank_rviz.launch.py
    ```
    2) LiDAR 채널 수를 바꾼 경우 (예: channels=16):
    ```bash
    TANK_LIDAR_CHANNELS=16 ros2 launch rviz_visualization tank_rviz.launch.py
    ```
    3) recon_map만 RViz에 표시:
    ```bash
    ros2 launch rviz_visualization tank_recon_map_rviz.launch.py
    ```
    4) mission_map 포함 RViz:
    ```bash
    ros2 launch rviz_visualization tank_recon_mission_map_rviz.launch.py
    ```
*   **Terminal 5: local_path_node 실행** ("카메라+라이다 통합" & "local path 생성")
    ```bash
    ros2 run path_planning local_path_node
    ```
*   **주행 후 discovered map 저장**
    ```bash
    ros2 service call /tank/map/discovered/save std_srvs/srv/Trigger "{}"
    ```
*   **저장 파일 확인**
    ```bash
    ls -lh ~/tank_discovered_maps
    ```

---

### 2. 자율주행 테스트

**목적**
- ROS2 planner/control이 탱크 주행 명령을 생성
- ros_bridge가 ROS2 제어 명령을 Tank Simulator로 전달
- RViz에서 경로, 장애물, 주행 상태 확인
- LiDAR/YOLO 기반 장애물 인식 결과를 local path 및 자율주행 제어에 반영

*   **Terminal 1: ros_bridge auto 모드 실행**
    ```bash
    TANK_ALLOWED_CLIENTS=127.0.0.1,::1,192.168.0.?? TANK_MODE=auto ros2 run ros_bridge ros_bridge
    ```
*   **Terminal 2: LiDAR processor 실행**
    ```bash
    ros2 run lidar lidar_processor_node
    ```
*   **Terminal 3: LiDAR clustering 실행** (장애물 후보 cluster 생성)
    ```bash
    ros2 run tank_visual_perception lidar_dbscan_cluster_node
    ```
*   **Terminal 4: RViz 실행**
    1. 기본 RViz:
    ```bash
    ros2 launch rviz_visualization tank_rviz.launch.py
    ```
    2. LiDAR 채널 수를 바꾼 경우 (예: channels=16):
    ```bash
    TANK_LIDAR_CHANNELS=16 ros2 launch rviz_visualization tank_rviz.launch.py
    ```
    3. recon_map만 RViz에 표시:
    ```bash
    ros2 launch rviz_visualization tank_recon_map_rviz.launch.py
    ```
    4. mission_map 포함 RViz:
    ```bash
    ros2 launch rviz_visualization tank_recon_mission_map_rviz.launch.py
    ```
*   **Terminal 5: local_path_node 실행** ("카메라+라이다 통합" & "local path 생성" & "discovered map 누적")
    ```bash
    ros2 run path_planning local_path_node
    ```
*   **Terminal 6: autonomous control 실행** (자율주행 제어 명령 생성)
    ```bash
    ros2 launch control tank_autonomous_control.launch.py
    ```
*   **주행 후 discovered map 저장**
    ```bash
    ros2 service call /tank/map/discovered/save std_srvs/srv/Trigger "{}"
    ```
*   **저장 파일 확인**
    ```bash
    ls -lh ~/tank_discovered_maps
    ```

## 관련 문서

*   [프로젝트 상세 구조 문서](docs/README_REORGANIZED_STRUCTURE.md)
*   [비주얼 인지 통합 노트](docs/INTEGRATION_NOTES_VISUAL_PERCEPTION.md)
*   [LiDAR 설정 리팩토링 노트](docs/REFACTOR_NOTES_LIDAR_CONFIG.md)
*   [팀 시뮬레이션 관련 문서 모음](docs/TEAM_TANKSIMULATION_DOCS/README.md)
