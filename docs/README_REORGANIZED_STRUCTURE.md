# Reorganized Tank Challenge ROS2 src

이 `src`는 기존 `phone_sensor_bridge_tank_tensorrt` 계통을 제외하고, 전차 시뮬레이터용 ROS2 패키지를 기능별로 재정리한 구조다.

## 패키지 구분

| 구분 | ROS2 패키지명 | 담당 책임 |
|---|---|---|
| ROS_BRIDGE | `ros_bridge` | Tank Simulator HTTP endpoint `/init`, `/info`, `/detect`, `/get_action` 등을 ROS2 topic과 연결 |
| PATH_PLANNING | `path_planning` | A* global path 생성, YOLO+LiDAR 기반 `local_path_node` |
| CONTROL | `control` | `/tank/local_target/pose` 또는 `/tank/path/lookahead_pose` 추종, `/tank/control/command` 생성 |
| VISION | `vision` | YOLO 모델 로딩, `/detect` 이미지 추론, `/tank/perception/detections` 생성 |
| LIDAR | `lidar` | `/tank/api/info/raw`에서 LiDAR 원본 추출, map 좌표 변환, `/tank/sensor/lidar/*` 생성 |
| RVIZ | `rviz_visualization` | 정적 map, LiDAR, path, potential field, fused object marker 시각화 |
| POTENTIAL | `potential` | A* lookahead와 LiDAR 장애물 기반 APF local target 생성 |

> ROS2 패키지명은 빌드 안정성을 위해 모두 lowercase로 두었다. `RVIZ` 패키지는 `rviz/` 설정 폴더와 이름 충돌을 피하기 위해 `rviz_visualization`으로 둔다.

## 핵심 데이터 흐름

```text
Tank Simulator
  ├─ /info
  │   └─ ros_bridge
  │       ├─ /tank/api/info/raw
  │       ├─ /tank/player/pose
  │       └─ /tank/player/state
  │
  │       /tank/api/info/raw
  │              ↓
  │            lidar
  │       ├─ /tank/sensor/lidar/points
  │       ├─ /tank/sensor/lidar/points_count
  │       ├─ /tank/sensor/lidar/origin
  │       ├─ /tank/sensor/lidar/rotation
  │       └─ /tank/sensor/lidar/detected_points_map
  │
  ├─ /detect image
  │   └─ ros_bridge → vision YOLO
  │       └─ /tank/perception/detections
  │
  └─ /get_action
      ↑
      ros_bridge ← /tank/control/command ← control
```

```text
경로/회피 흐름

/tank/player/pose + /tank/goal/pose
        ↓
path_planning/map_astar_planner_node
        ↓
/tank/global_path + /tank/path/lookahead_pose
        ↓
potential/potential_field_node
        ↑
/tank/sensor/lidar/detected_points_map  ← lidar/lidar_processor_node
        ↓
/tank/local_target/pose
        ↓
control/tank_controller_node
        ↓
/tank/control/command
```

```text
카메라 + LiDAR 융합 흐름

vision YOLO → /tank/perception/detections
lidar       → /tank/sensor/lidar/detected_points_map
        ↓
path_planning/local_path_node
        ├─ /tank/perception/fused_objects
        ├─ /tank/map/discovered/objects
        ├─ /tank/rviz/fused_object_markers
        └─ /tank/rviz/discovered_object_markers
```

## 실행 예시

### 1. 빌드

```bash
cd ~/tankcc
rm -rf build install log
colcon build --symlink-install
source install/setup.bash
```

### 2. ROS bridge 실행

```bash
TANK_ALLOWED_CLIENTS=127.0.0.1,192.168.0.24 \
TANK_MODE=auto \
ros2 run ros_bridge ros_bridge
```

### 3. 자율주행 stack 실행

```bash
ros2 launch control tank_autonomous_control.launch.py
```

이 launch는 다음 노드를 함께 실행한다.

```text
lidar/lidar_processor_node
path_planning/map_astar_planner_node
path_planning/local_path_node
potential/potential_field_node
control/tank_controller_node
```

### 4. RViz 실행

```bash
ros2 launch rviz_visualization tank_rviz.launch.py   # (구 tank_recon_map_rviz.launch.py는 존재하지 않음)
```

## LiDAR 책임 정리

기존 구조에서는 `ros_bridge`, `path_planning`, `potential`, `vision`, `rviz_visualization`에서 모두 LiDAR 관련 코드가 보여 헷갈릴 수 있었다. 새 구조에서는 책임을 아래처럼 나눈다.

| 패키지 | LiDAR에 대한 역할 |
|---|---|
| `ros_bridge` | `/info` 원본을 받기만 함. high-level `/tank/sensor/lidar/*` 생성하지 않음 |
| `lidar` | 유일하게 LiDAR 원본 분리, 좌표 변환, detected point 생성 담당 |
| `path_planning` | 이미 변환된 `/tank/sensor/lidar/detected_points_map`을 소비해서 A* 동적 판단 또는 `local_path_node` fusion에 사용 |
| `potential` | 이미 변환된 `/tank/sensor/lidar/detected_points_map`을 소비해서 APF 반발력 계산 |
| `rviz_visualization` | 이미 생성된 `/tank/sensor/lidar/points`를 marker로 표시만 함 |

즉, LiDAR 데이터의 “생성/전처리”는 `lidar` 패키지 하나만 담당하고, 나머지는 그 결과를 subscribe해서 사용하는 구조다.
