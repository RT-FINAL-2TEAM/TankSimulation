# LiDAR/config 책임 분리 리팩터링 기록

## 적용 원칙

- 최상위 패키지 구조는 유지했다.
- LiDAR raw schema 해석, 좌표 변환, detected point 추출, ground filtering, clustering, bbox 생성, path-block 저수준 계산은 `lidar` 패키지로 모았다.
- `path_planning`, `potential`, `rviz_visualization`은 LiDAR topic을 입력으로 사용할 수 있지만, LiDAR raw payload 처리 알고리즘을 직접 보유하지 않도록 정리했다.
- 시뮬레이션에서 바꿀 수 있는 기본값은 각 패키지의 `config.py` 또는 기존 `config/*.yaml`에 모았다.

## 새로 추가/정리한 핵심 파일

```text
lidar/lidar/config.py
lidar/lidar/coordinate_utils.py
lidar/lidar/payloads.py
lidar/lidar/path_blocking.py
lidar/lidar/obstacle_memory.py

path_planning/path_planning/config.py
control/control/config.py
potential/potential/config.py
vision/vision/config.py
```

## LiDAR 책임 범위

```text
ros_bridge
  /tank/api/info/raw publish only

lidar
  /tank/api/info/raw subscribe
  lidarOrigin/lidarRotation/lidarPoints 분리
  Unity raw -> tank_map 좌표 변환
  isDetected=True point 추출
  optional ground filtering
  /tank/sensor/lidar/* publish
  LiDAR obstacle memory / clustering / bbox / path-block utility 제공

path_planning
  A* 전역 경로 생성
  필요한 경우 lidar.obstacle_memory를 통해 dynamic replan 판단
  YOLO+LiDAR local_path fusion 담당

potential
  /tank/sensor/lidar/detected_points_map을 APF obstacle input으로만 사용
```

## 검증

- 모든 Python 파일 `py_compile` 통과.
- 이 환경에는 ROS2/colcon 런타임이 없어서 실제 `colcon build`는 수행하지 못했다.
