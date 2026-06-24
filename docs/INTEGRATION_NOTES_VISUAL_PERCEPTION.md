# Visual Perception 통합 메모

## 반영 목적

팀원이 수동 튜닝한 `lidar_camera_overlay_node`와 LiDAR DBSCAN clustering 작업을 ROS2 패키지 구조에 통합했습니다.

기존에는 overlay node 파일이 `path_planning` 패키지 내부에 있었지만, 실제 역할은 경로 계획이 아니라 **LiDAR-camera calibration / visual perception / RViz 검증**입니다. 따라서 새 패키지 `tank_visual_perception`으로 이동했습니다.

## 추가된 패키지

```text
tank_visual_perception/
├── launch/visual_perception.launch.py
├── tank_visual_perception/lidar_camera_overlay_node.py
└── tank_visual_perception/lidar_dbscan_cluster_node.py
```

## 자동 실행 구조

`control/launch/tank_autonomous_control.launch.py`에 다음 노드를 포함했습니다.

```text
lidar/lidar_processor_node
tank_visual_perception/lidar_camera_overlay_node
tank_visual_perception/lidar_dbscan_cluster_node
path_planning/map_astar_planner_node
path_planning/local_path_node
potential/potential_field_node
control/tank_controller_node
```

따라서 자율주행 실행 시 별도 terminal에서 overlay/cluster node를 따로 실행하지 않아도 됩니다.

## ros_bridge 변경

`/detect` endpoint로 들어온 turret camera 이미지를 다음 ROS2 topic으로 publish하도록 수정했습니다.

```text
/tank/camera/image_compressed
/tank/api/detect/image_compressed
```

`lidar_camera_overlay_node`는 이 이미지와 `/tank/api/info/raw`의 LiDAR raw 정보를 사용해 projection overlay를 생성합니다.

## 주요 topic

```text
/tank/camera/image_compressed                  # ros_bridge가 publish하는 터렛 카메라 이미지
/tank/camera/lidar_projection/image            # overlay Image
/tank/camera/lidar_projection/compressed       # overlay compressed image
/tank/camera/lidar_projection/status           # projection 통계
/tank/visual_perception/lidar_clusters         # DBSCAN cluster JSON
/tank/rviz/lidar_cluster_markers               # RViz cluster marker
```

## 팀원 명령 호환

아래 명령도 계속 동작하도록 console script를 등록했습니다.

```bash
ros2 run tank_visual_perception lidar_dbscan_cluster_node \
  --ros-args \
  -p eps:=1.5 \
  -p min_samples:=2 \
  -p min_cluster_size:=2
```

## trash 처리

기존 `path_planning/path_planning/camera_lidar_overlay_node.py`는 `trash/path_planning/`에 백업하고 실제 빌드 경로에서는 제거했습니다.

Python `__pycache__` 파일들도 source tree에서 제거했습니다.
