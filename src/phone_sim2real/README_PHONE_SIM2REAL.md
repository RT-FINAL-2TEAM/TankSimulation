# phone_sim2real 패키지

이 패키지는 APF를 사용하지 않고 스마트폰 카메라 이미지만으로 가상 장애물을 생성해 기존 시나리오 1, 2의 회피 체계에 넣기 위한 패키지입니다.

## 핵심 구조

```text
Android PhoneSim2RealApp
  416x416 JPEG + IMU + control metadata
        ↓ HTTP POST /phone/detect
phone_yolo_gateway_node
  best_final.engine YOLO 추론
        ↓ /tank/phone_sim2real/detections
phone_virtual_obstacle_node
  bbox 높이 기반 거리 추정
  map 좌표 가상 장애물 생성
  confirmed 객체 map lock
        ↓
/tank/phone_sim2real/virtual_obstacles
/tank/visual_perception/lidar_clusters      # 기존 A* 회피 입력으로 사용
/tank/rviz/phone_sim2real_markers
/tank/rviz/phone_sim2real_image_cluster_markers
```

`phone_sim2real`은 `/tank/control/command`를 직접 발행하지 않습니다. APF도 켜지 않습니다. 기존 시나리오가 사용 중인 LiDAR cluster 기반 동적 재계획 입력을 흉내 내는 방식입니다.

## Android 앱

Android Studio에서는 이 폴더만 엽니다.

```text
~/tankcc/src/phone_sim2real/android/PhoneSim2RealApp
```

앱 기능:

```text
LINK START / STOP     : 카메라 프레임 전송 시작/중지
INJECT ON / OFF       : YOLO 탐지는 유지하되 ROS 가상 장애물 주입 ON/OFF
LOCK OBSTACLE         : 현재 active phone 장애물을 map 좌표에 고정
CLEAR OBSTACLE        : phone_sim2real active 장애물 제거
IMU TX                : IMU metadata 전송
```

SDK 경로 오류가 나면 다음을 실행합니다.

```bash
cd ~/tankcc/src/phone_sim2real/android/PhoneSim2RealApp
./fix_android_sdk.sh
```

또는 `local.properties`에 실제 Android SDK 경로를 지정합니다.

```properties
sdk.dir=/home/tankcc/Android/Sdk
```

## 실행

시나리오 1 실행 중:

```bash
cd ~/tankcc
./scripts/run_scenario1_auto_terminator.sh
```

다른 터미널:

```bash
cd ~/tankcc
source install/setup.bash
ros2 launch phone_sim2real phone_sim2real.launch.py phone_port:=5002
```

시나리오 2도 동일합니다.

```bash
cd ~/tankcc
./scripts/run_scenario2_auto_terminator.sh
```

Android 앱 설정:

```text
Ubuntu IP : 예) 192.168.0.32
Port      : 5002
Endpoint  : /phone/detect
Interval  : 300~500 ms
IMU TX    : ON
INJECT    : ON
```

## 거리 추정 보정

거리 보정은 `config/phone_sim2real.yaml`에서 합니다.

```yaml
distance_mode: "calibrated_table"
class_distance_table_json: >-
  {
    "rock": [
      {"bbox_height_px": 40, "distance_m": 7.5},
      {"bbox_height_px": 70, "distance_m": 5.5},
      {"bbox_height_px": 115, "distance_m": 3.8},
      {"bbox_height_px": 180, "distance_m": 2.8},
      {"bbox_height_px": 250, "distance_m": 2.5}
    ]
  }
```

실험 방법:

1. 스마트폰으로 객체를 찍습니다.
2. 아래 토픽에서 `bbox_size.height_px`와 `distance_m`을 확인합니다.

```bash
ros2 topic echo /tank/phone_sim2real/virtual_obstacles --once
```

3. 원하는 가상 거리로 YAML의 table을 수정합니다.

예를 들어 rock 사진의 bbox 높이가 73px일 때 5m로 넣고 싶으면:

```json
{"bbox_height_px": 73, "distance_m": 5.0}
```

## Map lock 방식

기본값은 다음입니다.

```yaml
obstacle_anchor_mode: "map_lock_on_confirmed"
lock_after_observations: 3
locked_obstacle_ttl_sec: 10.0
publish_locked_only_to_clusters: true
```

의미:

```text
1~2회 탐지: RViz/debug에는 보이지만 synthetic cluster로는 아직 안 나감
3회 이상 탐지: map 좌표에 LOCK
LOCK 이후: 전차가 움직여도 장애물 위치가 전차를 따라다니지 않음
LOCK된 장애물만 /tank/visual_perception/lidar_clusters로 발행
```

이 구조가 회피 검증에 가장 안정적입니다.

## 오탐 필터

상단에 작게 잡히는 car 같은 오탐은 YAML에서 제거합니다.

```yaml
class_filter_json: >-
  {
    "rock": {"min_conf": 0.70, "min_bbox_height_px": 40, "min_bbox_area_px": 2500},
    "car":  {"min_conf": 0.80, "min_bbox_height_px": 45, "min_bbox_area_px": 3000}
  }
ignore_top_region_ratio: 0.08
```

## 확인 토픽

```bash
ros2 topic echo /tank/phone_sim2real/detections --once
ros2 topic echo /tank/phone_sim2real/virtual_obstacles --once
ros2 topic echo /tank/phone_sim2real/virtual_status --once
ros2 topic echo /tank/visual_perception/lidar_clusters --once
```

RViz MarkerArray:

```text
/tank/rviz/phone_sim2real_markers
/tank/rviz/phone_sim2real_image_cluster_markers
/tank/rviz/fused_object_markers
```

## 주의

`/tank/visual_perception/lidar_clusters`는 기존 LiDAR cluster 입력을 흉내 내는 토픽입니다. 스마트폰 장애물이 locked 상태가 되어야 이 토픽으로 발행되도록 기본 설정했습니다. 즉 순간 오탐이 바로 회피로 들어가는 것을 막습니다.


## Cluster mux 적용 후 실행 방식

이 버전부터 `phone_sim2real`은 `/tank/visual_perception/lidar_clusters`에 직접 publish하지 않는다.
대신 아래 전용 토픽에 스마트폰 synthetic cluster를 발행한다.

```bash
/tank/phone_sim2real/synthetic_lidar_clusters
```

`phone_cluster_mux_node`는 실제 LiDAR cluster와 phone synthetic cluster를 합쳐 아래 토픽으로 발행한다.

```bash
/tank/phone_sim2real/muxed_lidar_clusters
```

따라서 planner가 mux 결과를 보게 하려면 시나리오 실행 전에 다음 환경변수를 붙인다.

```bash
cd ~/tankcc
TANK_TOPIC_LIDAR_CLUSTERS=/tank/phone_sim2real/muxed_lidar_clusters \
./scripts/run_scenario1_auto_terminator.sh
```

시나리오 2도 동일하다.

```bash
cd ~/tankcc
TANK_TOPIC_LIDAR_CLUSTERS=/tank/phone_sim2real/muxed_lidar_clusters \
./scripts/run_scenario2_auto_terminator.sh
```

다른 터미널에서 phone_sim2real을 실행한다.

```bash
cd ~/tankcc
source install/setup.bash
ros2 launch phone_sim2real phone_sim2real.launch.py phone_port:=5002
```

확인:

```bash
ros2 topic info /tank/visual_perception/lidar_clusters
ros2 topic echo /tank/phone_sim2real/synthetic_lidar_clusters --once --full
ros2 topic echo /tank/phone_sim2real/muxed_lidar_clusters --once --full
ros2 topic echo /tank/phone_sim2real/cluster_mux_status --once --full
ros2 topic echo /tank/planner/status --once --full
```

목표 구조:

```text
real lidar clusters
  -> /tank/visual_perception/lidar_clusters
     \
      -> phone_cluster_mux_node -> /tank/phone_sim2real/muxed_lidar_clusters -> planner
     /
phone synthetic clusters
  -> /tank/phone_sim2real/synthetic_lidar_clusters
```


## turret/camera 전방 기준 장애물 주입

스마트폰 자체는 거리 센서가 없기 때문에 bbox 높이 기반 보정표로 `distance_m`을 추정한다.
위치 방향은 기본적으로 simulator turret/camera 전방을 기준으로 잡는다.

```yaml
bearing_reference_mode: "turret"
turret_topic: "/tank/api/get_action/turret"
use_bbox_bearing: true
```

즉 스마트폰 카메라가 rock을 인식하면, 전차 현재 위치에서 turret/camera yaw 방향으로
`distance_m`만큼 떨어진 map 좌표에 가상 장애물을 주입한다. bbox가 화면 중앙에서
좌우로 벗어나면 `use_bbox_bearing: true`에 의해 turret 전방 기준 좌우 bearing이 반영된다.

planner는 직접 `/tank/visual_perception/lidar_clusters`를 보지 말고 mux 결과를 보게 실행한다.

```bash
TANK_TOPIC_LIDAR_CLUSTERS=/tank/phone_sim2real/muxed_lidar_clusters \
./scripts/run_scenario1_auto_terminator.sh
```


## 진행방향 전방 주입 모드

이 버전은 기본적으로 `bearing_reference_mode: "path"`, `use_bbox_bearing: false`를 사용한다.
스마트폰 카메라가 객체를 인식하면 실제 카메라 방향이나 turret topic이 아니라, 현재 `/tank/path/lookahead_pose` 방향 앞쪽에 가상 장애물을 생성한다.
이렇게 하면 장애물이 전차 옆이나 지나간 궤적에 생성되지 않고 현재 주행 경로 전방에 생성되어 A* emergency replan 반응을 확인하기 쉽다.

기본 배치 거리:

```yaml
placement_distance_scale: 1.8
placement_distance_bias_m: 6.0
min_placement_distance_m: 14.0
max_placement_distance_m: 22.0
```

`/tank/phone_sim2real/virtual_obstacles`에서 `bearing_reference_source: lookahead_path`가 나오면 경로 전방 기준으로 정상 배치 중이다.


## PHONE EMERGENCY BRAKE

스마트폰으로 객체가 감지되면 `phone_emergency_brake_node`가 `/tank/api/get_action/override`로 STOP 명령을 짧은 시간 반복 발행한다.
이 override는 ros_bridge가 다음 `/get_action` 응답에 1회만 쓰는 명령이므로, 기존 controller의 `/tank/control/command` 지속 발행자와 직접 경쟁하지 않는다.

흐름:

```text
/tank/phone_sim2real/detections
/tank/phone_sim2real/virtual_obstacles
/tank/phone_sim2real/synthetic_lidar_clusters
        -> phone_emergency_brake_node
        -> /tank/api/get_action/override  # STOP 반복 발행
        -> /tank/phone_sim2real/emergency_status
```

동시에 `phone_virtual_obstacle_node`가 synthetic cluster를 `/tank/phone_sim2real/synthetic_lidar_clusters`로 발행하고, `phone_cluster_mux_node`가 실제 LiDAR cluster와 합쳐 `/tank/phone_sim2real/muxed_lidar_clusters`로 낸다. planner는 이 mux 토픽을 봐야 동적 재계획한다.

시나리오 실행 예:

```bash
cd ~/tankcc
source install/setup.bash
TANK_TOPIC_LIDAR_CLUSTERS=/tank/phone_sim2real/muxed_lidar_clusters \
./scripts/run_scenario1_auto_terminator.sh
```

긴급정지 상태 확인:

```bash
ros2 topic echo /tank/phone_sim2real/emergency_status --once --full
ros2 topic echo /tank/phone_sim2real/cluster_mux_status --once --full
ros2 topic echo /tank/planner/status --once --full
```


## Emergency stop + snapshot path wall mode

This build treats smartphone detections as emergency evidence, not as ordinary map/fusion objects.

- `/tank/phone_sim2real/detections` triggers `phone_emergency_brake_node`.
- The brake node repeatedly publishes STOP to `/tank/api/get_action/override` while the planner replans.
- `phone_virtual_obstacle_node` publishes only RViz markers and `/tank/phone_sim2real/synthetic_lidar_clusters` by default.
- It does **not** mirror phone obstacles into `/tank/map/discovered/objects` or `/tank/perception/fused_objects` by default. This avoids duplicate evidence layers fighting the normal LiDAR/fusion planner.
- The synthetic cluster uses `emergency_wall_center_mode: lookahead_snapshot`, so the first detected phone hazard is snapped to the current lookahead/path point and remembered for TTL. It does not chase the newly replanned path.

Planner execution must still use the mux output:

```bash
TANK_TOPIC_LIDAR_CLUSTERS=/tank/phone_sim2real/muxed_lidar_clusters     ./scripts/run_scenario1_auto_terminator.sh
```


## turret_front_yellow_arrow_final

This build places smartphone emergency obstacles in front of the RViz yellow turret arrow.
It uses `/tank/api/get_action/turret` as `geometry_msgs/Vector3Stamped`, reads `vector.x` as simulator/RViz turret heading in degrees, and converts it to map math yaw with `math_yaw = 90deg - turret_heading`.
`bearing_reference_mode` is `turret`, `emergency_wall_center_mode` is `turret_snapshot`, and bbox lateral bearing is disabled by default so the obstacle appears on the turret/camera centerline.


## 2026-07-02 turret-front far emergency update

This version keeps `phone_sim2real` as a sensor/emergency adapter rather than a path planner.

- Smartphone detections are converted only into `/tank/phone_sim2real/synthetic_lidar_clusters` and RViz markers.
- `enable_discovered_objects_publish` and `enable_fused_objects_mirror` are disabled by default to avoid duplicated obstacle evidence in other planning layers.
- Emergency wall placement is based on the RViz yellow turret/camera direction and is pushed farther ahead of the tank.
- Default placement range is 14–22 m in front of the tank/turret so A* has time and space to replan.
- Emergency STOP is held while the phone signal is active, route change has not occurred, or planner is blocked/failed, up to the configured maximum.

Important: path generation still belongs to `path_planning`; this package only publishes high-priority synthetic cluster evidence and STOP override.
