# potential lecture-style APF update

이 패키지는 Tank Challenge ROS2 프로젝트의 `/tank/local_target/pose` 생성 node입니다.

## 핵심 변경

- 강의식 Potential Field 수식이 코드에 직접 드러나도록 `potential_field_node.py`를 재작성했습니다.
- 사용자 조정값은 파일 상단의 global variable로 모았습니다.
- 기존 topic 계약은 유지했습니다.
- 추가 debug topic을 제공합니다.

## 수식

```text
U_A = 1/2 * k_A * d^2
F_A = -grad(U_A) = k_A * (r_D - r_B)

U_R = 1/2 * k_R * (1/g - 1/g*)^2,  g <= g*
F_R = -grad(U_R) = k_R * (1/g - 1/g*) / g^3 * (r_B - r_O)

F = F_A + F_R + F_T + F_threat
v_S = ||F||
theta_D = atan2(F_y, F_x)
theta_dot_S = k_theta * wrap(theta_D - theta)
```

## 주요 global 변수

- `K_ATTRACTIVE`: 목표점 인력 gain
- `K_REPULSIVE`: 장애물 척력 gain
- `OBSTACLE_INFLUENCE_RADIUS`: 장애물 영향 반경 `g*`
- `LOCAL_TARGET_DISTANCE`: APF 결과 방향으로 local target을 둘 거리
- `USE_TANGENTIAL_FORCE`: 장애물 앞 진동/로컬 미니마 완화용 접선항 사용 여부
- `TANGENTIAL_GAIN_SCALE`: 접선항 gain
- `ANGULAR_GAIN_K_THETA`: 강의식 heading 제어 gain
- `ANGLE_EPSILON_DEG`: 회전 후 병진 허용 각도 오차

## 기존 주요 topic

- Subscribe
  - `/tank/player/pose`
  - `/tank/path/lookahead_pose`
  - `/tank/goal/pose`
  - `/tank/sensor/lidar/detected_points_map`
  - `/tank/map/discovered/objects`
- Publish
  - `/tank/local_target/pose`
  - `/tank/potential/attractive_vector`
  - `/tank/potential/repulsive_vector`
  - `/tank/potential/threat_vector`
  - `/tank/potential/result_vector`
  - `/tank/potential/status`
  - `/tank/rviz/potential_field_markers`

## 추가 topic

- `/tank/potential/tangential_vector`
- `/tank/potential/desired_motion`

## 빌드

```bash
cd ~/tankcc   # 또는 현재 ROS2 workspace
colcon build --packages-select potential --symlink-install
source install/setup.bash
```

## 실행

```bash
ros2 run potential potential_field_node
```

또는 기존 launch에서 그대로 사용 가능합니다.
