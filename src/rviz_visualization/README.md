# rviz_visualization

Tank Challenge 시뮬레이터 데이터를 RViz2에서 시각화하기 위한 패키지입니다.

## 역할

이 패키지는 판단 알고리즘을 수행하지 않습니다.

담당 범위:

- 아군 전차 위치 표시
- 적 전차 위치 표시
- 목표 지점 표시
- 장애물 표시
- LiDAR point 표시
- risk / complexity 값이 포함된 obstacle 또는 perception 결과 시각화
- RViz2 설정 파일 관리

## 입력 topic

- `/tank/player/pose`
- `/tank/enemy/pose`
- `/tank/goal/pose`
- `/tank/map/obstacles`
- `/tank/sensor/lidar/points`

## 출력 topic

- `/tank/rviz/object_markers`
- `/tank/rviz/obstacle_markers`
- `/tank/rviz/lidar_markers`
- `/tank/rviz/risk_markers`

## 실행

```bash
ros2 launch rviz_visualization tank_rviz.launch.py
RViz Fixed Frame
tank_map
주의

카메라/라이다 기반 위험도 판단은 추후 perception/planning 패키지에서 수행하고,
이 패키지는 그 결과를 RViz2에 표시하는 역할만 담당합니다.
