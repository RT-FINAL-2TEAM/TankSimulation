# 코딩 컨벤션 (TankSimulation ROS2 워크스페이스)

전체 리팩토링(Phase A) 기준. 팀 공통 규칙이며, 각자 자기 패키지에 적용한다.

## 주석 언어 — 한국어로 통일

- **주석(`#`)과 docstring(`"""..."""`)은 한국어**로 쓴다. 기존 영문/혼용 주석은 한글로 바꾼다.
- **영문 유지 대상**(번역하지 않음):
  - 식별자: 함수·변수·클래스·파라미터 이름 (`calc_repulsive_force`, `route_id` 등)
  - ROS 토픽/서비스명, 로그 메시지의 키, 코드 리터럴
  - 파일 상단 `# -*- coding: utf-8 -*-` 선언
  - 외부에서 그대로 인용하는 규약(예: Unity JSON 키 `moveWS`)
- 주석은 "무엇을"이 아니라 **"왜"**를 적는다. 코드가 자명하면 주석을 달지 않는다.

## 네이밍

- 함수·변수: `snake_case`. 클래스: `PascalCase`(ROS 노드는 `...Node` 접미사).
- 내부 전용 상태는 `_` 접두사로 일관(`self._last_yaw`처럼 — public/private 혼용 금지).
- ROS 토픽 계층 유지: `/tank/<영역>/<이름>` (`/tank/sensor/lidar/*`, `/tank/terrain/*` 등).

## 파일/모듈 구조 (지향점)

- 노드 파일은 **ROS 노드(파라미터·구독/발행·콜백·타이머)** 에 집중한다.
- 순수 로직(수식·기하·알고리즘)은 별도 모듈로 분리한다(예: `apf_math.py`, `astar_utils.py`).
- 한 파일에 책임이 과하게 몰리면(대략 700줄+) 분할을 검토한다. **단, 무테스트 환경이라 구조 변경은 신중히(Phase C).**

## 검증 (테스트 스위트 없음)

- 변경 후 반드시: `colcon build --symlink-install` + 수정 모듈 import 스모크 + 정찰 시뮬 1회(거동 보존).
- 주석/데드코드 정리(Phase A)는 거동 불변이어야 한다.

## 리팩토링 단계

- **A(안전) ✅완료**: 죽은코드 제거 + 주석 한글 통일. AST 불변.
- **B(중간) ✅완료**: 중복 함수 통합 + `terrain_record_finalize_node` 중복 해소.
  - 함수 통합: `pointcloud2_to_xyz_array`(8곳→신규 `tank_common` 패키지), `prefab_half_size`(2곳→`path_planning.config`), `distance`(2곳→`lidar.path_blocking`). AST·호출값 검증.
  - terrain 노드: **`ground_division`를 canonical**로 단일화(상위집합·dual-input). rviz copy 삭제, rviz launch 2개를 gd 노드로 repoint(`use_preclassified_lidar:False`로 단일입력 동작 보존). 두 노드의 finalize/save 로직이 갈라진 fork라 **지형 시뮬 1회로 동치 확인 필요**.
- **C(위험)**: 비대 파일 모듈 분할. 한 번에 한 파일 + 스모크.
