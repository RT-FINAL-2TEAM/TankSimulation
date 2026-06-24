# 정찰 포탑 좌우 스윕 — 구현 스펙 (팀원 할당)

> 상태: **할당 스펙(미구현)**. 작성 2026-06-23. 시나리오 owner(팀장)가 정의, **구현은 control/perception 팀원**.
> 목적: 정찰 중 **카메라(포탑) FOV를 좌우로 스윕**해 발견 커버리지↑. body weave(경로계획, 이미 강화됨)와 **병행**.

## 1. 배경

정찰 발견율을 높이는 카메라 커버리지 수단은 두 가지를 **함께** 쓴다(팀장 결정, 2026-06-23):
- **body weave** (✅ 구현·강화됨, 경로계획=팀장): `map_astar_planner_node._recon_scan_target` — lookahead를 좌우 사인 오프셋(진폭 6m/파장 12m, launch 인자 `recon_scan_amplitude_m`/`recon_scan_wavelength_m`로 튜닝).
- **포탑 스윕** (❌ 본 스펙, 제어=팀원): 주행선은 그대로 두고 **카메라만 좌우로 흔들어** 밀집 숲(route A)처럼 weave가 APF에 가려지는 구간에서도 커버리지 확보.

## 2. 담당 분리
- **control(팀원)**: controller가 turret 명령을 실제로 발행. 현재 `make_action()`의 `turretQE`/`turretRF`는 **no-op 고정**([src/control/control/tank_controller_node.py:382-384](../src/control/control/tank_controller_node.py#L382-L384)).
- **perception(팀원)**: 포탑이 움직이면 카메라 헤딩이 바뀌므로 **융합/투영이 live 포탑각을 반영**하는지 확인(아래 §5).

## 3. 동작 규격
- **활성 조건**: `mission_type == "recon"` 일 때만. `mission`/`return`/시나리오2에선 **OFF**(그땐 포탑을 교전 조준에 쓸 예정 — §6).
- **패턴**: `turretQE`를 좌(Q)↔우(E) 주기 진동. 단순 시간기반 오픈루프로 충분(정밀도 불필요).
  - 예: 주기 `T`초로 Q `weight=w` → E `weight=w` 교대, 또는 사인으로 weight 변조.
  - `turretRF`(상하)는 건드리지 않음(`""`,0).
- **파라미터(제안)**: `recon_turret_sweep`(bool, 기본 false), `turret_sweep_period_sec`(예 4.0), `turret_sweep_weight`(0.1~1.0, 예 0.5). 한계각 기반으로 하려면 포탑 현재각 피드백으로 폐루프(아래).

## 4. 인터페이스 (이미 존재)
- **출력**: `make_action()`의 `turretQE` = `{"command": "Q"|"E"|"", "weight": 0.0~1.0}` (검증 규약 [src/ros_bridge/ros_bridge/commands.py](../src/ros_bridge/ros_bridge/commands.py): Q=좌/E=우).
- **시뮬 규약**: [docs/SIMULATOR_API.md](SIMULATOR_API.md) `/get_action`의 `turretQE`.
- **포탑 현재각 피드백**: `/tank/api/get_action/turret`([src/ros_bridge/ros_bridge/bridge_node.py](../src/ros_bridge/ros_bridge/bridge_node.py)) — 한계각 기반 폐루프 스윕에 사용 가능(오픈루프면 불필요).

## 5. ⚠ Perception 정합 (필수 — 안 하면 발견맵 오염)
포탑이 스윕하면 **카메라 헤딩이 실시간으로 바뀐다**. LiDAR↔카메라 융합/투영이 **그 순간의 포탑각**을 반영하지 않으면, 발견객체의 map 좌표가 틀어져 `discovered` 맵이 오염된다(시나리오2 known-tank/장애물에도 직접 영향).
- `local_path_node`에 `heading_source` 옵션 존재(`turret` / `body_plus_turret` / `body`) — 융합 투영 시 포탑각을 쓰도록 설정/확인.
- 스윕 활성 시 `heading_source`가 포탑각을 포함하는지(예: `body_plus_turret`), 그리고 그 각이 `/tank/api/get_action/turret` 최신값과 동기인지 검증.

## 6. 시나리오2 충돌 주의
- 시나리오2 교전(팀원 fire 구현)에서 turret은 **표적 조준**에 쓸 예정. 정찰 스윕과 **모드 분리**(mission_type 게이트)로 충돌 방지.
- (참고) 정찰 weave도 mission/scenario2에선 자동 OFF(`recon_scan_enabled=false`)로 동일 원칙.

## 7. 검증
1. 정찰 주행 중 시뮬 화면에서 포탑이 좌우로 흔들리는지 + `ros2 topic echo /tank/api/get_action/turret` 각이 주기 진동하는지.
2. **스윕 중 발견객체 map 좌표가 안정적**인지(같은 객체가 좌표 튀지 않음) — §5 정합 확인의 핵심 지표.
3. body weave(이미 강화됨)와 함께 켰을 때 발견 수(`route_*.json` vision_yolo / discovered count)가 늘어나는지 비교.
