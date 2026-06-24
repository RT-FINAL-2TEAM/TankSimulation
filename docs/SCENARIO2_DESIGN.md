# 시나리오2 설계 — 위험도 기반 교전·복귀 임무

> 상태: **설계 합의용 초안** (코드 미구현). 정찰(시나리오1) 위에서 진행되는 본 임무 시나리오의 개념·아키텍처·인터페이스를 정의한다.
> 구현은 합의 후 별도 단계. 이 문서는 토픽/노드/파일을 실제 코드와 대조해 작성했다.

## 1. 개요

시나리오2는 단순 A→B 주행이 아니라, **정찰로 파악한 적 전차를 사격 용이 지점에서 제거하며 목적지까지 주행하고,
정찰에 없던 새 적 전차가 출현하면 위험도를 평가해 복귀**하는 임무다. 프로젝트 핵심(정찰 = 은밀성·위험도 정량화)을
실제 임무 의사결정으로 잇는다.

- **임무 루트 = A 고정**(설계 결정). 정찰 평가가 A를 추천하도록 구성·검증한다. 추천 근거(노출도/은밀성)는 실제
  정찰 산출물(노출지도·comparison) 기준으로 확정한다. `routes.yaml`의 A/B는 의도 설계물이며 B는 비교·대조용이다.
- **사격은 실제 시뮬 사격**이되 성공 기준은 "포탄 임팩트가 표적 근처에 낙하"(명중/킬이 아니라 근접). 표적이 정지
  상태라 정밀 탄도는 필요 없다.

## 2. 스토리 (시퀀스)

1. **정찰(시1)** — 맵에 `tank`를 배치한 상태로 정찰. 발견객체맵에 **알려진 적 전차 목록(known-tank)** + 노출/LoS 평가 축적.
2. **루트·사격지점 결정** — 정찰 평가로 **A 추천**. 정찰 노출/LoS 분석으로 **사격 용이 웨이포인트 후보 도출(보조)** → 사용자가 확정해 `routes.yaml`에 반영.
3. **임무 주행(A)** — A 루트를 주행하며:
   - 사격 웨이포인트에서 사거리·시선(LoS) 내 **알려진 적 전차 → 교전(사격)** → 임팩트 근접 시 격파 처리.
   - **일반 장애물(차·바위) → APF 국소 회피** 후 계속.
   - **정찰에 없던 새 적 전차 출현 → 위험도 평가 → 임계 초과 시 복귀, 이하 시 회피·계속.**
4. **목적지 교전** — 최종 목적지는 **마지막 적 전차 사격 용이 지역**. 도착 후 교전 → 성공으로 마무리.
5. **복귀(분기)** — 새 적 전차 위험이 임계를 넘으면 출발지로 복귀(임무 중단).

## 3. 맵·표적 모델

- **맵 변경**: 인식이 약한 `person`을 맵에서 제거하고, 정찰 단계에 `tank`를 배치한다.
- **표적 2종** (둘 다 정지):

| 표적 | 출처 | 특성 |
|---|---|---|
| 배치 tank (경로상) | 정찰 발견객체맵 + 주행 중 YOLO `tank` 탐지(지도좌표) | **체력 개념 없음** → 적중해도 시뮬상 사라지지 않음. 격파는 **논리 처리**(임팩트 근접). |
| 목적지 적전차 | 시뮬 `/info` → `/tank/enemy/pose`·`/tank/enemy/state` | **체력 있음**, 위치·상태 풀 정보. |

- **known vs new 구분**: 정찰 known-tank 목록과 주행 중 탐지를 **map 좌표로 매칭**(발견객체맵은 본래 지도좌표 기반 식별).
  매칭 실패 = **새 적 전차** → 위험도 평가 트리거.

## 4. 아키텍처 — 계층형 자율 (Hierarchical Autonomy)

"로컬 AI가 스스로 판단하는 전차" 비전을 **안전하게** 구현하는 표준 구조. LLM을 **결정**에 쓰되, 제어 루프는 결정론이 지킨다.

| 계층 | 주기 | 담당 | LLM |
|---|---|---|---|
| **반사층** | ~100ms (10Hz) | 조향·충돌회피(APF/A*/제어) — 항상 안전 유지 | ❌ 금지 (지연·변동·환각) |
| **전술층** | 이벤트당 (수 초) | 돌파/복귀 결정, 교전 트리거, 근거 생성 | ✅ 가능 (소형 구조화 출력) |

**실시간 가능성 근거**: 현재 정찰 LLM 추론이 ~14초인 건 `qwen3:0.6b` + `num_predict: 768`(리포트 생성)이기 때문이다
([llm_reporter.py](../src/risk_analysis/risk_analysis/llm_reporter.py)). 전술 **결정**은 `{"decision":"복귀","reason":"..."}`
수준 30~50토큰이면 충분하므로 같은 모델로도 1~2초(GPU면 1초 이내) — **이벤트 기반 전술 결정엔 충분**하다.
100ms 제어 루프엔 어떤 LLM도 부적합하므로 절대 넣지 않는다.

**안전망**: LLM이 느리거나·부재·이상하면 **기하 위험도 점수**로 즉시 결정(아래 6장).

## 5. 교전(사격) — 인터페이스 중심

사격의 **저수준 제어(포탑 조준 + 발사)는 control 도메인(팀원)이 구현**한다. 시나리오는 **교전 인터페이스(계약)로 호출**만 한다.
이 분리 덕에 시나리오는 사격 내부구현 없이 완성·테스트(mock)할 수 있다.

**교전 인터페이스 계약(제안)**
- 시나리오 decision 노드 → `/tank/engage/request` 발행 (표적 map 좌표 + `target_id`).
- 팀원 turret 제어가 구독 → 조준(`turretQE`/`turretRF`) → 정렬 + 사거리 내 → `fire=True`.
- 결과 → `/tank/engage/result` (impact 좌표·표적거리·성공여부). 또는 기존 임팩트 토픽 활용.

**이미 존재하는 입력(조사 확인)** — 팀원 구현이 쓸 재료는 다 발행 중:
- 적: `/tank/enemy/pose`, `/tank/enemy/state`, `/tank/enemy/heading` ([bridge_node.py:167-183](../src/ros_bridge/ros_bridge/bridge_node.py#L167-L183))
- 내 pose/heading: `/tank/player/pose`, `/tank/player/heading`
- 포탑 현재각: `/tank/api/get_action/turret` ([bridge_node.py:118](../src/ros_bridge/ros_bridge/bridge_node.py#L118))
- 임팩트 피드백: `/tank/api/update_bullet/*` ([bridge_node.py:131-138](../src/ros_bridge/ros_bridge/bridge_node.py#L131))

**현재 상태**: `make_action()`의 `turretQE`/`turretRF`/`fire`는 빈값·False로 고정(no-op)
([tank_controller_node.py:382-384](../src/control/control/tank_controller_node.py#L382-L384)). `fire_cmd`는 선언만 되고 미사용.
→ 팀원이 인터페이스에 맞춰 배선·구현.

**성공 판정(시나리오 책임)**: `/update_bullet` 임팩트 ↔ 표적 거리 < R 이면 격파 처리. 정지표적 + 근접성공이라 정밀 탄도 불요.

> 선행과제(팀원): **포탑 조준 특성 실험**(포탑 회전속도·탄도 낙차·발사주기) — E1~E7식 실측. 임팩트 피드백으로 폐루프 튜닝.

## 6. 위험도 평가 & 의사결정 (새 적 전차 대응)

새 적 전차 출현 시 **돌파(회피·계속) vs 복귀**를 결정한다.

- **기하 위험도(빠름·안전망)**: 거리 + 내가 적 FOV에 노출됐는가 + LoS 엄폐 여부 + 위협 수 → score.
  기존 위협 로직 재활용: `check_los()`·`is_threat_active()` ([potential_field_node.py:322-340](../src/potential/potential/potential_field_node.py#L322-L340)),
  타입별 위협 반경(Tank 20m).
- **LLM 전술 결정(이벤트당)**: 상황 요약(거리·노출·엄폐·잔여경로 위협)을 입력해 `{"decision","reason"}` 소형 구조화 출력.
  `risk_analysis`의 LLM 호출 재활용하되 `num_predict` 대폭 축소·JSON 강제 ([llm_reporter.py](../src/risk_analysis/risk_analysis/llm_reporter.py)).
- **우선순위**: 시간 내 LLM 응답 = LLM 결정 / 지연·부재 = 기하 위험도 fallback.
- **표시**: 결정·근거를 콕핏 MFD(웹, [live_view.py](../src/ros_bridge/ros_bridge/live_view.py))에 — 기존 LLM 조언 표시 자리 재활용.

> 교전 모드에서는 "접근·조준"과 "위협 척력 회피"가 상충할 수 있다. 알려진 교전 표적은 위협 척력에서 제외하거나
> 교전 거리에서 척력을 낮추는 모드 전환이 필요(구현 단계 상세화).

## 7. 임무 FSM

```text
[출발]
   │
   ▼
[전진]  ──일반장애물──▶ APF 회피 ──▶ [전진]
   │
   ├─ known tank 사거리+LoS ─▶ [교전] engage요청 → 성공판정 ─▶ [전진]
   │
   ├─ new tank 탐지 ─▶ [위험도평가] ──임계초과──▶ [복귀] goal=출발지
   │                                └─이하──▶ 회피·[전진]
   ▼
[목적지 도달] ─▶ [최종 교전] ─▶ [성공]
```

## 8. 노드·토픽 인터페이스

- **재활용(기존)**: 적/포탑/임팩트 토픽(위 5장), FOV/LoS·위협반경(potential), 발견객체맵(local_path_node, 지도좌표 식별),
  전역 goal `/tank/goal/pose`(planner), MFD(live_view).
- **신규(본인 구현)**:
  - `decision 노드` — 표적 분류(known/new)·위험도·LLM → 돌파/복귀 결정 + `/tank/engage/request` 발행.
  - **복귀 goal-swap** — 현재 미구현(goal은 launch 고정). 후보: [tank_controller_node.py:304-317](../src/control/control/tank_controller_node.py#L304-L317),
    [map_astar_planner_node.py:540-549](../src/path_planning/path_planning/map_astar_planner_node.py#L540-L549) (도착 시 goal=start로 재계획).
  - **사격 웨이포인트 도출기** — 정찰 노출/LoS → 후보 제안.
  - **교전 인터페이스 계약**(5장).
- **위임(팀원)**: turret-aim/fire 제어(계약 구독측).

## 9. 역할 분담

- **본인(시나리오 owner)**: FSM, 표적 선정(known/new 매칭), 위험도·돌파/복귀 결정, 사격 WP 도출·확정, 복귀 goal-swap,
  오케스트레이션, MFD, 통합 + 교전 인터페이스 정의.
- **팀원(control)**: 포탑 조준 실험 + turret-aim/fire 구현(인터페이스 구독측). 미완성 시 시나리오는 **mock 노드로 병행 개발**.

## 10. 로드맵·의존성

1. 교전 인터페이스 계약 확정 → mock으로 시나리오 골격 검증.
2. known-tank 목록 산출(정찰) + known/new 매칭.
3. 복귀 goal-swap 구현.
4. 위험도 평가(기하) + LLM 전술결정(실시간화: `num_predict`↓·JSON·가능 시 GPU).
5. 사격 웨이포인트 도출기.
6. (팀원 병행) turret-aim/fire + 포탑 조준 실험.
7. 위험도 수식(6장)과 정량 연결.

## 11. 미해결 / 결정 필요

- 사격 웨이포인트 후보의 도출 기준(노출 가중치·표적 LoS·사거리) 구체 수치.
- 복귀 임계(위험도 score) 값 — 정찰 노출 통계로 캘리브레이션.
- 교전 중 위협 척력 모드 전환 방식.
- 교전 인터페이스 메시지 타입(커스텀 msg vs JSON String) — 팀원과 합의.
- A 추천을 정찰 평가가 실제로 산출하는지 검증(노출지도 기준).

---

## 12. 구현 준비 노트 (2026-06-22 재검토)

문서↔코드 표류·오늘 정찰 위험도 재설계와의 정합·오프라인 구현 가능성을 코드 대조로 재검토한 결과. **설계는 타당하고
그대로 구현 가능**하다. 토대는 준비됐고, FSM·교전·복귀는 설계대로 신규 구현(미배선)이다.

### 12.1 현황 (준비됨 vs 미구현)

| 구성요소 | 상태 | 위치 |
|---|---|---|
| 적/포탑/임팩트 토픽 발행 | ✅ 준비됨 | `/tank/enemy/{pose,state,heading}`, `/tank/api/update_bullet/*` ([bridge_node.py](../src/ros_bridge/ros_bridge/bridge_node.py)) |
| 기하 위협 함수(거리+LoS) | ✅ 준비됨 | `check_los`·`is_threat_active`(Tank 20m+LoS) ([potential_field_node.py](../src/potential/potential/potential_field_node.py)), 미러 [threat_geometry.py](../scripts/recon_eval/threat_geometry.py) |
| goal 변경 구독 → 자동 재계획 | ✅ 준비됨 | [map_astar_planner_node.py](../src/path_planning/path_planning/map_astar_planner_node.py) `goal_pose_cb`(약 558-569) |
| scenario2_map `targets`(tank) 생성 | ✅ 준비됨 | [build_scenario2_map.py](../scripts/build_scenario2_map.py) 약 175-183 |
| decision/FSM 노드 | ❌ 미구현 | 신규 |
| known/new 매칭 | ❌ 미구현 | 신규(아래 재활용) |
| `/tank/engage/{request,result}` | ❌ 미존재 | 신규 |
| `targets` 런타임 로드 | ❌ 미구현 | 신규(파일에만 기록) |
| 복귀 goal 발행 | ❌ 미구현 | 신규(구독측은 준비됨) |
| turret aim/fire | ⏳ 팀원 | `make_action`의 turret/fire는 no-op |

### 12.2 보정 3가지 (재설계 아님)

1. **⚠️ known-tank 목록이 빈약 — perception이 약해서.** 현재 발견 tank는 2개(route_A 0, route_B 2; obs_count=1, conf~0.7).
   YOLO는 많이 봤지만 중복이라 **센서퓨전 확정분만** 발견맵에 남는 게 정상. FSM이 "known→교전 / new→위험도평가"인데
   **known이 비면 주행 중 대부분 tank가 'new'로 분류**돼 위험도 평가만 돈다. ⇒ (a) **제대로 된 정찰 run 뒤에야** known
   목록이 채워지고(현 데이터는 시뮬 검증 전이라 무의미), (b) FSM은 "known 빈약"에 **강건**해야 한다(known 0이어도 동작).
2. **교전 메시지 = `std_msgs/String` + JSON.** 이 repo는 커스텀 `.msg`를 안 쓰고 전부 String+JSON 관행. 계약 스키마:
   - `/tank/engage/request` → `{"target_id": str, "pose": {"x": float, "y": float}, "distance_m": float, "los": bool}`
   - `/tank/engage/result` → `{"target_id": str, "impact": {"x": float, "y": float}, "success": bool, "dist_to_target_m": float}`
3. **복귀 goal-swap = `/tank/goal/pose`에 출발지 발행만 하면 됨.** planner가 이미 구독→재계획(12.1). 설계 §8이 우려한 것보다 단순.
   - (자잘) controller에 `mission_type→fire_cmd` 토글이 있으나 `make_action`에서 미사용(죽은 토글, [tank_controller_node.py](../src/control/control/tank_controller_node.py) 약 305-308) — 교전 배선 때 정리.

**오늘 작업과 정합 ✓:** 위험도는 **perception(탐지된 위협)** 으로 산정하고 GT(정답맵)는 안 쓴다(정찰 위험도 재설계와 동일 원칙).
Tank는 heading이 없어 **반경+LoS**(FOV 콘 없음)로 판정 — `is_threat_active`의 Tank001 규칙과 일치.

### 12.3 오프라인 구현 순서 (다음 세션, 시뮬 불필요)

① decision 노드 골격(FSM: `FORWARD` / `ENGAGE` / `RETURN`) → ② known/new 매칭(map 좌표, [local_path_node.py](../src/path_planning/path_planning/local_path_node.py)
`_find_existing_discovered`·`merge_radius_by_class` 패턴 재활용) → ③ 기하 위험도 score(거리+LoS+노출, `threat_geometry`/potential
함수 재활용, **0~1 정규화**) → ④ engage 계약 + **mock turret 노드**(폐루프 골격 검증) → ⑤ 복귀 goal 발행 → ⑥ 단위테스트.
**실제 fire·turret 조준·임팩트 피드백만 시뮬+팀원 대기.**

- **패키지 배치(권장):** scenario/통합은 본인 도메인 → 신규 패키지(예 `mission`) 또는 `path_planning`에 decision 노드.
  ament_python `entry_points` 패턴([path_planning/setup.py](../src/path_planning/setup.py)). launch는 [tank_scenario2.launch.py](../src/control/launch/tank_scenario2.launch.py)에 노드 추가.
- **교전 중 위협 척력 상충(§6):** decision 노드가 현재 교전 표적 id를 발행 → APF가 그 표적을 위협 목록에서 제외(또는 교전 거리에서 척력 스케일↓).
- **복귀 임계:** 0~1 위험도 score 기준, 정찰 노출 통계(길이비)로 캘리브레이션.

> 정리: **설계대로 진행 가능.** 12.2의 3보정만 반영하면 됨. 단 known-tank는 **제대로 된 정찰 시뮬 run이 선행**돼야 의미 있음.
