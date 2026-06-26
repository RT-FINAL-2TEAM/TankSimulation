# TankSimulation 기술 상세 (Technical Deep-Dive)

> **이 문서는** 각 서브시스템이 **어떤 이론으로 · 어떤 방식으로 · 어떻게 · 어디에** 쓰이는지 코드 레벨까지 파고든 기술 문서다. 각 항목은 ① 이론(평이하게) → ② 이 프로젝트의 구현·핵심 파라미터/수식 → ③ 어디에 쓰이는지(데이터 흐름) 순서. 정확한 코드는 `파일:줄` 포인터로 연결. (갱신 2026-06-22)

**전체 폐루프:** `시뮬(Windows) ──HTTP── 브릿지 → [인지] → [판단] → [회피] → [제어] → 브릿지 ──HTTP── 시뮬`. 시뮬은 DDS가 아니라 **HTTP(Flask)** 로 통신.

**좌표 규약(필수):** Unity raw → map 변환은 [coordinate_utils.py](../src/lidar/lidar/coordinate_utils.py)에만 정의 — `map.x=raw.x`, `map.y=raw.z`, `map.z=raw.y`(Unity 수직축 raw.y가 map의 높이 z로). map 프레임은 2D(x,y) 평면 + 높이 z.

---

# 1. 센서 · 인지 (Perception)

브릿지가 받은 raw LiDAR·카메라 이미지를 **지형/장애물 분리 → YOLO 탐지 → LiDAR-카메라 융합 → 발견객체맵**으로 가공한다.

## 1.1 LiDAR — 지형/장애물 분리

**① 이론.** 3D 클러스터링은 비싸다. 대신 **격자 기반 국소 지면 추정(local ground height)** 으로 싸게 분리한다: 바닥을 xy 격자로 쪼개고, 각 셀의 "지면 높이 = 셀 내 최저점"으로 잡은 뒤, **높이차가 큰(가파른) 셀**에서 지면 위로 솟은 점을 장애물로 본다. 나무·벽 같은 수직 구조와 지형 기복을 모두 처리.

**② 구현.** [lidar_processor_node.py](../src/lidar/lidar/lidar_processor_node.py) → [terrain_utils.py:123-189](../src/lidar/lidar/terrain_utils.py) `split_terrain_obstacle_points`:
- raw 입력: `/tank/api/info/raw`(브릿지가 시뮬 `/info`의 `lidarPoints` 중계).
- 격자 셀 = `(floor(x/q), floor(z/q))`, `q = terrain_grid_resolution = 0.5m`.
- 셀 지면 = 셀 내 최저 raw.y. **steep 셀** = 셀 내 높이 span > `climb_limit(0.4m)` **또는** 이웃 셀 지면차 > climb_limit.
- 분류: steep 셀에서 `local_ground + obstacle_min_height(0.2m)` 위 = 장애물, 아래 = 지형. non-steep 셀은 전부 지형.
- 부산물 `roughness_score = mean_span / climb_limit`(지형 거칠기 지표).

**③ 어디 쓰이나.** 출력 토픽(map 프레임 PointCloud2):
| 토픽 | 내용 | 소비처 |
|---|---|---|
| `/tank/sensor/lidar/detected_points_map` | 장애물 점 | DBSCAN(tank_visual_perception), APF, planner |
| `/tank/sensor/lidar/terrain_points_map` | 지면 점 | ground_division(지형맵 누적) |
| `/tank/sensor/lidar/all_detected_points_map` | 전체 점 | 디버그/시각화 |

> **lidar 패키지가 LiDAR 전처리의 유일한 출처.** 다른 노드는 raw를 다시 안 건드린다.

## 1.2 카메라 + YOLO — 객체 탐지

**① 이론.** YOLO(v8, ultralytics)가 카메라 프레임에서 객체 bbox + 클래스 + confidence를 한 번에 추론(single-shot). NMS로 겹친 박스 정리.

**② 구현.** 설정 [yolo_detection.yaml](../src/vision/config/yolo_detection.yaml), 추론 [yolo_detector.py](../src/vision/vision/yolo_detector.py):
- 모델 `best_final.engine`(TensorRT/GPU) 우선, 없으면 `best_final.pt`. 입력 **416×416**.
- **클래스 4종** car(0)/tank(2)/rock(3)/house(4) + tent(5) 레거시. `wall`·`person`은 **ignored**(절대 반환 안 함).
- conf 임계 `model_confidence 0.10`(tank는 0.25), `iou 0.70`(NMS), `max_det 20`, `max_return 5`.
- **그림자 제거 전처리**(`remove_shadow_with_gaussian`): V채널 Gaussian blur로 조명 추정→정규화 + CLAHE → 그늘에서도 탐지 안정.
- 추적(ByteTrack/Kalman)은 기본 off → 객체 식별은 **지도좌표(tank_map) 기반**(프레임 끊김에 강함).

**③ 어디 쓰이나.** 브릿지 `/detect`로 이미지 받아 추론 → bbox 리스트(`className`/`bbox`/`confidence`) 반환 → **융합(1.3)** 에서 LiDAR 클러스터와 매칭.

## 1.3 LiDAR ↔ 카메라 융합

**① 이론.** 카메라는 "무엇(class)"을 알지만 "어디(3D 위치)"는 모르고, LiDAR는 반대다. 둘을 합쳐 **분류된 3D 객체**를 만든다: (a) LiDAR 점을 DBSCAN으로 군집화, (b) 3D 군집을 카메라 이미지에 **투영**, (c) YOLO bbox와 투영된 군집을 **기하 점수로 매칭**.

**② 구현.**
- **DBSCAN** [lidar_dbscan_cluster_node.py](../src/tank_visual_perception/tank_visual_perception/lidar_dbscan_cluster_node.py): `eps=1.5`, `min_samples=2`, **xy평면**(2D)에서 군집화 → `/tank/visual_perception/lidar_clusters`(centroid/bbox/count).
- **투영(핀홀 카메라)** [projection.py](../src/tank_visual_perception/tank_visual_perception/projection.py):
  - 초점거리 `fx = image_w / (2·tan(hfov/2))`, `fy = image_h / (2·tan(vfov/2))` (hfov 86°, vfov 60.2°).
  - 카메라 pose = `lidarOrigin + R(yaw,pitch,roll)·offset`. yaw=`turret_yaw + yaw_offset(-0.9)`, pitch=`turret_pitch + pitch_offset(1.5) + body_pitch_gain·playerBodyY`, offset=`[tx0.28, ty0.02, tz11.8]`. (카메라가 **포탑에 장착** → 포탑각이 투영에 들어감.)
  - 점 투영: `p_cam = Rᵀ·(p_world − cam_pos)`, `u = fx·x_cam/z_cam + cx`, `v = cy − fy·y_cam/z_cam`. z_cam≤0이면 뒤쪽(버림).
- **매칭 점수** [local_path_node.py:1097-1255](../src/path_planning/path_planning/local_path_node.py): 군집 centroid가 bbox(여유 35px) 안 → `center_norm`(앵커 기준 정규화 거리) + `0.025·거리` + `0.45·bbox면적` = score, **≤1.55**면 후보. class별 앵커(person은 bbox 하단 90%, 그 외 중앙 50%). 전역 1:1 그리디 배정(모호하면 버림, `ambiguity_delta 0.35`).
- **융합 strict화**([fusion_mapping.yaml](../src/path_planning/config/fusion_mapping.yaml)): `allow_angle_fallback=false`(각도-only 폴백 금지), `semantic_requires_cluster=true`(YOLO+클러스터 동시 매칭만 인정), `max_fusion_range 45m`, `min_detection_confidence 0.2`. → 오탐 억제.

**③ 어디 쓰이나.** 융합 결과(분류된 3D 객체) → **발견객체맵(1.4)** 으로 누적.

> **커버리지 한계(중요).** 발견맵엔 **YOLO 분류 + 클러스터 매칭된 것만** 들어간다. 카메라 FOV가 ~48°(전방)이고 포탑이 고정이라, 측면/미분류 장애물은 라이다가 봐도 발견맵에 안 들어간다. (정찰 weave §2.2가 이걸 일부 완화.)

## 1.4 발견객체맵 (Discovered Object Map)

**① 이론.** 프레임마다 들어오는 융합 객체를 **누적·평활·확정**한다: 가까우면 같은 객체로 보고 위치를 EMA로 갱신, 일정 관측 충족 시 confirmed, 오래 안 보이면 메모리에서 제거.

**② 구현** [local_path_node.py:720-790](../src/path_planning/path_planning/local_path_node.py) `update_discovered_map_locked`:
- `DiscoveredObject`(dataclass:115): object_id(`detected_{class}_{NNNN}`), class_name, map_x/y/z, confidence, observation_count, class_votes, is_confirmed 등.
- 매칭: class별 merge반경(rock 1.5/car 2.5/tank 3.0/house 4.0/tent 3.0m) 내 기존 객체 찾기(+track_id 보조).
- 갱신: 위치 EMA(`α=0.35`), class voting(confidence 가중 최다 class 채택), observation_count++.
- **confirmation**(`_refresh_confirmation:1370`): `observation_count ≥ min_confirm_observations(1)` & `age ≥ min_confirm_age_sec(0)` → confirmed.
- **memory decay**(:494): unconfirmed인데 `memory_decay_sec(10s)` 넘게 안 보이면 제거.
- 저장(`make_map_payload:1463`, `/tank/map/discovered/save` Trigger): confirmed만(`save_confirmed_only`) JSON으로(`position`은 다시 Unity raw 규약). 발행 `/tank/map/discovered/objects`.

**③ 어디 쓰이나.** (a) 실시간 → **APF** 회피, (b) RViz 마커, (c) 정찰 종료 시 저장 → **시나리오2 맵 생성**(§5)의 입력.

---

# 2. 주행 (판단 · 회피 · 제어)

**A\*(전역 큰 길) → lookahead(국소 목표) → APF(실시간 회피) → 컨트롤러(조향)** 의 4계층.

## 2.1 전역 경로 — A\* (가중)

**① 이론.** 단순 최단거리가 아니라 **가중 A\***: 못 가는 곳(장애물=∞)에 더해 "가기 싫은 곳"을 셀 비용으로 얹어 균형 잡힌 길을 찾는다.

**② 구현** [team_path_planning.py:21-450](../src/path_planning/path_planning/team_path_planning.py):
1. **격자** `create_grid(300,300,1.0)` (1m 셀).
2. **장애물 마킹** `add_obstacles`: bbox + inflate(정적 나무 **1.0m** / 동적 라이다 **5.0m**) → 셀 `1`(통행불가). 정적 출처 = `load_static_obstacles_from_map`(finalmap의 Tree/Rock/Wall + 발견 class별 반경 rock5/car3/house6/tent2.5/tank4).
3. **비용맵** `_build_cost_map`:
   - **① clearance**(통로 중앙): BFS로 각 셀→최근접 장애물 거리, `< CLEARANCE_DESIRED(5m)`면 `CLEARANCE_WEIGHT(0.4)·(5−거리)` 가산 → 벽에서 멀어짐.
   - **② side-bias**(루트 유지): 웨이포인트 x 기준선에서 의도 채널 반대로 `SIDE_TOL(7m)` 넘어가면 `SIDE_WEIGHT(2.0)·초과` 가산. A=서/B=동.
   - **③ 지형 거칠기**(게이트형): `terrain_grid` 있을 때만 셀별 `terrain_weight·roughness` 가산. 정찰엔 grid 없어 무영향. **(2026-06-24: 시나리오2도 `terrain_weight=0.0`으로 off — 주행을 정찰과 일치시킴; 지형격자는 경로 z-lift·시각화용으로만 로드. 험지 회피 재활성화는 launch에서 값↑.)**
4. **A\* 탐색** `astar_search`: 이동비용 직선1.0/대각1.414 `+ cost_map`, heuristic=대각거리. 막힌 셀 회피.
5. **스무딩** `smooth_path`: 시야 트인(`has_line_of_sight`) 두 점 직선 단축.
- **루트 웨이포인트** `plan_path_through_waypoints`: `routes.yaml`의 A/B를 순서 경유(채널 힌트), `valid_waypoints`로 이미 지나친(전차 뒤, z<start−1m) 것 버림. 지형격자/노드 로딩은 [map_astar_planner_node.py](../src/path_planning/path_planning/map_astar_planner_node.py) `_load_terrain_grid`.
- **route checkpoint 진행도 추적**(fix/control2 `947d578`): A\* path-point index는 replan마다 의미가 바뀌므로, 전차가 실제로 도달한 웨이포인트를 `route_checkpoint_index`(단조 증가, never_decrease, reach_radius 8m)로 별도 관리 → dynamic/emergency replan 후에도 **지난 checkpoint를 through에 다시 안 넣음**(backward-hook 방지).

**핵심 파라미터**(launch): inflate 5.0, route_clearance_weight 0.35, lookahead_distance 13.0, goal_tolerance 10.0, use_route_waypoints True, use_gt_obstacles False, **enable_dynamic_replan True**(emergency cluster replan 포함), straight_ws_weight 0.34(crawl pivot). (fix/control2 기준; 옛 0.8m/APF-on 값은 폐기.)

**③ 어디 쓰이나.** 전역경로 `/tank/global_path` + lookahead → 컨트롤러(APF 비활성 시 A\* 직접 추종). **시나리오2에선 경로 점 z를 지형 z_median+0.4m로 lift**(`publish_path`)해 RViz 지형 면 위에 표시; 정찰은 z=0.

## 2.2 로컬 경로 — lookahead + 정찰 weave

**① 이론.** APF/컨트롤러는 먼 목적지가 아니라 경로 위 **8m 앞 점(lookahead)** 을 쫓는다(시야 내 추종).

**② 구현** [map_astar_planner_node.py:336-381,843-890](../src/path_planning/path_planning/map_astar_planner_node.py): `find_lookahead_along_path`(현재위치를 경로에 투영→8m 전진 보간) → `/tank/path/lookahead_pose`.
- **정찰 weave**(`_recon_scan_target`, recon만 `recon_scan_enabled=true`): lookahead를 진행방향 수직으로 `A·sin(2π·s/λ)`(s=누적 호장, A 진폭/λ 파장) 오프셋 → APF가 그쪽으로 조향 → 전차가 **S주행** → 전방 카메라가 더 넓게 훑어 발견 커버리지↑. 시나리오2(mission)는 off → 직진.

**③ 어디 쓰이나.** APF의 인력 타깃, 컨트롤러의 목표.

## 2.3 국소 회피 — APF (Artificial Potential Field)

> **⚠️ 현재 상태(2026-06-24, fix/control2 `947d578`): APF 비활성.** `tank_autonomous_control.launch.py`에서
> `potential_field_node`를 주석 처리 → 회피는 **A\* 전역경로 only**(APF 합벡터↔heading 충돌로 W를 끊어 멈칫하던 것 제거).
> 아래 이론·수식·게인은 재활성화(launch 주석 해제) 시 그대로 유효. potential 패키지 코드는 유지됨.

**① 이론.** 전차를 **힘의 장** 위 입자로 본다: 목표는 **당기고(인력)**, 장애물은 **밀고(척력)**, 합력 방향으로 간다. 국소 최소(장애물 뒤 갇힘)는 **접선력**으로 빠져나오고, 적 위협은 **강한 위협 척력**으로 크게 우회.

**② 수식·구현** [potential_field_node.py:192-464,945-1096](../src/potential/potential/potential_field_node.py):
- **인력** `F_A = k_att·(target − pos)` (k_att 3.0).
- **척력** 각 장애물: `g=거리`, `g ≤ g*(influence 9m)`이면 `F_R = k_rep·(1/g − 1/g*)/g³·(pos−obs)` (k_rep 60.0). g* 밖은 0.
- **접선력** 척력 있는 장애물의 수직 방향 중 목표쪽으로 `tangent_gain_scale·‖F_R‖` (국소최소 탈출).
- **위협 척력**(`is_threat_active`/`check_los`): House002(25m·±30°FOV·LoS) / Tank001(20m·LoS) 활성 시 `k_threat 2000`(척력의 33배)로 크게 회피.
- **합력** `F = F_A+F_R+F_T+F_threat`(‖·‖≤20 클램프). **passthrough_when_clear**: 척력/위협이 `repulsive_eps(0.5)` 미만이면 lookahead를 그대로 통과(APF 투명) → 채터링 방지. 아니면 `local_target = pos + (F/‖F‖)·8m`.
- 전처리 `filter_obstacles_for_apf`: 전방 sector 140°·corridor 7m·voxel 1m·최대 300점.

**핵심 게인**(launch): k_att 3·k_rep 60·influence 9·threat_radius 25·k_threat 2000·local_target_distance 8·front_sector 140·corridor 7. (척력 160배 압도 진동을 k_att↑/k_rep↓로 재균형한 결과.)

**③ 어디 쓰이나.** `/tank/local_target/pose` → 컨트롤러. + 힘 벡터 마커(RViz).

## 2.4 제어 — 헤딩 PD + 속도 로직

**① 이론.** 목표 방향과 현재 헤딩의 **오차(yaw_error)** 를 줄이도록 좌/우 회전. 진동(weaving)은 **D항(각속도 피드백)** 으로 감쇠.

**② 구현** [tank_controller_node.py:264-457](../src/control/control/tank_controller_node.py):
- **조향 PD** `calculate_steering`: `desired_yaw=atan2(dx,dz)`, `yaw_error=norm(desired−current)`. `u = yaw_error − steering_kd(0.2)·yaw_rate`. deadband 5°(이하 직진), full 45°에서 weight 1.0 → `D`(우)/`A`(좌).
- **속도** `calculate_speed`: 목적지 10m 내 → STOP(도착). 오차 >60° → 제자리회전(STOP+조향). >30° → 감속전진(W 0.4). 그 외 → 순항(W, 오차 비례 감속).
- **끼임 탈출** `escape_command_if_needed`: 5s간 1.5m 미만 이동 → 후진 1.5s + pivot 1.5s.
- **목표 선택** `choose_target`: APF local_target(우선) > A* lookahead > goal (TTL 2s).
- `make_action` → `{moveWS, moveAD, turretQE/RF=0, fire=False}`. **fire는 현재 no-op**(시나리오2 미구현).

**핵심 파라미터**(launch): controller_hz 10, steering_kd 0.2, heading_deadband 5°, steering_full_error 45°, slowdown 30°, rotate_in_place 60°, goal_tolerance 10, stuck_check 5s.

**③ 어디 쓰이나.** `/tank/control/command` → 브릿지 → 시뮬 `/get_action`.

---

# 3. LLM — 정찰 루트 위험도 평가

**① 이론.** LLM은 **느린(이벤트당) 전술 판단**에만 쓴다(100ms 제어 루프엔 절대 안 넣음 — 지연/환각). 정찰 후 A/B 루트 비교 지표를 주고 추천·근거를 받는다.

**② 구현.**
- 모델 **qwen3:0.6b**(로컬 ollama `localhost:11434`), `num_predict 768`, `temperature 0`, `format json`.
- 파이프라인: `comparison.json` → [make_llm_input.py](../scripts/make_llm_input.py)(13지표 요약: reached/distance/sim_time/collision/enemy_count/exposure/obstacle_density/terrain σ/yolo_counts…) → [route_risk_node.py](../src/risk_analysis/risk_analysis/route_risk_node.py) → [llm_reporter.py](../src/risk_analysis/risk_analysis/llm_reporter.py)(전술참모 프롬프트, 입력값만 근거·환각금지) → JSON 출력 `{selected_route, risk_level{A,B}, confidence, summary, key_risks, recommended_behavior, used_evidence}`.
- **안전망:** 출력 스키마 검증·결측 보정, `used_evidence`는 입력 truth로 강제 덮어씀. `TANK_FORCE_ROUTE`로 강제 가능.
- 현재 ~14초(0.6b + 768토큰). 시나리오2 전술결정용은 `num_predict 50`으로 1~2초 설계(§5).

**③ 어디 쓰이나.** `/tank/risk/route_report` → 브릿지 MFD(웹 [live_view.py](../src/ros_bridge/ros_bridge/live_view.py))에 추천·근거 표시. 파일 `route_risk_result.json`.

---

# 4. 정찰 위험도 · 은밀성 평가 기준 (핵심)

> 정찰의 **핵심 산출물**. 위험도는 **5요소 가중 합 수식**으로 실제 구현돼 있다.

**① 이론.** 루트의 위험도를 5개 정규화 지표의 가중합으로 0~1 점수화. 은밀성↑ = 위험도↓.

**② 수식** [generate_recon_report.py:49-359](../scripts/generate_recon_report.py): `Risk = Σ Wᵢ·normᵢ`

| 요소 | 가중 | 의미 | 정규화 기준 |
|---|---|---|---|
| **W1 위협 발견** | 0.35 | YOLO 위협클래스(person/tank/house) 가중합 + GT 자산 | / 20 |
| **W2 노출 시간** | 0.30 | 위협 FOV+LoS **활성 누적 시간**(가장 핵심 은밀성) | / 60s |
| **W3 우회 비율** | 0.10 | 실제거리 / 직선거리(현재 proxy, replan 미로깅) | (r−1)/(2−1) |
| **W4 지형 거칠기** | 0.15 | σ(pitch) + σ(roll) (차체 흔들림) | / 8° |
| **W5 실행 시간** | 0.10 | sim_time | / 300s |

- 가중치 설계: **W1+W2=65%**(위협·노출 우선=은밀성), W3~W5=35%(효율·안정).

**위협 기하** [scripts/recon_eval/threat_geometry.py](../scripts/recon_eval/threat_geometry.py):
- **House002**: 반경 25m + 적 시야 **±30° FOV** + **LoS 비차폐** 모두 만족해야 "활성". **Tank001**: 20m + LoS.
- `check_los`: 전차→위협 선분이 GT 장애물 bbox와 교차하면 차폐(segment-AABB 레이캐스트). `is_threat_active`: 위 조건 종합(APF의 동명 로직과 일치).
- [recon_logger.py](../src/path_planning/path_planning/recon_logger.py)가 주행 중 trajectory(0.5m 간격)·exposure·body각(pitch/roll)·YOLO 카운트를 기록 → `route_{A,B}.json`. 보고서 생성 시 `compute_exposure`가 trajectory에 위협기하를 재적용해 노출 산정 + 노출지도 PNG(`exposure_A/B.png`).

**③ 현재 상태 (중요 주의).** finalmap은 **의도적으로 "깨끗한 정찰 base"**(위협 House002·적전차 제거, 나무만 유지)다. 그래서 **맵에 활성 위협이 없어 W2 노출 = 0**(실측 `route_A.json` exposure.total_dwell_s=0.0). 즉:
- **활성:** W1(YOLO 발견 카운트), W4(지형 σ), W5(시간), LLM 비교.
- **코드 완비·미관측:** W2(노출) — 위협을 맵에 배치하면 즉시 동작(시나리오2가 tank 배치).
- **proxy:** W3(우회) — replan 카운트 미로깅, 거리비로 대체.

**route_{A,B}.json 스키마**: `result`(reached/collisions/sim_time/distance) · `trajectory[[t,x,z,yaw]]` · `obstacles_detected` · `exposure`(events/dwell) · `vision_yolo`(counts/detections) · `terrain_roughness`(pitch_std/roll_std) · `diagnostics`. 실측 route_A 예: distance 344m, sim 60s, collisions 0, YOLO rock232/car86/house61/tank110.

---

# 5. 시나리오2 — 정확한 계획

> 정본 [SCENARIO2_DESIGN.md](SCENARIO2_DESIGN.md). 정찰 위에서 **교전 + 조건부 복귀** 임무. **현재 맵 인계 토대만 구현(d1b8e8a), 교전 FSM은 설계만.**

**① 개념.** 정찰로 파악한 적 전차를 사격 용이 지점에서 제거하며 목적지까지 주행, 정찰에 없던 새 적 전차 출현 시 위험도 평가 후 복귀. **임무 루트 = A 고정.**

**② FSM:**
```
[출발]→[전진] ──일반장애물──▶ APF 회피 ──▶[전진]
          ├─ known tank 사거리+LoS ─▶[교전]사격→임팩트근접→격파 ─▶[전진]
          ├─ new tank 탐지 ─▶[위험도평가] ──임계초과──▶[복귀] goal=출발지
          │                                  └─이하──▶ 회피·[전진]
          ▼
      [목적지]─▶[최종 교전]─▶[성공]
```

**표적 분류:**
- **known tank**(경로상): 정찰 발견객체맵의 tank + 주행 탐지를 **map 좌표 매칭(±3m)**. **체력 없음** → 임팩트 근접 시 *논리 격파*.
- **new tank**: 매칭 실패 → 위험도 평가 트리거.
- **목적지 적전차**: 시뮬 `/info`(`/tank/enemy/pose`·state) **체력 있음**.

**교전 인터페이스(계약):** decision 노드 → `/tank/engage/request`(표적 map좌표+id) → **팀원 turret 제어**가 구독·조준(`turretQE`/`turretRF`)·발사(`fire`) → 결과 `/tank/engage/result`(임팩트 좌표·거리·성공). 성공 = 임팩트↔표적 거리 < R(정지표적이라 정밀탄도 불요). 재료(적/포탑/임팩트 토픽)는 브릿지가 이미 발행.

**위험도 결정(새 적 전차):** ① **기하 위험도**(거리+적 FOV 노출+LoS 엄폐+위협 수, APF `is_threat_active`/`check_los` 재활용) = 빠른 안전망, ② **LLM 전술결정**(상황 요약→`{decision: 복귀/돌파, reason}`, `num_predict 50`으로 1~2초). 시간 내 LLM=LLM / 지연·부재=기하 fallback. MFD 표시.

**복귀 goal-swap:** 위험 임계 초과 → goal 목적지→출발지 스왑→planner 재계획. **미구현**(후보: controller goal 훅 / planner 도착 재계획).

**역할:** 본인 = FSM·표적선정·위험도/복귀 결정·사격 WP 도출·goal-swap·통합 + 인터페이스 정의. 팀원(control) = turret 조준·발사. 미완 시 mock 노드로 병행.

**③ 완료 vs 미구현:**
| 항목 | 상태 |
|---|---|
| 맵 인계(scenario2_map: 발견장애물 회피 + tank `targets` + 지형비용) | ✅ d1b8e8a |
| 시나리오2 launch(route A, mission_type) | ✅ |
| 교전 decision 노드 / known·new 매칭 | ❌ 설계만 |
| fire 하달(현재 controller `fire=False` 고정) | ❌ (팀원 turret) |
| 복귀 goal-swap | ❌ |
| LLM 전술결정 연동 | ❌ |
| 기하 위험도 점수 노드화 | ⚠ 로직 재활용 가능, 미배선 |

**맵 인계 파이프라인(구현됨):** 정찰 `discovered_objects_route_{A,B}.map` + `terrain_map_route_{A,B}.npz` → [build_scenario2_map.py](../scripts/build_scenario2_map.py)(A+B 합본) → `scenario2_map.map`(A* 장애물) + `scenario2_terrain.json`(A* 비용) + `.npz`(뷰) → `ros2 launch control tank_scenario2.launch.py`.

---

# 6. 데이터 흐름 종합 (폐루프 토픽)

| 단계 | 노드 | 주요 입력 → 출력 토픽 |
|---|---|---|
| 시뮬↔ROS | ros_bridge | `/info`·`/detect` → `/tank/api/info/raw`·player/enemy/turret pose; `/tank/control/command` → `/get_action` |
| 인지(라이다) | lidar_processor | info/raw → `/tank/sensor/lidar/{detected,terrain,all}_points_map` |
| 인지(YOLO) | vision | `/detect` 이미지 → bbox 응답 |
| 인지(융합) | tank_visual_perception | detected_points_map → `/tank/visual_perception/lidar_clusters`; + 투영 |
| 판단(A*) | map_astar_planner | player/pose + 정적맵 + (시나리오2)지형 → `/tank/global_path`·`/tank/path/lookahead_pose` |
| 판단(발견) | local_path | 융합+클러스터 → `/tank/map/discovered/objects` + 정찰 로깅(route_*.json) |
| 회피(APF) | potential | lookahead + 장애물 + 위협 → `/tank/local_target/pose` + 힘 벡터 |
| 제어 | control | local_target/lookahead/goal → `/tank/control/command`(W/A/S/D) |
| 보조 | rviz_visualization / ground_division / risk_analysis | 마커 / 지형 메쉬·NPZ / 루트 LLM 위험도 |

---

## 더 볼 문서
- [SIMULATOR_API.md](SIMULATOR_API.md) — 시뮬 HTTP API 규약
- [README_REORGANIZED_STRUCTURE.md](README_REORGANIZED_STRUCTURE.md) — 패키지·토픽 그래프
- [SCENARIO2_DESIGN.md](SCENARIO2_DESIGN.md) — 시나리오2 설계 정본
- [src/potential/README_LECTURE_APF_UPDATE.md](../src/potential/README_LECTURE_APF_UPDATE.md) — APF 수식 상세
- [ROS_파일들_실행_방법.txt](ROS_파일들_실행_방법.txt) — 실행 절차
