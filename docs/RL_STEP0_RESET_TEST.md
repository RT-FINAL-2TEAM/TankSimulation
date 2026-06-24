# Step 0 — 시뮬 리셋 게이팅 테스트 (RL 도입 전제 검증)

> 목적: **ROS가 시뮬 에피소드를 프로그램적으로 리셋할 수 있는가**를 라이브로 확정한다.
> 이 결과가 강화학습 도입 경로를 가른다(계획: `~/.claude/plans/indexed-squishing-wadler.md`).
> - **리셋 O** → 실제 시뮬에서 온라인 RL 가능 → control-first(M1)부터 시도 가능.
> - **리셋 X** → 오프라인 2D gym으로 APF 대체(L1)를 첫 타깃으로.

## 배경 (왜 이게 핵심인가)

공식 API(`docs/SIMULATOR_API.md`)상 **`/info` 응답의 `control` 필드**는 에피소드 제어 채널이고,
범위가 `pause`(일시정지)·`reset`(초기화)이다. 브릿지에는 이 훅이 있었지만 지금까지 `control:""`(미사용)만
보내왔다. 이번에 `/tank/episode/control` 토픽으로 `reset/pause/start`를 큐잉하면 **다음 `/info` 응답**에
실어 보내도록 배선했다(기본 off). 남은 미지수는 **실제 Unity 빌드가 `control:reset`을 존중하는지**뿐 —
그게 이 테스트로 확인할 전부다.

관련 코드: [config.py `EPISODE_CONTROL_ENABLED`](../src/ros_bridge/ros_bridge/config.py),
[bridge_node.py `on_episode_control`/`take_episode_control`](../src/ros_bridge/ros_bridge/bridge_node.py),
[app_routes.py `/info`](../src/ros_bridge/ros_bridge/app_routes.py).

## 전제조건

- 시뮬 PC가 켜져 있고, 브릿지 IP/포트가 맞다(`TANK_ALLOWED_CLIENTS`에 시뮬 IP 포함).
- `logMode`는 init_config에서 항상 `True`라 `/info`가 주기적으로 들어온다(이 채널이 동작하는 조건).
- 브릿지를 **`auto` 모드 + `TANK_EPISODE_CONTROL=true`** 로 띄운다.

## 실행 절차

```bash
# 터미널 1 — 브릿지(에피소드 제어 켜고). 시뮬 IP는 실제값으로.
source install/setup.bash
TANK_ALLOWED_CLIENTS=127.0.0.1,::1,192.168.0.30 TANK_MODE=auto TANK_EPISODE_CONTROL=true \
  ros2 run ros_bridge ros_bridge
```

브릿지가 뜬 뒤 **시뮬레이터를 (재)시작**해 `/init` 핸드셰이크가 닿게 한다. 전차가 출발해 잠깐 주행하게 둔 다음:

```bash
# 터미널 2 — 리셋 1회 요청. 브릿지가 "episode control queued..."를 찍고,
# 다음 /info 응답에서 "[info] sending episode control to sim: reset"을 찍어야 한다.
source install/setup.bash
ros2 topic pub --once /tank/episode/control std_msgs/msg/String "{data: reset}"

# (선택) 일시정지 / 재개도 같은 방식으로 확인
ros2 topic pub --once /tank/episode/control std_msgs/msg/String "{data: pause}"
ros2 topic pub --once /tank/episode/control std_msgs/msg/String "{data: start}"
```

## 관찰·기록할 것 (4가지)

| # | 항목 | 보는 법 |
|---|---|---|
| 1 | **reset이 동작하는가** | 시뮬 화면에서 전차가 시작 위치(`blStartX/Y/Z`)로 되돌아가는가? |
| 2 | **무엇이 복원되는가** | 전차 pose만? health·장애물·적전차·경과시간(`time`)도 초기화되는가? (`/info`의 `playerHealth`,`time`,`enemyPos` 관찰) |
| 3 | **리셋 지연시간** | 토픽 발행 → 시뮬 초기화까지 몇 초? (에피소드 throughput을 좌우 → RL 학습 속도 추정) |
| 4 | **스폰 랜덤화 가능한가** | 리셋이 `/init` 재요청을 유발하면, 브릿지가 `blStart*`를 바꿔 초기상태 랜덤화 가능(`TANK_BLUE_START_X/Y/Z` env로 실험) |

`pause`/`start`도 honor되는지 같이 메모(에피소드 경계를 깔끔히 끊는 데 유용).

## 결과에 따른 분기

- **리셋 O (전차가 초기화됨)** → **분기 A**: 실시뮬 온라인 RL 가능. `BridgeTankEnv.reset()`이 이 토픽을
  그대로 호출하게 만들고, M1(control-first)로 RL 루프 전체를 검증 → M2(APF 대체)로 확장.
- **리셋 X (무시됨)** → **분기 B**: 온라인 RL 비실용. 자립형 `Kinematic2DEnv`로 오프라인 학습,
  L1(APF 대체)을 첫 타깃으로. (이 경우 control-first는 sim2real 위험으로 보류.)

> 안전: `TANK_EPISODE_CONTROL`을 빼면(기본 off) `/info`는 예전처럼 `control:""`만 보낸다 —
> 이 배선은 평상시 주행에 아무 영향이 없다.
