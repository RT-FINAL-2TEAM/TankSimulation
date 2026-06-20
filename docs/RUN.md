# 실행 가이드 (RUN) — 정본

> 팀 공통 실행 방법의 **단일 출처**. 모든 설정은 `.env` 하나에 모으고, 실행은 `scripts/run_*.sh` 래퍼로 통일한다.
> (CLAUDE.md·README의 실행절은 이 문서를 가리킨다.)

## 1. 연결 모델 (왜 IP가 두 개인가)

```
   [ Windows: Tank Simulator ]  --HTTP-->  [ Ubuntu: ros_bridge ]  <--topics-->  [ ROS2 자율주행 스택 ]
        (클라이언트)                              (서버, 포트 5000)
```

- **시뮬(윈도우)이 브릿지(우분투)로 접속**한다. 그래서 IP가 둘:
  1. **시뮬 UI 엔드포인트** = `내우분투IP:5000`  ← 우분투 IP 확인: `hostname -I`
  2. **`.env`의 `TANK_ALLOWED_CLIENTS`** = 접속을 허용할 **시뮬 PC IP**(또는 서브넷 `192.168.0.*`)
- 포트는 **각자 PC의 로컬 포트**라 팀 전원 `5000`을 써도 안 섞인다(IP로 구분). 팀원끼리 ROS 그래프가 섞이는 건 `ROS_DOMAIN_ID`/`ROS_LOCALHOST_ONLY`로 격리(`~/.bashrc`에 이미 설정) — HTTP 포트와 무관.

## 2. 최초 1회 설정

```bash
cp .env.example .env          # 템플릿 복사
# .env 상단 "여기만 수정"에서:
#   - TANK_ALLOWED_CLIENTS = 시뮬 PC IP 또는 서브넷(192.168.0.*)  ← IP 바뀌어도 그대로
#   - TANK_YOLO_MODEL_PATH = 본인 절대경로
hostname -I                   # 내 우분투 IP 확인 → 시뮬 UI 엔드포인트에 "그IP:5000" 입력
colcon build --symlink-install && source install/setup.bash
```
※ `.env`는 워크스페이스 루트에서 `ros2 run` 시 **자동 로드**된다 — 명령에 `TANK_*=...`를 붙일 필요 없다.

## 3. 실행 (래퍼 스크립트 — 모두 같은 명령을 씀)

> **실행 순서가 중요**: 브릿지를 먼저 띄운 뒤 시뮬을 (재)시작해야 `/init` 핸드셰이크가 닿는다.

| 역할 | 명령 | 설명 |
|---|---|---|
| **브릿지** | `scripts/run_bridge.sh --mode auto` | HTTP↔ROS 중계. `monitor`(수동)/`auto`(자율). 시뮬 UI에 넣을 주소를 출력해 줌 |
| 브릿지(정찰) | `scripts/run_bridge.sh --mode auto --reset` | + 루트 사이 시뮬 자동 리셋(`TANK_EPISODE_CONTROL`) |
| **자율 스택** | `scripts/run_stack.sh --route A` | lidar+인지+A*+APF+컨트롤러. `--route A\|B`, `--mission recon\|mission\|return` |
| **RViz** | `scripts/run_rviz.sh` | 경로/클러스터/힘벡터/지형 시각화 |
| **정찰 A→B** | `scripts/run_recon.sh` | A→B 자동 시퀀스(전제: 위 `run_bridge.sh --mode auto --reset` + 시뮬 시작) |

스크립트는 ROS·install·`.env`를 알아서 적용한다. 맨몸 명령으로도 동일하게 동작한다(예: `ros2 run ros_bridge ros_bridge`).

**전형적 흐름(자율주행)**: 터미널1 `run_bridge.sh --mode auto` → 시뮬 시작 → 터미널2 `run_stack.sh --route A` → (선택) 터미널3 `run_rviz.sh`.

## 4. 설정 변수 (.env)

전체 목록·기본값은 [`.env.example`](../.env.example) 참고. 핵심만:

| 변수 | 의미 |
|---|---|
| `TANK_ALLOWED_CLIENTS` | 접속 허용 시뮬 IP. 정확IP `192.168.0.30` / 와일드카드 `192.168.0.*` / CIDR `192.168.0.0/24`. loopback 항상 허용 |
| `TANK_MODE` | `monitor`(수동) \| `auto`(ROS 자율). 잘못된 값은 monitor로 강제 |
| `TANK_BRIDGE_PORT` | 브릿지 포트(시뮬 UI와 일치). **팀 표준 5000** |
| `TANK_EPISODE_CONTROL` | `true`면 루트 사이 시뮬 자동 리셋(정찰·RL). `run_recon.sh`/`--reset`가 켬 |
| `TANK_YOLO_MODEL_PATH` | YOLO 모델(.pt 권장) 절대경로 |

## 5. 서비스 콜 (수동)

```bash
ros2 service call /tank/terrain/finalize_map std_srvs/srv/Trigger "{}"   # 지형맵 저장(주행 중 누적 → 압축저장)
ros2 service call /tank/terrain/reset_map    std_srvs/srv/Trigger "{}"   # 지형 누적 리셋
ros2 service call /tank/map/discovered/save  std_srvs/srv/Trigger "{}"   # 발견객체 맵 저장
ros2 service call /tank/map/discovered/clear std_srvs/srv/Trigger "{}"   # 발견객체 맵 클리어
# 에피소드 제어(reset 검증/수동): 브릿지가 --reset로 떠 있어야 동작
ros2 topic pub --once /tank/episode/control std_msgs/msg/String "{data: reset}"
```

## 6. 안 될 때 (연결 체크리스트)

1. **포트**: 시뮬 UI 포트 == `.env`의 `TANK_BRIDGE_PORT`(5000) 인가.
2. **허용 IP**: 브릿지 로그에 `[BLOCKED OTHER CLIENT] <ip>` 가 뜨면 그 `<ip>`가 시뮬 PC IP → `.env` `TANK_ALLOWED_CLIENTS`에 추가하거나 서브넷(`192.168.0.*`)으로.
3. **우분투 IP**: 시뮬 UI 엔드포인트가 `hostname -I` 결과와 맞는가. (DHCP로 바뀌었을 수 있음)
4. **순서**: 브릿지를 먼저 → 그다음 시뮬 (재)시작.
5. **모드**: 자율주행인데 안 움직이면 `--mode auto`인지(브릿지 `/health`의 `mode`), 좀비 노드 `pkill` 후 재시작.
6. **방화벽**: 우분투에서 5000 포트 인바운드 허용.

> 사용 가능한 RViz launch는 `tank_rviz.launch.py`(기본), terrain 변형 `tank_recon_collect_terrain.launch.py`·`tank_recon_apply_terrain.launch.py`. (구 문서의 `tank_recon_map_rviz`·`tank_recon_mission_map_rviz`는 **존재하지 않음**.)
