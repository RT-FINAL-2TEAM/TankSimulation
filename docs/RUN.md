# 실행 가이드 (RUN) — 정본

> 팀 공통 실행 방법의 **단일 출처**. 모든 설정은 `.env` 하나에 모으고, 실행은 `scripts/run_*.sh` 래퍼로 통일한다.
> (CLAUDE.md·README의 실행절은 이 문서를 가리킨다.)

## 1. 연결 모델 (시뮬은 포트만, IP는 프록시가 맞춘다)

시뮬레이터에는 **포트 설정만** 있고 IP 입력이 없다(localhost로만 접속). 그래서 **윈도우에서 `tank_proxy.py`**를 띄워
시뮬의 요청을 우분투 브릿지로 중계한다. IP를 맞추는 곳은 시뮬이 아니라 **프록시의 `UBUNTU_SERVER`** 다.

```
[ Windows PC ]                                                    [ Ubuntu PC ]
 Tank Simulator ──127.0.0.1:5000──▶ tank_proxy.py ──http://<우분투IP>:5000──▶ ros_bridge ──topics──▶ 자율 스택
  (포트만 설정)                       (UBUNTU_SERVER=우분투IP)                 (TANK_ALLOWED_CLIENTS=윈도우 IP 허용)
```

- **시뮬**: 엔드포인트 포트 = 프록시 포트(기본 `5000`). IP는 localhost 고정(설정 없음).
- **`tank_proxy.py`(윈도우에서 실행)**: `UBUNTU_SERVER=http://<우분투IP>:5000` ← **여기서 우분투 IP 지정**. 우분투 IP 확인: 우분투에서 `hostname -I`. (프록시는 repo 루트의 `tank_proxy.py`, 같은 폴더 `.env`의 `UBUNTU_SERVER`/`PROXY_PORT`도 읽음.)
- **`ros_bridge`(우분투)**: `.env`의 `TANK_ALLOWED_CLIENTS`에 **윈도우 PC IP**(또는 서브넷 `192.168.0.*`)를 허용. (프록시가 윈도우에서 요청을 보내므로 출발 IP = 윈도우 IP.)
- 포트는 **각자 PC 로컬**이라 팀 전원 `5000`을 써도 안 섞인다(IP로 구분). 팀원끼리 ROS 그래프 섞임은 `ROS_DOMAIN_ID`/`ROS_LOCALHOST_ONLY`로 격리 — HTTP 포트와 무관.

## 2. 최초 1회 설정

**우분투(브릿지)**:
```bash
cp .env.example .env          # 템플릿 복사
# .env "여기만 수정": TANK_ALLOWED_CLIENTS = 윈도우 IP 또는 서브넷(192.168.0.*)  ← IP 바뀌어도 그대로
#                    TANK_YOLO_MODEL_PATH = 본인 절대경로(GPU면 best_final.engine / GPU 없으면 best_final.pt)
hostname -I                   # 내 우분투 IP 확인 → 윈도우 tank_proxy.py의 UBUNTU_SERVER에 "http://그IP:5000"
colcon build --symlink-install && source install/setup.bash
```
**윈도우(프록시·시뮬)**: `tank_proxy.py`를 띄우고(`UBUNTU_SERVER=http://<우분투IP>:5000`), 시뮬 엔드포인트 포트를 프록시 포트(5000)로.
```cmd
set UBUNTU_SERVER=http://<우분투IP>:5000
python tank_proxy.py            REM 127.0.0.1:5000 에서 받아 우분투 브릿지로 중계
```
※ 우분투 `.env`는 워크스페이스 루트에서 `ros2 run` 시 **자동 로드**된다 — 명령에 `TANK_*=...`를 붙일 필요 없다.

## 3. 실행 (래퍼 스크립트 — 모두 같은 명령을 씀)

> **실행 순서가 중요**: 브릿지를 먼저 띄운 뒤 시뮬을 (재)시작해야 `/init` 핸드셰이크가 닿는다.

| 역할 | 명령 | 설명 |
|---|---|---|
| **브릿지** | `scripts/run_bridge.sh --mode auto` | HTTP↔ROS 중계. `monitor`(수동)/`auto`(자율). 윈도우 프록시 `UBUNTU_SERVER`에 넣을 우분투 주소를 출력해 줌 |
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
| `TANK_ALLOWED_CLIENTS` | 접속 허용 IP(=프록시가 도는 윈도우 PC IP). 정확IP `192.168.0.30` / 와일드카드 `192.168.0.*` / CIDR `192.168.0.0/24`. loopback 항상 허용 |
| `TANK_MODE` | `monitor`(수동) \| `auto`(ROS 자율). 잘못된 값은 monitor로 강제 |
| `TANK_BRIDGE_PORT` | 브릿지 포트(프록시 `UBUNTU_SERVER` 포트와 일치). **팀 표준 5000** |
| `TANK_EPISODE_CONTROL` | `true`면 루트 사이 시뮬 자동 리셋(정찰·RL). `run_recon.sh`/`--reset`가 켬 |
| `TANK_YOLO_MODEL_PATH` | YOLO 모델 절대경로. **기본 `best_final.engine`(TensorRT, GPU 필요)** / GPU 없는 PC는 `best_final.pt` |
| (윈도우 프록시) `UBUNTU_SERVER` | `tank_proxy.py`가 중계할 우분투 브릿지 주소 `http://<우분투IP>:5000` |

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

1. **포트**: 시뮬 포트 == 프록시 포트(`PROXY_PORT`) == `.env` `TANK_BRIDGE_PORT`(5000) == 프록시 `UBUNTU_SERVER`의 포트, 4곳 일치.
2. **프록시 IP**: `tank_proxy.py`의 `UBUNTU_SERVER`가 `http://<우분투IP>:5000` (우분투 `hostname -I`와 일치)인가. 프록시 로그 `[PROXY] ... -> <target>` 확인.
3. **허용 IP**: 브릿지 로그 `[BLOCKED OTHER CLIENT] <ip>` 가 뜨면 그 `<ip>`(=윈도우 PC IP)를 `.env` `TANK_ALLOWED_CLIENTS`에 추가하거나 서브넷(`192.168.0.*`)으로.
4. **순서**: 브릿지를 먼저 → 그다음 (윈도우)프록시 → 시뮬 (재)시작.
5. **모드**: 자율주행인데 안 움직이면 `--mode auto`인지(브릿지 `/health`의 `mode`), 좀비 노드 `pkill` 후 재시작.
6. **방화벽**: 우분투에서 5000 포트 인바운드 허용(윈도우→우분투).

> 사용 가능한 RViz launch는 `tank_rviz.launch.py`(기본), terrain 변형 `tank_recon_collect_terrain.launch.py`·`tank_recon_apply_terrain.launch.py`. (구 문서의 `tank_recon_map_rviz`·`tank_recon_mission_map_rviz`는 **존재하지 않음**.)
