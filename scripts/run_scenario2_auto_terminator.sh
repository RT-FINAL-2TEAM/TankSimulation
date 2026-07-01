#!/usr/bin/env bash
set -Eeuo pipefail

# run_scenario2_auto_terminator_v2.sh
#
# 목적:
#   Terminator 4분할 창으로 시나리오2를 자동 실행하면서 각 pane 로그를 실시간 확인한다.
#
# v2 수정점:
#   - scenario2_map.map이 없으면 manager pane에서 자동으로 build_scenario2_map.py 실행
#   - RViz pane은 scenario2_map.map이 생길 때까지 기다린 뒤 launch
#   - 따라서 static_map_loader_node가 map file not found로 죽는 문제 방지
#
# Pane 구성:
#   좌상: T1 ros_bridge
#   우상: T2 scenario2 RViz, map 파일 대기 후 실행
#   좌하: T3 scenario2 자동 manager
#   우하: T4 debug monitor
#
# 사용:
#   cd ~/tankcc
#   ./scripts/run_scenario2_auto_terminator.sh
#
# 옵션:
#   --no-rviz        RViz 실행 안 함
#   --skip-reset     scenario2 실행 전 simulator reset 생략
#   --rebuild-map    기존 map이 있어도 build_scenario2_map.py 재실행
#   --no-build-map   map 자동 생성 안 함. 없으면 에러
#
# 주의:
#   scripts/run_scenario2_scenario.py가 scenario2 자율/교전 스택을 내부에서 실행한다.
#   따라서 이 스크립트는 tank_scenario2.launch.py를 별도로 실행하지 않는다.

WORKSPACE="${TANK_WS:-$HOME/tankcc}"
LOG_ROOT="${LOG_ROOT:-$WORKSPACE/logs}"
RUN_ID="scenario2_terminator_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$LOG_ROOT/$RUN_ID"

USE_RVIZ="true"
SKIP_RESET="false"
BUILD_MODE="auto"     # auto | rebuild | never
BRIDGE_HEALTH_URL="${BRIDGE_HEALTH_URL:-http://127.0.0.1:5000/health}"
SCENARIO2_MAP_REL="recon_reports/recon_map/scenario2_map.map"
SCENARIO2_TERRAIN_REL="recon_reports/recon_map/scenario2_terrain.npz"
MAP_WAIT_SEC="${MAP_WAIT_SEC:-180}"

usage() {
  cat <<USAGE
Usage: $0 [--no-rviz] [--skip-reset] [--rebuild-map] [--no-build-map]

Options:
  --no-rviz       RViz pane에서 RViz를 실행하지 않음
  --skip-reset    run_scenario2_scenario.py 실행 전 simulator reset 요청 생략
  --rebuild-map   기존 scenario2_map.map이 있어도 build_scenario2_map.py 재실행
  --no-build-map  map 자동 생성을 하지 않음. map이 없으면 에러
  -h, --help      도움말 출력

Environment:
  TANK_WS             ROS2 workspace 경로(default: ~/tankcc)
  LOG_ROOT            로그 저장 상위 경로(default: ~/tankcc/logs)
  MAP_WAIT_SEC        RViz pane의 scenario2_map.map 대기 시간(default: 180)
  BRIDGE_HEALTH_URL   bridge health URL(default: http://127.0.0.1:5000/health)
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-rviz)
      USE_RVIZ="false"
      shift
      ;;
    --skip-reset)
      SKIP_RESET="true"
      shift
      ;;
    --rebuild-map)
      BUILD_MODE="rebuild"
      shift
      ;;
    --no-build-map)
      BUILD_MODE="never"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown option: $1"
      usage
      exit 2
      ;;
  esac
done

if ! command -v terminator >/dev/null 2>&1; then
  echo "[ERROR] terminator가 설치되어 있지 않습니다."
  echo "설치:"
  echo "  sudo apt update && sudo apt install -y terminator"
  exit 1
fi

if [[ ! -d "$WORKSPACE" ]]; then
  echo "[ERROR] workspace not found: $WORKSPACE"
  exit 1
fi

mkdir -p "$LOG_DIR"

RUNTIME_DIR="$WORKSPACE/scripts/.terminator_runtime"
mkdir -p "$RUNTIME_DIR"

COMMON_ENV="$RUNTIME_DIR/common_env.sh"
cat > "$COMMON_ENV" <<EOF
#!/usr/bin/env bash
set -Eeuo pipefail

safe_source() {
  local setup_file="\$1"
  set +u
  source "\$setup_file"
  set -u
}

cd "$WORKSPACE"

if [[ ! -f /opt/ros/humble/setup.bash ]]; then
  echo "[ERROR] /opt/ros/humble/setup.bash not found"
  exec bash
fi

safe_source /opt/ros/humble/setup.bash

if [[ ! -f "$WORKSPACE/install/setup.bash" ]]; then
  echo "[ERROR] $WORKSPACE/install/setup.bash not found"
  echo "먼저 빌드하세요:"
  echo "  cd $WORKSPACE && colcon build"
  exec bash
fi

safe_source "$WORKSPACE/install/setup.bash"

if [[ -f "$WORKSPACE/src/vision/models/best_final.engine" ]]; then
  export TANK_YOLO_MODEL_PATH="$WORKSPACE/src/vision/models/best_final.engine"
fi
EOF
chmod +x "$COMMON_ENV"

BRIDGE_SCRIPT="$RUNTIME_DIR/s2_bridge_pane.sh"
cat > "$BRIDGE_SCRIPT" <<EOF
#!/usr/bin/env bash
set -Eeuo pipefail
printf '\\033]0;S2-T1 ros_bridge\\007'
source "$COMMON_ENV"

echo "============================================================"
echo "[T1] ros_bridge"
echo "============================================================"
echo "Command:"
echo "  TANK_MODE=auto TANK_EPISODE_CONTROL=true ros2 run ros_bridge ros_bridge"
echo
echo "[YOLO MODEL]"
echo "  TANK_YOLO_MODEL_PATH=\${TANK_YOLO_MODEL_PATH:-<not set>}"
echo
echo "[LOG] $LOG_DIR/bridge.log"
echo

TANK_MODE=auto TANK_EPISODE_CONTROL=true ros2 run ros_bridge ros_bridge 2>&1 | tee "$LOG_DIR/bridge.log"

echo
echo "[EXIT] ros_bridge 종료됨. 창을 닫거나 Enter를 누르세요."
exec bash
EOF
chmod +x "$BRIDGE_SCRIPT"

RVIZ_SCRIPT="$RUNTIME_DIR/s2_rviz_pane.sh"
if [[ "$USE_RVIZ" == "true" ]]; then
  cat > "$RVIZ_SCRIPT" <<EOF
#!/usr/bin/env bash
set -Eeuo pipefail
printf '\\033]0;S2-T2 Scenario2 RViz\\007'
source "$COMMON_ENV"

MAP_FILE="$WORKSPACE/$SCENARIO2_MAP_REL"

echo "============================================================"
echo "[T2] Scenario2 RViz"
echo "============================================================"
echo "RViz는 scenario2_map.map이 생긴 뒤 실행합니다."
echo "대기 파일:"
echo "  \$MAP_FILE"
echo
echo "[LOG] $LOG_DIR/rviz_scenario2.log"
echo

for i in \$(seq 1 "$MAP_WAIT_SEC"); do
  if [[ -f "\$MAP_FILE" ]]; then
    echo "[OK] scenario2 map found:"
    ls -lh "\$MAP_FILE"
    break
  fi
  if [[ "\$i" -eq "$MAP_WAIT_SEC" ]]; then
    echo "[ERROR] scenario2 map wait timeout: \$MAP_FILE"
    echo "        manager pane에서 build_scenario2_map.py 로그를 확인하세요."
    exec bash
  fi
  echo "[WAIT] scenario2_map.map 대기 중... \$i/$MAP_WAIT_SEC"
  sleep 1
done

echo
echo "Command:"
echo "  ros2 launch rviz_visualization tank_scenario2_map_view.launch.py"
echo

ros2 launch rviz_visualization tank_scenario2_map_view.launch.py 2>&1 | tee "$LOG_DIR/rviz_scenario2.log" &
RVIZ_LAUNCH_PID=\$!

echo "[WAIT] RViz 창 감지 후 최대화 시도"
for _ in \$(seq 1 60); do
  if command -v wmctrl >/dev/null 2>&1; then
    WIN_ID="\$(wmctrl -lx 2>/dev/null | awk 'BEGIN{IGNORECASE=1} /rviz|rviz2/ {print \$1; exit}')"
    if [[ -n "\${WIN_ID:-}" ]]; then
      wmctrl -ir "\$WIN_ID" -b add,maximized_vert,maximized_horz >/dev/null 2>&1 || true
      wmctrl -ia "\$WIN_ID" >/dev/null 2>&1 || true
      echo "[OK] RViz window maximized"
      break
    fi
  elif command -v xdotool >/dev/null 2>&1; then
    WIN_ID="\$(xdotool search --name 'RViz' 2>/dev/null | head -n 1 || true)"
    if [[ -n "\${WIN_ID:-}" ]]; then
      xdotool windowactivate "\$WIN_ID" >/dev/null 2>&1 || true
      xdotool key F11 >/dev/null 2>&1 || true
      echo "[OK] RViz window activated/fullscreen attempt"
      break
    fi
  else
    echo "[WARN] wmctrl/xdotool 없음. RViz 자동 최대화 생략."
    echo "       설치 권장: sudo apt install -y wmctrl"
    break
  fi
  sleep 0.5
done

wait "\$RVIZ_LAUNCH_PID"

echo
echo "[EXIT] scenario2 RViz launch 종료됨. 창을 닫거나 Enter를 누르세요."
exec bash
EOF
else
  cat > "$RVIZ_SCRIPT" <<EOF
#!/usr/bin/env bash
set -Eeuo pipefail
printf '\\033]0;S2-T2 No RViz\\007'
source "$COMMON_ENV"
echo "============================================================"
echo "[T2] Scenario2 RViz disabled"
echo "============================================================"
echo "--no-rviz 옵션으로 RViz 실행을 생략했습니다."
echo
echo "수동 실행:"
echo "  ros2 launch rviz_visualization tank_scenario2_map_view.launch.py"
echo
exec bash
EOF
fi
chmod +x "$RVIZ_SCRIPT"

MANAGER_SCRIPT="$RUNTIME_DIR/s2_manager_pane.sh"
cat > "$MANAGER_SCRIPT" <<EOF
#!/usr/bin/env bash
set -Eeuo pipefail
printf '\\033]0;S2-T3 Auto Pipeline\\007'
source "$COMMON_ENV"

MAP_FILE="$WORKSPACE/$SCENARIO2_MAP_REL"
TERRAIN_FILE="$WORKSPACE/$SCENARIO2_TERRAIN_REL"

echo "============================================================"
echo "[T3] Scenario 2 Auto Pipeline"
echo "============================================================"
echo "순서:"
echo "  1) bridge health 확인"
echo "  2) scenario2_map.map 없으면 자동 생성"
echo "  3) 3초 대기"
echo "  4) simulator reset/restart 요청"
echo "  5) 3초 대기"
echo "  6) python3 scripts/run_scenario2_scenario.py"
echo
echo "[LOG DIR] $LOG_DIR"
echo "[BUILD_MODE] $BUILD_MODE"
echo

run_step() {
  local name="\$1"
  shift

  echo
  echo "============================================================"
  echo "[RUN] \$name"
  echo "============================================================"
  echo "Command: \$*"
  echo

  "\$@" 2>&1 | tee "$LOG_DIR/\$name.log"
  local code="\${PIPESTATUS[0]}"

  if [[ "\$code" -ne 0 ]]; then
    echo
    echo "[ERROR] \$name failed with exit code \$code"
    echo "로그: $LOG_DIR/\$name.log"
    exec bash
  fi

  echo
  echo "[OK] \$name completed"
}

wait_for_bridge() {
  echo "[WAIT] ros_bridge health: $BRIDGE_HEALTH_URL"

  for _ in \$(seq 1 60); do
    if command -v curl >/dev/null 2>&1; then
      if curl -fsS "$BRIDGE_HEALTH_URL" >"$LOG_DIR/bridge_health.json" 2>/dev/null; then
        echo "[OK] bridge health: \$(cat "$LOG_DIR/bridge_health.json")"
        return 0
      fi
    else
      if ros2 topic list 2>/dev/null | grep -q "/tank"; then
        echo "[OK] ROS /tank topic detected"
        return 0
      fi
    fi
    sleep 0.5
  done

  echo "[WARN] bridge health 확인 실패. 그래도 계속 진행합니다."
  echo "       bridge pane 로그를 확인하세요."
  return 0
}

request_simulator_reset() {
  if [[ "$SKIP_RESET" == "true" ]]; then
    echo "[SKIP] simulator reset skipped"
    return 0
  fi

  echo "[RESET] simulator restart/reset request"
  echo "        topic: /tank/episode/control"
  echo "        data : reset"

  for i in 1 2 3; do
    echo "        publish reset attempt \$i/3"
    timeout 6s ros2 topic pub --once /tank/episode/control std_msgs/msg/String "{data: 'reset'}" \\
      2>&1 | tee "$LOG_DIR/reset_attempt_\${i}.log" || true
    sleep 0.5
  done
}

ensure_scenario2_map() {
  echo "[CHECK] scenario2 map:"
  echo "        \$MAP_FILE"

  if [[ "$BUILD_MODE" == "rebuild" ]]; then
    echo "[BUILD] --rebuild-map 지정됨. 기존 map 여부와 관계없이 다시 생성합니다."
    run_step "build_scenario2_map" python3 scripts/build_scenario2_map.py
  elif [[ -f "\$MAP_FILE" ]]; then
    echo "[OK] existing scenario2 map found:"
    ls -lh "\$MAP_FILE"
  elif [[ "$BUILD_MODE" == "auto" ]]; then
    echo "[BUILD] scenario2_map.map 없음 → 자동 생성합니다."
    run_step "build_scenario2_map" python3 scripts/build_scenario2_map.py
  else
    echo "[ERROR] scenario2_map.map 없음, 그리고 --no-build-map 지정됨."
    echo "        먼저 시나리오1 또는 build_scenario2_map.py를 실행하세요."
    exec bash
  fi

  if [[ ! -f "\$MAP_FILE" ]]; then
    echo
    echo "[ERROR] build 후에도 scenario2_map.map 없음"
    echo "        build 로그: $LOG_DIR/build_scenario2_map.log"
    echo "        recon_reports/recon_map 입력 파일들을 확인하세요:"
    find recon_reports -maxdepth 3 -type f 2>/dev/null | sort || true
    exec bash
  fi

  echo
  echo "[OK] scenario2 map ready:"
  ls -lh "\$MAP_FILE"
  if [[ -f "\$TERRAIN_FILE" ]]; then
    echo "[OK] scenario2 terrain ready:"
    ls -lh "\$TERRAIN_FILE"
  else
    echo "[WARN] scenario2 terrain npz 없음:"
    echo "       \$TERRAIN_FILE"
    echo "       지형 없이 map만으로 진행할 수 있는 구조면 계속 진행합니다."
  fi
}

wait_for_bridge
ensure_scenario2_map

echo
echo "[WAIT] bridge/RViz 실행 후 3초 대기"
sleep 3

request_simulator_reset

echo
echo "[WAIT] reset 후 3초 대기"
sleep 3

run_step "run_scenario2_scenario" python3 scripts/run_scenario2_scenario.py

echo
echo "============================================================"
echo "[DONE] Scenario 2 completed"
echo "============================================================"
echo "결과 확인:"
echo "  recon_reports/scenario2/"
echo "  recon_reports/scenario2/scenario2_result.json"
echo
echo "로그:"
echo "  $LOG_DIR"
echo
echo "이 창은 확인용으로 유지됩니다. 닫아도 됩니다."
exec bash
EOF
chmod +x "$MANAGER_SCRIPT"

DEBUG_SCRIPT="$RUNTIME_DIR/s2_debug_pane.sh"
cat > "$DEBUG_SCRIPT" <<EOF
#!/usr/bin/env bash
set -Eeuo pipefail
printf '\\033]0;S2-T4 Debug\\007'
source "$COMMON_ENV"

echo "============================================================"
echo "[T4] Scenario2 Debug Monitor"
echo "============================================================"
echo "2초마다 핵심 node/topic/file을 표시합니다."
echo "종료: Ctrl+C"
echo
echo "[LOG DIR] $LOG_DIR"
echo

sleep 4

watch -n 2 "
echo '[nodes]';
ros2 node list 2>/dev/null | grep -E 'ros_bridge|static_map|rviz_visualizer|terrain|lidar|overlay|dbscan|astar|local_path|controller|potential|turret|decision|scenario' || true;
echo;
echo '[key topics]';
ros2 topic list 2>/dev/null | grep -E '/tank/(api/info/raw|player/pose|enemy/pose|global_path|path/lookahead_pose|control/command|control/status|mission/goal_pose|turret/status|turret/override|engage/result|decision/status|map/discovered/objects|episode/control)' || true;
echo;
echo '[scenario2 files]';
ls -lh $SCENARIO2_MAP_REL $SCENARIO2_TERRAIN_REL recon_reports/scenario2/scenario2_result.json 2>/dev/null || true;
echo;
echo '[recon map inputs]';
find recon_reports/recon_map recon_reports/terrain_maps -maxdepth 1 -type f 2>/dev/null | sort | sed 's#^#  #';
"

exec bash
EOF
chmod +x "$DEBUG_SCRIPT"

CONFIG_FILE="$RUNTIME_DIR/scenario2_auto_terminator_config"
cat > "$CONFIG_FILE" <<EOF
[global_config]
[keybindings]
[profiles]
  [[default]]
    scrollback_infinite = True
[layouts]
  [[scenario2_auto]]
    [[[window0]]]
      type = Window
      parent = ""
      order = 0
      maximised = True
      title = Tank Scenario 2 Auto
    [[[hpaned0]]]
      type = HPaned
      parent = window0
      order = 0
      position = 960
    [[[vpaned_left]]]
      type = VPaned
      parent = hpaned0
      order = 0
      position = 520
    [[[terminal_bridge]]]
      type = Terminal
      parent = vpaned_left
      order = 0
      command = bash -lc '$BRIDGE_SCRIPT'
    [[[terminal_manager]]]
      type = Terminal
      parent = vpaned_left
      order = 1
      command = bash -lc '$MANAGER_SCRIPT'
    [[[vpaned_right]]]
      type = VPaned
      parent = hpaned0
      order = 1
      position = 520
    [[[terminal_rviz]]]
      type = Terminal
      parent = vpaned_right
      order = 0
      command = bash -lc '$RVIZ_SCRIPT'
    [[[terminal_debug]]]
      type = Terminal
      parent = vpaned_right
      order = 1
      command = bash -lc '$DEBUG_SCRIPT'
[plugins]
EOF

echo "[RUN] Terminator Scenario 2 Auto v2"
echo "      workspace : $WORKSPACE"
echo "      log dir   : $LOG_DIR"
echo "      rviz      : $USE_RVIZ"
echo "      skip_reset: $SKIP_RESET"
echo "      build_mode: $BUILD_MODE"
echo "      config    : $CONFIG_FILE"
echo
echo "Terminator 4분할 창이 열립니다:"
echo "  좌상: ros_bridge"
echo "  우상: scenario2 RViz, map 파일 대기 후 launch"
echo "  좌하: 자동 순차 실행 manager"
echo "  우하: debug monitor"
echo

terminator -g "$CONFIG_FILE" -l scenario2_auto
