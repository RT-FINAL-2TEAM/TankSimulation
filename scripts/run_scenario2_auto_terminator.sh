#!/usr/bin/env bash
set -Eeuo pipefail

# Scenario 2 auto runner.
# - Default: no local /view web, no local desktop RViz2 GUI.
# - --web : enable ros_bridge /view on this PC.
# - --rviz: open desktop RViz2 map view on this PC.
# rviz_web / rosbridge_websocket is intentionally not launched.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="${TANK_WS:-$(cd "$SCRIPT_DIR/.." && pwd)}"
LOG_ROOT="${LOG_ROOT:-$WORKSPACE/logs}"
RUN_ID="scenario2_terminator_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$LOG_ROOT/$RUN_ID"
RUNTIME_DIR="$WORKSPACE/scripts/.terminator_runtime"

WEB_ENABLED="false"
RVIZ_ENABLED="false"
SKIP_RESET="false"
BUILD_MODE="auto"  # auto | rebuild | never
BRIDGE_HEALTH_URL="${BRIDGE_HEALTH_URL:-http://127.0.0.1:5000/health}"
SCENARIO2_MAP_REL="recon_reports/recon_map/scenario2_map.map"
SCENARIO2_TERRAIN_REL="recon_reports/recon_map/scenario2_terrain.npz"
SCENARIO2_TERRAIN_JSON_REL="recon_reports/recon_map/scenario2_terrain.json"
MISSION_PLAN_REL="recon_reports/mission_plan.json"
MAP_WAIT_SEC="${MAP_WAIT_SEC:-180}"
PHONE_PORT="${PHONE_PORT:-5002}"

# Scenario1 정찰 결과로 생성된 mission_plan.json을 Scenario2 교전 시퀀스에 사용한다.
export TANK_USE_MISSION_PLAN=true
export TANK_MISSION_PLAN_FILE="$WORKSPACE/$MISSION_PLAN_REL"

usage() {
  cat <<USAGE
Usage: $0 [--web] [--rviz] [--skip-reset] [--rebuild-map] [--no-build-map]

Options:
  --web          이 PC의 ros_bridge에서 /view live web을 켬. 기본값은 꺼짐.
  --rviz         이 PC에서 데스크톱 RViz2 scenario2 map view를 실행. 기본값은 RViz2 GUI 꺼짐.
  --skip-reset   run_scenario2_scenario.py 실행 전 simulator reset 요청 생략
  --rebuild-map  기존 scenario2_map.map이 있어도 build_scenario2_map.py 재실행
  --no-build-map map 자동 생성을 하지 않음. map이 없거나 stale이어도 에러 처리
  --no-rviz      호환용 no-op. 기본값이 이미 RViz2 GUI off.
  -h, --help     도움말 출력

Panes:
  T1 ros_bridge
  T2 scenario2 visualization backend, plus local RViz2 GUI only when --rviz is set
  T3 scenario2 manager
  T4 phone_sim2real, phone_port:=5002
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --web)
      WEB_ENABLED="true"; shift ;;
    --rviz)
      RVIZ_ENABLED="true"; shift ;;
    --no-rviz)
      RVIZ_ENABLED="false"; shift ;;
    --skip-reset)
      SKIP_RESET="true"; shift ;;
    --rebuild-map)
      BUILD_MODE="rebuild"; shift ;;
    --no-build-map)
      BUILD_MODE="never"; shift ;;
    --web-rviz)
      echo "[ERROR] rviz_web/Web RViz는 더 이상 이 스크립트에서 실행하지 않습니다."
      echo "        live web은 --web, 데스크톱 RViz2는 --rviz를 사용하세요."
      exit 2 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "[ERROR] Unknown option: $1"; usage; exit 2 ;;
  esac
done

if ! command -v terminator >/dev/null 2>&1; then
  echo "[ERROR] terminator가 설치되어 있지 않습니다."
  echo "  sudo apt update && sudo apt install -y terminator"
  exit 1
fi
if [[ ! -d "$WORKSPACE" ]]; then
  echo "[ERROR] workspace not found: $WORKSPACE"
  exit 1
fi


preflight_cleanup_local_processes() {
  echo "[PRE-FLIGHT] 이전 실행에서 남은 로컬 프로세스를 정리합니다."
  echo "             stale bridge/RViz/terrain/autonomy 노드가 남으면 reset, marker, map 상태가 섞입니다."

  # Old bridge on :5000 is especially dangerous: the new bridge may fail to bind,
  # while manager health checks the old bridge and episode reset is ignored.
  pkill -9 -f 'ros2 run ros_bridge ros_bridge' 2>/dev/null || true
  pkill -9 -f 'install/ros_bridge/lib/ros_bridge/ros_bridge' 2>/dev/null || true
  pkill -9 -f 'ros_bridge.main' 2>/dev/null || true
  pkill -9 -f '/ros_bridge/ros_bridge' 2>/dev/null || true

  # Local visualization/terrain nodes. Remote RViz on another PC is not affected.
  pkill -9 -f 'rviz2' 2>/dev/null || true
  pkill -9 -f 'ros2 launch rviz_visualization' 2>/dev/null || true
  pkill -9 -f 'ros2 launch ground_division' 2>/dev/null || true
  pkill -9 -f 'clear_rviz_runtime_state.py' 2>/dev/null || true
  pkill -9 -f 'static_map_loader_node' 2>/dev/null || true
  pkill -9 -f 'rviz_visualizer_node' 2>/dev/null || true
  pkill -9 -f 'terrain_record_finalize_node' 2>/dev/null || true
  pkill -9 -f 'terrain_saved_map_visualizer_node' 2>/dev/null || true

  # Old autonomy stack and phone bridge can keep publishing stale path/markers or occupy phone_port.
  for key in \
    'tank_autonomous_control.launch.py' \
    'lidar_processor_node' \
    'lidar_camera_overlay' \
    'lidar_dbscan_cluster_node' \
    'map_astar_planner_node' \
    'local_path_node' \
    'potential_field_node' \
    'tank_controller_node' \
    'phone_sim2real.launch.py' \
    'phone_sim2real'; do
    pkill -9 -f "$key" 2>/dev/null || true
  done

  sleep 1.0
  local leftovers
  leftovers="$(pgrep -af 'ros_bridge|rviz2|static_map_loader_node|rviz_visualizer_node|terrain_record_finalize_node|terrain_saved_map_visualizer_node|phone_sim2real|tank_autonomous_control.launch.py|lidar_processor_node|lidar_dbscan_cluster_node|map_astar_planner_node|local_path_node|tank_controller_node' 2>/dev/null || true)"
  if [[ -n "$leftovers" ]]; then
    echo "[PRE-FLIGHT][WARN] 아직 남은 관련 프로세스가 있습니다. 한 번 더 정리합니다:"
    echo "$leftovers"
    pkill -9 -f 'ros_bridge|rviz2|static_map_loader_node|rviz_visualizer_node|terrain_record_finalize_node|terrain_saved_map_visualizer_node|phone_sim2real|tank_autonomous_control.launch.py|lidar_processor_node|lidar_dbscan_cluster_node|map_astar_planner_node|local_path_node|tank_controller_node' 2>/dev/null || true
    sleep 1.0
  fi
}
preflight_cleanup_runtime_outputs() {
  echo "[PRE-FLIGHT] 시나리오2 runtime 결과만 Terminator 실행 전에 먼저 정리합니다."
  mkdir -p "$WORKSPACE/recon_reports/scenario2"            "$WORKSPACE/tank_discovered_maps"            "$WORKSPACE/tank_terrain_maps"

  # Scenario2 입력인 scenario2_map.map, scenario2_terrain.*, mission_plan.json,
  # route_A/B 정찰 산출물은 절대 삭제하지 않는다.
  rm -f     "$WORKSPACE/recon_reports/scenario2/route_A.json"     "$WORKSPACE/recon_reports/scenario2/scenario2_result.json"     "$WORKSPACE/tank_discovered_maps/discovered_objects_latest.map"     "$WORKSPACE/tank_terrain_maps/terrain_map_latest.npz"
}
preflight_cleanup_local_processes
preflight_cleanup_runtime_outputs

# 이전 실행에서 생성된 pane script/config를 지워 stale command가 재사용되지 않게 한다.
rm -rf "$RUNTIME_DIR"
mkdir -p "$LOG_DIR" "$RUNTIME_DIR"

COMMON_ENV="$RUNTIME_DIR/common_env.sh"
cat > "$COMMON_ENV" <<EOF_ENV
#!/usr/bin/env bash
set -Eeuo pipefail

safe_source() {
  local setup_file="\$1"
  set +u
  source "\$setup_file"
  set -u
}

cd "$WORKSPACE"
export TANK_PROJECT_ROOT="$WORKSPACE"
# 기본값: 수동 실행과 동일하게 planner cluster topic env override 없음.
export TANK_USE_MISSION_PLAN="true"
export TANK_MISSION_PLAN_FILE="$WORKSPACE/$MISSION_PLAN_REL"

if [[ ! -f /opt/ros/humble/setup.bash ]]; then
  echo "[ERROR] /opt/ros/humble/setup.bash not found"
  exec bash
fi
safe_source /opt/ros/humble/setup.bash

if [[ ! -f "$WORKSPACE/install/setup.bash" ]]; then
  echo "[ERROR] $WORKSPACE/install/setup.bash not found"
  echo "먼저 빌드하세요: cd $WORKSPACE && colcon build"
  exec bash
fi
safe_source "$WORKSPACE/install/setup.bash"

if [[ -f "$WORKSPACE/src/vision/models/best_final.engine" ]]; then
  export TANK_YOLO_MODEL_PATH="$WORKSPACE/src/vision/models/best_final.engine"
fi
EOF_ENV
chmod +x "$COMMON_ENV"

BRIDGE_SCRIPT="$RUNTIME_DIR/s2_bridge_pane.sh"
cat > "$BRIDGE_SCRIPT" <<EOF_BRIDGE
#!/usr/bin/env bash
set -Eeuo pipefail
printf '\\033]0;S2-T1 ros_bridge\\007'
source "$COMMON_ENV"

echo "============================================================"
echo "[T1] ros_bridge"
echo "============================================================"
echo "live web       : $WEB_ENABLED"
echo "command        : TANK_MODE=auto TANK_EPISODE_CONTROL=true TANK_LIVE_VIEW=$WEB_ENABLED ros2 run ros_bridge ros_bridge"
echo "log            : $LOG_DIR/bridge.log"
echo

set +e
TANK_MODE=auto \
TANK_EPISODE_CONTROL=true \
TANK_LIVE_VIEW=$WEB_ENABLED \
ros2 run ros_bridge ros_bridge 2>&1 | tee "$LOG_DIR/bridge.log"
code=\${PIPESTATUS[0]}
set -e

echo
echo "[EXIT] ros_bridge 종료됨 (exit=\$code)."
echo "[LOG] $LOG_DIR/bridge.log"
exec bash
EOF_BRIDGE
chmod +x "$BRIDGE_SCRIPT"

RVIZ_SCRIPT="$RUNTIME_DIR/s2_rviz_pane.sh"
if [[ "$RVIZ_ENABLED" == "true" ]]; then
  cat > "$RVIZ_SCRIPT" <<EOF_RVIZ
#!/usr/bin/env bash
set -Eeuo pipefail
printf '\\033]0;S2-T2 RViz2 Map View\\007'
source "$COMMON_ENV"

MAP_FILE="$WORKSPACE/$SCENARIO2_MAP_REL"

echo "============================================================"
echo "[T2] Scenario2 Desktop RViz2 + Visualization Backend (--rviz)"
echo "============================================================"
echo "RViz2는 scenario2_map.map이 준비된 뒤 실행합니다."
echo "map : \$MAP_FILE"
echo "log : $LOG_DIR/rviz_scenario2.log"
echo
for i in \$(seq 1 "$MAP_WAIT_SEC"); do
  if [[ -f "\$MAP_FILE" ]]; then
    echo "[OK] scenario2 map found:"; ls -lh "\$MAP_FILE"; break
  fi
  if [[ "\$i" -eq "$MAP_WAIT_SEC" ]]; then
    echo "[ERROR] scenario2 map wait timeout: \$MAP_FILE"
    echo "        manager pane의 build_scenario2_map.py 로그를 확인하세요."
    exec bash
  fi
  echo "[WAIT] scenario2_map.map 대기 중... \$i/$MAP_WAIT_SEC"
  sleep 1
done

sleep 0.7
ros2 launch rviz_visualization tank_scenario2_map_view.launch.py 2>&1 | tee "$LOG_DIR/rviz_scenario2.log"
echo
echo "[EXIT] scenario2 RViz2 종료."
exec bash
EOF_RVIZ
else
  cat > "$RVIZ_SCRIPT" <<EOF_RVIZ
#!/usr/bin/env bash
set -Eeuo pipefail
printf '\\033]0;S2-T2 RViz Disabled\\007'
source "$COMMON_ENV"

echo "============================================================"
echo "[T2] Scenario2 Visualization Backend Only"
echo "============================================================"
echo "이 PC에서는 RViz2 GUI를 실행하지 않습니다."
echo "이 PC에서 보려면 다음 실행 때 --rviz를 붙이세요."
echo "마커 발행 노드는 이 pane에서 단일 출처로 실행합니다."
echo
echo "다른 PC/터미널에서는 중복 publisher가 생기지 않도록 viewer-only launch만 실행하세요:"
echo "  ros2 launch rviz_visualization tank_scenario2_rviz_viewer_only.launch.py"
echo
MAP_FILE="$WORKSPACE/$SCENARIO2_MAP_REL"
echo "대기 파일: \$MAP_FILE"
for i in \$(seq 1 "$MAP_WAIT_SEC"); do
  if [[ -f "\$MAP_FILE" ]]; then
    echo "[OK] scenario2 map found:"; ls -lh "\$MAP_FILE"; break
  fi
  if [[ "\$i" -eq "$MAP_WAIT_SEC" ]]; then
    echo "[ERROR] scenario2 map wait timeout: \$MAP_FILE"
    exec bash
  fi
  echo "[WAIT] scenario2_map.map 대기 중... \$i/$MAP_WAIT_SEC"
  sleep 1
done

sleep 0.7
ros2 launch rviz_visualization tank_scenario2_map_view.launch.py use_rviz:=false 2>&1 | tee "$LOG_DIR/rviz_scenario2_backend.log"
echo
echo "[EXIT] scenario2 visualization backend 종료."
exec bash
EOF_RVIZ
fi
chmod +x "$RVIZ_SCRIPT"

MANAGER_SCRIPT="$RUNTIME_DIR/s2_manager_pane.sh"
cat > "$MANAGER_SCRIPT" <<EOF_MANAGER
#!/usr/bin/env bash
set -Eeuo pipefail
printf '\\033]0;S2-T3 Auto Pipeline\\007'
source "$COMMON_ENV"

MAP_FILE="$WORKSPACE/$SCENARIO2_MAP_REL"
TERRAIN_FILE="$WORKSPACE/$SCENARIO2_TERRAIN_REL"
TERRAIN_JSON_FILE="$WORKSPACE/$SCENARIO2_TERRAIN_JSON_REL"
MISSION_PLAN_FILE="$WORKSPACE/$MISSION_PLAN_REL"

echo "============================================================"
echo "[T3] Scenario 2 Auto Pipeline"
echo "============================================================"
echo "log dir    : $LOG_DIR"
echo "build mode : $BUILD_MODE"
echo "map        : \$MAP_FILE"
echo "mission    : \$MISSION_PLAN_FILE"
echo

run_step() {
  local name="\$1"; shift
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
  echo "[OK] \$name completed"
}

run_optional_step() {
  local name="\$1"; shift
  echo
  echo "============================================================"
  echo "[RUN/OPTIONAL] \$name"
  echo "============================================================"
  echo "Command: \$*"
  echo
  set +e
  "\$@" 2>&1 | tee "$LOG_DIR/\$name.log"
  local code="\${PIPESTATUS[0]}"
  set -e
  if [[ "\$code" -ne 0 ]]; then
    echo "[WARN] optional step failed: \$name (exit=\$code). 계속 진행합니다."
    echo "       로그: $LOG_DIR/\$name.log"
  else
    echo "[OK] \$name completed"
  fi
}

wait_for_bridge() {
  echo "[WAIT] ros_bridge health: $BRIDGE_HEALTH_URL"
  for _ in \$(seq 1 60); do
    if command -v curl >/dev/null 2>&1 && curl -fsS "$BRIDGE_HEALTH_URL" >"$LOG_DIR/bridge_health.json" 2>/dev/null; then
      echo "[OK] bridge health: \$(cat "$LOG_DIR/bridge_health.json")"
      return 0
    fi
    sleep 0.5
  done
  echo "[WARN] bridge health 확인 실패. 그래도 계속 진행합니다."
}

request_simulator_reset() {
  if [[ "$SKIP_RESET" == "true" ]]; then
    echo "[SKIP] simulator reset skipped"
    return 0
  fi
  echo "[RESET] simulator restart/reset request -> /tank/episode/control reset"
  for i in 1 2 3; do
    timeout 6s ros2 topic pub --once /tank/episode/control std_msgs/msg/String "{data: 'reset'}" \
      2>&1 | tee "$LOG_DIR/reset_attempt_\${i}.log" || true
    sleep 0.5
  done
}

cleanup_scenario2_runtime_outputs() {
  echo "[CLEAN] 시나리오2 실행 결과만 정리. 정찰 입력 map/terrain/mission_plan은 보존합니다."
  mkdir -p recon_reports/scenario2
  rm -f recon_reports/scenario2/route_A.json recon_reports/scenario2/scenario2_result.json
}

map_inputs_newer_than_output() {
  python3 - <<'PY'
from pathlib import Path
import sys
out = Path("recon_reports/recon_map/scenario2_map.map")
inputs = [
    Path("recon_reports/recon_map/discovered_objects_route_A.map"),
    Path("recon_reports/recon_map/discovered_objects_route_B.map"),
    Path("recon_reports/terrain_maps/terrain_map_route_A.npz"),
    Path("recon_reports/terrain_maps/terrain_map_route_B.npz"),
]
if not out.exists():
    sys.exit(0)
out_m = out.stat().st_mtime
newer = [p for p in inputs if p.exists() and p.stat().st_mtime > out_m + 0.1]
if newer:
    print("[STALE] scenario2_map.map보다 최신 입력:", ", ".join(str(p) for p in newer))
    sys.exit(0)
sys.exit(1)
PY
}

mission_plan_stale() {
  python3 - <<'PY'
from pathlib import Path
import sys
map_file = Path("recon_reports/recon_map/scenario2_map.map")
terrain_file = Path("recon_reports/recon_map/scenario2_terrain.json")
plan = Path("recon_reports/mission_plan.json")
if not plan.exists():
    sys.exit(0)
base = plan.stat().st_mtime
for p in (map_file, terrain_file):
    if p.exists() and p.stat().st_mtime > base + 0.1:
        print(f"[STALE] mission_plan.json보다 최신 입력: {p}")
        sys.exit(0)
sys.exit(1)
PY
}

ensure_scenario2_map() {
  echo "[CHECK] scenario2 map: \$MAP_FILE"
  local need_build="false"
  if [[ "$BUILD_MODE" == "rebuild" ]]; then
    echo "[BUILD] --rebuild-map 지정됨."
    need_build="true"
  elif [[ ! -f "\$MAP_FILE" ]]; then
    if [[ "$BUILD_MODE" == "never" ]]; then
      echo "[ERROR] scenario2_map.map 없음, 그리고 --no-build-map 지정됨."
      exec bash
    fi
    echo "[BUILD] scenario2_map.map 없음."
    need_build="true"
  elif map_inputs_newer_than_output; then
    if [[ "$BUILD_MODE" == "never" ]]; then
      echo "[ERROR] scenario2_map.map이 입력보다 오래됐지만 --no-build-map 지정됨."
      exec bash
    fi
    echo "[BUILD] route A/B 정찰 입력이 scenario2_map.map보다 최신입니다. 자동 재생성합니다."
    need_build="true"
  else
    echo "[OK] existing scenario2 map is current enough."
  fi

  if [[ "\$need_build" == "true" ]]; then
    run_step "build_scenario2_map" python3 scripts/build_scenario2_map.py
  fi

  if [[ ! -f "\$MAP_FILE" ]]; then
    echo "[ERROR] build 후에도 scenario2_map.map 없음: \$MAP_FILE"
    find recon_reports -maxdepth 3 -type f 2>/dev/null | sort || true
    exec bash
  fi
  echo "[OK] scenario2 map ready:"; ls -lh "\$MAP_FILE"
  [[ -f "\$TERRAIN_FILE" ]] && { echo "[OK] terrain npz:"; ls -lh "\$TERRAIN_FILE"; } || true
  [[ -f "\$TERRAIN_JSON_FILE" ]] && { echo "[OK] terrain json:"; ls -lh "\$TERRAIN_JSON_FILE"; } || true
}

ensure_mission_plan() {
  if mission_plan_stale; then
    echo "[BUILD] mission_plan.json 없음 또는 scenario2_map/terrain보다 오래됨. 재생성합니다."
    run_optional_step "build_mission_plan" python3 scripts/build_mission_plan.py
  else
    echo "[OK] mission_plan.json is current enough."
  fi
  if [[ -f "\$MISSION_PLAN_FILE" ]]; then
    echo "[OK] mission plan:"; ls -lh "\$MISSION_PLAN_FILE"
  else
    echo "[WARN] mission_plan.json 없음. tank_scenario2.launch.py의 fallback engagement로 진행할 수 있습니다."
  fi
}

wait_for_bridge
cleanup_scenario2_runtime_outputs
ensure_scenario2_map
ensure_mission_plan
echo "[WAIT] bridge 준비 대기 3초"
sleep 3
request_simulator_reset
echo "[WAIT] reset 후 3초 대기"
sleep 3
run_step "run_scenario2_scenario" python3 scripts/run_scenario2_scenario.py

echo
echo "============================================================"
echo "[DONE] Scenario 2 completed"
echo "============================================================"
echo "결과: recon_reports/scenario2/scenario2_result.json"
echo "로그: $LOG_DIR"
exec bash
EOF_MANAGER
chmod +x "$MANAGER_SCRIPT"

PHONE_SCRIPT="$RUNTIME_DIR/s2_phone_sim2real_pane.sh"
cat > "$PHONE_SCRIPT" <<EOF_PHONE
#!/usr/bin/env bash
set -Eeuo pipefail
printf '\\033]0;S2-T4 phone_sim2real\\007'
source "$COMMON_ENV"

echo "============================================================"
echo "[T4] phone_sim2real"
echo "============================================================"
echo "Command: ros2 launch phone_sim2real phone_sim2real.launch.py phone_port:=$PHONE_PORT"
echo "Log    : $LOG_DIR/phone_sim2real.log"
echo
ros2 launch phone_sim2real phone_sim2real.launch.py phone_port:=$PHONE_PORT 2>&1 | tee "$LOG_DIR/phone_sim2real.log"
echo
echo "[EXIT] phone_sim2real 종료."
exec bash
EOF_PHONE
chmod +x "$PHONE_SCRIPT"

CONFIG_FILE="$RUNTIME_DIR/scenario2_auto_terminator_config"
cat > "$CONFIG_FILE" <<EOF_CFG
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
    [[[terminal_phone]]]
      type = Terminal
      parent = vpaned_right
      order = 1
      command = bash -lc '$PHONE_SCRIPT'
[plugins]
EOF_CFG

echo "[RUN] Terminator Scenario 2 Auto"
echo "      workspace       : $WORKSPACE"
echo "      log dir         : $LOG_DIR"
echo "      local web /view : $WEB_ENABLED"
echo "      desktop RViz2   : $RVIZ_ENABLED"
echo "      skip_reset      : $SKIP_RESET"
echo "      build_mode      : $BUILD_MODE"
echo "      phone_port      : $PHONE_PORT"
echo
echo "Terminator 4분할 창:"
echo "  좌상: ros_bridge"
echo "  우상: RViz2 map view(--rviz) 또는 안내 pane"
echo "  좌하: scenario2 manager"
echo "  우하: phone_sim2real"
echo
if [[ "$WEB_ENABLED" != "true" ]]; then
  echo "[WEB] 이 ros_bridge의 /view는 꺼져 있습니다. 이 PC에서 보려면 --web을 붙이세요."
fi
if [[ "$RVIZ_ENABLED" != "true" ]]; then
  echo "[RVIZ] 이 PC에서 RViz2 GUI는 열지 않습니다. 다른 PC/터미널은 tank_scenario2_rviz_viewer_only.launch.py를 사용하세요."
fi

terminator -u -g "$CONFIG_FILE" -l scenario2_auto
