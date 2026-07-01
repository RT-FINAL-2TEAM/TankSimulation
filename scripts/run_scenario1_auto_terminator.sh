#!/usr/bin/env bash
set -Eeuo pipefail

# run_scenario1_auto_terminator_v2.sh
#
# 목적:
#   Terminator 4분할 창을 열고 시나리오 1 전체 절차를 자동 실행한다.
#   각 터미널 pane에서 bridge/RViz/manager/debug 출력이 실시간으로 보인다.
#
# v2 핵심:
#   - run_recon_scenario.py 종료 후 반드시 postprocess 실행
#       1) python3 scripts/analyze_run.py
#       2) python3 scripts/build_scenario2_map.py
#   - terrain_map_route_A/B.npz가 0 byte 또는 깨진 파일이면 즉시 중단
#   - scenario2_map.map / scenario2_terrain.npz 생성 여부 검증
#   - scenario2_terrain.npz metadata에 route_A, route_B 둘 다 들어있는지 검증
#
# Pane 구성:
#   좌상: T1 ros_bridge
#   우상: T2 RViz launch + RViz 창 최대화
#   좌하: T3 자동 순차 실행 manager
#   우하: T4 debug monitor
#
# 사용:
#   cd ~/tankcc
#   ./scripts/run_scenario1_auto_terminator.sh
#
# 옵션:
#   --no-rviz          RViz 실행 안 함
#   --skip-reset       run_recon_scenario.py 전 simulator reset 생략
#   --postprocess-only 정찰은 생략하고 analyze_run.py + build_scenario2_map.py만 실행
#   --keep-old-output  기존 recon_reports 산출물 삭제하지 않음

WORKSPACE="${TANK_WS:-$HOME/tankcc}"
LOG_ROOT="${LOG_ROOT:-$WORKSPACE/logs}"
RUN_ID="scenario1_terminator_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$LOG_ROOT/$RUN_ID"

USE_RVIZ="true"
SKIP_RESET="false"
POSTPROCESS_ONLY="false"
KEEP_OLD_OUTPUT="false"
WEB_DEBUG_URL="false"
BRIDGE_HEALTH_URL="${BRIDGE_HEALTH_URL:-http://127.0.0.1:5000/health}"

usage() {
  cat <<USAGE
Usage: $0 [--no-rviz] [--skip-reset] [--postprocess-only] [--keep-old-output]

Options:
  --no-rviz          RViz pane에서 RViz를 실행하지 않음
  --skip-reset       run_recon_scenario.py 실행 전 simulator reset 요청 생략
  --postprocess-only run_recon_scenario.py는 생략하고 analyze/build만 실행
  --keep-old-output  시작 전에 기존 route/scenario2 산출물을 삭제하지 않음
  --web-debug-url    debug pane에 rviz_web URL 안내도 표시
  -h, --help         도움말 출력

Environment:
  TANK_WS             ROS2 workspace 경로(default: ~/tankcc)
  LOG_ROOT            로그 저장 상위 경로(default: ~/tankcc/logs)
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
    --postprocess-only)
      POSTPROCESS_ONLY="true"
      shift
      ;;
    --keep-old-output)
      KEEP_OLD_OUTPUT="true"
      shift
      ;;
    --web-debug-url)
      WEB_DEBUG_URL="true"
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

# YOLO engine 경로가 예전 workspace로 남는 문제 방지.
if [[ -f "$WORKSPACE/src/vision/models/best_final.engine" ]]; then
  export TANK_YOLO_MODEL_PATH="$WORKSPACE/src/vision/models/best_final.engine"
fi
EOF
chmod +x "$COMMON_ENV"

BRIDGE_SCRIPT="$RUNTIME_DIR/s1_bridge_pane.sh"
cat > "$BRIDGE_SCRIPT" <<EOF
#!/usr/bin/env bash
set -Eeuo pipefail
printf '\\033]0;S1-T1 ros_bridge\\007'
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

RVIZ_SCRIPT="$RUNTIME_DIR/s1_rviz_pane.sh"
if [[ "$USE_RVIZ" == "true" ]]; then
  cat > "$RVIZ_SCRIPT" <<EOF
#!/usr/bin/env bash
set -Eeuo pipefail
printf '\\033]0;S1-T2 RViz\\007'
source "$COMMON_ENV"

echo "============================================================"
echo "[T2] RViz"
echo "============================================================"
echo "Command:"
echo "  ros2 launch rviz_visualization tank_rviz.launch.py"
echo
echo "[LOG] $LOG_DIR/rviz.log"
echo

ros2 launch rviz_visualization tank_rviz.launch.py 2>&1 | tee "$LOG_DIR/rviz.log" &
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
echo "[EXIT] RViz launch 종료됨. 창을 닫거나 Enter를 누르세요."
exec bash
EOF
else
  cat > "$RVIZ_SCRIPT" <<EOF
#!/usr/bin/env bash
set -Eeuo pipefail
printf '\\033]0;S1-T2 No RViz\\007'
source "$COMMON_ENV"
echo "============================================================"
echo "[T2] RViz disabled"
echo "============================================================"
echo "--no-rviz 옵션으로 RViz 실행을 생략했습니다."
echo
echo "수동 실행:"
echo "  ros2 launch rviz_visualization tank_rviz.launch.py"
echo
exec bash
EOF
fi
chmod +x "$RVIZ_SCRIPT"

MANAGER_SCRIPT="$RUNTIME_DIR/s1_manager_pane.sh"
cat > "$MANAGER_SCRIPT" <<EOF
#!/usr/bin/env bash
set -Eeuo pipefail
printf '\\033]0;S1-T3 Auto Pipeline\\007'
source "$COMMON_ENV"

echo "============================================================"
echo "[T3] Scenario 1 Auto Pipeline v2"
echo "============================================================"
echo "순서:"
echo "  1) bridge health 확인"
echo "  2) 기존 산출물 정리"
echo "  3) 3초 대기"
echo "  4) simulator reset/restart 요청"
echo "  5) 3초 대기"
echo "  6) python3 scripts/run_recon_scenario.py"
echo "  7) terrain_map_route_A/B.npz 유효성 검증"
echo "  8) python3 scripts/analyze_run.py"
echo "  9) python3 scripts/build_scenario2_map.py"
echo " 10) scenario2_map.map / scenario2_terrain.npz 검증"
echo
echo "[LOG DIR] $LOG_DIR"
echo "[POSTPROCESS_ONLY] $POSTPROCESS_ONLY"
echo "[KEEP_OLD_OUTPUT]  $KEEP_OLD_OUTPUT"
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

cleanup_old_outputs() {
  if [[ "$KEEP_OLD_OUTPUT" == "true" ]] || [[ "$POSTPROCESS_ONLY" == "true" ]]; then
    echo "[KEEP] 기존 산출물 유지"
    return 0
  fi

  echo "[CLEAN] 기존 route/scenario2 산출물 정리"
  rm -f recon_reports/terrain_maps/terrain_map_route_A.npz
  rm -f recon_reports/terrain_maps/terrain_map_route_B.npz
  rm -f recon_reports/recon_map/scenario2_terrain.npz
  rm -f recon_reports/recon_map/scenario2_terrain.json
  rm -f recon_reports/recon_map/scenario2_map.map
  rm -f tank_terrain_maps/terrain_map_latest.npz
  mkdir -p recon_reports/terrain_maps recon_reports/recon_map
}

verify_route_terrain() {
  echo
  echo "============================================================"
  echo "[VERIFY] route terrain npz"
  echo "============================================================"

  python3 - <<'PY'
import sys
from pathlib import Path
import numpy as np

files = [
    Path("recon_reports/terrain_maps/terrain_map_route_A.npz"),
    Path("recon_reports/terrain_maps/terrain_map_route_B.npz"),
]

ok = True
for path in files:
    print(f"[CHECK] {path}")
    if not path.exists():
        print("  ERROR: missing")
        ok = False
        continue
    size = path.stat().st_size
    print(f"  size: {size} bytes")
    if size <= 0:
        print("  ERROR: 0 byte")
        ok = False
        continue
    try:
        d = np.load(path, allow_pickle=True)
        print(f"  keys: {d.files}")
        for key in ("accumulated", "ground", "non_ground"):
            if key in d.files:
                arr = d[key]
                print(f"  {key}: {arr.shape}")
                if arr.ndim != 2 or arr.shape[1] != 3 or arr.shape[0] == 0:
                    print(f"  ERROR: invalid {key} shape")
                    ok = False
    except Exception as e:
        print(f"  ERROR: invalid npz: {e}")
        ok = False

if not ok:
    sys.exit(1)
PY

  local code=\$?
  if [[ "\$code" -ne 0 ]]; then
    echo
    echo "[ERROR] route terrain 검증 실패"
    echo "        A/B 지형 파일이 없거나 깨져 있습니다."
    echo "        시나리오1 정찰을 다시 확인하세요."
    exec bash
  fi

  echo "[OK] route terrain A/B valid"
}

verify_scenario2_outputs() {
  echo
  echo "============================================================"
  echo "[VERIFY] scenario2 outputs"
  echo "============================================================"

  python3 - <<'PY'
import sys
from pathlib import Path
import numpy as np

map_file = Path("recon_reports/recon_map/scenario2_map.map")
terrain_file = Path("recon_reports/recon_map/scenario2_terrain.npz")
terrain_json = Path("recon_reports/recon_map/scenario2_terrain.json")

ok = True

for path in [map_file, terrain_file, terrain_json]:
    print(f"[CHECK] {path}")
    if not path.exists():
        print("  ERROR: missing")
        ok = False
    else:
        size = path.stat().st_size
        print(f"  size: {size} bytes")
        if size <= 0:
            print("  ERROR: 0 byte")
            ok = False

if terrain_file.exists() and terrain_file.stat().st_size > 0:
    try:
        d = np.load(terrain_file, allow_pickle=True)
        print("  keys:", d.files)
        for key in ("accumulated", "ground", "non_ground"):
            if key in d.files:
                print(f"  {key}: {d[key].shape}")
        metadata = str(d["metadata_json"]) if "metadata_json" in d.files else ""
        print("  metadata:", metadata)
        if "terrain_map_route_A.npz" not in metadata or "terrain_map_route_B.npz" not in metadata:
            print("  ERROR: metadata does not include both route_A and route_B")
            ok = False
    except Exception as e:
        print(f"  ERROR: invalid scenario2 terrain npz: {e}")
        ok = False

if not ok:
    sys.exit(1)
PY

  local code=\$?
  if [[ "\$code" -ne 0 ]]; then
    echo
    echo "[ERROR] scenario2 output 검증 실패"
    echo "        build_scenario2_map.py 로그를 확인하세요:"
    echo "        $LOG_DIR/build_scenario2_map.log"
    exec bash
  fi

  echo "[OK] scenario2_map.map / scenario2_terrain.npz valid"
}

wait_for_bridge
cleanup_old_outputs

if [[ "$POSTPROCESS_ONLY" != "true" ]]; then
  echo
  echo "[WAIT] bridge/RViz 실행 후 3초 대기"
  sleep 3

  request_simulator_reset

  echo
  echo "[WAIT] reset 후 3초 대기"
  sleep 3

  run_step "run_recon_scenario" python3 scripts/run_recon_scenario.py
else
  echo "[SKIP] run_recon_scenario.py (--postprocess-only)"
fi

verify_route_terrain
run_step "analyze_run" python3 scripts/analyze_run.py
run_step "build_scenario2_map" python3 scripts/build_scenario2_map.py
verify_scenario2_outputs

echo
echo "============================================================"
echo "[DONE] Scenario 1 full pipeline completed"
echo "============================================================"
echo "결과 확인:"
echo "  recon_reports/route_A.json"
echo "  recon_reports/route_B.json"
echo "  recon_reports/analysis/run_diagnosis.md"
echo "  recon_reports/recon_map/scenario2_map.map"
echo "  recon_reports/recon_map/scenario2_terrain.npz"
echo
echo "다음 단계:"
echo "  ./scripts/run_scenario2_auto_terminator.sh"
echo
echo "로그:"
echo "  $LOG_DIR"
echo
echo "이 창은 확인용으로 유지됩니다. 닫아도 됩니다."
exec bash
EOF
chmod +x "$MANAGER_SCRIPT"

DEBUG_SCRIPT="$RUNTIME_DIR/s1_debug_pane.sh"
cat > "$DEBUG_SCRIPT" <<EOF
#!/usr/bin/env bash
set -Eeuo pipefail
printf '\\033]0;S1-T4 Debug\\007'
source "$COMMON_ENV"

echo "============================================================"
echo "[T4] Debug Monitor"
echo "============================================================"
echo "2초마다 핵심 node/topic/file을 표시합니다."
echo "종료: Ctrl+C"
echo
echo "[LOG DIR] $LOG_DIR"
echo
if [[ "$WEB_DEBUG_URL" == "true" ]]; then
  echo "rviz_web URL 예시:"
  echo "  http://127.0.0.1:5055/rviz3d?frame=tank_map&cloud=off&rays=0&vectors=0"
  echo
fi

sleep 4

watch -n 2 "
echo '[nodes]';
ros2 node list 2>/dev/null | grep -E 'ros_bridge|static_map|rviz_visualizer|terrain|lidar|overlay|dbscan|astar|local_path|controller|potential|recon' || true;
echo;
echo '[key topics]';
ros2 topic list 2>/dev/null | grep -E '/tank/(api/info/raw|player/pose|sensor/lidar/detected_points_map|sensor/lidar/terrain_points_map|visual_perception/lidar_clusters|global_path|path/lookahead_pose|control/command|control/status|map/discovered/objects|perception/fused_objects|episode/control)' || true;
echo;
echo '[files]';
ls -lh recon_reports/terrain_maps/terrain_map_route_A.npz recon_reports/terrain_maps/terrain_map_route_B.npz recon_reports/recon_map/scenario2_map.map recon_reports/recon_map/scenario2_terrain.npz 2>/dev/null || true;
echo;
echo '[services]';
ros2 service list 2>/dev/null | grep -E '/tank/(map/discovered/save|map/discovered/clear|terrain/finalize_map|terrain/reset_map)' || true;
"

exec bash
EOF
chmod +x "$DEBUG_SCRIPT"

CONFIG_FILE="$RUNTIME_DIR/scenario1_auto_terminator_config"
cat > "$CONFIG_FILE" <<EOF
[global_config]
[keybindings]
[profiles]
  [[default]]
    scrollback_infinite = True
[layouts]
  [[scenario1_auto]]
    [[[window0]]]
      type = Window
      parent = ""
      order = 0
      maximised = True
      title = Tank Scenario 1 Auto
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

echo "[RUN] Terminator Scenario 1 Auto v2"
echo "      workspace       : $WORKSPACE"
echo "      log dir         : $LOG_DIR"
echo "      rviz            : $USE_RVIZ"
echo "      skip_reset      : $SKIP_RESET"
echo "      postprocess_only: $POSTPROCESS_ONLY"
echo "      keep_old_output : $KEEP_OLD_OUTPUT"
echo "      config          : $CONFIG_FILE"
echo
echo "Terminator 4분할 창이 열립니다:"
echo "  좌상: ros_bridge"
echo "  우상: RViz"
echo "  좌하: 자동 순차 실행 manager + postprocess 검증"
echo "  우하: debug monitor"
echo

terminator -g "$CONFIG_FILE" -l scenario1_auto
