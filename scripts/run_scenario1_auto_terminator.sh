#!/usr/bin/env bash
set -Eeuo pipefail

# Scenario 1 auto runner.
# - Default: no local /view web, no local desktop RViz GUI.
# - --web : enable ros_bridge /view on this PC.
# - --rviz: open desktop RViz2 on this PC.
# Visualization backend is intentionally kept independent from RViz GUI: when --rviz is
# omitted, T2 still runs marker publishers + terrain recorder with use_rviz:=false.
# A second PC/terminal should open viewer-only RViz to avoid duplicate publishers.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="${TANK_WS:-$(cd "$SCRIPT_DIR/.." && pwd)}"
LOG_ROOT="${LOG_ROOT:-$WORKSPACE/logs}"
RUN_ID="scenario1_terminator_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$LOG_ROOT/$RUN_ID"
RUNTIME_DIR="$WORKSPACE/scripts/.terminator_runtime"

WEB_ENABLED="false"
RVIZ_ENABLED="false"
SKIP_RESET="false"
POSTPROCESS_ONLY="false"
KEEP_OLD_OUTPUT="false"
BRIDGE_HEALTH_URL="${BRIDGE_HEALTH_URL:-http://127.0.0.1:5000/health}"
PHONE_PORT="${PHONE_PORT:-5002}"

usage() {
  cat <<USAGE
Usage: $0 [--web] [--rviz] [--skip-reset] [--postprocess-only] [--keep-old-output]

Options:
  --web              이 PC의 ros_bridge에서 /view live web을 켬. 기본값은 꺼짐.
  --rviz             이 PC에서 데스크톱 RViz2를 함께 실행. 기본값은 RViz2 GUI 꺼짐.
  --skip-reset       run_recon_scenario.py 실행 전 simulator reset 요청 생략
  --postprocess-only run_recon_scenario.py는 생략하고 analyze/build만 실행
  --keep-old-output  시작 전에 기존 정찰/시나리오2 산출물을 삭제하지 않음
  --no-rviz          호환용 no-op. 기본값이 이미 RViz2 GUI off.
  -h, --help         도움말 출력

Panes:
  T1 ros_bridge
  T2 visualization backend only, or backend + RViz2 GUI when --rviz is set
  T3 scenario1 manager
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
    --postprocess-only)
      POSTPROCESS_ONLY="true"; shift ;;
    --keep-old-output)
      KEEP_OLD_OUTPUT="true"; shift ;;
    --web-rviz|--web-debug-url)
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
  if [[ "${KEEP_OLD_OUTPUT:-false}" == "true" ]] || [[ "${POSTPROCESS_ONLY:-false}" == "true" ]]; then
    echo "[PRE-FLIGHT] 기존 산출물 유지"
    return 0
  fi

  echo "[PRE-FLIGHT] 이전 report/map/latest 산출물을 Terminator 실행 전에 먼저 정리합니다."
  mkdir -p "$WORKSPACE/recon_reports/analysis" \
           "$WORKSPACE/recon_reports/recon_map" \
           "$WORKSPACE/recon_reports/terrain_maps" \
           "$WORKSPACE/recon_reports/scenario2" \
           "$WORKSPACE/tank_discovered_maps" \
           "$WORKSPACE/tank_terrain_maps"

  rm -f \
    "$WORKSPACE/recon_reports/route_A.json" \
    "$WORKSPACE/recon_reports/route_B.json" \
    "$WORKSPACE/recon_reports/comparison.json" \
    "$WORKSPACE/recon_reports/route_comparison.json" \
    "$WORKSPACE/recon_reports/risk_features.json" \
    "$WORKSPACE/recon_reports/risk_comparison.json" \
    "$WORKSPACE/recon_reports/risk_comparison.md" \
    "$WORKSPACE/recon_reports/route_risk_result.json" \
    "$WORKSPACE/recon_reports/route_analysis_report.txt" \
    "$WORKSPACE/recon_reports/mission_plan.json" \
    "$WORKSPACE/recon_reports/analysis/recon_report.md" \
    "$WORKSPACE/recon_reports/analysis/mission_plan.md" \
    "$WORKSPACE/recon_reports/recon_map/discovered_objects_route_A.map" \
    "$WORKSPACE/recon_reports/recon_map/discovered_objects_route_B.map" \
    "$WORKSPACE/recon_reports/recon_map/scenario2_map.map" \
    "$WORKSPACE/recon_reports/recon_map/scenario2_terrain.json" \
    "$WORKSPACE/recon_reports/recon_map/scenario2_terrain.npz" \
    "$WORKSPACE/recon_reports/terrain_maps/terrain_map_route_A.npz" \
    "$WORKSPACE/recon_reports/terrain_maps/terrain_map_route_B.npz" \
    "$WORKSPACE/recon_reports/scenario2/route_A.json" \
    "$WORKSPACE/recon_reports/scenario2/scenario2_result.json" \
    "$WORKSPACE/tank_discovered_maps/discovered_objects_latest.map" \
    "$WORKSPACE/tank_terrain_maps/terrain_map_latest.npz"
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
# 시나리오 기본 주행 입력은 수동 실행과 동일하게 유지한다.
# phone_sim2real은 4번째 pane에서 실행하지만, planner cluster topic을 강제로 mux 토픽으로 바꾸지 않는다.
# 필요 시 별도 터미널에서 TANK_TOPIC_LIDAR_CLUSTERS=/tank/phone_sim2real/muxed_lidar_clusters 로 실험한다.

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

BRIDGE_SCRIPT="$RUNTIME_DIR/s1_bridge_pane.sh"
cat > "$BRIDGE_SCRIPT" <<EOF_BRIDGE
#!/usr/bin/env bash
set -Eeuo pipefail
printf '\\033]0;S1-T1 ros_bridge\\007'
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

VIZ_SCRIPT="$RUNTIME_DIR/s1_viz_or_terrain_pane.sh"
if [[ "$RVIZ_ENABLED" == "true" ]]; then
  cat > "$VIZ_SCRIPT" <<EOF_VIZ
#!/usr/bin/env bash
set -Eeuo pipefail
printf '\\033]0;S1-T2 RViz2 + Terrain\\007'
source "$COMMON_ENV"

echo "============================================================"
echo "[T2] Scenario1 RViz2 + Visualization Backend (--rviz)"
echo "============================================================"
echo "Command: ros2 launch rviz_visualization tank_rviz.launch.py"
echo "Log    : $LOG_DIR/rviz.log"
echo
sleep 0.7
ros2 launch rviz_visualization tank_rviz.launch.py 2>&1 | tee "$LOG_DIR/rviz.log"
echo
echo "[EXIT] RViz2/marker/terrain launch 종료."
exec bash
EOF_VIZ
else
  cat > "$VIZ_SCRIPT" <<EOF_VIZ
#!/usr/bin/env bash
set -Eeuo pipefail
printf '\\033]0;S1-T2 Terrain Recorder\\007'
source "$COMMON_ENV"

echo "============================================================"
echo "[T2] Visualization Backend Only"
echo "============================================================"
echo "RViz2 GUI는 실행하지 않습니다. 이 PC에서 RViz2를 보려면 다음 실행 때 --rviz를 붙이세요."
echo "마커 발행 노드와 지형 recorder는 이 pane에서 단일 출처로 실행합니다."
echo "다른 PC/터미널에서는 중복 publisher가 생기지 않도록 viewer-only launch만 실행하세요:"
echo "  ros2 launch rviz_visualization tank_rviz_viewer_only.launch.py"
echo
echo "Command: ros2 launch rviz_visualization tank_rviz.launch.py use_rviz:=false"
echo "Log    : $LOG_DIR/rviz_backend.log"
echo
sleep 0.7
ros2 launch rviz_visualization tank_rviz.launch.py use_rviz:=false 2>&1 | tee "$LOG_DIR/rviz_backend.log"
echo
echo "[EXIT] visualization backend 종료."
exec bash
EOF_VIZ
fi
chmod +x "$VIZ_SCRIPT"

MANAGER_SCRIPT="$RUNTIME_DIR/s1_manager_pane.sh"
cat > "$MANAGER_SCRIPT" <<EOF_MANAGER
#!/usr/bin/env bash
set -Eeuo pipefail
printf '\\033]0;S1-T3 Auto Pipeline\\007'
source "$COMMON_ENV"

echo "============================================================"
echo "[T3] Scenario 1 Auto Pipeline"
echo "============================================================"
echo "log dir          : $LOG_DIR"
echo "postprocess only : $POSTPROCESS_ONLY"
echo "keep old output  : $KEEP_OLD_OUTPUT"
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

cleanup_old_outputs() {
  if [[ "$KEEP_OLD_OUTPUT" == "true" ]] || [[ "$POSTPROCESS_ONLY" == "true" ]]; then
    echo "[KEEP] 기존 산출물 유지"
    return 0
  fi

  echo "[CLEAN] 시나리오1/시나리오2 파생 산출물 및 latest runtime 파일 정리"
  mkdir -p recon_reports/analysis recon_reports/recon_map recon_reports/terrain_maps recon_reports/scenario2 \
           tank_discovered_maps tank_terrain_maps

  rm -f \
    recon_reports/route_A.json \
    recon_reports/route_B.json \
    recon_reports/comparison.json \
    recon_reports/route_comparison.json \
    recon_reports/risk_features.json \
    recon_reports/risk_comparison.json \
    recon_reports/risk_comparison.md \
    recon_reports/route_risk_result.json \
    recon_reports/route_analysis_report.txt \
    recon_reports/mission_plan.json \
    recon_reports/analysis/recon_report.md \
    recon_reports/analysis/mission_plan.md \
    recon_reports/recon_map/discovered_objects_route_A.map \
    recon_reports/recon_map/discovered_objects_route_B.map \
    recon_reports/recon_map/scenario2_map.map \
    recon_reports/recon_map/scenario2_terrain.json \
    recon_reports/recon_map/scenario2_terrain.npz \
    recon_reports/terrain_maps/terrain_map_route_A.npz \
    recon_reports/terrain_maps/terrain_map_route_B.npz \
    recon_reports/scenario2/route_A.json \
    recon_reports/scenario2/scenario2_result.json \
    tank_discovered_maps/discovered_objects_latest.map \
    tank_terrain_maps/terrain_map_latest.npz
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
files = [Path("recon_reports/terrain_maps/terrain_map_route_A.npz"), Path("recon_reports/terrain_maps/terrain_map_route_B.npz")]
ok = True
for path in files:
    print(f"[CHECK] {path}")
    if not path.exists():
        print("  ERROR: missing"); ok = False; continue
    size = path.stat().st_size
    print(f"  size: {size} bytes")
    if size <= 0:
        print("  ERROR: 0 byte"); ok = False; continue
    try:
        d = np.load(path, allow_pickle=True)
        print(f"  keys: {d.files}")
        for key in ("accumulated", "ground", "non_ground"):
            if key in d.files:
                arr = d[key]
                print(f"  {key}: {arr.shape}")
                if arr.ndim != 2 or arr.shape[1] != 3 or arr.shape[0] == 0:
                    print(f"  ERROR: invalid {key} shape"); ok = False
    except Exception as e:
        print(f"  ERROR: invalid npz: {e}"); ok = False
if not ok:
    sys.exit(1)
PY
}

verify_scenario2_outputs() {
  echo
  echo "============================================================"
  echo "[VERIFY] scenario2 map/terrain outputs"
  echo "============================================================"
  python3 - <<'PY'
import json, sys
from pathlib import Path
import numpy as np
map_file = Path("recon_reports/recon_map/scenario2_map.map")
terrain_file = Path("recon_reports/recon_map/scenario2_terrain.npz")
terrain_json = Path("recon_reports/recon_map/scenario2_terrain.json")
ok = True
for path in [map_file, terrain_file, terrain_json]:
    print(f"[CHECK] {path}")
    if not path.exists():
        print("  ERROR: missing"); ok = False
    elif path.stat().st_size <= 0:
        print("  ERROR: 0 byte"); ok = False
    else:
        print(f"  size: {path.stat().st_size} bytes")
if map_file.exists() and map_file.stat().st_size > 0:
    try:
        data = json.loads(map_file.read_text(encoding="utf-8"))
        print("  object_count:", data.get("object_count"), "target_count:", data.get("target_count"))
        src = data.get("source_files", {}) or {}
        print("  discovered_maps:", [Path(p).name for p in src.get("discovered_maps", [])])
        if len(src.get("discovered_maps", [])) < 2:
            print("  WARN: route A/B discovered maps 중 일부가 빠져 있습니다.")
    except Exception as e:
        print("  ERROR: invalid map json:", e); ok = False
if terrain_file.exists() and terrain_file.stat().st_size > 0:
    try:
        d = np.load(terrain_file, allow_pickle=True)
        print("  keys:", d.files)
        metadata = str(d["metadata_json"]) if "metadata_json" in d.files else ""
        print("  metadata:", metadata)
        if "terrain_map_route_A.npz" not in metadata or "terrain_map_route_B.npz" not in metadata:
            print("  ERROR: metadata does not include both route_A and route_B"); ok = False
    except Exception as e:
        print(f"  ERROR: invalid scenario2 terrain npz: {e}"); ok = False
if not ok:
    sys.exit(1)
PY
}

wait_for_bridge
cleanup_old_outputs
if [[ "$POSTPROCESS_ONLY" != "true" ]]; then
  echo "[WAIT] bridge/terrain recorder 준비 대기 3초"
  sleep 3
  request_simulator_reset
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
# build_scenario2_map을 마지막에 다시 실행했으므로 mission_plan도 최신 scenario2_map 기준으로 다시 생성한다.
run_optional_step "build_mission_plan" python3 scripts/build_mission_plan.py

echo
echo "============================================================"
echo "[DONE] Scenario 1 full pipeline completed"
echo "============================================================"
echo "결과: recon_reports/route_A.json, route_B.json, recon_map/scenario2_map.map, mission_plan.json"
echo "다음 단계: ./scripts/run_scenario2_auto_terminator.sh"
echo "로그: $LOG_DIR"
exec bash
EOF_MANAGER
chmod +x "$MANAGER_SCRIPT"

PHONE_SCRIPT="$RUNTIME_DIR/s1_phone_sim2real_pane.sh"
cat > "$PHONE_SCRIPT" <<EOF_PHONE
#!/usr/bin/env bash
set -Eeuo pipefail
printf '\\033]0;S1-T4 phone_sim2real\\007'
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

CONFIG_FILE="$RUNTIME_DIR/scenario1_auto_terminator_config"
cat > "$CONFIG_FILE" <<EOF_CFG
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
    [[[terminal_viz_or_terrain]]]
      type = Terminal
      parent = vpaned_right
      order = 0
      command = bash -lc '$VIZ_SCRIPT'
    [[[terminal_phone]]]
      type = Terminal
      parent = vpaned_right
      order = 1
      command = bash -lc '$PHONE_SCRIPT'
[plugins]
EOF_CFG

echo "[RUN] Terminator Scenario 1 Auto"
echo "      workspace       : $WORKSPACE"
echo "      log dir         : $LOG_DIR"
echo "      local web /view : $WEB_ENABLED"
echo "      desktop RViz2   : $RVIZ_ENABLED"
echo "      skip_reset      : $SKIP_RESET"
echo "      postprocess_only: $POSTPROCESS_ONLY"
echo "      keep_old_output : $KEEP_OLD_OUTPUT"
echo "      phone_port      : $PHONE_PORT"
echo
echo "Terminator 4분할 창:"
echo "  좌상: ros_bridge"
echo "  우상: terrain recorder 또는 RViz2(--rviz)"
echo "  좌하: scenario1 manager"
echo "  우하: phone_sim2real"
echo
if [[ "$WEB_ENABLED" != "true" ]]; then
  echo "[WEB] 이 ros_bridge의 /view는 꺼져 있습니다. 이 PC에서 보려면 --web을 붙이세요."
fi
if [[ "$RVIZ_ENABLED" != "true" ]]; then
  echo "[RVIZ] 이 PC에서 RViz2 GUI는 열지 않습니다. 다른 PC/터미널은 tank_rviz_viewer_only.launch.py를 사용하세요."
fi

terminator -u -g "$CONFIG_FILE" -l scenario1_auto
