#!/usr/bin/env bash
set -Eeuo pipefail
printf '\033]0;S2-T3 Auto Pipeline\007'
source "/home/tankcc/tankcc/scripts/.terminator_runtime/common_env.sh"

MAP_FILE="/home/tankcc/tankcc/recon_reports/recon_map/scenario2_map.map"
TERRAIN_FILE="/home/tankcc/tankcc/recon_reports/recon_map/scenario2_terrain.npz"
TERRAIN_JSON_FILE="/home/tankcc/tankcc/recon_reports/recon_map/scenario2_terrain.json"
MISSION_PLAN_FILE="/home/tankcc/tankcc/recon_reports/mission_plan.json"

echo "============================================================"
echo "[T3] Scenario 2 Auto Pipeline"
echo "============================================================"
echo "log dir    : /home/tankcc/tankcc/logs/scenario2_terminator_20260708_164553"
echo "build mode : auto"
echo "map        : $MAP_FILE"
echo "mission    : $MISSION_PLAN_FILE"
echo

run_step() {
  local name="$1"; shift
  echo
  echo "============================================================"
  echo "[RUN] $name"
  echo "============================================================"
  echo "Command: $*"
  echo
  "$@" 2>&1 | tee "/home/tankcc/tankcc/logs/scenario2_terminator_20260708_164553/$name.log"
  local code="${PIPESTATUS[0]}"
  if [[ "$code" -ne 0 ]]; then
    echo
    echo "[ERROR] $name failed with exit code $code"
    echo "로그: /home/tankcc/tankcc/logs/scenario2_terminator_20260708_164553/$name.log"
    exec bash
  fi
  echo "[OK] $name completed"
}

run_optional_step() {
  local name="$1"; shift
  echo
  echo "============================================================"
  echo "[RUN/OPTIONAL] $name"
  echo "============================================================"
  echo "Command: $*"
  echo
  set +e
  "$@" 2>&1 | tee "/home/tankcc/tankcc/logs/scenario2_terminator_20260708_164553/$name.log"
  local code="${PIPESTATUS[0]}"
  set -e
  if [[ "$code" -ne 0 ]]; then
    echo "[WARN] optional step failed: $name (exit=$code). 계속 진행합니다."
    echo "       로그: /home/tankcc/tankcc/logs/scenario2_terminator_20260708_164553/$name.log"
  else
    echo "[OK] $name completed"
  fi
}

wait_for_bridge() {
  echo "[WAIT] ros_bridge health: http://127.0.0.1:5000/health"
  for _ in $(seq 1 60); do
    if command -v curl >/dev/null 2>&1 && curl -fsS "http://127.0.0.1:5000/health" >"/home/tankcc/tankcc/logs/scenario2_terminator_20260708_164553/bridge_health.json" 2>/dev/null; then
      echo "[OK] bridge health: $(cat "/home/tankcc/tankcc/logs/scenario2_terminator_20260708_164553/bridge_health.json")"
      return 0
    fi
    sleep 0.5
  done
  echo "[WARN] bridge health 확인 실패. 그래도 계속 진행합니다."
}

request_simulator_reset() {
  if [[ "false" == "true" ]]; then
    echo "[SKIP] simulator reset skipped"
    return 0
  fi
  echo "[RESET] simulator restart/reset request -> /tank/episode/control reset"
  for i in 1 2 3; do
    timeout 6s ros2 topic pub --once /tank/episode/control std_msgs/msg/String "{data: 'reset'}"       2>&1 | tee "/home/tankcc/tankcc/logs/scenario2_terminator_20260708_164553/reset_attempt_${i}.log" || true
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
  echo "[CHECK] scenario2 map: $MAP_FILE"
  local need_build="false"
  if [[ "auto" == "rebuild" ]]; then
    echo "[BUILD] --rebuild-map 지정됨."
    need_build="true"
  elif [[ ! -f "$MAP_FILE" ]]; then
    if [[ "auto" == "never" ]]; then
      echo "[ERROR] scenario2_map.map 없음, 그리고 --no-build-map 지정됨."
      exec bash
    fi
    echo "[BUILD] scenario2_map.map 없음."
    need_build="true"
  elif map_inputs_newer_than_output; then
    if [[ "auto" == "never" ]]; then
      echo "[ERROR] scenario2_map.map이 입력보다 오래됐지만 --no-build-map 지정됨."
      exec bash
    fi
    echo "[BUILD] route A/B 정찰 입력이 scenario2_map.map보다 최신입니다. 자동 재생성합니다."
    need_build="true"
  else
    echo "[OK] existing scenario2 map is current enough."
  fi

  if [[ "$need_build" == "true" ]]; then
    run_step "build_scenario2_map" python3 scripts/build_scenario2_map.py
  fi

  if [[ ! -f "$MAP_FILE" ]]; then
    echo "[ERROR] build 후에도 scenario2_map.map 없음: $MAP_FILE"
    find recon_reports -maxdepth 3 -type f 2>/dev/null | sort || true
    exec bash
  fi
  echo "[OK] scenario2 map ready:"; ls -lh "$MAP_FILE"
  [[ -f "$TERRAIN_FILE" ]] && { echo "[OK] terrain npz:"; ls -lh "$TERRAIN_FILE"; } || true
  [[ -f "$TERRAIN_JSON_FILE" ]] && { echo "[OK] terrain json:"; ls -lh "$TERRAIN_JSON_FILE"; } || true
}

ensure_mission_plan() {
  if mission_plan_stale; then
    echo "[BUILD] mission_plan.json 없음 또는 scenario2_map/terrain보다 오래됨. 재생성합니다."
    run_optional_step "build_mission_plan" python3 scripts/build_mission_plan.py
  else
    echo "[OK] mission_plan.json is current enough."
  fi
  if [[ -f "$MISSION_PLAN_FILE" ]]; then
    echo "[OK] mission plan:"; ls -lh "$MISSION_PLAN_FILE"
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
echo "로그: /home/tankcc/tankcc/logs/scenario2_terminator_20260708_164553"
exec bash
