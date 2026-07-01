#!/usr/bin/env bash
set -Eeuo pipefail
printf '\033]0;S2-T3 Auto Pipeline\007'
source "/home/tankcc/tankcc/scripts/.terminator_runtime/common_env.sh"

MAP_FILE="/home/tankcc/tankcc/recon_reports/recon_map/scenario2_map.map"
TERRAIN_FILE="/home/tankcc/tankcc/recon_reports/recon_map/scenario2_terrain.npz"

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
echo "[LOG DIR] /home/tankcc/tankcc/logs/scenario2_terminator_20260701_123248"
echo "[BUILD_MODE] auto"
echo

run_step() {
  local name="$1"
  shift

  echo
  echo "============================================================"
  echo "[RUN] $name"
  echo "============================================================"
  echo "Command: $*"
  echo

  "$@" 2>&1 | tee "/home/tankcc/tankcc/logs/scenario2_terminator_20260701_123248/$name.log"
  local code="${PIPESTATUS[0]}"

  if [[ "$code" -ne 0 ]]; then
    echo
    echo "[ERROR] $name failed with exit code $code"
    echo "로그: /home/tankcc/tankcc/logs/scenario2_terminator_20260701_123248/$name.log"
    exec bash
  fi

  echo
  echo "[OK] $name completed"
}

wait_for_bridge() {
  echo "[WAIT] ros_bridge health: http://127.0.0.1:5000/health"

  for _ in $(seq 1 60); do
    if command -v curl >/dev/null 2>&1; then
      if curl -fsS "http://127.0.0.1:5000/health" >"/home/tankcc/tankcc/logs/scenario2_terminator_20260701_123248/bridge_health.json" 2>/dev/null; then
        echo "[OK] bridge health: $(cat "/home/tankcc/tankcc/logs/scenario2_terminator_20260701_123248/bridge_health.json")"
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
  if [[ "false" == "true" ]]; then
    echo "[SKIP] simulator reset skipped"
    return 0
  fi

  echo "[RESET] simulator restart/reset request"
  echo "        topic: /tank/episode/control"
  echo "        data : reset"

  for i in 1 2 3; do
    echo "        publish reset attempt $i/3"
    timeout 6s ros2 topic pub --once /tank/episode/control std_msgs/msg/String "{data: 'reset'}" \
      2>&1 | tee "/home/tankcc/tankcc/logs/scenario2_terminator_20260701_123248/reset_attempt_${i}.log" || true
    sleep 0.5
  done
}

ensure_scenario2_map() {
  echo "[CHECK] scenario2 map:"
  echo "        $MAP_FILE"

  if [[ "auto" == "rebuild" ]]; then
    echo "[BUILD] --rebuild-map 지정됨. 기존 map 여부와 관계없이 다시 생성합니다."
    run_step "build_scenario2_map" python3 scripts/build_scenario2_map.py
  elif [[ -f "$MAP_FILE" ]]; then
    echo "[OK] existing scenario2 map found:"
    ls -lh "$MAP_FILE"
  elif [[ "auto" == "auto" ]]; then
    echo "[BUILD] scenario2_map.map 없음 → 자동 생성합니다."
    run_step "build_scenario2_map" python3 scripts/build_scenario2_map.py
  else
    echo "[ERROR] scenario2_map.map 없음, 그리고 --no-build-map 지정됨."
    echo "        먼저 시나리오1 또는 build_scenario2_map.py를 실행하세요."
    exec bash
  fi

  if [[ ! -f "$MAP_FILE" ]]; then
    echo
    echo "[ERROR] build 후에도 scenario2_map.map 없음"
    echo "        build 로그: /home/tankcc/tankcc/logs/scenario2_terminator_20260701_123248/build_scenario2_map.log"
    echo "        recon_reports/recon_map 입력 파일들을 확인하세요:"
    find recon_reports -maxdepth 3 -type f 2>/dev/null | sort || true
    exec bash
  fi

  echo
  echo "[OK] scenario2 map ready:"
  ls -lh "$MAP_FILE"
  if [[ -f "$TERRAIN_FILE" ]]; then
    echo "[OK] scenario2 terrain ready:"
    ls -lh "$TERRAIN_FILE"
  else
    echo "[WARN] scenario2 terrain npz 없음:"
    echo "       $TERRAIN_FILE"
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
echo "  /home/tankcc/tankcc/logs/scenario2_terminator_20260701_123248"
echo
echo "이 창은 확인용으로 유지됩니다. 닫아도 됩니다."
exec bash
