#!/usr/bin/env bash
set -Eeuo pipefail
printf '\033]0;S2-T2 Scenario2 RViz\007'
source "/home/tankcc/tankcc/scripts/.terminator_runtime/common_env.sh"

MAP_FILE="/home/tankcc/tankcc/recon_reports/recon_map/scenario2_map.map"

echo "============================================================"
echo "[T2] Scenario2 RViz"
echo "============================================================"
echo "RViz는 scenario2_map.map이 생긴 뒤 실행합니다."
echo "대기 파일:"
echo "  $MAP_FILE"
echo
echo "[LOG] /home/tankcc/tankcc/logs/scenario2_terminator_20260701_123248/rviz_scenario2.log"
echo

for i in $(seq 1 "180"); do
  if [[ -f "$MAP_FILE" ]]; then
    echo "[OK] scenario2 map found:"
    ls -lh "$MAP_FILE"
    break
  fi
  if [[ "$i" -eq "180" ]]; then
    echo "[ERROR] scenario2 map wait timeout: $MAP_FILE"
    echo "        manager pane에서 build_scenario2_map.py 로그를 확인하세요."
    exec bash
  fi
  echo "[WAIT] scenario2_map.map 대기 중... $i/180"
  sleep 1
done

echo
echo "Command:"
echo "  ros2 launch rviz_visualization tank_scenario2_map_view.launch.py"
echo

ros2 launch rviz_visualization tank_scenario2_map_view.launch.py 2>&1 | tee "/home/tankcc/tankcc/logs/scenario2_terminator_20260701_123248/rviz_scenario2.log" &
RVIZ_LAUNCH_PID=$!

echo "[WAIT] RViz 창 감지 후 최대화 시도"
for _ in $(seq 1 60); do
  if command -v wmctrl >/dev/null 2>&1; then
    WIN_ID="$(wmctrl -lx 2>/dev/null | awk 'BEGIN{IGNORECASE=1} /rviz|rviz2/ {print $1; exit}')"
    if [[ -n "${WIN_ID:-}" ]]; then
      wmctrl -ir "$WIN_ID" -b add,maximized_vert,maximized_horz >/dev/null 2>&1 || true
      wmctrl -ia "$WIN_ID" >/dev/null 2>&1 || true
      echo "[OK] RViz window maximized"
      break
    fi
  elif command -v xdotool >/dev/null 2>&1; then
    WIN_ID="$(xdotool search --name 'RViz' 2>/dev/null | head -n 1 || true)"
    if [[ -n "${WIN_ID:-}" ]]; then
      xdotool windowactivate "$WIN_ID" >/dev/null 2>&1 || true
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

wait "$RVIZ_LAUNCH_PID"

echo
echo "[EXIT] scenario2 RViz launch 종료됨. 창을 닫거나 Enter를 누르세요."
exec bash
