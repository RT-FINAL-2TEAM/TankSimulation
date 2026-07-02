#!/usr/bin/env bash
set -Eeuo pipefail
printf '\033]0;S1-T2 RViz\007'
source "/home/tankcc/tankcc/scripts/.terminator_runtime/common_env.sh"

echo "============================================================"
echo "[T2] RViz"
echo "============================================================"
echo "Command:"
echo "  ros2 launch rviz_visualization tank_rviz.launch.py"
echo
echo "[LOG] /home/tankcc/tankcc/logs/scenario1_terminator_20260701_122729/rviz.log"
echo

ros2 launch rviz_visualization tank_rviz.launch.py 2>&1 | tee "/home/tankcc/tankcc/logs/scenario1_terminator_20260701_122729/rviz.log" &
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
echo "[EXIT] RViz launch 종료됨. 창을 닫거나 Enter를 누르세요."
exec bash
