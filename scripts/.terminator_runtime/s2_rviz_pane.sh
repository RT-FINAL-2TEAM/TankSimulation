#!/usr/bin/env bash
set -Eeuo pipefail
printf '\033]0;S2-T2 RViz2 Map View\007'
source "/home/tankcc/tankcc/scripts/.terminator_runtime/common_env.sh"

MAP_FILE="/home/tankcc/tankcc/recon_reports/recon_map/scenario2_map.map"

echo "============================================================"
echo "[T2] Scenario2 Desktop RViz2 + Visualization Backend (--rviz)"
echo "============================================================"
echo "RViz2는 scenario2_map.map이 준비된 뒤 실행합니다."
echo "map : $MAP_FILE"
echo "log : /home/tankcc/tankcc/logs/scenario2_terminator_20260710_095630/rviz_scenario2.log"
echo
for i in $(seq 1 "180"); do
  if [[ -f "$MAP_FILE" ]]; then
    echo "[OK] scenario2 map found:"; ls -lh "$MAP_FILE"; break
  fi
  if [[ "$i" -eq "180" ]]; then
    echo "[ERROR] scenario2 map wait timeout: $MAP_FILE"
    echo "        manager pane의 build_scenario2_map.py 로그를 확인하세요."
    exec bash
  fi
  echo "[WAIT] scenario2_map.map 대기 중... $i/180"
  sleep 1
done

sleep 0.7
ros2 launch rviz_visualization tank_scenario2_map_view.launch.py 2>&1 | tee "/home/tankcc/tankcc/logs/scenario2_terminator_20260710_095630/rviz_scenario2.log"
echo
echo "[EXIT] scenario2 RViz2 종료."
exec bash
