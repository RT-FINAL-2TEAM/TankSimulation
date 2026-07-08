#!/usr/bin/env bash
set -Eeuo pipefail
printf '\033]0;S2-T2 Scenario2 Marker Publishers\007'
source "/home/tankcc/tankcc/scripts/.terminator_runtime/common_env.sh"

MAP_FILE="/home/tankcc/tankcc/recon_reports/recon_map/scenario2_map.map"

echo "============================================================"
echo "[T2] RViz2 창 생략 (--no-rviz) — scenario2 marker publisher만 실행"
echo "============================================================"
echo "대기 파일: $MAP_FILE"
echo

for i in $(seq 1 "180"); do
  if [[ -f "$MAP_FILE" ]]; then
    echo "[OK] scenario2 map found:"; ls -lh "$MAP_FILE"; break
  fi
  if [[ "$i" -eq "180" ]]; then
    echo "[ERROR] scenario2 map wait timeout: $MAP_FILE"; exec bash
  fi
  echo "[WAIT] scenario2_map.map 대기 중... $i/180"; sleep 1
done

echo
echo "Command:"
echo "  ros2 launch rviz_visualization tank_scenario2_map_view.launch.py use_rviz:=false"
echo

ros2 launch rviz_visualization tank_scenario2_map_view.launch.py use_rviz:=false   2>&1 | tee "/home/tankcc/tankcc/logs/scenario2_terminator_20260708_120036/rviz_scenario2.log"

echo
echo "[EXIT] scenario2 marker publisher 종료됨. 창을 닫거나 Enter를 누르세요."
exec bash
