#!/usr/bin/env bash
set -Eeuo pipefail
printf '\033]0;S2-T2 Scenario2 RViz\007'
source "/home/acorn/TankSimulation/scripts/.terminator_runtime/common_env.sh"

MAP_FILE="/home/acorn/TankSimulation/recon_reports/recon_map/scenario2_map.map"

echo "============================================================"
echo "[T2] Scenario2 RViz"
echo "============================================================"
echo "RViz는 scenario2_map.map이 생긴 뒤 실행합니다."
echo "대기 파일:"
echo "  $MAP_FILE"
echo
echo "[LOG] /home/acorn/TankSimulation/logs/scenario2_terminator_20260704_180237/rviz_scenario2.log"
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
echo "  ros2 launch rviz_web rviz_web_scenario2_map_view.launch.py  # 웹 RViz 3D(rosbridge:9090 포함)"
echo

# 웹 RViz 3D(/view의 'RVIZ 3D' 탭)가 붙는 rosbridge(:9090) + scenario2 맵/지형 마커를 함께 띄운다.
# (rviz_visualization 데스크톱 launch는 rosbridge를 안 띄워 브라우저에 맵이 안 떴음 — 웹 launch로 교체.)
ros2 launch rviz_web rviz_web_scenario2_map_view.launch.py 2>&1 | tee "/home/acorn/TankSimulation/logs/scenario2_terminator_20260704_180237/rviz_scenario2.log"

echo
echo "[EXIT] scenario2 웹 RViz(rosbridge+마커) 종료됨. 창을 닫거나 Enter를 누르세요."
exec bash
