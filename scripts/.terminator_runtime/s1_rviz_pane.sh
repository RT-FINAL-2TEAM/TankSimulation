#!/usr/bin/env bash
set -Eeuo pipefail
printf '\033]0;S1-T2 Marker Publishers (remote RViz)\007'
source "/home/tankcc/tankcc/scripts/.terminator_runtime/common_env.sh"

echo "============================================================"
echo "[T2] Marker / Map / Terrain Publisher Only"
echo "============================================================"
echo "이 PC에서는 데스크톱 RViz2와 Web RViz를 실행하지 않습니다."
echo "시각화 전용 다른 PC에서 같은 ROS_DOMAIN_ID / DDS 네트워크로"
echo "RViz를 실행해 이 노드가 발행하는 토픽을 구독하세요."
echo
echo "Command:"
echo "  ros2 launch rviz_visualization tank_rviz.launch.py use_rviz:=false"
echo
echo "[LOG] /home/tankcc/tankcc/logs/scenario1_terminator_20260708_115330/rviz.log"
echo

# 실제 RViz2 창은 생략하고, 정적맵/객체/지형/차체 marker publisher만 실행한다.
ros2 launch rviz_visualization tank_rviz.launch.py use_rviz:=false   2>&1 | tee "/home/tankcc/tankcc/logs/scenario1_terminator_20260708_115330/rviz.log"

echo
echo "[EXIT] marker publisher 종료됨. 창을 닫거나 Enter를 누르세요."
exec bash
