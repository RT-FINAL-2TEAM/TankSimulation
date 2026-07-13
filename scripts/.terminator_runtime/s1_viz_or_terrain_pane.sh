#!/usr/bin/env bash
set -Eeuo pipefail
printf '\033]0;S1-T2 RViz2 + Terrain\007'
source "/home/tankcc/tankcc/scripts/.terminator_runtime/common_env.sh"

echo "============================================================"
echo "[T2] Scenario1 RViz2 + Visualization Backend (--rviz)"
echo "============================================================"
echo "Command: ros2 launch rviz_visualization tank_rviz.launch.py"
echo "Log    : /home/tankcc/tankcc/logs/scenario1_terminator_20260713_175510/rviz.log"
echo
sleep 0.7
ros2 launch rviz_visualization tank_rviz.launch.py 2>&1 | tee "/home/tankcc/tankcc/logs/scenario1_terminator_20260713_175510/rviz.log"
echo
echo "[EXIT] RViz2/marker/terrain launch 종료."
exec bash
