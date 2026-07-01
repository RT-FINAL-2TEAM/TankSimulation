#!/usr/bin/env bash
cd "/home/tankcc/tankcc"
printf '\033]0;S1 T2 RViz\007'
source "/home/tankcc/tankcc/scripts/.terminator_runtime/common_env.sh"
echo "[S1/T2] desktop RViz"
echo "Command:"
echo "  ros2 launch rviz_visualization tank_rviz.launch.py"
echo
ros2 launch rviz_visualization tank_rviz.launch.py
echo
echo "[EXIT] RViz launch finished. Press Enter or close pane."
exec bash
