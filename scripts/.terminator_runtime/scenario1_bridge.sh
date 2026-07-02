#!/usr/bin/env bash
cd "/home/tankcc/tankcc"
printf '\033]0;S1 T1 Bridge\007'
source "/home/tankcc/tankcc/scripts/.terminator_runtime/common_env.sh"
echo "[S1/T1] ros_bridge auto + episode control"
echo "Command:"
echo "  TANK_MODE=auto TANK_EPISODE_CONTROL=true ros2 run ros_bridge ros_bridge"
echo
TANK_MODE=auto TANK_EPISODE_CONTROL=true ros2 run ros_bridge ros_bridge
echo
echo "[EXIT] bridge finished. Press Enter or close pane."
exec bash
