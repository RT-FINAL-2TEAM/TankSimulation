#!/usr/bin/env bash
set -Eeuo pipefail
printf '\033]0;S2-T1 ros_bridge\007'
source "/home/tankcc/tankcc/scripts/.terminator_runtime/common_env.sh"

echo "============================================================"
echo "[T1] ros_bridge"
echo "============================================================"
echo "live web       : true"
echo "command        : TANK_MODE=auto TANK_EPISODE_CONTROL=true TANK_LIVE_VIEW=true ros2 run ros_bridge ros_bridge"
echo "log            : /home/tankcc/tankcc/logs/scenario2_terminator_20260709_165846/bridge.log"
echo

set +e
TANK_MODE=auto TANK_EPISODE_CONTROL=true TANK_LIVE_VIEW=true ros2 run ros_bridge ros_bridge 2>&1 | tee "/home/tankcc/tankcc/logs/scenario2_terminator_20260709_165846/bridge.log"
code=${PIPESTATUS[0]}
set -e

echo
echo "[EXIT] ros_bridge 종료됨 (exit=$code)."
echo "[LOG] /home/tankcc/tankcc/logs/scenario2_terminator_20260709_165846/bridge.log"
exec bash
