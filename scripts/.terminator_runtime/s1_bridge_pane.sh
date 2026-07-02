#!/usr/bin/env bash
set -Eeuo pipefail
printf '\033]0;S1-T1 ros_bridge\007'
source "/home/tankcc/tankcc/scripts/.terminator_runtime/common_env.sh"

echo "============================================================"
echo "[T1] ros_bridge"
echo "============================================================"
echo "Command:"
echo "  TANK_MODE=auto TANK_EPISODE_CONTROL=true ros2 run ros_bridge ros_bridge"
echo
echo "[YOLO MODEL]"
echo "  TANK_YOLO_MODEL_PATH=${TANK_YOLO_MODEL_PATH:-<not set>}"
echo
echo "[LOG] /home/tankcc/tankcc/logs/scenario1_terminator_20260701_122729/bridge.log"
echo

TANK_MODE=auto TANK_EPISODE_CONTROL=true ros2 run ros_bridge ros_bridge 2>&1 | tee "/home/tankcc/tankcc/logs/scenario1_terminator_20260701_122729/bridge.log"

echo
echo "[EXIT] ros_bridge 종료됨. 창을 닫거나 Enter를 누르세요."
exec bash
