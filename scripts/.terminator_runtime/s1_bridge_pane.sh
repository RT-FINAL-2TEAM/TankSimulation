#!/usr/bin/env bash
set -Eeuo pipefail
printf '\033]0;S1-T1 ros_bridge\007'
source "/home/tankcc/tankcc/scripts/.terminator_runtime/common_env.sh"

echo "============================================================"
echo "[T1] ros_bridge"
echo "============================================================"
echo "Command:"
echo "  TANK_MODE=auto TANK_EPISODE_CONTROL=true TANK_LIVE_VIEW=true ros2 run ros_bridge ros_bridge"
echo
echo "[YOLO MODEL]"
echo "  TANK_YOLO_MODEL_PATH=${TANK_YOLO_MODEL_PATH:-<not set>}"
echo
echo "[LOG] /home/tankcc/tankcc/logs/scenario1_terminator_20260708_115330/bridge.log"
echo

# set -e/pipefail 상태에서는 ros_bridge가 비정상 종료될 때 pane 자체도 즉시 닫힌다.
# 종료 코드를 보존하면서 마지막 로그와 원인을 확인할 수 있도록 여기서만 errexit를 끈다.
set +e
TANK_MODE=auto TANK_EPISODE_CONTROL=true TANK_LIVE_VIEW=true ros2 run ros_bridge ros_bridge 2>&1 | tee "/home/tankcc/tankcc/logs/scenario1_terminator_20260708_115330/bridge.log"
BRIDGE_EXIT_CODE=${PIPESTATUS[0]}
set -e

echo
echo "[EXIT] ros_bridge 종료됨 (exit=$BRIDGE_EXIT_CODE)."
echo "[LOG] /home/tankcc/tankcc/logs/scenario1_terminator_20260708_115330/bridge.log"
echo "위 로그의 마지막 Traceback/ERROR를 확인한 뒤 Enter를 누르거나 창을 닫으세요."
exec bash
