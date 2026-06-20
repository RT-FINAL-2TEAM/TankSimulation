#!/usr/bin/env bash
# 자율주행 스택(lidar+인지+A*+APF+컨트롤러)을 launch 하나로 실행.
#   사용법: scripts/run_stack.sh [--route A|B] [--mission recon|mission|return]
#   route_side는 route_id로 자동(A=west / B=east).
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ROUTE="A"; MISSION="recon"
while [ $# -gt 0 ]; do
  case "$1" in
    --route) ROUTE="$2"; shift 2;;
    --route=*) ROUTE="${1#*=}"; shift;;
    --mission) MISSION="$2"; shift 2;;
    --mission=*) MISSION="${1#*=}"; shift;;
    -h|--help) sed -n '2,4p' "$0"; exit 0;;
    *) echo "알 수 없는 인자: $1 (--help)"; exit 1;;
  esac
done

SIDE="west"; [ "$ROUTE" = "B" ] && SIDE="east"

source /opt/ros/humble/setup.bash
source install/setup.bash 2>/dev/null || { echo "[run_stack] install/ 없음 — colcon build 먼저"; exit 1; }

echo "[run_stack] mission=$MISSION route=$ROUTE side=$SIDE"
exec ros2 launch control tank_autonomous_control.launch.py \
  mission_type:="$MISSION" route_id:="$ROUTE" route_side:="$SIDE"
