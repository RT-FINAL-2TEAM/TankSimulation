#!/usr/bin/env bash
# ros_bridge 실행. 모든 설정의 단일 출처는 .env(워크스페이스 루트, 자동 로드).
# --mode / --reset 로 이번 실행만 일시 override.
#   사용법: scripts/run_bridge.sh [--mode auto|monitor] [--reset]
#   예) 자율주행:       scripts/run_bridge.sh --mode auto
#       정찰(자동리셋):  scripts/run_bridge.sh --mode auto --reset
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODE=""; RESET=""
while [ $# -gt 0 ]; do
  case "$1" in
    --mode) MODE="$2"; shift 2;;
    --mode=*) MODE="${1#*=}"; shift;;
    --reset) RESET=1; shift;;
    -h|--help) sed -n '2,7p' "$0"; exit 0;;
    *) echo "알 수 없는 인자: $1 (--help)"; exit 1;;
  esac
done

source /opt/ros/humble/setup.bash
source install/setup.bash 2>/dev/null || { echo "[run_bridge] install/ 없음 — colcon build 먼저"; exit 1; }

[ -n "$MODE" ]  && export TANK_MODE="$MODE"
[ -n "$RESET" ] && export TANK_EPISODE_CONTROL=true

PORT="$(grep -E '^TANK_BRIDGE_PORT=' .env 2>/dev/null | tail -1 | cut -d= -f2 | tr -d ' ')"; PORT="${PORT:-5000}"
echo "[run_bridge] mode=${TANK_MODE:-.env}  episode_control=${TANK_EPISODE_CONTROL:-.env}"
echo "[run_bridge] 윈도우 tank_proxy.py의 UBUNTU_SERVER에 입력 → http://$(hostname -I | awk '{print $1}'):${PORT}"
exec ros2 run ros_bridge ros_bridge
