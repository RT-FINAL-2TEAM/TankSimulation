#!/usr/bin/env bash
# 정찰 A→B 자동 시퀀스(루트 사이 시뮬 자동 리셋).
#   전제: 다른 터미널에서  scripts/run_bridge.sh --mode auto --reset  로 브릿지를 먼저 띄우고 시뮬 시작.
#   사용법: scripts/run_recon.sh
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source /opt/ros/humble/setup.bash
source install/setup.bash 2>/dev/null || { echo "[run_recon] install/ 없음 — colcon build 먼저"; exit 1; }
echo "[run_recon] scripts/run_recon_scenario.py  (전제: run_bridge.sh --mode auto --reset + 시뮬 시작)"
exec python3 scripts/run_recon_scenario.py "$@"
