#!/usr/bin/env bash
set -Eeuo pipefail

safe_source() {
  local setup_file="$1"
  set +u
  source "$setup_file"
  set -u
}

cd "/home/tankcc/tankcc"
export TANK_PROJECT_ROOT="/home/tankcc/tankcc"
# 기본값: 수동 실행과 동일하게 planner cluster topic env override 없음.
export TANK_USE_MISSION_PLAN="true"
export TANK_MISSION_PLAN_FILE="/home/tankcc/tankcc/recon_reports/mission_plan.json"

if [[ ! -f /opt/ros/humble/setup.bash ]]; then
  echo "[ERROR] /opt/ros/humble/setup.bash not found"
  exec bash
fi
safe_source /opt/ros/humble/setup.bash

if [[ ! -f "/home/tankcc/tankcc/install/setup.bash" ]]; then
  echo "[ERROR] /home/tankcc/tankcc/install/setup.bash not found"
  echo "먼저 빌드하세요: cd /home/tankcc/tankcc && colcon build"
  exec bash
fi
safe_source "/home/tankcc/tankcc/install/setup.bash"

if [[ -f "/home/tankcc/tankcc/src/vision/models/best_final.engine" ]]; then
  export TANK_YOLO_MODEL_PATH="/home/tankcc/tankcc/src/vision/models/best_final.engine"
fi
