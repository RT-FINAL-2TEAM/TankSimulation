#!/usr/bin/env bash
set -Eeuo pipefail

safe_source() {
  local setup_file="$1"
  set +u
  source "$setup_file"
  set -u
}

cd "/home/tankcc/tankcc"

if [[ ! -f /opt/ros/humble/setup.bash ]]; then
  echo "[ERROR] /opt/ros/humble/setup.bash not found"
  exec bash
fi

safe_source /opt/ros/humble/setup.bash

if [[ ! -f "/home/tankcc/tankcc/install/setup.bash" ]]; then
  echo "[ERROR] /home/tankcc/tankcc/install/setup.bash not found"
  echo "먼저 빌드하세요:"
  echo "  cd /home/tankcc/tankcc && colcon build"
  exec bash
fi

safe_source "/home/tankcc/tankcc/install/setup.bash"

if [[ -f "/home/tankcc/tankcc/src/vision/models/best_final.engine" ]]; then
  export TANK_YOLO_MODEL_PATH="/home/tankcc/tankcc/src/vision/models/best_final.engine"
fi
