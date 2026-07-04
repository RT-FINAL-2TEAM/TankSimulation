#!/usr/bin/env bash
set -Eeuo pipefail

safe_source() {
  local setup_file="$1"
  set +u
  source "$setup_file"
  set -u
}

cd "/home/acorn/TankSimulation"

if [[ ! -f /opt/ros/humble/setup.bash ]]; then
  echo "[ERROR] /opt/ros/humble/setup.bash not found"
  exec bash
fi

safe_source /opt/ros/humble/setup.bash

if [[ ! -f "/home/acorn/TankSimulation/install/setup.bash" ]]; then
  echo "[ERROR] /home/acorn/TankSimulation/install/setup.bash not found"
  echo "먼저 빌드하세요:"
  echo "  cd /home/acorn/TankSimulation && colcon build"
  exec bash
fi

safe_source "/home/acorn/TankSimulation/install/setup.bash"

if [[ -f "/home/acorn/TankSimulation/src/vision/models/best_final.engine" ]]; then
  export TANK_YOLO_MODEL_PATH="/home/acorn/TankSimulation/src/vision/models/best_final.engine"
fi
