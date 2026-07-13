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
# 시나리오 기본 주행 입력은 수동 실행과 동일하게 유지한다.
# phone_sim2real은 4번째 pane에서 실행하지만, planner cluster topic을 강제로 mux 토픽으로 바꾸지 않는다.
# 필요 시 별도 터미널에서 TANK_TOPIC_LIDAR_CLUSTERS=/tank/phone_sim2real/muxed_lidar_clusters 로 실험한다.

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
