#!/usr/bin/env bash
# RViz2 시각화(경로/클러스터/힘벡터/지형). fixed frame: tank_map.
#   사용법: scripts/run_rviz.sh
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source /opt/ros/humble/setup.bash
source install/setup.bash 2>/dev/null || { echo "[run_rviz] install/ 없음 — colcon build 먼저"; exit 1; }
echo "[run_rviz] tank_rviz.launch.py"
exec ros2 launch rviz_visualization tank_rviz.launch.py "$@"
