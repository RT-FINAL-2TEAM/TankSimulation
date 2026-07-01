#!/usr/bin/env bash
set -Eeuo pipefail
printf '\033]0;S2-T4 Debug\007'
source "/home/tankcc/tankcc/scripts/.terminator_runtime/common_env.sh"

echo "============================================================"
echo "[T4] Scenario2 Debug Monitor"
echo "============================================================"
echo "2초마다 핵심 node/topic/file을 표시합니다."
echo "종료: Ctrl+C"
echo
echo "[LOG DIR] /home/tankcc/tankcc/logs/scenario2_terminator_20260701_123248"
echo

sleep 4

watch -n 2 "
echo '[nodes]';
ros2 node list 2>/dev/null | grep -E 'ros_bridge|static_map|rviz_visualizer|terrain|lidar|overlay|dbscan|astar|local_path|controller|potential|turret|decision|scenario' || true;
echo;
echo '[key topics]';
ros2 topic list 2>/dev/null | grep -E '/tank/(api/info/raw|player/pose|enemy/pose|global_path|path/lookahead_pose|control/command|control/status|mission/goal_pose|turret/status|turret/override|engage/result|decision/status|map/discovered/objects|episode/control)' || true;
echo;
echo '[scenario2 files]';
ls -lh recon_reports/recon_map/scenario2_map.map recon_reports/recon_map/scenario2_terrain.npz recon_reports/scenario2/scenario2_result.json 2>/dev/null || true;
echo;
echo '[recon map inputs]';
find recon_reports/recon_map recon_reports/terrain_maps -maxdepth 1 -type f 2>/dev/null | sort | sed 's#^#  #';
"

exec bash
