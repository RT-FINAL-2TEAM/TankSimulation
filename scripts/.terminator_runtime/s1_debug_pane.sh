#!/usr/bin/env bash
set -Eeuo pipefail
printf '\033]0;S1-T4 Debug\007'
source "/home/tankcc/tankcc/scripts/.terminator_runtime/common_env.sh"

echo "============================================================"
echo "[T4] Debug Monitor"
echo "============================================================"
echo "2초마다 핵심 node/topic/file을 표시합니다."
echo "종료: Ctrl+C"
echo
echo "[LOG DIR] /home/tankcc/tankcc/logs/scenario1_terminator_20260701_122729"
echo
if [[ "false" == "true" ]]; then
  echo "rviz_web URL 예시:"
  echo "  http://127.0.0.1:5055/rviz3d?frame=tank_map&cloud=off&rays=0&vectors=0"
  echo
fi

sleep 4

watch -n 2 "
echo '[nodes]';
ros2 node list 2>/dev/null | grep -E 'ros_bridge|static_map|rviz_visualizer|terrain|lidar|overlay|dbscan|astar|local_path|controller|potential|recon' || true;
echo;
echo '[key topics]';
ros2 topic list 2>/dev/null | grep -E '/tank/(api/info/raw|player/pose|sensor/lidar/detected_points_map|sensor/lidar/terrain_points_map|visual_perception/lidar_clusters|global_path|path/lookahead_pose|control/command|control/status|map/discovered/objects|perception/fused_objects|episode/control)' || true;
echo;
echo '[files]';
ls -lh recon_reports/terrain_maps/terrain_map_route_A.npz recon_reports/terrain_maps/terrain_map_route_B.npz recon_reports/recon_map/scenario2_map.map recon_reports/recon_map/scenario2_terrain.npz 2>/dev/null || true;
echo;
echo '[services]';
ros2 service list 2>/dev/null | grep -E '/tank/(map/discovered/save|map/discovered/clear|terrain/finalize_map|terrain/reset_map)' || true;
"

exec bash
