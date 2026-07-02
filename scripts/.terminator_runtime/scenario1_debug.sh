#!/usr/bin/env bash
cd "${TANK_WS:-$HOME/tankcc}"
printf '\033]0;S1 Debug\007'
source scripts/.terminator_runtime/common_env.sh
echo "[S1/DEBUG] 2초마다 핵심 노드/토픽 확인"
echo "종료: Ctrl+C"
echo
sleep 5
watch -n 2 "
echo '[nodes]';
ros2 node list 2>/dev/null | grep -E 'ros_bridge|lidar|dbscan|astar|local_path|controller|terrain|rviz|potential' || true;
echo;
echo '[key topics]';
ros2 topic list 2>/dev/null | grep -E '/tank/(api/info/raw|player/pose|sensor/lidar/detected_points_map|visual_perception/lidar_clusters|global_path|path/lookahead_pose|control/command|control/status|map/discovered/objects|perception/fused_objects)' || true;
"
exec bash
