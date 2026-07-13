#!/usr/bin/env bash
set -Eeuo pipefail

# 사용법:
# ./scripts/monitor_performance.sh <결과 디렉터리>
#
# 예:
# ./scripts/monitor_performance.sh logs/perf_test_01

OUTPUT_DIR="${1:-logs/performance_$(date +%Y%m%d_%H%M%S)}"
INTERVAL="${PERF_INTERVAL:-1}"

mkdir -p "$OUTPUT_DIR"

echo "Performance monitoring started"
echo "Output directory: $OUTPUT_DIR"
echo "Interval: ${INTERVAL}s"

# 실험 환경 저장
{
    echo "date=$(date --iso-8601=seconds)"
    echo "hostname=$(hostname)"
    echo "kernel=$(uname -a)"
    echo "cpu=$(lscpu | grep 'Model name' | head -n 1)"
    echo "memory=$(free -h | grep Mem)"
    echo "ros_domain_id=${ROS_DOMAIN_ID:-not_set}"
} > "$OUTPUT_DIR/environment.txt"

# 현재 실행 중인 ROS 노드 목록
ros2 node list > "$OUTPUT_DIR/ros_nodes_start.txt" 2>&1 || true

# 전체 CPU 사용률
mpstat "$INTERVAL" > "$OUTPUT_DIR/cpu_total.log" 2>&1 &
PID_MPSTAT=$!

# CPU 코어별 사용률
mpstat -P ALL "$INTERVAL" > "$OUTPUT_DIR/cpu_per_core.log" 2>&1 &
PID_MPSTAT_CORE=$!

# 프로세스별 CPU, 메모리, 디스크 I/O
pidstat -u -r -d -h "$INTERVAL" > "$OUTPUT_DIR/process_usage.log" 2>&1 &
PID_PIDSTAT=$!

# 메모리 상태
(
    while true; do
        echo "timestamp=$(date +%s.%N)"
        free -b
        echo
        sleep "$INTERVAL"
    done
) > "$OUTPUT_DIR/memory.log" 2>&1 &
PID_MEMORY=$!

# 네트워크 인터페이스별 송수신량
sar -n DEV "$INTERVAL" > "$OUTPUT_DIR/network.log" 2>&1 &
PID_NETWORK=$!

# ROS 프로세스 스냅샷
(
    while true; do
        echo "===== $(date --iso-8601=seconds) ====="
        ps -eo pid,ppid,pcpu,pmem,rss,vsz,etimes,cmd \
            --sort=-pcpu |
        grep -E \
            'ros_bridge|lidar_processor|lidar_camera_overlay|lidar_dbscan|map_astar|local_path|tank_controller|ballistic_turret|phone_sim2real|rviz2|yolo' |
        grep -v grep || true

        sleep "$INTERVAL"
    done
) > "$OUTPUT_DIR/ros_processes.log" 2>&1 &
PID_ROS_PS=$!

# NVIDIA GPU가 있는 컴퓨터에서만 기록
PID_GPU=""
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi \
        --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu \
        --format=csv \
        --loop="$INTERVAL" \
        > "$OUTPUT_DIR/gpu.log" 2>&1 &
    PID_GPU=$!
fi

cleanup() {
    echo
    echo "Stopping performance monitoring..."

    kill "$PID_MPSTAT" 2>/dev/null || true
    kill "$PID_MPSTAT_CORE" 2>/dev/null || true
    kill "$PID_PIDSTAT" 2>/dev/null || true
    kill "$PID_MEMORY" 2>/dev/null || true
    kill "$PID_NETWORK" 2>/dev/null || true
    kill "$PID_ROS_PS" 2>/dev/null || true

    if [[ -n "$PID_GPU" ]]; then
        kill "$PID_GPU" 2>/dev/null || true
    fi

    ros2 node list > "$OUTPUT_DIR/ros_nodes_end.txt" 2>&1 || true

    echo "Logs saved to: $OUTPUT_DIR"
}

trap cleanup EXIT INT TERM

while true; do
    sleep 1
done