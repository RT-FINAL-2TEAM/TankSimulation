#!/usr/bin/env bash
set -Eeuo pipefail
printf '\033]0;S1-T4 phone_sim2real\007'
source "/home/tankcc/tankcc/scripts/.terminator_runtime/common_env.sh"

echo "============================================================"
echo "[T4] phone_sim2real"
echo "============================================================"
echo "Command: ros2 launch phone_sim2real phone_sim2real.launch.py phone_port:=5002"
echo "Log    : /home/tankcc/tankcc/logs/scenario1_terminator_20260713_175510/phone_sim2real.log"
echo
ros2 launch phone_sim2real phone_sim2real.launch.py phone_port:=5002 2>&1 | tee "/home/tankcc/tankcc/logs/scenario1_terminator_20260713_175510/phone_sim2real.log"
echo
echo "[EXIT] phone_sim2real 종료."
exec bash
