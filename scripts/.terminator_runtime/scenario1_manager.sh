#!/usr/bin/env bash
cd "/home/tankcc/tankcc"
printf '\033]0;S1 T3 Recon Manager\007'
source "/home/tankcc/tankcc/scripts/.terminator_runtime/common_env.sh"
echo "[S1/T3] recon scenario manager"
echo "bridge 초기화 대기 3초..."
sleep 3
echo "Command:"
echo "  python3 scripts/run_recon_scenario.py"
echo
python3 scripts/run_recon_scenario.py
echo
echo "[EXIT] scenario1 manager finished. Press Enter or close pane."
exec bash
