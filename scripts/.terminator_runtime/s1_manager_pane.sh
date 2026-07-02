#!/usr/bin/env bash
set -Eeuo pipefail
printf '\033]0;S1-T3 Auto Pipeline\007'
source "/home/tankcc/tankcc/scripts/.terminator_runtime/common_env.sh"

echo "============================================================"
echo "[T3] Scenario 1 Auto Pipeline v2"
echo "============================================================"
echo "순서:"
echo "  1) bridge health 확인"
echo "  2) 기존 산출물 정리"
echo "  3) 3초 대기"
echo "  4) simulator reset/restart 요청"
echo "  5) 3초 대기"
echo "  6) python3 scripts/run_recon_scenario.py"
echo "  7) terrain_map_route_A/B.npz 유효성 검증"
echo "  8) python3 scripts/analyze_run.py"
echo "  9) python3 scripts/build_scenario2_map.py"
echo " 10) scenario2_map.map / scenario2_terrain.npz 검증"
echo
echo "[LOG DIR] /home/tankcc/tankcc/logs/scenario1_terminator_20260701_122729"
echo "[POSTPROCESS_ONLY] false"
echo "[KEEP_OLD_OUTPUT]  false"
echo

run_step() {
  local name="$1"
  shift

  echo
  echo "============================================================"
  echo "[RUN] $name"
  echo "============================================================"
  echo "Command: $*"
  echo

  "$@" 2>&1 | tee "/home/tankcc/tankcc/logs/scenario1_terminator_20260701_122729/$name.log"
  local code="${PIPESTATUS[0]}"

  if [[ "$code" -ne 0 ]]; then
    echo
    echo "[ERROR] $name failed with exit code $code"
    echo "로그: /home/tankcc/tankcc/logs/scenario1_terminator_20260701_122729/$name.log"
    exec bash
  fi

  echo
  echo "[OK] $name completed"
}

wait_for_bridge() {
  echo "[WAIT] ros_bridge health: http://127.0.0.1:5000/health"

  for _ in $(seq 1 60); do
    if command -v curl >/dev/null 2>&1; then
      if curl -fsS "http://127.0.0.1:5000/health" >"/home/tankcc/tankcc/logs/scenario1_terminator_20260701_122729/bridge_health.json" 2>/dev/null; then
        echo "[OK] bridge health: $(cat "/home/tankcc/tankcc/logs/scenario1_terminator_20260701_122729/bridge_health.json")"
        return 0
      fi
    else
      if ros2 topic list 2>/dev/null | grep -q "/tank"; then
        echo "[OK] ROS /tank topic detected"
        return 0
      fi
    fi
    sleep 0.5
  done

  echo "[WARN] bridge health 확인 실패. 그래도 계속 진행합니다."
  echo "       bridge pane 로그를 확인하세요."
  return 0
}

request_simulator_reset() {
  if [[ "false" == "true" ]]; then
    echo "[SKIP] simulator reset skipped"
    return 0
  fi

  echo "[RESET] simulator restart/reset request"
  echo "        topic: /tank/episode/control"
  echo "        data : reset"

  for i in 1 2 3; do
    echo "        publish reset attempt $i/3"
    timeout 6s ros2 topic pub --once /tank/episode/control std_msgs/msg/String "{data: 'reset'}" \
      2>&1 | tee "/home/tankcc/tankcc/logs/scenario1_terminator_20260701_122729/reset_attempt_${i}.log" || true
    sleep 0.5
  done
}

cleanup_old_outputs() {
  if [[ "false" == "true" ]] || [[ "false" == "true" ]]; then
    echo "[KEEP] 기존 산출물 유지"
    return 0
  fi

  echo "[CLEAN] 기존 route/scenario2 산출물 정리"
  rm -f recon_reports/terrain_maps/terrain_map_route_A.npz
  rm -f recon_reports/terrain_maps/terrain_map_route_B.npz
  rm -f recon_reports/recon_map/scenario2_terrain.npz
  rm -f recon_reports/recon_map/scenario2_terrain.json
  rm -f recon_reports/recon_map/scenario2_map.map
  rm -f tank_terrain_maps/terrain_map_latest.npz
  mkdir -p recon_reports/terrain_maps recon_reports/recon_map
}

verify_route_terrain() {
  echo
  echo "============================================================"
  echo "[VERIFY] route terrain npz"
  echo "============================================================"

  python3 - <<'PY'
import sys
from pathlib import Path
import numpy as np

files = [
    Path("recon_reports/terrain_maps/terrain_map_route_A.npz"),
    Path("recon_reports/terrain_maps/terrain_map_route_B.npz"),
]

ok = True
for path in files:
    print(f"[CHECK] {path}")
    if not path.exists():
        print("  ERROR: missing")
        ok = False
        continue
    size = path.stat().st_size
    print(f"  size: {size} bytes")
    if size <= 0:
        print("  ERROR: 0 byte")
        ok = False
        continue
    try:
        d = np.load(path, allow_pickle=True)
        print(f"  keys: {d.files}")
        for key in ("accumulated", "ground", "non_ground"):
            if key in d.files:
                arr = d[key]
                print(f"  {key}: {arr.shape}")
                if arr.ndim != 2 or arr.shape[1] != 3 or arr.shape[0] == 0:
                    print(f"  ERROR: invalid {key} shape")
                    ok = False
    except Exception as e:
        print(f"  ERROR: invalid npz: {e}")
        ok = False

if not ok:
    sys.exit(1)
PY

  local code=$?
  if [[ "$code" -ne 0 ]]; then
    echo
    echo "[ERROR] route terrain 검증 실패"
    echo "        A/B 지형 파일이 없거나 깨져 있습니다."
    echo "        시나리오1 정찰을 다시 확인하세요."
    exec bash
  fi

  echo "[OK] route terrain A/B valid"
}

verify_scenario2_outputs() {
  echo
  echo "============================================================"
  echo "[VERIFY] scenario2 outputs"
  echo "============================================================"

  python3 - <<'PY'
import sys
from pathlib import Path
import numpy as np

map_file = Path("recon_reports/recon_map/scenario2_map.map")
terrain_file = Path("recon_reports/recon_map/scenario2_terrain.npz")
terrain_json = Path("recon_reports/recon_map/scenario2_terrain.json")

ok = True

for path in [map_file, terrain_file, terrain_json]:
    print(f"[CHECK] {path}")
    if not path.exists():
        print("  ERROR: missing")
        ok = False
    else:
        size = path.stat().st_size
        print(f"  size: {size} bytes")
        if size <= 0:
            print("  ERROR: 0 byte")
            ok = False

if terrain_file.exists() and terrain_file.stat().st_size > 0:
    try:
        d = np.load(terrain_file, allow_pickle=True)
        print("  keys:", d.files)
        for key in ("accumulated", "ground", "non_ground"):
            if key in d.files:
                print(f"  {key}: {d[key].shape}")
        metadata = str(d["metadata_json"]) if "metadata_json" in d.files else ""
        print("  metadata:", metadata)
        if "terrain_map_route_A.npz" not in metadata or "terrain_map_route_B.npz" not in metadata:
            print("  ERROR: metadata does not include both route_A and route_B")
            ok = False
    except Exception as e:
        print(f"  ERROR: invalid scenario2 terrain npz: {e}")
        ok = False

if not ok:
    sys.exit(1)
PY

  local code=$?
  if [[ "$code" -ne 0 ]]; then
    echo
    echo "[ERROR] scenario2 output 검증 실패"
    echo "        build_scenario2_map.py 로그를 확인하세요:"
    echo "        /home/tankcc/tankcc/logs/scenario1_terminator_20260701_122729/build_scenario2_map.log"
    exec bash
  fi

  echo "[OK] scenario2_map.map / scenario2_terrain.npz valid"
}

wait_for_bridge
cleanup_old_outputs

if [[ "false" != "true" ]]; then
  echo
  echo "[WAIT] bridge/RViz 실행 후 3초 대기"
  sleep 3

  request_simulator_reset

  echo
  echo "[WAIT] reset 후 3초 대기"
  sleep 3

  run_step "run_recon_scenario" python3 scripts/run_recon_scenario.py
else
  echo "[SKIP] run_recon_scenario.py (--postprocess-only)"
fi

verify_route_terrain
run_step "analyze_run" python3 scripts/analyze_run.py
run_step "build_scenario2_map" python3 scripts/build_scenario2_map.py
verify_scenario2_outputs

echo
echo "============================================================"
echo "[DONE] Scenario 1 full pipeline completed"
echo "============================================================"
echo "결과 확인:"
echo "  recon_reports/route_A.json"
echo "  recon_reports/route_B.json"
echo "  recon_reports/analysis/run_diagnosis.md"
echo "  recon_reports/recon_map/scenario2_map.map"
echo "  recon_reports/recon_map/scenario2_terrain.npz"
echo
echo "다음 단계:"
echo "  ./scripts/run_scenario2_auto_terminator.sh"
echo
echo "로그:"
echo "  /home/tankcc/tankcc/logs/scenario1_terminator_20260701_122729"
echo
echo "이 창은 확인용으로 유지됩니다. 닫아도 됩니다."
exec bash
