#!/usr/bin/env bash
set -Eeuo pipefail
printf '\033]0;S1-T3 Auto Pipeline\007'
source "/home/tankcc/tankcc/scripts/.terminator_runtime/common_env.sh"

echo "============================================================"
echo "[T3] Scenario 1 Auto Pipeline"
echo "============================================================"
echo "log dir          : /home/tankcc/tankcc/logs/scenario1_terminator_20260713_175510"
echo "postprocess only : false"
echo "keep old output  : false"
echo

run_step() {
  local name="$1"; shift
  echo
  echo "============================================================"
  echo "[RUN] $name"
  echo "============================================================"
  echo "Command: $*"
  echo
  "$@" 2>&1 | tee "/home/tankcc/tankcc/logs/scenario1_terminator_20260713_175510/$name.log"
  local code="${PIPESTATUS[0]}"
  if [[ "$code" -ne 0 ]]; then
    echo
    echo "[ERROR] $name failed with exit code $code"
    echo "로그: /home/tankcc/tankcc/logs/scenario1_terminator_20260713_175510/$name.log"
    exec bash
  fi
  echo "[OK] $name completed"
}

run_optional_step() {
  local name="$1"; shift
  echo
  echo "============================================================"
  echo "[RUN/OPTIONAL] $name"
  echo "============================================================"
  echo "Command: $*"
  echo
  set +e
  "$@" 2>&1 | tee "/home/tankcc/tankcc/logs/scenario1_terminator_20260713_175510/$name.log"
  local code="${PIPESTATUS[0]}"
  set -e
  if [[ "$code" -ne 0 ]]; then
    echo "[WARN] optional step failed: $name (exit=$code). 계속 진행합니다."
    echo "       로그: /home/tankcc/tankcc/logs/scenario1_terminator_20260713_175510/$name.log"
  else
    echo "[OK] $name completed"
  fi
}

wait_for_bridge() {
  echo "[WAIT] ros_bridge health: http://127.0.0.1:5000/health"
  for _ in $(seq 1 60); do
    if command -v curl >/dev/null 2>&1 && curl -fsS "http://127.0.0.1:5000/health" >"/home/tankcc/tankcc/logs/scenario1_terminator_20260713_175510/bridge_health.json" 2>/dev/null; then
      echo "[OK] bridge health: $(cat "/home/tankcc/tankcc/logs/scenario1_terminator_20260713_175510/bridge_health.json")"
      return 0
    fi
    sleep 0.5
  done
  echo "[WARN] bridge health 확인 실패. 그래도 계속 진행합니다."
}

request_simulator_reset() {
  if [[ "false" == "true" ]]; then
    echo "[SKIP] simulator reset skipped"
    return 0
  fi
  echo "[RESET] simulator restart/reset request -> /tank/episode/control reset"
  for i in 1 2 3; do
    timeout 6s ros2 topic pub --once /tank/episode/control std_msgs/msg/String "{data: 'reset'}"       2>&1 | tee "/home/tankcc/tankcc/logs/scenario1_terminator_20260713_175510/reset_attempt_${i}.log" || true
    sleep 0.5
  done
}

cleanup_old_outputs() {
  if [[ "false" == "true" ]] || [[ "false" == "true" ]]; then
    echo "[KEEP] 기존 산출물 유지"
    return 0
  fi

  echo "[CLEAN] 시나리오1/시나리오2 파생 산출물 및 latest runtime 파일 정리"
  mkdir -p recon_reports/analysis recon_reports/recon_map recon_reports/terrain_maps recon_reports/scenario2            tank_discovered_maps tank_terrain_maps

  rm -f     recon_reports/route_A.json     recon_reports/route_B.json     recon_reports/comparison.json     recon_reports/route_comparison.json     recon_reports/risk_features.json     recon_reports/risk_comparison.json     recon_reports/risk_comparison.md     recon_reports/route_risk_result.json     recon_reports/route_analysis_report.txt     recon_reports/mission_plan.json     recon_reports/analysis/recon_report.md     recon_reports/analysis/mission_plan.md     recon_reports/recon_map/discovered_objects_route_A.map     recon_reports/recon_map/discovered_objects_route_B.map     recon_reports/recon_map/scenario2_map.map     recon_reports/recon_map/scenario2_terrain.json     recon_reports/recon_map/scenario2_terrain.npz     recon_reports/terrain_maps/terrain_map_route_A.npz     recon_reports/terrain_maps/terrain_map_route_B.npz     recon_reports/scenario2/route_A.json     recon_reports/scenario2/scenario2_result.json     tank_discovered_maps/discovered_objects_latest.map     tank_terrain_maps/terrain_map_latest.npz
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
files = [Path("recon_reports/terrain_maps/terrain_map_route_A.npz"), Path("recon_reports/terrain_maps/terrain_map_route_B.npz")]
ok = True
for path in files:
    print(f"[CHECK] {path}")
    if not path.exists():
        print("  ERROR: missing"); ok = False; continue
    size = path.stat().st_size
    print(f"  size: {size} bytes")
    if size <= 0:
        print("  ERROR: 0 byte"); ok = False; continue
    try:
        d = np.load(path, allow_pickle=True)
        print(f"  keys: {d.files}")
        for key in ("accumulated", "ground", "non_ground"):
            if key in d.files:
                arr = d[key]
                print(f"  {key}: {arr.shape}")
                if arr.ndim != 2 or arr.shape[1] != 3 or arr.shape[0] == 0:
                    print(f"  ERROR: invalid {key} shape"); ok = False
    except Exception as e:
        print(f"  ERROR: invalid npz: {e}"); ok = False
if not ok:
    sys.exit(1)
PY
}

verify_scenario2_outputs() {
  echo
  echo "============================================================"
  echo "[VERIFY] scenario2 map/terrain outputs"
  echo "============================================================"
  python3 - <<'PY'
import json, sys
from pathlib import Path
import numpy as np
map_file = Path("recon_reports/recon_map/scenario2_map.map")
terrain_file = Path("recon_reports/recon_map/scenario2_terrain.npz")
terrain_json = Path("recon_reports/recon_map/scenario2_terrain.json")
ok = True
for path in [map_file, terrain_file, terrain_json]:
    print(f"[CHECK] {path}")
    if not path.exists():
        print("  ERROR: missing"); ok = False
    elif path.stat().st_size <= 0:
        print("  ERROR: 0 byte"); ok = False
    else:
        print(f"  size: {path.stat().st_size} bytes")
if map_file.exists() and map_file.stat().st_size > 0:
    try:
        data = json.loads(map_file.read_text(encoding="utf-8"))
        print("  object_count:", data.get("object_count"), "target_count:", data.get("target_count"))
        src = data.get("source_files", {}) or {}
        print("  discovered_maps:", [Path(p).name for p in src.get("discovered_maps", [])])
        if len(src.get("discovered_maps", [])) < 2:
            print("  WARN: route A/B discovered maps 중 일부가 빠져 있습니다.")
    except Exception as e:
        print("  ERROR: invalid map json:", e); ok = False
if terrain_file.exists() and terrain_file.stat().st_size > 0:
    try:
        d = np.load(terrain_file, allow_pickle=True)
        print("  keys:", d.files)
        metadata = str(d["metadata_json"]) if "metadata_json" in d.files else ""
        print("  metadata:", metadata)
        if "terrain_map_route_A.npz" not in metadata or "terrain_map_route_B.npz" not in metadata:
            print("  ERROR: metadata does not include both route_A and route_B"); ok = False
    except Exception as e:
        print(f"  ERROR: invalid scenario2 terrain npz: {e}"); ok = False
if not ok:
    sys.exit(1)
PY
}

wait_for_bridge
cleanup_old_outputs
if [[ "false" != "true" ]]; then
  echo "[WAIT] bridge/terrain recorder 준비 대기 3초"
  sleep 3
  request_simulator_reset
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
# build_scenario2_map을 마지막에 다시 실행했으므로 mission_plan도 최신 scenario2_map 기준으로 다시 생성한다.
run_optional_step "build_mission_plan" python3 scripts/build_mission_plan.py

echo
echo "============================================================"
echo "[DONE] Scenario 1 full pipeline completed"
echo "============================================================"
echo "결과: recon_reports/route_A.json, route_B.json, recon_map/scenario2_map.map, mission_plan.json"
echo "다음 단계: ./scripts/run_scenario2_auto_terminator.sh"
echo "로그: /home/tankcc/tankcc/logs/scenario1_terminator_20260713_175510"
exec bash
