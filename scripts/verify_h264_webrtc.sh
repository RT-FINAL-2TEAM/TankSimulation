#!/usr/bin/env bash
set -euo pipefail

BASE="${1:-http://127.0.0.1:5000}"
echo "--- live view ---"
curl -fsS "$BASE/debug/live_view" | python3 -m json.tool || true
echo "--- WebRTC ---"
curl -fsS "$BASE/debug/webrtc" | python3 -m json.tool || true
echo
echo "Open $BASE/view, then run this again. Check lastConnectionState and the negotiated codec in the browser feed status."
