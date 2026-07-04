#!/usr/bin/env bash
set -euo pipefail

PROJECT="${1:-$HOME/tankcc}"
ENV_FILE="$PROJECT/.env"
STAMP="$(date +%Y%m%d_%H%M%S)"

if [ -f "$ENV_FILE" ]; then
  cp "$ENV_FILE" "$ENV_FILE.backup_h264_webrtc_$STAMP"
else
  : > "$ENV_FILE"
fi

for key in \
  TANK_WEB_STREAM_MODE TANK_WEBRTC_ENABLED TANK_WEBRTC_FPS \
  TANK_WEBRTC_MAX_SIDE TANK_WEBRTC_SYNC_TO_YOLO TANK_WEBRTC_PREFER_H264 \
  TANK_WEBRTC_OFFER_TIMEOUT_SEC TANK_LIVE_VIEW_RAW_JPEG \
  TANK_LIVE_VIEW_BROWSER_OVERLAY TANK_YOLO_ASYNC; do
  sed -i "/^${key}=/d" "$ENV_FILE"
done

cat >> "$ENV_FILE" <<'EOF'

# Browser camera preview: WebRTC with H.264 first in codec negotiation.
TANK_WEB_STREAM_MODE=webrtc
TANK_WEBRTC_ENABLED=true
TANK_WEBRTC_FPS=10
TANK_WEBRTC_MAX_SIDE=640
TANK_WEBRTC_SYNC_TO_YOLO=true
TANK_WEBRTC_PREFER_H264=true
TANK_WEBRTC_OFFER_TIMEOUT_SEC=12
TANK_LIVE_VIEW_RAW_JPEG=true
TANK_LIVE_VIEW_BROWSER_OVERLAY=true
# Keep the video frame and YOLO overlay result paired.
TANK_YOLO_ASYNC=false
EOF

echo "H.264-preferred WebRTC mode written to $ENV_FILE"
echo "Restart ros_bridge after this command."
