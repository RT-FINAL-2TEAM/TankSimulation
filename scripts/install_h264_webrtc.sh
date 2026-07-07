#!/usr/bin/env bash
set -euo pipefail

PROJECT="${1:-$HOME/tankcc}"
python3 -m pip install --user --upgrade -r "$PROJECT/requirements-webrtc.txt"
python3 - <<'PYCODE'
import av
import aiortc
from aiortc import RTCRtpSender

print("aiortc:", aiortc.__version__)
print("PyAV:", av.__version__)
print("video codecs:", sorted({codec.mimeType for codec in RTCRtpSender.getCapabilities("video").codecs}))
PYCODE
