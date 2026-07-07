# -*- coding: utf-8 -*-
"""Optional low-latency WebRTC video output for the tank live view.

YOLO inference is not executed here. The module only takes the newest JPEG
already received by ``POST /detect`` and publishes it as a WebRTC video track.
The default mode waits until synchronous YOLO has completed for that frame so
browser video and detection metadata refer to the same image.

The implementation is optional: if aiortc / PyAV are not installed, the Flask
bridge continues to run and the browser automatically falls back to MJPEG.
"""

from __future__ import annotations

import asyncio
from collections import deque
from fractions import Fraction
import os
import threading
import time
from typing import Any, Dict, Optional, Set

import numpy as np
from . import live_view

try:
    import av
    from aiortc import RTCConfiguration, RTCPeerConnection, RTCRtpSender, RTCSessionDescription, VideoStreamTrack
except Exception as exc:  # pragma: no cover
    av = None
    RTCConfiguration = None
    RTCPeerConnection = None
    RTCRtpSender = None
    RTCSessionDescription = None
    VideoStreamTrack = object
    _IMPORT_ERROR: Optional[Exception] = exc
else:
    _IMPORT_ERROR = None

try:
    import cv2
except Exception as exc:  # pragma: no cover
    cv2 = None
    if _IMPORT_ERROR is None:
        _IMPORT_ERROR = exc

_ENABLED = os.getenv("TANK_WEBRTC_ENABLED", "false").strip().lower() in {"1", "true", "yes", "y", "on"}
_FPS = max(1.0, float(os.getenv("TANK_WEBRTC_FPS", "10")))
_MAX_SIDE = max(0, int(os.getenv("TANK_WEBRTC_MAX_SIDE", "640")))
_SYNC_TO_YOLO = os.getenv("TANK_WEBRTC_SYNC_TO_YOLO", "true").strip().lower() in {"1", "true", "yes", "y", "on"}
_PREFER_H264 = os.getenv("TANK_WEBRTC_PREFER_H264", "true").strip().lower() in {"1", "true", "yes", "y", "on"}
_OFFER_TIMEOUT_SEC = max(2.0, float(os.getenv("TANK_WEBRTC_OFFER_TIMEOUT_SEC", "12")))

_loop: Optional[asyncio.AbstractEventLoop] = None
_loop_thread: Optional[threading.Thread] = None
_loop_lock = threading.Lock()
_peers: Set[Any] = set()
_state_lock = threading.Lock()
_frame_times = deque(maxlen=300)

_state: Dict[str, Any] = {
    "offerCount": 0,
    "connectedCount": 0,
    "lastOfferWall": None,
    "lastConnectionState": "idle",
    "lastFrameSeq": 0,
    "lastFrameWall": None,
    "lastDecodeMs": None,
    "lastError": None,
    "codecPreference": "H264" if _PREFER_H264 else "default",
}


def is_available() -> bool:
    return bool(_ENABLED and _IMPORT_ERROR is None and RTCPeerConnection is not None and av is not None and cv2 is not None)


def _set_state(**kwargs: Any) -> None:
    with _state_lock:
        _state.update(kwargs)


def debug_state() -> Dict[str, Any]:
    now = time.time()
    with _state_lock:
        payload = dict(_state)
        recent = [stamp for stamp in _frame_times if now - stamp <= 5.0]
    if len(recent) >= 2:
        payload["sourceFrameFps5s"] = round((len(recent) - 1) / max(1e-6, recent[-1] - recent[0]), 3)
    else:
        payload["sourceFrameFps5s"] = 0.0
    last_wall = payload.get("lastFrameWall")
    payload["lastFrameAgeMs"] = None if not last_wall else round((now - float(last_wall)) * 1000.0, 3)
    payload.update({
        "enabled": _ENABLED,
        "available": is_available(),
        "importError": None if _IMPORT_ERROR is None else str(_IMPORT_ERROR),
        "peerCount": len(_peers),
        "fps": _FPS,
        "maxSide": _MAX_SIDE,
        "syncToYolo": _SYNC_TO_YOLO,
        "preferH264": _PREFER_H264,
    })
    return payload


def _ensure_loop() -> asyncio.AbstractEventLoop:
    global _loop, _loop_thread
    with _loop_lock:
        if _loop is not None and _loop.is_running():
            return _loop
        loop = asyncio.new_event_loop()
        def _runner() -> None:
            asyncio.set_event_loop(loop)
            loop.run_forever()
        thread = threading.Thread(target=_runner, daemon=True, name="tank-webrtc-loop")
        thread.start()
        _loop = loop
        _loop_thread = thread
        return loop


def _resize(frame: np.ndarray) -> np.ndarray:
    if cv2 is None or _MAX_SIDE <= 0:
        return frame
    height, width = frame.shape[:2]
    max_side = max(height, width)
    if max_side <= _MAX_SIDE:
        return frame
    scale = _MAX_SIDE / float(max_side)
    size = (max(2, int(width * scale)), max(2, int(height * scale)))
    size = (max(2, size[0] - size[0] % 2), max(2, size[1] - size[1] % 2))
    return cv2.resize(frame, size, interpolation=cv2.INTER_AREA)


def _blank_frame() -> np.ndarray:
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    if cv2 is not None:
        cv2.putText(frame, "Waiting for synchronous YOLO frame...", (28, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


class LatestTankVideoTrack(VideoStreamTrack):
    """Latest-frame-only video track. It never stores a historical frame queue."""
    kind = "video"

    def __init__(self) -> None:
        super().__init__()
        self._interval = 1.0 / _FPS
        self._next_at = time.monotonic()
        self._clock_start = time.monotonic()
        self._last_seq = -1
        self._last_frame: Optional[np.ndarray] = None

    async def recv(self):  # type: ignore[override]
        # Cap the maximum send rate, but do not repeatedly encode an unchanged
        # source image. recv() may block until the next latest frame arrives.
        now = time.monotonic()
        if now < self._next_at:
            await asyncio.sleep(self._next_at - now)

        deadline = time.monotonic() + 1.0
        snapshot = live_view.get_stream_snapshot(sync_to_yolo=_SYNC_TO_YOLO)
        seq = int(snapshot.get("frameSeq") or 0)
        while seq == self._last_seq and time.monotonic() < deadline:
            await asyncio.sleep(min(0.01, self._interval))
            snapshot = live_view.get_stream_snapshot(sync_to_yolo=_SYNC_TO_YOLO)
            seq = int(snapshot.get("frameSeq") or 0)

        jpeg = snapshot.get("jpeg")
        if jpeg and seq != self._last_seq:
            started = time.perf_counter()
            decoded = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR) if cv2 is not None else None
            decode_ms = (time.perf_counter() - started) * 1000.0
            if decoded is not None:
                self._last_frame = _resize(decoded)
                self._last_seq = seq
                frame_wall = time.time()
                with _state_lock:
                    _frame_times.append(frame_wall)
                    _state["lastFrameSeq"] = seq
                    _state["lastFrameWall"] = frame_wall
                    _state["lastDecodeMs"] = decode_ms
                    _state["lastError"] = None
            else:
                _set_state(lastError="WebRTC JPEG decode failed")

        self._next_at = time.monotonic() + self._interval
        frame_array = self._last_frame if self._last_frame is not None else _blank_frame()
        video_frame = av.VideoFrame.from_ndarray(frame_array, format="bgr24")
        video_frame.pts = int((time.monotonic() - self._clock_start) * 90000)
        video_frame.time_base = Fraction(1, 90000)
        return video_frame


def _prefer_h264(pc: Any) -> None:
    if not _PREFER_H264 or RTCRtpSender is None:
        return
    codecs = list(RTCRtpSender.getCapabilities("video").codecs)
    h264 = [codec for codec in codecs if str(codec.mimeType).lower() == "video/h264"]
    if not h264:
        _set_state(lastError="aiortc H.264 capability not found; negotiated fallback codec will be used")
        return
    ordered = h264 + [codec for codec in codecs if codec not in h264]
    for transceiver in pc.getTransceivers():
        if transceiver.kind == "video":
            transceiver.setCodecPreferences(ordered)


async def _create_answer_async(offer_sdp: str, offer_type: str) -> Dict[str, str]:
    if not is_available():
        raise RuntimeError(f"WebRTC unavailable: {debug_state().get('importError') or 'disabled'}")
    pc = RTCPeerConnection(configuration=RTCConfiguration(iceServers=[]))
    _peers.add(pc)
    with _state_lock:
        offer_count = int(_state.get("offerCount") or 0) + 1
    _set_state(offerCount=offer_count, lastOfferWall=time.time(), lastConnectionState="new")
    pc.addTransceiver(LatestTankVideoTrack(), direction="sendonly")
    _prefer_h264(pc)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange() -> None:
        state = pc.connectionState
        _set_state(lastConnectionState=state)
        if state == "connected":
            with _state_lock:
                _state["connectedCount"] = int(_state.get("connectedCount") or 0) + 1
        if state in {"failed", "closed", "disconnected"}:
            await pc.close()
            _peers.discard(pc)

    try:
        await pc.setRemoteDescription(RTCSessionDescription(sdp=offer_sdp, type=offer_type))
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        local = pc.localDescription
        return {"sdp": local.sdp, "type": local.type}
    except Exception:
        await pc.close()
        _peers.discard(pc)
        raise


def create_answer(payload: Dict[str, Any]) -> Dict[str, str]:
    if not _ENABLED:
        raise RuntimeError("TANK_WEBRTC_ENABLED=false")
    if not is_available():
        raise RuntimeError(f"WebRTC dependencies unavailable: {debug_state().get('importError')}")
    sdp = str(payload.get("sdp") or "")
    offer_type = str(payload.get("type") or "offer")
    if not sdp:
        raise ValueError("Missing WebRTC offer SDP")
    future = asyncio.run_coroutine_threadsafe(_create_answer_async(sdp, offer_type), _ensure_loop())
    try:
        return future.result(timeout=_OFFER_TIMEOUT_SEC)
    except Exception as exc:
        _set_state(lastError=str(exc))
        future.cancel()
        raise
