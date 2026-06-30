# -*- coding: utf-8 -*-
"""Lightweight live camera view for ros_bridge.

This module does not run YOLO. It only displays the latest /detect frame and the
latest detection list already produced by the bridge/vision path.
"""

from __future__ import annotations

import time
import os
from copy import deepcopy
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from flask import Response, render_template_string

try:
    import cv2
except Exception:  # pragma: no cover - runtime optional guard
    cv2 = None

_state_lock = Lock()
_latest_frame: Optional[np.ndarray] = None
_latest_frame_seq = 0
_latest_frame_timestamp = 0.0
_latest_frame_shape: Optional[List[int]] = None
_latest_source_frame_shape: Optional[List[int]] = None
_latest_detections: List[Dict[str, Any]] = []
_latest_detection_metadata: Dict[str, Any] = {}
_latest_detection_timestamp = 0.0
_latest_error: Optional[str] = None
_latest_live_decode_ms = 0.0
_skipped_live_decode_count = 0

_LIVE_VIEW_DECODE_FPS = float(os.getenv("TANK_LIVE_VIEW_DECODE_FPS", "6"))
_LIVE_VIEW_DECODE_INTERVAL = 1.0 / max(0.1, _LIVE_VIEW_DECODE_FPS)
_LIVE_VIEW_MAX_SIDE = int(os.getenv("TANK_LIVE_VIEW_MAX_SIDE", "960"))
_LIVE_VIEW_BROWSER_OVERLAY = os.getenv("TANK_LIVE_VIEW_BROWSER_OVERLAY", "true").strip().lower() in ("1", "true", "yes", "y")

_CLASS_COLORS_BGR = {
    "tank": (0, 0, 255),
    "rock": (0, 255, 255),
    "person": (136, 255, 57),
    "car": (0, 140, 255),
    "unknown": (255, 255, 255),
}
_COLOR_PALETTE_BGR = [
    (0, 255, 0),
    (0, 0, 255),
    (255, 0, 0),
    (0, 255, 255),
    (255, 0, 255),
    (255, 255, 0),
    (255, 255, 255),
]


def _decode_jpeg(image_bytes: bytes) -> Optional[np.ndarray]:
    if cv2 is None or not image_bytes:
        return None
    buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    return cv2.imdecode(buffer, cv2.IMREAD_COLOR)


def _resize_for_live_view(frame: np.ndarray) -> np.ndarray:
    if cv2 is None or _LIVE_VIEW_MAX_SIDE <= 0:
        return frame
    height, width = frame.shape[:2]
    max_side = max(height, width)
    if max_side <= _LIVE_VIEW_MAX_SIDE:
        return frame
    scale = _LIVE_VIEW_MAX_SIDE / float(max_side)
    resized_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return cv2.resize(frame, resized_size, interpolation=cv2.INTER_AREA)


def update_frame(image_bytes: bytes) -> Optional[List[int]]:
    """Store a throttled display frame. Returns source frame shape [h, w, c] if known."""
    global _latest_frame, _latest_frame_seq, _latest_frame_timestamp, _latest_frame_shape, _latest_source_frame_shape, _latest_error, _latest_live_decode_ms, _skipped_live_decode_count
    now = time.time()
    with _state_lock:
        if _latest_frame_timestamp and now - _latest_frame_timestamp < _LIVE_VIEW_DECODE_INTERVAL:
            _skipped_live_decode_count += 1
            return deepcopy(_latest_source_frame_shape or _latest_frame_shape)
    decode_started = time.perf_counter()
    frame = _decode_jpeg(image_bytes)
    decode_ms = (time.perf_counter() - decode_started) * 1000.0
    if frame is None:
        with _state_lock:
            _latest_error = "live_view: failed to decode frame or cv2 unavailable"
        return None
    source_shape = [int(v) for v in frame.shape]
    display_frame = _resize_for_live_view(frame)
    display_shape = [int(v) for v in display_frame.shape]
    with _state_lock:
        _latest_frame = display_frame
        _latest_frame_seq += 1
        _latest_frame_timestamp = time.time()
        _latest_frame_shape = display_shape
        _latest_source_frame_shape = source_shape
        _latest_live_decode_ms = decode_ms
        _latest_error = None
    return source_shape


def update_detections(detections: Any, metadata: Optional[Dict[str, Any]] = None) -> None:
    """Store latest detection list for overlay."""
    global _latest_detections, _latest_detection_metadata, _latest_detection_timestamp
    with _state_lock:
        _latest_detections = deepcopy(detections) if isinstance(detections, list) else []
        _latest_detection_metadata = deepcopy(metadata) if isinstance(metadata, dict) else {}
        _latest_detection_timestamp = time.time()


def _class_color(class_name: str, class_id: int = 0) -> Tuple[int, int, int]:
    key = str(class_name).strip().lower()
    if key in _CLASS_COLORS_BGR:
        return _CLASS_COLORS_BGR[key]
    return _COLOR_PALETTE_BGR[int(class_id) % len(_COLOR_PALETTE_BGR)]


def _blend_rect(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int, color: Tuple[int, int, int], alpha: float) -> None:
    height, width = frame.shape[:2]
    left = max(0, min(width, x1))
    right = max(0, min(width, x2))
    top = max(0, min(height, y1))
    bottom = max(0, min(height, y2))
    if right <= left or bottom <= top:
        return
    roi = frame[top:bottom, left:right]
    fill = np.full_like(roi, color, dtype=np.uint8)
    cv2.addWeighted(fill, alpha, roi, 1.0 - alpha, 0, roi)


def _draw_refined_box(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int, color: Tuple[int, int, int]) -> None:
    height, width = frame.shape[:2]
    x1 = max(0, min(width - 1, x1))
    x2 = max(0, min(width - 1, x2))
    y1 = max(0, min(height - 1, y1))
    y2 = max(0, min(height - 1, y2))
    if x2 <= x1 or y2 <= y1:
        return
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
    corner = max(10, min(24, int(min(x2 - x1, y2 - y1) * 0.22)))
    for start, end in (
        ((x1, y1), (x1 + corner, y1)),
        ((x1, y1), (x1, y1 + corner)),
        ((x2, y1), (x2 - corner, y1)),
        ((x2, y1), (x2, y1 + corner)),
        ((x1, y2), (x1 + corner, y2)),
        ((x1, y2), (x1, y2 - corner)),
        ((x2, y2), (x2 - corner, y2)),
        ((x2, y2), (x2, y2 - corner)),
    ):
        cv2.line(frame, start, end, color, 1, cv2.LINE_AA)


def _draw_refined_label(frame: np.ndarray, label: str, x: int, y: int, color: Tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    thickness = 1
    padding_x = 6
    padding_y = 4
    text_size, baseline = cv2.getTextSize(label, font, font_scale, thickness)
    text_w, text_h = text_size
    height, width = frame.shape[:2]

    label_x = max(4, min(width - text_w - padding_x * 2 - 4, x))
    label_y = y - text_h - baseline - padding_y * 2 - 4
    if label_y < 4:
        label_y = min(height - text_h - baseline - padding_y * 2 - 4, y + 6)
    label_y = max(4, label_y)

    bg_left = label_x
    bg_top = label_y
    bg_right = label_x + text_w + padding_x * 2
    bg_bottom = label_y + text_h + baseline + padding_y * 2
    _blend_rect(frame, bg_left, bg_top, bg_right, bg_bottom, (4, 8, 6), 0.72)
    cv2.rectangle(frame, (bg_left, bg_top), (bg_right, bg_bottom), color, 1, cv2.LINE_AA)
    text_origin = (label_x + padding_x, label_y + padding_y + text_h)
    shadow_origin = (text_origin[0] + 1, text_origin[1] + 1)
    cv2.putText(frame, label, shadow_origin, font, font_scale, (12, 18, 14), thickness, cv2.LINE_AA)
    cv2.putText(frame, label, text_origin, font, font_scale, color, thickness, cv2.LINE_AA)


def _draw_detections(frame: np.ndarray, detections: List[Dict[str, Any]], metadata: Dict[str, Any]) -> np.ndarray:
    if cv2 is None:
        return frame
    drawn = frame.copy()
    for det in detections:
        if not isinstance(det, dict):
            continue
        bbox = det.get("bbox", [])
        if not isinstance(bbox, list) or len(bbox) < 4:
            continue
        try:
            x1, y1, x2, y2 = [int(float(v)) for v in bbox[:4]]
        except Exception:
            continue
        source_shape = metadata.get("image_shape") or metadata.get("latestFrameShape")
        if isinstance(source_shape, list) and len(source_shape) >= 2:
            source_h = max(1.0, float(source_shape[0]))
            source_w = max(1.0, float(source_shape[1]))
            frame_h, frame_w = drawn.shape[:2]
            scale_x = frame_w / source_w
            scale_y = frame_h / source_h
            x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
            y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
        class_name = str(det.get("className", det.get("class_name", "object"))).strip().lower()
        class_id = int(det.get("classId") or 0)
        track_id = det.get("trackId", det.get("track_id"))
        fixed_id = det.get("classFixedId", det.get("id"))
        conf = float(det.get("confidence") or 0.0)
        color = _class_color(class_name, class_id)
        cv2.rectangle(drawn, (x1, y1), (x2, y2), color, 2)
        id_text = ""
        if fixed_id is not None:
            id_text += f" ID:{fixed_id}"
        if track_id is not None:
            id_text += f" T:{track_id}"
        label = f"{class_name}{id_text} {conf:.2f}"
        _draw_refined_box(drawn, x1, y1, x2, y2, color)
        _draw_refined_label(drawn, label, x1, y1, color)
    return drawn


def _blank_frame(message: str = "Waiting for /detect image...") -> np.ndarray:
    frame = np.zeros((480, 854, 3), dtype=np.uint8)
    if cv2 is not None:
        cv2.putText(frame, message, (40, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return frame


def render_view_page(poll_ms: int = 1000) -> str:
    html = """
    <!doctype html>
    <html lang="ko">
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>TANK-CV MFD</title>
        <style>
            :root {
                color-scheme: dark;
                /* 현대로템 지휘통제(C2) 팔레트 — Rotem Blue(#0A2369)·Rotem Red(#FF0000) 기반 다크 네이비 콘솔 */
                --bg: #06101f;
                --panel: #0c1c3a;
                --line: #2a539a;       /* Rotem 블루 라인 */
                --line-dim: #1a3358;
                --text: #e8eefc;
                --muted: #8ea6cf;      /* 블루-그레이 */
                --green: #4f8dff;      /* 주 강조 = Rotem 블루(밝게). 타이틀/보더/활성 */
                --amber: #ffb74d;      /* 경고 */
                --red: #ff4444;        /* Rotem 레드 = 위협/경보/임계 */
                --cyan: #5ac8ff;       /* 정보 */
                --ok: #36d17a;         /* 정상/안전(상태표시 전용) */
            }
            /* 패널 더블클릭 확대 + 여백/폰트 ↑ (C2 가독성) */
            .panel.maximized {
                position: fixed; top: 54px; left: 10px; right: 10px; bottom: 42px; z-index: 60;
                box-shadow: 0 0 0 1px var(--line), 0 14px 44px rgba(0, 0, 0, 0.65);
            }
            .main-grid { gap: 10px; padding: 10px; }
            .panel-title { height: 34px; font-size: 13px; letter-spacing: 0.4px; }
            .scroll { padding: 12px; }
            .tele .value { font-size: 16px; }
            * { box-sizing: border-box; }
            html, body { width: 100%; height: 100%; }
            body {
                margin: 0;
                overflow: hidden;
                background:
                    linear-gradient(90deg, rgba(57,255,136,0.03) 1px, transparent 1px),
                    linear-gradient(rgba(57,255,136,0.025) 1px, transparent 1px),
                    var(--bg);
                background-size: 28px 28px;
                color: var(--text);
                font-family: "Cascadia Mono", "Consolas", "SFMono-Regular", monospace;
                letter-spacing: 0;
            }
            .mfd {
                width: 100vw;
                height: 100vh;
                display: grid;
                grid-template-rows: 48px minmax(0, 1fr) 36px;
            }
            .header, .bottom {
                display: flex;
                align-items: center;
                gap: 14px;
                background: rgba(5, 11, 8, 0.96);
                padding: 0 14px;
                min-width: 0;
            }
            .header { border-bottom: 1px solid var(--line); }
            .bottom {
                border-top: 1px solid var(--line);
                color: var(--muted);
                font-size: 12px;
                white-space: nowrap;
                overflow: hidden;
            }
            .brand {
                color: var(--green);
                font-weight: 800;
                font-size: 18px;
                flex: 0 0 auto;
            }
            .header-metrics {
                display: flex;
                align-items: center;
                justify-content: flex-end;
                gap: 10px;
                min-width: 0;
                flex: 1 1 auto;
            }
            .metric, .bottom span {
                border: 1px solid var(--line-dim);
                background: rgba(9, 22, 15, 0.86);
                padding: 5px 8px;
                min-width: 0;
            }
            .metric strong, .bottom strong {
                color: var(--muted);
                font-weight: 700;
                margin-right: 5px;
            }
            .metric b, .bottom b {
                color: var(--text);
                font-weight: 800;
            }
            .status-ok { color: var(--ok) !important; }
            .status-warn { color: var(--amber) !important; }
            .status-error { color: var(--red) !important; }
            .main-grid {
                min-height: 0;
                display: grid;
                grid-template-columns: 1fr 1fr;
                grid-template-rows: 1fr 1fr;
                gap: 8px;
                padding: 8px;
            }
            .panel {
                min-width: 0;
                min-height: 0;
                background: rgba(6, 15, 10, 0.92);
                border: 1px solid var(--line);
                box-shadow: inset 0 0 0 1px rgba(57,255,136,0.05);
                display: flex;
                flex-direction: column;
            }
            .panel-title {
                height: 32px;
                flex: 0 0 auto;
                display: flex;
                align-items: center;
                justify-content: space-between;
                border-bottom: 1px solid var(--line-dim);
                color: var(--green);
                padding: 0 10px;
                font-size: 12px;
                font-weight: 800;
            }
            .feed-status-text {
                min-width: 0;
                max-width: 72%;
                overflow: hidden;
                text-overflow: ellipsis;
                white-space: nowrap;
                color: var(--cyan);
                font-weight: 800;
            }
            .left-tabs {
                height: 42px;
                display: grid;
                grid-template-columns: repeat(4, minmax(0, 1fr));
                gap: 6px;
                padding: 6px;
                border-bottom: 1px solid var(--line-dim);
            }
            .tab-button {
                min-width: 0;
                border: 1px solid var(--line-dim);
                background: #08130d;
                color: var(--muted);
                font: inherit;
                font-size: 12px;
                font-weight: 800;
                cursor: pointer;
            }
            .tab-button.active {
                border-color: var(--green);
                color: var(--green);
                background: #0d2115;
            }
            .map-tabs {
                height: 38px;
                flex: 0 0 auto;
                display: grid;
                grid-template-columns: repeat(3, minmax(0, 1fr));
                gap: 6px;
                padding: 6px;
                border-bottom: 1px solid var(--line-dim);
                background: rgba(4, 10, 7, 0.72);
            }
            .scroll {
                min-height: 0;
                overflow: auto;
                padding: 10px;
            }
            .feed-wrap, .map-wrap {
                min-height: 0;
                flex: 1 1 auto;
                position: relative;
                background: #020403;
            }
            #driveFeed {
                width: 100%;
                height: 100%;
                object-fit: contain;
                display: block;
                background: #000;
            }
            #driveOverlay {
                position: absolute;
                inset: 0;
                width: 100%;
                height: 100%;
                pointer-events: none;
            }
            #mapCanvas {
                width: 100%;
                height: 100%;
                display: block;
                background: #050806;
            }
            #rvizFrame {
                position: absolute;
                inset: 0;
                width: 100%;
                height: 100%;
                border: 0;
                background: #050806;
            }
            .tele-grid {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 6px;
            }
            .tele {
                border: 1px solid var(--line-dim);
                background: rgba(7, 18, 12, 0.82);
                padding: 6px 8px;
            }
            .tele .label { color: var(--muted); font-size: 11px; }
            .tele .value { color: var(--text); font-size: 15px; font-weight: 800; }
            .bar { height: 8px; background: #0c2417; border-radius: 3px; overflow: hidden; margin-top: 3px; }
            .bar > span { display: block; height: 100%; background: var(--green); }
            .sec-title { font-weight: 800; color: var(--green); margin: 8px 0 5px; font-size: 12px; }
            /* CSS 3D 전차 자세 인디케이터 */
            .tank3d-stage {
                position: relative;
                perspective: 600px;
                height: 210px;
                display: flex;
                align-items: center;
                justify-content: center;
                background: radial-gradient(circle at 50% 72%, #0b1e14, #050806);
                border: 1px solid var(--line-dim);
                margin-bottom: 8px;
                overflow: hidden;
            }
            .tank3d-view { transform-style: preserve-3d; transform: rotateX(-24deg) rotateY(-32deg); }
            .tank3d { position: relative; width: 0; height: 0; transform-style: preserve-3d; transition: transform 0.15s linear; }
            .t3d-grp { position: absolute; left: 0; top: 0; transform-style: preserve-3d; }
            .t3d-face { position: absolute; left: 0; top: 0; box-sizing: border-box; border: 1px solid #2a539a; background: rgba(26, 56, 110, 0.62); }
            .t3d-face.turret { background: rgba(40, 78, 150, 0.85); border-color: #4f8dff; }
            .t3d-face.barrel { background: #4f8dff; border-color: #aaccff; }
            /* 3D 전차 위 HUD 오버레이 */
            .t3d-hud { position: absolute; z-index: 2; pointer-events: none; line-height: 1.25; }
            .t3d-hud .k { color: var(--muted); font-size: 10px; letter-spacing: 0.5px; }
            .t3d-hud .v { color: var(--text); font-weight: 800; font-size: 13px; }
            .t3d-hud .big { font-size: 22px; color: var(--cyan); }
            .t3d-hud.tl { top: 8px; left: 10px; }
            .t3d-hud.tr { top: 8px; right: 10px; text-align: right; }
            .t3d-hud.bl { bottom: 26px; left: 10px; }
            .t3d-hud.br { bottom: 26px; right: 10px; text-align: right; }
            .t3d-hud.bc { left: 0; right: 0; bottom: 6px; text-align: center; font-size: 11px; color: var(--muted); }
            .t3d-hud.bc b { color: var(--cyan); font-weight: 800; }
            .readout-list { display: grid; gap: 8px; }
            .route-compare {
                display: grid;
                gap: 8px;
            }
            .route-decision {
                border: 1px solid var(--line-dim);
                background: rgba(4, 12, 8, 0.84);
                padding: 8px;
                font-size: 11px;
                color: var(--muted);
            }
            .route-decision strong {
                display: block;
                color: var(--green);
                font-size: 12px;
                margin-bottom: 3px;
            }
            .route-card {
                border: 1px solid var(--line-dim);
                background: rgba(7, 18, 12, 0.82);
                padding: 8px;
                min-width: 0;
            }
            .route-card.selected {
                border-color: var(--green);
                box-shadow: inset 0 0 0 1px rgba(57, 255, 136, 0.12);
            }
            .route-head {
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 8px;
                margin-bottom: 6px;
            }
            .route-name {
                min-width: 0;
                color: var(--text);
                font-size: 12px;
                font-weight: 800;
                overflow: hidden;
                text-overflow: ellipsis;
                white-space: nowrap;
            }
            .route-chip {
                flex: 0 0 auto;
                border: 1px solid currentColor;
                padding: 2px 5px;
                font-size: 10px;
                font-weight: 800;
            }
            .route-summary {
                color: var(--muted);
                font-size: 11px;
                line-height: 1.35;
                margin-bottom: 8px;
            }
            .route-factor {
                display: grid;
                grid-template-columns: 46px minmax(0, 1fr) 42px;
                align-items: center;
                gap: 6px;
                min-height: 18px;
                color: var(--muted);
                font-size: 10px;
            }
            .route-meter {
                height: 4px;
                background: rgba(216, 255, 233, 0.12);
                overflow: hidden;
            }
            .route-meter span {
                display: block;
                height: 100%;
                width: var(--score);
                background: var(--factor-color);
            }
            .factor-low { --factor-color: var(--green); }
            .factor-mid { --factor-color: var(--amber); }
            .factor-high { --factor-color: var(--red); }
            .factor-critical { --factor-color: var(--red); }
            .factor-pending { --factor-color: var(--muted); }
            .route-value {
                text-align: right;
                color: var(--text);
                font-weight: 800;
            }
            .readout {
                border-left: 2px solid var(--line);
                background: rgba(14, 30, 20, 0.58);
                padding: 8px;
                min-width: 0;
            }
            .readout .label {
                color: var(--muted);
                font-size: 11px;
                margin-bottom: 3px;
            }
            .readout .value {
                color: var(--text);
                font-size: 12px;
                overflow-wrap: anywhere;
            }
            .recon-actions {
                display: grid;
                grid-template-columns: repeat(2, minmax(0, 1fr));
                gap: 6px;
                margin-bottom: 8px;
            }
            .action-button {
                min-width: 0;
                border: 1px solid var(--line-dim);
                background: rgba(8, 19, 13, 0.92);
                color: var(--muted);
                font: inherit;
                font-size: 11px;
                font-weight: 800;
                padding: 7px 6px;
                cursor: pointer;
            }
            .action-button:hover {
                border-color: var(--green);
                color: var(--green);
                background: rgba(13, 33, 21, 0.92);
            }
            .empty {
                color: var(--muted);
                border: 1px dashed var(--line-dim);
                padding: 10px;
                font-size: 12px;
            }
            .map-legend {
                position: absolute;
                left: 10px;
                bottom: 10px;
                display: flex;
                gap: 8px;
                flex-wrap: wrap;
                color: var(--muted);
                font-size: 11px;
                pointer-events: none;
            }
            .map-legend span {
                background: rgba(3, 8, 5, 0.78);
                border: 1px solid var(--line-dim);
                padding: 3px 6px;
            }
            @media (max-width: 980px) {
                body { overflow: auto; }
                .mfd { min-height: 100vh; height: auto; grid-template-rows: auto auto auto; }
                .header { min-height: 48px; align-items: flex-start; padding: 8px; flex-direction: column; gap: 8px; }
                .header-metrics { width: 100%; justify-content: flex-start; flex-wrap: wrap; }
                .main-grid { grid-template-columns: 1fr; grid-auto-rows: minmax(260px, auto); }
                .center-panel { min-height: 58vh; }
                .right-panel { min-height: 360px; }
                .bottom { min-height: 36px; flex-wrap: wrap; padding: 8px; white-space: normal; }
            }
            /* === 현대로템 C2 정체성: 헤더/타이포/패널 크롬 (오버라이드) === */
            body { font-family: "Pretendard", "Pretendard Variable", "Noto Sans KR", system-ui, sans-serif; }
            .metric b, .bottom span b, .tele .value, .feed-status-text, #timeValue, .t3d-hud .v { font-family: "Cascadia Mono", "Consolas", monospace; }
            .mfd { grid-template-rows: 64px minmax(0, 1fr) 36px; }
            .header { flex-direction: column; gap: 0; padding: 0; align-items: stretch; }
            .header-main { display: flex; align-items: center; gap: 14px; padding: 0 14px; flex: 1 1 auto; min-width: 0; }
            .brand { display: flex; align-items: center; gap: 14px; flex: 0 0 auto; }
            .logo { display: flex; align-items: center; gap: 2px; }
            .logo-h { color: #5a93ff; font-weight: 700; font-size: 12px; letter-spacing: 2.5px; }
            .logo-r { color: #ffffff; font-weight: 800; font-size: 19px; letter-spacing: 0.3px; }
            .logo-accent { display: inline-block; width: 13px; height: 14px; margin-left: 4px; background: var(--red); clip-path: polygon(35% 0, 100% 0, 65% 100%, 0 100%); }
            .sysname { color: var(--text); font-weight: 700; font-size: 15px; padding-left: 14px; border-left: 2px solid var(--red); letter-spacing: 0.5px; }
            .classbar { display: flex; align-items: center; justify-content: space-between; height: 20px; padding: 0 14px; font-size: 10px; letter-spacing: 1.5px; text-transform: uppercase; color: #b9c8e8; background: linear-gradient(90deg, rgba(255, 68, 68, 0.18), rgba(42, 83, 154, 0.14) 55%, transparent); border-top: 1px solid var(--line-dim); }
            .panel { position: relative; }
            .panel-title { border-left: 3px solid var(--green); padding-right: 34px; }
            .panel::before, .panel::after { content: ""; position: absolute; width: 10px; height: 10px; border-color: var(--line); border-style: solid; opacity: 0.5; pointer-events: none; z-index: 3; }
            .panel::before { top: 0; left: 0; border-width: 1px 0 0 1px; }
            .panel::after { bottom: 0; right: 0; border-width: 0 1px 1px 0; }
            .maxbtn { position: absolute; top: 5px; right: 7px; z-index: 6; background: rgba(8, 19, 40, 0.9); border: 1px solid var(--line-dim); color: var(--muted); cursor: pointer; font-size: 13px; line-height: 1; padding: 3px 7px; }
            .maxbtn:hover { color: var(--green); border-color: var(--green); }
            .panel.maximized { top: 68px; }
            .rosbtn { background: #0a1730; color: var(--muted); border: 1px solid var(--line-dim); font: inherit; font-size: 10px; font-weight: 700; padding: 3px 9px; cursor: pointer; letter-spacing: 0.5px; }
            .rosbtn.on { color: var(--green); border-color: var(--green); }
        </style>
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.min.css">
        <script src="https://cdn.jsdelivr.net/npm/cytoscape@3.28.1/dist/cytoscape.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/dagre@0.8.5/dist/dagre.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/cytoscape-dagre@2.5.0/cytoscape-dagre.min.js"></script>
    </head>
    <body>
        <div class="mfd">
            <header class="header">
                <div class="header-main">
                    <div class="brand">
                        <span class="logo"><span class="logo-h">HYUNDAI</span><span class="logo-r">Rotem</span><i class="logo-accent"></i></span>
                        <span class="sysname">AI 자율전차 지휘통제</span>
                    </div>
                    <div class="header-metrics">
                        <div class="metric"><strong>MODE</strong><b id="modeValue">WAIT</b></div>
                        <div class="metric"><strong>YOLO</strong><b id="yoloValue">WAIT</b></div>
                        <div class="metric"><strong>ROS</strong><b id="rosValue">WAIT</b></div>
                        <div class="metric"><strong>TIME</strong><b id="timeValue">--:--:--</b></div>
                        <div class="metric"><strong>STATUS</strong><b id="statusValue">BOOT</b></div>
                    </div>
                </div>
                <div class="classbar"><span>통제용 · OFFICIAL USE ONLY</span><span>AI 자율전차 지휘통제 체계 · 실시간 운용</span></div>
            </header>
            <main class="main-grid">
                <section class="panel">
                    <div class="panel-title"><span>① CAMERA · YOLO</span><span id="feedStatusText" class="feed-status-text">det=0 sync</span></div>
                    <div class="feed-wrap">
                        <img id="driveFeed" src="/video_feed" alt="drive feed">
                        <canvas id="driveOverlay"></canvas>
                    </div>
                </section>
                <section class="panel">
                    <div class="panel-title"><span>② RVIZ 3D / MAP</span><span id="mapPanelTitle">TERRAIN MAP</span></div>
                    <div class="map-tabs">
                        <button id="map-tab-terrain" class="tab-button active" type="button" onclick="setMapTab('terrain')">TERRAIN</button>
                        <button id="map-tab-ros" class="tab-button" type="button" onclick="setMapTab('ros')">ROS</button>
                        <button id="map-tab-rviz" class="tab-button" type="button" onclick="setMapTab('rviz')">RVIZ 3D</button>
                    </div>
                    <div class="map-wrap">
                        <canvas id="mapCanvas"></canvas>
                        <iframe id="rvizFrame" title="RViz 3D" style="display:none;"></iframe>
                        <div id="rosWrap" style="display:none;position:absolute;inset:0;flex-direction:column;background:var(--bg);">
                            <div class="map-tabs" style="grid-template-columns:repeat(3,minmax(0,1fr));height:32px;">
                                <button id="ros-sub-graph" class="tab-button active" type="button" onclick="setRosSub('graph')">GRAPH</button>
                                <button id="ros-sub-services" class="tab-button" type="button" onclick="setRosSub('services')">SERVICES</button>
                                <button id="ros-sub-params" class="tab-button" type="button" onclick="setRosSub('params')">PARAMS</button>
                            </div>
                            <div style="position:relative;flex:1 1 auto;min-height:0;">
                                <div id="rosGraph" style="position:absolute;inset:0;"></div>
                                <div id="rosGraphTools" style="position:absolute;top:6px;right:8px;z-index:5;display:flex;gap:5px;align-items:center;">
                                    <span style="color:var(--muted);font-size:10px">보기</span>
                                    <button id="rosFlowBtn" class="rosbtn on" type="button" onclick="toggleRosFlow()" title="노드↔노드 / 흐름 / 전체 전환">노드</button>
                                    <button class="rosbtn" type="button" onclick="cyRosFit()">FIT</button>
                                </div>
                                <div id="rosEdgeInfo" style="display:none;position:absolute;left:8px;bottom:8px;max-width:46%;max-height:62%;overflow:auto;z-index:6;background:rgba(8,19,40,0.96);border:1px solid var(--line);padding:8px 10px;font-size:10px;line-height:1.6;"></div>
                                <div id="rosServices" class="scroll" style="position:absolute;inset:0;display:none;"></div>
                                <div id="rosParams" class="scroll" style="position:absolute;inset:0;display:none;"></div>
                            </div>
                        </div>
                        <div id="mapLegend" class="map-legend">
                            <span>SELF</span><span>ENEMY</span><span>TARGET</span><span>WATER</span><span>RIDGE</span><span>HIGH</span><span>TREE</span><span>ROCK</span><span>HOUSE</span><span>ROUTE</span>
                        </div>
                    </div>
                </section>
                <section class="panel">
                    <div class="panel-title"><span>③ TANK STATE · SYSTEM</span><span id="tankStatusText" class="feed-status-text">-</span></div>
                    <div id="tankSystemContent" class="scroll"></div>
                </section>
                <section class="panel">
                    <div class="panel-title"><span>④ LLM · RECON RISK</span><span id="riskStatusText" class="feed-status-text">-</span></div>
                    <div id="riskContent" class="scroll"></div>
                </section>
            </main>
            <footer class="bottom">
                <span><strong>latestYoloMs</strong><b id="bottomYolo">-</b></span>
                <span><strong>objectCount</strong><b id="bottomObjects">-</b></span>
                <span><strong>cache</strong><b id="bottomCache">-</b></span>
                <span><strong>route</strong><b id="bottomRoute">-</b></span>
                <span><strong>warning</strong><b id="bottomWarning">-</b></span>
            </footer>
        </div>
        <script>
            let activeTab = "route";
            let activeMapTab = "terrain";
            let latestState = null;
            let lastFetchOk = false;
            let staticMap = null;
            let staticMapLoadError = null;
            let staticTerrainCache = null;
            let overviewImage = null;
            let overviewImageLoaded = false;
            let overviewImageError = null;
            let lastWindowsReconAction = null;

            function byId(id) { return document.getElementById(id); }
            function safe(value, fallback = "-") { return value === undefined || value === null || value === "" ? fallback : value; }
            function numberText(value, digits = 1) {
                const n = Number(value);
                return Number.isFinite(n) ? n.toFixed(digits) : "-";
            }
            function escapeHtml(value) {
                return String(value ?? "").replace(/[&<>"']/g, (char) => ({
                    "&": "&amp;",
                    "<": "&lt;",
                    ">": "&gt;",
                    '"': "&quot;",
                    "'": "&#39;"
                }[char]));
            }
            function setStatusClass(element, status) {
                element.classList.remove("status-ok", "status-warn", "status-error");
                element.classList.add(status);
            }
            function setTab(tabName) {
                activeTab = tabName;
                for (const tab of ["route", "risk", "ai", "recon", "sensor"]) {
                    byId(`tab-${tab}`).classList.toggle("active", tab === tabName);
                }
                byId("leftPanelTitle").textContent = tabName === "route" ? "ROUTE" : tabName === "risk" ? "RECON RISK" : tabName === "ai" ? "AI LOG" : tabName === "recon" ? "RECON" : "SENSOR";
                updateLeftPanel(latestState || {});
            }
            let cyRos = null, cyRosSig = "", cyDashTimer = null;
            let rosGraphMode = "node";  // node(노드↔노드, 토픽 접음) | flow(노드↔노드 토픽) | all(전체)
            const ROS_MODES = ["node", "flow", "all"], ROS_MODE_LABEL = { node: "노드", flow: "흐름", all: "전체" };
            function cyRosFit() { if (cyRos) { try { cyRos.resize(); cyRos.fit(undefined, 12); } catch (e) {} } }
            function toggleRosFlow() {
                rosGraphMode = ROS_MODES[(ROS_MODES.indexOf(rosGraphMode) + 1) % ROS_MODES.length];
                byId("rosFlowBtn").textContent = ROS_MODE_LABEL[rosGraphMode];
                hideRosEdgeInfo(); cyRosSig = "";
                renderRosGraph(latestState || {});
            }
            function showRosEdgeInfo(src, dst, topics) {
                const d = byId("rosEdgeInfo"); if (!d) return;
                d.innerHTML = `<div style="color:var(--green);font-weight:700;margin-bottom:5px">${escapeHtml(rosShort(src))} → ${escapeHtml(rosShort(dst))} · ${topics.length}개 토픽</div>`
                    + topics.slice().sort().map((t) => `<div style="color:#bfe0ff">${escapeHtml(t)}</div>`).join("");
                d.style.display = "block";
            }
            function hideRosEdgeInfo() { const d = byId("rosEdgeInfo"); if (d) d.style.display = "none"; }
            function rosShort(id) {
                if (id.startsWith("t:")) { const p = id.slice(2).split("/").filter(Boolean); return "/" + p.slice(-2).join("/"); }
                const p = id.split("/").filter(Boolean); return p[p.length - 1] || id;
            }
            function hzWidth(hz) { return (!hz || hz <= 0) ? 1.2 : Math.min(11, 1.6 + Math.log10(1 + hz) * 3.4); }
            function hzColor(hz) { if (!hz || hz <= 0) return "#2c5a44"; if (hz < 2) return "#39ff88"; if (hz < 15) return "#ffd34d"; return "#ff8a3d"; }
            function ensureCyRos() {
                if (cyRos) return cyRos;
                if (typeof cytoscape === "undefined") return null;
                try { if (window.cytoscapeDagre) cytoscape.use(window.cytoscapeDagre); } catch (e) {}
                cyRos = cytoscape({
                    container: byId("rosGraph"), elements: [], wheelSensitivity: 0.2,
                    style: [
                        { selector: 'node[kind="node"]', style: { shape: "round-rectangle", "background-color": "#0d2147", "border-color": "#4f8dff", "border-width": 1.5, label: "data(label)", color: "#dbe6ff", "font-size": 10, "font-weight": "bold", "text-valign": "center", "text-halign": "center", width: "label", height: 20, "padding": "7px", "text-wrap": "wrap", "text-max-width": "130px" } },
                        { selector: 'node[kind="topic"]', style: { shape: "round-rectangle", "background-color": "#0a1730", "border-color": "#5ac8ff", "border-width": 1, label: "data(label)", color: "#bfe0ff", "font-size": 9, "text-valign": "center", "text-halign": "center", width: "label", height: 15, "padding": "4px" } },
                        { selector: "edge", style: { "curve-style": "bezier", "target-arrow-shape": "triangle", width: "data(w)", "line-color": "data(c)", "target-arrow-color": "data(c)", "arrow-scale": 0.8, opacity: 0.78, label: "data(elabel)", "font-size": 8, color: "#9fd8ff", "text-rotation": "autorotate", "text-background-color": "#06101f", "text-background-opacity": 0.85, "text-background-padding": 1 } },
                        { selector: "edge[active = 1]", style: { "line-style": "dashed", "line-dash-pattern": [6, 4] } },
                        { selector: "edge:selected", style: { "line-color": "#ffffff", "target-arrow-color": "#ffffff", "z-index": 99 } },
                    ],
                    layout: { name: "preset" },
                });
                cyRos.on("tap", "edge", (ev) => {
                    const t = ev.target.data("topics");
                    if (t && t.length) showRosEdgeInfo(ev.target.data("src2"), ev.target.data("dst2"), t);
                    else hideRosEdgeInfo();
                });
                cyRos.on("tap", (ev) => { if (ev.target === cyRos) hideRosEdgeInfo(); });
                if (!cyDashTimer) {
                    let off = 0;
                    cyDashTimer = setInterval(() => {
                        if (!cyRos || byId("rosGraph").style.display === "none") return;
                        off -= 1; cyRos.edges("[active = 1]").style("line-dash-offset", off);
                    }, 80);
                }
                return cyRos;
            }
            function renderRosGraph(state) {
                const cont = byId("rosGraph");
                const g = state && state.rosGraph;
                if (!g || !g.available) {
                    if (cyRos) { cyRos.destroy(); cyRos = null; cyRosSig = ""; }
                    cont.innerHTML = '<div style="color:#5a6b62;font-size:12px;padding:14px">ROS 그래프 대기 — 자율 스택 실행 시 노드/토픽 표시</div>';
                    return;
                }
                const cy = ensureCyRos();
                if (!cy) { cont.innerHTML = '<div style="color:#5a6b62;font-size:12px;padding:14px">cytoscape 로드 실패(CDN 확인)</div>'; return; }
                const nodes = g.nodes || [], topics = g.topics || [], edges = g.edges || [];
                const els = [];
                if (rosGraphMode === "node") {
                    // 토픽을 접고 노드→노드 직결(엣지 라벨=토픽 수, 클릭하면 토픽 목록 펼침)
                    const pubs = {}, subs = {}, thz = {};
                    edges.forEach((e) => {
                        if (String(e.target).startsWith("t:")) { (pubs[e.target] = pubs[e.target] || []).push(e.source); thz[e.target] = e.hz || 0; }
                        else if (String(e.source).startsWith("t:")) { (subs[e.source] = subs[e.source] || []).push(e.target); }
                    });
                    const n2n = {};
                    Object.keys(pubs).forEach((t) => (pubs[t] || []).forEach((P) => (subs[t] || []).forEach((S) => {
                        if (P === S) return;
                        const k = P + "" + S, o = n2n[k] || (n2n[k] = { src: P, dst: S, topics: [], maxHz: 0 });
                        o.topics.push(t.slice(2)); o.maxHz = Math.max(o.maxHz, thz[t] || 0);
                    })));
                    const used = new Set();
                    Object.values(n2n).forEach((o) => { used.add(o.src); used.add(o.dst); });
                    nodes.forEach((n) => { if (used.has(n.id)) els.push({ data: { id: n.id, kind: "node", label: rosShort(n.id) } }); });
                    Object.keys(n2n).forEach((k) => { const o = n2n[k]; els.push({ data: { id: "e:" + k, source: o.src, target: o.dst, src2: o.src, dst2: o.dst, topics: o.topics, elabel: String(o.topics.length), w: hzWidth(o.maxHz), c: hzColor(o.maxHz), active: o.maxHz > 0 ? 1 : 0 } }); });
                    if (!els.length) nodes.forEach((n) => els.push({ data: { id: n.id, kind: "node", label: rosShort(n.id) } }));
                } else {
                    let tps = topics, eds = edges, nds = nodes;
                    if (rosGraphMode === "flow" && nodes.length >= 2) {
                        const hasPub = new Set(), hasSub = new Set();
                        edges.forEach((e) => { if (String(e.target).startsWith("t:")) hasPub.add(e.target); if (String(e.source).startsWith("t:")) hasSub.add(e.source); });
                        const keep = new Set(topics.filter((t) => hasPub.has(t.id) && hasSub.has(t.id)).map((t) => t.id));
                        if (keep.size) {
                            tps = topics.filter((t) => keep.has(t.id));
                            eds = edges.filter((e) => keep.has(e.source) || keep.has(e.target));
                            const u = new Set(); eds.forEach((e) => { if (!String(e.source).startsWith("t:")) u.add(e.source); if (!String(e.target).startsWith("t:")) u.add(e.target); });
                            nds = nodes.filter((n) => u.has(n.id));
                        }
                    }
                    nds.forEach((n) => els.push({ data: { id: n.id, kind: "node", label: rosShort(n.id) } }));
                    tps.forEach((t) => { const hz = t.hz; els.push({ data: { id: t.id, kind: "topic", label: rosShort(t.id) + (hz ? ` ${hz}Hz` : ""), hz: hz || 0 } }); });
                    eds.forEach((e) => { const hz = e.hz || 0; els.push({ data: { id: e.id, source: e.source, target: e.target, w: hzWidth(hz), c: hzColor(hz), active: hz > 0 ? 1 : 0, elabel: "" } }); });
                }
                const sig = rosGraphMode + "|" + els.map((x) => x.data.id).sort().join("|");
                if (sig !== cyRosSig) {
                    cyRos.json({ elements: els });
                    cyRosSig = sig;
                    try { cyRos.layout({ name: "dagre", rankDir: "LR", nodeSep: 18, rankSep: 70, edgeSep: 8, fit: true, padding: 12 }).run(); }
                    catch (e) { try { cyRos.layout({ name: "breadthfirst", directed: true, spacingFactor: 0.9, fit: true, padding: 12 }).run(); } catch (e2) {} }
                } else {
                    cyRos.batch(() => {
                        els.forEach((x) => {
                            const el = cyRos.getElementById(x.data.id);
                            if (!el || !el.length) return;
                            if (x.data.kind === "topic") { el.data("label", x.data.label); el.data("hz", x.data.hz); }
                            if (x.data.w != null) { el.data("w", x.data.w); el.data("c", x.data.c); el.data("active", x.data.active); if (x.data.elabel != null) el.data("elabel", x.data.elabel); }
                        });
                    });
                }
            }
            let rosSubTab = "graph", rosParamNode = "";
            function setRosSub(s) {
                rosSubTab = ["graph", "services", "params"].includes(s) ? s : "graph";
                ["graph", "services", "params"].forEach((k) => byId("ros-sub-" + k).classList.toggle("active", k === rosSubTab));
                byId("rosGraph").style.display = rosSubTab === "graph" ? "block" : "none";
                byId("rosGraphTools").style.display = rosSubTab === "graph" ? "flex" : "none";
                byId("rosServices").style.display = rosSubTab === "services" ? "block" : "none";
                byId("rosParams").style.display = rosSubTab === "params" ? "block" : "none";
                byId("rosServices").dataset.sig = ""; byId("rosParams").dataset.count = "";
                renderRosActive(latestState || {});
                if (rosSubTab === "graph" && cyRos) { setTimeout(() => { try { cyRos.resize(); cyRos.fit(undefined, 8); } catch (e) {} }, 50); }
            }
            function renderRosActive(state) {
                if (rosSubTab === "services") renderRosServices(state);
                else if (rosSubTab === "params") renderRosParams(state);
                else renderRosGraph(state);
            }
            function renderRosServices(state) {
                const c = byId("rosServices"); const svc = (state && state.rosGraph && state.rosGraph.services) || [];
                const sig = svc.map((s) => s.name + (s.nodes || []).join()).join("|");
                if (c.dataset.sig === sig) return; c.dataset.sig = sig;
                if (!svc.length) { c.innerHTML = '<div style="color:var(--muted);padding:12px;line-height:1.6">서비스 없음<br><span style="color:#5a6b62;font-size:11px">자율 스택(run_recon_scenario.py / tank_autonomous_control.launch) 실행 시 표시됩니다.<br>지금은 브릿지 노드만 동작 중입니다.</span></div>'; return; }
                let h = '<table style="width:100%;border-collapse:collapse;font-size:11px"><tr style="color:var(--muted);text-align:left"><th style="padding:4px">서비스</th><th>타입</th><th>노드</th></tr>';
                svc.forEach((s) => { h += `<tr style="border-top:1px solid var(--line-dim)"><td style="padding:4px;color:var(--cyan)">${escapeHtml(rosShort("t:" + s.name))}</td><td style="color:var(--muted)">${escapeHtml((s.type || "").split("/").pop())}</td><td>${escapeHtml((s.nodes || []).map((n) => n.split("/").pop()).join(", "))}</td></tr>`; });
                c.innerHTML = h + "</table>";
            }
            function renderRosParams(state) {
                const c = byId("rosParams"); const nodes = (state && state.rosGraph && state.rosGraph.nodes) || [];
                if (c.dataset.count === String(nodes.length)) return;  // 노드 수 동일하면 재구성 안 함(편집 중 깜빡임 방지)
                c.dataset.count = String(nodes.length);
                const opts = nodes.map((n) => `<option value="${escapeHtml(n.id)}" ${n.id === rosParamNode ? "selected" : ""}>${escapeHtml(n.id)}</option>`).join("");
                c.innerHTML = `<div style="padding:8px">
                    <select id="rosParamSel" onchange="loadRosParams(this.value)" style="width:100%;background:#0a1730;color:var(--text);border:1px solid var(--line);padding:6px;font:inherit;font-size:12px">
                    <option value="">노드 선택…</option>${opts}</select>
                    <div id="rosParamTable" style="margin-top:8px"></div></div>`;
                if (rosParamNode) loadRosParams(rosParamNode);
            }
            async function loadRosParams(node) {
                rosParamNode = node; const t = byId("rosParamTable"); if (!t) return;
                if (!node) { t.innerHTML = ""; return; }
                t.innerHTML = '<div style="color:var(--muted)">로딩…</div>';
                try {
                    const d = await (await fetch("/api/ros/params?node=" + encodeURIComponent(node), { cache: "no-store" })).json();
                    if (!d.ok) { t.innerHTML = `<div style="color:var(--red)">${escapeHtml(d.error || "실패")}</div>`; return; }
                    if (!d.params.length) { t.innerHTML = '<div style="color:var(--muted)">파라미터 없음</div>'; return; }
                    let h = '<table style="width:100%;border-collapse:collapse;font-size:11px"><tr style="color:var(--muted);text-align:left"><th style="padding:3px">이름</th><th>타입</th><th>값</th><th></th></tr>';
                    d.params.forEach((p, i) => { const val = Array.isArray(p.value) ? p.value.join(",") : String(p.value); h += `<tr style="border-top:1px solid var(--line-dim)"><td style="padding:3px;color:var(--cyan)">${escapeHtml(p.name)}</td><td style="color:var(--muted)">${escapeHtml(p.type)}</td><td><input id="pp${i}" value="${escapeHtml(val)}" style="width:88px;background:#0a1730;color:var(--text);border:1px solid var(--line-dim);font:inherit;font-size:11px"></td><td><button class="tab-button" style="padding:2px 7px" onclick="setRosParam('${escapeHtml(node)}','${escapeHtml(p.name)}','pp${i}')">SET</button></td></tr>`; });
                    t.innerHTML = h + "</table>";
                } catch (e) { t.innerHTML = `<div style="color:var(--red)">오류: ${escapeHtml(String(e))}</div>`; }
            }
            async function setRosParam(node, name, inputId) {
                const inp = byId(inputId); if (!inp) return;
                try {
                    const d = await (await fetch("/api/ros/params/set", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ node, name, value: inp.value }) })).json();
                    inp.style.borderColor = d.ok ? "var(--ok)" : "var(--red)";
                    if (!d.ok) alert("설정 실패: " + (d.reason || d.error || ""));
                } catch (e) { alert("오류: " + e); }
            }
            function setMapTab(tabName) {
                activeMapTab = ["terrain", "ros", "rviz"].includes(tabName) ? tabName : "terrain";
                byId("map-tab-terrain").classList.toggle("active", activeMapTab === "terrain");
                byId("map-tab-ros").classList.toggle("active", activeMapTab === "ros");
                byId("map-tab-rviz").classList.toggle("active", activeMapTab === "rviz");
                const isTerrain = activeMapTab === "terrain", isRos = activeMapTab === "ros", isRviz = activeMapTab === "rviz";
                byId("mapCanvas").style.display = isTerrain ? "block" : "none";
                byId("mapLegend").style.display = isTerrain ? "flex" : "none";
                byId("rosWrap").style.display = isRos ? "flex" : "none";
                const frame = byId("rvizFrame");
                frame.style.display = isRviz ? "block" : "none";
                if (isRviz && !frame.src) { frame.src = "/rviz3d?frame=tank_map&cloud=detected&rays=1"; }
                byId("mapPanelTitle").textContent = isRviz ? "RVIZ 3D" : (isRos ? "ROS MONITOR" : "TERRAIN MAP");
                if (isTerrain) { updateMapLegend(); drawMap(latestState || {}); }
                if (isRos) { renderRosActive(latestState || {}); }
            }
            function updateMapLegend() {
                const terrain = ["SELF", "ENEMY", "WATER", "RIDGE", "HIGH", "TREE", "ROCK"];
                const ros = ["SELF", "ENEMY", "TARGET", "OBS", "ROUTE", "YOLO"];
                byId("mapLegend").innerHTML = (activeMapTab === "ros" ? ros : terrain)
                    .map((label) => `<span>${label}</span>`)
                    .join("");
            }
            function latestBridge(state) { return state?.bridge?.latest || {}; }
            function routeCounts(state) { return state?.bridge?.routeCounts || state?.bridge?.route_counts || {}; }
            function readPoint(obj) {
                if (!obj || typeof obj !== "object") return null;
                const x = obj.x ?? obj.position?.x ?? obj.pose?.position?.x;
                const y = obj.y ?? obj.z ?? obj.position?.y ?? obj.position?.z ?? obj.pose?.position?.y ?? obj.pose?.position?.z;
                if (x === undefined || y === undefined) return null;
                const px = Number(x);
                const py = Number(y);
                if (!Number.isFinite(px) || !Number.isFinite(py)) return null;
                return { x: px, y: py };
            }
            function extractArray(value) {
                if (Array.isArray(value)) return value;
                if (Array.isArray(value?.data)) return value.data;
                if (Array.isArray(value?.obstacles)) return value.obstacles;
                if (Array.isArray(value?.points)) return value.points;
                if (Array.isArray(value?.route)) return value.route;
                if (Array.isArray(value?.path)) return value.path;
                return [];
            }
            function mapObjectCategory(name) {
                const text = String(name || "").toLowerCase();
                if (text.startsWith("tree")) return "tree";
                if (text.startsWith("rock")) return "rock";
                if (text.startsWith("house")) return "house";
                if (text.startsWith("human")) return "human";
                if (text.startsWith("car")) return "car";
                if (text.startsWith("tank")) return "tank";
                return "unknown";
            }
            function readStaticObjectPoint(obj) {
                const pos = obj?.position || {};
                const x = Number(pos.x);
                const y = Number(pos.z);
                const height = Number(pos.y);
                if (!Number.isFinite(x) || !Number.isFinite(y)) return null;
                return {
                    x,
                    y,
                    height: Number.isFinite(height) ? height : null,
                    category: mapObjectCategory(obj?.prefabName),
                    name: obj?.prefabName || "object"
                };
            }
            function mapBoundsFromMap(mapData) {
                const bounds = mapData?.bounds || {};
                return {
                    minX: Number(bounds.min_x ?? bounds.minX ?? 0),
                    maxX: Number(bounds.max_x ?? bounds.maxX ?? 300),
                    minY: Number(bounds.min_y ?? bounds.min_z ?? bounds.minZ ?? 0),
                    maxY: Number(bounds.max_y ?? bounds.max_z ?? bounds.maxZ ?? 300)
                };
            }
            function staticMapMapper(width, height, mapData) {
                const bounds = mapBoundsFromMap(mapData);
                const minX = bounds.minX;
                const maxX = bounds.maxX;
                const minY = bounds.minY;
                const maxY = bounds.maxY;
                const pad = 20;
                const usableW = Math.max(1, width - pad * 2);
                const usableH = Math.max(1, height - pad * 2);
                const worldW = Math.max(1, maxX - minX);
                const worldH = Math.max(1, maxY - minY);
                const scale = Math.min(usableW / worldW, usableH / worldH);
                const offsetX = (width - worldW * scale) * 0.5;
                const offsetY = (height - worldH * scale) * 0.5;
                return (p) => ({
                    x: offsetX + (p.x - minX) * scale,
                    y: height - offsetY - (p.y - minY) * scale
                });
            }
            function clamp01(value) {
                const n = Number(value);
                if (!Number.isFinite(n)) return 0;
                return Math.max(0, Math.min(1, n));
            }
            function heightSummaryFromMap(mapData, points) {
                const summary = mapData?.heightSummary || {};
                let min = Number(summary.min);
                let max = Number(summary.max);
                let avg = Number(summary.avg);
                const heights = points.map((p) => Number(p.height)).filter(Number.isFinite);
                if ((!Number.isFinite(min) || !Number.isFinite(max)) && heights.length) {
                    min = Math.min(...heights);
                    max = Math.max(...heights);
                }
                if (!Number.isFinite(avg) && heights.length) {
                    avg = heights.reduce((sum, h) => sum + h, 0) / heights.length;
                }
                if (!Number.isFinite(min) || !Number.isFinite(max)) return null;
                const span = Math.max(0.001, max - min);
                const surface = mapData?.surfaceSummary || {};
                const low = Number(surface.lowThreshold);
                const high = Number(surface.highThreshold);
                return {
                    min,
                    max,
                    avg: Number.isFinite(avg) ? avg : min + span * 0.5,
                    span,
                    lowThreshold: Number.isFinite(low) ? low : min + span * 0.18,
                    highThreshold: Number.isFinite(high) ? high : min + span * 0.78
                };
            }
            function topoColor(t) {
                const v = clamp01(t);
                if (v < 0.12) return "#0b1b12";
                if (v < 0.26) return "#102c1b";
                if (v < 0.42) return "#174226";
                if (v < 0.58) return "#1f5b33";
                if (v < 0.72) return "#486535";
                if (v < 0.86) return "#6d6036";
                return "#83583c";
            }
            function terrainCacheKey(mapData, staticObjects, terrain) {
                return [
                    mapData?.mapFile || "static-map",
                    safe(mapData?.objectCount, staticObjects.length),
                    staticObjects.length,
                    numberText(terrain?.min, 3),
                    numberText(terrain?.max, 3)
                ].join("|");
            }
            function buildConvexHull(points) {
                const sorted = points
                    .filter((p) => Number.isFinite(p.x) && Number.isFinite(p.y))
                    .map((p) => ({ x: p.x, y: p.y }))
                    .sort((a, b) => a.x === b.x ? a.y - b.y : a.x - b.x);
                if (sorted.length <= 2) return sorted;
                const cross = (o, a, b) => (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
                const lower = [];
                for (const p of sorted) {
                    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) lower.pop();
                    lower.push(p);
                }
                const upper = [];
                for (let i = sorted.length - 1; i >= 0; i -= 1) {
                    const p = sorted[i];
                    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0) upper.pop();
                    upper.push(p);
                }
                lower.pop();
                upper.pop();
                return lower.concat(upper);
            }
            function buildTerrainGrid(staticObjects, mapData) {
                const terrain = heightSummaryFromMap(mapData, staticObjects);
                if (!terrain) return;
                const samples = staticObjects.filter((obj) => Number.isFinite(Number(obj.height)));
                if (!samples.length) return;
                const key = terrainCacheKey(mapData, staticObjects, terrain);
                if (staticTerrainCache?.key === key) return staticTerrainCache;
                const bounds = mapBoundsFromMap(mapData);
                const cols = 104;
                const rows = 104;
                const worldW = Math.max(1, bounds.maxX - bounds.minX);
                const worldH = Math.max(1, bounds.maxY - bounds.minY);
                const smoothing = Math.max(8, Math.min(worldW, worldH) / 22);
                const values = [];
                for (let row = 0; row <= rows; row += 1) {
                    const y = bounds.minY + (worldH * row) / rows;
                    for (let col = 0; col <= cols; col += 1) {
                        const x = bounds.minX + (worldW * col) / cols;
                        let numerator = 0;
                        let denominator = 0;
                        let exact = null;
                        for (const sample of samples) {
                            const dx = x - sample.x;
                            const dy = y - sample.y;
                            const d2 = dx * dx + dy * dy;
                            if (d2 < 0.01) {
                                exact = Number(sample.height);
                                break;
                            }
                            const weight = 1 / Math.pow(d2 + smoothing * smoothing, 1.22);
                            numerator += Number(sample.height) * weight;
                            denominator += weight;
                        }
                        values.push(exact ?? (denominator ? numerator / denominator : terrain.avg));
                    }
                }
                staticTerrainCache = {
                    key,
                    bounds,
                    cols,
                    rows,
                    values,
                    terrain,
                    hull: buildConvexHull(samples)
                };
                return staticTerrainCache;
            }
            function terrainValue(grid, row, col) {
                return grid.values[row * (grid.cols + 1) + col];
            }
            function terrainWorldPoint(grid, row, col) {
                return {
                    x: grid.bounds.minX + ((grid.bounds.maxX - grid.bounds.minX) * col) / grid.cols,
                    y: grid.bounds.minY + ((grid.bounds.maxY - grid.bounds.minY) * row) / grid.rows
                };
            }
            function screenRectFromBounds(mapper, bounds) {
                const p0 = mapper({ x: bounds.minX, y: bounds.minY });
                const p1 = mapper({ x: bounds.maxX, y: bounds.maxY });
                return {
                    x: Math.min(p0.x, p1.x),
                    y: Math.min(p0.y, p1.y),
                    w: Math.abs(p1.x - p0.x),
                    h: Math.abs(p1.y - p0.y)
                };
            }
            function terrainZonesOf(mapData, type) {
                const zones = mapData?.terrainZones?.zones;
                if (!Array.isArray(zones)) return [];
                return zones.filter((zone) => zone?.type === type && Array.isArray(zone.points) && zone.points.length >= 3);
            }
            function drawZonePath(ctx, mapper, zone) {
                const points = zone.points
                    .map((point) => ({ x: Number(point.x), y: Number(point.y ?? point.z) }))
                    .filter((point) => Number.isFinite(point.x) && Number.isFinite(point.y))
                    .map(mapper);
                if (points.length < 3) return null;
                ctx.beginPath();
                points.forEach((point, index) => {
                    if (index === 0) ctx.moveTo(point.x, point.y);
                    else ctx.lineTo(point.x, point.y);
                });
                ctx.closePath();
                return points;
            }
            function drawRockyZones(ctx, mapper, mapData) {
                const zones = terrainZonesOf(mapData, "rocky");
                for (const zone of zones) {
                    ctx.save();
                    const points = drawZonePath(ctx, mapper, zone);
                    if (!points) {
                        ctx.restore();
                        continue;
                    }
                    ctx.fillStyle = "rgba(75, 84, 74, 0.72)";
                    ctx.fill();
                    ctx.strokeStyle = "rgba(143, 157, 134, 0.48)";
                    ctx.lineWidth = 1.2;
                    ctx.stroke();
                    ctx.clip();
                    const xs = points.map((p) => p.x);
                    const ys = points.map((p) => p.y);
                    const minX = Math.min(...xs);
                    const maxX = Math.max(...xs);
                    const minY = Math.min(...ys);
                    const maxY = Math.max(...ys);
                    ctx.strokeStyle = "rgba(185, 195, 174, 0.22)";
                    ctx.lineWidth = 1;
                    for (let x = minX - 24; x < maxX + 28; x += 18) {
                        ctx.beginPath();
                        ctx.moveTo(x, maxY + 8);
                        ctx.lineTo(x + 42, minY - 8);
                        ctx.stroke();
                    }
                    ctx.restore();
                }
            }
            function drawWaterZones(ctx, mapper, mapData) {
                const zones = terrainZonesOf(mapData, "water");
                for (const zone of zones) {
                    ctx.save();
                    const points = drawZonePath(ctx, mapper, zone);
                    if (!points) {
                        ctx.restore();
                        continue;
                    }
                    ctx.fillStyle = "rgba(2, 39, 47, 0.98)";
                    ctx.fill();
                    ctx.strokeStyle = "rgba(68, 217, 255, 0.56)";
                    ctx.lineWidth = 2;
                    ctx.stroke();
                    ctx.clip();
                    const xs = points.map((p) => p.x);
                    const ys = points.map((p) => p.y);
                    const minX = Math.min(...xs);
                    const maxX = Math.max(...xs);
                    const minY = Math.min(...ys);
                    const maxY = Math.max(...ys);
                    const gradient = ctx.createLinearGradient(minX, minY, maxX, maxY);
                    gradient.addColorStop(0, "rgba(20, 118, 134, 0.32)");
                    gradient.addColorStop(0.48, "rgba(0, 18, 24, 0.54)");
                    gradient.addColorStop(1, "rgba(11, 88, 104, 0.28)");
                    ctx.fillStyle = gradient;
                    ctx.fillRect(minX, minY, maxX - minX, maxY - minY);
                    ctx.strokeStyle = "rgba(122, 218, 224, 0.12)";
                    ctx.lineWidth = 1;
                    for (let y = minY + 12; y < maxY; y += 18) {
                        ctx.beginPath();
                        ctx.moveTo(minX - 10, y);
                        ctx.quadraticCurveTo((minX + maxX) * 0.5, y + 7, maxX + 10, y - 2);
                        ctx.stroke();
                    }
                    ctx.restore();
                }
            }
            function drawPassageZones(ctx, mapper, mapData, mode = "terrain") {
                const zones = terrainZonesOf(mapData, "passage");
                for (const zone of zones) {
                    ctx.save();
                    const points = drawZonePath(ctx, mapper, zone);
                    if (!points) {
                        ctx.restore();
                        continue;
                    }
                    ctx.fillStyle = mode === "ros" ? "rgba(4, 12, 8, 0.96)" : "rgba(18, 56, 31, 0.96)";
                    ctx.fill();
                    ctx.restore();
                }
            }
            function fillTerrainCell(ctx, grid, mapper, row, col, fillStyle, bleed = 0.8) {
                const a = mapper(terrainWorldPoint(grid, row, col));
                const b = mapper(terrainWorldPoint(grid, row + 1, col + 1));
                ctx.fillStyle = fillStyle;
                ctx.fillRect(
                    Math.min(a.x, b.x) - bleed,
                    Math.min(a.y, b.y) - bleed,
                    Math.abs(b.x - a.x) + bleed * 2,
                    Math.abs(b.y - a.y) + bleed * 2
                );
            }
            function drawContourLevel(ctx, grid, mapper, level) {
                const crosses = (a, b) => (a < level && b >= level) || (a >= level && b < level);
                const interp = (pa, pb, va, vb) => {
                    const denom = vb - va;
                    const t = Math.abs(denom) < 0.0001 ? 0.5 : (level - va) / denom;
                    return { x: pa.x + (pb.x - pa.x) * t, y: pa.y + (pb.y - pa.y) * t };
                };
                for (let row = 0; row < grid.rows; row += 1) {
                    for (let col = 0; col < grid.cols; col += 1) {
                        const v00 = terrainValue(grid, row, col);
                        const v10 = terrainValue(grid, row, col + 1);
                        const v11 = terrainValue(grid, row + 1, col + 1);
                        const v01 = terrainValue(grid, row + 1, col);
                        const p00 = mapper(terrainWorldPoint(grid, row, col));
                        const p10 = mapper(terrainWorldPoint(grid, row, col + 1));
                        const p11 = mapper(terrainWorldPoint(grid, row + 1, col + 1));
                        const p01 = mapper(terrainWorldPoint(grid, row + 1, col));
                        const hits = [];
                        if (crosses(v00, v10)) hits.push(interp(p00, p10, v00, v10));
                        if (crosses(v10, v11)) hits.push(interp(p10, p11, v10, v11));
                        if (crosses(v11, v01)) hits.push(interp(p11, p01, v11, v01));
                        if (crosses(v01, v00)) hits.push(interp(p01, p00, v01, v00));
                        if (hits.length === 2) {
                            ctx.moveTo(hits[0].x, hits[0].y);
                            ctx.lineTo(hits[1].x, hits[1].y);
                        } else if (hits.length === 4) {
                            ctx.moveTo(hits[0].x, hits[0].y);
                            ctx.lineTo(hits[1].x, hits[1].y);
                            ctx.moveTo(hits[2].x, hits[2].y);
                            ctx.lineTo(hits[3].x, hits[3].y);
                        }
                    }
                }
            }
            function drawContours(ctx, grid, mapper) {
                const interval = Math.max(1.5, grid.terrain.span / 9);
                const first = Math.ceil(grid.terrain.min / interval) * interval;
                for (let level = first; level <= grid.terrain.max; level += interval) {
                    const major = Math.round((level - first) / interval) % 2 === 0;
                    ctx.beginPath();
                    ctx.strokeStyle = major ? "rgba(255, 202, 79, 0.26)" : "rgba(216, 255, 233, 0.13)";
                    ctx.lineWidth = major ? 1.15 : 0.75;
                    drawContourLevel(ctx, grid, mapper, level);
                    ctx.stroke();
                }
            }
            function drawOverviewTexture(ctx, rect, mode = "terrain") {
                if (!overviewImageLoaded || !overviewImage) return false;
                ctx.save();
                ctx.drawImage(overviewImage, rect.x, rect.y, rect.w, rect.h);
                ctx.fillStyle = mode === "ros" ? "rgba(3, 12, 9, 0.56)" : "rgba(3, 14, 8, 0.34)";
                ctx.fillRect(rect.x, rect.y, rect.w, rect.h);
                ctx.strokeStyle = "rgba(57, 255, 136, 0.08)";
                ctx.lineWidth = 1;
                const grid = 30;
                for (let x = rect.x; x <= rect.x + rect.w; x += rect.w / (300 / grid)) {
                    ctx.beginPath();
                    ctx.moveTo(x, rect.y);
                    ctx.lineTo(x, rect.y + rect.h);
                    ctx.stroke();
                }
                for (let y = rect.y; y <= rect.y + rect.h; y += rect.h / (300 / grid)) {
                    ctx.beginPath();
                    ctx.moveTo(rect.x, y);
                    ctx.lineTo(rect.x + rect.w, y);
                    ctx.stroke();
                }
                ctx.restore();
                return true;
            }
            function drawTopographicLayer(ctx, mapper, staticObjects, mapData, width, height) {
                const grid = buildTerrainGrid(staticObjects, mapData);
                if (!grid) return false;
                const rect = screenRectFromBounds(mapper, grid.bounds);
                ctx.save();
                const usedTexture = drawOverviewTexture(ctx, rect, "terrain");
                if (!usedTexture) {
                    ctx.fillStyle = "#06110b";
                    ctx.fillRect(rect.x, rect.y, rect.w, rect.h);
                }
                ctx.beginPath();
                ctx.rect(rect.x, rect.y, rect.w, rect.h);
                ctx.clip();
                if (!usedTexture) {
                    for (let row = 0; row < grid.rows; row += 1) {
                        for (let col = 0; col < grid.cols; col += 1) {
                            const h00 = terrainValue(grid, row, col);
                            const h10 = terrainValue(grid, row, col + 1);
                            const h11 = terrainValue(grid, row + 1, col + 1);
                            const h01 = terrainValue(grid, row + 1, col);
                            const h = (h00 + h10 + h11 + h01) * 0.25;
                            fillTerrainCell(ctx, grid, mapper, row, col, topoColor((h - grid.terrain.min) / grid.terrain.span));
                            const shade = clamp01(((h10 + h11) - (h00 + h01)) / grid.terrain.span + 0.5) - 0.5;
                            if (Math.abs(shade) > 0.025) {
                                fillTerrainCell(
                                    ctx,
                                    grid,
                                    mapper,
                                    row,
                                    col,
                                    shade > 0 ? `rgba(216, 255, 233, ${Math.min(0.08, shade * 0.16)})` : `rgba(0, 0, 0, ${Math.min(0.16, Math.abs(shade) * 0.28)})`,
                                    0.8
                                );
                            }
                        }
                    }
                }
                let waterLabel = usedTexture ? "texture" : "overview";
                if (!usedTexture) {
                    drawRockyZones(ctx, mapper, mapData);
                    const hasOverviewWater = terrainZonesOf(mapData, "water").length > 0;
                    if (hasOverviewWater) {
                        drawWaterZones(ctx, mapper, mapData);
                    } else {
                    const lowWaterLimit = grid.terrain.min + grid.terrain.span * 0.34;
                    const deepWaterLimit = grid.terrain.min + grid.terrain.span * 0.24;
                    waterLabel = `<=${numberText(lowWaterLimit, 1)}`;
                    for (let row = 0; row < grid.rows; row += 1) {
                        for (let col = 0; col < grid.cols; col += 1) {
                            const h = (
                                terrainValue(grid, row, col) +
                                terrainValue(grid, row, col + 1) +
                                terrainValue(grid, row + 1, col + 1) +
                                terrainValue(grid, row + 1, col)
                            ) * 0.25;
                            if (h <= lowWaterLimit) {
                                fillTerrainCell(ctx, grid, mapper, row, col, h <= deepWaterLimit ? "rgba(14, 101, 122, 0.64)" : "rgba(49, 139, 144, 0.28)", 1.1);
                            }
                        }
                    }
                    ctx.beginPath();
                    ctx.strokeStyle = "rgba(68, 217, 255, 0.48)";
                    ctx.lineWidth = 2.1;
                    drawContourLevel(ctx, grid, mapper, lowWaterLimit);
                    ctx.stroke();
                    ctx.beginPath();
                    ctx.strokeStyle = "rgba(68, 217, 255, 0.28)";
                    ctx.lineWidth = 1.2;
                    drawContourLevel(ctx, grid, mapper, deepWaterLimit);
                    ctx.stroke();
                    }
                    drawPassageZones(ctx, mapper, mapData, "terrain");
                    drawContours(ctx, grid, mapper);
                }
                ctx.restore();

                ctx.save();
                ctx.fillStyle = "rgba(216, 255, 233, 0.82)";
                ctx.font = "11px Consolas, monospace";
                ctx.textAlign = "left";
                ctx.fillText(`TOPO MAP objects=${staticObjects.length}`, rect.x + 10, rect.y + 18);
                ctx.textAlign = "right";
                ctx.fillText(`elev ${numberText(grid.terrain.min, 1)}..${numberText(grid.terrain.max, 1)}`, rect.x + rect.w - 10, rect.y + 18);
                ctx.fillStyle = "rgba(68, 217, 255, 0.86)";
                ctx.fillText(`water ${waterLabel}`, rect.x + rect.w - 10, rect.y + 34);
                ctx.restore();
                return true;
            }
            function getDetections(state) {
                const yoloDetections = state?.yolo?.latestReturnedDetections;
                if (Array.isArray(yoloDetections) && yoloDetections.length) return yoloDetections;
                const liveDetections = state?.liveView?.latestDetections;
                if (Array.isArray(liveDetections) && liveDetections.length) return liveDetections;
                const detect = latestBridge(state)?.detect_result;
                if (Array.isArray(detect?.detections)) return detect.detections;
                return [];
            }
            function overlayClassColor(className) {
                const key = String(className || "").toLowerCase();
                const colors = { person: "#39ff88", car: "#ff8c00", tank: "#ff5b64", rock: "#ffca4f", house: "#b084ff" };
                return colors[key] || "#d8ffe9";
            }
            function driveImageBox(canvas, sourceW, sourceH) {
                const w = canvas.clientWidth || 1;
                const h = canvas.clientHeight || 1;
                const imageAspect = sourceW / Math.max(1, sourceH);
                const boxAspect = w / Math.max(1, h);
                if (boxAspect > imageAspect) {
                    const drawH = h;
                    const drawW = h * imageAspect;
                    return { x: (w - drawW) * 0.5, y: 0, w: drawW, h: drawH };
                }
                const drawW = w;
                const drawH = w / Math.max(0.001, imageAspect);
                return { x: 0, y: (h - drawH) * 0.5, w: drawW, h: drawH };
            }
            function drawOverlayBox(ctx, box, color) {
                ctx.save();
                ctx.strokeStyle = color;
                ctx.lineWidth = 1;
                ctx.beginPath();
                ctx.rect(box.x1, box.y1, box.x2 - box.x1, box.y2 - box.y1);
                ctx.stroke();
                const corner = Math.max(10, Math.min(24, Math.min(box.x2 - box.x1, box.y2 - box.y1) * 0.22));
                const segments = [
                    [box.x1, box.y1, box.x1 + corner, box.y1],
                    [box.x1, box.y1, box.x1, box.y1 + corner],
                    [box.x2, box.y1, box.x2 - corner, box.y1],
                    [box.x2, box.y1, box.x2, box.y1 + corner],
                    [box.x1, box.y2, box.x1 + corner, box.y2],
                    [box.x1, box.y2, box.x1, box.y2 - corner],
                    [box.x2, box.y2, box.x2 - corner, box.y2],
                    [box.x2, box.y2, box.x2, box.y2 - corner],
                ];
                for (const [x1, y1, x2, y2] of segments) {
                    ctx.beginPath();
                    ctx.moveTo(x1, y1);
                    ctx.lineTo(x2, y2);
                    ctx.stroke();
                }
                ctx.restore();
            }
            function drawOverlayLabel(ctx, text, x, y, color, canvasW, canvasH) {
                ctx.save();
                ctx.font = "11px Consolas, monospace";
                ctx.textBaseline = "top";
                const padX = 6;
                const padY = 4;
                const metrics = ctx.measureText(text);
                const textW = metrics.width;
                const textH = 12;
                let left = Math.max(4, Math.min(canvasW - textW - padX * 2 - 4, x));
                let top = y - textH - padY * 2 - 7;
                if (top < 4) top = Math.min(canvasH - textH - padY * 2 - 4, y + 6);
                top = Math.max(4, top);
                ctx.fillStyle = "rgba(4, 8, 6, 0.72)";
                ctx.fillRect(left, top, textW + padX * 2, textH + padY * 2);
                ctx.strokeStyle = color;
                ctx.lineWidth = 1;
                ctx.strokeRect(left, top, textW + padX * 2, textH + padY * 2);
                ctx.fillStyle = color;
                ctx.fillText(text, left + padX, top + padY);
                ctx.restore();
            }
            function drawFeedOverlay(state) {
                const canvas = byId("driveOverlay");
                if (!canvas) return;
                const rect = canvas.getBoundingClientRect();
                const dpr = window.devicePixelRatio || 1;
                const width = Math.max(1, Math.floor(rect.width * dpr));
                const height = Math.max(1, Math.floor(rect.height * dpr));
                if (canvas.width !== width || canvas.height !== height) {
                    canvas.width = width;
                    canvas.height = height;
                }
                const ctx = canvas.getContext("2d");
                ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
                ctx.clearRect(0, 0, rect.width, rect.height);
                const metadata = state?.liveView?.latestDetectionMetadata || latestBridge(state)?.detect_result || {};
                const shape = metadata.image_shape || state?.liveView?.latestSourceFrameShape || state?.yolo?.latestFrameShape || [];
                const sourceW = Number(metadata.image?.width || shape[1] || byId("driveFeed")?.naturalWidth || 1920);
                const sourceH = Number(metadata.image?.height || shape[0] || byId("driveFeed")?.naturalHeight || 1080);
                if (!Number.isFinite(sourceW) || !Number.isFinite(sourceH) || sourceW <= 0 || sourceH <= 0) return;
                const imageBox = driveImageBox(canvas, sourceW, sourceH);
                const detections = getDetections(state);
                for (const det of detections) {
                    const bbox = det?.bbox;
                    if (!Array.isArray(bbox) || bbox.length < 4) continue;
                    const x1 = Number(bbox[0]);
                    const y1 = Number(bbox[1]);
                    const x2 = Number(bbox[2]);
                    const y2 = Number(bbox[3]);
                    if (![x1, y1, x2, y2].every(Number.isFinite)) continue;
                    const mapped = {
                        x1: imageBox.x + (x1 / sourceW) * imageBox.w,
                        y1: imageBox.y + (y1 / sourceH) * imageBox.h,
                        x2: imageBox.x + (x2 / sourceW) * imageBox.w,
                        y2: imageBox.y + (y2 / sourceH) * imageBox.h,
                    };
                    const className = safe(det.className || det.class_name || det.modelClassName, "object");
                    const color = overlayClassColor(className);
                    drawOverlayBox(ctx, mapped, color);
                    drawOverlayLabel(ctx, `${className} ${numberText(det.confidence, 2)}`, mapped.x1, mapped.y1, color, rect.width, rect.height);
                }
            }
            function feedStatusText(state) {
                const liveView = state?.liveView || {};
                const metadata = liveView.latestDetectionMetadata || latestBridge(state)?.detect_result || {};
                const detections = getDetections(state);
                const count = Number.isFinite(Number(liveView.latestDetectionCount))
                    ? Number(liveView.latestDetectionCount)
                    : detections.length;
                let text = `det=${count}`;
                if (metadata.asyncYolo) {
                    const frameSeq = safe(metadata.frameSeq, "-");
                    const processedSeq = safe(metadata.processedFrameSeq, "-");
                    const age = Number(metadata.resultAgeMs);
                    text += Number.isFinite(age)
                        ? ` async frame=${frameSeq} yolo=${processedSeq} age=${age.toFixed(0)}ms`
                        : ` async frame=${frameSeq} yolo=${processedSeq}`;
                } else {
                    text += liveView.asyncYoloEnabled ? " async waiting" : " sync";
                }
                const sourceFps = Number(liveView.liveViewSourceFps);
                if (Number.isFinite(sourceFps) && sourceFps > 0) text += ` src=${sourceFps.toFixed(1)}fps`;
                return text;
            }
            function updateFeedStatus(state) {
                const element = byId("feedStatusText");
                if (element) element.textContent = feedStatusText(state);
            }
            async function runWindowsRecon(runLlm = false) {
                lastWindowsReconAction = {
                    running: true,
                    message: runLlm ? "Starting Windows LLM pipeline..." : "Building Windows recon files..."
                };
                updateLeftPanel(latestState || {});
                const url = runLlm ? "/api/recon/windows/run?llm=true&force=true" : "/api/recon/windows/run";
                try {
                    const response = await fetch(url, { method: "POST", cache: "no-store" });
                    const data = await response.json().catch(() => ({ ok: false, error: `HTTP ${response.status}` }));
                    const errors = Array.isArray(data.errors) ? data.errors.filter(Boolean) : [];
                    lastWindowsReconAction = {
                        running: false,
                        ok: data.ok === true && response.ok,
                        message: errors.length ? errors.join(" / ") : data.error || (data.ok ? "Windows recon pipeline updated." : `HTTP ${response.status}`)
                    };
                    await fetchDashboardState();
                } catch (err) {
                    lastWindowsReconAction = { running: false, ok: false, message: err.message };
                    updateLeftPanel(latestState || {});
                }
            }
            function renderReadouts(items) {
                if (!items.length) return '<div class="empty">No data</div>';
                return `<div class="readout-list">${items.map((item) => `
                    <div class="readout"><div class="label">${escapeHtml(item.label)}</div><div class="value">${escapeHtml(item.value)}</div></div>
                `).join("")}</div>`;
            }
            function routeCandidateData(state) {
                const payload = state?.routeCandidates;
                return payload?.candidates?.length ? payload : { selected: null, decisionNote: "Waiting for route candidate data.", candidates: [] };
            }
            function selectedRouteCandidate(payload) {
                if (!payload?.selected) return null;
                return payload?.candidates?.find((candidate) => candidate.selected || candidate.id === payload.selected) || null;
            }
            function riskColorForScore(scoreValue, fallback = "#39ff88") {
                const score = Number(scoreValue);
                if (!Number.isFinite(score)) return fallback;
                if (score >= 85) return "#ff3448";
                if (score >= 65) return "#ff8a3d";
                if (score >= 35) return "#ffd34d";
                return "#39ff88";
            }
            function riskLevelForScore(scoreValue, fallback = "pending") {
                const score = Number(scoreValue);
                if (!Number.isFinite(score)) return fallback;
                if (score >= 85) return "critical";
                if (score >= 65) return "high";
                if (score >= 35) return "mid";
                return "low";
            }
            function renderRouteComparison(state) {
                const payload = routeCandidateData(state);
                const selected = selectedRouteCandidate(payload);
                const metricHtml = (label, scoreValue, displayValue, levelValue = "pending") => {
                    const hasScore = scoreValue !== null && scoreValue !== undefined && Number.isFinite(Number(scoreValue));
                    const score = hasScore ? Math.max(0, Math.min(100, Number(scoreValue))) : 0;
                    const level = ["low", "mid", "high", "critical", "pending"].includes(levelValue) ? levelValue : "pending";
                    const value = displayValue ?? (hasScore ? numberText(score, 0) : "AI");
                    return `
                        <div class="route-factor factor-${level}">
                            <span>${escapeHtml(label)}</span>
                            <div class="route-meter"><span style="--score:${score}%"></span></div>
                            <span class="route-value">${escapeHtml(value)}</span>
                        </div>
                    `;
                };
                const cards = (payload.candidates || []).map((candidate) => {
                    const isSelected = candidate.selected || (payload.selected && candidate.id === payload.selected);
                    const color = escapeHtml(candidate.riskColor || riskColorForScore(candidate.riskScore, candidate.color || "#39ff88"));
                    const factors = Array.isArray(candidate.factors) ? candidate.factors : [];
                    const factorHtml = factors.map((factor) => {
                        return metricHtml(factor.label, factor.score, factor.value, factor.level);
                    }).join("");
                    return `
                        <div class="route-card ${isSelected ? "selected" : ""}">
                            <div class="route-head">
                                <div class="route-name" style="color:${color}">${escapeHtml(candidate.name || candidate.id)}</div>
                                <div class="route-chip" style="color:${color}">${escapeHtml(candidate.role || candidate.id)}</div>
                            </div>
                            <div class="route-summary">${escapeHtml(candidate.summary || "-")}</div>
                            ${metricHtml("SCORE", candidate.riskScore, candidate.riskLabel || null, candidate.scoreLevel || (isSelected ? "low" : "pending"))}
                            ${factorHtml}
                        </div>
                    `;
                }).join("");
                const title = selected
                    ? `${selected.side || selected.id} ROUTE SELECTED`
                    : (payload.candidates || []).length ? "ROUTE RISK ASSESSMENT" : "AI DECISION PENDING";
                return `
                    <div class="route-compare">
                        <div class="route-decision">
                            <strong>${escapeHtml(title)}</strong>
                            ${escapeHtml(payload.decisionNote || "Route comparison is waiting for candidate data.")}
                        </div>
                        ${cards || '<div class="empty">No route candidates</div>'}
                    </div>
                `;
            }
            function updateHeader(state) {
                const yolo = state?.yolo || {};
                const bridge = state?.bridge || {};
                const liveView = state?.liveView || {};
                byId("modeValue").textContent = safe(state?.mode, "monitor").toString().toUpperCase();
                const yoloValue = byId("yoloValue");
                const yoloError = yolo.error || yolo.importError || yolo.loadError;
                const yoloLoaded = yolo.ready === true || yolo.loaded === true;
                yoloValue.textContent = yoloError ? "ERROR" : yoloLoaded ? "READY" : "WAIT";
                setStatusClass(yoloValue, yoloError ? "status-error" : yoloLoaded ? "status-ok" : "status-warn");
                const rosValue = byId("rosValue");
                const rosReady = !bridge.error && !!bridge.latest;
                rosValue.textContent = bridge.error ? "ERROR" : rosReady ? "CONNECTED" : "WAITING";
                setStatusClass(rosValue, bridge.error ? "status-error" : rosReady ? "status-ok" : "status-warn");
                byId("timeValue").textContent = new Date((state?.serverTime || Date.now() / 1000) * 1000).toLocaleTimeString();
                const statusValue = byId("statusValue");
                const hasFrame = Number(liveView.latestFrameSeq || 0) > 0;
                statusValue.textContent = lastFetchOk ? (hasFrame ? "LIVE" : "NO FRAME") : "API ERROR";
                setStatusClass(statusValue, lastFetchOk ? (hasFrame ? "status-ok" : "status-warn") : "status-error");
                updateFeedStatus(state);
            }
            function riskBandColor(band) {
                return ({ low: "#39ff88", medium: "#ffd34d", high: "#ff8a3d", critical: "#ff3448" })[band] || "#5a6b62";
            }
            function riskOk(v) { return v === true ? "✅" : v === false ? "❌" : "—"; }
            function renderExposureBars(profile, color) {
                if (!Array.isArray(profile) || !profile.length) {
                    return '<div style="color:#5a6b62;font-size:11px;">노출 프로파일 없음</div>';
                }
                const bars = profile.map((p) => {
                    const ratio = Math.max(0, Math.min(1, Number(p.exposed) || 0));
                    const h = Math.max(2, Math.round(ratio * 26));
                    return `<span title="pos ${safe(p.pos)}: ${safe(p.exposed)}" style="display:inline-block;width:4px;height:${h}px;background:${color};margin-right:1px;vertical-align:bottom;opacity:${0.35 + 0.65 * ratio};"></span>`;
                }).join("");
                return `<div style="height:28px;line-height:0;margin-top:3px;" title="경로 출발→목적지 구간별 노출">${bars}</div>`;
            }
            function renderRiskPanel(state) {
                const cmp = state?.riskComparison;
                const feat = state?.riskFeatures;
                if (!cmp) {
                    return '<div class="empty">RISK 비교 대기 — generate_recon_report.py + route_risk_node 실행 후 표시.</div>';
                }
                const w = cmp.winner || {};
                const per = cmp.per_route || {};
                let html = '';

                // 블록 1: 정찰 결과 요약
                if (feat) {
                    html += '<div style="margin-bottom:10px;"><div style="font-weight:700;color:#9ec5f0;margin-bottom:4px;">정찰 결과</div>';
                    for (const r of ["A", "B"]) {
                        const f = feat[`route_${r}`];
                        if (!f) continue;
                        const th = f.threat || {};
                        const eff = f.efficiency || {};
                        const reached = f.reached ? "✅도착" : "❌미도착";
                        const bc = th.by_class || {};
                        const bcStr = Object.keys(bc).length ? Object.entries(bc).map(([k, v]) => `${k}${v}`).join("/") : "0";
                        const near = (th.nearest_dist_m === null || th.nearest_dist_m === undefined) ? "—" : `${numberText(th.nearest_dist_m, 0)}m`;
                        html += `<div style="font-size:12px;margin:2px 0;"><b>route_${r}</b> ${reached} · ${numberText(eff.distance_m, 0)}m · 확정위협 ${safe(th.confirmed_count, 0)}(${escapeHtml(bcStr)}) · 최근접 ${near}</div>`;
                    }
                    html += '</div>';
                }

                // 블록 2: 수식 vs LLM 한눈
                html += '<div style="margin-bottom:10px;"><div style="font-weight:700;color:#9ec5f0;margin-bottom:4px;">수식 vs LLM</div>';
                html += `<div style="font-size:12px;margin-bottom:6px;">선택 — 수식 <b>${safe(w.formula)}</b> / LLM <b>${safe(w.llm)}</b> ${riskOk(w.agreement)} · 순위일치 ${riskOk(cmp.rank_agreement)}</div>`;
                for (const r of ["A", "B"]) {
                    const pr = per[r];
                    if (!pr) continue;
                    const f = pr.formula || {};
                    const l = pr.llm || {};
                    const rt = (typeof f.risk_total === "number") ? f.risk_total : null;
                    const fcol = riskBandColor(f.band);
                    const lcol = riskBandColor(l.risk_level);
                    const gw = rt === null ? 0 : Math.round(rt * 100);
                    const exp = (feat && feat[`route_${r}`] && feat[`route_${r}`].exposure) || {};
                    html += `<div style="margin:4px 0 8px 0;">
                        <div style="font-size:12px;"><b>route_${r}</b> ${riskOk(pr.band_match)}</div>
                        <div style="height:8px;background:#0c2417;border-radius:3px;overflow:hidden;margin:3px 0;"><div style="height:100%;width:${gw}%;background:${fcol};"></div></div>
                        <div style="font-size:11px;">수식 ${rt === null ? "—" : rt.toFixed(3)} <span style="color:${fcol}">[${(f.band || "—").toUpperCase()}]</span> · LLM <span style="color:${lcol}">[${(l.risk_level || "—").toUpperCase()}]</span></div>
                        ${renderExposureBars(exp.profile, fcol)}
                    </div>`;
                }
                html += '</div>';

                // 블록 3: 근거/발산
                const n = cmp.narrative || {};
                html += '<div><div style="font-weight:700;color:#9ec5f0;margin-bottom:4px;">판단 근거</div>';
                html += `<div style="font-size:11px;margin:2px 0;"><b>수식</b>: ${escapeHtml(safe(n.formula_reason))}</div>`;
                html += `<div style="font-size:11px;margin:2px 0;"><b>LLM</b>: ${escapeHtml(safe(n.llm_decision_reason || n.llm_summary))}</div>`;
                if (Array.isArray(cmp.divergence) && cmp.divergence.length) {
                    const dv = cmp.divergence.map((d) => `route_${d.route} 수식 ${d.formula_band}↔LLM ${d.llm_band}`).join(", ");
                    html += `<div style="font-size:11px;margin:4px 0;color:#ffd34d;">발산: ${escapeHtml(dv)}</div>`;
                }
                html += '</div>';
                return html;
            }
            function fmtCmd(cmd) {
                if (!cmd || typeof cmd !== "object") return "STOP";
                const pick = (v) => (v && typeof v === "object") ? (v.command ?? v.dir ?? "") : (v ?? "");
                const ws = pick(cmd.moveWS), ad = pick(cmd.moveAD);
                const parts = [ws, ad].filter(Boolean).join(" ");
                return (parts || "STOP") + (cmd.fire ? " · FIRE" : "");
            }
            function normDeg(v) {
                v = Number(v);
                if (!Number.isFinite(v)) return 0;
                return ((v % 360) + 540) % 360 - 180;  // 0~360 → -180~180 (0=수평)
            }
            function t3dCube(w, h, d, cls) {
                const f = (W, H, tf) => `<div class="t3d-face ${cls}" style="width:${W}px;height:${H}px;transform:translate(-50%,-50%) ${tf}"></div>`;
                return f(w, h, `translateZ(${d / 2}px)`)
                     + f(w, h, `rotateY(180deg) translateZ(${d / 2}px)`)
                     + f(d, h, `rotateY(90deg) translateZ(${w / 2}px)`)
                     + f(d, h, `rotateY(-90deg) translateZ(${w / 2}px)`)
                     + f(w, d, `rotateX(90deg) translateZ(${h / 2}px)`)
                     + f(w, d, `rotateX(-90deg) translateZ(${h / 2}px)`);
            }
            function renderTank3d(pitch, roll, yaw, tYaw, tPitch) {
                const hull = `<div class="t3d-grp">${t3dCube(60, 15, 38, "")}</div>`;
                const barrel = `<div class="t3d-grp" style="transform:translateZ(11px) rotateX(${-tPitch}deg)">`
                    + `<div class="t3d-grp" style="transform:translateZ(16px)">${t3dCube(5, 5, 30, "barrel")}</div></div>`;
                const turret = `<div class="t3d-grp" style="transform:translateY(-15px) rotateY(${tYaw}deg)">`
                    + t3dCube(28, 13, 22, "turret") + barrel + `</div>`;
                return `<div class="tank3d-view">`
                    + `<div class="tank3d" style="transform:rotateY(${yaw}deg) rotateX(${-pitch}deg) rotateZ(${roll}deg)">`
                    + hull + turret + `</div></div>`;
            }
            function renderTankState(state) {
                const latest = latestBridge(state);
                const sim = latest.sim_status || {};
                const ps = latest.player_state || {};
                const body = ps.body || {};
                const turret = ps.turret || {};
                const pose = ps.pose_map || latest.player_pose_map || {};
                const cmd = (latest.get_action_response || {}).command;
                const speed = (ps.speed ?? sim.player_speed);
                const health = (ps.health ?? sim.player_health);
                const pitch = normDeg(body.z), roll = normDeg(body.y), yaw = normDeg(body.x);
                const tYaw = normDeg((Number(turret.x) || 0) - (Number(body.x) || 0)), tPitch = normDeg(turret.y);
                const posTxt = (pose && pose.x != null) ? `${numberText(pose.x,0)}, ${numberText(pose.y,0)}` : "-";
                const hud =
                    `<div class="t3d-hud tl"><div class="k">속도 m/s</div><div class="v big">${numberText(speed,2)}</div></div>`
                  + `<div class="t3d-hud tr"><div class="k">체력</div><div class="v big">${numberText(health,0)}</div></div>`
                  + `<div class="t3d-hud bl"><div class="k">자세 P/R/Y°</div><div class="v">${numberText(pitch,1)} / ${numberText(roll,1)} / ${numberText(yaw,1)}</div></div>`
                  + `<div class="t3d-hud br"><div class="k">포탑 H/P°</div><div class="v">${numberText(turret.x,1)} / ${numberText(turret.y,1)}</div></div>`
                  + `<div class="t3d-hud bc">제어 <b>${escapeHtml(fmtCmd(cmd))}</b> · 위치 <b>${posTxt}</b> · <b>${numberText(sim.sim_time,0)}s</b> / <b>${numberText(sim.distance,0)}m</b></div>`;
                const stage = `<div class="tank3d-stage">${renderTank3d(pitch, roll, yaw, tYaw, tPitch)}${hud}</div>`;
                return `<div class="sec-title">전차 상태</div>${stage}`;
            }
            function renderSystem(state) {
                const m = (state?.sensor || {}).systemMetrics || {};
                const bar = (label, pct, extra) => {
                    const p = Number.isFinite(Number(pct)) ? Math.max(0, Math.min(100, Number(pct))) : null;
                    const col = p===null ? "#5a6b62" : (p>85?"#ff3448":p>60?"#ffd34d":"#39ff88");
                    return `<div class="tele"><div class="label">${escapeHtml(label)} ${p===null?"(N/A)":(p.toFixed(0)+"%")}${extra?(" · "+escapeHtml(extra)):""}</div><div class="bar"><span style="width:${p||0}%;background:${col}"></span></div></div>`;
                };
                let html = '<div class="sec-title">시스템 자원</div><div class="tele-grid" style="grid-template-columns:1fr;">';
                html += bar("CPU", m.cpuPercent);
                html += bar("MEM", m.memoryPercent, (m.memoryUsedMb!=null)?`${numberText(m.memoryUsedMb,0)}/${numberText(m.memoryTotalMb,0)}MB`:"");
                if (m.gpuAvailable) {
                    const gmem = (m.gpuMemoryTotalMb)? (100*m.gpuMemoryUsedMb/m.gpuMemoryTotalMb):null;
                    html += bar("GPU", m.gpuPercent);
                    html += bar("GPU MEM", gmem, `${numberText(m.gpuMemoryUsedMb,0)}/${numberText(m.gpuMemoryTotalMb,0)}MB`);
                } else {
                    html += `<div class="tele"><div class="label">GPU</div><div class="value" style="font-size:12px;color:#5a6b62;">없음(이 PC)</div></div>`;
                }
                html += '</div>';
                return html;
            }
            function updateLeftPanel(state) {
                // C2 4분할: 패널 ③(전차상태+시스템)·④(LLM/위험도) 렌더. 아래 legacy 5탭 코드는 unreachable.
                byId("tankSystemContent").innerHTML = renderTankState(state) + renderSystem(state);
                byId("riskContent").innerHTML = renderRiskPanel(state);
                return;
                const latest = latestBridge(state);
                const yolo = state?.yolo || {};
                const liveView = state?.liveView || {};
                const sensor = state?.sensor || {};
                if (activeTab === "route") {
                    byId("leftContent").innerHTML = renderRouteComparison(state);
                    return;
                }
                if (activeTab === "risk") {
                    byId("leftContent").innerHTML = renderRiskPanel(state);
                    return;
                }
                if (activeTab === "ai") {
                    const ai = state?.aiLog || latest.ai_log || latest.llm_log || latest.decision;
                    const values = Array.isArray(ai) ? ai : ai ? [ai] : [];
                    byId("leftContent").innerHTML = values.length
                        ? renderReadouts(values.slice(-8).map((entry, index) => ({
                            label: `AI ${index + 1}`,
                            value: typeof entry === "string" ? entry : JSON.stringify(entry)
                        })))
                        : '<div class="empty">AI explanation is not connected yet.</div>';
                    return;
                    /*
                    const entry = Array.isArray(ai) ? ai[ai.length - 1] : ai;
                    if (!entry) {
                        byId("leftContent").innerHTML = '<div class="empty">AI explanation is not connected yet.</div>';
                        return;
                    }
                    if (typeof entry === "string") {
                        byId("leftContent").innerHTML = renderReadouts([{ label: "AI", value: entry }]);
                        return;
                    }
                    const res = entry.result || {};
                    const rl = res.risk_level || {};
                    const rb = res.recommended_behavior || {};
                    const kr = res.key_risks || {};
                    const arr = (v) => Array.isArray(v) ? v.join(" / ") : safe(v, "-");
                    byId("leftContent").innerHTML = renderReadouts([
                        { label: "추천 루트", value: safe(res.selected_route, "-") },
                        { label: "위험도 A/B", value: `${safe(rl.A, "-")} / ${safe(rl.B, "-")}` },
                        { label: "확신도", value: safe(res.confidence, "-") },
                        { label: "요약", value: safe(res.summary || entry.summary, "-") },
                        { label: "판단 근거", value: safe(res.decision_reason, "-") },
                        { label: "속도 정책", value: safe(rb.speed_policy, "-") },
                        { label: "주의 지점", value: arr(rb.caution_points) },
                        { label: "전술 코멘트", value: safe(rb.tactical_note, "-") },
                        { label: "A 위험요인", value: arr(kr.A) },
                        { label: "B 위험요인", value: arr(kr.B) },
                    ]);
                    return;
                    */
                }
                if (activeTab === "recon") {
                    const detections = getDetections(state);
                    byId("leftContent").innerHTML = detections.length
                        ? renderReadouts(detections.slice(0, 12).map((det, index) => ({
                            label: `${safe(det.className || det.class_name || det.modelClassName, "object")} #${index + 1}`,
                            value: `conf=${numberText(det.confidence, 2)} ts=${numberText(latest.detect_result?.timestamp_wall || state?.serverTime, 3)}`
                        })))
                        : '<div class="empty">No detection event</div>';
                    return;
                }
                const playerPose = latest.player_pose_map || latest.get_action_pose_map || sensor.playerPose;
                const bridge = state?.bridge || {};
                const mapSummary = state?.staticMap || {};
                const heightSummary = mapSummary.heightSummary || staticMap?.heightSummary || {};
                const surfaceSummary = mapSummary.surfaceSummary || staticMap?.surfaceSummary || {};
                const windowsRecon = state?.windowsRecon || {};
                const reconFiles = windowsRecon.files || {};
                const fileOk = (key) => reconFiles?.[key]?.exists ? "OK" : "MISS";
                const reconMessages = Array.isArray(windowsRecon.messages) && windowsRecon.messages.length
                    ? windowsRecon.messages.join(" / ")
                    : "-";
                const actionMessage = lastWindowsReconAction
                    ? `${lastWindowsReconAction.running ? "RUNNING" : lastWindowsReconAction.ok ? "OK" : "CHECK"}: ${lastWindowsReconAction.message || "-"}`
                    : "-";
                const actionHtml = `
                    <div class="recon-actions">
                        <button class="action-button" type="button" onclick="runWindowsRecon(false)">WIN BUILD</button>
                        <button class="action-button" type="button" onclick="runWindowsRecon(true)">WIN LLM</button>
                    </div>
                `;
                byId("leftContent").innerHTML = actionHtml + renderReadouts([
                    { label: "Windows recon", value: windowsRecon.mode ? `${windowsRecon.mode} comparison=${windowsRecon.readyForComparison ? "ready" : "wait"} llm=${windowsRecon.readyForLlm ? "ready" : "wait"}` : "-" },
                    { label: "Recon files", value: `A=${fileOk("routeA")} B=${fileOk("routeB")} cmp=${fileOk("comparison")} llmIn=${fileOk("routeComparison")} risk=${fileOk("riskResult")} txt=${fileOk("txtReport")}` },
                    { label: "Recon action", value: actionMessage },
                    { label: "Recon message", value: reconMessages },
                    { label: "YOLO latest ms", value: `${numberText(yolo.latestYoloMs ?? yolo.latestDetectMs, 1)} ms` },
                    { label: "YOLO returned count", value: safe(yolo.latestReturnedDetectionCount ?? liveView.latestDetectionCount, 0) },
                    { label: "YOLO loaded", value: yolo.loaded === true ? "true" : "false" },
                    { label: "YOLO status", value: safe(yolo.status, yolo.error || yolo.importError || yolo.loadError ? "error" : "wait") },
                    { label: "YOLO error", value: safe(yolo.error || yolo.importError || yolo.loadError, "-") },
                    { label: "YOLO model", value: safe(yolo.modelPath, "-") },
                    { label: "Static map", value: mapSummary.loaded ? `${safe(mapSummary.objectCount, 0)} objects` : safe(mapSummary.error || staticMapLoadError, "loading") },
                    { label: "Elevation", value: heightSummary.sampleCount ? `y=${numberText(heightSummary.min, 1)}..${numberText(heightSummary.max, 1)} avg=${numberText(heightSummary.avg, 1)}` : "-" },
                    { label: "Surface zones", value: surfaceSummary.waterDataAvailable ? `water=${safe(surfaceSummary.waterZoneCount, 0)} ridge=${safe(surfaceSummary.rockyZoneCount, 0)}` : `low<=${numberText(surfaceSummary.lowThreshold, 1)}` },
                    { label: "ROS status", value: bridge.error ? `ERROR: ${bridge.error}` : latest && Object.keys(latest).length ? "CONNECTED" : "WAITING" },
                    { label: "Player pose", value: playerPose ? JSON.stringify(playerPose) : "-" },
                    { label: "Live view", value: `frame=${safe(liveView.latestFrameSeq, 0)} age=${numberText(liveView.latestFrameAgeMs, 1)}ms` }
                ]);
            }
            function canvasPointMapper(points, width, height) {
                const valid = points.filter(Boolean);
                if (!valid.length) return (p) => ({ x: width / 2 + (p?.x || 0), y: height / 2 - (p?.y || 0) });
                let minX = Math.min(...valid.map((p) => p.x));
                let maxX = Math.max(...valid.map((p) => p.x));
                let minY = Math.min(...valid.map((p) => p.y));
                let maxY = Math.max(...valid.map((p) => p.y));
                if (Math.abs(maxX - minX) < 20) { minX -= 10; maxX += 10; }
                if (Math.abs(maxY - minY) < 20) { minY -= 10; maxY += 10; }
                const pad = 34;
                const scale = Math.min((width - pad * 2) / (maxX - minX), (height - pad * 2) / (maxY - minY));
                return (p) => ({ x: pad + (p.x - minX) * scale, y: height - pad - (p.y - minY) * scale });
            }
            function drawSymbol(ctx, point, color, label, shape = "circle") {
                if (!point) return;
                ctx.save();
                ctx.fillStyle = color;
                ctx.strokeStyle = color;
                ctx.lineWidth = 2;
                if (shape === "diamond") {
                    ctx.beginPath();
                    ctx.moveTo(point.x, point.y - 7);
                    ctx.lineTo(point.x + 7, point.y);
                    ctx.lineTo(point.x, point.y + 7);
                    ctx.lineTo(point.x - 7, point.y);
                    ctx.closePath();
                    ctx.fill();
                } else if (shape === "square") {
                    ctx.strokeRect(point.x - 5, point.y - 5, 10, 10);
                } else {
                    ctx.beginPath();
                    ctx.arc(point.x, point.y, 6, 0, Math.PI * 2);
                    ctx.fill();
                }
                if (label) {
                    ctx.font = "10px Consolas, monospace";
                    const labelX = point.x + 10;
                    const labelY = point.y - 24;
                    const labelW = Math.ceil(ctx.measureText(label).width) + 10;
                    const labelH = 16;
                    ctx.fillStyle = "rgba(3, 8, 5, 0.72)";
                    ctx.strokeStyle = color;
                    ctx.lineWidth = 1;
                    ctx.fillRect(labelX, labelY, labelW, labelH);
                    ctx.strokeRect(labelX + 0.5, labelY + 0.5, labelW - 1, labelH - 1);
                    ctx.fillStyle = color;
                    ctx.fillText(label, labelX + 5, labelY + 11);
                }
                ctx.restore();
            }
            function drawMapTag(ctx, text, point, color, width, height) {
                if (!point || !text) return;
                ctx.save();
                ctx.font = "10px Consolas, monospace";
                const labelW = Math.ceil(ctx.measureText(text).width) + 10;
                const labelH = 16;
                const x = Math.max(4, Math.min(width - labelW - 4, point.x + 8));
                const y = Math.max(4, Math.min(height - labelH - 4, point.y - 20));
                ctx.fillStyle = "rgba(3, 8, 5, 0.74)";
                ctx.strokeStyle = color;
                ctx.lineWidth = 1;
                ctx.fillRect(x, y, labelW, labelH);
                ctx.strokeRect(x + 0.5, y + 0.5, labelW - 1, labelH - 1);
                ctx.fillStyle = color;
                ctx.fillText(text, x + 5, y + 11);
                ctx.restore();
            }
            function drawRouteCandidateOverlay(ctx, mapper, state, width, height) {
                const payload = routeCandidateData(state);
                const candidates = Array.isArray(payload.candidates) ? payload.candidates : [];
                if (!mapper || !candidates.length) return;
                const ordered = [...candidates].sort((a, b) => Number(a.selected || a.id === payload.selected) - Number(b.selected || b.id === payload.selected));
                for (const candidate of ordered) {
                    const isSelected = candidate.selected || (payload.selected && candidate.id === payload.selected);
                    const routePoints = Array.isArray(candidate.points)
                        ? candidate.points.map(readPoint).filter(Boolean)
                        : [];
                    if (routePoints.length < 2) continue;
                    const points = routePoints.map(mapper);
                    const color = candidate.color || (isSelected ? "#39ff88" : "#44d9ff");
                    ctx.save();
                    ctx.lineJoin = "round";
                    ctx.lineCap = "round";
                    ctx.globalAlpha = isSelected ? 0.92 : 0.66;
                    ctx.strokeStyle = "rgba(0, 0, 0, 0.78)";
                    ctx.lineWidth = isSelected ? 6.5 : 5;
                    ctx.beginPath();
                    points.forEach((p, index) => index === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y));
                    ctx.stroke();
                    ctx.strokeStyle = color;
                    ctx.lineWidth = isSelected ? 3.2 : 2.2;
                    ctx.setLineDash(isSelected ? [] : [7, 5]);
                    ctx.beginPath();
                    points.forEach((p, index) => index === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y));
                    ctx.stroke();
                    ctx.restore();
                    const labelPoint = points[Math.max(1, Math.floor(points.length * 0.45))];
                    const label = isSelected ? `${candidate.side || candidate.id} SELECTED` : `${candidate.side || candidate.id} ${candidate.id || ""}`.trim();
                    drawMapTag(ctx, label, labelPoint, color, width, height);
                }
                const selected = selectedRouteCandidate(payload);
                const start = readPoint(payload.start || selected?.points?.[0]);
                const destination = readPoint(payload.destination || selected?.points?.[selected?.points?.length - 1]);
                drawSymbol(ctx, start ? mapper(start) : null, "#d8ffe9", "", "circle");
                drawSymbol(ctx, destination ? mapper(destination) : null, "#ffca4f", "", "diamond");
            }
            function drawStaticObject(ctx, point, category) {
                if (!point) return;
                const colors = {
                    tree: "#39ff88",
                    rock: "#d8ffe9",
                    house: "#b084ff",
                    human: "#ffca4f",
                    car: "#ff8c00",
                    tank: "#ff5b64",
                    unknown: "#9ad8b4"
                };
                const color = colors[category] || colors.unknown;
                ctx.save();
                ctx.fillStyle = color;
                ctx.strokeStyle = color;
                ctx.globalAlpha = category === "tree" ? 0.58 : 0.86;
                if (category === "house" || category === "car") {
                    ctx.strokeRect(point.x - 3.5, point.y - 3.5, 7, 7);
                } else if (category === "rock") {
                    ctx.beginPath();
                    ctx.arc(point.x, point.y, 3.2, 0, Math.PI * 2);
                    ctx.stroke();
                } else {
                    ctx.fillRect(point.x - 2, point.y - 2, 4, 4);
                }
                ctx.restore();
            }
            function drawRosMapBase(ctx, mapper, mapData) {
                const bounds = mapBoundsFromMap(mapData);
                const rect = screenRectFromBounds(mapper, bounds);
                ctx.save();
                const usedTexture = drawOverviewTexture(ctx, rect, "ros");
                if (!usedTexture) {
                    ctx.fillStyle = "rgba(4, 12, 8, 0.88)";
                    ctx.fillRect(rect.x, rect.y, rect.w, rect.h);
                }
                ctx.beginPath();
                ctx.rect(rect.x, rect.y, rect.w, rect.h);
                ctx.clip();
                ctx.strokeStyle = "rgba(57, 255, 136, 0.12)";
                ctx.lineWidth = 1;
                const step = 30;
                for (let x = Math.ceil(bounds.minX / step) * step; x <= bounds.maxX; x += step) {
                    const p0 = mapper({ x, y: bounds.minY });
                    const p1 = mapper({ x, y: bounds.maxY });
                    ctx.beginPath();
                    ctx.moveTo(p0.x, p0.y);
                    ctx.lineTo(p1.x, p1.y);
                    ctx.stroke();
                }
                for (let y = Math.ceil(bounds.minY / step) * step; y <= bounds.maxY; y += step) {
                    const p0 = mapper({ x: bounds.minX, y });
                    const p1 = mapper({ x: bounds.maxX, y });
                    ctx.beginPath();
                    ctx.moveTo(p0.x, p0.y);
                    ctx.lineTo(p1.x, p1.y);
                    ctx.stroke();
                }
                if (!usedTexture) {
                    drawRockyZones(ctx, mapper, mapData);
                    drawWaterZones(ctx, mapper, mapData);
                    drawPassageZones(ctx, mapper, mapData, "ros");
                }
                ctx.restore();
                ctx.save();
                ctx.fillStyle = "rgba(216, 255, 233, 0.82)";
                ctx.font = "11px Consolas, monospace";
                ctx.fillText("ROS MAP", rect.x + 10, rect.y + 18);
                ctx.restore();
            }
            function drawPanelGrid(ctx, width, height) {
                ctx.fillStyle = "#050806";
                ctx.fillRect(0, 0, width, height);
                ctx.strokeStyle = "rgba(57,255,136,0.16)";
                ctx.lineWidth = 1;
                for (let x = 0; x < width; x += 28) {
                    ctx.beginPath();
                    ctx.moveTo(x, 0);
                    ctx.lineTo(x, height);
                    ctx.stroke();
                }
                for (let y = 0; y < height; y += 28) {
                    ctx.beginPath();
                    ctx.moveTo(0, y);
                    ctx.lineTo(width, y);
                    ctx.stroke();
                }
            }
            function drawRosStatus(ctx, bridge, latest, width) {
                const routeCount = Object.keys(routeCounts({ bridge })).length;
                const hasLatest = latest && Object.keys(latest).length > 0;
                const connected = !bridge.error && bridge.available !== false;
                const text = connected ? "ROS CONNECTED" : hasLatest ? "ROS DATA / BRIDGE FALLBACK" : "ROS WAITING";
                ctx.save();
                ctx.textAlign = "right";
                ctx.font = "11px Consolas, monospace";
                ctx.fillStyle = connected ? "rgba(57,255,136,0.9)" : hasLatest ? "rgba(255,202,79,0.88)" : "rgba(255,91,100,0.86)";
                ctx.fillText(text, width - 14, 24);
                if (bridge.error) {
                    ctx.fillStyle = "rgba(255,202,79,0.82)";
                    ctx.fillText(String(bridge.error).slice(0, 46), width - 14, 40);
                } else if (routeCount) {
                    ctx.fillStyle = "rgba(116,169,140,0.9)";
                    ctx.fillText(`routes ${routeCount}`, width - 14, 40);
                }
                ctx.restore();
            }
            function drawMap(state) {
                if (activeMapTab !== "terrain") return;  // ros=그래프, rviz=iframe → 캔버스는 terrain 전용
                try {
                    const canvas = byId("mapCanvas");
                    const rect = canvas.getBoundingClientRect();
                    const dpr = window.devicePixelRatio || 1;
                    const width = Math.max(1, Math.floor(rect.width * dpr));
                    const height = Math.max(1, Math.floor(rect.height * dpr));
                    if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height; }
                    const ctx = canvas.getContext("2d");
                    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
                    const w = rect.width;
                    const h = rect.height;
                    ctx.clearRect(0, 0, w, h);
                    drawPanelGrid(ctx, w, h);
                    const staticObjects = Array.isArray(staticMap?.obstacles)
                        ? staticMap.obstacles.map(readStaticObjectPoint).filter(Boolean)
                        : [];
                    const staticPoint = staticMap ? staticMapMapper(w, h, staticMap) : null;
                    if (staticPoint) {
                        if (activeMapTab === "terrain") {
                            drawTopographicLayer(ctx, staticPoint, staticObjects, staticMap, w, h);
                            for (const obj of staticObjects) drawStaticObject(ctx, staticPoint(obj), obj.category);
                        } else {
                            drawRosMapBase(ctx, staticPoint, staticMap);
                        }
                    } else if (staticMapLoadError) {
                        ctx.fillStyle = "#ff5b64";
                        ctx.font = "12px Consolas, monospace";
                        ctx.fillText(`STATIC MAP ERROR: ${staticMapLoadError.slice(0, 54)}`, 18, 28);
                    }
                    const bridge = state?.bridge || {};
                    const latest = latestBridge(state);
                    const player = readPoint(latest.player_pose_map || latest.get_action_pose_map || latest.info_compact?.player_pose_map);
                    const enemy = readPoint(latest.enemy_pose_map || latest.info_compact?.enemy_pose_map);
                    const destination = readPoint(latest.destination?.pose_map || latest.destination?.pose_raw || latest.goal || latest.target);
                    const obstacles = extractArray(latest.obstacles).map(readPoint).filter(Boolean);
                    const route = extractArray(latest.route || latest.path || latest.planned_route).map(readPoint).filter(Boolean);
                    const routeCandidates = Array.isArray(state?.routeCandidates?.candidates) ? state.routeCandidates.candidates : [];
                    const routeCandidatePoints = routeCandidates.flatMap((candidate) =>
                        Array.isArray(candidate.points) ? candidate.points.map(readPoint).filter(Boolean) : []
                    );
                    const detections = getDetections(state);
                    const imageInfo = state?.liveView?.latestDetectionMetadata?.image || {};
                    const frameShape = state?.liveView?.latestFrameShape || state?.yolo?.latestFrameShape || [];
                    const imageW = Number(imageInfo.width || frameShape[1] || 1920);
                    const imageH = Number(imageInfo.height || frameShape[0] || 1080);
                    const detectionContacts = detections.map((det, index) => {
                        const box = det?.bbox;
                        if (!Array.isArray(box) || box.length < 4) return null;
                        const x = (Number(box[0]) + Number(box[2])) * 0.5;
                        const y = (Number(box[1]) + Number(box[3])) * 0.5;
                        if (!Number.isFinite(x) || !Number.isFinite(y)) return null;
                        return {
                            x,
                            y,
                            label: safe(det.className || det.class_name || det.modelClassName, `OBJ${index + 1}`),
                            confidence: Number(det.confidence || 0)
                        };
                    }).filter(Boolean);
                    const mapPoint = staticPoint || canvasPointMapper([player, enemy, destination, ...obstacles, ...route, ...routeCandidatePoints].filter(Boolean), w, h);
                    drawRouteCandidateOverlay(ctx, mapPoint, state, w, h);
                    if (activeMapTab === "ros" && route.length >= 2) {
                        ctx.strokeStyle = "#ffca4f";
                        ctx.lineWidth = 2;
                        ctx.beginPath();
                        route.map(mapPoint).forEach((p, index) => index === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y));
                        ctx.stroke();
                    } else {
                        if (activeMapTab === "ros" && !staticPoint) {
                            ctx.fillStyle = bridge.error ? "#ff5b64" : "rgba(255,202,79,0.75)";
                            ctx.font = "16px Consolas, monospace";
                            ctx.fillText(bridge.error ? "ROS UNAVAILABLE" : "NO ROUTE", 18, 28);
                            if (bridge.error) {
                                ctx.fillStyle = "rgba(255,202,79,0.9)";
                                ctx.font = "12px Consolas, monospace";
                                ctx.fillText(String(bridge.error).slice(0, 62), 18, 48);
                            }
                        }
                    }
                    if (activeMapTab === "ros" && !player && !enemy && !destination && !obstacles.length && !route.length && detectionContacts.length) {
                        const pad = 34;
                        const classColors = { house: "#b084ff", person: "#39ff88", tank: "#ff5b64", rock: "#ffca4f", car: "#ff8c00" };
                        ctx.fillStyle = "#44d9ff";
                        ctx.font = "13px Consolas, monospace";
                        ctx.fillText("YOLO CONTACTS", 18, bridge.error ? (staticPoint ? 92 : 70) : (staticPoint ? 70 : 52));
                        for (const contact of detectionContacts.slice(0, 10)) {
                            const point = {
                                x: pad + (contact.x / Math.max(1, imageW)) * (w - pad * 2),
                                y: pad + (contact.y / Math.max(1, imageH)) * (h - pad * 2)
                            };
                            const cls = String(contact.label).toLowerCase();
                            drawSymbol(ctx, point, classColors[cls] || "#39ff88", `${contact.label} ${numberText(contact.confidence, 2)}`, cls === "house" ? "square" : "circle");
                        }
                    }
                    if (activeMapTab === "ros") {
                        for (const obstacle of obstacles) drawSymbol(ctx, mapPoint(obstacle), "#6aa884", "OBS", "square");
                        drawSymbol(ctx, destination ? mapPoint(destination) : null, "#ffca4f", "TARGET", "diamond");
                    }
                    drawSymbol(ctx, enemy ? mapPoint(enemy) : null, "#ff5b64", "ENEMY", "circle");
                    if (player) {
                        const selfPoint = mapPoint(player);
                        drawSymbol(ctx, selfPoint, "#39ff88", `SELF ${numberText(player.x, 1)},${numberText(player.y, 1)}`, "circle");
                    } else {
                        ctx.fillStyle = "rgba(57,255,136,0.82)";
                        ctx.font = "12px Consolas, monospace";
                        ctx.fillText("SELF WAITING: /info or /get_action", 18, staticPoint ? 112 : 92);
                    }
                    if (activeMapTab === "ros") drawRosStatus(ctx, bridge, latest, w);
                } catch (err) {
                    const canvas = byId("mapCanvas");
                    const ctx = canvas.getContext("2d");
                    ctx.fillStyle = "#050806";
                    ctx.fillRect(0, 0, canvas.width, canvas.height);
                    ctx.fillStyle = "#ff5b64";
                    ctx.font = "14px Consolas, monospace";
                    ctx.fillText("MAP ERROR", 16, 26);
                }
            }
            function updateBottomStatus(state) {
                const yolo = state?.yolo || {};
                const latest = latestBridge(state);
                const detections = getDetections(state);
                const counts = routeCounts(state);
                byId("bottomYolo").textContent = `${numberText(yolo.latestYoloMs ?? yolo.latestDetectMs, 1)} ms`;
                byId("bottomObjects").textContent = safe(yolo.latestReturnedDetectionCount ?? detections.length, 0);
                byId("bottomCache").textContent = safe(yolo.latestDetectCached ?? latest.detect_result?.yolo_cached, "-");
                byId("bottomRoute").textContent = `/detect ${safe(counts["/detect"], 0)} /info ${safe(counts["/info"], 0)}`;
                byId("bottomWarning").textContent = state?.bridge?.error || state?.yolo?.error || state?.liveView?.latestError || "-";
            }
            async function fetchDashboardState() {
                try {
                    const response = await fetch("/api/dashboard/state", { cache: "no-store" });
                    if (!response.ok) throw new Error(`HTTP ${response.status}`);
                    latestState = await response.json();
                    lastFetchOk = true;
                    updateHeader(latestState);
                    updateLeftPanel(latestState);
                    drawMap(latestState);
                    if (activeMapTab === "ros") renderRosActive(latestState);
                    updateBottomStatus(latestState);
                    drawFeedOverlay(latestState);
                } catch (err) {
                    lastFetchOk = false;
                    const fallback = latestState || {};
                    fallback.bridge = fallback.bridge || {};
                    fallback.bridge.error = `API ERROR: ${err.message}`;
                    updateHeader(fallback);
                    updateLeftPanel(fallback);
                    drawMap(fallback);
                    updateBottomStatus(fallback);
                    drawFeedOverlay(fallback);
                }
            }
            async function fetchStaticMap() {
                try {
                    const response = await fetch("/api/static-map", { cache: "no-store" });
                    if (!response.ok) throw new Error(`HTTP ${response.status}`);
                    staticMap = await response.json();
                    staticMapLoadError = staticMap?.error || null;
                    staticTerrainCache = null;
                    loadOverviewImage(staticMap);
                    drawMap(latestState || {});
                } catch (err) {
                    staticMap = null;
                    staticMapLoadError = err.message;
                    staticTerrainCache = null;
                    overviewImage = null;
                    overviewImageLoaded = false;
                    overviewImageError = err.message;
                    drawMap(latestState || {});
                }
            }
            function loadOverviewImage(mapData) {
                const info = mapData?.overviewImage || {};
                if (!info.available || !info.url) {
                    overviewImage = null;
                    overviewImageLoaded = false;
                    overviewImageError = info.available === false ? "overview image not found" : null;
                    return;
                }
                const img = new Image();
                overviewImageLoaded = false;
                overviewImageError = null;
                img.onload = () => {
                    overviewImage = img;
                    overviewImageLoaded = true;
                    drawMap(latestState || {});
                };
                img.onerror = () => {
                    overviewImage = null;
                    overviewImageLoaded = false;
                    overviewImageError = "overview image failed to load";
                    drawMap(latestState || {});
                };
                img.src = `${info.url}?t=${Date.now()}`;
            }
            window.addEventListener("resize", () => {
                drawMap(latestState || {});
                drawFeedOverlay(latestState || {});
            });
            updateMapLegend();
            if (new URLSearchParams(window.location.search).get("map") === "ros") setMapTab("ros");
            fetchStaticMap();
            function toggleMax(panel) {
                if (!panel) return;
                const on = panel.classList.toggle("maximized");
                document.querySelectorAll(".main-grid > .panel").forEach((p) => { if (p !== panel) p.style.visibility = on ? "hidden" : ""; });
                setTimeout(() => {
                    try { if (cyRos) { cyRos.resize(); cyRos.fit(undefined, 8); } } catch (e) {}
                    if (activeMapTab === "terrain") drawMap(latestState || {});
                }, 60);
            }
            function initPanelMax() {
                document.querySelectorAll(".main-grid > .panel").forEach((p) => {
                    const t = p.querySelector(".panel-title");
                    if (t) { t.title = "더블클릭: 패널 확대/복원"; t.addEventListener("dblclick", () => toggleMax(p)); }
                    const b = document.createElement("button");
                    b.className = "maxbtn"; b.type = "button"; b.textContent = "⛶"; b.title = "확대 / 복원";
                    b.addEventListener("click", (e) => { e.stopPropagation(); toggleMax(p); });
                    p.appendChild(b);
                });
            }
            initPanelMax();
            fetchDashboardState();
            setInterval(fetchDashboardState, 300);
        </script>
    </body>
    </html>
    """
    # 폴링 간격을 설정값으로 주입한다(대시보드는 사람이 보는 화면이라 기본 1초로 완화).
    html = html.replace(
        "setInterval(fetchDashboardState, 300)",
        f"setInterval(fetchDashboardState, {int(poll_ms)})",
    )
    return render_template_string(html)


def generate_video_stream(web_fps: float = 20.0, jpeg_quality: int = 80):
    interval = 1.0 / max(1.0, float(web_fps))
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)] if cv2 is not None else []
    while True:
        with _state_lock:
            frame = None if _latest_frame is None else _latest_frame.copy()
            detections = deepcopy(_latest_detections)
            metadata = deepcopy(_latest_detection_metadata)
        if frame is None:
            frame = _blank_frame()
        elif not _LIVE_VIEW_BROWSER_OVERLAY:
            frame = _draw_detections(frame, detections, metadata)
        if cv2 is not None:
            ok, buffer = cv2.imencode(".jpg", frame, encode_params)
            if ok:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        time.sleep(interval)


def video_response(web_fps: float = 20.0, jpeg_quality: int = 80) -> Response:
    return Response(generate_video_stream(web_fps=web_fps, jpeg_quality=jpeg_quality), mimetype="multipart/x-mixed-replace; boundary=frame")


def debug_state() -> Dict[str, Any]:
    with _state_lock:
        frame_age = time.time() - _latest_frame_timestamp if _latest_frame_timestamp else None
        det_age = time.time() - _latest_detection_timestamp if _latest_detection_timestamp else None
        return {
            "enabled": True,
            "opencvAvailable": cv2 is not None,
            "latestFrameSeq": _latest_frame_seq,
            "latestFrameShape": deepcopy(_latest_frame_shape),
            "latestSourceFrameShape": deepcopy(_latest_source_frame_shape),
            "liveViewDecodeFps": _LIVE_VIEW_DECODE_FPS,
            "liveViewMaxSide": _LIVE_VIEW_MAX_SIDE,
            "latestLiveDecodeMs": _latest_live_decode_ms,
            "skippedLiveDecodeCount": _skipped_live_decode_count,
            "latestFrameAgeMs": None if frame_age is None else frame_age * 1000.0,
            "latestDetectionCount": len(_latest_detections),
            "latestDetections": deepcopy(_latest_detections[:10]),
            "latestDetectionAgeMs": None if det_age is None else det_age * 1000.0,
            "latestDetectionMetadata": deepcopy(_latest_detection_metadata),
            "latestError": _latest_error,
            "rawStream": _LIVE_VIEW_BROWSER_OVERLAY,
        }
