# -*- coding: utf-8 -*-
"""Legacy Team TankSimulation YOLO helper, adapted for the `vision` ROS2 package.

This module is intentionally separate from `vision.yolo_detector` so the active
/detect bridge keeps using the optimized TensorRT/ONNX/PT detector. It is useful
for offline tests or quick comparisons with the team's original PIL-based path.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Dict, List

from PIL import Image

_model = None


def _default_weights_path() -> Path:
    try:
        from ament_index_python.packages import get_package_share_directory
        return Path(get_package_share_directory("vision")) / "models" / "best_300.pt"
    except Exception:
        return Path(__file__).resolve().parents[1] / "models" / "best_300.pt"


def _get_model(weights_path: str | None = None):
    global _model
    if _model is not None:
        return _model
    from ultralytics import YOLO
    path = Path(weights_path).expanduser() if weights_path else _default_weights_path()
    if not path.exists():
        raise FileNotFoundError(f"YOLO 가중치 파일이 없습니다: {path}")
    _model = YOLO(str(path))
    return _model


def _decode_image(raw_data) -> Image.Image:
    if isinstance(raw_data, str):
        img_bytes = base64.b64decode(raw_data)
    elif isinstance(raw_data, bytes):
        img_bytes = raw_data
    else:
        raise ValueError("image 필드의 타입이 올바르지 않습니다.")
    return Image.open(io.BytesIO(img_bytes))


def _format_single_result(box, class_name: str, conf: float) -> dict:
    x1, y1, x2, y2 = [int(v) for v in box]
    return {
        "className": class_name,
        "bbox": [x1, y1, x2, y2],
        "confidence": round(float(conf), 4),
        "color": "#00FF00",
        "filled": False,
        "updateBoxWhileMoving": False,
    }


def run_inference(image_data, weights_path: str | None = None) -> List[Dict]:
    model = _get_model(weights_path)
    img = _decode_image(image_data)
    results = model(img, verbose=False)
    detections: List[Dict] = []
    if not results:
        return detections
    for r in results:
        boxes = r.boxes
        if boxes is None:
            continue
        for i in range(len(boxes)):
            conf = float(boxes.conf[i])
            cls_id = int(boxes.cls[i])
            class_name = model.names.get(cls_id, str(cls_id))
            box = boxes.xyxy[i].tolist()
            detections.append(_format_single_result(box, class_name, conf))
    return detections
