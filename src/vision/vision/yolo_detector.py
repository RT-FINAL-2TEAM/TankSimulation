# -*- coding: utf-8 -*-
"""
YOLO detector wrapper for Tank Challenge.

This module is intentionally independent from Flask and ROS2 so it can be used by:
- ros_bridge.app_routes /detect endpoint
- a standalone debug server
- future ROS2 perception nodes

The detector returns the official Tank Challenge /detect response format:
[
  {
    "className": "person",
    "bbox": [x1, y1, x2, y2],
    "confidence": 0.85,
    "color": "#00FFFF",
    "filled": false,
    "updateBoxWhileMoving": false
  }
]
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import torch
import yaml

try:
    from ament_index_python.packages import get_package_share_directory
except Exception:  # pragma: no cover - only unavailable outside ROS2 runtime
    get_package_share_directory = None

from vision.config import DEFAULT_CLASS_COLORS, DEFAULT_CONFIG_FILENAME, DEFAULT_MODEL_FILENAME


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on", "y"}


def _load_yaml(path: Optional[Path]) -> Dict[str, Any]:
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
    return loaded if isinstance(loaded, dict) else {}


def _deep_get(data: Dict[str, Any], keys: Iterable[str], default: Any) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _resolve_package_share_file(package_name: str, *parts: str) -> Optional[Path]:
    if get_package_share_directory is None:
        return None
    try:
        return Path(get_package_share_directory(package_name), *parts)
    except Exception:
        return None


def _resolve_source_file(*parts: str) -> Path:
    # .../vision/vision/yolo_detector.py
    # source root package dir is parent.parent
    return Path(__file__).resolve().parents[1].joinpath(*parts)


def resolve_default_config_path() -> Optional[Path]:
    env_path = os.getenv("TANK_YOLO_CONFIG") or os.getenv("YOLO_CONFIG")
    if env_path:
        return Path(env_path).expanduser().resolve()
    share_path = _resolve_package_share_file("vision", "config", DEFAULT_CONFIG_FILENAME)
    if share_path and share_path.exists():
        return share_path
    source_path = _resolve_source_file("config", DEFAULT_CONFIG_FILENAME)
    return source_path if source_path.exists() else None


def resolve_default_model_path(config: Dict[str, Any]) -> Path:
    env_path = os.getenv("TANK_YOLO_MODEL_PATH") or os.getenv("YOLO_MODEL_PATH")
    if env_path:
        return Path(env_path).expanduser().resolve()

    default_filename = str(_deep_get(config, ["model", "default_filename"], DEFAULT_MODEL_FILENAME))
    share_path = _resolve_package_share_file("vision", "models", default_filename)
    if share_path and share_path.exists():
        return share_path

    source_path = _resolve_source_file("models", default_filename)
    if source_path.exists():
        return source_path
    if source_path.suffix.lower() in {".engine", ".onnx"}:
        pt_fallback = source_path.with_suffix(".pt")
        if pt_fallback.exists():
            return pt_fallback
    return source_path


def normalize_model_names(names: Any) -> Dict[int, str]:
    if isinstance(names, dict):
        return {int(class_id): str(name) for class_id, name in names.items()}
    return {class_id: str(name) for class_id, name in enumerate(names)}


@dataclass
class YoloRuntimeConfig:
    model_path: Path
    config_path: Optional[Path]
    imgsz: int = 416
    device: str = "cpu"
    use_cuda: bool = False
    half: bool = False
    iou: float = 0.70
    max_det: int = 20
    max_return: int = 5
    model_conf: float = 0.5
    fallback_model_conf: float = 0.5
    low_conf_fallback: bool = False
    return_fallback_detections: bool = True
    tracking_enabled: bool = True
    tracker: str = "bytetrack.yaml"
    track_persist: bool = True
    enable_cache: bool = True
    min_interval_sec: float = 0.12
    warmup_runs: int = 2
    aliases: Dict[str, str] = field(default_factory=lambda: {"blue": "person", "red": "person", "tank": "tank", "Tank": "tank", "car": "car", "house": "house"})
    canonical_classes: set = field(default_factory=lambda: {"rock", "person", "tank", "car", "house"})
    canonical_class_order: List[str] = field(default_factory=lambda: ["rock", "person", "tank", "car", "house"])
    ignored_classes: set = field(default_factory=set)
    class_fixed_ids: Dict[str, int] = field(default_factory=lambda: {"tank": 1, "rock": 2, "person": 3, "car": 4, "house": 5})
    class_colors: Dict[str, str] = field(default_factory=lambda: dict(DEFAULT_CLASS_COLORS))
    default_box_color: str = "#00FF00"
    default_confidence: float = 0.5
    class_confidence_thresholds: Dict[str, float] = field(default_factory=dict)
    shadow_filter_enabled: bool = False
    shadow_sigma: float = 35.0
    shadow_strength: float = 0.75
    shadow_work_scale: float = 0.35
    shadow_max_side: int = 960
    clahe_enabled: bool = False
    clahe_clip: float = 1.5
    timing_log: bool = False
    recognition_log: bool = True
    recognition_log_cache: bool = False
    recognition_log_empty: bool = False
    debug_detections: bool = False

    @classmethod
    def from_environment(cls) -> "YoloRuntimeConfig":
        config_path = resolve_default_config_path()
        raw = _load_yaml(config_path)
        model_path = resolve_default_model_path(raw)

        device_default = "0" if torch.cuda.is_available() else "cpu"
        device = os.getenv("YOLO_DEVICE", str(_deep_get(raw, ["model", "device"], device_default)))
        use_cuda = torch.cuda.is_available() and device.lower() != "cpu"
        half_default = bool(_deep_get(raw, ["model", "half_if_cuda"], True)) and use_cuda

        colors = dict(DEFAULT_CLASS_COLORS)
        colors.update(dict(_deep_get(raw, ["classes", "colors"], {}) or {}))
        aliases = dict(_deep_get(raw, ["classes", "aliases"], {}) or {"blue": "person", "red": "person", "tank": "tank", "Tank": "tank", "car": "car", "house": "house"})
        canonical_order = [
            str(x).strip().lower()
            for x in (_deep_get(raw, ["classes", "canonical"], ["rock", "person", "tank", "car", "house"]) or [])
            if str(x).strip()
        ]
        canonical = set(canonical_order)
        ignored = set(str(x).strip().lower() for x in (_deep_get(raw, ["classes", "ignored"], []) or []))
        fixed_ids_raw = dict(_deep_get(raw, ["classes", "fixed_ids"], {}) or {"tank": 1, "rock": 2, "person": 3, "car": 4, "house": 5})
        fixed_ids = {str(k).strip().lower(): int(v) for k, v in fixed_ids_raw.items()}

        thresholds = dict(_deep_get(raw, ["classes", "confidence_thresholds"], {}) or {})
        return cls(
            model_path=model_path,
            config_path=config_path,
            imgsz=int(os.getenv("YOLO_IMGSZ", _deep_get(raw, ["model", "imgsz"], 416))),
            device=device,
            use_cuda=use_cuda,
            half=_env_flag("YOLO_HALF", half_default),
            iou=float(os.getenv("YOLO_IOU", _deep_get(raw, ["model", "iou"], 0.70))),
            max_det=int(os.getenv("YOLO_MAX_DET", _deep_get(raw, ["model", "max_det"], 20))),
            max_return=int(os.getenv("YOLO_MAX_RETURN", _deep_get(raw, ["model", "max_return"], 5))),
            model_conf=float(os.getenv("YOLO_MODEL_CONF", _deep_get(raw, ["model", "model_confidence"], 0.5))),
            fallback_model_conf=float(os.getenv("YOLO_FALLBACK_MODEL_CONF", _deep_get(raw, ["model", "fallback_model_confidence"], 0.5))),
            low_conf_fallback=_env_flag("YOLO_LOW_CONF_FALLBACK", bool(_deep_get(raw, ["model", "enable_low_conf_fallback"], False))),
            return_fallback_detections=_env_flag("YOLO_RETURN_FALLBACK_DETECTIONS", True),
            tracking_enabled=_env_flag("YOLO_TRACKING", bool(_deep_get(raw, ["tracking", "enabled"], True))),
            tracker=str(os.getenv("YOLO_TRACKER", _deep_get(raw, ["tracking", "tracker"], "bytetrack.yaml"))),
            track_persist=_env_flag("YOLO_TRACK_PERSIST", bool(_deep_get(raw, ["tracking", "persist"], True))),
            enable_cache=_env_flag("YOLO_DETECT_CACHE", bool(_deep_get(raw, ["model", "enable_cache"], True))),
            min_interval_sec=float(os.getenv("YOLO_MIN_INTERVAL", _deep_get(raw, ["model", "min_interval_sec"], 0.12))),
            warmup_runs=int(os.getenv("YOLO_WARMUP_RUNS", _deep_get(raw, ["model", "warmup_runs"], 2))),
            aliases=aliases,
            canonical_classes=canonical,
            canonical_class_order=canonical_order,
            ignored_classes=ignored,
            class_fixed_ids=fixed_ids,
            class_colors=colors,
            default_box_color=str(_deep_get(raw, ["classes", "default_box_color"], "#00FF00")),
            default_confidence=float(os.getenv("YOLO_DEFAULT_CONF", _deep_get(raw, ["classes", "default_confidence"], 0.5))),
            class_confidence_thresholds={k: float(v) for k, v in thresholds.items()},
            shadow_filter_enabled=_env_flag("YOLO_SHADOW_FILTER", bool(_deep_get(raw, ["preprocess", "shadow_filter_enabled"], False))),
            shadow_sigma=float(os.getenv("YOLO_SHADOW_SIGMA", _deep_get(raw, ["preprocess", "shadow_sigma"], 35.0))),
            shadow_strength=float(os.getenv("YOLO_SHADOW_STRENGTH", _deep_get(raw, ["preprocess", "shadow_strength"], 0.75))),
            shadow_work_scale=float(os.getenv("YOLO_SHADOW_WORK_SCALE", _deep_get(raw, ["preprocess", "shadow_work_scale"], 0.35))),
            shadow_max_side=int(os.getenv("YOLO_SHADOW_MAX_SIDE", _deep_get(raw, ["preprocess", "shadow_max_side"], 960))),
            clahe_enabled=_env_flag("YOLO_SHADOW_CLAHE", bool(_deep_get(raw, ["preprocess", "clahe_enabled"], False))),
            clahe_clip=float(os.getenv("YOLO_SHADOW_CLAHE_CLIP", _deep_get(raw, ["preprocess", "clahe_clip"], 1.5))),
            timing_log=_env_flag("YOLO_TIMING", bool(_deep_get(raw, ["runtime", "timing_log"], False))),
            recognition_log=_env_flag("YOLO_RECOGNITION_LOG", bool(_deep_get(raw, ["runtime", "recognition_log"], True))),
            recognition_log_cache=_env_flag("YOLO_RECOGNITION_LOG_CACHE", bool(_deep_get(raw, ["runtime", "recognition_log_cache"], False))),
            recognition_log_empty=_env_flag("YOLO_RECOGNITION_LOG_EMPTY", bool(_deep_get(raw, ["runtime", "recognition_log_empty"], False))),
            debug_detections=_env_flag("YOLO_DETECT_DEBUG", bool(_deep_get(raw, ["runtime", "debug_detections"], False))),
        )


class TankYoloDetector:
    """Thread-safe YOLO detector for /detect images."""

    def __init__(self, config: Optional[YoloRuntimeConfig] = None) -> None:
        self.config = config or YoloRuntimeConfig.from_environment()
        self._state_lock = Lock()
        self._predict_lock = Lock()
        self._model = None
        self._model_names: Dict[int, str] = {}
        self._public_names: Dict[int, str] = {}
        self._state: Dict[str, Any] = {
            "loaded": False,
            "load_error": None,
            "latest_detections": [],
            "latest_detection_timestamp": 0.0,
            "latest_detect_cached": False,
            "latest_cache_reason": None,
            "latest_detect_ms": 0.0,
            "latest_decode_ms": 0.0,
            "latest_preprocess_ms": 0.0,
            "latest_yolo_ms": 0.0,
            "latest_postprocess_ms": 0.0,
            "latest_raw_detection_count": 0,
            "latest_returned_detection_count": 0,
            "latest_raw_detections": [],
            "latest_rejected_detections": [],
            "latest_frame_shape": None,
            "latest_model_conf_used": self.config.model_conf,
            "latest_fallback_used": False,
        }
        self._load_model()

    @property
    def loaded(self) -> bool:
        with self._state_lock:
            return bool(self._state.get("loaded"))

    def _load_model(self) -> None:
        try:
            from ultralytics import YOLO
        except Exception as exc:
            self._set_load_error(f"ultralytics import failed: {exc}")
            return

        if not self.config.model_path.exists():
            self._set_load_error(f"YOLO model not found: {self.config.model_path}")
            return

        try:
            self._model = YOLO(str(self.config.model_path), task="detect")
            self._model_names = normalize_model_names(self._model.names)
            self._public_names = {
                class_id: self.normalize_public_class_name(name)
                for class_id, name in self._model_names.items()
            }
            if self.config.use_cuda:
                torch.backends.cudnn.benchmark = _env_flag("YOLO_CUDNN_BENCHMARK", True)
                tf32_enabled = _env_flag("YOLO_TF32", True)
                torch.backends.cuda.matmul.allow_tf32 = tf32_enabled
                torch.backends.cudnn.allow_tf32 = tf32_enabled
                try:
                    torch.set_float32_matmul_precision("high")
                except Exception:
                    pass
                warmup_image = np.zeros((max(32, self.config.imgsz), max(32, self.config.imgsz), 3), dtype=np.uint8)
                warmup_image, _, _ = self.preprocess_frame(warmup_image)
                for _ in range(max(1, self.config.warmup_runs)):
                    with torch.inference_mode():
                        self._model.predict(
                            source=warmup_image,
                            conf=self.config.model_conf,
                            imgsz=self.config.imgsz,
                            device=self.config.device,
                            half=self.config.half,
                            iou=self.config.iou,
                            max_det=self.config.max_det,
                            verbose=False,
                        )
                torch.cuda.synchronize()
            with self._state_lock:
                self._state["loaded"] = True
                self._state["load_error"] = None
            print(
                "[vision] YOLO loaded: "
                f"model={self.config.model_path}, labels={self._model_names}, public={self._public_names}, "
                f"backend={self.runtime_backend()}, device={self.config.device}, half={self.config.half}, imgsz={self.config.imgsz}, "
                f"tracking={self.config.tracking_enabled}, tracker={self.config.tracker}, persist={self.config.track_persist}"
            )
        except Exception as exc:
            self._set_load_error(str(exc))

    def _set_load_error(self, message: str) -> None:
        with self._state_lock:
            self._state["loaded"] = False
            self._state["load_error"] = message
        print(f"[vision] YOLO disabled: {message}")

    def normalize_public_class_name(self, class_name: Any) -> str:
        normalized = str(class_name).strip()
        lowered = normalized.lower()
        # YAML may contain aliases with original case or lower case. Always return canonical lower-case class names.
        alias_value = self.config.aliases.get(normalized, self.config.aliases.get(lowered, lowered))
        return str(alias_value).strip().lower()

    def get_class_fixed_id(self, class_name: str) -> Optional[int]:
        return self.config.class_fixed_ids.get(str(class_name).strip().lower())

    def get_box_color(self, class_name: str) -> str:
        return self.config.class_colors.get(str(class_name).strip().lower(), self.config.default_box_color)

    def runtime_backend(self) -> str:
        suffix = self.config.model_path.suffix.lower()
        if suffix == ".engine":
            return "tensorrt"
        if suffix == ".onnx":
            return "onnx"
        if suffix == ".pt":
            return "pytorch"
        return suffix.lstrip(".") or "unknown"

    def decode_image_bytes(self, image_bytes: bytes) -> Optional[np.ndarray]:
        if not image_bytes:
            return None
        image_buffer = np.frombuffer(image_bytes, dtype=np.uint8)
        return cv2.imdecode(image_buffer, cv2.IMREAD_COLOR)

    def _clamp_float(self, value: float, lower: float, upper: float) -> float:
        return max(lower, min(upper, value))

    def remove_shadow_with_gaussian(self, frame: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hue, saturation, value = cv2.split(hsv)
        value_float = value.astype(np.float32)
        frame_height, frame_width = value.shape[:2]
        work_scale = self._clamp_float(self.config.shadow_work_scale, 0.1, 1.0)
        sigma = max(self.config.shadow_sigma * work_scale, 1.0)

        if work_scale < 1.0:
            work_width = max(16, int(frame_width * work_scale))
            work_height = max(16, int(frame_height * work_scale))
            value_for_blur = cv2.resize(value_float, (work_width, work_height), interpolation=cv2.INTER_AREA)
        else:
            value_for_blur = value_float

        illumination = cv2.GaussianBlur(value_for_blur, (0, 0), sigmaX=sigma, sigmaY=sigma)
        if work_scale < 1.0:
            illumination = cv2.resize(illumination, (frame_width, frame_height), interpolation=cv2.INTER_LINEAR)

        illumination = np.maximum(illumination, 1.0)
        scale = max(float(np.mean(illumination)), 1.0)
        normalized_value = cv2.divide(value_float, illumination, scale=scale)
        normalized_value = np.clip(normalized_value, 0, 255).astype(np.uint8)

        if self.config.clahe_enabled:
            clahe = cv2.createCLAHE(clipLimit=self.config.clahe_clip, tileGridSize=(8, 8))
            normalized_value = clahe.apply(normalized_value)

        strength = self._clamp_float(self.config.shadow_strength, 0.0, 1.0)
        corrected_value = cv2.addWeighted(value, 1.0 - strength, normalized_value, strength, 0)
        corrected_hsv = cv2.merge((hue, saturation, corrected_value))
        return cv2.cvtColor(corrected_hsv, cv2.COLOR_HSV2BGR)

    def resize_for_shadow_filter(self, frame: np.ndarray) -> Tuple[np.ndarray, float, float]:
        frame_height, frame_width = frame.shape[:2]
        max_side = max(frame_height, frame_width)
        if self.config.shadow_max_side <= 0 or max_side <= self.config.shadow_max_side:
            return frame, 1.0, 1.0
        resize_scale = self.config.shadow_max_side / float(max_side)
        resized_width = max(16, int(frame_width * resize_scale))
        resized_height = max(16, int(frame_height * resize_scale))
        resized_frame = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
        return resized_frame, frame_width / float(resized_width), frame_height / float(resized_height)

    def preprocess_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, float, float]:
        if not self.config.shadow_filter_enabled:
            return frame, 1.0, 1.0
        resized_frame, scale_x, scale_y = self.resize_for_shadow_filter(frame)
        return self.remove_shadow_with_gaussian(resized_frame), scale_x, scale_y

    def scale_box_to_original_frame(self, box: np.ndarray, scale_x: float, scale_y: float) -> np.ndarray:
        scaled_box = box.copy()
        scaled_box[0] *= scale_x
        scaled_box[2] *= scale_x
        scaled_box[1] *= scale_y
        scaled_box[3] *= scale_y
        return scaled_box

    def is_valid_box(self, box: Any) -> bool:
        if len(box) < 4:
            return False
        x1, y1, x2, y2 = (float(value) for value in box[:4])
        return x2 > x1 and y2 > y1

    def evaluate_detection_for_return(self, class_name: Optional[str], confidence: float, box: Any, frame_shape: Tuple[int, ...], bypass: bool) -> Tuple[bool, Optional[str], Optional[float]]:
        if class_name is None:
            return False, "class_name_none", None
        if not self.is_valid_box(box):
            return False, "invalid_box", None
        if class_name in self.config.ignored_classes:
            return False, "ignored_class", None
        if self.config.canonical_classes and class_name not in self.config.canonical_classes:
            return False, "non_canonical_class", None
        if bypass:
            return True, None, None
        threshold = self.config.class_confidence_thresholds.get(class_name, self.config.default_confidence)
        if confidence >= threshold:
            return True, None, threshold
        if class_name in self.config.class_confidence_thresholds:
            return False, "below_class_threshold", threshold
        return False, "below_default_threshold", threshold

    def make_detection_response(
        self,
        class_name: str,
        box: Any,
        confidence: float,
        class_id: Optional[int] = None,
        track_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        x1, y1, x2, y2 = [float(coord) for coord in box[:4]]
        class_fixed_id = self.get_class_fixed_id(class_name)
        return {
            "id": class_fixed_id,
            "className": class_name,
            "classId": None if class_id is None else int(class_id),
            "classFixedId": class_fixed_id,
            "trackId": None if track_id is None else int(track_id),
            "bbox": [x1, y1, x2, y2],
            "center": [0.5 * (x1 + x2), 0.5 * (y1 + y2)],
            "confidence": float(confidence),
            "color": self.get_box_color(class_name),
            "filled": False,
            "updateBoxWhileMoving": False,
        }

    def _result_box_count(self, results: Any) -> int:
        if not results:
            return 0
        boxes = results[0].boxes
        if boxes is None:
            return 0
        return len(boxes)

    def _get_cached(self, now_seconds: float) -> Optional[Tuple[List[Dict[str, Any]], float]]:
        if not self.config.enable_cache:
            return None
        with self._state_lock:
            latest_ts = float(self._state.get("latest_detection_timestamp") or 0.0)
            if latest_ts <= 0.0 or now_seconds - latest_ts > self.config.min_interval_sec:
                return None
            return list(self._state.get("latest_detections") or []), latest_ts

    def _latest_for_busy(self) -> Tuple[List[Dict[str, Any]], float]:
        with self._state_lock:
            return list(self._state.get("latest_detections") or []), float(self._state.get("latest_detection_timestamp") or 0.0)

    def _update_state(self, **kwargs: Any) -> None:
        with self._state_lock:
            self._state.update(kwargs)

    def _run_model(self, frame: np.ndarray, confidence: float) -> Any:
        if self.config.tracking_enabled:
            return self._model.track(
                source=frame,
                conf=confidence,
                imgsz=self.config.imgsz,
                device=self.config.device,
                half=self.config.half,
                iou=self.config.iou,
                max_det=self.config.max_det,
                tracker=self.config.tracker,
                persist=self.config.track_persist,
                verbose=False,
            )
        return self._model.predict(
            source=frame,
            conf=confidence,
            imgsz=self.config.imgsz,
            device=self.config.device,
            half=self.config.half,
            iou=self.config.iou,
            max_det=self.config.max_det,
            verbose=False,
        )

    def _iter_result_boxes(self, results: Any) -> List[Dict[str, Any]]:
        if not results or results[0].boxes is None:
            return []
        boxes = results[0].boxes
        xyxy = boxes.xyxy.detach().cpu().numpy()
        confs = boxes.conf.detach().cpu().numpy()
        clss = boxes.cls.detach().cpu().numpy()
        if getattr(boxes, "id", None) is not None:
            track_ids = boxes.id.detach().cpu().numpy()
        else:
            track_ids = [None] * len(xyxy)

        out: List[Dict[str, Any]] = []
        for bbox, conf, cls_id, track_id in zip(xyxy, confs, clss, track_ids):
            out.append(
                {
                    "bbox": bbox,
                    "confidence": float(conf),
                    "class_id": int(cls_id),
                    "track_id": None if track_id is None else int(track_id),
                }
            )
        return out

    def _log_detections(self, detections: List[Dict[str, Any]], cached: bool = False) -> None:
        if not self.config.recognition_log:
            return
        if cached and not self.config.recognition_log_cache:
            return
        if not detections:
            if self.config.recognition_log_empty:
                print("[detect] no object recognized")
            return
        print(f"[detect] {len(detections)} object(s) recognized")
        for det in detections:
            bbox = det.get("bbox", [])
            bbox_text = ", ".join(f"{float(coord):.1f}" for coord in bbox[:4])
            print(f"[detect] class={det.get('className')} conf={float(det.get('confidence', 0.0)):.2f} bbox=[{bbox_text}]")

    def detect_bytes(self, image_bytes: bytes) -> List[Dict[str, Any]]:
        started_at = time.perf_counter()
        if not self.loaded or self._model is None:
            return []

        cached = self._get_cached(time.time())
        if cached is not None:
            detections, ts = cached
            elapsed_ms = (time.perf_counter() - started_at) * 1000.0
            self._update_state(
                latest_detect_ms=elapsed_ms,
                latest_detect_cached=True,
                latest_cache_reason="fresh_interval",
                latest_returned_detection_count=len(detections),
                latest_detections=list(detections),
                latest_detection_timestamp=ts,
            )
            self._log_detections(detections, cached=True)
            return detections

        if not self._predict_lock.acquire(blocking=False):
            detections, ts = self._latest_for_busy()
            elapsed_ms = (time.perf_counter() - started_at) * 1000.0
            self._update_state(
                latest_detect_ms=elapsed_ms,
                latest_detect_cached=True,
                latest_cache_reason="inference_busy",
                latest_returned_detection_count=len(detections),
                latest_detections=list(detections),
                latest_detection_timestamp=ts,
            )
            self._log_detections(detections, cached=True)
            return detections

        try:
            decode_started = time.perf_counter()
            frame = self.decode_image_bytes(image_bytes)
            decode_ms = (time.perf_counter() - decode_started) * 1000.0
            if frame is None:
                return []
            original_shape = frame.shape
            frame_shape_list = [int(value) for value in original_shape]

            preprocess_started = time.perf_counter()
            processed_frame, scale_x, scale_y = self.preprocess_frame(frame)
            preprocess_ms = (time.perf_counter() - preprocess_started) * 1000.0

            yolo_started = time.perf_counter()
            model_conf_used = self.config.model_conf
            fallback_used = False
            with torch.inference_mode():
                results = self._run_model(processed_frame, self.config.model_conf)
                if (
                    self.config.low_conf_fallback
                    and self.config.fallback_model_conf < self.config.model_conf
                    and self._result_box_count(results) == 0
                ):
                    fallback_used = True
                    model_conf_used = self.config.fallback_model_conf
                    results = self._run_model(processed_frame, self.config.fallback_model_conf)
            yolo_ms = (time.perf_counter() - yolo_started) * 1000.0
        finally:
            self._predict_lock.release()

        post_started = time.perf_counter()
        result_items = self._iter_result_boxes(results)
        filtered: List[Dict[str, Any]] = []
        raw_debug: List[Dict[str, Any]] = []
        rejected: List[Dict[str, Any]] = []

        for item in result_items:
            class_id = int(item["class_id"])
            model_class_name = self._model_names.get(class_id)
            class_name = self._public_names.get(class_id)
            confidence = float(item["confidence"])
            track_id = item.get("track_id")
            scaled_box = self.scale_box_to_original_frame(np.asarray(item["bbox"], dtype=float), scale_x, scale_y)
            bypass = fallback_used and self.config.return_fallback_detections
            returned, reason, threshold = self.evaluate_detection_for_return(class_name, confidence, scaled_box, original_shape, bypass)
            debug_item = {
                "modelClassName": model_class_name,
                "className": class_name,
                "classId": class_id,
                "classFixedId": self.get_class_fixed_id(class_name) if class_name else None,
                "trackId": track_id,
                "bbox": [float(coord) for coord in scaled_box[:4]],
                "center": [float(0.5 * (scaled_box[0] + scaled_box[2])), float(0.5 * (scaled_box[1] + scaled_box[3]))],
                "confidence": confidence,
                "returned": returned,
                "rejectReason": reason,
                "threshold": threshold,
            }
            raw_debug.append(debug_item)
            if not returned:
                rejected.append(debug_item)
                continue
            filtered.append(self.make_detection_response(class_name, scaled_box, confidence, class_id=class_id, track_id=track_id))

        filtered.sort(key=lambda item: item["confidence"], reverse=True)
        filtered = filtered[: self.config.max_return]
        post_ms = (time.perf_counter() - post_started) * 1000.0
        elapsed_ms = (time.perf_counter() - started_at) * 1000.0

        self._update_state(
            latest_detections=list(filtered),
            latest_detection_timestamp=time.time(),
            latest_detect_cached=False,
            latest_cache_reason=None,
            latest_detect_ms=elapsed_ms,
            latest_decode_ms=decode_ms,
            latest_preprocess_ms=preprocess_ms,
            latest_yolo_ms=yolo_ms,
            latest_postprocess_ms=post_ms,
            latest_raw_detection_count=len(raw_debug),
            latest_returned_detection_count=len(filtered),
            latest_raw_detections=raw_debug[:10],
            latest_rejected_detections=rejected[:10],
            latest_frame_shape=frame_shape_list,
            latest_model_conf_used=model_conf_used,
            latest_fallback_used=fallback_used,
        )

        if self.config.timing_log:
            print(
                "[perf:/detect] "
                f"decode={decode_ms:.1f}ms preprocess={preprocess_ms:.1f}ms "
                f"yolo={yolo_ms:.1f}ms post={post_ms:.1f}ms total={elapsed_ms:.1f}ms "
                f"raw={len(raw_debug)} returned={len(filtered)}"
            )
        if self.config.debug_detections:
            print("[detect] raw:", raw_debug)
            print("[detect] rejected:", rejected)
            print("[detect] returned:", filtered)
        self._log_detections(filtered, cached=False)
        return filtered

    def debug_state(self) -> Dict[str, Any]:
        with self._state_lock:
            state = dict(self._state)
        return {
            "serverMode": "ros_bridge_embedded_yolo",
            "loaded": state.get("loaded"),
            "loadError": state.get("load_error"),
            "modelPath": str(self.config.model_path),
            "runtimeBackend": self.runtime_backend(),
            "tensorRtEnabled": self.runtime_backend() == "tensorrt",
            "configPath": str(self.config.config_path) if self.config.config_path else None,
            "modelNames": self._model_names,
            "publicNames": self._public_names,
            "classColors": self.config.class_colors,
            "canonicalClasses": list(self.config.canonical_class_order),
            "ignoredClasses": sorted(self.config.ignored_classes),
            "classFixedIds": self.config.class_fixed_ids,
            "trackingEnabled": self.config.tracking_enabled,
            "tracker": self.config.tracker,
            "trackPersist": self.config.track_persist,
            "imgsz": self.config.imgsz,
            "device": self.config.device,
            "useCuda": self.config.use_cuda,
            "half": self.config.half,
            "modelConf": self.config.model_conf,
            "defaultConf": self.config.default_confidence,
            "classConfidenceThresholds": self.config.class_confidence_thresholds,
            "maxDet": self.config.max_det,
            "maxReturn": self.config.max_return,
            "cacheEnabled": self.config.enable_cache,
            "minIntervalSec": self.config.min_interval_sec,
            "latestDetectMs": state.get("latest_detect_ms"),
            "latestDecodeMs": state.get("latest_decode_ms"),
            "latestPreprocessMs": state.get("latest_preprocess_ms"),
            "latestYoloMs": state.get("latest_yolo_ms"),
            "latestPostprocessMs": state.get("latest_postprocess_ms"),
            "latestRawDetectionCount": state.get("latest_raw_detection_count"),
            "latestReturnedDetectionCount": state.get("latest_returned_detection_count"),
            "latestRawDetections": state.get("latest_raw_detections"),
            "latestRejectedDetections": state.get("latest_rejected_detections"),
            "latestFrameShape": state.get("latest_frame_shape"),
            "latestDetectCached": state.get("latest_detect_cached"),
            "latestCacheReason": state.get("latest_cache_reason"),
            "cudaAvailable": torch.cuda.is_available(),
            "cudaDeviceName": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }


_DETECTOR_SINGLETON: Optional[TankYoloDetector] = None
_DETECTOR_LOCK = Lock()


def get_detector() -> TankYoloDetector:
    global _DETECTOR_SINGLETON
    with _DETECTOR_LOCK:
        if _DETECTOR_SINGLETON is None:
            _DETECTOR_SINGLETON = TankYoloDetector()
        return _DETECTOR_SINGLETON
