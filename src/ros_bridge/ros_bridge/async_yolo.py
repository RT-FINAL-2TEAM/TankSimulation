# -*- coding: utf-8 -*-
"""Optional asynchronous YOLO worker for production /detect.

The worker processes only the newest frame. /detect can return immediately with
the most recent completed detections, while metadata tells downstream nodes how
old that result is.
"""

from __future__ import annotations

import time
from copy import deepcopy
from threading import Condition, Lock, Thread
from typing import Any, Callable, Dict, List, Optional, Tuple


class AsyncYoloService:
    def __init__(
        self,
        detector_factory: Callable[[], Any],
        *,
        min_interval_sec: float = 0.0,
        max_result_age_ms: float = 300.0,
        log_interval_sec: float = 2.0,
    ) -> None:
        self.detector_factory = detector_factory
        self.min_interval_sec = max(0.0, float(min_interval_sec))
        self.max_result_age_ms = max(1.0, float(max_result_age_ms))
        self.log_interval_sec = max(0.0, float(log_interval_sec))
        self._lock = Lock()
        self._condition = Condition(self._lock)
        self._thread: Optional[Thread] = None
        self._latest_image_bytes: Optional[bytes] = None
        self._latest_frame_seq = 0
        self._processed_frame_seq = 0
        self._latest_detections: List[Dict[str, Any]] = []
        self._latest_result_timestamp = 0.0
        self._latest_yolo_ms = 0.0
        self._latest_error: Optional[str] = None
        self._worker_count = 0
        self._last_run_time = 0.0
        self._last_log_time = 0.0
        self._dropped_frame_count = 0

    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._thread = Thread(target=self._worker_loop, daemon=True, name="AsyncYoloWorker")
            self._thread.start()

    def enqueue(self, image_bytes: bytes) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        self.start()
        with self._condition:
            if self._latest_frame_seq > self._processed_frame_seq:
                self._dropped_frame_count += 1
            self._latest_frame_seq += 1
            frame_seq = self._latest_frame_seq
            self._latest_image_bytes = image_bytes
            detections = deepcopy(self._latest_detections)
            metadata = self._metadata_locked(frame_seq=frame_seq)
            self._condition.notify()
        return detections, metadata

    def debug_state(self) -> Dict[str, Any]:
        with self._lock:
            return self._metadata_locked(frame_seq=self._latest_frame_seq)

    def _metadata_locked(self, frame_seq: int) -> Dict[str, Any]:
        now = time.time()
        result_age_ms = None
        if self._latest_result_timestamp > 0.0:
            result_age_ms = (now - self._latest_result_timestamp) * 1000.0
        stale = result_age_ms is None or result_age_ms > self.max_result_age_ms
        return {
            "asyncYolo": True,
            "frameSeq": int(frame_seq),
            "processedFrameSeq": int(self._processed_frame_seq),
            "resultAgeMs": result_age_ms,
            "staleAsyncResult": bool(stale),
            "asyncLatestDetectionCount": len(self._latest_detections),
            "asyncWorkerCount": int(self._worker_count),
            "asyncLatestYoloMs": float(self._latest_yolo_ms),
            "asyncMaxResultAgeMs": float(self.max_result_age_ms),
            "asyncDroppedFrameCount": int(self._dropped_frame_count),
            "asyncMinIntervalSec": float(self.min_interval_sec),
            "asyncLatestError": self._latest_error,
        }

    def _worker_loop(self) -> None:
        print("[ros_bridge] Async YOLO worker started")
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._latest_frame_seq > self._processed_frame_seq)
                image_bytes = self._latest_image_bytes
                seq_to_process = self._latest_frame_seq
            if not image_bytes:
                continue
            if self.min_interval_sec > 0:
                wait_sec = self.min_interval_sec - (time.time() - self._last_run_time)
                if wait_sec > 0:
                    time.sleep(wait_sec)
                self._last_run_time = time.time()
            started = time.perf_counter()
            try:
                detector = self.detector_factory()
                detections = detector.detect_bytes(image_bytes)
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                with self._lock:
                    self._processed_frame_seq = int(seq_to_process)
                    self._latest_detections = deepcopy(detections) if isinstance(detections, list) else []
                    self._latest_result_timestamp = time.time()
                    self._latest_yolo_ms = elapsed_ms
                    self._latest_error = None
                    self._worker_count += 1
                now = time.time()
                if self.log_interval_sec > 0 and now - self._last_log_time >= self.log_interval_sec:
                    self._last_log_time = now
                    print(f"[ros_bridge] async yolo seq={seq_to_process} det={len(detections) if isinstance(detections, list) else 0} yolo={elapsed_ms:.1f}ms dropped={self._dropped_frame_count}")
            except Exception as exc:  # noqa: BLE001
                with self._lock:
                    self._latest_error = str(exc)
                print(f"[ros_bridge] async yolo error: {exc}")
                time.sleep(0.05)
