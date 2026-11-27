"""Perception pipeline: raw camera frame -> YOLO -> position -> schema."""

from __future__ import annotations

import time
from typing import Iterator

import numpy as np

from .detector import ObjectDetector
from .localizer import estimate_position
from .schema import DetectedObject, PerceptionFrame
from .stream import IPWebcamStream


class PerceptionPipeline:
    def __init__(
        self,
        stream_url: str,
        model_name: str = "yolov8s.pt",
        confidence_threshold: float = 0.25,
        input_size: tuple[int, int] | None = None,  # retained for compatibility; no manual resizing
    ) -> None:
        self._stream = IPWebcamStream(stream_url)
        self._detector = ObjectDetector(
            model_name=model_name,
            confidence_threshold=confidence_threshold,
            image_size=960,
        )
        self._frame_id = 0

    def run(self, frame_timeout: float = 1.0) -> Iterator[PerceptionFrame]:
        with self._stream:
            for raw_frame in self._stream.frames(timeout=frame_timeout):
                result = self._process_frame(raw_frame)
                if result is not None:
                    yield result

    def process_image(self, frame: np.ndarray) -> PerceptionFrame:
        result = self._process_frame(frame)
        if result is None:
            raise RuntimeError("Frame processing failed.")
        return result

    def _process_frame(self, raw_frame: np.ndarray) -> PerceptionFrame | None:
        try:
            src_h, src_w = raw_frame.shape[:2]
            raw_detections = self._detector.detect(raw_frame)
            objects = [
                DetectedObject(
                    label=str(det["label"]),
                    confidence=float(det["confidence"]),
                    bounding_box=det["bounding_box"],  # type: ignore[arg-type]
                    position=estimate_position(
                        det["bounding_box"],  # type: ignore[arg-type]
                        frame_width=src_w,
                    ),
                )
                for det in raw_detections
            ]
            self._frame_id += 1
            return PerceptionFrame(
                frame_id=self._frame_id,
                timestamp=time.time(),
                frame_width=src_w,
                frame_height=src_h,
                objects=objects,
            )
        except Exception as exc:
            print(f"[M2] Frame processing error (frame skipped): {exc}", flush=True)
            return None
