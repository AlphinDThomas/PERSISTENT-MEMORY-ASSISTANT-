"""PerceptionPipeline — top-level orchestrator for M2.

Composes :class:`~perception.stream.IPWebcamStream`,
:class:`~perception.preprocessor.FramePreprocessor`,
:class:`~perception.detector.ObjectDetector`, and
:func:`~perception.localizer.estimate_position` into a single object that
yields :class:`~perception.schema.PerceptionFrame` instances.
"""

from __future__ import annotations

import time
from typing import Iterator

import numpy as np

from .detector import ObjectDetector
from .localizer import estimate_position
from .preprocessor import FramePreprocessor
from .schema import DetectedObject, PerceptionFrame
from .stream import IPWebcamStream


class PerceptionPipeline:
    """End-to-end perception pipeline for M2.

    Acquires frames from an IP Webcam stream, preprocesses them, runs YOLO
    object detection, estimates object positions, and yields
    :class:`~perception.schema.PerceptionFrame` instances.

    Args:
        stream_url:           IP Webcam MJPEG URL,
                              e.g. ``"http://192.168.1.5:8080/video"``.
        model_name:           YOLOv8 model identifier or ``.pt`` path.
                              Defaults to ``"yolov8n.pt"`` (nano).
        confidence_threshold: Minimum detection confidence.  Defaults to ``0.5``.
        input_size:           Preprocessor resize target ``(w, h)``.
                              Pass ``None`` to skip resizing.

    Example::

        pipeline = PerceptionPipeline("http://192.168.1.5:8080/video")
        for frame in pipeline.run():
            for obj in frame.objects:
                print(obj.label, obj.position, f"{obj.confidence:.2f}")
    """

    def __init__(
        self,
        stream_url: str,
        model_name: str = "yolov8n.pt",
        confidence_threshold: float = 0.5,
        input_size: tuple[int, int] | None = (640, 640),
    ) -> None:
        self._stream = IPWebcamStream(stream_url)
        self._preprocessor = FramePreprocessor(input_size=input_size)
        self._detector = ObjectDetector(
            model_name=model_name,
            confidence_threshold=confidence_threshold,
        )
        self._frame_id = 0

    def run(self, frame_timeout: float = 1.0) -> Iterator[PerceptionFrame]:
        """Yield :class:`~perception.schema.PerceptionFrame` instances indefinitely.

        Call :meth:`stop` or use the pipeline as a context manager to release
        resources.

        Args:
            frame_timeout: Seconds to wait for each new frame before skipping.
        """
        with self._stream:
            for raw_frame in self._stream.frames(timeout=frame_timeout):
                perception_frame = self._process_frame(raw_frame)
                if perception_frame is not None:
                    yield perception_frame

    def process_image(self, frame: np.ndarray) -> PerceptionFrame:
        """Process a single BGR numpy array and return a :class:`PerceptionFrame`.

        Useful for testing or one-shot file/image processing without a live
        stream.

        Args:
            frame: BGR uint8 numpy array (e.g. from ``cv2.imread``).
        """
        result = self._process_frame(frame)
        assert result is not None
        return result

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _process_frame(self, raw_frame: np.ndarray) -> PerceptionFrame | None:
        try:
            src_h, src_w = raw_frame.shape[:2]
            preprocessed = self._preprocessor.process(raw_frame)
            raw_detections = self._detector.detect(preprocessed)

            objects = [
                DetectedObject(
                    label=det["label"],  # type: ignore[arg-type]
                    confidence=det["confidence"],  # type: ignore[arg-type]
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
        except Exception as exc:  # noqa: BLE001
            # Log and continue — a single bad frame should not crash the pipeline
            print(f"[M2] Frame processing error (frame skipped): {exc}", flush=True)
            return None

