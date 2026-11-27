"""YOLO-based object detection for M2 — Real-Time Environmental Perception.

Wraps the Ultralytics YOLOv8 model to run inference on preprocessed frames
and return raw detection results filtered by a confidence threshold.
"""

from __future__ import annotations

import numpy as np
from ultralytics import YOLO


class ObjectDetector:
    """Run YOLOv8 inference and filter detections by confidence.

    Args:
        model_name:           Ultralytics model identifier or path to a local
                              ``.pt`` weights file.  Defaults to ``"yolov8s.pt"``
                              (small — good accuracy/speed balance on CPU).
        confidence_threshold: Minimum confidence score [0.0, 1.0] for a
                              detection to be included in the output.
                              Defaults to ``0.35``.
        iou_threshold:        IoU threshold for non-maximum suppression.
                              Lower values reduce overlapping boxes.
                              Defaults to ``0.45``.
        device:               Inference device.  ``None`` lets Ultralytics pick
                              automatically (GPU when available, else CPU).
    """

    def __init__(
        self,
        model_name: str = "yolov8s.pt",
        confidence_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        device: str | None = None,
    ) -> None:
        print(
            f"Loading YOLO model '{model_name}'. "
            "The first run will download weights if not already cached.",
            flush=True,
        )
        self._model: YOLO = YOLO(model_name)
        if device is not None:
            self._model.to(device)
        self._confidence_threshold = confidence_threshold
        self._iou_threshold = iou_threshold
        print(f"YOLO model '{model_name}' is ready.", flush=True)

    @property
    def confidence_threshold(self) -> float:
        """The minimum confidence score used to filter detections."""
        return self._confidence_threshold

    def detect(
        self, frame: np.ndarray
    ) -> list[dict[str, object]]:
        """Run detection on *frame* and return filtered results.

        Args:
            frame: Preprocessed RGB uint8 numpy array (output of
                   :class:`~perception.preprocessor.FramePreprocessor`).

        Returns:
            List of detection dicts, each containing:

            * ``"label"``       — COCO class name string.
            * ``"confidence"``  — float in [0.0, 1.0].
            * ``"bounding_box"``— ``(x1, y1, x2, y2)`` int tuple in pixels.
        """
        results = self._model(
            frame,
            verbose=False,
            conf=self._confidence_threshold,
            iou=self._iou_threshold,
        )
        detections: list[dict[str, object]] = []

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                confidence = float(box.conf[0])
                if confidence < self._confidence_threshold:
                    continue

                class_id = int(box.cls[0])
                label: str = self._model.names[class_id]
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])

                detections.append(
                    {
                        "label": label,
                        "confidence": confidence,
                        "bounding_box": (x1, y1, x2, y2),
                    }
                )

        return detections

