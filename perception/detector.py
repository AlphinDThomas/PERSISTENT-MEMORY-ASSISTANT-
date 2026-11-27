"""YOLO object detection using original BGR frames and original-frame coordinates."""

from __future__ import annotations

import numpy as np
from ultralytics import YOLO


class ObjectDetector:
    def __init__(
        self,
        model_name: str = "yolov8s.pt",
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str | None = None,
        image_size: int = 960,
    ) -> None:
        print(f"Loading YOLO model '{model_name}'...", flush=True)
        self._model = YOLO(model_name)
        if device is not None:
            self._model.to(device)
        self._confidence_threshold = confidence_threshold
        self._iou_threshold = iou_threshold
        self._image_size = image_size
        print(f"YOLO model '{model_name}' is ready.", flush=True)

    def detect(self, frame: np.ndarray) -> list[dict[str, object]]:
        """Infer directly on an original OpenCV BGR frame.

        Ultralytics handles resizing/letterboxing internally. Returned xyxy
        coordinates are mapped to the original input image dimensions.
        """
        results = self._model.predict(
            source=frame,
            verbose=False,
            conf=self._confidence_threshold,
            iou=self._iou_threshold,
            imgsz=self._image_size,
        )
        detections: list[dict[str, object]] = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                label = self._model.names[class_id]
                x1, y1, x2, y2 = (int(round(float(v))) for v in box.xyxy[0])
                h, w = frame.shape[:2]
                # Clamp to original frame boundaries.
                x1 = max(0, min(x1, w - 1))
                x2 = max(0, min(x2, w - 1))
                y1 = max(0, min(y1, h - 1))
                y2 = max(0, min(y2, h - 1))
                if x2 <= x1 or y2 <= y1:
                    continue
                detections.append({
                    "label": label,
                    "confidence": confidence,
                    "bounding_box": (x1, y1, x2, y2),
                })
        return detections
