"""Frame preprocessing for M2 — Real-Time Environmental Perception.

Converts raw BGR frames from the IP Webcam stream into the format expected
by the YOLO detector: RGB, resized to the model's input resolution, with
pixel values kept as uint8 (Ultralytics handles normalisation internally).
"""

from __future__ import annotations

import cv2
import numpy as np


class FramePreprocessor:
    """Prepare a raw BGR frame for YOLO inference.

    Args:
        input_size: ``(width, height)`` to resize frames to before passing
                    them to the detector.  Defaults to ``(640, 640)``, which
                    is the standard YOLOv8 input resolution.  Pass ``None``
                    to skip resizing (useful when the stream already delivers
                    the correct resolution).
    """

    def __init__(self, input_size: tuple[int, int] | None = (640, 640)) -> None:
        self._input_size = input_size

    def process(self, frame: np.ndarray) -> np.ndarray:
        """Return a preprocessed copy of *frame* ready for YOLO inference.

        Steps applied:
        1. Resize to ``input_size`` (if set), preserving aspect ratio via
           letterboxing so objects are not distorted.
        2. Convert BGR → RGB (Ultralytics expects RGB).

        Args:
            frame: Raw BGR uint8 frame from OpenCV / IP Webcam stream.

        Returns:
            Preprocessed RGB uint8 numpy array.
        """
        if self._input_size is not None:
            frame = self._letterbox(frame, self._input_size)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return rgb

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _letterbox(
        frame: np.ndarray,
        target_size: tuple[int, int],
        fill_color: tuple[int, int, int] = (114, 114, 114),
    ) -> np.ndarray:
        """Resize *frame* to fit inside *target_size* without distortion.

        Pads remaining space with *fill_color* (grey by default, matching
        the standard YOLO letterbox convention).
        """
        target_w, target_h = target_size
        src_h, src_w = frame.shape[:2]

        scale = min(target_w / src_w, target_h / src_h)
        new_w = int(src_w * scale)
        new_h = int(src_h * scale)

        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        canvas = np.full((target_h, target_w, 3), fill_color, dtype=np.uint8)
        pad_top = (target_h - new_h) // 2
        pad_left = (target_w - new_w) // 2
        canvas[pad_top : pad_top + new_h, pad_left : pad_left + new_w] = resized
        return canvas

