"""Compatibility preprocessor.

Intentionally does not resize or change color channels. OpenCV frames are BGR
and should be passed directly to Ultralytics, which performs its own preprocessing.
"""

from __future__ import annotations

import numpy as np


class FramePreprocessor:
    def __init__(self, input_size: tuple[int, int] | None = None) -> None:
        self._input_size = input_size

    def process(self, frame: np.ndarray) -> np.ndarray:
        return frame
