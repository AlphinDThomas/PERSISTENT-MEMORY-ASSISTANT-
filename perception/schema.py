"""Shared output data contracts for M2 — Real-Time Environmental Perception.

These dataclasses form the interface boundary between M2 and its consumers
(M3 — Object Tracking, M11 — Observation Processing).  Do not embed M2-internal
implementation details here; keep the schema stable so downstream modules can
rely on it without change.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DetectedObject:
    """A single object detected within one video frame.

    Attributes:
        label:        COCO class label string, e.g. ``"chair"``, ``"person"``.
        confidence:   Detection confidence score in the range [0.0, 1.0].
        bounding_box: Pixel coordinates ``(x1, y1, x2, y2)`` of the detection
                      box, where ``(x1, y1)`` is the top-left corner and
                      ``(x2, y2)`` is the bottom-right corner.
        position:     Coarse horizontal region of the object within the frame.
                      One of ``"left"``, ``"center-left"``, ``"center"``,
                      ``"center-right"``, or ``"right"``.
    """

    label: str
    confidence: float
    bounding_box: tuple[int, int, int, int]
    position: str


@dataclass
class PerceptionFrame:
    """All detections produced from a single captured video frame.

    Attributes:
        frame_id:     Monotonically increasing integer assigned by the pipeline.
        timestamp:    Unix timestamp (``time.time()``) at the moment of capture.
        frame_width:  Width of the original frame in pixels.
        frame_height: Height of the original frame in pixels.
        objects:      List of :class:`DetectedObject` instances found in the frame.
    """

    frame_id: int
    timestamp: float
    frame_width: int
    frame_height: int
    objects: list[DetectedObject] = field(default_factory=list)

