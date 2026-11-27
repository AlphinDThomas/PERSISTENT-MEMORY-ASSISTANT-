"""Relative position estimation for M2 — Real-Time Environmental Perception.

Converts bounding box pixel coordinates into a coarse horizontal region label
that describes where in the frame an object appears.  This is the primary
spatial cue communicated to visually impaired users.
"""

from __future__ import annotations

# Horizontal region boundaries as fractions of frame width.
# Five equal zones: left | center-left | center | center-right | right
_ZONES: list[tuple[float, str]] = [
    (0.20, "left"),
    (0.40, "center-left"),
    (0.60, "center"),
    (0.80, "center-right"),
    (1.00, "right"),
]


def estimate_position(
    bounding_box: tuple[int, int, int, int],
    frame_width: int,
) -> str:
    """Return the horizontal region label for a detected object.

    The position is determined by the horizontal midpoint of the bounding box
    relative to the total frame width, divided into five equal zones.

    Args:
        bounding_box: ``(x1, y1, x2, y2)`` pixel coordinates of the detection.
        frame_width:  Width of the original (pre-letterbox) frame in pixels.

    Returns:
        One of ``"left"``, ``"center-left"``, ``"center"``,
        ``"center-right"``, or ``"right"``.
    """
    x1, _, x2, _ = bounding_box
    center_x = (x1 + x2) / 2.0
    relative = center_x / max(frame_width, 1)

    for threshold, label in _ZONES:
        if relative <= threshold:
            return label

    return "right"  # Fallback for floating-point edge cases

