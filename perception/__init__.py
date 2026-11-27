"""M2 — Real-Time Environmental Perception package.

Public API
----------
:class:`PerceptionPipeline`
    The single entry point for external consumers (M3, M11, M8).
    Composes stream acquisition, preprocessing, detection, and position
    estimation into one object.

:class:`~perception.schema.DetectedObject`
:class:`~perception.schema.PerceptionFrame`
    Shared output data contracts.
"""

from __future__ import annotations

from .schema import DetectedObject, PerceptionFrame
from .pipeline import PerceptionPipeline

__all__ = [
    "DetectedObject",
    "PerceptionFrame",
    "PerceptionPipeline",
]

