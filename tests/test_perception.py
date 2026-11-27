"""Tests for M2 — Real-Time Environmental Perception.

Run with:
    pytest tests/test_perception.py -v

To skip tests that require a live IP Webcam connection:
    pytest tests/test_perception.py -v --no-camera
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fixture to check --no-camera flag (option registered in conftest.py)
# ---------------------------------------------------------------------------

@pytest.fixture
def no_camera(request: pytest.FixtureRequest) -> bool:
    return request.config.getoption("--no-camera")  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# schema
# ---------------------------------------------------------------------------

class TestSchema:
    def test_detected_object_fields(self) -> None:
        from perception.schema import DetectedObject

        obj = DetectedObject(
            label="chair",
            confidence=0.87,
            bounding_box=(10, 20, 100, 200),
            position="center",
        )
        assert obj.label == "chair"
        assert obj.confidence == pytest.approx(0.87)
        assert obj.bounding_box == (10, 20, 100, 200)
        assert obj.position == "center"

    def test_perception_frame_defaults(self) -> None:
        from perception.schema import PerceptionFrame

        frame = PerceptionFrame(
            frame_id=1,
            timestamp=time.time(),
            frame_width=640,
            frame_height=480,
        )
        assert frame.objects == []

    def test_perception_frame_with_objects(self) -> None:
        from perception.schema import DetectedObject, PerceptionFrame

        obj = DetectedObject("person", 0.9, (0, 0, 100, 200), "left")
        frame = PerceptionFrame(1, time.time(), 640, 480, objects=[obj])
        assert len(frame.objects) == 1
        assert frame.objects[0].label == "person"


# ---------------------------------------------------------------------------
# preprocessor
# ---------------------------------------------------------------------------

class TestFramePreprocessor:
    def test_output_shape(self) -> None:
        from perception.preprocessor import FramePreprocessor

        pre = FramePreprocessor(input_size=(640, 640))
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        result = pre.process(dummy)
        assert result.shape == (640, 640, 3)

    def test_output_dtype(self) -> None:
        from perception.preprocessor import FramePreprocessor

        pre = FramePreprocessor(input_size=(640, 640))
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        result = pre.process(dummy)
        assert result.dtype == np.uint8

    def test_no_resize(self) -> None:
        from perception.preprocessor import FramePreprocessor

        pre = FramePreprocessor(input_size=None)
        dummy = np.zeros((480, 320, 3), dtype=np.uint8)
        result = pre.process(dummy)
        # Only channel conversion; spatial dims unchanged
        assert result.shape == (480, 320, 3)

    def test_bgr_to_rgb_conversion(self) -> None:
        """Red BGR pixel (0, 0, 255) should become (255, 0, 0) in RGB."""
        from perception.preprocessor import FramePreprocessor

        pre = FramePreprocessor(input_size=None)
        frame = np.zeros((10, 10, 3), dtype=np.uint8)
        frame[:, :] = (0, 0, 255)  # BGR red
        result = pre.process(frame)
        assert result[0, 0, 0] == 255  # R channel
        assert result[0, 0, 2] == 0    # B channel

    def test_letterbox_preserves_aspect_ratio(self) -> None:
        """A wide frame letterboxed to a square should be padded, not squashed."""
        from perception.preprocessor import FramePreprocessor

        pre = FramePreprocessor(input_size=(640, 640))
        wide_frame = np.full((240, 640, 3), 200, dtype=np.uint8)
        result = pre.process(wide_frame)
        assert result.shape == (640, 640, 3)
        # Padding rows (top/bottom) should be grey (114)
        assert result[0, 0, 0] == 114


# ---------------------------------------------------------------------------
# localizer
# ---------------------------------------------------------------------------

class TestLocalizer:
    @pytest.mark.parametrize(
        "center_x, frame_width, expected",
        [
            (50,  1000, "left"),
            (250, 1000, "center-left"),
            (450, 1000, "center"),
            (700, 1000, "center-right"),
            (950, 1000, "right"),
            (0,   640,  "left"),
            (640, 640,  "right"),
        ],
    )
    def test_position_zones(
        self, center_x: int, frame_width: int, expected: str
    ) -> None:
        from perception.localizer import estimate_position

        # Build a bounding box whose horizontal centre is at center_x
        x1 = center_x - 10
        x2 = center_x + 10
        result = estimate_position((x1, 0, x2, 100), frame_width=frame_width)
        assert result == expected

    def test_zero_width_frame_does_not_crash(self) -> None:
        from perception.localizer import estimate_position

        # Should not raise ZeroDivisionError
        result = estimate_position((0, 0, 10, 10), frame_width=0)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# detector (mocked — avoids downloading weights in CI)
# ---------------------------------------------------------------------------

class TestObjectDetector:
    def test_confidence_filtering(self) -> None:
        """Detections below the threshold must be excluded."""
        from perception.detector import ObjectDetector

        with patch("perception.detector.YOLO") as MockYOLO:
            # Build a fake result with one high-conf and one low-conf detection
            fake_box_high = MagicMock()
            fake_box_high.conf = [0.9]
            fake_box_high.cls = [0]
            fake_box_high.xyxy = [[10, 20, 100, 200]]

            fake_box_low = MagicMock()
            fake_box_low.conf = [0.2]
            fake_box_low.cls = [1]
            fake_box_low.xyxy = [[200, 200, 300, 300]]

            fake_result = MagicMock()
            fake_result.boxes = [fake_box_high, fake_box_low]

            mock_model = MockYOLO.return_value
            mock_model.return_value = [fake_result]
            mock_model.names = {0: "person", 1: "bicycle"}

            detector = ObjectDetector(confidence_threshold=0.5)
            dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
            detections = detector.detect(dummy_frame)

        assert len(detections) == 1
        assert detections[0]["label"] == "person"
        assert detections[0]["confidence"] == pytest.approx(0.9)

    def test_bounding_box_format(self) -> None:
        """Bounding box must be a 4-int tuple."""
        from perception.detector import ObjectDetector

        with patch("perception.detector.YOLO") as MockYOLO:
            fake_box = MagicMock()
            fake_box.conf = [0.8]
            fake_box.cls = [0]
            fake_box.xyxy = [[5, 10, 50, 100]]

            fake_result = MagicMock()
            fake_result.boxes = [fake_box]

            mock_model = MockYOLO.return_value
            mock_model.return_value = [fake_result]
            mock_model.names = {0: "cat"}

            detector = ObjectDetector(confidence_threshold=0.5)
            detections = detector.detect(np.zeros((640, 640, 3), dtype=np.uint8))

        bbox = detections[0]["bounding_box"]
        assert isinstance(bbox, tuple)
        assert len(bbox) == 4
        assert all(isinstance(v, int) for v in bbox)


# ---------------------------------------------------------------------------
# pipeline (integration — mocked stream, real preprocessor + localizer)
# ---------------------------------------------------------------------------

class TestPerceptionPipeline:
    def test_process_image_returns_perception_frame(self) -> None:
        """process_image on a dummy frame should return a valid PerceptionFrame."""
        from perception.pipeline import PerceptionPipeline
        from perception.schema import PerceptionFrame

        with patch("perception.pipeline.ObjectDetector") as MockDetector, \
             patch("perception.pipeline.IPWebcamStream"):

            mock_detector = MockDetector.return_value
            mock_detector.detect.return_value = [
                {
                    "label": "bottle",
                    "confidence": 0.75,
                    "bounding_box": (100, 50, 200, 300),
                }
            ]

            pipeline = PerceptionPipeline("http://dummy-url/video")
            dummy_bgr = np.zeros((480, 640, 3), dtype=np.uint8)
            result = pipeline.process_image(dummy_bgr)

        assert isinstance(result, PerceptionFrame)
        assert result.frame_id == 1
        assert result.frame_width == 640
        assert result.frame_height == 480
        assert len(result.objects) == 1
        assert result.objects[0].label == "bottle"
        assert result.objects[0].position in {
            "left", "center-left", "center", "center-right", "right"
        }

    def test_frame_id_increments(self) -> None:
        from perception.pipeline import PerceptionPipeline

        with patch("perception.pipeline.ObjectDetector") as MockDetector, \
             patch("perception.pipeline.IPWebcamStream"):

            MockDetector.return_value.detect.return_value = []
            pipeline = PerceptionPipeline("http://dummy-url/video")
            dummy = np.zeros((480, 640, 3), dtype=np.uint8)
            f1 = pipeline.process_image(dummy)
            f2 = pipeline.process_image(dummy)

        assert f2.frame_id == f1.frame_id + 1


# ---------------------------------------------------------------------------
# stream (requires live camera — skipped with --no-camera)
# ---------------------------------------------------------------------------

class TestIPWebcamStream:
    def test_stream_connect(self, no_camera: bool) -> None:
        if no_camera:
            pytest.skip("--no-camera flag set; skipping live stream test.")

        import os
        url = os.environ.get("IPWEBCAM_URL", "http://192.168.1.5:8080/video")

        from perception.stream import IPWebcamStream

        stream = IPWebcamStream(url, timeout=10.0)
        stream.start()
        try:
            frame = stream.latest_frame(timeout=5.0)
            assert frame is not None, "No frame received from IP Webcam stream."
            assert isinstance(frame, np.ndarray)
            assert frame.ndim == 3
            assert frame.shape[2] == 3
        finally:
            stream.stop()

