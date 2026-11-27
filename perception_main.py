"""Standalone M2 entry point for development and demonstration.

Usage
-----
Live stream from IP Webcam::

    python perception_main.py --url http://192.168.1.5:8080/video

Single image file (no camera needed)::

    python perception_main.py --image path/to/image.jpg

Options::

    --url URL             IP Webcam stream URL (default: http://192.168.1.5:8080/video)
    --image PATH          Path to a static image file (skips live stream)
    --model MODEL         YOLOv8 model name or .pt path (default: yolov8n.pt)
    --confidence FLOAT    Detection confidence threshold (default: 0.5)
    --no-display          Suppress OpenCV preview window
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="M2 Real-Time Environmental Perception — development runner."
    )
    parser.add_argument(
        "--url",
        default="http://192.168.1.5:8080/video",
        help="IP Webcam stream URL (default: %(default)s)",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Path to a static image file; skips the live stream.",
    )
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="YOLOv8 model name or .pt path (default: %(default)s)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Minimum detection confidence threshold (default: %(default)s)",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Suppress the OpenCV preview window.",
    )
    return parser.parse_args()


def _print_frame(frame: object) -> None:  # noqa: ANN001
    """Pretty-print a PerceptionFrame to stdout."""
    from perception.schema import PerceptionFrame

    assert isinstance(frame, PerceptionFrame)
    print(
        f"\n[Frame {frame.frame_id}] "
        f"{frame.frame_width}x{frame.frame_height} | "
        f"t={frame.timestamp:.3f} | "
        f"{len(frame.objects)} object(s)"
    )
    for obj in frame.objects:
        print(
            f"  • {obj.label:<20} conf={obj.confidence:.2f}  "
            f"pos={obj.position}  box={obj.bounding_box}"
        )


def _draw_detections(frame_bgr: object, perception_frame: object) -> object:  # noqa: ANN001
    """Draw bounding boxes and labels onto a BGR frame for preview."""
    import cv2
    import numpy as np
    from perception.schema import DetectedObject, PerceptionFrame

    assert isinstance(frame_bgr, np.ndarray)
    assert isinstance(perception_frame, PerceptionFrame)

    canvas = frame_bgr.copy()
    for obj in perception_frame.objects:
        assert isinstance(obj, DetectedObject)
        x1, y1, x2, y2 = obj.bounding_box
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label_text = f"{obj.label} {obj.confidence:.2f} [{obj.position}]"
        cv2.putText(
            canvas,
            label_text,
            (x1, max(y1 - 8, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    return canvas


def main() -> None:
    args = _parse_args()

    from perception import PerceptionPipeline

    pipeline = PerceptionPipeline(
        stream_url=args.url,
        model_name=args.model,
        confidence_threshold=args.confidence,
    )

    if args.image is not None:
        # ── Static image mode ──────────────────────────────────────────
        import cv2

        bgr = cv2.imread(str(args.image))
        if bgr is None:
            raise SystemExit(f"Could not read image: {args.image}")

        frame = pipeline.process_image(bgr)
        _print_frame(frame)

        if not args.no_display:
            annotated = _draw_detections(bgr, frame)
            cv2.imshow("M2 — Perception (image mode)", annotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return

    # ── Live stream mode ───────────────────────────────────────────────
    import cv2
    from perception.stream import IPWebcamStream

    print(f"Connecting to IP Webcam stream at {args.url} …")
    print("Press Ctrl+C to stop.\n")

    # We need raw frames for display; run stream and pipeline in tandem.
    stream = IPWebcamStream(args.url)
    stream.start()

    try:
        for raw_frame in stream.frames():
            perception_frame = pipeline.process_image(raw_frame)
            _print_frame(perception_frame)

            if not args.no_display:
                annotated = _draw_detections(raw_frame, perception_frame)
                cv2.imshow("M2 — Perception (live)", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        stream.stop()
        if not args.no_display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

