"""IP Webcam frame acquisition for M2 — Real-Time Environmental Perception.

Connects to the MJPEG HTTP stream served by the IP Webcam Android application
and delivers decoded BGR frames via a thread-safe queue.

Typical IP Webcam stream URL:  ``http://<phone-ip>:8080/video``
"""

from __future__ import annotations

import queue
import threading
import time
from typing import Iterator

import cv2
import numpy as np


class IPWebcamStream:
    """Acquire frames from an IP Webcam MJPEG stream in a background thread.

    The background thread continuously reads frames from the stream URL and
    places them onto an internal queue.  Callers consume frames via
    :meth:`frames` or :meth:`latest_frame`.

    Args:
        url:       Full URL of the IP Webcam stream, e.g.
                   ``"http://192.168.1.5:8080/video"``.
        maxsize:   Maximum number of frames buffered in the internal queue.
                   Older frames are dropped when the queue is full to avoid
                   unbounded memory growth.
        timeout:   Seconds to wait for the stream to open before raising
                   :class:`RuntimeError`.
    """

    def __init__(
        self,
        url: str,
        maxsize: int = 2,
        timeout: float = 10.0,
    ) -> None:
        self._url = url
        self._queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=maxsize)
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._capture_loop,
            name="ipwebcam-capture",
            daemon=True,
        )
        self._timeout = timeout
        self._last_error: Exception | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> "IPWebcamStream":
        """Start the background capture thread.  Returns *self* for chaining."""
        self._thread.start()
        return self

    def stop(self) -> None:
        """Signal the capture thread to stop and wait for it to finish."""
        self._stop_event.set()
        self._thread.join(timeout=5.0)

    def latest_frame(self, timeout: float = 1.0) -> np.ndarray | None:
        """Return the most recently captured frame, or *None* on timeout.

        Drains older buffered frames so the returned frame is as fresh as
        possible.
        """
        frame = None
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                frame = self._queue.get_nowait()
            except queue.Empty:
                if frame is not None:
                    break
                time.sleep(0.005)
        return frame

    def frames(self, timeout: float = 1.0) -> Iterator[np.ndarray]:
        """Yield frames indefinitely until :meth:`stop` is called."""
        while not self._stop_event.is_set():
            frame = self.latest_frame(timeout=timeout)
            if frame is not None:
                yield frame

    @property
    def url(self) -> str:
        """The stream URL this instance is connected to."""
        return self._url

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _capture_loop(self) -> None:
        """Background thread: open the stream and push frames onto the queue."""
        cap = cv2.VideoCapture(self._url)
        deadline = time.monotonic() + self._timeout
        while not cap.isOpened():
            if time.monotonic() > deadline:
                self._last_error = RuntimeError(
                    f"Could not open IP Webcam stream at '{self._url}' "
                    f"within {self._timeout}s. "
                    "Check that the phone and laptop are on the same network "
                    "and that the URL is correct."
                )
                return
            time.sleep(0.1)
            cap = cv2.VideoCapture(self._url)

        try:
            while not self._stop_event.is_set():
                ok, frame = cap.read()
                if not ok:
                    # Transient read failure — try to reconnect once
                    cap.release()
                    cap = cv2.VideoCapture(self._url)
                    continue

                # Drop the oldest frame if the consumer is too slow
                if self._queue.full():
                    try:
                        self._queue.get_nowait()
                    except queue.Empty:
                        pass

                self._queue.put(frame)
        finally:
            cap.release()

    def __enter__(self) -> "IPWebcamStream":
        return self.start()

    def __exit__(self, *_: object) -> None:
        self.stop()

