"""Silero voice activity detection helpers for the speech application."""

from __future__ import annotations

import math
import queue
import time
from collections.abc import Callable
from typing import TypedDict

import numpy as np

from .event_log import log_speech_event


class SpeechTimestamp(TypedDict):
    start: int
    end: int


class SileroVAD:
    """Load Silero VAD and provide speech scoring and timestamp helpers."""

    def __init__(self, sample_rate: int = 16_000) -> None:
        import torch

        self.torch = torch
        self.sample_rate = sample_rate
        print("Loading Silero VAD model...", flush=True)
        self.model, utilities = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
        )
        self.model.eval()
        self._get_speech_timestamps: Callable[..., list[SpeechTimestamp]] = utilities[0]
        print("Silero VAD model is ready.", flush=True)

    def speech_probability(self, audio_frame: np.ndarray) -> float:
        """Return the model's speech probability for one audio frame."""
        frame_tensor = self.torch.from_numpy(audio_frame)
        with self.torch.inference_mode():
            return float(self.model(frame_tensor, self.sample_rate).item())

    def get_speech_timestamps(self, audio: np.ndarray) -> list[SpeechTimestamp]:
        """Return speech start and end sample indexes for an audio array."""
        return self._get_speech_timestamps(
            self.torch.from_numpy(audio),
            self.model,
            sampling_rate=self.sample_rate,
        )

    def reset_states(self) -> None:
        """Reset recurrent model state between microphone speech segments."""
        self.model.reset_states()

    def listen_to_microphone(self, input_device: int | None = None) -> None:
        """Run microphone input through VAD only and report speech activity."""
        import sounddevice as sd

        frame_size = 512
        audio_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=16)
        dropped_frames = 0
        input_overflows = 0

        def audio_callback(
            input_data: np.ndarray,
            frame_count: int,
            time_info: object,
            status: object,
        ) -> None:
            nonlocal dropped_frames, input_overflows

            if status:
                input_overflows += 1
            frame = input_data[:, 0].copy()
            try:
                audio_queue.put_nowait(frame)
            except queue.Full:
                try:
                    audio_queue.get_nowait()
                except queue.Empty:
                    pass
                dropped_frames += 1
                try:
                    audio_queue.put_nowait(frame)
                except queue.Full:
                    dropped_frames += 1

        print(
            "VAD-only mode: speak into the microphone. "
            "Periodic health status is printed every 5 seconds. Press Ctrl+C to stop."
        )
        is_speaking = False
        processed_audio_frames = 0
        speech_start_frame: int | None = None
        speech_end_frame: int | None = None
        last_frame_time = time.monotonic()
        last_report_time = last_frame_time
        last_dropped_frames = 0
        last_input_overflows = 0
        processed_frames = 0
        latest_probability = 0.0
        latest_rms = 0.0
        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=frame_size,
            device=input_device,
            callback=audio_callback,
        ):
            try:
                while True:
                    try:
                        frame = audio_queue.get(timeout=1.0)
                    except queue.Empty:
                        now = time.monotonic()
                        if now - last_frame_time >= 2.0:
                            print(
                                "VAD warning: no microphone frames received "
                                f"for {now - last_frame_time:.1f}s.",
                                flush=True,
                            )
                            last_frame_time = now
                        continue

                    now = time.monotonic()
                    last_frame_time = now
                    if (
                        dropped_frames != last_dropped_frames
                        or input_overflows != last_input_overflows
                    ):
                        self.reset_states()
                        print(
                            "VAD warning: audio frames were lost "
                            f"(queue drops: {dropped_frames}, "
                            f"input overflows: {input_overflows}); "
                            "resetting VAD state.",
                            flush=True,
                        )
                        last_dropped_frames = dropped_frames
                        last_input_overflows = input_overflows

                    processed_audio_frames += 1
                    latest_probability = self.speech_probability(frame)
                    latest_rms = math.sqrt(float(np.mean(np.square(frame))))
                    processed_frames += 1
                    detected = latest_probability >= 0.5
                    if detected and not is_speaking:
                        is_speaking = True
                        speech_start_frame = processed_audio_frames
                        speech_end_frame = processed_audio_frames
                        print(
                            "Speech detected.",
                            flush=True,
                        )
                    elif detected:
                        speech_end_frame = processed_audio_frames
                    elif is_speaking:
                        assert speech_start_frame is not None
                        assert speech_end_frame is not None
                        duration = (
                            (speech_end_frame - speech_start_frame + 1)
                            * frame_size
                            / self.sample_rate
                        )
                        log_speech_event(
                            event="speech_segment",
                            source="vad_only",
                            duration_seconds=duration,
                        )
                        print(
                            f"Silence detected. Speech duration: {duration:.2f}s.",
                            flush=True,
                        )
                        is_speaking = False
                        speech_start_frame = None
                        speech_end_frame = None

                    if now - last_report_time >= 5.0:
                        print(
                            "VAD alive: "
                            f"processed {processed_frames} frames; "
                            f"audio RMS={latest_rms:.4f}; "
                            f"speech probability={latest_probability:.3f}; "
                            f"dropped frames={dropped_frames}; "
                            f"input overflows={input_overflows}.",
                            flush=True,
                        )
                        processed_frames = 0
                        last_report_time = now
            except KeyboardInterrupt:
                if is_speaking:
                    assert speech_start_frame is not None
                    assert speech_end_frame is not None
                    duration = (
                        (speech_end_frame - speech_start_frame + 1)
                        * frame_size
                        / self.sample_rate
                    )
                    log_speech_event(
                        event="speech_segment",
                        source="vad_only",
                        duration_seconds=duration,
                    )
                print("\nStopped VAD.")
