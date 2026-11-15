"""Silero voice activity detection helpers for the speech application."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypedDict

import numpy as np


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
        import queue

        import sounddevice as sd

        frame_size = 512
        audio_queue: queue.Queue[np.ndarray] = queue.Queue()

        def audio_callback(
            input_data: np.ndarray,
            frame_count: int,
            time_info: object,
            status: object,
        ) -> None:
            audio_queue.put(input_data[:, 0].copy())

        print("VAD-only mode: speak into the microphone. Press Ctrl+C to stop.")
        is_speaking = False
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
                    probability = self.speech_probability(audio_queue.get())
                    detected = probability >= 0.5
                    if detected != is_speaking:
                        print(
                            "Speech detected." if detected else "Silence detected.",
                            flush=True,
                        )
                        is_speaking = detected
            except KeyboardInterrupt:
                print("\nStopped VAD.")
