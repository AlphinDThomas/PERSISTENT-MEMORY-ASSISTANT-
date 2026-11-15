"""Silero voice activity detection helpers."""

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
        self.model, utilities = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            trust_repo=True,
        )
        self.model.eval()
        self._get_speech_timestamps: Callable[..., list[SpeechTimestamp]] = utilities[0]

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
