"""Low-latency microphone and file transcription with faster-whisper and Silero VAD.

Install Python dependencies with:
    python -m pip install -r requirements.txt

On Linux, sounddevice may also need PortAudio:
    sudo apt install libportaudio2

The first run downloads the selected Whisper model and Silero VAD from Torch Hub.
"""

from __future__ import annotations

import argparse
import queue
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from silero_vad import SileroVAD


class WhisperSTT:
    """Load Whisper and Silero VAD for file or live microphone transcription."""

    sample_rate = 16_000
    frame_size = 512
    silence_duration = 0.4
    pre_roll_duration = 0.2
    max_speech_duration = 15

    def __init__(self, whisper_model_name: str = "tiny") -> None:
        import torch
        from faster_whisper import WhisperModel

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if self.device == "cuda" else "int8"
        self.whisper_model = WhisperModel(
            whisper_model_name,
            device=self.device,
            compute_type=compute_type,
        )
        self.vad = SileroVAD(sample_rate=self.sample_rate)

    def _transcribe(self, audio: np.ndarray, **options: Any) -> str:
        settings: dict[str, Any] = {
            "beam_size": 1,
            "condition_on_previous_text": False,
        }
        settings.update(options)
        settings["language"] = "en"
        settings["task"] = "transcribe"
        segments, _ = self.whisper_model.transcribe(audio, **settings)
        return " ".join(segment.text.strip() for segment in segments).strip()

    def transcribe_file(
        self, audio_path: str | Path, **transcribe_options: Any
    ) -> str:
        """Transcribe speech regions in an audio file and return their combined text."""
        from faster_whisper.audio import decode_audio

        audio = decode_audio(str(audio_path), sampling_rate=self.sample_rate)
        speech_regions = self.vad.get_speech_timestamps(audio)
        transcripts = [
            self._transcribe(
                audio[region["start"] : region["end"]],
                **transcribe_options,
            )
            for region in speech_regions
            if region["end"] > region["start"]
        ]
        return " ".join(text for text in transcripts if text)

    def transcribe_microphone(self, input_device: int | None = None) -> None:
        """Continuously listen to the default microphone and print recognized phrases."""
        import sounddevice as sd

        audio_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=8)
        pre_roll: deque[np.ndarray] = deque(
            maxlen=round(self.pre_roll_duration * self.sample_rate / self.frame_size)
        )
        silence_frames_required = round(
            self.silence_duration * self.sample_rate / self.frame_size
        )
        max_speech_frames = round(
            self.max_speech_duration * self.sample_rate / self.frame_size
        )
        speech_frames: list[np.ndarray] = []
        silence_frames = 0
        is_speaking = False

        def audio_callback(
            input_data: np.ndarray,
            frame_count: int,
            time_info: Any,
            status: Any,
        ) -> None:
            frame = input_data[:, 0].copy()
            try:
                audio_queue.put_nowait(frame)
            except queue.Full:
                try:
                    audio_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    audio_queue.put_nowait(frame)
                except queue.Full:
                    pass

        print("Listening... Speak into the microphone. Press Ctrl+C to stop.")
        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.frame_size,
            device=input_device,
            callback=audio_callback,
        ):
            try:
                while True:
                    frame = audio_queue.get()
                    speech_probability = self.vad.speech_probability(frame)

                    if not is_speaking:
                        pre_roll.append(frame)
                        if speech_probability >= 0.5:
                            is_speaking = True
                            speech_frames = list(pre_roll)
                            silence_frames = 0
                    else:
                        speech_frames.append(frame)
                        if speech_probability >= 0.5:
                            silence_frames = 0
                        else:
                            silence_frames += 1

                    should_finalize = is_speaking and (
                        silence_frames >= silence_frames_required
                        or len(speech_frames) >= max_speech_frames
                    )
                    if should_finalize:
                        speech_audio = np.concatenate(speech_frames)
                        self.vad.reset_states()
                        transcript = self._transcribe(speech_audio)
                        if transcript:
                            print(transcript, flush=True)
                        speech_frames = []
                        silence_frames = 0
                        is_speaking = False
                        pre_roll.clear()
            except KeyboardInterrupt:
                if speech_frames:
                    transcript = self._transcribe(np.concatenate(speech_frames))
                    if transcript:
                        print(transcript, flush=True)
                print("\nStopped listening.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transcribe live microphone audio or an audio file."
    )
    parser.add_argument(
        "audio_file",
        nargs="?",
        type=Path,
        help="Optional audio file; if omitted, listen continuously to the microphone.",
    )
    parser.add_argument(
        "--model",
        default="tiny",
        help=(
            "faster-whisper model name (default: tiny for lower latency; "
            "e.g. base, small, medium, large-v3)"
        ),
    )
    parser.add_argument(
        "--input-device",
        type=int,
        help="Microphone device index (default: system default input device)",
    )
    args = parser.parse_args()

    stt = WhisperSTT(whisper_model_name=args.model)
    if args.audio_file is not None:
        print(stt.transcribe_file(args.audio_file))
    else:
        stt.transcribe_microphone(input_device=args.input_device)


if __name__ == "__main__":
    main()
