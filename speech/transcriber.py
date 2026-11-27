"""Low-latency transcription using faster-whisper and the speech package's VAD.

Install Python dependencies with:
    python -m pip install -r requirements.txt

On Linux, sounddevice may also need PortAudio:
    sudo apt install libportaudio2

The first run downloads the selected Whisper model and Silero VAD from Torch Hub.
"""

from __future__ import annotations

import asyncio
import queue
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from .event_log import log_speech_event
from .vad import SileroVAD


def _send_transcript_to_assistant(transcript: str) -> None:
    if not transcript.strip():
        return

    from vlm_client import generate_response

    asyncio.run(generate_response(transcript))


class WhisperTranscriber:
    """Load faster-whisper for English transcription without requiring VAD."""

    sample_rate = 16_000

    def __init__(self, whisper_model_name: str = "tiny") -> None:
        import torch
        from faster_whisper import WhisperModel

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.compute_type = "float16" if self.device == "cuda" else "int8"
        print(
            f"Loading faster-whisper model '{whisper_model_name}' "
            f"on {self.device} ({self.compute_type}). "
            "The first run may download several gigabytes for larger models.",
            flush=True,
        )
        self.whisper_model = WhisperModel(
            whisper_model_name,
            device=self.device,
            compute_type=self.compute_type,
        )
        print(f"Whisper model '{whisper_model_name}' is ready.", flush=True)

    def transcribe_audio(self, audio: np.ndarray, **options: Any) -> str:
        """Transcribe an audio array as English speech."""
        settings: dict[str, Any] = {
            "beam_size": 1,
            "condition_on_previous_text": False,
            "language": "en",
            "task": "transcribe",
        }
        settings.update(options)
        settings["language"] = "en"
        settings["task"] = "transcribe"
        segments, _ = self.whisper_model.transcribe(audio, **settings)
        return " ".join(segment.text.strip() for segment in segments).strip()

    def transcribe_file(self, audio_path: str | Path) -> str:
        """Transcribe an entire audio file without running voice activity detection."""
        from faster_whisper.audio import decode_audio

        audio = decode_audio(str(audio_path), sampling_rate=self.sample_rate)
        started = time.perf_counter()
        try:
            transcript = self.transcribe_audio(audio)
        except Exception as error:
            log_speech_event(
                event="transcription",
                source="whisper_file",
                duration_seconds=audio.size / self.sample_rate,
                transcription_time_seconds=time.perf_counter() - started,
                error=str(error),
            )
            raise
        log_speech_event(
            event="transcription",
            source="whisper_file",
            duration_seconds=audio.size / self.sample_rate,
            transcript=transcript,
            transcription_time_seconds=time.perf_counter() - started,
        )
        _send_transcript_to_assistant(transcript)
        return transcript


class WhisperSTT(WhisperTranscriber):
    """Combine Whisper transcription with Silero VAD for live microphone input."""

    frame_size = 512
    silence_duration = 0.4
    pre_roll_duration = 0.2
    max_speech_duration = 15

    def __init__(self, whisper_model_name: str = "tiny") -> None:
        super().__init__(whisper_model_name)
        self.vad = SileroVAD(sample_rate=self.sample_rate)

    def _transcribe(self, audio: np.ndarray, **options: Any) -> str:
        return self.transcribe_audio(audio, **options)

    def transcribe_file(
        self, audio_path: str | Path, **transcribe_options: Any
    ) -> str:
        """Transcribe speech regions in an audio file and return their combined text."""
        from faster_whisper.audio import decode_audio

        audio = decode_audio(str(audio_path), sampling_rate=self.sample_rate)
        speech_regions = self.vad.get_speech_timestamps(audio)
        transcripts = []
        for region in speech_regions:
            if region["end"] <= region["start"]:
                continue
            speech_audio = audio[region["start"] : region["end"]]
            duration = speech_audio.size / self.sample_rate
            started = time.perf_counter()
            try:
                transcript = self.transcribe_audio(
                    speech_audio,
                    **transcribe_options,
                )
            except Exception as error:
                log_speech_event(
                    event="speech_segment",
                    source="file",
                    duration_seconds=duration,
                    transcription_time_seconds=time.perf_counter() - started,
                    error=str(error),
                )
                raise
            log_speech_event(
                event="speech_segment",
                source="file",
                duration_seconds=duration,
                transcript=transcript,
                transcription_time_seconds=time.perf_counter() - started,
            )
            _send_transcript_to_assistant(transcript)
            transcripts.append(transcript)
        return " ".join(text for text in transcripts if text)

    def transcribe_microphone(self, input_device: int | None = None) -> None:
        """Continuously listen to the default microphone and print recognized phrases."""
        import sounddevice as sd

        audio_queue: queue.Queue[np.ndarray] = queue.Queue()
        transcription_queue: queue.Queue[tuple[np.ndarray, float] | None] = queue.Queue()
        transcription_errors: queue.Queue[Exception] = queue.Queue(maxsize=1)
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
        processed_audio_frames = 0
        speech_start_frame: int | None = None
        speech_end_frame: int | None = None

        def transcribe_in_background() -> None:
            while True:
                item = transcription_queue.get()
                if item is None:
                    return
                speech_audio, duration = item
                started = time.perf_counter()
                try:
                    print(f"Transcribing detected speech ({duration:.1f}s)...", flush=True)
                    transcript = self._transcribe(speech_audio)
                except Exception as error:
                    log_speech_event(
                        event="speech_segment",
                        source="live",
                        duration_seconds=duration,
                        transcription_time_seconds=time.perf_counter() - started,
                        error=str(error),
                    )
                    transcription_errors.put(error)
                    return
                log_speech_event(
                    event="speech_segment",
                    source="live",
                    duration_seconds=duration,
                    transcript=transcript,
                    transcription_time_seconds=time.perf_counter() - started,
                )
                if transcript:
                    print(transcript, flush=True)
                    _send_transcript_to_assistant(transcript)

        def audio_callback(
            input_data: np.ndarray,
            frame_count: int,
            time_info: Any,
            status: Any,
        ) -> None:
            audio_queue.put(input_data[:, 0].copy())

        transcription_thread = threading.Thread(
            target=transcribe_in_background,
            name="whisper-transcription",
            daemon=True,
        )
        transcription_thread.start()
        print(
            "Listening... Speak into the microphone. "
            "Transcription will print when each phrase is processed. Press Ctrl+C to stop.",
            flush=True,
        )
        try:
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
                        try:
                            error = transcription_errors.get_nowait()
                        except queue.Empty:
                            error = None
                        if error is not None:
                            raise RuntimeError("Whisper transcription failed.") from error

                        try:
                            frame = audio_queue.get(timeout=0.1)
                        except queue.Empty:
                            continue
                        processed_audio_frames += 1
                        speech_probability = self.vad.speech_probability(frame)

                        if not is_speaking:
                            pre_roll.append(frame)
                            if speech_probability >= 0.5:
                                is_speaking = True
                                speech_frames = list(pre_roll)
                                silence_frames = 0
                                speech_start_frame = processed_audio_frames
                                speech_end_frame = processed_audio_frames
                        else:
                            speech_frames.append(frame)
                            if speech_probability >= 0.5:
                                silence_frames = 0
                                speech_end_frame = processed_audio_frames
                            else:
                                silence_frames += 1

                        should_finalize = is_speaking and (
                            silence_frames >= silence_frames_required
                            or len(speech_frames) >= max_speech_frames
                        )
                        if should_finalize:
                            assert speech_start_frame is not None
                            assert speech_end_frame is not None
                            duration = (
                                (speech_end_frame - speech_start_frame + 1)
                                * self.frame_size
                                / self.sample_rate
                            )
                            transcription_queue.put(
                                (np.concatenate(speech_frames), duration)
                            )
                            self.vad.reset_states()
                            speech_frames = []
                            silence_frames = 0
                            is_speaking = False
                            speech_start_frame = None
                            speech_end_frame = None
                            pre_roll.clear()
                except KeyboardInterrupt:
                    if speech_frames:
                        assert speech_start_frame is not None
                        assert speech_end_frame is not None
                        duration = (
                            (speech_end_frame - speech_start_frame + 1)
                            * self.frame_size
                            / self.sample_rate
                        )
                        transcription_queue.put(
                            (np.concatenate(speech_frames), duration)
                        )
                    print("\nStopped listening.")
        finally:
            transcription_queue.put(None)
            transcription_thread.join()

        try:
            error = transcription_errors.get_nowait()
        except queue.Empty:
            error = None
        if error is not None:
            raise RuntimeError("Whisper transcription failed.") from error
