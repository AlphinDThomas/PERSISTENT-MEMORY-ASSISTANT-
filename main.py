"""Application entry point for microphone and file speech transcription."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transcribe live English microphone audio or an audio file."
    )
    parser.add_argument(
        "audio_file",
        nargs="?",
        type=Path,
        help="Optional audio file; if omitted, listen continuously to the microphone.",
    )
    parser.add_argument(
        "--component",
        choices=("live", "transcription", "vad"),
        default="live",
        help=(
            "Run live transcription (default), Whisper-only file transcription, "
            "or microphone VAD-only mode."
        ),
    )
    parser.add_argument(
        "--model",
        default="medium",
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

    if args.component == "vad":
        if args.audio_file is not None:
            parser.error("--component vad uses the microphone and does not take an audio file.")
        from speech import SileroVAD

        SileroVAD().listen_to_microphone(input_device=args.input_device)
        return

    if args.component == "transcription":
        if args.audio_file is None:
            parser.error("--component transcription requires an audio file.")
        from speech import WhisperTranscriber

        transcriber = WhisperTranscriber(whisper_model_name=args.model)
        print(transcriber.transcribe_file(args.audio_file))
        return

    from speech import WhisperSTT

    stt = WhisperSTT(whisper_model_name=args.model)
    if args.audio_file is not None:
        print(stt.transcribe_file(args.audio_file))
        return
    stt.transcribe_microphone(input_device=args.input_device)


if __name__ == "__main__":
    main()
