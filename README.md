# English Speech Transcription

This project listens to live microphone audio and prints English transcripts
continuously. It can also transcribe an audio file.

## Setup

From the project directory, create and activate a virtual environment, then
install the Python dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
```

On Linux, install PortAudio if microphone support requires it:

```bash
sudo apt install libportaudio2
```

## Run

Start live microphone transcription:

```bash
python main.py
```

Transcribe an audio file:

```bash
python main.py path/to/audio.wav
```

Choose a different faster-whisper model or microphone device with `--model` or
`--input-device`. The default model is `tiny` for lower latency. Model loading
and downloads are reported in the terminal; medium and large models may take
several minutes to download and need substantially more RAM. Live audio capture
continues while Whisper processes each detected phrase. Press Ctrl+C to stop.

Run components independently:

```bash
# Whisper only (does not load Silero VAD)
python main.py --component transcription path/to/audio.wav --model medium

# Silero VAD only (does not load Whisper)
python main.py --component vad
```

The transcription-only mode accepts an audio file. The VAD-only mode tests
microphone speech detection and prints when speech or silence is detected.
While running VAD-only mode, a health line appears every five seconds with the
audio RMS level, current speech probability, and any dropped audio frames or
input overflows. If the RMS stays near zero while you speak, check the selected
microphone with `--input-device`.

Speech events and transcripts are written locally as JSON Lines to
`logs/speech.log`. VAD-only mode logs each detected speech segment and its
duration. Live and VAD-enabled file transcription also log the segment duration
and transcript together, including `transcription_time_seconds` for Whisper
inference. Whisper-only file transcription logs the transcript, audio duration,
and inference time. The log is rotated when it grows and is excluded from Git;
transcripts may contain private speech, so review the log before sharing it.

## Module layout

- `main.py` is the single application entry point.
- `speech/transcriber.py` handles microphone capture and faster-whisper.
- `speech/vad.py` encapsulates Silero VAD loading and inference.

Keeping VAD in the `speech` package avoids shadowing the third-party
`silero_vad` package that Torch Hub loads.