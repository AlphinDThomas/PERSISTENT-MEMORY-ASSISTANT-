"""Structured, rotating log for speech detection and transcription events."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_FILE = Path(__file__).resolve().parents[1] / "logs" / "speech.log"
_LOGGER_NAME = "speech.events"
_MAX_LOG_BYTES = 5 * 1024 * 1024
_BACKUP_COUNT = 3


def log_speech_event(
    *,
    event: str,
    source: str,
    duration_seconds: float,
    transcript: str | None = None,
    transcription_time_seconds: float | None = None,
    error: str | None = None,
) -> None:
    """Write one speech event as a JSON line to the local rotating log."""
    logger = logging.getLogger(_LOGGER_NAME)
    if not logger.handlers:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(
            LOG_FILE,
            maxBytes=_MAX_LOG_BYTES,
            backupCount=_BACKUP_COUNT,
            encoding="utf-8",
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False

    record: dict[str, str | float] = {
        "timestamp": datetime.now().astimezone().isoformat(timespec="milliseconds"),
        "event": event,
        "source": source,
        "duration_seconds": round(duration_seconds, 3),
    }
    if transcript is not None:
        record["transcript"] = transcript
    if transcription_time_seconds is not None:
        record["transcription_time_seconds"] = round(transcription_time_seconds, 3)
    if error is not None:
        record["error"] = error
    logger.info(json.dumps(record, ensure_ascii=False))
