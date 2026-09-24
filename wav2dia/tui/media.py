from __future__ import annotations

import math
import wave
from pathlib import Path


def duration_seconds(path: Path) -> float | None:
    """Return an audio duration without loading the ASR or diarization stack."""
    if path.suffix.lower() == ".wav":
        try:
            with wave.open(str(path), "rb") as source:
                frame_rate = source.getframerate()
                if frame_rate:
                    return source.getnframes() / frame_rate
        except (OSError, wave.Error):
            pass

    try:
        from mutagen import File as open_audio
    except ModuleNotFoundError:
        return None
    try:
        audio = open_audio(path)
        length = getattr(getattr(audio, "info", None), "length", None)
        return float(length) if length is not None and length >= 0 else None
    except Exception:
        return None


def format_duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds):
        return "unavailable"
    total_seconds = max(0, round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{secs:02}"
