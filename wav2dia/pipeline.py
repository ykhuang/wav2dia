from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .models import Segment, SpeakerTurn, Transcript, Word

SUPPORTED_MEDIA_SUFFIXES = {".wav", ".mp3", ".m4a", ".flac", ".mp4", ".mkv", ".webm"}


class DependencyError(RuntimeError):
    """Raised when an optional local speech dependency has not been installed."""


def validate_media(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Input file does not exist: {path}")
    if path.suffix.lower() not in SUPPORTED_MEDIA_SUFFIXES:
        supported = ", ".join(sorted(SUPPORTED_MEDIA_SUFFIXES))
        raise ValueError(f"Unsupported media type {path.suffix!r}. Supported: {supported}")


def transcribe_audio(
    path: Path,
    *,
    model_name: str,
    device: str,
    compute_type: str,
    language: str | None,
    hotwords: str | None,
    word_timestamps: bool,
) -> Transcript:
    try:
        from faster_whisper import WhisperModel
    except ModuleNotFoundError as exc:
        raise DependencyError(
            "faster-whisper is not installed. Create a Python 3.11/3.12 virtual environment and run "
            "`pip install -e .[transcription]`."
        ) from exc

    validate_media(path)
    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    source_segments, info = model.transcribe(
        str(path),
        language=language,
        beam_size=5,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
        hotwords=hotwords or None,
        condition_on_previous_text=True,
        word_timestamps=word_timestamps,
    )
    segments: list[Segment] = []
    for source in source_segments:
        words = [
            Word(
                start=getattr(word, "start", None),
                end=getattr(word, "end", None),
                text=word.word,
                probability=getattr(word, "probability", None),
            )
            for word in (source.words or [])
        ]
        segments.append(Segment(start=source.start, end=source.end, text=source.text.strip(), words=words))
    return Transcript(
        source=str(path),
        language=getattr(info, "language", None),
        language_probability=getattr(info, "language_probability", None),
        duration=getattr(info, "duration", None),
        model=model_name,
        segments=segments,
    )


def diarize_audio(
    path: Path,
    *,
    model_name: str,
    token_env: str | None,
    min_speakers: int | None,
    max_speakers: int | None,
) -> list[SpeakerTurn]:
    try:
        from pyannote.audio import Pipeline
    except ModuleNotFoundError as exc:
        raise DependencyError(
            "pyannote.audio is not installed. Create a Python 3.11/3.12 virtual environment and run "
            "`pip install -e .[diarization]`."
        ) from exc

    validate_media(path)
    token = os.environ.get(token_env) if token_env else None
    try:
        pipeline = Pipeline.from_pretrained(model_name, token=token)
    except TypeError:  # Compatibility with older pyannote.audio releases.
        pipeline = Pipeline.from_pretrained(model_name, use_auth_token=token)

    options: dict[str, Any] = {}
    if min_speakers is not None:
        options["min_speakers"] = min_speakers
    if max_speakers is not None:
        options["max_speakers"] = max_speakers
    output = pipeline(str(path), **options)
    annotation = getattr(output, "speaker_diarization", output)
    if hasattr(annotation, "itertracks"):
        return [
            SpeakerTurn(start=turn.start, end=turn.end, speaker=speaker)
            for turn, _track, speaker in annotation.itertracks(yield_label=True)
        ]
    return [SpeakerTurn(start=turn.start, end=turn.end, speaker=speaker) for turn, speaker in annotation]


def overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def speaker_for_interval(start: float, end: float, turns: list[SpeakerTurn]) -> str | None:
    best: SpeakerTurn | None = None
    best_overlap = 0.0
    for turn in turns:
        shared = overlap(start, end, turn.start, turn.end)
        if shared > best_overlap:
            best, best_overlap = turn, shared
    return best.speaker if best else None


def assign_speakers(segments: list[Segment], turns: list[SpeakerTurn]) -> list[Segment]:
    """Assign speakers using word timestamps when present, otherwise segment overlap.

    Consecutive words assigned to different speakers become separate subtitle segments.
    This avoids labelling an entire long subtitle with the first speaker when a handoff
    occurs inside it. Text without usable word timestamps remains a single segment.
    """
    aligned: list[Segment] = []
    for segment in segments:
        timed_words = [word for word in segment.words if word.start is not None and word.end is not None]
        if not timed_words:
            aligned.append(
                Segment(
                    start=segment.start,
                    end=segment.end,
                    text=segment.text,
                    speaker=speaker_for_interval(segment.start, segment.end, turns),
                    words=segment.words,
                )
            )
            continue

        groups: list[tuple[str | None, list[Word]]] = []
        for word in timed_words:
            speaker = speaker_for_interval(float(word.start), float(word.end), turns)
            if groups and groups[-1][0] == speaker:
                groups[-1][1].append(word)
            else:
                groups.append((speaker, [word]))

        for speaker, words in groups:
            text = "".join(word.text for word in words).strip()
            if not text:
                continue
            aligned.append(
                Segment(
                    start=float(words[0].start),
                    end=float(words[-1].end),
                    text=text,
                    speaker=speaker,
                    words=words,
                )
            )
    return aligned
