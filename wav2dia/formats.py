from __future__ import annotations

import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from .models import Segment, SpeakerTurn, Transcript


def srt_timestamp(seconds: float) -> str:
    milliseconds = max(0, round(seconds * 1000))
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    secs, milliseconds = divmod(milliseconds, 1_000)
    return f"{hours:02}:{minutes:02}:{secs:02},{milliseconds:03}"


def display_timestamp(seconds: float) -> str:
    milliseconds = max(0, round(seconds * 1000))
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    secs, milliseconds = divmod(milliseconds, 1_000)
    return f"{hours:02}:{minutes:02}:{secs:02}.{milliseconds:03}"


def parse_srt_timestamp(value: str) -> float:
    match = re.fullmatch(r"(\d+):(\d{2}):(\d{2})[,.](\d{3})", value.strip())
    if not match:
        raise ValueError(f"Invalid SRT timestamp: {value!r}")
    hours, minutes, seconds, milliseconds = (int(item) for item in match.groups())
    return hours * 3600 + minutes * 60 + seconds + milliseconds / 1000


def write_srt(path: Path, segments: list[Segment], include_speakers: bool = True) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for index, segment in enumerate(segments, start=1):
            text = segment.text.strip()
            if include_speakers and segment.speaker:
                text = f"{segment.speaker}: {text}"
            handle.write(
                f"{index}\n{srt_timestamp(segment.start)} --> {srt_timestamp(segment.end)}\n{text}\n\n"
            )


def _escape_markdown(value: str) -> str:
    return " ".join(value.replace("|", "\\|").splitlines())


def write_markdown(path: Path, transcript: Transcript) -> None:
    generated_at = datetime.now(UTC).isoformat(timespec="seconds")
    diarized = bool(transcript.diarization)
    header = [
        "---",
        f"source: {json.dumps(transcript.source, ensure_ascii=False)}",
        f"generated_at: {generated_at}",
        f"language: {transcript.language or 'unknown'}",
        f"asr_model: {transcript.model}",
        f"diarization: {'true' if diarized else 'false'}",
        "schema: wav2dia.transcript/v1",
        "---",
        "",
        "# Transcript",
        "",
        "| # | Start | End | Speaker | Text |",
        "| ---: | --- | --- | --- | --- |",
    ]
    rows = []
    for index, segment in enumerate(transcript.segments, start=1):
        speaker = segment.speaker or "UNKNOWN"
        rows.append(
            f"| {index} | {display_timestamp(segment.start)} | "
            f"{display_timestamp(segment.end)} | {speaker} | {_escape_markdown(segment.text.strip())} |"
        )
    path.write_text("\n".join(header + rows) + "\n", encoding="utf-8", newline="\n")


def write_speaker_turns(path: Path, turns: list[SpeakerTurn]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for turn in turns:
            handle.write(f"[{turn.start:8.2f} -> {turn.end:8.2f}] {turn.speaker}\n")


def read_speaker_turns(path: Path) -> list[SpeakerTurn]:
    pattern = re.compile(r"\[\s*([\d.]+)\s*->\s*([\d.]+)\s*\]\s+(\S+)")
    turns = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        match = pattern.search(line)
        if match:
            turns.append(
                SpeakerTurn(start=float(match.group(1)), end=float(match.group(2)), speaker=match.group(3))
            )
    return turns


def read_srt(path: Path) -> list[Segment]:
    blocks = re.split(r"\r?\n\s*\r?\n", path.read_text(encoding="utf-8-sig").strip())
    segments: list[Segment] = []
    for block in blocks:
        lines = [line.strip() for line in block.splitlines()]
        if len(lines) < 3:
            continue
        match = re.fullmatch(r"(.+?)\s+-->\s+(.+)", lines[1])
        if not match:
            continue
        segments.append(
            Segment(
                start=parse_srt_timestamp(match.group(1)),
                end=parse_srt_timestamp(match.group(2)),
                text=" ".join(lines[2:]),
            )
        )
    return segments


def write_artifacts(output_prefix: Path, transcript: Transcript, include_speakers: bool = True) -> dict[str, Path]:
    """Stage every artifact before replacing prior output with the completed set."""

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    paths = {
        "srt": output_prefix.parent / f"{output_prefix.name}.srt",
        "markdown": output_prefix.parent / f"{output_prefix.name}.md",
        "json": output_prefix.parent / f"{output_prefix.name}.segments.json",
    }
    if transcript.diarization:
        paths["speakers"] = output_prefix.parent / f"{output_prefix.name}.speakers.txt"
    temporary = {
        name: path.with_name(f".{path.name}.{uuid4().hex}.tmp") for name, path in paths.items()
    }
    try:
        write_srt(temporary["srt"], transcript.segments, include_speakers=include_speakers)
        write_markdown(temporary["markdown"], transcript)
        temporary["json"].write_text(
            json.dumps(transcript.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        if "speakers" in temporary:
            write_speaker_turns(temporary["speakers"], transcript.diarization)
        for name, destination in paths.items():
            os.replace(temporary[name], destination)
    finally:
        for path in temporary.values():
            if path.exists():
                path.unlink(missing_ok=True)
    return paths


def read_transcript(path: Path) -> Transcript:
    return Transcript.from_dict(json.loads(path.read_text(encoding="utf-8-sig")))
