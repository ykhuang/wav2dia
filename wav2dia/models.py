from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class Word:
    start: float | None
    end: float | None
    text: str
    probability: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Word":
        return cls(**value)


@dataclass(slots=True)
class Segment:
    start: float
    end: float
    text: str
    speaker: str | None = None
    words: list[Word] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "speaker": self.speaker,
            "words": [word.to_dict() for word in self.words],
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Segment":
        return cls(
            start=float(value["start"]),
            end=float(value["end"]),
            text=str(value["text"]),
            speaker=value.get("speaker"),
            words=[Word.from_dict(item) for item in value.get("words", [])],
        )


@dataclass(slots=True)
class SpeakerTurn:
    start: float
    end: float
    speaker: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SpeakerTurn":
        return cls(start=float(value["start"]), end=float(value["end"]), speaker=str(value["speaker"]))


@dataclass(slots=True)
class Transcript:
    source: str
    language: str | None
    language_probability: float | None
    duration: float | None
    model: str
    segments: list[Segment]
    diarization: list[SpeakerTurn] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "source": self.source,
            "language": self.language,
            "language_probability": self.language_probability,
            "duration": self.duration,
            "model": self.model,
            "segments": [segment.to_dict() for segment in self.segments],
            "diarization": [turn.to_dict() for turn in self.diarization],
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Transcript":
        return cls(
            source=str(value["source"]),
            language=value.get("language"),
            language_probability=value.get("language_probability"),
            duration=value.get("duration"),
            model=str(value.get("model", "unknown")),
            segments=[Segment.from_dict(item) for item in value.get("segments", [])],
            diarization=[SpeakerTurn.from_dict(item) for item in value.get("diarization", [])],
        )
