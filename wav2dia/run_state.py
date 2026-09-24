"""Durable state for an individual transcription and diarization run."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

RunStatus = Literal["running", "completed", "cancelled", "failed", "abandoned", "invalid"]
RunStage = Literal["none", "transcribed", "diarized"]


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def state_path(output_prefix: Path) -> Path:
    """Return the sidecar that records the latest attempt for one source output."""

    return output_prefix.parent / f"{output_prefix.name}.run-status.json"


def raw_transcript_path(output_prefix: Path) -> Path:
    """Return the canonical transcript retained after a successful ASR phase."""

    return output_prefix.parent / f"{output_prefix.name}.raw.segments.json"


def source_fingerprint(source: Path) -> dict[str, str | int]:
    stat = source.stat()
    return {
        "path": str(source.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def make_run_state(
    source: Path,
    *,
    status: RunStatus,
    last_completed_stage: RunStage,
    mode: Literal["full", "diarize_only"],
    message: str = "",
    raw_artifacts: dict[str, Path] | None = None,
) -> dict[str, Any]:
    """Create a state document; callers update and persist it between phases."""

    return {
        "schema_version": 1,
        "status": status,
        "last_completed_stage": last_completed_stage,
        "mode": mode,
        "source": source_fingerprint(source),
        "started_at": utc_now(),
        "updated_at": utc_now(),
        "message": message,
        "raw_artifacts": {name: str(path) for name, path in (raw_artifacts or {}).items()},
    }


def update_run_state(state: dict[str, Any], **changes: Any) -> dict[str, Any]:
    state.update(changes)
    state["updated_at"] = utc_now()
    return state


def write_run_state(output_prefix: Path, state: dict[str, Any]) -> Path:
    """Atomically replace the run-state sidecar so interrupted writes stay detectable."""

    destination = state_path(output_prefix)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(state, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink(missing_ok=True)
    return destination


def load_run_state(output_prefix: Path) -> dict[str, Any] | None:
    """Load the latest state; malformed sidecars are represented as an invalid run."""

    path = state_path(output_prefix)
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "schema_version": 1,
            "status": "invalid",
            "last_completed_stage": "none",
            "message": f"Cannot read run status: {type(exc).__name__}: {exc}",
        }
    if not isinstance(value, dict):
        return {
            "schema_version": 1,
            "status": "invalid",
            "last_completed_stage": "none",
            "message": "Run status is not a JSON object.",
        }
    return value


def matches_source(state: dict[str, Any], source: Path) -> bool:
    """Require path, size, and modification time to match before allowing resume."""

    recorded = state.get("source")
    if not isinstance(recorded, dict):
        return False
    try:
        current = source_fingerprint(source)
    except OSError:
        return False
    return all(recorded.get(key) == current[key] for key in ("path", "size", "mtime_ns"))


def has_existing_artifacts(output_prefix: Path) -> bool:
    """Detect old output without trusting it as resumable when its sidecar is absent."""

    suffixes = (".srt", ".md", ".segments.json", ".speakers.txt", ".raw.srt", ".raw.md", ".raw.segments.json")
    return any((output_prefix.parent / f"{output_prefix.name}{suffix}").is_file() for suffix in suffixes)
