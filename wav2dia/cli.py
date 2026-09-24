from __future__ import annotations

import argparse
import os
import sys
import tomllib
from pathlib import Path
from typing import Any

from .formats import read_speaker_turns, read_srt, read_transcript, write_artifacts, write_speaker_turns
from .llm import LLMError, post_process
from .models import Transcript
from .pipeline import DependencyError, assign_speakers, diarize_audio, transcribe_audio, validate_media

DEFAULT_CONFIG: dict[str, Any] = {
    "asr": {
        "model": "medium",
        "device": "cpu",
        "compute_type": "int8",
        "language": None,
        "hotwords": "",
        "word_timestamps": True,
    },
    "diarization": {
        "model": "pyannote/speaker-diarization-community-1",
        "token_env": "HF_TOKEN",
        "min_speakers": None,
        "max_speakers": None,
    },
    "llm": {
        "backend": "ollama",
        "max_input_chars": 180000,
        "ollama": {"base_url": "http://localhost:11434", "model": "", "timeout_seconds": 600},
        "codex": {"executable": "codex", "timeout_seconds": 900},
        "agy": {"executable": "agy", "timeout_seconds": 900, "model": ""},
    },
}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = {key: value.copy() if isinstance(value, dict) else value for key, value in base.items()}
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | None) -> dict[str, Any]:
    if path is None:
        default = Path("config.toml")
        return deep_merge(DEFAULT_CONFIG, tomllib.loads(default.read_text(encoding="utf-8")) if default.is_file() else {})
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {config_path}")
    return deep_merge(DEFAULT_CONFIG, tomllib.loads(config_path.read_text(encoding="utf-8")))


def option(args: argparse.Namespace, name: str, config: dict[str, Any], section: str) -> Any:
    value = getattr(args, name, None)
    return config[section].get(name) if value is None else value


def prefix_for(input_path: Path, output_dir: str | None, output_name: str | None) -> Path:
    directory = Path(output_dir) if output_dir else input_path.parent
    name = output_name or input_path.stem
    return directory / name


def asr_settings(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_name": option(args, "model", config, "asr"),
        "device": option(args, "device", config, "asr"),
        "compute_type": option(args, "compute_type", config, "asr"),
        "language": option(args, "language", config, "asr"),
        "hotwords": option(args, "hotwords", config, "asr"),
        "word_timestamps": not args.no_word_timestamps
        if args.no_word_timestamps
        else bool(config["asr"].get("word_timestamps", True)),
    }


def diarization_settings(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_name": args.diarization_model
        if args.diarization_model is not None
        else config["diarization"]["model"],
        "token_env": option(args, "token_env", config, "diarization"),
        "min_speakers": option(args, "min_speakers", config, "diarization"),
        "max_speakers": option(args, "max_speakers", config, "diarization"),
    }


def print_paths(paths: dict[str, Path]) -> None:
    for label, path in paths.items():
        print(f"{label:9}: {path}")


def command_transcribe(args: argparse.Namespace, config: dict[str, Any]) -> None:
    source = Path(args.input)
    transcript = transcribe_audio(source, **asr_settings(args, config))
    paths = write_artifacts(prefix_for(source, args.output_dir, args.output_name), transcript, include_speakers=False)
    print_paths(paths)


def command_diarize(args: argparse.Namespace, config: dict[str, Any]) -> None:
    source = Path(args.input)
    turns = diarize_audio(source, **diarization_settings(args, config))
    output_prefix = prefix_for(source, args.output_dir, args.output_name)
    output = output_prefix.parent / f"{output_prefix.name}.speakers.txt"
    output.parent.mkdir(parents=True, exist_ok=True)
    write_speaker_turns(output, turns)
    print(f"speakers : {output}")


def command_run(args: argparse.Namespace, config: dict[str, Any]) -> None:
    source = Path(args.input)
    transcript = transcribe_audio(source, **asr_settings(args, config))
    target = prefix_for(source, args.output_dir, args.output_name)
    raw_paths = write_artifacts(target.with_name(target.name + ".raw"), transcript, include_speakers=False)
    turns = diarize_audio(source, **diarization_settings(args, config))
    transcript.diarization = turns
    transcript.segments = assign_speakers(transcript.segments, turns)
    final_paths = write_artifacts(target, transcript, include_speakers=True)
    print("Raw transcription")
    print_paths(raw_paths)
    print("Speaker-aligned transcription")
    print_paths(final_paths)


def command_merge(args: argparse.Namespace, _config: dict[str, Any]) -> None:
    srt_path = Path(args.srt)
    speaker_path = Path(args.speakers)
    segments = read_srt(srt_path)
    turns = read_speaker_turns(speaker_path)
    if not segments:
        raise ValueError(f"No valid subtitle blocks found in {srt_path}")
    if not turns:
        raise ValueError(f"No speaker turns found in {speaker_path}")
    transcript = Transcript(
        source=str(srt_path),
        language=None,
        language_probability=None,
        duration=max(segment.end for segment in segments),
        model="external-srt",
        segments=assign_speakers(segments, turns),
        diarization=turns,
    )
    target = prefix_for(srt_path, args.output_dir, args.output_name)
    print_paths(write_artifacts(target, transcript, include_speakers=True))


def command_llm(args: argparse.Namespace, config: dict[str, Any]) -> None:
    source = Path(args.input)
    if not source.is_file():
        raise FileNotFoundError(f"Input file does not exist: {source}")
    backend = args.backend or config["llm"]["backend"]
    default_suffix = ".md" if args.task == "markdown" else source.suffix
    output = Path(args.output) if args.output else source.with_name(f"{source.stem}.{args.task}{default_suffix}")
    post_process(
        source=source,
        output=output,
        task=args.task,
        backend=backend,
        config=config["llm"],
        target_language=args.target_language,
        max_input_chars=int(config["llm"]["max_input_chars"]),
    )
    print(f"artifact : {output}")
    print(f"metadata : {output.with_suffix(output.suffix + '.llm-meta.json')}")


def add_config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", help="TOML configuration file; defaults to ./config.toml when present")


def add_asr_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", help="faster-whisper model name")
    parser.add_argument("--device", help="faster-whisper device, e.g. cpu or cuda")
    parser.add_argument("--compute-type", dest="compute_type", help="e.g. int8, float16")
    parser.add_argument("--language", help="BCP-47-ish language code, e.g. zh, en, ja; default auto-detect")
    parser.add_argument("--hotwords", help="Space-separated terminology prompt")
    parser.add_argument("--no-word-timestamps", action="store_true", help="Disable word timestamps and speaker handoff splitting")


def add_diarization_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--diarization-model", help="pyannote pipeline model")
    parser.add_argument("--token-env", help="Environment variable containing a Hugging Face token")
    parser.add_argument("--min-speakers", type=int, help="Known minimum number of speakers")
    parser.add_argument("--max-speakers", type=int, help="Known maximum number of speakers")


def add_output_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-dir", help="Directory for generated artifacts")
    parser.add_argument("--output-name", help="Artifact basename; defaults to input filename")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="wav2dia",
        description="Transcribe WAV/MP3 media, diarize speakers, and generate SRT/Markdown/JSON artifacts.",
    )
    add_config_argument(parser)
    subparsers = parser.add_subparsers(dest="command", required=True)

    transcribe = subparsers.add_parser("transcribe", help="Transcribe audio/video into raw SRT, Markdown, and JSON")
    transcribe.add_argument("input", help="Input .wav, .mp3, or other supported media")
    add_output_arguments(transcribe)
    add_asr_arguments(transcribe)

    diarize = subparsers.add_parser("diarize", help="Identify speaker turns in media")
    diarize.add_argument("input", help="Input .wav, .mp3, or other supported media")
    add_output_arguments(diarize)
    add_diarization_arguments(diarize)

    run = subparsers.add_parser("run", help="Run transcription, diarization, and speaker alignment")
    run.add_argument("input", help="Input .wav, .mp3, or other supported media")
    add_output_arguments(run)
    add_asr_arguments(run)
    add_diarization_arguments(run)

    merge = subparsers.add_parser("merge", help="Add speakers from a pyannote TXT file to an existing SRT")
    merge.add_argument("srt", help="Input SRT")
    merge.add_argument("speakers", help="Speaker turn TXT, e.g. [ 0.00 -> 4.20] SPEAKER_00")
    add_output_arguments(merge)

    llm = subparsers.add_parser("llm", help="Optional LLM correction, translation, or Markdown normalization")
    llm.add_argument("task", choices=("correct", "translate", "markdown"))
    llm.add_argument("input", help="Existing .srt, .md, or .json transcript artifact")
    llm.add_argument("--backend", choices=("ollama", "codex", "agy"), help="LLM backend")
    llm.add_argument("--target-language", help="Required for translate, e.g. English or ja")
    llm.add_argument("--output", help="Output artifact path")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        if args.command in {"transcribe", "diarize", "run"}:
            validate_media(Path(args.input))
        commands = {
            "transcribe": command_transcribe,
            "diarize": command_diarize,
            "run": command_run,
            "merge": command_merge,
            "llm": command_llm,
        }
        commands[args.command](args, config)
    except (DependencyError, LLMError, FileNotFoundError, ValueError, OSError, tomllib.TOMLDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
