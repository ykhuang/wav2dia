#!/usr/bin/env python3

import argparse
import os
import sys
from datetime import timedelta

from faster_whisper import WhisperModel


# ------------------------------------------------------------
# Settings
# ------------------------------------------------------------

MODEL_SIZE = "medium"
DEVICE = "cpu"
COMPUTE_TYPE = "int8"

# 常見專有名詞。
# 不需要可以設成空字串 ""。
HOTWORDS = (
    "EtherCAT KINGSTAR RTX64 PDO SDO FoE EoE "
    "CoE SoE CiA402 Motion Control Servo Drive"
)


def format_timestamp(seconds: float) -> str:
    """TXT timestamp: HH:MM:SS"""
    seconds = int(seconds)
    return str(timedelta(seconds=seconds))


def format_srt_timestamp(seconds: float) -> str:
    """SRT timestamp: HH:MM:SS,mmm"""
    milliseconds = int(seconds * 1000)

    hours = milliseconds // 3600000
    milliseconds %= 3600000

    minutes = milliseconds // 60000
    milliseconds %= 60000

    secs = milliseconds // 1000
    milliseconds %= 1000

    return f"{hours:02}:{minutes:02}:{secs:02},{milliseconds:03}"


def main():
    parser = argparse.ArgumentParser(
        description="Long meeting transcription using faster-whisper"
    )

    parser.add_argument(
        "input",
        help="Input audio/video file, e.g. meeting.wav"
    )

    parser.add_argument(
        "--language",
        default=None,
        help="Force language, e.g. zh, en, ja. Default: auto detect"
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Output basename. Default: same as input filename"
    )

    args = parser.parse_args()

    input_file = args.input

    if not os.path.isfile(input_file):
        print(f"ERROR: File not found: {input_file}")
        sys.exit(1)

    if args.output:
        output_base = args.output
    else:
        output_base = os.path.splitext(input_file)[0]

    txt_file = output_base + ".txt"
    srt_file = output_base + ".srt"

    print("=" * 70)
    print("faster-whisper meeting transcription")
    print("=" * 70)
    print(f"Input        : {input_file}")
    print(f"Model        : {MODEL_SIZE}")
    print(f"Device       : {DEVICE}")
    print(f"Compute type : {COMPUTE_TYPE}")
    print(f"Language     : {args.language or 'auto detect'}")
    print(f"TXT output   : {txt_file}")
    print(f"SRT output   : {srt_file}")
    print("=" * 70)

    print("\nLoading Whisper model...")

    model = WhisperModel(
        MODEL_SIZE,
        device=DEVICE,
        compute_type=COMPUTE_TYPE
    )

    print("Model loaded.")
    print("Starting transcription...\n")

    segments, info = model.transcribe(
        input_file,

        # None = Whisper 自動判斷語言
        language=args.language,

        # 對準確率有幫助，但 CPU 會稍微慢一些
        beam_size=5,

        # 移除長時間靜音
        vad_filter=True,

        vad_parameters=dict(
            # 超過 500 ms 視為靜音區段
            min_silence_duration_ms=500
        ),

        # 技術名詞提示
        hotwords=HOTWORDS if HOTWORDS else None,

        # 避免某些情況下重複輸出前面的內容
        condition_on_previous_text=True,
    )

    print(f"Detected language: {info.language}")
    print(f"Language probability: {info.language_probability:.3f}")

    if info.duration:
        print(
            f"Audio duration: "
            f"{format_timestamp(info.duration)}"
        )

    print("\n--- Transcript ---\n")

    segment_count = 0

    # line-buffered，確保每一段立即寫入磁碟
    with (
        open(
            txt_file,
            "w",
            encoding="utf-8",
            buffering=1
        ) as txt,
        open(
            srt_file,
            "w",
            encoding="utf-8",
            buffering=1
        ) as srt
    ):

        # 寫入基本資料
        txt.write(
            f"Source: {input_file}\n"
            f"Model: {MODEL_SIZE} / {COMPUTE_TYPE} / {DEVICE}\n"
            f"Language: {info.language}\n"
            "\n"
        )

        for segment in segments:
            segment_count += 1

            start = segment.start
            end = segment.end
            text = segment.text.strip()

            # TXT
            txt_line = (
                f"[{format_timestamp(start)} "
                f"-> {format_timestamp(end)}] "
                f"{text}"
            )

            print(txt_line)
            txt.write(txt_line + "\n")
            txt.flush()

            # SRT
            srt.write(f"{segment_count}\n")
            srt.write(
                f"{format_srt_timestamp(start)} --> "
                f"{format_srt_timestamp(end)}\n"
            )
            srt.write(text + "\n\n")
            srt.flush()

    print("\n" + "=" * 70)
    print("Transcription finished.")
    print(f"Segments   : {segment_count}")
    print(f"TXT output : {txt_file}")
    print(f"SRT output : {srt_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
