import re
import argparse


def srt_time_to_seconds(t):
    h, m, rest = t.split(":")
    s, ms = rest.split(",")

    return (
        int(h) * 3600
        + int(m) * 60
        + int(s)
        + int(ms) / 1000
    )


def seconds_to_hms(seconds):
    seconds = int(seconds)

    h = seconds // 3600
    seconds %= 3600

    m = seconds // 60
    s = seconds % 60

    return f"{h:02}:{m:02}:{s:02}"


def parse_srt(filename):
    with open(filename, "r", encoding="utf-8-sig") as f:
        content = f.read().strip()

    blocks = re.split(r"\n\s*\n", content)

    subtitles = []

    for block in blocks:
        lines = block.splitlines()

        if len(lines) < 3:
            continue

        index = lines[0].strip()

        time_match = re.match(
            r"(.+?)\s+-->\s+(.+)",
            lines[1].strip()
        )

        if not time_match:
            continue

        start_str = time_match.group(1)
        end_str = time_match.group(2)

        text = " ".join(
            line.strip()
            for line in lines[2:]
        )

        subtitles.append({
            "index": index,
            "start_str": start_str,
            "end_str": end_str,
            "start": srt_time_to_seconds(start_str),
            "end": srt_time_to_seconds(end_str),
            "text": text
        })

    return subtitles


def parse_speakers(filename):
    speakers = []

    pattern = re.compile(
        r"\[\s*([\d.]+)\s*->\s*([\d.]+)\s*\]\s+(\S+)"
    )

    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            match = pattern.search(line)

            if not match:
                continue

            speakers.append({
                "start": float(match.group(1)),
                "end": float(match.group(2)),
                "speaker": match.group(3)
            })

    return speakers


def overlap(a_start, a_end, b_start, b_end):
    return max(
        0,
        min(a_end, b_end)
        - max(a_start, b_start)
    )


def find_speaker(subtitle, speakers):
    best_speaker = "UNKNOWN"
    best_overlap = 0

    for sp in speakers:
        duration = overlap(
            subtitle["start"],
            subtitle["end"],
            sp["start"],
            sp["end"]
        )

        if duration > best_overlap:
            best_overlap = duration
            best_speaker = sp["speaker"]

    return best_speaker


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "srt",
        help="Whisper SRT file"
    )

    parser.add_argument(
        "speakers",
        help="pyannote speaker TXT file"
    )

    parser.add_argument(
        "--output",
        default="meeting_with_speakers",
        help="Output basename"
    )

    args = parser.parse_args()

    subtitles = parse_srt(args.srt)
    speakers = parse_speakers(args.speakers)

    txt_output = args.output + ".txt"
    srt_output = args.output + ".srt"

    with open(
        txt_output,
        "w",
        encoding="utf-8"
    ) as txt, open(
        srt_output,
        "w",
        encoding="utf-8"
    ) as srt:

        for i, sub in enumerate(subtitles, 1):

            speaker = find_speaker(
                sub,
                speakers
            )

            # LLM-friendly TXT
            txt.write(
                f"[{seconds_to_hms(sub['start'])} - "
                f"{seconds_to_hms(sub['end'])}] "
                f"{speaker}: {sub['text']}\n"
            )

            # SRT
            srt.write(f"{i}\n")

            srt.write(
                f"{sub['start_str']} --> "
                f"{sub['end_str']}\n"
            )

            srt.write(
                f"{speaker}: "
                f"{sub['text']}\n\n"
            )

    print(f"Created: {txt_output}")
    print(f"Created: {srt_output}")


if __name__ == "__main__":
    main()
