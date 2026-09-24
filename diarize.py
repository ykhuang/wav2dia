from pyannote.audio import Pipeline
from datetime import datetime

print(f"[{datetime.now()}] Loading pipeline...", flush=True)

pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-community-1"
)

print(f"[{datetime.now()}] Starting diarization...", flush=True)

output = pipeline(
    "CRA_Day2.wav",
    min_speakers=3,
    max_speakers=7
)

print(f"[{datetime.now()}] Diarization finished. Writing results...", flush=True)

with open("CRA_Day2_speakers.txt", "w", encoding="utf-8") as f:
    for turn, speaker in output.speaker_diarization:
        line = f"[{turn.start:8.2f} -> {turn.end:8.2f}] {speaker}"
        print(line, flush=True)
        f.write(line + "\n")

print(f"[{datetime.now()}] Done.", flush=True)
