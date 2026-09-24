# Wav2Dia

Wav2Dia consolidates the original faster-whisper transcription, pyannote speaker diarization, and SRT merge scripts into a reproducible local pipeline. It accepts `.wav` and `.mp3` at minimum (and also `.m4a`, `.flac`, `.mp4`, `.mkv`, `.webm` when the decoder supports them).

The speaker-aligned `run` command writes four artifacts with one basename:

- `meeting.srt` — video-ready subtitles, with `SPEAKER_00:` labels.
- `meeting.md` — analysis-friendly Markdown with YAML metadata and one row per timed segment.
- `meeting.segments.json` — the canonical machine-readable data; preserve this for later re-rendering.
- `meeting.speakers.txt` — diarization turns in the legacy-compatible text format.

It also retains `meeting.raw.*` artifacts before diarization so ASR output and speaker alignment can be reviewed separately.

## Setup

Use Python 3.11 or 3.12. The repository was scanned on Python 3.14, where the speech dependencies were not installed and may not be supported.

### WSL / Ubuntu transcription and diarization prerequisites

The TUI-only installation is not enough to run `transcribe`, `diarize`, or `run`. For the default CPU configuration, prepare the following inside WSL:

| Requirement | Why it is needed | Install / setup |
| --- | --- | --- |
| Python 3.11 or 3.12 with `venv` | Project runtime and isolated dependencies | `sudo apt update && sudo apt install -y python3-venv` |
| FFmpeg | `pyannote.audio` `community-1` audio decoding | `sudo apt install -y ffmpeg` |
| `faster-whisper` | Local speech-to-text; pulls CTranslate2 and PyAV through pip | included in the `transcription` extra |
| `pyannote.audio` | Local speaker diarization; pulls its PyTorch dependencies through pip | included in the `diarization` extra |
| Hugging Face account, accepted model conditions, and access token | Downloads the default `pyannote/speaker-diarization-community-1` pipeline | accept the model conditions, create a read-capable token, and export `HF_TOKEN` |
| Internet access on the first run | Downloads the Whisper model and pyannote pipeline into the local Hugging Face cache | required once per model/pipeline unless pre-downloaded |

`faster-whisper` can decode audio through its PyAV dependency without a system FFmpeg executable. However, the selected pyannote `community-1` pipeline uses `torchcodec`, whose official setup requires FFmpeg; install it for the combined pipeline.

From the repository root, create or activate the WSL virtual environment and install every project extra:

```bash
cd /mnt/c/Users/DELL/Document/Wav2Dia
python3 --version                    # must be 3.11 or 3.12
python3 -m venv .venv                # omit when .venv already exists
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[all]'
cp config.example.toml config.toml    # omit when config.toml already exists
```

If the TUI extra was installed earlier, the smaller follow-up install is enough:

```bash
. .venv/bin/activate
python -m pip install -e '.[transcription,diarization]'
```

### Install the speech packages explicitly

The project-extra command above installs both packages. The equivalent direct package commands are provided here for visibility and troubleshooting:

```bash
. .venv/bin/activate
python -m pip install 'faster-whisper>=1.1,<2'
python -m pip install 'pyannote.audio>=4,<5'
```

For normal project use, prefer the single project-extra command because it records the intended version ranges and keeps the `wav2dia` command installed:

```bash
python -m pip install -e '.[transcription,diarization]'
```

If you use the two direct commands on a fresh environment instead, also install this project and its TUI dependencies:

```bash
python -m pip install -e '.[tui]'
```

Before the first diarization run, visit [the default pyannote model page](https://huggingface.co/pyannote/speaker-diarization-community-1), accept its user conditions, and create an access token. The TUI checks the shell environment first and then the ignored repository-root `.env` file. Copy the supplied example and put the token only in `.env`:

```bash
cp .env.example .env
# Edit .env and set: HF_TOKEN=hf_your_token_here
```

For CLI-only sessions, exporting it in the active shell also works:

```bash
export HF_TOKEN='hf_your_token_here'
```

Run this inexpensive preflight after installation. It checks the executable, package imports, and CLI without downloading models or processing audio:

```bash
ffmpeg -version
python -c "from faster_whisper import WhisperModel; from pyannote.audio import Pipeline; print('Speech dependencies: OK')"
wav2dia --help
```

GPU acceleration is optional and is not needed by the included default configuration (`device = "cpu"`, `compute_type = "int8"`). If you later change ASR to CUDA, first confirm `nvidia-smi` works inside WSL and install CUDA 12-compatible cuBLAS and cuDNN 9 for faster-whisper/CTranslate2. Keep the CPU configuration until that setup has been verified.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e '.[all]'
Copy-Item config.example.toml config.toml
```

The PowerShell commands above are the Windows-native equivalent. For the WSL workflow described by this project, use the Bash commands above. A Hugging Face token is required to download or access the configured pyannote pipeline:

```powershell
$env:HF_TOKEN = "..."
```

## Main workflow

```powershell
wav2dia run .\meeting.mp3 --output-dir .\output --language zh --min-speakers 3 --max-speakers 7
```

Or run the phases individually:

```powershell
wav2dia transcribe .\meeting.wav --output-dir .\output
wav2dia diarize .\meeting.wav --output-dir .\output
wav2dia merge .\output\meeting.raw.srt .\output\meeting.speakers.txt --output-dir .\output --output-name meeting
```

The `merge` command preserves compatibility with the old `merge_speaker.py` workflow. When word timestamps are available, `run` improves on the old largest-overlap rule by splitting a transcription segment when the detected speaker changes.

## TUI

The TUI scans the workspace `files/` directory and displays `Files` and `Setup` in the left tree. It creates `files/` on first launch. Install all extras before running real transcription and diarization:

```bash
python3 -m pip install -e '.[all]'
python3 -m wav2dia.tui
# Equivalent after installation: wav2dia-tui
```

Press `q` to close and `F5` to reload the workspace and re-check dependencies. Under `Setup`, select `Diarize` and press `Enter` to set `min_speakers` and `max_speakers`; they default to `3` and `6` and apply only to jobs launched during the current TUI session (they do not rewrite `config.toml`). The tree handles `↑`/`↓`; hovering a media file displays its duration in the lower pane. Press `Enter` on `Dependency` to expand the dependency list: failed Python package/version checks and system-command checks (currently `ffmpeg -version`) are red, including the parent item. `Hugging Face access` is a separate entry; it verifies only that the configured token variable is present in the shell or `.env`, never exposing the token. Startup checks never import speech-model packages; import and native-library errors are reported only when a real job runs.

Press `Enter` on a WAV/MP3 file to first review a confirmation dialog with audio duration and the selected `min_speakers` / `max_speakers`, then choose **OK** to run **Transcribing → Diarizing** or **Cancel** to leave it untouched. `Ctrl+C` requests cancellation: the current model call is blocking, so cancellation takes effect at the next safe stage boundary. A cancellation during transcription never starts diarization. A cancellation after a raw transcript is complete retains the raw `*.raw.segments.json` and offers **Skip to diarize** on the next attempt.

Every run writes `<basename>.run-status.json`. This sidecar stores the source fingerprint, status, completed stage, and message without changing SRT/Markdown/TXT syntax. Results from cancelled or abandoned runs display orange; failed or source-mismatched runs display red. When matching complete output exists, the TUI offers **Overwrite / Cancel**. When a matching raw transcript exists after cancellation or failure, it offers **Overwrite / Skip to diarize / Cancel**. Artifacts are staged in temporary files and replaced only after each artifact set has been fully generated.

Use `Tab` to move between panes; the focused pane receives a high-contrast border. In `Transcription results`, press `Enter` on a source group to expand or collapse its artifacts. Moving the cursor onto an SRT or Markdown artifact displays its file content in the lower preview pane.

Press `p` to open the independent pipeline-progress visual test. It is a centered 60%-width dialog: the ASCII spinner rotates continuously and the bar advances one segment per 10 seconds, returning to zero after it fills. Press `d` or `t` to preview the `Diarizing` and `Transcribing` labels; `Esc` closes this visual test.

## LLM post-processing

Core transcription and diarization are deterministic and do not require a cloud LLM. The separate `llm` command applies a chosen backend to an existing artifact and saves a small provenance sidecar (`.llm-meta.json`). It never overwrites the source unless you explicitly choose that output path.

```powershell
# Correct the transcript while retaining SRT indices/timestamps/speaker labels.
wav2dia llm correct .\output\meeting.srt --backend ollama --output .\output\meeting.corrected.srt

# Translate subtitles while preserving timing and speaker labels.
wav2dia llm translate .\output\meeting.srt --backend agy --target-language English --output .\output\meeting.en.srt

# Normalize an existing transcript into a Markdown analysis artifact.
wav2dia llm markdown .\output\meeting.segments.json --backend codex --output .\output\meeting.reviewed.md
```

Backends are configured in `config.toml`:

- **Ollama** calls its local `/api/generate` endpoint using `base_url` and `model`.
- **Codex CLI** passes the full prompt through standard input to `codex exec --ephemeral --skip-git-repo-check -`. `codex exec` is the documented non-interactive interface; it uses saved CLI authentication by default and runs read-only by default. See [OpenAI’s non-interactive Codex guide](https://learn.chatgpt.com/docs/non-interactive-mode).
- **Antigravity CLI (`agy`)** uses its documented `stream-json` stdin/stdout mode. This avoids Windows command-line length limits for long transcripts. Authenticate once interactively before automating it. See [Antigravity headless mode](https://www.antigravity.google/docs/cli/headless/).

For long transcripts, `llm.max_input_chars` prevents accidentally sending an unbounded prompt. Split the JSON canonical artifact into time chunks before raising that limit.

## Legacy scripts

`transcribe.py`, `diarize.py`, and `merge_speaker.py` remain unchanged as historical references. New work should use `wav2dia` so configuration and outputs stay consistent.
