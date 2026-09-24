from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version as distribution_version
import os
from pathlib import Path
from shutil import which
import subprocess
from typing import Any, Literal

from packaging.specifiers import SpecifierSet
from packaging.version import InvalidVersion, Version
from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Static, Tree
from textual.worker import get_current_worker

from ..cli import load_config
from ..formats import read_transcript, write_artifacts
from ..pipeline import (
    SUPPORTED_MEDIA_SUFFIXES,
    assign_speakers,
    diarize_audio,
    transcribe_audio,
)
from ..run_state import (
    has_existing_artifacts,
    load_run_state,
    make_run_state,
    matches_source,
    raw_transcript_path,
    state_path,
    update_run_state,
    write_run_state,
)
from .media import duration_seconds, format_duration
from .progress import (
    DiarizeSettingsScreen,
    PipelineProgressScreen,
    ProgressTestScreen,
    RunChoiceScreen,
    RunConfirmScreen,
)


@dataclass(frozen=True, slots=True)
class DependencyCheck:
    """Result of verifying an installable package needed by the local TUI workflow."""

    distribution: str
    module: str
    specifier: str
    installed_version: str | None
    error: str | None
    category: Literal["Python package", "System command"] = "Python package"

    @property
    def passed(self) -> bool:
        return self.error is None

    @property
    def requirement(self) -> str:
        return f"{self.distribution}{self.specifier}"


@dataclass(frozen=True, slots=True)
class HuggingFaceAccess:
    """Presence-only check for the diarization token; its value is never displayed."""

    environment_variable: str
    source: Literal["environment", ".env"] | None
    error: str | None

    @property
    def passed(self) -> bool:
        return self.error is None


@dataclass(frozen=True, slots=True)
class DiarizeSettings:
    """Speaker bounds selected in the TUI without modifying config.toml."""

    min_speakers: int = 3
    max_speakers: int = 6


@dataclass(frozen=True, slots=True)
class RunAssessment:
    """The safe choices available when a source has prior output or a state sidecar."""

    kind: Literal["new", "completed", "resumable", "restart_only"]
    state: dict[str, Any] | None
    detail: str

    @property
    def can_skip_to_diarize(self) -> bool:
        return self.kind == "resumable"


def _read_dotenv_value(path: Path, name: str) -> str | None:
    """Read one simple .env assignment without logging or exposing its secret value."""

    if not path.is_file():
        return None
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except OSError:
        return None
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").lstrip()
        key, separator, value = line.partition("=")
        if separator != "=" or key.strip() != name:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        else:
            value = value.split(" #", 1)[0].strip()
        return value or None
    return None


def check_huggingface_access(workspace: Path, environment_variable: str) -> HuggingFaceAccess:
    """Load a token from the environment first, then a local ignored .env file."""

    if os.environ.get(environment_variable):
        return HuggingFaceAccess(environment_variable, "environment", None)
    dotenv_value = _read_dotenv_value(workspace / ".env", environment_variable)
    if dotenv_value:
        os.environ[environment_variable] = dotenv_value
        return HuggingFaceAccess(environment_variable, ".env", None)
    return HuggingFaceAccess(environment_variable, None, "token is missing from the environment and .env")


DEPENDENCY_REQUIREMENTS = (
    ("faster-whisper", "faster_whisper", ">=1.1,<2"),
    ("pyannote.audio", "pyannote.audio", ">=4,<5"),
    ("textual", "textual", ">=0.58"),
    ("mutagen", "mutagen", ">=1.47"),
)

SYSTEM_DEPENDENCY_COMMANDS = (("ffmpeg", "ffmpeg", "-version"),)


def check_dependencies() -> tuple[DependencyCheck, ...]:
    """Quick startup checks that never import optional speech-model packages."""

    checks: list[DependencyCheck] = []
    for distribution, module, specifier in DEPENDENCY_REQUIREMENTS:
        try:
            installed_version = distribution_version(distribution)
        except PackageNotFoundError:
            checks.append(DependencyCheck(distribution, module, specifier, None, "package is not installed"))
            continue

        try:
            if Version(installed_version) not in SpecifierSet(specifier):
                checks.append(
                    DependencyCheck(
                        distribution,
                        module,
                        specifier,
                        installed_version,
                        f"installed version {installed_version} does not satisfy {specifier}",
                    )
                )
                continue
        except InvalidVersion:
            checks.append(
                DependencyCheck(distribution, module, specifier, installed_version, "installed version is invalid")
            )
            continue

        # Do not import `faster_whisper` or `pyannote.audio` here. pyannote imports
        # PyTorch, which makes a UI startup check expensive; the real pipeline will
        # report import/native-library errors when the user starts a job.
        checks.append(DependencyCheck(distribution, module, specifier, installed_version, None))

    for label, command, version_argument in SYSTEM_DEPENDENCY_COMMANDS:
        executable = which(command)
        if executable is None:
            checks.append(
                DependencyCheck(
                    label,
                    f"{command} {version_argument}",
                    "",
                    None,
                    "command is not available on PATH",
                    "System command",
                )
            )
            continue
        try:
            completed = subprocess.run(
                [executable, version_argument],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            checks.append(
                DependencyCheck(
                    label,
                    f"{command} {version_argument}",
                    "",
                    None,
                    f"command check failed: {type(exc).__name__}: {exc}",
                    "System command",
                )
            )
            continue
        if completed.returncode != 0:
            detail = completed.stderr.strip().splitlines()
            checks.append(
                DependencyCheck(
                    label,
                    f"{command} {version_argument}",
                    "",
                    None,
                    f"command returned exit {completed.returncode}: {detail[0] if detail else 'no diagnostic output'}",
                    "System command",
                )
            )
            continue
        version_line = completed.stdout.strip().splitlines()
        checks.append(
            DependencyCheck(
                label,
                f"{command} {version_argument}",
                "",
                version_line[0] if version_line else "available",
                None,
                "System command",
            )
        )
    return tuple(checks)


@dataclass(frozen=True, slots=True)
class TreeItem:
    """The operation associated with one node in a navigation tree."""

    kind: Literal[
        "directory",
        "audio",
        "setup",
        "dependency_group",
        "dependency",
        "diarize_settings",
        "huggingface_access",
        "result_group",
        "artifact",
        "run_status",
        "empty",
    ]
    path: Path | None = None
    dependency: DependencyCheck | None = None
    dependencies: tuple[DependencyCheck, ...] = ()
    huggingface_access: HuggingFaceAccess | None = None
    state: dict[str, Any] | None = None


class Wav2DiaTui(App[None]):
    """Workspace browser and background runner for the Wav2Dia pipeline."""

    TITLE = "Wav2Dia"
    CSS_PATH = "layout.tcss"
    BINDINGS = [
        ("q", "quit", "Quit"),
        ("ctrl+q", "quit", "Quit"),
        ("ctrl+c", "cancel_active_job", "Cancel job"),
        ("f5", "refresh_workspace", "Reload"),
        ("p", "open_progress_test", "Progress test"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.workspace = Path.cwd()
        self.files_directory = self.workspace / "files"
        self.output_directory = self.workspace / "output"
        self.active_worker: Any | None = None
        self.active_source: Path | None = None
        self.active_progress: PipelineProgressScreen | None = None
        self.diarize_settings = DiarizeSettings()

    def compose(self) -> ComposeResult:
        with Horizontal(id="upper-row"):
            with Vertical(id="files-pane", classes="pane"):
                yield Static("[b]Files / Setup[/b]", classes="pane-heading")
                yield Tree[TreeItem]("Navigation", id="workspace-tree")
            with Vertical(id="results-pane", classes="pane"):
                yield Static("[b]Transcription results[/b]", classes="pane-heading")
                yield Tree[TreeItem]("Results", id="results-tree")
            with VerticalScroll(id="reserved-pane", classes="pane"):
                yield Static("[b]Reserved[/b]\n\n[dim]Future workspace[/dim]", classes="reserved-copy")
        with VerticalScroll(id="status-pane", classes="pane"):
            yield Static(id="status-content")

    def on_mount(self) -> None:
        self.files_directory.mkdir(parents=True, exist_ok=True)
        self.refresh_workspace_trees()
        self.set_status(
            "Workspace ready",
            "Use ↑/↓ to move in Files / Setup. Press Enter on a folder to expand or collapse it, "
            "or on a WAV/MP3 file to start transcription and diarization.",
        )
        self._focus_files_tree()

    def action_refresh_workspace(self) -> None:
        self.refresh_workspace_trees()
        self._focus_files_tree()
        self.set_status("Workspace refreshed", f"Scanned {self.files_directory} and {self.output_directory}.")

    def action_open_progress_test(self) -> None:
        self.push_screen(ProgressTestScreen())

    def action_cancel_active_job(self) -> None:
        worker = self.active_worker
        source = self.active_source
        if worker is None or source is None or getattr(worker, "is_finished", False):
            self.set_status("No active job", "Ctrl+C only cancels an active transcription or diarization job.")
            return
        worker.cancel()
        state = load_run_state(self._output_prefix(source))
        if state is not None:
            update_run_state(state, cancellation_requested_at=True, message="Cancellation requested by user.")
            write_run_state(self._output_prefix(source), state)
        if self.active_progress is not None and self.active_progress.is_mounted:
            self.active_progress.show_cancelling()
        self.set_status(
            "Cancelling…",
            f"{source}\n\nThe current model call is blocking. Diarization will not start, or continue, after this stage returns.",
        )

    def run_pipeline_simulation(self, source: Path) -> None:
        """Preview the progress dialog without touching the audio or output directory."""
        self.push_screen(
            ProgressTestScreen(simulate_pipeline=True),
            lambda outcome: self._simulation_finished(source, outcome),
        )

    def _simulation_finished(self, source: Path, outcome: str) -> None:
        if outcome == "completed":
            self.set_status(
                "Simulation completed",
                f"{source}\n\nTranscribing and Diarizing each ran as a 15-second visual simulation. "
                "No model was loaded and no artifacts were written.",
            )
        else:
            self.set_status("Simulation cancelled", f"{source}\n\nNo model was loaded and no artifacts were written.")

    def _focus_files_tree(self) -> None:
        workspace_tree = self.query_one("#workspace-tree", Tree)
        workspace_tree.focus()
        # The Tree has no cursor until one is explicitly selected. Files is the
        # first visible node, so it is the intended initial Enter target.
        workspace_tree.cursor_line = 0

    def refresh_workspace_trees(self) -> None:
        self._populate_workspace_tree()
        self._populate_results_tree()

    def _prepare_tree(self, tree_id: str) -> Tree[TreeItem]:
        tree = self.query_one(tree_id, Tree)
        tree.clear()
        tree.show_root = False
        tree.auto_expand = False
        tree.root.expand()
        return tree

    def _populate_workspace_tree(self) -> None:
        tree = self._prepare_tree("#workspace-tree")
        files_node = tree.root.add("Files", TreeItem("directory", self.files_directory), expand=True)
        self._add_media_nodes(files_node, self.files_directory)
        setup_node = tree.root.add("Setup", TreeItem("setup"), expand=True)
        setup_node.add_leaf(self._diarize_settings_label(), TreeItem("diarize_settings"))
        checks = check_dependencies()
        dependency_node = setup_node.add(
            self._dependency_group_label(checks),
            TreeItem("dependency_group", dependencies=checks),
            expand=False,
        )
        for check in checks:
            dependency_node.add_leaf(self._dependency_label(check), TreeItem("dependency", dependency=check))
        access = self._huggingface_access()
        setup_node.add_leaf(
            self._huggingface_access_label(access),
            TreeItem("huggingface_access", huggingface_access=access),
        )

    def _diarize_settings_label(self) -> Text:
        settings = self.diarize_settings
        return Text(
            f"Diarize  min={settings.min_speakers}  max={settings.max_speakers}",
            style="cyan",
        )

    @staticmethod
    def _dependency_group_label(checks: tuple[DependencyCheck, ...]) -> Text:
        failed_count = sum(not check.passed for check in checks)
        if failed_count:
            return Text(f"Dependency — {failed_count} check failed", style="bold red")
        return Text("Dependency — all checks passed", style="green")

    @staticmethod
    def _dependency_label(check: DependencyCheck) -> Text:
        if check.passed:
            state = "available" if check.category == "System command" else "installed"
            version = f" {check.installed_version}" if check.category == "Python package" else ""
            return Text(f"{check.requirement}  ✓ {state}{version}", style="green")
        return Text(f"{check.requirement}  × check failed", style="red")

    def _huggingface_access(self) -> HuggingFaceAccess:
        try:
            config = load_config(None)
            environment_variable = str(config["diarization"]["token_env"])
        except (KeyError, OSError, ValueError) as exc:
            return HuggingFaceAccess("HF_TOKEN", None, f"configuration error: {exc}")
        return check_huggingface_access(self.workspace, environment_variable)

    @staticmethod
    def _huggingface_access_label(access: HuggingFaceAccess) -> Text:
        if access.passed:
            return Text(f"Hugging Face access  ✓ available via {access.source}", style="green")
        return Text("Hugging Face access  × token missing", style="bold red")

    def _add_media_nodes(self, parent: object, directory: Path) -> None:
        """Add directories and supported media, keeping non-media files out of the TUI."""
        try:
            entries = sorted(directory.iterdir(), key=lambda path: (not path.is_dir(), path.name.casefold()))
        except OSError as exc:
            parent.add_leaf(f"[unreadable: {exc.strerror or exc}]", TreeItem("empty"))
            return
        for entry in entries:
            if entry.is_dir():
                child = parent.add(entry.name, TreeItem("directory", entry), expand=False)
                self._add_media_nodes(child, entry)
            elif entry.is_file() and entry.suffix.lower() in SUPPORTED_MEDIA_SUFFIXES:
                parent.add_leaf(entry.name, TreeItem("audio", entry))

    def _populate_results_tree(self) -> None:
        tree = self._prepare_tree("#results-tree")
        media_files = self._media_files()
        if not media_files:
            tree.root.add_leaf("No supported media in files/", TreeItem("empty"))
            return
        for source in media_files:
            relative = source.relative_to(self.files_directory)
            prefix = self._output_prefix(source)
            state = load_run_state(prefix)
            group = tree.root.add(
                self._result_group_label(relative.as_posix(), source, state),
                TreeItem("result_group", path=state_path(prefix), state=state),
                expand=False,
            )
            final_artifacts = (
                ("srt", prefix.parent / f"{prefix.name}.srt"),
                ("md", prefix.parent / f"{prefix.name}.md"),
                ("txt  (speaker turns)", prefix.parent / f"{prefix.name}.speakers.txt"),
            )
            incomplete = state is not None and str(state.get("status")) != "completed"
            if state is not None:
                status = str(state.get("status", "unknown"))
                stage = str(state.get("last_completed_stage", "none"))
                group.add_leaf(
                    self._run_status_label(status, stage, str(state.get("message", ""))),
                    TreeItem("run_status", path=state_path(prefix), state=state),
                )
            found = False
            for label, artifact in final_artifacts:
                if artifact.is_file():
                    group.add_leaf(
                        self._result_artifact_label(label, incomplete),
                        TreeItem("artifact", artifact),
                    )
                    found = True
            if incomplete and str(state.get("last_completed_stage")) == "transcribed":
                raw_artifacts = (
                    ("srt  (raw transcription)", prefix.parent / f"{prefix.name}.raw.srt"),
                    ("md  (raw transcription)", prefix.parent / f"{prefix.name}.raw.md"),
                )
                for label, artifact in raw_artifacts:
                    if artifact.is_file():
                        group.add_leaf(Text(label, style="orange1"), TreeItem("artifact", artifact))
                        found = True
                if not (prefix.parent / f"{prefix.name}.speakers.txt").is_file():
                    group.add_leaf(
                        Text("txt  (speaker turns — incomplete)", style="orange1"),
                        TreeItem("run_status", path=state_path(prefix), state=state),
                    )
                    found = True
            if not found:
                group.add_leaf("No completed artifacts", TreeItem("empty"))

    @staticmethod
    def _result_artifact_label(label: str, incomplete: bool) -> str | Text:
        return Text(f"{label}  (previous output)", style="orange1") if incomplete else label

    @staticmethod
    def _run_status_label(status: str, stage: str, message: str) -> Text:
        if status in {"cancelled", "abandoned", "running"}:
            return Text(f"⚠ {status} after {stage}", style="orange1")
        if status in {"failed", "invalid"}:
            return Text(f"× {status} after {stage}", style="red")
        return Text(f"✓ completed ({stage})", style="green")

    @staticmethod
    def _result_group_label(name: str, source: Path, state: dict[str, Any] | None) -> str | Text:
        if state is None:
            return name
        status = str(state.get("status", "invalid"))
        if status in {"cancelled", "abandoned", "running"}:
            return Text(f"{name}  — incomplete", style="orange1")
        if status in {"failed", "invalid"} or not matches_source(state, source):
            return Text(f"{name}  — needs attention", style="bold red")
        return name

    def _media_files(self) -> list[Path]:
        if not self.files_directory.is_dir():
            return []
        return sorted(
            (
                path
                for path in self.files_directory.rglob("*")
                if path.is_file() and path.suffix.lower() in SUPPORTED_MEDIA_SUFFIXES
            ),
            key=lambda path: path.relative_to(self.files_directory).as_posix().casefold(),
        )

    def _output_prefix(self, source: Path) -> Path:
        relative = source.relative_to(self.files_directory)
        artifact_directory = self.output_directory / relative.parent / relative.stem
        return artifact_directory / relative.stem

    def _assess_run(self, source: Path) -> RunAssessment:
        prefix = self._output_prefix(source)
        state = load_run_state(prefix)
        if state is None:
            if has_existing_artifacts(prefix):
                return RunAssessment(
                    "restart_only",
                    None,
                    "Existing output has no Wav2Dia run-status JSON, so it cannot be resumed safely.",
                )
            return RunAssessment("new", None, "No previous output found.")

        status = str(state.get("status", "invalid"))
        if status == "running":
            update_run_state(
                state,
                status="abandoned",
                message="Previous run did not close cleanly and is treated as abandoned.",
            )
            write_run_state(prefix, state)
            status = "abandoned"
        if status == "invalid":
            return RunAssessment("restart_only", state, str(state.get("message", "Run status is invalid.")))
        if not matches_source(state, source):
            return RunAssessment(
                "restart_only",
                state,
                "The source path, size, or modification time differs from the previous run; resume is unsafe.",
            )
        if status == "completed":
            return RunAssessment("completed", state, "A complete result already exists for this unchanged source.")
        if (
            str(state.get("last_completed_stage")) == "transcribed"
            and raw_transcript_path(prefix).is_file()
        ):
            return RunAssessment(
                "resumable",
                state,
                "A complete raw transcript is available. You may skip transcription and retry diarization.",
            )
        return RunAssessment("restart_only", state, "The prior run has no complete raw transcript to resume from.")

    def _pipeline_prerequisites_ready(self) -> bool:
        failed = [check.requirement for check in check_dependencies() if not check.passed]
        if failed:
            self.set_status("Cannot start pipeline", f"Dependency checks failed: {', '.join(failed)}")
            return False
        access = self._huggingface_access()
        if not access.passed:
            self.set_status(
                "Cannot start diarization",
                f"Hugging Face access is required. Add {access.environment_variable}=… to .env or the environment.",
            )
            return False
        return True

    def request_pipeline(self, source: Path) -> None:
        if self.active_worker is not None and not getattr(self.active_worker, "is_finished", False):
            self.set_status("Pipeline already running", f"{self.active_source}\n\nPress Ctrl+C to request cancellation.")
            return
        if not self._pipeline_prerequisites_ready():
            return
        assessment = self._assess_run(source)
        if assessment.kind == "new":
            self._start_pipeline(source, "full")
            return
        title = "Existing output detected"
        self.push_screen(
            RunChoiceScreen(title, assessment.detail, can_skip_to_diarize=assessment.can_skip_to_diarize),
            lambda choice: self._handle_run_choice(source, assessment, choice),
        )

    def confirm_pipeline(self, source: Path) -> None:
        """Show input metadata and selected speaker bounds before work can begin."""

        if self.active_worker is not None and not getattr(self.active_worker, "is_finished", False):
            self.set_status("Pipeline already running", f"{self.active_source}\n\nPress Ctrl+C to request cancellation.")
            return
        settings = self.diarize_settings
        self.push_screen(
            RunConfirmScreen(
                source.name,
                format_duration(duration_seconds(source)),
                settings.min_speakers,
                settings.max_speakers,
            ),
            lambda confirmed: self._handle_pipeline_confirmation(source, confirmed),
        )

    def _handle_pipeline_confirmation(self, source: Path, confirmed: bool) -> None:
        if confirmed:
            self.request_pipeline(source)
        else:
            self.set_status("Pipeline not started", f"{source}\n\nCancelled before transcription began.")

    def _handle_run_choice(self, source: Path, assessment: RunAssessment, choice: str) -> None:
        if choice == "overwrite":
            self._start_pipeline(source, "full")
        elif choice == "skip" and assessment.can_skip_to_diarize:
            self._start_pipeline(source, "diarize_only")
        else:
            self.set_status("Pipeline not started", f"{source}\n\nExisting output was left unchanged.")

    def _start_pipeline(self, source: Path, mode: Literal["full", "diarize_only"]) -> None:
        self.active_source = source
        self.active_progress = PipelineProgressScreen(source.name)
        self.push_screen(self.active_progress)
        self.active_worker = self.run_full_pipeline(source, mode, self.diarize_settings)

    def on_tree_node_selected(self, event: Tree.NodeSelected[TreeItem]) -> None:
        item = event.node.data
        if item is None:
            return
        if item.kind == "directory":
            event.node.toggle()
            action = "expanded" if event.node.is_expanded else "collapsed"
            self.set_status(f"Folder {action}", str(item.path))
        elif item.kind == "setup":
            self.show_setup()
        elif item.kind == "dependency_group":
            event.node.toggle()
            failed_count = sum(not check.passed for check in item.dependencies)
            self.set_status(
                "Dependency checks",
                f"{failed_count} failed of {len(item.dependencies)} checked. "
                "Press Enter on an item for its installed version or failure detail.",
            )
        elif item.kind == "dependency" and item.dependency is not None:
            self.show_dependency_status(item.dependency)
        elif item.kind == "diarize_settings":
            self.show_diarize_settings()
        elif item.kind == "huggingface_access" and item.huggingface_access is not None:
            self.show_huggingface_access(item.huggingface_access)
        elif item.kind == "result_group":
            event.node.toggle()
            action = "expanded" if event.node.is_expanded else "collapsed"
            self.set_status(f"Result group {action}", str(item.path))
        elif item.kind == "audio" and item.path is not None:
            if event.control.id == "workspace-tree":
                self.confirm_pipeline(item.path)
            else:
                self.set_status("Result source selected", str(item.path))
        elif item.kind == "artifact" and item.path is not None:
            self.show_artifact_status(item.path)
        elif item.kind == "run_status" and item.state is not None:
            self.show_run_status(item.state, item.path)

    def on_tree_node_highlighted(self, event: Tree.NodeHighlighted[TreeItem]) -> None:
        """Show selected input-audio metadata while the cursor moves in Files."""
        if event.control.id == "results-tree":
            item = event.node.data
            if item is not None and item.kind == "artifact" and item.path is not None:
                if item.path.suffix.lower() in {".srt", ".md"}:
                    self.show_artifact_preview(item.path)
            return
        if event.control.id != "workspace-tree":
            return
        item = event.node.data
        if item is None or item.kind != "audio" or item.path is None:
            return
        duration = format_duration(duration_seconds(item.path))
        try:
            file_size = item.path.stat().st_size
        except OSError:
            file_size = 0
        self.set_status(
            "Audio selected",
            f"{item.path}\nDuration: {duration}\nSize: {file_size:,} bytes\n\nPress Enter to review job settings before starting the pipeline.",
        )

    def show_setup(self) -> None:
        try:
            config = load_config(None)
        except (OSError, ValueError) as exc:
            self.set_status("Setup configuration error", str(exc))
            return
        asr = config["asr"]
        diarization = config["diarization"]
        llm = config["llm"]
        self.set_status(
            "Setup",
            "\n".join(
                (
                    f"ASR: model={asr['model']}, device={asr['device']}, compute_type={asr['compute_type']}",
                    f"Diarization: model={diarization['model']}, token env={diarization['token_env']}",
                    f"Diarize bounds (this TUI session): min={self.diarize_settings.min_speakers}, max={self.diarize_settings.max_speakers}",
                    f"LLM default: {llm['backend']}",
                    "Select Diarize and press Enter to change the speaker bounds for this session.",
                )
            ),
        )

    def show_diarize_settings(self) -> None:
        settings = self.diarize_settings
        self.push_screen(
            DiarizeSettingsScreen(settings.min_speakers, settings.max_speakers),
            self._apply_diarize_settings,
        )

    def _apply_diarize_settings(self, result: tuple[int, int] | None) -> None:
        if result is None:
            self.set_status("Diarize settings unchanged", "No speaker-bound setting was changed.")
            return
        self.diarize_settings = DiarizeSettings(*result)
        self.refresh_workspace_trees()
        self.set_status(
            "Diarize settings updated",
            "\n".join(
                (
                    f"min_speakers={self.diarize_settings.min_speakers}",
                    f"max_speakers={self.diarize_settings.max_speakers}",
                    "The values apply to jobs started in this TUI session and do not modify config.toml.",
                )
            ),
        )

    def show_dependency_status(self, check: DependencyCheck) -> None:
        if check.passed:
            detail = "\n".join(
                (
                    f"Requirement: {check.requirement}",
                    f"Category: {check.category}",
                    f"Package module: {check.module}",
                    f"Detected version: {check.installed_version}",
                    "Startup check: package is installed and its version matches; module import is deferred until a job runs.",
                )
            )
            self.set_status("Dependency ready", detail)
            return
        detail = "\n".join(
            (
                f"Requirement: {check.requirement}",
                f"Category: {check.category}",
                f"Check target: {check.module}",
                f"Detected version: {check.installed_version or 'not available'}",
                f"Dependency check failed: {check.error}",
            )
        )
        self.set_status("Dependency check failed", detail)

    def show_huggingface_access(self, access: HuggingFaceAccess) -> None:
        if access.passed:
            self.set_status(
                "Hugging Face access ready",
                f"{access.environment_variable} is available via {access.source}. The token value is intentionally hidden.",
            )
            return
        self.set_status(
            "Hugging Face access missing",
            f"Set {access.environment_variable} in the shell environment or in {self.workspace / '.env'}. "
            "The token value is never shown in the TUI.",
        )

    def show_run_status(self, state: dict[str, Any], path: Path | None) -> None:
        detail = "\n".join(
            (
                f"Status: {state.get('status', 'unknown')}",
                f"Last completed stage: {state.get('last_completed_stage', 'none')}",
                f"Updated: {state.get('updated_at', 'unknown')}",
                f"Message: {state.get('message', '') or 'none'}",
                f"State file: {path or 'unknown'}",
            )
        )
        self.set_status("Run status", detail)

    def show_artifact_status(self, artifact: Path) -> None:
        if artifact.suffix.lower() in {".srt", ".md"}:
            self.show_artifact_preview(artifact)
            return
        try:
            size = artifact.stat().st_size
        except OSError as exc:
            self.set_status("Artifact unavailable", str(exc))
            return
        self.set_status("Artifact selected", f"{artifact}\n{size:,} bytes\n\nPreview rendering is the next UI step.")

    def show_artifact_preview(self, artifact: Path) -> None:
        try:
            content = artifact.read_text(encoding="utf-8-sig")
            size = artifact.stat().st_size
        except (OSError, UnicodeError) as exc:
            self.set_status("Artifact unavailable", f"{artifact}\n\n{type(exc).__name__}: {exc}")
            return
        detail = Text()
        detail.append(f"{artifact}\n{size:,} bytes\n")
        detail.append("─" * 76 + "\n\n")
        detail.append(content)
        self.set_status(f"Preview: {artifact.suffix.lower()[1:].upper()}", detail)

    @work(thread=True, exclusive=True, group="pipeline", exit_on_error=False)
    def run_full_pipeline(
        self,
        source: Path,
        mode: Literal["full", "diarize_only"],
        diarize_settings: DiarizeSettings,
    ) -> None:
        """Run a durable, cancellable pipeline without blocking Textual's event loop."""

        worker = get_current_worker()
        target = self._output_prefix(source)
        raw_target = target.with_name(target.name + ".raw")
        state: dict[str, Any] | None = None
        try:
            initial_stage: Literal["none", "transcribed"] = "transcribed" if mode == "diarize_only" else "none"
            state = make_run_state(
                source,
                status="running",
                last_completed_stage=initial_stage,
                mode=mode,
                message="Pipeline started.",
                raw_artifacts={"json": raw_transcript_path(target)} if mode == "diarize_only" else None,
            )
            update_run_state(
                state,
                diarize_settings={
                    "min_speakers": diarize_settings.min_speakers,
                    "max_speakers": diarize_settings.max_speakers,
                },
            )
            write_run_state(target, state)
            config = load_config(None)
            asr = config["asr"]
            diarization = config["diarization"]

            if mode == "full":
                self._set_pipeline_stage_from_worker("Transcribing", f"{source.name}\nModel: {asr['model']}")
                transcript = transcribe_audio(
                    source,
                    model_name=asr["model"],
                    device=asr["device"],
                    compute_type=asr["compute_type"],
                    language=asr["language"],
                    hotwords=asr["hotwords"],
                    word_timestamps=bool(asr["word_timestamps"]),
                )
                if worker.is_cancelled:
                    self._finish_cancelled_from_worker(target, state, source, "Cancelled during transcription.")
                    return

                self._set_pipeline_stage_from_worker("Saving raw transcript", str(raw_target.parent))
                raw_paths = write_artifacts(raw_target, transcript, include_speakers=False)
                update_run_state(
                    state,
                    last_completed_stage="transcribed",
                    raw_artifacts={name: str(path) for name, path in raw_paths.items()},
                    message="Transcription completed; ready for diarization.",
                )
                write_run_state(target, state)
                if worker.is_cancelled:
                    self._finish_cancelled_from_worker(target, state, source, "Cancelled after transcription completed.")
                    return
            else:
                raw_json = raw_transcript_path(target)
                if not raw_json.is_file():
                    raise FileNotFoundError(f"Cannot skip transcription: raw transcript is missing: {raw_json}")
                self._set_pipeline_stage_from_worker("Loading raw transcript", str(raw_json))
                transcript = read_transcript(raw_json)
                if worker.is_cancelled:
                    self._finish_cancelled_from_worker(target, state, source, "Cancelled before diarization started.")
                    return

            self._set_pipeline_stage_from_worker("Diarizing", f"{source.name}\nPipeline: {diarization['model']}")
            turns = diarize_audio(
                source,
                model_name=diarization["model"],
                token_env=diarization["token_env"],
                min_speakers=diarize_settings.min_speakers,
                max_speakers=diarize_settings.max_speakers,
            )
            if worker.is_cancelled:
                self._finish_cancelled_from_worker(target, state, source, "Cancelled during diarization.")
                return

            transcript.diarization = turns
            transcript.segments = assign_speakers(transcript.segments, turns)
            self._set_pipeline_stage_from_worker("Saving speaker-aligned results", str(target.parent))
            paths = write_artifacts(target, transcript, include_speakers=True)
            update_run_state(
                state,
                status="completed",
                last_completed_stage="diarized",
                message="Transcription and diarization completed.",
                final_artifacts={name: str(path) for name, path in paths.items()},
            )
            write_run_state(target, state)
            self._finish_pipeline_from_worker(
                "Completed",
                "\n".join(f"{label}: {path}" for label, path in paths.items()),
            )
        except Exception as exc:  # Optional dependency, model, network, and decoder errors are reported in the TUI.
            if state is not None and worker.is_cancelled:
                self._finish_cancelled_from_worker(target, state, source, "Cancelled while the current operation was stopping.")
                return
            if state is not None:
                update_run_state(
                    state,
                    status="failed",
                    message=f"{type(exc).__name__}: {exc}",
                )
                write_run_state(target, state)
            self._finish_pipeline_from_worker("Pipeline failed", f"{type(exc).__name__}: {exc}")

    def _finish_cancelled_from_worker(
        self,
        target: Path,
        state: dict[str, Any],
        source: Path,
        message: str,
    ) -> None:
        update_run_state(state, status="cancelled", message=message)
        write_run_state(target, state)
        stage = state.get("last_completed_stage", "none")
        detail = f"{source}\n\n{message}\nLast complete stage: {stage}."
        if stage == "transcribed":
            detail += "\nRaw transcript was retained; choose Skip to diarize to resume."
        self._finish_pipeline_from_worker("Cancelled by user", detail)

    def _set_pipeline_stage_from_worker(self, headline: str, detail: str) -> None:
        self.call_from_thread(self._set_pipeline_stage, headline, detail)

    def _set_pipeline_stage(self, headline: str, detail: str) -> None:
        if self.active_progress is not None and self.active_progress.is_mounted:
            self.active_progress.set_phase(headline)
        self.set_status(headline, detail)

    def _finish_pipeline_from_worker(self, headline: str, detail: str) -> None:
        self.call_from_thread(self._finish_pipeline, headline, detail)

    def _finish_pipeline(self, headline: str, detail: str) -> None:
        if self.active_progress is not None and self.active_progress.is_mounted:
            self.active_progress.dismiss()
        self.active_worker = None
        self.active_source = None
        self.active_progress = None
        self.refresh_workspace_trees()
        self.set_status(headline, detail)

    def set_status(self, headline: str, detail: str | Text) -> None:
        content = Text(
            "Status / preview\n\n"
            f"Workspace: {self.workspace}\n"
            f"State: {headline}\n"
            "────────────────────────────────────────────────────────────────────────────\n\n"
        )
        if isinstance(detail, Text):
            content.append_text(detail)
        else:
            content.append(detail)
        self.query_one("#status-content", Static).update(content)


def main() -> int:
    Wav2DiaTui().run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
