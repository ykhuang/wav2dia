from __future__ import annotations

from time import monotonic

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, ProgressBar, Static


class ProgressTestScreen(ModalScreen[str]):
    """A visual test of the repeating, stage-aware pipeline progress display."""

    STEP_COUNT = 6
    SPINNER_FRAMES = ("|", "/", "-", "\\")
    BINDINGS = [
        ("escape", "cancel", "Close"),
        ("d", "switch_to_diarizing", "Diarizing"),
        ("t", "switch_to_transcribing", "Transcribing"),
    ]

    def __init__(self, *, simulate_pipeline: bool = False, stage_seconds: int = 15) -> None:
        super().__init__()
        self.simulate_pipeline = simulate_pipeline
        self.stage_seconds = stage_seconds
        self.phase = "Transcribing"
        self.spinner_index = 0
        self.started_at: float | None = None

    def compose(self) -> ComposeResult:
        with Vertical(id="progress-dialog"):
            yield Static(self.phase, id="progress-phase")
            with Horizontal(id="progress-row"):
                yield Static(self.SPINNER_FRAMES[0], id="ascii-spinner")
                yield ProgressBar(
                    total=self.STEP_COUNT,
                    show_percentage=False,
                    show_eta=False,
                    id="pipeline-progress",
                )
            yield Static("Elapsed time: 00:00:00", id="elapsed-time")
            help_text = (
                "Simulation · Transcribing 15s → Diarizing 15s → finish\n[Esc] Cancel"
                if self.simulate_pipeline
                else "Test mode · advances one segment every 10 seconds · restarts after filling\n"
                "[D] Diarizing  [T] Transcribing  [Esc] Close"
            )
            yield Static(help_text, id="progress-help")

    def on_mount(self) -> None:
        self.started_at = monotonic()
        self._update_elapsed_time()
        self.set_interval(0.15, self.advance_spinner)
        self.set_interval(10, self.advance_progress)
        if self.simulate_pipeline:
            self.set_timer(self.stage_seconds, self._begin_diarizing)
            self.set_timer(self.stage_seconds * 2, self._finish_simulation)

    def advance_spinner(self) -> None:
        self.spinner_index = (self.spinner_index + 1) % len(self.SPINNER_FRAMES)
        self.query_one("#ascii-spinner", Static).update(self.SPINNER_FRAMES[self.spinner_index])
        self._update_elapsed_time()

    def advance_progress(self) -> None:
        progress = self.query_one("#pipeline-progress", ProgressBar)
        if progress.progress >= self.STEP_COUNT:
            progress.update(progress=0)
        else:
            progress.advance(1)

    def action_switch_to_diarizing(self) -> None:
        self._set_phase("Diarizing")

    def action_switch_to_transcribing(self) -> None:
        self._set_phase("Transcribing")

    def action_cancel(self) -> None:
        self.dismiss("cancelled")

    def _begin_diarizing(self) -> None:
        self._set_phase("Diarizing")

    def _finish_simulation(self) -> None:
        self.dismiss("completed")

    def _set_phase(self, phase: str) -> None:
        self.phase = phase
        self.query_one("#progress-phase", Static).update(phase)
        self.query_one("#pipeline-progress", ProgressBar).update(progress=0)

    def _update_elapsed_time(self) -> None:
        if self.started_at is None:
            return
        elapsed_seconds = int(monotonic() - self.started_at)
        hours, remainder = divmod(elapsed_seconds, 3_600)
        minutes, seconds = divmod(remainder, 60)
        self.query_one("#elapsed-time", Static).update(
            f"Elapsed time: {hours:02}:{minutes:02}:{seconds:02}"
        )


class PipelineProgressScreen(ModalScreen[None]):
    """Live stage display for a real pipeline run; cancellation is cooperative."""

    SPINNER_FRAMES = ("|", "/", "-", "\\")
    BINDINGS = [("ctrl+c", "request_cancel", "Cancel job")]

    def __init__(self, source_name: str) -> None:
        super().__init__()
        self.source_name = source_name
        self.phase = "Preparing"
        self.spinner_index = 0
        self.started_at: float | None = None
        self.cancelling = False

    def compose(self) -> ComposeResult:
        with Vertical(id="progress-dialog"):
            yield Static(self.phase, id="progress-phase")
            with Horizontal(id="progress-row"):
                yield Static(self.SPINNER_FRAMES[0], id="ascii-spinner")
                yield ProgressBar(total=6, show_percentage=False, show_eta=False, id="pipeline-progress")
            yield Static("Elapsed time: 00:00:00", id="elapsed-time")
            yield Static(f"{self.source_name}\n[Ctrl+C] Cancel job", id="progress-help")

    def on_mount(self) -> None:
        self.started_at = monotonic()
        self.set_interval(0.15, self.advance_spinner)
        self.set_interval(10, self.advance_progress)

    def action_request_cancel(self) -> None:
        self.app.action_cancel_active_job()

    def advance_spinner(self) -> None:
        self.spinner_index = (self.spinner_index + 1) % len(self.SPINNER_FRAMES)
        self.query_one("#ascii-spinner", Static).update(self.SPINNER_FRAMES[self.spinner_index])
        self._update_elapsed_time()

    def advance_progress(self) -> None:
        progress = self.query_one("#pipeline-progress", ProgressBar)
        if progress.progress >= 6:
            progress.update(progress=0)
        else:
            progress.advance(1)

    def set_phase(self, phase: str) -> None:
        self.phase = phase
        self.query_one("#progress-phase", Static).update(phase)
        self.query_one("#pipeline-progress", ProgressBar).update(progress=0)

    def show_cancelling(self) -> None:
        self.cancelling = True
        self.query_one("#progress-phase", Static).update("Cancelling…")
        self.query_one("#progress-help", Static).update(
            f"{self.source_name}\nCancellation will take effect after the current model stage returns."
        )

    def _update_elapsed_time(self) -> None:
        if self.started_at is None:
            return
        elapsed_seconds = int(monotonic() - self.started_at)
        hours, remainder = divmod(elapsed_seconds, 3_600)
        minutes, seconds = divmod(remainder, 60)
        self.query_one("#elapsed-time", Static).update(
            f"Elapsed time: {hours:02}:{minutes:02}:{seconds:02}"
        )


class RunChoiceScreen(ModalScreen[str]):
    """Choose whether an existing source output should be replaced or resumed."""

    BINDINGS = [
        ("o", "overwrite", "Overwrite"),
        ("s", "skip_to_diarize", "Skip to diarize"),
        ("escape", "cancel", "Cancel"),
        ("c", "cancel", "Cancel"),
    ]

    def __init__(self, title: str, detail: str, *, can_skip_to_diarize: bool) -> None:
        super().__init__()
        self.title = title
        self.detail = detail
        self.can_skip_to_diarize = can_skip_to_diarize

    def compose(self) -> ComposeResult:
        with Vertical(id="run-choice-dialog"):
            yield Static(self.title, id="choice-title")
            yield Static(self.detail, id="choice-detail")
            with Horizontal(id="choice-actions"):
                yield Button("Overwrite", id="overwrite", classes="choice-button", variant="warning")
                if self.can_skip_to_diarize:
                    yield Button("Skip to diarize", id="skip", classes="choice-button", variant="primary")
                yield Button("Cancel", id="cancel", classes="choice-button")
            keys = "[O] Overwrite"
            if self.can_skip_to_diarize:
                keys += "  [S] Skip to diarize"
            yield Static(f"{keys}  [Esc] Cancel", id="choice-help")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(str(event.button.id))

    def action_overwrite(self) -> None:
        self.dismiss("overwrite")

    def action_skip_to_diarize(self) -> None:
        if self.can_skip_to_diarize:
            self.dismiss("skip")

    def action_cancel(self) -> None:
        self.dismiss("cancel")


class DiarizeSettingsScreen(ModalScreen[tuple[int, int] | None]):
    """Edit the speaker bounds used by jobs started from this TUI session."""

    BINDINGS = [
        ("enter", "apply", "Apply"),
        ("escape", "cancel", "Cancel"),
    ]

    def __init__(self, min_speakers: int, max_speakers: int) -> None:
        super().__init__()
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers

    def compose(self) -> ComposeResult:
        with Vertical(id="diarize-settings-dialog"):
            yield Static("Diarize settings", id="diarize-settings-title")
            yield Static(
                "Set the speaker range for jobs started in this TUI session.",
                id="diarize-settings-detail",
            )
            with Horizontal(classes="diarize-setting-row"):
                yield Static("Min speakers", classes="diarize-setting-label")
                yield Input(str(self.min_speakers), id="min-speakers", classes="diarize-setting-input")
            with Horizontal(classes="diarize-setting-row"):
                yield Static("Max speakers", classes="diarize-setting-label")
                yield Input(str(self.max_speakers), id="max-speakers", classes="diarize-setting-input")
            yield Static("", id="diarize-settings-error")
            with Horizontal(id="diarize-settings-actions"):
                yield Button("OK", id="apply", classes="choice-button", variant="primary")
                yield Button("Cancel", id="cancel", classes="choice-button")
            yield Static("[Enter] OK  [Esc] Cancel", id="diarize-settings-help")

    def on_mount(self) -> None:
        self.query_one("#min-speakers", Input).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "apply":
            self.action_apply()
        elif event.button.id == "cancel":
            self.action_cancel()

    def on_input_submitted(self, _event: Input.Submitted) -> None:
        """Treat Enter from either numeric field exactly like clicking OK."""

        self.action_apply()

    def action_apply(self) -> None:
        try:
            min_speakers = int(self.query_one("#min-speakers", Input).value.strip())
            max_speakers = int(self.query_one("#max-speakers", Input).value.strip())
        except ValueError:
            self._show_error("Min speakers and max speakers must both be whole numbers.")
            return
        if min_speakers < 1:
            self._show_error("Min speakers must be at least 1.")
            return
        if max_speakers < min_speakers:
            self._show_error("Max speakers must be greater than or equal to min speakers.")
            return
        self.dismiss((min_speakers, max_speakers))

    def action_cancel(self) -> None:
        self.dismiss(None)

    def _show_error(self, message: str) -> None:
        self.query_one("#diarize-settings-error", Static).update(message)


class RunConfirmScreen(ModalScreen[bool]):
    """Require an explicit confirmation before starting a media pipeline job."""

    BINDINGS = [
        ("enter", "confirm", "OK"),
        ("escape", "cancel", "Cancel"),
        ("c", "cancel", "Cancel"),
    ]

    def __init__(self, source_name: str, duration: str, min_speakers: int, max_speakers: int) -> None:
        super().__init__()
        self.source_name = source_name
        self.duration = duration
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers

    def compose(self) -> ComposeResult:
        with Vertical(id="run-confirm-dialog"):
            yield Static("Start transcription and diarization?", id="confirm-title")
            yield Static(
                "\n".join(
                    (
                        f"Audio: {self.source_name}",
                        f"Duration: {self.duration}",
                        f"Diarize: min_speakers={self.min_speakers}, max_speakers={self.max_speakers}",
                    )
                ),
                id="confirm-detail",
            )
            with Horizontal(id="confirm-actions"):
                yield Button("OK", id="confirm", classes="choice-button", variant="primary")
                yield Button("Cancel", id="cancel", classes="choice-button")
            yield Static("[Enter] OK  [Esc] Cancel", id="confirm-help")

    def on_mount(self) -> None:
        self.query_one("#confirm", Button).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "confirm":
            self.action_confirm()
        else:
            self.action_cancel()

    def action_confirm(self) -> None:
        self.dismiss(True)

    def action_cancel(self) -> None:
        self.dismiss(False)
