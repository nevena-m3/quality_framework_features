from __future__ import annotations

import html
import os
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from .segmentation import (
    MANUAL_SEGMENTATION_COLUMNS,
    SEGMENTATION_ADJUDICATION_COLUMNS,
)


def parse_manual_intervals_text(text: str) -> list[tuple[float, float]]:
    """Parse one ``start_sec,end_sec`` speech interval per nonblank line."""
    intervals: list[tuple[float, float]] = []
    for line_number, raw_line in enumerate(str(text).splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        pieces = [piece.strip() for piece in line.replace("\t", ",").split(",")]
        if len(pieces) != 2:
            raise ValueError(
                f"Line {line_number} must contain exactly start_sec,end_sec: {raw_line!r}"
            )
        try:
            start, end = map(float, pieces)
        except ValueError as exc:
            raise ValueError(
                f"Line {line_number} contains a non-numeric boundary: {raw_line!r}"
            ) from exc
        if not np.isfinite(start) or not np.isfinite(end):
            raise ValueError(f"Line {line_number} contains a non-finite boundary.")
        intervals.append((start, end))
    return intervals


def validate_manual_interval_list(
    intervals: list[tuple[float, float]],
    *,
    duration_sec: float,
) -> list[tuple[float, float]]:
    """Validate and sort manual speech intervals against recording duration."""
    if not intervals:
        raise ValueError("KEEP/MANUAL requires at least one speech interval.")
    ordered = sorted((float(start), float(end)) for start, end in intervals)
    for index, (start, end) in enumerate(ordered):
        if start < 0 or end <= start:
            raise ValueError(f"Invalid manual interval {index}: ({start}, {end})")
        if end > float(duration_sec) + 1e-6:
            raise ValueError(
                f"Manual interval {index} ends at {end:.6f}s, beyond "
                f"the {duration_sec:.6f}s recording."
            )
        if index and start < ordered[index - 1][1] - 1e-9:
            raise ValueError(f"Manual intervals {index - 1} and {index} overlap.")
    return ordered


def format_manual_intervals(
    intervals: pd.DataFrame | list[tuple[float, float]],
) -> str:
    """Format intervals for the notebook editor."""
    if isinstance(intervals, pd.DataFrame):
        values = list(
            intervals.sort_values("segment_index")[["start_sec", "end_sec"]]
            .astype(float)
            .itertuples(index=False, name=None)
        )
    else:
        values = intervals
    return "\n".join(f"{start:.6f},{end:.6f}" for start, end in values)


def _atomic_csv_write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def save_segmentation_review_entry(
    *,
    review_path: str | Path,
    overrides_path: str | Path,
    logical_recording_id: str,
    duration_sec: float,
    decision: str,
    boundary_source: str,
    reviewer: str,
    notes: str = "",
    manual_intervals_text: str = "",
    review_date: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Atomically save one eligibility/boundary decision and its manual intervals."""
    review_path = Path(review_path)
    overrides_path = Path(overrides_path)
    review = pd.read_csv(review_path, dtype=str, keep_default_na=False)
    missing = [column for column in SEGMENTATION_ADJUDICATION_COLUMNS if column not in review]
    if missing:
        raise ValueError(f"Review sheet is missing columns: {missing}")
    logical_id = str(logical_recording_id)
    selected = review["logical_recording_id"].astype(str).eq(logical_id)
    if selected.sum() != 1:
        raise ValueError(f"Expected one review row for {logical_id}; found {int(selected.sum())}.")

    decision = str(decision).strip().upper()
    source = str(boundary_source).strip().upper()
    reviewer = str(reviewer).strip()
    notes = str(notes).strip()
    required = str(review.loc[selected, "review_required"].iloc[0]).strip().lower() in {
        "true",
        "1",
        "yes",
        "y",
    }
    system_excluded = str(
        review.loc[selected, "automatic_task_exclusion"].iloc[0]
    ).strip().lower() in {"true", "1", "yes", "y"}
    if system_excluded:
        raise ValueError(
            "This recording is locked to EXCLUDE because frozen metadata says "
            "Task Completed as Instructed = NO."
        )
    if decision not in {"KEEP", "EXCLUDE"}:
        raise ValueError("Decision must be KEEP or EXCLUDE.")
    if decision == "KEEP" and source not in {"AUTO", "MANUAL"}:
        raise ValueError("KEEP requires boundary_source AUTO or MANUAL.")
    if decision == "EXCLUDE" and source != "NONE":
        raise ValueError("EXCLUDE requires boundary_source NONE.")
    if (required or decision == "EXCLUDE" or source == "MANUAL") and not reviewer:
        raise ValueError("This decision requires a reviewer name.")
    if (decision == "EXCLUDE" or source == "MANUAL") and not notes:
        raise ValueError("Exclusions and manual boundary changes require notes.")
    review_date = str(review_date or date.today().isoformat()).strip()

    if overrides_path.exists():
        try:
            overrides = pd.read_csv(overrides_path, dtype=str, keep_default_na=False)
        except pd.errors.EmptyDataError:
            overrides = pd.DataFrame(columns=MANUAL_SEGMENTATION_COLUMNS)
    else:
        overrides = pd.DataFrame(columns=MANUAL_SEGMENTATION_COLUMNS)
    for column in MANUAL_SEGMENTATION_COLUMNS:
        if column not in overrides:
            overrides[column] = ""
    overrides = overrides[MANUAL_SEGMENTATION_COLUMNS]
    overrides = overrides.loc[~overrides["logical_recording_id"].astype(str).eq(logical_id)].copy()

    if decision == "KEEP" and source == "MANUAL":
        parsed = parse_manual_intervals_text(manual_intervals_text)
        ordered = validate_manual_interval_list(parsed, duration_sec=float(duration_sec))
        file_name = review.loc[selected, "file_name"].iloc[0]
        new_rows = pd.DataFrame(
            [
                {
                    "logical_recording_id": logical_id,
                    "file_name": file_name,
                    "segment_index": index,
                    "start_sec": start,
                    "end_sec": end,
                    "reviewer": reviewer,
                    "review_date": review_date,
                    "notes": notes,
                }
                for index, (start, end) in enumerate(ordered)
            ],
            columns=MANUAL_SEGMENTATION_COLUMNS,
        )
        overrides = (
            new_rows if overrides.empty else pd.concat([overrides, new_rows], ignore_index=True)
        )

    review.loc[selected, "decision"] = decision
    review.loc[selected, "boundary_source"] = source
    review.loc[selected, "reviewer"] = reviewer
    review.loc[selected, "review_date"] = review_date
    review.loc[selected, "notes"] = notes
    _atomic_csv_write(review[SEGMENTATION_ADJUDICATION_COLUMNS], review_path)
    _atomic_csv_write(
        overrides.sort_values(["file_name", "segment_index"])[MANUAL_SEGMENTATION_COLUMNS],
        overrides_path,
    )
    return review, overrides


def launch_segmentation_review_widget(
    *,
    summary: pd.DataFrame,
    automatic_intervals: pd.DataFrame,
    review_path: str | Path,
    overrides_path: str | Path,
    default_reviewer: str = "",
    ffmpeg: str = "ffmpeg",
    ffprobe: str = "ffprobe",
):
    """Build a scrollable Jupyter curation interface for every recording.

    Ordinary recordings can be retained with one click. Manual boundaries remain
    optional, and frozen-metadata task exclusions are visible but locked.
    """
    import ipywidgets as widgets
    import matplotlib.pyplot as plt
    from IPython.display import Audio, Image, clear_output, display

    review_path = Path(review_path)
    overrides_path = Path(overrides_path)
    state: dict[str, pd.DataFrame] = {}

    queue_filter = widgets.Dropdown(
        options=[
            ("All recordings", "ALL"),
            ("Pending decisions", "PENDING"),
            ("Mandatory review queue", "MANDATORY"),
            ("Kept with automatic Silero", "KEEP_AUTO"),
            ("Kept with manual boundaries", "KEEP_MANUAL"),
            ("Excluded", "EXCLUDED"),
            ("Task-invalid (system excluded)", "TASK_EXCLUDED"),
        ],
        value="ALL",
        description="Show:",
        layout=widgets.Layout(width="48%"),
    )
    search = widgets.Text(
        value="",
        description="Find:",
        placeholder="filename, participant ID, QC flag, or review reason",
        layout=widgets.Layout(width="48%"),
        continuous_update=False,
    )
    selector = widgets.Select(
        description="",
        rows=12,
        layout=widgets.Layout(width="98%", height="245px"),
    )
    decision = widgets.ToggleButtons(
        options=["UNRESOLVED", "KEEP", "EXCLUDE"],
        description="Decision:",
    )
    source = widgets.ToggleButtons(
        options=["UNRESOLVED", "AUTO", "MANUAL", "NONE"],
        description="Boundaries:",
    )
    reviewer = widgets.Text(
        value=default_reviewer,
        description="Reviewer:",
        layout=widgets.Layout(width="70%"),
    )
    notes = widgets.Textarea(
        description="Notes:",
        layout=widgets.Layout(width="95%", height="90px"),
    )
    manual_text = widgets.Textarea(
        description="Speech intervals:",
        placeholder="One start_sec,end_sec interval per line",
        layout=widgets.Layout(width="95%", height="180px"),
    )
    save_button = widgets.Button(description="Save current", icon="save")
    keep_auto_button = widgets.Button(
        description="Keep Silero + next",
        button_style="success",
        icon="check",
        tooltip="Accept the automatic Silero boundaries and advance.",
    )
    exclude_button = widgets.Button(
        description="Exclude + next",
        button_style="danger",
        icon="ban",
        tooltip="Exclude this recording; a reason is required.",
    )
    save_manual_button = widgets.Button(
        description="Save manual + next",
        button_style="warning",
        icon="pencil",
        tooltip="Use the complete manual interval list; a reason is required.",
    )
    preview_button = widgets.Button(
        description="Preview manual boundaries", button_style="info", icon="eye"
    )
    previous_button = widgets.Button(description="Previous", icon="arrow-left")
    next_button = widgets.Button(description="Next", icon="arrow-right")
    progress = widgets.HTML()
    status = widgets.HTML()
    recording_output = widgets.Output()
    preview_output = widgets.Output()
    manual_box = widgets.VBox([manual_text, preview_button])

    def reload_tables() -> None:
        state["review"] = pd.read_csv(review_path, dtype=str, keep_default_na=False)
        if overrides_path.exists():
            try:
                state["overrides"] = pd.read_csv(overrides_path, dtype=str, keep_default_na=False)
            except pd.errors.EmptyDataError:
                state["overrides"] = pd.DataFrame(columns=MANUAL_SEGMENTATION_COLUMNS)
        else:
            state["overrides"] = pd.DataFrame(columns=MANUAL_SEGMENTATION_COLUMNS)

    def is_true(value: object) -> bool:
        return str(value).strip().lower() in {"true", "1", "yes", "y"}

    def row_complete(row: pd.Series) -> bool:
        row_decision = str(row["decision"]).strip().upper()
        row_source = str(row["boundary_source"]).strip().upper()
        system_excluded = is_true(row.get("automatic_task_exclusion", False))
        if system_excluded:
            return (
                row_decision == "EXCLUDE"
                and row_source == "NONE"
                and bool(str(row.get("notes", "")).strip())
            )
        manual_review = (
            is_true(row["review_required"]) or row_decision == "EXCLUDE" or row_source == "MANUAL"
        )
        return (
            row_decision in {"KEEP", "EXCLUDE"}
            and (
                (row_decision == "KEEP" and row_source in {"AUTO", "MANUAL"})
                or (row_decision == "EXCLUDE" and row_source == "NONE")
            )
            and (not manual_review or bool(str(row["reviewer"]).strip()))
            and (not manual_review or bool(str(row["review_date"]).strip()))
            and (
                row_decision != "EXCLUDE"
                and row_source != "MANUAL"
                or bool(str(row["notes"]).strip())
            )
        )

    def row_state(row: pd.Series) -> str:
        if is_true(row.get("automatic_task_exclusion", False)):
            return "SYSTEM EXCLUDED"
        if not row_complete(row):
            return "PENDING"
        row_decision = str(row["decision"]).strip().upper()
        row_source = str(row["boundary_source"]).strip().upper()
        if row_decision == "EXCLUDE":
            return "EXCLUDED"
        if row_source == "MANUAL":
            return "KEEP MANUAL"
        return "KEEP AUTO"

    def refresh_options(*_args) -> None:
        current = selector.value
        work = state["review"].copy()
        filter_value = str(queue_filter.value)
        if filter_value == "PENDING":
            work = work.loc[~work.apply(row_complete, axis=1)]
        elif filter_value == "MANDATORY":
            work = work.loc[work["review_required"].map(is_true)]
        elif filter_value == "KEEP_AUTO":
            work = work.loc[
                work["decision"].astype(str).str.upper().eq("KEEP")
                & work["boundary_source"].astype(str).str.upper().eq("AUTO")
            ]
        elif filter_value == "KEEP_MANUAL":
            work = work.loc[
                work["decision"].astype(str).str.upper().eq("KEEP")
                & work["boundary_source"].astype(str).str.upper().eq("MANUAL")
            ]
        elif filter_value == "EXCLUDED":
            work = work.loc[
                work["decision"].astype(str).str.upper().eq("EXCLUDE")
            ]
        elif filter_value == "TASK_EXCLUDED":
            work = work.loc[work["automatic_task_exclusion"].map(is_true)]
        query = search.value.strip().lower()
        if query:
            searchable = work[
                [
                    "logical_recording_id",
                    "file_name",
                    "automatic_qc_status",
                    "qc_flags",
                    "review_reasons",
                    "task_completed_as_instructed",
                ]
            ].fillna("").astype(str).agg(" | ".join, axis=1).str.lower()
            work = work.loc[searchable.str.contains(query, regex=False)]
        options = []
        for row in work.itertuples():
            row_series = pd.Series(row._asdict())
            marker = row_state(row_series)
            label = f"[{marker}] {row.automatic_qc_status} | {row.file_name}"
            options.append((label, str(row.logical_recording_id)))
        selector.options = options
        values = [value for _, value in options]
        if current in values:
            selector.value = current
        elif values:
            selector.value = values[0]
        total = len(state["review"])
        total_complete = int(state["review"].apply(row_complete, axis=1).sum())
        total_required = int(state["review"]["review_required"].map(is_true).sum())
        required_rows = state["review"].loc[state["review"]["review_required"].map(is_true)]
        completed = int(required_rows.apply(row_complete, axis=1).sum())
        task_excluded = int(
            state["review"]["automatic_task_exclusion"].map(is_true).sum()
        )
        progress.value = (
            f"<b>All decisions:</b> {total_complete}/{total} complete &nbsp; | &nbsp; "
            f"<b>Mandatory:</b> {completed}/{total_required} complete &nbsp; | &nbsp; "
            f"<b>Locked task exclusions:</b> {task_excluded}"
        )

    def current_rows() -> tuple[pd.Series, pd.Series]:
        logical_id = str(selector.value)
        review_row = (
            state["review"]
            .loc[state["review"]["logical_recording_id"].astype(str).eq(logical_id)]
            .iloc[0]
        )
        summary_row = summary.loc[summary["logical_recording_id"].astype(str).eq(logical_id)].iloc[
            0
        ]
        return review_row, summary_row

    def render_recording(change=None) -> None:
        if change is not None and change.get("name") != "value":
            return
        if selector.value is None:
            return
        review_row, summary_row = current_rows()
        saved_decision = str(review_row["decision"]).strip().upper()
        decision.value = saved_decision if saved_decision in {"KEEP", "EXCLUDE"} else "UNRESOLVED"
        saved_source = str(review_row["boundary_source"]).strip().upper()
        source.value = saved_source if saved_source in {"AUTO", "MANUAL", "NONE"} else "UNRESOLVED"
        reviewer.value = str(review_row["reviewer"]).strip() or default_reviewer
        notes.value = str(review_row["notes"]).strip()
        locked = is_true(review_row.get("automatic_task_exclusion", False))
        for control in [
            decision,
            source,
            reviewer,
            notes,
            manual_text,
            save_button,
            keep_auto_button,
            exclude_button,
            save_manual_button,
            preview_button,
        ]:
            control.disabled = locked
        manual_box.layout.display = (
            "" if (not locked and source.value == "MANUAL") else "none"
        )
        logical_id = str(review_row["logical_recording_id"])
        saved_manual = state["overrides"].loc[
            state["overrides"]["logical_recording_id"].astype(str).eq(logical_id)
        ]
        if not saved_manual.empty:
            manual_text.value = format_manual_intervals(saved_manual)
        else:
            automatic = automatic_intervals.loc[
                automatic_intervals["logical_recording_id"].astype(str).eq(logical_id)
                & automatic_intervals["profile"].eq("primary")
                & automatic_intervals["view"].eq("primary_speech")
            ].sort_values("start_sec")
            manual_text.value = "\n".join(
                f"{float(row.start_sec):.6f},{float(row.end_sec):.6f}"
                for row in automatic.itertuples()
            )

        with recording_output:
            clear_output(wait=True)
            fields = [
                "file_name",
                "automatic_qc_status",
                "qc_flags",
                "task_completed_as_instructed",
                "automatic_task_exclusion",
                "automatic_exclusion_reason",
                "accepted_outlier",
                "accepted_outlier_max_abs_robust_z",
                "review_reasons",
            ]
            display(pd.DataFrame([review_row[fields].to_dict()]))
            quantitative = [
                "duration_sec",
                "speech_fraction",
                "n_speech_segments",
                "n_internal_nonspeech_segments",
                "leading_nonspeech_sec",
                "trailing_nonspeech_sec",
                "longest_internal_nonspeech_sec",
                "rms_db_median",
                "rms_db_std",
            ]
            display(pd.DataFrame([summary_row[quantitative].to_dict()]))
            plot_path = Path(str(summary_row["plot_path"]))
            if plot_path.exists():
                display(Image(filename=str(plot_path), width=1100))
            else:
                print("Missing diagnostic figure:", plot_path)
            boundary_plot_path = Path(str(summary_row.get("boundary_plot_path", "")))
            if boundary_plot_path.is_file():
                display(Image(filename=str(boundary_plot_path), width=1100))
            media_path = Path(str(summary_row["file_path"]))
            if media_path.exists() and media_path.suffix.lower() == ".wav":
                display(Audio(filename=str(media_path)))
            elif media_path.exists():
                from .media import decode_audio_views

                decoded = decode_audio_views(media_path, ffmpeg=ffmpeg, ffprobe=ffprobe)
                display(Audio(decoded.analysis_16k, rate=16000))
            else:
                print("Missing media file:", media_path)
        preview_output.clear_output()
        if locked:
            status.value = (
                "<span style='color:#8a4b08'><b>Locked system exclusion.</b> "
                "Frozen metadata records Task Completed as Instructed = NO. "
                "This file remains browsable and auditable but cannot be included here.</span>"
            )
        else:
            status.value = ""

    def preview_manual(_button) -> None:
        with preview_output:
            clear_output(wait=True)
            try:
                _, summary_row = current_rows()
                intervals = validate_manual_interval_list(
                    parse_manual_intervals_text(manual_text.value),
                    duration_sec=float(summary_row["duration_sec"]),
                )
                from .media import decode_audio_views

                decoded = decode_audio_views(
                    str(summary_row["file_path"]),
                    ffmpeg=ffmpeg,
                    ffprobe=ffprobe,
                )
                waveform = decoded.analysis_16k
                time = np.arange(len(waveform)) / 16000
                if len(waveform) > 100_000:
                    indices = np.linspace(0, len(waveform) - 1, 100_000).astype(int)
                    time = time[indices]
                    waveform = waveform[indices]
                fig, ax = plt.subplots(figsize=(16, 4))
                ax.plot(time, waveform, linewidth=0.45, color="0.25")
                for index, (start, end) in enumerate(intervals):
                    ax.axvspan(
                        start,
                        end,
                        color="#66bb6a",
                        alpha=0.30,
                        label="manual speech" if index == 0 else None,
                    )
                ax.set(
                    title=f"Manual boundary preview: {summary_row['file_name']}",
                    xlabel="Time (s)",
                    ylabel="Amplitude",
                )
                ax.legend(frameon=False)
                plt.show()
            except Exception as exc:
                print("PREVIEW BLOCKED:", exc)

    def update_manual_visibility(change=None) -> None:
        if change is not None and change.get("name") != "value":
            return
        if selector.value is None:
            manual_box.layout.display = "none"
            return
        review_row, _ = current_rows()
        locked = is_true(review_row.get("automatic_task_exclusion", False))
        manual_box.layout.display = (
            "" if (not locked and source.value == "MANUAL") else "none"
        )

    def save_current(
        _button=None,
        *,
        desired_decision: str | None = None,
        desired_source: str | None = None,
        advance: bool = False,
    ) -> None:
        try:
            review_row, summary_row = current_rows()
            chosen_decision = desired_decision or decision.value
            chosen_source = desired_source or source.value
            save_segmentation_review_entry(
                review_path=review_path,
                overrides_path=overrides_path,
                logical_recording_id=str(review_row["logical_recording_id"]),
                duration_sec=float(summary_row["duration_sec"]),
                decision=chosen_decision,
                boundary_source=chosen_source,
                reviewer=reviewer.value,
                notes=notes.value,
                manual_intervals_text=manual_text.value,
            )
            status.value = (
                "<span style='color:#137333'><b>Saved.</b> "
                "This is still editable until the freeze command is run.</span>"
            )
            reload_tables()
            current = selector.value
            refresh_options()
            current_values = [value for _, value in selector.options]
            if current in current_values:
                selector.value = current
                if advance:
                    move(1)
            elif selector.value is not None:
                render_recording()
        except Exception as exc:
            status.value = (
                "<span style='color:#b00020'><b>SAVE BLOCKED:</b> "
                f"{html.escape(str(exc))}</span>"
            )

    def move(offset: int) -> None:
        values = [value for _, value in selector.options]
        if not values or selector.value not in values:
            return
        index = values.index(selector.value)
        selector.value = values[(index + offset) % len(values)]

    reload_tables()
    queue_filter.observe(refresh_options, names="value")
    search.observe(refresh_options, names="value")
    selector.observe(render_recording, names="value")
    save_button.on_click(save_current)
    keep_auto_button.on_click(
        lambda button: save_current(
            button,
            desired_decision="KEEP",
            desired_source="AUTO",
            advance=True,
        )
    )
    exclude_button.on_click(
        lambda button: save_current(
            button,
            desired_decision="EXCLUDE",
            desired_source="NONE",
            advance=True,
        )
    )
    save_manual_button.on_click(
        lambda button: save_current(
            button,
            desired_decision="KEEP",
            desired_source="MANUAL",
            advance=True,
        )
    )
    preview_button.on_click(preview_manual)
    previous_button.on_click(lambda _button: move(-1))
    next_button.on_click(lambda _button: move(1))
    source.observe(update_manual_visibility, names="value")
    refresh_options()
    render_recording()

    instructions = widgets.HTML(
        """
        <b>Review rules</b><br>
        • Browse every recording in the scrollable list; filter or search when useful.<br>
        • <b>Keep Silero + next</b>: retain automatic boundaries and advance.<br>
        • KEEP + MANUAL: enter one speech interval per line as
          <code>start_sec,end_sec</code>; preview, then save with a reason.<br>
        • EXCLUDE + NONE: exclude the recording and document the reason.<br>
        • Task Completed as Instructed = NO is a visible, locked system exclusion.<br>
        • Do not correct acoustic noise itself here; change only speech boundaries.
        """
    )
    controls = widgets.VBox(
        [
            instructions,
            widgets.HBox([queue_filter, search]),
            progress,
            selector,
            recording_output,
            decision,
            source,
            reviewer,
            notes,
            manual_box,
            widgets.HBox(
                [
                    keep_auto_button,
                    save_manual_button,
                    exclude_button,
                    save_button,
                ]
            ),
            widgets.HBox([previous_button, next_button]),
            status,
            preview_output,
        ]
    )
    return controls
