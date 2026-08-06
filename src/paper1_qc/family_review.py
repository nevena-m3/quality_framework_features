from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Audio, clear_output, display
from scipy import signal

from .human_qc import load_interval_human_qc
from .media import decode_native_audio


@dataclass(frozen=True)
class ReviewConfig:
    code: str
    folder: str
    human_category: str
    title: str
    ranking: tuple[tuple[str, str], ...]
    event_ledger: str | None = None


CONFIGS = {
    "QGAIN": ReviewConfig(
        "QGAIN", "01_QGAIN", "gain_dynamics", "Gain and level dynamics",
        (("qgain_within_segment_iqr_db", "high"), ("qgain_between_segment_mad_db", "high")),
    ),
    "QADD": ReviewConfig(
        "QADD", "02_QADD", "additive_interference", "Additive interference",
        (("qadd_pause_ac_level_dbfs_median", "high"), ("qadd_speech_pause_level_contrast_db", "low")),
    ),
    "QREV": ReviewConfig(
        "QREV", "03_QREV", "reverberation_tail", "Reverberation and residual tails",
        (("qrev_tail_excess_100ms_db", "high"), ("qrev_srmr_norm", "low")),
    ),
    "QCHAN": ReviewConfig(
        "QCHAN", "04_QCHAN", "channel_device", "Channel/device spectral effects",
        (("qchan_ltas_distance_db", "high"), ("qchan_rolloff95_deficit_hz", "high")),
    ),
    "QDIST": ReviewConfig(
        "QDIST", "05_QDIST", "nonlinear_distortion", "Hard-clipping morphology",
        (("qdist_hard_clipped_sample_fraction", "positive"),),
        "MAIN outputs/02_FEATURE_REVIEWED/06_family_freezes/nonlinear_distortion/"
        "qdist-v4.1.0/source_candidate/tables/qdist_v410_episode_ledger.csv",
    ),
}

RATERS = ["Abbas", "Liya", "Samaana", "Samara"]


def _human_tables(human_root: Path, category: str):
    parts = []
    interval_parts = []
    for source, design, excluded in (
        (human_root, "distributed_main", ["Reliability"]),
        (human_root / "Reliability", "crossed_reliability", []),
    ):
        ratings, _, intervals, issues = load_interval_human_qc(
            source,
            rater_strategy="parent_directory",
            rater_directory_names=RATERS,
            exclude_path_parts=excluded,
            interval_time_base="absolute",
        )
        if len(issues):
            raise ValueError(f"Human-QC parsing issues in {design}: {issues.to_dict('records')[:5]}")
        ratings = ratings.loc[ratings["category"].eq(category)].copy()
        intervals = intervals.loc[intervals["family"].eq(category)].copy()
        ratings["design"] = design
        intervals["design"] = design
        parts.append(ratings)
        interval_parts.append(intervals)
    ratings = pd.concat(parts, ignore_index=True)
    intervals = pd.concat(interval_parts, ignore_index=True)
    for frame in (ratings, intervals):
        frame["media_file_name"] = frame["file_name"].astype(str).str.split("__").str[-1]
        frame["logical_recording_id"] = frame["media_file_name"].map(lambda x: Path(x).stem)
    positive = ratings.loc[ratings["rating"].eq(1)].copy()
    positive_intervals = intervals.merge(
        positive[["file_name", "rater_id", "design", "logical_recording_id"]].drop_duplicates(),
        on=["file_name", "rater_id", "design", "logical_recording_id"],
        how="inner",
        validate="many_to_one",
    )
    return ratings, positive, positive_intervals


def _review_media(project_root: Path, data_root: Path, human_ids: set[str]) -> pd.DataFrame:
    freeze = pd.read_csv(
        project_root / "MAIN outputs" / "00_DATA_FREEZE" / "v1" / "frozen_bamboo_recordings.csv"
    )[["logical_recording_id", "SubjectID", "selected_media_file_name", "media_path"]]
    missing = human_ids - set(freeze["logical_recording_id"])
    missing |= set(
        freeze.loc[
            freeze["logical_recording_id"].isin(human_ids) & freeze["media_path"].isna(),
            "logical_recording_id",
        ]
    )
    if missing:
        index = {}
        for path in (data_root / "Bamboo_passage_only").rglob("*"):
            if path.is_file() and path.suffix.lower() in {".wav", ".webm", ".mp4"}:
                index.setdefault(path.stem.casefold(), path)
        rows = []
        for logical_id in sorted(missing):
            path = index.get(logical_id.casefold())
            if path is None:
                raise FileNotFoundError(f"Human-rated audio not found: {logical_id}")
            rows.append(
                {
                    "logical_recording_id": logical_id,
                    "SubjectID": logical_id.split("_")[0],
                    "selected_media_file_name": path.name,
                    "media_path": str(path),
                }
            )
        freeze = pd.concat([freeze.loc[~freeze["logical_recording_id"].isin(missing)], pd.DataFrame(rows)])
    return freeze.drop_duplicates("logical_recording_id")


def prepare_family_review(
    project_root: str | Path,
    family_code: str,
    *,
    data_root: str | Path,
    extreme_per_feature: int = 12,
) -> dict[str, object]:
    root = Path(project_root).resolve()
    data_root = Path(data_root)
    config = CONFIGS[family_code.upper()]
    features = pd.read_csv(
        root / "MAIN outputs" / "02_FEATURE_LATEST" / "families" / config.folder / "features.csv"
    )
    ratings, human_positive, human_intervals = _human_tables(
        data_root / "Bamboo_passage_HumanQC", config.human_category
    )
    human_ids = set(human_positive["logical_recording_id"])
    media = _review_media(root, data_root, human_ids)

    selections = []
    for feature, direction in config.ranking:
        values = pd.to_numeric(features[feature], errors="coerce")
        if direction == "positive":
            chosen = features.loc[values.gt(0), ["logical_recording_id"]].copy()
        else:
            ordered = values.sort_values(ascending=direction == "low", na_position="last")
            chosen = features.loc[ordered.head(extreme_per_feature).index, ["logical_recording_id"]].copy()
        chosen["objective_selection_reason"] = f"{feature}:{direction}"
        selections.append(chosen)
    objective = pd.concat(selections, ignore_index=True).groupby("logical_recording_id", as_index=False).agg(
        objective_selection_reason=("objective_selection_reason", lambda x: "; ".join(sorted(set(x))))
    )
    human = pd.DataFrame({"logical_recording_id": sorted(human_ids), "human_positive": True})
    candidates = objective.assign(objective_selected=True).merge(human, on="logical_recording_id", how="outer")
    for column in ("objective_selected", "human_positive"):
        candidates[column] = candidates[column].map(lambda value: value is True)
    candidates = candidates.merge(media, on="logical_recording_id", how="left", validate="one_to_one")
    candidates = candidates.merge(features, on="logical_recording_id", how="left", validate="one_to_one")
    summary = human_positive.groupby("logical_recording_id").agg(
        human_positive_ratings=("rating", "sum"),
        human_raters=("rater_id", lambda x: ", ".join(sorted(set(x)))),
        human_designs=("design", lambda x: ", ".join(sorted(set(x)))),
    ).reset_index()
    candidates = candidates.merge(summary, on="logical_recording_id", how="left")
    candidates["source_group"] = np.select(
        [candidates["objective_selected"] & candidates["human_positive"], candidates["objective_selected"]],
        ["objective + human", "objective only"],
        default="human only",
    )
    candidates = candidates.sort_values(["source_group", "selected_media_file_name"]).reset_index(drop=True)
    if candidates["media_path"].isna().any():
        raise ValueError("Review candidates include unresolved audio paths")

    events = pd.read_csv(root / config.event_ledger) if config.event_ledger else pd.DataFrame()
    out = root / "outputs" / "06_family_manual_review" / config.folder
    out.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(out / "review_recordings.csv", index=False)
    human_positive.to_csv(out / "human_positive_ratings.csv", index=False)
    human_intervals.to_csv(out / "human_intervals.csv", index=False)
    if len(events):
        events.to_csv(out / "objective_event_intervals.csv", index=False)
    adjudication = out / "manual_verification.csv"
    if not adjudication.exists():
        candidates[["logical_recording_id", "selected_media_file_name", "source_group"]].assign(
            reviewer="", review_status="PENDING", feature_works="", artifact_present="",
            artifact_type="", notes=""
        ).to_csv(adjudication, index=False)
    return {
        "config": config, "features": features, "candidates": candidates,
        "human_ratings": ratings, "human_positive": human_positive,
        "human_intervals": human_intervals, "events": events, "output": out,
        "adjudication": adjudication,
    }


def launch_family_review(review: dict[str, object]) -> None:
    config = review["config"]
    candidates = review["candidates"]
    events = review["events"]
    human_intervals = review["human_intervals"]
    adjudication = review["adjudication"]
    feature_names = [
        column
        for column in review["features"].columns
        if column.startswith(config.code.lower() + "_")
        and not column.endswith("_status")
        and not any(token in column for token in ("version", "role", "available"))
    ]

    def selector_options(frame):
        return [
            (f"{row.source_group} | {row.selected_media_file_name}", row.logical_recording_id)
            for row in frame.itertuples()
        ]

    source_filter = widgets.Dropdown(
        options=["all", "objective selected", "human positive", "objective + human"],
        value="objective selected", description="Cases:"
    )
    selector = widgets.Dropdown(
        options=selector_options(candidates.loc[candidates["objective_selected"]]),
        description="Recording:", layout=widgets.Layout(width="95%")
    )
    mode = widgets.ToggleButtons(options=["full recording", "event windows"], value="event windows", description="View:")
    reviewer = widgets.Text(description="Reviewer:")
    status = widgets.Dropdown(options=["PENDING", "WORKS", "DOES_NOT_WORK", "UNCERTAIN"], description="Feature:")
    present = widgets.Dropdown(options=["", "yes", "no", "uncertain"], description="Audible:")
    kind = widgets.Text(description="Type:")
    notes = widgets.Textarea(description="Notes:", layout=widgets.Layout(width="80%"))
    save = widgets.Button(description="Save verification", button_style="success")
    viewer, message = widgets.Output(), widgets.Output()

    def spans_for(logical_id):
        spans = []
        if len(events):
            local = events.loc[events["logical_recording_id"].eq(logical_id)]
            for row in local.itertuples():
                spans.append((float(row.start_sec_native), float(row.end_sec_native), "objective", "tab:red"))
        local_human = human_intervals.loc[human_intervals["logical_recording_id"].eq(logical_id)]
        for row in local_human.itertuples():
            spans.append((float(row.start_sec), float(row.end_sec), f"human:{row.rater_id}", "tab:blue"))
        return spans, local_human

    def render(*_):
        with viewer:
            clear_output(wait=True)
            if selector.value is None:
                print("No recordings in this source group.")
                return
            row = candidates.loc[candidates["logical_recording_id"].eq(selector.value)].iloc[0]
            decoded = decode_native_audio(row["media_path"], ffmpeg="ffmpeg", ffprobe="ffprobe")
            y = decoded.native.mean(axis=1, dtype=np.float64)
            fs = decoded.sample_rate_native
            spans, local_human = spans_for(row["logical_recording_id"])
            duration = len(y) / fs
            if mode.value == "event windows" and spans:
                lo, hi = max(0, min(x[0] for x in spans) - 1), min(duration, max(x[1] for x in spans) + 1)
            else:
                lo, hi = 0.0, duration
            i0, i1 = int(lo * fs), int(hi * fs)
            view = y[i0:i1]
            times = np.arange(i0, i1) / fs
            fig, axes = plt.subplots(2, 1, figsize=(15, 7), sharex=True, constrained_layout=True)
            axes[0].plot(times, view, lw=.45, color="black")
            axes[0].set_ylabel("Amplitude")
            nperseg = min(1024, max(64, len(view)))
            freq, time, power = signal.spectrogram(view, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
            axes[1].pcolormesh(time + lo, freq, 10 * np.log10(power + 1e-12), shading="auto", cmap="magma")
            axes[1].set_ylim(0, min(8000, fs / 2))
            axes[1].set(ylabel="Frequency (Hz)", xlabel="Time (s)")
            used = set()
            for start, end, label, color in spans:
                for axis in axes:
                    axis.axvspan(start, end, color=color, alpha=.25, label=label if label not in used else None)
                used.add(label)
            if used:
                axes[0].legend(loc="upper right")
            fig.suptitle(f"{config.code}: {row['selected_media_file_name']} | {row['source_group']}")
            plt.show()
            display(pd.DataFrame([{name: row.get(name) for name in ["logical_recording_id", "source_group", "objective_selection_reason", "human_raters", *feature_names]}]))
            if len(local_human):
                display(local_human[["design", "rater_id", "subcategory", "start_sec", "end_sec", "duration_sec"]])
            print("Full recording:")
            display(Audio(y, rate=fs, normalize=False))
            if spans:
                print(f"Review window {lo:.2f}–{hi:.2f} s:")
                display(Audio(view, rate=fs, normalize=False))

    def save_review(_):
        table = pd.read_csv(adjudication, keep_default_na=False)
        mask = table["logical_recording_id"].eq(selector.value)
        table.loc[mask, ["reviewer", "review_status", "feature_works", "artifact_present", "artifact_type", "notes"]] = [
            reviewer.value, status.value, status.value, present.value, kind.value, notes.value
        ]
        table.to_csv(adjudication, index=False)
        with message:
            clear_output(wait=True)
            print("Saved:", selector.value, "→", adjudication)

    def apply_filter(change=None):
        if source_filter.value == "objective selected":
            local = candidates.loc[candidates["objective_selected"]]
        elif source_filter.value == "human positive":
            local = candidates.loc[candidates["human_positive"]]
        elif source_filter.value == "objective + human":
            local = candidates.loc[candidates["objective_selected"] & candidates["human_positive"]]
        else:
            local = candidates
        selector.options = selector_options(local)
        if not len(local):
            with viewer:
                clear_output(wait=True)
                print("No recordings in this source group.")

    selector.observe(render, names="value")
    mode.observe(render, names="value")
    source_filter.observe(apply_filter, names="value")
    save.on_click(save_review)
    display(widgets.VBox([source_filter, selector, mode, widgets.HBox([reviewer, status, present]), kind, notes, save, message]), viewer)
    render()
