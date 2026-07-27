import json
from pathlib import Path

import pandas as pd
import pytest

from paper1_qc import cli
from paper1_qc.segmentation import (
    MANUAL_SEGMENTATION_COLUMNS,
    segmentation_adjudication_template,
)


def test_segment_adjudicate_writes_immutable_main_outputs_freeze(
    tmp_path, monkeypatch
):
    output = tmp_path / "outputs" / "01_segmentation"
    output.mkdir(parents=True)
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config_path = config_dir / "project.yaml"
    config_path.write_text("project: {name: test}\n", encoding="utf-8")
    automatic_segments = output / "recording-1_segments.csv"
    automatic_frames = output / "recording-1_frames.csv"
    automatic_plot = output / "recording-1_silero.png"
    pd.DataFrame({"start_sec": [1.0], "end_sec": [4.0]}).to_csv(
        automatic_segments, index=False
    )
    pd.DataFrame({"time_sec": [0.0], "is_speech": [False]}).to_csv(
        automatic_frames, index=False
    )
    automatic_plot.write_bytes(b"test-image")

    summary = pd.DataFrame(
        {
            "logical_recording_id": ["recording-1"],
            "file_name": ["recording-1.wav"],
            "file_path": [str(tmp_path / "recording-1.wav")],
            "qc_status": ["accepted"],
            "qc_flags": [""],
            "Task Completed as Instructed": ["Yes"],
            "duration_sec": [10.0],
            "speech_fraction": [0.6],
            "n_speech_segments": [2],
            "n_internal_nonspeech_segments": [1],
            "leading_nonspeech_sec": [1.0],
            "trailing_nonspeech_sec": [1.0],
            "longest_internal_nonspeech_sec": [1.0],
            "rms_db_median": [-30.0],
            "rms_db_std": [5.0],
            "segments_path": [str(automatic_segments)],
            "frames_path": [str(automatic_frames)],
            "plot_path": [str(automatic_plot)],
            "boundary_audit_path": [""],
            "boundary_plot_path": [""],
        }
    )
    summary.to_csv(output / "bamboo_segmentation_summary.csv", index=False)
    intervals = pd.DataFrame(
        [
            {
                "file_name": "recording-1.wav",
                "view": view,
                "interval_index": 0,
                "start_sec": start,
                "end_sec": end,
                "duration_sec": end - start,
                "profile": "primary",
                "logical_recording_id": "recording-1",
            }
            for view, start, end in [
                ("raw_speech", 1.0, 4.0),
                ("primary_speech", 1.0, 4.0),
                ("strict_speech", 1.05, 3.95),
                ("strict_internal_nonspeech", 4.2, 4.8),
            ]
        ]
    )
    intervals.to_csv(output / "bamboo_segmentation_intervals.csv", index=False)

    review = segmentation_adjudication_template(
        summary, {"accepted_guardrails": {}}
    )
    review_path = config_dir / "segmentation_adjudication.csv"
    manual_path = config_dir / "manual_segmentation_overrides.csv"
    review.to_csv(review_path, index=False)
    pd.DataFrame(columns=MANUAL_SEGMENTATION_COLUMNS).to_csv(
        manual_path, index=False
    )
    cfg = {
        "_output_root": str(tmp_path / "outputs"),
        "_main_output_root": str(tmp_path / "MAIN outputs"),
        "_project_root": str(tmp_path),
        "_config_path": str(config_path),
        "data_freeze": {
            "version": "cohort-v1",
            "segmentation_adjudication": str(review_path),
            "manual_segmentation_overrides": str(manual_path),
        },
        "segmentation_freeze": {"version": "segments-v1"},
        "segmentation_review": {"accepted_guardrails": {}},
        "software": {"ffmpeg": "auto", "ffprobe": "auto"},
        "vad": {
            "threshold": 0.5,
            "min_speech_ms": 250,
            "min_silence_ms": 100,
            "strict_speech_edge_ms": 50,
            "strict_nonspeech_edge_ms": 200,
        },
    }

    def fake_manifest(path, **_kwargs):
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps({"test": True}), encoding="utf-8")
        return {"test": True}

    monkeypatch.setattr(cli, "write_run_manifest", fake_manifest)
    monkeypatch.setattr(
        cli, "resolve_executable", lambda value, default_name: default_name
    )
    cli.command_segment_adjudicate(cfg)

    freeze = (
        tmp_path
        / "MAIN outputs"
        / "01_SEGMENTATION_FREEZE"
        / "segments-v1"
    )
    assert (freeze / "frozen_segmentation_decisions.csv").exists()
    assert (freeze / "frozen_segmentation_intervals.csv").exists()
    assert (freeze / "frozen_manual_segmentation_overrides.csv").exists()
    assert (freeze / "segmentation_decision_summary.csv").exists()
    assert (freeze / "segmentation_freeze_manifest.json").exists()
    frozen_intervals = pd.read_csv(freeze / "frozen_segmentation_intervals.csv")
    assert frozen_intervals["segmentation_boundary_source"].eq(
        "automatic_silero"
    ).all()
    reviewed = tmp_path / "outputs" / "01_segmentation_after_review"
    reviewed_summary = pd.read_csv(
        reviewed
        / "segmentation"
        / "silero"
        / "summary"
        / "silero_after_review_summary.csv"
    )
    assert reviewed_summary.loc[0, "final_review_status"] == "accepted"
    assert bool(reviewed_summary.loc[0, "analysis_included"])
    assert (
        reviewed
        / "segmentation"
        / "silero"
        / "segments"
        / "accepted"
        / "recording-1_segments.csv"
    ).exists()
    assert (
        reviewed
        / "figures"
        / "segmentation"
        / "silero"
        / "accepted"
        / "recording-1_silero.png"
    ).exists()

    with pytest.raises(FileExistsError, match="will not be overwritten"):
        cli.command_segment_adjudicate(cfg)


def test_reviewed_output_keeps_flagged_and_preserves_excluded_for_audit(
    tmp_path, monkeypatch
):
    source = tmp_path / "automatic"
    source.mkdir()
    rows = []
    for logical_id, automatic_status, decision, source_name in [
        ("accepted-1", "accepted", "KEEP", "AUTO"),
        ("flagged-1", "flagged", "KEEP", "AUTO"),
        ("excluded-1", "accepted", "EXCLUDE", "NONE"),
    ]:
        segments = source / f"{logical_id}_segments.csv"
        frames = source / f"{logical_id}_frames.csv"
        figure = source / f"{logical_id}_silero.png"
        pd.DataFrame({"start_sec": [1.0], "end_sec": [2.0]}).to_csv(
            segments, index=False
        )
        pd.DataFrame({"time_sec": [0.0]}).to_csv(frames, index=False)
        figure.write_bytes(b"test-image")
        rows.append(
            {
                "logical_recording_id": logical_id,
                "file_name": f"{logical_id}.wav",
                "automatic_qc_status": automatic_status,
                "decision": decision,
                "boundary_source": source_name,
                "review_required": automatic_status == "flagged",
                "segmentation_decision_source": (
                    "manual_exclusion"
                    if decision == "EXCLUDE"
                    else (
                        "reviewed_auto_boundaries"
                        if automatic_status == "flagged"
                        else "automatic_accepted"
                    )
                ),
                "segments_path": str(segments),
                "frames_path": str(frames),
                "plot_path": str(figure),
                "boundary_audit_path": "",
                "boundary_plot_path": "",
            }
        )
    frozen = pd.DataFrame(rows)

    def fake_manifest(path, **_kwargs):
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps({"test": True}), encoding="utf-8")
        return {"test": True}

    monkeypatch.setattr(cli, "write_run_manifest", fake_manifest)
    staging = tmp_path / "staging"
    published = tmp_path / "outputs" / "01_segmentation_after_review"
    reviewed, _ = cli._materialize_reviewed_segmentation_output(
        destination=staging,
        published_destination=published,
        frozen=frozen,
        frozen_intervals=pd.DataFrame(),
        manual_artifacts=pd.DataFrame(),
        cfg={"_config_path": str(tmp_path / "project.yaml")},
        input_paths=[],
    )
    by_id = reviewed.set_index("logical_recording_id")
    assert by_id.loc["accepted-1", "final_review_status"] == "accepted"
    assert by_id.loc["flagged-1", "final_review_status"] == "flagged"
    assert bool(by_id.loc["flagged-1", "analysis_included"])
    assert by_id.loc["excluded-1", "final_review_status"] == "excluded"
    assert not bool(by_id.loc["excluded-1", "analysis_included"])
    assert (
        staging
        / "segmentation"
        / "silero"
        / "segments"
        / "flagged"
        / "flagged-1_segments.csv"
    ).exists()
    assert (
        staging
        / "figures"
        / "segmentation"
        / "silero"
        / "excluded"
        / "excluded-1_silero.png"
    ).exists()
