import numpy as np
import pandas as pd
import pytest

from paper1_qc.segmentation import (
    LEGACY_SILERO_FRAME_COLUMNS,
    LEGACY_SILERO_SEGMENT_COLUMNS,
    Interval,
    apply_segmentation_adjudication,
    boundary_alignment_diagnostics,
    build_segmentation_views,
    classify_reading_segmentation,
    freeze_segmentation_intervals,
    legacy_silero_artifacts,
    plot_segmentation_diagnostic,
    segmentation_adjudication_template,
    segmentation_pending_reviews,
    segmentation_review_selection,
    summarize_legacy_silero_artifacts,
)
from paper1_qc.segmentation_review import (
    parse_manual_intervals_text,
    save_segmentation_review_entry,
)


def test_segmentation_views_are_distinct_and_guarded():
    raw = [Interval(0.10, 1.00), Interval(1.05, 2.00), Interval(3.00, 4.00)]
    views = build_segmentation_views(
        raw,
        duration_sec=5.0,
        bridge_gap_ms=100,
        min_speech_ms=250,
        strict_speech_edge_ms=50,
        strict_nonspeech_edge_ms=200,
    )
    assert len(views["raw_speech"]) == 3
    assert len(views["primary_speech"]) == 2
    assert views["primary_speech"][0] == Interval(0.10, 2.00)
    assert abs(views["strict_speech"][0].start_sec - 0.15) < 1e-12
    assert abs(views["strict_speech"][0].end_sec - 1.95) < 1e-12
    assert len(views["strict_internal_nonspeech"]) == 1
    assert abs(views["strict_internal_nonspeech"][0].start_sec - 2.20) < 1e-12
    assert abs(views["strict_internal_nonspeech"][0].end_sec - 2.80) < 1e-12


def test_reading_segmentation_triage_preserves_original_hard_and_soft_rules():
    excluded = classify_reading_segmentation(
        {
            "speech_fraction": 0.01,
            "n_speech_segments": 1,
            "duration_sec": 30,
            "longest_internal_nonspeech_sec": 0,
            "rms_db_median": -30,
            "rms_db_std": 5,
        }
    )
    flagged = classify_reading_segmentation(
        {
            "speech_fraction": 0.6,
            "n_speech_segments": 26,
            "duration_sec": 30,
            "longest_internal_nonspeech_sec": 0,
            "rms_db_median": -30,
            "rms_db_std": 5,
        }
    )
    assert excluded["qc_status"] == "excluded"
    assert flagged["qc_status"] == "flagged"


def test_legacy_silero_frames_segments_and_labels_match_original_contract(tmp_path):
    sample_rate = 16000
    waveform = np.zeros(int(0.12 * sample_rate), dtype=np.float32)
    waveform[480:1440] = 0.25
    timestamps = [{"start": 480, "end": 1440}]

    frames, segments = legacy_silero_artifacts(
        waveform,
        sample_rate,
        timestamps,
        threshold=0.5,
        frame_ms=30,
    )
    assert list(frames.columns) == LEGACY_SILERO_FRAME_COLUMNS
    assert list(segments.columns) == LEGACY_SILERO_SEGMENT_COLUMNS
    assert frames["speech_vad_smooth"].tolist() == [False, True, True, False]
    assert frames["speech_vad_raw"].equals(frames["speech_vad_smooth"])
    assert frames["speech_mask_strict"].equals(frames["speech_vad_smooth"])
    assert frames["nonspeech_mask_strict"].equals(~frames["speech_vad_smooth"])
    assert segments["segment_role"].tolist() == [
        "leading_nonspeech",
        "speech",
        "trailing_nonspeech",
    ]
    assert np.allclose(segments["start_sec"], [0.00, 0.03, 0.09])
    assert np.allclose(segments["end_sec"], [0.03, 0.09, 0.12])

    summary = summarize_legacy_silero_artifacts(
        waveform,
        sample_rate,
        frames,
        segments,
        threshold=0.5,
        frame_ms=30,
        min_speech_ms=250,
        min_silence_ms=100,
        speech_pad_ms=50,
    )
    output = tmp_path / "example_silero.png"
    fig, axes = plot_segmentation_diagnostic(
        waveform,
        sample_rate,
        frames,
        segments,
        summary,
        file_name="example.wav | silero_vad",
        save_path=output,
    )
    assert output.exists()
    assert [axis.get_ylabel() for axis in axes] == [
        "Amplitude",
        "RMS dB",
        "Masks",
        "Segments",
    ]
    assert [tick.get_text() for tick in axes[3].get_yticklabels()] == [
        "non-speech",
        "speech",
    ]
    assert [text.get_text() for text in axes[3].get_legend().get_texts()] == [
        "leading_nonspeech",
        "speech",
        "trailing_nonspeech",
    ]
    assert fig._suptitle.get_text().startswith("example.wav | silero_vad")


def test_boundary_audit_separates_exact_samples_from_30ms_display_bins():
    sample_rate = 16000
    waveform = np.zeros(sample_rate, dtype=np.float32)
    exact = [Interval(0.305, 0.705)]
    waveform[int(0.305 * sample_rate) : int(0.705 * sample_rate)] = 0.2
    timestamps = [
        {
            "start": int(exact[0].start_sec * sample_rate),
            "end": int(exact[0].end_sec * sample_rate),
        }
    ]
    _, displayed = legacy_silero_artifacts(
        waveform,
        sample_rate,
        timestamps,
        threshold=0.5,
        frame_ms=30,
    )
    audit = boundary_alignment_diagnostics(
        waveform,
        sample_rate,
        exact,
        displayed_segments=displayed,
        window_ms=120,
        guard_ms=20,
        minimum_contrast_db=3,
    )
    assert len(audit) == 1
    assert abs(audit.loc[0, "start_sec"] - 0.305) < 1e-12
    assert abs(audit.loc[0, "end_sec"] - 0.705) < 1e-12
    assert abs(audit.loc[0, "display_onset_delta_ms"]) <= 30
    assert abs(audit.loc[0, "display_offset_delta_ms"]) <= 30
    assert audit.loc[0, "onset_contrast_db"] > 20
    assert audit.loc[0, "offset_contrast_db"] > 20
    assert not bool(audit.loc[0, "boundary_review_flag"])


def test_boundary_audit_flags_ambiguous_edges_without_moving_them():
    sample_rate = 16000
    waveform = np.full(sample_rate, 0.02, dtype=np.float32)
    exact = [Interval(0.30, 0.70)]
    audit = boundary_alignment_diagnostics(
        waveform,
        sample_rate,
        exact,
        window_ms=120,
        guard_ms=20,
        minimum_contrast_db=3,
    )
    assert bool(audit.loc[0, "ambiguous_onset"])
    assert bool(audit.loc[0, "ambiguous_offset"])
    assert bool(audit.loc[0, "boundary_review_flag"])
    assert audit.loc[0, "start_sec"] == 0.30
    assert audit.loc[0, "end_sec"] == 0.70


def test_segmentation_adjudication_blocks_blank_review_decisions():
    summary = pd.DataFrame(
        {
            "logical_recording_id": ["a", "b"],
            "file_name": ["a.wav", "b.wav"],
            "qc_status": ["accepted", "flagged"],
            "qc_flags": ["", "extreme_fragmentation"],
        }
    )
    template = segmentation_adjudication_template(summary)
    assert template.loc[template["file_name"].eq("a.wav"), "decision"].iloc[0] == "KEEP"
    with pytest.raises(ValueError, match="KEEP or EXCLUDE"):
        apply_segmentation_adjudication(summary, template)
    template.loc[template["file_name"].eq("b.wav"), "decision"] = "KEEP"
    template.loc[template["file_name"].eq("b.wav"), "boundary_source"] = "AUTO"
    template.loc[template["file_name"].eq("b.wav"), "reviewer"] = "Nev"
    template.loc[template["file_name"].eq("b.wav"), "review_date"] = "2026-07-23"
    frozen = apply_segmentation_adjudication(summary, template)
    assert frozen["segmentation_analysis_eligible"].all()
    assert (
        frozen.loc[frozen["file_name"].eq("b.wav"), "segmentation_decision_source"].iloc[0]
        == "reviewed_auto_boundaries"
    )


def test_task_not_completed_is_locked_automatic_exclusion():
    summary = pd.DataFrame(
        {
            "logical_recording_id": ["task-no", "task-yes"],
            "file_name": ["task-no.wav", "task-yes.wav"],
            "qc_status": ["accepted", "accepted"],
            "qc_flags": ["", ""],
            "Task Completed as Instructed": [" No ", "Yes"],
        }
    )
    template = segmentation_adjudication_template(summary)
    no_row = template.loc[template["file_name"].eq("task-no.wav")].iloc[0]
    assert bool(no_row["automatic_task_exclusion"])
    assert no_row["decision"] == "EXCLUDE"
    assert no_row["boundary_source"] == "NONE"
    assert "Task Completed as Instructed = NO" in no_row["notes"]
    assert not bool(no_row["review_required"])
    assert segmentation_pending_reviews(template).empty

    frozen = apply_segmentation_adjudication(summary, template)
    no_frozen = frozen.loc[frozen["file_name"].eq("task-no.wav")].iloc[0]
    assert not bool(no_frozen["segmentation_analysis_eligible"])
    assert (
        no_frozen["segmentation_decision_source"]
        == "automatic_task_not_performed"
    )

    edited = template.copy()
    edited.loc[edited["file_name"].eq("task-no.wav"), "decision"] = "KEEP"
    edited.loc[
        edited["file_name"].eq("task-no.wav"), "boundary_source"
    ] = "AUTO"
    assert len(segmentation_pending_reviews(edited)) == 1
    with pytest.raises(ValueError, match="locked to EXCLUDE"):
        apply_segmentation_adjudication(summary, edited)


def test_accepted_segmentation_outlier_enters_mandatory_review_queue():
    durations = np.linspace(20, 40, 21)
    durations[-1] = 300
    summary = pd.DataFrame(
        {
            "logical_recording_id": [f"id-{index}" for index in range(21)],
            "file_name": [f"{index}.wav" for index in range(21)],
            "qc_status": ["accepted"] * 21,
            "qc_flags": [""] * 21,
            "duration_sec": durations,
        }
    )
    selected = segmentation_review_selection(
        summary,
        {
            "robust_z_threshold": 4.5,
            "minimum_accepted_reference_n": 20,
            "accepted_outlier_features": ["duration_sec"],
            "accepted_guardrails": {},
        },
    )
    outlier = selected.iloc[-1]
    assert bool(outlier["accepted_outlier"])
    assert bool(outlier["review_required"])
    assert "accepted_robust_outlier:duration_sec" in outlier["review_reasons"]
    template = segmentation_adjudication_template(
        summary,
        {
            "robust_z_threshold": 4.5,
            "minimum_accepted_reference_n": 20,
            "accepted_outlier_features": ["duration_sec"],
            "accepted_guardrails": {},
        },
    )
    assert template.loc[template["file_name"].eq("20.wav"), "decision"].iloc[0] == ""


def test_manual_primary_boundaries_replace_only_primary_profile():
    summary = pd.DataFrame(
        {
            "logical_recording_id": ["a", "b"],
            "file_name": ["a.wav", "b.wav"],
            "qc_status": ["accepted", "flagged"],
            "qc_flags": ["", "extreme_fragmentation"],
            "duration_sec": [10.0, 10.0],
        }
    )
    review = segmentation_adjudication_template(summary)
    selected = review["file_name"].eq("b.wav")
    review.loc[selected, "decision"] = "KEEP"
    review.loc[selected, "boundary_source"] = "MANUAL"
    review.loc[selected, "reviewer"] = "Nev"
    review.loc[selected, "review_date"] = "2026-07-23"
    review.loc[selected, "notes"] = "Corrected VAD boundary after listening."
    decisions = apply_segmentation_adjudication(summary, review)

    interval_rows = []
    for logical_id, file_name in [("a", "a.wav"), ("b", "b.wav")]:
        for profile in ["primary", "conservative"]:
            for view in ["raw_speech", "primary_speech", "strict_speech"]:
                interval_rows.append(
                    {
                        "file_name": file_name,
                        "view": view,
                        "interval_index": 0,
                        "start_sec": 1.0,
                        "end_sec": 3.0,
                        "duration_sec": 2.0,
                        "profile": profile,
                        "logical_recording_id": logical_id,
                    }
                )
    automatic = pd.DataFrame(interval_rows)
    overrides = pd.DataFrame(
        {
            "logical_recording_id": ["b", "b"],
            "file_name": ["b.wav", "b.wav"],
            "segment_index": [0, 1],
            "start_sec": [2.0, 6.0],
            "end_sec": [4.0, 8.0],
            "reviewer": ["Nev", "Nev"],
            "review_date": ["2026-07-23", "2026-07-23"],
            "notes": [
                "Corrected VAD boundary after listening.",
                "Corrected VAD boundary after listening.",
            ],
        }
    )
    frozen = freeze_segmentation_intervals(automatic, decisions, overrides)
    manual_primary = frozen.loc[
        frozen["logical_recording_id"].eq("b")
        & frozen["profile"].eq("primary")
        & frozen["view"].eq("primary_speech")
    ]
    assert manual_primary["start_sec"].tolist() == [2.0, 6.0]
    assert manual_primary["segmentation_boundary_source"].eq("manual_override").all()
    conservative = frozen.loc[
        frozen["logical_recording_id"].eq("b")
        & frozen["profile"].eq("conservative")
        & frozen["view"].eq("primary_speech")
    ]
    assert conservative["start_sec"].tolist() == [1.0]
    assert conservative["segmentation_boundary_source"].eq("automatic_silero").all()


def test_notebook_review_save_is_atomic_and_validates_manual_intervals(tmp_path):
    summary = pd.DataFrame(
        {
            "logical_recording_id": ["b"],
            "file_name": ["b.wav"],
            "qc_status": ["flagged"],
            "qc_flags": ["extreme_fragmentation"],
            "duration_sec": [10.0],
        }
    )
    review = segmentation_adjudication_template(summary)
    review_path = tmp_path / "review.csv"
    overrides_path = tmp_path / "manual.csv"
    review.to_csv(review_path, index=False)
    saved_review, saved_overrides = save_segmentation_review_entry(
        review_path=review_path,
        overrides_path=overrides_path,
        logical_recording_id="b",
        duration_sec=10.0,
        decision="KEEP",
        boundary_source="MANUAL",
        reviewer="Nev",
        notes="Listened and corrected speech onset/offset.",
        manual_intervals_text="1.0,3.0\n4.0,8.5",
        review_date="2026-07-23",
    )
    assert saved_review.iloc[0]["boundary_source"] == "MANUAL"
    assert saved_overrides["segment_index"].astype(int).tolist() == [0, 1]
    assert parse_manual_intervals_text("1,2\n# note\n3,4") == [(1.0, 2.0), (3.0, 4.0)]
    with pytest.raises(ValueError, match="overlap"):
        save_segmentation_review_entry(
            review_path=review_path,
            overrides_path=overrides_path,
            logical_recording_id="b",
            duration_sec=10.0,
            decision="KEEP",
            boundary_source="MANUAL",
            reviewer="Nev",
            notes="Invalid test.",
            manual_intervals_text="1,4\n3,5",
            review_date="2026-07-23",
        )
