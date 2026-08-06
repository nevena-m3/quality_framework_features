from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from paper1_qc_reviewed.qchan_v400 import (
    ANALYSIS_FEATURES,
    DEFAULT_PARAMETERS,
    TimeInterval,
    build_subject_balanced_loso_references,
    compute_reference_relative_features,
    extract_recording_spectrum,
    full_span_interval,
    lowpass_filter,
    synthetic_speech_like,
)
from paper1_qc_reviewed.qchan_v400_cohort import (
    CANONICAL_PROFILE,
    CANONICAL_STRICT_VIEW,
    COHORT_ORCHESTRATION_VERSION,
    as_bool,
    canonical_interval_contract,
    construct_reference_from_members,
    deterministic_stratified_sample,
    deterministic_gallery_selection,
    gallery_linked_view_source,
    REQUIRED_GALLERY_LINKED_VIEWS,
    empirical_feature_summary,
    eligible_reference_members,
    ltas_source_frame,
    model_interface_frame,
    pairwise_redundancy,
    participant_balanced_resampling,
    participant_balanced_summary,
    reference_inventory_frame,
    reference_robustness_grid,
    remove_global_dc,
    repeated_recording_persistence,
    resolve_media_path,
    save_recording_spectrum,
    load_recording_spectrum,
    save_reference_spectrum,
    load_reference_spectrum,
    status_missingness_summary,
    summarize_reference_robustness,
    support_availability_summary,
    task_stratum_series,
)

FS = DEFAULT_PARAMETERS.analysis_sample_rate_hz


def synthetic_bundle(subject_count: int = 8, recordings_per_subject: int = 2):
    waveform = synthetic_speech_like(duration_sec=5.0, sample_rate_hz=FS)
    spectra = {}
    rows = []
    for subject_index in range(subject_count):
        for recording_index in range(recordings_per_subject):
            recording_id = f"s{subject_index}_r{recording_index}"
            transformed = waveform
            if subject_index == subject_count - 1:
                transformed = lowpass_filter(waveform, FS, 4500.0)
            spectrum = extract_recording_spectrum(
                transformed,
                FS,
                strict_speech=full_span_interval(transformed, FS),
                logical_recording_id=recording_id,
                source_sample_rate_hz=48000,
            )
            spectra[recording_id] = spectrum
            rows.append(
                {
                    "logical_recording_id": recording_id,
                    "subject_id": f"s{subject_index}",
                    "task_stratum": "BAMBOO_PASSAGE",
                }
            )
    metadata = pd.DataFrame(rows)
    references = build_subject_balanced_loso_references(spectra, metadata)
    return waveform, spectra, metadata, references


def test_orchestration_identity_and_no_scalar():
    assert COHORT_ORCHESTRATION_VERSION == "qchan-v4.0.0-cohort-orchestration-v3"
    assert len(ANALYSIS_FEATURES) == 4
    assert not any("scalar" in feature.lower() for feature in ANALYSIS_FEATURES)


def test_global_dc_removal_is_exact():
    waveform = np.array([2.0, 4.0, 6.0])
    corrected, before, after = remove_global_dc(waveform)
    assert before == 4.0
    assert abs(after) < 1e-12
    assert np.allclose(corrected, [-2.0, 0.0, 2.0])


def test_canonical_strict_interval_contract():
    decisions = pd.DataFrame(
        {
            "logical_recording_id": ["a", "b"],
            "segmentation_analysis_eligible": [True, True],
        }
    )
    intervals = pd.DataFrame(
        {
            "logical_recording_id": ["a", "a", "b"],
            "view": [CANONICAL_STRICT_VIEW] * 3,
            "profile": [CANONICAL_PROFILE] * 3,
            "interval_index": [0, 1, 0],
            "start_sec": [0.0, 2.0, 0.0],
            "end_sec": [1.0, 3.0, 2.0],
            "decision": ["KEEP"] * 3,
            "segmentation_analysis_eligible": [True] * 3,
        }
    )
    strict, contract = canonical_interval_contract(decisions, intervals)
    assert len(strict) == 3
    assert contract["contract_pass"].all()
    assert strict["frozen_interval_id"].is_unique


def test_recording_spectrum_npz_roundtrip(tmp_path):
    waveform = synthetic_speech_like(duration_sec=4.0, sample_rate_hz=FS)
    spectrum = extract_recording_spectrum(
        waveform,
        FS,
        strict_speech=full_span_interval(waveform, FS),
        logical_recording_id="recording",
        source_sample_rate_hz=48000,
    )
    path = tmp_path / "spectrum.npz"
    save_recording_spectrum(spectrum, path)
    restored = load_recording_spectrum(path)
    assert restored.logical_recording_id == spectrum.logical_recording_id
    assert restored.status == spectrum.status
    assert np.array_equal(restored.frequencies_hz, spectrum.frequencies_hz)
    assert np.array_equal(
        restored.normalized_psd_per_hz, spectrum.normalized_psd_per_hz
    )


def test_reference_npz_roundtrip(tmp_path):
    _, spectra, metadata, references = synthetic_bundle()
    reference = references[metadata.iloc[0]["logical_recording_id"]]
    path = tmp_path / "reference.npz"
    save_reference_spectrum(reference, path)
    restored = load_reference_spectrum(path)
    assert restored.reference_key == reference.reference_key
    assert restored.member_subject_ids == reference.member_subject_ids
    assert np.array_equal(
        restored.normalized_psd_per_hz, reference.normalized_psd_per_hz
    )


def test_task_stratum_falls_back_to_declared_bamboo_task():
    frame = pd.DataFrame({"logical_recording_id": ["a", "b"]})
    series, source = task_stratum_series(frame)
    assert source == "constant:BAMBOO_PASSAGE"
    assert set(series.astype(str)) == {"BAMBOO_PASSAGE"}


def test_reference_inventory_confirms_target_exclusion():
    _, spectra, metadata, references = synthetic_bundle()
    inventory = reference_inventory_frame(references, metadata)
    assert inventory["target_subject_excluded"].all()
    assert inventory["reference_subject_count"].min() >= 5
    assert inventory["reference_recording_count"].min() >= 8


def test_recording_weighted_reference_is_explicit_alternative():
    waveform = synthetic_speech_like(duration_sec=5.0, sample_rate_hz=FS)
    spectra = {}
    rows = []
    # s0 has five strongly low-passed recordings; other subjects have one dry
    # recording. Subject balancing prevents s0 from dominating.
    for subject_index in range(6):
        count = 5 if subject_index == 0 else 1
        for recording_index in range(count):
            recording_id = f"s{subject_index}_r{recording_index}"
            local = (
                lowpass_filter(waveform, FS, 3500.0)
                if subject_index == 0
                else waveform
            )
            spectra[recording_id] = extract_recording_spectrum(
                local,
                FS,
                strict_speech=full_span_interval(local, FS),
                logical_recording_id=recording_id,
                source_sample_rate_hz=48000,
            )
            rows.append(
                {
                    "logical_recording_id": recording_id,
                    "subject_id": f"s{subject_index}",
                    "task_stratum": "BAMBOO_PASSAGE",
                }
            )
    metadata = pd.DataFrame(rows)
    members = metadata.loc[~metadata["subject_id"].eq("s5")]
    balanced = construct_reference_from_members(
        spectra,
        members,
        task_stratum="BAMBOO_PASSAGE",
        excluded_subject_id="s5",
        mode="subject_balanced",
    )
    weighted = construct_reference_from_members(
        spectra,
        members,
        task_stratum="BAMBOO_PASSAGE",
        excluded_subject_id="s5",
        mode="recording_weighted",
    )
    assert balanced.reference_sha256 != weighted.reference_sha256


def test_empirical_and_status_summaries_preserve_one_sided_zero_mass():
    frame = pd.DataFrame(
        {
            "qchan_ltas_distance_db": [1.0, 2.0, np.nan],
            "qchan_rolloff95_deficit_hz": [0.0, 100.0, np.nan],
            "qchan_highband_ratio_deficit": [0.0, 0.1, np.nan],
            "qchan_tilt_steepening_db_per_oct": [0.0, 0.2, np.nan],
            "qchan_support_tier": ["minimum", "high", "unavailable"],
        }
    )
    for feature in ANALYSIS_FEATURES:
        frame[f"{feature}_status"] = ["measured", "measured", "reference_unavailable"]
    summary = empirical_feature_summary(frame)
    rolloff = summary.loc[
        summary["feature"].eq("qchan_rolloff95_deficit_hz")
    ].iloc[0]
    assert rolloff["available_n"] == 2
    assert rolloff["zero_n"] == 1
    assert len(status_missingness_summary(frame)) >= 8
    assert len(support_availability_summary(frame)) == 16


def test_deterministic_stratified_sample_is_repeatable():
    frame = pd.DataFrame(
        {
            "logical_recording_id": [f"r{i}" for i in range(30)],
            "tier": ["a", "b", "c"] * 10,
        }
    )
    first = deterministic_stratified_sample(
        frame, maximum_rows=12, stratum_columns=["tier"]
    )
    second = deterministic_stratified_sample(
        frame, maximum_rows=12, stratum_columns=["tier"]
    )
    assert first["logical_recording_id"].tolist() == second["logical_recording_id"].tolist()
    assert len(first) == 12


def test_repeated_recording_persistence_and_redundancy():
    rows = []
    for subject_index in range(10):
        for repeat in range(2):
            rows.append(
                {
                    "logical_recording_id": f"s{subject_index}_r{repeat}",
                    "SubjectID": f"s{subject_index}",
                    "recording_date_analysis": f"2026-01-{1 + repeat:02d}",
                    "qchan_ltas_distance_db": subject_index + 0.1 * repeat,
                    "qchan_rolloff95_deficit_hz": 100 * subject_index + repeat,
                    "qchan_highband_ratio_deficit": 0.01 * subject_index,
                    "qchan_tilt_steepening_db_per_oct": 0.02 * subject_index,
                }
            )
    frame = pd.DataFrame(rows)
    persistence = repeated_recording_persistence(
        frame,
        subject_column="SubjectID",
        date_column="recording_date_analysis",
    )
    assert persistence["paired_subject_count"].eq(10).all()
    assert persistence["first_second_spearman"].min() > 0.99
    redundancy = pairwise_redundancy(frame, ANALYSIS_FEATURES)
    assert len(redundancy) == 6


def test_participant_balanced_summary_and_ml_contract():
    _, spectra, metadata, references = synthetic_bundle()
    rows = []
    for metadata_row in metadata.itertuples(index=False):
        result = compute_reference_relative_features(
            spectra[metadata_row.logical_recording_id],
            references[metadata_row.logical_recording_id],
        )
        result["SubjectID"] = metadata_row.subject_id
        result["recording_date_analysis"] = "2026-01-01"
        rows.append(result)
    frame = pd.DataFrame(rows)
    resampling = participant_balanced_resampling(
        frame,
        subject_column="SubjectID",
        iterations=20,
    )
    summary = participant_balanced_summary(resampling)
    assert len(summary) == 4
    interface = model_interface_frame(frame)
    for feature in ANALYSIS_FEATURES:
        assert feature in interface
        assert f"{feature}__available" in interface
        assert f"{feature}__status" in interface
        assert f"{feature}__missing_reason" in interface
    assert not interface["qchan_family_scalar_available"].any()
    assert not interface["qchan_standalone_reject_allowed"].any()


def test_reference_robustness_grid_contains_all_comparisons():
    _, spectra, metadata, references = synthetic_bundle(subject_count=8)
    target_id = metadata.iloc[0]["logical_recording_id"]
    targets = pd.DataFrame({"logical_recording_id": [target_id]})
    grid = reference_robustness_grid(
        targets,
        spectra=spectra,
        metadata=metadata,
        baseline_references=references,
        bootstrap_iterations=5,
        maximum_delete_subjects=3,
    )
    assert {
        "recording_weighted",
        "vintage80_1",
        "vintage80_2",
        "delete_one_reference_subject",
        "subject_bootstrap",
    }.issubset(set(grid["comparison"]))
    summary = summarize_reference_robustness(grid)
    assert len(summary) == 5 * 4


def test_ltas_source_frame_is_complete():
    _, spectra, metadata, references = synthetic_bundle()
    target_id = metadata.iloc[-1]["logical_recording_id"]
    source = ltas_source_frame(spectra[target_id], references[target_id])
    assert len(source) >= 10
    assert {
        "center_frequency_hz",
        "observation_ltas_db",
        "reference_ltas_db",
        "difference_db",
    }.issubset(source.columns)


def test_resolve_media_path_with_override(tmp_path):
    root = tmp_path / "Bamboo_passage_only"
    nested = root / "site" / "file.mp4"
    nested.parent.mkdir(parents=True)
    nested.write_bytes(b"x")
    raw = r"C:\old\Bamboo_passage_only\site\file.mp4"
    assert resolve_media_path(raw, media_root_override=root) == nested


def test_cohort_notebook_uses_resolvable_parameter_grid_and_explicit_schemas():
    notebook_path = (
        ROOT
        / "notebooks/02_feature_extraction"
        / "04_QCHAN"
        / "02_extract_cohort.ipynb"
    )
    text = notebook_path.read_text(encoding="utf-8")
    assert "octave_fraction_1" in text
    assert "octave_fraction_2" in text
    assert "octave_fraction_6" not in text
    assert "PARAMETER_SENSITIVITY_COLUMNS" in text
    assert "parameter_variant" in text



def _gallery_test_frame(recording_count: int = 20) -> pd.DataFrame:
    ids = [f"r{i:02d}" for i in range(recording_count)]
    frame = pd.DataFrame(
        {
            "logical_recording_id": ids,
            "qchan_family_status": ["measured"] * recording_count,
            "qchan_support_tier": (
                ["minimum", "moderate", "high", "high", "moderate"]
                * ((recording_count + 4) // 5)
            )[:recording_count],
            "qchan_source_bandwidth_limited": [
                index % 7 == 0 for index in range(recording_count)
            ],
            "qchan_ltas_distance_db": np.linspace(0.1, 10.0, recording_count),
            "qchan_rolloff95_deficit_hz": np.linspace(
                0.0, 1000.0, recording_count
            ),
            "qchan_highband_ratio_deficit": np.linspace(
                0.0, 0.4, recording_count
            ),
            "qchan_tilt_steepening_db_per_oct": np.linspace(
                0.0, 2.0, recording_count
            ),
            "qchan_rolloff95_signed_difference_hz": np.linspace(
                -1200.0, 1000.0, recording_count
            ),
        }
    )
    # Force the same recording to be the maximum for three related features.
    frame.loc[recording_count - 1, [
        "qchan_ltas_distance_db",
        "qchan_rolloff95_deficit_hz",
        "qchan_highband_ratio_deficit",
    ]] = [50.0, 5000.0, 1.0]
    return frame


def test_gallery_selection_recovers_unique_strata_when_extremes_overlap():
    frame = _gallery_test_frame()
    selection = deterministic_gallery_selection(
        frame,
        maximum_rows=10,
        minimum_rows=8,
    )
    assert len(selection) >= 8
    assert len(selection) <= 10
    assert selection["logical_recording_id"].is_unique
    assert selection["selection_reason"].is_unique
    assert "high_highband_deficit" in set(selection["selection_reason"])
    assert {
        "low_ltas_distance",
        "high_ltas_distance",
        "high_rolloff_deficit",
        "high_tilt_steepening",
        "median_measured_profile",
        "strong_upward_signed_rolloff_difference",
    }.issubset(set(selection["selection_reason"]))


def test_gallery_selection_is_deterministic_and_label_blind():
    frame = _gallery_test_frame()
    frame["diagnosis_analysis"] = ["ALS", "Control"] * (len(frame) // 2)
    frame["human_qc_label"] = np.arange(len(frame)) % 3
    first = deterministic_gallery_selection(frame)
    shuffled = deterministic_gallery_selection(
        frame.sample(frac=1.0, random_state=991).reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(
        first.reset_index(drop=True),
        shuffled.reset_index(drop=True),
    )


def test_gallery_selection_does_not_duplicate_undersized_inputs():
    frame = _gallery_test_frame(recording_count=5)
    selection = deterministic_gallery_selection(
        frame,
        maximum_rows=10,
        minimum_rows=8,
    )
    assert len(selection) == 5
    assert selection["logical_recording_id"].is_unique


def test_cohort_notebook_uses_governed_unique_gallery_selector():
    notebook_path = (
        ROOT
        / "notebooks/02_feature_extraction"
        / "04_QCHAN"
        / "02_extract_cohort.ipynb"
    )
    text = notebook_path.read_text(encoding="utf-8")
    assert "deterministic_gallery_selection" in text
    assert "unique_reason_prioritized_deterministic_fill_v1" in text
    assert "gallery recording identities are unique" in text


def test_gallery_source_declares_all_five_linked_views_when_ltas_available():
    waveform = pd.DataFrame(
        {
            "row_type": ["waveform", "waveform"],
            "logical_recording_id": ["r1", "r1"],
            "selection_reason": ["low_ltas_distance", "low_ltas_distance"],
            "time_sec": [0.0, 0.01],
            "amplitude": [0.1, -0.1],
        }
    )
    spectrogram = pd.DataFrame(
        {
            "row_type": ["spectrogram"],
            "logical_recording_id": ["r1"],
            "selection_reason": ["low_ltas_distance"],
            "time_sec": [0.0],
            "frequency_hz": [1000.0],
            "power_db": [-20.0],
        }
    )
    ltas = pd.DataFrame(
        {
            "row_type": ["ltas", "ltas"],
            "logical_recording_id": ["r1", "r1"],
            "selection_reason": ["low_ltas_distance", "low_ltas_distance"],
            "frequency_hz": [250.0, 500.0],
            "observation_ltas_db": [-4.0, -6.0],
            "reference_ltas_db": [-5.0, -5.5],
            "difference_db": [1.0, -0.5],
        }
    )
    features = pd.DataFrame(
        {
            "row_type": ["features"],
            "logical_recording_id": ["r1"],
            "selection_reason": ["low_ltas_distance"],
        }
    )

    source = gallery_linked_view_source(
        waveform,
        spectrogram,
        ltas,
        features,
    )
    assert set(REQUIRED_GALLERY_LINKED_VIEWS).issubset(
        set(source["view"].astype(str))
    )
    for view in ("target_ltas", "reference_ltas", "ltas_difference"):
        local = source.loc[source["view"] == view]
        assert len(local) == 2
        assert local["view_available"].astype(bool).all()
        assert pd.to_numeric(local["linked_value"], errors="coerce").notna().all()


def test_gallery_source_retains_explicit_unavailable_ltas_views():
    waveform = pd.DataFrame(
        {
            "row_type": ["waveform"],
            "logical_recording_id": ["r2"],
            "selection_reason": ["unavailable_state"],
            "time_sec": [0.0],
            "amplitude": [0.0],
        }
    )
    spectrogram = pd.DataFrame(
        {
            "row_type": ["spectrogram"],
            "logical_recording_id": ["r2"],
            "selection_reason": ["unavailable_state"],
            "time_sec": [0.0],
            "frequency_hz": [1000.0],
            "power_db": [-30.0],
        }
    )
    features = pd.DataFrame(
        {
            "row_type": ["features"],
            "logical_recording_id": ["r2"],
            "selection_reason": ["unavailable_state"],
        }
    )

    source = gallery_linked_view_source(
        waveform,
        spectrogram,
        pd.DataFrame(),
        features,
        unavailable_reason="insufficient_reference_support",
    )
    assert set(REQUIRED_GALLERY_LINKED_VIEWS).issubset(
        set(source["view"].astype(str))
    )
    unavailable = source.loc[
        source["view"].isin(
            ["target_ltas", "reference_ltas", "ltas_difference"]
        )
    ]
    assert len(unavailable) == 3
    assert not unavailable["view_available"].astype(bool).any()
    assert set(unavailable["view_unavailable_reason"]) == {
        "insufficient_reference_support"
    }


def test_cohort_notebook_uses_explicit_gallery_linked_view_contract():
    notebook_path = (
        ROOT
        / "notebooks/02_feature_extraction"
        / "04_QCHAN"
        / "02_extract_cohort.ipynb"
    )
    text = notebook_path.read_text(encoding="utf-8")
    assert "gallery_linked_view_source" in text
    assert "five_explicit_linked_views_v1" in text
    assert "gallery-linked-source-r3" in text
