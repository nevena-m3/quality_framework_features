import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from paper1_qc_reviewed import qdist_v400_cohort as cohort


def project_root() -> Path:
    value = os.environ.get("QDIST_PROJECT_ROOT")
    if value:
        return Path(value)
    here = Path.cwd().resolve()
    while here.parent != here:
        if (here / "src reviewed").exists() and (here / "MAIN outputs").exists():
            return here
        here = here.parent
    pytest.skip("QDIST project root not available")


def notebook_text() -> str:
    root = project_root()
    path = root / "notebooks reviewed" / "05_QDIST" / "05_nonlinear_distortion_QDIST_v4_0_0_REVIEWED_COHORT_SOURCE.ipynb"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join("".join(cell.get("source", [])) for cell in payload.get("cells", []))


def test_feature_registry_is_exact_and_scalar_free():
    assert cohort.ANALYSIS_FEATURES == (
        "qdist_hard_clipped_frame_fraction",
        "qdist_hard_clip_event_rate_per_min",
        "qdist_hard_clipped_sample_fraction",
    )
    assert "scalar" not in " ".join(cohort.ANALYSIS_FEATURES).lower()


def test_panel_contract_includes_event_verification():
    assert set(cohort.REQUIRED_MAIN_PANELS) == {
        "A", "B", "C", "D1", "D2", "D3", "E1", "E2", "E3", "F",
        "H1", "H2", "H3", "I", "J",
    }
    assert cohort.GALLERY_MINIMUM == 8


def test_event_review_requires_five_declared_views():
    assert set(cohort.EVENT_REVIEW_REQUIRED_VIEWS) == {
        "waveform", "pcm_derivative", "amplitude_distribution", "spectrogram", "audio_excerpt"
    }


@pytest.mark.parametrize(
    "recording_id,participant",
    [
        ("TIBD63_270_1_20221214_240_PSG_BAMBOO", "TIBD63"),
        ("CAPT0000019_272_3_20230927_240_PSG_BAMBOO", "CAPT0000019"),
    ],
)
def test_participant_derivation(recording_id, participant):
    assert cohort.derive_participant_id(recording_id) == participant


def test_acquisition_date_derivation():
    date = cohort.derive_acquisition_date("TIBD63_270_1_20221214_240_PSG_BAMBOO")
    assert str(date.date()) == "2022-12-14"


def test_interval_merging_is_half_open_and_gap_aware():
    intervals = [(0, 4), (4, 7), (9, 12)]
    assert cohort._merge_intervals(intervals, gap_samples=0) == [(0, 7), (9, 12)]
    assert cohort._merge_intervals(intervals, gap_samples=2) == [(0, 12)]


def test_independent_reconstruction_uses_union_and_episode_ledger():
    row = {
        "logical_recording_id": "R1",
        "qdist_frame_length_samples": 10,
        "qdist_complete_frame_count": 10,
        "qdist_finite_channel_sample_count": 200,
        "qdist_finite_exposure_sec": 2.0,
    }
    accepted = pd.DataFrame(
        [
            {"logical_recording_id": "R1", "channel_index": 0, "start_sample_task": 8, "end_sample_task_exclusive": 14},
            {"logical_recording_id": "R1", "channel_index": 0, "start_sample_task": 12, "end_sample_task_exclusive": 18},
            {"logical_recording_id": "R1", "channel_index": 1, "start_sample_task": 20, "end_sample_task_exclusive": 24},
        ]
    )
    episodes = pd.DataFrame([{"logical_recording_id": "R1"}, {"logical_recording_id": "R1"}])
    rebuilt = cohort.reconstruct_recording_features(row, accepted, episodes)
    assert rebuilt["qdist_reconstructed_channel_sample_count"] == 14
    assert rebuilt["qdist_reconstructed_affected_frame_count"] == 3
    assert rebuilt["qdist_hard_clipped_frame_fraction_reconstructed"] == pytest.approx(0.3)
    assert rebuilt["qdist_hard_clipped_sample_fraction_reconstructed"] == pytest.approx(0.07)
    assert rebuilt["qdist_hard_clip_event_rate_per_min_reconstructed"] == pytest.approx(60.0)


def test_wilson_interval_is_bounded():
    low, high = cohort._wilson_interval(6, 519)
    assert 0.0 <= low <= 6 / 519 <= high <= 1.0


def test_figure_index_rejects_missing_gallery():
    rows = []
    for panel in cohort.REQUIRED_MAIN_PANELS:
        rows.append({"panel": panel, "stem": panel, **{field: __file__ for field in ["png", "svg", "pdf", "source_csv", "caption", "provenance"]}})
    checks = cohort.verify_figure_index(pd.DataFrame(rows), minimum_gallery=8)
    gallery = checks.loc[checks["check"].eq("at least eight deterministic signal examples")].iloc[0]
    assert bool(gallery["passed"]) is False


def test_notebook_controls_are_safe_and_explicit():
    text = notebook_text()
    assert "RUN_COHORT_STANDARDIZATION = True" in text
    assert "RECOMPUTE_FEATURE_EXTRACTION = False" in text
    assert "BUILD_EVENT_REVIEW = True" in text
    assert "REBUILD_EVENT_REVIEW = False" in text
    assert 'SCIENTIFIC_REVIEW_DECISION = "PENDING"' in text
    assert "PUBLISH_AND_FREEZE = False" in text


def test_notebook_declares_completion_and_no_freeze():
    text = notebook_text()
    assert "QDIST v4.0.0 REVIEWED COHORT STANDARDIZATION COMPLETE" in text
    assert 'manifest["freeze_allowed"] is False' in text
    assert 'manifest["feature_values_recomputed"] is False' in text


def test_project_frozen_baseline_and_preflight_contracts():
    paths = cohort.CohortPaths.from_project_root(project_root())
    preflight_manifest, preflight_index = cohort.verify_preflight_bundle(paths.preflight_root)
    frozen_manifest, data = cohort.verify_frozen_baseline(paths.frozen_root)
    assert preflight_manifest["preflight_blocking_checks_pass"] is True
    assert set(preflight_index["panel"].astype(str)) == {"A", "B", "C"}
    assert frozen_manifest["measurement_version"] == "qdist-v3.1.1"
    assert len(data["recordings"]) == 519
    assert len(data["accepted"]) == 30
    assert len(data["episodes"]) == 15


def test_project_reconstruction_is_machine_precision_equivalent():
    paths = cohort.CohortPaths.from_project_root(project_root())
    _, data = cohort.verify_frozen_baseline(paths.frozen_root)
    _, summary = cohort.build_reconstruction_audit(data["recordings"], data["accepted"], data["episodes"])
    assert summary["recording_count"].eq(519).all()
    assert summary["maximum_absolute_difference"].le(2e-15).all()
    assert summary["passed"].all()


def test_project_event_review_contract_has_sixty_items():
    paths = cohort.CohortPaths.from_project_root(project_root())
    _, data = cohort.verify_frozen_baseline(paths.frozen_root)
    adjudicated, summary = cohort.adjudication_summary(data["gallery_index"], data["gallery_review"])
    assert len(adjudicated) == 60
    assert summary.set_index("stratum").loc["accepted_plateau", "hard_clip_positive_fraction"] == 1.0
    assert summary.set_index("stratum").loc["valid_zero", "hard_clip_positive_fraction"] == 0.0
    assert summary.set_index("stratum").loc["rejected_candidate", "hard_clip_positive_fraction"] <= 0.20
