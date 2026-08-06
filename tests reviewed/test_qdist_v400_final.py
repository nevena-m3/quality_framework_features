from __future__ import annotations

import json
import os
from pathlib import Path

import nbformat
import pandas as pd

from paper1_qc_reviewed import qdist_v400_final as final


def candidate_root() -> Path:
    override = os.environ.get("QDIST_TEST_CANDIDATE_ROOT")
    if override:
        return Path(override)
    project = Path(__file__).resolve().parents[1]
    return project / "outputs reviewed" / "nonlinear_distortion" / "qdist-v4.0.0-candidate"


def test_identity_and_acceptance_token():
    assert final.FINAL_MEASUREMENT_VERSION == "qdist-v4.0.0"
    assert final.SOURCE_MEASUREMENT_VERSION == "qdist-v4.0.0-candidate"
    assert final.ACCEPTANCE_TOKEN == "ACCEPT_QDIST_V400"


def test_exact_three_feature_registry_and_roles():
    frame = final.final_decisions_frame()
    assert frame["feature"].tolist() == list(final.ANALYSIS_FEATURES)
    assert frame["final_decision"].tolist() == [
        "RETAIN_PRIMARY_RELATED_VIEW",
        "RETAIN_PRIMARY_EVENT_WITH_UNCERTAINTY",
        "RETAIN_SECONDARY_RELATED_VIEW",
    ]


def test_no_scalar_or_standalone_gate():
    frame = final.final_decisions_frame()
    assert not frame["family_scalar_allowed"].astype(bool).any()
    assert not frame["standalone_gate_allowed"].astype(bool).any()


def test_joint_model_policy_is_feature_specific():
    frame = final.final_decisions_frame().set_index("feature")
    assert bool(frame.loc[final.ANALYSIS_FEATURES[0], "default_joint_model_inclusion"])
    assert bool(frame.loc[final.ANALYSIS_FEATURES[1], "default_joint_model_inclusion"])
    assert not bool(frame.loc[final.ANALYSIS_FEATURES[2], "default_joint_model_inclusion"])


def test_ten_domain_dashboard_complete():
    frame = final.ten_domain_dashboard_frame()
    assert len(frame) == 10
    assert frame["domain"].nunique() == 10


def test_gate_summary_complete():
    frame = final.final_gate_summary_frame()
    assert frame["gate"].tolist() == [f"G{i}" for i in range(1, 11)]
    assert frame.loc[frame.gate.eq("G10"), "final_status"].item() == "PASS"


def test_interval_helpers():
    lo, hi = final.clopper_pearson(0, 10)
    assert lo == 0 and 0 < hi < 1
    lo, hi = final.wilson_interval(6, 519)
    assert lo < 6/519 < hi


def test_analysis_equality_tolerance():
    root = candidate_root()
    frame = pd.read_csv(root / "tables" / "qdist_v400_recording_features.csv")
    assert final.analysis_values_equal(frame, frame.copy())
    changed = frame.copy()
    changed.loc[0, final.ANALYSIS_FEATURES[0]] += 1e-6
    assert not final.analysis_values_equal(frame, changed)


def test_candidate_manifest_contract():
    root = candidate_root()
    manifest = json.loads((root / "manifests" / "qdist_v400_cohort_candidate_manifest.json").read_text(encoding="utf-8"))
    assert manifest["cohort_evidence_complete"] is True
    assert manifest["recording_count"] == 519
    assert manifest["positive_recording_count"] == 6
    assert manifest["event_review_item_count"] == 60
    assert manifest["figure_bundle_count"] == 23
    assert manifest["feature_values_recomputed"] is False


def test_event_review_five_view_contract():
    root = candidate_root()
    index = pd.read_csv(root / "tables" / "qdist_v400_event_review_index.csv")
    assert len(index) == 60
    required = {"waveform", "pcm_derivative", "amplitude_distribution", "spectrogram", "audio_excerpt"}
    for row in index.itertuples(index=False):
        source = pd.read_csv(root / "event_review" / f"qdist_review_{row.review_item_id}.source.csv")
        assert required.issubset(set(source["view"].astype(str)))


def test_ai_assisted_review_is_explicit():
    root = candidate_root()
    reviewer = pd.read_csv(root / "tables" / "qdist_v400_event_review_index.csv")
    joined = " ".join(reviewer["reviewer"].astype(str).fillna("").tolist()).lower()
    assert "ai" in joined or "gpt" in joined


def test_all_cohort_checks_pass():
    checks = pd.read_csv(candidate_root() / "validation" / "qdist_v400_cohort_checks.csv")
    assert checks["passed"].astype(bool).all()


def test_finalization_notebook_controls_safe():
    project = Path(__file__).resolve().parents[1]
    path = project / "notebooks reviewed" / "05_QDIST" / "05_nonlinear_distortion_QDIST_v4_0_0_FINALIZATION_SOURCE.ipynb"
    nb = nbformat.read(path, as_version=4)
    text = "\n".join("".join(cell.source) for cell in nb.cells)
    assert 'SCIENTIFIC_REVIEW_DECISION = "PENDING"' in text
    assert "PUBLISH_AND_FREEZE = False" in text
    assert "QDIST v4.0.0 FINALIZATION COMPLETE" in text
