from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
for folder in (PROJECT / "src", PROJECT / "src"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from paper1_qc_reviewed import qdist_v410_computational_verification as verification


def test_wilson_interval_is_bounded_and_non_degenerate() -> None:
    low, high = verification.wilson_interval(12, 12)
    assert 0 < low < 1
    assert high == 1


def test_feature_decisions_are_complete_and_do_not_claim_full_nonlinear_distortion() -> None:
    decisions = verification.final_feature_decisions()
    assert set(decisions["decision"]) == {
        "RETAIN", "RETAIN_CONDITIONALLY", "RETAIN_AS_STATUS"
    }
    assert decisions["rationale"].str.strip().ne("").all()
    prohibited = " ".join(decisions["prohibited_interpretation"].astype(str)).lower()
    assert "complete nonlinear distortion" in prohibited
    assert "independent biomarker" in prohibited
    assert "recording acceptability" in prohibited


def test_exact_mask_challenge_audit_reports_uncertainty_and_micro_metrics() -> None:
    rows = []
    for geometry in ("negative_only", "positive_only", "symmetric"):
        for dose in (.0003, .001, .003, .01):
            for carrier in range(12):
                rows.append({
                    "logical_recording_id": f"r{carrier}",
                    "geometry": geometry,
                    "target_fraction": dose,
                    "occurrence_detected": True,
                    "true_positive_samples": 90,
                    "false_positive_samples": 1,
                    "false_negative_samples": 10,
                    "sample_precision": 90 / 91,
                    "sample_recall": .9,
                    "sample_f1": 2 * (90 / 91) * .9 / ((90 / 91) + .9),
                })
    audit, summary = verification.build_challenge_audit(pd.DataFrame(rows))
    assert len(audit) == 12
    assert audit["carrier_count"].eq(12).all()
    assert audit["occurrence_wilson95_low"].between(0, 1).all()
    assert np.isclose(float(summary.iloc[0]["micro_precision"]), 90 / 91)
    assert np.isclose(float(summary.iloc[0]["micro_recall"]), .9)


def test_morphology_audit_never_creates_human_labels() -> None:
    decisions = verification.final_feature_decisions()
    joined = " ".join(decisions.astype(str).to_numpy().ravel()).lower()
    assert "human-confirmed" not in joined
    assert "complete nonlinear distortion" in joined
    assert not any("review_label" in column for column in decisions.columns)
