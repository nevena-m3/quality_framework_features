import numpy as np
import pandas as pd

from paper1_qc.statistics import (
    compare_binary_label_systems,
    direction_oriented_family_indices,
    family_alignment_matrix,
    matched_family_specificity,
    rater_stratified_family_alignment,
)


def _feature_frame(n=30):
    x = np.linspace(0.0, 1.0, n)
    return pd.DataFrame(
        {
            "file_name": [f"f{i}.wav" for i in range(n)],
            "SubjectID": [f"s{i // 2}" for i in range(n)],
            "qadd_nonspeech_level_dbfs": -60 + 30 * x,
            # lower is worse; the reverse trend should be oriented to higher burden
            "qadd_snr_proxy_db": 30 - 20 * x,
            "qadd_nonspeech_variability_db": 10 * x,
            "qadd_hum_prominence_db": 5 * x,
            "qadd_transient_rate_per_min": 3 * x,
        }
    )


def test_family_index_respects_mixed_metric_directions():
    frame = _feature_frame()
    indices, audit = direction_oriented_family_indices(frame)
    score = indices["qfamily__additive_interference"]
    assert score.corr(pd.Series(np.linspace(0, 1, len(score))), method="spearman") > 0.99
    snr = audit.loc[audit["feature"] == "qadd_snr_proxy_db"].iloc[0]
    assert snr["orientation_applied"] == "reverse_percentile"


def test_family_alignment_and_paired_label_system_comparison():
    frame = _feature_frame()
    indices, _ = direction_oriented_family_indices(frame)
    labels_a = pd.DataFrame(
        {
            "file_name": frame["file_name"],
            "category": "additive_interference",
            "consensus_rating": [0] * 15 + [1] * 15,
        }
    )
    labels_b = labels_a.copy()
    labels_b["consensus_rating"] = [0, 1] * 15
    matrix = family_alignment_matrix(
        indices,
        labels_a,
        label_system="four_ra",
        bootstrap_replicates=20,
    )
    matched = matrix.loc[matrix["matched_family"]].iloc[0]
    assert matched["roc_auc"] > 0.95
    specificity = matched_family_specificity(
        indices,
        labels_a,
        label_system="four_ra",
        bootstrap_replicates=20,
    )
    # Only one objective/human family is present, so a matched-vs-mismatched
    # specificity contrast is correctly blocked rather than invented.
    assert specificity.loc[0, "status"] == "under_supported_specificity_estimand"
    comparison = compare_binary_label_systems(
        indices,
        labels_a,
        labels_b,
        label_a_name="four_ra",
        label_b_name="two_ra",
        shared_families=["additive_interference"],
        bootstrap_replicates=20,
    )
    assert comparison.loc[0, "delta_auc_a_minus_b"] > 0


def test_distributed_alignment_is_estimated_within_rater():
    frame = _feature_frame(n=40)
    indices, _ = direction_oriented_family_indices(frame)
    labels = pd.DataFrame(
        {
            "file_name": frame["file_name"],
            "rater_id": ["r1"] * 20 + ["r2"] * 20,
            "category": "additive_interference",
            "rating": ([0] * 10 + [1] * 10) * 2,
        }
    )
    result = rater_stratified_family_alignment(
        indices,
        labels,
        label_system="distributed",
        minimum_class_recordings_per_rater=3,
        bootstrap_replicates=30,
    )
    matched = result.loc[result["matched_family"]].iloc[0]
    assert matched["raters_estimable"] == 2
    assert matched["effect"] > 0.95
    assert matched["estimable"]
