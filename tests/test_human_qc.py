import json

import numpy as np
import pandas as pd

from paper1_qc.human_qc import (
    agreement_summary,
    gwet_ac1,
    load_interval_human_qc,
    make_consensus,
    rating_design_coverage,
)


def test_gwet_ac1_perfect_agreement():
    matrix = pd.DataFrame({"r1": [0, 1, 1], "r2": [0, 1, 1], "r3": [0, 1, 1]})
    assert np.isclose(gwet_ac1(matrix), 1.0)


def test_consensus_tie_requires_adjudication():
    ratings = pd.DataFrame(
        {
            "file_name": ["a.wav", "a.wav"],
            "rater_id": ["r1", "r2"],
            "category": ["noise", "noise"],
            "rating": [0, 1],
        }
    )
    consensus = make_consensus(ratings)
    assert consensus.loc[0, "requires_adjudication"]
    assert pd.isna(consensus.loc[0, "consensus_rating"])


def test_agreement_summary_keeps_reliability_distinct_from_consensus():
    ratings = pd.DataFrame(
        [
            {"file_name": f"f{i}.wav", "rater_id": rater, "category": "noise", "rating": value}
            for i, triplet in enumerate([(0, 0, 0), (1, 1, 1), (0, 0, 1), (1, 1, 1)])
            for rater, value in zip(["r1", "r2", "r3"], triplet)
        ]
    )
    result = agreement_summary(ratings)
    assert result.loc[0, "raters_total"] == 3
    assert result.loc[0, "items_complete_all_raters"] == 4
    assert result.loc[0, "observed_pair_agreement"] > 0.7


def test_interval_gui_parser_maps_perceptual_families_and_unions_overlap(tmp_path):
    rater = tmp_path / "RA01"
    rater.mkdir()
    source = rater / "sample_segments.csv"
    empty_context = {
        "Coughing": [],
        "Extra or filler word": [],
    }
    row = {
        "file_name": "sample.wav",
        "onset_seconds_absolute": 0.0,
        "offset_seconds_absolute": 5.0,
        "duration_seconds": 5.0,
        "Environmental noise": json.dumps(
            {
                "HVAC": [[0.0, 3.0]],
                "Traffic": [[2.0, 4.0]],
            }
        ),
        "Any non-task related content": json.dumps(empty_context),
        "Competing speech": json.dumps({"Other human speakers": []}),
        "Volume unstable": json.dumps({"Volume changes": []}),
        "Clipping": json.dumps({"Crackling on loud syllables": []}),
        "Reverberation/echo": json.dumps({"Echo": [], "Reverb": []}),
        "Platform effects": json.dumps({"Muffled": []}),
        "Temporal discontinuities": json.dumps({"Audio glitching": []}),
    }
    pd.DataFrame([row]).to_csv(source, index=False)

    ratings, context, intervals, issues = load_interval_human_qc(
        tmp_path, rater_strategy="parent_directory"
    )
    additive = ratings.loc[ratings["category"] == "additive_interference"].iloc[0]
    assert additive["rater_id"] == "RA01"
    assert additive["rating"] == 1
    assert np.isclose(additive["annotated_duration_sec"], 4.0)
    assert np.isclose(additive["annotated_fraction"], 0.8)
    assert len(intervals) == 2
    assert context["rating"].sum() == 0
    assert not issues["issue"].eq("rater_identity_unresolved").any()


def test_named_ra_directories_exclude_crossed_reliability_subfolder(tmp_path):
    row = {
        "file_name": "sample.wav",
        "onset_seconds_absolute": 0.0,
        "offset_seconds_absolute": 5.0,
        "duration_seconds": 5.0,
        "Environmental noise": json.dumps({"HVAC": []}),
        "Any non-task related content": json.dumps({"Coughing": []}),
        "Competing speech": json.dumps({"Other human speakers": []}),
        "Volume unstable": json.dumps({"Volume changes": []}),
        "Clipping": json.dumps({"Crackling on loud syllables": []}),
        "Reverberation/echo": json.dumps({"Echo": [], "Reverb": []}),
        "Platform effects": json.dumps({"Muffled": []}),
        "Temporal discontinuities": json.dumps({"Audio glitching": []}),
    }
    for relative in [
        ("Abbas", "main_segments.csv"),
        ("Reliability", "Abbas", "reliability_segments.csv"),
    ]:
        folder = tmp_path.joinpath(*relative[:-1])
        folder.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([row]).to_csv(folder / relative[-1], index=False)

    main, _, _, issues = load_interval_human_qc(
        tmp_path,
        rater_strategy="parent_directory",
        rater_directory_names=["Abbas", "Liya", "Samaana", "Samara"],
        exclude_path_parts=["Reliability"],
    )
    reliability, _, _, reliability_issues = load_interval_human_qc(
        tmp_path / "Reliability",
        rater_strategy="parent_directory",
        rater_directory_names=["Abbas", "Liya", "Samaana", "Samara"],
    )

    assert main["source_file"].nunique() == 1
    assert reliability["source_file"].nunique() == 1
    assert set(main["rater_id"]) == {"Abbas"}
    assert set(reliability["rater_id"]) == {"Abbas"}
    assert issues.empty
    assert reliability_issues.empty


def test_four_ra_coverage_and_primary_consensus_gate():
    ratings = pd.DataFrame(
        [
            {
                "file_name": "a.wav",
                "rater_id": rater,
                "category": "additive_interference",
                "rating": value,
            }
            for rater, value in zip(["r1", "r2", "r3"], [1, 1, 0])
        ]
    )
    _, summary = rating_design_coverage(ratings, expected_raters=4)
    assert summary.loc[0, "design_status"] == "blocked_no_item_has_all_expected_raters"
    consensus = make_consensus(ratings, expected_raters=4, minimum_ratings=4)
    assert consensus.loc[0, "consensus_method"] == "insufficient_rater_coverage"
    assert pd.isna(consensus.loc[0, "consensus_rating"])
