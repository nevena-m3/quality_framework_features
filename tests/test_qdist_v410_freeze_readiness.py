from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "paper1_qc_reviewed"
    / "qdist_v410_freeze_readiness.py"
)
if not MODULE_PATH.exists():
    MODULE_PATH = (
        Path(__file__).resolve().parents[1]
        / "src_reviewed"
        / "paper1_qc_reviewed"
        / "qdist_v410_freeze_readiness.py"
    )
spec = importlib.util.spec_from_file_location("qdist_v410_freeze_readiness", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


def test_true_runs_and_cross_channel_merging() -> None:
    mask = np.zeros((30, 2), dtype=bool)
    mask[2:5, 0] = True
    mask[8:10, 1] = True
    mask[22:25, 0] = True
    assert module.true_runs(mask[:, 0]) == [(2, 5), (22, 25)]
    assert module.merged_mask_episodes(mask, gap_samples=3) == [
        (2, 10),
        (22, 25),
    ]


def test_event_matching_is_one_to_one_and_reports_temporal_iou() -> None:
    result = module.match_events(
        [(0, 10), (20, 30)],
        [(0, 9), (19, 31), (50, 60)],
    )
    assert result["reference_event_count"] == 2
    assert result["predicted_event_count"] == 3
    assert result["matched_event_count"] == 2
    assert result["event_precision"] == 2 / 3
    assert result["event_recall"] == 1.0
    assert 0.8 < result["matched_iou_median"] < 1.0


def test_reference_frame_fraction_uses_only_complete_frames() -> None:
    truth = np.zeros((25, 2), dtype=bool)
    truth[2, 0] = True
    truth[12, 1] = True
    truth[24, 0] = True
    count, fraction = module.reference_frame_fraction(
        truth, frame_length_samples=10, complete_frame_count=2
    )
    assert count == 2
    assert fraction == 1.0


def _qualification_tables(*, event_pass: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_cell = pd.DataFrame(
        [
            {
                "geometry": geometry,
                "target_fraction": dose,
                "carrier_count": 12,
                "occurrence_sensitivity": 1.0,
                "sample_micro_precision": 0.99,
                "sample_micro_recall": 0.93,
                "event_micro_precision": 0.90 if event_pass else 0.50,
                "event_micro_recall": 0.90 if event_pass else 0.50,
                "event_f1_median": 0.90 if event_pass else 0.50,
                "matched_iou_median": 0.90 if event_pass else 0.30,
                "event_count_error_median": 0.0,
                "event_count_absolute_error_median": 0.0,
                "event_rate_error_per_min_median": 0.0,
                "frame_fraction_absolute_error_median": 0.001,
            }
            for geometry in ("negative_only", "positive_only", "symmetric")
            for dose in (0.0003, 0.001, 0.003, 0.01)
        ]
    )
    summary = pd.DataFrame(
        [
            {
                "challenge_rows": 144,
                "carrier_count": 12,
                "geometry_count": 3,
                "dose_count": 4,
                "sample_micro_precision": 0.999,
                "sample_micro_recall": 0.936,
                "reference_event_count": 100,
                "predicted_event_count": 100,
                "matched_event_count": 90 if event_pass else 50,
                "event_micro_precision": 0.90 if event_pass else 0.50,
                "event_micro_recall": 0.90 if event_pass else 0.50,
                "event_f1_median": 0.90 if event_pass else 0.50,
                "matched_iou_median": 0.90 if event_pass else 0.30,
                "event_rate_spearman": 0.90 if event_pass else 0.40,
                "event_count_absolute_error_median": 0.0,
                "frame_fraction_absolute_error_median": 0.001,
                "frame_fraction_absolute_error_max": 0.005,
            }
        ]
    )
    return by_cell, summary


def test_event_rate_is_retained_when_all_exact_reference_gates_pass() -> None:
    by_cell, summary = _qualification_tables(event_pass=True)
    checks, decisions, freeze_allowed = module.decide_roles(by_cell, summary)
    event = decisions.loc[
        decisions["feature"].eq("qdist_hard_clip_event_rate_per_min")
    ].iloc[0]
    assert freeze_allowed
    assert event["decision"] == "RETAIN"
    assert event["final_role"] == "SECONDARY"
    assert not checks["status"].eq("FAIL").any()


def test_failed_event_gate_demotes_event_rate_without_blocking_primary_freeze() -> None:
    by_cell, summary = _qualification_tables(event_pass=False)
    checks, decisions, freeze_allowed = module.decide_roles(by_cell, summary)
    event = decisions.loc[
        decisions["feature"].eq("qdist_hard_clip_event_rate_per_min")
    ].iloc[0]
    assert freeze_allowed
    assert event["decision"] == "RETAIN_CONDITIONALLY"
    assert event["final_role"] == "CONDITIONAL_AUDIT"
    event_check = checks.loc[
        checks["check"].eq("exact-reference event-rate qualification")
    ].iloc[0]
    assert event_check["status"] == "CONDITIONAL_DEMOTION"
    assert not checks["status"].eq("FAIL").any()


def test_failed_primary_sample_gate_blocks_measurement_freeze() -> None:
    by_cell, summary = _qualification_tables(event_pass=True)
    summary.loc[0, "sample_micro_recall"] = 0.50
    checks, decisions, freeze_allowed = module.decide_roles(by_cell, summary)
    assert not freeze_allowed
    assert checks["status"].eq("FAIL").any()
    sample_check = checks.loc[
        checks["check"].eq("exact-mask primary sample burden")
    ].iloc[0]
    assert sample_check["status"] == "FAIL"
    assert not decisions.empty


def test_checklist_cleanup_removes_stale_manual_review_requirement() -> None:
    source = pd.DataFrame(
        [
            {
                "item_id": item_id,
                "status": "PENDING",
                "evidence_path_notes": "old",
                "reviewer_note": "old",
            }
            for item_id in ("C4", "C5", "C6", "X2", "V2", "V3", "G10", "G11", "G12")
        ]
    )
    cleaned = module.clean_checklist(source, event_pass=True)
    v2 = cleaned.loc[cleaned["item_id"].eq("V2")].iloc[0]
    assert v2["status"] == "N/A"
    assert "No human or AI labels" in v2["reviewer_note"]
    assert not cleaned["reviewer_note"].str.contains(
        "two independent|required.*reviewer", case=False, regex=True
    ).any()
