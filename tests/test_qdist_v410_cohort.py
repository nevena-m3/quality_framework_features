from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from paper1_qc import qdist_v410_candidate as detector
from paper1_qc_reviewed import qdist_v410_cohort as cohort
from paper1_qc_reviewed import qdist_v410_panels as panels


def clipped_signal(fs: int = 16000, duration: float = 4.0) -> np.ndarray:
    time = np.arange(int(fs * duration)) / fs
    clean = .82 * np.sin(2 * np.pi * 211 * time) + .13 * np.sin(2 * np.pi * 431 * time)
    return np.clip(clean, -.62, .62)[:, None]


class QDISTV410CohortTests(unittest.TestCase):
    def test_import_has_no_soundfile_dependency(self) -> None:
        self.assertEqual(cohort.COHORT_VERSION, "qdist-v4.1.0-candidate-cohort-r1")

    def test_matched_clip_truth_mask_and_polarity(self) -> None:
        values = np.linspace(-1, 1, 10000)[:, None]
        for geometry in ["symmetric", "positive_only", "negative_only"]:
            altered, truth, metadata = cohort.inject_matched_hard_clip(values, .01, geometry)
            self.assertAlmostEqual(metadata["realized_fraction"], .01, delta=3 / values.size)
            self.assertTrue(np.array_equal(truth, altered != values))
            if geometry == "positive_only":
                self.assertFalse((truth & (values <= 0)).any())
            if geometry == "negative_only":
                self.assertFalse((truth & (values >= 0)).any())

    def test_truth_metrics(self) -> None:
        truth = np.array([[True], [True], [False], [False]])
        detected = np.array([[True], [False], [True], [False]])
        result = cohort._truth_metrics(truth, detected)
        self.assertEqual(result["true_positive_samples"], 1)
        self.assertEqual(result["false_positive_samples"], 1)
        self.assertEqual(result["false_negative_samples"], 1)
        self.assertAlmostEqual(result["sample_precision"], .5)
        self.assertAlmostEqual(result["sample_recall"], .5)

    def test_reconstruction_audit_matches_extractor(self) -> None:
        waveform = clipped_signal()
        extraction = detector.extract_qdist(waveform, 16000, logical_recording_id="P01_X_V1_20250101_A_B_C")
        recordings = pd.DataFrame([extraction.recording])
        long, summary = cohort.build_reconstruction_audit(
            recordings, extraction.accepted_plateau_ledger, extraction.episode_ledger
        )
        self.assertEqual(len(long), 3)
        self.assertTrue(summary["passed"].all())

    def test_prepare_recording_roles_and_valid_zero(self) -> None:
        waveform = np.sin(2 * np.pi * 220 * np.arange(64000) / 16000)[:, None] * .2
        extraction = detector.extract_qdist(waveform, 16000, logical_recording_id="P01_X_V1_20250101_A_B_C")
        table = cohort.prepare_recording_table(pd.DataFrame([extraction.recording]))
        self.assertTrue(bool(table.loc[0, "qdist_valid_zero"]))
        self.assertEqual(table.loc[0, "qdist_feature_role_primary"], "qdist_hard_clipped_sample_fraction")
        self.assertEqual(int(table.loc[0, "qdist_occurrence"]), 0)

    def test_legacy_comparison_labels_transitions(self) -> None:
        candidate = pd.DataFrame({
            "logical_recording_id": ["a", "b"],
            "qdist_positive": [False, True],
            "qdist_hard_clipped_frame_fraction": [0., .1],
            "qdist_hard_clip_event_rate_per_min": [0., 2.],
            "qdist_hard_clipped_sample_fraction": [0., .01],
        })
        legacy = {"analysis": pd.DataFrame({
            "logical_recording_id": ["a", "b"],
            "qdist_hard_clipped_frame_fraction": [0., 0.],
            "qdist_hard_clip_event_rate_per_min": [0., 0.],
            "qdist_hard_clipped_sample_fraction": [0., 0.],
        })}
        long, summary = cohort.legacy_comparison(candidate, legacy)
        self.assertEqual(set(long["occurrence_transition"]), {"zero_to_zero", "new_v410_positive"})
        self.assertEqual(summary.loc[summary["comparison"].eq("occurrence"), "new_v410_positive"].iloc[0], 1)

    def test_parameter_grid_is_unique_one_factor_at_a_time(self) -> None:
        variants = cohort.parameter_variants()
        names = [name for name, _, _ in variants]
        self.assertEqual(len(names), len(set(names)))
        base = detector.DEFAULT_PARAMETERS.to_dict()
        for name, _, parameters in variants[1:]:
            changed = [key for key, value in parameters.to_dict().items() if value != base[key]]
            self.assertEqual(len(changed), 1, name)

    def test_merge_gap_sensitivity_counts_occurrence(self) -> None:
        recordings = pd.DataFrame([{
            "logical_recording_id": "r", "participant_id": "p",
            "qdist_native_sample_rate_hz": 1000, "qdist_finite_exposure_sec": 10.0,
        }])
        accepted = pd.DataFrame({
            "logical_recording_id": ["r", "r"],
            "start_sample_task": [100, 125], "end_sample_task_exclusive": [110, 135],
        })
        long, summary = cohort.merge_gap_sensitivity(recordings, accepted)
        self.assertEqual(len(long), 4)
        self.assertTrue(summary["occurrence_agreement"].eq(1).all())
        counts = long.set_index("merge_gap_ms")["event_count"].to_dict()
        self.assertEqual(counts[10.0], 2)
        self.assertEqual(counts[20.0], 1)

    def test_participant_weighting(self) -> None:
        recordings = pd.DataFrame({
            "logical_recording_id": ["a", "b", "c"],
            "participant_id": ["p1", "p1", "p2"],
            "qdist_positive": [True, False, False],
            "qdist_hard_clipped_sample_fraction": [.1, 0, 0],
            "qdist_hard_clip_event_rate_per_min": [1, 0, 0],
            "qdist_hard_clipped_frame_fraction": [.2, 0, 0],
        })
        participant, summary = cohort.participant_weighting(recordings)
        self.assertEqual(len(participant), 2)
        self.assertAlmostEqual(summary.loc[summary["analysis_level"].eq("recording_weighted"), "positive_fraction"].iloc[0], 1 / 3)
        self.assertAlmostEqual(summary.loc[summary["analysis_level"].eq("participant_ever_positive"), "positive_fraction"].iloc[0], .5)

    def test_candidate_decisions_have_correct_roles(self) -> None:
        decisions = cohort.candidate_feature_decisions().set_index("feature")
        self.assertEqual(decisions.loc["qdist_hard_clipped_sample_fraction", "candidate_role"], "PRIMARY")
        self.assertEqual(decisions.loc["qdist_hard_clip_event_rate_per_min", "candidate_role"], "SECONDARY")
        self.assertTrue(decisions.loc["qdist_hard_clipped_frame_fraction", "candidate_role"].startswith("CONDITIONAL"))
        self.assertTrue(decisions["status"].str.startswith("PENDING").all())

    def test_checkpoint_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "checkpoint.pkl.gz"
            payload = {"signature": "abc", "recording": {"x": 1}, "table": pd.DataFrame({"a": [1, 2]})}
            cohort.write_checkpoint(path, payload)
            restored = cohort.read_checkpoint(path)
            self.assertEqual(restored["signature"], "abc")
            pd.testing.assert_frame_equal(restored["table"], payload["table"])

    def test_adjudication_requires_two_complete_humans(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            blind = root / "blind_review"
            restricted = blind / "restricted"
            restricted.mkdir(parents=True)
            index = pd.DataFrame({"blind_id": ["x", "y"]})
            key = pd.DataFrame({
                "blind_id": ["x", "y"],
                "review_stratum": ["accepted_plateau", "valid_zero_window"],
            })
            index.to_csv(blind / "qdist_v410_blind_review_index.csv", index=False)
            key.to_csv(restricted / "qdist_v410_blind_review_key.csv", index=False)
            for number, labels in [(1, ["DEFINITE_HARD_CLIP", "NOT_HARD_CLIP"]), (2, ["PROBABLE_HARD_CLIP", "NOT_HARD_CLIP"])]:
                review = pd.DataFrame({
                    "blind_id": ["x", "y"], "reviewer_id": [f"r{number}"] * 2,
                    "review_label": labels, "confidence_1_to_5": [5, 4],
                })
                review.to_csv(root / f"r{number}.csv", index=False)
            result = cohort.adjudicate_blind_review(root, root / "r1.csv", root / "r2.csv")
            self.assertEqual(result["summary"].loc[0, "binary_agreement"], 1.0)
            self.assertEqual(len(result["disagreements"]), 0)
            self.assertEqual(len(result["exact_disagreements"]), 1)
            adjudication = pd.DataFrame({
                "blind_id": ["x"], "adjudicator_id": ["a1"],
                "adjudicated_label": ["DEFINITE_HARD_CLIP"],
                "rationale": ["Plateau and bilateral transitions are visible."],
            })
            adjudication.to_csv(root / "adjudication.csv", index=False)
            completed = cohort.adjudicate_blind_review(
                root, root / "r1.csv", root / "r2.csv", root / "adjudication.csv"
            )
            self.assertTrue(completed["review_checks"].loc[
                completed["review_checks"]["check"].eq("all exact-label disagreements adjudicated"),
                "status",
            ].eq("PASS").all())

    def test_figure_bundle_has_six_artifacts_and_relative_index(self) -> None:
        import matplotlib.pyplot as plt
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            fig, ax = plt.subplots()
            ax.plot([0, 1], [0, 1])
            row = panels._save_bundle(
                root, "Z", "test", fig, pd.DataFrame({"x": [1]}),
                "Caption.", {"parameter_hash": "abc"},
            )
            for field in ["png", "svg", "pdf", "source_csv", "caption", "provenance"]:
                self.assertFalse(Path(row[field]).is_absolute())
                self.assertTrue((root / row[field]).exists())

    def test_publish_and_freeze_is_hard_blocked(self) -> None:
        with self.assertRaises(ValueError):
            cohort.run_candidate_cohort("does-not-exist", publish_and_freeze=True)


if __name__ == "__main__":
    unittest.main()
