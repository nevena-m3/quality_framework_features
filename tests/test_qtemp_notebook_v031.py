from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = (
    ROOT
    / "notebooks"
    / "02_feature_extraction"
    / "02f_temporal_discontinuity_QTEMP_v0_3_1_FINALIZATION_SOURCE.ipynb"
)
GENERATOR = ROOT / "scripts" / "generate_qtemp_v031_notebook.py"


def load_generator():
    spec = importlib.util.spec_from_file_location(
        "generate_qtemp_v031_notebook",
        GENERATOR,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def notebook():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def all_source():
    return "\n".join(
        "".join(cell.get("source", []))
        if isinstance(cell.get("source", []), list)
        else str(cell.get("source", ""))
        for cell in notebook()["cells"]
    )


def test_generator_reproduces_committed_source_notebook():
    assert load_generator().build_notebook() == notebook()


def test_source_notebook_is_clean_and_unexecuted():
    for cell in notebook()["cells"]:
        if cell["cell_type"] == "code":
            assert cell.get("execution_count") is None
            assert cell.get("outputs", []) == []


def test_source_and_executed_review_notebooks_are_separate():
    source = all_source()
    assert "FINALIZATION_SOURCE.ipynb" in source
    assert "FINALIZATION_EXECUTED_REVIEW.ipynb" in source
    assert "SOURCE_NOTEBOOK" in source
    assert "EXECUTED_NOTEBOOK" in source


def test_finalization_retains_four_features_and_drops_splice():
    source = all_source()
    assert "RETAINED_ANALYSIS_FEATURES" in source
    assert "qtemp_dropout_duration_fraction" in source
    assert "qtemp_dropout_event_rate_per_min" in source
    assert "qtemp_frozen_audio_duration_fraction" in source
    assert "qtemp_frozen_audio_event_rate_per_min" in source
    assert "DROP_FAILED_ANALYTICAL_VALIDATION" in source
    assert "qtemp_splice_discontinuity_rate_per_min" in source


def test_final_duplicate_scope_is_explicit_and_supported():
    source = all_source()
    assert "FINAL_DUPLICATE_MIN_DURATION_MS = 40.0" in source
    assert "FINAL_DUPLICATE_PARAMETERS" in source
    assert "duplicate_min_sequence_ms=FINAL_DUPLICATE_MIN_DURATION_MS" in source
    assert "final_duplicate_min_duration_ms" in source


def test_runtime_uses_non_shadowable_time_alias():
    source = all_source()
    assert "import time as pytime" in source
    assert "pytime.perf_counter()" in source
    assert re.search(r"(?<!py)time\.perf_counter\(", source) is None


def test_subject_identity_resolution_is_case_insensitive_and_audited():
    source = all_source()
    assert "SubjectID" in source
    assert "_case_insensitive_columns" in source
    assert "qtemp_subject_id_source" in source
    assert "subject_identity_audit" in source


def test_parameter_sensitivity_is_retained_detector_only_and_bounded():
    source = all_source()
    assert "SENSITIVITY_WORKERS" in source
    assert "MAX_SENSITIVITY_RECORDINGS = 12" in source
    assert '"dropout": {' in source
    assert '"frozen_audio": {' in source
    assert "enabled_event_types=(detector,)" in source
    assert "RUN_DROPPED_SPLICE_AUDIT = False" in source


def test_blinded_review_is_mandatory_and_has_negative_controls():
    source = all_source()
    assert "PENDING_BLINDED_ADJUDICATION" in source
    assert "candidate_free" in source
    assert "hard_negative" in source
    assert "ACCEPT_QTEMP_V1" in source
    assert "accepted_event_observable_yes_fraction" in source


def test_freeze_is_review_gated_non_overwriting_and_retained_only():
    source = all_source()
    assert "PUBLISH_AND_FREEZE_QTEMP_V1 = False" in source
    assert "Refusing to overwrite immutable" in source
    assert "G1–G9" in source
    assert "qtemp-v1.0.0" in source
    assert "RETAINED_ANALYSIS_FEATURES" in source
    assert "qtemp_v10_features.csv" in source


def test_finalization_reconstructs_after_filtering_development_ledgers():
    source = all_source()
    assert "Reconstruct the retained recording-level outputs" in source
    assert "_filter_short_duplicate_rows" in source
    assert "dropped_splice_audit_summary" in source
    assert "duplicate_scope_audit" in source


def test_no_auto_freeze_or_clinical_label_threshold_tuning():
    source = all_source()
    assert "PUBLISH_AND_FREEZE_QTEMP_V1 = False" in source
    assert "clinical" not in "\n".join(
        line.lower()
        for line in source.splitlines()
        if "threshold" in line.lower()
    )


def test_registry_discloses_same_ledger_redundancy_for_both_event_rates():
    source = all_source()
    assert "same-ledger frequency view of near-exact consecutive decoded-waveform" in source
    assert "same-ledger event-frequency view" in source
    assert "same-ledger redundancy is disclosed" in source


def test_nonretained_splice_dependencies_are_resolved_not_silently_failed():
    source = all_source()
    assert "QDIST accepted-event arbitration requirement is resolved" in source
    assert '"splice" not in RETAINED_EVENT_TYPES' in source
    assert "failed held-out source-replacement hypothesis is explicitly" in source
    assert "final disposition=dropped" in source


def test_identifier_comparisons_are_index_aligned():
    source = all_source()
    assert "subject_disagreement = pd.Series" in source
    assert "inferred_disagreement = pd.Series" in source
    assert "subject_disagreement.loc[comparable_subject]" in source
    assert "inferred_disagreement.loc[explicit_and_inferred]" in source


def test_real_speech_regex_and_gallery_spectrogram_are_warning_safe():
    source = all_source()
    assert r'dropout_(?:zero|constant)' in source
    assert "na=False" in source
    assert ".specgram(" not in source
    assert "signal.spectrogram(" in source
    assert "np.finfo(np.float64).tiny" in source


def test_g1_reports_exact_failed_components():
    source = all_source()
    assert "_failed_blocking_check_names" in source
    assert "g1_registry_failures" in source
    assert "g1_input_failures" in source
