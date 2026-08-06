from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from paper1_qc.qchan import ANALYSIS_FEATURES


ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "scripts" / "generate_qchan_v3_notebook.py"
NOTEBOOK = (
    ROOT / "notebooks" / "02_feature_extraction"
    / "02d_channel_device_QCHAN_v3_0_1.ipynb"
)


def load_generator():
    spec = importlib.util.spec_from_file_location(
        "generate_qchan_v3_notebook", GENERATOR
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
        for cell in notebook()["cells"]
    )


def test_generator_reproduces_committed_notebook():
    generated = load_generator().build_notebook()
    assert generated == notebook()


def test_all_code_cells_compile():
    for index, cell in enumerate(notebook()["cells"]):
        if cell["cell_type"] != "code":
            continue
        compile(
            "".join(cell["source"]),
            f"qchan_notebook_cell_{index}",
            "exec",
        )


def test_notebook_is_clean_and_unexecuted():
    for cell in notebook()["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None
            assert cell["outputs"] == []


def test_required_scientific_sections_are_present():
    source = all_source()
    for heading in [
        "Formula, gain, polarity, and determinism validation",
        "Synthetic construct validity and discriminant controls",
        "Source-bandwidth, spectral-floor, and codec characterization",
        "Frozen subject-balanced LOSO references and feature table",
        "Reference resampling, subject balancing, and vintage stability",
        "Boundary, frame, support, and availability robustness",
        "Empirical distributions, availability, and redundancy",
        "Label-blind scientific gallery",
        "Immutable freeze",
        "Central analysis-feature table export",
    ]:
        assert heading in source


def test_exact_analysis_features_are_declared():
    source = all_source()
    for feature in ANALYSIS_FEATURES:
        assert feature in source
    assert "qchan_score" in source  # explicit prohibited-column check
    assert '"qchan_score":' not in source


def test_reference_contract_has_no_global_or_cross_task_fallback():
    source = all_source()
    assert "build_subject_balanced_loso_references" in source
    assert "def bootstrap_reference" in source
    assert "REFERENCE_BOOTSTRAP_REPLICATES" in source
    assert "task_matching_required" in source
    assert "global_leave_one_subject_out_fallback" not in source
    assert "cross-task or global fallback" in source
    assert "reference_vintage_sha256" in source


def test_reference_identity_excludes_labels():
    source = all_source()
    assert "no clinical or human-QC fields enter reference identity" in source
    assert "diagnosis" in source
    assert "alsfrs" in source
    assert "human_qc" in source
    assert "no clinical or human-QC labels used" in source


def test_gallery_uses_cached_spectra_and_does_not_redecode():
    cells = notebook()["cells"]
    gallery_cell = next(
        "".join(cell["source"])
        for cell in cells
        if cell["cell_type"] == "code"
        and "def choose_gallery" in "".join(cell["source"])
    )
    assert "checkpoint_paths" in gallery_cell
    assert "np.load" in gallery_cell
    assert "decode_audio_views" not in gallery_cell


def test_final_cells_do_not_launch_tests_or_recompute():
    cells = notebook()["cells"]
    freeze_cell = next(
        "".join(cell["source"])
        for cell in cells
        if cell["cell_type"] == "code"
        and "freeze_requested_safely" in "".join(cell["source"])
    )
    assert "subprocess" not in freeze_cell
    assert "pytest" not in freeze_cell
    assert "decode_audio_views" not in freeze_cell
    assert "extract_recording_spectrum" not in freeze_cell


def test_candidate_cannot_enter_central_feature_tables():
    source = all_source()
    assert "if FROZEN_MANIFEST.exists()" in source
    assert "No unfrozen QCHAN table was copied" in source
    assert "qchan_v301_analysis_features.csv" in source
    assert "qchan_v301_analysis_features.parquet" in source


def test_freeze_is_non_overwriting_and_review_gated():
    source = all_source()
    assert "ACCEPT_QCHAN_V301" in source
    assert "Refusing to overwrite immutable freeze" in source
    assert "Failed gates cannot be waived" in source
    assert "PUBLISH_AND_FREEZE_QCHAN_V301 = False" in source


def test_r1_frozen_input_contract_is_explicit_and_schema_safe():
    source = all_source()
    assert 'NOTEBOOK_REVISION = "qchan-v3.0.1-r1"' in source
    assert 'INPUT_CONTRACT_VERSION = "qchan-frozen-input-v3"' in source
    assert '"SubjectID", "subject_id"' in source
    assert 'frozen_segmentation_intervals' in source
    assert 'discover_interval_table' not in source
    assert 'def normalize_text' in source
    assert 'def validate_checkpoint' in source
    assert 'complete eligible-ID checkpoint coverage' in source
    assert 'reference metadata is identity-only' not in source
    assert 'no clinical or human-QC fields enter reference identity' in source


def test_r1_export_verifies_manifest_and_hashes():
    source = all_source()
    assert 'Frozen manifest notebook revision mismatch' in source
    assert 'stage_file_sha256' in source
    assert 'Frozen source hash mismatch or absent from manifest' in source


def test_v301_task_freeze_floor_and_gallery_repairs_are_present():
    source = all_source()
    assert '["task_stratum", "task_name", "task", "protocol_task"]' in source
    assert '["task_stratum", "task_name", "task", "protocol_task", "protocol"]' not in source
    assert 'DATA_FREEZE_VERSION = "v1"' in source
    assert 'SEGMENTATION_FREEZE_VERSION = "v1"' in source
    assert 'notebook_constant:DATA_FREEZE_VERSION' in source
    assert 'columns=["logical_recording_id", "condition", "error_type", "message"]' in source
    assert 'qchan_v301_empirical_floor_sensitivity_summary' in source
    assert 'LTAS-distance scale is explicitly floor-dependent' in source
    assert 'positive_q{int(100 * quantile):02d}' in source
    assert 'zero plus positive q10/q50/q90' in source


def test_notebook_has_unique_stable_cell_ids():
    cells = notebook()["cells"]
    cell_ids = [cell.get("id") for cell in cells]
    assert all(cell_ids)
    assert len(cell_ids) == len(set(cell_ids))


def test_r1_parquet_contract_fails_early_and_writes_atomically():
    source = all_source()
    assert "def resolve_parquet_engine" in source
    assert "QCHAN requires a parquet engine" in source
    assert "engine=PARQUET_ENGINE" in source
    assert 'csv_tmp = csv_path.with_name' in source
    assert 'parquet_tmp = parquet_path.with_name' in source


def test_scientific_citation_registry_uses_verified_dois():
    source = all_source()
    for doi in [
        "10.1044/jshr.3606.1177",
        "10.1080/1401543051006721",
        "10.1044/1058-0360(2010/09-0091)",
        "10.1121/10.0005132",
        "10.1044/2024_AJSLP-23-00372",
    ]:
        assert doi in source
    assert "10.1044/jshr.3605.1092" not in source
    assert "10.3109/14015430902839961" not in source
    assert "10.1044/2023_AJSLP-23-00131" not in source


def test_signed_directional_precursors_and_single_task_scope_are_audited():
    source = all_source()
    assert "one-sided deficits reconstruct from retained signed precursors" in source
    assert "qchan_rolloff95_signed_difference_hz" in source
    assert "qchan_highband_ratio_signed_difference" in source
    assert "qchan_tilt_signed_difference_db_per_oct" in source
    assert "No cross-task generalization claim" in source
    assert "each new task requires its own frozen reference" in source
