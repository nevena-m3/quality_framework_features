from __future__ import annotations

import ast
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = (
    ROOT
    / "notebooks"
    / "02_feature_extraction"
    / "02e_nonlinear_distortion_QDIST_v3_1_1.ipynb"
)
GENERATOR = ROOT / "scripts" / "generate_qdist_v311_notebook.py"


def test_qdist_notebook_and_generator_exist():
    assert GENERATOR.exists()
    assert NOTEBOOK.exists()


def test_all_qdist_notebook_code_cells_parse():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "code":
            ast.parse(cell.source, filename=f"qdist-cell-{index}")


def test_notebook_has_complete_scientific_validation_structure():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    text = "\n".join(cell.source for cell in notebook.cells)
    required = [
        "Immutable feature registry",
        "Hard-clipping construct recovery",
        "Quantization, natural-extrema",
        "Native-view transformation and codec characterization",
        "Parameter-neighborhood",
        "Frozen cohort and native-source provenance contract",
        "reconstruct_qdist_features",
        "Sparse-event cohort",
        "Empirical parameter robustness",
        "Label-blind accepted, rejected, and valid-zero waveform gallery",
        "G1–G10",
        "qdist_v311_candidate_manifest.json",
        "PUBLISH_AND_FREEZE_QDIST_V311 = False",
    ]
    for phrase in required:
        assert phrase in text


def test_notebook_uses_only_three_analysis_features_and_rejects_legacy_proxies():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    text = "\n".join(cell.source for cell in notebook.cells)
    for feature in [
        "qdist_hard_clipped_frame_fraction",
        "qdist_hard_clip_event_rate_per_min",
        "qdist_hard_clipped_sample_fraction",
    ]:
        assert feature in text
    assert "removed_from_analysis_set" in text
    assert "qdist_near_fullscale_fraction" in text
    assert "qdist_edge_histogram_spike" in text


def test_notebook_does_not_use_clinical_labels_for_measurement_approval():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    code = "\n".join(
        cell.source for cell in notebook.cells if cell.cell_type == "code"
    ).lower()
    forbidden = [
        "alsfrs",
        "diagnosis",
        "bulbar_score",
        "human_qc_label",
        "case_control",
    ]
    for term in forbidden:
        assert term not in code


def test_manual_gallery_review_is_versioned_and_blocks_freeze():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    text = "\n".join(cell.source for cell in notebook.cells)
    assert "qdist_v311_gallery_review.csv" in text
    assert "DEFINITE_HARD_CLIP" in text
    assert "accepted_precision >= 0.90" in text
    assert "accepted_recording_precision >= 0.90" in text
    assert "QDIST_REVIEW_DECISION == \"ACCEPT_QDIST_V311\"" in text


def test_notebook_uses_native_only_decode_and_compact_single_file_checkpoints():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    code = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
    assert "decode_native_audio" in code
    assert "decode_audio_views" not in code
    assert ".qdist.pkl.gz" in code
    assert "flat_run_prefilter_reduction_fraction" in code


def test_g8_is_decoupled_from_nonmerge_g6_and_all_accepted_plateaus_are_reviewed():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    code = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
    assert "g8_merge_checks" in code
    assert "all accepted plateaus are included" in code.lower()
    assert "accepted_plateau" in code
    assert "p90_clipped_duration_change_ms_per_min" in code


def test_construct_grid_has_nonzero_truth_and_sample_accurate_recovery_gates():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    code = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
    for phrase in [
        "highest_energy_window",
        "true_clipped_sample_count",
        "sample_precision",
        "sample_recall",
        "sample_f1",
        "qdist_v311_hard_clip_sample_recovery",
    ]:
        assert phrase in code
    assert "minimum=0.99" not in code  # thresholds are computed from outputs, not hard-coded results


def test_generator_reproduces_committed_notebook_exactly():
    import copy
    import importlib.util

    spec = importlib.util.spec_from_file_location("generate_qdist_v311_notebook", GENERATOR)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    generated = module.build_notebook()
    committed = nbformat.read(NOTEBOOK, as_version=4)

    def governed_view(notebook):
        notebook = copy.deepcopy(notebook)
        for cell in notebook.cells:
            if cell.cell_type == "code":
                cell.execution_count = None
                cell.outputs = []
        return notebook

    assert governed_view(generated) == governed_view(committed)
