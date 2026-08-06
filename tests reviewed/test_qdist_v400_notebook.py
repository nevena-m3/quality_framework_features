from __future__ import annotations

import ast
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = (
    ROOT
    / "notebooks reviewed"
    / "05_QDIST"
    / "05_nonlinear_distortion_QDIST_v4_0_0_REVIEWED_PREFLIGHT_SOURCE.ipynb"
)


def notebook():
    return nbformat.read(NOTEBOOK, as_version=4)


def source_text():
    return "\n".join(cell.source for cell in notebook().cells)


def literal_assignments():
    assignments = {}
    for cell in notebook().cells:
        if cell.cell_type != "code":
            continue
        tree = ast.parse(cell.source)
        for node in tree.body:
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            target = node.targets[0]
            if not isinstance(target, ast.Name):
                continue
            try:
                assignments[target.id] = ast.literal_eval(node.value)
            except Exception:
                pass
    return assignments


def test_notebook_exists():
    assert NOTEBOOK.exists()


def test_candidate_controls_are_safe():
    values = literal_assignments()
    assert values["RUN_COHORT_EXTRACTION"] is False
    assert values["PUBLISH_AND_FREEZE"] is False
    assert values["SCIENTIFIC_REVIEW_DECISION"] == "PENDING"


def test_notebook_uses_reviewed_wrapper():
    assert "paper1_qc_reviewed.qdist_v400" in source_text()


def test_no_family_scalar():
    text = source_text()
    assert "family_scalar_constructed" in text
    assert "standalone_gate_allowed" in text


def test_panels_a_to_c_declared_semantically():
    values = literal_assignments()
    assert tuple(values["DECLARED_PREFLIGHT_PANELS"]) == (
        "A_construct_response",
        "B_discriminant_specificity",
        "C_transformation_contract",
    )


def test_event_verification_remains_pending():
    assert "APPLICABLE_pending_event_verification" in source_text()
