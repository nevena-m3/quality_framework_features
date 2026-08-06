from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK = (
    Path(__file__).resolve().parents[1]
    / "notebooks"
    / "02_feature_extraction"
    / "02c_reverberation_QREV_v3_1_1.ipynb"
)


def notebook_payload() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def notebook_source() -> str:
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook_payload()["cells"]
    )


def test_notebook_is_versioned_for_the_pinned_estimator() -> None:
    payload = notebook_payload()
    assert payload["metadata"]["qrev_release"] == {
        "validation_release": "qrev-v3.1.1",
        "estimator_implementation": "qrev-v3.1.0",
        "scope": "validation and gallery correction only",
    }


def test_short_support_variant_is_internally_feasible() -> None:
    source = notebook_source()
    assert '"short_support_900ms"' in source
    assert "persistence_horizon_ms=900.0" in source
    assert "floor_start_ms=600.0" in source
    assert "floor_end_ms=900.0" in source
    assert "horizon_800ms" not in source


def test_robustness_gates_follow_declared_feature_roles() -> None:
    source = notebook_source()
    assert "primary-feature availability retained at +/-20 ms" in source
    assert "secondary decay remains estimable and rank-stable" in source
    assert ">=0.90 for both primary conditional features" in source
    assert "rho>=0.80; availability reported" in source
    assert ">=0.90 for every conditional feature" not in source


def test_gallery_contract_includes_support_failures() -> None:
    source = notebook_source()
    assert "availability_audit" in source
    assert "minimum_support_audit" in source
    assert "gallery construction contract" in source
    assert ".head(18)" in source
    assert ".head(12)" not in source


def test_freeze_and_export_use_validation_release() -> None:
    source = notebook_source()
    assert 'VALIDATION_RELEASE = "qrev-v3.1.1"' in source
    assert "/ VALIDATION_RELEASE" in source
    assert "ACCEPT_QREV_V311" in source
    assert "PUBLISH_AND_FREEZE_QREV_V311" in source
    assert "qrev_v311_frozen_manifest.json" in source
