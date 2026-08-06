from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

EXPECTED_PANELS = set("ABCDEFGH")
EXPECTED_FIGURE_COUNT = 32


def _find_project_root() -> Path:
    start = Path.cwd().resolve()
    for candidate in [start, *start.parents]:
        if (candidate / "MAIN outputs/reviewed").exists() and (candidate / "outputs/reviewed").exists():
            return candidate
    raise RuntimeError("Could not locate project root")


def _candidate_root() -> Path:
    return _find_project_root() / "outputs/reviewed" / "gain_dynamics" / "qgain-v4.1.0-figures-v1.0.0-candidate"


def test_candidate_manifest_and_source_freeze_contract():
    root = _candidate_root()
    manifest = json.loads((root / "manifests" / "qgain_v410_figure_package_manifest.json").read_text(encoding="utf-8"))
    assert manifest["measurement_version"] == "qgain-v4.1.0"
    assert manifest["figure_package_version"] == "qgain-v4.1.0-figures-v1.0.0"
    assert manifest["feature_values_recomputed"] is False
    assert manifest["standalone_gate_allowed"] is False
    assert manifest["family_scalar_constructed"] is False
    assert manifest["required_panels_complete"] is True


def test_figure_index_is_complete_and_every_artifact_exists():
    root = _candidate_root()
    index = pd.read_csv(root / "tables" / "qgain_v410_figure_gallery_index.csv")
    assert len(index) == EXPECTED_FIGURE_COUNT
    assert EXPECTED_PANELS.issubset(set(index["panel"]))
    assert set(index["status"]) == {"PASS"}
    for column in ["figure_png", "figure_svg", "figure_pdf", "source_data", "caption", "provenance_json"]:
        for relative in index[column].astype(str):
            assert (root / relative).is_file(), (column, relative)


def test_completed_workbook_and_checklists_are_attached():
    root = _candidate_root()
    expected = [
        root / "docs" / "QGAIN_Family_Evaluation_Workbook_v1_0.docx",
        root / "tables" / "QGAIN_Validation_Checklist_v1_0.csv",
        root / "tables" / "QGAIN_Ten_Domain_Dashboard_v1_0.csv",
        root / "tables" / "QGAIN_Figure_Gallery_Index_v1_0.csv",
    ]
    for path in expected:
        assert path.is_file(), path


def test_no_measurement_values_are_recomputed_or_overwritten():
    project = _find_project_root()
    freeze_manifest = json.loads((project / "MAIN outputs/reviewed" / "06_family_freezes" / "gain_dynamics" / "qgain-v4.1.0" / "manifests" / "qgain_v410_freeze_manifest.json").read_text(encoding="utf-8"))
    candidate_manifest = json.loads((_candidate_root() / "manifests" / "qgain_v410_figure_package_manifest.json").read_text(encoding="utf-8"))
    assert candidate_manifest["source_freeze_inventory_sha256"] == freeze_manifest["freeze_inventory_sha256"]
    assert candidate_manifest["source_executed_notebook_sha256"] == freeze_manifest["executed_notebook_sha256"]
