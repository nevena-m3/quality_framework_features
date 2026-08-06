from __future__ import annotations

import pandas as pd
import pytest

from paper1_qc.reviewed_features import (
    FAMILIES,
    build_latest_feature_release,
    load_latest_feature_release,
    reviewed_registry_frame,
)


def _write_reviewed_sources(root, *, mismatched_family: str | None = None) -> None:
    reviewed = root / "MAIN outputs" / "02_FEATURE_REVIEWED"
    tables = reviewed / "01_analysis_features"
    registries = reviewed / "00_feature_registry"
    tables.mkdir(parents=True)
    registries.mkdir(parents=True)
    registry = reviewed_registry_frame()
    for family in FAMILIES:
        ids = ["recording-1", "recording-2"]
        if family.code == mismatched_family:
            ids[1] = "different-recording"
        frame = pd.DataFrame({"logical_recording_id": ids})
        if family.code == "QADD":
            frame["SubjectID"] = ["S1", "S2"]
            frame["file_name"] = ["one.wav", "two.wav"]
        for index, spec in (
            registry.loc[registry["family_code"].eq(family.code)].reset_index(drop=True).iterrows()
        ):
            frame[spec["feature"]] = [float(index), float(index + 1)]
            frame[spec["status_field"]] = "measured"
        frame.to_csv(tables / family.table_name, index=False)
        pd.DataFrame({"feature": ["source-registry-present"]}).to_csv(
            registries / family.registry_name, index=False
        )


def test_latest_release_is_compact_and_complete(tmp_path) -> None:
    _write_reviewed_sources(tmp_path)

    result = build_latest_feature_release(tmp_path)
    features, registry = load_latest_feature_release(tmp_path)
    release_root = tmp_path / "MAIN outputs" / "02_FEATURE_LATEST"

    assert result["recording_count"] == 2
    assert result["feature_count"] == 20
    assert features["logical_recording_id"].is_unique
    assert set(registry["feature"]).issubset(features.columns)
    assert sorted(path.name for path in (release_root / "families").iterdir()) == [
        "01_QGAIN",
        "02_QADD",
        "03_QREV",
        "04_QCHAN",
        "05_QDIST",
        "06_QTEMP",
    ]
    disposition = pd.read_csv(release_root / "families" / "06_QTEMP" / "feature_disposition.csv")
    assert disposition.loc[0, "validated_primary_feature_count"] == 0


def test_latest_release_rejects_mismatched_family_cohort(tmp_path) -> None:
    _write_reviewed_sources(tmp_path, mismatched_family="QREV")

    with pytest.raises(ValueError, match="QREV reviewed cohort"):
        build_latest_feature_release(tmp_path)
