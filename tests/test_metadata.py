from pathlib import Path

import pandas as pd

from paper1_qc.metadata import audit_metadata_workbook


def _row(file_name: str, diagnosis="ALS", score=40, recording="2024-01-01"):
    return {
        "Raw Media File name": file_name,
        "Extension": Path(file_name).suffix,
        "SubjectID": file_name.split("_")[0],
        "Recording date": recording,
        "Task Name": "Bamboo passage",
        "Diagnosis": diagnosis,
        "Assessment date": "2024-01-01",
        "Date of Birth": "1970-01-01",
        "ALSFRS total score": score,
        "ALSFRS bulbar subscore": 10,
        "EAT10 total score": 3,
        "Sentence intelligibility percent": 90,
        "Speaking rate": 150,
    }


def test_metadata_sentinel_and_diagnosis_provenance(tmp_path):
    rows = [
        _row("ALS001_272_1_20240101_240_PSG_BAMBOO.wav", score=999),
        _row("CNEC001_272_1_20240101_240_PSG_BAMBOO.wav", diagnosis=None),
    ]
    path = tmp_path / "metadata.xlsx"
    pd.DataFrame(rows).to_excel(path, index=False)
    audit = audit_metadata_workbook(path)
    assert pd.isna(audit.clean_media_rows.loc[0, "ALSFRS total score"])
    assert audit.clean_media_rows.loc[1, "diagnosis_inferred_from_id"] == "CONTROLS"
    assert pd.isna(audit.clean_media_rows.loc[1, "diagnosis_analysis"])
    assert audit.clean_media_rows.loc[1, "diagnosis_requires_review"]
    assert "clinical_sentinel" in set(audit.issues["rule"])
    assert "diagnosis_control_id_candidate" in set(audit.issues["rule"])


def test_metadata_detects_filename_date_mismatch(tmp_path):
    path = tmp_path / "metadata.xlsx"
    pd.DataFrame(
        [_row("ALS001_272_1_20240102_240_PSG_BAMBOO.wav", recording="2024-01-01")]
    ).to_excel(path, index=False)
    audit = audit_metadata_workbook(path)
    assert "filename_recording_date_mismatch" in set(audit.issues["rule"])

