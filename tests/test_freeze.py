import pandas as pd
import pytest

from paper1_qc.freeze import (
    attach_media_freeze_gate,
    diagnosis_adjudication_template,
    load_adjudications,
    resolve_diagnoses,
)


PATTERNS = [r"^CNEC\d+$", r"^C\d+$", r"^TCV\d+$"]


def _canonical():
    return pd.DataFrame(
        {
            "SubjectID": ["ALS01", "CNEC001", "C05-1", "PENDING004"],
            "diagnosis_reported": ["ALS", pd.NA, pd.NA, pd.NA],
            "Raw Media File name": ["a.wav", "b.wav", "c.wav", "d.wav"],
            "logical_recording_id": ["a", "b", "c", "d"],
        }
    )


def test_template_omits_confirmed_exact_control_patterns():
    template = diagnosis_adjudication_template(
        [_canonical()],
        control_id_patterns=PATTERNS,
        confirm_control_patterns=True,
    )
    assert set(template["SubjectID"]) == {"C05-1", "PENDING004"}


def test_template_omits_investigator_confirmed_exceptional_controls():
    template = diagnosis_adjudication_template(
        [_canonical()],
        control_id_patterns=PATTERNS,
        confirm_control_patterns=True,
        confirmed_control_subject_ids=["C05-1"],
    )
    assert set(template["SubjectID"]) == {"PENDING004"}


def test_resolution_keeps_provenance_and_manual_exclusion():
    manual = pd.DataFrame(
        [
            {
                "SubjectID": "C05-1",
                "diagnosis_analysis": "CONTROLS",
                "evidence_source": "data manager confirmation",
                "reviewer": "PI",
                "review_date": "2026-07-23",
                "notes": "",
            },
            {
                "SubjectID": "PENDING004",
                "diagnosis_analysis": "EXCLUDE",
                "evidence_source": "not confirmed in target cohort",
                "reviewer": "PI",
                "review_date": "2026-07-23",
                "notes": "",
            },
        ]
    )
    resolved = resolve_diagnoses(
        _canonical(),
        adjudications=manual,
        allowed_diagnoses=["ALS", "CONTROLS"],
        control_id_patterns=PATTERNS,
        confirm_control_patterns=True,
        control_rule_evidence="data-manager rule",
    ).set_index("SubjectID")
    assert resolved.loc["CNEC001", "diagnosis_analysis"] == "CONTROLS"
    assert resolved.loc["CNEC001", "diagnosis_resolution"] == "confirmed_control_id_rule"
    assert resolved.loc["C05-1", "diagnosis_analysis"] == "CONTROLS"
    assert resolved.loc["PENDING004", "diagnosis_resolution"] == "manual_excluded"
    assert not bool(resolved.loc["PENDING004", "target_cohort_eligible"])


def test_blank_adjudication_is_rejected(tmp_path):
    path = tmp_path / "adjudication.csv"
    pd.DataFrame(
        [
            {
                "SubjectID": "PENDING004",
                "diagnosis_analysis": "",
                "evidence_source": "",
                "reviewer": "",
                "review_date": "",
                "notes": "",
            }
        ]
    ).to_csv(path, index=False)
    with pytest.raises(ValueError, match="ALS, CONTROLS, or EXCLUDE"):
        load_adjudications(path)


def test_media_gate_requires_one_decodable_path():
    canonical = _canonical().iloc[:2].copy()
    inventory = pd.DataFrame(
        {
            "file_name": ["a.wav", "b.wav", "b.wav"],
            "file_path": ["/a.wav", "/one/b.wav", "/two/b.wav"],
            "probe_ok": [True, True, True],
            "full_decode_ok": [True, True, True],
            "sha256": ["a", "b", "c"],
        }
    )
    gated = attach_media_freeze_gate(canonical, inventory).set_index("SubjectID")
    assert bool(gated.loc["ALS01", "media_freeze_eligible"])
    assert not bool(gated.loc["CNEC001", "media_freeze_eligible"])
    assert (
        gated.loc["CNEC001", "media_freeze_reason"]
        == "canonical_filename_resolves_to_multiple_paths"
    )


def test_media_gate_prefers_wav_over_webm_when_both_decode():
    canonical = _canonical().iloc[:1].copy()
    canonical.loc[:, "Raw Media File name"] = "a.webm"
    media_rows = pd.DataFrame(
        {
            "logical_recording_id": ["a", "a"],
            "Raw Media File name": ["a.webm", "a.wav"],
            "extension_parsed": [".webm", ".wav"],
        }
    )
    inventory = pd.DataFrame(
        {
            "file_name": ["a.webm", "a.wav"],
            "file_path": ["/a.webm", "/a.wav"],
            "probe_ok": [True, True],
            "full_decode_ok": [True, True],
            "sha256": ["webm", "wav"],
        }
    )
    gated = attach_media_freeze_gate(
        canonical,
        inventory,
        media_rows=media_rows,
        media_preference=[".wav", ".webm"],
    )
    assert gated.loc[0, "Raw Media File name"] == "a.wav"
    assert gated.loc[0, "media_path"] == "/a.wav"
