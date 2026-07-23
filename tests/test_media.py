import pandas as pd

from paper1_qc.media import audit_native_metadata_consistency


def test_native_metadata_consistency_flags_rate_and_duration():
    inventory = pd.DataFrame(
        [
            {
                "file_name": "a.wav",
                "file_path": "/tmp/a.wav",
                "probe_ok": True,
                "full_decode_ok": True,
                "full_decode_warning": "",
                "sample_rate_hz": 48000,
                "stream_duration_sec": 10.0,
                "container_duration_sec": 10.0,
            }
        ]
    )
    metadata = pd.DataFrame(
        [{"Raw Media File name": "a.wav", "Sampling Rate": 44100, "Duration (s)": 8.0}]
    )
    issues = audit_native_metadata_consistency(inventory, metadata)
    assert set(issues["rule"]) == {
        "native_metadata_sample_rate_mismatch",
        "native_metadata_duration_mismatch",
    }

