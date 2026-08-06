"""Governed four-feature QTEMP candidate built on the audited v0.3 detector.

This module does not freeze QTEMP.  It makes the finalization contract
executable: splice is excluded after failed analytical validation and accepted
near-exact repetition must last at least 40 ms.  Publication remains blocked
until the predeclared blinded G9 review passes.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from paper1_qc import qtemp as legacy


MEASUREMENT_VERSION = "qtemp-v1.0.0-candidate-g9-pending"
DETECTOR_VERSION = legacy.MEASUREMENT_VERSION
ANALYSIS_FEATURES = (
    "qtemp_dropout_duration_fraction",
    "qtemp_dropout_event_rate_per_min",
    "qtemp_frozen_audio_duration_fraction",
    "qtemp_frozen_audio_event_rate_per_min",
)
PRIMARY_FEATURES = (
    "qtemp_dropout_duration_fraction",
    "qtemp_frozen_audio_duration_fraction",
)
SECONDARY_SAME_LEDGER_FEATURES = (
    "qtemp_dropout_event_rate_per_min",
    "qtemp_frozen_audio_event_rate_per_min",
)
RETAINED_EVENT_TYPES = ("dropout", "frozen_audio")
FINAL_DUPLICATE_MIN_DURATION_MS = 40.0
# Four-millisecond comparison frames advanced every two milliseconds recover a
# 40-ms injected target as 38 ms of directly evidenced support.  The detector
# threshold is therefore the preregistered truth-scope boundary minus one hop;
# the ledger retains the observed (not inflated) support duration.
# Covers one hop plus native-sample rounding at 44.1 kHz and non-grid event
# alignment. This inclusion tolerance is tested across native rates; observed
# ledger support is never padded.
DUPLICATE_BOUNDARY_TOLERANCE_MS = 2.5
DETECTOR_DUPLICATE_MIN_EVIDENCE_MS = (
    FINAL_DUPLICATE_MIN_DURATION_MS - DUPLICATE_BOUNDARY_TOLERANCE_MS
)
ELIGIBILITY_NUMERIC_TOLERANCE_SEC = 1e-9
FINAL_PARAMETERS = replace(
    legacy.DEFAULT_PARAMETERS,
    duplicate_min_sequence_ms=DETECTOR_DUPLICATE_MIN_EVIDENCE_MS,
    minimum_eligible_duration_sec=(
        legacy.DEFAULT_PARAMETERS.minimum_eligible_duration_sec
        - ELIGIBILITY_NUMERIC_TOLERANCE_SEC
    ),
)


@dataclass
class QTEMPCandidateExtraction:
    recording: dict
    candidate_ledger: pd.DataFrame
    disposition_ledger: pd.DataFrame
    event_ledger: pd.DataFrame
    exposure_ledger: pd.DataFrame


def feature_registry_frame() -> pd.DataFrame:
    registry = legacy.feature_registry_frame()
    registry = registry.loc[registry["name"].isin(ANALYSIS_FEATURES)].copy()
    registry["measurement_version"] = MEASUREMENT_VERSION
    registry["finalization_disposition"] = "candidate_retain_pending_g9"
    registry["evidence_maturity"] = "study-specific event detector"
    frozen = registry["name"].str.startswith("qtemp_frozen_audio")
    registry.loc[frozen, "minimum_support"] = (
        "successful native decode/provenance, non-silent bilateral evidence, "
        "at least 1 s eligible exposure; preregistered target scope >=40 ms "
        "with 2.5 ms frame/hop boundary tolerance (>=37.5 ms directly evidenced support)"
    )
    registry.loc[frozen, "claim_boundary"] = (
        "near-exact consecutive decoded-waveform repetition lasting at least "
        "40 ms; not all freezes, buffering failures, packet-loss concealment, "
        "or periodic voiced speech"
    )
    registry.loc[registry["name"].isin(SECONDARY_SAME_LEDGER_FEATURES), "role"] = (
        "secondary same-ledger event-frequency view"
    )
    return registry.reset_index(drop=True)


def _version_ledger(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    if len(output):
        output["qtemp_detector_version"] = output.get(
            "qtemp_measurement_version", DETECTOR_VERSION
        )
        output["qtemp_measurement_version"] = MEASUREMENT_VERSION
    return output


def extract_qtemp(
    waveform: np.ndarray,
    sample_rate_hz: int,
    *,
    analysis_intervals: Iterable[legacy.TimeInterval] | None = None,
    speech_intervals: Iterable[legacy.TimeInterval] | None = None,
    logical_recording_id: str = "",
    native_source_confirmed: bool = True,
    preprocessing_provenance_ok: bool = True,
    parameters: legacy.QTEMPParameters = FINAL_PARAMETERS,
    enabled_event_types: Sequence[str] | None = None,
) -> QTEMPCandidateExtraction:
    """Extract the retained QTEMP contract without splice or post-hoc filtering."""

    enabled = RETAINED_EVENT_TYPES if enabled_event_types is None else tuple(enabled_event_types)
    unsupported = sorted(set(enabled).difference(RETAINED_EVENT_TYPES))
    if unsupported:
        raise ValueError(f"Non-retained QTEMP event type(s): {unsupported}")
    if parameters.duplicate_min_sequence_ms < DETECTOR_DUPLICATE_MIN_EVIDENCE_MS:
        raise ValueError("Final candidate requires duplicate evidence >=37.5 ms for the 40-ms truth scope")

    result = legacy.extract_qtemp(
        waveform,
        sample_rate_hz,
        analysis_intervals=analysis_intervals,
        speech_intervals=speech_intervals,
        logical_recording_id=logical_recording_id,
        native_source_confirmed=native_source_confirmed,
        preprocessing_provenance_ok=preprocessing_provenance_ok,
        parameters=parameters,
        enabled_event_types=enabled,
    )
    recording = dict(result.recording)
    recording["qtemp_detector_version"] = recording.get(
        "qtemp_measurement_version", DETECTOR_VERSION
    )
    recording["qtemp_measurement_version"] = MEASUREMENT_VERSION
    recording["qtemp_publication_ready"] = False
    recording["qtemp_validation_state"] = "G1_G8_PRIOR_EVIDENCE__G9_PENDING__G10_BLOCKED"
    recording["qtemp_final_duplicate_min_duration_ms"] = FINAL_DUPLICATE_MIN_DURATION_MS
    recording["qtemp_duplicate_boundary_tolerance_ms"] = DUPLICATE_BOUNDARY_TOLERANCE_MS
    for key in list(recording):
        if "splice" in key:
            del recording[key]
    return QTEMPCandidateExtraction(
        recording=recording,
        candidate_ledger=_version_ledger(result.candidate_ledger),
        disposition_ledger=_version_ledger(result.disposition_ledger),
        event_ledger=_version_ledger(result.event_ledger),
        exposure_ledger=_version_ledger(result.exposure_ledger),
    )


def reconstruct_recording_features(
    event_ledger: pd.DataFrame,
    eligible_duration_sec: float,
) -> dict[str, float]:
    values = legacy.reconstruct_recording_features(event_ledger, eligible_duration_sec)
    return {name: values[name] for name in ANALYSIS_FEATURES}


TimeInterval = legacy.TimeInterval
inject_dropout = legacy.inject_dropout
inject_consecutive_duplicate = legacy.inject_consecutive_duplicate
poisson_rate_interval = legacy.poisson_rate_interval
