"""QREV v4.0 reviewed residual-tail and SRMR measurements.

This module measures observable post-speech residual behavior compatible with
reverberation or echo. It does not estimate RT60, EDT, C50/C80, D50, DRR, STI,
a room impulse response, or echo identity.

The boundary estimators use *primary-speech* offsets. The frozen strict-speech
view is a separately eroded support view and must never be substituted for the
natural speech offset. SRMR is computed over the primary task span while strict
speech duration is retained as its support quantity.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from math import ceil, floor, log10, sqrt

import numpy as np
import pandas as pd
from scipy import stats

MEASUREMENT_VERSION = "qrev-v4.0.0-candidate"
SRMR_UPSTREAM_COMMIT = "fee009779cef96bed34db3a7e31d10f3ad1ea133"
SRMR_VARIANT = "SRMRpy normalized-fast; norm=True; fast=True; max_cf=30"
SRMR_GAMMATONE_VERSION = "1.0.3"
SRMR_PINNED_REGRESSION_VALUE = 3.7158141034373164

ANALYSIS_FEATURES = (
    "qrev_tail_excess_100ms_db",
    "qrev_tail_persistence_median_sec",
    "qrev_downward_decay_rate_db_per_sec",
    "qrev_srmr_norm",
)
CONDITIONAL_BOUNDARY_FEATURES = ANALYSIS_FEATURES[:3]
BROADLY_AVAILABLE_COMPARATOR_FEATURES = (ANALYSIS_FEATURES[3],)


@dataclass(frozen=True, order=True)
class SpeechInterval:
    start_sec: float
    end_sec: float
    interval_id: str = ""
    interval_index: int | None = None
    view: str = "primary_speech"
    profile: str = "primary"

    @property
    def duration_sec(self) -> float:
        return max(0.0, float(self.end_sec) - float(self.start_sec))


# Backward-friendly alias for synthetic tests.
TimeInterval = SpeechInterval


@dataclass(frozen=True)
class QREVParameters:
    measurement_version: str = MEASUREMENT_VERSION
    analysis_sample_rate_hz: int = 16_000
    frame_length_ms: float = 30.0
    frame_hop_ms: float = 10.0
    dbfs_floor_db: float = -120.0
    maximum_floor_frame_fraction: float = 0.10
    maximum_floor_iqr_db: float = 12.0

    early_tail_start_ms: float = 0.0
    early_tail_end_ms: float = 100.0
    decay_start_ms: float = 0.0
    decay_end_ms: float = 300.0
    floor_start_ms: float = 700.0
    floor_end_ms: float = 1000.0
    persistence_horizon_ms: float = 600.0
    persistence_threshold_db: float = 3.0
    persistence_consecutive_frames: int = 3
    minimum_early_frame_count: int = 5
    minimum_floor_frame_count: int = 20
    minimum_decay_frame_count: int = 20
    minimum_persistence_frame_count: int = 50
    minimum_decay_dynamic_range_db: float = 3.0

    # Provisional Master-v1 support policy. The cohort notebook will compare
    # 2-, 3-, and 4-boundary policies before G10; raw estimates are always kept.
    minimum_tail_boundary_count: int = 2
    minimum_persistence_boundary_count: int = 2
    minimum_decay_boundary_count: int = 2
    moderate_boundary_count: int = 4
    high_boundary_count: int = 6

    minimum_srmr_speech_support_sec: float = 3.0
    minimum_srmr_task_span_sec: float = 3.0
    srmr_n_cochlear_filters: int = 23
    srmr_low_frequency_hz: float = 125.0
    srmr_min_modulation_cf_hz: float = 4.0
    srmr_max_modulation_cf_hz: float = 30.0
    srmr_normalized: bool = True
    srmr_fast: bool = True
    maximum_srmr_estimated_memory_mb: float = 512.0
    random_seed: int = 20260803

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT_PARAMETERS = QREVParameters()

FEATURE_DEFINITIONS = (
    {
        "feature": "qrev_tail_excess_100ms_db",
        "display_name": "Early post-offset tail excess",
        "role": "primary conditional candidate",
        "unit": "dB",
        "maturity": "study-specific residual-tail proxy",
        "estimand": "Median signed early 0-100-ms AC level minus independent 700-1000-ms late-pause floor across eligible primary-speech offsets.",
        "orientation": "Higher means stronger early residual energy above the local late-pause baseline.",
        "claim_limit": "Not RT60, DRR, RIR recovery, echo identity, or an ordinal quality score.",
    },
    {
        "feature": "qrev_tail_persistence_median_sec",
        "display_name": "Bounded tail persistence",
        "role": "primary conditional candidate",
        "unit": "s",
        "maturity": "study-specific bounded/censored persistence estimator",
        "estimand": "Median observed time until the post-offset envelope remains within 3 dB of the local floor for three frames, right-censored at 0.6 s. The late floor is estimated independently from 0.7-1.0 s.",
        "orientation": "Higher means longer observable above-floor persistence within the fixed 0.6-s horizon.",
        "claim_limit": "Not reverberation time; horizon values are right-censored lower bounds.",
    },
    {
        "feature": "qrev_downward_decay_rate_db_per_sec",
        "display_name": "Conditional downward tail-decay rate",
        "role": "secondary conditional candidate",
        "unit": "dB/s",
        "maturity": "study-specific robust slope",
        "estimand": "Median magnitude of a negative Theil-Sen slope during 0-300 ms when robust dynamic range is at least 3 dB.",
        "orientation": "Lower positive magnitude means slower valid downward decay; unavailable is not zero.",
        "claim_limit": "Not a Schroeder decay or RT estimate; nondecaying/rising/nonsmooth boundaries remain unavailable.",
    },
    {
        "feature": "qrev_srmr_norm",
        "display_name": "Normalized-fast SRMR",
        "role": "secondary established comparator",
        "unit": "ratio",
        "maturity": "published no-reference metric; pinned implementation",
        "estimand": "Pinned normalized-fast SRMRpy ratio over the primary task span with internal pauses preserved.",
        "orientation": "Typically lower means greater reverberation-related modulation smearing for the pinned variant.",
        "claim_limit": "Reverberation-sensitive but not reverberation-specific and not a direct RT60 measure.",
    },
)


@dataclass
class QREVExtraction:
    recording: dict
    boundary_ledger: pd.DataFrame


def feature_registry_frame() -> pd.DataFrame:
    frame = pd.DataFrame(FEATURE_DEFINITIONS)
    frame.insert(0, "measurement_version", MEASUREMENT_VERSION)
    frame["signal_view"] = "mono, globally DC-removed, deterministic 16-kHz analysis view"
    frame["boundary_view"] = ["primary_speech"] * 3 + ["primary task span; strict speech support"]
    frame["support_policy"] = "provisional 2-boundary minimum; 2/3/4 policies compared before G10"
    frame["missing_value_behavior"] = "NaN when unavailable; raw estimate, count, status, censoring, and reason retained"
    frame["family_scalar_prohibited"] = True
    frame["standalone_gate_prohibited"] = True
    return frame


def _as_interval(item, default_view: str) -> SpeechInterval:
    if isinstance(item, SpeechInterval):
        return item
    if hasattr(item, "start_sec") and hasattr(item, "end_sec"):
        return SpeechInterval(
            float(item.start_sec), float(item.end_sec),
            str(getattr(item, "interval_id", "")),
            getattr(item, "interval_index", None),
            str(getattr(item, "view", default_view)),
            str(getattr(item, "profile", "primary")),
        )
    if isinstance(item, Sequence) and len(item) >= 2:
        return SpeechInterval(float(item[0]), float(item[1]), view=default_view)
    raise TypeError(f"Cannot interpret {type(item).__name__} as a speech interval")


def validate_speech_intervals(
    intervals: Iterable[SpeechInterval],
    duration_sec: float,
    *,
    required_view: str,
    required_profile: str = "primary",
) -> list[SpeechInterval]:
    """Clip, sort, and validate a frozen speech view without silent merging."""
    clean: list[SpeechInterval] = []
    seen_ids: set[str] = set()
    for raw in intervals:
        item = _as_interval(raw, required_view)
        if item.view != required_view:
            raise ValueError(f"Expected {required_view!r}; received {item.view!r}")
        if item.profile != required_profile:
            raise ValueError(f"Expected profile {required_profile!r}; received {item.profile!r}")
        start = min(max(float(item.start_sec), 0.0), float(duration_sec))
        end = min(max(float(item.end_sec), 0.0), float(duration_sec))
        if not np.isfinite(start) or not np.isfinite(end) or end <= start:
            continue
        interval_id = item.interval_id or f"{required_view}:{len(clean):05d}"
        if interval_id in seen_ids:
            raise ValueError(f"Duplicate frozen interval identity: {interval_id}")
        seen_ids.add(interval_id)
        clean.append(SpeechInterval(start, end, interval_id, item.interval_index, item.view, item.profile))
    clean.sort(key=lambda x: (x.start_sec, x.end_sec, x.interval_id))
    for left, right in zip(clean[:-1], clean[1:]):
        if right.start_sec < left.end_sec - 1e-12:
            raise ValueError("Frozen speech intervals overlap; resolve upstream instead of merging")
    return clean


def internal_pause_boundaries(primary_speech: Iterable[SpeechInterval], duration_sec: float) -> list[dict]:
    speech = validate_speech_intervals(
        primary_speech, duration_sec, required_view="primary_speech"
    )
    rows = []
    for index, (left, right) in enumerate(zip(speech[:-1], speech[1:])):
        if right.start_sec <= left.end_sec:
            continue
        rows.append({
            "boundary_index": int(index),
            "boundary_source_view": "primary_speech",
            "boundary_source_profile": "primary",
            "left_interval_id": left.interval_id,
            "left_interval_index": left.interval_index,
            "right_interval_id": right.interval_id,
            "right_interval_index": right.interval_index,
            "previous_speech_start_sec": left.start_sec,
            "speech_offset_sec": left.end_sec,
            "pause_start_sec": left.end_sec,
            "pause_end_sec": right.start_sec,
            "next_speech_onset_sec": right.start_sec,
            "pause_duration_sec": right.start_sec - left.end_sec,
        })
    return rows


def remove_global_dc(waveform: np.ndarray) -> np.ndarray:
    """Return a finite mono float64 waveform with its global mean removed.

    QREV's scientific contract defines a globally DC-removed 16-kHz analysis
    view. This preprocessing belongs to the QREV feature pipeline rather than
    to the pinned upstream SRMRpy implementation regression.
    """
    values = np.asarray(waveform, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("QREV requires mono audio")
    if not np.isfinite(values).all():
        raise ValueError("Waveform contains non-finite samples")
    if not len(values):
        return values.copy()
    return values - float(np.mean(values))


def ac_rms_dbfs(samples: np.ndarray, floor_db: float = -120.0) -> tuple[float, bool]:
    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        return np.nan, False
    centered = values - float(np.mean(values))
    rms = float(sqrt(float(np.mean(centered * centered))))
    if rms <= 0.0 or not np.isfinite(rms):
        return float(floor_db), True
    return float(max(floor_db, 20.0 * log10(rms))), False


def _frame_levels(waveform, fs, start_sec, end_sec, parameters):
    frame_n = round(parameters.frame_length_ms * fs / 1000.0)
    hop_n = round(parameters.frame_hop_ms * fs / 1000.0)
    first = max(0, int(ceil(float(start_sec) * fs - 1e-9)))
    final_sample = min(len(waveform), int(floor(float(end_sec) * fs + 1e-9)))
    final_start = final_sample - frame_n
    columns = ["frame_start_sec","frame_end_sec","frame_mid_sec","ac_rms_dbfs","at_digital_floor"]
    if final_start < first:
        return pd.DataFrame(columns=columns)
    rows=[]
    for start in np.arange(first, final_start + 1, hop_n, dtype=np.int64):
        end=int(start+frame_n)
        level, at_floor = ac_rms_dbfs(waveform[int(start):end], parameters.dbfs_floor_db)
        rows.append({
            "frame_start_sec": start/fs,
            "frame_end_sec": end/fs,
            "frame_mid_sec": (start+frame_n/2)/fs,
            "ac_rms_dbfs": level,
            "at_digital_floor": bool(at_floor),
        })
    return pd.DataFrame(rows, columns=columns)


def boundary_envelope_trace(waveform, fs, speech_offset_sec, pause_end_sec, *, parameters=DEFAULT_PARAMETERS):
    # The trace must include both the persistence horizon and the independent
    # late-pause floor window. Truncating at the persistence horizon would make
    # floor estimation impossible whenever the horizon precedes the floor.
    required_end_ms = max(
        parameters.persistence_horizon_ms,
        parameters.floor_end_ms,
        parameters.decay_end_ms,
        parameters.early_tail_end_ms,
    )
    end = min(float(pause_end_sec), float(speech_offset_sec) + required_end_ms/1000.0)
    trace = _frame_levels(waveform, fs, speech_offset_sec, end, parameters)
    if len(trace):
        for stem in ["start","end","mid"]:
            trace[f"relative_{stem}_sec"] = trace[f"frame_{stem}_sec"] - float(speech_offset_sec)
    return trace


def _select_relative(trace, start, end):
    if trace.empty: return trace.copy()
    return trace.loc[
        trace.relative_start_sec.ge(float(start)-1e-12)
        & trace.relative_end_sec.le(float(end)+1e-12)
    ].copy()


def _nonfloor_levels(frame):
    if frame.empty: return np.array([],dtype=float)
    values=pd.to_numeric(frame.loc[~frame.at_digital_floor.astype(bool),"ac_rms_dbfs"],errors="coerce")
    return values[np.isfinite(values)].to_numpy(float)


def _bounded_persistence(trace, floor_dbfs, parameters):
    horizon=parameters.persistence_horizon_ms/1000.0
    local=_select_relative(trace,0.0,horizon)
    if len(local)<parameters.minimum_persistence_frame_count:
        return np.nan, False
    within=pd.to_numeric(local.ac_rms_dbfs,errors="coerce").to_numpy(float) <= float(floor_dbfs)+parameters.persistence_threshold_db
    run=0
    for index,flag in enumerate(within):
        run=run+1 if bool(flag) else 0
        if run>=parameters.persistence_consecutive_frames:
            first=index-parameters.persistence_consecutive_frames+1
            return float(local.iloc[first].relative_mid_sec), False
    return float(horizon), True


def measure_boundary(waveform, fs, boundary, *, logical_recording_id, parameters=DEFAULT_PARAMETERS):
    row={
        "logical_recording_id": logical_recording_id,
        "boundary_id": f"{logical_recording_id}:primary-offset-{int(boundary['boundary_index']):04d}",
        **boundary,
        "tail_eligible":False,"persistence_eligible":False,"decay_eligible":False,
        "exclusion_reason":"","early_frame_count":0,"floor_frame_count":0,
        "persistence_frame_count":0,"decay_frame_count":0,"floor_frame_fraction":np.nan,
        "floor_dbfs":np.nan,"floor_iqr_db":np.nan,"floor_stable":False,
        "early_level_dbfs":np.nan,"tail_excess_100ms_db":np.nan,
        "tail_persistence_sec":np.nan,"tail_persistence_right_censored":False,
        "signed_decay_slope_db_per_sec":np.nan,"decay_dynamic_range_db":np.nan,
        "downward_decay_rate_db_per_sec":np.nan,"nondecreasing_decay":False,
    }
    pause=float(boundary["pause_duration_sec"])
    if pause < parameters.floor_end_ms/1000.0:
        row["exclusion_reason"]="pause_shorter_than_floor_window"; return row
    offset=float(boundary["speech_offset_sec"])
    trace=boundary_envelope_trace(waveform,fs,offset,boundary["pause_end_sec"],parameters=parameters)
    early=_select_relative(trace,parameters.early_tail_start_ms/1000,parameters.early_tail_end_ms/1000)
    floor_frames=_select_relative(trace,parameters.floor_start_ms/1000,parameters.floor_end_ms/1000)
    decay=_select_relative(trace,parameters.decay_start_ms/1000,parameters.decay_end_ms/1000)
    early_levels=_nonfloor_levels(early); floor_levels=_nonfloor_levels(floor_frames); decay_levels=_nonfloor_levels(decay)
    floor_fraction=float(floor_frames.at_digital_floor.astype(bool).mean()) if len(floor_frames) else np.nan
    row.update({"early_frame_count":len(early_levels),"floor_frame_count":len(floor_levels),"persistence_frame_count":len(trace),"decay_frame_count":len(decay_levels),"floor_frame_fraction":floor_fraction})
    if len(early_levels)<parameters.minimum_early_frame_count:
        row["exclusion_reason"]="insufficient_early_frames"; return row
    if len(floor_levels)<parameters.minimum_floor_frame_count:
        row["exclusion_reason"]="insufficient_nonfloor_floor_frames"; return row
    if np.isfinite(floor_fraction) and floor_fraction>parameters.maximum_floor_frame_fraction:
        row["exclusion_reason"]="digital_floor_censored"; return row
    floor_dbfs=float(np.median(floor_levels)); floor_iqr=float(np.quantile(floor_levels,.75)-np.quantile(floor_levels,.25))
    row.update({"floor_dbfs":floor_dbfs,"floor_iqr_db":floor_iqr,"floor_stable":floor_iqr<=parameters.maximum_floor_iqr_db,"early_level_dbfs":float(np.median(early_levels))})
    if not row["floor_stable"]:
        row["exclusion_reason"]="unstable_late_pause_floor"; return row
    row["tail_eligible"]=True
    row["tail_excess_100ms_db"]=row["early_level_dbfs"]-floor_dbfs
    if len(decay_levels)>=parameters.minimum_decay_frame_count:
        valid=decay.loc[~decay.at_digital_floor.astype(bool)].copy()
        valid=valid.loc[np.isfinite(pd.to_numeric(valid.ac_rms_dbfs,errors="coerce"))]
        times=valid.relative_mid_sec.to_numpy(float); levels=valid.ac_rms_dbfs.to_numpy(float)
        dynamic=float(np.quantile(levels,.90)-np.quantile(levels,.10))
        slope=float(stats.theilslopes(levels,times,alpha=.95).slope)
        row.update({"signed_decay_slope_db_per_sec":slope,"decay_dynamic_range_db":dynamic,"nondecreasing_decay":slope>=0})
        if slope<0 and dynamic>=parameters.minimum_decay_dynamic_range_db:
            row["decay_eligible"]=True; row["downward_decay_rate_db_per_sec"]=-slope
    if pause>=parameters.persistence_horizon_ms/1000.0:
        value,censored=_bounded_persistence(trace,floor_dbfs,parameters)
        if np.isfinite(value):
            row["persistence_eligible"]=True; row["tail_persistence_sec"]=value; row["tail_persistence_right_censored"]=bool(censored)
    return row


def estimate_srmr_working_set_mb(sample_count, parameters=DEFAULT_PARAMETERS):
    samples=max(0,int(sample_count)); gt_steps=int(ceil(samples/40.0))
    modulation_frames=max(1,int(ceil((gt_steps-ceil(.256*400.0))/ceil(.064*400.0)))+1)
    values=samples+parameters.srmr_n_cochlear_filters*gt_steps+8*gt_steps+parameters.srmr_n_cochlear_filters*8*modulation_frames
    return float(values*8.0*2.5/(1024.0**2))


def compute_srmr_norm(waveform, fs, *, parameters=DEFAULT_PARAMETERS):
    if int(fs)!=parameters.analysis_sample_rate_hz: raise ValueError("SRMR requires 16000 Hz")
    values=np.asarray(waveform,dtype=np.float64)
    if values.ndim!=1 or not len(values) or not np.isfinite(values).all(): raise ValueError("SRMR requires finite mono audio")
    try: installed=version("Gammatone")
    except PackageNotFoundError as exc: raise ModuleNotFoundError("Gammatone 1.0.3 is required") from exc
    if installed!=SRMR_GAMMATONE_VERSION: raise RuntimeError(f"Expected Gammatone {SRMR_GAMMATONE_VERSION}; found {installed}")
    from paper1_qc._vendor.srmrpy import srmr
    score,energy=srmr(values,int(fs),n_cochlear_filters=parameters.srmr_n_cochlear_filters,low_freq=parameters.srmr_low_frequency_hz,min_cf=parameters.srmr_min_modulation_cf_hz,max_cf=parameters.srmr_max_modulation_cf_hz,fast=parameters.srmr_fast,norm=parameters.srmr_normalized)
    result=float(score); del energy
    if not np.isfinite(result) or result<=0: raise RuntimeError(f"Invalid SRMR: {result}")
    return result


def _tier(count, parameters):
    if count>=parameters.high_boundary_count: return "high"
    if count>=parameters.moderate_boundary_count: return "moderate"
    if count>=2: return "minimum"
    return "unavailable"


def extract_qrev(waveform, fs, *, primary_speech, strict_speech, logical_recording_id="recording", parameters=DEFAULT_PARAMETERS, compute_srmr=True):
    raw_samples=np.asarray(waveform,dtype=np.float64)
    if raw_samples.ndim!=1: raise ValueError("QREV requires mono audio")
    if int(fs)!=parameters.analysis_sample_rate_hz: raise ValueError("QREV requires 16000 Hz")
    if not np.isfinite(raw_samples).all(): raise ValueError("Waveform contains non-finite samples")
    input_mean_before_dc_removal=float(np.mean(raw_samples)) if len(raw_samples) else 0.0
    samples=remove_global_dc(raw_samples)
    input_mean_after_dc_removal=float(np.mean(samples)) if len(samples) else 0.0
    duration=len(samples)/float(fs)
    primary=validate_speech_intervals(primary_speech,duration,required_view="primary_speech")
    strict=validate_speech_intervals(strict_speech,duration,required_view="strict_speech")
    boundaries=internal_pause_boundaries(primary,duration)
    rows=[measure_boundary(samples,fs,b,logical_recording_id=logical_recording_id,parameters=parameters) for b in boundaries]
    ledger=pd.DataFrame(rows)
    if ledger.empty:
        tail=pd.DataFrame(); persistence=pd.DataFrame(); decay=pd.DataFrame()
    else:
        tail=ledger.loc[ledger.tail_eligible.astype(bool)].copy()
        persistence=ledger.loc[ledger.persistence_eligible.astype(bool)].copy()
        decay=ledger.loc[ledger.decay_eligible.astype(bool)].copy()
    counts={"tail":len(tail),"persistence":len(persistence),"decay":len(decay)}
    support_pause_sec={
        "tail": float(pd.to_numeric(tail.get("pause_duration_sec", pd.Series(dtype=float)), errors="coerce").sum()) if len(tail) else 0.0,
        "persistence": float(pd.to_numeric(persistence.get("pause_duration_sec", pd.Series(dtype=float)), errors="coerce").sum()) if len(persistence) else 0.0,
        "decay": float(pd.to_numeric(decay.get("pause_duration_sec", pd.Series(dtype=float)), errors="coerce").sum()) if len(decay) else 0.0,
    }
    observed_persistence_sec = float(
        pd.to_numeric(persistence.get("tail_persistence_sec", pd.Series(dtype=float)), errors="coerce").sum()
    ) if len(persistence) else 0.0
    tail_raw=float(np.median(tail.tail_excess_100ms_db)) if len(tail) else np.nan
    persistence_raw=float(np.median(persistence.tail_persistence_sec)) if len(persistence) else np.nan
    decay_raw=float(np.median(decay.downward_decay_rate_db_per_sec)) if len(decay) else np.nan
    censored_fraction=float(persistence.tail_persistence_right_censored.astype(bool).mean()) if len(persistence) else np.nan
    recording_median_censored=bool(np.isfinite(persistence_raw) and np.isclose(persistence_raw,parameters.persistence_horizon_ms/1000.0))
    tail_available=counts["tail"]>=parameters.minimum_tail_boundary_count
    persistence_available=counts["persistence"]>=parameters.minimum_persistence_boundary_count
    decay_available=counts["decay"]>=parameters.minimum_decay_boundary_count

    primary_span=(primary[-1].end_sec-primary[0].start_sec) if primary else 0.0
    strict_support=float(sum(x.duration_sec for x in strict))
    left=max(0,int(floor(primary[0].start_sec*fs))) if primary else 0
    right=min(len(samples),int(ceil(primary[-1].end_sec*fs))) if primary else 0
    srmr_samples=max(0,right-left); srmr_memory=estimate_srmr_working_set_mb(srmr_samples,parameters)
    srmr_value=np.nan; srmr_status="not_requested"
    if compute_srmr:
        if strict_support<parameters.minimum_srmr_speech_support_sec or primary_span<parameters.minimum_srmr_task_span_sec: srmr_status="insufficient_support"
        elif srmr_memory>parameters.maximum_srmr_estimated_memory_mb: srmr_status="resource_limit"
        else:
            try: srmr_value=compute_srmr_norm(samples[left:right],fs,parameters=parameters); srmr_status="measured"
            except ModuleNotFoundError: srmr_status="dependency_unavailable"
            except (ValueError,RuntimeError,FloatingPointError): srmr_status="computation_failed"

    values={
        ANALYSIS_FEATURES[0]:tail_raw if tail_available else np.nan,
        ANALYSIS_FEATURES[1]:persistence_raw if persistence_available else np.nan,
        ANALYSIS_FEATURES[2]:decay_raw if decay_available else np.nan,
        ANALYSIS_FEATURES[3]:srmr_value,
    }
    statuses={
        ANALYSIS_FEATURES[0]:"measured" if tail_available else "insufficient_support",
        ANALYSIS_FEATURES[1]:"right_censored_at_horizon" if persistence_available and recording_median_censored else "measured" if persistence_available else "insufficient_support",
        ANALYSIS_FEATURES[2]:"measured" if decay_available else "no_valid_downward_decay" if counts["tail"]>=parameters.minimum_tail_boundary_count else "insufficient_support",
        ANALYSIS_FEATURES[3]:srmr_status,
    }
    recording={
        "logical_recording_id":logical_recording_id,
        "qrev_measurement_version":MEASUREMENT_VERSION,
        "qrev_signal_view":"mono_globally_dc_removed_16k_analysis; framewise_AC_RMS",
        "qrev_global_dc_removal_applied":True,
        "qrev_input_mean_before_dc_removal":input_mean_before_dc_removal,
        "qrev_input_mean_after_dc_removal":input_mean_after_dc_removal,
        "qrev_boundary_source_view":"primary_speech",
        "qrev_boundary_source_profile":"primary",
        "qrev_srmr_task_span_view":"primary_speech natural task span",
        "qrev_srmr_support_view":"strict_speech",
        "qrev_support_policy":"provisional_2_boundary_minimum_compare_2_3_4_before_G10",
        "qrev_internal_boundary_count":len(boundaries),
        "qrev_tail_valid_boundary_count":counts["tail"],
        "qrev_tail_valid_pause_support_sec":support_pause_sec["tail"],
        "qrev_persistence_valid_boundary_count":counts["persistence"],
        "qrev_persistence_valid_pause_support_sec":support_pause_sec["persistence"],
        "qrev_persistence_observed_duration_support_sec":observed_persistence_sec,
        "qrev_decay_valid_boundary_count":counts["decay"],
        "qrev_decay_valid_pause_support_sec":support_pause_sec["decay"],
        "qrev_persistence_right_censored_fraction":censored_fraction,
        "qrev_persistence_recording_median_censored":recording_median_censored,
        "qrev_srmr_variant":SRMR_VARIANT,
        "qrev_srmr_upstream_commit":SRMR_UPSTREAM_COMMIT,
        "qrev_srmr_primary_task_span_sec":primary_span,
        "qrev_srmr_strict_speech_support_sec":strict_support,
        "qrev_srmr_estimated_working_set_mb":srmr_memory,
        "qrev_tail_excess_100ms_db_raw_estimate":tail_raw,
        "qrev_tail_persistence_median_sec_raw_estimate":persistence_raw,
        "qrev_downward_decay_rate_db_per_sec_raw_estimate":decay_raw,
        **values,
    }
    for feature,count in zip(ANALYSIS_FEATURES[:3],[counts["tail"],counts["persistence"],counts["decay"]]):
        recording[f"{feature}_status"]=statuses[feature]
        recording[f"{feature}_support_tier"]=_tier(count,parameters)
    recording[f"{ANALYSIS_FEATURES[3]}_status"]=srmr_status
    recording[f"{ANALYSIS_FEATURES[3]}_support_tier"]="minimum" if srmr_status=="measured" else "unavailable"
    recording["qrev_primary_available_count"]=int(np.isfinite(values[ANALYSIS_FEATURES[0]])+np.isfinite(values[ANALYSIS_FEATURES[1]]))
    recording["qrev_family_status"]="all_primary_available" if recording["qrev_primary_available_count"]==2 else "partial_primary_available" if recording["qrev_primary_available_count"]==1 else "primary_unavailable"
    return QREVExtraction(recording,ledger)


def shift_intervals(intervals: Iterable[SpeechInterval], shift_sec: float) -> list[SpeechInterval]:
    """Apply a common time shift while preserving frozen interval identities."""
    delta=float(shift_sec)
    return [
        SpeechInterval(
            start_sec=float(item.start_sec)+delta,
            end_sec=float(item.end_sec)+delta,
            interval_id=item.interval_id,
            interval_index=item.interval_index,
            view=item.view,
            profile=item.profile,
        )
        for item in intervals
    ]


def apply_gain_db(waveform, gain_db):
    return np.asarray(waveform,dtype=np.float64)*(10.0**(float(gain_db)/20.0))
