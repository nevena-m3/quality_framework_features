from __future__ import annotations

import pandas as pd


# "Primary" means pre-specified for the paper's family-level characterization. It does
# not imply that measures form a reflective latent scale or may be averaged.
METRIC_REGISTRY = [
    # Additive interference
    dict(feature="qadd_nonspeech_level_dbfs", family="additive_interference", role="primary", unit="dBFS", worse="higher", signal_region="guarded internal nonspeech", minimum_support="0.5 s", interpretation="background/interference level; floor-limited", confounding="breathing, leakage from speech, room tone"),
    dict(feature="qadd_snr_proxy_db", family="additive_interference", role="primary", unit="dB", worse="lower", signal_region="strict speech vs guarded internal nonspeech", minimum_support="3 s speech and 0.5 s nonspeech", interpretation="within-recording speech-to-background level contrast", confounding="dysarthric intensity, mic distance, VAD error"),
    dict(feature="qadd_nonspeech_variability_db", family="additive_interference", role="primary", unit="dB", worse="higher", signal_region="guarded internal nonspeech", minimum_support="20 frames", interpretation="robust IQR of nonspeech frame level", confounding="breathing and other legitimate nonspeech"),
    dict(feature="qadd_hum_prominence_db", family="additive_interference", role="primary", unit="dB", worse="higher", signal_region="guarded internal nonspeech", minimum_support="1 s", interpretation="maximum 50/60-Hz harmonic prominence above adjacent local spectrum", confounding="very short pauses, low-frequency biological noise"),
    dict(feature="qadd_transient_rate_per_min", family="additive_interference", role="primary", unit="events/min", worse="higher", signal_region="guarded internal nonspeech", minimum_support="0.5 s", interpretation="exposure-normalized high-energy nonspeech transients", confounding="coughs, breath, table/contact sounds"),
    dict(feature="qadd_spectral_flatness", family="additive_interference", role="secondary", unit="ratio", worse="contextual", signal_region="guarded internal nonspeech", minimum_support="1 s", interpretation="geometric/arithmetic mean PSD ratio", confounding="noise type; not an ordinal severity measure"),
    # Gain/amplitude dynamics
    dict(feature="qgain_active_level_dbfs", family="gain_dynamics", role="primary_descriptor", unit="dBFS", worse="contextual", signal_region="strict speech", minimum_support="3 s", interpretation="median active-speech frame level", confounding="speech physiology, mic distance, device gain"),
    dict(feature="qgain_level_iqr_db", family="gain_dynamics", role="primary", unit="dB", worse="higher", signal_region="strict speech", minimum_support="100 frames", interpretation="robust within-recording speech-level spread", confounding="prosody and dysarthria"),
    dict(feature="qgain_segment_sd_db", family="gain_dynamics", role="primary", unit="dB", worse="higher", signal_region="strict speech segments", minimum_support="3 segments", interpretation="between-segment level variability", confounding="linguistic/prosodic variation"),
    dict(feature="qgain_abs_drift_db_per_min", family="gain_dynamics", role="primary", unit="dB/min", worse="higher", signal_region="strict speech in original time", minimum_support="10 s span", interpretation="absolute robust level drift", confounding="posture, mic distance, fatigue"),
    dict(feature="qgain_step_rate_per_min", family="gain_dynamics", role="secondary", unit="events/min", worse="higher", signal_region="adjacent speech frames within segments", minimum_support="3 s", interpretation="large local level-change rate", confounding="plosives, prosody, segmentation"),
    dict(feature="qgain_crest_factor_db", family="gain_dynamics", role="secondary", unit="dB", worse="contextual", signal_region="strict speech", minimum_support="3 s", interpretation="waveform peak-to-RMS ratio", confounding="phonation and articulation"),
    # Reverberation tail
    dict(feature="qrev_tail_excess_db", family="reverberation_tail", role="primary", unit="dB", worse="higher", signal_region="speech offsets followed by guarded pauses", minimum_support="3 offsets", interpretation="early post-offset level above late-pause floor", confounding="VAD boundary error, breath/noise after offset"),
    dict(feature="qrev_decay_time_proxy_sec", family="reverberation_tail", role="primary", unit="s", worse="higher", signal_region="speech offsets followed by guarded pauses", minimum_support="3 offsets", interpretation="time to fall within 3 dB of late-pause floor; right-censored", confounding="noise floor and pause duration"),
    dict(feature="qrev_decay_slope_db_per_sec", family="reverberation_tail", role="secondary", unit="dB/s", worse="higher", signal_region="first 300 ms after speech offset", minimum_support="3 offsets", interpretation="median robust decay slope; values nearer zero are slower", confounding="floor effects and boundary error"),
    dict(feature="qrev_srmr", family="reverberation_tail", role="secondary_optional", unit="ratio", worse="lower", signal_region="recording", minimum_support="implementation-defined", interpretation="non-intrusive speech-to-reverberation modulation energy ratio", confounding="phonation, speaking rate, noise; optional dependency"),
    # Channel/device descriptors
    dict(feature="qchan_effective_bandwidth_hz", family="channel_device", role="primary_descriptor", unit="Hz", worse="lower", signal_region="strict speech", minimum_support="3 s", interpretation="frequency containing 99% of long-term speech spectral power", confounding="speaker spectrum and source physiology"),
    dict(feature="qchan_highband_ratio", family="channel_device", role="primary_descriptor", unit="ratio", worse="contextual", signal_region="strict speech", minimum_support="3 s", interpretation="power above 3 kHz divided by 0.1-Nyquist power", confounding="speaker spectrum, frication, sample rate"),
    dict(feature="qchan_spectral_tilt_db_per_oct", family="channel_device", role="secondary", unit="dB/octave", worse="contextual", signal_region="strict speech", minimum_support="3 s", interpretation="robust LTAS slope from 300 Hz to available upper band", confounding="voice quality and disease"),
    # Nonlinear distortion
    dict(feature="qdist_hard_clip_sample_fraction", family="nonlinear_distortion", role="primary", unit="fraction", worse="higher", signal_region="native strict speech", minimum_support="3 s", interpretation="samples in sustained plateaus at the observed digital edge", confounding="lossy-codec ringing can hide clipping"),
    dict(feature="qdist_clip_event_rate_per_min", family="nonlinear_distortion", role="primary", unit="events/min", worse="higher", signal_region="native strict speech", minimum_support="3 s", interpretation="exposure-normalized edge-plateau events", confounding="quantization and codec"),
    dict(feature="qdist_clipped_frame_fraction", family="nonlinear_distortion", role="primary", unit="fraction", worse="higher", signal_region="native strict speech", minimum_support="100 frames", interpretation="speech frames containing a hard-clip event", confounding="codec and very high natural peaks"),
    dict(feature="qdist_near_fullscale_fraction", family="nonlinear_distortion", role="secondary", unit="fraction", worse="contextual", signal_region="native strict speech", minimum_support="3 s", interpretation="samples at or above 0.98 full scale; not sufficient evidence of clipping", confounding="legitimate peaks"),
    dict(feature="qdist_edge_histogram_spike", family="nonlinear_distortion", role="secondary", unit="ratio", worse="higher", signal_region="native strict speech", minimum_support="3 s", interpretation="occupancy at extreme amplitude bins relative to neighboring bins", confounding="sample depth and codec"),
    # Temporal discontinuity
    dict(feature="qtemp_zero_dropout_fraction", family="temporal_discontinuity", role="primary", unit="fraction", worse="higher", signal_region="strict speech in original time", minimum_support="3 s", interpretation="speech samples contained in near-zero runs of at least 10 ms", confounding="VAD error and truly unvoiced intervals"),
    dict(feature="qtemp_zero_dropout_rate_per_min", family="temporal_discontinuity", role="primary", unit="events/min", worse="higher", signal_region="strict speech in original time", minimum_support="3 s", interpretation="exposure-normalized near-zero runs", confounding="VAD error"),
    dict(feature="qtemp_duplicate_window_rate_per_min", family="temporal_discontinuity", role="primary", unit="events/min", worse="higher", signal_region="contiguous strict-speech intervals", minimum_support="3 s", interpretation="near-identical adjacent non-silent 20-ms windows", confounding="periodic voiced speech; threshold intentionally stringent"),
    dict(feature="qtemp_energy_jump_rate_per_min", family="temporal_discontinuity", role="secondary", unit="events/min", worse="higher", signal_region="adjacent frames within strict-speech intervals", minimum_support="3 s", interpretation="frame-level changes >=18 dB", confounding="plosives, prosody, gain changes"),
    dict(feature="qtemp_continuity_break_rate_per_min", family="temporal_discontinuity", role="secondary", unit="events/min", worse="higher", signal_region="within contiguous strict-speech intervals", minimum_support="3 s", interpretation="robust sample-difference outliers; interval boundaries excluded", confounding="impulsive speech sounds"),
]


def metric_registry_frame() -> pd.DataFrame:
    frame = pd.DataFrame(METRIC_REGISTRY)
    if frame["feature"].duplicated().any():
        raise ValueError("Metric registry contains duplicate feature names")
    return frame


def features_for_role(prefix: str = "primary") -> list[str]:
    return [row["feature"] for row in METRIC_REGISTRY if row["role"].startswith(prefix)]

