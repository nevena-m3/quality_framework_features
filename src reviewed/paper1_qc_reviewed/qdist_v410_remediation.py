"""Blocking-remediation preflight for the QDIST v4.1.0 candidate.

This layer addresses the frame-grid and asymmetric-clipping failures found in
the proposed qdist-v4.0.0 package. It cannot freeze or publish a measurement.
Full-cohort rerun, real-speech injection, output inspection, and independent
blinded human/technical review remain mandatory.
"""
from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal

from paper1_qc import qdist_v410_candidate as detector


MEASUREMENT_VERSION = "qdist-v4.1.0-candidate"
VALIDATION_VERSION = "qdist-v4.1.0-remediation-preflight-r1"
ANALYSIS_FEATURES = detector.ANALYSIS_FEATURES
DIRECT_FEATURES = (
    "qdist_hard_clip_event_rate_per_min",
    "qdist_hard_clipped_sample_fraction",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)


def save_table(frame: pd.DataFrame, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)
    return path


def save_bundle(
    figure: plt.Figure,
    root: Path,
    stem: str,
    *,
    source: pd.DataFrame,
    caption: str,
    provenance: dict[str, Any],
) -> dict[str, str]:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "png": root / f"{stem}.png",
        "svg": root / f"{stem}.svg",
        "pdf": root / f"{stem}.pdf",
        "source_csv": root / f"{stem}.source.csv",
        "caption": root / f"{stem}.caption.md",
        "provenance": root / f"{stem}.provenance.json",
    }
    figure.savefig(paths["png"], dpi=300, bbox_inches="tight")
    figure.savefig(paths["svg"], bbox_inches="tight")
    figure.savefig(paths["pdf"], bbox_inches="tight")
    plt.close(figure)
    source.to_csv(paths["source_csv"], index=False)
    paths["caption"].write_text(caption.strip() + "\n", encoding="utf-8")
    write_json(paths["provenance"], provenance)
    return {name: str(path) for name, path in paths.items()}


def synthetic_speech_like(
    *,
    sample_rate_hz: int,
    duration_sec: float,
    seed: int,
) -> np.ndarray:
    fs = int(sample_rate_hz)
    n_samples = int(round(float(duration_sec) * fs))
    time = np.arange(n_samples, dtype=float) / fs
    rng = np.random.default_rng(int(seed))
    f0 = 118.0 + 18.0 * np.sin(2 * np.pi * 0.63 * time)
    f0 += 5.0 * np.sin(2 * np.pi * 1.37 * time + 0.4)
    phase = 2 * np.pi * np.cumsum(f0) / fs
    voiced = np.zeros(n_samples, dtype=float)
    phases = rng.uniform(-np.pi, np.pi, size=10)
    for harmonic in range(1, 11):
        voiced += np.sin(harmonic * phase + phases[harmonic - 1]) / harmonic**1.15
    syllabic = 0.22 + 0.78 * (
        0.5 + 0.5 * np.sin(2 * np.pi * 2.7 * time + 0.3)
    ) ** 1.4
    phrase = 0.75 + 0.25 * np.sin(2 * np.pi * 0.31 * time - 0.2)
    noise = rng.normal(size=n_samples)
    if fs > 6_000:
        sos = signal.butter(4, 2_000, btype="highpass", fs=fs, output="sos")
        noise = signal.sosfilt(sos, noise)
    waveform = syllabic * phrase * (voiced + 0.08 * noise)
    waveform += 3e-4 * rng.normal(size=n_samples)
    guard = min(int(round(0.25 * fs)), n_samples // 5)
    waveform[:guard] = 0.0
    waveform[-guard:] = 0.0
    peak = float(np.max(np.abs(waveform)))
    if not np.isfinite(peak) or peak <= 0:
        raise RuntimeError("Degenerate synthetic carrier")
    return 0.98 * waveform / peak


def analyze(waveform: np.ndarray, fs: int, recording_id: str):
    pcm = detector.quantize_pcm(np.asarray(waveform, dtype=float), 16)
    provenance = detector.NativeSignalProvenance(
        native_view_verified=True,
        known_preprocessing_applied=False,
        codec_name="pcm_s16le",
        sample_format="s16",
        bits_per_raw_sample=16,
    )
    return detector.extract_qdist(
        pcm,
        int(fs),
        logical_recording_id=str(recording_id),
        provenance=provenance,
    ), pcm


def feature_values(result) -> dict[str, float]:
    return {
        feature: float(result.recording[feature])
        for feature in ANALYSIS_FEATURES
    } | {
        "event_count": int(result.recording["qdist_hard_clip_event_count"]),
        "plateau_count": int(result.recording["qdist_accepted_plateau_count"]),
    }


def plateau_mask(result, n_samples: int) -> np.ndarray:
    mask = np.zeros(int(n_samples), dtype=bool)
    for row in result.accepted_plateau_ledger.itertuples(index=False):
        left = max(0, int(row.start_sample_task))
        right = min(len(mask), int(row.end_sample_task_exclusive))
        mask[left:right] = True
    return mask


def inject_clip(
    waveform: np.ndarray,
    limit: float,
    geometry: str,
) -> tuple[np.ndarray, np.ndarray]:
    output = np.asarray(waveform, dtype=float).copy()
    if geometry == "symmetric":
        truth = np.abs(output) > limit
        output = np.clip(output, -limit, limit)
    elif geometry == "positive_only":
        truth = output > limit
        output[truth] = limit
    elif geometry == "negative_only":
        truth = output < -limit
        output[truth] = -limit
    else:
        raise ValueError(geometry)
    return output, truth


def recovery_row(
    waveform: np.ndarray,
    truth: np.ndarray,
    *,
    fs: int,
    seed: int,
    geometry: str,
    limit: float,
) -> dict[str, Any]:
    result, pcm = analyze(waveform, fs, f"dose_{fs}_{seed}_{geometry}_{limit}")
    predicted = plateau_mask(result, len(pcm))
    truth = np.asarray(truth, dtype=bool)[: len(predicted)]
    tp = int(np.count_nonzero(predicted & truth))
    fp = int(np.count_nonzero(predicted & ~truth))
    fn = int(np.count_nonzero(~predicted & truth))
    precision = tp / (tp + fp) if tp + fp else np.nan
    recall = tp / (tp + fn) if tp + fn else np.nan
    return {
        "sample_rate_hz": fs,
        "seed": seed,
        "geometry": geometry,
        "limit": limit,
        "limit_dbfs": 20.0 * np.log10(limit),
        "true_clipped_sample_fraction": float(truth.mean()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "positive": bool(result.recording["qdist_hard_clip_event_count"] > 0),
        **feature_values(result),
    }


def dose_grid() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for fs in [8_000, 16_000, 22_050, 44_100, 48_000]:
        for seed in [5101, 5102, 5103]:
            carrier = synthetic_speech_like(
                sample_rate_hz=fs,
                duration_sec=4.0,
                seed=seed,
            )
            for limit in [0.90, 0.75, 0.60]:
                for geometry in ["symmetric", "positive_only", "negative_only"]:
                    clipped, truth = inject_clip(carrier, limit, geometry)
                    rows.append(
                        recovery_row(
                            clipped,
                            truth,
                            fs=fs,
                            seed=seed,
                            geometry=geometry,
                            limit=limit,
                        )
                    )
    return pd.DataFrame(rows)


def low_level_state_grid() -> pd.DataFrame:
    fs = 48_000
    source = synthetic_speech_like(sample_rate_hz=fs, duration_sec=6.0, seed=6201)
    midpoint = len(source) // 2
    rows: list[dict[str, Any]] = []
    for gain, limit in [(0.8, 0.4), (0.6, 0.3), (0.5, 0.25), (0.4, 0.2), (0.3, 0.15)]:
        waveform = source.copy()
        waveform[:midpoint] *= gain
        waveform[:midpoint] = np.clip(waveform[:midpoint], -limit, limit)
        result, _ = analyze(waveform, fs, f"state_{gain}_{limit}")
        accepted = result.accepted_plateau_ledger
        rows.append(
            {
                "gain_state": gain,
                "clip_limit": limit,
                "clip_limit_to_recording_peak": limit / np.max(np.abs(waveform)),
                "event_count": int(result.recording["qdist_hard_clip_event_count"]),
                "accepted_plateau_count": len(accepted),
                "magnitude_paths": (
                    "|".join(sorted(set(accepted["magnitude_path"].astype(str))))
                    if len(accepted)
                    else "none"
                ),
                **{
                    feature: result.recording[feature]
                    for feature in ANALYSIS_FEATURES
                },
            }
        )
    return pd.DataFrame(rows)


def smooth_compressor(waveform: np.ndarray, threshold: float, ratio: float) -> np.ndarray:
    samples = np.asarray(waveform, dtype=float)
    magnitude = np.abs(samples)
    compressed = np.where(
        magnitude <= threshold,
        magnitude,
        threshold + (magnitude - threshold) / ratio,
    )
    return np.sign(samples) * compressed


def discriminant_grid() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for fs in [8_000, 16_000, 48_000]:
        time = np.arange(4 * fs) / fs
        rng = np.random.default_rng(7300 + fs)
        controls: dict[str, np.ndarray] = {}
        for frequency in [80, 137, 440, 1_000]:
            controls[f"sine_{frequency}"] = 0.995 * np.sin(2 * np.pi * frequency * time)
            controls[f"triangle_{frequency}"] = 0.95 * signal.sawtooth(
                2 * np.pi * frequency * time,
                width=0.5,
            )
            controls[f"square_{frequency}"] = 0.80 * signal.square(
                2 * np.pi * frequency * time
            )
        speech = synthetic_speech_like(sample_rate_hz=fs, duration_sec=4.0, seed=7401)
        controls["clean_speech"] = speech
        controls["dc_offset_speech"] = 0.72 * speech + 0.14
        controls["broadband_noise"] = np.clip(rng.normal(0, 0.2, len(time)), -0.95, 0.95)
        impulse = np.zeros_like(time)
        impulse[len(impulse) // 2] = 0.99
        controls["single_impulse"] = impulse
        click_train = np.zeros_like(time)
        click_train[:: max(1, fs // 2)] = 0.99
        controls["click_train"] = click_train
        for drive in [1.0, 2.0, 4.0, 8.0]:
            controls[f"smooth_tanh_{drive}"] = np.tanh(drive * speech) / np.tanh(drive)
        for ratio in [2.0, 4.0, 10.0]:
            controls[f"compressor_{ratio}"] = smooth_compressor(speech, 0.50, ratio)
        for name, waveform in controls.items():
            result, _ = analyze(waveform, fs, f"control_{fs}_{name}")
            rows.append(
                {
                    "sample_rate_hz": fs,
                    "control": name,
                    "positive": bool(result.recording["qdist_hard_clip_event_count"] > 0),
                    **feature_values(result),
                }
            )
    return pd.DataFrame(rows)


def _translated_frame_fraction(
    ledger: pd.DataFrame,
    *,
    offset_samples: int,
    frame_length_samples: int,
    complete_frame_count: int,
) -> float:
    affected = np.zeros(complete_frame_count, dtype=bool)
    for row in ledger.itertuples(index=False):
        start = int(row.start_sample_task) + offset_samples
        end = int(row.end_sample_task_exclusive) + offset_samples
        first = max(0, start // frame_length_samples)
        last = min(
            complete_frame_count - 1,
            max(start, end - 1) // frame_length_samples,
        )
        if last >= first:
            affected[first : last + 1] = True
    return float(affected.mean())


def frame_phase_grid() -> tuple[pd.DataFrame, dict[str, Any]]:
    fs = 48_000
    carrier = synthetic_speech_like(sample_rate_hz=fs, duration_sec=6.0, seed=4101)
    clipped, _ = inject_clip(carrier, 0.65, "symmetric")
    baseline, _ = analyze(clipped, fs, "phase_baseline")
    frame_length = int(baseline.recording["qdist_frame_length_samples"])
    complete_frames = int(baseline.recording["qdist_complete_frame_count"])
    rows = [
        {
            "offset_samples": offset,
            "offset_ms": 1000.0 * offset / fs,
            "frame_fraction": _translated_frame_fraction(
                baseline.accepted_plateau_ledger,
                offset_samples=offset,
                frame_length_samples=frame_length,
                complete_frame_count=complete_frames,
            ),
        }
        for offset in range(frame_length)
    ]
    grid = pd.DataFrame(rows)
    changed = grid.loc[
        ~np.isclose(
            grid["frame_fraction"],
            float(baseline.recording["qdist_hard_clipped_frame_fraction"]),
            rtol=0,
            atol=1e-15,
        )
    ]
    confirmation_offset = int(changed.iloc[0]["offset_samples"])
    shifted = np.concatenate(
        [np.zeros(confirmation_offset), clipped[:-confirmation_offset]]
    )
    translated, _ = analyze(shifted, fs, "phase_shifted")
    summary = {
        "frame_length_samples": frame_length,
        "confirmation_offset_samples": confirmation_offset,
        "confirmation_offset_ms": 1000.0 * confirmation_offset / fs,
        "baseline": feature_values(baseline),
        "shifted": feature_values(translated),
        "minimum_frame_fraction": float(grid["frame_fraction"].min()),
        "maximum_frame_fraction": float(grid["frame_fraction"].max()),
        "distinct_frame_fractions": int(grid["frame_fraction"].nunique()),
    }
    return grid, summary


def transformation_grid() -> pd.DataFrame:
    fs = 48_000
    carrier = synthetic_speech_like(sample_rate_hz=fs, duration_sec=4.0, seed=5103)
    negative, _ = inject_clip(carrier, 0.60, "negative_only")
    cases = {"negative_only": negative, "polarity_inverted": -negative}
    for gain in [0.25, 0.50, 0.75, 1.00]:
        cases[f"post_clip_gain_{gain}"] = gain * negative
    rows = []
    for condition, waveform in cases.items():
        result, _ = analyze(waveform, fs, condition)
        rows.append({"condition": condition, **feature_values(result)})
    return pd.DataFrame(rows)


def reconstruction_audit() -> pd.DataFrame:
    fs = 48_000
    carrier = synthetic_speech_like(sample_rate_hz=fs, duration_sec=4.0, seed=5103)
    clipped, _ = inject_clip(carrier, 0.60, "negative_only")
    result, _ = analyze(clipped, fs, "reconstruction")
    rebuilt = detector.reconstruct_qdist_features(
        result.accepted_plateau_ledger,
        result.episode_ledger,
        finite_channel_sample_count=result.recording["qdist_finite_channel_sample_count"],
        finite_time_sample_count=result.recording["qdist_finite_time_sample_count"],
        finite_exposure_sec=result.recording["qdist_task_span_duration_sec"],
        frame_length_samples=result.recording["qdist_frame_length_samples"],
        complete_frame_count=result.recording["qdist_complete_frame_count"],
    )
    return pd.DataFrame(
        [
            {
                "feature": feature,
                "stored": float(result.recording[feature]),
                "reconstructed": float(rebuilt[feature]),
                "absolute_error": abs(
                    float(result.recording[feature]) - float(rebuilt[feature])
                ),
            }
            for feature in ANALYSIS_FEATURES
        ]
    )


def validation_checks(
    dose: pd.DataFrame,
    controls: pd.DataFrame,
    transformations: pd.DataFrame,
    phase_summary: dict[str, Any],
    reconstruction: pd.DataFrame,
    low_level: pd.DataFrame,
) -> pd.DataFrame:
    moderate = dose.loc[dose["true_clipped_sample_fraction"] >= 0.001]
    detected = moderate.loc[moderate["positive"]]
    negative_row = transformations.loc[
        transformations["condition"].eq("negative_only")
    ].iloc[0]
    inverted_row = transformations.loc[
        transformations["condition"].eq("polarity_inverted")
    ].iloc[0]
    direct_shift_equal = all(
        np.isclose(
            phase_summary["baseline"][feature],
            phase_summary["shifted"][feature],
            rtol=0,
            atol=1e-15,
        )
        for feature in DIRECT_FEATURES
    )
    polarity_equal = all(
        np.isclose(
            float(negative_row[feature]),
            float(inverted_row[feature]),
            rtol=0,
            atol=1e-15,
        )
        for feature in ANALYSIS_FEATURES
    )
    low_path_present = low_level["magnitude_paths"].str.contains(
        "repeated_low_level_saturation",
        regex=False,
    ).any()
    return pd.DataFrame(
        [
            {
                "gate": "G1",
                "check": "candidate measurement identity and corrected roles",
                "passed": detector.MEASUREMENT_VERSION == MEASUREMENT_VERSION
                and detector.PRIMARY_FEATURES
                == ("qdist_hard_clipped_sample_fraction",),
                "observed": detector.MEASUREMENT_VERSION,
                "required": MEASUREMENT_VERSION,
            },
            {
                "gate": "G2",
                "check": "all feature values reconstruct from candidate ledgers",
                "passed": bool(reconstruction["absolute_error"].le(1e-15).all()),
                "observed": reconstruction.to_dict("records"),
                "required": "absolute error <=1e-15",
            },
            {
                "gate": "G3",
                "check": "one-sided polarity inversion is exact",
                "passed": polarity_equal,
                "observed": transformations.loc[
                    transformations["condition"].isin(
                        ["negative_only", "polarity_inverted"]
                    )
                ].to_dict("records"),
                "required": "all three outputs exact",
            },
            {
                "gate": "G3",
                "check": "direct burden and event rate survive arbitrary common shift",
                "passed": direct_shift_equal,
                "observed": phase_summary,
                "required": "event rate and sample fraction exact",
            },
            {
                "gate": "G3",
                "check": "frame-grid phase dependence is detected and prevents primary role",
                "passed": phase_summary["distinct_frame_fractions"] > 1
                and detector.CONDITIONAL_FEATURES
                == ("qdist_hard_clipped_frame_fraction",),
                "observed": phase_summary,
                "required": "sensitivity visible; feature conditional",
            },
            {
                "gate": "G4",
                "check": "all moderate known-truth doses are detected",
                "passed": bool(len(moderate) and moderate["positive"].all()),
                "observed": {
                    "rows": len(moderate),
                    "positive": int(moderate["positive"].sum()),
                },
                "required": "all rows with true burden >=0.001 positive",
            },
            {
                "gate": "G4",
                "check": "detected moderate doses have sample precision >=0.99",
                "passed": bool(len(detected) and detected["precision"].min() >= 0.99),
                "observed": float(detected["precision"].min()),
                "required": ">=0.99",
            },
            {
                "gate": "G4",
                "check": "repeated low-level saturation path is exercised",
                "passed": bool(low_path_present),
                "observed": low_level.to_dict("records"),
                "required": "at least one accepted repeated_low_level_saturation case",
            },
            {
                "gate": "G5",
                "check": "periodic, clean, impulse/noise, smooth saturation and compressor controls are negative",
                "passed": bool(~controls["positive"].astype(bool).any()),
                "observed": controls.loc[
                    controls["positive"], ["sample_rate_hz", "control"]
                ].to_dict("records"),
                "required": "no synthetic-control positives",
            },
        ]
    )


def figure_bundles(
    output: Path,
    *,
    dose: pd.DataFrame,
    controls: pd.DataFrame,
    transformations: pd.DataFrame,
    phase: pd.DataFrame,
    low_level: pd.DataFrame,
    provenance: dict[str, Any],
) -> pd.DataFrame:
    figures = output / "figures"
    rows: list[dict[str, Any]] = []

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    for geometry, local in dose.groupby("geometry", sort=True):
        axes[0, 0].scatter(
            local["true_clipped_sample_fraction"],
            local["qdist_hard_clipped_sample_fraction"],
            s=18,
            alpha=0.65,
            label=geometry,
        )
        axes[0, 1].scatter(
            local["true_clipped_sample_fraction"],
            local["recall"],
            s=18,
            alpha=0.65,
            label=geometry,
        )
    maximum = max(
        dose["true_clipped_sample_fraction"].max(),
        dose["qdist_hard_clipped_sample_fraction"].max(),
    )
    axes[0, 0].plot([0, maximum], [0, maximum], "--", color="black", linewidth=1)
    axes[0, 0].set(
        xlabel="True clipped-sample fraction",
        ylabel="Detected channel-sample fraction",
        title="Known-truth burden recovery",
    )
    axes[0, 0].legend()
    axes[0, 1].axvline(0.001, linestyle="--", color="black", linewidth=1)
    axes[0, 1].set(
        xlabel="True clipped-sample fraction",
        ylabel="Sample recall",
        ylim=(-0.03, 1.03),
        title="Detection limit by clipping geometry",
    )
    summary = (
        dose.groupby(["sample_rate_hz", "geometry"], as_index=False)
        .agg(positive_fraction=("positive", "mean"))
    )
    for geometry, local in summary.groupby("geometry", sort=True):
        axes[1, 0].plot(
            local["sample_rate_hz"],
            local["positive_fraction"],
            marker="o",
            label=geometry,
        )
    axes[1, 0].set(
        xlabel="Native sample rate (Hz)",
        ylabel="Positive fraction across dose rows",
        ylim=(-0.03, 1.03),
        title="Source-rate and polarity coverage",
    )
    axes[1, 0].legend()
    axes[1, 1].plot(
        low_level["clip_limit_to_recording_peak"],
        low_level["event_count"],
        marker="o",
    )
    axes[1, 1].axvline(
        detector.DEFAULT_PARAMETERS.candidate_generation_minimum_edge_to_robust_peak_ratio,
        linestyle="--",
        color="black",
        linewidth=1,
    )
    axes[1, 1].set(
        xlabel="Clip rail / recording robust-peak proxy",
        ylabel="Accepted episode count",
        title="Repeated low-level path and candidate floor",
    )
    a_source = pd.concat(
        [dose.assign(source_section="dose"), low_level.assign(source_section="low_level")],
        ignore_index=True,
        sort=False,
    )
    paths = save_bundle(
        fig,
        figures,
        "qdist_v410_panel-A_construct-response",
        source=a_source,
        caption=(
            "QDIST v4.1.0 candidate construct response. Known-truth symmetric and "
            "one-sided hard clipping is evaluated across five native sample rates and "
            "three carriers. Direct sample burden, recall, source-rate coverage, and the "
            "prespecified low-level candidate floor are shown. Because accepted runs must "
            "satisfy minimum-duration and morphology rules, detected burden is a conservative "
            "subset of all synthetically rail-limited samples. The dashed vertical line marks "
            "the preflight's 0.001 known-burden coverage target, not a validated clinical "
            "operating point. This synthetic preflight does not replace held-out real-speech "
            "injection."
        ),
        provenance={**provenance, "panel": "A"},
    )
    rows.append({"panel": "A", "stem": "qdist_v410_panel-A_construct-response", **paths})

    control_summary = (
        controls.groupby("control", as_index=False)
        .agg(
            tested_rates=("sample_rate_hz", "nunique"),
            positive_count=("positive", "sum"),
            maximum_event_count=("event_count", "max"),
        )
        .sort_values(["positive_count", "control"])
    )
    control_summary["negative_fraction"] = (
        1.0 - control_summary["positive_count"] / control_summary["tested_rates"]
    )
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), constrained_layout=True)
    axes[0].barh(
        control_summary["control"],
        control_summary["negative_fraction"],
        color="#2a7f62",
    )
    for index, row in control_summary.reset_index(drop=True).iterrows():
        axes[0].text(
            0.97,
            index,
            f"{int(row['tested_rates'] - row['positive_count'])}/{int(row['tested_rates'])}",
            ha="right",
            va="center",
            color="white",
            fontsize=7,
        )
    axes[0].set(
        xlabel="Negative fraction across tested sample rates",
        xlim=(0, 1.02),
        title="Synthetic discriminant rejection",
    )
    axes[1].scatter(
        control_summary["maximum_event_count"],
        control_summary["control"],
        marker="|",
        s=140,
        color="#8c2d3e",
    )
    maximum_control_events = int(control_summary["maximum_event_count"].max())
    axes[1].set_xlim(-0.1, max(1.0, maximum_control_events + 0.5))
    axes[1].axvline(0, color="black", linewidth=0.8)
    axes[1].text(
        0.02,
        0.985,
        f"All {len(controls)} rate-condition rows: 0 accepted episodes",
        transform=axes[1].transAxes,
        ha="left",
        va="top",
        fontsize=8,
    )
    axes[1].set(
        xlabel="Maximum accepted episode count",
        title="Worst control response",
    )
    axes[1].set_title("Worst control response", pad=14)
    paths = save_bundle(
        fig,
        figures,
        "qdist_v410_panel-B_discriminant-specificity",
        source=controls,
        caption=(
            "QDIST v4.1.0 candidate synthetic discriminant sweep across native rates. "
            "Periodic signals, clean speech-like carriers, impulses, clicks, noise, DC "
            "offset, smooth tanh saturation, and non-flat compression are controls. Real "
            "ALS/dysarthric, plosive, music, and neighboring-family controls remain required."
        ),
        provenance={**provenance, "panel": "B"},
    )
    rows.append({"panel": "B", "stem": "qdist_v410_panel-B_discriminant-specificity", **paths})

    fig, axes = plt.subplots(2, 2, figsize=(14, 8.5), constrained_layout=True)
    axes[0, 0].plot(phase["offset_ms"], phase["frame_fraction"])
    axes[0, 0].set(
        xlabel="Frame-grid origin offset (ms)",
        ylabel="Hard-clipped frame fraction",
        title="Declared grid-phase sensitivity",
    )
    base = transformations.loc[transformations["condition"].eq("negative_only")].iloc[0]
    deviations = transformations.copy()
    for feature in ANALYSIS_FEATURES:
        deviations[f"delta_{feature}"] = deviations[feature] - float(base[feature])

    axes[0, 1].axis("off")
    transformation_summary = []
    for feature in ANALYSIS_FEATURES:
        transformation_summary.append(
            [
                feature.replace("qdist_", ""),
                f"{float(base[feature]):.8g}",
                f"{float(deviations[f'delta_{feature}'].abs().max()):.3g}",
            ]
        )
    table = axes[0, 1].table(
        cellText=transformation_summary,
        colLabels=["Feature", "Baseline", "max |change|"],
        cellLoc="center",
        loc="center",
        colWidths=[0.55, 0.22, 0.22],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.5)
    axes[0, 1].set_title("Exact polarity/gain transformation audit")

    baseline_phase = float(phase.loc[phase["offset_samples"].eq(0), "frame_fraction"].iloc[0])
    confirmation_offset = min(261, int(phase["offset_samples"].max()))
    shifted_phase = float(
        phase.loc[
            phase["offset_samples"].eq(confirmation_offset), "frame_fraction"
        ].iloc[0]
    )
    baseline_values = np.array(
        [
            baseline_phase,
            float(base["qdist_hard_clip_event_rate_per_min"]),
            float(base["qdist_hard_clipped_sample_fraction"]),
        ]
    )
    shifted_values = np.array(
        [
            shifted_phase,
            float(base["qdist_hard_clip_event_rate_per_min"]),
            float(base["qdist_hard_clipped_sample_fraction"]),
        ]
    )
    relative = shifted_values / baseline_values
    short_names = ["30-ms frame\nfraction", "event rate", "sample\nfraction"]
    axes[1, 0].bar(short_names, relative, color=["#b04a5a", "#2a7f62", "#2a7f62"])
    axes[1, 0].axhline(1.0, linestyle="--", color="black", linewidth=1)
    axes[1, 0].set_ylim(0.94, 1.01)
    axes[1, 0].set(
        ylabel="Shifted / baseline value",
        title=f"Common {confirmation_offset / 48:.3f}-ms shift confirmation",
    )
    for index, value in enumerate(relative):
        axes[1, 0].text(index, value - 0.002, f"{value:.4f}", ha="center", va="top")
    phase_counts = phase.groupby("frame_fraction", as_index=False).size()
    axes[1, 1].bar(phase_counts["frame_fraction"].astype(str), phase_counts["size"])
    axes[1, 1].set(
        xlabel="Distinct frame fraction",
        ylabel="Grid origins",
        title="Frame-origin distribution",
    )
    c_source = pd.concat(
        [phase.assign(source_section="frame_phase"), transformations.assign(source_section="transformations")],
        ignore_index=True,
        sort=False,
    )
    paths = save_bundle(
        fig,
        figures,
        "qdist_v410_panel-C_transformation-contract",
        source=c_source,
        caption=(
            "QDIST v4.1.0 candidate transformation contract. Same-polarity prominence "
            "restores exact one-sided polarity inversion and uniform-gain behavior for "
            "direct burden/event views. The legacy 30-ms frame fraction is explicitly "
            "grid-phase dependent and is therefore conditional, not invariant."
        ),
        provenance={**provenance, "panel": "C"},
    )
    rows.append({"panel": "C", "stem": "qdist_v410_panel-C_transformation-contract", **paths})
    return pd.DataFrame(rows)


def run_remediation_preflight(output_root: str | Path) -> dict[str, Any]:
    output = Path(output_root)
    for name in ["tables", "validation", "figures", "manifests"]:
        (output / name).mkdir(parents=True, exist_ok=True)

    dose = dose_grid()
    controls = discriminant_grid()
    low_level = low_level_state_grid()
    phase, phase_summary = frame_phase_grid()
    transformations = transformation_grid()
    reconstruction = reconstruction_audit()
    checks = validation_checks(
        dose,
        controls,
        transformations,
        phase_summary,
        reconstruction,
        low_level,
    )

    save_table(dose, output / "tables" / "qdist_v410_dose_grid.csv")
    save_table(controls, output / "tables" / "qdist_v410_discriminant_grid.csv")
    save_table(low_level, output / "tables" / "qdist_v410_low_level_state_grid.csv")
    save_table(phase, output / "tables" / "qdist_v410_frame_phase_grid.csv")
    save_table(transformations, output / "tables" / "qdist_v410_transformations.csv")
    save_table(reconstruction, output / "validation" / "qdist_v410_reconstruction.csv")
    save_table(checks, output / "validation" / "qdist_v410_remediation_checks.csv")
    write_json(output / "validation" / "qdist_v410_frame_phase_summary.json", phase_summary)

    detector_path = Path(detector.__file__).resolve()
    module_path = Path(__file__).resolve()
    provenance = {
        "created_utc": utc_now(),
        "measurement_version": MEASUREMENT_VERSION,
        "validation_version": VALIDATION_VERSION,
        "detector_path": str(detector_path),
        "detector_sha256": sha256_file(detector_path),
        "validation_module_path": str(module_path),
        "validation_module_sha256": sha256_file(module_path),
        "parameter_hash": detector.DEFAULT_PARAMETERS.parameter_hash(),
        "clinical_labels_used": False,
        "human_qc_labels_used": False,
        "feature_values_from_cohort_recomputed": False,
        "candidate_only": True,
    }
    figure_index = figure_bundles(
        output,
        dose=dose,
        controls=controls,
        transformations=transformations,
        phase=phase,
        low_level=low_level,
        provenance=provenance,
    )
    save_table(figure_index, output / "tables" / "qdist_v410_figure_index_preflight.csv")

    manifest = {
        **provenance,
        "candidate_only": True,
        "freeze_allowed": False,
        "blocking_remediation_checks_pass": bool(checks["passed"].astype(bool).all()),
        "cohort_rerun_required": True,
        "real_speech_injection_required": True,
        "independent_blinded_human_review_required": True,
        "external_output_inspection_required": True,
        "panel_count": len(figure_index),
        "panels": figure_index["panel"].tolist(),
        "analysis_features": list(ANALYSIS_FEATURES),
        "primary_features": list(detector.PRIMARY_FEATURES),
        "secondary_features": list(detector.SECONDARY_FEATURES),
        "conditional_features": list(detector.CONDITIONAL_FEATURES),
        "prohibited_action": "No freeze, publication export, or overwrite from remediation preflight.",
    }
    write_json(
        output / "manifests" / "qdist_v410_remediation_preflight_manifest.json",
        manifest,
    )
    return {
        "manifest": manifest,
        "checks": checks,
        "dose": dose,
        "controls": controls,
        "low_level": low_level,
        "phase": phase,
        "phase_summary": phase_summary,
        "transformations": transformations,
        "reconstruction": reconstruction,
        "figure_index": figure_index,
    }
