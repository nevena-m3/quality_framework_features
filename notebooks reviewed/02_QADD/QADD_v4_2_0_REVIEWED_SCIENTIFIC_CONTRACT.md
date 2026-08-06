# QADD v4.2.0 Reviewed Scientific Contract (Candidate)

## Family identity

**Family:** QADD  
**Reviewed candidate version:** `qadd-v4.2.0-candidate`  
**Recommended display name:** *Recorded pause-region additive-interference manifestations*  

QADD quantifies observable digital energy and spectral structure in guarded internal nonspeech regions, plus one explicitly mixed speech-pause contrast. It does not identify a physical source, estimate calibrated SPL, or recover physical SNR from a single remote recording.

## Frozen input views

- Comparable mono, DC-removed, deterministically resampled 16-kHz analysis waveform.
- No peak/loudness normalization, denoising, interpolation, compression, or AGC.
- `profile == primary`, `decision == KEEP`, and `segmentation_analysis_eligible == True`.
- Pause measurements: canonical `strict_internal_nonspeech` intervals, already carrying the frozen 200-ms edge erosion; minimum residual pause support remains feature-specific.
- Speech contrast: canonical `strict_speech` intervals directly. **No second 50-ms edge erosion is permitted.**
- Source view/profile/interval identity and freeze hashes must be preserved in ledgers.

## Retained candidate feature vector

1. `qadd_pause_ac_level_dbfs_median` - primary contextual recorded pause-energy measurement.
2. `qadd_pause_level_iqr_db` - secondary nonstationarity descriptor.
3. `qadd_speech_pause_level_contrast_db` - secondary mixed speech/acquisition descriptor; never physical SNR.
4. `qadd_pause_spectral_flatness` - secondary nonordinal spectral-type descriptor over 80-7000 Hz.
5. `qadd_mains_hum_comb_score_db` - targeted hum-like harmonic-structure descriptor with count-matched null companions.

No family scalar, accept/reject threshold, transient event rate, or confirmatory competing-speaker detector is authorized.

## Required numerical corrections from legacy v4.1.0

1. Remove duplicate strict-speech erosion in the cohort extraction.
2. Store separate 50- and 60-Hz harmonic-support counts per window and use the recording-level winner's support companion.
3. Preserve canonical source interval identity in every lower-level ledger.
4. Make the 80-7000-Hz flatness band a formal versioned contract and update the governing registry.

## Required new validation evidence

- Independent speech-gain x pause-noise factorial control for contrast.
- Hum false-positive controls for voiced leakage/low F0, musical single tones, fan-like near-grid combs, and frequency drift.
- Polarity, DC-offset, time-shift, and resampling behavior.
- Whole-pause support truncation/bootstrap calibration and a 100-300-ms guard grid.
- Frame/window/band sensitivity.
- Repeated-recording persistence and participant-balanced summaries.
- Standardized A-H/J figure package with source data, caption, provenance, and index.
- Support-aware scientific and ML exports.

## G10 rule

The existing `qadd-v4.1.0` freeze is treated as legacy evidence, not as the reviewed family freeze. The reviewed candidate may freeze only after every blocking checklist row is PASS or scientifically justified CONDITIONAL, feature-specific decisions are signed, the executed notebook is sealed, and an immutable artifact inventory is generated.
