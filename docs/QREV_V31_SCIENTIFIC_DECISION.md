# QREV v3.1 scientific decision record

## Construct and claim boundary

QREV v3.1 measures observable temporal smearing and post-speech residual
behavior compatible with reverberation or echo. It does not estimate RT60,
EDT, C50/C80, D50, DRR, STI, or a room impulse response. It also does not
contain a discrete-delay echo detector and cannot identify echo as the cause
of a residual. Breath, changing noise, delayed echo, speech-offset error,
codec behavior, and speech content remain explicit confounds.

No scalar QREV score is constructed.

## Retained analysis profile

1. `qrev_tail_excess_100ms_db` is the recording median of the signed
   difference between the first 100-ms post-offset AC-RMS level and a
   700-1000-ms local pause baseline. Negative values are retained; they are not
   clipped to zero.
2. `qrev_tail_persistence_median_sec` is the recording median time until the
   envelope first remains within 3 dB of the local floor for three frames,
   right-censored at 1.0 s. A recording median at the horizon is explicitly
   labeled censored.
3. `qrev_downward_decay_rate_db_per_sec` is the median magnitude of a negative
   Theil-Sen slope over the first 300 ms. It is defined only for boundaries
   with at least 3 dB robust dynamic range. Rising, flat, and unsupported
   boundaries remain unavailable rather than becoming zero.
4. `qrev_srmr_norm` is a normalized-fast SRMR comparator computed over the
   natural task span with internal pauses preserved.

The first three measures are conditional natural-boundary features: they
require internal speech-to-pause transitions with sufficiently long following
pauses. The default recording support threshold is four valid boundaries.
`qrev_srmr_norm` is a broadly available established comparator and requires at
least 3 s of frozen speech and a 3-s task span. It is not called a direct
estimate of reverberation severity.

Availability is part of the measurement result. A boundary feature that cannot
be supported is unavailable, not zero and not imputed. Because the occurrence
of suitable speech offsets and pauses can depend on speech phenotype, the
analysis stage must test feature availability against prespecified ALS
severity and cohort variables. Clinical labels are intentionally absent from
this feature-extraction notebook.

## SRMR implementation decision

The production implementation is the official SRMRpy code at commit
`fee009779cef96bed34db3a7e31d10f3ad1ea133`, using:

- `fast=True`
- `norm=True`
- `max_cf=30`
- 23 cochlear filters
- 125-Hz lower acoustic-filter frequency
- 4-Hz first modulation center frequency
- Gammatone 1.0.3

This configuration is the normalized-fast case exercised by the upstream test
suite. It was selected over TorchMetrics because TorchMetrics labels its SRMR
implementation experimental, adds PyTorch/torchaudio memory overhead, and
still derives from SRMRpy.

The upstream repository's stored 2014 normalized-fast fixture value does not
reproduce under the later Python-3-compatible Gammatone implementation. The
unmodified upstream code returns the same value as the vendored implementation
under Gammatone 1.0.3. QREV therefore pins the full executable contract and
stores both the current regression value and the historical discrepancy. It
does not silently waive or overwrite that evidence.

## Corrections relative to QREV v2.2

QREV v3.1 supersedes rather than overwrites v2.2.

- Feature algorithms live in `paper1_qc.qrev`; the notebook does not duplicate
  them.
- Post-offset frames require strict sample containment.
- AC-RMS removes frame DC.
- Tail excess remains signed.
- Decay is conditional on a negative robust slope and sufficient dynamic range.
- Censoring and floor instability are explicit.
- SRMR is added as a pinned published comparator.
- RIR validation uses the same stochastic tail across dose levels, controls
  tail energy and decay separately, prepares headroom once, and forbids
  condition-specific peak normalization.
- Breath and delayed-echo positive controls expose residual-tail confounding;
  they do not license artifact identity claims. An additive-noise dose grid
  explicitly characterizes SRMR noise sensitivity, and codec/resampling
  sensitivity is audited separately.
- Failed scientific gates cannot be converted into "documented exceptions."
- Extraction is recording-at-a-time with resumable checkpoints; waveforms and
  full frame ledgers are never accumulated in memory.
- Boundary robustness uses a deterministic stratified sample covering support
  tiers, feature magnitude, and baseline-unavailable recordings.
- Every perturbation reports baseline availability, perturbed availability,
  paired sample size, rank stability, and magnitude change. Zero-pair cells
  cannot pass by being dropped.
- The sensitivity grid varies early-tail, late-floor, and 0.8/1.2-s
  persistence-horizon definitions around the prespecified 1.0-s estimator.
- Delete-one-boundary precision includes recordings at the minimum four-boundary
  support threshold.
- The empirical availability table includes Wilson 95% confidence intervals,
  support-tier summaries, and pairwise sample sizes for correlations.

## Freeze rule

The first cohort run is a candidate. Freeze is permitted only after package
tests, reference regression, synthetic RIR controls, discriminant controls,
floor/censoring calibration, codec checks, support and boundary robustness,
cohort extraction, empirical characterization, and the label-blind gallery
all complete without a failed blocking gate. A failed gate requires a new
scientific decision or measurement version; it cannot be waived inside the
notebook.
