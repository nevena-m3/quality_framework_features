# QREV v3.0 scientific decision record

## Construct and claim boundary

QREV v3.0 measures observable temporal smearing and post-speech residual
behavior compatible with reverberation or echo. It does not estimate RT60,
EDT, C50/C80, D50, DRR, STI, or a room impulse response. Breath, changing
noise, delayed echo, speech-offset error, codec behavior, and speech content
remain explicit confounds.

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

Boundary features require natural internal speech-to-pause transitions. The
default recording support threshold is four valid boundaries. SRMR requires at
least 3 s of frozen speech and a 3-s task span.

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

QREV v3.0 supersedes rather than overwrites v2.2.

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
- Failed scientific gates cannot be converted into "documented exceptions."
- Extraction is recording-at-a-time with resumable checkpoints; waveforms and
  full frame ledgers are never accumulated in memory.

## Freeze rule

The first cohort run is a candidate. Freeze is permitted only after package
tests, reference regression, synthetic RIR controls, discriminant controls,
floor/censoring calibration, codec checks, support and boundary robustness,
cohort extraction, empirical characterization, and the label-blind gallery
all complete without a failed blocking gate. A failed gate requires a new
scientific decision or measurement version; it cannot be waived inside the
notebook.
