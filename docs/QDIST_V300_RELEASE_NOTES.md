# QDIST v3.0.0 release notes

## Replaces the legacy nonlinear-distortion notebook

The legacy notebook summarized pre-existing columns but did not perform a
governed native-waveform extraction. QDIST v3.0.0 replaces it with a complete
measurement package: pure estimator module, deterministic notebook generator,
analytical tests, event ledgers, cohort checkpoints, gallery review, central
export, and immutable freeze.

## Scientific changes

- Narrows the claim from generic nonlinear distortion to evidence compatible
  with hard clipping or saturation.
- Uses the native decoded multichannel waveform before any transformation.
- Analyzes the continuous frozen natural task span with internal pauses intact.
- Replaces global-maximum and per-frame heuristics with candidate-level
  adaptive histogram evidence plus plateau and boundary morphology.
- Supports asymmetric clipping, clipping below final digital full scale, and
  multiple clipping levels in one recording.
- Derives frame fraction, episode rate, and sample fraction from one exact
  event ledger.
- Retains near-full-scale occupancy and histogram strength only as diagnostics.
- Does not create a retained soft-clipping or compression feature.
- Treats coarse regular quantization as detector unavailability rather than a
  clipping event or a valid zero.
- Preserves valid zeros for detector-available no-event recordings.
- Reports exact Poisson intervals and downstream suitability for sparse event
  rates.
- Requires independent feature-level pass/drop decisions.

## Validation included

- 77 automated QDIST module/notebook tests.
- Hand-computable sample, frame, and episode denominators.
- Exact event-ledger reconstruction.
- Polarity and post-clipping attenuation invariance.
- One-sided and multiple-level clipping recovery.
- Ten-seed hard-clipping dose grid with sample- and event-level recovery.
- Unclipped speech, periodic signals, impulses, music, percussive signals, DC
  offset, coarse quantization, compression, and tanh saturation controls.
- Lossless, resampling, Opus, and AAC characterization.
- Frame, histogram, plateau, boundary, and episode-merge sensitivity.
- Restart-safe source/code/parameter-bound cohort checkpoints.
- Sparse-distribution suitability audit and blinded event/non-event gallery.

## Container validation result

The clean package tests passed:

```text
77 passed
```

The R4 module and generated-notebook governance tests passed. Focused
30-second native-PCM benchmarks confirmed bounded candidate enumeration, and
effectively 12-bit or coarser regular lattices fail closed before candidate
construction. The complete real-cohort notebook run must be performed in the
project repository because the frozen media, frozen segmentation outputs, and
the repository Parquet environment are not included in this patch.


## Revision r2 — parameter-table serialization fix

- Canonically serializes every parameter value as JSON before CSV/Parquet export.
- Prevents mixed Python scalars and tuple-valued parameters from producing a PyArrow schema error.
- Preserves the estimator, parameters, and all scientific validation definitions unchanged.


## Revision r3 — cohort runtime, checkpoint, and Jupyter-root repair

- Replaces unbounded near-flat-pair gap chaining with bounded consecutive-flat
  plateau seeds. Runtime is now approximately linear in native waveform length.
- Defines the candidate ledger as morphology-positive plateau candidates rather
  than every isolated equal-value sample pair. Clean 30-second native PCM controls
  produce a bounded candidate table instead of hundreds of non-candidates.
- Evaluates local histogram evidence only after the inexpensive morphology gate.
- Expands the fail-closed coarse-quantization audit through effective 12-bit
  regular lattices, while requiring a minimum adjacent-repeat fraction so clean
  analytic triangle waves are not mislabeled as quantized audio.
- Uses a native-only cohort decoder, avoiding unnecessary mono construction and
  16-kHz resampling during QDIST extraction.
- Writes checkpoints atomically, writes the manifest last, and automatically
  invalidates incomplete/corrupt/identity-mismatched checkpoints.
- Adds per-recording progress, phase timings, slow-recording markers, ETA, and a
  live JSON progress file.
- Adds a repository-root-safe PowerShell Jupyter launcher to prevent duplicated
  notebook paths, failed autosave, and stale-kernel browser sessions.

## Revision r4 — exact fixed-window evidence and rare-event specificity

- Replaces candidate-centred histogram rescanning with exact binary-search counts
  in deterministic fixed 250-ms and 1,000-ms windows on a 50%-overlap grid.
  Each occupied native window is sorted once and cached, so evidence evaluation
  no longer scales with candidate count multiplied by context length.
- Tightens the scale-relative plateau tolerance from a near-16-bit-LSB value to
  a genuinely near-flat value, preventing ordinary smooth extrema in native or
  codec-decoded speech from becoming plateau candidates.
- Requires at least four consecutive near-flat plateau samples and every
  morphology-positive plateau to be a local amplitude extremum before histogram
  evidence is evaluated. Three-sample smooth or quantized extrema are therefore
  not promoted to hard-clipping candidates.
- Adds minimum edge-zone occupancy and edge-excess requirements. A short flat
  run cannot pass solely because its own samples occupy an otherwise empty
  histogram bin.
- Uses zero internal-gap chaining for plateau construction. Nearby plateaus are
  still combined only at the separately frozen 20-ms episode-merging stage.
- Retains every accepted plateau but caps saved rejected-candidate rows
  deterministically; complete candidate/rejection totals and stage counts remain
  in the recording summary.
- Adds stage-level diagnostics for flat runs, morphology-positive candidates,
  local-extremum candidates, edge-evaluated candidates, and saved audit rows.
- Prints source-identity/hash, decode, detector, and checkpoint-write times
  separately, with distinct detector-slow versus I/O-or-decode-slow labels.
- Adds smooth oversampled negative-control and bounded-audit tests. These changes
  preserve hard-clipping dose recovery, polarity/gain behavior, multi-level
  clipping recovery, exact ledger reconstruction, and the hard-clipping-only
  claim boundary.
