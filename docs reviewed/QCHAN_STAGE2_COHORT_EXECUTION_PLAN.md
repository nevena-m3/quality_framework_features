# QCHAN v4.0.0 — Stage 2 corrected cohort execution plan

## Authorization state

The local QCHAN v4.0.0 analytical preflight is accepted. All blocking G1–G6 preflight checks passed, the reviewed package tests passed, Panels A–C are complete, Panel I is explicitly not applicable, and no family scalar or standalone rejection rule was created.

The corrected cohort run is authorized as a candidate-only evidence-generation stage. Publication and freezing remain prohibited until independent post-cohort review completes G6–G8 and assigns G10 feature-specific decisions.

## Frozen inputs

The cohort notebook consumes the frozen Bamboo recording table and the frozen `strict_speech / primary` intervals. It must resolve exactly 519 eligible recordings and 224 participants. Every available media SHA-256 value is verified before a spectrum is accepted.

The waveform contract is deterministic mono 16-kHz `AudioViews.analysis_16k`, with explicit global DC removal and no amplitude normalization, spectral equalization or dereverberation. Native sample rate and Nyquist support are retained.

## Restart-safe extraction

Each recording receives independent spectrum, metadata, media-audit and error checkpoints. A completed checkpoint can be reused without re-decoding the recording. Failed checkpoints remain explicit and cannot be mistaken for successful extraction.

The extraction saves:

- normalized long-term speech spectrum and frequency grid;
- target-support duration and valid-frame count;
- native sample rate and source Nyquist;
- media path and observed/expected hashes;
- strict-speech interval provenance;
- explicit unavailable status and missing reason.

## Reference construction

For every target recording, the notebook creates a task-matched, leave-one-subject-out, subject-balanced reference:

1. exclude every recording from the target participant;
2. retain only the exact task stratum;
3. take the median spectrum within each reference participant;
4. take the median across reference participants;
5. require at least five other participants and eight recordings;
6. require declared full-band support when applicable;
7. prohibit any global or cross-task fallback.

The reference ledger records target, task, reference subjects, reference recordings, source-bandwidth eligibility, parameters, membership SHA-256 and reference-spectrum SHA-256. References are saved separately from recording spectra so every feature can be reconstructed.

## G2 numerical evidence

The cohort stage verifies that all four recording-level measurements reconstruct from the saved target spectrum and frozen reference spectrum. The reconstruction audit includes signed precursors for the three one-sided features. Insufficient target or reference support remains unavailable rather than numerical zero.

## G5–G6 robustness evidence

The notebook implements:

- signed precursor versus max-zero feature characterization;
- native-bandwidth and source-Nyquist stratification;
- frame, hop, speech-boundary guard and interval perturbations;
- delete-one-segment target sensitivity;
- spectral-floor, analysis-band, high-band, rolloff-fraction and smoothing variants;
- deterministic alternative reference vintages;
- recording-weighted versus subject-balanced references;
- delete-one-reference-subject sensitivity;
- subject-bootstrap reference uncertainty.

These analyses characterize sensitivity. They do not redefine the frozen-default estimators.

## G7 empirical evidence

The run produces feature distributions, explicit availability and missingness, one-sided zero masses, tails and outliers, target and reference support, task-stratum support, and native-bandwidth context. At least eight signal-linked examples are selected deterministically without diagnosis, ALSFRS or human-QC labels.

Each Panel G example includes waveform, spectrogram, target LTAS, reference LTAS and target-minus-reference spectrum, with a full PNG/SVG/PDF/source/caption/provenance bundle.

## G8 reliability and redundancy

The run evaluates:

- repeated-recording Spearman persistence and ICC where estimable;
- participant-balanced resampling;
- recording-weighted versus participant-weighted summaries;
- pairwise redundancy among the four retained features;
- relationships between one-sided features and their signed precursors;
- reference-composition and membership-vintage robustness.

## Standardized figure package

Panels A–C are inherited unchanged from the accepted preflight. The cohort notebook creates:

- D1 target support and availability;
- D2 reference support and status;
- D3 native source bandwidth;
- E1 window, boundary and common-support sensitivity;
- E2 reference bootstrap, delete-one-subject, membership-vintage and weighting sensitivity;
- E3 estimator-parameter sensitivity;
- F empirical distributions and one-sided zero masses;
- G at least eight deterministic signal-linked examples;
- H1 repeated-recording persistence;
- H2 redundancy and signed-versus-truncated evidence;
- H3 participant and reference weighting;
- J support-aware, non-imputed ML handoff.

Panel I remains `N/A_no_retained_event_detector`.

Every applicable figure requires 300-dpi PNG, editable SVG, PDF, source-data CSV, scientific caption and provenance JSON.

## ML handoff

Every feature is exported with value, availability, status, missing reason, support class, target-support metrics, native source bandwidth, reference-support metrics and reference vintage/hash. No missing feature value is imputed. No QCHAN scalar or generic rejection threshold is produced.

## Acceptance checks before packaging

The local run must report:

- 519 recordings and 224 participants;
- zero extraction, target-robustness, reference-robustness and gallery errors;
- all available media hashes verified;
- complete recording-spectrum and reference ledgers;
- complete reconstruction audit;
- Panels A–H and J complete, with Panel I explicitly N/A;
- at least eight complete Panel G bundles;
- `scientific_review_decision = PENDING`;
- `freeze_allowed = false`;
- `family_scalar_constructed = false`;
- `standalone_gate_allowed = false`;
- `device_identity_claim_allowed = false`.

Only the independent post-cohort audit may assign final feature roles and authorize an atomic numerical and figure freeze.
