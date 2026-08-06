# QDIST v4.0.0 — Cohort Execution Plan

## Inputs

1. Accepted reviewed preflight candidate at `outputs reviewed/nonlinear_distortion/qdist-v4.0.0-candidate`.
2. Immutable numerical baseline at `MAIN outputs/02_FEATURE_FREEZE/nonlinear_distortion/qdist-v3.1.1`.
3. Frozen v3.1.1 tables, candidate/accepted/episode ledgers, gallery PNG/WAV files, and adjudication tables.

No raw-media path is required. Feature extraction is not recomputed.

## Execution stages

1. Verify the preflight manifest, Panels A–C, immutable manifest, feature registry, and frozen counts.
2. Reconstruct frame fraction, event rate, and channel-sample fraction independently for all 519 recordings.
3. Characterize 861 candidate margins and 30 accepted plateau margins.
4. Evaluate 10/20/30/50-ms episode merge gaps, delete-one-plateau and delete-one-episode influence, and exact Poisson event-rate intervals.
5. Quantify availability, valid-zero mass, native sample rate, channel count, sample format, codec, and acquisition vintage.
6. Quantify repeated-recording occurrence; do not estimate positive-part magnitude reliability when fewer than five participant pairs are positive at both visits.
7. Build the 60-item label-blind event-review package and 8 deterministic Panel G examples.
8. Generate all Panels A–J, with Panel I applicable.
9. Write a support-aware, non-imputed ML interface and a candidate manifest with freeze disabled.

## Prespecified pass conditions

- 519 recordings and 224 participants.
- Numerical equivalence to v3.1.1 within 2e-15 after CSV roundtrip.
- 60 event-review items, each declaring waveform, PCM/derivative, amplitude-distribution, spectrogram, and audio-excerpt views.
- Accepted-event positive fraction >=0.90; rejected-candidate adjudicable positive fraction <=0.20; valid-zero positive fraction =0.
- At least eight deterministic label-blind Panel G examples.
- 23 complete six-artifact figure bundles.
- No scalar, standalone threshold, broad nonlinear-distortion claim, imputation, publication, or freeze.

## Expected dry-run evidence

The exact immutable baseline yielded 6 positive recordings (5 participants), 513 valid-zero recordings, 30 accepted plateaus, and 15 merged episodes. Occurrence was unchanged at all tested merge gaps. A 10-ms gap changed event counts in 2/519 recordings but not positive occurrence; 20/30/50 ms were identical. Positive-part repeated magnitude was not estimable because only one participant pair was positive at both visits.
