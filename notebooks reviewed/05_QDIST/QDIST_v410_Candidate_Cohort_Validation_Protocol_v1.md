# QDIST v4.1 candidate-cohort validation protocol

## Scientific construct and boundary

QDIST v4.1 measures visible hard-plateau morphology in the stored, first-stream,
native-rate decoded waveform over the frozen continuous task span. It is an
observation-model family for acquisition-related contamination. It does not by
itself identify an analog or digital causal stage, prove physical clipping,
measure soft clipping, characterize automatic gain control or dynamic-range
compression, quantify codec nonlinearity, or cover all nonlinear distortion.

The candidate feature roles are:

1. `qdist_hard_clipped_sample_fraction` — primary: unioned accepted plateau
   support divided by finite channel-sample exposure.
2. `qdist_hard_clip_event_rate_per_min` — secondary: accepted episodes after a
   prespecified 20-ms cross-channel merge rule per finite exposure minute.
3. `qdist_hard_clipped_frame_fraction` — conditional/audit: complete 30-ms
   frames intersected by accepted plateaus. This view is frame-origin dependent.
4. `qdist_occurrence` and governed status — companion descriptors, not a
   standalone recording-acceptability gate.

No family scalar is permitted. Missing/unavailable is never imputed to zero.

## Locked inputs and leakage controls

- Frozen eligibility and frozen segmentation define the cohort and task span.
- The first audio stream is decoded at its native sample rate with channels
  preserved and without normalization, mono reduction, resampling, filtering,
  interpolation, denoising, or fallback processing.
- Candidate extraction, carrier selection, parameter selection, validation
  sampling, and figures use no diagnosis, ALS severity, clinical outcome, or
  human recording-quality label.
- Code, parameter, source-file, decoded-waveform, and frozen-input hashes are
  retained. Per-recording checkpoints are content-addressed by these inputs.

## Computational validation sequence

1. Verify the accepted v4.1 remediation preflight, candidate detector hash, and
   frozen input identities.
2. Recompute every eligible recording from native decoded media. Do not copy
   qdist-v3.1.1 values.
3. Reconstruct all three analysis values independently from accepted-plateau and
   episode ledgers. Maximum absolute difference must be at most `2e-15`.
4. Compare qdist-v3.1.1 and qdist-v4.1 recording identities, occurrence
   transitions, feature differences, and rank correspondence. Differences are
   detector-version effects, not reconstruction error.
5. Audit the actual conjunctive predicates, including the corrected
   same-polarity local-context ratio (`>=0.90`), both context sides, plateau
   length/duration, edge occupancy, interior contrast, terminality,
   quantization guard, and square-like guard.
6. Apply known hard limits to deterministic, label-blind, candidate-valid-zero
   cohort speech carriers at exact target burdens `0.0003`, `0.001`, `0.003`,
   and `0.01`, separately for symmetric, positive-only, and negative-only
   geometry. The altered-sample mask is exact and is the reference for
   occurrence sensitivity, sample precision, sample recall, F1, and burden
   response. This intervention validates decoded-waveform morphology and does
   not localize a physical stage.
7. Repeat a fixed symmetric `0.003` intervention at 3, 5, 10, 20, and 30 seconds
   to quantify availability and recovery versus exposure.
8. Rerun a label-blind enriched subset under 23 prespecified one-factor settings
   (baseline plus lower/higher values for 11 detector parameters). The subset
   contains every baseline positive, near-boundary rejected candidates, and a
   deterministic valid-zero sample.
9. Quantify 10/20/30/50-ms episode merge-gap behavior, single-plateau and
   single-episode deletion influence, participant weighting, first-pair
   within-participant persistence, and association among the three related views.
10. Generate A–J figure bundles. Each bundle must contain high-resolution PNG,
    SVG, PDF, source CSV, caption draft, and provenance JSON using relative paths.

## Prespecified computational gates

- All frozen eligible identities recomputed; zero extraction errors.
- Exact ledger reconstruction for all three features and all recordings.
- Complete 3-geometry × 4-dose challenge; zero carrier errors.
- At target burden `>=0.001`, occurrence sensitivity `>=0.90` in every geometry.
- Range of mean sensitivity across the three geometries `<=0.10` at those doses.
- Median sample precision `>=0.90` in every geometry/dose cell at target
  `>=0.001`. Recall is reported as construct coverage and is not forced to one.
- All five exposure durations present; zero support-calibration carrier errors.
- All 23 parameter settings complete; zero parameter-rerun errors.
- Occurrence agreement `>=0.95` for every nonbaseline one-factor setting.
- Occurrence agreement `>=0.99` for 10, 30, and 50 ms versus the 20-ms merge rule.
- Every finite accepted predicate margin is nonnegative.
- All required A–J and at least eight G gallery bundles complete.

Failure of a prespecified gate blocks finalization. It is not repaired by
relaxing the gate after inspecting the same results.

## Blinded morphology review

The review package includes every accepted plateau, up to 30 near-threshold
rejections, and up to 20 deterministic valid-zero windows. Each item has an
opaque identifier, randomized deterministic order, waveform context,
sample-level zoom, first difference, amplitude occupancy, spectrogram,
empirical CDF, and audio. The restricted key is withheld from reviewers.

Two independent reviewers label every item as one of:

- `DEFINITE_HARD_CLIP`
- `PROBABLE_HARD_CLIP`
- `AMBIGUOUS`
- `NOT_HARD_CLIP`
- `CANNOT_DETERMINE`

The first two labels form the binary morphology-positive category. Reviewers
also record confidence (1–5), view completeness, and comments. No AI-generated
review label is permitted.

Prespecified human-review gates are:

- 100% completion by two independently identified reviewers;
- binary Cohen kappa `>=0.80` (lower agreement requires investigation/revision);
- every exact-label disagreement receives a documented adjudicated label;
- adjudicated morphology-positive fraction `>=0.90` among accepted plateaus;
- adjudicated morphology-positive fraction `<=0.20` among near-threshold rejections;
- adjudicated morphology-positive fraction `<=0.05` among valid-zero windows.

These are morphology-review proportions, not population prevalence, diagnostic
performance, or independent physical ground truth. Candidate plateaus from one
recording are not assumed statistically independent.

If review evidence motivates detector revision, the code and measurement
version must change, existing v4.1 candidate outputs must remain immutable, and
the revised detector must undergo a new blinded review using a fresh or held-out
sample. Reviewer feedback must not be used to silently tune and then validate on
the same items.

## Finalization boundary

The candidate-cohort notebook and adjudication notebook cannot freeze QDIST.
After all computational and human gates are complete, investigators must still:

1. characterize disagreements and failure modes by magnitude path, polarity,
   codec, sample rate, support, recording, and participant clustering;
2. resolve joint arbitration with QGAIN, QCHAN, QTEMP, and other neighboring
   families without outcome-label tuning;
3. make explicit retain/revise/conditional/exploratory/drop decisions for every
   feature and companion;
4. reconcile construct wording and the manuscript feature census;
5. execute a separate, immutable, hash-sealed finalization workflow.

Until then: `scientific_review_decision=PENDING`, `freeze_allowed=false`, and
`publish_and_freeze=false`.
