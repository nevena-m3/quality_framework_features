# QDIST v4.0.0 final scientific audit

## Final decision

QDIST v4.0.0 is accepted for immutable numerical and figure-package freeze, with explicit qualifications. The accepted construct is native-waveform plateau morphology compatible with hard clipping or saturation. It is not a complete nonlinear-distortion measure and must not be described as total harmonic distortion, soft clipping, compression, limiting, automatic gain control, codec distortion, perceptual distortion, or proof of where clipping occurred.

The reviewed family preserves the immutable qdist-v3.1.1 numerical detector and standardizes its evidence, figures, event review, feature roles, provenance, and support-aware handoff. Finalization changes version and governance metadata only. It does not recompute or alter any recording-level QDIST value.

## Cohort and provenance

- 519 recordings from 224 participants.
- 519/519 recordings available under the native-waveform input contract.
- 6 positive recordings and 513 valid-zero recordings.
- 861 candidate plateaus, 30 accepted plateaus, and 15 merged episodes.
- First decoded native-rate multichannel waveform; no resampling, channel averaging, amplitude normalization, filtering, denoising, interpolation, DC removal, or re-encoding before detection.
- Exact equivalence to the frozen qdist-v3.1.1 feature table.
- Maximum cohort reconstruction discrepancy no greater than 2 x 10^-15, attributable to CSV double-precision round trip.
- Zero extraction, ledger, cohort-standardization, gallery, or event-review errors.

## Final feature decisions

### qdist_hard_clipped_frame_fraction

Decision: RETAIN_PRIMARY_RELATED_VIEW.

This is the fraction of complete native-waveform 30-ms frames intersecting at least one accepted hard-clipping plateau. It is the primary merge-gap-independent occurrence-burden view. It should be included in the manuscript and may enter the default QDIST model set. It is not independent of the event-rate and sample-burden views.

### qdist_hard_clip_event_rate_per_min

Decision: RETAIN_PRIMARY_EVENT_WITH_UNCERTAINTY.

This is the number of merged accepted hard-clipping episodes per finite task-span minute. It must always be accompanied by the underlying event count, analyzed exposure, frozen 20-ms merge gap, and exact Poisson confidence interval. Sparse counts make the rate sensitive to deletion of a single episode. The feature remains primary because it answers a distinct operational question: how often accepted clipping episodes occur.

### qdist_hard_clipped_sample_fraction

Decision: RETAIN_SECONDARY_RELATED_VIEW.

This is accepted plateau channel-sample support divided by finite channel-sample exposure. It should be translated to clipped channel-ms/min for human interpretation. It is retained as a direct burden view but is not recommended as a simultaneous default model input when both frame prevalence and event rate are already present.

No QDIST scalar, composite severity score, or standalone rejection threshold is authorized.

## Empirical behavior

Only six recordings were positive. Across positive recordings:

- hard-clipped frame fraction: median 0.001852; IQR 0.001641-0.002862; maximum 0.004600;
- hard-clip event rate: median 3.677658 episodes/min; IQR 3.281166-3.842449; maximum 7.353838 episodes/min;
- hard-clipped sample fraction: median 0.000009; IQR 0.000006-0.000017; maximum 0.000061.

The six positive recordings contained 30 accepted plateaus and 15 merged episodes. Five participants were positive; one participant contributed two positive recordings. Recording-weighted occurrence was 6/519 = 1.156% with Wilson 95% interval 0.531%-2.499%. Participant-weighted occurrence was 5/224 = 2.232% with Wilson 95% interval 0.957%-5.118%.

## Analytical validity

### Numerical reconstruction

All three features were independently reconstructed from saved ledgers and exposure fields for all 519 recordings. Frame prevalence reconstructed from accepted plateau/frame intersections. Event rate reconstructed from merged episode count and finite task-span exposure. Sample fraction reconstructed from accepted plateau channel-sample support and finite channel-sample exposure. Valid zero remained distinct from unavailable or indeterminate status.

### Controlled construct response

Progressive hard clipping produced ordered increases in frame and sample burden. Accepted detections had synthetic precision 1.000. Mean recall across the full dose series was approximately 0.753 because the detector intentionally does not promote some extremely sparse smallest-dose plateaus. This is a conservative detection limit, not a numerical failure.

### Discriminant controls

Clean speech-like signals, natural extrema, sinusoidal and non-sinusoidal controls, impulses, click trains, broadband noise, DC offset, clean 8/10/12/16/24-bit PCM, and moderate smooth tanh saturation were not promoted as accepted hard-clipping episodes. The family therefore has specificity within its declared hard-plateau construct. Smooth saturation and broader nonlinear distortion remain outside scope.

### Transformation contract

Polarity reversal, aligned non-wrapping translation, post-clipping attenuation, and lossless PCM round trip behaved as declared. Resampling and lossy Opus/AAC encoding can erase exact plateau morphology. QDIST must therefore run on the first decoded native-rate waveform, before analysis resampling or lossy transformation.

## Robustness and uncertainty

A deterministic 46-recording sensitivity set was used. Occurrence was unchanged for most declared variants. One recording changed class under a 0.40 recording-relative floor and one under a minimum-five-sample plateau rule. These results show that a small number of events are threshold-adjacent; they do not invalidate the frozen definition. Threshold margins are retained in the audit.

Occurrence was invariant under 10-, 20-, 30-, and 50-ms episode merge gaps. The 10-ms rule changed episode count in two recordings and changed maximum rate by 1.838 episodes/min. The 20-, 30-, and 50-ms results were identical in this cohort. The 20-ms merge rule is frozen as feature identity.

Deleting one accepted plateau or episode can change event rate by as much as approximately 1.927 episodes/min. Exact Poisson intervals are therefore mandatory for rate interpretation.

## Reliability and redundancy

Among 158 repeated-recording pairs, 154 were zero-zero, two were first-positive only, one was second-positive only, and one was positive-positive. Overall agreement was 0.981, negative agreement 0.990, positive agreement 0.400, and Cohen kappa 0.391. Positive-part magnitude reliability cannot be estimated from only one double-positive pair.

All-recording rank correlations among the three features are near one because 513 recordings share a valid zero. Positive-only correlations among six recordings were 0.943 for frame fraction versus event rate, 0.771 for frame fraction versus sample fraction, and 0.657 for event rate versus sample fraction. The outputs are related views of one detector system, not independent mechanisms.

## Event verification (G9)

The event-review package contains 60 deterministic, label-blind items:

- all 30 accepted real plateaus;
- 20 rejected near-threshold candidates;
- 10 valid-zero controls.

Every item includes waveform, PCM-code/first-difference context, amplitude occupancy, spectrogram, and a linked audio excerpt. Accepted events were 30/30 positive under the review rubric. Among seven adjudicable rejected candidates, one was judged positive (14.3%); 13 rejected candidates remained ambiguous. Valid-zero controls were 0/10 positive.

The point-estimate gates are satisfied: accepted positive fraction at least 0.90, adjudicable rejected positive fraction at most 0.20, and valid-zero positive fraction zero. However, the review was AI-assisted blinded morphology review, not independent human ground truth. An independent expert signoff is recommended before manuscript claims about event-level precision are made. This qualification does not prevent freezing the numerical measurement family because the detector scope, event ledgers, uncertainty, and review status are all explicit.

## Figures and gallery

The standardized package contains 23 complete figure bundles: 15 main A-J bundles and eight deterministic Panel G examples. Panel I is applicable and complete. Every indexed bundle includes 300-dpi PNG, editable SVG, PDF, source-data CSV, caption, and provenance JSON. The 60-item event-review package additionally contains standardized review PNG, source CSV, WAV excerpt, caption, and provenance for each item.

Finalization regenerates D2, E1, E2, E3, F, H1, H2, H3, I, and J from saved cohort tables only. This improves sparse-positive visibility, physical units, uncertainty, ambiguity reporting, and non-estimability language. A-C and unaffected cohort/gallery bundles remain unchanged. No feature value is recomputed.

## Final G1-G10 state

- G1 PASS.
- G2 PASS.
- G3 PASS WITH QUALIFICATION.
- G4 PASS WITH QUALIFICATION.
- G5 PASS WITH SCOPE LIMIT.
- G6 PASS WITH QUALIFICATION.
- G7 PASS WITH QUALIFICATION.
- G8 PASS WITH MAJOR QUALIFICATION.
- G9 PASS WITH QUALIFICATION.
- G10 PASS.

## Freeze authorization

QDIST v4.0.0 is authorized for atomic numerical freeze and standardized figure-package freeze if and only if the finalization notebook completes without error, confirms exact numerical equivalence, writes the accepted final manifest, and both freeze scripts pass all contract checks. Freeze scripts refuse overwrite. Any numerical or semantic change requires a new measurement version.
