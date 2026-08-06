# QTEMP freeze-readiness review of the local preflight run

## Decision

The uploaded `QTEMP_v100_LOCAL_RUN_FOR_REVIEW.zip` executed without notebook errors, but it is not a freeze package. It contains a synthetic/preflight layer and cohort summary counts copied from a prior executed notebook. It does not contain the 519-recording feature table, retained candidate/disposition/event/exposure ledgers, native media, full real-speech and signal-chain source tables, complete G9 gallery, or a complete master checklist.

Publication freeze is therefore prohibited at this stage.

## Figure audit

| Panel | Current artifact | Freeze-standard decision | Required remediation |
|---|---|---|---|
| A | Single synthetic event-count curves | CONDITIONAL | Plot feature estimates and recovery error across repeated carriers, dose, sample rate and alignment; include uncertainty, expected response and detection boundary. |
| B | Four all-zero synthetic controls | FAIL for freeze | Add stop closures, long pauses, weak/breathy/voiceless speech, periodic sustained phonation, repeated linguistic material, click/impulse, clipping and gain-step controls; include real-speech hard negatives and uncertainty on false-positive rate. |
| C | Dropout gain/polarity only | FAIL for freeze | Add both retained detectors, common time shift, native sample rate, resampling/interpolation damage, mono handling, normalization, and codec/re-encoding characterization. |
| D | Six synthetic exposure points | CONDITIONAL | Add recovery/bias versus independent exposure and the actual cohort support/status distribution; preserve unavailable separately from measured zero. |
| E | Duplicate-duration threshold only | FAIL for freeze | Add dropout thresholds, context/edge guards, merge rules, duplicate similarity, lag grid, frame/hop, periodicity guard and event grouping; show recording-level change distributions. |
| F | Positive-versus-zero counts only | FAIL for freeze | Use the actual feature table; show zero mass, nonzero ECDF/rug, missingness, eligible duration, exact counts/Poisson intervals, task/device concentration and outliers. |
| G | Missing | FAIL | Generate signal-linked WAV/PNG examples for every accepted event plus ambiguous/rejected/candidate-free controls with detector components and exact ledger decomposition. |
| H | Two-row text table | FAIL for freeze | Show exact reconstructability, same-ledger relationships, event grouping sensitivity, subject recurrence and repeat-recording persistence where available. |
| I | Prior 3/6 accepted events observable, 3/6 uncertain | FAIL | Diagnose the three uncertain accepted events using native audio and waveform evidence. If the detector changes, evaluate a newly held-out blinded set; do not relabel or tune on diagnosis. |
| J | Schema description | CONDITIONAL | Validate the schema against the actual central feature table, including status, support, count intervals, version and abstention fields. |

Every final panel requires SVG, PDF, 300-dpi PNG, machine-readable source data, caption draft, provenance JSON and a gallery-index row.

## Checklist audit

The uploaded checklist contains 20 selected rows. The shared master checklist contains construct, provenance, estimator, implementation, transformation, dose-response, discriminant, support, empirical, reliability, event-verification, interpretation, ML-readiness, figure and freeze rows. A family cannot freeze from a selected subset.

The final QTEMP checklist must include every shared ID: C1–C6, P1–P4, E1–E5, I1–I4, T1–T5, D1–D4, X1–X4, S1–S5, Plaus1–Plaus3, R1–R5, V1–V3, INT1–INT3, ML1–ML3, F1–F3, and G10–G12. Each row must be PASS, CONDITIONAL, FAIL, PENDING or N/A and point to an exact artifact rather than a narrative assertion.

## Blocking scientific issue

The prior G9 result is 3 clearly observable accepted dropout events out of 6:

\[
\hat p = \frac{3}{6}=0.50,
\]

below the prespecified requirement of at least 0.80. With six accepted events, at least five would need to be clearly verified to meet the raw fraction rule. The three uncertain events cannot simply be converted to positive or excluded after unblinding. They must be diagnosed. A detector revision must be locked using development evidence and assessed on new held-out blinded material.

The frozen-audio detector had zero cohort positives. That is valid measured absence, not analytical validation of positive-event precision. It may be retained only as a zero-variation candidate after its injected recovery and real periodic-speech specificity are fully documented.

## Required evidence package

Run `COLLECT_QTEMP_FREEZE_EVIDENCE.cmd` from the project root. It collects compact CSV/JSON/figure/gallery/audio evidence from the existing `qtemp-v0.3.1-finalization` stage, excludes large caches and Parquet files, creates SHA-256 inventory and requirement tables, and writes `QTEMP_FREEZE_EVIDENCE_FOR_REVIEW.zip` for upload.
