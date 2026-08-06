# QTEMP v1.0.0 final scientific audit

## Final decision

QTEMP v1.0.0 is closed as an immutable analytical implementation and evidence freeze with **no retained primary validated features**. This is a completed negative/limited feature-selection result, not a pending workflow.

The accepted family construct is restricted to observable temporal-discontinuity patterns in the native decoded audio stream. QTEMP does not identify packet loss, network failure, buffering, packet-loss concealment, missing speech content, device failure, or physiology. The four deterministic outputs remain available only for explicitly labeled exploratory, descriptive, monitoring, and sensitivity analyses. The splice-discontinuity output is dropped.

## Cohort and provenance

- 519 recordings from 224 participants.
- Native decoded signal view and preprocessing order are pinned.
- 519 unique recording rows in the final exploratory table.
- Two recordings contain accepted dropout-like support; no recording contains an accepted frozen-audio event.
- The dropout ledger contains six accepted events across the two positive recordings.
- Exact reconstruction, determinism, missingness semantics, support checks, transformation checks, parameter sensitivity, figures, captions, provenance, and SHA-256 hashes are sealed.
- The validated primary feature table is not modified.

## Final feature decisions

### qtemp_dropout_duration_fraction

Decision: `EXPLORATORY_NOT_PRIMARY_CONSTRUCT_SPECIFICITY_LIMITED`.

This is the fraction of eligible decoded support occupied by algorithmically accepted bracketed dropout-like events. It may be reported descriptively, with the pause-boundary ambiguity and 2/519-positive support stated explicitly. It is not a validated packet-loss measure or primary inferential biomarker.

### qtemp_dropout_event_rate_per_min

Decision: `EXPLORATORY_NOT_PRIMARY_CONSTRUCT_SPECIFICITY_LIMITED`.

This is the frequency of the same accepted dropout-like events per eligible minute. It is a same-ledger view, not an independent dimension. It must not be described as a calibrated packet-loss or technical-failure rate.

### qtemp_frozen_audio_duration_fraction

Decision: `MONITORING_ONLY_ZERO_VARIATION_POSITIVE_BEHAVIOR_UNVERIFIED`.

Controlled synthetic and participant-disjoint real-speech injections recover the registered near-exact repetition target, but the cohort contains 0/519 positives. The output therefore documents that no registered event was observed; it does not establish robust real-world positive-event behavior.

### qtemp_frozen_audio_event_rate_per_min

Decision: `MONITORING_ONLY_ZERO_VARIATION_POSITIVE_BEHAVIOR_UNVERIFIED`.

This is the same-ledger event-frequency view and is zero for all 519 recordings. It is retained only for monitoring and sensitivity work, never as independent evidence or a primary analysis feature.

### qtemp_splice_discontinuity_rate_per_min

Decision: `DROP_FAILED_ANALYTICAL_VALIDATION`.

Held-out real-speech boundary recovery failed. The earlier estimator produced 318/519 positives and displayed sample-rate dependence. It must not be exported or analyzed as a retained QTEMP feature.

## Analytical validity

### Numerical correctness

The final four-output table is deterministic and contains one unique row per recording. Values reconstruct from frozen event ledgers and eligible exposure. Valid zero remains distinct from unavailable status. Duration fractions and event rates preserve their shared-ledger relationship.

### Controlled target recovery

Registered dropout-like and near-exact repetition targets are recovered in controlled synthetic tests and participant-disjoint real-speech injections. These results validate the implemented observable within the tested operating range; they do not validate a physical network or device cause.

### Discriminant specificity

The dropout observable remains vulnerable to natural pause-boundary and stop-closure ambiguity. The repetition observable must distinguish near-exact duplicate support from periodic speech. Those limitations prevent primary promotion. The splice estimator failed its held-out real-speech validation and was removed.

### Transformation and support contracts

Gain, polarity, common time shift, native-view behavior, sample-rate behavior, minimum eligible exposure, event merging, and parameter boundaries are characterized. Unsupported invariance or interchangeability claims are prohibited. All values retain recording identity, support, status, exposure, version, and provenance.

## Empirical behavior and uncertainty

The cohort prevalence is extremely sparse: 2/519 recordings are dropout-positive and 0/519 are frozen-positive. The absence of observed frozen-audio events does not establish population absence or real-world sensitivity. Positive-part reliability cannot be established from zero or near-zero positive support. Repeated-recording summaries are dominated by shared zeros and cannot rescue construct specificity.

## Reliability and redundancy

The duration and event-rate outputs within each detector are two summaries of the same event ledger. They must not be presented as independent confirmations. Persistence estimates are zero-dominated. Consequently, no QTEMP scalar, composite severity score, standalone exclusion gate, or default multivariable feature block is authorized.

## Figures

Panels A-J are complete. Each panel has editable SVG, PDF, 300-dpi PNG, source-data CSV, scientific caption, and provenance JSON. Panel I explicitly records that event verification is not applicable because no event feature is retained in the validated primary set. This is an analytical disposition, not unfinished evidence collection.

## Final G1-G10 state

- G1 PASS.
- G2 PASS.
- G3 PASS.
- G4 PASS.
- G5 CONDITIONAL.
- G6 PASS.
- G7 PASS.
- G8 CONDITIONAL.
- G9 N/A — no retained primary event feature.
- G10 FINALIZED — no retained primary QTEMP features.

## Publication and downstream handoff

Paper 1 must count **zero QTEMP features in the validated primary set**. It may report, as a transparent negative/limited result, that 2/519 recordings contained algorithmically accepted bracketed dropout-like support and 0/519 contained an event meeting the registered near-exact repetition rule. These statements must remain exploratory or monitoring-only and must preserve the physical-cause prohibitions above.

Downstream sensitivity work may use the four frozen outputs only with their role, status, exposure, detector version, and same-ledger dependencies intact. Primary models, primary biomarker-robustness claims, and standalone recording rejection must exclude them.

## Freeze authorization

QTEMP v1.0.0 is authorized as `qtemp-v1.0.0-analytical-final-no-retained`. The implementation, notebooks, tables, figures, decisions, checklist, dashboard, audit, provenance, and hashes are frozen. Any numerical, semantic, preprocessing, threshold, role, or claim-boundary change requires a new measurement version.
