# Feature-family notebook validation standard

Version: 1.0  
Reference implementation: QADD v4.1

## Purpose

Every feature-family notebook must produce a small, versioned measurement
vector and enough evidence to defend each estimand. The notebook is an
orchestrator and scientific validation record. The authoritative estimators
must live in `src/paper1_qc/`; algorithm code must not be copied into the
notebook.

The standard distinguishes:

- **technical correctness**: the implementation matches its formula;
- **measurement validity**: the estimator responds to its intended mechanism
  and not to prespecified alternatives;
- **sensitivity and robustness evidence**: clustered perturbations, missingness,
  boundaries, and signal-chain choices are quantified without confusing
  construct variation with numerical error;
- **downstream correspondence**: human perceptual or independent reference
  evidence is evaluated after estimator locking, in a separate analysis;
- **scientific review**: the audit gallery and claim limits have been reviewed;
- **integration**: the accepted implementation is the only implementation used
  by the registry, CLI, dataset assembly, and manuscript.

Running without error is not a scientific validation result.

## Required notebook sections

| Section | Common requirement | Family-specific requirement |
|---|---|---|
| 0. Environment and controls | Version, source root, safe defaults, output folders | Optional dependencies |
| 1. Measurement contract | Estimand, unit, role, direction, formula, support/status, establishment, confounding, claim limit | Family construct and feature crosswalk |
| 2. Frozen inputs | Eligible IDs, selected media, interval identity, hashes | Required signal regions |
| 3. Formula verification | Analytical controls, transform behavior, ranges, ledger reconstruction | Exact expected formula responses |
| 4. Mechanism validity | Positive and negative/discriminant controls | Simulated mechanism and plausible alternatives |
| 5. Censoring/support | Cluster-aware precision; no independent-frame bootstrap | Floor, clipping, censoring, event exposure, or other family mechanism |
| 6. Signal chain | Native/analysis view, sample rate, channel, codec checks | Family-relevant transformations |
| 7. Full extraction | One row per eligible ID, errors/statuses, reconstructable ledgers | Feature-specific ledgers |
| 8. Sensitivity/robustness | Full-estimator perturbation, availability transitions, clustered resampling with one population contribution per recording | Boundaries and parameters relevant to the family |
| 9. Empirical structure | Distributions, availability, redundancy, repeated-measure reliability where estimable | Family-specific distributional behavior |
| 10. Gallery | Label-blind algorithmic selection and an index | Feature-specific diagnostic panels |
| 11. Gates/freeze | Separate technical, scientific, empirical, review, and integration gates | Prespecified acceptance criteria where a threshold has a scientific basis; complete reporting otherwise |

## Minimum feature registry fields

Each analysis feature requires:

1. immutable feature name and measurement version;
2. display name and unit;
3. primary, secondary, mixed, targeted, or non-ordinal role;
4. exact mathematical estimator;
5. signal view and signal region;
6. minimum support and a support-tier field;
7. status field and explicit missing-value behavior;
8. expected positive-control response;
9. expected negative/discriminant-control response;
10. mathematical range;
11. establishment status: established estimator, adapted application, or novel
    study-specific estimator;
12. interpretation and known confounding;
13. explicit claim limit;
14. analysis eligibility and whether composite use is prohibited.

No scalar family score may be created merely because features share a family
name.

## Required validation principles

### Deterministic verification

- Use signals with analytical expected values.
- Test absolute features for the correct gain or scale equivariance.
- Test relative/shape features for invariance where theoretically expected.
- Reconstruct every stored raw estimate from saved ledgers.
- Test exact-zero, floor, empty-support, and boundary cases.

### Mechanism and discriminant validity

- Prespecify at least one dose-response positive control for each intended
  construct.
- Include plausible competing mechanisms, not only an easy null.
- Do not use diagnosis, outcome, or human-QC labels to tune the estimator.
- Novel estimators require independent calibration and evaluation simulations.

### Support and missingness

- Overlapping frames are dependent; resample whole intervals, utterances, or
  recordings as appropriate.
- Store total and eligible support, counts, support tier, and status.
- Support-class labels describe signal quantity only; do not call a class
  `robust` unless robustness itself was independently established.
- Treat availability transitions as a sensitivity and missingness result.
- Never publish a censored raw estimate under an `ok` status.
- Support thresholds are precision requirements, not clean/noisy thresholds.

### Boundary and signal-view robustness

- Report all finite paired estimates, not only pairs that remain available.
- Report unavailable-to-available and available-to-unavailable transitions.
- Distinguish full-estimator effects from common-support effects.
- State whether a perturbation is intended to test numerical repeatability or
  deliberately changes the operational construct. Do not impose an arbitrary
  stability threshold on the latter merely to create a pass/fail gate.
- State whether the measurement is native-stream, analysis-view, or calibrated
  physical level.
- Test transformations that can alter the feature: resampling, mono conversion,
  codec, filtering, normalization, or channel selection as relevant.

### Downstream correspondence (separate analysis)

- Do not place human-QC correspondence inside extraction notebooks or feature-freeze gates.
- Lock estimators before evaluating construct-specific blinded ratings or references.
- Use participant-clustered uncertainty when participants contribute repeated
  recordings.
- Report inconclusive or contrary evidence; do not retune on the same ratings
  and relabel the result validation.

## Figure and gallery contract

Every family must produce:

1. a construct/mechanism figure;
2. a support and robustness figure;
3. an empirical distribution/availability figure;
4. a within-family structure/reliability figure when cohort data permit;
5. a reviewer-facing gallery with a saved selection index.

Publication figures are saved as SVG, PDF, and 600-dpi PNG, with:

- physical units and unambiguous axes;
- sample sizes in the caption or companion table;
- color-vision-deficiency-safe colors;
- no diagnosis-based aesthetic unless diagnosis is the prespecified analysis;
- a caption and alt-text sidecar;
- no “representative” recording selected by hand.

The gallery is selected without diagnosis or human-QC labels using feature
quantiles, support extremes, censoring/status extremes, and family-specific
descriptor extremes. Gallery review may flag problems, but the gallery is not
used as a hidden training set.

## Freeze rule

A family remains `candidate_only` until every blocking gate passes and a named
reviewer records an acceptance rationale. Only then may its authoritative
implementation replace the legacy central registry/CLI implementation.

The freeze manifest must contain:

- measurement version and feature vector;
- frozen parameters;
- implementation hash;
- input-table hashes;
- gate-table hash;
- review decision, reviewer, and rationale;
- explicit statement that the manifest is frozen rather than candidate.
