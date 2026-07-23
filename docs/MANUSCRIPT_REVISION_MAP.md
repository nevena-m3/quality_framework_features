# Paper 1 manuscript revision map

## Recommended paper identity

Suggested working title:

> A multidimensional measurement and quality-control framework for remotely collected speech in ALS

The paper should be the measurement foundation for the later audio papers. It should not be presented as an ALS detector, severity predictor, progression model, or proof that one global “Q” variable exists.

## Recommended aims

### Aim 1 — Feasibility and empirical behavior

Quantify support, failure, occurrence, and distributions of pre-specified quality/acquisition proxies in remote Bamboo-passage recordings, including dependence on native stream characteristics and repeated acquisition.

Primary outputs: per-metric support/missingness, robust distribution estimates, sparse-artifact prevalence/rates, and participant-clustered confidence intervals.

### Aim 2 — Participant persistence

Estimate how strongly participant ordering persists across recording occasions.

Primary language: “participant persistence.” Avoid “reliability” unless a true same-condition test–retest design is introduced.

### Aim 3 — Multidimensional structure and robustness

Quantify within- versus between-family structure, participant-collapsed stability,
segmentation/encoding sensitivity, and the exact-session Rest contextual reference.

### Aim 4 — Perceptual family alignment

First quantify reliability of the independent four-RA interval annotations. Then test
pre-specified matched family links, excluding source/task annotations. Analyze the older
merged two-RA metadata QC separately and compare it with 4RA consensus only on shared
recordings and overlapping explicit families after scale/direction normalization.

Primary links should be category-specific; overall poor-audio-quality correspondence is secondary.

## Confirmatory versus exploratory hierarchy

- **Descriptive primary:** denominators, failure rates, distributions, sparse-event prevalence.
- **Targeted validation primary:** pre-mapped detailed-human-family links using direction-oriented family indices and feature-level secondary analyses.
- **Secondary:** broad metadata QC correspondence, exact-session Rest checks, native-stream strata.
- **Exploratory:** full correlation matrix, clustering/PCA, ALS/control contrasts, clinical associations.

Do not define confirmatory hypotheses after viewing the detailed human-QC associations. The category–metric map and metric directions must be signed off first.

## Methods sections to replace

### Participants/data

Generate all counts from the final canonical-recording and eligibility tables. Distinguish media rows, logical recordings, sessions, and participants. Explain the WAV/WEBM technical-replicate policy, unresolved diagnosis workflow, repeated visits, and exact clinical-date window.

### Metadata/audio audit

Add a dedicated subsection describing schema, filename, disk inventory, native probe/decode, sentinel/range, chronology, cross-workbook, and error-ledger gates. Include a participant/recording flow diagram.

### Signal representation

State explicitly that native multichannel audio is retained and 16-kHz mono is a separate VAD/analysis view. Remove any methods language implying that all QC was measured after normalization/resampling.

### Segmentation

Report installed Silero version, parameters, raw timestamps, distinct primary/strict masks, guarded internal pauses, manual visual review design, and sensitivity profiles. Segmentation is part of the measurement model, not a neutral preprocessing footnote.

### Q metrics

Use the registry table as the authoritative count and definition source. For each metric report unit, signal region, support, aggregation, direction, and confounding. Do not describe channel-spectrum measures as device identifiers or reverberation-tail proxies as RT60.

### Statistics

Replace automated skew transforms/winsorization/family medians with the frozen statistical analysis plan. Explain participant clustering, pair-specific denominators, sparse-event policy, group-stratified bootstrap, no SMOTE, rater agreement before consensus, and familywise multiplicity.

## Results structure

1. Cohort/data integrity and flow.
2. Native media heterogeneity and extraction support/failures.
3. Metric distributions and sparse-event prevalence.
4. Redundancy/internal structure.
5. Participant persistence (clearly not reliability).
6. Four-RA agreement, crossed-design coverage, and consensus yield.
7. Paired 4RA versus merged 2RA alignment on shared families/recordings.
8. Category-specific perceptual correspondence.
9. Broad metadata QC correspondence (separate subsection).
10. Sensitivity analyses.

Do not put a global Q score or diagnostic-performance table in the main results.

## Tables and figures

- Table 1: cohort and media flow with participant/recording denominators.
- Table 2: frozen metric registry (family, metric, unit, support, direction, confounding).
- Table 3: distribution/support/prevalence estimates.
- Table 4: four-RA agreement and consensus completeness.
- Table 5: paired 4RA versus merged 2RA family alignment with direction/scale audit.
- Table 6: pre-specified perceptual links with clustered CIs and class counts.
- Figure 1: data/measurement pipeline and audit gates.
- Figure 2: distributions plus missing/support annotations.
- Figure 3: clustered Spearman matrix with pairwise denominators.
- Figure 4: human-category effects, not an undifferentiated correlation heatmap.
- Figure 5: segmentation/encoding/one-recording sensitivity summary.

## Claims to remove or qualify

- “Q is a latent recording property” → “Q metrics are observed, partially confounded proxies.”
- “High ICC demonstrates reliability” → “random-intercept variance estimates participant persistence across occasions.”
- “Channel/device feature” → “source-confounded channel/device spectral proxy.”
- “Reverberation/echo measurement” → “non-intrusive reverberation-tail proxy; not direct RT60/echo estimation.”
- “No clipping/dropout” → “no detected evidence after adequate support.”
- “Validated quality framework” → use only after the four-RA family analysis, paired 2RA sensitivity, and robustness gates pass.
- “Clinical biomarker correction” → reserve for a later paper; Paper 1 supplies the measurement layer.

## Reproducibility rule

The manuscript must never contain a manually maintained cohort or feature count. Counts, definitions, tables, and figures are generated from frozen run outputs and the registry. If the registry changes, the version, tests, outputs, and Methods table change together.
