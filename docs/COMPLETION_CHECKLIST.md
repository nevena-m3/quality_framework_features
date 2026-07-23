# Completion checklist before manuscript results are frozen

## Files still required in this workspace

- all Bamboo audio files used for Paper 1;
- Rest audio files for exact-session sensitivity;
- the detailed HumanQC exports with independent rater identity for all four RAs and a shared four-rater subset;
- the rubric/codebook and scale direction for every detailed category;
- confirmation whether broad metadata QC contains independent ratings or only a merged/adjudicated value;
- adjudication of missing/inferred diagnosis and date/clinical anomalies from the metadata ledger.

## Decisions requiring investigator sign-off

- which encoding is original when both WAV and WEBM exist;
- whether `Task Completed as Instructed = No` is always a hard task-comparability exclusion;
- exact meaning/direction of `Needs Parsing` and why `Background Noise = Yes` occurs in ~90% of canonical Bamboo recordings;
- whether the supplied control-ID patterns are sufficient after manual review;
- whether ±14 days is acceptable for every clinical instrument or must be score-specific;
- human-QC consensus/adjudication rules and whether raters were blinded to metadata/model outputs.
- the two-RA broad-QC direction codebook (`Yes=artifact present` versus any alternative coding);
- the mapping from perceptual categories to Q families, with source/task annotations explicitly excluded from matched-family validity.

## Empirical validation gates

- [ ] zero unresolved metadata errors in the frozen primary cohort;
- [ ] 100% disk inventory reconciliation or documented disposition;
- [ ] zero silent decode/segmentation/feature failures;
- [ ] stratified waveform/spectrogram review across metric quantiles and statuses;
- [ ] synthetic monotonicity tests expanded to every primary metric;
- [ ] paired WAV/WEBM agreement report;
- [ ] primary/conservative/permissive segmentation robustness report;
- [ ] detailed RA marginal distributions and item-level completeness report;
- [ ] agreement CIs reviewed before consensus;
- [ ] category–metric map frozen before perceptual-link output is opened;
- [ ] participant/recording flow counts generated from code;
- [ ] all manuscript tables/figures regenerated from the final run manifests.

## Items deliberately deferred

- diagnostic or progression prediction;
- SMOTE/oversampling or predictive class balancing;
- clinical biomarker correction using learned Q weights;
- a single overall Q score;
- causal claims that technical quality produces clinical severity differences.
