# Statistical analysis plan

Version: 0.3.0  
Status: freeze before inspecting Q–clinical or Q–human associations

## 1. Analysis population and dependence

The recording is the measurement unit and the participant is the resampling/inference cluster. Alternate encodings of one logical recording are technical replicates, not independent observations. Repeated visits are retained for distribution and persistence analyses but never treated as independent participants.

Primary measurement eligibility requires an available/decodable Bamboo recording, task completed as instructed, usable speech support, successful extraction, and no severe segmentation failure. Diagnosis is not required to characterize metric behavior, but diagnosis contrasts require confirmed reported ALS/control status. ID-inferred controls remain excluded until reviewed.

There is no global complete-case Q cohort. Each metric/family has its own support denominator. A flow table reports participant and recording counts at every gate.

## 2. Preprocessing

- No outcome-informed transformation selection.
- No automatic 1st/99th-percentile winsorization.
- No imputation of failed/unsupported Q metrics.
- Sparse-event absence is zero only when support is adequate; otherwise missing.
- Spearman associations operate on raw metric ordering.
- Participant persistence uses one fixed rank-normal transform for eligible non-sparse measures and is labeled rank persistence.
- Metric directions come only from the frozen registry. Contextual descriptors are not forced into a worse-is-higher composite.

## 3. Goal 1: occurrence, distributions, and acquisition variability

For each registered metric report:

- recordings and unique participants with adequate support;
- missing fraction and family status counts;
- median, Q1, Q3, full range, and zero fraction where meaningful;
- participant-clustered percentile-bootstrap 95% CI for the median;
- distribution by native codec/sample rate/channel count, confirmed diagnosis, and visit number as descriptive stratifications.

Sparse artifacts additionally require recording prevalence and event rate with participant-clustered confidence intervals. Counts without exposure denominators are never primary.

ALS/control contrasts are exploratory in Paper 1. To neutralize unequal visits, each participant contributes their median metric. Report the median difference and Cliff’s delta with group-stratified participant bootstrap CIs. Do not use raw recording counts as the effective sample size.

## 4. Goal 2: participant persistence

For non-sparse eligible metrics, fit a random-intercept model to fixed rank-normal scores:

\[
z(Q_{ij}) = \beta_0 + b_i + \epsilon_{ij}, \quad b_i\sim N(0,\sigma_b^2).
\]

The variance ratio \(\sigma_b^2/(\sigma_b^2+\sigma_e^2)\) is called **participant rank persistence**. It is not test–retest reliability because recordings differ in occasion, clinical state, device, environment, and potentially task implementation. Require at least 20 participants and 10 with repeated recordings. Skip measures with >80% zeros or inadequate unique values.

The original three-goal manuscript grouped persistence with distributional characterization. Version 0.3.0 separates it as Goal 2 so the repeated-measures estimand, eligibility, and interpretation are explicit.

## 5. Goal 3: multidimensional structure and reference robustness

### 5.1 Redundancy/structure

Compute pairwise Spearman correlations with participant-clustered bootstrap CIs and pair-specific denominators. Display a clustered correlation matrix, with a separate missingness/support matrix. Do not interpret a correlation cluster as a latent construct automatically.

Exploratory PCA/factor analysis is permitted only if:

- features meet pre-specified nonmissing/variance thresholds;
- sparse zero-inflated measures are not forced into a Gaussian PCA;
- one-recording-per-participant and repeated-recording analyses agree materially;
- component number is supported by parallel analysis;
- loading stability is evaluated by participant bootstrap;
- the manuscript calls the result exploratory.

No global Q score is permitted. Direction-oriented within-family indices are introduced
only for Goal 4 convergent-validity summaries. They are formative, sample-relative
percentile medians and remain secondary to the individual registered metrics.

### 5.2 Robustness/reference analyses

- Compare primary, conservative, and permissive segmentation profiles on the same logical recordings.
- Treat WAV and WEBM versions of one logical recording as technical replicates.
- Use Rest only as an exact-session acquisition-context sensitivity. Do not run speech VAD on Rest and do not substitute Rest for Bamboo speech pauses.
- For additive interference, compare Rest level/context metrics with Bamboo guarded-nonspeech metrics using exact participant/date/protocol/iteration pairs.
- Report matched-pair support and selection differences; the Rest subset is not assumed representative of all Bamboo recordings.

## 6. Goal 4: perceptual family alignment

### 6.1 Two distinct label systems

Analyze the four-RA detailed interval annotations separately from broad metadata QC performed by two different RAs. Never pool them as if they were interchangeable. If only a final broad metadata label is available, report prevalence/association but do not claim its inter-rater agreement.

The supplied detailed CSV structure contains one annotation layer per file and no
`rater_id`. Rater identity must therefore come from an explicit manifest or a validated
one-folder-per-rater design. Never infer the rater from the recording filename. The
primary detailed-label subset requires the expected four independent ratings for the
same recording/family.

### 6.2 Rater reliability before consensus

For each detailed category report:

- number of items per rater and complete across all raters;
- rating-level marginals and missingness by rater;
- observed pairwise agreement;
- Gwet AC1 and Fleiss kappa for nominal categories with item-bootstrap 95% CIs;
- ICC(2,1) absolute agreement and ICC(2,k) average-rating agreement for genuinely numeric/ordinal ratings, using complete items and item-bootstrap CIs;
- a category-by-rater confusion/marginal table.

Nominal Gwet AC1 is not labeled AC2. If the final scale is ordinal and AC2 is desired, add and validate an explicit weight matrix before the analysis freeze. ICC form, unit, agreement definition, and single/average rating must always be stated, following reliability reporting guidance.

Consensus is computed only after agreement reporting. Binary family presence uses
four-rater majority mode; a 2–2 tie remains missing and requires adjudication. Primary
consensus requires all four ratings. Three-of-four majority consensus is saved as a
separate sensitivity analysis. The median annotated fraction/duration across four raters
is a secondary extent label. The adjudication decision and blind status are recorded.

### 6.3 Family—not source—alignment

Map perceptual categories to mechanistic Q families before viewing associations:
environmental noise→additive interference, volume instability→gain dynamics,
reverberation/echo→reverberation tail, platform effects→channel/device, clipping→nonlinear
distortion, and temporal discontinuities→temporal discontinuity.

`Any non-task related content` and `Competing speech` remain contextual annotations but
are excluded from the primary matched-family estimand. “Poor overall audio quality” is
also excluded from matched-family validation because it is multidimensional.

Primary family alignment uses an analysis-specific objective family index:

1. include only pre-specified registry metrics with an explicit higher-worse or
   lower-worse direction;
2. convert each metric to its empirical percentile among eligible recordings;
3. reverse lower-worse metrics so larger always means greater artifact burden;
4. take the median only when at least half of eligible family metrics are observed.

This index is a formative, sample-relative summary for convergent validity, not a latent
factor or universal Q scale. Save the metric-direction and support audit.

- Binary human labels: ROC AUC, rank-biserial effect \(2AUC-1\), average precision,
  prevalence, and participant-clustered bootstrap CI.
- Ordinal/extent labels with ≥3 levels: direction-oriented Spearman rho with
  participant-clustered bootstrap CI.
- Report both class counts and participant counts.
- Do not estimate an AUC when either class has fewer than five recordings or clustered support is inadequate; report the category as not estimable.
- No SMOTE, oversampling, undersampling, or synthetic labels.

The full objective-family × human-family matrix is reported. The matched diagonal tests
convergent correspondence; off-diagonal cells are discriminant/specificity checks. A
generic association with every family is not evidence of family validity. Summarize
specificity as the mean matched rank-biserial effect minus the mean estimable mismatched
effect, with a participant-clustered bootstrap CI and the numbers of estimable matched
and mismatched pairs.

### 6.4 Paired 4RA versus merged 2RA comparison

Compare systems only on the same recordings, participants, direction, binary scale, and
overlapping explicit families. In the current metadata these are expected to be additive
interference (`Background Noise`) and gain dynamics (`Volume is Unstable`).

- Confirm from the 2RA codebook that `Yes=artifact present` and `No=artifact absent`.
- Normalize both systems to 0=absent, 1=present, higher=worse.
- For each shared family compute AUC against the same objective family index.
- Estimate paired \(\Delta AUC=AUC_{4RA}-AUC_{2RA}\) with participant-clustered bootstrap CI.
- Report prevalence and shared-recording/participant counts for each system.

Because individual 2RA decisions/reliability are unavailable, \(\Delta AUC\) is a
conditional observed-alignment comparison, not an intrinsic proof that one annotation
system is better. Do not compare binary AUC directly with the secondary continuous-extent
Spearman result.

If later regression is needed, use participant-clustered GEE or a mixed model appropriate to the outcome. Rare-event separation requires penalized/Firth methods or exact descriptive reporting, not unstable maximum-likelihood coefficients.

## 7. Imbalance policy

Diagnosis imbalance and artifact-label imbalance are different problems:

- **Unequal ALS/control participants:** participant-level effects and group-stratified bootstrap; no attempt to manufacture balance.
- **Unequal recordings per participant:** retain all recordings only with participant clustering; confirm with one-recording-per-participant sensitivity.
- **Rare human artifact categories:** report prevalence and uncertainty; block under-supported AUC/models rather than oversample.
- **Sparse Q events:** separate detection/prevalence from positive magnitude; do not run Gaussian models on zero-heavy counts.

Accuracy is not an acceptable headline metric for any imbalanced classification sensitivity. If prediction is ever introduced, report participant-split ROC AUC, PR AUC, sensitivity, specificity, PPV, NPV, balanced accuracy, calibration, and CIs—outside the primary Paper 1 measurement aims.

## 8. Clinical/confounding analyses

Clinical associations are secondary construct/confounding checks. Use only assessments within ±14 days; ±30 days is sensitivity. ALSFRS total and bulbar subscores must pass range/sentinel/chronology audits.

Pre-specify whether the question is:

- acquisition confounding by disease/severity;
- concurrent association with clinical state;
- modification of biomarker validity by quality.

Do not interpret a Q–severity association as proof that poor audio causes severity or that Q is purely technical. Adjusted analyses should include a minimal, non-collinear set (age, sex where appropriate, diagnosis/severity, codec/sample rate) and participant clustering. This line should remain subordinate to the measurement-validation aims; clinical prediction belongs in later papers.

## 9. Multiplicity and uncertainty

Effect sizes and 95% CIs are primary. If p-values are produced, Benjamini–Hochberg correction is applied **within a predeclared hypothesis family** (e.g., one human category across its mapped metrics), not across a changing list selected after results are seen. Report raw p, q, effect, CI, and denominator.

Bootstrap resamples participants for recording-level analyses and items for rater agreement. Default is 2,000 replicates with seed 20260713. Record successful replicates; unstable or non-estimable intervals are reported, not replaced.

## 10. Sensitivity analyses

Required:

1. conservative and permissive segmentation guards;
2. one randomly selected recording per participant using the frozen seed;
3. WAV versus WEBM technical-replicate comparison where both exist;
4. native versus 16-kHz estimates for metrics where resampling could matter;
5. removal of task-not-completed/severe segmentation failures only;
6. confirmed diagnosis only versus reviewed ID-pattern controls;
7. clinical alignment ±14 versus ±30 days;
8. exact-session Bamboo–Rest pairs only—never nearest-date pairing as primary;
9. broad merged 2RA metadata QC versus detailed four-RA QC, with paired shared-recording comparison only for overlapping families;
10. four-of-four primary consensus versus three-of-four sensitivity consensus;
11. binary presence versus median annotated-fraction extent labels, reported on their appropriate scales.

Robustness is summarized by denominator changes, Spearman agreement, median signed/absolute differences, and conclusion changes—not only by whether a p-value crosses 0.05.

## 11. Reproducible reporting

All manuscript counts/tables/figures must be generated from saved stage outputs. Report software/model versions, metric registry version, missingness, support criteria, failures, duplicate-encoding policy, participant clustering, assessment-date windows, human-rating design, and all deviations from this plan.

Useful reporting references:

- Koo and Li (2016), ICC selection/reporting: <https://pmc.ncbi.nlm.nih.gov/articles/PMC4913118/>
- Quarfoot and Levine (2016), prevalence sensitivity of multirater indices: <https://sci.sdsu.edu/crmse/msed/papers/quarfoot-levine.pdf>
- STARD 2015 transparency principles: <https://www.equator-network.org/reporting-guidelines/stard/>
