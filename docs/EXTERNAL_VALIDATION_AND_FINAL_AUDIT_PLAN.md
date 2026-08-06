# External validation and final family audit

Version: `external-validation-audit-v0.1.0`

## Purpose

This stage answers a question that the family-specific analytical validation did
not, and should not, answer on its own:

> How do the reviewed indicators relate to established software outputs, how
> dependent are they on local windowing and support, and does any extractor need
> revision before scientific release?

The stage is deliberately downstream of the reviewed feature freezes. It never
edits frozen values in place and never interprets an external model as ground
truth.

## Why this audit is needed

Many reviewed indicators use short local windows. Short windows are appropriate
when the phenomenon itself is local, such as a clipping plateau, speech-offset
tail, or discontinuity. The scientifically relevant question is whether enough
independent local evidence contributes to the recording-level estimate and
whether the aggregation remains stable under reasonable changes in window,
guard, boundary, and support rules.

The audit therefore separates:

1. local estimator correctness;
2. effective recording-level evidence;
3. sensitivity to windowing and segmentation;
4. agreement with related existing metrics;
5. discrimination from competing mechanisms;
6. perceptual correspondence; and
7. the final permitted claim.

## Comparator classes

Every external relationship must be assigned one of these classes before
results are inspected.

| Class | Meaning | Permitted inference |
|---|---|---|
| Mathematical equivalent | Same estimand and implementation contract | Numerical equivalence can be tested directly |
| Related primitive | Same low-level quantity, different region or aggregation | Convergence is expected but equality is not |
| Algorithmic comparator | Different detector aimed at a related event | Agreement and disagreement cases require review |
| Perceptual comparator | Model or listener score for perceived quality | Supports perceptual relevance only |
| Complementary measure | Different construct expected to covary | Association is informative but non-validation |
| Negative control | Should remain insensitive to the target perturbation | Detects generic or confounded responses |

NISQA, DNSMOS, and SQUIM are primarily perceptual comparators. FFmpeg, librosa,
pyloudnorm, SRMR implementations, and Essentia can provide algorithmic or
primitive-level comparisons, depending on the exact output.

## Notebook order

1. `00_audit_registry.ipynb`
   - freezes the comparator hypotheses before results;
   - inventories every reviewed indicator and its role;
   - creates the feature-level audit ledger.

2. `01_external_comparators.ipynb`
   - resolves the audio manifest;
   - checks tool availability;
   - runs FFmpeg, native-sample, librosa, pyloudnorm, and optional Essentia
     comparators;
   - imports versioned NISQA, DNSMOS, SQUIM, and SRMR tables when configured.

3. `02_window_support_sensitivity.ipynb`
   - profiles feature availability and support;
   - inventories existing window, parameter, deletion, resampling, and
     invariance evidence from the reviewed family outputs;
   - identifies missing evidence that must be generated before a final verdict.

4. `03_convergent_discriminant_audit.ipynb`
   - tests prespecified indicator–comparator relationships;
   - uses participant-clustered bootstrap when participant identity is
     available;
   - controls FDR across tested comparator relationships;
   - exports disagreement cases for blinded inspection.

5. `04_final_family_verdict.ipynb`
   - builds the human-signed final evidence table;
   - assigns one of the following feature-level actions:
     `RETAIN`, `RETAIN_WITH_CLAIM_BOUNDARY`, `REVISE_AND_RERUN`, `DEMOTE`, or
     `REMOVE`;
   - requires a new measurement version before any extractor change.

## Required evidence for each indicator

A final verdict should not be signed until the following are documented:

- exact implementation re-audit against the mathematical specification;
- known-truth synthetic response and negative controls;
- window/hop/guard sensitivity;
- leave-one-pause, leave-one-boundary, leave-one-segment, or leave-one-event
  stability as appropriate;
- boundary and segmentation sensitivity;
- support distribution and missingness;
- comparison with the closest existing implementation where one exists;
- review of major disagreement cases;
- phenotype-confounding assessment;
- final claim and prohibited claim.

## Family-specific priorities

### QADD

Primary concern: source and gain ambiguity, not merely short frames.

Required additions:
- compare pause-level estimates with independently computed decoded-level/noise
  descriptors;
- repeat the estimator across plausible frame, hop, and pause-guard settings;
- perform leave-one-pause stability;
- test competing speech, breathing, vocal leakage, tonal hum, colored noise,
  gain, and channel filtering separately.

A competing-talker model may be added as a separate detector. It must not be
presented as equivalent to generic additive interference.

### QGAIN

Primary concern: physiological and task entanglement.

Required additions:
- compare guarded speech level with FFmpeg level and calibrated loudness
  summaries;
- assess frame and speech-edge guard sensitivity;
- perform leave-one-segment stability for within- and between-segment measures;
- test gain, microphone-distance proxies, compression/AGC-like transformations,
  fatigue-like level trends, and altered speech intensity separately.

Typical level remains nonordinal unless a downstream use defines a valid
two-sided operating range.

### QREV

Primary concern: boundary dependence and causal specificity.

Required additions:
- compare normalized SRMR with an independent reference implementation;
- test post-offset measures over multiple tail windows and boundary guards;
- perform leave-one-boundary stability;
- inspect disagreement among tail excess, persistence, SRMR, and perceptual
  reverberation;
- challenge the measures with trailing phonation, breath noise, additive noise,
  channel filtering, and inaccurate speech offsets.

Boundary measures may support residual-tail claims, not direct RT60 recovery.

### QCHAN

Primary concern: reference dependence and phenotype sensitivity.

Required additions:
- compare raw rolloff and high-band ratios with independent spectral
  implementations;
- verify invariance to STFT settings within a reasonable range;
- repeat reference construction under participant bootstrap and frozen
  reference vintages;
- inspect ALS/control and bulbar-stratified disagreement without using those
  labels to tune the metric.

A zero one-sided deficit does not establish absence of channel influence.

### QDIST

Primary concern: scope coverage, not local-window duration.

Required additions:
- compare sample occupancy near full scale and external peak statistics with the
  accepted plateau ledger;
- repeat detector parameters across plateau length, edge, singleton, and
  tolerance settings;
- test hard clipping, asymmetric clipping, limiting, soft clipping, compression,
  quantization, codec round trips, and naturally flat peaks;
- inspect every rare positive or a prespecified enriched sample.

The current family measures hard-clipping morphology only.

### QTEMP

Primary concern: detector validity and rarity.

Required additions:
- compare with an independent discontinuity detector;
- test deletions, insertions, repeated blocks, zero-filled gaps, packet-loss
  concealment, clicks, genuine pauses, weak dysarthric speech, and segmentation
  errors;
- require event localization agreement, not only recording-level correlation.

No primary QTEMP score should be released unless these gates support a new
validated indicator.

## Change-control rule

An external disagreement does not automatically authorize an extractor change.
A change is permitted only when all of the following are present:

1. a documented failure of the current estimand or implementation;
2. evidence that the proposed revision improves known-truth and natural-case
   performance without creating a worse competing-mechanism failure;
3. a signed feature-level verdict;
4. a new measurement version;
5. rerun of the affected family validation gates and cohort extraction; and
6. a migration note describing numerical non-equivalence with the previous
   freeze.

## Installation

Base audit functionality uses the repository dependencies and FFmpeg.

For lightweight optional comparators:

```powershell
python -m pip install -e ".[audit]"
```

NISQA, DNSMOS, SQUIM, SRMR reference implementations, and Essentia should be run
in version-pinned environments appropriate to those projects. Import their
recording-level outputs through `config/external_validation.yaml`; do not copy
model weights or third-party repositories into this repository.
