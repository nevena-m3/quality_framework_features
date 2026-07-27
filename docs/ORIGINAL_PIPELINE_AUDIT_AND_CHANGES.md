# Original pipeline audit and implemented changes

This document records why the original implementation could not be used as the Paper 1 measurement engine without substantive repair.

## Repository-level findings

- The useful stage order existed (setup → segmentation → six families → assembly → analysis), so the rebuild preserves it.
- Core formulas were duplicated across large notebooks rather than versioned/tested functions.
- FFmpeg paths were hard-coded to inconsistent Windows locations.
- Silero was fetched from an unpinned `torch.hub` branch.
- The three segmentation masks named raw/smoothed/strict were assigned identically, so downstream notebooks implied boundary robustness that did not exist.
- Analysis notebook 06 contained nearly 10,000 source lines; notebooks 07 and 08 were zero-byte files.
- README and analysis-roadmap files were empty.
- The environment freeze captured an entire machine-specific environment rather than a minimal reproducible project specification.
- Old output directories contained large derived artifacts that could drift from current formulas and cohort metadata.
- The original Silero notebook's accepted/flagged/excluded diagnostic plots were useful
  for audit, but the downstream pipeline did not require a complete documented
  KEEP/EXCLUDE decision table.

The rebuild retains the original four-panel Silero diagnostic concept, saves one figure
per logical recording under `accepted/`, `flagged/`, or `excluded/`, and blocks Q-metric
extraction until every non-accepted recording is explicitly adjudicated.

## Measurement findings

### Audio decoding

Most feature notebooks decoded directly to 16-kHz mono. This erased native sample rate/channel/bandwidth evidence and could alter clipping/dropout morphology before QC was measured. The rebuild retains native multichannel audio and creates a separate VAD view.

### Additive interference

- internal nonspeech lacked a speech-boundary/reverberation guard;
- 50/60-Hz ratios used fixed narrow bands divided by total spectrum rather than local spectral prominence/harmonics;
- spectral flux carried state across separate pauses, creating boundary artifacts;
- a 95th-percentile pause value was named “peak median”;
- pause centroid was treated as a primary severity metric despite context-dependent direction;
- kurtosis threshold semantics were inconsistent with excess-kurtosis output.

The rebuild uses guarded pauses, local hum prominence, per-interval PSD aggregation, exposure-normalized transient rates, and explicitly contextual secondary spectral descriptors.

### Gain dynamics

- rolling analysis closed time gaps by compressing speech frames;
- slope was per frame rather than per unit time;
- “crest factor” was based on frame percentiles rather than waveform peak/RMS;
- segment-level coefficient of variation divided SD by the absolute mean of logarithmic dB values, which is not a meaningful CV;
- level was implicitly oriented as artifact severity despite physiological/mic-distance confounding.

The rebuild uses original time, dB/min drift, actual waveform crest factor, robust level spread, and no dB-scale CV.

### Reverberation/echo

- a heuristic boundary-blur formula had no interpretable physical scale;
- early/late ratios duplicated the same information;
- the late reference could remain contaminated by the same reverberant tail;
- global SRMR failures were swallowed into missing values;
- features were reverberation proxies but family language also claimed echo.

The rebuild narrows the construct to reverberation-tail evidence, estimates the floor at the end of the entire pause, requires three valid offsets, makes censoring/support explicit, and records SRMR dependency/failure status.

### Channel/device

- all acoustic evidence was measured after imposing 16-kHz mono;
- a corpus-wide LTAS reference was sample-composition dependent;
- speech spectral differences were at risk of being called device effects even though phonation/articulation can generate them.

The rebuild makes native stream properties primary descriptors and labels acoustic LTAS measures as source-confounded channel proxies. Corpus-wide LTAS distance is not primary.

### Nonlinear distortion

- any sample above 0.95/0.995 full scale could drive clipping metrics;
- near-full-scale occupancy was over-interpreted as nonlinear distortion;
- a 20-dB crest-factor reference was heuristic and source-dependent;
- metrics were computed after resampling.

The rebuild requires sustained edge plateaus on native channels and treats near-full-scale/histogram evidence as secondary.

### Temporal discontinuity

- speech samples were concatenated before waveform differencing, creating artificial discontinuities at every speech-interval join;
- raw event counts were primary despite unequal speech duration;
- repeated-window analysis was not restricted consistently to contiguous speech;
- natural energy changes could be conflated with glitches.

The rebuild performs all temporal operations inside original contiguous intervals, makes rates primary, and labels energy/difference events secondary.

## Statistical findings

- sample-driven skew transforms and 1st/99th percentile winsorization could make estimates dataset-dependent and suppress real rare artifacts;
- median family scores assumed commensurability and common direction without construct validation;
- participant random-intercept variance was labeled ICC/reliability although it quantified participant persistence across changing occasions;
- several analyses risked treating repeated recordings as independent;
- human correspondence and sensitivity workflows were placeholders;
- no complete strategy existed for rare human labels, rater agreement, clinical-date alignment, or class imbalance.

The rebuild uses raw-order robust statistics, participant clustering, group-stratified participant bootstrap, no synthetic balancing, explicit support missingness, separate rater reliability and consensus, and the term participant persistence rather than reliability.

## Intentional changes to the paper claim

Version 0.6.0 does **not** claim a single latent Q vector or a validated overall quality score. It establishes a multidimensional registry of observed proxies and tests their behavior, support, redundancy, persistence, robustness, and perceptual family correspondence. Direction-oriented within-family percentile summaries are used only as secondary formative convergent-validity indices; they are not a global Q score. A latent/composite score would require a later construct-validation step with stable loadings, measurement invariance, and external validation.
