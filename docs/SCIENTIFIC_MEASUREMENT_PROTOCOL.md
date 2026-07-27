# Scientific measurement protocol

Version: 0.10.0  
Status: implementation-complete; empirical execution pending local audio and annotations

## 1. Scope and claims

Paper 1 establishes an auditable measurement framework for technical/acquisition conditions in remote Bamboo-passage recordings. The target is not a disease classifier and not a claim that all audio-quality proxies measure one latent construct.

For recording (j) from participant (i), each observed metric is treated as:

\[
Q_{ijk}=f_k(y_{ij}, M_{ij}, S_{ij}; \theta_k),
\]

where (y) is the decoded waveform, (M) is native stream metadata, (S) is a pre-specified segmentation view, and \(\theta_k\) is a frozen metric-specific parameter set. Each (Q_k) has a separate support denominator and failure status. Metrics are not interchangeable indicators of a single reflective factor.

Several speech-region measures can contain both acquisition and physiology. This is a limitation to estimate and report, not something that can be removed by renaming the variable “quality.” The registry records this confounding explicitly.

## 2. Data sources and unit of analysis

- **Primary measurement task:** Bamboo passage.
- **Primary technical unit:** one logical Bamboo recording, after collapsing alternate media encodings.
- **Dependence unit:** participant. All interval estimates retain participant clustering.
- **Rest data:** inventory and exact participant/date/protocol/iteration pairing only. Rest is a session-reference/sensitivity source; it is not passed through speech VAD as if it were a speech task.
- **Combined Rest_Bamboo workbook:** reconciliation check only. It is not appended to standalone workbooks.
- **Detailed human QC:** a distributed main set with one independent RA per recording,
  plus a crossed Reliability subset containing the same approximately 70 recordings for
  all four RAs. Both are preserved in item–rater–family long format with event-level
  intervals and are never pooled as a single crossed design. Rater identity comes from
  the four predeclared RA subfolder names, never the recording filename.
- **Broad metadata QC:** treated as a separate external label system. If individual ratings from its two RAs are unavailable, its inter-rater reliability is not estimable.

## 3. Immutable ingestion and provenance

Raw media and workbooks remain read-only. Each run records:

- absolute input path, byte size, and SHA-256 for files;
- full configuration hash;
- Python/platform/package versions;
- native codec, container, sample rate, sample format, channel count, bit rate, and durations;
- FFmpeg decode warnings/errors;
- source workbook and Excel row for every metadata record.

Filename, metadata, and disk inventory are three independent sources that must reconcile. A missing or duplicated path is an error, not a row to drop silently.

## 4. Metadata quality gates

Checks precede any analysis:

1. schema and required-value checks;
2. filename parse plus subject/date/protocol/iteration/task/extension agreement;
3. WAV/WEBM logical-recording duplication and cross-encoding metadata agreement;
4. within-participant conflicts in diagnosis and static demographics;
5. configured clinical sentinels converted to missing with an error ledger;
6. score range and subscore–total checks;
7. birth/symptom/diagnosis/assessment/recording chronology;
8. assessment-to-recording delta saved in days;
9. technical plausibility checks for duration, frame rate, dimensions, and sample rate;
10. column-level missingness, cardinality, numeric coverage, and extrema;
11. standalone-versus-combined export reconciliation;
12. metadata-versus-native-stream comparison after FFprobe.

Clinical primary analyses require a non-sentinel ALSFRS score with assessment date within ±14 days of the recording. ±30 days is sensitivity only. Scores are session-specific; participant-level carry-forward is prohibited.

Identifier patterns supplied by the data manager first create
`diagnosis_inferred_from_id`. They are promoted to `diagnosis_analysis` only in the
versioned freeze when the configuration records that the rule was reviewed and gives its
evidence source. Nonmatching missing diagnoses require an explicit `ALS`, `CONTROLS`, or
`EXCLUDE` adjudication. The source `Diagnosis` and `diagnosis_reported` fields are never
overwritten.

## 5. Audio views

### 5.1 Native view

FFmpeg decodes the first native audio stream to float32 while preserving native sample rate and all channels. There is no peak normalization, dynamic-range processing, amplitude clipping, or resampling. Native data are required for clipping, dropout, duplicate-window, channel, and stream-integrity evidence.

### 5.2 Analysis/VAD view

Channels are averaged only after the native multichannel view is retained. DC is removed, then polyphase resampling creates a 16-kHz mono view. This view is used for Silero VAD, comparable frame levels, and long-term spectral proxies.

The distinction is mandatory: resampling before QC can smooth clipping plateaus, alter exact zeros, hide channel differences, and impose an artificial bandwidth ceiling.

## 6. Segmentation measurement model

Silero VAD is installed as version 6.2.1 and loaded from the installed package, not
from an unpinned `torch.hub` branch. The primary candidate rule is the package default:
16-kHz mono input, threshold 0.5, 250-ms minimum speech, and 100-ms minimum silence.
The model is loaded once and reset by the official timestamp function between
recordings.

The scientifically timed output differs from the original display implementation in
two deliberate ways. First, primary Silero regions use `speech_pad_ms=0`; padding
expands a region by a requested amount and is not evidence for an acoustic onset or
offset. Second, there is no second 100-ms bridge or 250-ms filter after Silero has
already applied its duration rules. The primary table therefore retains the returned
sample indices (62.5-µs resolution at 16 kHz), subject only to clipping/normalization at
recording limits. This removes the systematic 50-ms expansion and avoids double
post-processing of pauses.

The familiar visible artifact layer remains: non-overlapping 30-ms diagnostic frames,
one frame CSV and one segment CSV per recording, and the original four-panel
waveform/RMS/mask/segment plot. Segment roles are `leading_nonspeech`,
`internal_nonspeech`, `trailing_nonspeech`, and `speech`; the plotted rows are labelled
`non-speech` and `speech`. The displayed mask can differ from an exact boundary by up
to one 30-ms bin. It is not the timing source used by the frozen interval table.

The legacy-named frame columns `speech_vad_raw`, `speech_vad_smooth`, and
`speech_mask_strict` intentionally remain identical because this is the original
artifact contract. They are diagnostic compatibility aliases and are not treated as
three sensitivity definitions.

Four distinct interval views are saved:

- `raw_speech`: direct unpadded, sample-indexed version-pinned model output;
- `primary_speech`: normalized direct output with no second bridge/filter pass;
- `strict_speech`: primary speech eroded by 50 ms at each edge;
- `strict_internal_nonspeech`: only pauses between speech regions, eroded by 200 ms away from both speech boundaries.

The nonspeech guard reduces speech leakage and reverberant-tail contamination. It does not turn pauses into guaranteed noise-only calibration intervals. Breathing, coughs, room sounds, and another speaker remain possible.

The original-style plot is accompanied by a boundary-audit table and figure. For each
exact onset/offset, a 120-ms local window separated from the boundary by a 20-ms guard
quantifies inside-versus-outside RMS contrast and the difference between the exact
sample edge and its 30-ms displayed representation. Contrast below 3 dB is a review
flag only. It never snaps or trims a boundary automatically because weak, breathy, or
gradually decaying ALS speech can have genuinely low energy contrast.

Two pre-specified sensitivity profiles rerun Silero rather than merely relabeling the
same timestamps. All profiles remain unpadded and have no second bridge/filter pass:

| Profile | Threshold | Minimum speech | Minimum silence | Speech-edge guard | Nonspeech-edge guard |
|---|---:|---:|---:|---:|---:|
| Primary | 0.50 | 250 ms | 100 ms | 50 ms | 200 ms |
| Conservative | 0.65 | 250 ms | 100 ms | 100 ms | 300 ms |
| Permissive | 0.35 | 100 ms | 200 ms | 25 ms | 100 ms |

These values are pre-specified operating points, not a claim of universally optimal
ALS boundaries. VAD performance is known to differ for dysarthric speech and by task.
Any manuscript claim about boundary accuracy therefore requires an independent,
diagnosis-blind manual boundary reference subset, participant-disjoint parameter
selection/validation, and reporting of onset error, offset error, speech overlap, false
speech duration, and missed-speech duration separately for ALS and controls. Parameter
changes made after viewing clinical or human-quality associations require a new
segmentation version.

### 6.1 Mandatory review selection

One prespecified task-validity rule is applied before segmentation adjudication:
an exact normalized frozen-metadata value
`Task Completed as Instructed = NO` is a locked automatic exclusion. This rule concerns
whether Bamboo was performed, not perceived acoustic quality. Missing values are not
silently interpreted as `NO`.

Among task-valid recordings, mandatory review selection is based only on segmentation
diagnostics. Diagnosis, clinical scores, Q metrics, and perceptual quality-family labels
are not used by the selection rule or displayed as review fields. This is not claimed as
fully blinded because the required recording filename/subject identifier can sometimes
reveal or suggest cohort membership.
Every automatically flagged or excluded recording is mandatory. An accepted recording
is also mandatory if it crosses a prespecified near-threshold guardrail or has an
extreme robust value among accepted recordings.

For each prespecified segmentation summary \(x\), the transparent robust score is

\[
z_i^* = 0.67448975\frac{x_i-\operatorname{median}(x)}
{\operatorname{median}(|x-\operatorname{median}(x)|)}.
\]

The accepted-reference set must contain at least 20 nonmissing recordings and have
nonzero median absolute deviation. The review threshold is
\(|z_i^*|\ge 4.5\). The features are duration, speech fraction, counts of speech/internal
nonspeech segments, leading/trailing nonspeech duration, longest internal pause,
frame-RMS median/SD, and boundary-contrast summaries. Additional accepted guardrails
are speech fraction <0.10, at least 20 speech segments, longest internal pause ≥4 s,
local low-contrast boundary fraction ≥0.50, or duration <2 s. These rules prompt review;
they never automatically exclude a recording.

### 6.2 Review and manual-boundary policy

The reviewer can scroll/search across every recording and sees the original four-panel
plot, exact-edge boundary audit, segmentation quantities, review reason, and an audio
player. Exactly three human adjudication states are allowed:

- `KEEP + AUTO`: retain the automatic primary Silero boundaries;
- `KEEP + MANUAL`: replace primary speech boundaries with documented manual intervals;
- `EXCLUDE + NONE`: exclude the recording with a documented reason.

Manual speech intervals must be finite, ordered, non-overlapping, within recording
duration, and accompanied by reviewer/date/reason provenance. Manual edits correct
speech onset/offset errors only. They must not remove background noise, clipping,
reverberation, gain changes, or other quality phenomena that Paper 1 is intended to
measure. A manually corrected primary view is eroded using the same 50-ms strict-speech
and 200-ms internal-nonspeech guards. Its raw/primary compatibility views both represent
the investigator-adjudicated speech support and are explicitly marked
`manual_override`.

Manual correction replaces only the primary profile. Conservative and permissive
profiles retain the original automatic Silero results, preserving a valid segmentation
sensitivity analysis.

### 6.3 Segmentation freeze

`segment-adjudicate` validates the complete review sheet and manual interval ledger,
then atomically writes a versioned immutable directory under
`MAIN outputs/01_SEGMENTATION_FREEZE/<version>` containing
`frozen_segmentation_decisions` and `frozen_segmentation_intervals`.
Every kept recording must contain positive-duration frozen primary speech. Feature
extraction reads only the frozen interval table; the unfrozen automatic table cannot be
used by the extraction command. Manual-review frame/segment CSVs, diagnostic PNG, and
exact-edge boundary audit are saved for every manually corrected recording. A separate
immutable publication tree, `outputs/01_segmentation_after_review`, materializes every
final frame/segment/figure and boundary audit under accepted, flagged, or excluded.
Accepted and flagged are analysis-eligible; excluded is audit-only.

## 7. Q-metric principles

The machine-readable registry is `src/paper1_qc/registry.py`. It defines role, units, artifact direction, signal region, minimum support, interpretation, and confounding for every metric.

### Additive interference

- guarded internal-nonspeech median level (dBFS);
- within-recording speech-to-nonspeech level contrast (SNR proxy);
- robust nonspeech-level IQR;
- 50/60-Hz harmonic prominence against local adjacent spectral bins;
- exposure-normalized high-energy nonspeech transient rate;
- spectral flatness as a secondary noise-type descriptor, not a monotonic severity score.

Support is metric-specific rather than an all-or-none family gate:

- nonspeech level, variability, and transient rate require at least 0.5 s of total
  guarded internal nonspeech and 20 complete nonspeech frames;
- the SNR proxy additionally requires at least 3 s and 100 complete frames of strict
  speech;
- spectral flatness requires at least 1 s total from pauses that are each at least
  250 ms;
- narrowband hum prominence requires at least 1 s from pauses that are each at least
  1 s, preventing zero-padding of short pauses from being mistaken for genuine frequency
  resolution.

Every metric has its own status field. An `ok` status must correspond to one finite
value, and a non-`ok` status must correspond to a missing value. The family status is
`ok` only when all five primary measures are available, `partial_support` when at least
one primary measure is available, and `insufficient_support` otherwise. Transient runs
are counted within each pause so events cannot merge across discontinuous intervals.
Relative spectral ratios use a machine-level numerical floor rather than a fixed
power offset, preserving global-gain invariance. The SNR proxy is explicitly not a
calibrated acoustic SNR.

### Gain and amplitude dynamics

- median active-speech level as a contextual descriptor;
- robust within-recording level IQR;
- SD of speech-segment median levels;
- absolute Theil–Sen level drift in dB/min using original recording time;
- large local level-step rate (secondary);
- true waveform peak/RMS crest factor (secondary).

No coefficient of variation is computed on logarithmic dB values. Speech level is not automatically oriented as worse because low intensity can reflect bulbar physiology, microphone distance, or gain.

### Reverberation tail

Speech offsets followed by ≥500 ms of pause provide early post-offset and late-pause windows. The noise floor comes from the end of the entire pause rather than the tail window itself; otherwise a slow tail raises its own reference and biases the effect downward.

- early tail excess above late-pause floor;
- time to remain within 3 dB of the late-pause floor, right-censored at 500 ms;
- robust early decay slope (secondary);
- SRMR as optional secondary evidence.

At least three valid offsets are required. These are reverberation-tail proxies, not RT60 or room impulse-response measurements. SRMR is also sensitive to speech production and noise.

### Channel/device

Native codec, container, sample rate, channels, and bit rate are retained as technical descriptors. Acoustic descriptors include effective speech bandwidth, high-band power ratio, and LTAS tilt. Because all speech-spectrum measures are source-dependent, they are not called device identifiers. The original corpus-wide LTAS distance is not primary; it was sample-dependent and susceptible to disease/source composition.

### Nonlinear distortion

Hard clipping requires sustained near-edge plateaus, not a single sample above an arbitrary 0.95 threshold. Metrics are computed on native channels and the worst channel is retained:

- hard-clip sample fraction;
- clip-event rate per speech minute;
- fraction of speech frames containing hard-clip evidence;
- near-full-scale fraction and edge-histogram spike as secondary evidence.

Lossy coding can obscure a previously clipped plateau, so a zero value is “no detected native-waveform evidence,” not proof that clipping never occurred upstream.

### Temporal discontinuity

All operations remain inside original contiguous speech intervals. Speech clips are never concatenated before differencing, because concatenation creates artificial discontinuities.

- fraction and event rate of near-zero runs ≥10 ms;
- near-identical adjacent non-silent window event rate;
- large frame-energy jump rate (secondary);
- robust sample-difference outlier rate (secondary).

Counts are support variables; exposure-normalized rates are primary. Natural plosives and periodic voicing remain possible confounders for secondary measures.

## 8. Failure and missingness semantics

Every family emits a categorical status. Insufficient support produces missing metric values, never zeros. True absence of a detected sparse event produces zero only after adequate support. Analyses use pairwise denominators and report missing/support rates by diagnosis, task, rater category, and participant.

Hard exclusions are limited to unavailable/undecodable media, task not completed, no usable speech, or severe segmentation failure. Broad labels such as “Poor Audio Quality,” background noise, and volume instability are outcomes/validation variables and must not be used to remove the very artifacts under study.

## 9. Validation ladder

1. deterministic unit tests for interval boundaries and metadata rules;
2. synthetic perturbations for additive noise, hum, gain drift, reverberation tails, clipping, dropout, repeated windows, and low-pass filtering;
3. native-versus-resampled and WAV-versus-WEBM comparison on paired encodings;
4. blinded stratified waveform/spectrogram review across metric quantiles and failure statuses;
5. detailed four-rater agreement, then matched perceptual-family linking;
6. paired shared-recording comparison with the separate merged two-RA broad labels for overlapping families only;
7. exact-session Rest sensitivity where scientifically appropriate;
8. segmentation-profile robustness;
9. physiology/confounding analyses using temporally aligned clinical values.

Passing a synthetic monotonicity test establishes implementation behavior, not clinical validity. Human correspondence establishes perceptual association, not pure technical causality. Competing speech and non-task content remain contextual annotations and are not counted as matched Q-family evidence.

## 10. Primary references

- ITU-T P.56, *Objective measurement of active speech level*: <https://www.itu.int/rec/t-rec-p.56-201112-i/en>
- Falk, Zheng, and Chan (2010), SRMR: <https://musaelab.ca/pdfs/J19.pdf>
- SRMRpy reference implementation: <https://github.com/jfsantos/SRMRpy>
- Silero VAD installed-package usage and supported rates: <https://github.com/snakers4/silero-vad>
- Hansen et al. (2021), waveform clipping assessment/detection: <https://pubmed.ncbi.nlm.nih.gov/35784517/>
- Laguna and Lerch (2016), histogram-based clipping detection: <https://musicinformatics.gatech.edu/wp-content_nondefault/uploads/2016/09/Laguna_Lerch_2016_An-Efficient-Algorithm-For-Clipping-Detection-And-Declipping-Audio.pdf>
