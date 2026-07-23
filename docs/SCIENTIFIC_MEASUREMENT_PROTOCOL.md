# Scientific measurement protocol

Version: 0.3.0  
Status: implementation-complete; empirical execution pending audio and complete detailed four-rater QC access

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
- **Detailed human QC:** four independent RA interval-annotation layers, preserved in item–rater–family long format with event-level intervals. Rater identity comes from an explicit manifest or validated RA subfolder, never the recording filename.
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

Identifier patterns supplied by the data manager can create `diagnosis_inferred_from_id`, but `diagnosis_analysis` remains missing until reviewed. An inferred label never overwrites the reported field.

## 5. Audio views

### 5.1 Native view

FFmpeg decodes the first native audio stream to float32 while preserving native sample rate and all channels. There is no peak normalization, dynamic-range processing, amplitude clipping, or resampling. Native data are required for clipping, dropout, duplicate-window, channel, and stream-integrity evidence.

### 5.2 Analysis/VAD view

Channels are averaged only after the native multichannel view is retained. DC is removed, then polyphase resampling creates a 16-kHz mono view. This view is used for Silero VAD, comparable frame levels, and long-term spectral proxies.

The distinction is mandatory: resampling before QC can smooth clipping plateaus, alter exact zeros, hide channel differences, and impose an artificial bandwidth ceiling.

## 6. Segmentation measurement model

Silero VAD is installed as version 6.2.1 and loaded from the installed package, not from an unpinned `torch.hub` branch. Raw timestamps are saved. Primary post-processing bridges gaps ≤100 ms and removes speech islands <250 ms.

Four distinct interval views are saved:

- `raw_speech`: direct version-pinned model output;
- `primary_speech`: bridged/filtered output;
- `strict_speech`: primary speech eroded by 50 ms at each edge;
- `strict_internal_nonspeech`: only pauses between speech regions, eroded by 200 ms away from both speech boundaries.

The nonspeech guard reduces speech leakage and reverberant-tail contamination. It does not turn pauses into guaranteed noise-only calibration intervals. Breathing, coughs, room sounds, and another speaker remain possible.

Two pre-specified sensitivity profiles alter only post-processing guards:

| Profile | Bridge | Speech-edge erosion | Nonspeech-edge erosion |
|---|---:|---:|---:|
| Primary | 100 ms | 50 ms | 200 ms |
| Conservative | 50 ms | 100 ms | 300 ms |
| Permissive | 150 ms | 25 ms | 100 ms |

## 7. Q-metric principles

The machine-readable registry is `src/paper1_qc/registry.py`. It defines role, units, artifact direction, signal region, minimum support, interpretation, and confounding for every metric.

### Additive interference

- guarded internal-nonspeech median level (dBFS);
- within-recording speech-to-nonspeech level contrast (SNR proxy);
- robust nonspeech-level IQR;
- 50/60-Hz harmonic prominence against local adjacent spectral bins;
- exposure-normalized high-energy nonspeech transient rate;
- spectral flatness as a secondary noise-type descriptor, not a monotonic severity score.

At least 3 s of strict speech, 0.5 s of guarded internal nonspeech, and 20 nonspeech frames are required. The SNR proxy is explicitly not a calibrated acoustic SNR.

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
