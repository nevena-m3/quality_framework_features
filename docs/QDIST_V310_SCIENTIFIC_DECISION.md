# QDIST v3.1.0 scientific decision record

## Decision

Proceed with QDIST v3.1.0 as a **candidate hard-clipping measurement release**. Do not freeze the family until the complete frozen cohort, empirical event morphology, parameter robustness, and blinded event/non-event review have been examined.

QDIST v3.1.0 supersedes the failed v3.0.0 candidate detector. The v3.0.0 run was technically complete but scientifically invalid: it admitted low-amplitude PCM quantization plateaus and ordinary locally flat extrema as apparent clipping. In that run, 40% of available recordings were positive, accepted edges were concentrated near digital zero, and the review gallery did not show convincing limiting levels. Those outputs remain a transparent negative-result audit and must not be used in downstream analysis.

## Construct and scope

QDIST v3.1.0 measures waveform intervals compatible with **hard clipping or saturation**. An accepted interval must jointly satisfy:

1. a short plateau at a polarity-specific waveform extremum;
2. meaningful amplitude relative to the recording and local context;
3. quantization-aware edge-distribution concentration; and
4. negligible waveform support beyond the proposed limiting level.

The detector does not measure total harmonic distortion, soft clipping, limiting, dynamic-range compression, automatic gain control, or the complete nonlinear-distortion process. Smooth saturation is characterized only as a competing mechanism. Lossy encoding or resampling may erase pre-existing plateaus; therefore, absence of detected evidence does not prove that clipping never occurred upstream.

## Authoritative input and exposure

The authoritative input is the first native-rate decoded audio stream, preserving source sample rate and channel geometry. No resampling, channel averaging, amplitude normalization, filtering, denoising, interpolation, or DC removal is allowed before QDIST inspection.

For declared integer PCM, the decoder uses an integer pipe and converts codes to float64 only as a numerical representation. The notebook verifies that 16- and 24-bit PCM geometry and quantization lattices are preserved. For lossy or floating-point sources, no integer lattice is invented.

The analysis exposure is all finite native samples within the frozen natural task span, from the first frozen speech onset to the final frozen speech offset, with internal pauses preserved. This avoids dilution by arbitrary leading and trailing idle time while retaining continuous waveform context. Whole-file identity and task-span boundaries remain auditable.

## Retained candidate outputs

- `qdist_hard_clipped_frame_fraction`: primary prevalence view; fraction of complete native task-span frames intersecting an accepted plateau on any channel.
- `qdist_hard_clip_event_rate_per_min`: primary event-frequency view; merged accepted episodes per minute of finite task-span exposure.
- `qdist_hard_clipped_sample_fraction`: secondary support-burden view; fraction of native channel-samples occupied by accepted plateaus.

All three outputs are reconstructed exactly from one accepted plateau/episode ledger. They are related views, not independent detectors. Final retention may differ by feature after empirical redundancy and review. No QDIST scalar is constructed.

## Quantization-aware detector contract

The v3.1.0 detector includes the following protections against the v3.0.0 failure mode:

- exact native PCM code-space analysis when a source lattice is declared;
- exact-code plateau runs for integer PCM;
- minimum plateau support, with a stricter rule for isolated levels;
- maximum plateau duration to reject frozen or square-like regions;
- boundary-step and local-extremum morphology requirements;
- minimum candidate magnitude relative to recording and local robust peaks;
- a minimum absolute PCM-code magnitude;
- histogram shells defined in representable code units rather than sub-LSB amplitude widths;
- minimum edge-zone support, excess over interior shells, concentration, and ratio;
- a 99.95th-percentile tail requirement with a strict allowance for samples beyond the proposed edge;
- repeated-level cluster support;
- fail-closed handling for source bit depths of 12 bits or less;
- fail-closed handling for low-entropy square-like, frozen, or malformed waveforms;
- deterministic bounded storage of rejected candidates while preserving complete counts.

A valid technically available recording with no accepted event receives numerical zero. A recording for which the detector cannot distinguish clipping from coarse quantization or low-entropy structure remains unavailable, not zero.

## Validation completed before cohort execution

The clean package passes 67 automated tests. Controlled notebook execution also passes the formula, construct, discriminant, transformation, and support contracts.

Key controlled findings:

- clean 16-bit quantized speech is a valid zero;
- hard clipping produces accepted events;
- all three features reconstruct exactly from the event ledger;
- polarity inversion and time translation preserve the measurements;
- post-clipping attenuation preserves detector interpretation;
- frame and sample burden are monotonic across the clipping-dose grid;
- sample-level precision is effectively 1.0 in the controlled grid;
- moderate/severe clipped-sample recall is approximately 0.84–0.94;
- detection generalizes across 8, 16, 44.1, and 48 kHz and across 16- and 24-bit PCM;
- clean speech, low-gain speech, sine, triangle, sawtooth, music-like, percussive, broadband-noise, impulse, click-train, DC-offset, and smooth-compression controls do not trigger;
- a 64-condition realistic 16-bit PCM null grid produces zero false detections;
- square-like low-entropy signals fail closed;
- moderate smooth saturation is not promoted to hard clipping;
- 8-, 10-, and 12-bit sources remain unavailable, while 16- and 24-bit clean controls remain zero and hard-clipped controls remain detectable;
- native 16- and 24-bit PCM decoding preserves exact geometry and source lattice;
- lossless PCM round trips are exact;
- resampling and lossy-codec effects are characterized rather than conflated with native measurements.

The v3.1.0 detector was also re-applied to the 16 examples from the failed v3.0.0 review gallery. It rejected or failed closed on all ten real false-positive/rejected examples, detected all three synthetic hard-clipping positives, and rejected all three synthetic clean-peak negatives. See `QDIST_V310_R4_GALLERY_RECHECK.csv`.

## Scientific validation figures

The notebook produces the same governed validation suite used by the other feature families:

1. formula and transformation validation;
2. synthetic construct validity and dose recovery;
3. discriminant, quantization, soft-scope, and bit-depth validation;
4. native/lossless/resampling/codec and real-recording parameter robustness;
5. empirical availability, structural-zero mass, and positive-part distributions;
6. accepted-event morphology, including code magnitude, recording-relative amplitude, tail percentile, edge concentration, more-extreme-sample contract, and histogram ratio;
7. feature redundancy and repeated-recording behavior;
8. blinded accepted/rejected/synthetic event review contact sheet.

Each governed figure is saved as PNG and vector PDF with machine-readable caption/alt text. Figure source tables are saved in CSV and Parquet alongside the estimates. Unavailable observations are not coerced to zero.

## Remaining blocking work

The full 519-recording v3.1.0 cohort has not been run in this environment because the frozen local media are not available here. The first local execution must remain a candidate run. Before freezing, review:

- extraction completeness and provenance;
- availability and reasons for unavailability;
- positive-recording prevalence without tuning thresholds to prevalence;
- accepted code magnitude and recording/context-relative amplitude;
- tail and edge-contract distributions;
- parameter and episode-merge robustness;
- source codec/sample-rate/bit-depth stratification;
- repeated-recording behavior;
- feature near-redundancy;
- blinded accepted-event, rejected-candidate, clean-peak, and synthetic-control gallery.

Final feature decisions are made independently. Failure of event rate, for example, need not invalidate frame fraction. At least one primary feature must pass for the family to freeze.
