# QDIST v4.0.0 — Preflight Acceptance and Cohort Authorization

## Decision

The uploaded reviewed preflight package is accepted. The local notebook contains 6/6 executed code cells, no saved Python errors, 28 passing tests, 18/18 blocking G1–G6 checks, and complete six-artifact bundles for Panels A–C. Cohort standardization and event verification are authorized; final scientific acceptance and freezing are not.

## Accepted preflight evidence

- Exact three-feature contract inherited from immutable `qdist-v3.1.1`.
- Clean speech, natural extrema, impulses/clicks, tones, noise, DC, clean 8/10/12/16/24-bit PCM, and smooth tanh saturation controls did not produce accepted hard-clipping episodes.
- Progressive hard clipping increased sample burden, frame prevalence, and episode rate; synthetic precision was 1.0.
- The detector is intentionally conservative at extremely sparse clipping burdens. Across the full synthetic dose set, mean recall was approximately 0.753; the smallest burden was below the detector's intended support threshold.
- Polarity, aligned time shifts, post-clipping attenuation, and lossless PCM preserved evidence. Resampling and lossy Opus/AAC processing erased native plateau evidence, confirming that QDIST must operate on the first decoded native-rate waveform.

## Cohort-stage design

The cohort notebook does not re-decode media and does not alter the detector. It reuses the immutable 519-recording `qdist-v3.1.1` freeze, independently reconstructs all three features from accepted plateau and episode ledgers, evaluates detector and merge-gap sensitivity, creates the complete 60-item event-review package, and produces standardized Panels D–J plus eight Panel G examples.

A full dry run against the exact immutable freeze produced: 519 recordings, 224 participants, 861 candidates, 30 accepted plateaus, 15 merged episodes, 6 positive recordings, 513 valid zeros, 60 event-review items, 23 figure bundles, and zero failed cohort checks. This dry run authorizes local execution but is not a substitute for packaging and post-run scientific review.

## Governing qualifications

QDIST detects accepted native-waveform plateau morphology compatible with hard clipping or saturation. It does not estimate total harmonic distortion, soft clipping, compression, limiting, AGC, codec distortion, perceptual distortion, or the physical acquisition stage at which the plateau was introduced. The three outputs are related views of one detector and may not be combined into a family scalar.
