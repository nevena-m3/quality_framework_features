# QDIST v3.1.0 — Candidate implementation notes

## Scientific scope

QDIST v3.1.0 measures native-waveform evidence compatible with hard clipping or saturation. It is not a general nonlinear-distortion estimator. It does not estimate THD, intermodulation distortion, soft clipping, compression, limiting, AGC, general codec distortion, or perceptual distortion.

The three analysis outputs are reconstructable views of one accepted plateau/event system:

- `qdist_hard_clipped_frame_fraction`
- `qdist_hard_clip_event_rate_per_min`
- `qdist_hard_clipped_sample_fraction`

Near-full-scale occupancy and histogram-edge concentration remain diagnostics only.

## Authoritative signal contract

The detector accepts the first native-rate decoded stream with all channels preserved. The canonical path applies no resampling, channel averaging, amplitude normalization, filtering, denoising, interpolation, DC removal, or codec re-encoding. Frozen speech intervals define only the continuous outer task span; internal pauses remain in place.

Production extraction requires a fully finite decoded task span. Any non-finite native sample makes the measurement indeterminate rather than silently changing denominators.

## Detector architecture

1. Generate polarity-specific flat-run candidates using quantization-aware tolerances.
2. Preserve local morphology: duration, unique levels, plateau range/slope, bilateral context, and directional entry/exit transitions.
3. Cluster candidates by channel, polarity, and limiting level.
4. Evaluate edge support, edge-to-interior concentration, and terminality within temporally contiguous candidate components. This local-component rule permits transient clipping at one gain state even when other recording regions legitimately exceed the same amplitude.
5. Apply explicit coarse-quantization and square-like/two-level ambiguity guards.
6. Accept a plateau only when every required criterion passes; weighted compensation is prohibited.
7. Merge accepted plateaus into episodes with a frozen 20-ms gap rule.
8. Reconstruct all three recording outputs from accepted plateau and episode ledgers.

## Candidate-release status

This implementation is intentionally not frozen. The notebook defaults to:

```python
PUBLISH_AND_FREEZE_QDIST_V31 = False
QDIST_REVIEW_DECISION = "PENDING"
```

A freeze requires all G1–G10 gates, package tests, full-cohort extraction, exact reconstruction, empirical parameter sensitivity, and a completed label-blind waveform review.

## Added files

- `src/paper1_qc/qdist.py`
- `tests/test_qdist_v31.py`
- `tests/test_qdist_notebook_v310.py`
- `scripts/generate_qdist_v31_notebook.py`
- `notebooks/02_feature_extraction/02e_nonlinear_distortion_QDIST_v3_1_0.ipynb`
- `docs/QDIST_v3_1_0_IMPLEMENTATION_NOTES.md`
- `docs/QDIST_v3_1_0_RUN_GUIDE.md`

## Validation completed before delivery

The focused automated suite contains 24 passing tests: 18 estimator tests and 6 notebook-governance tests. It covers valid zeros, hard-clipping recovery, separate bursts, merge-rule isolation, exact reconstruction, multichannel denominators, polarity inversion, post-clipping attenuation, soft-clipping scope, 8/12/16/24-bit quantization controls, sine and square-like controls, task-span mapping, unavailable-versus-zero handling, non-finite support, Poisson intervals, determinism, and notebook structure.

A separate Monte Carlo smoke audit used 40 clean speech-like signals and 80 clipped variants across 8, 16, 44.1, and 48 kHz. Clean false positives were 0/40; both full and transient hard-clipping conditions were detected in 40/40 cases per condition. This is development evidence only and does not replace the cohort and blinded-review gates in the notebook.
