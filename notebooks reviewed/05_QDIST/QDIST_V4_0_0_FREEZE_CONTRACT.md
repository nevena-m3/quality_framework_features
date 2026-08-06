# QDIST v4.0.0 immutable-freeze contract

## Measurement identity

- Family: QDIST - native-waveform hard-clipping morphology.
- Final measurement version: qdist-v4.0.0.
- Numerical baseline: immutable qdist-v3.1.1.
- Figure package: qdist-v4.0.0-figures-v1.0.0.
- Finalization revision: qdist-v4.0.0-finalization-r1.

## Frozen features

1. qdist_hard_clipped_frame_fraction
2. qdist_hard_clip_event_rate_per_min
3. qdist_hard_clipped_sample_fraction

The three outputs are related views of one accepted plateau-and-episode detector. No family scalar is part of the measurement.

## Frozen input and preprocessing contract

QDIST operates on the first decoded native-rate audio stream while preserving native channels and sample geometry. No resampling, channel averaging, amplitude normalization, filtering, denoising, interpolation, DC removal, or re-encoding may occur before detection. Source and decoded hashes, native sample rate, channel count, sample format, codec/container, task-span exposure, and parameter hash are part of provenance.

## Frozen event construction

- Plateau detection and all morphology thresholds are inherited exactly from qdist-v3.1.1.
- Frame prevalence uses complete 30-ms native-waveform frames.
- Event rate uses the frozen 20-ms episode merge gap.
- Event rate must be accompanied by count, exposure, and exact Poisson interval.
- Sample fraction must retain channel-sample exposure and should be translated to clipped channel-ms/min for interpretation.

## Scientific claim boundary

Permitted claim: evidence of native-waveform plateau morphology compatible with hard clipping or saturation.

Prohibited claims: total harmonic distortion, complete nonlinear distortion, soft clipping, compression, limiting, AGC, codec distortion, perceptual distortion, causal device attribution, or proof of where clipping occurred.

## Support and missingness

Valid zero is not missing. Unavailable or indeterminate values remain NaN with explicit status and exposure. No imputation to zero is permitted. Support, counts, uncertainty, native geometry, hashes, version, and status must accompany ML handoff.

## Event-review qualification

The frozen package retains the complete 60-item label-blind event review and its AI-assisted adjudication status. It must not be represented as independent human ground truth. Thirteen rejected candidates remain ambiguous. Independent expert signoff is recommended before event-level precision claims are made.

## Immutability

The numerical and figure freeze scripts are atomic and refuse overwrite. Any change to a feature value, detector threshold, signal-view contract, frame rule, merge gap, exposure definition, status semantics, final feature role, or reference evidence requires a new semantic measurement or package version. Caption-only or formatting changes also require a new figure-package version.
