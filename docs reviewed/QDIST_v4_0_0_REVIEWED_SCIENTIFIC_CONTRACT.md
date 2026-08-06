# QDIST v4.0.0 Reviewed Scientific Contract

## Estimand

Native-waveform evidence compatible with hard clipping or saturation during the continuous frozen natural task span. Internal pauses remain in place.

## Authoritative signal

The first decoded native-rate audio stream with original channel geometry. No resampling, channel averaging, amplitude normalization, filtering, denoising, interpolation, DC removal, or codec re-encoding occurs before QDIST inspection.

## Detection rule

A plateau is accepted only when every required morphology, amplitude, context, edge-distribution, terminality, and quantization guard passes. Exact signed-PCM rail plateaus and sub-rail plateaus follow the frozen qdist-v3.1.1 evidence pathways. Weighted compensation is prohibited.

## Outputs

- qdist_hard_clipped_frame_fraction
- qdist_hard_clip_event_rate_per_min
- qdist_hard_clipped_sample_fraction

These are related views of one event ledger. No family scalar or standalone reject threshold is allowed.

## Prohibited claims

No claim of total harmonic distortion, soft-clipping quantification, compression/limiting/AGC detection, source-stage identification, device identification, or globally distortion-free audio from a valid zero.

## Version continuity

qdist-v4.0.0 may freeze only if every recording-level feature and supporting event/plateau identity required by the equivalence contract matches the frozen qdist-v3.1.1 baseline.
