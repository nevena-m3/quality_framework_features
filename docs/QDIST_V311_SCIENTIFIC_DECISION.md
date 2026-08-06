# QDIST v3.1.1 scientific decision

QDIST measures evidence compatible with hard clipping or digital saturation in the native decoded waveform. It does not measure soft clipping, limiting, compression, THD, or complete nonlinear distortion.

Two pathways are frozen:

1. **Exact PCM rail:** exact signed-integer rail, at least three consecutive native samples, valid flatness, entry/exit morphology, local extremum and activity support, and no samples beyond the rail. Repeated same-level support is not required because the rail is a known numerical boundary.
2. **Sub-rail saturation:** arbitrary limiting levels retain the stricter repeated-level/singleton-duration, quantization-aware histogram, tail-percentile, edge-concentration, and negligible-beyond-edge contract.

One- and two-sample rail touches are not events. Soft-clipping experiments remain scope characterization only. The three ledger-derived recording values are related views, not independent evidence. Final inferential retention remains feature-specific after cohort review; event rate may be audit-only if it adds no stable information beyond frame prevalence.
