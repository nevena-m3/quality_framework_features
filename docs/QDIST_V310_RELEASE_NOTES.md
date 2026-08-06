# QDIST v3.1.0 release notes

## Why the version changed

QDIST v3.1.0 is a new estimator, not a patch to v3.0.0. The v3.0.0 detector was invalidated by empirical review because low-level PCM quantization plateaus were accepted as hard clipping. Existing v3.0.0 outputs and checkpoints are incompatible and must not be reused.

## Major changes

- New quantization-aware native-code detector.
- Integer-preserving native decoder for PCM16/PCM24/PCM32 sources.
- Task-span exposure with internal pauses preserved.
- Strict amplitude, morphology, edge-distribution, and beyond-edge support contract.
- Coarse-bit-depth and low-entropy fail-closed statuses.
- Exact plateau and episode ledgers.
- Atomic, identity-validated restart checkpoints.
- Exact Poisson intervals for event rate.
- Explicit rarity/structural-zero downstream guidance.
- Expanded synthetic, null, soft-scope, bit-depth, codec, resampling, and parameter validation.
- Eight governed validation/review figures aligned with QADD, QGAIN, QREV, and QCHAN.
- Blinded four-panel event review items with waveform, PCM-code/derivative context, amplitude distribution, spectrogram, and audio excerpt.
- Feature-specific G1–G10 decisions and immutable freeze safeguards.

## Compatibility

- New output root: `outputs\02_features\nonlinear_distortion\qdist-v3.1.0`.
- New immutable freeze root: `MAIN outputs\02_FEATURE_FREEZE\nonlinear_distortion\qdist-v3.1.0`.
- New central export stem after acceptance: `qdist_v310_analysis_features`.
- Do not delete the old v3.0.0 candidate output if it is needed as the negative-result audit, but do not analyze it.
