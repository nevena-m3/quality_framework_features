# QDIST Stage 0 Integrated Audit

## Decision

Proceed with a standardized reviewed QDIST preflight, using the frozen qdist-v3.1.1 detector as the numerical baseline. Do not overwrite or invalidate the existing qdist-v3.1.1 freeze. The reviewed release is named qdist-v4.0.0-candidate because the governance, output, checklist, and figure-package contracts are changing. A future qdist-v4.0.0 freeze is allowed only after exact cohort numerical equivalence to qdist-v3.1.1 is proven.

## Scientific construct

QDIST measures native-waveform intervals whose plateau morphology and polarity-specific amplitude-distribution evidence are jointly compatible with hard clipping or saturation. It does not measure the complete nonlinear-distortion term. Soft clipping, compression, limiting, AGC, THD, intermodulation distortion, general codec distortion, and perceptual distortion remain excluded.

## Retained measurements

1. qdist_hard_clipped_frame_fraction - primary prevalence view.
2. qdist_hard_clip_event_rate_per_min - primary event-frequency view, dependent on the frozen merge-gap rule.
3. qdist_hard_clipped_sample_fraction - secondary channel-sample burden view.

The three values are reconstructable views of one accepted plateau/episode system and are not independent detectors. No scalar is permitted.

## Legacy disposition

The qdist-v3.0.0 detector is a transparent negative-result audit because it admitted low-level PCM quantization plateaus. The qdist-v3.1.1 detector corrected that failure with native quantization-aware morphology, recording-relative and local prominence requirements, edge concentration, beyond-edge support, coarse-quantization guards, and square-like ambiguity handling. It was frozen successfully. The standardized review does not loosen those safeguards.

## Standardized evaluation

The reviewed family uses the common 50-item G1-G10 checklist and A-J figure framework. Panels A-C are analytical preflight figures. Panels D-H/J require the 519-recording cohort. Panel I is applicable because QDIST is an event detector and requires complete label-blind event verification.
