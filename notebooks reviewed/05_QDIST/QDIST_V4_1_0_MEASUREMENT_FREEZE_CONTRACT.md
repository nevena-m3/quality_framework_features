# QDIST v4.1.0 immutable measurement-freeze contract

## Measurement identity

- Family: QDIST — native-waveform hard-clipping morphology.
- Version: `qdist-v4.1.0`.
- Criterion: exact altered-sample masks on label-blind cohort-derived speech.
- Real-cohort positives: operational detections, not human-confirmed physical clipping.

## Frozen roles

- `qdist_hard_clipped_sample_fraction`: primary direct burden.
- `qdist_hard_clip_event_rate_per_min`: secondary only if all prespecified exact-reference episode gates pass; otherwise automatically conditional/audit-only.
- `qdist_hard_clipped_frame_fraction`: conditional audit/legacy view; excluded from default models.
- `qdist_occurrence`: companion status; not counted as an independent feature.

No QDIST family scalar is constructed.

## Frozen input contract

The detector operates on the first decoded native-rate stream while preserving native channels. No resampling, channel averaging, amplitude normalization, filtering, denoising, interpolation, DC removal, or re-encoding may precede QDIST extraction. Source and decoded hashes, native geometry, task-span exposure, parameter hash, and status accompany the measurement.

## Frozen episode and frame rules

- Episode construction uses the governed 20-ms merge gap.
- Event rate is interpreted only as a detector-defined episode rate.
- Frame occupancy uses complete 30-ms frames anchored at the declared task-span origin.
- Frame occupancy remains conditional because alternative grid origins can change it.

## Claim boundary

Permitted: conservative accepted hard-plateau support in the stored native decoded waveform.

Prohibited: complete nonlinear distortion, THD, soft clipping, compression, limiting, AGC/DRC, codec distortion, perceptual distortion, causal acquisition-stage localization, disease independence, diagnosis, or standalone recording acceptance/rejection.

## Reviewer policy

No manual reviewers are required for the exact altered-mask criterion. The workflow must not synthesize reviewer forms, human labels, or AI morphology labels. Absence of human confirmation remains explicit.

## Measurement freeze versus publication integration

This contract permits an immutable measurement freeze after the automated gates pass. It does not represent manuscript wording, the global feature census, phenotype/content interactions, or empirical cross-family overlap as complete. Those remain downstream publication-integration tasks.

## Immutability

The freeze refuses overwrite. Any change to the detector, threshold, signal view, merge rule, frame rule, exposure definition, feature value, status semantics, feature role, or criterion evidence requires a new semantic version.

