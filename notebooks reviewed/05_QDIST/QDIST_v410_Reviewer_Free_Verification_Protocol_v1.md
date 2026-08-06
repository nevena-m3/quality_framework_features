# QDIST v4.1 reviewer-free verification protocol

## Purpose

This revision removes the requirement for two unavailable manual reviewers without inventing reviewer labels or weakening the interpretation boundary. It is a post-cohort verification layer: the detector, thresholds, recording features, accepted plateau ledger, episode ledger, and cohort values do not change.

## Scientific basis

QDIST v4.1 is restricted to a digital, sample-level construct: accepted hard-plateau morphology in the stored native decoded waveform. A visual reviewer cannot provide an exact altered-sample mask and is not a calibrated reference for the causal acquisition stage. The strongest available criterion evidence is therefore the known transformation applied to the signal itself.

The verification hierarchy is:

1. **Criterion reference:** exact altered-sample masks from hard-limit interventions on label-blind cohort-derived speech.
2. **Analytical traceability:** exact reconstruction and exhaustive linkage of every accepted, boundary-rejected, and valid-zero audit item to governed ledgers.
3. **Real-cohort evidence:** operational detector outputs, explicitly not human-confirmed and not attributed to a particular device or analog stage.

Human review is recorded as **not performed / not used as the criterion reference**. It is not silently treated as completed, and AI labels are prohibited.

## G9 acceptance rules

- The 3-geometry × 4-dose × 12-carrier exact-mask grid must be complete.
- The previously declared moderate-dose occurrence sensitivity threshold must pass in every applicable cell.
- The previously declared moderate-dose median sample-precision threshold must pass in every applicable cell.
- Recall must be reported, including worst-case under-recovery; recall is not forced to unity.
- Every selected accepted plateau, boundary rejection, and valid-zero item must link exactly to the public index, source recording state, candidate decision, accepted ledger where applicable, episode ledger where applicable, and stored rejection reason.
- No human or AI morphology label may be generated.

## Feature decisions

- `qdist_hard_clipped_sample_fraction`: retain as the primary burden view.
- `qdist_hard_clip_event_rate_per_min`: retain as the secondary temporal-occurrence view, conditional on the 20-ms merge rule.
- `qdist_hard_clipped_frame_fraction`: retain only for audit/legacy compatibility; exclude from primary models because it is frame-origin dependent and redundant.
- `qdist_occurrence`: retain as a companion availability-aware status; do not count it as an independent family feature.

## Required wording

Use **QDIST hard-clipping morphology** or **accepted hard-plateau morphology in the stored native decoded waveform**. Do not call these outputs a complete measure of nonlinear distortion. Do not infer microphone/codec/analog-stage cause, full clipping sensitivity, disease, recording acceptability, or diagnostic status.

## Remaining tasks before immutable freeze

- Cross-family arbitration with QGAIN, QCHAN, and QTEMP.
- Manuscript wording and feature-census reconciliation.
- Separate immutable freeze after those integration tasks.

