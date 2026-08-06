# QTEMP v0.3.1-r2 finalization guide


## Purpose

This release finalizes the analytically supportable QTEMP subset without rewriting the validated v0.3 detector core.

Retained:
- bracketed dropout-like duration fraction;
- bracketed dropout-like event rate;
- near-exact consecutive decoded-repetition duration fraction, restricted to accepted targets of at least 40 ms;
- near-exact consecutive decoded-repetition event rate, restricted to accepted targets of at least 40 ms.

Dropped:
- abrupt splice-like discontinuity rate.

The splice feature failed held-out real-speech recovery and produced an implausibly prevalent, sample-rate-dependent cohort signal. It is formally excluded rather than threshold-tuned against the cohort.

## Runtime

The notebook reuses hash-compatible v0.3:
- participant-disjoint real-speech injection results;
- signal-chain characterization;
- all 519 recording caches.

New computation is limited mainly to:
- 12-recording, retained-detector parameter sensitivity;
- compact retained-event review gallery;
- final tables and gates.

Do not delete `outputs/02_features/temporal_discontinuity/qtemp-v0.3.0-measurement-development/checkpoints/recording_cache`.

## Exact workflow

1. Run the `FINALIZATION_EXECUTED_REVIEW.ipynb` notebook from top to bottom and save it.
2. Confirm G1 through G8 pass. G9 must remain pending until manual review.
3. Open:
   `outputs/02_features/temporal_discontinuity/qtemp-v0.3.1-finalization/gallery/qtemp_v031_blinded_adjudication_sheet.csv`
4. Review each row in randomized order using only its linked WAV and PNG:
   - `observable_discontinuity_yes_no_uncertain`: yes, no, or uncertain.
   - `observed_type_none_dropout_duplicate_other`: none, dropout, duplicate, or other.
   - localization fields: mark the apparent event within the 1.8 s clip when present.
   - `localization_acceptable_yes_no_uncertain`.
   - `confidence_1_to_5`.
   - `competing_mechanism`.
   - `audio_usable_yes_no`.
   - `review_comment`.
5. Save the CSV without changing `review_id`.
6. In Section 0, enter reviewer, ISO date, rationale, and set:
   `QTEMP_REVIEW_DECISION = "ACCEPT_QTEMP_V1"`
7. Rerun Sections 15–17. G9 passes only if the prespecified precision/negative-control thresholds pass.
8. Only after G1–G9 pass, set:
   `PUBLISH_AND_FREEZE_QTEMP_V1 = True`
   and rerun Section 16 once.
9. The notebook creates immutable `qtemp-v1.0.0` freeze outputs and a central four-feature table. It refuses overwrite.

Never edit the clean `_SOURCE.ipynb` notebook and never tune thresholds after viewing adjudication decisions.

## r2 correction note

This corrected freeze-candidate release removes the false G1 failure caused by
an incomplete same-ledger claim boundary. It also resolves nonretained splice
dependencies explicitly and removes execution warnings from identity
reconciliation, real-speech regex matching, and gallery spectrogram creation.

After a clean full run, all programmatic gates G1–G8 should pass. G9 remains
manual by design: review the blinded adjudication sheet before requesting the
immutable freeze.

## Clean-output deployment

The r2 deployment script moves any existing
`outputs/02_features/temporal_discontinuity/qtemp-v0.3.1-finalization`
folder to a timestamped project-root backup before opening Jupyter. This
prevents stale r1 artifacts from being mixed with the corrected run. The
expensive v0.3 development recording cache is not removed.
