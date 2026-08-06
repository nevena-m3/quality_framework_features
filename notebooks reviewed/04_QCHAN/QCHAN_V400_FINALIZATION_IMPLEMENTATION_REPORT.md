# QCHAN v4.0.0 — Finalization implementation report

## Purpose

The finalization patch converts the scientifically reviewed `qchan-v4.0.0-candidate` into a freeze-ready `qchan-v4.0.0` candidate. It does not decode audio, reconstruct target spectra, rebuild LOSO references, or recompute recording-level QCHAN values.

## Inputs verified

The finalization module requires the completed R3 cohort candidate and verifies:

- 519 recordings and 224 participants;
- 519 saved spectra and 519 reference-ledger rows;
- 224 unique LOSO reference identities and one reference vintage;
- all cohort evidence complete;
- eight Panel G bundles using the five-linked-view source contract;
- empty extraction, reference, target-robustness, reference-robustness, and gallery error tables;
- no scalar, standalone gate, or device identity.

## Finalization actions

1. Copy the completed candidate to `outputs reviewed/channel_device/qchan-v4.0.0`.
2. Remove transient `_archive` content from the final candidate.
3. Verify exact equality for the four analysis features and three signed precursors.
4. Write final G10 feature decisions, the ten-domain dashboard, final gate summary, and the resolved 50-item checklist.
5. Compute participant-pair bootstrap confidence intervals for repeated-recording Spearman correlation and ICC(1), using 5,000 iterations.
6. Compute iteration-wise reference-bootstrap rank stability from the saved 100-iteration reference-robustness ledger.
7. Regenerate Panels E1, E2, E3, H1, and H3 from saved validation tables.
8. Build a standardized 22-bundle figure index plus Panel I N/A.
9. Write four feature passports and the final freeze-ready manifest.
10. Copy the scientific audit, decisions, checklist, workbook, dashboard, and freeze contract into final-candidate provenance.

## Corrected audit figures

The following figures are regenerated because the original cohort versions were visually incomplete for zero-median or uncertainty evidence:

- `E1_window_boundary_sensitivity`
- `E2_reference_robustness`
- `E3_parameter_sensitivity`
- `H1_repeated_recordings`
- `H3_weighting`

The regeneration uses only saved analysis summaries. The final manifest records `feature_values_recomputed = false` and `audit_summaries_recomputed_from_saved_outputs = true`.

## Test coverage

The release contains the original reviewed QCHAN preflight and cohort tests plus seven finalization tests. Expected local result:

`47 passed, 0 failed`

The finalization tests verify feature roles, no-scalar governance, dashboard structure, checklist resolution, exact feature equivalence, participant-bootstrap confidence intervals, reference-bootstrap rank summaries, and the 22-bundle figure contract.

## Atomic freezes

`freeze_qchan_v400.ps1` validates the accepted final candidate and executed notebook, stages a complete copy, writes a hash inventory and freeze manifest, and atomically moves the sealed directory into `MAIN outputs reviewed/06_family_freezes/channel_device/qchan-v4.0.0`.

`freeze_qchan_figure_package_v100.ps1` validates the frozen measurement, copies the 22 figure/example bundles and final scientific documentation, writes a second independent hash inventory and manifest, and atomically moves the package into `MAIN outputs reviewed/07_figure_packages/channel_device/qchan-v4.0.0-figures-v1.0.0`.
