# QDIST v4.0.0 finalization implementation report

## Purpose

This patch converts the accepted qdist-v4.0.0 cohort candidate into a scientifically adjudicated, freeze-ready family while preserving exact numerical equivalence to qdist-v3.1.1.

## Implemented outputs

- final feature registry and feature-specific roles;
- completed 50-item G1-G10 checklist;
- ten-domain dashboard and final gate summary;
- final scientific audit and immutable-freeze contract;
- feature passports;
- standardized support-aware ML handoff metadata;
- regenerated sparse-positive, robustness, uncertainty, reliability, redundancy, weighting, event-verification, and handoff figures;
- 60 standardized event-review PNGs linked to source CSV and WAV excerpts;
- finalization notebook with acceptance token and numerical equivalence checks;
- atomic numerical freeze script;
- atomic figure-package freeze script;
- refusal to overwrite existing immutable outputs.

## Numerical policy

The candidate recording-feature table is copied and only qdist_measurement_version metadata is updated. The three analysis columns are compared before and after finalization with a 2 x 10^-15 tolerance. No detector execution, media decoding, plateau extraction, or episode reconstruction occurs during finalization.

## Figure policy

A-C and unaffected cohort/gallery figures are preserved. D2, E1, E2, E3, F, H1, H2, H3, I, and J are regenerated from saved candidate tables only. Each indexed bundle remains a six-artifact bundle. Panel I is applicable. All 60 event-review items receive a standardized four-panel visual; the linked WAV excerpt is the fifth review view.

## Release tests

The final test suite verifies identity, feature roles, prohibited scalar/gate behavior, ten-domain and G1-G10 completeness, interval helpers, candidate-manifest contract, five-view event-review contract, AI-assisted-review disclosure, cohort checks, safe notebook controls, and finalization equivalence.

## Required local workflow

1. Install the patch and run all reviewed QDIST tests.
2. Run the finalization notebook with ACCEPT_QDIST_V400 and PUBLISH_AND_FREEZE=False.
3. Save the fully executed notebook.
4. Run freeze_qdist_v400.ps1.
5. Only after numerical freeze succeeds, run freeze_qdist_figure_package_v100.ps1.
