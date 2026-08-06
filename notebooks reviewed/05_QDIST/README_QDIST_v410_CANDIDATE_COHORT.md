# QDIST v4.1 full candidate-cohort package

This package continues from the accepted qdist-v4.1 remediation preflight. It
does not replace or modify frozen qdist-v3.1.1 outputs and cannot publish or
freeze qdist-v4.1.

## What it installs

- the full native-media candidate-cohort validation engine;
- the standardized A–J figure generator;
- the full cohort notebook and the later human-review adjudication notebook;
- 14 cohort/orchestration tests plus the existing 28 candidate-detector tests;
- the candidate-cohort scientific protocol and checklist source.

## Computational stages

The full cohort notebook performs:

1. accepted-preflight and frozen-input verification;
2. native-rate, channel-preserving recomputation for every eligible recording;
3. content-addressed per-recording checkpoints and exact ledger reconstruction;
4. governed qdist-v3.1.1 versus qdist-v4.1 comparison;
5. corrected same-polarity morphology-margin audit;
6. four matched realized burdens across symmetric, positive-only, and
   negative-only hard limits on label-blind cohort speech carriers;
7. support calibration at 3, 5, 10, 20, and 30 seconds;
8. 23 prespecified one-factor detector settings, 10/20/30/50-ms merge gaps,
   and deletion influence;
9. participant weighting, repeated-recording evidence, and related-view audit;
10. a human-only blinded review package containing every accepted plateau,
    near-threshold rejections, and valid-zero windows;
11. all A–J and candidate G gallery bundles, each with PNG, SVG, PDF, source
    CSV, caption, and provenance JSON;
12. a candidate checklist, ML interface, artifact hashes, and manifest.

## Construct boundary

The valid construct is visible hard-plateau morphology in the stored native
decoded waveform. The primary candidate feature is accepted channel-sample
support. Event rate is secondary. The 30-ms frame fraction is conditional/audit
because it depends on frame-grid origin. These outputs do not cover every form
of nonlinear distortion and do not localize an analog, codec, or processing
cause.

## Windows installation

Use the supplied `install_qdist_v410_candidate_cohort_r1.ps1`. It verifies every
package-file hash, backs up differing destinations, compiles the Python modules,
runs all 42 tests, verifies FFmpeg/FFprobe and the accepted preflight, and opens
the full cohort notebook in JupyterLab. It deliberately uses `jupyter lab`, not
`jupyter notebook`, because the latter command is not installed in the current
environment.

The installer does not run the long cohort workflow automatically. In the
opened notebook, run cells in order. The main cell is restartable with
`RESUME=True`.

## Outputs

Candidate outputs are written under:

`outputs reviewed/nonlinear_distortion/qdist_v410_candidate_cohort`

Important subfolders are:

- `tables` — recomputed values, ledgers, figure index, candidate ML interface;
- `validation` — reconstruction, injection, support, robustness, participant,
  repeat, checklist, and decision tables;
- `figures` — A–J and candidate G six-file bundles;
- `blind_review/items` — opaque review images and audio;
- `blind_review/restricted` — the withheld selection key;
- `audit` — errors, runtimes, provenance, and content-addressed checkpoints;
- `manifests` — candidate manifest and artifact hashes.

## Required human stage

Give `reviewer_1_TEMPLATE.csv` and `reviewer_2_TEMPLATE.csv` to two independent
reviewers together with the `items` folder. Reviewers must not see the
`restricted` key or compare sheets before both are complete. Rename completed
sheets as instructed by the adjudication notebook. Exact-label disagreements
require a documented adjudicated label and rationale.

Human review validates visible morphology only. If it motivates detector
revision, increment the measurement version and repeat review on a fresh or
held-out sample; do not tune and revalidate on the same items.

## Stop boundary

Even a computationally clean run remains candidate evidence. Cross-family
arbitration, two-reviewer completion, disagreement/failure-mode analysis, final
feature decisions, manuscript feature-census reconciliation, and a separate
immutable freeze workflow are still required. The code enforces:

- `scientific_review_decision = PENDING`
- `freeze_allowed = false`
- `publish_and_freeze = false`

