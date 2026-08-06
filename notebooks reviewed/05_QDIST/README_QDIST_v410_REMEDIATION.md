# QDIST v4.1.0 scientific-remediation package

## Decision

`qdist-v4.0.0` should not be frozen. The package reproduced two blocking construct/implementation failures: sub-frame phase dependence of the 30-ms frame fraction and systematic failure of the original local-prominence rule for ordinary negative-only clipping.

`qdist-v4.1.0-candidate` is a non-overwriting development correction. It uses same-polarity local context for the local-prominence test, proposes clipped channel-sample fraction as the primary direct burden, retains episode rate as a secondary event view, and demotes frame fraction to a conditional/audit-only grid view.

Passing the included synthetic preflight does not authorize a freeze.

## Package contents

- `QDIST_Scientific_Audit_and_Remediation_v1.md`: scientific decision, evidence, blocking findings, feature-role recommendations, and G1–G10 remediation design.
- `QDIST_Master_Validation_Checklist_v1_1_REMEDIATION.csv`: the exact 60-item shared family checklist populated with PASS/CONDITIONAL/FAIL/PENDING/N/A decisions and item-specific evidence/action notes.
- `pipeline_files/src/paper1_qc/qdist_v410_candidate.py`: candidate detector; the original canonical detector is not overwritten.
- `pipeline_files/src_reviewed/paper1_qc_reviewed/qdist_v410_remediation.py`: executable synthetic/algebraic remediation preflight and A–C evidence-bundle generator.
- `pipeline_files/tests_reviewed/test_qdist_v410_candidate.py`: candidate regression and adversarial tests.
- `pipeline_files/notebooks_reviewed/05_QDIST/05_nonlinear_distortion_QDIST_v4_1_0_REMEDIATION_PREFLIGHT_SOURCE.ipynb`: source notebook with explicit freeze guards.
- `adversarial_audit/`: independently reproduced frame-phase, original-detector asymmetry, periodic-control, and event-review uncertainty results.
- `remediation_preflight_outputs/`: candidate dose, control, transformation, reconstruction, frame-phase, figure, caption, provenance, and manifest artifacts.

## Run from the pipeline project root

Open and run:

`notebooks reviewed/05_QDIST/05_nonlinear_distortion_QDIST_v4_1_0_REMEDIATION_PREFLIGHT_SOURCE.ipynb`

The notebook writes only to:

`outputs reviewed/nonlinear_distortion/qdist_v410_remediation_preflight/`

It must leave `freeze_allowed=false`.

The same preflight can be called from Python after placing both `src` and `src reviewed` on `sys.path`:

```python
from paper1_qc_reviewed.qdist_v410_remediation import run_remediation_preflight

evidence = run_remediation_preflight(
    "outputs reviewed/nonlinear_distortion/qdist_v410_remediation_preflight"
)
```

## Completed checks in this package

- exact feature reconstruction from accepted plateau and episode ledgers;
- 135 known-truth dose conditions across five native sample rates, three carriers, three clipping geometries, and three rails;
- one-sided polarity inversion and uniform post-clip gain behavior;
- all 1,440 possible 30-ms frame-grid origins at 48 kHz;
- repeated low-level saturation states spanning the candidate-generation floor;
- 72 synthetic discriminant conditions covering periodic, clean speech-like, DC-offset, noise, impulse, click, smooth-saturation, and compressor controls;
- source-linked Panels A–C in PNG, SVG, and PDF, each with caption and provenance JSON.

## Remaining freeze blockers

1. Rerun all cohort recordings under the candidate detector; prior v3.1.1 values cannot be copied forward.
2. Inject known hard clipping into independently screened held-out real control and ALS/dysarthric speech.
3. Add asymmetric unequal rails, time-varying rails, burst-duration, bit-depth, channel-geometry, native/resampled, and cohort-codec characterization.
4. Complete support-tier and full parameter-neighborhood calibration on locked evaluation data.
5. Inspect the complete output bundle, plotted source tables, event images, and WAVs.
6. Conduct randomized blinded review by at least two independent human/technical reviewers, preserve ambiguity/disagreement, and adjudicate without clinical labels.
7. Complete cohort occurrence, conditional-positive burden, participant-aware uncertainty, persistence, redundancy, and cross-family arbitration analyses.
8. Reconcile manuscript scope, causal wording, references, and the total feature census.
9. Populate Panels D–J and move every blocking shared-checklist item to a justified PASS before creating a new, non-overwriting freeze version.

## Environment note

The analytical module was executed successfully and all nine included blocking synthetic checks passed. Python syntax and notebook JSON/code-cell syntax were also verified. The current review runtime did not include `pytest`, so the pytest file was not run through the pytest command here; its covered checks were exercised directly by the executable preflight. Run the notebook's package-test cell in the project environment before cohort work.
