# Remote Speech QC — Reviewed Pipeline, Stage 1

This package is a **parallel patch**. Extract it at the project root so that
`notebooks reviewed`, `src reviewed`, `tests reviewed`, `outputs reviewed`, and
`MAIN outputs reviewed` sit next to the existing folders. It does not overwrite
`notebooks`, `src`, `tests`, `outputs`, or `MAIN outputs`.

Stage 1 establishes the reviewed repository contract and a candidate redesign
of QGAIN. The QGAIN candidate is intentionally **not publication-frozen**. It
must be executed against the local frozen cohort, inspected, and explicitly
accepted before a freeze can be produced.

## Run order

1. Place these folders at the project root.
2. Open `notebooks reviewed/01_QGAIN/02b_gain_dynamics_QGAIN_v4_0_0_REVIEWED_SOURCE.ipynb`.
3. Confirm the project paths in the environment cell.
4. Run first with `RUN_COHORT_EXTRACTION = False` to reproduce synthetic gates.
5. Set `RUN_COHORT_EXTRACTION = True` only when the frozen media paths resolve.
6. Review the migration, support, non-identifiability, reliability, and gallery outputs.
7. Do not set `PUBLISH_AND_FREEZE = True` until the G10 decision table is approved.

## Scientific position of QGAIN v4 candidate

The family is renamed conceptually to **recorded level and level dynamics**.
The four retained measurements describe observable level behavior. They are
not source-identifying estimates of automatic gain control or device gain.
They are non-ordinal measurement-context variables and candidate inputs to
future biomarker-specific reliability models; none is a standalone recording
rejection rule.
