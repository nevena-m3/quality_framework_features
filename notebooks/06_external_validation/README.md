# Stage 06 — External validation and final family audit

This folder adds a downstream audit layer for the reviewed QGAIN, QADD, QREV,
QCHAN, QDIST, and QTEMP implementations.

It does **not** replace the family validation notebooks, edit frozen values, or
assume that any external quality model is ground truth.

Run in order:

1. `00_audit_registry.ipynb`
2. `01_external_comparators.ipynb`
3. `02_window_support_sensitivity.ipynb`
4. `03_convergent_discriminant_audit.ipynb`
5. `04_final_family_verdict.ipynb`

Before running:

```powershell
Copy-Item config\external_validation.example.yaml config\external_validation.yaml
Copy-Item config\external_validation_manifest.example.csv config\external_validation_manifest.csv
```

Edit both local files. Do not commit participant paths or generated model
outputs.

The full scientific and change-control contract is in
`docs/EXTERNAL_VALIDATION_AND_FINAL_AUDIT_PLAN.md`.
