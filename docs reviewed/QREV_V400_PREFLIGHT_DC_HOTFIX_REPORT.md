# QREV v4.0.0 Preflight DC-Contract Hotfix 1

## Diagnostic finding

The uploaded preflight completed normally, with 19 package tests passed, all nine notebook code cells executed, and no saved Python errors. Exactly one blocking check failed:

- Gate: G3
- Check: pinned SRMR gain/polarity/DC behavior
- Observed maximum relative difference: 0.0186192622 (1.862%)
- Required: <=1%

Gain and polarity were effectively invariant. The entire discrepancy came from adding a +0.25 constant DC offset to the raw SRMR fixture.

## Root cause

The QREV scientific contract defines the feature input as a globally DC-removed 16-kHz analysis waveform. The boundary estimators already use AC RMS and were invariant to the added constant. The preflight applied the DC perturbation directly to the upstream SRMRpy routine without first applying QREV's declared preprocessing. The module also described the input as DC-removed without enforcing that step inside `extract_qrev`.

This was a contract-enforcement and test-harness mismatch, not a failure of the pinned SRMR implementation, RIR-dose response, tail estimators, or dependency pinning.

## Correction

1. Add deterministic `remove_global_dc`.
2. Enforce it once at the start of `extract_qrev`.
3. Preserve the raw upstream SRMR fixture as the separate G2 regression.
4. Evaluate G3-G5 feature-level SRMR after QREV preprocessing.
5. Record pre- and post-removal waveform means.
6. Add two unit tests for constant-offset removal and SRMR-path enforcement.

No cohort values exist, so no numerical migration or measurement-version change is required.

## Expected rerun

- 21 tests passed, with no skipped pinned SRMR test.
- G1 PASS
- G2 PASS
- G3 PASS
- G4 PASS
- G5 CONDITIONAL
- G6 PREFLIGHT_PASS
- `preflight_blocking_checks_pass: true`
- cohort extraction disabled
- freeze disabled
