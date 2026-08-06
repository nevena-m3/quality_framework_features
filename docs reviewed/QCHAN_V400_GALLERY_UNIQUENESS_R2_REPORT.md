# QCHAN v4.0.0 gallery uniqueness hotfix R2

## Diagnostic finding

The completed QCHAN cohort run was numerically and scientifically successful
through G1-G8 except for one figure-contract requirement:

- 14/14 notebook code cells executed;
- zero saved notebook errors;
- 519 recordings and 224 participants;
- 519 target spectra and 519 reference-ledger rows;
- zero extraction, reference, target-robustness, reference-robustness, and
  gallery-generation errors;
- every required main panel A-H/J present;
- all indexed figure artifacts present and nonempty;
- only 7 deterministic Panel G examples were indexed, while the declared
  minimum is 8.

The candidate manifest therefore correctly retained
`cohort_evidence_complete = false`.

## Root cause

The gallery code selected one top recording for each prespecified signal
stratum and then removed duplicate recording identities. Several related
spectral extrema occurred in the same recordings. Removing duplicates after
selection silently removed whole strata and reduced the gallery from the
intended 8-10 unique examples to 7.

This was not a failure of the QCHAN estimators, references, media, spectra,
feature values, robustness analyses, or figure-generation runtime.

## Correction

R2 introduces a tested, centralized deterministic gallery selector.

1. Prespecified strata are evaluated in fixed order.
2. Each stratum selects the highest-ranked unused recording.
3. When the leading recording is already represented, the next-ranked unused
   recording is selected instead of dropping the stratum.
4. If fewer than eight prespecified strata are available, remaining measured
   recordings are selected by deterministic support-, bandwidth-, and
   LTAS-stratified sampling.
5. A recording is never duplicated merely to satisfy the minimum.
6. Diagnosis, clinical outcomes, ALSFRS-R, and human-QC labels are not used.
7. The gallery gate now verifies both the minimum example count and unique
   recording identities.

The cohort orchestration identifier is advanced from v1 to v2. The QCHAN
measurement version remains `qchan-v4.0.0-candidate`.

## Scientific impact

No numerical feature definition, recording spectrum, LOSO reference, reference
vintage, support rule, missingness rule, parameter sensitivity result, or
recording-level feature value changes.

The installer preserves all spectrum and reference checkpoints. It archives
only the previous gallery and its dependent figure-contract/manifest outputs.
A complete notebook rerun reloads the checkpoints and regenerates the governed
gallery, final figure index, cohort checks, checklist state, and candidate
manifest.
