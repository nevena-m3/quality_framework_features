# Assessment of uploaded QTEMP v0.3.0 execution

## Decision

The uploaded v0.3.0 run is not freeze-ready.

Gates:
| gate   | state   | passed   | evidence                                                                              |
|:-------|:--------|:---------|:--------------------------------------------------------------------------------------|
| G1     | PASS    | True     | registry + provenance + subject identity                                              |
| G2     | PASS    | True     | package_tests=True                                                                    |
| G3     | PASS    | True     | native/mono/normalization/resampling/codec characterization                           |
| G4     | FAIL    | False    | detector-specific synthetic grids + held-out real speech                              |
| G5     | PASS    | True     | stop/low-level, periodic/tone, impulse, clipping, gain, speech-boundary controls      |
| G6     | FAIL    | False    | exposure ledger + stratified one-at-a-time profiles                                   |
| G7     | PASS    | True     | recording/candidate/disposition/event/exposure + prevalence/technical concentration   |
| G8     | PASS    | True     | exact reconstruction + participant identity + suitability                             |
| G9     | PENDING | False    | gallery_complete=False; sheet_complete=False; decision=PENDING_BLINDED_ADJUDICATION   |
| G10    | BLOCKED | False    | intentionally blocked for v0.3 development; create a new freeze version only after G9 |

## What passed

- 519/519 eligible recordings have a measured status.
- Native-waveform extraction, provenance, channel handling, cohort ledgers, exact feature reconstruction, subject identity, recurrence infrastructure, and empirical rare-event summaries completed.
- G1, G2, G3, G5, G7, and G8 passed.

## Blocking findings

### G4

The retained dropout detector passed. Near-exact repetition passed only in a narrowed duration scope. The splice detector failed held-out real-speech recovery and is dropped in v0.3.1 rather than threshold-tuned against the cohort.

### G6

All 17 parameter-sensitivity attempts failed because `time` had been shadowed by a NumPy array. v0.3.1 uses `import time as pytime` in every timed path and performs only a bounded retained-detector audit.

### G9

Blinded review was not completed. The prior gallery also contained 4 empty-clip failures. v0.3.1 clamps native times, generates candidate-free controls, preserves stable review IDs, and computes prespecified review metrics.

## Empirical evidence

- Dropout: 6 accepted events in 2 recordings; aggregate rate 0.015636 events/min.
- Near-exact decoded repetition: 0 accepted cohort events.
- Splice: 1,226 accepted events in 318 recordings (61.3%), with one-sample median support.
- Splice event rate was 5.653 events/min at 44.1 kHz versus 1.336 events/min at 48 kHz.

## Final scientific disposition

Retain:
1. `qtemp_dropout_duration_fraction`
2. `qtemp_dropout_event_rate_per_min`
3. `qtemp_frozen_audio_duration_fraction`, narrowed to near-exact consecutive decoded repetition >=40 ms
4. `qtemp_frozen_audio_event_rate_per_min`, same narrowed ledger

Drop:
- `qtemp_splice_discontinuity_rate_per_min`

Manual blinded G9 remains mandatory before the immutable qtemp-v1.0.0 freeze.

## r2 gate correction

The prior G1 failure was not a data or detector failure. It arose because the
near-exact repetition event-rate claim boundary did not contain the explicit
same-ledger wording required by the registry gate. The corrected notebook adds
that wording and reports exact failed gate components.
