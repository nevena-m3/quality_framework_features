# QREV Stage 0 Integrated Audit

## Executive decision

The QREV scientific idea is valuable, and the four-feature architecture remains defensible: three conditional post-offset residual-tail measurements plus a broadly available, pinned SRMR comparator. However, the legacy `qrev-v3.1.1` boundary-dependent cohort values cannot be carried into the reviewed pipeline. They were calculated from the frozen `strict_speech / primary` view rather than natural `primary_speech / primary` offsets.

A second, independent design conflict was identified in bounded persistence. The legacy 1.0-s persistence horizon overlapped the 0.7–1.0-s late-floor window used to define return to floor. A tail persisting into the horizon therefore contaminated its own baseline, preventing honest 1.0-s right-censoring. The reviewed candidate resolves this by using a 0.6-s horizon and the independent 0.7–1.0-s floor window.

The existing numerical freeze should remain preserved as historical provenance, but it is superseded for the reviewed framework. A corrected extraction is required after analytical preflight.

## Intended construct

QREV asks whether energy persists after a natural speech offset in a pattern compatible with temporal smearing, reflections, or echo. It measures observable residual-tail and modulation-spectrum manifestations. It does not estimate RT60, EDT, C50/C80, D50, DRR, STI, a room impulse response, or causal room/echo identity.

## Legacy implementation defect 1: boundary source

The legacy notebook selected the first available view in the order `strict_speech`, `primary_speech`, `final_speech`, `speech`; because `strict_speech` existed, it was used for both SRMR support and post-offset boundary placement.

The frozen segmentation audit shows:

- 8,017 `primary_speech / primary` intervals.
- 8,017 matching `strict_speech / primary` intervals.
- Every strict interval starts exactly 50 ms later and ends exactly 50 ms earlier.
- All 7,473 internal speech-to-pause boundaries were therefore shifted 50 ms early.
- Every pause was inflated by 100 ms.
- 273 boundaries met the legacy 1.0-s pause requirement only because of strict-view inflation.

Consequences:

1. The first 100-ms “tail” window included approximately 50 ms of actual primary speech and only approximately 50 ms of true post-offset audio.
2. Pause eligibility and floor-window support were overstated.
3. Tail excess, persistence, and decay must all be recomputed.
4. The legacy boundary ledger did not preserve frozen left/right interval identities, reducing auditability.

Reviewed correction:

- `primary_speech / primary` defines the natural speech offset and next onset.
- `strict_speech / primary` is retained only as the speech-support quantity for SRMR.
- Boundary ledgers preserve left/right frozen interval indices and deterministic identities.
- Wrong view/profile, duplicate identities, and overlapping intervals are blocking errors; no silent merging or fallback is permitted.

## Legacy implementation defect 2: persistence horizon versus floor

The legacy persistence definition used:

- persistence horizon: 0–1.0 s after speech offset;
- independent late floor: 0.7–1.0 s after speech offset.

These windows overlap. If residual energy truly persists toward 1.0 s, the “floor” is itself residual energy. The threshold then rises with the tail, causing premature return-to-floor estimates rather than valid right-censoring. Consistent with this conflict, the legacy cohort had no right-censored boundaries and a maximum recording median of approximately 0.725 s.

Reviewed correction:

- persistence horizon: 0.6 s;
- late floor: 0.7–1.0 s;
- 100-ms separation between the horizon and floor;
- horizon values are explicit right-censored lower bounds;
- threshold, consecutive-frame rule, and horizon sensitivity remain cohort-stage validation items.

This is a justified registry amendment to the original Master Design. It narrows the observable horizon to preserve estimator validity.

## Feature-by-feature audit

### Early post-offset tail excess

Status: **REVISE AND RETAIN AS CANDIDATE**.

The signed difference between early 0–100-ms AC level and an independent 700–1000-ms late-pause floor is coherent as a residual-tail proxy. The legacy values are invalid because the early window started inside primary speech. The reviewed estimator preserves signed values, including negative estimates, and never clips to zero.

Claim boundary: residual energy above a local baseline; not RT60, DRR, or RIR recovery.

### Bounded tail persistence

Status: **REDEFINE AND RETAIN AS CANDIDATE**.

The persistence concept is valuable, but the legacy 1.0-s horizon was not independent of the floor. The reviewed version uses a 0.6-s right-censoring horizon and 0.7–1.0-s floor. It stores censoring per boundary and recording-level censored fraction.

Claim boundary: observable above-floor persistence within a fixed 0.6-s horizon; not reverberation time.

### Conditional downward tail-decay rate

Status: **REVISE AND RETAIN AS CANDIDATE**.

The negative-only Theil–Sen slope with a minimum dynamic-range gate is scientifically honest. Nondecaying, rising, nonsmooth, or low-range traces remain unavailable rather than zero. It must be recomputed from corrected primary offsets and tested for frame, resampling, codec, boundary, and floor sensitivity.

Claim boundary: conditional local envelope slope; not Schroeder decay, RT60, or a room parameter.

### Normalized-fast SRMR

Status: **RETAIN AS PINNED COMPARATOR PENDING RUNTIME VALIDATION**.

The implementation identity is frozen as SRMRpy normalized-fast with `norm=True`, `fast=True`, `max_cf=30`, 23 cochlear filters, low frequency 125 Hz, minimum modulation center frequency 4 Hz, upstream commit `fee009779cef96bed34db3a7e31d10f3ad1ea133`, and Gammatone 1.0.3. The official regression fixture must reproduce the pinned value exactly. RIR dose, additive-noise, codec, bandwidth, and duration behavior must be characterized.

Claim boundary: published no-reference reverberation-sensitive comparator; not reverberation-specific and not direct RT60.

## Legacy empirical context — not reviewed results

The legacy 519-recording table contained:

| Feature | Available | Availability | Median |
|---|---:|---:|---:|
| Tail excess | 86 | 16.6% | 5.63 dB |
| Bounded persistence | 86 | 16.6% | 0.075 s |
| Conditional decay | 72 | 13.9% | 34.09 dB/s |
| Normalized-fast SRMR | 519 | 100% | 3.50 |

These values are retained only to document migration. The three boundary-dependent distributions must not be used in the manuscript or downstream analyses. SRMR must still be regenerated within the reviewed provenance environment.

The legacy four-boundary policy materially reduced conditional availability. Based on legacy counts alone, two-boundary availability would have been approximately 171 recordings for tail/persistence and 145 for decay; three-boundary availability approximately 123 and 101; four-boundary availability 86 and 72. These are only planning estimates because the underlying boundaries were wrong. The corrected cohort will compare two-, three-, and four-boundary policies before G10.

## Common G1–G10 plan

- **G1:** corrected primary-boundary contract, separate strict support, pinned SRMR variant/dependencies, immutable registry.
- **G2:** hand-computable formulas, deterministic extraction, signed tail behavior, missing-not-zero, exact boundary-ledger reconstruction, official SRMR fixture.
- **G3:** gain/polarity/DC/common-shift invariance, deterministic source-rate resampling, codec characterization, SRMR duration behavior.
- **G4:** controlled RIR dose ordering, bounded-persistence recovery/censoring, exponential-slope recovery, SRMR RIR response.
- **G5:** stationary noise, changing late floor, breath, rising/nondecaying trace, delayed echo scope, SRMR noise/codec/bandwidth sensitivity.
- **G6:** two/three/four-boundary policy, boundary shifts, horizon/threshold/frame/floor-window variants, censoring and availability.
- **G7:** corrected cohort distributions, missingness/support, boundary masses, representative typical/high/confounded/failure examples.
- **G8:** repeated-recording persistence, participant-balanced summaries, redundancy among the three tail views, convergence with SRMR without requiring correlation.
- **G9:** N/A; no discrete event detector is retained.
- **G10:** per-feature retain/revise/demote/drop decisions, no family scalar, no standalone threshold, immutable numerical and figure freezes.

## Figure plan

- **A:** controlled RIR, persistence, decay, and SRMR responses.
- **B:** RIR target versus stationary noise, changing floor, breath, rising trace, delayed echo, and SRMR confounds.
- **C:** gain/polarity/DC/time shift, source-rate resampling, codec, and SRMR duration contract.
- **D:** corrected cohort support, availability, and censoring.
- **E:** boundary, horizon, threshold, frame, and floor-window sensitivity.
- **F:** conditional-feature distributions separately from broadly available SRMR; missingness never zero-filled.
- **G:** deterministic signal-linked examples: typical, high tail, censored, breath/noise confound, delayed echo, and failure.
- **H:** participant-aware persistence and redundancy.
- **I:** N/A unless a discrete echo detector is introduced later.
- **J:** support-aware ML handoff.

## Stage 0 decision

Proceed with `qrev-v4.0.0-candidate` analytical preflight. Do not use or freeze legacy boundary-dependent outputs. Do not run the corrected cohort until G1–G6 preflight passes, including the pinned SRMR runtime.
