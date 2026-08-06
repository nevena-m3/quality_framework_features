# QREV v4.0.0 Reviewed Preflight — Implementation Report

## Decision

Proceed with a new `qrev-v4.0.0-candidate` analytical preflight. Preserve the legacy `qrev-v3.1.1` freeze as historical provenance, but do not use its three boundary-dependent cohort features in the reviewed pipeline or manuscript. A corrected cohort extraction is required after the local pinned-SRMR preflight passes.

## Common validation framework

QREV follows the same family governance used for QGAIN and QADD:

- ten-domain family evaluation;
- G1–G10 gate record;
- feature-specific retain/revise/demote/drop decisions;
- Panels A–J with explicit N/A where scientifically inapplicable;
- source CSV, caption, provenance JSON, PNG, SVG, and PDF per figure;
- support-aware, non-imputed ML handoff;
- immutable numerical and figure freezes;
- no family scalar and no standalone accept/reject threshold.

## Critical legacy corrections

### 1. Natural-boundary contract

The legacy notebook placed post-offset measurements at the end of `strict_speech / primary`. That view is already eroded by 50 ms at each edge. Across the frozen interval audit:

- all 8,017 strict intervals ended 50 ms earlier than their matched primary intervals;
- all 7,473 internal speech-to-pause offsets were therefore 50 ms early;
- pauses were inflated by 100 ms;
- 273 boundaries met the legacy 1.0-s pause requirement only because of that inflation.

The reviewed contract uses:

- `primary_speech / primary` for the natural offset and next onset;
- `strict_speech / primary` only for SRMR speech-support duration;
- no fallback, merging, or profile pooling;
- preserved left/right frozen interval identities in the boundary ledger.

The legacy tail excess, persistence, and decay cohort values are superseded and must be recomputed.

### 2. Persistence horizon/floor independence

The legacy persistence horizon was 0–1.0 s, while its local floor was estimated from 0.7–1.0 s. A long tail could therefore contaminate its own baseline. The reviewed candidate uses:

- observation horizon: 0–0.6 s;
- independent late floor: 0.7–1.0 s;
- 100-ms separation;
- explicit boundary- and recording-level right-censoring.

The 0.6-s horizon is a deliberate scientific amendment that restores estimator validity. It is not a reverberation-time estimate.

### 3. Pinned SRMR identity

The comparator is pinned as:

- feature ID: `qrev_srmr_norm`;
- SRMRpy normalized-fast;
- `norm=True`, `fast=True`, `max_cf=30`;
- 23 cochlear filters;
- 125-Hz lowest acoustic center frequency;
- 4-Hz minimum modulation center frequency;
- upstream commit `fee009779cef96bed34db3a7e31d10f3ad1ea133`;
- Gammatone `1.0.3`;
- regression fixture value `3.7158141034373164`.

The installer enforces the dependency version and the local test suite requires exact fixture regression before the preflight can pass.

## Candidate features

| Feature | Candidate role | Reviewed action |
|---|---|---|
| `qrev_tail_excess_100ms_db` | Primary conditional residual magnitude | Revise boundary contract and recompute |
| `qrev_tail_persistence_median_sec` | Primary conditional bounded persistence | Redefine to 0.6-s horizon and recompute |
| `qrev_downward_decay_rate_db_per_sec` | Secondary conditional decay shape | Revise boundary contract and recompute |
| `qrev_srmr_norm` | Secondary published comparator | Retain only under pinned runtime identity |

No discrete-delay echo detector is included. Delayed echo outside the early window is shown as an explicit scope limitation rather than silently attributed to QREV.

## Reviewed analytical preflight

The preflight includes:

### G1 — Contract and provenance

- exact four-feature registry;
- primary-versus-strict interval separation;
- pinned SRMR implementation identity;
- no family scalar or threshold.

### G2 — Numerical correctness

- deterministic extraction;
- exact reconstruction from boundary ledger;
- signed tail values retained;
- nondecaying/rising traces remain unavailable, not zero;
- exact SRMR fixture regression in the local pinned environment.

### G3 — Transformation behavior

- uniform gain, polarity, DC offset, and common waveform/interval shift;
- 48-to-16-kHz source-rate conversion;
- FLAC, Opus, and AAC characterization on an above-quantization fixture;
- SRMR gain/polarity/DC and repeated-content duration behavior.

### G4 — Controlled construct response

- increasing controlled RIR doses;
- bounded persistence recovery and right-censoring;
- exponential-envelope slope recovery;
- normalized-fast SRMR RIR response.

The 0.8-s RIR condition is displayed as a late-floor/horizon stress condition and excluded from the monotonic operating-range acceptance test.

### G5 — Discriminant specificity

- dry signal;
- stationary pause noise;
- changing late floor;
- breath-like residual;
- rising/nondecaying trace;
- delayed echo outside the early-tail scope;
- SRMR additive-noise sensitivity.

Breath, changing noise, and echo remain explicit confounds. SRMR is treated as reverberation-sensitive rather than reverberation-specific.

### G6 — Support and estimator sensitivity

- provisional 2-, 3-, and 4-boundary support policies;
- raw estimates preserved when analysis values are suppressed;
- persistence horizon and threshold grid;
- primary-boundary shift grid;
- independent floor-window variants;
- explicit censoring.

## Figure package at preflight

Panels A–C are generated, each with:

- 300-dpi PNG;
- SVG;
- PDF;
- source CSV;
- scientific caption;
- provenance JSON.

Panels D–H and J require the corrected cohort extraction. Panel I is N/A because no retained event detector exists.

## Independent reference execution

In the container reference environment:

- non-SRMR tests: 18 passed;
- SRMR runtime test: skipped because Gammatone was unavailable in the isolated environment;
- all non-SRMR notebook checks passed;
- Panels A–C executed and were visually inspected;
- codec conditions preserved all three conditional feature measurements.

The local Windows installer enforces Gammatone `1.0.3`. Expected local result: **19 passed** and no skipped SRMR fixture.

## Current gate state

| Gate | Status before local execution |
|---|---|
| G1 | Contract pass |
| G2 | Pending local pinned SRMR runtime |
| G3 | Reference pass; local confirmation required |
| G4 | Reference partial pass; SRMR dose pending local runtime |
| G5 | Conditional by design; local SRMR noise characterization pending |
| G6 | Preflight pass; cohort support-policy decision pending |
| G7 | Pending corrected cohort |
| G8 | Pending corrected cohort |
| G9 | N/A |
| G10 | Pending |

## Next sequence

1. Install the reviewed preflight patch.
2. Run the local notebook with Gammatone `1.0.3` and the pinned SRMR fixture.
3. Package the executed notebook and candidate outputs.
4. Audit the local preflight.
5. Build the corrected 519-recording cohort extraction.
6. Complete Panels D–H/J and the empirical portions of G6–G8.
7. Make feature-specific G10 decisions.
8. Finalize and freeze the numerical family.
9. Seal the standardized figure supplement.

Freeze is currently prohibited.
