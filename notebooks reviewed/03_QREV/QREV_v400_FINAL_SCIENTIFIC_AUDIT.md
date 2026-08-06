# QREV v4.0.0 — Final scientific audit

## Decision

QREV v4.0.0 is scientifically ready for finalization and immutable freeze after one audit-only correction to the shorter-horizon sensitivity evidence and four figure-presentation corrections. The four retained outputs remain separate. No family scalar, source-identity claim, room-parameter estimate, standalone reject rule, or operational threshold is authorized.

## Cohort and implementation integrity

- Corrected cohort: **519 recordings from 224 participants**.
- Frozen media hashes verified: **519/519**.
- Internal natural speech-to-pause boundaries: **7,219**.
- Paired primary/strict frozen intervals: **7,738**.
- Boundary-to-recording reconstruction checks: **1,557**, all passing.
- Extraction, robustness, bandwidth and gallery error tables: **0 rows**.
- Global DC removal applied: **519/519**.
- Natural offsets come only from `primary_speech / primary`; `strict_speech / primary` is used only for SRMR support.
- Pinned SRMR identity: normalized-fast SRMRpy, `norm=True`, `fast=True`, `max_cf=30`, commit `fee0097...`, Gammatone 1.0.3.

## Final feature decisions

### 1. Early residual-tail excess — RETAIN_PRIMARY_CONDITIONAL

Availability is **132/519 (25.4%)**, representing 73 participants. The cohort median is **0.783 dB** (IQR **-0.919 to 4.726 dB**). Synthetic RIR ordering, numerical reconstruction and transformation behavior are strong. The feature remains conditional because 6,490 of 7,219 internal boundaries lack the required one-second pause/floor support, and because breathing, residual articulation, offset error and changing noise floors can mimic or alter the measured residual. It is an observable signed early-minus-late level contrast, not RT60, EDT, DRR or proof of room reverberation.

### 2. Bounded residual-tail persistence — RETAIN_SECONDARY_CONDITIONAL_NONINDEPENDENT

Availability is **132/519 (25.4%)**. The median is **0.0275 s** (IQR **0.0150–0.0913 s**). The estimator correctly recovers known persistence and explicitly right-censors at 0.6 s. At the boundary level, **19/639 (3.0%)** eligible values are censored. The raw one-boundary-or-more summary contains **5/208** recording medians at the horizon, whereas the final two-boundary analysis table contains one right-censored available recording; these denominators answer different questions and are preserved separately. Persistence is strongly related to tail excess (Spearman rho **0.832**, n=132), so it is retained as secondary non-independent evidence rather than a co-primary independent dimension. It is not reverberation time.

### 3. Downward residual decay rate — RETAIN_EXPLORATORY_CONDITIONAL

Availability is **88/519 (17.0%)**, representing 51 participants. The median is **21.17 dB/s** (IQR **11.18–36.96 dB/s**). The exponential-slope recovery is numerically excellent, but empirical repeated-recording persistence is weak-to-moderate (Spearman **0.415**, 14 paired subjects), delete-one-boundary sensitivity is material (median **2.56 dB/s**, 95th percentile **27.10 dB/s**), and the result is sensitive to the decay-window definition. It is retained for exploratory and quality-aware modeling with full support metadata, but is excluded from default confirmatory manuscript analyses.

### 4. Normalized-fast SRMR — RETAIN_ESTABLISHED_COMPARATOR

SRMR is available for **519/519 recordings**. The cohort median is **3.494** (IQR **3.143–3.880**). It responds in the expected direction to increasing simulated RIR dose and shows moderate repeated-recording persistence (Spearman **0.675**, 158 paired subjects). It is retained as a pinned published comparator, not as a reverberation-specific physical measurement. Additive noise, bandwidth, codec, speech content, pitch and dysarthria may influence it.

## Support-policy decision

The final conditional-feature policy remains a minimum of **two eligible boundaries**, consistent with the prespecified design. Tightening the policy to three or four boundaries lowers availability from 25.4% to 17.7% and 12.0% for tail/persistence, and from 17.0% to 9.6% and 5.4% for decay. Every eligible boundary already requires a stable one-second internal pause, so the two-boundary rule represents at least approximately two seconds of eligible pause support. The support classes are quantity descriptors only; bootstrap widths did not monotonically decrease with support, so they are not called precision tiers.

## Corrected shorter-horizon sensitivity

The cohort notebook's 0.4-s and 0.5-s persistence variants inherited the default minimum frame count of 50. At 30-ms frames and 10-ms hop, those shorter horizons contained too few frames and were incorrectly marked unavailable. This does not affect the default 0.6-s estimator or any recording-level QREV feature. Finalization replaces only those sensitivity rows. Because the default ledger stores either the first sustained return time or a 0.6-s censor value, shorter-horizon persistence is derived exactly as `min(default value, new horizon)`, with censoring updated at the new horizon. No raw audio or analysis feature is recomputed.

## Figure audit and corrections

All 22 applicable figure bundles were present with PNG, SVG, PDF, source CSV, caption and provenance. Panel I is explicitly N/A. Before freeze, finalization corrects four presentation issues:

1. Panel D3 is renamed and recaptioned as support versus bootstrap uncertainty; it does not call support a calibrated precision tier.
2. Panel E1 uses separate axes and units for tail excess, persistence and decay.
3. Panels E2/E3 incorporate the corrected shorter-horizon evidence.
4. Panel H3 uses feature-specific panels and units rather than plotting incomparable absolute differences on one axis.

## G1–G10 conclusion

G1–G4 and G7 pass. G5 remains conditional because source identity is not identifiable from single-channel no-reference audio. G6 passes with the explicit support/censoring qualification. G8 passes with qualification because decay remains exploratory and tail/persistence are non-independent. G9 is N/A. G10 passes under the feature-specific roles above. The numerical feature table remains exactly equivalent to the reviewed cohort candidate.
