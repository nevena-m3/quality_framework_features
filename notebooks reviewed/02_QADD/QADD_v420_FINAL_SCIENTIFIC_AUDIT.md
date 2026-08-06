# QADD v4.2.0 — Final Post-Cohort Scientific Audit

## Executive decision

The corrected QADD v4.2.0 cohort extraction is technically valid and scientifically useful. It uses the intended frozen interval contract, preserves missingness and floor censoring, reconstructs every raw estimand from saved ledgers, and completes the standardized A–H/J evidence package.

QADD should proceed to semantic/governance finalization and freeze with five retained measurements. No numerical estimator requires another raw-audio rerun.

One bookkeeping defect must be corrected before freeze: the cohort hum summary counts 50-Hz and 60-Hz winners across all finite raw estimates, including recordings that failed the final hum-support eligibility rule. The corrected eligible counts are 50 recordings with a 50-Hz winner and 163 with a 60-Hz winner, totaling the 213 eligible recordings. This does not change any recording-level feature value or joint-evidence decision.

## Cohort integrity

- Recordings: 519
- Participants: 224
- Primary and strict-speech intervals available: 519/519
- Strict internal pauses available: 513/519
- Media SHA-256 matches: 519/519
- Extraction errors: 0
- Robustness errors: 0
- Gallery errors: 0
- Raw-estimand reconstruction failures: 0
- Family scalar: not constructed
- Standalone reject threshold: not calibrated and prohibited

The production views are `primary_speech / primary`, `strict_speech / primary`, and `strict_internal_nonspeech / primary`. Strict speech and strict pauses are already guarded; no second guard is applied.

## Empirical feature summary

| Feature | Available | Availability | Median | IQR |
|---|---:|---:|---:|---:|
| Guarded-pause AC level | 462 | 89.0% | -65.50 dBFS | 15.38 dB |
| Guarded-pause level IQR | 440 | 84.8% | 5.96 dB | 8.18 dB |
| Speech–pause level contrast | 462 | 89.0% | 38.25 dB | 14.25 dB |
| Guarded-pause spectral flatness | 371 | 71.5% | 0.080 | 0.113 |
| 50/60-Hz hum-like comb prominence | 213 | 41.0% | 2.82 dB | 2.90 dB |

Seventeen recordings are floor-censored for the pause-level estimand. Floor-censored values remain unavailable in the analysis table while floor-inclusive and raw audit companions are retained.

## Repeated-recording persistence

These estimates describe within-subject persistence under real repeated recordings; they are not pure technical test–retest reliability.

| Feature | First–second Spearman | ICC(1,1) | Median absolute difference |
|---|---:|---:|---:|
| Guarded-pause AC level | 0.752 | 0.675 | 4.39 dB |
| Guarded-pause level IQR | 0.571 | 0.606 | 2.24 dB |
| Speech–pause contrast | 0.714 | 0.657 | 4.74 dB |
| Spectral flatness | 0.665 | 0.739 | 0.027 |
| Hum-comb prominence | 0.212 | 0.363 | 1.68 dB |

The hum feature's low persistence does not by itself invalidate the estimator because mains-like or machinery-like interference may be intermittent across sessions.

## Support and robustness

Whole-pause deletion was summarized relative to the cohort IQR.

| Feature | Median relative change | Population 90th percentile |
|---|---:|---:|
| Pause AC level | 0.050 | 0.263 |
| Pause-level IQR | 0.142 | 0.646 |
| Speech–pause contrast | 0.054 | 0.284 |
| Spectral flatness | 0.055 | 0.338 |
| Hum-comb prominence | 0.064 | 0.420 |

Pause level, contrast, and flatness are robust to deleting one whole pause for most recordings. Pause-level IQR and hum prominence are more sensitive in the upper tail and therefore require explicit support tiers.

Additional 100-ms pause erosion produced moderate feature changes but also changed availability for 15–28% of the audited recordings. Additional 200-ms erosion was intentionally severe and removed enough support to change availability in 22–53% of cases. These results support the fixed canonical pause contract rather than claiming parameter invariance.

Adding an unnecessary 50-ms speech erosion changed only the speech–pause contrast, with a 95th-percentile change of 0.22 dB. This confirms that the corrected v4.2.0 contrast removes the legacy duplicate speech erosion while remaining numerically close to v4.1.0.

## Redundancy and feature roles

The maximum absolute within-family Spearman correlation is 0.919 between pause AC level and speech–pause contrast. This relationship is expected because the contrast is the difference between speech and pause medians and speech level varies less than pause level across the cohort.

The contrast is retained because it has a distinct transformation property: it is invariant to uniform recording gain, whereas pause dBFS is gain-equivariant. It must nevertheless be labeled as mixed and non-independent, quarantined from any acquisition-only composite, and handled carefully in multivariable analyses.

Spectral flatness and hum-comb prominence are empirically distinct from the level features. Neither is an ordinal quality severity measure.

## Hum-like evidence

The raw hum-comb score is available in 213 recordings. A stricter interpretive companion requires both:

1. a score above the count-matched simulated colored-noise P95, and
2. at least three supported harmonics for the winning 50- or 60-Hz comb.

Eleven of 213 eligible recordings (5.16%) satisfy this joint evidence rule.

Among eligible recordings, 50 have a 50-Hz winner and 163 have a 60-Hz winner. The current cohort summary reports 74 and 218 because it counts raw winners in recordings that fail final support eligibility; this summary-only defect must be corrected before freeze.

The raw score remains a targeted descriptor. The joint-evidence field is an audit and interpretation companion, not a calibrated clinical or recording-rejection threshold.

## Final feature decisions

1. `qadd_pause_ac_level_dbfs_median` — **RETAIN_PRIMARY_CONTEXTUAL**
2. `qadd_pause_level_iqr_db` — **RETAIN_SECONDARY_CONDITIONAL**
3. `qadd_speech_pause_level_contrast_db` — **RETAIN_SECONDARY_MIXED_NONINDEPENDENT**
4. `qadd_pause_spectral_flatness` — **RETAIN_SECONDARY_NONORDINAL**
5. `qadd_mains_hum_comb_score_db` — **RETAIN_TARGETED_CONDITIONAL**

No QADD scalar is approved. No standalone accept/reject threshold is approved. Human-QC correspondence, ALS association, prediction, and biomarker-robustness analyses remain downstream scientific analyses and are not analytical-validation gates.

## Figure-package decision

Panels A–H and J are complete and auditable. Panel I is explicitly not applicable because no event detector is retained.

The final figure index must combine the preflight A–C panels, cohort D–F/H/J panels, and signal-linked G galleries using relative rather than machine-specific absolute paths. Every indexed figure must retain PNG, SVG, PDF, source CSV, caption, and provenance JSON.

## ML handoff

All five features may be exposed to future quality-aware multimodal pipelines only with:

- value
- availability mask
- status
- support tier
- support amount
- floor/censoring companions where applicable
- measurement version
- input-view identity
- hum calibration companions where applicable

The ML interface must not convert unavailable values to zero, expose a QADD scalar, or define a generic good/bad recording threshold.

## Finalization required before freeze

The finalization patch should:

1. preserve all five numerical feature columns exactly;
2. change candidate metadata to `qadd-v4.2.0`;
3. write the final feature roles and claim boundaries;
4. correct the eligible hum-winner summary;
5. mark G9 explicitly `N/A`;
6. generate the completed ten-domain dashboard and G1–G10 checklist;
7. create feature passports;
8. generate a combined relative-path figure index;
9. include the executed finalization notebook;
10. atomically freeze the family and publish canonical scientific, support, registry, and ML exports;
11. seal a separately versioned standardized figure package linked to the measurement-freeze hashes.

## Conclusion

QADD is ready for finalization, not another extraction redesign. The five features provide complementary views of recorded pause energy, nonstationarity, speech–pause separation, spectral structure, and targeted hum-like harmonic structure. Their value lies in exposing acquisition context and uncertainty, not in uniquely identifying environmental sources or declaring recordings globally good or bad.
