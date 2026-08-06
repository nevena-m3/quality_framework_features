# QCHAN v4.0.0 — Final scientific audit

## Executive decision

QCHAN v4.0.0 is scientifically accepted for finalization and immutable freeze. The completed candidate contains 519 recordings from 224 participants, 519 saved target spectra, 519 task-matched leave-one-subject-out reference assignments, one governed reference vintage, zero extraction/reference/robustness/gallery errors, and complete A–H/J validation evidence. All 50 checklist requirements are resolved. Panel I is explicitly not applicable because QCHAN contains no retained event detector or device classifier.

No raw audio, target spectrum, reference, or recording-level feature value needs to be recomputed. The post-cohort audit identified five presentation/evidence improvements—Panels E1, E2, E3, H1, and H3—which are regenerated from saved validation tables. These changes do not alter the four QCHAN analysis features or their signed precursors.

## Scientific construct and claim boundary

QCHAN measures spectral manifestations compatible with channel, device, platform, codec, source-bandwidth, or other spectral coloration effects. The measurements are computed from guarded strict speech and compared with a frozen task-matched, subject-balanced, leave-one-subject-out reference. They describe cohort-relative spectral deviation; they do not identify a device, recover a microphone transfer function, label a codec, or establish that a technical channel was the physical cause.

Speech spectra are jointly influenced by the capture chain and by phonetic content, anatomy, age, sex, vocal effort, dysarthria, articulation, additive noise, and task execution. Consequently, QCHAN is valid as a quality-context family only when interpreted with its reference provenance, target support, source-rate metadata, signed precursor values, and explicit confounders.

## Final retained features

### 1. Reference-relative LTAS distance

`qchan_ltas_distance_db`

Final role: **RETAIN_PRIMARY_NONORDINAL**.

The feature is the RMS distance, in dB, between the target one-third-octave log-LTAS and the frozen subject-balanced LOSO reference. Higher values indicate greater spectral departure from the reference, but not intrinsically poorer quality. It responds to both attenuation and positive/negative coloration, so it is the broadest QCHAN measurement and must remain nonordinal.

Cohort summary: available in 519/519 recordings; median 5.241 dB RMS; IQR 4.147–6.766 dB RMS; range 1.968–16.713 dB RMS. Repeated-recording Spearman correlation was 0.751 (participant-pair bootstrap 95% CI 0.663–0.817), and ICC(1) was 0.788 (95% CI 0.690–0.861).

### 2. Reference-relative rolloff-95 deficit

`qchan_rolloff95_deficit_hz`

Final role: **RETAIN_PRIMARY_ONE_SIDED**.

The feature is the positive part of reference minus target 95%-power rolloff, in Hz. Its signed difference is retained. A value of zero means that the target was equal to or higher than the reference on the signed scale; it does not mean absence of channel effects. The 95% rolloff fraction is part of feature identity.

Cohort summary: available in 519/519 recordings; median 0 Hz; IQR 0–560.35 Hz; range 0–1150.91 Hz; zero mass 50.87%. Repeated-recording Spearman correlation was 0.740 (95% CI 0.641–0.825), and ICC(1) was 0.743 (95% CI 0.634–0.830).

### 3. Reference-relative high-band ratio deficit

`qchan_highband_ratio_deficit`

Final role: **RETAIN_SECONDARY_NONINDEPENDENT**.

The feature is the positive part of reference minus target 3–7.5-kHz power share. Its signed difference is retained. It provides a complementary high-frequency-energy description, but is moderately redundant with rolloff deficit (Spearman ρ = 0.616) and showed the weakest repeated-recording persistence of the four QCHAN features.

Cohort summary: available in 519/519 recordings; median 0.000560; IQR 0–0.006878; range 0–0.012833; zero mass 48.17%. Repeated-recording Spearman correlation was 0.652 (95% CI 0.531–0.756), and ICC(1) was 0.701 (95% CI 0.585–0.796).

### 4. Reference-relative spectral-tilt steepening

`qchan_tilt_steepening_db_per_oct`

Final role: **RETAIN_EXPLORATORY_PHENOTYPE_SENSITIVE**.

The feature is the positive part of reference minus target robust log-LTAS slope, in dB/octave. The signed precursor is retained. Although repeatability was strong, tilt is highly sensitive to glottal source, vocal effort, dysarthria, phonetic composition, smoothing width, and fitting range. It is therefore retained for exploratory and sensitivity analyses but excluded from the default confirmatory manuscript feature set.

Cohort summary: available in 519/519 recordings; median 0.0746 dB/octave; IQR 0–1.483 dB/octave; range 0–6.351 dB/octave; zero mass 47.78%. Repeated-recording Spearman correlation was 0.774 (95% CI 0.681–0.848), and ICC(1) was 0.767 (95% CI 0.660–0.849).

## Target and reference support

All 519 recordings met the QCHAN target measurement contract. Of these, 518 were high-support and one was moderate-support. Guarded strict-speech support ranged from 7.028 to 93.64 seconds, with median 26.54 seconds. Every target had a task-matched LOSO reference containing all 223 other participants and 514–518 recordings. No global, cross-task, or insufficient-support fallback was used.

The cohort contained one task stratum, `PSG_BAMBOO`, and one governed reference vintage:

`74c0b334f69b5b7aa5d5fd8919810bbf8e40c7991f4485b754fcfabf0acd622e`

This vintage is part of feature identity. Values computed under another reference membership or parameter vintage are not silently interchangeable with qchan-v4.0.0.

Native source rates were 44.1 kHz for 215 recordings and 48 kHz for 304 recordings. Both provide Nyquist support above the 7.5-kHz analysis ceiling, so no cohort recording was classified as native-bandwidth-limited. The rate groups nevertheless showed different QCHAN distributions and must remain contextual metadata because sample rate can co-vary with platform, device, era, site, or participant composition.

## Numerical and implementation validity

The four features reconstructed from saved target spectra and reference ledgers across all 519 recordings. Signed precursors were preserved before one-sided truncation. Missing support remained unavailable rather than becoming numerical zero. The analysis used the canonical globally DC-removed mono 16-kHz waveform, guarded strict-speech frames, native-rate metadata, target-participant exclusion, subject-balanced reference aggregation, and exact reference hashes.

The reviewed test suite contains 47 tests after finalization additions. The local release expectation is 47 passed. The completed cohort notebook contained 14/14 executed code cells and no saved errors.

## Controlled response and discriminant validity

Progressive low-pass restriction produced ordered increases in LTAS distance, rolloff deficit, and high-band deficit. Shelves and notches demonstrated that LTAS distance also responds to nonordinal spectral coloration. Gain, polarity, DC offset, and common time shift were invariant. Source-rate and codec transformations were characterized rather than assumed invariant.

High-frequency noise could mask apparent low-pass deficits, and positive high-frequency coloration could create large LTAS distance while leaving one-sided deficits at zero. These results justify G5 as conditional: QCHAN detects spectral manifestations but cannot uniquely identify their source from single-channel no-reference speech.

## Robustness and uncertainty

Target-side sensitivity was evaluated on a deterministic 72-recording sample. Availability agreement was 100% across frame, hop, guard, boundary, and segment-deletion variants. Rank correlations were generally very high. The largest upper-tail changes remained feature-specific and are displayed in physical units in the corrected Panel E1.

Parameter variants confirmed that estimator definitions are part of measurement identity. One-octave and half-octave smoothing materially changed LTAS values; alternative 90% and 99% rolloff fractions materially changed rolloff deficits; high-band split and tilt-range variants changed their respective estimands. These tests do not invalidate the defaults—they establish that the frozen definitions must be pinned and reported.

Reference robustness was evaluated for 12 deterministic target recordings using 100 subject-bootstrap references, delete-one-reference-subject perturbations, 80% reference vintages, and recording-weighted alternatives. Iteration-wise bootstrap rank stability remained strong: median Spearman ρ was 1.000 for LTAS, rolloff, and high-band deficit and 0.992 for tilt; the minimum values were 0.979, 0.983, 0.928, and 0.922, respectively.

Participant-balanced and recording-weighted summaries were similar for LTAS and had small absolute shifts for one-sided features. Corrected Panel H3 displays participant-balanced 95% intervals so zero-median features are not visually blank.

## Redundancy and feature-set decision

The one-sided QCHAN descriptors overlap but are not identical. Rolloff deficit correlated with high-band deficit at ρ = 0.616 and with tilt steepening at ρ = 0.703. LTAS distance correlated only modestly with the one-sided features (ρ = 0.179–0.354), supporting its nonordinal complementary role. No family scalar is justified. A scalar would obscure direction, reference dependence, zero semantics, phenotype sensitivity, and different robustness classes.

The default manuscript set is:

- `qchan_ltas_distance_db`
- `qchan_rolloff95_deficit_hz`
- `qchan_highband_ratio_deficit`

The exploratory feature is:

- `qchan_tilt_steepening_db_per_oct`

## Figure-package audit

The candidate contains 22 applicable figure/example bundles: 14 main figures and eight deterministic label-blind Panel G examples. Every bundle includes PNG, SVG, PDF, source CSV, scientific caption, and provenance JSON. Panel I is explicitly N/A.

Panels A–D, F, G, H2, and J were accepted as produced. Five audit-only improvements are applied during finalization:

1. Panel E1 shows median and 95th-percentile target-side changes in separate physical units, preventing one-sided zero medians from appearing as absent evidence.
2. Panel E2 similarly shows median and 95th-percentile reference-robustness changes.
3. Panel E3 displays upper-tail parameter sensitivity in addition to median changes.
4. Panel H1 adds participant-pair bootstrap 95% confidence intervals for Spearman correlation and ICC(1).
5. Panel H3 adds participant-balanced 95% intervals, including for zero-median one-sided features.

The figures are regenerated exclusively from saved validation tables. `feature_values_recomputed = false` remains part of the final manifest.

## Final G1–G10 decisions

| Gate | Final status | Conclusion |
|---|---|---|
| G1 | PASS | Construct, provenance, reference identity, prohibited claims, and no-scalar contract are explicit. |
| G2 | PASS | Formula tests, signed precursors, missingness behavior, and 519-recording reconstruction are complete. |
| G3 | PASS | Gain, polarity, DC, time shift, source-rate, and codec behavior are characterized. |
| G4 | PASS | Low-pass dose response and coloration controls support construct response. |
| G5 | CONDITIONAL | Physical source is non-identifiable from speech spectra alone. |
| G6 | PASS_WITH_QUALIFICATION | Target, parameter, reference, vintage, and weighting robustness are complete; definitions remain part of identity. |
| G7 | PASS_WITH_QUALIFICATION | Cohort evidence, support, hashes, distributions, and eight examples are complete; rate-group structure remains contextual. |
| G8 | PASS_WITH_QUALIFICATION | Repeatability, uncertainty, redundancy, and weighting are complete; high-band is secondary and tilt exploratory. |
| G9 | N/A | No retained event detector or device classifier. |
| G10 | PASS | Feature-specific roles are accepted; no scalar, threshold, or device identity is authorized. |

## Final governance rules

- QCHAN values must travel with target support, source sample rate/Nyquist, reference subject and recording counts, task stratum, reference vintage hash, and signed precursors.
- One-sided zeros must not be interpreted as absence of channel effects.
- Missing values must never be replaced with zero.
- QCHAN must not be used as a standalone recording rejection rule.
- Clinical labels and human-QC labels remain downstream validation variables, not extraction gates.
- Any change to the four estimators, frequency limits, smoothing, target support, reference membership, aggregation, or reference vintage requires a new semantic measurement version.
- The numerical measurement freeze and standardized figure-package freeze are separate immutable artifacts.

## Final conclusion

QCHAN v4.0.0 is scientifically defensible as a reference-relative spectral-manifestation family. It is complete, fully supported in the present cohort, numerically auditable, reference-governed, robust to ordinary implementation perturbations, and appropriately qualified against source non-identifiability and phenotype confounding. It is accepted for finalization and immutable freeze without rerunning audio extraction.
