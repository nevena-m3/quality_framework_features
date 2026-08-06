# QDIST scientific audit and remediation record

**Review target:** proposed qdist-v4.0.0 final package, derived without numerical change from qdist-v3.1.1  
**Corrected development target:** qdist-v4.1.0-candidate  
**Review scope:** feature construct, estimator, detector implementation, analytical validation, figure/checklist governance, scientific lineage, and manuscript handoff  
**Current decision:** **DO NOT FREEZE qdist-v4.0.0**

## 1. Executive scientific decision

The overall research program is coherent and valuable. Paper 1 can provide the acquisition-measurement layer required by the later shortcut-learning, biomarker-robustness, and fit-for-purpose studies. QDIST, however, is not presently ready to be frozen.

The current package establishes exact reconstruction of its saved ledgers and preserves native-waveform provenance well. Those are important strengths. They prove that the recording-level values reproduce the frozen qdist-v3.1.1 implementation. They do **not** establish that the detector is scientifically valid across the intended hard-clipping construct.

Two blocking failures were independently reproduced:

1. The 30-ms frame fraction is phase-dependent. A 5.44-ms common shift preserved accepted plateaus, event count, event rate, and channel-sample burden but changed the frame fraction from 0.170 to 0.165. Across all 1,440 sub-frame origins at 48 kHz, the value ranged from 0.150 to 0.170 (11.8% maximum relative change). The existing G3 “common time-shift invariance” PASS tests only a shift of exactly one 30-ms frame and therefore cannot detect this failure.

2. The detector misses asymmetric clipping under ordinary one-sided conditions. In the original detector, all 45 one-sided negative-clipping fixtures across 8–48 kHz were negative, including cases with approximately 0.6–0.7% truly clipped samples. The candidates existed but were rejected because local prominence compared the negative rail against the largest **absolute** local excursion, including the unaffected positive polarity. This is inconsistent with asymmetric clipping and with the detector's stated polarity-specific architecture.

A candidate correction now computes local prominence against same-polarity context. It restores moderate one-sided negative clipping across 8, 16, 22.05, 44.1, and 48 kHz and preserves exact polarity inversion. A 72-condition synthetic discriminant sweep—including 36 periodic sine, triangle, and square conditions—remained negative. This is a development result only; the complete cohort and all validation layers must be rerun. Detection at low known burden remains geometry- and carrier-dependent, and accepted plateau support is a conservative subset of all synthetically rail-limited samples rather than an unbiased estimate of every clipped sample.

## 2. What QDIST can scientifically claim

Hard clipping itself is an established nonlinear waveform phenomenon: amplitudes beyond a threshold are mapped to a fixed rail, producing plateau-like extrema and altered amplitude distributions. Consecutive extreme samples, time-domain plateau morphology, and amplitude-histogram evidence all have literature precedent. The most relevant detector-specific lineage is [Hansen, Stauffer, and Xia (2021)](https://doi.org/10.1016/j.specom.2021.07.007), [Xia and Hansen (2018)](https://www.isca-archive.org/interspeech_2018/xia18b_interspeech.pdf), [Laguna and Lerch (2016)](https://musicinformatics.gatech.edu/wp-content_nondefault/uploads/2016/09/Laguna_Lerch_2016_An-Efficient-Algorithm-For-Clipping-Detection-And-Declipping-Audio.pdf), and the coded-speech work of [Eaton and Naylor (2014)](https://spiral.imperial.ac.uk/bitstreams/58ceea9f-4e83-43c7-b32a-8d1607a3395d/download).

The exact QDIST detector is **not** an established published metric. Its conjunction of plateau length, flatness, local prominence, edge occupancy, terminality, quantization guards, repeated-level support, and a 20-ms episode merge is study-specific. Its maturity should therefore be stated as:

> Literature-informed hard-clipping detector using established waveform primitives and study-specific thresholds, event construction, and recording-level aggregation.

The valid construct is also narrower than the manuscript's nonlinear operator phi. QDIST measures **visible plateau morphology in the stored native decoded waveform that is compatible with hard clipping or saturation**. It does not measure all nonlinear distortion, soft clipping, limiting, dynamic-range compression, AGC, THD, intermodulation distortion, quantization distortion, perceptual severity, or the physical stage at which clipping occurred. Lossy coding can smear upstream plateaus; absence of QDIST evidence is not evidence that upstream clipping never occurred.

## 3. Feature-level decision recommendation

| Feature | Current role | Recommended role before rerun | Scientific reason |
|---|---|---|---|
| `qdist_hard_clipped_sample_fraction` | Secondary | **Candidate primary direct burden** | Direct channel-sample support from the accepted ledger; shift-invariant; translate to clipped channel-ms/min for interpretation. Still conditional on detector repair and validation. |
| `qdist_hard_clip_event_rate_per_min` | Primary event; default joint model | **Candidate secondary event view** | Reconstructable and exposure-aware, but depends on the 20-ms merge rule and on speech excursions through the clipping rail. With only 15 events in six recordings, count, exposure, and model-based uncertainty must accompany it. Do not include by default alongside another burden view. |
| `qdist_hard_clipped_frame_fraction` | Primary; default joint model | **REVISE or audit-only legacy view** | A tiny accepted plateau expands to an entire 30-ms frame; the result depends on grid origin and is not direct physical duration. It should not be called clipping “prevalence” or treated as time-shift invariant. If retained, call it a prespecified analysis-window contamination view and report frame-origin sensitivity. |
| `qdist_positive` / status | Export companion | **Retain as occurrence/status companion, not a fourth feature** | With 513/519 valid zeros, occurrence is the most stable cohort descriptor. It must remain separate from missing/unavailable and should not be presented as a calibrated reject threshold. |

An alternative to the legacy frame fraction is a newly defined, shift-invariant analysis-window impact measure based on dilating accepted intervals by the intended downstream window support. That would be a new estimand and must not be introduced without a new semantic measurement version and its own dose/interpretation validation.

## 4. Blocking findings

| ID | Gate | Finding | Why blocking | Required correction |
|---|---|---|---|---|
| B01 | G1/G5/G10 | Manuscript phi and Table 2 include clipping, saturation, nonlinear compression, and quantization, while QDIST measures hard-plateau morphology only. | Family label and observation-model claim exceed the estimator. | Narrow the operational wording everywhere or add separately validated nonlinear-distortion observables. Do not imply complete coverage. |
| B02 | G3 | Sub-frame common shifts change frame fraction by up to 11.8% in the reproduced fixture. | Current G3 PASS is false for a primary feature. | Test all/random frame origins; demote/revise the frame feature or change the estimand. |
| B03 | G4/G5 | Original detector missed all tested one-sided negative-clipping fixtures. | Intended hard-clipping construct includes asymmetric rails. | Use polarity-specific local prominence; validate positive-only, negative-only, unequal rails, and polarity inversion across rates/severities. |
| B04 | G4 | Existing dose study uses only three synthetic carriers, 48 kHz, symmetric clipping, five thresholds, and 16-bit PCM. | It does not establish the operating range or detection limits of a native-rate detector. | Add multiple rates, carriers, asymmetric/time-varying rails, burst durations, bit depths, SNRs, channel geometries, and real-speech injection. |
| B05 | G4/G6 | The repeated low-level saturation path is absent from all 30 accepted cohort plateaus and is not exercised by the v4 preflight dose grid. | A claimed detector pathway is unvalidated. | Add within-recording gain-state fixtures spanning the 0.25 candidate floor and characterize the fail-closed region below it. |
| B06 | G5 | Synthetic specificity controls omit real plosives/stop releases, ALS/dysarthric extrema, music, realistic limiter/compressor outputs, and matched neighboring-family perturbations. | Study-specific thresholds may confuse physiology/content with clipping. | Use blinded real negative controls and matched QADD/QGAIN/QREV/QCHAN/QTEMP perturbations at comparable severity. |
| B07 | G6 | Support tiers (3, 10, and 30 s) are declared but not calibrated; preflight tests only 1 s versus 4 s. | Exported support tiers lack evidence. | Plot recovery/false-positive behavior versus duration, event count, sample rate, and channel count; justify or remove tiers. |
| B08 | G6/G8 | Parameter robustness is based on a deterministic 46-recording subset and detector-margin summaries; the original thresholds evolved on the same cohort. | Generalization and threshold overfitting are unresolved. | Separate detector development from locked evaluation; use held-out or external recordings and report full rerun results. |
| B09 | G9 | Review is AI-assisted, not independent human/technical adjudication; 13/20 rejected candidates are ambiguous. | The mandatory event-detector gate is incomplete. | Use a randomized, stratum-blind review form with at least two independent reviewers and adjudication; retain disagreement and ambiguity. |
| B10 | G9 | The rejected-candidate gate uses 1/7 after excluding 13 ambiguous items; candidate review cannot detect events missed before candidate generation. | It cannot establish real-data sensitivity or a reliable false-negative rate. | Use injected ground truth for sensitivity and separate real-data PPV/error-mode review. Do not call rejected-candidate review sensitivity. |
| B11 | G9 | Accepted-event point estimate is 30/30, but its binomial 95% lower bound is 0.884 and events are clustered within six recordings. | A point-estimate >=0.90 gate is not equivalent to precision established with uncertainty. | Predeclare event- and recording-level criteria; report cluster-aware uncertainty and reviewer agreement. |
| B12 | Governance | Current final statuses include `PASS_WITH_QUALIFICATION`, `PASS_WITH_SCOPE_LIMIT`, and `PASS_WITH_MAJOR_QUALIFICATION`, outside the shared vocabulary. G3/G4/G6 are not PASS. | The shared minimum freeze rule requires G1–G4 and G6 PASS; G9 is mandatory. | Use only PASS/CONDITIONAL/FAIL/PENDING/N/A and enforce the common rule programmatically. |
| B13 | Governance | QDIST's 50-row checklist is not the shared master checklist and has no concrete evidence-path column. Finalization can copy any CSV with the expected filename without validating contents. | “50/50” is not auditable evidence. | Instantiate the exact master rows, require an existing artifact path and item-specific status, and validate schema/content before freeze. |
| B14 | Governance | `final_checklist_frame` blanket-assigns a gate status to every item and sets `scientific_review_complete=True`; finalization does not require a nonempty rationale. | Software can manufacture scientific completion without item review. | Remove blanket promotion; require signed decision records, nonempty rationale, artifact hashes, and human review metadata. |
| B15 | Governance | Finalization deletes an existing final root before copying the candidate and hard-codes cohort counts. | It is not a non-overwriting scientific freeze and cannot safely accommodate a corrected rerun. | Refuse overwrite; derive counts from verified tables; atomic-freeze only a newly versioned candidate. |
| B16 | G1/G10 | Registry citations are `Li et al. (2014); Patel et al. (2018); Goldsack et al. (2020)`, none of which establishes this clipping detector. | Scientific lineage is inadequate and may be misleading. | Cite detector-specific work; retain Goldsack only for V3 validation governance. |
| B17 | G10 | The manuscript reports 26 indicators and five QGAIN indicators, but the reviewed QGAIN package retains four. | The feature census is already inconsistent before QDIST is finalized. | Reconcile Table 3, Supplement S1, registry, and frozen exports after QDIST decisions. Current reviewed total is 25 before any QDIST demotion/revision. |

## 5. Additional implementation inconsistencies

- The cohort morphology-margin audit compares `candidate_to_context_ratio` with 0.50, although `local_magnitude_pass` uses 0.90. It also does not separately audit bilateral context thresholds. The accepted-margin PASS therefore does not reconstruct the actual local-prominence rule.
- The production detector names the low-level path `repeated_low_level_saturation`, while the reviewed cohort gate allows `low_level_repeated_edge`. The current cohort passes only because every accepted plateau uses `strong_recording_edge`.
- The frame feature's unit should be “fraction of complete 30-ms grid frames,” not merely “fraction.” Its grid origin belongs to the measurement identity.
- “Exact Poisson interval” should be called a model-based Poisson count-rate interval. Events are clustered within recordings and tied to nonhomogeneous speech amplitude; the interval is not assumption-free uncertainty.
- High negative repeat agreement is dominated by 513 shared zeros. Positive persistence is a different scientific question from technical repeatability and is not estimable here.
- The present v4 package validates numerical equivalence to v3.1.1, not independent scientific validity. Equivalence is a provenance result, not a construct-validity result.

## 6. Corrected validation design

### G1 — Construct, input, and lineage

- Name the family “visible native-decoded hard-clipping morphology,” with `QDIST` retained as the family code.
- State hard-clipping-only scope in manuscript Methods, Table 2, Table 3, Supplement S1e, figure captions, registry, and ML schema.
- Pin native first decoded stream, channels preserved, task-span time map, codec/container/bit depth, and no preprocessing.
- Classify each feature as established primitive plus study-specific detector/aggregation; include detector-specific references.

### G2 — Numerical correctness

- Hand-computable rail plateaus for mono and multichannel signals.
- Exact reconstruction of channel-sample burden, any-channel time burden, frame-grid view, episode count/rate, and all statuses.
- Explicit tests for overlapping channels, cross-channel episode merging, boundary intervals, partial final frames, NaN/unavailable, valid zero, and quantization lattice.

### G3 — Transformation behavior

- Exact polarity inversion for symmetric and one-sided clipping.
- Sub-frame origin sweep from 0 to frame length minus one; direct burden/event invariance and explicit frame-grid phase sensitivity.
- Uniform post-clip attenuation at multiple gains.
- Lossless PCM round-trip at supported bit depths.
- Source-rate grid and native-versus-resampled comparison.
- Opus/AAC/other cohort codec characterization as evidence-erasing transformations, never as invariances.

### G4 — Controlled construct response

- At least three carrier classes: synthetic speech-like, held-out real control speech, and held-out disease speech manually screened for injected-truth use.
- Source rates matching the cohort; mono and multichannel cases.
- Symmetric, positive-only, negative-only, unequal-rail, time-varying, and within-recording gain-state clipping.
- Dose by true clipped-sample fraction and rail dBFS, with detection-limit curves, precision, recall, F1, boundary error, event-count behavior, and uncertainty by speaker/recording.

### G5 — Discriminant validity

- Real and synthetic clean speech, ALS/dysarthric high-amplitude extrema, plosives, stop releases, breaths, clicks, impulses, tones, music, low-frequency waveforms, square/two-level signals, DC offsets, coarse quantization, smooth saturation, limiting, compression, AGC, lossy coding, additive noise, reverberation, channel filtering, and temporal glitches.
- Equal-severity target versus competing controls and explicit cross-family arbitration.

### G6 — Support and robustness

- Duration/exposure and sample-rate recovery curves.
- Full parameter neighborhood, including 0.25 recording floor, 0.45 strong path, 0.90 same-polarity prominence, plateau length/flatness, edge occupancy, terminality, and merge gap.
- Full locked evaluation set rather than only development positives plus 40 zeros.
- Absolute and relative changes, class changes, unavailable fraction, and detection-limit movement.

### G7/G8 — Cohort behavior and redundancy

- Rerun all 519 recordings under the corrected version; do not assume six positives or 30 plateaus.
- Stratify availability and occurrence by source rate, codec/container, bit depth, channel count, task, participant, and acquisition vintage.
- Report occurrence separately from conditional positive burden.
- Treat repeated recordings as persistence, not technical repeatability; do not interpret zero-dominated agreement as reliability.

### G9 — Event verification

- Randomized IDs and order; hide detector stratum, detector decision, source identity, diagnosis, clinical outcomes, human-QC labels, and prior AI labels.
- Two independent reviewers using waveform, sample-code/derivative, amplitude distribution, spectrogram, and audio.
- Review every accepted event plus stratified near-threshold rejections and high-risk valid-zero windows; retain `CANNOT_DETERMINE` separately.
- Report event- and recording-level agreement, PPV-like evidence, ambiguity, adjudication changes, and clustered uncertainty.
- State explicitly that real-data sensitivity is not estimable without known truth.

### G10 and freeze

- One exact master checklist with concrete artifact paths and permitted status vocabulary.
- One feature worksheet per output.
- A–J bundles with editable vector, 300-dpi PNG, source table, caption, provenance JSON, input/code/parameter hashes, and gallery index.
- No overwrite; no acceptance token can bypass a failed or pending blocking item.

## 7. Figure corrections

| Panel | Current problem | Required revision |
|---|---|---|
| A | Three 48-kHz symmetric synthetic carriers; no uncertainty/detection-limit surface. | Plot recovery versus known burden by rate, carrier, polarity geometry, and rail; show uncertainty and fail-closed region. |
| B | Single synthetic traces per control and incomplete confound set. | Use matched-severity target/competing mechanisms, real negative controls, false-positive rate with uncertainty, and cross-family arbitration. |
| C | Frame-aligned shift makes the invariance check tautological; bars do not show observed-minus-expected or tolerance bands. | Add sub-frame phase sweep, polarity-asymmetric inversion, gain, native/resampled/codec views, deviations from expected response, and tolerance bands. |
| D | Availability is shown, but support-tier calibration is absent. | Add recovery/availability versus independent exposure, sample rate, and channel geometry; separate zero from unavailable. |
| E | Margin threshold for local prominence is mis-specified; low-level path unrepresented. | Reconstruct every actual acceptance predicate and show full-detector class changes for prespecified variants. |
| F | Extreme zero inflation is present. | Use occurrence plus conditional positive distributions; report participant-aware prevalence and exact counts. |
| G | Eight rendered examples are not enough for mandatory adjudication. | Keep publication examples, but link the complete randomized review package and failure-mode exemplars. |
| H | Zero-dominated agreement can look reassuring. | Separate occurrence persistence, conditional positive burden, technical reconstruction, and related-view redundancy. |
| I | AI labels and point-estimate gates are insufficient. | Replace with independent reviewer agreement/adjudication and cluster-aware uncertainty; retain ambiguity. |
| J | Current default joint inclusion authorizes frame plus event rate. | Export status, occurrence, direct burden, count, exposure, uncertainty, native geometry, and version; no default simultaneous use of related views. |

## 8. Manuscript corrections required now

1. Replace claims that QDIST measures the full phi operator with wording that it operationalizes one observable subset: visible hard-clipping morphology.
2. Remove or qualify causal language such as “nonlinear distortion from overload or coding.” The detector cannot identify the clipping stage, and codec processing can erase the evidence.
3. Do not describe the frame fraction as direct clipping prevalence without naming the 30-ms grid and its phase dependence.
4. Do not call the current acceptance criteria “preregistered” unless an external time-stamped preregistration exists. “Predeclared in the validation protocol before the locked evaluation” is supportable only after a genuinely locked evaluation.
5. Reconcile the feature census: reviewed QGAIN has four retained indicators, so the draft's five-QGAIN/26-total count is stale.
6. Keep the manuscript's current “QDIST pending” status until qdist-v4.1.0 completes G1–G10 and the final registry is reconciled.

## 9. Evidence produced in this audit

- `results/frame_grid_translation.csv`: all 1,440 sub-frame origins for the reproduced 48-kHz fixture.
- `results/frame_grid_translation_summary.json`: baseline and shifted feature values plus phase range.
- `results/sample_rate_asymmetry_recovery.csv`: 135 original-detector conditions across five rates, three seeds, three clipping geometries, and three rails.
- `results/periodic_false_positive_sweep.csv`: original-detector sine/square control sweep.
- `results/event_review_uncertainty.csv`: binomial interval audit of reported adjudicable counts.
- `candidate/qdist_v410_candidate.py`: non-overwriting candidate detector with same-polarity local prominence and corrected registry roles/lineage.
- Pipeline candidate module: `src/paper1_qc/qdist_v410_candidate.py`.
- Candidate tests: `tests reviewed/test_qdist_v410_candidate.py`.
- Candidate remediation module: `src reviewed/paper1_qc_reviewed/qdist_v410_remediation.py`.
- Candidate source notebook: `notebooks reviewed/05_QDIST/05_nonlinear_distortion_QDIST_v4_1_0_REMEDIATION_PREFLIGHT_SOURCE.ipynb`.
- Exact shared-master remediation checklist: `notebooks reviewed/05_QDIST/QDIST_Master_Validation_Checklist_v1_1_REMEDIATION.csv`.
- `remediation_preflight_outputs/tables/`: 135-condition dose/recovery grid, 72-condition discriminant grid, low-level-state, transformation, frame-phase, and figure-index tables.
- `remediation_preflight_outputs/validation/`: reconstruction and blocking synthetic-check records.
- `remediation_preflight_outputs/figures/`: source-linked preflight Panels A–C in PNG, SVG, and PDF with captions and provenance JSON.
- `remediation_preflight_outputs/manifests/qdist_v410_remediation_preflight_manifest.json`: candidate identity, hashes, feature roles, and explicit freeze blockers.

## 10. Current limitations of this audit

The uploaded pipeline contains code, notebooks, tests, workbooks, and embedded execution summaries, but not the external candidate/final output directories. The 23 figure bundles, their plotted source tables, the 60 review packages/WAV files, and the 519-recording feature/ledger tables therefore could not yet be independently inspected or rerun. Claims derived only from embedded outputs remain provisionally accepted as reported, not independently verified.
