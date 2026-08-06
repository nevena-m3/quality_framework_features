# QTEMP v0.2.0 Scientific Measurement-Development Decision

## Decision

QTEMP v0.2.0 is a **candidate measurement-development release**, not a publication freeze. It replaces the v0.1 implementation scaffold with a native-waveform, event-ledger-first system designed to determine—rather than assume—which temporal-discontinuity indicators are scientifically defensible.

The family measures **observable continuity violations in decoded audio consistent with omitted, held, repeated, or abruptly joined waveform support**. It does not identify packet loss, browser failure, buffering failure, transport failure, or a unique upstream mechanism.

## Candidate analysis features

1. `qtemp_dropout_duration_fraction`
2. `qtemp_dropout_event_rate_per_min`
3. `qtemp_frozen_audio_duration_fraction`
4. `qtemp_frozen_audio_event_rate_per_min`
5. `qtemp_splice_discontinuity_rate_per_min`

Duration fraction and event rate are complementary summaries of the same detector ledger. They are not independent evidence. No QTEMP scalar score or family composite is created.

## Governing signal contract

QTEMP operates on the **native decoded channels before resampling, mono conversion, normalization, interpolation, filtering, denoising, or codec re-encoding**. Frozen task and speech intervals are mapped into native sample time. Channels are inspected independently; mono averaging is prohibited in the production detector path.

The primary denominator is finite eligible native-stream duration within the frozen task interval after symmetric edge guards. Strict-speech duration is not used as the main denominator because a temporal artifact can alter VAD output and therefore its own denominator.

## Event and status governance

The implementation creates five auditable levels:

- native-source/exposure ledger;
- all generated candidate ledger;
- candidate disposition ledger (`accepted`, `indeterminate`, `rejected`);
- accepted recording-level event ledger after cross-channel merging and arbitration;
- recording-level features reconstructed exactly from the accepted event and exposure ledgers.

A zero is reported only after a complete eligible native-stream inspection. Decode, provenance, support, or adapter failures remain unavailable and do not become zero.

## Detector scope

### Bracketed dropout-like runs

Detects exact-zero or constant-low-information runs with active bilateral context. Edge silence, attenuated active speech, natural closure-like intervals, and overlong silence are explicit controls. The output is not packet-loss percentage.

### Frozen/duplicated decoded support

Detects consecutive near-exact repeated decoded waveform sequences over an explicit lag grid. Periodic voicing, vowel-like harmonic signals, tones, repeated linguistic material, low entropy, and weak boundary novelty are blocking competing explanations. The claim is intentionally narrower than general packet-loss concealment or all possible freezes.

### Abrupt splice-like joins

Detects localized bilateral prediction failures after QDIST clipping-edge, QADD impulse-like, QGAIN persistent-level-step, frozen speech-boundary, and accepted-QTEMP event guards. Plain deletion duration is characterized but is not assumed to be a monotonic severity dose because observability depends on the compatibility of the newly joined waveform contexts.

## Analytical-validation program

The notebook contains:

- deterministic formula, interval, channel, status, and reconstruction tests;
- multi-seed synthetic dose grids for all three detectors;
- sample-rate, source-level, event-count, event-merging, and localization characterization;
- periodic, low-level, closure-like, gain-step, click, clipping, and speech-boundary controls;
- participant-disjoint development and held-out validation injections on real Bamboo speech;
- native-versus-mono/normalized/DC-removed/resampled/Opus/AAC observability characterization;
- full frozen-cohort extraction with candidate, disposition, event, exposure, error, and recording tables;
- one-at-a-time parameter sensitivity on rare-positive/candidate and event-free recordings;
- exact cohort reconstruction;
- prevalence, exposure, confidence intervals, technical concentration, recurrence, redundancy, and downstream-suitability audits;
- stratified accepted/indeterminate/rejected/event-free gallery with a detector-label-blind adjudication sheet;
- computed G1–G10 gates and feature-specific decisions.

## Feature decision policy

Each feature independently receives one of the following outcomes:

- `CANDIDATE_RETAIN_PENDING_G9`
- `CANDIDATE_SECONDARY_PENDING_G9`
- `CANDIDATE_PENDING_COHORT`
- `CANDIDATE_PENDING_SENSITIVITY`
- `CANDIDATE_PENDING_GALLERY`
- `REVISE`
- `ANALYTICALLY_ACCEPTED_FOR_NEXT_FREEZE_VERSION`
- `ANALYTICALLY_ACCEPTED_SECONDARY_FOR_NEXT_FREEZE_VERSION`

No v0.2 feature is marked publication-ready before blinded adjudication. Even a successful v0.2 review authorizes a new immutable freeze candidate; v0.2 itself remains a development record.

## Rare-event interpretation

Sparse or absent events are scientifically valid outcomes. QTEMP features may be unsuitable for ICC, robust z-scoring, PCA, or continuous correlations while remaining valid binary/count rare-event measurements. Thresholds must not be weakened to manufacture variance or disease associations.

## Residual limitations

- Natural-event sensitivity cannot be identified without exhaustive waveform-level ground truth or transport metadata.
- Codec processing may smear or erase pre-encoding corruption.
- Near-exact duplicate detection intentionally under-covers nonliteral concealment and broader freeze mechanisms.
- Abrupt joins that happen to be waveform-continuous may be unobservable from the decoded waveform.
- Blinded adjudication is required to estimate positive predictive credibility on real cohort candidates.
