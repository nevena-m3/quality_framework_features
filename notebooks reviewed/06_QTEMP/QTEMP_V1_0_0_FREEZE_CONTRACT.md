# QTEMP v1.0.0 immutable-freeze contract

## Measurement identity

- Family: QTEMP — decoded-waveform temporal-discontinuity observables.
- Final measurement version: `qtemp-v1.0.0-analytical-final-no-retained`.
- Source candidate: `qtemp-v1.0.0-candidate-g9-pending`.
- Final state: analytical implementation freeze with no retained primary validated features.

## Frozen dispositions

1. `qtemp_dropout_duration_fraction` — exploratory descriptive burden only.
2. `qtemp_dropout_event_rate_per_min` — exploratory same-ledger frequency only.
3. `qtemp_frozen_audio_duration_fraction` — monitoring-only, zero variation.
4. `qtemp_frozen_audio_event_rate_per_min` — monitoring same-ledger frequency, zero variation.
5. `qtemp_splice_discontinuity_rate_per_min` — dropped; do not export or analyze.

The validated primary QTEMP feature registry is empty. No family scalar is defined.

## Frozen input and preprocessing contract

QTEMP operates on the pinned native decoded waveform view before transformations that could create, erase, or alter the target temporal patterns. Signal-view identity, channel handling, sample rate, preprocessing order, eligible exposure, detector parameters, media provenance, and artifact hashes are part of measurement identity.

## Frozen scientific claim boundary

Permitted claim: observable decoded-waveform temporal-discontinuity patterns meeting the registered algorithmic rules.

Prohibited claims: packet loss, network failure, buffering, concealment, missing speech content, physical device failure, physiological biomarker, complete temporal-quality measure, or validated real-world event detector.

## Support, missingness, and dependence

Valid zero is not missing. Unavailable or insufficient-support states remain explicit and are never imputed to zero. Event count, eligible exposure, status, and detector version accompany event-rate use. Duration and rate summaries from the same event ledger are dependent views and must not be treated as independent evidence.

## G9 disposition

G9 is `N/A_NO_RETAINED_PRIMARY_EVENT_FEATURES`. It is not passed, failed, blocked, or pending. The family closes through the negative primary-feature disposition.

## Downstream contract

- Primary Paper 1 feature census: zero QTEMP features.
- Primary statistical or machine-learning models: exclude QTEMP outputs.
- Exploratory or monitoring sensitivity work: allowed only with frozen role, provenance, support, and claim limitations.
- Standalone exclusion thresholds, family composites, and causal source labels: prohibited.

## Immutability

The final archive and standardized family wrapper are immutable. Any change to a value, detector, threshold, signal view, exposure definition, merge rule, missingness meaning, feature role, or claim boundary requires a new semantic measurement version. Main validated feature tables must remain unchanged.
