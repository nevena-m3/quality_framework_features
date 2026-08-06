# QREV v4.0.0 Reviewed Scientific Contract

## Identity

- Family: QREV
- Display name: Reverberation / observable residual tail
- Candidate version: `qrev-v4.0.0-candidate`
- Measurement regime: three conditional natural-boundary estimators plus one broadly available pinned comparator
- Waveform: mono, globally DC-removed, deterministically resampled 16-kHz analysis waveform
- Prohibited preprocessing: amplitude normalization, denoising, dereverberation, interpolation, or uncontrolled filtering before QREV

## Family construct

QREV measures observable temporal smearing and post-offset energy patterns compatible with reflections or echo. It does not identify a room, recover a room impulse response, or estimate standard room-acoustic parameters.

Permitted wording: residual-tail magnitude, bounded above-floor persistence, conditional downward envelope decay, and normalized-fast SRMR.

Prohibited wording: RT60, EDT, C50/C80, D50, DRR, STI, physical reverberation time, recovered RIR, or confirmed echo source.

## Frozen interval contract

1. Select exactly `primary_speech / primary` for natural speech offsets and internal pause endpoints.
2. Select exactly `strict_speech / primary` for SRMR speech-support duration only.
3. Require frozen segmentation eligibility and retained decision status.
4. Preserve logical recording ID, frozen interval index, deterministic interval identity, view, and profile.
5. Reject duplicate identities and overlapping intervals.
6. Never pool profiles or silently fall back to another speech view.
7. Never use the already-eroded strict interval end as the post-offset boundary.

## Feature 1 — Early post-offset tail excess

Feature ID: `qrev_tail_excess_100ms_db`

Operational definition: for each eligible primary-speech offset, calculate framewise AC RMS dBFS in the first 0–100 ms and subtract the median AC RMS dBFS in the independent 700–1000-ms late-pause floor. Aggregate the signed boundary values by the median.

Unit: dB.

Orientation: higher indicates stronger early residual energy above the local late-pause baseline.

Eligibility: pause extends through 1.0 s; at least five non-floor early frames; at least 20 non-floor floor frames; digital-floor fraction <=0.10; floor IQR <=12 dB. Recording support policy is provisionally >=2 valid boundaries and will be compared with >=3 and >=4 in the corrected cohort.

Claim boundary: blind residual-tail proxy only; not RT60, DRR, or RIR estimation. Breath, early noise changes, echo, speech leakage, and segmentation error may increase it.

## Feature 2 — Bounded tail persistence

Feature ID: `qrev_tail_persistence_median_sec`

Operational definition: using the independent 700–1000-ms floor, find the first 30-ms frame midpoint after the primary speech offset at which the envelope remains within floor +3 dB for three consecutive frames. Observe only through 0.6 s. If return to floor is not observed by 0.6 s, store 0.6 s and mark the boundary right-censored. Aggregate boundary observations by the median and retain censored fraction and recording-median censoring.

Unit: s.

Orientation: higher indicates longer observable above-floor persistence within the 0.6-s horizon.

Eligibility: same stable-floor boundary contract as tail excess and sufficient frame support through the 0.6-s horizon. Recording support policy provisionally >=2 valid boundaries, compared with >=3 and >=4 before G10.

Claim boundary: not reverberation time. A horizon value is a lower bound, not an exact duration.

Scientific amendment: the legacy 1.0-s horizon overlapped the 700–1000-ms floor. The reviewed 0.6-s horizon restores independence between the observation horizon and floor.

## Feature 3 — Conditional downward tail-decay rate

Feature ID: `qrev_downward_decay_rate_db_per_sec`

Operational definition: fit a Theil–Sen slope to valid non-floor frame levels in 0–300 ms after the primary speech offset. A boundary is eligible only if the signed slope is negative and the 90th–10th percentile dynamic range is >=3 dB. Store the signed slope and report the positive magnitude `-slope`. Aggregate by the median.

Unit: dB/s.

Orientation: among valid downward traces, lower positive magnitude means slower observed decay.

Eligibility: at least 20 valid frames per boundary; provisionally >=2 valid downward-decay boundaries per recording, compared with >=3 and >=4 before G10.

Claim boundary: not a Schroeder decay and not RT. Nondecaying, rising, nonsmooth, or low-range boundaries are unavailable, not zero.

## Feature 4 — Normalized-fast SRMR

Feature ID: `qrev_srmr_norm`

Pinned identity:

- SRMRpy normalized-fast implementation
- `norm=True`
- `fast=True`
- `max_cf=30 Hz`
- 23 cochlear filters
- low acoustic center frequency 125 Hz
- minimum modulation center frequency 4 Hz
- upstream commit `fee009779cef96bed34db3a7e31d10f3ad1ea133`
- Gammatone `1.0.3`
- primary task span from first primary onset through last primary offset, internal pauses preserved
- strict-speech union duration retained as support

Unit: ratio.

Orientation: for this pinned variant, lower values are generally compatible with greater reverberation-related modulation smearing.

Eligibility: >=3 s primary task span, >=3 s strict speech support, finite mono 16-kHz waveform, working-set estimate <=512 MB, exact pinned dependencies.

Claim boundary: published no-reference reverberation-sensitive comparator; not reverberation-specific, not direct RT60, and sensitive to additive noise, codec, bandwidth, speech content, pitch, and enhancement artifacts.

## Support, status, and missingness

Every feature travels with:

- feature value;
- raw estimate where a recording-level support policy suppresses the analysis value;
- availability boolean;
- measurement status and missing reason;
- valid boundary count or speech duration;
- valid pause-support duration;
- precision tier;
- censoring fields where applicable;
- measurement version and input-view identity;
- SRMR implementation identity.

Unavailable, censored, and zero are distinct. No unavailable value is coerced to zero.

## Provisional support policy

The Master Design proposed >=2 valid boundaries. The legacy implementation used >=4. The reviewed extraction retains raw estimates at every support count and compares two-, three-, and four-boundary policies using availability, delete-one-boundary sensitivity, bootstrap/precision behavior, repeated-recording persistence, and redundancy. G10 will select the final policy without optimizing against clinical group effects or human labels.

## Required validation

G1–G10 and Panels A–J follow the common family template. G9 is N/A because no discrete event detector is retained. A delayed-echo detector is explicitly outside the current scope.

## ML handoff

QREV will export each feature with availability, support, censoring, precision, version, and status. The future quality-aware pipeline may use these fields to predict biomarker-specific reliability or abstention. QREV does not provide a generic good/bad label, family scalar, or standalone reject threshold.

## Freeze rule

Freeze is prohibited until:

1. all G1–G6 preflight checks pass in the pinned local SRMR environment;
2. corrected 519-recording extraction completes with exact media and interval provenance;
3. support policy, censoring, boundary sensitivity, empirical plausibility, persistence, and redundancy are reviewed;
4. all features receive explicit G10 decisions;
5. Panels A–H/J and their source/caption/provenance bundles are complete;
6. executed notebook, tests, registry, passports, ML export, and immutable manifests are sealed.
