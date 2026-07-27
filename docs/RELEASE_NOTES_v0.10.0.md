# Release notes — v0.10.0

## Additive-interference measurement repair

- Replaced the family-wide all-or-none support gate with metric-specific eligibility.
- Added one explicit status per additive metric plus speech, nonspeech, frame,
  interval, flatness-spectrum, and hum-spectrum support fields.
- Defined `qadd_status` as `ok`, `partial_support`, or `insufficient_support` from
  the five primary metric statuses.
- Counted transient runs within each guarded pause so separate pauses cannot merge
  into one event.
- Required one second of cumulative analyzable pause support for flatness and at
  least one continuous one-second pause for narrowband 50/60-Hz hum estimation.
- Replaced the fixed PSD epsilon in relative spectral ratios with a machine-level
  numerical floor, restoring global-gain invariance.

## Transparent additive notebook

- Expanded 02a to 20 auditable cells covering formulas, units, directions,
  mathematical ranges, support rules, status fields, confounding, and expected
  control responses.
- Added an exact frozen-recording extraction contract, duplicate/missing/unexpected
  ledgers, metric-specific support summaries, status/value consistency checks, hard
  range checks, and minimum analysis support checks.
- Added deterministic broadband-noise, 60-Hz hum, separated-transient, broadband
  versus tonal flatness, and global-gain controls with saved pass/fail criteria.
- Added empirical distribution, support-dependence, codec/sample-rate/channel, and
  representative low/middle/high/robust-extreme audits.
- Added optional in-notebook playback for empirical examples and explicit guidance
  that genuine high-artifact recordings are retained.
- Preserved the efficient extraction contract: 02a launches extraction once, while
  02b–02f audit the other families from the same recording-level table.

## Verification

- Added tests for metric-specific support, spectral-support/status consistency,
  segment-aware transient counting, and global-gain invariance.
- Added the complete additive measurement specification to the scientific protocol.
