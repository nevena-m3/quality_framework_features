# Release notes — v0.8.0

## Segmentation hierarchy

- Moved every Silero-stage artifact under `outputs/01_segmentation`.
- Kept separate `figures/` and `segmentation/` branches.
- Preserved one original-style frame CSV, segment CSV, and four-panel PNG per
  recording, grouped as accepted, flagged, or excluded.
- Moved manual-review artifacts into the same stage hierarchy.

## Boundary correctness and auditability

- Removed the 50-ms analysis padding that systematically expanded Silero speech
  regions.
- Removed the redundant second 100-ms gap bridge and second 250-ms short-region
  filter. Silero already applies its configured duration rules.
- Frozen automatic boundaries now come from unpadded Silero sample indices, not the
  30-ms visualization bins.
- Added one boundary-audit CSV and PNG per recording. They quantify 30-ms display
  deltas and local onset/offset RMS contrast.
- Low local contrast is review evidence only; it never automatically moves an edge,
  which protects weak, breathy, or gradually decaying ALS speech from energy-based
  truncation.
- Accepted recordings with extreme boundary summaries or at least 50% low-contrast
  edges enter the mandatory review queue.

## Sensitivity and performance

- Conservative and permissive profiles now rerun Silero with distinct
  threshold/minimum-duration settings rather than reusing the primary timestamps.
- The Silero model is loaded once per command and reused safely across recordings and
  profiles.
- Added tests for exact-versus-displayed edge separation and low-contrast review
  flags.

The defaults are defensible prespecified operating points, not a claim of perfect
boundary accuracy. A manuscript accuracy claim still requires an independent,
diagnosis-blind manual boundary reference subset and group-specific error reporting.
