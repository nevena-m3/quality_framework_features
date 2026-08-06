# Release notes v0.10.1

## QADD v4.1 measurement-development repair

- preserves the five QADD feature formulas and the fixed 200-ms guard;
- replaces misleading `limited`/`robust` support labels with
  `minimum`/`high`;
- adds whole-pause deletion diagnostics for all five raw estimands;
- summarizes clustered sensitivity once per recording and feature;
- replaces the redundant 100/200/300-ms guard comparison with a genuine
  additional 100-ms erosion of fixed reference pauses;
- reports numerical sensitivity and availability without post-hoc acceptance
  thresholds;
- repairs unavailable-descriptor gallery panels;
- separates downstream integration from the measurement-freeze gate;
- creates an immutable, checksummed freeze snapshot and refuses overwrite;
- adds two regression tests for clustered diagnostics and support terminology.
