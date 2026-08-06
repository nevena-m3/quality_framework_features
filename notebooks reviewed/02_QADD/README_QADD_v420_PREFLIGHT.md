# QADD v4.2.0 reviewed analytical preflight

This patch installs a parallel reviewed QADD implementation. It does not overwrite the legacy notebook or freeze.

Run order:

1. Install with `install_qadd_reviewed_preflight_v420.ps1`.
2. Confirm `18 passed`.
3. Run all cells in `02a_additive_interference_QADD_v4_2_0_REVIEWED_LOCAL_PREFLIGHT.ipynb`.
4. Keep publication and freeze disabled.
5. Save the executed notebook and package the candidate outputs for review.

Expected final notebook status:

- `preflight_blocking_checks_pass: true`
- `cohort_extraction_completed: false`
- `publish_and_freeze: false`
- G7, G8, and G10 pending
