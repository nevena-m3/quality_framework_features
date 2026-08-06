# Repository maintenance and retention policy

## Purpose

This policy keeps the working repository understandable without changing scientific or computational behavior. Scientific versions are retained when they remain part of provenance; incidental working copies are not.

## Canonical material

Keep and version-control:

- Python source under `src/` and `src reviewed/`;
- source/finalization notebooks and intentionally retained execution evidence;
- tests under `tests/` and `tests reviewed/`;
- configuration examples, protocols, specifications, manifests, and scientific decisions;
- scripts required to reproduce, validate, or freeze an implementation;
- the small `MAIN outputs/README.md` directory contract.

Keep locally but do not version-control:

- participant data and media;
- local configuration and adjudication files;
- generated `outputs/`, `outputs reviewed/`, `MAIN outputs/`, and `MAIN outputs reviewed/` trees;
- local evidence bundles containing derived data;
- virtual environments.

Move out of the active repository or delete when independently backed up:

- ZIP/tar snapshots of the repository;
- `*.bak`, timestamped `pre_*`, and ad-hoc backup trees;
- `.ipynb_checkpoints`, Python bytecode, pytest caches, and package metadata;
- explicitly superseded notebooks stored under an `old/` directory;
- broken virtual environments.

## Version selection rule

“Latest” is determined per feature family from release/finalization evidence, not solely from filesystem modification time. A newer reviewed implementation does not automatically replace its frozen predecessor: both may be required to reproduce a migration comparison or immutable release. Within the same implementation line, retain the canonical source and finalization artifacts and remove timestamped working copies.

## Cleanup record: 2026-08-06

The active repository was inventoried against Git history and the available chronological snapshot/patch archives. The cleanup:

- retained both the shared core and reviewed code layers because reviewed notebooks import both;
- retained the latest reviewed family lines and their tests/documentation;
- retained generated output trees locally and excluded them from Git;
- moved repository ZIPs, the broken virtual environment, backup directories, timestamped `.bak` files, Jupyter checkpoints, and `notebooks/02_feature_extraction/old/` to a dated sibling recovery directory;
- added a single entry-point README and explicit Git hygiene rules;
- made no intentional changes to measurement or analysis logic.

The recovery directory is:

```text
../paper_1_cleanup_recovery_20260806/
```

After the cleaned repository and its Git history have been independently backed up, that directory can be archived to external storage or removed.

## Future maintenance

Create changes on a topic branch, run both test suites, and record scientific behavior changes in the appropriate release/decision document. Never reuse a frozen version identifier for changed behavior. Avoid putting manual copies beside active source; Git already provides rollback and comparison.
