# Paper 1 Remote Speech Quality-Control Pipeline

This repository contains the auditable measurement and validation pipeline used for Paper 1. It evaluates acquisition-related variation in remote speech recordings and maintains frozen, traceable feature-family outputs.

The repository has two intentional code layers:

- `src/paper1_qc/` is the shared core and the last frozen implementations.
- `src reviewed/paper1_qc_reviewed/` contains the newer reviewed feature-family implementations and imports shared utilities from the core layer.

Do not remove the core layer merely because a reviewed implementation exists. Reviewed notebooks rely on both source roots.

## Current feature-family implementations

| Family | Core/frozen line | Latest reviewed line |
| --- | --- | --- |
| Gain dynamics | `qgain.py` (v3.1) | `qgain_v410.py` |
| Additive interference | `qadd.py` (v4.1) | `qadd_v420*.py` |
| Reverberation | `qrev.py` (v3.1.1) | `qrev_v400*.py` |
| Channel/device | `qchan.py` (v3.0.1) | `qchan_v400*.py` |
| Nonlinear distortion | `qdist.py` (v3.1.1) | `qdist_v410*.py` |
| Temporal discontinuity | `qtemp.py` (v0.3.1) | `qtemp_v100_candidate.py` plus the final analytical-disposition notebook |

Version suffixes in this table are part of the scientific provenance. They are not interchangeable APIs.

## Repository map

```text
config/                 Local configuration templates and adjudication inputs
docs/                   Core protocols, specifications, release notes, and run guides
docs reviewed/          Reviewed-family scientific decisions and validation records
notebooks/              Core pipeline, frozen feature notebooks, assembly, and analyses
notebooks reviewed/     Latest reviewed source/finalization notebooks by feature family
scripts/                Core notebook generators and command-line helpers
scripts reviewed/       Reviewed-family verification and orchestration helpers
src/paper1_qc/          Installable core Python package
src reviewed/           Reviewed Python package used by reviewed notebooks
tests/                  Core and frozen-line tests
tests reviewed/         Reviewed-family tests
visualization/          Result-audit and manuscript visualization notebooks
outputs*/               Generated local work products (not committed)
MAIN outputs*/          Immutable/local freeze products (not committed, except README)
```

See `docs/REPOSITORY_MAINTENANCE.md` for the retention policy and cleanup record.

## Setup

Python 3.11 is the supported baseline.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev,reverb]"
```

Copy the example configuration files before running cohort work. Local configuration and participant-derived data are intentionally ignored by Git.

## Run and test

Run the core suite:

```powershell
python -m pytest tests
```

Run reviewed-family tests with both source roots available:

```powershell
$env:PYTHONPATH = "$PWD\src;$PWD\src reviewed"
python -m pytest "tests reviewed"
```

For notebook execution, begin with the relevant `*_SOURCE.ipynb` or `*_FINALIZATION_SOURCE.ipynb`. Files labeled `LOCAL`, `EXECUTED`, or `REFERENCE` are evidence/results, not the canonical editable source.

QTEMP has a Windows runner at `RUN_QTEMP_FINALIZE.cmd`. Read `QTEMP_v100_FINAL_READ_ME.md` before using it.

## Scientific status

The reviewed layer is not uniformly publication-frozen. Each feature family records its own validation gates and disposition. In particular, QTEMP v1.0.0 concludes with no retained primary features; its dropout measures are exploratory and frozen-audio measures are monitoring-only. Consult the corresponding decision and finalization documents before downstream use.

## Verification baseline

The reviewed suite currently passes completely. The core suite has four known notebook-governance failures involving committed execution metadata/source-generator parity; all remaining tests pass. See `docs/TEST_STATUS.md` for exact results. Do not “fix” those failures by clearing or regenerating scientific evidence without first confirming the intended canonical notebook.

## Data and Git policy

Never commit participant media, local configuration, generated outputs, virtual environments, notebook checkpoints, or recovery archives. The `.gitignore` encodes these rules. Large local outputs remain in their established paths because notebooks and manifests use those paths.

Before committing:

```powershell
git status --short
python -m pytest tests
$env:PYTHONPATH = "$PWD\src;$PWD\src reviewed"
python -m pytest "tests reviewed"
```

A dated recovery folder may exist beside this checkout. It is deliberately outside Git and contains superseded snapshots and backups retained during repository cleanup.
