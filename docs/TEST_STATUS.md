# Verification status

Last verified: 2026-08-06

Environment: repository `.venv`, with `src/` and `src reviewed/` on `PYTHONPATH`.

## Results

- Reviewed suite: **327 passed**.
- Combined current core and reviewed suites: **540 passed, 4 failed**.
- Source/runtime tests passed; the four failures are notebook-governance assertions.

## Known notebook-governance failures

1. `tests/test_qchan_notebook_v300.py::test_generator_reproduces_committed_notebook`
2. `tests/test_qchan_notebook_v300.py::test_notebook_is_clean_and_unexecuted`
3. `tests/test_qchan_notebook_v300.py::test_freeze_is_non_overwriting_and_review_gated`
4. `tests/test_qdist_notebook_v311.py::test_generator_reproduces_committed_notebook_exactly`

The committed QCHAN notebook contains execution metadata and differs from its generator/governance expectations. The committed QDIST v3.1.1 notebook differs from its generator after outputs and execution counts are normalized. These mismatches predate the repository cleanup and were preserved to avoid changing scientific code or evidence.

## Commands

```powershell
$env:PYTHONPATH = "$PWD\src;$PWD\src reviewed"
python -m pytest tests "tests reviewed"
python -m pytest "tests reviewed"
```

Before resolving a notebook mismatch, explicitly designate whether the generator, source notebook, or executed evidence is canonical. Record any resulting scientific-content change under a new version.
