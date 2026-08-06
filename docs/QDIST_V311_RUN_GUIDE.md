# QDIST v3.1.1 Windows run guide

Stop old Jupyter servers and install the patch into the repository root. Delete only `outputs/02_features/nonlinear_distortion/qdist-v3.1.1` if restarting this version; never reuse v3.1.0 checkpoints.

Run:

```powershell
$repo = "C:\Users\musikicn\Desktop\Nevena_project\Paper_1\paper_1"
Set-Location $repo
& "$repo\.venv\Scripts\python.exe" scripts\generate_qdist_v3_notebook.py
& "$repo\.venv\Scripts\python.exe" -m pytest tests\test_qdist_v30.py tests\test_qdist_notebook_v300.py -q
& powershell.exe -ExecutionPolicy Bypass -File "$repo\scripts\run_qdist_jupyter.ps1" -Repo "$repo"
```

Keep `PUBLISH_AND_FREEZE_QDIST_V311 = False`, `QDIST_REVIEW_DECISION = "PENDING"`, and all feature decisions pending for the first run. Use Restart Kernel and Clear All Outputs, then Run All Cells. Review the randomized gallery and complete the reviewer form before enabling freeze.
