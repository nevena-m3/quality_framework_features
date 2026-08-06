# QCHAN v3.0.1 — install and run

From PowerShell, extract the patch into the repository root, run the two QCHAN test files, and open the new notebook in JupyterLab.

```powershell
$repo = "C:\Users\musikicn\Desktop\Nevena_project\Paper_1\paper_1"
$downloads = Join-Path $env:USERPROFILE "Downloads"
$tmp = Join-Path $env:TEMP "QCHAN_v301"

Remove-Item $tmp -Recurse -Force -ErrorAction SilentlyContinue
Expand-Archive "$downloads\QCHAN_v3_0_1_patch.zip" $tmp -Force
Copy-Item "$tmp\QCHAN_v3_0_1_patch\*" $repo -Recurse -Force

Set-Location $repo
& "$repo\.venv\Scripts\python.exe" -m pytest tests\test_qchan_v30.py tests\test_qchan_notebook_v300.py -q
& "$repo\.venv\Scripts\python.exe" -m jupyter lab "notebooks\02_feature_extraction\02d_channel_device_QCHAN_v3_0_1.ipynb"
```

After the test command reports `40 passed`, set `PACKAGE_TESTS_CONFIRMED = True` in the first code cell. Keep `PUBLISH_AND_FREEZE_QCHAN_V301 = False` for the first complete run. Use **Restart Kernel and Run All Cells**.
