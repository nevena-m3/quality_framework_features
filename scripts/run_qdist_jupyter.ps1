param(
    [string]$Repo = "C:\Users\musikicn\Desktop\Nevena_project\Paper_1\paper_1"
)

$ErrorActionPreference = "Stop"
Set-Location $Repo
$env:PYTHONPATH = "$Repo\src"
$python = "$Repo\.venv\Scripts\python.exe"
$target = "/lab/tree/notebooks/02_feature_extraction/02e_nonlinear_distortion_QDIST_v3_1_1.ipynb"

& $python -m jupyter lab `
    --ServerApp.root_dir="$Repo" `
    --ServerApp.default_url="$target"
