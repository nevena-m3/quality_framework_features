param(
    [Parameter(Mandatory=$true)]
    [string]$ProjectRoot
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$MeasurementFreeze = Join-Path $ProjectRoot "MAIN outputs\reviewed\06_family_freezes\gain_dynamics\qgain-v4.1.0"
$Candidate = Join-Path $ProjectRoot "outputs\reviewed\gain_dynamics\qgain-v4.1.0-figures-v1.0.0-candidate"
$FinalParent = Join-Path $ProjectRoot "MAIN outputs\reviewed\07_figure_packages\gain_dynamics"
$Final = Join-Path $FinalParent "qgain-v4.1.0-figures-v1.0.0"
$Staging = Join-Path $FinalParent "qgain-v4.1.0-figures-v1.0.0.__staging__"
$Notebook = Join-Path $ProjectRoot "notebooks\02_feature_extraction\01_QGAIN\02_figures_executed.ipynb"
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$TestFile = Join-Path $ProjectRoot "tests\test_qgain_figure_completion_v100.py"

foreach ($Path in @($MeasurementFreeze, $Candidate, $Notebook, $Python, $TestFile)) {
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "Required path is missing: $Path"
    }
}
if (Test-Path -LiteralPath $Final) {
    throw "Immutable figure package already exists and will not be overwritten: $Final"
}
if (Test-Path -LiteralPath $Staging) {
    Remove-Item -LiteralPath $Staging -Recurse -Force
}

# Validate executed notebook and candidate manifest.
$env:QGAIN_FIG_NOTEBOOK = $Notebook
$env:QGAIN_FIG_CANDIDATE = $Candidate
$NotebookCheck = @'
import json
import os
from pathlib import Path

nb_path = Path(os.environ["QGAIN_FIG_NOTEBOOK"])
candidate = Path(os.environ["QGAIN_FIG_CANDIDATE"])
nb = json.loads(nb_path.read_text(encoding="utf-8"))
code = [c for c in nb.get("cells", []) if c.get("cell_type") == "code"]
unexecuted = [i for i,c in enumerate(code) if c.get("execution_count") is None]
errors = []
for i,c in enumerate(code):
    for output in c.get("outputs", []):
        if output.get("output_type") == "error":
            errors.append((i, output.get("ename"), output.get("evalue")))
if unexecuted:
    raise SystemExit(f"Notebook has unexecuted code cells: {unexecuted}")
if errors:
    raise SystemExit(f"Notebook contains error outputs: {errors}")
manifest = json.loads((candidate / "manifests" / "qgain_v410_figure_package_manifest.json").read_text(encoding="utf-8"))
assert manifest["measurement_version"] == "qgain-v4.1.0"
assert manifest["figure_package_version"] == "qgain-v4.1.0-figures-v1.0.0"
assert manifest["required_panels_complete"] is True
assert manifest["feature_values_recomputed"] is False
assert manifest["standalone_gate_allowed"] is False
assert manifest["family_scalar_constructed"] is False
print("Validated executed figure-completion notebook and candidate manifest.")
'@
$NotebookCheck | & $Python -
if ($LASTEXITCODE -ne 0) { throw "Notebook/manifest validation failed." }

Push-Location $ProjectRoot
try {
    & $Python -m pytest $TestFile -q --disable-warnings
    if ($LASTEXITCODE -ne 0) { throw "Figure-package tests failed." }
}
finally { Pop-Location }

New-Item -ItemType Directory -Path $FinalParent -Force | Out-Null
Copy-Item -LiteralPath $Candidate -Destination $Staging -Recurse -Force

$Prov = Join-Path $Staging "provenance"
New-Item -ItemType Directory -Path $Prov -Force | Out-Null
Copy-Item -LiteralPath $Notebook -Destination (Join-Path $Prov "02_figures_executed.ipynb") -Force
Copy-Item -LiteralPath (Join-Path $ProjectRoot "src\paper1_qc_reviewed\qgain_figure_completion_v100.py") -Destination $Prov -Force
Copy-Item -LiteralPath $TestFile -Destination $Prov -Force
Copy-Item -LiteralPath (Join-Path $ProjectRoot "docs reviewed\QGAIN_FIGURE_COMPLETION_CONTRACT_v1_0.md") -Destination $Prov -Force

function Get-Sha256([string]$Path) {
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

# Create inventory before seal files.
$InventoryPath = Join-Path $Staging "manifests\qgain_v410_figure_package_freeze_inventory.csv"
$Files = Get-ChildItem -LiteralPath $Staging -Recurse -File | Sort-Object FullName
$Rows = foreach ($File in $Files) {
    [PSCustomObject]@{
        relative_path = $File.FullName.Substring($Staging.Length + 1).Replace('\','/')
        bytes = $File.Length
        sha256 = Get-Sha256 $File.FullName
    }
}
$Rows | Export-Csv -LiteralPath $InventoryPath -NoTypeInformation -Encoding UTF8
$InventorySha = Get-Sha256 $InventoryPath
$NotebookSha = Get-Sha256 (Join-Path $Prov "02_figures_executed.ipynb")

$CandidateManifestPath = Join-Path $Staging "manifests\qgain_v410_figure_package_manifest.json"
$CandidateManifest = Get-Content -LiteralPath $CandidateManifestPath -Raw | ConvertFrom-Json
$FinalManifest = [ordered]@{
    family = "QGAIN"
    measurement_version = "qgain-v4.1.0"
    figure_package_version = "qgain-v4.1.0-figures-v1.0.0"
    status = "frozen"
    source_measurement_freeze = $MeasurementFreeze
    source_freeze_manifest_sha256 = $CandidateManifest.source_freeze_manifest_sha256
    source_freeze_inventory_sha256 = $CandidateManifest.source_freeze_inventory_sha256
    source_executed_notebook_sha256 = $CandidateManifest.source_executed_notebook_sha256
    recording_count = $CandidateManifest.recording_count
    participant_count = $CandidateManifest.participant_count
    figure_count = $CandidateManifest.figure_count
    required_panels_complete = $CandidateManifest.required_panels_complete
    event_panel_I = $CandidateManifest.event_panel_I
    optional_panel_J_complete = $CandidateManifest.optional_panel_J_complete
    feature_values_recomputed = $false
    family_scalar_constructed = $false
    standalone_gate_allowed = $false
    executed_notebook_relative_path = "provenance/02_figures_executed.ipynb"
    executed_notebook_sha256 = $NotebookSha
    artifact_count_excluding_seal_files = $Rows.Count
    freeze_inventory_sha256 = $InventorySha
    freeze_created_utc = [DateTime]::UtcNow.ToString("o")
    immutability_policy = "never overwrite; create a new figure-package semantic version for any change"
}
$FinalManifestPath = Join-Path $Staging "manifests\qgain_v410_figure_package_freeze_manifest.json"
$FinalManifest | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $FinalManifestPath -Encoding UTF8

Move-Item -LiteralPath $Staging -Destination $Final

$PublishRoot = Join-Path $ProjectRoot "MAIN outputs\reviewed\08_validation_workbooks"
New-Item -ItemType Directory -Path $PublishRoot -Force | Out-Null
Copy-Item -LiteralPath (Join-Path $Final "docs\QGAIN_Family_Evaluation_Workbook_v1_0.docx") -Destination (Join-Path $PublishRoot "QGAIN_Family_Evaluation_Workbook_v1_0.docx") -Force
Copy-Item -LiteralPath (Join-Path $Final "tables\QGAIN_Validation_Checklist_v1_0.csv") -Destination (Join-Path $PublishRoot "QGAIN_Validation_Checklist_v1_0.csv") -Force
Copy-Item -LiteralPath (Join-Path $Final "tables\QGAIN_Ten_Domain_Dashboard_v1_0.csv") -Destination (Join-Path $PublishRoot "QGAIN_Ten_Domain_Dashboard_v1_0.csv") -Force

Write-Host ""
Write-Host "QGAIN STANDARDIZED VALIDATION AND FIGURE PACKAGE FROZEN SUCCESSFULLY" -ForegroundColor Green
Write-Host "Figure package: $Final" -ForegroundColor Cyan
Write-Host "Measurement freeze was not modified: $MeasurementFreeze" -ForegroundColor Cyan
