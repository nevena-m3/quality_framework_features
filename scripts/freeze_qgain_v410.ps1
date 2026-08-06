param(
    [Parameter(Mandatory = $true)]
    [string]$ProjectRoot
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path -LiteralPath $ProjectRoot).Path
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Candidate = Join-Path $ProjectRoot "outputs\reviewed\gain_dynamics\qgain-v4.1.0-candidate"
$NotebookDir = Join-Path $ProjectRoot "notebooks\02_feature_extraction\01_QGAIN"
$Module = Join-Path $ProjectRoot "src\paper1_qc_reviewed\qgain_v410.py"
$Tests = Join-Path $ProjectRoot "tests\test_qgain_v410.py"
$Contract = Join-Path $NotebookDir "QGAIN_V4_1_0_FREEZE_CONTRACT.md"
$Audit = Join-Path $NotebookDir "QGAIN_v401_FINAL_SCIENTIFIC_AUDIT.md"
$Decisions = Join-Path $NotebookDir "QGAIN_v401_FEATURE_DECISIONS.csv"

foreach ($Path in @($Python, $Candidate, $NotebookDir, $Module, $Tests, $Contract)) {
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "Required QGAIN freeze input is missing: $Path"
    }
}

$ExecutedNotebook = Get-ChildItem `
    -LiteralPath $NotebookDir `
    -File `
    -Filter "*QGAIN_v4_1_0*LOCAL_FINALIZE*.ipynb" |
Sort-Object LastWriteTime -Descending |
Select-Object -First 1

if (-not $ExecutedNotebook) {
    throw "Could not locate the saved QGAIN v4.1.0 local finalization notebook."
}

$CandidateManifest = Join-Path $Candidate "manifests\qgain_v410_candidate_manifest.json"
if (-not (Test-Path -LiteralPath $CandidateManifest)) {
    throw "Candidate manifest is missing: $CandidateManifest"
}

$env:QGAIN_NOTEBOOK = $ExecutedNotebook.FullName
$env:QGAIN_CANDIDATE_MANIFEST = $CandidateManifest

$ValidationScript = @'
import json
import os
from pathlib import Path

notebook_path = Path(os.environ["QGAIN_NOTEBOOK"])
manifest_path = Path(os.environ["QGAIN_CANDIDATE_MANIFEST"])

with notebook_path.open("r", encoding="utf-8") as handle:
    notebook = json.load(handle)

code_cells = [cell for cell in notebook.get("cells", []) if cell.get("cell_type") == "code"]
unexecuted = [index for index, cell in enumerate(code_cells) if cell.get("execution_count") is None]
errors = []
text_outputs = []
for index, cell in enumerate(code_cells):
    for output in cell.get("outputs", []):
        if output.get("output_type") == "error":
            errors.append({
                "cell_index": index,
                "ename": output.get("ename"),
                "evalue": output.get("evalue"),
            })
        if output.get("output_type") == "stream":
            text_outputs.append(str(output.get("text", "")))

if unexecuted:
    raise SystemExit(f"Notebook has unexecuted code cells: {unexecuted}")
if errors:
    raise SystemExit(f"Notebook contains error outputs: {errors}")
if "QGAIN v4.1.0 FINALIZATION COMPLETE" not in "\n".join(text_outputs):
    raise SystemExit("Notebook completion marker was not found. Save the fully executed notebook first.")

manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
if manifest.get("measurement_version") != "qgain-v4.1.0":
    raise SystemExit("Candidate manifest has the wrong measurement version.")
if manifest.get("scientific_review_decision") != "ACCEPT_QGAIN_V410":
    raise SystemExit("The exact QGAIN v4.1.0 acceptance token is absent.")
if not manifest.get("freeze_allowed", False):
    raise SystemExit("Candidate manifest does not authorize freeze.")
if not manifest.get("numerical_equivalence_to_v401", False):
    raise SystemExit("Numerical equivalence to validated v4.0.1 was not established.")

print(f"Validated executed notebook: {notebook_path.name}")
print("Candidate manifest authorizes atomic freeze.")
'@

$ValidationScript | & $Python -
if ($LASTEXITCODE -ne 0) {
    throw "QGAIN notebook/manifest validation failed."
}

# Rerun package tests immediately before freeze.
Push-Location $ProjectRoot
try {
    & $Python -m pytest "tests\test_qgain_v410.py" -q
    if ($LASTEXITCODE -ne 0) {
        throw "QGAIN v4.1.0 tests failed immediately before freeze."
    }
}
finally {
    Pop-Location
}

$FreezeParent = Join-Path $ProjectRoot "MAIN outputs\reviewed\06_family_freezes\gain_dynamics"
$FreezeTarget = Join-Path $FreezeParent "qgain-v4.1.0"
if (Test-Path -LiteralPath $FreezeTarget) {
    throw "Refusing to overwrite immutable QGAIN freeze: $FreezeTarget"
}

$MainReviewed = Join-Path $ProjectRoot "MAIN outputs/reviewed"
$CanonicalDestinations = @(
    "01_analysis_features\qgain_v410_analysis_features.csv",
    "01_analysis_features\qgain_v410_analysis_features.parquet",
    "02_support_and_availability\qgain_v410_measurements_long.csv",
    "02_support_and_availability\qgain_v410_measurements_long.parquet",
    "04_model_ready_features\qgain_v410_model_interface.csv",
    "04_model_ready_features\qgain_v410_model_interface.parquet",
    "00_feature_registry\qgain_v410_feature_registry.csv",
    "00_feature_registry\qgain_v410_feature_registry.parquet",
    "05_feature_passports\qgain-v4.1.0"
)
foreach ($RelativePath in $CanonicalDestinations) {
    $DestinationPath = Join-Path $MainReviewed $RelativePath
    if (Test-Path -LiteralPath $DestinationPath) {
        throw "Refusing to overwrite existing canonical reviewed output: $DestinationPath"
    }
}
New-Item -ItemType Directory -Path $FreezeParent -Force | Out-Null

$Staging = Join-Path $FreezeParent (".qgain-v4.1.0.staging." + [guid]::NewGuid().ToString("N"))
Copy-Item -LiteralPath $Candidate -Destination $Staging -Recurse -Force

$Provenance = Join-Path $Staging "provenance"
New-Item -ItemType Directory -Path $Provenance -Force | Out-Null
Copy-Item -LiteralPath $ExecutedNotebook.FullName -Destination (Join-Path $Provenance "02b_gain_dynamics_QGAIN_v4_1_0_EXECUTED_FINAL.ipynb") -Force
Copy-Item -LiteralPath $Module -Destination $Provenance -Force
Copy-Item -LiteralPath $Tests -Destination $Provenance -Force
Copy-Item -LiteralPath $Contract -Destination $Provenance -Force
if (Test-Path -LiteralPath $Audit) { Copy-Item -LiteralPath $Audit -Destination $Provenance -Force }
if (Test-Path -LiteralPath $Decisions) { Copy-Item -LiteralPath $Decisions -Destination $Provenance -Force }

$env:QGAIN_STAGING = $Staging
$env:QGAIN_EXECUTED_NOTEBOOK = (Join-Path $Provenance "02b_gain_dynamics_QGAIN_v4_1_0_EXECUTED_FINAL.ipynb")

$SealScript = @'
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

root = Path(os.environ["QGAIN_STAGING"])
notebook = Path(os.environ["QGAIN_EXECUTED_NOTEBOOK"])

registry_csv = root / "tables" / "qgain_v410_feature_registry.csv"
registry = pd.read_csv(registry_csv)
registry["publication_status"] = "frozen"
registry.to_csv(registry_csv, index=False)
try:
    registry.to_parquet(root / "tables" / "qgain_v410_feature_registry.parquet", index=False)
except Exception:
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

excluded = {
    "manifests/qgain_v410_freeze_manifest.json",
    "manifests/qgain_v410_freeze_inventory.csv",
}
rows = []
for path in sorted(item for item in root.rglob("*") if item.is_file()):
    relative = path.relative_to(root).as_posix()
    if relative in excluded:
        continue
    rows.append({
        "relative_path": relative,
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    })
inventory = pd.DataFrame(rows)
inventory_path = root / "manifests" / "qgain_v410_freeze_inventory.csv"
inventory.to_csv(inventory_path, index=False)

candidate_manifest_path = root / "manifests" / "qgain_v410_candidate_manifest.json"
candidate_manifest = json.loads(candidate_manifest_path.read_text(encoding="utf-8"))
freeze_manifest = {
    **candidate_manifest,
    "candidate_only": False,
    "freeze_status": "frozen",
    "freeze_allowed": True,
    "freeze_created_utc": datetime.now(timezone.utc).isoformat(),
    "executed_notebook_relative_path": notebook.relative_to(root).as_posix(),
    "executed_notebook_sha256": sha256(notebook),
    "artifact_count_excluding_seal_files": int(len(inventory)),
    "freeze_inventory_sha256": sha256(inventory_path),
    "immutability_policy": "never overwrite; create a new semantic version for any change",
}
(root / "manifests" / "qgain_v410_freeze_manifest.json").write_text(
    json.dumps(freeze_manifest, indent=2), encoding="utf-8"
)
(root / "FROZEN_QGAIN_V4_1_0.txt").write_text(
    "QGAIN v4.1.0 is frozen. Do not modify this directory.\n",
    encoding="utf-8",
)
print(json.dumps(freeze_manifest, indent=2))
'@

$SealScript | & $Python -
if ($LASTEXITCODE -ne 0) {
    Remove-Item -LiteralPath $Staging -Recurse -Force -ErrorAction SilentlyContinue
    throw "Failed to seal QGAIN staging freeze."
}

# Atomic rename on the same volume.
Move-Item -LiteralPath $Staging -Destination $FreezeTarget

# Refuse to overwrite canonical reviewed outputs.
$CanonicalPairs = @(
    @{ Source = "tables\qgain_v410_recording_features"; Destination = "01_analysis_features\qgain_v410_analysis_features" },
    @{ Source = "tables\qgain_v410_measurements_long"; Destination = "02_support_and_availability\qgain_v410_measurements_long" },
    @{ Source = "tables\qgain_v410_model_interface"; Destination = "04_model_ready_features\qgain_v410_model_interface" },
    @{ Source = "tables\qgain_v410_feature_registry"; Destination = "00_feature_registry\qgain_v410_feature_registry" }
)

foreach ($Pair in $CanonicalPairs) {
    foreach ($Extension in @(".csv", ".parquet")) {
        $SourcePath = Join-Path $FreezeTarget ($Pair.Source + $Extension)
        if (Test-Path -LiteralPath $SourcePath) {
            $DestinationPath = Join-Path $MainReviewed ($Pair.Destination + $Extension)
            if (Test-Path -LiteralPath $DestinationPath) {
                throw "Canonical reviewed output already exists; freeze is preserved but canonical copy was not overwritten: $DestinationPath"
            }
            New-Item -ItemType Directory -Path (Split-Path $DestinationPath -Parent) -Force | Out-Null
            Copy-Item -LiteralPath $SourcePath -Destination $DestinationPath -Force
        }
    }
}

$PassportSource = Join-Path $FreezeTarget "feature_passports"
$PassportDestination = Join-Path $MainReviewed "05_feature_passports\qgain-v4.1.0"
if (Test-Path -LiteralPath $PassportDestination) {
    throw "Feature-passport destination already exists: $PassportDestination"
}
New-Item -ItemType Directory -Path (Split-Path $PassportDestination -Parent) -Force | Out-Null
Copy-Item -LiteralPath $PassportSource -Destination $PassportDestination -Recurse -Force

Write-Host ""
Write-Host "QGAIN v4.1.0 FROZEN SUCCESSFULLY" -ForegroundColor Green
Write-Host "Freeze: $FreezeTarget" -ForegroundColor Cyan
Write-Host "Canonical reviewed outputs were published under MAIN outputs/reviewed." -ForegroundColor Cyan
