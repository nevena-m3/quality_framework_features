param(
    [Parameter(Mandatory = $true)]
    [string]$ProjectRoot
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path -LiteralPath $ProjectRoot).Path
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Candidate = Join-Path $ProjectRoot `
    "outputs\reviewed\additive_interference\qadd-v4.2.0-final-candidate"
$NotebookDir = Join-Path $ProjectRoot "notebooks\02_feature_extraction\02_QADD"

$RequiredInputs = @(
    $Python,
    $Candidate,
    (Join-Path $ProjectRoot "src\paper1_qc_reviewed\qadd_v420.py"),
    (Join-Path $ProjectRoot "src\paper1_qc_reviewed\qadd_v420_cohort.py"),
    (Join-Path $ProjectRoot "src\paper1_qc_reviewed\qadd_v420_final.py"),
    (Join-Path $ProjectRoot "tests\test_qadd_v420.py"),
    (Join-Path $ProjectRoot "tests\test_qadd_v420_cohort.py"),
    (Join-Path $ProjectRoot "tests\test_qadd_v420_final.py"),
    (Join-Path $NotebookDir "QADD_V4_2_0_FREEZE_CONTRACT.md"),
    (Join-Path $NotebookDir "QADD_v420_FINAL_SCIENTIFIC_AUDIT.md"),
    (Join-Path $NotebookDir "QADD_v420_FINAL_FEATURE_DECISIONS.csv"),
    (Join-Path $NotebookDir "QADD_Family_Evaluation_Workbook_v1_0.docx"),
    (Join-Path $NotebookDir "QADD_Validation_Checklist_v1_0.csv"),
    (Join-Path $NotebookDir "QADD_Ten_Domain_Dashboard_v1_0.csv")
)
foreach ($Path in $RequiredInputs) {
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "Required QADD freeze input is missing: $Path"
    }
}

$ExecutedNotebook = Get-ChildItem `
    -LiteralPath $NotebookDir `
    -File `
    -Filter "*QADD_v4_2_0*LOCAL_FINALIZE*.ipynb" |
Sort-Object LastWriteTime -Descending |
Select-Object -First 1
if (-not $ExecutedNotebook) {
    throw "Could not locate the saved QADD v4.2.0 finalization notebook."
}

$CandidateManifest = Join-Path $Candidate `
    "manifests\qadd_v420_final_candidate_manifest.json"
if (-not (Test-Path -LiteralPath $CandidateManifest)) {
    throw "Final candidate manifest is missing: $CandidateManifest"
}

$env:QADD_NOTEBOOK = $ExecutedNotebook.FullName
$env:QADD_CANDIDATE_MANIFEST = $CandidateManifest

$ValidationScript = @'
import json
import os
from pathlib import Path

notebook_path = Path(os.environ["QADD_NOTEBOOK"])
manifest_path = Path(os.environ["QADD_CANDIDATE_MANIFEST"])

notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
code_cells = [
    cell for cell in notebook.get("cells", [])
    if cell.get("cell_type") == "code"
]
unexecuted = [
    index for index, cell in enumerate(code_cells)
    if cell.get("execution_count") is None
]
errors = []
streams = []
for index, cell in enumerate(code_cells):
    for output in cell.get("outputs", []):
        if output.get("output_type") == "error":
            errors.append({
                "cell_index": index,
                "ename": output.get("ename"),
                "evalue": output.get("evalue"),
            })
        if output.get("output_type") == "stream":
            text = output.get("text", "")
            if isinstance(text, list):
                text = "".join(text)
            streams.append(str(text))

if unexecuted:
    raise SystemExit(f"Notebook has unexecuted code cells: {unexecuted}")
if errors:
    raise SystemExit(f"Notebook contains error outputs: {errors}")
if "QADD v4.2.0 FINALIZATION COMPLETE" not in "\n".join(streams):
    raise SystemExit("QADD finalization completion marker was not found.")

manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
required = {
    "measurement_version": "qadd-v4.2.0",
    "scientific_review_decision": "ACCEPT_QADD_V420",
    "freeze_allowed": True,
    "freeze_status": "ready_for_atomic_freeze",
    "numerical_equivalence_to_cohort_candidate": True,
    "hum_winner_counts_match_eligible": True,
    "required_panels_complete": True,
    "panel_i_status": "N/A_no_retained_event_detector",
    "feature_values_recomputed": False,
}
for key, expected in required.items():
    observed = manifest.get(key)
    if observed != expected:
        raise SystemExit(f"Manifest mismatch for {key}: {observed!r} != {expected!r}")

print(f"Validated executed notebook: {notebook_path.name}")
print("QADD final candidate manifest authorizes atomic freeze.")
'@

$ValidationScript | & $Python -
if ($LASTEXITCODE -ne 0) {
    throw "QADD notebook/manifest validation failed."
}

Push-Location $ProjectRoot
try {
    & $Python -m pytest `
        "tests\test_qadd_v420.py" `
        "tests\test_qadd_v420_cohort.py" `
        "tests\test_qadd_v420_final.py" `
        -q `
        --disable-warnings
    if ($LASTEXITCODE -ne 0) {
        throw "QADD v4.2.0 tests failed immediately before freeze."
    }
}
finally {
    Pop-Location
}

$FreezeParent = Join-Path $ProjectRoot `
    "MAIN outputs\reviewed\06_family_freezes\additive_interference"
$FreezeTarget = Join-Path $FreezeParent "qadd-v4.2.0"
if (Test-Path -LiteralPath $FreezeTarget) {
    throw "Refusing to overwrite immutable QADD freeze: $FreezeTarget"
}

$MainReviewed = Join-Path $ProjectRoot "MAIN outputs/reviewed"
$CanonicalDestinations = @(
    "00_feature_registry\qadd_v420_feature_registry.csv",
    "00_feature_registry\qadd_v420_feature_registry.parquet",
    "01_analysis_features\qadd_v420_analysis_features.csv",
    "01_analysis_features\qadd_v420_analysis_features.parquet",
    "02_support_and_availability\qadd_v420_measurements_long.csv",
    "02_support_and_availability\qadd_v420_measurements_long.parquet",
    "04_model_ready_features\qadd_v420_model_interface.csv",
    "04_model_ready_features\qadd_v420_model_interface.parquet",
    "05_feature_passports\qadd-v4.2.0"
)
foreach ($RelativePath in $CanonicalDestinations) {
    $DestinationPath = Join-Path $MainReviewed $RelativePath
    if (Test-Path -LiteralPath $DestinationPath) {
        throw "Refusing to overwrite existing canonical reviewed output: $DestinationPath"
    }
}

New-Item -ItemType Directory -Path $FreezeParent -Force | Out-Null
$Staging = Join-Path $FreezeParent `
    (".qadd-v4.2.0.staging." + [guid]::NewGuid().ToString("N"))
Copy-Item -LiteralPath $Candidate -Destination $Staging -Recurse -Force

$Provenance = Join-Path $Staging "provenance"
New-Item -ItemType Directory -Path $Provenance -Force | Out-Null
Copy-Item `
    -LiteralPath $ExecutedNotebook.FullName `
    -Destination (Join-Path $Provenance `
        "02a_additive_interference_QADD_v4_2_0_EXECUTED_FINAL.ipynb") `
    -Force

$ProvenanceFiles = @(
    "src\paper1_qc_reviewed\qadd_v420.py",
    "src\paper1_qc_reviewed\qadd_v420_cohort.py",
    "src\paper1_qc_reviewed\qadd_v420_final.py",
    "tests\test_qadd_v420.py",
    "tests\test_qadd_v420_cohort.py",
    "tests\test_qadd_v420_final.py",
    "notebooks\02_feature_extraction\02_QADD\QADD_V4_2_0_FREEZE_CONTRACT.md",
    "notebooks\02_feature_extraction\02_QADD\QADD_v420_FINAL_SCIENTIFIC_AUDIT.md",
    "notebooks\02_feature_extraction\02_QADD\QADD_v420_FINAL_FEATURE_DECISIONS.csv",
    "notebooks\02_feature_extraction\02_QADD\QADD_Family_Evaluation_Workbook_v1_0.docx",
    "notebooks\02_feature_extraction\02_QADD\QADD_Validation_Checklist_v1_0.csv",
    "notebooks\02_feature_extraction\02_QADD\QADD_Ten_Domain_Dashboard_v1_0.csv"
)
foreach ($RelativePath in $ProvenanceFiles) {
    Copy-Item `
        -LiteralPath (Join-Path $ProjectRoot $RelativePath) `
        -Destination $Provenance `
        -Force
}

$env:QADD_STAGING = $Staging
$env:QADD_EXECUTED_NOTEBOOK = Join-Path $Provenance `
    "02a_additive_interference_QADD_v4_2_0_EXECUTED_FINAL.ipynb"

$SealScript = @'
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

root = Path(os.environ["QADD_STAGING"])
notebook = Path(os.environ["QADD_EXECUTED_NOTEBOOK"])

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

registry_path = root / "tables" / "qadd_v420_feature_registry.csv"
registry = pd.read_csv(registry_path)
registry["publication_status"] = "frozen"
registry.to_csv(registry_path, index=False)
try:
    registry.to_parquet(root / "tables" / "qadd_v420_feature_registry.parquet", index=False)
except Exception:
    pass

excluded = {
    "manifests/qadd_v420_freeze_manifest.json",
    "manifests/qadd_v420_freeze_inventory.csv",
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
inventory_path = root / "manifests" / "qadd_v420_freeze_inventory.csv"
inventory.to_csv(inventory_path, index=False)

candidate_manifest_path = root / "manifests" / "qadd_v420_final_candidate_manifest.json"
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
(root / "manifests" / "qadd_v420_freeze_manifest.json").write_text(
    json.dumps(freeze_manifest, indent=2),
    encoding="utf-8",
)
(root / "FROZEN_QADD_V4_2_0.txt").write_text(
    "QADD v4.2.0 is frozen. Do not modify this directory.\n",
    encoding="utf-8",
)
print(json.dumps(freeze_manifest, indent=2))
'@

$SealScript | & $Python -
if ($LASTEXITCODE -ne 0) {
    Remove-Item `
        -LiteralPath $Staging `
        -Recurse `
        -Force `
        -ErrorAction SilentlyContinue
    throw "Failed to seal QADD staging freeze."
}

Move-Item -LiteralPath $Staging -Destination $FreezeTarget

$CanonicalPairs = @(
    @{ Source = "tables\qadd_v420_recording_features"; Destination = "01_analysis_features\qadd_v420_analysis_features" },
    @{ Source = "tables\qadd_v420_measurements_long"; Destination = "02_support_and_availability\qadd_v420_measurements_long" },
    @{ Source = "tables\qadd_v420_model_interface"; Destination = "04_model_ready_features\qadd_v420_model_interface" },
    @{ Source = "tables\qadd_v420_feature_registry"; Destination = "00_feature_registry\qadd_v420_feature_registry" }
)
foreach ($Pair in $CanonicalPairs) {
    foreach ($Extension in @(".csv", ".parquet")) {
        $SourcePath = Join-Path $FreezeTarget ($Pair.Source + $Extension)
        if (Test-Path -LiteralPath $SourcePath) {
            $DestinationPath = Join-Path $MainReviewed `
                ($Pair.Destination + $Extension)
            if (Test-Path -LiteralPath $DestinationPath) {
                throw "Canonical output already exists: $DestinationPath"
            }
            New-Item `
                -ItemType Directory `
                -Path (Split-Path -Parent $DestinationPath) `
                -Force |
            Out-Null
            Copy-Item `
                -LiteralPath $SourcePath `
                -Destination $DestinationPath `
                -Force
        }
    }
}

$PassportSource = Join-Path $FreezeTarget "feature_passports"
$PassportDestination = Join-Path $MainReviewed `
    "05_feature_passports\qadd-v4.2.0"
New-Item `
    -ItemType Directory `
    -Path (Split-Path -Parent $PassportDestination) `
    -Force |
Out-Null
Copy-Item `
    -LiteralPath $PassportSource `
    -Destination $PassportDestination `
    -Recurse `
    -Force

Write-Host ""
Write-Host "QADD v4.2.0 FROZEN SUCCESSFULLY" -ForegroundColor Green
Write-Host "Freeze: $FreezeTarget" -ForegroundColor Cyan
Write-Host "Canonical reviewed outputs were published under MAIN outputs/reviewed." `
    -ForegroundColor Cyan
