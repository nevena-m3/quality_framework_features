param(
    [Parameter(Mandatory = $true)]
    [string]$ProjectRoot
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path -LiteralPath $ProjectRoot).Path
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$MeasurementFreeze = Join-Path $ProjectRoot `
    "MAIN outputs reviewed\06_family_freezes\additive_interference\qadd-v4.2.0"
$FigureTargetParent = Join-Path $ProjectRoot `
    "MAIN outputs reviewed\07_figure_packages\additive_interference"
$FigureTarget = Join-Path $FigureTargetParent `
    "qadd-v4.2.0-figures-v1.0.0"
$WorkbookSource = Join-Path $ProjectRoot `
    "notebooks reviewed\02_QADD\QADD_Family_Evaluation_Workbook_v1_0.docx"

foreach ($Path in @($Python, $MeasurementFreeze, $WorkbookSource)) {
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "Required QADD figure-package input is missing: $Path"
    }
}
if (Test-Path -LiteralPath $FigureTarget) {
    throw "Refusing to overwrite immutable QADD figure package: $FigureTarget"
}

$MeasurementManifest = Join-Path $MeasurementFreeze `
    "manifests\qadd_v420_freeze_manifest.json"
$FigureIndex = Join-Path $MeasurementFreeze `
    "figures\qadd_v420_standardized_figure_index.csv"
foreach ($Path in @($MeasurementManifest, $FigureIndex)) {
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "Required sealed QADD artifact is missing: $Path"
    }
}

$env:QADD_MEASUREMENT_FREEZE = $MeasurementFreeze
$env:QADD_MEASUREMENT_MANIFEST = $MeasurementManifest
$env:QADD_FIGURE_INDEX = $FigureIndex

$ValidationScript = @'
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

root = Path(os.environ["QADD_MEASUREMENT_FREEZE"])
manifest_path = Path(os.environ["QADD_MEASUREMENT_MANIFEST"])
index_path = Path(os.environ["QADD_FIGURE_INDEX"])

manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
required = {
    "measurement_version": "qadd-v4.2.0",
    "freeze_status": "frozen",
    "scientific_review_decision": "ACCEPT_QADD_V420",
    "required_panels_complete": True,
}
for key, expected in required.items():
    if manifest.get(key) != expected:
        raise SystemExit(f"Measurement manifest mismatch for {key}")

index = pd.read_csv(index_path)
required_panels = set("ABCDEFGHJ")
observed = set(index.loc[index["panel"] != "I", "panel"])
if observed != required_panels:
    raise SystemExit(f"Figure panels mismatch: {observed} != {required_panels}")
if index.loc[index["panel"] == "I", "purpose"].item() != "no retained event detector":
    raise SystemExit("Panel I is not explicitly marked N/A")

artifact_columns = ["png", "svg", "pdf", "source_csv", "caption", "provenance"]
for _, row in index.loc[index["panel"] != "I"].iterrows():
    for column in artifact_columns:
        path = root / str(row[column])
        if not path.exists():
            raise SystemExit(f"Missing figure artifact: {path}")

print(f"Verified {len(index) - 1} indexed QADD figures plus explicit Panel I N/A.")
'@

$ValidationScript | & $Python -
if ($LASTEXITCODE -ne 0) {
    throw "QADD figure-package validation failed."
}

New-Item -ItemType Directory -Path $FigureTargetParent -Force | Out-Null
$Staging = Join-Path $FigureTargetParent `
    (".qadd-v4.2.0-figures-v1.0.0.staging." + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $Staging -Force | Out-Null

foreach ($Folder in @("figures", "galleries")) {
    Copy-Item `
        -LiteralPath (Join-Path $MeasurementFreeze $Folder) `
        -Destination (Join-Path $Staging $Folder) `
        -Recurse `
        -Force
}

$ValidationDestination = Join-Path $Staging "validation"
New-Item -ItemType Directory -Path $ValidationDestination -Force | Out-Null
$ValidationFiles = @(
    "validation\qadd_v420_ten_domain_dashboard.csv",
    "validation\qadd_v420_gate_summary_final.csv",
    "validation\qadd_v420_g10_feature_decisions.csv",
    "validation\qadd_v420_figure_contract.csv",
    "tables\qadd_v420_hum_joint_evidence_summary.csv",
    "tables\qadd_v420_empirical_summary.csv",
    "tables\qadd_v420_repeated_recording_persistence.csv",
    "tables\qadd_v420_weighting_comparison.csv",
    "tables\qadd_v420_spearman_correlations.csv"
)
foreach ($RelativePath in $ValidationFiles) {
    $Source = Join-Path $MeasurementFreeze $RelativePath
    if (Test-Path -LiteralPath $Source) {
        Copy-Item -LiteralPath $Source -Destination $ValidationDestination -Force
    }
}

$Provenance = Join-Path $Staging "provenance"
New-Item -ItemType Directory -Path $Provenance -Force | Out-Null
Copy-Item -LiteralPath $WorkbookSource -Destination $Provenance -Force

$ProvenanceFiles = @(
    "QADD_v420_FINAL_SCIENTIFIC_AUDIT.md",
    "QADD_v420_FINAL_FEATURE_DECISIONS.csv",
    "QADD_V4_2_0_FREEZE_CONTRACT.md",
    "QADD_Validation_Checklist_v1_0.csv",
    "QADD_Ten_Domain_Dashboard_v1_0.csv"
)
foreach ($Name in $ProvenanceFiles) {
    $Source = Join-Path $ProjectRoot "notebooks reviewed\02_QADD\$Name"
    if (Test-Path -LiteralPath $Source) {
        Copy-Item -LiteralPath $Source -Destination $Provenance -Force
    }
}

$ExecutedNotebook = Join-Path $MeasurementFreeze `
    "provenance\02a_additive_interference_QADD_v4_2_0_EXECUTED_FINAL.ipynb"
if (Test-Path -LiteralPath $ExecutedNotebook) {
    Copy-Item -LiteralPath $ExecutedNotebook -Destination $Provenance -Force
}

$env:QADD_FIGURE_STAGING = $Staging
$env:QADD_WORKBOOK = Join-Path $Provenance `
    "QADD_Family_Evaluation_Workbook_v1_0.docx"

$SealScript = @'
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

root = Path(os.environ["QADD_FIGURE_STAGING"])
measurement_root = Path(os.environ["QADD_MEASUREMENT_FREEZE"])
measurement_manifest_path = Path(os.environ["QADD_MEASUREMENT_MANIFEST"])
workbook = Path(os.environ["QADD_WORKBOOK"])

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

measurement_manifest = json.loads(
    measurement_manifest_path.read_text(encoding="utf-8")
)
index_path = root / "figures" / "qadd_v420_standardized_figure_index.csv"
index = pd.read_csv(index_path)

excluded = {
    "manifests/qadd_v420_figure_package_manifest.json",
    "manifests/qadd_v420_figure_package_inventory.csv",
}
manifest_dir = root / "manifests"
manifest_dir.mkdir(parents=True, exist_ok=True)

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
inventory_path = manifest_dir / "qadd_v420_figure_package_inventory.csv"
inventory.to_csv(inventory_path, index=False)

figure_manifest = {
    "package_version": "qadd-v4.2.0-figures-v1.0.0",
    "measurement_version": "qadd-v4.2.0",
    "freeze_status": "frozen",
    "measurement_freeze_manifest_sha256": sha256(measurement_manifest_path),
    "measurement_freeze_inventory_sha256": measurement_manifest[
        "freeze_inventory_sha256"
    ],
    "measurement_executed_notebook_sha256": measurement_manifest[
        "executed_notebook_sha256"
    ],
    "figure_count": int((index["panel"] != "I").sum()),
    "required_panels_complete": True,
    "panel_i_status": "N/A_no_retained_event_detector",
    "workbook_relative_path": workbook.relative_to(root).as_posix(),
    "workbook_sha256": sha256(workbook),
    "artifact_count_excluding_seal_files": int(len(inventory)),
    "figure_package_inventory_sha256": sha256(inventory_path),
    "feature_values_recomputed": False,
    "family_scalar_constructed": False,
    "standalone_gate_allowed": False,
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "immutability_policy": "never overwrite; create a new package version for any change",
}
(manifest_dir / "qadd_v420_figure_package_manifest.json").write_text(
    json.dumps(figure_manifest, indent=2),
    encoding="utf-8",
)
(root / "FROZEN_QADD_V4_2_0_FIGURES_V1_0_0.txt").write_text(
    "QADD v4.2.0 standardized figure package v1.0.0 is frozen.\n",
    encoding="utf-8",
)
print(json.dumps(figure_manifest, indent=2))
'@

$SealScript | & $Python -
if ($LASTEXITCODE -ne 0) {
    Remove-Item `
        -LiteralPath $Staging `
        -Recurse `
        -Force `
        -ErrorAction SilentlyContinue
    throw "Failed to seal QADD figure package."
}

Move-Item -LiteralPath $Staging -Destination $FigureTarget

$WorkbookDestinationRoot = Join-Path $ProjectRoot `
    "MAIN outputs reviewed\08_validation_workbooks"
New-Item -ItemType Directory -Path $WorkbookDestinationRoot -Force | Out-Null
$WorkbookDestination = Join-Path $WorkbookDestinationRoot `
    "QADD_Family_Evaluation_Workbook_v1_0.docx"
if (Test-Path -LiteralPath $WorkbookDestination) {
    throw "Refusing to overwrite existing validation workbook: $WorkbookDestination"
}
Copy-Item `
    -LiteralPath $WorkbookSource `
    -Destination $WorkbookDestination `
    -Force

Write-Host ""
Write-Host "QADD STANDARDIZED VALIDATION AND FIGURE PACKAGE FROZEN SUCCESSFULLY" `
    -ForegroundColor Green
Write-Host "Figure package: $FigureTarget" -ForegroundColor Cyan
Write-Host "Workbook: $WorkbookDestination" -ForegroundColor Cyan
