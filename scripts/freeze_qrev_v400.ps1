param(
    [Parameter(Mandatory = $true)]
    [string]$ProjectRoot
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path -LiteralPath $ProjectRoot).Path
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$FinalCandidate = Join-Path $ProjectRoot "outputs\reviewed\reverberation\qrev-v4.0.0"
$FinalManifest = Join-Path $FinalCandidate "manifests\qrev_v400_final_candidate_manifest.json"
$ExecutedNotebook = Join-Path $ProjectRoot "notebooks\02_feature_extraction\03_QREV\03_reverberation_QREV_v4_0_0_LOCAL_FINALIZE.ipynb"
$FreezeParent = Join-Path $ProjectRoot "MAIN outputs\reviewed\06_family_freezes\reverberation"
$FreezeTarget = Join-Path $FreezeParent "qrev-v4.0.0"

foreach ($Path in @($Python, $FinalCandidate, $FinalManifest, $ExecutedNotebook)) {
    if (-not (Test-Path -LiteralPath $Path)) { throw "Required QREV freeze input is missing: $Path" }
}
if (Test-Path -LiteralPath $FreezeTarget) { throw "Refusing to overwrite immutable QREV freeze: $FreezeTarget" }

$env:QREV_FINAL_CANDIDATE = $FinalCandidate
$env:QREV_FINAL_MANIFEST = $FinalManifest
$env:QREV_EXECUTED_NOTEBOOK = $ExecutedNotebook

$Validate = @'
import json, os
from pathlib import Path
import pandas as pd

root=Path(os.environ['QREV_FINAL_CANDIDATE'])
manifest=json.loads(Path(os.environ['QREV_FINAL_MANIFEST']).read_text(encoding='utf-8'))
required={
 'measurement_version':'qrev-v4.0.0',
 'freeze_allowed':True,
 'freeze_status':'ready_for_atomic_freeze',
 'scientific_review_decision':'ACCEPT_QREV_V400',
 'numerical_equivalence_to_cohort_candidate':True,
 'horizon_sensitivity_corrected':True,
 'required_panels_complete':True,
 'panel_i_status':'N/A_no_retained_event_detector',
 'feature_values_recomputed':False,
 'standalone_reject_allowed':False,
 'support_tier_is_precision':False,
}
for key,expected in required.items():
    if manifest.get(key)!=expected:
        raise SystemExit(f'Manifest mismatch for {key}: {manifest.get(key)!r}')
nb=json.loads(Path(os.environ['QREV_EXECUTED_NOTEBOOK']).read_text(encoding='utf-8'))
code=[c for c in nb.get('cells',[]) if c.get('cell_type')=='code']
if any(c.get('execution_count') is None for c in code): raise SystemExit('Finalization notebook has unexecuted cells')
errors=[o for c in code for o in c.get('outputs',[]) if o.get('output_type')=='error']
if errors: raise SystemExit(f'Finalization notebook contains errors: {errors}')
index=pd.read_csv(root/'figures/qrev_v400_standardized_figure_index.csv')
if set(index.loc[index.panel!='I','panel'])!=set('ABCDEFGHJ'): raise SystemExit('Figure panels are incomplete')
if len(index.loc[index.panel!='I'])!=22: raise SystemExit('Expected 22 applicable QREV figure bundles')
print('QREV final candidate and executed notebook authorize atomic freeze.')
'@
$Validate | & $Python -
if ($LASTEXITCODE -ne 0) { throw "QREV freeze validation failed." }

New-Item -ItemType Directory -Path $FreezeParent -Force | Out-Null
$Staging = Join-Path $FreezeParent (".qrev-v4.0.0.staging." + [guid]::NewGuid().ToString("N"))
Copy-Item -LiteralPath $FinalCandidate -Destination $Staging -Recurse -Force
$Provenance = Join-Path $Staging "provenance"
New-Item -ItemType Directory -Path $Provenance -Force | Out-Null
$FrozenNotebook = Join-Path $Provenance "03_reverberation_QREV_v4_0_0_EXECUTED_FINAL.ipynb"
Copy-Item -LiteralPath $ExecutedNotebook -Destination $FrozenNotebook -Force

$env:QREV_FREEZE_STAGING = $Staging
$env:QREV_FROZEN_NOTEBOOK = $FrozenNotebook
$Seal = @'
from __future__ import annotations
import hashlib,json,os
from datetime import datetime,timezone
from pathlib import Path
import pandas as pd

root=Path(os.environ['QREV_FREEZE_STAGING'])
notebook=Path(os.environ['QREV_FROZEN_NOTEBOOK'])
manifest_path=root/'manifests/qrev_v400_final_candidate_manifest.json'
source=json.loads(manifest_path.read_text(encoding='utf-8'))

def sha(path):
 d=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''): d.update(chunk)
 return d.hexdigest()

excluded={'manifests/qrev_v400_freeze_manifest.json','manifests/qrev_v400_freeze_inventory.csv'}
rows=[]
for path in sorted(p for p in root.rglob('*') if p.is_file()):
 rel=path.relative_to(root).as_posix()
 if rel in excluded: continue
 rows.append({'relative_path':rel,'bytes':path.stat().st_size,'sha256':sha(path)})
inv=pd.DataFrame(rows)
inv_path=root/'manifests/qrev_v400_freeze_inventory.csv'; inv.to_csv(inv_path,index=False)
freeze=dict(source)
freeze.update({
 'candidate_only':False,
 'freeze_status':'frozen',
 'freeze_allowed':True,
 'freeze_created_utc':datetime.now(timezone.utc).isoformat(),
 'executed_notebook_relative_path':notebook.relative_to(root).as_posix(),
 'executed_notebook_sha256':sha(notebook),
 'artifact_count_excluding_seal_files':len(inv),
 'freeze_inventory_sha256':sha(inv_path),
})
(root/'manifests/qrev_v400_freeze_manifest.json').write_text(json.dumps(freeze,indent=2),encoding='utf-8')
(root/'FROZEN_QREV_V4_0_0.txt').write_text('QREV v4.0.0 is frozen. Never overwrite.\n',encoding='utf-8')
print(json.dumps(freeze,indent=2))
'@
$Seal | & $Python -
if ($LASTEXITCODE -ne 0) { Remove-Item $Staging -Recurse -Force -ErrorAction SilentlyContinue; throw "Failed to seal QREV freeze." }
Move-Item -LiteralPath $Staging -Destination $FreezeTarget

# Publish canonical reviewed outputs without overwriting an existing QREV version.
$Publications = @(
    @{Source="tables\qrev_v400_feature_registry.csv"; Destination="00_feature_registry\qrev_v400_feature_registry.csv"},
    @{Source="tables\qrev_v400_analysis_features.csv"; Destination="01_analysis_features\qrev_v400_analysis_features.csv"},
    @{Source="validation\qrev_v400_status_missingness_summary.csv"; Destination="02_support_and_availability\qrev_v400_status_missingness_summary.csv"},
    @{Source="validation\qrev_v400_support_policy_availability.csv"; Destination="02_support_and_availability\qrev_v400_support_policy_availability.csv"},
    @{Source="tables\qrev_v400_model_ready_features.csv"; Destination="04_model_ready_features\qrev_v400_model_ready_features.csv"}
)
foreach ($Item in $Publications) {
    $Source=Join-Path $FreezeTarget $Item.Source
    $Destination=Join-Path (Join-Path $ProjectRoot "MAIN outputs/reviewed") $Item.Destination
    if (Test-Path -LiteralPath $Destination) { throw "Refusing to overwrite canonical reviewed output: $Destination" }
    New-Item -ItemType Directory -Path (Split-Path -Parent $Destination) -Force | Out-Null
    Copy-Item -LiteralPath $Source -Destination $Destination -Force
}
$PassportDestination=Join-Path $ProjectRoot "MAIN outputs\reviewed\05_feature_passports\reverberation\qrev-v4.0.0"
if (Test-Path -LiteralPath $PassportDestination) { throw "Refusing to overwrite feature passports: $PassportDestination" }
New-Item -ItemType Directory -Path (Split-Path -Parent $PassportDestination) -Force | Out-Null
Copy-Item -LiteralPath (Join-Path $FreezeTarget "feature_passports") -Destination $PassportDestination -Recurse -Force

Write-Host ""
Write-Host "QREV v4.0.0 FROZEN SUCCESSFULLY" -ForegroundColor Green
Write-Host "Freeze: $FreezeTarget" -ForegroundColor Cyan
