param(
    [Parameter(Mandatory = $true)]
    [string]$ProjectRoot
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path -LiteralPath $ProjectRoot).Path
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$FinalCandidate = Join-Path $ProjectRoot "outputs reviewed\channel_device\qchan-v4.0.0"
$FinalManifest = Join-Path $FinalCandidate "manifests\qchan_v400_final_candidate_manifest.json"
$ExecutedNotebook = Join-Path $ProjectRoot "notebooks reviewed\04_QCHAN\04_channel_device_QCHAN_v4_0_0_LOCAL_FINALIZE.ipynb"
$FreezeParent = Join-Path $ProjectRoot "MAIN outputs reviewed\06_family_freezes\channel_device"
$FreezeTarget = Join-Path $FreezeParent "qchan-v4.0.0"

foreach ($Path in @($Python, $FinalCandidate, $FinalManifest, $ExecutedNotebook)) {
    if (-not (Test-Path -LiteralPath $Path)) { throw "Required QCHAN freeze input is missing: $Path" }
}
if (Test-Path -LiteralPath $FreezeTarget) { throw "Refusing to overwrite immutable QCHAN freeze: $FreezeTarget" }

$env:QCHAN_FINAL_CANDIDATE = $FinalCandidate
$env:QCHAN_FINAL_MANIFEST = $FinalManifest
$env:QCHAN_EXECUTED_NOTEBOOK = $ExecutedNotebook

$Validate = @'
import json, os
from pathlib import Path
import pandas as pd
root=Path(os.environ['QCHAN_FINAL_CANDIDATE'])
manifest=json.loads(Path(os.environ['QCHAN_FINAL_MANIFEST']).read_text(encoding='utf-8'))
required={
 'measurement_version':'qchan-v4.0.0','freeze_allowed':True,
 'freeze_status':'ready_for_atomic_freeze','scientific_review_decision':'ACCEPT_QCHAN_V400',
 'recording_count':519,'participant_count':224,'all_features_available_n':519,
 'reference_ledger_count':519,'reference_vintage_count':1,
 'numerical_equivalence_to_cohort_candidate':True,'required_panels_complete':True,
 'figure_count_excluding_na':22,'panel_i_status':'N/A_no_retained_event_detector',
 'feature_values_recomputed':False,'standalone_reject_allowed':False,
 'device_identity_estimated':False,'support_tier_is_precision':False,
 'reference_vintage_is_feature_identity':True,'signed_precursors_retained':True,
}
for key,expected in required.items():
 if manifest.get(key)!=expected: raise SystemExit(f'Manifest mismatch for {key}: {manifest.get(key)!r}')
nb=json.loads(Path(os.environ['QCHAN_EXECUTED_NOTEBOOK']).read_text(encoding='utf-8'))
code=[c for c in nb.get('cells',[]) if c.get('cell_type')=='code']
if any(c.get('execution_count') is None for c in code): raise SystemExit('Finalization notebook has unexecuted cells')
errors=[o for c in code for o in c.get('outputs',[]) if o.get('output_type')=='error']
if errors: raise SystemExit(f'Finalization notebook contains errors: {errors}')
index=pd.read_csv(root/'figures/qchan_v400_standardized_figure_index.csv')
app=index[index.panel!='I']
if len(app)!=22 or set(app.panel)!=set(['A','B','C','D1','D2','D3','E1','E2','E3','F','G','H1','H2','H3','J']): raise SystemExit('Figure contract is incomplete')
if index.loc[index.panel=='I','selection_reason'].item()!='no retained event detector': raise SystemExit('Panel I is not explicit N/A')
features=pd.read_csv(root/'tables/qchan_v400_analysis_features.csv')
if len(features)!=519: raise SystemExit('Final analysis table count is not 519')
print('QCHAN final candidate and executed notebook authorize atomic freeze.')
'@
$Validate | & $Python -
if ($LASTEXITCODE -ne 0) { throw "QCHAN freeze validation failed." }

New-Item -ItemType Directory -Path $FreezeParent -Force | Out-Null
$Staging = Join-Path $FreezeParent (".qchan-v4.0.0.staging." + [guid]::NewGuid().ToString("N"))
Copy-Item -LiteralPath $FinalCandidate -Destination $Staging -Recurse -Force
$Provenance=Join-Path $Staging "provenance"; New-Item -ItemType Directory -Path $Provenance -Force|Out-Null
Copy-Item -LiteralPath $ExecutedNotebook -Destination (Join-Path $Provenance "04_channel_device_QCHAN_v4_0_0_EXECUTED_FINAL.ipynb") -Force

$env:QCHAN_FREEZE_STAGING=$Staging
$Seal=@'
import hashlib,json,os
from datetime import datetime,timezone
from pathlib import Path
import pandas as pd
root=Path(os.environ['QCHAN_FREEZE_STAGING'])
final_manifest_path=root/'manifests/qchan_v400_final_candidate_manifest.json'
final=json.loads(final_manifest_path.read_text(encoding='utf-8'))
def sha(p):
 d=hashlib.sha256()
 with p.open('rb') as f:
  for c in iter(lambda:f.read(1024*1024),b''): d.update(c)
 return d.hexdigest()
excluded={'manifests/qchan_v400_freeze_manifest.json','manifests/qchan_v400_freeze_inventory.csv','FROZEN_QCHAN_V4_0_0.txt'}
rows=[]
for p in sorted(x for x in root.rglob('*') if x.is_file()):
 rel=p.relative_to(root).as_posix()
 if rel in excluded: continue
 rows.append({'relative_path':rel,'bytes':p.stat().st_size,'sha256':sha(p)})
inv=pd.DataFrame(rows); inv_path=root/'manifests/qchan_v400_freeze_inventory.csv'; inv.to_csv(inv_path,index=False)
notebook=root/'provenance/04_channel_device_QCHAN_v4_0_0_EXECUTED_FINAL.ipynb'
freeze=dict(final)
freeze.update({
 'freeze_status':'frozen','freeze_allowed':False,
 'frozen_utc':datetime.now(timezone.utc).isoformat(),
 'executed_notebook_relative_path':notebook.relative_to(root).as_posix(),
 'executed_notebook_sha256':sha(notebook),
 'artifact_count_excluding_seal_files':len(inv),
 'freeze_inventory_sha256':sha(inv_path),
})
(root/'manifests/qchan_v400_freeze_manifest.json').write_text(json.dumps(freeze,indent=2),encoding='utf-8')
(root/'FROZEN_QCHAN_V4_0_0.txt').write_text('QCHAN v4.0.0 is frozen. Never overwrite.\n',encoding='utf-8')
print(json.dumps(freeze,indent=2))
'@
$Seal | & $Python -
if ($LASTEXITCODE -ne 0) { Remove-Item $Staging -Recurse -Force -ErrorAction SilentlyContinue; throw "Failed to seal QCHAN freeze." }
Move-Item -LiteralPath $Staging -Destination $FreezeTarget

$Publications = @(
    @{Source="tables\qchan_v400_feature_registry.csv"; Destination="00_feature_registry\qchan_v400_feature_registry.csv"},
    @{Source="tables\qchan_v400_analysis_features.csv"; Destination="01_analysis_features\qchan_v400_analysis_features.csv"},
    @{Source="validation\qchan_v400_status_missingness_summary.csv"; Destination="02_support_and_availability\qchan_v400_status_missingness_summary.csv"},
    @{Source="validation\qchan_v400_support_availability.csv"; Destination="02_support_and_availability\qchan_v400_support_availability.csv"},
    @{Source="validation\qchan_v400_native_bandwidth_summary.csv"; Destination="02_support_and_availability\qchan_v400_native_bandwidth_summary.csv"},
    @{Source="tables\qchan_v400_model_ready_features.csv"; Destination="04_model_ready_features\qchan_v400_model_ready_features.csv"}
)
foreach ($Item in $Publications) {
    $Source=Join-Path $FreezeTarget $Item.Source
    $Destination=Join-Path (Join-Path $ProjectRoot "MAIN outputs reviewed") $Item.Destination
    if (Test-Path -LiteralPath $Destination) { throw "Refusing to overwrite canonical reviewed output: $Destination" }
    New-Item -ItemType Directory -Path (Split-Path -Parent $Destination) -Force | Out-Null
    Copy-Item -LiteralPath $Source -Destination $Destination -Force
}
$PassportDestination=Join-Path $ProjectRoot "MAIN outputs reviewed\05_feature_passports\channel_device\qchan-v4.0.0"
if (Test-Path -LiteralPath $PassportDestination) { throw "Refusing to overwrite feature passports: $PassportDestination" }
New-Item -ItemType Directory -Path (Split-Path -Parent $PassportDestination) -Force | Out-Null
Copy-Item -LiteralPath (Join-Path $FreezeTarget "feature_passports") -Destination $PassportDestination -Recurse -Force

Write-Host ""
Write-Host "QCHAN v4.0.0 FROZEN SUCCESSFULLY" -ForegroundColor Green
Write-Host "Freeze: $FreezeTarget" -ForegroundColor Cyan
