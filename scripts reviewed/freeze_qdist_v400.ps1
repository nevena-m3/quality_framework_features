param(
    [Parameter(Mandatory = $true)]
    [string]$ProjectRoot
)
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProjectRoot=(Resolve-Path -LiteralPath $ProjectRoot).Path
$Python=Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$FinalCandidate=Join-Path $ProjectRoot "outputs reviewed\nonlinear_distortion\qdist-v4.0.0"
$FinalManifest=Join-Path $FinalCandidate "manifests\qdist_v400_final_candidate_manifest.json"
$ExecutedNotebook=Join-Path $ProjectRoot "notebooks reviewed\05_QDIST\05_nonlinear_distortion_QDIST_v4_0_0_LOCAL_FINALIZE.ipynb"
$FreezeParent=Join-Path $ProjectRoot "MAIN outputs reviewed\06_family_freezes\nonlinear_distortion"
$FreezeTarget=Join-Path $FreezeParent "qdist-v4.0.0"
foreach($Path in @($Python,$FinalCandidate,$FinalManifest,$ExecutedNotebook)){if(-not(Test-Path -LiteralPath $Path)){throw "Required QDIST freeze input missing: $Path"}}
if(Test-Path -LiteralPath $FreezeTarget){throw "Refusing to overwrite immutable QDIST freeze: $FreezeTarget"}
$CanonicalDestinations=@(
 (Join-Path $ProjectRoot 'MAIN outputs reviewed\00_feature_registry\qdist_v400_feature_registry.csv'),
 (Join-Path $ProjectRoot 'MAIN outputs reviewed\01_analysis_features\qdist_v400_analysis_features.csv'),
 (Join-Path $ProjectRoot 'MAIN outputs reviewed\02_support_and_availability\qdist_v400_status_missingness_summary.csv'),
 (Join-Path $ProjectRoot 'MAIN outputs reviewed\02_support_and_availability\qdist_v400_empirical_feature_summary.csv'),
 (Join-Path $ProjectRoot 'MAIN outputs reviewed\04_model_ready_features\qdist_v400_ml_interface.csv'),
 (Join-Path $ProjectRoot 'MAIN outputs reviewed\05_feature_passports\nonlinear_distortion\qdist-v4.0.0')
)
foreach($Destination in $CanonicalDestinations){if(Test-Path -LiteralPath $Destination){throw "Refusing partial freeze because a canonical reviewed destination already exists: $Destination"}}
$env:QDIST_FINAL_CANDIDATE=$FinalCandidate;$env:QDIST_FINAL_MANIFEST=$FinalManifest;$env:QDIST_EXECUTED_NOTEBOOK=$ExecutedNotebook
$Validate=@'
import json,os
from pathlib import Path
import pandas as pd
root=Path(os.environ['QDIST_FINAL_CANDIDATE'])
m=json.loads(Path(os.environ['QDIST_FINAL_MANIFEST']).read_text(encoding='utf-8'))
required={'measurement_version':'qdist-v4.0.0','freeze_status':'ready_for_atomic_freeze','freeze_allowed':True,'scientific_review_decision':'ACCEPT_QDIST_V400','recording_count':519,'participant_count':224,'available_recording_count':519,'positive_recording_count':6,'valid_zero_recording_count':513,'candidate_plateau_count':861,'accepted_plateau_count':30,'episode_count':15,'event_review_item_count':60,'event_review_standardized_png_count':60,'figure_count':23,'main_figure_bundle_count':15,'gallery_bundle_count':8,'required_panels_complete':True,'panel_i_status':'APPLICABLE_complete_event_verification','numerical_equivalence_to_cohort_candidate':True,'numerical_equivalence_to_qdist_v311':True,'feature_values_recomputed':False,'family_scalar_constructed':False,'standalone_gate_allowed':False,'complete_nonlinear_distortion_claim_allowed':False,'missing_values_imputed':False}
for k,e in required.items():
 if m.get(k)!=e: raise SystemExit(f'Manifest mismatch {k}: {m.get(k)!r}')
nb=json.loads(Path(os.environ['QDIST_EXECUTED_NOTEBOOK']).read_text(encoding='utf-8'))
code=[c for c in nb.get('cells',[]) if c.get('cell_type')=='code']
if len(code)!=6 or any(c.get('execution_count') is None for c in code): raise SystemExit('Finalization notebook must contain six executed code cells')
errors=[o for c in code for o in c.get('outputs',[]) if o.get('output_type')=='error']
if errors: raise SystemExit(f'Finalization notebook contains errors: {errors}')
text='\n'.join(''.join(o.get('text',[])) if isinstance(o.get('text'),list) else str(o.get('text','')) for c in code for o in c.get('outputs',[]))
if 'QDIST v4.0.0 FINALIZATION COMPLETE' not in text: raise SystemExit('Completion marker not saved')
features=pd.read_csv(root/'tables/qdist_v400_recording_features.csv')
if len(features)!=519 or int(features.qdist_positive.astype(bool).sum())!=6: raise SystemExit('Final feature table count mismatch')
index=pd.read_csv(root/'figures/qdist_v400_standardized_figure_index.csv')
if len(index)!=23 or int((index.panel=='G').sum())!=8 or int((index.panel=='I').sum())!=1: raise SystemExit('Figure index mismatch')
review=pd.read_csv(root/'tables/qdist_v400_event_review_index.csv')
if len(review)!=60: raise SystemExit('Event-review count mismatch')
pngs=list((root/'event_review').glob('qdist_review_*.png'))
standard=[p for p in pngs if not p.name.endswith('.legacy.png')]
if len(standard)!=60: raise SystemExit(f'Standardized event PNG count mismatch: {len(standard)}')
print('QDIST final candidate and executed notebook authorize atomic freeze.')
'@
$Validate | & $Python -
if($LASTEXITCODE -ne 0){throw "QDIST freeze validation failed."}
New-Item -ItemType Directory -Path $FreezeParent -Force|Out-Null
$Staging=Join-Path $FreezeParent (".qdist-v4.0.0.staging."+[guid]::NewGuid().ToString('N'))
Copy-Item -LiteralPath $FinalCandidate -Destination $Staging -Recurse -Force
$Prov=Join-Path $Staging 'provenance';New-Item -ItemType Directory -Path $Prov -Force|Out-Null
Copy-Item -LiteralPath $ExecutedNotebook -Destination (Join-Path $Prov '05_nonlinear_distortion_QDIST_v4_0_0_EXECUTED_FINAL.ipynb') -Force
$env:QDIST_FREEZE_STAGING=$Staging
$Seal=@'
import hashlib,json,os
from datetime import datetime,timezone
from pathlib import Path
import pandas as pd
root=Path(os.environ['QDIST_FREEZE_STAGING'])
final=json.loads((root/'manifests/qdist_v400_final_candidate_manifest.json').read_text(encoding='utf-8'))
def sha(p):
 d=hashlib.sha256()
 with p.open('rb') as f:
  for c in iter(lambda:f.read(1024*1024),b''):d.update(c)
 return d.hexdigest()
excluded={'manifests/qdist_v400_freeze_manifest.json','manifests/qdist_v400_freeze_inventory.csv','FROZEN_QDIST_V4_0_0.txt'}
rows=[]
for p in sorted(x for x in root.rglob('*') if x.is_file()):
 rel=p.relative_to(root).as_posix()
 if rel not in excluded:rows.append({'relative_path':rel,'bytes':p.stat().st_size,'sha256':sha(p)})
inv=pd.DataFrame(rows);ip=root/'manifests/qdist_v400_freeze_inventory.csv';inv.to_csv(ip,index=False)
nb=root/'provenance/05_nonlinear_distortion_QDIST_v4_0_0_EXECUTED_FINAL.ipynb'
freeze=dict(final);freeze.update({'freeze_status':'frozen','freeze_allowed':False,'frozen_utc':datetime.now(timezone.utc).isoformat(),'executed_notebook_relative_path':nb.relative_to(root).as_posix(),'executed_notebook_sha256':sha(nb),'artifact_count_excluding_seal_files':len(inv),'freeze_inventory_sha256':sha(ip)})
(root/'manifests/qdist_v400_freeze_manifest.json').write_text(json.dumps(freeze,indent=2),encoding='utf-8')
(root/'FROZEN_QDIST_V4_0_0.txt').write_text('QDIST v4.0.0 is frozen. Never overwrite.\n',encoding='utf-8')
print(json.dumps(freeze,indent=2))
'@
$Seal | & $Python -
if($LASTEXITCODE -ne 0){Remove-Item $Staging -Recurse -Force -ErrorAction SilentlyContinue;throw "Failed to seal QDIST freeze."}
Move-Item -LiteralPath $Staging -Destination $FreezeTarget
$Publications=@(
 @{Source='tables\qdist_v400_feature_registry.csv';Destination='00_feature_registry\qdist_v400_feature_registry.csv'},
 @{Source='tables\qdist_v400_analysis_features.csv';Destination='01_analysis_features\qdist_v400_analysis_features.csv'},
 @{Source='validation\qdist_v400_status_missingness_summary.csv';Destination='02_support_and_availability\qdist_v400_status_missingness_summary.csv'},
 @{Source='validation\qdist_v400_empirical_feature_summary.csv';Destination='02_support_and_availability\qdist_v400_empirical_feature_summary.csv'},
 @{Source='tables\qdist_v400_ml_interface.csv';Destination='04_model_ready_features\qdist_v400_ml_interface.csv'}
)
foreach($Item in $Publications){$Source=Join-Path $FreezeTarget $Item.Source;$Destination=Join-Path (Join-Path $ProjectRoot 'MAIN outputs reviewed') $Item.Destination;if(Test-Path -LiteralPath $Destination){throw "Refusing to overwrite canonical reviewed output: $Destination"};New-Item -ItemType Directory -Path (Split-Path -Parent $Destination) -Force|Out-Null;Copy-Item $Source -Destination $Destination -Force}
$PassportDestination=Join-Path $ProjectRoot 'MAIN outputs reviewed\05_feature_passports\nonlinear_distortion\qdist-v4.0.0'
if(Test-Path -LiteralPath $PassportDestination){throw "Refusing to overwrite feature passports: $PassportDestination"}
New-Item -ItemType Directory -Path (Split-Path -Parent $PassportDestination) -Force|Out-Null
Copy-Item -LiteralPath (Join-Path $FreezeTarget 'feature_passports') -Destination $PassportDestination -Recurse -Force
Write-Host "";Write-Host "QDIST v4.0.0 FROZEN SUCCESSFULLY" -ForegroundColor Green;Write-Host "Freeze: $FreezeTarget" -ForegroundColor Cyan
