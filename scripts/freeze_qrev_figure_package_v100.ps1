param(
    [Parameter(Mandatory = $true)]
    [string]$ProjectRoot
)
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProjectRoot=(Resolve-Path -LiteralPath $ProjectRoot).Path
$Python=Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$MeasurementFreeze=Join-Path $ProjectRoot "MAIN outputs\reviewed\06_family_freezes\reverberation\qrev-v4.0.0"
$FigureParent=Join-Path $ProjectRoot "MAIN outputs\reviewed\07_figure_packages\reverberation"
$FigureTarget=Join-Path $FigureParent "qrev-v4.0.0-figures-v1.0.0"
$Workbook=Join-Path $ProjectRoot "notebooks\02_feature_extraction\03_QREV\support/QREV_Family_Evaluation_Workbook_v1_0.docx"
foreach ($Path in @($Python,$MeasurementFreeze,$Workbook)) { if (-not(Test-Path -LiteralPath $Path)){throw "Required QREV figure input missing: $Path"} }
if (Test-Path -LiteralPath $FigureTarget){throw "Refusing to overwrite immutable QREV figure package: $FigureTarget"}
$MeasurementManifest=Join-Path $MeasurementFreeze "manifests\qrev_v400_freeze_manifest.json"
$FigureIndex=Join-Path $MeasurementFreeze "figures\qrev_v400_standardized_figure_index.csv"
$env:QREV_MEASUREMENT_FREEZE=$MeasurementFreeze
$env:QREV_MEASUREMENT_MANIFEST=$MeasurementManifest
$env:QREV_FIGURE_INDEX=$FigureIndex
$Validate=@'
import json,os
from pathlib import Path
import pandas as pd
root=Path(os.environ['QREV_MEASUREMENT_FREEZE'])
manifest=json.loads(Path(os.environ['QREV_MEASUREMENT_MANIFEST']).read_text(encoding='utf-8'))
for key,expected in {'measurement_version':'qrev-v4.0.0','freeze_status':'frozen','scientific_review_decision':'ACCEPT_QREV_V400','required_panels_complete':True}.items():
 if manifest.get(key)!=expected: raise SystemExit(f'Measurement manifest mismatch for {key}')
index=pd.read_csv(Path(os.environ['QREV_FIGURE_INDEX']))
app=index[index.panel!='I']
if len(app)!=22 or set(app.panel)!=set('ABCDEFGHJ'): raise SystemExit('QREV figure index is incomplete')
if index.loc[index.panel=='I','purpose'].item()!='no retained event detector': raise SystemExit('Panel I is not explicit N/A')
for _,row in app.iterrows():
 for col in ['png','svg','pdf','source_csv','caption','provenance']:
  if not (root/str(row[col])).exists(): raise SystemExit(f'Missing figure artifact: {row[col]}')
print('Verified 22 indexed QREV figures plus explicit Panel I N/A.')
'@
$Validate | & $Python -
if($LASTEXITCODE -ne 0){throw "QREV figure-package validation failed."}
New-Item -ItemType Directory -Path $FigureParent -Force | Out-Null
$Staging=Join-Path $FigureParent (".qrev-v4.0.0-figures-v1.0.0.staging."+[guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $Staging -Force | Out-Null
foreach($Folder in @('figures','galleries')){Copy-Item -LiteralPath (Join-Path $MeasurementFreeze $Folder) -Destination (Join-Path $Staging $Folder) -Recurse -Force}
$Validation=Join-Path $Staging 'validation'; New-Item -ItemType Directory -Path $Validation -Force|Out-Null
foreach($Name in @('qrev_v400_ten_domain_dashboard.csv','qrev_v400_gate_summary_final.csv','qrev_v400_g10_feature_decisions.csv','qrev_v400_empirical_feature_summary.csv','qrev_v400_repeated_recording_persistence.csv','qrev_v400_pairwise_redundancy.csv','qrev_v400_recording_vs_participant_weighting.csv','qrev_v400_corrected_horizon_sensitivity.csv')){
 $Source=Join-Path $MeasurementFreeze ("validation\"+$Name); if(Test-Path -LiteralPath $Source){Copy-Item $Source -Destination $Validation -Force}
}
$Prov=Join-Path $Staging 'provenance'; New-Item -ItemType Directory -Path $Prov -Force|Out-Null
Copy-Item -LiteralPath $Workbook -Destination $Prov -Force
foreach($Name in @('support/QREV_v400_FINAL_SCIENTIFIC_AUDIT.md','support/QREV_v400_FINAL_FEATURE_DECISIONS.csv','support/QREV_V4_0_0_FREEZE_CONTRACT.md','support/QREV_Validation_Checklist_v1_0.csv','support/QREV_Ten_Domain_Dashboard_v1_0.csv','support/QREV_V400_FINALIZATION_IMPLEMENTATION_REPORT.md')){
 $Source=Join-Path $ProjectRoot ("notebooks\02_feature_extraction\03_QREV\"+$Name); if(Test-Path -LiteralPath $Source){Copy-Item $Source -Destination $Prov -Force}
}
$Executed=Join-Path $MeasurementFreeze 'provenance\03_finalize_executed.ipynb'; if(Test-Path $Executed){Copy-Item $Executed -Destination $Prov -Force}
$env:QREV_FIGURE_STAGING=$Staging
$env:QREV_WORKBOOK=Join-Path $Prov 'support/QREV_Family_Evaluation_Workbook_v1_0.docx'
$Seal=@'
import hashlib,json,os
from datetime import datetime,timezone
from pathlib import Path
import pandas as pd
root=Path(os.environ['QREV_FIGURE_STAGING']); measurement=Path(os.environ['QREV_MEASUREMENT_FREEZE']); workbook=Path(os.environ['QREV_WORKBOOK'])
mm_path=Path(os.environ['QREV_MEASUREMENT_MANIFEST']); mm=json.loads(mm_path.read_text(encoding='utf-8'))
def sha(p):
 d=hashlib.sha256()
 with p.open('rb') as f:
  for c in iter(lambda:f.read(1024*1024),b''): d.update(c)
 return d.hexdigest()
manifest_dir=root/'manifests'; manifest_dir.mkdir(parents=True,exist_ok=True)
excluded={'manifests/qrev_v400_figure_package_manifest.json','manifests/qrev_v400_figure_package_inventory.csv'}
rows=[]
for p in sorted(x for x in root.rglob('*') if x.is_file()):
 rel=p.relative_to(root).as_posix()
 if rel in excluded: continue
 rows.append({'relative_path':rel,'bytes':p.stat().st_size,'sha256':sha(p)})
inv=pd.DataFrame(rows); inv_path=manifest_dir/'qrev_v400_figure_package_inventory.csv'; inv.to_csv(inv_path,index=False)
index=pd.read_csv(root/'figures/qrev_v400_standardized_figure_index.csv')
fm={'package_version':'qrev-v4.0.0-figures-v1.0.0','measurement_version':'qrev-v4.0.0','freeze_status':'frozen','measurement_freeze_manifest_sha256':sha(mm_path),'measurement_freeze_inventory_sha256':mm['freeze_inventory_sha256'],'measurement_executed_notebook_sha256':mm['executed_notebook_sha256'],'figure_count':int((index.panel!='I').sum()),'required_panels_complete':True,'panel_i_status':'N/A_no_retained_event_detector','workbook_relative_path':workbook.relative_to(root).as_posix(),'workbook_sha256':sha(workbook),'artifact_count_excluding_seal_files':len(inv),'figure_package_inventory_sha256':sha(inv_path),'feature_values_recomputed':False,'horizon_sensitivity_corrected':True,'support_tier_is_precision':False,'family_scalar_constructed':False,'standalone_gate_allowed':False,'created_utc':datetime.now(timezone.utc).isoformat(),'immutability_policy':'never overwrite; create a new package version for any change'}
(manifest_dir/'qrev_v400_figure_package_manifest.json').write_text(json.dumps(fm,indent=2),encoding='utf-8')
(root/'FROZEN_QREV_V4_0_0_FIGURES_V1_0_0.txt').write_text('QREV v4.0.0 figure package v1.0.0 is frozen.\n',encoding='utf-8')
print(json.dumps(fm,indent=2))
'@
$Seal | & $Python -
if($LASTEXITCODE -ne 0){Remove-Item $Staging -Recurse -Force -ErrorAction SilentlyContinue;throw "Failed to seal QREV figure package."}
Move-Item -LiteralPath $Staging -Destination $FigureTarget
$WorkbookRoot=Join-Path $ProjectRoot 'MAIN outputs\reviewed\08_validation_workbooks'; New-Item -ItemType Directory -Path $WorkbookRoot -Force|Out-Null
$WorkbookDestination=Join-Path $WorkbookRoot 'support/QREV_Family_Evaluation_Workbook_v1_0.docx'
if(Test-Path -LiteralPath $WorkbookDestination){throw "Refusing to overwrite validation workbook: $WorkbookDestination"}
Copy-Item -LiteralPath $Workbook -Destination $WorkbookDestination -Force
Write-Host ""
Write-Host "QREV STANDARDIZED VALIDATION AND FIGURE PACKAGE FROZEN SUCCESSFULLY" -ForegroundColor Green
Write-Host "Figure package: $FigureTarget" -ForegroundColor Cyan
Write-Host "Workbook: $WorkbookDestination" -ForegroundColor Cyan
