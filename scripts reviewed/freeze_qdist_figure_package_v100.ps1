param(
 [Parameter(Mandatory=$true)][string]$ProjectRoot
)
Set-StrictMode -Version Latest
$ErrorActionPreference='Stop'
$ProjectRoot=(Resolve-Path -LiteralPath $ProjectRoot).Path
$Python=Join-Path $ProjectRoot '.venv\Scripts\python.exe'
$MeasurementFreeze=Join-Path $ProjectRoot 'MAIN outputs reviewed\06_family_freezes\nonlinear_distortion\qdist-v4.0.0'
$FigureParent=Join-Path $ProjectRoot 'MAIN outputs reviewed\07_figure_packages\nonlinear_distortion'
$FigureTarget=Join-Path $FigureParent 'qdist-v4.0.0-figures-v1.0.0'
$Workbook=Join-Path $ProjectRoot 'notebooks reviewed\05_QDIST\QDIST_Family_Evaluation_Workbook_v1_0.docx'
$WorkbookRoot=Join-Path $ProjectRoot 'MAIN outputs reviewed\08_validation_workbooks'
$WorkbookDestination=Join-Path $WorkbookRoot 'QDIST_Family_Evaluation_Workbook_v1_0.docx'
foreach($Path in @($Python,$MeasurementFreeze,$Workbook)){if(-not(Test-Path -LiteralPath $Path)){throw "Required QDIST figure input missing: $Path"}}
if(Test-Path -LiteralPath $FigureTarget){throw "Refusing to overwrite immutable QDIST figure package: $FigureTarget"}
if(Test-Path -LiteralPath $WorkbookDestination){throw "Refusing partial figure freeze because the validation workbook already exists: $WorkbookDestination"}
$MeasurementManifest=Join-Path $MeasurementFreeze 'manifests\qdist_v400_freeze_manifest.json'
$FigureIndex=Join-Path $MeasurementFreeze 'figures\qdist_v400_standardized_figure_index.csv'
$env:QDIST_MEASUREMENT_FREEZE=$MeasurementFreeze;$env:QDIST_MEASUREMENT_MANIFEST=$MeasurementManifest;$env:QDIST_FIGURE_INDEX=$FigureIndex
$Validate=@'
import json,os
from pathlib import Path
import pandas as pd
root=Path(os.environ['QDIST_MEASUREMENT_FREEZE']);m=json.loads(Path(os.environ['QDIST_MEASUREMENT_MANIFEST']).read_text(encoding='utf-8'))
for k,e in {'measurement_version':'qdist-v4.0.0','freeze_status':'frozen','scientific_review_decision':'ACCEPT_QDIST_V400','required_panels_complete':True,'feature_values_recomputed':False}.items():
 if m.get(k)!=e:raise SystemExit(f'Measurement manifest mismatch {k}: {m.get(k)!r}')
index=pd.read_csv(Path(os.environ['QDIST_FIGURE_INDEX']))
expected={'A','B','C','D1','D2','D3','E1','E2','E3','F','G','H1','H2','H3','I','J'}
if len(index)!=23 or set(index.panel)!=expected or int((index.panel=='G').sum())!=8 or int((index.panel=='I').sum())!=1:raise SystemExit('QDIST figure index incomplete')
for _,row in index.iterrows():
 for col in ['png','svg','pdf','source_csv','caption','provenance']:
  p=root/str(row[col])
  if not p.exists() or p.stat().st_size==0:raise SystemExit(f'Missing figure artifact {row[col]}')
review=pd.read_csv(root/'tables/qdist_v400_event_review_index.csv')
required={'waveform','pcm_derivative','amplitude_distribution','spectrogram','audio_excerpt'}
if len(review)!=60:raise SystemExit('Event-review index count mismatch')
for row in review.itertuples(index=False):
 stem=f'qdist_review_{row.review_item_id}'
 for ext in ['.png','.source.csv','.wav','.caption.md','.provenance.json']:
  p=root/'event_review'/f'{stem}{ext}'
  if not p.exists() or p.stat().st_size==0:raise SystemExit(f'Missing event-review artifact {p.name}')
 source=pd.read_csv(root/'event_review'/f'{stem}.source.csv')
 if not required.issubset(set(source.view.astype(str))):raise SystemExit(f'Five-view contract failed {stem}')
print('Verified 23 QDIST figure bundles and 60 complete event-review records.')
'@
$Validate | & $Python -
if($LASTEXITCODE -ne 0){throw 'QDIST figure-package validation failed.'}
New-Item -ItemType Directory -Path $FigureParent -Force|Out-Null
$Staging=Join-Path $FigureParent ('.qdist-v4.0.0-figures-v1.0.0.staging.'+[guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $Staging -Force|Out-Null
foreach($Folder in @('figures','event_review')){Copy-Item -LiteralPath (Join-Path $MeasurementFreeze $Folder) -Destination (Join-Path $Staging $Folder) -Recurse -Force}
$Validation=Join-Path $Staging 'validation';New-Item -ItemType Directory -Path $Validation -Force|Out-Null
foreach($Name in @('qdist_v400_ten_domain_dashboard.csv','qdist_v400_gate_summary_final.csv','qdist_v400_g10_feature_decisions.csv','qdist_v400_empirical_feature_summary.csv','qdist_v400_parameter_sensitivity_summary.csv','qdist_v400_merge_gap_sensitivity_summary.csv','qdist_v400_deletion_influence_summary.csv','qdist_v400_repeated_recording_summary.csv','qdist_v400_related_view_redundancy.csv','qdist_v400_weighting_summary.csv','qdist_v400_event_adjudication_summary.csv')){$Source=Join-Path $MeasurementFreeze ('validation\'+$Name);if(Test-Path $Source){Copy-Item $Source -Destination $Validation -Force}}
$Prov=Join-Path $Staging 'provenance';New-Item -ItemType Directory -Path $Prov -Force|Out-Null
Copy-Item $Workbook -Destination $Prov -Force
foreach($Name in @('QDIST_v400_FINAL_SCIENTIFIC_AUDIT.md','QDIST_v400_FINAL_FEATURE_DECISIONS.csv','QDIST_V4_0_0_FREEZE_CONTRACT.md','QDIST_Validation_Checklist_v1_0.csv','QDIST_Ten_Domain_Dashboard_v1_0.csv','QDIST_Gate_Summary_FINAL_v1_0.csv','QDIST_V400_FINALIZATION_IMPLEMENTATION_REPORT.md')){$Source=Join-Path $ProjectRoot ('notebooks reviewed\05_QDIST\'+$Name);if(Test-Path $Source){Copy-Item $Source -Destination $Prov -Force}}
$Executed=Join-Path $MeasurementFreeze 'provenance\05_nonlinear_distortion_QDIST_v4_0_0_EXECUTED_FINAL.ipynb';if(Test-Path $Executed){Copy-Item $Executed -Destination $Prov -Force}
$env:QDIST_FIGURE_STAGING=$Staging;$env:QDIST_WORKBOOK=Join-Path $Prov 'QDIST_Family_Evaluation_Workbook_v1_0.docx'
$Seal=@'
import hashlib,json,os
from datetime import datetime,timezone
from pathlib import Path
import pandas as pd
root=Path(os.environ['QDIST_FIGURE_STAGING']);workbook=Path(os.environ['QDIST_WORKBOOK']);mm_path=Path(os.environ['QDIST_MEASUREMENT_MANIFEST']);mm=json.loads(mm_path.read_text(encoding='utf-8'))
def sha(p):
 d=hashlib.sha256()
 with p.open('rb') as f:
  for c in iter(lambda:f.read(1024*1024),b''):d.update(c)
 return d.hexdigest()
md=root/'manifests';md.mkdir(parents=True,exist_ok=True)
excluded={'manifests/qdist_v400_figure_package_manifest.json','manifests/qdist_v400_figure_package_inventory.csv','FROZEN_QDIST_V4_0_0_FIGURES_V1_0_0.txt'}
rows=[]
for p in sorted(x for x in root.rglob('*') if x.is_file()):
 rel=p.relative_to(root).as_posix()
 if rel not in excluded:rows.append({'relative_path':rel,'bytes':p.stat().st_size,'sha256':sha(p)})
inv=pd.DataFrame(rows);ip=md/'qdist_v400_figure_package_inventory.csv';inv.to_csv(ip,index=False)
index=pd.read_csv(root/'figures/qdist_v400_standardized_figure_index.csv')
fm={'package_version':'qdist-v4.0.0-figures-v1.0.0','measurement_version':'qdist-v4.0.0','freeze_status':'frozen','measurement_freeze_manifest_sha256':sha(mm_path),'measurement_freeze_inventory_sha256':mm['freeze_inventory_sha256'],'measurement_executed_notebook_sha256':mm['executed_notebook_sha256'],'figure_count':len(index),'main_figure_bundle_count':int((index.panel!='G').sum()),'gallery_bundle_count':int((index.panel=='G').sum()),'panel_i_status':'APPLICABLE_complete_event_verification','event_review_item_count':60,'event_review_adjudication_type':'AI-assisted blinded morphology review','event_review_independent_human_ground_truth':False,'workbook_relative_path':workbook.relative_to(root).as_posix(),'workbook_sha256':sha(workbook),'artifact_count_excluding_seal_files':len(inv),'figure_package_inventory_sha256':sha(ip),'feature_values_recomputed':False,'family_scalar_constructed':False,'standalone_gate_allowed':False,'complete_nonlinear_distortion_claim_allowed':False,'created_utc':datetime.now(timezone.utc).isoformat(),'immutability_policy':'never overwrite; create a new package version for any change'}
(md/'qdist_v400_figure_package_manifest.json').write_text(json.dumps(fm,indent=2),encoding='utf-8')
(root/'FROZEN_QDIST_V4_0_0_FIGURES_V1_0_0.txt').write_text('QDIST v4.0.0 figure package v1.0.0 is frozen.\n',encoding='utf-8')
print(json.dumps(fm,indent=2))
'@
$Seal | & $Python -
if($LASTEXITCODE -ne 0){Remove-Item $Staging -Recurse -Force -ErrorAction SilentlyContinue;throw 'Failed to seal QDIST figure package.'}
Move-Item $Staging -Destination $FigureTarget
$WorkbookRoot=Join-Path $ProjectRoot 'MAIN outputs reviewed\08_validation_workbooks';New-Item -ItemType Directory -Path $WorkbookRoot -Force|Out-Null
$WorkbookDestination=Join-Path $WorkbookRoot 'QDIST_Family_Evaluation_Workbook_v1_0.docx'
if(Test-Path $WorkbookDestination){throw "Refusing to overwrite validation workbook: $WorkbookDestination"}
Copy-Item $Workbook -Destination $WorkbookDestination -Force
Write-Host "";Write-Host 'QDIST STANDARDIZED VALIDATION AND FIGURE PACKAGE FROZEN SUCCESSFULLY' -ForegroundColor Green;Write-Host "Figure package: $FigureTarget" -ForegroundColor Cyan;Write-Host "Workbook: $WorkbookDestination" -ForegroundColor Cyan
