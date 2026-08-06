@echo off
setlocal EnableExtensions

set "PROJECT_ROOT=%~dp0"
set "NOTEBOOK_DIR=%PROJECT_ROOT%notebooks\02_feature_extraction\06_QTEMP"
set "SOURCE_NOTEBOOK=%NOTEBOOK_DIR%\06_temporal_discontinuity_QTEMP_v1_0_0_FINAL_ANALYTICAL_DISPOSITION_SOURCE.ipynb"
set "EXECUTED_NAME=06_temporal_discontinuity_QTEMP_v1_0_0_FINAL_ANALYTICAL_DISPOSITION_EXECUTED.ipynb"
set "EXECUTED_NOTEBOOK=%NOTEBOOK_DIR%\%EXECUTED_NAME%"
set "CANDIDATE_ROOT=%PROJECT_ROOT%outputs\reviewed\06_QTEMP\qtemp-v1.0.0-candidate-g9-pending"
set "FINAL_ROOT=%PROJECT_ROOT%outputs\reviewed\06_QTEMP\qtemp-v1.0.0-analytical-final-no-retained"

if not exist "%PROJECT_ROOT%config\project.yaml" (
    echo ERROR: config\project.yaml was not found under:
    echo %PROJECT_ROOT%
    exit /b 1
)

if not exist "%SOURCE_NOTEBOOK%" (
    echo ERROR: The final QTEMP source notebook is missing:
    echo %SOURCE_NOTEBOOK%
    exit /b 1
)

if not exist "%CANDIDATE_ROOT%\manifests\qtemp_v100_artifact_sha256.csv" (
    echo ERROR: The completed QTEMP validation evidence is missing:
    echo %CANDIDATE_ROOT%
    exit /b 1
)

where python >nul 2>&1
if not errorlevel 1 (
    set "PYTHON_CMD=python"
) else (
    where py >nul 2>&1
    if errorlevel 1 (
        echo ERROR: Python was not found on PATH.
        exit /b 1
    )
    set "PYTHON_CMD=py -3"
)

set "PYTHONPATH=%PROJECT_ROOT%src;%PROJECT_ROOT%src;%PYTHONPATH%"

echo.
echo [1/4] Checking the Python environment...
%PYTHON_CMD% -c "import numpy, pandas, matplotlib, scipy, IPython"
if errorlevel 1 (
    echo ERROR: Required Python packages are missing from this environment.
    exit /b 1
)

%PYTHON_CMD% -m jupyter nbconvert --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Jupyter nbconvert is unavailable in this Python environment.
    exit /b 1
)

echo.
echo [2/4] Executing the final QTEMP notebook from beginning to end...
%PYTHON_CMD% -m jupyter nbconvert ^
    --to notebook ^
    --execute "%SOURCE_NOTEBOOK%" ^
    --output "%EXECUTED_NAME%" ^
    --output-dir "%NOTEBOOK_DIR%" ^
    --ExecutePreprocessor.timeout=-1
if errorlevel 1 (
    echo ERROR: QTEMP finalization failed. The final archive was not accepted.
    exit /b 1
)

echo.
echo [3/4] Verifying the final status and immutable manifest...
%PYTHON_CMD% -c "from pathlib import Path; import csv,hashlib,json; r=Path(r'%FINAL_ROOT%'); s=json.loads((r/'manifests/qtemp_v100_final_status.json').read_text(encoding='utf-8')); assert s['finalization_state']=='FINAL_ANALYTICAL_IMPLEMENTATION_FREEZE_NO_RETAINED_PRIMARY_FEATURES'; assert s['validated_primary_features']==[]; assert s['g9_status']=='N/A_NO_RETAINED_PRIMARY_EVENT_FEATURES'; m=r/'manifests/qtemp_v100_final_artifact_sha256.csv'; rows=list(csv.DictReader(m.open(encoding='utf-8'))); assert rows; bad=[x['relative_path'] for x in rows if hashlib.sha256((r/x['relative_path']).read_bytes()).hexdigest()!=x['sha256']]; assert not bad, bad; print('Verified final artifacts:',len(rows))"
if errorlevel 1 (
    echo ERROR: Final status or manifest verification failed.
    exit /b 1
)

echo.
echo [4/4] QTEMP finalization completed successfully.
echo Final archive:
echo %FINAL_ROOT%
echo Executed notebook:
echo %EXECUTED_NOTEBOOK%
echo.
echo Opening the executed notebook now...

%PYTHON_CMD% -m jupyter lab --version >nul 2>&1
if not errorlevel 1 (
    start "" %PYTHON_CMD% -m jupyter lab "%EXECUTED_NOTEBOOK%"
) else (
    start "" %PYTHON_CMD% -m jupyter notebook "%EXECUTED_NOTEBOOK%"
)

exit /b 0
