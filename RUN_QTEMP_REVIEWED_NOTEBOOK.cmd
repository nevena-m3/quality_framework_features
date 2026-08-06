@echo off
setlocal

cd /d "%~dp0"
set "QTEMP_PROJECT_ROOT=%CD%"
set "PYTHONPATH=%CD%\src;%CD%\src reviewed;%PYTHONPATH%"

set "QTEMP_SOURCE=%CD%\notebooks reviewed\06_QTEMP\06_temporal_discontinuity_QTEMP_v1_0_0_REVIEWED_SOURCE.ipynb"
set "QTEMP_LOCAL=%CD%\notebooks reviewed\06_QTEMP\06_temporal_discontinuity_QTEMP_v1_0_0_REVIEWED_LOCAL_EXECUTED.ipynb"

if not exist "%QTEMP_SOURCE%" (
    echo ERROR: QTEMP source notebook was not found:
    echo %QTEMP_SOURCE%
    pause
    exit /b 1
)

where python >nul 2>nul
if errorlevel 1 (
    echo ERROR: Python is not available on PATH.
    pause
    exit /b 1
)

python -c "import jupyterlab" >nul 2>nul
if errorlevel 1 (
    echo ERROR: JupyterLab is not installed for this Python environment.
    echo Install it with: python -m pip install jupyterlab
    pause
    exit /b 1
)

if not exist "%QTEMP_LOCAL%" (
    copy /Y "%QTEMP_SOURCE%" "%QTEMP_LOCAL%" >nul
    if errorlevel 1 (
        echo ERROR: Could not create the local executed notebook copy.
        pause
        exit /b 1
    )
)

echo Project root: %QTEMP_PROJECT_ROOT%
echo Opening: %QTEMP_LOCAL%
echo.

python -m jupyterlab --notebook-dir="%QTEMP_PROJECT_ROOT%" "%QTEMP_LOCAL%"

if errorlevel 1 (
    echo.
    echo ERROR: JupyterLab exited with an error.
    pause
    exit /b 1
)

endlocal
