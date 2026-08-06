@echo off
setlocal
cd /d "%~dp0"

set "COLLECTOR=%CD%\scripts reviewed\collect_qtemp_freeze_evidence.ps1"

if not exist "%COLLECTOR%" (
    echo ERROR: QTEMP evidence collector was not found:
    echo %COLLECTOR%
    pause
    exit /b 1
)

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%COLLECTOR%" -ProjectRoot "%CD%"

if errorlevel 1 (
    echo.
    echo ERROR: QTEMP evidence collection failed.
    pause
    exit /b 1
)

echo.
pause
endlocal

