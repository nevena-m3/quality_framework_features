param(
    [Parameter(Mandatory = $true)]
    [string]$ProjectRoot,

    [Parameter(Mandatory = $true)]
    [string]$ExecutedNotebook
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path -LiteralPath $ProjectRoot).Path
$ExecutedNotebook = (Resolve-Path -LiteralPath $ExecutedNotebook).Path
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    $Python = (Get-Command python -ErrorAction Stop).Source
}

$Required = @(
    $ExecutedNotebook,
    (Join-Path $ProjectRoot "src\paper1_qc\qdist_v410_candidate.py"),
    (Join-Path $ProjectRoot "src\paper1_qc_reviewed\qdist_v410_cohort.py"),
    (Join-Path $ProjectRoot "src\paper1_qc_reviewed\qdist_v410_computational_verification.py"),
    (Join-Path $ProjectRoot "src\paper1_qc_reviewed\qdist_v410_freeze_readiness.py"),
    (Join-Path $ProjectRoot "tests\test_qdist_v410_candidate.py"),
    (Join-Path $ProjectRoot "tests\test_qdist_v410_computational_verification.py"),
    (Join-Path $ProjectRoot "tests\test_qdist_v410_freeze_readiness.py"),
    (Join-Path $ProjectRoot "notebooks\02_feature_extraction\05_QDIST\support/QDIST_V4_1_0_MEASUREMENT_FREEZE_CONTRACT.md"),
    (Join-Path $ProjectRoot "notebooks\02_feature_extraction\05_QDIST\support/QDIST_v410_AUTOMATED_FREEZE_PROTOCOL.md")
)
foreach ($Path in $Required) {
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "Required QDIST freeze input is missing: $Path"
    }
}

$PreviousPythonPath = $env:PYTHONPATH
$env:PYTHONPATH = (
    (Join-Path $ProjectRoot "src") + ";" +
    (Join-Path $ProjectRoot "src")
)
if ($PreviousPythonPath) {
    $env:PYTHONPATH += ";$PreviousPythonPath"
}

Push-Location $ProjectRoot
try {
    & $Python `
        -m pytest `
        "tests\test_qdist_v410_candidate.py" `
        "tests\test_qdist_v410_computational_verification.py" `
        "tests\test_qdist_v410_freeze_readiness.py" `
        -q `
        --disable-warnings
    if ($LASTEXITCODE -ne 0) {
        throw "QDIST governed tests failed immediately before freeze."
    }

    & $Python `
        -m paper1_qc_reviewed.qdist_v410_freeze_readiness `
        seal `
        --project-root $ProjectRoot `
        --executed-notebook $ExecutedNotebook
    if ($LASTEXITCODE -ne 0) {
        throw "QDIST v4.1.0 atomic measurement freeze failed."
    }
}
finally {
    Pop-Location
    $env:PYTHONPATH = $PreviousPythonPath
}

Write-Host ""
Write-Host "QDIST v4.1.0 MEASUREMENT FROZEN SUCCESSFULLY" -ForegroundColor Green
Write-Host (
    "Freeze: " +
    (Join-Path $ProjectRoot `
        "MAIN outputs\02_FEATURE_REVIEWED\06_family_freezes\nonlinear_distortion\qdist-v4.1.0")
) -ForegroundColor Cyan
Write-Host "No manual review or reviewer labels were used."
Write-Host "Manuscript and joint-family integration remain explicitly pending."

