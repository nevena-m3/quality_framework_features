param(
    [Parameter(Mandatory=$true)]
    [string]$ProjectRoot,
    [Parameter(Mandatory=$true)]
    [string]$PatchZip
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ExpectedInventorySha = "6f6c579a37a5ca7e53ac429334b93d942f0ad8dc99c0b8134a57fdd1badf88cc"
$FreezeManifest = Join-Path $ProjectRoot "MAIN outputs reviewed\06_family_freezes\gain_dynamics\qgain-v4.1.0\manifests\qgain_v410_freeze_manifest.json"
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

foreach ($Path in @($ProjectRoot, $PatchZip, $FreezeManifest, $Python)) {
    if (-not (Test-Path -LiteralPath $Path)) { throw "Required path is missing: $Path" }
}
$Manifest = Get-Content -LiteralPath $FreezeManifest -Raw | ConvertFrom-Json
if ($Manifest.measurement_version -ne "qgain-v4.1.0" -or $Manifest.freeze_status -ne "frozen") {
    throw "The required immutable qgain-v4.1.0 source freeze is not present."
}
if ($Manifest.freeze_inventory_sha256 -ne $ExpectedInventorySha) {
    throw "The local qgain-v4.1.0 freeze inventory does not match the audited freeze. Refusing installation."
}

$Temp = Join-Path $env:TEMP ("qgain_figure_completion_" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $Temp -Force | Out-Null
try {
    Expand-Archive -LiteralPath $PatchZip -DestinationPath $Temp -Force
    $Root = Get-ChildItem -LiteralPath $Temp -Directory | Select-Object -First 1
    if ($Root -and (Test-Path -LiteralPath (Join-Path $Root.FullName "src reviewed"))) {
        $PatchRoot = $Root.FullName
    } else {
        $PatchRoot = $Temp
    }
    foreach ($Folder in @("src reviewed", "notebooks reviewed", "tests reviewed", "scripts reviewed", "docs reviewed")) {
        $Source = Join-Path $PatchRoot $Folder
        if (-not (Test-Path -LiteralPath $Source)) { throw "Patch folder missing: $Source" }
        Copy-Item -LiteralPath $Source -Destination $ProjectRoot -Recurse -Force
        Write-Host "Installed: $Folder" -ForegroundColor Green
    }
}
finally {
    if (Test-Path -LiteralPath $Temp) { Remove-Item -LiteralPath $Temp -Recurse -Force }
}

& $Python -m ipykernel install --user --name "paper1-qc-reviewed" --display-name "Python 3.12 (paper1-qc-reviewed)" | Out-Null

$SourceNotebook = Join-Path $ProjectRoot "notebooks reviewed\01_QGAIN\02b_gain_dynamics_QGAIN_v4_1_0_FIGURE_COMPLETION_SOURCE.ipynb"
$LocalNotebook = Join-Path $ProjectRoot "notebooks reviewed\01_QGAIN\02b_gain_dynamics_QGAIN_v4_1_0_FIGURE_COMPLETION_LOCAL_RUN.ipynb"
Copy-Item -LiteralPath $SourceNotebook -Destination $LocalNotebook -Force

Write-Host ""
Write-Host "QGAIN figure-completion patch installed." -ForegroundColor Green
Write-Host "Opening notebook: $LocalNotebook" -ForegroundColor Cyan
Write-Host "Run all cells, save, close JupyterLab, then execute:" -ForegroundColor Yellow
Write-Host "  & `"$ProjectRoot\scripts reviewed\freeze_qgain_figure_package_v100.ps1`" -ProjectRoot `"$ProjectRoot`""

Set-Location -LiteralPath $ProjectRoot
& $Python -m jupyterlab --notebook-dir="$ProjectRoot" "$LocalNotebook"
