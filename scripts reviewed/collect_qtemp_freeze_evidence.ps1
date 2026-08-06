param(
    [Parameter(Mandatory = $true)]
    [string]$ProjectRoot
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path $ProjectRoot).Path
$Stage = Join-Path $ProjectRoot "outputs\02_features\temporal_discontinuity\qtemp-v0.3.1-finalization"
$DevelopmentStage = Join-Path $ProjectRoot "outputs\02_features\temporal_discontinuity\qtemp-v0.3.0-measurement-development"
$BundleRoot = Join-Path $ProjectRoot "QTEMP_FREEZE_EVIDENCE_FOR_REVIEW"
$BundleZip = Join-Path $ProjectRoot "QTEMP_FREEZE_EVIDENCE_FOR_REVIEW.zip"

if (-not (Test-Path $Stage)) {
    throw "Required QTEMP finalization stage was not found: $Stage"
}

if (Test-Path $BundleRoot) {
    Remove-Item $BundleRoot -Recurse -Force
}
if (Test-Path $BundleZip) {
    Remove-Item $BundleZip -Force
}

New-Item -ItemType Directory -Path $BundleRoot | Out-Null

$IncludedExtensions = @(
    ".csv", ".json", ".md", ".txt", ".yaml", ".yml",
    ".png", ".svg", ".pdf", ".wav", ".flac"
)
$ExcludedDirectoryNames = @(
    "checkpoints", "recording_cache", ".ipynb_checkpoints", "__pycache__"
)

function Copy-EvidenceTree {
    param(
        [string]$SourceRoot,
        [string]$DestinationName
    )

    if (-not (Test-Path $SourceRoot)) {
        return
    }

    $DestinationRoot = Join-Path $BundleRoot $DestinationName

    Get-ChildItem -Path $SourceRoot -Recurse -File | ForEach-Object {
        $SourceFile = $_
        $RelativePath = $SourceFile.FullName.Substring($SourceRoot.Length).TrimStart("\")
        $PathParts = $RelativePath -split "\\"
        $Excluded = $false

        foreach ($Part in $PathParts) {
            if ($ExcludedDirectoryNames -contains $Part) {
                $Excluded = $true
                break
            }
        }

        if ($Excluded) {
            return
        }

        if ($IncludedExtensions -notcontains $SourceFile.Extension.ToLowerInvariant()) {
            return
        }

        $DestinationFile = Join-Path $DestinationRoot $RelativePath
        $DestinationDirectory = Split-Path $DestinationFile -Parent
        New-Item -ItemType Directory -Path $DestinationDirectory -Force | Out-Null
        Copy-Item $SourceFile.FullName $DestinationFile -Force
    }
}

Copy-EvidenceTree -SourceRoot $Stage -DestinationName "qtemp-v0.3.1-finalization"

# Only compact validation evidence is needed from the development stage.
if (Test-Path $DevelopmentStage) {
    $DevelopmentDestination = Join-Path $BundleRoot "qtemp-v0.3.0-development-evidence"
    Get-ChildItem -Path $DevelopmentStage -Recurse -File | Where-Object {
        $Extension = $_.Extension.ToLowerInvariant()
        $Name = $_.Name.ToLowerInvariant()
        $Path = $_.FullName.ToLowerInvariant()
        ($IncludedExtensions -contains $Extension) -and
        ($Path -notmatch "recording_cache|checkpoints") -and
        ($Name -match "real_speech|signal_chain|synthetic|validation|parameter|manifest|provenance|registry")
    } | ForEach-Object {
        $RelativePath = $_.FullName.Substring($DevelopmentStage.Length).TrimStart("\")
        $DestinationFile = Join-Path $DevelopmentDestination $RelativePath
        New-Item -ItemType Directory -Path (Split-Path $DestinationFile -Parent) -Force | Out-Null
        Copy-Item $_.FullName $DestinationFile -Force
    }
}

$ExplicitFiles = @(
    "notebooks\02_feature_extraction\02f_temporal_discontinuity_QTEMP_v0_3_1_FINALIZATION_SOURCE.ipynb",
    "notebooks\02_feature_extraction\02f_temporal_discontinuity_QTEMP_v0_3_1_FINALIZATION_EXECUTED_REVIEW.ipynb",
    "notebooks\02_feature_extraction\02f_temporal_discontinuity_QTEMP_v0_3_1_FINALIZATION_EXECUTED_REVIEW(3) (1).ipynb",
    "notebooks reviewed\06_QTEMP\06_temporal_discontinuity_QTEMP_v1_0_0_REVIEWED_LOCAL_EXECUTED.ipynb",
    "notebooks reviewed\06_QTEMP\QTEMP_v100_SCIENTIFIC_AUDIT.md",
    "src\paper1_qc\qtemp.py",
    "src reviewed\paper1_qc_reviewed\qtemp_v100_candidate.py",
    "tests\test_qtemp_v03.py",
    "tests\test_qtemp_notebook_v031.py",
    "tests reviewed\test_qtemp_v100_candidate.py",
    "config\project.yaml"
)

foreach ($RelativePath in $ExplicitFiles) {
    $SourceFile = Join-Path $ProjectRoot $RelativePath
    if (-not (Test-Path $SourceFile)) {
        continue
    }
    $DestinationFile = Join-Path $BundleRoot (Join-Path "pipeline_files" $RelativePath)
    New-Item -ItemType Directory -Path (Split-Path $DestinationFile -Parent) -Force | Out-Null
    Copy-Item $SourceFile $DestinationFile -Force
}

# Capture any existing QTEMP provisional or overridden freeze metadata without
# copying large Parquet tables or duplicating the full MAIN outputs tree.
$MainRoots = @(
    (Join-Path $ProjectRoot "MAIN outputs\02_FEATURE_FAMILY_SNAPSHOTS\temporal_discontinuity"),
    (Join-Path $ProjectRoot "MAIN outputs\02_FEATURE_FAMILY_FREEZES\temporal_discontinuity"),
    (Join-Path $ProjectRoot "MAIN outputs\02_FEATURE_TABLES_PROVISIONAL"),
    (Join-Path $ProjectRoot "MAIN outputs\02_FEATURE_TABLES")
)

foreach ($MainRoot in $MainRoots) {
    if (-not (Test-Path $MainRoot)) {
        continue
    }
    Get-ChildItem -Path $MainRoot -Recurse -File | Where-Object {
        $Extension = $_.Extension.ToLowerInvariant()
        ($Extension -in @(".csv", ".json", ".md", ".txt")) -and
        ($_.Name.ToLowerInvariant() -match "qtemp|manifest|sha256|gate|decision|status")
    } | ForEach-Object {
        $RelativePath = $_.FullName.Substring($ProjectRoot.Length).TrimStart("\")
        $DestinationFile = Join-Path $BundleRoot (Join-Path "main_outputs_metadata" $RelativePath)
        New-Item -ItemType Directory -Path (Split-Path $DestinationFile -Parent) -Force | Out-Null
        Copy-Item $_.FullName $DestinationFile -Force
    }
}

$Inventory = Get-ChildItem -Path $BundleRoot -Recurse -File | ForEach-Object {
    [PSCustomObject]@{
        relative_path = $_.FullName.Substring($BundleRoot.Length).TrimStart("\")
        size_bytes = $_.Length
        sha256 = (Get-FileHash -Algorithm SHA256 -Path $_.FullName).Hash.ToLowerInvariant()
    }
}

$InventoryPath = Join-Path $BundleRoot "QTEMP_FREEZE_EVIDENCE_SHA256.csv"
$Inventory | Sort-Object relative_path | Export-Csv -Path $InventoryPath -NoTypeInformation -Encoding UTF8

$RequiredPatterns = @(
    "*analysis_features*.csv",
    "*accepted_event_ledger*.csv",
    "*candidate_disposition_ledger*.csv",
    "*exposure_ledger*.csv",
    "*real_speech*.csv",
    "*signal_chain*.csv",
    "*parameter_sensitivity*.csv",
    "*blinded_adjudication_sheet*.csv",
    "*gallery_index*.csv",
    "*gate_summary*.csv"
)

$RequirementRows = foreach ($Pattern in $RequiredPatterns) {
    $Matches = Get-ChildItem -Path $BundleRoot -Recurse -File -Filter $Pattern
    [PSCustomObject]@{
        required_pattern = $Pattern
        found = [bool]$Matches
        match_count = @($Matches).Count
        matches = (($Matches | ForEach-Object {
            $_.FullName.Substring($BundleRoot.Length).TrimStart("\")
        }) -join " | ")
    }
}

$RequirementsPath = Join-Path $BundleRoot "QTEMP_FREEZE_EVIDENCE_REQUIREMENTS.csv"
$RequirementRows | Export-Csv -Path $RequirementsPath -NoTypeInformation -Encoding UTF8

Compress-Archive -Path "$BundleRoot\*" -DestinationPath $BundleZip -CompressionLevel Optimal

Write-Host ""
Write-Host "QTEMP freeze evidence package created:" -ForegroundColor Green
Write-Host $BundleZip -ForegroundColor Cyan
Write-Host "Package size (MB):" ([math]::Round((Get-Item $BundleZip).Length / 1MB, 2))
Write-Host ""
Write-Host "Upload this ZIP for the scientific freeze review."

