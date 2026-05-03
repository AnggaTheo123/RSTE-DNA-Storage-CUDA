#!/usr/bin/env pwsh
# run_dataset.ps1 - Interactive experiment runner with dataset selection

$ErrorActionPreference = "Continue"

Write-Output "=================================="
Write-Output "   RSTE Experiment Runner"
Write-Output "   Select dataset(s) to run"
Write-Output "=================================="
Write-Output ""

# Verify executable
if (-not (Test-Path "rste_seq.exe")) {
    Write-Error "rste_seq.exe not found. Run: .\build.ps1"
    exit 1
}

if (-not (Test-Path "results")) {
    New-Item -ItemType Directory -Path "results" | Out-Null
}

# Show available datasets
Write-Output "Available Datasets:"
Write-Output ""
Write-Output "  [1] straybird.txt           (50 KB)   - QUICK (few seconds)"
Write-Output "  [2] internationale.pdf      (35 KB)   - PDF format"
Write-Output "  [3] vangoghmuseum.jpg       (118 KB)  - Image format"
Write-Output "  [4] o henry.txt             (5.2 MB)  - LARGE (~1 minute)"
Write-Output ""
Write-Output "  [A] Run all datasets"
Write-Output ""

# Get user input
$Input = Read-Host "Select (1 or 1,2,4 or A) [default: 1]"
if ([string]::IsNullOrWhiteSpace($Input)) { $Input = "1" }

# Parse and validate input
$Selected = @()
if ($Input -eq "A" -or $Input -eq "a") {
    $Selected = 1, 2, 3, 4
} else {
    $Nums = $Input.Split(",") | ForEach-Object { $_.Trim() }
    foreach ($N in $Nums) {
        if ($N -match "^\d+$") {
            $Num = [int]$N
            if ($Num -ge 1 -and $Num -le 4) {
                $Selected += $Num
            }
        }
    }
}

if ($Selected.Count -eq 0) {
    Write-Output "No valid selection."
    exit 1
}

# Map selection to files
$DatasetMap = @{
    1 = "straybird.txt"
    2 = "internationale.pdf"
    3 = "vangoghmuseum-s0031V1962-800.jpg"
    4 = "o henry.txt"
}

Write-Output ""
Write-Output "Selected datasets:"
foreach ($S in $Selected) {
    Write-Output "  - $($DatasetMap[$S])"
}

Write-Output ""
$Confirm = Read-Host "Proceed? (y/n) [default: y]"
if ($Confirm -eq "n" -or $Confirm -eq "N") {
    Write-Output "Cancelled."
    exit 0
}

# Run experiments
Write-Output ""
Write-Output "=================================="
Write-Output "   Running Experiments"
Write-Output "=================================="
Write-Output ""

$TimeStamp = Get-Date -Format "yyyyMMdd_HHmmss"
$CsvFiles = @()

foreach ($S in $Selected) {
    $Dataset = $DatasetMap[$S]
    $InputPath = "src/dataset/$Dataset"
    
    if (-not (Test-Path $InputPath)) {
        Write-Output "ERROR: Not found: $Dataset"
        continue
    }

    $BaseName = $Dataset -replace '\.[^.]*$', ''
    $OutputCsv = "results/${TimeStamp}_seq_$BaseName.csv"
    
    Write-Output "---"
    Write-Output "File: $Dataset"
    Write-Output ""

    $T1 = Get-Date
    & .\rste_seq.exe "$InputPath" "seq_test" "$OutputCsv"
    $T2 = Get-Date
    
    if ($LASTEXITCODE -eq 0) {
        $ElapsedSecs = [Math]::Round(($T2 - $T1).TotalSeconds, 2)
        Write-Output ""
        Write-Output "  Status: SUCCESS"
        Write-Output "  Time: ${ElapsedSecs}s"
        Write-Output "  Output: $OutputCsv"
        $CsvFiles += $OutputCsv
    } else {
        Write-Output ""
        Write-Output "  Status: FAILED"
    }
    Write-Output ""
}

# Final summary
Write-Output "=================================="
Write-Output "   Complete"
Write-Output "=================================="
Write-Output ""

if ($CsvFiles.Count -gt 0) {
    Write-Output "Generated CSV files:"
    foreach ($F in $CsvFiles) {
        Write-Output "  $F"
    }
}
