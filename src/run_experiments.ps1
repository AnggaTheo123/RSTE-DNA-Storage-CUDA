#!/usr/bin/env pwsh
# run_experiments.ps1 - Run RSTE on all datasets and validate CPU vs GPU hashes

param(
    [string]$DatasetDir = "src/dataset",
    [string]$ResultsDir = "results"
)

$ErrorActionPreference = "Continue"

Write-Output "=== RSTE Experiment Runner ==="
Write-Output "Dataset dir: $DatasetDir"
Write-Output "Results dir: $ResultsDir"

# Create results directory
if (-not (Test-Path $ResultsDir)) {
    New-Item -ItemType Directory -Path $ResultsDir | Out-Null
}

# Check if executables exist
$SeqExe = "rste_seq.exe"
$GpuExe = "rste_cuda.exe"

if (-not (Test-Path $SeqExe)) {
    Write-Error "Sequential executable not found: $SeqExe"
    Write-Output "Run: .\build.ps1"
    exit 1
}

Write-Output "Found: $SeqExe"
if (Test-Path $GpuExe) {
    Write-Output "Found: $GpuExe"
} else {
    Write-Output "Warning: GPU executable not found; will skip GPU runs"
}

# Find dataset files
$Datasets = @()
if (Test-Path $DatasetDir) {
    Get-ChildItem -Path $DatasetDir -File | Where-Object { 
        $_.Extension -in @(".txt", ".pdf", ".jpg") 
    } | ForEach-Object {
        $Datasets += @{
            Name = $_.BaseName
            Path = $_.FullName
            Tag  = $_.BaseName -replace "-", "_"
        }
    }
}

if ($Datasets.Count -eq 0) {
    Write-Error "No datasets found in $DatasetDir"
    exit 1
}

Write-Output "Found $($Datasets.Count) dataset(s)"

# Run experiments
$Results = @()

foreach ($DS in $Datasets) {
    Write-Output ""
    Write-Output "--- Processing: $($DS.Name) ---"
    
    $OutSeq = "$ResultsDir/$($DS.Tag)_seq.csv"
    $OutGpu = "$ResultsDir/$($DS.Tag)_gpu.csv"
    
    # Sequential
    Write-Output "Running sequential..."
    & $SeqExe "$($DS.Path)" "exp_seq" $OutSeq
    
    if ($LASTEXITCODE -ne 0) {
        Write-Output "  ERROR: Sequential run failed"
        continue
    }
    
    $HashSeq = ""
    if (Test-Path $OutSeq) {
        $CSV = Get-Content $OutSeq | Select-Object -Last 1
        $HashSeq = ($CSV -split ",")[-2]  # dna_hash_payload is second-to-last
        Write-Output "  Sequential hash: $HashSeq"
    }
    
    # GPU (if available)
    $HashGpu = ""
    $Match = "N/A"
    
    if (Test-Path $GpuExe) {
        Write-Output "Running GPU..."
        & $GpuExe "$($DS.Path)" "exp_gpu" $OutGpu
        
        if ($LASTEXITCODE -eq 0 -and (Test-Path $OutGpu)) {
            $CSV = Get-Content $OutGpu | Select-Object -Last 1
            $HashGpu = ($CSV -split ",")[-2]
            Write-Output "  GPU hash: $HashGpu"
            
            if ($HashSeq -eq $HashGpu) {
                $Match = "MATCH ✓"
                Write-Output "  $Match"
            } else {
                $Match = "MISMATCH ✗"
                Write-Output "  $Match (potential bug!)"
            }
        } else {
            Write-Output "  GPU run failed"
        }
    }
    
    $Results += @{
        Dataset   = $DS.Name
        HashSeq   = $HashSeq
        HashGpu   = $HashGpu
        Match     = $Match
    }
}

# Summary
Write-Output ""
Write-Output "=== Experiment Summary ==="
Write-Output ""
$Results | Format-Table -AutoSize

# Write summary CSV
$SummaryCSV = "$ResultsDir/summary.csv"
"Dataset,HashSeq,HashGpu,Match" | Out-File $SummaryCSV
foreach ($R in $Results) {
    "$($R.Dataset),$($R.HashSeq),$($R.HashGpu),$($R.Match)" | Out-File $SummaryCSV -Append
}
Write-Output "Summary written to: $SummaryCSV"

# Check for mismatches
$Mismatches = $Results | Where-Object { $_.Match -eq "MISMATCH ✗" }
if ($Mismatches.Count -gt 0) {
    Write-Output ""
    Write-Output "WARNING: $($Mismatches.Count) hash mismatch(es) detected!"
    exit 1
} else {
    Write-Output ""
    Write-Output "All experiments completed successfully!"
    exit 0
}
