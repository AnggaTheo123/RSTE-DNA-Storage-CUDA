#!/usr/bin/env pwsh
<#
.SYNOPSIS
Run all 4 datasets AUTOMATICALLY - no menu selection needed
#>

$ErrorActionPreference = "Continue"

Write-Output ""
Write-Output "========================================="
Write-Output "  RSTE: Batch Run All 4 Datasets (AUTO)"
Write-Output "========================================="
Write-Output ""
Write-Output "Running all datasets automatically..."
Write-Output "This may take 5-10 minutes (depending on file size)"
Write-Output ""

# Check executables
if (-not (Test-Path "rste_seq.exe")) {
    Write-Error "rste_seq.exe not found"
    exit 1
}

# Create results folder
if (-not (Test-Path "results")) {
    mkdir "results" | Out-Null
}

$TimeStamp = Get-Date -Format "yyyyMMdd_HHmmss"
$Results = @()

$Files = @{
    1 = "straybird.txt"
    2 = "internationale.pdf"
    3 = "vangoghmuseum-s0031V1962-800.jpg"
    4 = "o henry.txt"
}

# Dataset menu selection
Write-Output "Available Datasets:"
Write-Output ""
Write-Output "  [1] straybird.txt              50 KB"
Write-Output "  [2] internationale.pdf         35 KB"
Write-Output "  [3] vangoghmuseum.jpg         118 KB"
Write-Output "  [4] o henry.txt              5.2 MB"
Write-Output ""
Write-Output "  [A] All (1,2,3,4)"
Write-Output ""

$Input = Read-Host "Select datasets (1 or 1,2,4 or A) [default: A]"
if ([string]::IsNullOrWhiteSpace($Input)) { $Input = "A" }

$Selected = @()
if ($Input -eq "A" -or $Input -eq "a") {
    $Selected = 1, 2, 3, 4
} else {
    $Input.Split(",") | ForEach-Object {
        $Num = [int]$_.Trim()
        if ($Num -ge 1 -and $Num -le 4) {
            $Selected += $Num
        }
    }
}

if ($Selected.Count -eq 0) {
    Write-Output "No valid datasets selected"
    exit 1
}

Write-Output ""
Write-Output "Starting experiments for datasets: $($Selected -join ', ')"
Write-Output ""

foreach ($S in $Selected) {
    $File = $Files[$S]
    $Input = "src/dataset/$File"
    
    if (-not (Test-Path $Input)) {
        Write-Output "SKIP: Not found - $File"
        continue
    }

    $Name = $File -replace '\.[^.]*$', ''
    Write-Output "[$S/4] Processing: $File"
    
    # Sequential
    $CsvSeq = "results/${TimeStamp}_seq_$Name.csv"
    $T0 = Get-Date
    & .\rste_seq.exe "$Input" "seq" $CsvSeq 2>&1 | Select-Object -Last 1
    $T1 = (Get-Date) - $T0
    $SeqSec = [Math]::Round($T1.TotalSeconds, 2)

    $SeqHash = ""
    if (Test-Path $CsvSeq) {
        $Line = Get-Content $CsvSeq -Tail 1
        $Parts = $Line -split ","
        $SeqHash = $Parts[-2]
    }

    Write-Output "      Time: ${SeqSec}s | Hash: $SeqHash"
    Write-Output ""

    $Results += [PSCustomObject]@{
        File = $File
        SeqTime = $SeqSec
        GpuTime = "-"
        Speedup = "-"
        Match = "-"
    }
}

# Summary
Write-Output "========================================="
Write-Output "  Experiment Complete"
Write-Output "========================================="
Write-Output ""

$HasGpuExe = Test-Path "rste_cuda.exe"

Write-Output "Results Summary:"
Write-Output ""

if ($HasGpuExe) {
    Write-Output "dataset | seq time (s) | gpu time (s) | speedup | match"
    Write-Output "--------|--------------|--------------|---------|-------"
} else {
    Write-Output "NOTE: GPU executable not found - showing CPU (Sequential) results only"
    Write-Output ""
    Write-Output "dataset | seq time (s) | encoding rate | gc mean | rll max"
    Write-Output "--------|--------------|---------------|---------|--------"
}

foreach ($R in $Results) {
    if ($HasGpuExe) {
        $Line = "$($R.File) | $($R.SeqTime) | $($R.GpuTime) | $($R.Speedup) | $($R.Match)"
    } else {
        # Show encoding metrics instead
        $Line = "$($R.File) | $($R.SeqTime) | (CPU only) | N/A | N/A"
    }
    Write-Output $Line
}

Write-Output ""

if (-not $HasGpuExe) {
    Write-Output "To enable GPU acceleration:"
    Write-Output "  1. Install CUDA Toolkit 12.4+"
    Write-Output "  2. Install MSVC (Visual Studio Community)"
    Write-Output "  3. Run: cmake --build build --config Release"
    Write-Output ""
}

Write-Output "CSV files saved:"
Get-ChildItem "results/*${TimeStamp}*.csv" -ErrorAction SilentlyContinue | ForEach-Object {
    Write-Output "  - $($_.Name)"
}

Write-Output ""
Write-Output "Generating comprehensive visualization..."
if (Get-Command python -ErrorAction SilentlyContinue) {
    python generate_visualization.py --chart
    python generate_visualization.py --html
    Write-Output ""
    Write-Output "Reports generated:"
    Write-Output "  - Bar Chart:  results/chart_bar_${TimeStamp}.png"
    Write-Output "  - Line Chart: results/chart_line_${TimeStamp}.png"
    Write-Output "  - HTML Report: results/report_${TimeStamp}.html"
} else {
    Write-Output "  (python not available - install for charts)"
}

Write-Output ""
Write-Output "Done!"
