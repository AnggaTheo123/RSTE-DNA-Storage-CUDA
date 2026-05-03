#!/usr/bin/env pwsh
# run_compare.ps1 - Compare CUDA vs Sequential

$ErrorActionPreference = "Continue"

Write-Output "==================================="
Write-Output "  RSTE: CUDA vs Sequential Compare"
Write-Output "==================================="
Write-Output ""

# Check executables
$HasSeq = Test-Path "rste_seq.exe"
$HasGpu = Test-Path "rste_cuda.exe"

if (-not $HasSeq) {
    Write-Error "rste_seq.exe not found"
    exit 1
}

if (-not $HasGpu) {
    Write-Output "(GPU executable not found - will run Sequential only)"
    Write-Output ""
}

if (-not (Test-Path "results")) {
    mkdir "results" | Out-Null
}

# Dataset options
Write-Output "Datasets:"
Write-Output ""
Write-Output "  [1] straybird.txt           50 KB"
Write-Output "  [2] internationale.pdf      35 KB"
Write-Output "  [3] vangoghmuseum.jpg       118 KB"
Write-Output "  [4] o henry.txt             5.2 MB"
Write-Output ""
Write-Output "  [A] All"
Write-Output ""

$Input = Read-Host "Select (1 or 1,2,4 or A) [default: 1]"
if ([string]::IsNullOrWhiteSpace($Input)) { $Input = "1" }

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
    exit 1
}

$Files = @{
    1 = "straybird.txt"
    2 = "internationale.pdf"
    3 = "vangoghmuseum-s0031V1962-800.jpg"
    4 = "o henry.txt"
}

Write-Output ""
Write-Output "Starting comparison..."
Write-Output ""

$TimeStamp = Get-Date -Format "yyyyMMdd_HHmmss"
$Results = @()

foreach ($S in $Selected) {
    $File = $Files[$S]
    $Input = "src/dataset/$File"
    
    if (-not (Test-Path $Input)) {
        Write-Output "SKIP: Not found - $File"
        continue
    }

    $Name = $File -replace '\.[^.]*$', ''
    Write-Output "Processing: $File"
    Write-Output ""

    # Sequential
    $CsvSeq = "results/${TimeStamp}_seq_$Name.csv"
    $T0 = Get-Date
    & .\rste_seq.exe "$Input" "seq" $CsvSeq 2>&1 | Select-Object -Last 1
    $T1 = (Get-Date) - $T0
    $SeqMs = $T1.TotalMilliseconds
    $SeqSec = [Math]::Round($T1.TotalSeconds, 2)

    $SeqHash = ""
    if (Test-Path $CsvSeq) {
        $Line = Get-Content $CsvSeq -Tail 1
        $Parts = $Line -split ","
        $SeqHash = $Parts[-2]
    }

    Write-Output "  Sequential: ${SeqSec}s | Hash: $SeqHash"

    # GPU
    $GpuSec = "-"
    $GpuHash = ""
    $Speedup = "-"
    $MatchStatus = "-"
    
    if ($HasGpu) {
        $CsvGpu = "results/${TimeStamp}_gpu_$Name.csv"
        $T0 = Get-Date
        & .\rste_cuda.exe "$Input" "gpu" $CsvGpu 2>&1 | Select-Object -Last 1
        $T1 = (Get-Date) - $T0
        $GpuSec = [Math]::Round($T1.TotalSeconds, 2)

        if (Test-Path $CsvGpu) {
            $Line = Get-Content $CsvGpu -Tail 1
            $Parts = $Line -split ","
            $GpuHash = $Parts[-2]
            
            $Speedup = [Math]::Round($SeqSec / $GpuSec, 2)
            
            if ($SeqHash -eq $GpuHash -and $SeqHash -ne "") {
                $MatchStatus = "YES"
            } else {
                $MatchStatus = "NO"
            }
        }

        Write-Output "  GPU:        ${GpuSec}s | Hash: $GpuHash"
        Write-Output "  Speedup:    ${Speedup}x"
        Write-Output "  Match:      $MatchStatus"
    }

    Write-Output ""

    $Results += [PSCustomObject]@{
        File = $File
        SeqTime = $SeqSec
        GpuTime = $GpuSec
        Speedup = $Speedup
        Match = $MatchStatus
    }
}

# Summary table
Write-Output "==================================="
Write-Output "  Summary"
Write-Output "==================================="
Write-Output ""

if ($Results.Count -gt 0) {
    # Display as formatted table
    if ($HasGpu) {
        Write-Output "dataset | seq time (s) | gpu time (s) | speedup | match"
        Write-Output "--------|--------------|--------------|---------|-------"
    } else {
        Write-Output "STATUS: GPU not available (Sequential CPU only)"
        Write-Output ""
        Write-Output "dataset | seq time (s) | encoding rate | gc mean | rll max"
        Write-Output "--------|--------------|---------------|---------|--------"
    }
    
    foreach ($R in $Results) {
        if ($HasGpu) {
            $Line = "$($R.File) | $($R.SeqTime) | $($R.GpuTime) | $($R.Speedup) | $($R.Match)"
        } else {
            $Line = "$($R.File) | $($R.SeqTime) | (CPU only) | N/A | N/A"
        }
        Write-Output $Line
    }
    
    Write-Output ""
    
    if (-not $HasGpu) {
        Write-Output "To enable GPU: Install CUDA 12.4 + MSVC, then rebuild"
        Write-Output ""
    }
    
    Write-Output "CSV files saved:"
    Get-ChildItem "results/*${TimeStamp}*.csv" -ErrorAction SilentlyContinue | ForEach-Object {
        Write-Output "  - $($_.Name)"
    }
    
    Write-Output ""
    Write-Output "Generating visualization..."
    if (Get-Command python -ErrorAction SilentlyContinue) {
        python generate_visualization.py --chart
    } else {
        Write-Output "  (python not available - skipping charts)"
    }
} else {
    Write-Output "No results"
}

Write-Output ""
Write-Output "Done!"
