#!/usr/bin/env pwsh
# build.ps1 - Build RSTE (sequential and optional CUDA)

param(
    [switch]$Clean,
    [switch]$Test,
    [switch]$GPU
)

$ErrorActionPreference = "Stop"

Write-Output "=== RSTE Build Script ==="

# Clean if requested
if ($Clean) {
    if (Test-Path "build") {
        Write-Output "Cleaning build directory..."
        Remove-Item -Recurse -Force build
    }
}

# Create build directory
if (-not (Test-Path "build")) {
    New-Item -ItemType Directory -Path build | Out-Null
}

$BuildDir = (Resolve-Path "build").Path
$SourceDir = (Resolve-Path ".").Path

Write-Output "Build directory: $BuildDir"
Write-Output "Source directory: $SourceDir"

# Try CMake if available
$CMakePath = (Get-Command cmake -ErrorAction SilentlyContinue).Source
if ($CMakePath) {
    Write-Output "CMake found at: $CMakePath"
    Write-Output "Configuring with CMake..."
    
    Push-Location $BuildDir
    
    $GenArgs = @("-G", "Visual Studio 17 2022", "..")
    if (-not $GPU) {
        $GenArgs += "-DCMAKE_DISABLE_FIND_PACKAGE_CUDA=ON"
    }
    
    & cmake @GenArgs
    if ($LASTEXITCODE -ne 0) {
        Write-Error "CMake configuration failed"
    }
    
    & cmake --build . --config Release
    if ($LASTEXITCODE -ne 0) {
        Write-Error "CMake build failed"
    }
    
    Pop-Location
    
    Write-Output "Build complete. Executables in: build/Release/"
    Get-ChildItem -Path "$BuildDir/Release/*.exe" -ErrorAction SilentlyContinue | ForEach-Object {
        Write-Output "  - $($_.Name)"
    }
} else {
    # Fallback: manual g++ build
    Write-Output "CMake not found; using g++ for sequential build..."
    
    $ExeName = "rste_seq.exe"
    $CompileCmd = @(
        "g++", "-O2", "-std=c++17",
        "rste_seq/main.cpp",
        "rste_seq/encoder.cpp",
        "rste_seq/lsbm.cpp",
        "rste_seq/huffman.cpp",
        "rste_seq/constraints.cpp",
        "rste_seq/metrics.cpp",
        "-o", $ExeName
    )
    
    Write-Output "Compiling: $([string]::Join(' ', $CompileCmd))"
    $CompileArgs = $CompileCmd[1..($CompileCmd.Length-1)]
    & $CompileCmd[0] @CompileArgs
    
    if ($LASTEXITCODE -eq 0) {
        Write-Output "Build successful: $ExeName"
    } else {
        Write-Error "Build failed"
    }
}

# Run tests if requested
if ($Test) {
    Write-Output ""
    Write-Output "=== Running Tests ==="
    
    $TestExe = "test_rste.exe"
    if (-not (Test-Path $TestExe)) {
        Write-Output "Compiling test..."
        $TestCompile = @(
            "g++", "-O2", "-std=c++17",
            "test_rste.cpp",
            "rste_seq/encoder.cpp",
            "rste_seq/lsbm.cpp",
            "rste_seq/huffman.cpp",
            "rste_seq/constraints.cpp",
            "-o", $TestExe
        )
        $TestArgs = $TestCompile[1..($TestCompile.Length-1)]
        & $TestCompile[0] @TestArgs
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Test compile failed"
        }
    }
    
    Write-Output "Running: .\$TestExe"
    & ".\$TestExe"
    if ($LASTEXITCODE -eq 0) {
        Write-Output "Tests PASSED"
    } else {
        Write-Error "Tests FAILED"
    }
}

Write-Output ""
Write-Output "=== Build Complete ==="
