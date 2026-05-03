@echo off
cd /d "%~dp0"
call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64

if not exist build mkdir build

echo === Building Sequential (CPU) ===
cl /nologo /O2 /std:c++17 /EHsc ^
  /Fe:build\rste_seq.exe ^
  /Fo:build\ ^
  src\rste_seq\main.cpp ^
  src\rste_seq\lsbm.cpp ^
  src\rste_seq\huffman.cpp ^
  src\rste_seq\encoder.cpp ^
  src\rste_seq\constraints.cpp ^
  src\rste_seq\metrics.cpp
if errorlevel 1 (
  echo [FAIL] Sequential build failed
  exit /b 1
)
echo [OK] Sequential built: build\rste_seq.exe

echo === Building CUDA Windows (GPU) ===
nvcc -O2 -std=c++17 ^
  -o build\rste_cuda.exe ^
  src\rste_cuda\main.cu ^
  src\rste_cuda\lsbm_kernel.cu ^
  src\rste_cuda\encode_kernel.cu ^
  src\rste_cuda\constraints.cu
if errorlevel 1 (
  echo [FAIL] CUDA build failed
  exit /b 1
)
echo [OK] CUDA built: build\rste_cuda.exe

echo === All builds successful ===
