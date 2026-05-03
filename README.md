# GPU-Accelerated DNA Storage Encoding (RSTE) — Comprehensive README

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Team Members](#2-team-members)
3. [Repository Structure](#3-repository-structure)
4. [How It Works](#4-how-it-works)
5. [Prerequisites](#5-prerequisites)
6. [Setup & Installation](#6-setup--installation)
7. [Building the Code](#7-building-the-code)
8. [Running the Programs](#8-running-the-programs)
9. [Understanding the Output](#9-understanding-the-output)
10. [Profiling with NVIDIA Nsight Systems](#10-profiling-with-nvidia-nsight-systems)
11. [Datasets](#11-datasets)
12. [Troubleshooting](#12-troubleshooting)
13. [Project Files Quick Reference](#13-project-files-quick-reference)

---

## 1. Project Overview
This project benchmarks two implementations of the **Repeating Substring Tree Encoding (RSTE)** algorithm for DNA Data Storage:

- **Sequential (CPU)**
- **Parallel (GPU/CUDA)**

DNA data storage offers extreme density and long-term durability over conventional silicon-based storage. The RSTE method encodes binary data into DNA nucleotide sequences (A, C, G, T) while satisfying biological constraints:

- **Run-Length Limit (RLL):** No nucleotide appears more than a set number of times consecutively.
- **GC-Content:** The ratio of G and C nucleotides stays within an acceptable biological range.

The computationally expensive **Longest Substring Backtracking Method (LSBM)** is the core bottleneck. The parallel implementation maps each LSBM segment search to a one-block-per-segment GPU kernel using a novel binary search optimization ($O(\log L)$), and accelerates greedy DNA encoding using bit-packed Huffman lookup tables in GPU shared memory.

---

## 2. Team Members
**Universitas Pelita Harapan**

| Name | Student ID |
|---|---|
| Angga Theo Kekuatanta Pasaribu | 01082230023 |
| Daniel Prawira | 01082230004 |
| Alexander Owen Marlin | 01082230048 |
| Rafelixa Reynard Isak | 01082230017 |

---

## 3. Repository Structure
```
KomparUAS/
├── docs/
│   ├── paper_lncs.tex            # Research paper (LNCS/Springer format)
│   ├── result_summary.xlsx       # Profiling results summary
│   └── result_summary.py         # Summary generator script
│
├── prof/
│   ├── Both/                     # Combined metrics & visual charts
│   │   ├── result_summary.xlsx   # Spreadsheet with embedded charts
│   │   ├── chart_runtime.png     # Visual runtime comparison
│   │   └── chart_speedup.png     # Visual speedup comparison
│   ├── Parallel/                 # NVIDIA Nsight Systems profiler output
│   │   ├── *.nsys-rep            # Nsight profiler traces
│   │   ├── *.sqlite              # Nsight SQLite databases
│   │   ├── *.csv                 # CUDA raw timing CSVs
│   │   └── chart_kernel.png      # CUDA per-kernel breakdown chart
│   └── Sequential/               # CPU execution reports
│       └── *.csv                 # CPU raw timing CSVs
│
├── src/                          # ALL source, dataset, and automation files
│   ├── dataset/                  # Input files used for benchmarking
│   │   ├── internationale_paper.pdf
│   │   ├── ohenry_paper.txt
│   │   ├── straybird_paper.txt
│   │   └── vangogh_paper.jpg
│   ├── main_seq.cpp              # Sequential entry point
│   ├── main_cuda.cu              # Parallel (CUDA) entry point
│   ├── lsbm.cpp / lsbm_kernel.cu # LSBM implementations
│   ├── encoder.cpp / encode_kernel.cu
│   ├── constraints.cpp / constraints.cu
│   ├── huffman.cpp               # CPU Huffman tree builder
│   ├── metrics.cpp               # CPU metrics output
│   ├── rste_seq.hpp / utils.cuh  # Header files
│   ├── CMakeLists.txt            # CMake build configuration
│   ├── build.ps1                 # PowerShell build automation
│   ├── build_all.cmd             # Windows batch build script
│   ├── run_experiments.ps1       # Single-dataset experiment runner
│   ├── run_all_experiments.ps1   # Full benchmark automation
│   ├── run_compare.ps1           # SEQ vs CUDA comparison script
│   └── run_dataset.ps1           # Dataset-specific runner
│
└── README.md                     # This file
```

---

## 4. How It Works

### Parallel GPU Architecture Overview
The GPU pipeline processes the RSTE encoding in overlapping phases to maximize throughput. To overcome the PCIe transfer bottleneck, memory is allocated on the device and transferred using **Pinned-Memory Asynchronous Transfers** with **Multi-Stream Overlap**:
1. **Host-to-Device (H2D)** copies the raw byte buffer to the GPU asynchronously.
2. **Compute Stream** executes the sequence of kernels (`bytes_to_bits`, `lsbm`, `encode`, `constraints`) concurrently with data transfers for other chunks.
3. **Device-to-Host (D2H)** copies the final encoded DNA strings and constraint validation metrics back to the host.

Synchronization between CPU mapping and GPU execution is handled via explicit CUDA synchronization barriers (`cudaDeviceSynchronize()`) to ensure the Huffman tree is ready before the encoding kernel fires.

### Parallel Feature Details & Optimizations
1. **Bytes to Bits Conversion (`bytes_to_bits_kernel`):**
   - A highly parallel kernel mapping 1 thread per byte.
   - Extracts 8 bits per byte and stores them in a flattened 1D array for easy $O(1)$ segment access.

2. **Longest Substring Backtracking Method (LSBM):**
   - **The Core Bottleneck**: Sequential profiling reveals that LSBM consumes **over 92.6%** of the total CPU runtime due to its quadratic dependence on segment length.
   - **CUDA Mapping**: Maps **1 block per segment** (500 bits) and utilizes **Cooperative Shared-Memory Loading** to broadcast the segment bits to all threads in the block, avoiding global memory latency.
   - **$O(\log L)$ Binary Search Optimization**: The predicate "a prefix or suffix of length $L$ occurs at least twice non-overlapping" is *monotone-decreasing*. Instead of a naïve $O(L)$ linear scan checking $L=499, 498 \dots$, the GPU performs a binary search. This cuts iterations per segment from $\approx 480$ down to $\approx 9$, yielding a **$68\times$ in-kernel speed-up** for LSBM alone.

3. **Huffman Tree & Map Generation (CPU-assisted):**
   - The GPU outputs substring frequencies. The CPU quickly builds a Huffman tree using a min-heap ($O(N \log N)$) and generates a `DnaCodeTable`.
   - The lookup table is transferred back to the GPU's **Constant Memory** for blazing-fast, cache-coherent access during encoding.

4. **Greedy DNA Encoding & Constraints (`rste_encode_kernel` & `gc_rll_end_kernel`):**
   - `rste_encode_kernel` translates the compressed bits into quaternary DNA bases.
   - Applies odd/even level rules to the Huffman tree (Odd: `G/C`, Even: `A/T`) to inherently prevent long homopolymers.
   - `gc_rll_end_kernel` performs a final verification pass using parallel reduction to validate:
     - **GC-Content**: Must remain between $40\% - 60\%$.
     - **Run-Length Limit (RLL)**: Maximum identical consecutive bases $\le 2$.
     - **End-Constraint**: At most three G/C bases in the last 5 positions.

### Sequential CPU Baseline
The sequential implementation (`src/main_seq.cpp`) serves as the explicit, single-threaded baseline:
- Executes the KMP-based linear LSBM, Huffman tree build, and DNA encoding in a strict sequential loop without any CUDA dependencies.
- Serves as the ground truth. It produces output that is **bit-exact identical** to the GPU version, verified through 64-bit FNV-1a hash equality.

### Parallel vs Sequential Comparison Methodology & Amdahl's Law
When running the automated benchmark, the script measures the end-to-end wall clock time, breaking down LSBM, Huffman, and Encoding phases.

**Amdahl's Law Analysis:**
Profiling reveals that **$\approx 93.6\%$** of the RSTE pipeline is fully parallelizable (LSBM and Encoding). The remaining $6.4\%$ is the residual sequential fraction attributable to Huffman tree construction and string assembly on the host. 

With $f = 0.936$, the theoretical speedup ceiling is governed by:
$$S(p) = \frac{1}{(1 - f) + f/p}$$
Across four heterogeneous datasets (from 205 KB to 4.31 MB), the implemented CUDA pipeline achieves empirical end-to-end **speedups between $4.0\times$ and $11.6\times$**, perfectly aligning with the Amdahl's Law theoretical limits, all while maintaining **zero biological-constraint violations**.

### Bagaimana Dataset Diproses (Mapping File $\rightarrow$ Binary $\rightarrow$ DNA)
Dataset (teks, PDF, JPG) diproses menjadi DNA melalui alur berikut:

1. **Pembacaan Byte Input:**
   Program membaca seluruh isi *file* input sebagai array *byte* murni (`uint8_t`).

2. **Ekstraksi Byte ke Bit (`bytes_to_bits_kernel`):**
   Setiap *byte* dipecah menjadi 8 bit secara sekuensial. Contoh untuk 1 byte:
   - `bit 7` $\rightarrow$ elemen array ke-0
   - `bit 6` $\rightarrow$ elemen array ke-1
   - ...
   - `bit 0` $\rightarrow$ elemen array ke-7

3. **Kompresi & Huffman Mapping (RSTE):**
   Array bit tersebut kemudian dibagi menjadi segmen (500-bit). Pola bit yang sering muncul dikonversi menjadi rute pohon Huffman. Untuk memenuhi batas *Run-Length* (RLL), simbol DNA dipilih berdasarkan kedalaman pohon:
   - Level ganjil: memilih antara **G** atau **C**
   - Level genap: memilih antara **A** atau **T**

4. **Quaternary Fallback Mapping:**
   Untuk sisa bit yang tidak terkompresi atau untuk blok *Address* pengenal data, konversi dilakukan dengan pemetaan *lossless* RLL-safe (2 bit $\rightarrow$ 2 nukleotida):
   - `00` $\rightarrow$ **GA**
   - `01` $\rightarrow$ **GT**
   - `10` $\rightarrow$ **CA**
   - `11` $\rightarrow$ **CT**
   
   Karena semua blok dimulai dengan G/C dan diakhiri dengan A/T, penggabungan blok mana pun tidak akan pernah menghasilkan run-length lebih dari 1 secara internal, dan maksimal run-length 2 di perbatasan Huffman-fallback. Ini **secara matematis menjamin RLL $\le 2$**.

---

## 5. Prerequisites

### Hardware
- An **NVIDIA GPU** with CUDA Compute Capability **6.0 or higher** is recommended.

### Software
| Tool | Version | Purpose |
|---|---|---|
| **CUDA Toolkit** | v12.4 or newer | Compiles `.cu` files with `nvcc` |
| **CMake** | v3.18+ | Generates the build system |
| **Microsoft Visual Studio** | 2022 (Community or higher) | C++17 compiler (`MSVC`) required by CMake |
| **NVIDIA Nsight Systems** | Latest | For viewing `.nsys-rep` profiler files |
| **Windows OS** | Windows 10/11 | The build script is a `.ps1` file (PowerShell) |

> **Note:** The provided `build.ps1` script is written for **Windows**. Running on Linux requires manual CMake compilation (see Section 7).

### Checking Your GPU
Open a terminal and run:

```cmd
nvidia-smi
```

Look for your GPU's **Compute Capability** in the output. Alternatively, check NVIDIA's [CUDA GPU List](https://developer.nvidia.com/cuda-gpus).

---

## 6. Setup & Installation

### Step 1 — Extract the Archive
Extract the ZIP to any folder. You should see the directory structure described in Section 3.

### Step 2 — Install CUDA Toolkit
Download from: https://developer.nvidia.com/cuda-downloads

Choose your OS → Architecture → Version. Run the installer and follow the setup wizard. After installation, verify:

```cmd
nvcc --version
```

### Step 3 — Install Visual Studio 2022 and CMake
Download VS from: https://visualstudio.microsoft.com/vs/community/
During installation, select the **"Desktop development with C++"** workload. 
Also, ensure **CMake** is installed and accessible from your terminal (`cmake --version`).

---

## 7. Building the Code

### Automated Build (Recommended — Windows)
Open PowerShell at the repository root and run:

```powershell
.\build.ps1
```

This script does the following automatically:

1. Configures the CMake build system.
2. Compiles the **sequential** implementation → `build/Release/rste_seq.exe`
3. Compiles the **CUDA parallel** implementation → `build/Release/rste_cuda.exe`

If either build fails, an error message is shown and the script exits. See Section 12 (Troubleshooting) for fixes.

### Manual Build (Windows / Linux)
If the automated script fails, open a terminal in the root directory and run:

```bash
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
```

---

## 8. Running the Programs

Both executables share the same command-line interface:

```
<executable> <input_file> <dataset_tag> <output_csv>
```

| Argument | Description | Example |
|---|---|---|
| `<input_file>` | Path to the file to encode (txt, pdf, jpg, etc.) | `dataset\straybird_paper.txt` |
| `<dataset_tag>` | A short label used to identify this run in the CSV output | `straybird_cuda` |
| `<output_csv>` | Path to a CSV file where metrics are appended | `prof\Parallel\cuda_straybird.csv` |

### Automated Benchmarking (All Datasets)
To run all 4 datasets on both CPU and GPU automatically, and to verify the FNV-1a Hash integrity:

```powershell
.\run_experiments.ps1
```

### Manual Run Example

**Running Sequential:**
```powershell
.\build\Release\rste_seq.exe dataset\straybird_paper.txt seq_straybird prof\Sequential\seq_straybird.csv
```

**Running Parallel (CUDA):**
```powershell
.\build\Release\rste_cuda.exe dataset\straybird_paper.txt cuda_straybird prof\Parallel\cuda_straybird.csv
```

> **Tip:** The `prof/Both/result_summary.xlsx` file aggregates these CSV outputs.

---

## 9. Understanding the Output

### Terminal Output
After each run, a summary line is printed showing the metrics:

| Field | Meaning |
|---|---|
| `dataset` | The dataset tag you passed in |
| `rate` | Encoding rate in bits per nucleotide (payload efficiency) |
| `gc_mean` | Mean GC-content across all storage sequences (target: ~0.50) |
| `rll_max` | Maximum run-length observed (lower is better for biological validity) |
| `hash` | FNV1a-64 hash of the DNA output — use this to verify reproducibility |
| `total_ms` | Total wall-clock time in milliseconds |

### CSV Output
Each run **appends** a row to the specified CSV file. The CSV contains detailed timing breakdowns for each pipeline phase:

- `h2d_ms` — Host-to-Device memory transfer time
- `bits_ms` — Byte-to-bit conversion kernel time
- `lsbm_ms` — LSBM kernel time (main bottleneck)
- `huffman_ms` — Huffman tree build time
- `encode_ms` — DNA encoding kernel time
- `constraint_ms` — Biological constraint check time
- `d2h_ms` — Device-to-Host memory transfer time
- `total_ms` — Total wall-clock time

Open the CSV in Excel or compare it directly using `prof/Both/result_summary.xlsx`.

---

## 10. Profiling with NVIDIA Nsight Systems
The `prof/Parallel/` folder contains pre-collected profiler traces for all four datasets. To view them:

### Open an Existing Profiler Report
1. Install **NVIDIA Nsight Systems** from https://developer.nvidia.com/nsight-systems
2. Launch Nsight Systems.
3. Go to **File → Open** and select any `.nsys-rep` file from `prof/Parallel/`.
4. The timeline view shows kernel execution, memory transfers, and CPU activity side-by-side.

### Collect a New Profile (Optional)
To re-profile your own run:

```cmd
nsys profile -o prof\Parallel\my_run .\build\Release\rste_cuda.exe dataset\straybird_paper.txt txt1_straybird results.csv
```

---

## 11. Datasets
The `dataset/` folder contains four input files representing different data types and sizes:

| File | Type | Description |
|---|---|---|
| `straybird_paper.txt` | Text | Poem collection (~205 KB) |
| `ohenry_paper.txt` | Text | O. Henry short stories collection (~758 KB) |
| `internationale_paper.pdf` | PDF | Document file (~479 KB) |
| `vangogh_paper.jpg` | JPEG Image | Van Gogh museum artwork (~4.3 MB) |

These were chosen to test the encoder across different file sizes and binary content patterns.

---

## 12. Troubleshooting

### "CMake Error: No CMAKE_CUDA_COMPILER could be found"
CUDA Toolkit is not in your PATH. Fix options:
- Re-run the CUDA Toolkit installer and ensure the **"Add to PATH"** option is selected.
- Or manually add `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\bin` to your system's `PATH` environment variable.

### "Cannot read: dataset\..."
The executable cannot find the input file. Ensure you run the command **from the repository root directory**, not from `build/`. The dataset paths are relative.

```powershell
cd path\to\FinalProject_Submission   ← must be here
.\build\Release\rste_cuda.exe dataset\straybird_paper.txt ...
```

### CUDA Runtime Error / GPU Crash
- Update your NVIDIA GPU driver to the latest version.
- Ensure the CUDA Toolkit version (12.4+) is compatible with your driver version.
- Try a smaller input file first to rule out memory issues.

### Results Don't Match `result_summary.xlsx`
- The `hash` field in the terminal output is a deterministic fingerprint of the DNA output. Compare hashes between runs to verify consistency.
- Timing results will vary by machine and GPU model — only the encoding rate, GC-content, and hash should be deterministic.

---

## 13. Project Files Quick Reference

| File | Purpose |
|---|---|
| `build.ps1` | One-click CMake build script for Windows |
| `run_experiments.ps1` | Automated testing script for all datasets |
| `src/main_seq.cpp` | Sequential program entry point |
| `src/main_cuda.cu` | Parallel CUDA program entry point |
| `src/lsbm_kernel.cu` | GPU kernel for binary-search LSBM segment processing |
| `src/encode_kernel.cu` | GPU kernel for Huffman-based DNA encoding |
| `src/constraints.cu` | GPU kernel for GC-content & run-length checks |
| `src/utils.cuh` | Shared CUDA utility macros and helpers |
| `src/lsbm.cpp` | CPU implementation of LSBM |
| `src/huffman.cpp` | CPU Huffman tree builder |
| `src/encoder.cpp` | CPU DNA encoder |
| `src/constraints.cpp` | CPU biological constraint checker |
| `src/metrics.cpp` | CPU metrics output helper |
| `docs/paper_lncs.tex` | Full research paper describing the method (Springer format) |
| `prof/Both/result_summary.xlsx` | Benchmark results spreadsheet with embedded charts |
| `prof/Parallel/*.nsys-rep` | Nsight Systems profiler traces (GPU runs) |
