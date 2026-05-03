cmake_minimum_required(VERSION 3.18)
project(RSTE LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Sequential (CPU) version
add_executable(rste_seq
    src/main_seq.cpp
    src/encoder.cpp
    src/lsbm.cpp
    src/huffman.cpp
    src/constraints.cpp
    src/metrics.cpp
)
target_compile_options(rste_seq PRIVATE -O2)

# CUDA version (if CUDA enabled)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    add_executable(rste_cuda
        src/main_cuda.cu
        src/lsbm_kernel.cu
        src/encode_kernel.cu
        src/constraints.cu
    )
    target_compile_options(rste_cuda PRIVATE -O2)
    target_link_libraries(rste_cuda PRIVATE cudart)

    # Set CUDA architecture (adjust as needed)
    set_target_properties(rste_cuda PROPERTIES CUDA_ARCHITECTURES "60-real;61-real;70-real;75-real;80-real;86-real")
else()
    message(STATUS "CUDA not found; skipping GPU build")
endif()
