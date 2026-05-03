#pragma once

#include <cuda_runtime.h>
#include <cstdio>

// Paper constants (Section II.A, Fig.1)
static constexpr int SEGMENT_BITS      = 500;
static constexpr int PAYLOAD_NT        = 120;
static constexpr int ADDRESS_NT        = 9;
static constexpr int PRIMER_NT         = 20;
static constexpr int MAX_KEY_LEN       = 248;  // max LSBM substring < SEGMENT_BITS/2
static constexpr int MAX_VAL_LEN       = 12;   // max Huffman depth (ceil(log2(512))+pad)
static constexpr int MAX_TABLE_ENTRIES = 512;  // filter caps at 512 entries

// encode kernel block size: 256 threads → 8 warps/block, ~67% SM occupancy
static constexpr int ENCODE_BLOCK = 256;

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err__ = (call);                                               \
        if (err__ != cudaSuccess) {                                               \
            printf("CUDA error %s:%d -> %s\n", __FILE__, __LINE__,               \
                   cudaGetErrorString(err__));                                    \
            return 1;                                                             \
        }                                                                         \
    } while (0)

// Step 1
extern "C" __global__ void bytes_to_bits_kernel(
    const unsigned char* in_bytes, int n_bytes,
    unsigned char* out_bits);

// Step 2-3 -- paper-faithful: returns L_pref_max and L_suf_max separately
// so the host can enumerate every valid prefix/suffix length and add them
// all to map1 (paper Step 5: "retain the suffix AND prefix strings").
extern "C" __global__ void lsbm_parallel_kernel(
    const unsigned char* bits, int n_bits,
    int* out_L_pref_max,   // [n_segs] largest L with prefix-L count >= 2
    int* out_L_suf_max);   // [n_segs] largest L with suffix-L count >= 2

// Step 7 – RSTE greedy encoder
// Launch: <<<(n_segs+ENCODE_BLOCK-1)/ENCODE_BLOCK, ENCODE_BLOCK>>>
// Shared memory: compact table loaded cooperatively by all threads in block.
// Each thread handles 1 segment; single sorted scan replaces O(max_klen*n) double loop.
extern "C" __global__ void rste_encode_kernel(
    const unsigned char* bits,      int n_bits,
    const unsigned char* d_keys,    // [n_entries * MAX_KEY_LEN] sorted longest-first
    const int*           d_key_len, // [n_entries]
    const unsigned char* d_vals,    // [n_entries * MAX_VAL_LEN]
    const int*           d_val_len, // [n_entries]
    int                  n_entries,
    int                  n_segs,
    const int*           seg_offsets,
    char*                out_dna,
    int*                 seg_dna_len);

// Fallback quaternary encoder
extern "C" __global__ void bits_to_dna_quaternary_kernel(
    const unsigned char* bits, int n_bits,
    char* out_dna, int n_nt);

// Step 8 helper — constraint check (1 thread per 120-nt block, padding-aware)
extern "C" __global__ void gc_rll_end_kernel(
    const char* payload_flat,
    const int*  real_lens,             // actual non-padded length per block
    int seq_count, int seq_len,
    int* gc_out, int* rll_out, int* endgc_out);
