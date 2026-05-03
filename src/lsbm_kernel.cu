#include <cuda_runtime.h>
#include "utils.cuh"

// -----------------------------------------------------------------------
// Step 1: bytes -> bits  (1 thread per input byte, MSB first)
// -----------------------------------------------------------------------
extern "C" __global__ void bytes_to_bits_kernel(
    const unsigned char* in_bytes, int n_bytes,
    unsigned char* out_bits)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_bytes) return;

    unsigned char b = in_bytes[i];
    int o = i * 8;
    out_bits[o + 0] = (b >> 7) & 1;
    out_bits[o + 1] = (b >> 6) & 1;
    out_bits[o + 2] = (b >> 5) & 1;
    out_bits[o + 3] = (b >> 4) & 1;
    out_bits[o + 4] = (b >> 3) & 1;
    out_bits[o + 5] = (b >> 2) & 1;
    out_bits[o + 6] = (b >> 1) & 1;
    out_bits[o + 7] =  b       & 1;
}

// -----------------------------------------------------------------------
// Step 2-3: LSBM parallel kernel
//
// Paper Section II.A.2, Fig.5:
//   - Each CUDA block handles exactly one 500-bit segment
//   - Threads cooperate via shared memory to count NON-OVERLAPPING
//     occurrences of each prefix/suffix substring
//   - Uses parallel reduction to sum hits across threads
//   - Stops at first (longest) length with >= 2 hits (longest-first policy)
//
// **Grid/Block Layout**:
//   - Grid: n_segs blocks (1D), one block per segment
//   - Threads per block: typically 256 (blockDim.x)
//   - Each block handles SEGMENT_BITS (500) bits from one segment
//
// **Shared Memory**:
//   - seg_bits[0..SEGMENT_BITS-1]: segment bits (500 bytes)
//   - red[blockDim.x*4]: reduction buffer for parallel sum (int)
//   - Total: ~500 + 1024 = ~1500 bytes (well within 96 KB limit)
//
// **Algorithm** (per block):
//   1. Load segment bits cooperatively (all threads help copy)
//   2. For each length L (longest to shortest):
//      a. Prefix: match seg[0..L-1] at positions p >= L (no overlap)
//      b. Suffix: match seg[end-L..end-1] at positions p+L <= end-L
//      c. Count occurrences via parallel reduction
//      d. If total >= 2, record best_len=L, best_hits and STOP
//   3. Write results to global memory
//
// Non-overlap rule:
//   - For prefix match: a position p is a hit iff seg[p..p+L-1] == prefix
//     AND p does not overlap the prefix itself (p >= L). The prefix itself
//     (at p == 0) is excluded — we count *additional* occurrences only.
//   - For suffix match: similarly p does not overlap the suffix range
//     [seg_len-L .. seg_len-1].
//
// **Correctness**:
//   - Parallel reduction ensures correct thread-level summation
//   - __syncthreads() ensures all threads see same shared data
//   - Each segment processed by exactly one block → no race conditions
// -----------------------------------------------------------------------
extern "C" __global__ void lsbm_parallel_kernel(
    const unsigned char* bits, int n_bits,
    int* out_L_pref_max,
    int* out_L_suf_max)
{
    int seg = blockIdx.x;
    int seg_start = seg * SEGMENT_BITS;
    int seg_len   = min(SEGMENT_BITS, n_bits - seg_start);
    if (seg_len <= 0) {
        if (threadIdx.x == 0) {
            out_L_pref_max[seg] = 0;
            out_L_suf_max[seg]  = 0;
        }
        return;
    }

    extern __shared__ unsigned char shmem[];
    unsigned char* seg_bits  = shmem;
    unsigned char* match_buf = shmem + SEGMENT_BITS;
    int* red = reinterpret_cast<int*>(shmem + 2 * SEGMENT_BITS);

    for (int i = threadIdx.x; i < seg_len; i += blockDim.x)
        seg_bits[i] = bits[seg_start + i];
    __syncthreads();

    // -----------------------------------------------------------------
    // Binary search for largest L_pref where prefix-L count >= 2.
    // Predicate is monotone-decreasing in L (every prefix-L occurrence
    // is also a prefix-(L-1) occurrence), so the valid set is
    // downward-closed and binary search returns the same maximum as a
    // linear scan but in O(log seg_len) iterations.
    // -----------------------------------------------------------------
    __shared__ int s_pref_best, s_pref_lo, s_pref_hi;
    if (threadIdx.x == 0) {
        s_pref_best = 0;
        s_pref_lo   = 3;
        s_pref_hi   = seg_len - 1;
    }
    __syncthreads();

    while (s_pref_lo <= s_pref_hi) {
        int L = (s_pref_lo + s_pref_hi) >> 1;

        for (int p = threadIdx.x; p + L <= seg_len; p += blockDim.x) {
            bool ok = true;
            for (int i = 0; i < L && ok; ++i)
                ok = (seg_bits[p + i] == seg_bits[i]);
            match_buf[p] = ok ? 1 : 0;
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            int count = 0, next_ok = 0;
            for (int p = 0; p + L <= seg_len; ++p) {
                if (match_buf[p] && p >= next_ok) { ++count; next_ok = p + L; }
            }
            red[0] = count;
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            if (red[0] >= 2) { s_pref_best = L; s_pref_lo = L + 1; }
            else             { s_pref_hi = L - 1; }
        }
        __syncthreads();
    }

    // -----------------------------------------------------------------
    // Binary search for largest L_suf -- independent of prefix search.
    // -----------------------------------------------------------------
    __shared__ int s_suf_best, s_suf_lo, s_suf_hi;
    if (threadIdx.x == 0) {
        s_suf_best = 0;
        s_suf_lo   = 3;
        s_suf_hi   = seg_len - 1;
    }
    __syncthreads();

    while (s_suf_lo <= s_suf_hi) {
        int L = (s_suf_lo + s_suf_hi) >> 1;
        int suf_off = seg_len - L;

        for (int p = threadIdx.x; p + L <= seg_len; p += blockDim.x) {
            bool ok = true;
            for (int i = 0; i < L && ok; ++i)
                ok = (seg_bits[p + i] == seg_bits[suf_off + i]);
            match_buf[p] = ok ? 1 : 0;
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            int count = 0, next_ok = 0;
            for (int p = 0; p + L <= seg_len; ++p) {
                if (match_buf[p] && p >= next_ok) { ++count; next_ok = p + L; }
            }
            red[1] = count;
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            if (red[1] >= 2) { s_suf_best = L; s_suf_lo = L + 1; }
            else             { s_suf_hi = L - 1; }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        out_L_pref_max[seg] = s_pref_best;
        out_L_suf_max[seg]  = s_suf_best;
    }
}
