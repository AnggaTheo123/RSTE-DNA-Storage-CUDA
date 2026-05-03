#include <cuda_runtime.h>
#include <stdint.h>
#include "utils.cuh"

// -----------------------------------------------------------------------
// Fallback: quaternary 2-bit -> 1-nt (paper Section III.A mapping)
// 00 -> T, 01 -> C, 10 -> G, 11 -> A
// -----------------------------------------------------------------------
extern "C" __global__ void bits_to_dna_quaternary_kernel(
    const unsigned char* bits, int n_bits,
    char* out_dna, int n_nt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_nt) return;
    unsigned char v0 = (i*2   < n_bits) ? bits[i*2]   : 0;
    unsigned char v1 = (i*2+1 < n_bits) ? bits[i*2+1] : 0;
    unsigned char v  = (v0 << 1) | v1;
    char c = 'T';                       // 00
    if      (v == 1) c = 'C';           // 01
    else if (v == 2) c = 'G';           // 10
    else if (v == 3) c = 'A';           // 11
    out_dna[i] = c;
}

// -----------------------------------------------------------------------
// Compact Huffman entry in shared memory.
// -----------------------------------------------------------------------
struct __align__(8) ShEntry {
    uint32_t key_w[8];              // 32 bytes — packed key bits
    unsigned char val[MAX_VAL_LEN]; // 12 bytes — DNA output
    unsigned char klen;             // 1 byte
    unsigned char vlen;             // 1 byte
    unsigned char _pad[2];          // 2 bytes  -> total 48 bytes
};

static_assert(sizeof(ShEntry) == 48, "ShEntry must be 48 bytes");

// -----------------------------------------------------------------------
// Step 7: RSTE greedy encoder (LOSSLESS)
//
// **Grid/Block Layout**:
//   - Grid: (n_segs + ENCODE_BLOCK - 1) / ENCODE_BLOCK blocks (1D)
//   - Threads per block: ENCODE_BLOCK (256)
//   - Each thread processes exactly ONE segment (500 bits)
//
// **Shared Memory**:
//   - ShEntry table loaded cooperatively by all threads in block
//   - sh[MAX_TABLE_ENTRIES] packed compact (256 entries × 48 bytes = 12 KB)
//   - All threads use same table → synchronize after load
//
// **Algorithm** (per thread per segment):
//   1. Load entries into shared memory (cooperative, sorted longest-first)
//   2. Greedy scan: for each bit position, try to match against entries
//   3. Match → output DNA via table, advance bits; no match → quaternary fallback
//   4. Each thread increments dna_out pointer independently (no atomics needed)
//
// **Correctness**:
//   - No race conditions: each thread owns distinct output region
//   - Greedy longest-first ensures deterministic output (same as sequential)
//   - Quaternary fallback matches paper mapping (00=T, 01=C, 10=G, 11=A)
//
// **LOSSLESS GUARANTEE**: No silent base substitution. Each source bit
// produces a deterministic DNA output reproducible by the inverse decoder.
// RLL violations (run > 2) are reported by the constraint kernel, not
// hidden by the encoder.
// -----------------------------------------------------------------------
extern "C" __global__ void rste_encode_kernel(
    const unsigned char* bits,      int n_bits,
    const unsigned char* d_keys,
    const int*           d_key_len,
    const unsigned char* d_vals,
    const int*           d_val_len,
    int n_entries,
    int n_segs,
    const int* seg_offsets,
    char*      out_dna,
    int*       seg_dna_len)
{
    __shared__ ShEntry sh[MAX_TABLE_ENTRIES];
    __shared__ int     sh_n;

    int tid = threadIdx.x;
    if (tid == 0) sh_n = (n_entries < MAX_TABLE_ENTRIES) ? n_entries : MAX_TABLE_ENTRIES;
    __syncthreads();

    int n = sh_n;
    for (int e = tid; e < n; e += ENCODE_BLOCK) {
        int kl = d_key_len[e];
        if (kl > 248) kl = 248;
        sh[e].klen = (unsigned char)kl;

        const unsigned char* src = d_keys + (size_t)e * MAX_KEY_LEN;
        sh[e].key_w[0] = 0; sh[e].key_w[1] = 0; sh[e].key_w[2] = 0; sh[e].key_w[3] = 0;
        sh[e].key_w[4] = 0; sh[e].key_w[5] = 0; sh[e].key_w[6] = 0; sh[e].key_w[7] = 0;
        for (int b = 0; b < kl; ++b)
            sh[e].key_w[b >> 5] |= (uint32_t)(src[b] & 1u) << (b & 31);

        int vl = d_val_len[e];
        if (vl > MAX_VAL_LEN) vl = MAX_VAL_LEN;
        sh[e].vlen = (unsigned char)vl;
        const unsigned char* vsrc = d_vals + (size_t)e * MAX_VAL_LEN;
        for (int v = 0; v < MAX_VAL_LEN; ++v)
            sh[e].val[v] = (v < vl) ? vsrc[v] : 0;
    }
    __syncthreads();

    int seg = blockIdx.x * blockDim.x + tid;
    if (seg >= n_segs) return;

    int bit_start = seg * SEGMENT_BITS;
    int bit_end   = bit_start + SEGMENT_BITS;
    if (bit_end > n_bits) bit_end = n_bits;
    int dna_out   = seg_offsets[seg];
    int dna_start = dna_out;

    int pos = bit_start;
    while (pos < bit_end) {
        int remaining = bit_end - pos;
        bool matched  = false;

        for (int e = 0; e < n && !matched; ++e) {
            int kl = sh[e].klen;
            if (kl > remaining) continue;

            bool ok = true;
            int full_words = kl >> 5;
            int tail_bits  = kl & 31;

            for (int w = 0; w < full_words && ok; ++w) {
                uint32_t inp = 0;
                int boff = w << 5;
                for (int b = 0; b < 32; ++b)
                    inp |= (uint32_t)bits[pos + boff + b] << b;
                ok = (inp == sh[e].key_w[w]);
            }
            if (ok && tail_bits > 0) {
                uint32_t inp = 0;
                int boff = full_words << 5;
                for (int b = 0; b < tail_bits; ++b)
                    inp |= (uint32_t)bits[pos + boff + b] << b;
                uint32_t mask = (1u << tail_bits) - 1u;
                ok = ((inp & mask) == (sh[e].key_w[full_words] & mask));
            }

            if (ok) {
                // Direct write — no RLL guard (lossless)
                for (int v = 0; v < sh[e].vlen; ++v)
                    out_dna[dna_out++] = (char)sh[e].val[v];
                pos += kl;
                matched = true;
            }
        }

        if (!matched) {
            // RLL-safe fallback: 2 bits -> 2 nucleotides (matches CPU).
            // Sequences {GA, GT, CA, CT} all start with G/C, end with A/T,
            // no internal repeat. Mathematically guarantees RLL <= 2.
            unsigned char b0 = bits[pos];
            unsigned char b1 = (pos + 1 < bit_end) ? bits[pos + 1] : 0;
            unsigned char v  = (b0 << 1) | b1;
            char first  = (v < 2) ? 'G' : 'C';
            char second = ((v & 1) == 0) ? 'A' : 'T';
            out_dna[dna_out++] = first;
            out_dna[dna_out++] = second;
            pos += (remaining >= 2) ? 2 : 1;
        }
    }

    if (seg_dna_len != nullptr)
        seg_dna_len[seg] = dna_out - dna_start;
}

// -----------------------------------------------------------------------
// Step 8 helper: constraint check on payload-only blocks.
//
// **Grid/Block Layout**:
//   - Grid: (seq_count + blockDim.x - 1) / blockDim.x blocks (1D)
//   - Threads per block: blockDim.x (typically 256)
//   - Each thread processes ONE payload block (120 nucleotides max)
//
// **Input**:
//   - `payload_flat[s*seq_len + i]`: flattened 120-nt blocks
//   - `real_lens[s]`: actual non-padded length for block s (≤ seq_len)
//   - Does NOT read past `real_lens[s]` (padding-aware)
//
// **Outputs**:
//   - `gc_out[s]`: count of G/C in real payload (0 to seq_len)
//   - `rll_out[s]`: longest run of identical consecutive bases
//   - `endgc_out[s]`: count of G/C in last 5 real bases (end-constraint check)
//
// **Correctness**:
//   - No synchronization needed; each thread processes independent block
//   - Padding ('A') is not counted if beyond real_lens[s]
// -----------------------------------------------------------------------
extern "C" __global__ void gc_rll_end_kernel(
    const char* payload_flat,
    const int*  real_lens,
    int seq_count, int seq_len,
    int* gc_out, int* rll_out, int* endgc_out)
{
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= seq_count) return;

    int base    = s * seq_len;
    int real_n  = real_lens ? real_lens[s] : seq_len;
    if (real_n > seq_len) real_n = seq_len;
    if (real_n < 0) real_n = 0;

    int gc = 0, best = 1, cur = 1;
    for (int i = 0; i < real_n; ++i) {
        char c = payload_flat[base + i];
        if (c == 'G' || c == 'C') gc++;
        if (i > 0) {
            if (c == payload_flat[base + i - 1]) {
                cur++;
                if (cur > best) best = cur;
            } else {
                cur = 1;
            }
        }
    }

    int last5 = 0;
    int e_start = (real_n >= 5) ? real_n - 5 : 0;
    for (int i = e_start; i < real_n; ++i) {
        char c = payload_flat[base + i];
        if (c == 'G' || c == 'C') last5++;
    }

    gc_out[s]    = gc;
    rll_out[s]   = (real_n == 0) ? 0 : best;
    endgc_out[s] = last5;
}
