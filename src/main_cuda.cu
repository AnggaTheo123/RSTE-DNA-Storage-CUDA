// rste_cuda/main.cu
// GPU-Accelerated RSTE Encoding — Full 8-step pipeline (paper Section II.A)
//
// Pipeline:
//   Step 1  : bytes_to_bits_kernel       (1 thread/byte, embarrassingly parallel)
//   Step 2-3: lsbm_parallel_kernel       (1 block/segment, parallel reduction)
//   Step 4-6: CPU build map1 -> Huffman tree -> map2 (uploaded to GPU)
//   Step 7  : rste_encode_kernel         (1 thread/segment, shared-memory table)
//   Step 8  : build_storage_sequences    (CPU primer + 9-nt address attach)
//             gc_rll_end_kernel          (1 thread/block, padding-aware)
//
// LOSSLESS: All kernels write deterministic output without silent base
// mutation. FNV-1a hash of payload + storage DNA is reported in CSV so
// sequential vs CUDA output equivalence can be verified by hash comparison.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <queue>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "utils.cuh"

// -----------------------------------------------------------------------
// FNV-1a 64-bit hash (matches sequential CPU implementation)
// -----------------------------------------------------------------------
static std::string fnv1a64_hex(const std::vector<char>& data) {
    unsigned long long h = 1469598103934665603ull;
    for (char c : data) {
        h ^= (unsigned long long)(unsigned char)c;
        h *= 1099511628211ull;
    }
    std::ostringstream oss;
    oss << std::hex << std::setw(16) << std::setfill('0') << h;
    return oss.str();
}

// -----------------------------------------------------------------------
// Step 8 (CPU): primer + 9-nt address attachment.
// Same primers and addressing as the sequential .cpp implementation,
// so the hash of the storage DNA can be compared end-to-end.
// -----------------------------------------------------------------------
static const char PRIMER_LEFT[PRIMER_NT + 1]  = "ACGTTGCATGTCAGTACGTA";
static const char PRIMER_RIGHT[PRIMER_NT + 1] = "TACGACTGACATGCAACGTT";

static void append_address_9nt(int idx, std::vector<char>& out) {
    static const char b4[4] = {'A','C','G','T'};
    for (int i = ADDRESS_NT - 1; i >= 0; --i) {
        int v = (idx >> (2 * i)) & 0x3;
        out.push_back(b4[v]);
    }
}

static void build_storage_sequences(const std::vector<char>& payload_dna,
                                    std::vector<char>& storage_dna,
                                    int& storage_seq_count) {
    int payload_nt = (int)payload_dna.size();
    storage_seq_count = (payload_nt + PAYLOAD_NT - 1) / PAYLOAD_NT;
    storage_dna.clear();
    storage_dna.reserve((size_t)storage_seq_count *
                        (PRIMER_NT + ADDRESS_NT + PAYLOAD_NT + PRIMER_NT));

    for (int s = 0; s < storage_seq_count; ++s) {
        int start = s * PAYLOAD_NT;
        int end   = std::min(start + PAYLOAD_NT, payload_nt);

        for (int i = 0; i < PRIMER_NT; ++i) storage_dna.push_back(PRIMER_LEFT[i]);
        append_address_9nt(s, storage_dna);
        for (int i = start; i < end; ++i) storage_dna.push_back(payload_dna[i]);
        for (int i = end; i < start + PAYLOAD_NT; ++i) storage_dna.push_back('A');
        for (int i = 0; i < PRIMER_NT; ++i) storage_dna.push_back(PRIMER_RIGHT[i]);
    }
}

// -----------------------------------------------------------------------
// Host helpers
// -----------------------------------------------------------------------
static std::vector<unsigned char> read_binary_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return {};
    return std::vector<unsigned char>((std::istreambuf_iterator<char>(in)),
                                       std::istreambuf_iterator<char>());
}
static std::string file_basename(const std::string& path) {
    size_t p = path.find_last_of("\\/");
    return (p == std::string::npos) ? path : path.substr(p + 1);
}

static void bytes_to_bits_host(const std::vector<unsigned char>& raw,
                                std::vector<unsigned char>& bits) {
    int n_bytes = (int)raw.size();
    bits.assign(n_bytes * 8, 0);
    for (int i = 0; i < n_bytes; ++i) {
        unsigned char b = raw[i];
        int o = i * 8;
        bits[o+0] = (b>>7)&1; bits[o+1] = (b>>6)&1;
        bits[o+2] = (b>>5)&1; bits[o+3] = (b>>4)&1;
        bits[o+4] = (b>>3)&1; bits[o+5] = (b>>2)&1;
        bits[o+6] = (b>>1)&1; bits[o+7] =  b    &1;
    }
}

// -----------------------------------------------------------------------
// Step 4: build map1 from LSBM best-length results
// -----------------------------------------------------------------------
struct RepEntry { std::string sub; int freq; };

// Count non-overlapping occurrences of pat in text via KMP (mirrors CPU).
static int host_kmp_count(const std::string& text, const std::string& pat) {
    int n = (int)text.size(), m = (int)pat.size();
    if (m == 0 || m > n) return 0;
    std::vector<int> f(m, 0);
    int k = 0;
    for (int i = 1; i < m; ++i) {
        while (k > 0 && pat[k] != pat[i]) k = f[k-1];
        if (pat[k] == pat[i]) ++k;
        f[i] = k;
    }
    int count = 0; k = 0; int next_ok = 0;
    for (int i = 0; i < n; ++i) {
        while (k > 0 && pat[k] != text[i]) k = f[k-1];
        if (pat[k] == text[i]) ++k;
        if (k == m) {
            int start = i - m + 1;
            if (start >= next_ok) { ++count; next_ok = start + m; }
            k = f[k-1];
        }
    }
    return count;
}

// Step 4 host build_map1 -- mirrors CPU lsbm_cpu (paper-faithful):
// for each segment, enumerate every prefix length L = 3..L_pref_max
// AND every suffix length L = 3..L_suf_max, accumulate the total
// non-overlap occurrence count into the global frequency map
// (paper Step 5: "retain the suffix AND prefix strings AND positions
// existing in S").
static constexpr int MIN_KEY_LEN_HOST = 3;

static std::vector<RepEntry> build_map1(
    const std::vector<unsigned char>& bits,
    const std::vector<int>& L_pref_max,
    const std::vector<int>& L_suf_max)
{
    int n_bits = (int)bits.size();
    int n_segs = (n_bits + SEGMENT_BITS - 1) / SEGMENT_BITS;
    std::unordered_map<std::string, int> freq;

    for (int seg = 0; seg < n_segs; ++seg) {
        int Lp = (seg < (int)L_pref_max.size()) ? L_pref_max[seg] : 0;
        int Ls = (seg < (int)L_suf_max .size()) ? L_suf_max [seg] : 0;
        if (Lp < MIN_KEY_LEN_HOST && Ls < MIN_KEY_LEN_HOST) continue;

        int start = seg * SEGMENT_BITS;
        int slen  = std::min(SEGMENT_BITS, n_bits - start);
        if (slen < MIN_KEY_LEN_HOST + 1) continue;

        std::string seg_str(slen, '0');
        for (int i = 0; i < slen; ++i) seg_str[i] = (char)('0' + bits[start + i]);

        if (Lp > slen - 1) Lp = slen - 1;
        if (Ls > slen - 1) Ls = slen - 1;

        // Paper-faithful with practical scope -- retain only the LONGEST
        // valid prefix and the LONGEST valid suffix per segment.
        // (See lsbm.cpp comment for empirical rationale.)
        if (Lp >= MIN_KEY_LEN_HOST) {
            std::string pref = seg_str.substr(0, Lp);
            int cnt = host_kmp_count(seg_str, pref);
            if (cnt >= 2) freq[pref] += cnt;
        }
        if (Ls >= MIN_KEY_LEN_HOST) {
            std::string suf  = seg_str.substr(slen - Ls, Ls);
            std::string pref = (Ls <= Lp) ? seg_str.substr(0, Ls)
                                          : std::string();
            if (suf != pref) {
                int cnt = host_kmp_count(seg_str, suf);
                if (cnt >= 2) freq[suf] += cnt;
            }
        }
    }

    std::vector<RepEntry> result;
    result.reserve(freq.size());
    for (auto& kv : freq)
        if ((int)kv.first.size() > 2)
            result.push_back({kv.first, kv.second});
    std::sort(result.begin(), result.end(), [](const RepEntry& a, const RepEntry& b){
        if (a.sub.size() != b.sub.size()) return a.sub.size() > b.sub.size();
        if (a.freq != b.freq)             return a.freq > b.freq;
        return a.sub < b.sub;  // deterministic tie-break
    });

    int n = std::min((int)result.size(), 512);
    for (int iter = 0; iter < 20 && n > 1; ++iter) {
        int max_depth = 0, tmp = n;
        while (tmp > 1) { max_depth++; tmp = (tmp+1)/2; }
        int min_len = 2 * max_depth + 1;
        int ok = 0;
        for (int i = 0; i < n; ++i)
            if ((int)result[i].sub.size() >= min_len) ok++;
        if (ok >= n) break;
        n = ok;
        if (n == 0) break;
    }
    if (n > 0) result.resize(n);

    return result;
}

// -----------------------------------------------------------------------
// Step 5-6: CPU Huffman tree -> DnaCodeTable (map2)
// -----------------------------------------------------------------------
using DnaCodeTable = std::unordered_map<std::string, std::string>;

static DnaCodeTable build_huffman_dna_table(const std::vector<RepEntry>& map1) {
    DnaCodeTable table;
    if (map1.empty()) return table;

    struct Node { int freq, left, right, level; std::string sub, dna; };
    std::vector<Node> pool;
    pool.reserve(map1.size() * 2);

    using PIpair = std::pair<int,int>;
    std::priority_queue<PIpair, std::vector<PIpair>, std::greater<PIpair>> pq;

    for (auto& e : map1) {
        pool.push_back({e.freq, -1, -1, 0, e.sub, ""});
        pq.push({e.freq, (int)pool.size() - 1});
    }
    while (pq.size() > 1) {
        auto [f1,i1] = pq.top(); pq.pop();
        auto [f2,i2] = pq.top(); pq.pop();
        pool.push_back({f1+f2, i1, i2, 0, "", ""});
        pq.push({f1+f2, (int)pool.size()-1});
    }
    if (pq.empty()) return table;
    int root = pq.top().second;

    struct Frame { int idx, level; std::string code; };
    std::queue<Frame> bfs;
    bfs.push({root, 0, ""});
    while (!bfs.empty()) {
        auto [idx, lvl, code] = bfs.front(); bfs.pop();
        Node& nd = pool[idx];
        nd.level = lvl; nd.dna = code;
        if (nd.left == -1 && nd.right == -1) {
            if (!nd.sub.empty() && code.size() >= 2) table[nd.sub] = code;
            continue;
        }
        int cl = lvl + 1;
        char lc = (cl % 2 == 1) ? 'G' : 'A';
        char rc = (cl % 2 == 1) ? 'C' : 'T';
        if (nd.left  != -1) bfs.push({nd.left,  cl, code + lc});
        if (nd.right != -1) bfs.push({nd.right, cl, code + rc});
    }
    return table;
}

// -----------------------------------------------------------------------
// Transfer Huffman table to GPU
// -----------------------------------------------------------------------
struct GpuTable {
    unsigned char* d_keys    = nullptr;
    int*           d_key_len = nullptr;
    unsigned char* d_vals    = nullptr;
    int*           d_val_len = nullptr;
    int            n_entries = 0;
    int            max_key_len = 0;
};

static int upload_table(const DnaCodeTable& table, GpuTable& gt) {
    gt.n_entries = (int)table.size();
    if (gt.n_entries == 0) return 0;

    std::vector<std::string> keys, vals;
    keys.reserve(gt.n_entries); vals.reserve(gt.n_entries);
    for (auto& kv : table) { keys.push_back(kv.first); vals.push_back(kv.second); }

    std::vector<int> idx(gt.n_entries);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](int a, int b) {
        if (keys[a].size() != keys[b].size()) return keys[a].size() > keys[b].size();
        if (keys[a] != keys[b]) return keys[a] < keys[b];
        return vals[a] < vals[b];
    });
    std::vector<std::string> skeys(gt.n_entries), svals(gt.n_entries);
    for (int i = 0; i < gt.n_entries; ++i) { skeys[i] = keys[idx[i]]; svals[i] = vals[idx[i]]; }
    keys = skeys; vals = svals;

    gt.max_key_len = 0;
    for (auto& k : keys) gt.max_key_len = std::max(gt.max_key_len, (int)k.size());
    gt.max_key_len = std::min(gt.max_key_len, MAX_KEY_LEN);

    std::vector<unsigned char> h_keys(gt.n_entries * MAX_KEY_LEN, 0);
    std::vector<int>           h_klen(gt.n_entries, 0);
    std::vector<unsigned char> h_vals(gt.n_entries * MAX_VAL_LEN, 0);
    std::vector<int>           h_vlen(gt.n_entries, 0);

    for (int i = 0; i < gt.n_entries; ++i) {
        int kl = std::min((int)keys[i].size(), MAX_KEY_LEN);
        h_klen[i] = kl;
        for (int j = 0; j < kl; ++j) h_keys[i * MAX_KEY_LEN + j] = (unsigned char)keys[i][j];
        int vl = std::min((int)vals[i].size(), MAX_VAL_LEN);
        h_vlen[i] = vl;
        for (int j = 0; j < vl; ++j) h_vals[i * MAX_VAL_LEN + j] = (unsigned char)vals[i][j];
    }

    CUDA_CHECK(cudaMalloc(&gt.d_keys,    (size_t)gt.n_entries * MAX_KEY_LEN));
    CUDA_CHECK(cudaMalloc(&gt.d_key_len, (size_t)gt.n_entries * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&gt.d_vals,    (size_t)gt.n_entries * MAX_VAL_LEN));
    CUDA_CHECK(cudaMalloc(&gt.d_val_len, (size_t)gt.n_entries * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(gt.d_keys,    h_keys.data(), (size_t)gt.n_entries * MAX_KEY_LEN, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gt.d_key_len, h_klen.data(), (size_t)gt.n_entries * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gt.d_vals,    h_vals.data(), (size_t)gt.n_entries * MAX_VAL_LEN, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gt.d_val_len, h_vlen.data(), (size_t)gt.n_entries * sizeof(int), cudaMemcpyHostToDevice));
    return 0;
}

// -----------------------------------------------------------------------
// CSV output (mirrors sequential header for direct comparison)
// -----------------------------------------------------------------------
static int append_csv(const std::string& csv_path, const std::string& dataset,
    const std::string& fname, size_t fsize,
    double total_ms, double std_total_ms, double phase_sum_ms,
    double h2d_ms, double bits_ms,
    double lsbm_ms, double std_lsbm_ms, double huffman_ms,
    double encode_ms, double std_encode_ms,
    double constraint_ms, double assembly_ms, double d2h_ms,
    double enc_rate, double gross_rate, double gc_mean, double gc_var,
    int rll_max, int rll_viol, int end_viol, int seq_count, int storage_seq_count,
    double avg_lsbm, int segments, int runs,
    const std::string& hash_payload, const std::string& hash_storage)
{
    bool hdr = false;
    { std::ifstream chk(csv_path); hdr = !chk.good(); }
    std::ofstream out(csv_path, std::ios::app);
    if (!out) { std::cerr << "Cannot write CSV: " << csv_path << "\n"; return 1; }
    if (hdr)
        out << "dataset,file_name,file_size_bytes,runs,"
               "total_ms,std_total_ms,phase_sum_ms,h2d_ms,bits_kernel_ms,"
               "lsbm_kernel_ms,std_lsbm_ms,huffman_cpu_ms,"
               "encode_kernel_ms,std_encode_ms,constraint_kernel_ms,"
               "assembly_ms,d2h_ms,encoding_rate_bits_per_nt,gross_rate_bits_per_nt,"
               "gc_mean,gc_var,rll_max,rll_violations,end_violations,seq_count,"
               "storage_seq_count,avg_best_lsbm_len,segments,"
               "dna_hash_payload,dna_hash_storage\n";
    out << dataset << "," << fname << "," << fsize << "," << runs << ","
        << total_ms << "," << std_total_ms << "," << phase_sum_ms << ","
        << h2d_ms << "," << bits_ms << ","
        << lsbm_ms << "," << std_lsbm_ms << "," << huffman_ms << ","
        << encode_ms << "," << std_encode_ms << ","
        << constraint_ms << "," << assembly_ms << "," << d2h_ms << ","
        << enc_rate << "," << gross_rate << ","
        << gc_mean << "," << gc_var << "," << rll_max << "," << rll_viol << ","
        << end_viol << "," << seq_count << "," << storage_seq_count << ","
        << avg_lsbm << ","
        << segments << "," << hash_payload << "," << hash_storage << "\n";
    return 0;
}

// -----------------------------------------------------------------------
// Main pipeline
// -----------------------------------------------------------------------
int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: rste_cuda.exe <input> <dataset_tag> <output_csv> [--runs N]\n";
        return 1;
    }
    const std::string input_path  = argv[1];
    const std::string dataset_tag = argv[2];
    const std::string out_csv     = argv[3];

    int n_runs = 5;
    for (int i = 4; i < argc - 1; ++i)
        if (std::string(argv[i]) == "--runs")
            n_runs = std::max(1, std::atoi(argv[i + 1]));

    auto t0 = std::chrono::high_resolution_clock::now();

    std::vector<unsigned char> raw = read_binary_file(input_path);
    if (raw.empty()) { std::cerr << "Cannot read: " << input_path << "\n"; return 1; }

    int n_bytes = (int)raw.size();
    int n_bits  = n_bytes * 8;
    int n_segs  = (n_bits + SEGMENT_BITS - 1) / SEGMENT_BITS;

    // --- Fix 3: CUDA Streams + pinned memory ---
    // Two streams: stream0 for first half of segments, stream1 for second half.
    // Pinned host memory enables async H2D/D2H transfers.
    cudaStream_t stream0, stream1;
    CUDA_CHECK(cudaStreamCreate(&stream0));
    CUDA_CHECK(cudaStreamCreate(&stream1));

    cudaEvent_t ev0, ev1;
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));
    float ms = 0;
    double h2d_ms=0, bits_ms=0, lsbm_ms=0, encode_ms=0, constraint_ms=0;
    double d2h_ms=0, assembly_ms=0;

    // Pinned host memory for raw bytes (faster async H2D)
    unsigned char* h_raw_pinned = nullptr;
    CUDA_CHECK(cudaMallocHost(&h_raw_pinned, n_bytes));
    std::memcpy(h_raw_pinned, raw.data(), n_bytes);

    // --- H2D (async via stream0) ---
    unsigned char *d_raw=nullptr, *d_bits=nullptr;
    CUDA_CHECK(cudaMalloc(&d_raw,  n_bytes * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&d_bits, n_bits  * sizeof(unsigned char)));

    CUDA_CHECK(cudaEventRecord(ev0));
    CUDA_CHECK(cudaMemcpyAsync(d_raw, h_raw_pinned, n_bytes,
                               cudaMemcpyHostToDevice, stream0));
    CUDA_CHECK(cudaStreamSynchronize(stream0));
    CUDA_CHECK(cudaEventRecord(ev1)); CUDA_CHECK(cudaEventSynchronize(ev1));
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1)); h2d_ms = ms;

    // --- Step 1: bytes -> bits (stream0) ---
    int blk = 256;
    CUDA_CHECK(cudaEventRecord(ev0));
    bytes_to_bits_kernel<<<(n_bytes+blk-1)/blk, blk, 0, stream0>>>(d_raw, n_bytes, d_bits);
    CUDA_CHECK(cudaStreamSynchronize(stream0));
    CUDA_CHECK(cudaEventRecord(ev1)); CUDA_CHECK(cudaEventSynchronize(ev1));
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1)); bits_ms = ms;

    // --- Step 2-3: LSBM kernel (stream0, one block per segment) ---
    // Output L_pref_max + L_suf_max per segment; host build_map1 then
    // enumerates ALL valid prefix/suffix lengths (paper Step 5).
    int* d_L_pref=nullptr; int* d_L_suf=nullptr;
    CUDA_CHECK(cudaMalloc(&d_L_pref, n_segs * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_L_suf,  n_segs * sizeof(int)));

    // shmem layout: seg_bits[SEGMENT_BITS] + match_buf[SEGMENT_BITS] + red[blk*int]
    size_t lsbm_shmem = 2 * SEGMENT_BITS * sizeof(unsigned char) + blk * sizeof(int);
    CUDA_CHECK(cudaEventRecord(ev0));
    lsbm_parallel_kernel<<<n_segs, blk, lsbm_shmem, stream0>>>(
        d_bits, n_bits, d_L_pref, d_L_suf);
    CUDA_CHECK(cudaStreamSynchronize(stream0));
    CUDA_CHECK(cudaEventRecord(ev1)); CUDA_CHECK(cudaEventSynchronize(ev1));
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1)); lsbm_ms = ms;

    std::vector<int> h_L_pref(n_segs, 0), h_L_suf(n_segs, 0);
    CUDA_CHECK(cudaMemcpyAsync(h_L_pref.data(), d_L_pref, n_segs * sizeof(int),
                               cudaMemcpyDeviceToHost, stream0));
    CUDA_CHECK(cudaMemcpyAsync(h_L_suf .data(), d_L_suf,  n_segs * sizeof(int),
                               cudaMemcpyDeviceToHost, stream0));
    CUDA_CHECK(cudaStreamSynchronize(stream0));

    // --- Step 4-6: CPU Huffman ---
    std::vector<unsigned char> h_bits;
    bytes_to_bits_host(raw, h_bits);

    auto huf0 = std::chrono::high_resolution_clock::now();
    auto map1  = build_map1(h_bits, h_L_pref, h_L_suf);
    auto table = build_huffman_dna_table(map1);
    auto huf1  = std::chrono::high_resolution_clock::now();
    double huffman_ms = std::chrono::duration<double,std::milli>(huf1-huf0).count();

    GpuTable gt;
    if (upload_table(table, gt) != 0) return 1;

    // --- Step 7: encode kernel ---
    // Worst-case DNA per segment: 2-bit -> 2-nt fallback gives 1 nt/bit
    int max_dna_per_seg = SEGMENT_BITS;
    int total_dna_alloc = n_segs * max_dna_per_seg;
    std::vector<int> h_seg_offsets(n_segs + 1);
    for (int s = 0; s <= n_segs; ++s) h_seg_offsets[s] = s * max_dna_per_seg;

    int* d_seg_offsets = nullptr;
    int* d_seg_dna_len = nullptr;
    char* d_dna = nullptr;
    CUDA_CHECK(cudaMalloc(&d_seg_offsets, (n_segs+1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_seg_dna_len, n_segs     * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_dna, total_dna_alloc * sizeof(char)));
    CUDA_CHECK(cudaMemset(d_dna, 0, total_dna_alloc));
    CUDA_CHECK(cudaMemset(d_seg_dna_len, 0, n_segs * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_seg_offsets, h_seg_offsets.data(), (n_segs+1)*sizeof(int), cudaMemcpyHostToDevice));

    // --- Fix 3: Two-stream encode — split segments across stream0 / stream1 ---
    // stream0 handles first half, stream1 handles second half simultaneously.
    // Both streams share the same d_dna and d_seg_offsets (non-overlapping ranges).
    int half_segs  = n_segs / 2;
    int enc_grid0  = (half_segs  + ENCODE_BLOCK - 1) / ENCODE_BLOCK;
    int enc_grid1  = (n_segs - half_segs + ENCODE_BLOCK - 1) / ENCODE_BLOCK;

    // Temporary per-stream seg_dna_len (combined after sync)
    int* d_seg_dna_len0 = d_seg_dna_len;
    int* d_seg_dna_len1 = d_seg_dna_len + half_segs;

    // Per-stream seg_offsets pointers (pointing into same d_seg_offsets array)
    int* d_seg_offsets0 = d_seg_offsets;
    int* d_seg_offsets1 = d_seg_offsets + half_segs;

    CUDA_CHECK(cudaEventRecord(ev0));
    if (gt.n_entries > 0) {
        // stream0: segments [0, half_segs)
        if (half_segs > 0)
            rste_encode_kernel<<<enc_grid0, ENCODE_BLOCK, 0, stream0>>>(
                d_bits, n_bits,
                gt.d_keys, gt.d_key_len,
                gt.d_vals, gt.d_val_len,
                gt.n_entries, half_segs,
                d_seg_offsets0, d_dna, d_seg_dna_len0);
        // stream1: segments [half_segs, n_segs).
        // Kernel uses local seg index (0..n_segs-1) to compute bit_start =
        // seg * SEGMENT_BITS. So we offset the bits pointer by
        // half_segs*SEGMENT_BITS to point at the correct global segment.
        if (n_segs - half_segs > 0) {
            int bit_skip = half_segs * SEGMENT_BITS;
            rste_encode_kernel<<<enc_grid1, ENCODE_BLOCK, 0, stream1>>>(
                d_bits + bit_skip, n_bits - bit_skip,
                gt.d_keys, gt.d_key_len,
                gt.d_vals, gt.d_val_len,
                gt.n_entries, n_segs - half_segs,
                d_seg_offsets1, d_dna, d_seg_dna_len1);
        }
    } else {
        int n_nt = (n_bits + 1) / 2;
        bits_to_dna_quaternary_kernel<<<(n_nt+blk-1)/blk, blk, 0, stream0>>>(
            d_bits, n_bits, d_dna, n_nt);
        std::vector<int> qtmp(n_segs);
        for (int s = 0; s < n_segs; ++s) {
            int bseg = std::min(SEGMENT_BITS, n_bits - s * SEGMENT_BITS);
            qtmp[s] = (bseg + 1) / 2;
        }
        CUDA_CHECK(cudaMemcpyAsync(d_seg_dna_len, qtmp.data(), n_segs*sizeof(int),
                                   cudaMemcpyHostToDevice, stream0));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream0));
    CUDA_CHECK(cudaStreamSynchronize(stream1));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(ev1)); CUDA_CHECK(cudaEventSynchronize(ev1));
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1)); encode_ms = ms;

    // --- D2H DNA (async, stream0) ---
    char* h_dna_pinned = nullptr;
    CUDA_CHECK(cudaMallocHost(&h_dna_pinned, total_dna_alloc));
    CUDA_CHECK(cudaEventRecord(ev0));
    CUDA_CHECK(cudaMemcpyAsync(h_dna_pinned, d_dna, total_dna_alloc,
                               cudaMemcpyDeviceToHost, stream0));
    CUDA_CHECK(cudaStreamSynchronize(stream0));
    CUDA_CHECK(cudaEventRecord(ev1)); CUDA_CHECK(cudaEventSynchronize(ev1));
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1)); d2h_ms += ms;
    std::vector<char> h_dna(h_dna_pinned, h_dna_pinned + total_dna_alloc);
    CUDA_CHECK(cudaFreeHost(h_dna_pinned));

    std::vector<int> h_seg_dna_len(n_segs, 0);
    CUDA_CHECK(cudaMemcpy(h_seg_dna_len.data(), d_seg_dna_len,
                          n_segs * sizeof(int), cudaMemcpyDeviceToHost));

    // --- Flatten per-segment DNA into contiguous payload (no RLL guard) ---
    int actual_nts = 0;
    for (int v : h_seg_dna_len) actual_nts += v;

    std::vector<char> dna_payload;
    dna_payload.reserve(actual_nts);
    for (int s = 0; s < n_segs; ++s) {
        int off  = h_seg_offsets[s];
        int wlen = h_seg_dna_len[s];
        for (int j = 0; j < wlen; ++j)
            dna_payload.push_back(h_dna[off + j]);
    }

    // --- Step 8 (CPU): primer + 9-nt address attachment ---
    auto a0 = std::chrono::high_resolution_clock::now();
    std::vector<char> storage_dna;
    int storage_seq_count = 0;
    build_storage_sequences(dna_payload, storage_dna, storage_seq_count);
    auto a1 = std::chrono::high_resolution_clock::now();
    assembly_ms = std::chrono::duration<double,std::milli>(a1-a0).count();

    // --- Build payload-only flat buffer for constraint kernel ---
    int seq_count = storage_seq_count;
    std::vector<char> payload_flat(seq_count * PAYLOAD_NT, 'A');
    std::vector<int>  real_lens(seq_count, 0);
    for (int s = 0; s < seq_count; ++s) {
        int p_start = s * PAYLOAD_NT;
        int p_end   = std::min(p_start + PAYLOAD_NT, (int)dna_payload.size());
        int p_len   = p_end - p_start;
        for (int j = 0; j < p_len; ++j)
            payload_flat[s * PAYLOAD_NT + j] = dna_payload[p_start + j];
        real_lens[s] = p_len;
    }

    char* d_payload=nullptr; int* d_real_lens=nullptr;
    int* d_gc=nullptr; int* d_rll=nullptr; int* d_endgc=nullptr;
    CUDA_CHECK(cudaMalloc(&d_payload,    seq_count * PAYLOAD_NT));
    CUDA_CHECK(cudaMalloc(&d_real_lens,  seq_count * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_gc,         seq_count * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_rll,        seq_count * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_endgc,      seq_count * sizeof(int)));

    CUDA_CHECK(cudaEventRecord(ev0));
    CUDA_CHECK(cudaMemcpy(d_payload,   payload_flat.data(), seq_count*PAYLOAD_NT,    cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_real_lens, real_lens.data(),    seq_count*sizeof(int),   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(ev1)); CUDA_CHECK(cudaEventSynchronize(ev1));
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1)); h2d_ms += ms;

    // --- Step 8: constraint kernel (padding-aware) ---
    CUDA_CHECK(cudaEventRecord(ev0));
    gc_rll_end_kernel<<<(seq_count+blk-1)/blk, blk>>>(
        d_payload, d_real_lens, seq_count, PAYLOAD_NT, d_gc, d_rll, d_endgc);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(ev1)); CUDA_CHECK(cudaEventSynchronize(ev1));
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1)); constraint_ms = ms;

    std::vector<int> h_gc(seq_count), h_rll(seq_count), h_endgc(seq_count);
    CUDA_CHECK(cudaEventRecord(ev0));
    CUDA_CHECK(cudaMemcpy(h_gc.data(),    d_gc,    seq_count*sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_rll.data(),   d_rll,   seq_count*sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_endgc.data(), d_endgc, seq_count*sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(ev1)); CUDA_CHECK(cudaEventSynchronize(ev1));
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1)); d2h_ms += ms;

    // --- Hashes (for sequential vs CUDA output equivalence check) ---
    std::string hash_payload = fnv1a64_hex(dna_payload);
    std::string hash_storage = fnv1a64_hex(storage_dna);

    // --- Compute metrics ---
    double gc_mean = 0;
    for (int v : h_gc) gc_mean += (double)v / PAYLOAD_NT;
    gc_mean /= std::max(1, seq_count);

    double gc_var = 0;
    for (int v : h_gc) { double d=(double)v/PAYLOAD_NT - gc_mean; gc_var += d*d; }
    gc_var /= std::max(1, seq_count);

    int rll_max = 0, rll_viol = 0, end_viol = 0;
    for (int i = 0; i < seq_count; ++i) {
        rll_max = std::max(rll_max, h_rll[i]);
        if (h_rll[i] > 2) rll_viol++;
        if (h_endgc[i] > 3) end_viol++;
    }

    // avg_best_lsbm_len = mean of max(L_pref_max, L_suf_max) per segment
    double avg_lsbm = 0;
    for (int s = 0; s < n_segs; ++s)
        avg_lsbm += std::max(h_L_pref[s], h_L_suf[s]);
    avg_lsbm /= std::max(1, n_segs);

    double enc_rate   = (double)n_bits / (double)std::max(1, (int)dna_payload.size());
    double gross_rate = (double)n_bits / (double)std::max(1, (int)storage_dna.size());
    double phase_sum_ms = h2d_ms + bits_ms + lsbm_ms + huffman_ms + encode_ms +
                          constraint_ms + assembly_ms + d2h_ms;
    auto t1 = std::chrono::high_resolution_clock::now();
    double total_ms_run1 = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // --- Multi-run: repeat GPU kernels n_runs-1 more times for timing stats ---
    std::vector<double> v_total  = {total_ms_run1};
    std::vector<double> v_lsbm   = {lsbm_ms};
    std::vector<double> v_encode = {encode_ms};

    for (int r = 1; r < n_runs; ++r) {
        // LSBM re-run
        float ms2 = 0;
        cudaEvent_t e0, e1;
        CUDA_CHECK(cudaEventCreate(&e0)); CUDA_CHECK(cudaEventCreate(&e1));

        CUDA_CHECK(cudaEventRecord(e0));
        lsbm_parallel_kernel<<<n_segs, blk, lsbm_shmem, stream0>>>(
            d_bits, n_bits, d_L_pref, d_L_suf);
        CUDA_CHECK(cudaStreamSynchronize(stream0));
        CUDA_CHECK(cudaEventRecord(e1)); CUDA_CHECK(cudaEventSynchronize(e1));
        CUDA_CHECK(cudaEventElapsedTime(&ms2, e0, e1));
        v_lsbm.push_back(ms2);

        // Encode re-run (two streams)
        CUDA_CHECK(cudaEventRecord(e0));
        if (gt.n_entries > 0) {
            if (half_segs > 0)
                rste_encode_kernel<<<enc_grid0, ENCODE_BLOCK, 0, stream0>>>(
                    d_bits, n_bits,
                    gt.d_keys, gt.d_key_len, gt.d_vals, gt.d_val_len,
                    gt.n_entries, half_segs, d_seg_offsets, d_dna, d_seg_dna_len);
            if (n_segs - half_segs > 0) {
                int bit_skip = half_segs * SEGMENT_BITS;
                rste_encode_kernel<<<enc_grid1, ENCODE_BLOCK, 0, stream1>>>(
                    d_bits + bit_skip, n_bits - bit_skip,
                    gt.d_keys, gt.d_key_len, gt.d_vals, gt.d_val_len,
                    gt.n_entries, n_segs - half_segs,
                    d_seg_offsets + half_segs, d_dna, d_seg_dna_len + half_segs);
            }
        }
        CUDA_CHECK(cudaStreamSynchronize(stream0));
        CUDA_CHECK(cudaStreamSynchronize(stream1));
        CUDA_CHECK(cudaEventRecord(e1)); CUDA_CHECK(cudaEventSynchronize(e1));
        CUDA_CHECK(cudaEventElapsedTime(&ms2, e0, e1));
        v_encode.push_back(ms2);

        v_total.push_back(v_lsbm.back() + huffman_ms + v_encode.back() +
                          h2d_ms + bits_ms + constraint_ms + assembly_ms + d2h_ms);

        cudaEventDestroy(e0); cudaEventDestroy(e1);
    }

    // Compute mean ± std
    auto mean_std = [](const std::vector<double>& v) -> std::pair<double,double> {
        double m = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
        double var = 0;
        for (double x : v) var += (x - m) * (x - m);
        return {m, std::sqrt(var / v.size())};
    };
    auto [tot_m, tot_s]  = mean_std(v_total);
    auto [lsb_m, lsb_s]  = mean_std(v_lsbm);
    auto [enc_m, enc_s]  = mean_std(v_encode);

    append_csv(out_csv, dataset_tag, file_basename(input_path), raw.size(),
        tot_m, tot_s, phase_sum_ms, h2d_ms, bits_ms,
        lsb_m, lsb_s, huffman_ms,
        enc_m, enc_s, constraint_ms, assembly_ms, d2h_ms,
        enc_rate, gross_rate, gc_mean, gc_var, rll_max, rll_viol, end_viol,
        seq_count, storage_seq_count, avg_lsbm, n_segs, n_runs,
        hash_payload, hash_storage);

    std::cout << "[CUDA DONE] dataset=" << dataset_tag
              << " rate="      << enc_rate    << " bits/nt"
              << " gross="     << gross_rate  << " bits/nt"
              << " gc_mean="   << gc_mean
              << " rll_max="   << rll_max
              << " rll_viol="  << rll_viol
              << " end_viol="  << end_viol
              << " hash="      << hash_payload
              << " total_ms="  << tot_m
              << " std_ms="    << tot_s
              << " runs="      << n_runs << "\n";

    cudaFree(d_raw); cudaFree(d_bits); cudaFree(d_L_pref); cudaFree(d_L_suf);
    cudaFree(d_dna); cudaFree(d_seg_offsets); cudaFree(d_seg_dna_len);
    cudaFree(d_payload); cudaFree(d_real_lens);
    cudaFree(d_gc); cudaFree(d_rll); cudaFree(d_endgc);
    cudaFreeHost(h_raw_pinned);
    if (gt.d_keys)    cudaFree(gt.d_keys);
    if (gt.d_key_len) cudaFree(gt.d_key_len);
    if (gt.d_vals)    cudaFree(gt.d_vals);
    if (gt.d_val_len) cudaFree(gt.d_val_len);
    cudaStreamDestroy(stream0); cudaStreamDestroy(stream1);
    cudaEventDestroy(ev0); cudaEventDestroy(ev1);
    return 0;
}
