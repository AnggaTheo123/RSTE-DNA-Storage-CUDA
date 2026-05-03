#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include "rste_seq.hpp"

static std::vector<unsigned char> read_binary_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return {};
    return std::vector<unsigned char>((std::istreambuf_iterator<char>(in)),
                                       std::istreambuf_iterator<char>());
}

static std::string dataset_basename(const std::string& path) {
    size_t p = path.find_last_of("\\/");
    return (p == std::string::npos) ? path : path.substr(p + 1);
}

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
    static const char primer_l[PRIMER_NT + 1] = "ACGTTGCATGTCAGTACGTA";
    static const char primer_r[PRIMER_NT + 1] = "TACGACTGACATGCAACGTT";

    int payload_nt = (int)payload_dna.size();
    storage_seq_count = (payload_nt + PAYLOAD_NT - 1) / PAYLOAD_NT;
    storage_dna.clear();
    storage_dna.reserve((size_t)storage_seq_count * (PRIMER_NT + ADDRESS_NT + PAYLOAD_NT + PRIMER_NT));

    for (int s = 0; s < storage_seq_count; ++s) {
        int start = s * PAYLOAD_NT;
        int end = std::min(start + PAYLOAD_NT, payload_nt);

        for (int i = 0; i < PRIMER_NT; ++i) storage_dna.push_back(primer_l[i]);
        append_address_9nt(s, storage_dna);

        for (int i = start; i < end; ++i) storage_dna.push_back(payload_dna[i]);
        for (int i = end; i < start + PAYLOAD_NT; ++i) storage_dna.push_back('A');

        for (int i = 0; i < PRIMER_NT; ++i) storage_dna.push_back(primer_r[i]);
    }
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: rste_seq.exe <input_file> <dataset_tag> <output_csv> [--runs N]\n";
        return 1;
    }

    const std::string input_path  = argv[1];
    const std::string dataset_tag = argv[2];
    const std::string out_csv     = argv[3];

    int n_runs = 5;
    for (int i = 4; i < argc - 1; ++i)
        if (std::string(argv[i]) == "--runs")
            n_runs = std::max(1, std::atoi(argv[i + 1]));

    // ----------------------------------------------------------------
    // Step 1: read file and convert to bits (done once)
    // ----------------------------------------------------------------
    std::vector<unsigned char> raw = read_binary_file(input_path);
    if (raw.empty()) {
        std::cerr << "Input read failed or empty: " << input_path << "\n";
        return 1;
    }
    std::vector<unsigned char> bits;
    bytes_to_bits_cpu(raw, bits);

    // ----------------------------------------------------------------
    // Multi-run loop — repeat compute pipeline n_runs times
    // ----------------------------------------------------------------
    std::vector<double> v_total, v_lsbm, v_huff, v_enc;
    std::vector<char> dna, storage_dna;
    std::vector<int>  gc, rll, endgc, best_lengths, best_hits;
    int storage_seq_count = 0;

    for (int r = 0; r < n_runs; ++r) {
        auto t0 = std::chrono::high_resolution_clock::now();

        // Steps 2-3: LSBM
        auto l0 = std::chrono::high_resolution_clock::now();
        std::vector<RepeatEntry> repeats = lsbm_cpu(bits);
        auto l1 = std::chrono::high_resolution_clock::now();

        // Steps 4-6: Huffman
        auto h0 = std::chrono::high_resolution_clock::now();
        DnaCodeTable dna_table = build_huffman_dna_table(repeats);
        auto h1 = std::chrono::high_resolution_clock::now();

        // Step 7: Encode
        auto e0 = std::chrono::high_resolution_clock::now();
        bits_to_dna_rste_cpu(bits, dna_table, dna);
        build_storage_sequences(dna, storage_dna, storage_seq_count);

        // Step 8: Constraints
        compute_constraints_cpu(dna, gc, rll, endgc);
        auto e1 = std::chrono::high_resolution_clock::now();

        auto t1 = std::chrono::high_resolution_clock::now();

        v_total.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        v_lsbm .push_back(std::chrono::duration<double, std::milli>(l1 - l0).count());
        v_huff .push_back(std::chrono::duration<double, std::milli>(h1 - h0).count());
        v_enc  .push_back(std::chrono::duration<double, std::milli>(e1 - e0).count());
    }

    // Profiling per-segment for avg_best_lsbm_len
    lsbm_profile_cpu(bits, best_lengths, best_hits);



    // Compute mean and std
    auto mean_std = [](const std::vector<double>& v) -> std::pair<double,double> {
        double m = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
        double var = 0;
        for (double x : v) var += (x - m) * (x - m);
        return {m, std::sqrt(var / v.size())};
    };

    auto [tot_m, tot_s] = mean_std(v_total);
    auto [lsb_m, lsb_s] = mean_std(v_lsbm);
    auto [huf_m, huf_s] = mean_std(v_huff);
    auto [enc_m, enc_s] = mean_std(v_enc);

    // ----------------------------------------------------------------
    // Collect stats
    // ----------------------------------------------------------------
    RsteStats st;
    st.runs        = n_runs;
    st.total_ms    = tot_m;  st.std_total_ms   = tot_s;
    st.lsbm_ms     = lsb_m;  st.std_lsbm_ms    = lsb_s;
    st.huffman_ms  = huf_m;  st.std_huffman_ms  = huf_s;
    st.encode_ms   = enc_m;  st.std_encode_ms   = enc_s;

    st.segments   = (int)best_lengths.size();
    st.seq_count  = (int)gc.size();

    // encoding rate: bits of input / nucleotides produced (paper formula)
    st.encoding_rate_bits_per_nt =
        (double)bits.size() / (double)std::max(1, (int)dna.size());
    st.gross_rate_bits_per_nt =
        (double)bits.size() / (double)std::max(1, (int)storage_dna.size());
    st.storage_seq_count = storage_seq_count;
    st.dna_hash_payload = fnv1a64_hex(dna);
    st.dna_hash_storage = fnv1a64_hex(storage_dna);

    // average best LSBM length across segments
    if (!best_lengths.empty()) {
        st.avg_best_lsbm_len =
            std::accumulate(best_lengths.begin(), best_lengths.end(), 0.0)
            / (double)best_lengths.size();
    }

    // GC mean and variance
    st.gc_mean = 0.0;
    for (int v : gc)
        st.gc_mean += (double)v / (double)PAYLOAD_NT;
    st.gc_mean /= std::max(1, st.seq_count);

    st.gc_var = 0.0;
    for (int v : gc) {
        double x = (double)v / (double)PAYLOAD_NT;
        double d = x - st.gc_mean;
        st.gc_var += d * d;
    }
    st.gc_var /= std::max(1, st.seq_count);

    // RLL max, RLL violations (run > 2), and end-constraint violations (> 3 GC in last 5)
    // Paper Section II.B.1: max consecutive identical bases <= 2
    st.rll_max        = 0;
    st.rll_violations = 0;
    st.end_violations = 0;
    for (int i = 0; i < st.seq_count; ++i) {
        st.rll_max = std::max(st.rll_max, rll[i]);
        if (rll[i] > 2) st.rll_violations++;
        if (endgc[i] > 3) st.end_violations++;
    }

    // ----------------------------------------------------------------
    // Write CSV row
    // ----------------------------------------------------------------
    int rc = append_seq_csv_row(out_csv, dataset_tag,
                                 dataset_basename(input_path),
                                 raw.size(), st);
    if (rc != 0) return rc;

    std::cout << "[RSTE_SEQ DONE] dataset=" << dataset_tag
              << " file="      << dataset_basename(input_path)
              << " rate="      << st.encoding_rate_bits_per_nt << " bits/nt"
              << " gross="     << st.gross_rate_bits_per_nt << " bits/nt"
              << " gc_mean="   << st.gc_mean
              << " rll_max="   << st.rll_max
              << " rll_viol="  << st.rll_violations
              << " end_viol="  << st.end_violations
              << " hash="      << st.dna_hash_payload
              << "\n";
    return 0;
}
