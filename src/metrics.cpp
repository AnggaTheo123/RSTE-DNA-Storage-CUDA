#include <fstream>
#include <iostream>
#include <string>

#include "rste_seq.hpp"

int append_seq_csv_row(const std::string& out_csv,
                       const std::string& dataset,
                       const std::string& file_name,
                       size_t file_size,
                       const RsteStats& st) {
    bool write_header = false;
    {
        std::ifstream chk(out_csv);
        write_header = !chk.good();
    }

    std::ofstream out(out_csv, std::ios::app);
    if (!out) {
        std::cerr << "Cannot write CSV: " << out_csv << "\n";
        return 1;
    }

    if (write_header) {
        out << "dataset,file_name,file_size_bytes,runs,"
               "total_ms,lsbm_ms,huffman_ms,encode_ms,"
               "std_total_ms,std_lsbm_ms,std_huffman_ms,std_encode_ms,"
               "peak_ram_mb,cpu_percent,encoding_rate_bits_per_nt,"
               "gross_rate_bits_per_nt,gc_mean,gc_var,rll_max,rll_violations,"
               "seq_count,storage_seq_count,end_violations,avg_best_lsbm_len,"
               "segments,dna_hash_payload,dna_hash_storage\n";
    }

    out << dataset           << ","
        << file_name         << ","
        << file_size         << ","
        << st.runs           << ","
        << st.total_ms       << ","
        << st.lsbm_ms        << ","
        << st.huffman_ms     << ","
        << st.encode_ms      << ","
        << st.std_total_ms   << ","
        << st.std_lsbm_ms    << ","
        << st.std_huffman_ms << ","
        << st.std_encode_ms  << ","
        << st.peak_ram_mb    << ","
        << st.cpu_percent    << ","
        << st.encoding_rate_bits_per_nt << ","
        << st.gross_rate_bits_per_nt    << ","
        << st.gc_mean        << ","
        << st.gc_var         << ","
        << st.rll_max        << ","
        << st.rll_violations << ","
        << st.seq_count      << ","
        << st.storage_seq_count << ","
        << st.end_violations << ","
        << st.avg_best_lsbm_len << ","
        << st.segments       << ","
        << st.dna_hash_payload << ","
        << st.dna_hash_storage << "\n";

    return 0;
}
