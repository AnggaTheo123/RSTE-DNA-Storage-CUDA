#pragma once

#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

// Paper constants (Section II.A)
static constexpr int SEGMENT_BITS = 500;  // max bits per LSBM segment
static constexpr int PAYLOAD_NT   = 120;  // data-bit nucleotides per sequence block
static constexpr int ADDRESS_NT   = 9;    // address block length
static constexpr int PRIMER_NT    = 20;   // primer length each end

// -----------------------------------------------------------------------
// Data structures
// -----------------------------------------------------------------------

// One repeated substring found by LSBM in a segment
struct RepeatEntry {
    std::string substr;   // the binary substring
    int         freq;     // how many non-overlapping occurrences
    std::vector<int> positions; // start indices in global bitstream
};

// Huffman tree node
struct HuffNode {
    std::string substr;  // non-empty only at leaf
    int         freq;
    int         left;    // index into node pool, -1 = none
    int         right;
    int         level;   // tree level (root = 0)
};

// A built Huffman table: maps binary substring -> DNA motif
using DnaCodeTable = std::unordered_map<std::string, std::string>;

// -----------------------------------------------------------------------
// Step 1: bytes -> bits (MSB first, paper Step 1)
// -----------------------------------------------------------------------
void bytes_to_bits_cpu(const std::vector<unsigned char>& raw,
                       std::vector<unsigned char>& bits);

// -----------------------------------------------------------------------
// Step 2-3: LSBM – find all repeated substrings in each 500-bit segment
// Returns the full list of repeated substrings across all segments,
// including frequency and global bit positions (paper Step 3),
// sorted longest-first then by freq descending.
// -----------------------------------------------------------------------
std::vector<RepeatEntry> lsbm_cpu(const std::vector<unsigned char>& bits);

// Profiling wrapper (for timing / CSV stats)
void lsbm_profile_cpu(const std::vector<unsigned char>& bits,
                      std::vector<int>& best_lengths,
                      std::vector<int>& best_hits);

// -----------------------------------------------------------------------
// Step 4-6: Build map1 (substr->freq) then Huffman tree -> map2 (substr->DNA)
// Paper rule: odd level  left=G right=C
//             even level left=A right=T
// -----------------------------------------------------------------------
DnaCodeTable build_huffman_dna_table(const std::vector<RepeatEntry>& repeats);

// Profiling wrapper
void huffman_profile_cpu(const std::vector<RepeatEntry>& repeats);

// -----------------------------------------------------------------------
// Step 7: Encode bits -> DNA using the RSTE code table
// Greedy: at each position try the longest matching key in the table first.
// Any remaining bits not covered by a repeat are encoded by fallback
// quaternary (00->T 01->C 10->G 11->A) as in the paper.
// -----------------------------------------------------------------------
void bits_to_dna_rste_cpu(const std::vector<unsigned char>& bits,
                           const DnaCodeTable& table,
                           std::vector<char>& dna);

// -----------------------------------------------------------------------
// Step 8: Constraint verification (Section II.B)
//   gc[i]    = number of G/C bases in payload block i
//   rll[i]   = longest run of identical bases in block i
//   endgc[i] = number of G/C in last 5 bases of block i
// -----------------------------------------------------------------------
void compute_constraints_cpu(const std::vector<char>& dna,
                              std::vector<int>& gc,
                              std::vector<int>& rll,
                              std::vector<int>& endgc);

// -----------------------------------------------------------------------
// CSV output
// -----------------------------------------------------------------------
struct RsteStats {
    double total_ms   = 0.0;
    double lsbm_ms    = 0.0;
    double huffman_ms = 0.0;
    double encode_ms  = 0.0;
    // multi-run std deviation (0 if runs=1)
    double std_total_ms   = 0.0;
    double std_lsbm_ms    = 0.0;
    double std_huffman_ms = 0.0;
    double std_encode_ms  = 0.0;
    int    runs       = 1;
    double gc_mean    = 0.0;
    double gc_var     = 0.0;
    int    rll_max        = 0;
    int    rll_violations = 0;  // blocks with run > 2 (paper RLL constraint)
    int    end_violations = 0;
    int    seq_count  = 0;
    int    storage_seq_count = 0;
    int    segments   = 0;
    double avg_best_lsbm_len     = 0.0;
    double encoding_rate_bits_per_nt = 0.0;
    double gross_rate_bits_per_nt = 0.0;
    double peak_ram_mb  = -1.0;
    double cpu_percent  = -1.0;
    std::string dna_hash_payload;
    std::string dna_hash_storage;
};

int append_seq_csv_row(const std::string& out_csv,
                       const std::string& dataset,
                       const std::string& file_name,
                       size_t file_size,
                       const RsteStats& st);
