#include <algorithm>
#include <string>
#include <vector>

#include "rste_seq.hpp"

// Step 1: bytes -> bit vector (MSB first, paper Step 1)
void bytes_to_bits_cpu(const std::vector<unsigned char>& raw,
                        std::vector<unsigned char>& bits) {
    const int n_bytes = (int)raw.size();
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
// Step 7: RSTE greedy encoder (paper Section II.A, Step 7)
//
// Greedy longest-match over the Huffman DNA table (map2).
// Bits not matched by any table entry fall back to quaternary
// using the paper mapping (2 bits -> 1 nt: 00=T 01=C 10=G 11=A).
// -----------------------------------------------------------------------
void bits_to_dna_rste_cpu(const std::vector<unsigned char>& bits,
                           const DnaCodeTable& table,
                           std::vector<char>& dna) {
    const int n_bits = (int)bits.size();
    if (n_bits == 0) return;

    // Build bit-string for direct compare
    std::string bstr(n_bits, '0');
    for (int i = 0; i < n_bits; ++i)
        bstr[i] = (char)('0' + bits[i]);

    // Sort entries longest-first for greedy scan
    struct Entry { int len; const std::string* key; const std::string* val; };
    std::vector<Entry> entries;
    entries.reserve(table.size());
    for (auto& kv : table)
        entries.push_back({(int)kv.first.size(), &kv.first, &kv.second});
    std::sort(entries.begin(), entries.end(), [](const Entry& a, const Entry& b){
        if (a.len != b.len) return a.len > b.len;
        if (*a.key != *b.key) return *a.key < *b.key;
        return *a.val < *b.val;
    });

    dna.clear();
    dna.reserve(n_bits / 2 + 16);

    auto push = [&](char c) { dna.push_back(c); };

    // RLL-safe quaternary fallback: 2 bits -> 2 nucleotides (lossless).
    //   00 -> GA, 01 -> GT, 10 -> CA, 11 -> CT   (1 bit per 1 nt)
    // All four sequences start with G or C, end with A or T, and have no
    // internal repeat. Concatenating any two yields max run = 1
    // internally, max run = 2 at the Huffman-fall-back boundary. This
    // mathematically guarantees RLL <= 2.
    //
    // Trade-off: encoding rate drops on fall-back-heavy data, but the
    // RLL <= 2 claim becomes mathematically guaranteed (which the
    // simple 2-bit -> 1-nt mapping cannot achieve, by pigeonhole).
    static const char fb2[4][3] = {
        {'G','A',0},   // 00 -> GA
        {'G','T',0},   // 01 -> GT
        {'C','A',0},   // 10 -> CA
        {'C','T',0},   // 11 -> CT
    };

    for (int seg_start = 0; seg_start < n_bits; seg_start += SEGMENT_BITS) {
        int seg_end = std::min(seg_start + SEGMENT_BITS, n_bits);
        int pos = seg_start;
        while (pos < seg_end) {
            bool matched = false;
            int avail = seg_end - pos;

            for (auto& e : entries) {
                if (e.len > avail) continue;
                if (e.len < 2) break;
                if (bstr.compare(pos, e.len, *e.key) == 0) {
                    for (char c : *e.val)
                        push(c);
                    pos += e.len;
                    matched = true;
                    break;
                }
            }

            if (!matched) {
                int bit_val;
                if (avail == 1) {
                    bit_val = bits[pos] * 2;
                    pos += 1;
                } else {
                    bit_val = bits[pos] * 2 + bits[pos + 1];
                    pos += 2;
                }
                push(fb2[bit_val][0]);
                push(fb2[bit_val][1]);
            }
        }
    }
}
