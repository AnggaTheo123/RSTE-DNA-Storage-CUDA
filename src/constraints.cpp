#include <algorithm>
#include <vector>

#include "rste_seq.hpp"

// Paper Section II.B – verify three biological constraints per 120-nt payload block:
//
//  RLL (run-length limit): max consecutive identical bases <= 2 (paper: "at most two")
//  GC-content: count G/C per block (should be 40-60%, ~60 of 120)
//  End-constraint: last 5 bases must have <= 3 G/C (paper Section II.B.3)

void compute_constraints_cpu(const std::vector<char>& dna,
                              std::vector<int>& gc,
                              std::vector<int>& rll,
                              std::vector<int>& endgc) {
    const int n_nt    = (int)dna.size();
    const int seq_count = (n_nt + PAYLOAD_NT - 1) / PAYLOAD_NT;

    gc.assign(seq_count, 0);
    rll.assign(seq_count, 1);
    endgc.assign(seq_count, 0);

    for (int s = 0; s < seq_count; ++s) {
        int start = s * PAYLOAD_NT;
        // Limit to real (non-padded) DNA positions in this block
        int block_end = std::min(start + PAYLOAD_NT, n_nt);
        int block_len = block_end - start;

        int gcn   = 0;
        int best  = 1;
        int cur   = 1;

        for (int i = 0; i < block_len; ++i) {
            int idx = start + i;
            char c  = dna[idx];
            if (c == 'G' || c == 'C') gcn++;
            if (i > 0) {
                char p = dna[start + i - 1];
                if (c == p) { cur++; best = std::max(best, cur); }
                else          cur = 1;
            }
        }

        // End constraint: last 5 real bases of this block (or fewer if incomplete)
        int last5 = 0;
        int end_start = block_len >= 5 ? block_len - 5 : 0;
        for (int i = end_start; i < block_len; ++i) {
            char c = dna[start + i];
            if (c == 'G' || c == 'C') last5++;
        }

        gc[s]    = gcn;
        rll[s]   = best;
        endgc[s] = last5;
    }
}
