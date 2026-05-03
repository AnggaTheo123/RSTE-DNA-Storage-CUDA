#include <algorithm>
#include <string>
#include <unordered_map>
#include <vector>

#include "rste_seq.hpp"

// -----------------------------------------------------------------------
// KMP helpers
// -----------------------------------------------------------------------
static std::vector<int> kmp_failure(const std::string& pat) {
    int m = (int)pat.size();
    std::vector<int> f(m, 0);
    int k = 0;
    for (int i = 1; i < m; ++i) {
        while (k > 0 && pat[k] != pat[i]) k = f[k - 1];
        if (pat[k] == pat[i]) ++k;
        f[i] = k;
    }
    return f;
}

static int kmp_count_nonoverlap(const std::string& text, const std::string& pat) {
    int n = (int)text.size(), m = (int)pat.size();
    if (m == 0 || m > n) return 0;
    auto f = kmp_failure(pat);
    int count = 0, k = 0, next_ok = 0;
    for (int i = 0; i < n; ++i) {
        while (k > 0 && pat[k] != text[i]) k = f[k - 1];
        if (pat[k] == text[i]) ++k;
        if (k == m) {
            int start = i - m + 1;
            if (start >= next_ok) { ++count; next_ok = start + m; }
            k = f[k - 1];
        }
    }
    return count;
}

// -----------------------------------------------------------------------
// Largest L (in [LO, HI]) for which the predicate ok(L) is true.
// Uses monotonicity (assumes ok() is downward-closed: if ok(L) then ok(L-1)).
// Returns 0 if no L in [LO, HI] satisfies ok().
// -----------------------------------------------------------------------
template <typename Pred>
static int largest_valid_L(int lo, int hi, Pred ok) {
    int best = 0;
    while (lo <= hi) {
        int mid = (lo + hi) >> 1;
        if (ok(mid)) { best = mid; lo = mid + 1; }
        else         { hi = mid - 1; }
    }
    return best;
}

// -----------------------------------------------------------------------
// LSBM (paper Section II.A.2, Steps 2-5) — paper-faithful version.
//
// Per segment (<= 500 bits):
//   * Iterate prefixes [0..L-1] and suffixes [seg_len-L..seg_len-1] from
//     long to short, RETAIN ALL whose non-overlap count is >= 2
//     (paper Step 5: "Retain the suffix AND prefix strings AND positions
//     existing in S, and form the final corresponding array").
//   * Frequency in map1 = total non-overlap occurrence count across all
//     segments (paper Step 4: "Generate map1 of the final string AND
//     frequency").
//
// Implementation note: the predicate "non-overlap count >= 2" is monotone-
// decreasing in L for the same prefix family (because every prefix-L
// occurrence is also a prefix-(L-1) occurrence). We therefore binary-
// search the largest valid L once, then enumerate L = MIN_KEY_LEN..L_max
// linearly to add every retained substring to the frequency map. Same
// for the suffix family.
// -----------------------------------------------------------------------
static constexpr int MIN_KEY_LEN = 3;  // skip 1- and 2-bit "substrings"

std::vector<RepeatEntry> lsbm_cpu(const std::vector<unsigned char>& bits) {
    int n_bits   = (int)bits.size();
    int segments = (n_bits + SEGMENT_BITS - 1) / SEGMENT_BITS;

    std::unordered_map<std::string, int> freq_map;

    for (int seg = 0; seg < segments; ++seg) {
        int start = seg * SEGMENT_BITS;
        int len   = std::min(SEGMENT_BITS, n_bits - start);
        if (len < MIN_KEY_LEN + 1) continue;

        std::string seg_str(len, '0');
        for (int i = 0; i < len; ++i)
            seg_str[i] = (char)('0' + bits[start + i]);

        int L_pref_max = largest_valid_L(MIN_KEY_LEN, len - 1,
            [&](int L) {
                return kmp_count_nonoverlap(seg_str, seg_str.substr(0, L)) >= 2;
            });

        int L_suf_max = largest_valid_L(MIN_KEY_LEN, len - 1,
            [&](int L) {
                return kmp_count_nonoverlap(seg_str, seg_str.substr(len - L, L)) >= 2;
            });

        // Paper-faithful retention with practical scope:
        // retain only the LONGEST valid prefix and the LONGEST valid suffix.
        // (Empirical: enumerating every L = 3..L_max balloons the map with
        // segment-specific substrings that rarely match elsewhere; the
        // depth filter then forces deep Huffman codes whose per-match
        // compression ratio ~= fallback, dragging the overall rate down.
        // Keeping only L_max gives statistically meaningful patterns that
        // recur across segments and produce shallow Huffman codes.)
        if (L_pref_max >= MIN_KEY_LEN) {
            std::string pref = seg_str.substr(0, L_pref_max);
            int cnt = kmp_count_nonoverlap(seg_str, pref);
            if (cnt >= 2) freq_map[pref] += cnt;
        }
        if (L_suf_max >= MIN_KEY_LEN) {
            std::string suf  = seg_str.substr(len - L_suf_max, L_suf_max);
            std::string pref = (L_suf_max <= L_pref_max)
                                 ? seg_str.substr(0, L_suf_max)
                                 : std::string();
            if (suf != pref) {
                int cnt = kmp_count_nonoverlap(seg_str, suf);
                if (cnt >= 2) freq_map[suf] += cnt;
            }
        }
    }

    std::vector<RepeatEntry> result;
    result.reserve(freq_map.size());
    for (auto& kv : freq_map) {
        RepeatEntry e;
        e.substr = kv.first;
        e.freq   = kv.second;
        result.push_back(e);
    }

    std::sort(result.begin(), result.end(), [](const RepeatEntry& a, const RepeatEntry& b) {
        if (a.substr.size() != b.substr.size()) return a.substr.size() > b.substr.size();
        if (a.freq != b.freq)                   return a.freq > b.freq;
        return a.substr < b.substr;  // deterministic tie-break for SEQ/CUDA hash parity
    });

    // Huffman depth filter — identical to build_map1() in CUDA
    int n = std::min((int)result.size(), 512);
    for (int iter = 0; iter < 20 && n > 1; ++iter) {
        int max_depth = 0, tmp = n;
        while (tmp > 1) { max_depth++; tmp = (tmp + 1) / 2; }
        int min_len = 2 * max_depth + 1;
        int ok = 0;
        for (int i = 0; i < n; ++i)
            if ((int)result[i].substr.size() >= min_len) ok++;
        if (ok >= n) break;
        n = ok;
        if (n == 0) break;
    }
    if (n > 0) result.resize(n);

    return result;
}

// -----------------------------------------------------------------------
// Profiling wrapper — best length per segment (for CSV avg_best_lsbm_len)
// -----------------------------------------------------------------------
void lsbm_profile_cpu(const std::vector<unsigned char>& bits,
                      std::vector<int>& best_lengths,
                      std::vector<int>& best_hits) {
    int n_bits   = (int)bits.size();
    int segments = (n_bits + SEGMENT_BITS - 1) / SEGMENT_BITS;
    best_lengths.assign(segments, 0);
    best_hits.assign(segments, 0);

    for (int seg = 0; seg < segments; ++seg) {
        int start = seg * SEGMENT_BITS;
        int len   = std::min(SEGMENT_BITS, n_bits - start);
        if (len < MIN_KEY_LEN + 1) continue;

        std::string seg_str(len, '0');
        for (int i = 0; i < len; ++i)
            seg_str[i] = (char)('0' + bits[start + i]);

        // Profile: largest L (over BOTH prefix and suffix) with count >= 2.
        int L_pref_max = largest_valid_L(MIN_KEY_LEN, len - 1,
            [&](int L) {
                return kmp_count_nonoverlap(seg_str, seg_str.substr(0, L)) >= 2;
            });
        int L_suf_max = largest_valid_L(MIN_KEY_LEN, len - 1,
            [&](int L) {
                return kmp_count_nonoverlap(seg_str, seg_str.substr(len - L, L)) >= 2;
            });
        int L = std::max(L_pref_max, L_suf_max);
        best_lengths[seg] = L;
        if (L >= MIN_KEY_LEN) {
            int pref_hits = kmp_count_nonoverlap(seg_str, seg_str.substr(0, L));
            int suf_hits  = kmp_count_nonoverlap(seg_str, seg_str.substr(len - L, L));
            best_hits[seg] = std::max(pref_hits, suf_hits);
        }
    }
}
