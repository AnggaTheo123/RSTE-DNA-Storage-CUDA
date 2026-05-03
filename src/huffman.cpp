#include <algorithm>
#include <queue>
#include <string>
#include <vector>

#include "rste_seq.hpp"

// Build the Huffman tree according to paper Section II.A.1:
//   - Sort repeated substrings longest-first (already done by lsbm_cpu).
//   - Assign codes by tree level using the rule:
//       odd  level: left child -> 'G', right child -> 'C'
//       even level: left child -> 'A', right child -> 'T'
//   - Root is level 0 (paper Fig.3: "the root node is the 0th level").
//
// The tree is a standard min-heap Huffman tree built on *frequency*.
// Nodes are assigned levels (depth) during BFS traversal after construction.
// Each leaf's DNA code is the concatenation of edge labels from root to leaf.

struct HNode {
    int  freq;
    int  left  = -1;
    int  right = -1;
    int  level = 0;
    std::string substr; // non-empty at leaves
    std::string dna;    // accumulated code path
};

DnaCodeTable build_huffman_dna_table(const std::vector<RepeatEntry>& repeats) {
    DnaCodeTable table;
    if (repeats.empty()) return table;

    // Pool of nodes
    std::vector<HNode> pool;
    pool.reserve(repeats.size() * 2);

    // Input is assumed already sorted (size desc, freq desc, substr asc) by
    // lsbm_cpu so we don't re-sort here — that would lose lexicographic tie
    // break and diverge from CUDA build_huffman_dna_table (no re-sort).
    const std::vector<RepeatEntry>& sorted = repeats;

    // Min-heap: (freq, pool_index)
    using PIpair = std::pair<int, int>;
    std::priority_queue<PIpair, std::vector<PIpair>, std::greater<PIpair>> pq;

    for (auto& e : sorted) {
        HNode nd;
        nd.freq   = e.freq;
        nd.substr = e.substr;
        pool.push_back(nd);
        pq.push({e.freq, (int)pool.size() - 1});
    }

    // Build Huffman tree
    while (pq.size() > 1) {
        auto [f1, i1] = pq.top(); pq.pop();
        auto [f2, i2] = pq.top(); pq.pop();

        HNode parent;
        parent.freq  = f1 + f2;
        parent.left  = i1;
        parent.right = i2;
        pool.push_back(parent);
        pq.push({parent.freq, (int)pool.size() - 1});
    }

    if (pq.empty()) return table;
    int root = pq.top().second;

    // BFS to assign level and DNA edge labels, then collect leaf codes
    // Paper rule: odd level  => left='G', right='C'
    //             even level => left='A', right='T'
    // Root is level 0; its children are level 1 (odd) => G/C
    struct Frame { int idx; int level; std::string code; };
    std::queue<Frame> bfs;
    bfs.push({root, 0, ""});

    while (!bfs.empty()) {
        auto [idx, lvl, code] = bfs.front();
        bfs.pop();

        HNode& nd = pool[idx];
        nd.level = lvl;
        nd.dna   = code;

        if (nd.left == -1 && nd.right == -1) {
            // Leaf node — skip depth-1 (single-base) codes that would
            // break paper's RLL <= 2 claim when matched repeatedly.
            // Affected bits fall through to position-rotated quaternary.
            if (!nd.substr.empty() && code.size() >= 2)
                table[nd.substr] = code;
            continue;
        }

        // Determine child labels based on current node's level (children are lvl+1)
        int child_level = lvl + 1;
        char left_base, right_base;
        if (child_level % 2 == 1) {
            // odd level => left=G, right=C
            left_base  = 'G';
            right_base = 'C';
        } else {
            // even level => left=A, right=T
            left_base  = 'A';
            right_base = 'T';
        }

        if (nd.left != -1)
            bfs.push({nd.left,  child_level, code + left_base});
        if (nd.right != -1)
            bfs.push({nd.right, child_level, code + right_base});
    }

    return table;
}

// Profiling wrapper (timing hook in main.cu)
void huffman_profile_cpu(const std::vector<RepeatEntry>& repeats) {
    build_huffman_dna_table(repeats);
}
