// Deterministic Dev-vs-Prev screen for speed and tree-size candidates.
//
// The repo's `test_bots depth N` mode is NOT an equivalence test: each worker
// reuses one engine pair across the two games of an opening pair, so a
// provably tree-identical Dev has measured -8.44 +/- 23.84 Elo at N=700. This
// tool instead constructs a fresh engine per position, which makes both node
// counts and scores deterministic functions of the engine alone.
//
//   equiv <positions> <depth> [seed]  identical score AND node count required
//   nodes <positions> <depth> [seed]  total nodes and wall time per engine
//
// `equiv` exits nonzero on the first divergence. `nodes` interleaves the two
// engines per position (A,B then B,A on alternate positions) so cache-warming
// order cannot favour either side.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <random>
#include <stack>
#include <vector>
#include <immintrin.h>

#include "global_board.hpp"
#ifndef CROSSFISH_PREV_HEADER
#define CROSSFISH_PREV_HEADER "crossfish_prev.hpp"
#endif
#ifndef CROSSFISH_DEV_HEADER
#define CROSSFISH_DEV_HEADER "crossfish_dev.hpp"
#endif
#include CROSSFISH_PREV_HEADER
#include CROSSFISH_DEV_HEADER

namespace {

// Random legal playout to a target ply, rejecting finished games so every
// sampled root still has legal moves.
std::vector<GlobalBoard> sample_positions(int count, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::vector<GlobalBoard> out;
    out.reserve(count);
    Move buf[81];
    while ((int)out.size() < count) {
        GlobalBoard board;
        const int target_ply = 4 + (int)(rng() % 25);
        bool ok = true;
        for (int ply = 0; ply < target_ply; ply++) {
            if (board.checkWinner() != -1) {
                ok = false;
                break;
            }
            const int n = board.fillLegalMoves(buf);
            if (n <= 0) {
                ok = false;
                break;
            }
            board.makeMove(buf[rng() % (uint64_t)n]);
        }
        if (!ok || board.checkWinner() != -1) continue;
        if (board.fillLegalMoves(buf) <= 0) continue;
        out.push_back(board);
    }
    return out;
}

struct Probe {
    bool ok = false;
    int score = 0;
    long long nodes = 0;
    double seconds = 0;
};

template <typename Engine>
Probe probe(GlobalBoard position, int depth) {
    // A fresh engine per position: search_fixed_depth clears killers, history
    // and correction history but NOT the transposition table, so a reused
    // engine would leak the previous position's tree into this one's node count.
    auto engine = std::make_unique<Engine>();
    Probe r;
    int score = 0;
    const auto t0 = std::chrono::steady_clock::now();
    r.ok = engine->search_fixed_depth(position, depth, score);
    const auto t1 = std::chrono::steady_clock::now();
    r.score = score;
    r.nodes = engine->nodes;
    r.seconds = std::chrono::duration<double>(t1 - t0).count();
    return r;
}

int run_equiv(int count, int depth, uint64_t seed) {
    const auto positions = sample_positions(count, seed);
    long long prev_nodes = 0;
    long long dev_nodes = 0;
    int compared = 0;
    int score_diffs = 0;
    int node_diffs = 0;
    for (size_t i = 0; i < positions.size(); i++) {
        const Probe p = probe<CrossfishPrev>(positions[i], depth);
        const Probe d = probe<CrossfishDev>(positions[i], depth);
        if (!p.ok || !d.ok) continue;
        compared++;
        prev_nodes += p.nodes;
        dev_nodes += d.nodes;
        if (p.score != d.score) {
            if (score_diffs < 10) {
                std::cout << "SCORE DIFF at position " << i
                          << ": prev=" << p.score << " dev=" << d.score
                          << std::endl;
            }
            score_diffs++;
        }
        if (p.nodes != d.nodes) {
            if (node_diffs < 10) {
                std::cout << "NODE DIFF at position " << i
                          << ": prev=" << p.nodes << " dev=" << d.nodes
                          << " (" << (100.0 * (d.nodes - p.nodes) / p.nodes)
                          << "%)" << std::endl;
            }
            node_diffs++;
        }
    }
    std::cout << "equiv depth=" << depth << " positions=" << compared
              << " prev_nodes=" << prev_nodes << " dev_nodes=" << dev_nodes
              << " score_diffs=" << score_diffs
              << " node_diffs=" << node_diffs << std::endl;
    if (score_diffs == 0 && node_diffs == 0) {
        std::cout << "IDENTICAL: dev computes the same tree as prev"
                  << std::endl;
        return 0;
    }
    std::cout << "NOT IDENTICAL" << std::endl;
    return 1;
}

int run_nodes(int count, int depth, uint64_t seed) {
    const auto positions = sample_positions(count, seed);
    long long prev_nodes = 0, dev_nodes = 0;
    double prev_secs = 0, dev_secs = 0;
    int compared = 0;
    int dev_better = 0, prev_better = 0;
    for (size_t i = 0; i < positions.size(); i++) {
        Probe p, d;
        if (i % 2 == 0) {
            p = probe<CrossfishPrev>(positions[i], depth);
            d = probe<CrossfishDev>(positions[i], depth);
        } else {
            d = probe<CrossfishDev>(positions[i], depth);
            p = probe<CrossfishPrev>(positions[i], depth);
        }
        if (!p.ok || !d.ok) continue;
        compared++;
        prev_nodes += p.nodes;
        dev_nodes += d.nodes;
        prev_secs += p.seconds;
        dev_secs += d.seconds;
        if (d.nodes < p.nodes) dev_better++;
        else if (p.nodes < d.nodes) prev_better++;
    }
    if (compared == 0 || prev_nodes == 0) {
        std::cerr << "no comparable positions" << std::endl;
        return 2;
    }
    const double prev_nps = prev_nodes / prev_secs;
    const double dev_nps = dev_nodes / dev_secs;
    std::cout.setf(std::ios::fixed);
    std::cout.precision(3);
    std::cout << "nodes depth=" << depth << " positions=" << compared
              << "\n  prev nodes=" << prev_nodes << " secs=" << prev_secs
              << " nps=" << (long long)prev_nps
              << "\n  dev  nodes=" << dev_nodes << " secs=" << dev_secs
              << " nps=" << (long long)dev_nps
              << "\n  node ratio dev/prev="
              << (100.0 * dev_nodes / prev_nodes) << "%"
              << "  nps ratio dev/prev=" << (100.0 * dev_nps / prev_nps) << "%"
              << "\n  time ratio dev/prev=" << (100.0 * dev_secs / prev_secs)
              << "%  positions dev_fewer_nodes=" << dev_better
              << " prev_fewer_nodes=" << prev_better << std::endl;
    return 0;
}

}  // namespace

int main(int argc, char **argv) {
    if (argc < 4) {
        std::cerr << "usage: bench_ab equiv|nodes <positions> <depth> [seed]"
                  << std::endl;
        return 2;
    }
    const int count = std::atoi(argv[2]);
    const int depth = std::atoi(argv[3]);
    const uint64_t seed = (argc >= 5) ? std::strtoull(argv[4], nullptr, 10) : 20260922ull;
    if (count <= 0 || depth <= 0) {
        std::cerr << "positions and depth must be positive" << std::endl;
        return 2;
    }
    if (std::strcmp(argv[1], "equiv") == 0) return run_equiv(count, depth, seed);
    if (std::strcmp(argv[1], "nodes") == 0) return run_nodes(count, depth, seed);
    std::cerr << "unknown mode: " << argv[1] << std::endl;
    return 2;
}
