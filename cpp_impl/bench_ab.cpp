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
//   sat <games> <plies> <depth>       persistent-TT walk, deterministic nodes
//   walk <games> <plies> <ms>         persistent-TT walk at a real move budget
//
// `equiv` exits nonzero on the first divergence. `nodes` interleaves the two
// engines per position (A,B then B,A on alternate positions) so cache-warming
// order cannot favour either side.
//
// `nodes` and `equiv` build a fresh engine per position, so the transposition
// table starts empty and a single fixed-depth search never comes close to
// filling its 262,144 entries. That makes them blind to anything about table
// capacity or replacement. `sat` and `walk` instead replay one scripted move
// sequence through a SINGLE engine instance, exactly as a real game does, so
// the table saturates the way it does on CodinGame (measured there: ~39%
// occupancy after the first move, 100% by move 9). Both engines see an
// identical position sequence.
//
// `sat` is the deterministic capacity screen: fixed depth, so node counts are
// reproducible and immune to machine load. `walk` is the strength-shaped one:
// a real per-move millisecond budget, reporting mean completed root depth,
// which is the quantity that actually converts into Elo. `walk` is timing
// based and therefore needs a quiet machine.

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

// One scripted game: a legal move sequence from the start position. Both
// engines are asked to think at every position in the same sequence, so the
// comparison is paired and the sequence itself is engine-independent.
std::vector<Move> sample_game(int plies, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::vector<Move> moves;
    GlobalBoard board;
    Move buf[81];
    for (int ply = 0; ply < plies; ply++) {
        if (board.checkWinner() != -1) break;
        const int n = board.fillLegalMoves(buf);
        if (n <= 0) break;
        const Move m = buf[rng() % (uint64_t)n];
        moves.push_back(m);
        board.makeMove(m);
    }
    return moves;
}

struct WalkStat {
    long long nodes = 0;
    long long depth_sum = 0;
    int searches = 0;
    double seconds = 0;
};

// Fixed depth, one engine instance for the whole game: the table fills up and
// stays full, and node counts stay deterministic.
template <typename Engine>
WalkStat walk_fixed(const std::vector<Move> &script, int depth) {
    auto engine = std::make_unique<Engine>();
    WalkStat s;
    GlobalBoard board;
    const auto t0 = std::chrono::steady_clock::now();
    for (const Move &m : script) {
        if (board.checkWinner() != -1) break;
        GlobalBoard probe_board = board;
        int score = 0;
        if (engine->search_fixed_depth(probe_board, depth, score)) {
            s.nodes += engine->nodes;
            s.searches++;
        }
        board.makeMove(m);
    }
    s.seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
            .count();
    return s;
}

// Real move budget, one engine instance for the whole game. completed_root_depth
// is the iteration the engine actually finished, which is the quantity a
// speed or capacity win has to convert into.
template <typename Engine>
WalkStat walk_timed(const std::vector<Move> &script, int ms) {
    auto engine = std::make_unique<Engine>();
    WalkStat s;
    GlobalBoard board;
    const auto t0 = std::chrono::steady_clock::now();
    for (const Move &m : script) {
        if (board.checkWinner() != -1) break;
        engine->getMove(board, std::chrono::milliseconds(ms));
        s.nodes += engine->nodes;
        s.depth_sum += engine->completed_root_depth;
        s.searches++;
        board.makeMove(m);
    }
    s.seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
            .count();
    return s;
}

int run_sat(int games, int plies, int depth) {
    WalkStat prev, dev;
    for (int g = 0; g < games; g++) {
        const auto script = sample_game(plies, 555000ull + (uint64_t)g);
        WalkStat p, d;
        if (g % 2 == 0) {
            p = walk_fixed<CrossfishPrev>(script, depth);
            d = walk_fixed<CrossfishDev>(script, depth);
        } else {
            d = walk_fixed<CrossfishDev>(script, depth);
            p = walk_fixed<CrossfishPrev>(script, depth);
        }
        prev.nodes += p.nodes;
        prev.searches += p.searches;
        prev.seconds += p.seconds;
        dev.nodes += d.nodes;
        dev.searches += d.searches;
        dev.seconds += d.seconds;
    }
    if (prev.nodes == 0) {
        std::cerr << "no searches completed" << std::endl;
        return 2;
    }
    std::cout.setf(std::ios::fixed);
    std::cout.precision(3);
    std::cout << "sat games=" << games << " plies=" << plies
              << " depth=" << depth
              << "\n  prev nodes=" << prev.nodes
              << " searches=" << prev.searches << " secs=" << prev.seconds
              << " nps=" << (long long)(prev.nodes / prev.seconds)
              << "\n  dev  nodes=" << dev.nodes
              << " searches=" << dev.searches << " secs=" << dev.seconds
              << " nps=" << (long long)(dev.nodes / dev.seconds)
              << "\n  node ratio dev/prev=" << (100.0 * dev.nodes / prev.nodes)
              << "%  time ratio dev/prev="
              << (100.0 * dev.seconds / prev.seconds) << "%" << std::endl;
    if (prev.searches != dev.searches) {
        std::cout << "  WARNING: different search counts, sequences diverged"
                  << std::endl;
    }
    return 0;
}

int run_walk(int games, int plies, int ms) {
    WalkStat prev, dev;
    for (int g = 0; g < games; g++) {
        const auto script = sample_game(plies, 555000ull + (uint64_t)g);
        WalkStat p, d;
        if (g % 2 == 0) {
            p = walk_timed<CrossfishPrev>(script, ms);
            d = walk_timed<CrossfishDev>(script, ms);
        } else {
            d = walk_timed<CrossfishDev>(script, ms);
            p = walk_timed<CrossfishPrev>(script, ms);
        }
        prev.nodes += p.nodes;
        prev.depth_sum += p.depth_sum;
        prev.searches += p.searches;
        prev.seconds += p.seconds;
        dev.nodes += d.nodes;
        dev.depth_sum += d.depth_sum;
        dev.searches += d.searches;
        dev.seconds += d.seconds;
    }
    if (prev.searches == 0 || dev.searches == 0) {
        std::cerr << "no searches completed" << std::endl;
        return 2;
    }
    const double prev_depth = (double)prev.depth_sum / prev.searches;
    const double dev_depth = (double)dev.depth_sum / dev.searches;
    std::cout.setf(std::ios::fixed);
    std::cout.precision(4);
    std::cout << "walk games=" << games << " plies=" << plies << " ms=" << ms
              << "\n  prev nodes=" << prev.nodes
              << " mean_depth=" << prev_depth
              << " nps=" << (long long)(prev.nodes / prev.seconds)
              << "\n  dev  nodes=" << dev.nodes
              << " mean_depth=" << dev_depth
              << " nps=" << (long long)(dev.nodes / dev.seconds)
              << "\n  mean depth delta dev-prev=" << (dev_depth - prev_depth)
              << "  node ratio dev/prev=" << (100.0 * dev.nodes / prev.nodes)
              << "%  searches=" << prev.searches << "/" << dev.searches
              << std::endl;
    return 0;
}

}  // namespace

int main(int argc, char **argv) {
    if (argc < 4) {
        std::cerr << "usage: bench_ab equiv <n> <depth> [seed] / nodes <n> <depth> [seed] / sat <games> <plies> <depth> / walk <games> <plies> <ms>\n       legacy: equiv|nodes <positions> <depth> [seed]"
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
    // For sat/walk the positional arguments are <games> <plies> and the
    // fourth is the depth or millisecond budget rather than a seed.
    if (std::strcmp(argv[1], "sat") == 0) {
        const int arg = (argc >= 5) ? std::atoi(argv[4]) : 10;
        return run_sat(count, depth, arg);
    }
    if (std::strcmp(argv[1], "walk") == 0) {
        const int arg = (argc >= 5) ? std::atoi(argv[4]) : 90;
        return run_walk(count, depth, arg);
    }
    std::cerr << "unknown mode: " << argv[1] << std::endl;
    return 2;
}
