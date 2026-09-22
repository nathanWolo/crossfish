// Identity gate for the CodinGame port.
//
// codingame_nnue.cpp keeps its own standalone copy of the board and the
// search, so porting an accepted crossfish_dev.hpp change into it is a manual
// edit that nothing in `make test` checks for behavioural equivalence. This
// driver pins that down: it renames the CG file's main() away, includes it
// whole, and drives the CG engine's own search_fixed_depth over a
// deterministic position set, printing a checksum over the selected move, the
// score and the node count of every search.
//
// Build it BEFORE a port and again AFTER. For a port of a tree-identical
// change the two checksums must match exactly. Nothing here is compiled into
// the submission: the file is never included by codingame_nnue.cpp and the
// minifier never sees it.
//
//   g++ -O3 -std=c++17 -mavx2 -mbmi -mbmi2 -mlzcnt -mpopcnt -pthread \
//       -Wno-unknown-pragmas -o bin/cg_selfcheck cg_selfcheck.cpp
//   ./bin/cg_selfcheck 120 7

#define main cg_shipped_main_unused
#include "codingame_nnue.cpp"
#undef main

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>

namespace {

// FNV-1a over the per-search results, so one number covers every position.
inline void mix(uint64_t &h, uint64_t v) {
    for (int i = 0; i < 8; i++) {
        h ^= (v >> (8 * i)) & 0xff;
        h *= 1099511628211ull;
    }
}

}  // namespace

int main(int argc, char **argv) {
    const int positions = (argc >= 2) ? std::atoi(argv[1]) : 120;
    const int depth = (argc >= 3) ? std::atoi(argv[2]) : 7;
    if (positions <= 0 || depth <= 0) {
        std::fprintf(stderr, "usage: cg_selfcheck <positions> <depth>\n");
        return 2;
    }

    std::mt19937_64 rng(987654321ull);
    uint64_t checksum = 1469598103934665603ull;
    long long total_nodes = 0;
    int searched = 0;
    Move buf[81];

    while (searched < positions) {
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

        // A fresh engine per position: search_fixed_depth does not clear the
        // transposition table, so a reused engine would make the node count
        // depend on the positions that came before it.
        CrossfishDev *engine = new CrossfishDev();
        GlobalBoard probe = board;
        int score = 0;
        const bool completed = engine->search_fixed_depth(probe, depth, score);
        if (completed) {
            searched++;
            total_nodes += engine->nodes;
            mix(checksum, (uint64_t)(int64_t)score);
            mix(checksum, (uint64_t)engine->nodes);
        }
        delete engine;
    }

    std::printf("cg_selfcheck positions=%d depth=%d nodes=%lld checksum=%llu\n",
                searched, depth, total_nodes,
                (unsigned long long)checksum);
    return 0;
}
