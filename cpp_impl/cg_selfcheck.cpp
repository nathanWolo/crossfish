// Identity gate for the CodinGame port.
//
// codingame_nnue.cpp keeps its own standalone copy of the board and the
// search, so porting an accepted crossfish_dev.hpp change into it is a manual
// edit that nothing in `make test` checks for behavioural equivalence. This
// driver pins that down: it renames the CG file's main() away, includes it
// whole, and drives the CG engine's own search_fixed_depth over a
// deterministic position set, printing a checksum over the score and the
// node count of every search.
// It also decodes the opening book and prints the table's entry count and
// checksum, which must match play_book_check on the local build.
//
// Build it BEFORE a port and again AFTER. For a port of a tree-identical
// change the two checksums must match exactly. Nothing here is compiled into
// the submission: the file is never included by codingame_nnue.cpp and the
// minifier never sees it.
//
//   g++ -O3 -std=c++17 -mavx2 -mbmi -mbmi2 -mlzcnt -mpopcnt -pthread \
//       -Wno-unknown-pragmas -o bin/cg_selfcheck cg_selfcheck.cpp
//   ./bin/cg_selfcheck 120 7
//
// It also prints the wall time of the searches alone. Built once at -O3 and
// once with CodinGame's flags (`make cg-speed` does both), the two compute
// the same tree, so the seconds ratio is the CodinGame speed gap.

#define main cg_shipped_main_unused
#include "codingame_nnue.cpp"
#undef main

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <random>
#include <vector>

// FNV-1a over the decoded book table in key order, so builds can be compared.
inline uint64_t play_book_table_checksum() {
    std::vector<std::pair<uint64_t, uint8_t>> rows(PB_TABLE.begin(), PB_TABLE.end());
    std::sort(rows.begin(), rows.end());
    uint64_t h = 1469598103934665603ull;
    for (auto &r : rows) {
        for (int i = 0; i < 8; i++) { h ^= (r.first >> (8 * i)) & 0xff; h *= 1099511628211ull; }
        h ^= r.second; h *= 1099511628211ull;
    }
    return h;
}

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

    // Warm the static tables (LUTs, macro table) outside the timed region, so
    // `seconds` measures search alone. The tables are pure functions of the
    // packed data, so this cannot change any score or node count below.
    {
        CrossfishDev *warm = new CrossfishDev();
        GlobalBoard start;
        int ignored = 0;
        warm->search_fixed_depth(start, 1, ignored);
        delete warm;
    }
    double search_seconds = 0;

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
        const auto t0 = std::chrono::steady_clock::now();
        const bool completed = engine->search_fixed_depth(probe, depth, score);
        search_seconds += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0).count();
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
    // Timing goes on its own line so the checksum line stays comparable
    // across builds. Compare seconds only between builds of the same tree on
    // the same machine: it is the CodinGame-flags vs -O3 speed gate.
    std::printf("cg_selfcheck seconds=%.3f nps=%.0f\n", search_seconds,
                search_seconds > 0 ? total_nodes / search_seconds : 0.0);
    bool book_ok = pb_init<GlobalBoard, Move>();
    std::printf("cg_selfcheck book=%s entries=%zu table_checksum=%llu\n",
                book_ok ? "ok" : "FAILED", PB_TABLE.size(),
                (unsigned long long)play_book_table_checksum());
    return 0;
}
