// cg_selfcheck's twin on the reference engines.
//
// cg_selfcheck pins the CodinGame port to itself: for a tree-identical change
// its checksum must not move. A change that alters the tree cannot be checked
// that way, so this runs the identical position set and checksum over
// CrossfishDev or CrossfishPrev instead. A correct port then prints the same
// line from `cg_selfcheck N D` (built from the ported codingame_nnue.cpp) as
// `engine_selfcheck dev N D`, and an unported CG file matches
// `engine_selfcheck prev N D`.
//
//   g++ -O3 -std=c++17 -mavx2 -mbmi -mbmi2 -mlzcnt -mpopcnt -pthread \
//       -Wno-unknown-pragmas -o bin/engine_selfcheck engine_selfcheck.cpp
//   ./bin/engine_selfcheck dev 120 7

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <random>
#include <stack>
#include <string>
#include <vector>
#include <immintrin.h>

#include "global_board.hpp"
#include "crossfish_prev.hpp"
#include "crossfish_dev.hpp"

namespace {

inline void mix(uint64_t &h, uint64_t v) {
    for (int i = 0; i < 8; i++) {
        h ^= (v >> (8 * i)) & 0xff;
        h *= 1099511628211ull;
    }
}

// Same sampling, fresh-engine rule and checksum as cg_selfcheck.cpp.
template <typename Engine>
int run(int positions, int depth) {
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

        auto engine = std::make_unique<Engine>();
        GlobalBoard probe = board;
        int score = 0;
        const bool completed = engine->search_fixed_depth(probe, depth, score);
        if (completed) {
            searched++;
            total_nodes += engine->nodes;
            mix(checksum, (uint64_t)(int64_t)score);
            mix(checksum, (uint64_t)engine->nodes);
        }
    }

    std::printf("cg_selfcheck positions=%d depth=%d nodes=%lld checksum=%llu\n",
                searched, depth, total_nodes,
                (unsigned long long)checksum);
    return 0;
}

}  // namespace

int main(int argc, char **argv) {
    const std::string which = (argc >= 2) ? argv[1] : "";
    const int positions = (argc >= 3) ? std::atoi(argv[2]) : 120;
    const int depth = (argc >= 4) ? std::atoi(argv[3]) : 7;
    if ((which != "dev" && which != "prev") || positions <= 0 || depth <= 0) {
        std::fprintf(stderr, "usage: engine_selfcheck dev|prev <positions> <depth>\n");
        return 2;
    }
    return which == "dev" ? run<CrossfishDev>(positions, depth)
                          : run<CrossfishPrev>(positions, depth);
}
