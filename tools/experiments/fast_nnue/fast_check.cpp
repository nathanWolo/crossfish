// Drives real searches of a -DFASTNNUE_CHECK build of the fast-NNUE Dev engine, so that
// fast_nnue.hpp's per-evaluation checks (incremental accumulators == scalar from-scratch
// int32 accumulators, AVX2 output == scalar dense reference) run on many real trees.
// Build in the candidate directory next to its patched crossfish_dev.hpp (build_cand.sh does):
//
//   fast_check POSITIONS.cfdg N_POS THREADS
//
// For N_POS records spread over POSITIONS (DgRec, tools/eval_data.py):
//   - search_fixed_depth at depths 6, 8 and 10 (10 on every third record), each on a
//     fresh engine (a new root refresh per search);
//   - on every fifth record, a self-play game to the end with one engine per side reused
//     across moves (persistent TT, iterative deepening with aspiration re-searches, the
//     time cutoff unwinding mid-tree) at 3 ms per move;
//   - evaluate(GlobalBoard&) (the from-scratch path datagen uses) on the record.
// Any mismatch aborts inside fast_nnue.hpp; the counters print at exit.
#define main tb_main
#include "test_bots.cpp"
#undef main

#include <atomic>
#include <ctime>
#include <thread>

#pragma pack(push, 1)
struct CheckRec {
    char s[93];
    uint8_t rest[35];
};
#pragma pack(pop)
static_assert(sizeof(CheckRec) == 128, "DgRec is 128 bytes");

int main(int argc, char **argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: fast_check POSITIONS.cfdg N_POS THREADS\n");
        return 2;
    }
#ifndef FASTNNUE_CHECK
    std::fprintf(stderr, "fast_check: built without -DFASTNNUE_CHECK, nothing would be checked\n");
    return 2;
#endif
    FILE *f = std::fopen(argv[1], "rb");
    if (!f) { std::fprintf(stderr, "cannot read %s\n", argv[1]); return 1; }
    std::vector<CheckRec> all;
    CheckRec r;
    while (std::fread(&r, sizeof(r), 1, f) == 1) all.push_back(r);
    std::fclose(f);
    const int want = std::min<int>(std::atoi(argv[2]), (int)all.size());
    const int threads = std::max(1, std::atoi(argv[3]));
    std::vector<CheckRec> pos;
    for (int i = 0; i < want; i++) pos.push_back(all[(size_t)i * all.size() / want]);
    CrossfishDev::init_mini_lut();
    std::atomic<int> next{0};
    std::atomic<long long> searches{0}, nodes{0}, games{0}, game_moves{0}, skipped{0}, scratch{0};
    const auto t0 = std::chrono::steady_clock::now();
    auto worker = [&](int) {
        for (;;) {
            const int i = next.fetch_add(1);
            if (i >= (int)pos.size()) break;
            GlobalBoard b;
            if (!prepare_board_for_search(b, pos[i].s)) { skipped++; continue; }
            {
                auto bot = std::make_unique<CrossfishDev>();
                (void)bot->evaluate(b);
                scratch++;
            }
            for (int d : {6, 8, 10}) {
                if (d == 10 && i % 3) continue;
                auto bot = std::make_unique<CrossfishDev>();
                GlobalBoard copy = b;
                int score = 0;
                bot->search_fixed_depth(copy, d, score);
                searches++;
                nodes += bot->nodes;
            }
            if (i % 5 == 0) {
                auto p0 = std::make_unique<CrossfishDev>();
                auto p1 = std::make_unique<CrossfishDev>();
                GlobalBoard g = b;
                while (g.checkWinner() == -1) {
                    CrossfishDev &bot = (g.n_moves & 1) ? *p1 : *p0;
                    Move m = bot.getMove(g, std::chrono::milliseconds(3));
                    nodes += bot.nodes;
                    g.makeMove(m);
                    game_moves++;
                }
                games++;
            }
        }
    };
    std::vector<std::thread> pool;
    for (int t = 0; t < threads; t++) pool.emplace_back(worker, t);
    std::atomic<bool> done{false};
    std::thread progress([&] {  // timestamped progress with ETA every 15 s
        auto last = std::chrono::steady_clock::now();
        while (!done) {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            auto now = std::chrono::steady_clock::now();
            if (now - last < std::chrono::seconds(15)) continue;
            last = now;
            const double el = std::chrono::duration<double>(now - t0).count();
            const int k = std::min<int>(next.load(), (int)pos.size());
            std::time_t tt = std::time(nullptr);
            char hms[16];
            std::strftime(hms, sizeof(hms), "%H:%M:%S", std::localtime(&tt));
            std::printf("[%s] %d/%zu positions, %.0f s elapsed, ETA %.0f s, evals checked %lld\n", hms, k,
                        pos.size(), el, k ? el * (pos.size() - k) / k : 0.0, fnnue::g_check.evals.load());
            std::fflush(stdout);
        }
    });
    for (auto &t : pool) t.join();
    done = true;
    progress.join();
    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("fast_check: %zu positions (%lld skipped), %lld fixed-depth searches, %lld games (%lld moves), "
                "%lld scratch evals, %lld nodes, %.1f s\n",
                pos.size(), skipped.load(), searches.load(), games.load(), game_moves.load(), scratch.load(),
                nodes.load(), secs);
    std::fflush(stdout);
    return 0;
}
