// Multi-threaded search throughput of the Dev (fast NNUE) and Prev (shipped) engines: the cache
// behaviour under the match setting (up to 14 threads sharing one process, one net table, one L3).
//
//   search_mt POSITIONS.cfdg N_POS DEPTH THREADS
//
// N_POS records spread over POSITIONS; each thread builds one engine and runs search_fixed_depth on
// the next position until none are left (the engine, its TT and eval cache are reused as in a game;
// fresh engines per position make the TT allocation dominate at 14 threads).
// Dev runs first on all threads, then Prev on all threads. Prints per engine the total nodes, the
// wall time of the phase, the aggregate nodes/s and the mean per-thread nodes/s (nodes over the summed
// search time, engine construction excluded), and the Dev/Prev ratios.
#define main tb_main
#include "test_bots.cpp"
#undef main

#include <atomic>
#include <thread>

#pragma pack(push, 1)
struct MtRec {
    char s[93];
    uint8_t rest[35];
};
#pragma pack(pop)

template <typename Engine>
static void phase(const char *name, const std::vector<MtRec> &pos, int depth, int threads, double &agg, double &per) {
    std::atomic<int> next{0};
    std::atomic<long long> nodes{0}, nsec{0};
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<std::thread> pool;
    for (int t = 0; t < threads; t++)
        pool.emplace_back([&] {
            auto e = std::make_unique<Engine>();  // one engine per thread, reused (as in games)
            for (;;) {
                const int i = next.fetch_add(1);
                if (i >= (int)pos.size()) break;
                GlobalBoard b;
                if (!prepare_board_for_search(b, pos[i].s)) continue;
                int score = 0;
                const auto s0 = std::chrono::steady_clock::now();
                e->search_fixed_depth(b, depth, score);
                nsec += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - s0).count();
                nodes += e->nodes;  // search_fixed_depth resets the counter
            }
        });
    for (auto &th : pool) th.join();
    const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    agg = nodes.load() / wall;
    per = nodes.load() / (nsec.load() * 1e-9);
    std::printf("%s: %lld nodes, %.2f s wall, aggregate %.1fM nodes/s, per thread %.2fM nodes/s\n", name, nodes.load(),
                wall, agg / 1e6, per / 1e6);
    std::fflush(stdout);
}

int main(int argc, char **argv) {
    if (argc < 5) {
        std::fprintf(stderr, "usage: search_mt POSITIONS.cfdg N_POS DEPTH THREADS\n");
        return 2;
    }
    FILE *f = std::fopen(argv[1], "rb");
    if (!f) return 1;
    std::vector<MtRec> all;
    MtRec r;
    while (std::fread(&r, sizeof(r), 1, f) == 1) all.push_back(r);
    std::fclose(f);
    const int want = std::min<int>(std::atoi(argv[2]), (int)all.size());
    const int depth = std::atoi(argv[3]), threads = std::atoi(argv[4]);
    std::vector<MtRec> pos;
    for (int i = 0; i < want; i++) pos.push_back(all[(size_t)i * all.size() / want]);
    CrossfishDev::init_mini_lut();
    CrossfishPrev::init_mini_lut();
    { auto warm = std::make_unique<CrossfishDev>(); }  // load the net before timing
    double da, dp, pa, pp;
    phase<CrossfishDev>("dev ", pos, depth, threads, da, dp);
    phase<CrossfishPrev>("prev", pos, depth, threads, pa, pp);
    std::printf("search_mt: %d threads, depth %d, %zu positions: dev/prev aggregate %.1f%%, per thread %.1f%%\n", threads,
                depth, pos.size(), 100 * da / pa, 100 * dp / pp);
    return 0;
}
