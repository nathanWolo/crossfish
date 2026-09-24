// Sample positions at which qsearch is called during self-play, for labeling.
#define main tb_main
#include "test_bots.cpp"
#undef main
#include "crossfish_leaf.hpp"
int main(int argc, char **argv) {
    const int games = atoi(argv[1]), ms = atoi(argv[2]); const char *out = argv[3];
    const uint64_t threshold = argc > 4 ? strtoull(argv[4], nullptr, 10) : 1;   // prob = threshold / 4096
    const int threads = 4;
    std::vector<std::vector<NnueDumpPos>> per(threads);
    std::atomic<int> next{0};
    std::vector<std::thread> pool;
    for (int t = 0; t < threads; t++) pool.emplace_back([&, t] {
        std::mt19937_64 rng(1000 + t);
        std::vector<std::string> sink;
        while (true) {
            int g = next.fetch_add(1); if (g >= games) break;
            auto a = std::make_unique<CrossfishLeaf>(), b = std::make_unique<CrossfishLeaf>();
            for (auto *e : {a.get(), b.get()}) { e->leaf_sink = &sink; e->leaf_threshold = threshold; e->leaf_rng = rng(); }
            GlobalBoard board; Move buf[81];
            int opening = 4 + rng() % 5;
            for (int p = 0; p < opening && board.checkWinner() == -1; p++) { int k = board.fillLegalMoves(buf); board.makeMove(buf[rng() % k]); }
            while (board.checkWinner() == -1) {
                auto &e = (board.n_moves & 1) ? *b : *a;
                board.makeMove(e.getMove(board, std::chrono::milliseconds(ms)));
            }
            for (auto &st : sink) {
                GlobalBoard chk; if (!load_utttai_state(chk, st.c_str()) || chk.checkWinner() != -1) continue;
                NnueDumpPos r{}; memcpy(r.s, st.data(), 93); per[t].push_back(r);
            }
            sink.clear();
            if (g % 200 == 0) fprintf(stderr, "game %d samples(thread %d)=%zu\n", g, t, per[t].size());
        }
    });
    for (auto &th : pool) th.join();
    std::vector<NnueDumpPos> all;
    for (auto &v : per) all.insert(all.end(), v.begin(), v.end());
    write_nnue_dump(out, all);
    printf("games %d samples %zu\n", games, all.size());
}
