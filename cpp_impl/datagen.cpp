// Training-data generator for the evaluation (documentation/eval_data.md).
//
// Records are fixed 128-byte DgRec structs with no file header, so files can
// be appended to, concatenated and memory-mapped (numpy dtype in
// tools/eval_data.py).
//
//   datagen play  OUT N_POSITIONS MS THREADS SEED [OPENINGS.txt]
//       Crossfish self-play until N_POSITIONS records exist in OUT (appends,
//       so an interrupted run resumes). Each game starts from one of:
//         - 0..8 uniform-random plies                      (source 0)
//         - a position from OPENINGS (93-char states)       (source 1)
//         - 9..20 uniform-random plies                      (source 2)
//       and Crossfish then plays every move at MS ms, except that before ply
//       40 a move is sometimes drawn uniformly from the moves a depth-4 search
//       puts within SOFT_MARGIN of the best. The result is valid (flag 1) for
//       every position after the last uniform-random move.
//
//   datagen label IN OUT DEPTH THREADS
//       Fills hce, static_eval and search (fixed-depth, full window, current
//       evaluator, mates clamped to +/-20000) for every record of IN, in
//       order. Appends to OUT in blocks, so an interrupted run resumes.
//       DEPTH 0 refreshes only hce and static_eval (for a changed evaluator)
//       and keeps each record's search label.
//
//   datagen diffs IN OUT
//       Writes the ten HCE feature differences (eval_diffs, player 0 minus
//       player 1) of every record as int8[10], for training the HCE weights.
#define main tb_main
#include "test_bots.cpp"
#undef main

#include <climits>
#include <ctime>
#include <functional>

#pragma pack(push, 1)
struct DgRec {
    char s[93];            // UTTTAI state, ASCII digits, s[92] = '0'
    uint8_t source;        // 0 light random, 1 uttt.ai opening, 2 heavy random, 3 uttt.ai self-play
    uint8_t ply;           // stones on the board
    int8_t result;         // game result for the side to move: 1, 0, -1
    uint8_t flags;         // 1 result valid, 2 search labeled, 4 uttt value, 8 uttt root q
    uint8_t pad[3];
    uint32_t game;         // game id, unique within a file
    int32_t hce;           // static HCE, side to move
    int32_t static_eval;   // HCE + MiniNet + macro, side to move
    int32_t search;        // fixed-depth search score, side to move
    int32_t game_score;    // in-game root score at the move (0 if none)
    float uttt_q;          // uttt.ai MCTS root value, side to move
    float uttt_v;          // uttt.ai network value, side to move
};
#pragma pack(pop)
static_assert(sizeof(DgRec) == 128, "DgRec must stay 128 bytes");

static constexpr int SOFT_MAX_PLY = 40;
static constexpr int SOFT_DEPTH = 4;
static constexpr int SOFT_MARGIN = 300;
static constexpr double SOFT_PROB = 0.08;

static std::string now_hms() {
    std::time_t t = std::time(nullptr);
    char buf[16];
    std::strftime(buf, sizeof(buf), "%H:%M:%S", std::localtime(&t));
    return buf;
}

static long long file_records(const char *path) {
    std::error_code ec;
    auto size = fs::file_size(path, ec);
    if (ec) return 0;
    return (long long)(size / sizeof(DgRec));
}

// Before appending to a file an interrupted run left behind: drop a partly
// written last record, or every record appended after it would be misaligned.
static long long resume_records(const char *path) {
    std::error_code ec;
    auto size = fs::file_size(path, ec);
    if (ec) return 0;
    long long n = (long long)(size / sizeof(DgRec));
    if (size % sizeof(DgRec) != 0) {
        fs::resize_file(path, (uintmax_t)n * sizeof(DgRec), ec);
        if (ec) { std::fprintf(stderr, "cannot trim the partial record of %s\n", path); std::exit(1); }
        std::printf("dropped a partial record at the end of %s\n", path);
    }
    return n;
}

static std::vector<std::string> read_openings(const char *path) {
    std::vector<std::string> out;
    std::ifstream in(path);
    std::string line;
    while (std::getline(in, line)) {
        while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) line.pop_back();
        if (line.size() == 93) out.push_back(line);
    }
    return out;
}

static void fill_position(DgRec &r, const GlobalBoard &b, NnueNet &enc) {
    std::memset(&r, 0, sizeof(r));
    enc.encode_state(b, r.s);
    r.ply = (uint8_t)b.n_moves;
}

// A move drawn uniformly from those within SOFT_MARGIN of the best at depth
// SOFT_DEPTH. An immediate game win is always taken.
static Move soft_random_move(GlobalBoard &b, CrossfishDev &scorer, std::mt19937_64 &rng) {
    Move legal[81];
    int n = b.fillLegalMoves(legal);
    int scores[81];
    int best = INT_MIN;
    const int mover = b.n_moves & 1;
    for (int i = 0; i < n; i++) {
        b.makeMove(legal[i]);
        int w = b.checkWinner();
        int s;
        if (w == -1) {
            int child = 0;
            s = scorer.search_fixed_depth(b, SOFT_DEPTH, child) ? -child : -30000;
        } else if (w == 2) {
            s = 0;
        } else {
            // Deciding the last live miniboard can lose on the count tiebreak.
            s = w == mover ? 30000 : -30000;
        }
        b.unmakeMove();
        scores[i] = s;
        best = std::max(best, s);
    }
    if (best >= 20000) {
        for (int i = 0; i < n; i++) if (scores[i] == best) return legal[i];
    }
    int pick[81], k = 0;
    for (int i = 0; i < n; i++) if (scores[i] >= best - SOFT_MARGIN) pick[k++] = i;
    return legal[pick[rng() % (uint64_t)k]];
}

static std::atomic<long long> g_written{0};
static std::atomic<long long> g_games{0};
static std::atomic<uint32_t> g_next_game{0};
static std::mutex g_out_mutex;

static void play_worker(const char *out_path, long long target, int ms, uint64_t seed, int tid,
                        const std::vector<std::string> &openings) {
    std::mt19937_64 rng(seed * 0x9E3779B97F4A7C15ull + (uint64_t)tid * 7919u + 1);
    auto bot = std::make_unique<CrossfishDev>();
    auto scorer = std::make_unique<CrossfishDev>();
    NnueNet enc;
    Move legal[81];
    std::uniform_real_distribution<double> unif(0.0, 1.0);
    while (g_written.load() < target) {
        GlobalBoard b;
        std::vector<DgRec> recs;
        double r = unif(rng);
        uint8_t source;
        int n_random;
        if (!openings.empty() && r < 0.35) {
            source = 1;
            n_random = 0;
            const std::string &st = openings[rng() % openings.size()];
            if (!prepare_board_for_search(b, st.c_str())) continue;
        } else if (r < 0.80) {
            source = 0;
            n_random = (int)(rng() % 9);
        } else {
            source = 2;
            n_random = 9 + (int)(rng() % 12);
        }
        int last_random_rec = -1;  // positions up to this index precede a random move
        for (int i = 0; i < n_random && b.checkWinner() == -1; i++) {
            DgRec rec;
            fill_position(rec, b, enc);
            recs.push_back(rec);
            last_random_rec = (int)recs.size() - 1;
            int n = b.fillLegalMoves(legal);
            b.makeMove(legal[rng() % (uint64_t)n]);
        }
        while (b.checkWinner() == -1) {
            DgRec rec;
            fill_position(rec, b, enc);
            Move m;
            if (b.n_moves < SOFT_MAX_PLY && unif(rng) < SOFT_PROB) {
                m = soft_random_move(b, *scorer, rng);
            } else {
                m = bot->getMove(b, std::chrono::milliseconds(ms));
                rec.game_score = bot->completed_root_score;
            }
            recs.push_back(rec);
            b.makeMove(m);
        }
        int winner = b.checkWinner();
        for (int i = 0; i < (int)recs.size(); i++) {
            DgRec &rec = recs[i];
            int stm = rec.ply & 1;
            rec.result = winner == 2 ? 0 : (winner == stm ? 1 : -1);
            rec.flags = i > last_random_rec ? 1 : 0;
            rec.source = source;
        }
        {
            std::lock_guard<std::mutex> lock(g_out_mutex);
            if (g_written.load() >= target) break;
            uint32_t game_id = g_next_game++;
            for (DgRec &rec : recs) rec.game = game_id;
            FILE *f = std::fopen(out_path, "ab");
            if (!f) { std::fprintf(stderr, "cannot append %s\n", out_path); std::exit(1); }
            std::fwrite(recs.data(), sizeof(DgRec), recs.size(), f);
            std::fclose(f);
            g_written += (long long)recs.size();
            g_games++;
        }
    }
}

static void progress_loop(long long start, long long target, const char *what,
                          std::function<long long()> extra_games) {
    auto t0 = std::chrono::steady_clock::now();
    while (g_written.load() < target) {
        for (int i = 0; i < 30 && g_written.load() < target; i++)
            std::this_thread::sleep_for(std::chrono::seconds(1));
        long long done = g_written.load();
        double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        double rate = (done - start) / std::max(1.0, sec);
        double eta = rate > 0 ? (target - done) / rate : 0;
        std::printf("[%s] %s %lld/%lld (%.1f%%) %.0f/s games=%lld eta %.0fm\n", now_hms().c_str(), what,
                    done, target, 100.0 * done / target, rate, extra_games(), eta / 60.0);
        std::fflush(stdout);
    }
}

static int cmd_play(int argc, char **argv) {
    if (argc < 7) {
        std::fprintf(stderr, "usage: datagen play OUT N_POSITIONS MS THREADS SEED [OPENINGS.txt]\n");
        return 2;
    }
    const char *out = argv[2];
    long long target = std::atoll(argv[3]);
    int ms = std::atoi(argv[4]);
    int threads = std::atoi(argv[5]);
    uint64_t seed = std::strtoull(argv[6], nullptr, 10);
    std::vector<std::string> openings;
    if (argc >= 8) {
        openings = read_openings(argv[7]);
        std::printf("openings: %zu from %s\n", openings.size(), argv[7]);
        if (openings.empty()) {  // a missing or malformed file would silently drop source 1
            std::fprintf(stderr, "no 93-character openings in %s\n", argv[7]);
            return 1;
        }
    }
    CrossfishDev::init_mini_lut();
    long long start = resume_records(out);
    g_written = start;
    if (start > 0) {  // resume: continue the game ids already written
        FILE *f = std::fopen(out, "rb");
        DgRec rec;
        uint32_t max_id = 0;
        while (std::fread(&rec, sizeof(rec), 1, f) == 1) max_id = std::max(max_id, rec.game);
        std::fclose(f);
        g_next_game = max_id + 1;
    }
    // Resume: a new seed, so a restart does not replay the games already written.
    seed += (uint64_t)start;
    std::printf("[%s] play -> %s: have %lld, target %lld, %d ms/move, %d threads, seed %llu\n",
                now_hms().c_str(), out, start, target, ms, threads, (unsigned long long)seed);
    std::fflush(stdout);
    std::vector<std::thread> pool;
    for (int t = 0; t < threads; t++)
        pool.emplace_back(play_worker, out, target, ms, seed, t,
                          std::cref(openings));
    progress_loop(start, target, "positions", [] { return g_games.load(); });
    for (auto &t : pool) t.join();
    std::printf("[%s] done: %lld records in %s\n", now_hms().c_str(), file_records(out), out);
    return 0;
}

static int cmd_label(int argc, char **argv) {
    if (argc < 6) {
        std::fprintf(stderr, "usage: datagen label IN OUT DEPTH THREADS\n");
        return 2;
    }
    const char *in_path = argv[2];
    const char *out_path = argv[3];
    int depth = std::atoi(argv[4]);
    int threads = std::atoi(argv[5]);
    long long total = file_records(in_path);
    long long have = resume_records(out_path);
    std::printf("[%s] label %s -> %s: depth %d, %d threads, %lld records, %lld already done\n",
                now_hms().c_str(), in_path, out_path, depth, threads, total, have);
    std::fflush(stdout);
    CrossfishDev::init_mini_lut();
    FILE *in = std::fopen(in_path, "rb");
    if (!in) { std::fprintf(stderr, "cannot read %s\n", in_path); return 1; }
    const long long BLOCK = 50000;
    std::vector<std::unique_ptr<CrossfishDev>> bots;
    for (int t = 0; t < threads; t++) bots.push_back(std::make_unique<CrossfishDev>());
    g_written = have;
    auto t0 = std::chrono::steady_clock::now();
    long long done_here = 0;
    long long bad_total = 0;
    for (long long base = have; base < total; base += BLOCK) {
        long long n = std::min(BLOCK, total - base);
        std::vector<DgRec> recs((size_t)n);
#ifdef _WIN32
        _fseeki64(in, base * (long long)sizeof(DgRec), SEEK_SET);
#else
        fseeko(in, (off_t)(base * (long long)sizeof(DgRec)), SEEK_SET);
#endif
        if ((long long)std::fread(recs.data(), sizeof(DgRec), (size_t)n, in) != n) {
            std::fprintf(stderr, "short read at %lld\n", base);
            return 1;
        }
        std::atomic<long long> next{0}, bad{0};
        auto worker = [&](int t) {
            CrossfishDev &bot = *bots[t];
            GlobalBoard b;
            for (;;) {
                long long i = next.fetch_add(1);
                if (i >= n) break;
                DgRec &r = recs[(size_t)i];
                if (!prepare_board_for_search(b, r.s)) { bad++; continue; }
                r.hce = bot.evaluate_hce(b);
                r.static_eval = bot.evaluate(b);
                int score = 0;
                if (depth == 0) continue;  // statics only: keep the existing search label
                if (bot.search_fixed_depth(b, depth, score)) {
                    r.search = score;
                    r.flags |= 2;
                } else {
                    bad++;
                }
            }
        };
        std::vector<std::thread> pool;
        for (int t = 0; t < threads; t++) pool.emplace_back(worker, t);
        for (auto &t : pool) t.join();
        FILE *out = std::fopen(out_path, "ab");
        if (!out) { std::fprintf(stderr, "cannot append %s\n", out_path); return 1; }
        std::fwrite(recs.data(), sizeof(DgRec), (size_t)n, out);
        std::fclose(out);
        done_here += n;
        bad_total += bad.load();
        double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        double rate = done_here / std::max(1e-9, sec);
        long long done = base + n;
        std::printf("[%s] labeled %lld/%lld (%.1f%%) %.0f/s unlabeled=%lld eta %.0fm\n", now_hms().c_str(),
                    done, total, 100.0 * done / total, rate, bad_total, (total - done) / rate / 60.0);
        std::fflush(stdout);
    }
    std::fclose(in);
    std::printf("[%s] done: %lld records in %s\n", now_hms().c_str(), file_records(out_path), out_path);
    return 0;
}

static int cmd_diffs(int argc, char **argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: datagen diffs IN OUT\n");
        return 2;
    }
    long long total = file_records(argv[2]);
    FILE *in = std::fopen(argv[2], "rb");
    FILE *out = std::fopen(argv[3], "wb");
    if (!in || !out) { std::fprintf(stderr, "cannot open files\n"); return 1; }
    CrossfishDev::init_mini_lut();
    auto bot = std::make_unique<CrossfishDev>();
    DgRec r;
    GlobalBoard b;
    long long bad = 0;
    while (std::fread(&r, sizeof(r), 1, in) == 1) {
        int d[CrossfishDev::N_EVAL_WEIGHTS] = {};
        int8_t row[CrossfishDev::N_EVAL_WEIGHTS] = {};
        if (prepare_board_for_search(b, r.s)) {
            bot->eval_diffs(b, d);
            for (int i = 0; i < CrossfishDev::N_EVAL_WEIGHTS; i++) row[i] = (int8_t)d[i];
        } else {
            bad++;
        }
        std::fwrite(row, 1, sizeof(row), out);
    }
    std::fclose(in);
    std::fclose(out);
    std::printf("diffs for %lld records (%lld unloadable, written as zeros) -> %s\n", total, bad, argv[3]);
    return 0;
}

int main(int argc, char **argv) {
    if (argc >= 2 && std::strcmp(argv[1], "diffs") == 0) return cmd_diffs(argc, argv);
    if (argc >= 2 && std::strcmp(argv[1], "play") == 0) return cmd_play(argc, argv);
    if (argc >= 2 && std::strcmp(argv[1], "label") == 0) return cmd_label(argc, argv);
    std::fprintf(stderr, "usage: datagen play|label|diffs ... (see the header comment)\n");
    return 2;
}
