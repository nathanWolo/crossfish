// Measures the shipped opening book (play_book_data.hpp) under the CodinGame
// protocol: the first player opens center-center and every other move gets
// 90 ms. In book, the book side still runs its normal search (warming its
// tables, exactly as the CodinGame bot does) and then plays the book move.
//
//   play_book_match <games> <mode> [threads=7] [seed=1]
//     mode 0: book side vs the plain engine (a direct head-to-head).
//     mode 1: diverse opponent: half the time it opens off center, and on its
//             first three moves it plays a random reply within 100 of its best
//             (depth-10 search) half the time. Each opening is played twice,
//             with and without the book; the book's value is the difference.
//     mode 2: a different engine as opponent, paired the same way. Build with
//             -DPLAY_BOOK_OPPONENT_HEADER='"path.hpp"' providing class
//             CrossfishOld (see documentation/play_book.md).
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>
#pragma GCC optimize("O3")
#pragma GCC target("avx2,bmi,bmi2,lzcnt,popcnt")
#include "global_board.hpp"
#include "crossfish_dev.hpp"
#include "play_book.hpp"
#ifdef PLAY_BOOK_OPPONENT_HEADER
#include PLAY_BOOK_OPPONENT_HEADER
#endif

static constexpr int MOVE_MS = 90;

struct Tally {
    int w = 0, d = 0, l = 0;
    long long book_moves = 0;
    long long exit_ply_sum = 0;
    void add(int r) { r > 0 ? w++ : (r == 0 ? d++ : l++); }
    void merge(const Tally &o) { w += o.w; d += o.d; l += o.l; book_moves += o.book_moves; exit_ply_sum += o.exit_ply_sum; }
    int n() const { return w + d + l; }
};

static void elo(const Tally &t, double &e, double &ci) {
    double n = t.n(), s = (t.w + 0.5 * t.d) / n;
    double var = (t.w * std::pow(1 - s, 2) + t.d * std::pow(0.5 - s, 2) + t.l * s * s) / n;
    double sd = std::sqrt(var / n);
    auto f = [](double p) { p = std::min(0.999, std::max(0.001, p)); return -400 * std::log10(1 / p - 1); };
    e = f(s);
    ci = (f(s + 1.96 * sd) - f(s - 1.96 * sd)) / 2;
}

static Move diverse_pick(GlobalBoard &b, std::mt19937 &rng, CrossfishDev &eng) {
    Move buf[81];
    int n = b.fillLegalMoves(buf);
    auto probe = std::make_unique<CrossfishDev>();
    std::vector<std::pair<int, Move>> scored;
    int best = -1000000;
    for (int i = 0; i < n; i++) {
        GlobalBoard c = b;
        c.makeMove(buf[i]);
        int v, w = c.checkWinner();
        if (w != -1) v = (w == 2) ? 0 : 100000;
        else { int s = 0; probe->search_fixed_depth(c, 10, s); v = -s; }
        scored.push_back({v, buf[i]});
        best = std::max(best, v);
    }
    std::vector<Move> ok;
    for (auto &x : scored) if (best - x.first <= 100) ok.push_back(x.second);
    Move m = eng.getMove(b, std::chrono::milliseconds(MOVE_MS));  // keep its tables warm
    return (rng() % 2) ? ok[rng() % ok.size()] : m;
}

// +1 book side wins, 0 draw, -1 loss.
template <class Opp>
static int play(bool book_first, bool use_book, int mode, uint32_t seed, Tally &t) {
    std::mt19937 rng(seed);
    auto me = std::make_unique<CrossfishDev>();
    auto opp = std::make_unique<Opp>();
    GlobalBoard b;
    int my_side = book_first ? 0 : 1, in_book = 0, exit_ply = -1, opp_early = 0;
    while (b.checkWinner() == -1) {
        bool mine = (b.n_moves % 2) == my_side;
        Move m;
        if (b.n_moves == 0) {
            m = (!mine && mode == 1 && rng() % 2) ? Move{(int)(rng() % 9), (int)(rng() % 9)} : Move{4, 4};
        } else if (mine) {
            m = me->getMove(b, std::chrono::milliseconds(MOVE_MS));
            Move bm;
            if (exit_ply < 0 && use_book && pb_lookup(b, bm)) { m = bm; in_book++; }
            else if (exit_ply < 0) exit_ply = b.n_moves;
        } else if (mode == 1 && opp_early < 3) {
            opp_early++;
            if constexpr (std::is_same<Opp, CrossfishDev>::value) m = diverse_pick(b, rng, *opp);
        } else {
            m = opp->getMove(b, std::chrono::milliseconds(MOVE_MS));
        }
        b.makeMove(m);
    }
    t.book_moves += in_book;
    t.exit_ply_sum += exit_ply < 0 ? b.n_moves : exit_ply;
    int w = b.checkWinner();
    return w == 2 ? 0 : (w == my_side ? 1 : -1);
}

int main(int argc, char **argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: play_book_match <games> <mode> [threads] [seed]\n"); return 2; }
    int games = std::atoi(argv[1]), mode = std::atoi(argv[2]);
    int threads = argc > 3 ? std::atoi(argv[3]) : 7;
    uint32_t seed0 = argc > 4 ? (uint32_t)std::atoi(argv[4]) : 1;
#ifndef PLAY_BOOK_OPPONENT_HEADER
    if (mode == 2) { std::fprintf(stderr, "mode 2 needs -DPLAY_BOOK_OPPONENT_HEADER\n"); return 2; }
#endif
    CrossfishDev::init_mini_lut();
    if (!pb_init<GlobalBoard, Move>()) { std::fprintf(stderr, "book failed to decode\n"); return 1; }
    std::printf("book %d positions, %d %s, mode %d\n", PLAY_BOOK_ENTRIES, games, mode == 0 ? "games" : "paired openings", mode);
    std::fflush(stdout);

    Tally with[2], without[2];  // [book side moves first]
    std::mutex mu;
    std::atomic<int> next{0};
    auto t0 = std::chrono::steady_clock::now();
    auto work = [&] {
        for (int i; (i = next++) < games;) {
            bool first = i % 2 == 0;
            uint32_t seed = seed0 * 1000003u + (uint32_t)(i / 2);
            Tally a, c;
            int r = 0, r2 = 0;
            if (mode == 2) {
#ifdef PLAY_BOOK_OPPONENT_HEADER
                r = play<CrossfishOld>(first, true, mode, seed, a);
                r2 = play<CrossfishOld>(first, false, mode, seed, c);
#endif
            } else {
                r = play<CrossfishDev>(first, true, mode, seed, a);
                if (mode == 1) r2 = play<CrossfishDev>(first, false, mode, seed, c);
            }
            std::lock_guard<std::mutex> lk(mu);
            a.add(r);
            with[first].merge(a);
            if (mode != 0) without[first].add(r2);
            Tally tw, tn;
            for (int f = 0; f < 2; f++) { tw.merge(with[f]); tn.merge(without[f]); }
            int n = tw.n();
            if (n % 100 == 0 || n == games) {
                double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                std::time_t now = std::time(nullptr), end = now + (std::time_t)(sec / n * (games - n));
                char ts[16], eta[16];
                std::strftime(ts, sizeof ts, "%H:%M:%S", std::localtime(&now));
                std::strftime(eta, sizeof eta, "%H:%M", std::localtime(&end));
                double e, ci;
                elo(tw, e, ci);
                if (mode == 0) {
                    std::printf("[%s] %d/%d games  ETA %s  book vs no book %+.1f +/- %.1f\n", ts, n, games, eta, e, ci);
                } else {
                    double e2, ci2;
                    elo(tn, e2, ci2);
                    std::printf("[%s] %d/%d openings  ETA %s  with book %+.1f +/- %.1f  without %+.1f +/- %.1f  book value %+.1f\n",
                                ts, n, games, eta, e, ci, e2, ci2, e - e2);
                }
                std::fflush(stdout);
            }
        }
    };
    std::vector<std::thread> ts;
    for (int i = 0; i < threads; i++) ts.emplace_back(work);
    for (auto &t : ts) t.join();

    for (int f = 1; f >= 0; f--) {
        const Tally &t = with[f];
        std::printf("book side moves %s: %.2f book moves per game, leaves book at ply %.1f on average\n",
                    f ? "first " : "second", (double)t.book_moves / t.n(), (double)t.exit_ply_sum / t.n());
    }
    Tally tw, tn;
    for (int f = 0; f < 2; f++) { tw.merge(with[f]); tn.merge(without[f]); }
    double e, ci;
    elo(tw, e, ci);
    std::printf("%s  W %d / D %d / L %d  Elo %+.1f +/- %.1f\n", mode == 0 ? "book vs no book    " : "with book vs opp   ",
                tw.w, tw.d, tw.l, e, ci);
    if (mode != 0) {
        double e2, ci2;
        elo(tn, e2, ci2);
        std::printf("without book vs opp  W %d / D %d / L %d  Elo %+.1f +/- %.1f\nbook value %+.1f\n",
                    tn.w, tn.d, tn.l, e2, ci2, e - e2);
    }
    return 0;
}
