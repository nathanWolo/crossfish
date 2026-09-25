// Generates the full-coverage opening book as text (see documentation/play_book.md).
//
// Every opponent reply is covered; each of our positions gets a long search
// for its book move. We move first: the bot opens center-center, then the book
// covers our next <depth_first> moves. We move second: the opponent's
// center-center, then our first <depth_second> moves. Positions equivalent under the 8
// board symmetries are searched once. Levels are processed shallowest first,
// so an interrupted run still leaves a complete shallower book on disk.
//
//   play_book_gen selftest
//   play_book_gen <out.txt> [depth_first=5] [depth_second=4] [search_ms=2000] [threads=7]
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <memory>
#include <mutex>
#include <random>
#include <set>
#include <thread>
#include <vector>
#pragma GCC optimize("O3")
#pragma GCC target("avx2,bmi,bmi2,lzcnt,popcnt")
#include "global_board.hpp"
#include "crossfish_dev.hpp"
#include "play_book_text.hpp"

// Symmetry sanity check: transformed keys, legal-move mapping and inverses.
static int selftest() {
    std::mt19937 rng(1234);
    Move buf[81], tbuf[81];
    int checked = 0;
    for (int g = 0; g < 400; g++) {
        GlobalBoard b;
        std::vector<Move> hist;
        int plies = 1 + rng() % 40;
        for (int i = 0; i < plies && b.checkWinner() == -1; i++) {
            int n = b.fillLegalMoves(buf);
            Move m = buf[rng() % n];
            b.makeMove(m);
            hist.push_back(m);
        }
        for (int t = 0; t < 8; t++) {
            GlobalBoard tb;
            for (Move m : hist) tb.makeMove(sym_move(t, m));  // aborts on an illegal mapped move
            if (key_in(b, t) != key_in(tb, 0)) { std::printf("FAIL key t=%d\n", t); return 1; }
            if (b.checkWinner() != -1) continue;
            int n = b.fillLegalMoves(buf), tn = tb.fillLegalMoves(tbuf);
            if (n != tn) { std::printf("FAIL move count t=%d\n", t); return 1; }
            std::set<int> tset;
            for (int i = 0; i < tn; i++) tset.insert(tbuf[i].mini_board * 9 + tbuf[i].square);
            for (int i = 0; i < n; i++) {
                Move tm = sym_move(t, buf[i]), back = sym_move(sym_inverse(t), tm);
                if (!tset.count(tm.mini_board * 9 + tm.square)) { std::printf("FAIL move map t=%d\n", t); return 1; }
                if (back.mini_board != buf[i].mini_board || back.square != buf[i].square) {
                    std::printf("FAIL inverse t=%d\n", t);
                    return 1;
                }
            }
            int ta, tt;
            if (canonical_key(b, ta) != canonical_key(tb, tt)) { std::printf("FAIL canonical t=%d\n", t); return 1; }
            checked++;
        }
    }
    std::printf("symmetry selftest OK (%d position/transform pairs)\n", checked);
    return 0;
}

struct Pending {
    GlobalBoard board;
    int our_idx;  // stored moves already made on this line
};

int main(int argc, char **argv) {
    if (argc >= 2 && std::string(argv[1]) == "selftest") return selftest();
    if (argc < 2) {
        std::fprintf(stderr, "usage: play_book_gen <out.txt> [depth_first] [depth_second] [search_ms] [threads]\n");
        return 2;
    }
    std::string out = argv[1];
    int depth_first = argc > 2 ? std::atoi(argv[2]) : 5;
    int depth_second = argc > 3 ? std::atoi(argv[3]) : 4;
    int search_ms = argc > 4 ? std::atoi(argv[4]) : 2000;
    int threads = argc > 5 ? std::atoi(argv[5]) : 7;
    CrossfishDev::init_mini_lut();

    Book book;
    std::set<std::string> claimed;
    std::mutex mu;
    auto t0 = std::chrono::steady_clock::now();
    int done_total = 0;

    // Opponent to move at `o`: queue every non-terminal reply we have not seen.
    auto expand = [&](GlobalBoard o, int our_idx, std::vector<Pending> &next) {
        Move buf[81];
        int n = o.fillLegalMoves(buf);
        for (int i = 0; i < n; i++) {
            GlobalBoard c = o;
            c.makeMove(buf[i]);
            if (c.checkWinner() != -1) continue;
            int t;
            if (claimed.insert(canonical_key(c, t)).second) next.push_back({c, our_idx});
        }
    };

    std::vector<Pending> level;
    GlobalBoard first;  // we move first: the bot opens center-center
    first.makeMove(Move{4, 4});
    expand(first, 0, level);
    {  // we move second: the book assumes the opponent opened center-center
        GlobalBoard second;
        second.makeMove(Move{4, 4});
        int t;
        claimed.insert(canonical_key(second, t));
        level.push_back({second, 0});
    }

    for (int depth = 0; !level.empty(); depth++) {
        std::vector<Pending> next;
        std::atomic<size_t> cursor{0};
        std::atomic<int> done{0};
        auto work = [&] {
            for (size_t i; (i = cursor++) < level.size();) {
                Pending &p = level[i];
                auto engine = std::make_unique<CrossfishDev>();
                Move m = engine->getMove(p.board, std::chrono::milliseconds(search_ms));
                int t;
                std::string k = canonical_key(p.board, t);
                bool we_first = p.board.n_moves % 2 == 0;
                std::lock_guard<std::mutex> lk(mu);
                book.entries[k] = BookEntry{sym_move(t, m), engine->root_score, p.board.n_moves, 0};
                if (p.our_idx + 1 < (we_first ? depth_first : depth_second)) {
                    GlobalBoard o = p.board;
                    o.makeMove(m);
                    if (o.checkWinner() == -1) expand(o, p.our_idx + 1, next);
                }
                int d = ++done;
                if (d % 100 == 0 || d == (int)level.size()) {
                    double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                    std::time_t now = std::time(nullptr);
                    std::time_t end = now + (std::time_t)(sec / (done_total + d) * (level.size() - d));
                    char a[16], e[16];
                    std::strftime(a, sizeof a, "%H:%M:%S", std::localtime(&now));
                    std::strftime(e, sizeof e, "%H:%M", std::localtime(&end));
                    std::printf("[%s] level %d: %d/%zu positions  (level ETA %s)\n", a, depth + 1, d, level.size(), e);
                    std::fflush(stdout);
                }
            }
        };
        std::vector<std::thread> ts;
        for (int i = 0; i < threads; i++) ts.emplace_back(work);
        for (auto &t : ts) t.join();
        done_total += (int)level.size();
        book.save(out);
        std::printf("level %d complete: book now %zu positions, next level %zu\n", depth + 1, book.entries.size(), next.size());
        std::fflush(stdout);
        level.swap(next);
    }
    std::printf("done: %zu positions -> %s\n", book.entries.size(), out.c_str());
    return 0;
}
