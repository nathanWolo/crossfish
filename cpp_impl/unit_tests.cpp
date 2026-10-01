#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <random>
#include <stack>
#include <string>
#include <vector>

#include "global_board.hpp"
#include "mini_eval.hpp"
// Candidate engines can be unit-tested without editing the tracked header:
//   -DCROSSFISH_DEV_HEADER='"/path/candidate.hpp"'
#ifndef CROSSFISH_DEV_HEADER
#define CROSSFISH_DEV_HEADER "crossfish_dev.hpp"
#endif
#include CROSSFISH_DEV_HEADER
#include "play_book.hpp"

// Frozen startpos perft, matched against the independent Python oracle
// in python_impl/test_rules.py (same UTTT send-to / finished-board rules).
static const uint64_t STARTPOS_PERFT[] = {
    0,
    81ull,
    720ull,
    6336ull,
    55080ull,
    473256ull,
};

struct TestCtx {
    const char *name;
    int fails = 0;

    void check(bool cond, const char *expr, const char *file, int line) {
        if (!cond) {
            std::cerr << "  FAIL " << name << ": " << expr
                      << " (" << file << ":" << line << ")" << std::endl;
            fails++;
        }
    }
};

#define CHECK(cond) ctx.check((bool)(cond), #cond, __FILE__, __LINE__)
#define CHECK_EQ(a, b) do { \
    auto _va = (a); auto _vb = (b); \
    if (_va != _vb) { \
        std::cerr << "  FAIL " << ctx.name << ": " << #a << " == " << #b \
                  << " (" << _va << " != " << _vb << ") (" \
                  << __FILE__ << ":" << __LINE__ << ")" << std::endl; \
        ctx.fails++; \
    } \
} while (0)

static const int WIN_LINES[8] = {
    (1 << 0) + (1 << 1) + (1 << 2),
    (1 << 3) + (1 << 4) + (1 << 5),
    (1 << 6) + (1 << 7) + (1 << 8),
    (1 << 0) + (1 << 3) + (1 << 6),
    (1 << 1) + (1 << 4) + (1 << 7),
    (1 << 2) + (1 << 5) + (1 << 8),
    (1 << 0) + (1 << 4) + (1 << 8),
    (1 << 2) + (1 << 4) + (1 << 6),
};

static bool scalar_has_win(int markers) {
    for (int w : WIN_LINES) {
        if ((markers & w) == w) return true;
    }
    return false;
}

static bool same_move(const Move &a, const Move &b) {
    return a.mini_board == b.mini_board && a.square == b.square;
}

static bool contains_move(const std::vector<Move> &v, const Move &m) {
    for (const Move &x : v) {
        if (same_move(x, m)) return true;
    }
    return false;
}

static void sort_moves(std::vector<Move> &v) {
    std::sort(v.begin(), v.end(), [](const Move &a, const Move &b) {
        if (a.mini_board != b.mini_board) return a.mini_board < b.mini_board;
        return a.square < b.square;
    });
}

static bool same_move_list(std::vector<Move> a, std::vector<Move> b) {
    sort_moves(a);
    sort_moves(b);
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); i++) {
        if (!same_move(a[i], b[i])) return false;
    }
    return true;
}

static std::vector<Move> buf_to_vec(Move *buf, int n) {
    return std::vector<Move>(buf, buf + n);
}

// Independent UTTT rules oracle. Does not call GlobalBoard movegen.
struct Oracle {
    int markers[2][9] = {};
    int state[3] = {};
    int last_sq = -1;
    int n = 0;
    bool passed = false;

    int out_of_play() const { return state[0] | state[1] | state[2]; }

    std::vector<Move> legal() const {
        std::vector<Move> out;
        out.reserve(81);
        if (n == 0) {
            for (int i = 0; i < 9; i++) {
                for (int j = 0; j < 9; j++) {
                    out.push_back(Move{i, j});
                }
            }
            return out;
        }
        int oop = out_of_play();
        auto add_board = [&](int mb) {
            int marked = markers[0][mb] | markers[1][mb];
            for (int sq = 0; sq < 9; sq++) {
                if ((marked & (1 << sq)) == 0) {
                    out.push_back(Move{mb, sq});
                }
            }
        };
        if (passed || (oop & (1 << last_sq)) != 0) {
            for (int mb = 0; mb < 9; mb++) {
                if ((oop & (1 << mb)) == 0) add_board(mb);
            }
        } else {
            add_board(last_sq);
        }
        return out;
    }

    std::vector<Move> captures() const {
        std::vector<Move> out;
        int stm = n % 2;
        for (const Move &m : legal()) {
            if (scalar_has_win(markers[stm][m.mini_board] | (1 << m.square))) {
                out.push_back(m);
            }
        }
        return out;
    }

    void make(Move m) {
        int stm = n % 2;
        markers[stm][m.mini_board] |= (1 << m.square);
        if (scalar_has_win(markers[stm][m.mini_board])) {
            state[stm] |= (1 << m.mini_board);
        } else if ((markers[0][m.mini_board] | markers[1][m.mini_board]) == 511) {
            state[2] |= (1 << m.mini_board);
        }
        last_sq = m.square;
        n++;
        passed = false;
    }

    int winner() const {
        if (scalar_has_win(state[0])) return 0;
        if (scalar_has_win(state[1])) return 1;
        if (out_of_play() == 511) {
            int c0 = __builtin_popcount(state[0]);
            int c1 = __builtin_popcount(state[1]);
            if (c0 > c1) return 0;
            if (c1 > c0) return 1;
            return 2;
        }
        return -1;
    }
};

static uint64_t rebuild_zobrist(const GlobalBoard &b) {
    uint64_t h = 0;
    for (int mb = 0; mb < 9; mb++) {
        for (int sq = 0; sq < 9; sq++) {
            if (b.mini_boards[mb].markers[0] & (1 << sq)) {
                h ^= b.move_hashes[0][mb][sq];
            }
            if (b.mini_boards[mb].markers[1] & (1 << sq)) {
                h ^= b.move_hashes[1][mb][sq];
            }
        }
    }
    for (int st = 0; st < 3; st++) {
        for (int mb = 0; mb < 9; mb++) {
            if (b.mini_board_states[st] & (1 << mb)) {
                h ^= b.mini_board_hashes[st][mb];
            }
        }
    }
    if (b.n_moves % 2 == 1) {
        h ^= b.player_to_move_hash;
    }
    if (b.n_moves > 0) {
        h ^= b.legal_mini_board_hashes[b.move_history.top().square];
    }
    return h;
}

struct Snap {
    std::array<MiniBoard, 9> mini;
    std::array<int, 3> states;
    uint64_t hash;
    int n;
    bool pass;
    bool has_last;
    Move last;
};

static Snap take_snap(const GlobalBoard &b) {
    Snap s;
    s.mini = b.mini_boards;
    s.states = b.mini_board_states;
    s.hash = b.zobrist_hash;
    s.n = b.n_moves;
    s.pass = b.prev_move_was_pass;
    s.has_last = !b.move_history.empty();
    if (s.has_last) s.last = b.move_history.top();
    return s;
}

static bool snap_eq(const Snap &a, const Snap &b) {
    if (a.n != b.n || a.hash != b.hash || a.pass != b.pass || a.has_last != b.has_last) {
        return false;
    }
    if (a.has_last && !same_move(a.last, b.last)) return false;
    if (a.states != b.states) return false;
    for (int i = 0; i < 9; i++) {
        if (a.mini[i].markers != b.mini[i].markers) return false;
    }
    return true;
}

static uint64_t perft(GlobalBoard &board, int depth) {
    if (depth == 0) return 1;
    if (board.checkWinner() != -1) return 0;
    Move buf[81];
    int n = board.fillLegalMoves(buf);
    if (depth == 1) return (uint64_t)n;
    uint64_t nodes = 0;
    for (int i = 0; i < n; i++) {
        board.makeMove(buf[i]);
        nodes += perft(board, depth - 1);
        board.unmakeMove();
    }
    return nodes;
}

static uint64_t oracle_perft(Oracle &o, int depth) {
    if (depth == 0) return 1;
    if (o.winner() != -1) return 0;
    std::vector<Move> moves = o.legal();
    if (depth == 1) return (uint64_t)moves.size();
    uint64_t nodes = 0;
    for (const Move &m : moves) {
        Oracle child = o;
        child.make(m);
        nodes += oracle_perft(child, depth - 1);
    }
    return nodes;
}

static void apply_moves(GlobalBoard &board, const std::vector<Move> &moves) {
    for (const Move &m : moves) {
        board.makeMove(m);
    }
}

static std::vector<Move> random_opening(std::mt19937 &rng, int n_plies, GlobalBoard *out = nullptr) {
    GlobalBoard board;
    std::vector<Move> hist;
    for (int i = 0; i < n_plies; i++) {
        if (board.checkWinner() != -1) break;
        std::vector<Move> legal = board.getLegalMoves();
        if (legal.empty()) break;
        Move m = legal[rng() % legal.size()];
        board.makeMove(m);
        hist.push_back(m);
    }
    if (out) *out = board;
    return hist;
}

static std::vector<Move> instant_wins(GlobalBoard &board) {
    std::vector<Move> wins;
    int stm = board.n_moves % 2;
    std::vector<Move> legal = board.getLegalMoves();
    for (const Move &m : legal) {
        board.makeMove(m);
        if (board.checkWinner() == stm) wins.push_back(m);
        board.unmakeMove();
    }
    return wins;
}

static void test_startpos_counts(TestCtx &ctx) {
    GlobalBoard board;
    CHECK_EQ(board.n_moves, 0);
    CHECK_EQ(board.checkWinner(), -1);
    CHECK_EQ((int)board.getLegalMoves().size(), 81);
    Move buf[81];
    CHECK_EQ(board.fillLegalMoves(buf), 81);
    CHECK_EQ(board.fillCaptures(buf), 0);
    CHECK_EQ((int)board.get_captures().size(), 0);
    CHECK_EQ(board.zobrist_hash, 0ull);
}

static void test_send_to_same_board(TestCtx &ctx) {
    GlobalBoard board;
    board.makeMove({4, 4});
    std::vector<Move> legal = board.getLegalMoves();
    CHECK_EQ((int)legal.size(), 8);
    for (const Move &m : legal) {
        CHECK_EQ(m.mini_board, 4);
        CHECK(m.square != 4);
    }
}

static void test_send_to_other_board(TestCtx &ctx) {
    GlobalBoard board;
    board.makeMove({0, 1});
    std::vector<Move> legal = board.getLegalMoves();
    CHECK_EQ((int)legal.size(), 9);
    for (const Move &m : legal) {
        CHECK_EQ(m.mini_board, 1);
    }
}

static void test_grid_coord_roundtrip(TestCtx &ctx) {
    for (int row = 0; row < 9; row++) {
        for (int col = 0; col < 9; col++) {
            int mb = (row / 3) * 3 + (col / 3);
            int sq = (row % 3) * 3 + (col % 3);
            int r2 = (mb / 3) * 3 + (sq / 3);
            int c2 = (mb % 3) * 3 + (sq % 3);
            CHECK_EQ(r2, row);
            CHECK_EQ(c2, col);
        }
    }
}

static void test_scalar_vs_avx_wins(TestCtx &ctx) {
    GlobalBoard board;
    for (int markers = 0; markers < 512; markers++) {
        board.mini_board_states[0] = markers;
        board.mini_board_states[1] = 0;
        CHECK(board.won_avx(0) == scalar_has_win(markers));
        board.mini_board_states[0] = 0;
        board.mini_board_states[1] = markers;
        CHECK(board.won_avx(1) == scalar_has_win(markers));
    }
    board.mini_board_states[0] = 0;
    board.mini_board_states[1] = 0;
}

static void test_miniboard_win_and_draw(TestCtx &ctx) {
    std::mt19937 rng(11);
    int wins_seen = 0;
    int draws_seen = 0;
    for (int game = 0; game < 400 && (wins_seen < 5 || draws_seen < 3); game++) {
        GlobalBoard board;
        for (int ply = 0; ply < 81; ply++) {
            if (board.checkWinner() != -1) break;
            std::vector<Move> legal = board.getLegalMoves();
            if (legal.empty()) break;
            Move m = legal[rng() % legal.size()];
            int stm = board.n_moves % 2;
            int before = board.mini_board_states[0] | board.mini_board_states[1] | board.mini_board_states[2];
            board.makeMove(m);
            int after = board.mini_board_states[0] | board.mini_board_states[1] | board.mini_board_states[2];
            int newly = after & ~before;
            if (!newly) continue;
            CHECK(__builtin_popcount(newly) == 1);
            int mb = __builtin_ctz(newly);
            CHECK_EQ(mb, m.mini_board);
            if (board.mini_board_states[stm] & (1 << mb)) {
                wins_seen++;
                CHECK(scalar_has_win(board.mini_boards[mb].markers[stm]));
            } else {
                draws_seen++;
                CHECK((board.mini_board_states[2] & (1 << mb)) != 0);
                int occ = board.mini_boards[mb].markers[0] | board.mini_boards[mb].markers[1];
                CHECK_EQ(occ, 511);
                CHECK(!scalar_has_win(board.mini_boards[mb].markers[0]));
                CHECK(!scalar_has_win(board.mini_boards[mb].markers[1]));
            }
            if (board.checkWinner() == -1) {
                std::vector<Move> next = board.getLegalMoves();
                for (const Move &nm : next) {
                    CHECK(nm.mini_board != mb);
                }
            }
        }
    }
    CHECK(wins_seen >= 1);
}

static void test_free_move_when_sent_to_finished(TestCtx &ctx) {
    std::mt19937 rng(17);
    int seen = 0;
    for (int game = 0; game < 500 && seen < 8; game++) {
        GlobalBoard board;
        for (int ply = 0; ply < 81; ply++) {
            if (board.checkWinner() != -1) break;
            std::vector<Move> legal = board.getLegalMoves();
            if (legal.empty()) break;
            int oop = board.mini_board_states[0] | board.mini_board_states[1] | board.mini_board_states[2];
            bool free = board.n_moves > 0 && ((oop & (1 << board.move_history.top().square)) != 0);
            if (free) {
                seen++;
                CHECK(legal.size() > 9);
                for (const Move &m : legal) {
                    CHECK((oop & (1 << m.mini_board)) == 0);
                    int marked = board.mini_boards[m.mini_board].markers[0] | board.mini_boards[m.mini_board].markers[1];
                    CHECK((marked & (1 << m.square)) == 0);
                }
                break;
            }
            board.makeMove(legal[rng() % legal.size()]);
        }
    }
    CHECK(seen >= 1);
}

static void test_global_win_by_three_miniboards(TestCtx &ctx) {
    GlobalBoard board;
    board.mini_board_states[0] = (1 << 0) | (1 << 1) | (1 << 2);
    CHECK_EQ(board.checkWinner(), 0);
    board.mini_board_states[0] = 0;
    board.mini_board_states[1] = (1 << 0) | (1 << 4) | (1 << 8);
    CHECK_EQ(board.checkWinner(), 1);
}

static void test_count_win_when_all_decided(TestCtx &ctx) {
    // Patterns with no 3-in-a-row, so the count tiebreak is what decides.
    GlobalBoard board;
    board.mini_board_states[0] = (1 << 0) | (1 << 2) | (1 << 3) | (1 << 7);
    board.mini_board_states[1] = (1 << 1) | (1 << 4) | (1 << 6);
    board.mini_board_states[2] = (1 << 5) | (1 << 8);
    CHECK(!scalar_has_win(board.mini_board_states[0]));
    CHECK(!scalar_has_win(board.mini_board_states[1]));
    CHECK_EQ(board.checkWinner(), 0);
    board.mini_board_states[0] = (1 << 0) | (1 << 2) | (1 << 7);
    board.mini_board_states[1] = (1 << 1) | (1 << 3) | (1 << 5) | (1 << 6);
    board.mini_board_states[2] = (1 << 4) | (1 << 8);
    CHECK(!scalar_has_win(board.mini_board_states[0]));
    CHECK(!scalar_has_win(board.mini_board_states[1]));
    CHECK_EQ(board.checkWinner(), 1);
    board.mini_board_states[0] = (1 << 0) | (1 << 2) | (1 << 3) | (1 << 7);
    board.mini_board_states[1] = (1 << 1) | (1 << 4) | (1 << 5) | (1 << 6);
    board.mini_board_states[2] = (1 << 8);
    CHECK(!scalar_has_win(board.mini_board_states[0]));
    CHECK(!scalar_has_win(board.mini_board_states[1]));
    CHECK_EQ(board.checkWinner(), 2);
}

static void test_fill_vs_vector(TestCtx &ctx) {
    std::mt19937 rng(12345);
    Move buf[81];
    Move cbuf[81];
    for (int game = 0; game < 400; game++) {
        GlobalBoard board;
        for (int ply = 0; ply < 90; ply++) {
            if (board.checkWinner() != -1) break;
            std::vector<Move> v = board.getLegalMoves();
            int n = board.fillLegalMoves(buf);
            CHECK(same_move_list(v, buf_to_vec(buf, n)));
            std::vector<Move> c = board.get_captures();
            int cn = board.fillCaptures(cbuf);
            CHECK(same_move_list(c, buf_to_vec(cbuf, cn)));
            if (v.empty()) break;
            board.makeMove(v[rng() % v.size()]);
        }
    }
}

static void test_oracle_agrees_random_games(TestCtx &ctx) {
    std::mt19937 rng(20260813);
    for (int game = 0; game < 500; game++) {
        GlobalBoard board;
        Oracle oracle;
        for (int ply = 0; ply < 90; ply++) {
            int w = board.checkWinner();
            CHECK_EQ(w, oracle.winner());
            if (w != -1) break;
            std::vector<Move> engine = board.getLegalMoves();
            std::vector<Move> naive = oracle.legal();
            CHECK(same_move_list(engine, naive));
            std::vector<Move> caps = board.get_captures();
            CHECK(same_move_list(caps, oracle.captures()));
            for (const Move &c : caps) {
                CHECK(contains_move(engine, c));
            }
            CHECK_EQ(rebuild_zobrist(board), board.zobrist_hash);
            if (engine.empty()) break;
            Move m = engine[rng() % engine.size()];
            int stm = board.n_moves % 2;
            bool expect_capture = contains_move(caps, m);
            Snap before = take_snap(board);
            board.makeMove(m);
            oracle.make(m);
            if (expect_capture) {
                CHECK((board.mini_board_states[stm] & (1 << m.mini_board)) != 0);
            }
            board.unmakeMove();
            CHECK(snap_eq(take_snap(board), before));
            board.makeMove(m);
        }
    }
}

static void test_make_unmake_restores(TestCtx &ctx) {
    std::mt19937 rng(7);
    for (int game = 0; game < 200; game++) {
        GlobalBoard board;
        std::vector<Snap> stack;
        stack.push_back(take_snap(board));
        for (int ply = 0; ply < 40; ply++) {
            if (board.checkWinner() != -1) break;
            std::vector<Move> legal = board.getLegalMoves();
            if (legal.empty()) break;
            board.makeMove(legal[rng() % legal.size()]);
            stack.push_back(take_snap(board));
        }
        while (board.n_moves > 0) {
            board.unmakeMove();
            stack.pop_back();
            CHECK(snap_eq(take_snap(board), stack.back()));
        }
        CHECK_EQ(board.zobrist_hash, 0ull);
        CHECK_EQ(board.n_moves, 0);
    }
}

static void test_copy_and_two_boards_same_hash(TestCtx &ctx) {
    std::mt19937 rng(99);
    GlobalBoard a;
    std::vector<Move> hist = random_opening(rng, 25, &a);
    GlobalBoard b;
    apply_moves(b, hist);
    CHECK_EQ(a.zobrist_hash, b.zobrist_hash);
    CHECK_EQ(a.n_moves, b.n_moves);
    CHECK_EQ(a.checkWinner(), b.checkWinner());
    GlobalBoard c = a;
    CHECK_EQ(c.zobrist_hash, a.zobrist_hash);
    if (a.checkWinner() == -1) {
        std::vector<Move> legal = a.getLegalMoves();
        if (!legal.empty()) {
            a.makeMove(legal[0]);
            c.makeMove(legal[0]);
            CHECK_EQ(a.zobrist_hash, c.zobrist_hash);
        }
    }
}

static void test_pass_unpass_hash(TestCtx &ctx) {
    GlobalBoard board;
    board.makeMove({4, 4});
    Snap before = take_snap(board);
    uint64_t h = board.zobrist_hash;
    board.pass();
    CHECK_EQ(board.n_moves, 2);
    CHECK(board.prev_move_was_pass);
    CHECK(board.zobrist_hash != h);
    std::vector<Move> legal = board.getLegalMoves();
    CHECK(legal.size() > 8);
    board.unpass();
    CHECK(snap_eq(take_snap(board), before));
}

static void test_perft_startpos(TestCtx &ctx) {
    for (int d = 1; d <= 5; d++) {
        GlobalBoard board;
        uint64_t n = perft(board, d);
        CHECK_EQ(n, STARTPOS_PERFT[d]);
        CHECK_EQ(board.n_moves, 0);
        CHECK_EQ(board.zobrist_hash, 0ull);
    }
    Oracle o;
    CHECK_EQ(oracle_perft(o, 4), STARTPOS_PERFT[4]);
}

static void test_perft_after_first_moves(TestCtx &ctx) {
    {
        GlobalBoard board;
        board.makeMove({4, 4});
        CHECK_EQ(perft(board, 1), 8ull);
        CHECK_EQ(perft(board, 2), 72ull);
    }
    {
        GlobalBoard board;
        board.makeMove({0, 1});
        CHECK_EQ(perft(board, 1), 9ull);
    }
    uint64_t sum = 0;
    for (int mb = 0; mb < 9; mb++) {
        for (int sq = 0; sq < 9; sq++) {
            GlobalBoard board;
            board.makeMove({mb, sq});
            sum += perft(board, 1);
        }
    }
    CHECK_EQ(sum, STARTPOS_PERFT[2]);
}

static void test_mini_index_and_lut(TestCtx &ctx) {
    CrossfishDev::init_mini_lut();
    int seen = 0;
    std::vector<uint8_t> used(CrossfishDev::MINI_LUT_SIZE, 0);
    for (int p0 = 0; p0 < 512; p0++) {
        for (int p1 = 0; p1 < 512; p1++) {
            if (p0 & p1) continue;
            int idx = CrossfishDev::mini_index(p0, p1);
            CHECK(idx >= 0 && idx < CrossfishDev::MINI_LUT_SIZE);
            if (!used[idx]) {
                used[idx] = 1;
                seen++;
            }
            int decoded0 = 0, decoded1 = 0;
            int t = idx;
            for (int s = 0; s < 9; s++) {
                int cell = t % 3;
                t /= 3;
                if (cell == 1) decoded0 |= (1 << s);
                else if (cell == 2) decoded1 |= (1 << s);
            }
            CHECK_EQ(decoded0, p0);
            CHECK_EQ(decoded1, p1);
        }
    }
    CHECK_EQ(seen, CrossfishDev::MINI_LUT_SIZE);

    const CrossfishDev::MiniLut &empty = CrossfishDev::mini_lut[0];
    CHECK(!empty.dead && !empty.p0_tiar && !empty.p1_tiar && !empty.p0_win1 && !empty.p0_sq);

    const CrossfishDev::MiniLut &row = CrossfishDev::mini_lut[4];
    CHECK(!row.dead && row.p0_win1 != 0 && row.p0_tiar >= 1);

    const CrossfishDev::MiniLut &blocked = CrossfishDev::mini_lut[22];
    CHECK(!blocked.p0_win1 && !blocked.p0_tiar);
}

static void test_eval_consistency(TestCtx &ctx) {
    CrossfishDev dev;
    CrossfishDev::init_mini_lut();
    std::mt19937 rng(999);
    GlobalBoard empty;
    CHECK_EQ(dev.evaluate_hce(empty), empty.n_moves % 2 == 0 ? dev.eval_weights[9] : -dev.eval_weights[9]);

    for (int g = 0; g < 300; g++) {
        GlobalBoard board;
        for (int ply = 0; ply < 90; ply++) {
            if (board.checkWinner() != -1) break;
            int d[CrossfishDev::N_EVAL_WEIGHTS];
            dev.eval_diffs(board, d);
            int stm = (board.n_moves % 2 == 0) ? 1 : -1;
            int val = 0;
            for (int i = 0; i < 9; i++) {
                val += dev.eval_weights[i] * d[i];
            }
            val += stm * dev.eval_weights[9];
            int extra = dev.eval_extra(board);
            int ev = dev.evaluate_hce(board);
            CHECK_EQ(ev, stm * val + extra);
            int16_t idx[9];
            int n = 0;
            int base = 0;
            dev.eval_parts(board, idx, n, base);
            int local = 0;
            for (int i = 0; i < n; i++) {
                local += CrossfishDev::mini_score[idx[i]];
            }
            CHECK_EQ(ev, base + stm * local + extra);
            std::vector<Move> moves = board.getLegalMoves();
            if (moves.empty()) break;
            board.makeMove(moves[rng() % moves.size()]);
        }
    }
}

static void test_eval_free_move_bonus(TestCtx &ctx) {
    CrossfishDev dev;
    GlobalBoard mid;
    mid.makeMove({4, 4});
    CHECK_EQ(dev.eval_extra(mid), 0);

    std::mt19937 rng(21);
    int seen = 0;
    for (int game = 0; game < 500 && seen < 5; game++) {
        GlobalBoard board;
        for (int ply = 0; ply < 81; ply++) {
            if (board.checkWinner() != -1) break;
            std::vector<Move> legal = board.getLegalMoves();
            if (legal.empty()) break;
            int oop = board.mini_board_states[0] | board.mini_board_states[1] | board.mini_board_states[2];
            bool free = board.n_moves > 0 && ((oop & (1 << board.move_history.top().square)) != 0);
            if (free) {
                seen++;
                CHECK_EQ(dev.eval_extra(board), CrossfishDev::W_FREE_MOVE);
                break;
            }
            board.makeMove(legal[rng() % legal.size()]);
        }
    }
    CHECK(seen >= 1);
}

static void test_search_returns_legal(TestCtx &ctx) {
    std::mt19937 rng(123);
    CrossfishDev bot;
    for (int i = 0; i < 12; i++) {
        GlobalBoard board;
        random_opening(rng, 8 + (int)(rng() % 12), &board);
        if (board.checkWinner() != -1) continue;
        std::vector<Move> legal = board.getLegalMoves();
        if (legal.empty()) continue;
        Move m = bot.getMove(board, std::chrono::milliseconds(8));
        CHECK(contains_move(legal, m));
        board.makeMove(m);
        CHECK_EQ(board.n_moves > 0, true);
    }
}

static void test_search_takes_instant_win(TestCtx &ctx) {
    std::mt19937 rng(4242);
    int found = 0;
    for (int game = 0; game < 300 && found < 6; game++) {
        GlobalBoard board;
        while (board.checkWinner() == -1) {
            std::vector<Move> wins = instant_wins(board);
            if (!wins.empty()) {
                found++;
                CrossfishDev bot;
                Move m = bot.getMove(board, std::chrono::milliseconds(30));
                CHECK(contains_move(wins, m));
                int stm = board.n_moves % 2;
                board.makeMove(m);
                CHECK_EQ(board.checkWinner(), stm);
                break;
            }
            std::vector<Move> legal = board.getLegalMoves();
            if (legal.empty()) break;
            std::vector<Move> caps = board.get_captures();
            if (!caps.empty() && (rng() % 3) == 0) {
                board.makeMove(caps[rng() % caps.size()]);
            } else {
                board.makeMove(legal[rng() % legal.size()]);
            }
        }
    }
    CHECK(found >= 1);
}

static void test_ttentry_layout_matches_store(TestCtx &ctx) {
    TTEntry e = {7, 1234, 2, 0xabcull, Move{3, 5}};
    CHECK_EQ(e.depth, 7);
    CHECK_EQ(e.score, 1234);
    CHECK_EQ(e.flag, 2);
    CHECK_EQ(e.zobrist_hash, 0xabcull);
    CHECK_EQ(e.best_move.mini_board, 3);
    CHECK_EQ(e.best_move.square, 5);

    CHECK_EQ(sizeof(CrossfishDev::CompactTTEntry), 16u);
    CHECK_EQ(sizeof(CrossfishDev::CompactTTBucket), 32u);
    CrossfishDev::CompactTTEntry compact = {
        0x123456789abcdef0ull, -12345, 77, TT_LOWER, 80
    };
    CHECK_EQ(compact.zobrist_hash, 0x123456789abcdef0ull);
    CHECK_EQ(compact.score, -12345);
    CHECK_EQ(compact.depth, 77);
    CHECK_EQ(compact.flag, TT_LOWER);
    CHECK_EQ(compact.best_move, 80);
    for (int score : {-99999, -90001, -123, 0, 123, 90001, 99999}) {
        int stored = CrossfishDev::tt_score_to_store(score, 17);
        CHECK_EQ(CrossfishDev::tt_score_from_store(stored, 17), score);
#ifdef CROSSFISH_NORMALIZE_TT_MATES
        if (score > 90000) CHECK_EQ(stored, score + 17);
        else if (score < -90000) CHECK_EQ(stored, score - 17);
        else CHECK_EQ(stored, score);
#else
        CHECK_EQ(stored, score);
#endif
    }
    for (int mb = 0; mb < 9; mb++) {
        for (int sq = 0; sq < 9; sq++) {
            Move move{mb, sq};
            Move roundtrip =
                CrossfishDev::unpack_tt_move(CrossfishDev::pack_tt_move(move));
            CHECK_EQ(roundtrip.mini_board, mb);
            CHECK_EQ(roundtrip.square, sq);
        }
    }
    Move invalid = CrossfishDev::unpack_tt_move(
        CrossfishDev::pack_tt_move(Move{99, 99}));
    CHECK_EQ(invalid.mini_board, 99);
    CHECK_EQ(invalid.square, 99);
}

static void test_mini_avx_matches_scalar(TestCtx &ctx) {
    mini_load_packed();
    std::mt19937 rng(20260816);
    int max_abs = 0;
    int disagree = 0;
    for (int g = 0; g < 40; g++) {
        GlobalBoard board;
        for (int ply = 0; ply < 30; ply++) {
            int a = evaluate_mini(board);
            int b = evaluate_mini_avx(board);
            int d = a - b;
            if (d < 0) d = -d;
            if (d > 1) disagree++;
            int abs_a = a < 0 ? -a : a;
            if (abs_a > max_abs) max_abs = abs_a;
            std::vector<Move> legal = board.getLegalMoves();
            if (legal.empty() || board.checkWinner() != -1) break;
            board.makeMove(legal[rng() % legal.size()]);
        }
    }
    CHECK_EQ(disagree, 0);
    CHECK(max_abs < 20000);
}

static void test_mini_fast_matches_scalar(TestCtx &ctx) {
    mini_load_packed();
    std::mt19937 rng(20260905);
    int max_diff = 0;
    for (int g = 0; g < 80; g++) {
        GlobalBoard board;
        for (int ply = 0; ply < 50; ply++) {
            int a = evaluate_mini(board);
            int b = evaluate_mini_fast(board);
            int d = std::abs(a - b);
            if (d > max_diff) max_diff = d;
            std::vector<Move> legal = board.getLegalMoves();
            if (legal.empty() || board.checkWinner() != -1) break;
            board.makeMove(legal[rng() % legal.size()]);
        }
    }
    CHECK(max_diff <= 2);
}

// The macro net evaluated directly from its embeddings: the reference for the
// exact lookup table the bot uses (macro_eval.hpp keeps only the table path).
template <typename Board>
static int evaluate_macro_fast(const Board &board) {
    if (!MACRO_READY && !macro_load_packed()) return 0;
    const int stm = board.n_moves & 1;
    const int constraint = d16_mini_board_constraint(board);
    __m256 h0 = _mm256_add_ps(
        _mm256_load_ps(MACRO_BASE),
        _mm256_load_ps(MACRO_CONSTR[constraint]));
    __m256 h1 = _mm256_add_ps(
        _mm256_load_ps(MACRO_BASE + 8),
        _mm256_load_ps(MACRO_CONSTR[constraint] + 8));
    for (int mb = 0; mb < 9; mb++) {
        const int bit = 1 << mb;
        int cls = 0;
        if (board.mini_board_states[stm] & bit) cls = 1;
        else if (board.mini_board_states[stm ^ 1] & bit) cls = 2;
        else if (board.mini_board_states[2] & bit) cls = 3;
        h0 = _mm256_add_ps(
            h0, _mm256_load_ps(MACRO_EMB[mb][cls]));
        h1 = _mm256_add_ps(
            h1, _mm256_load_ps(MACRO_EMB[mb][cls] + 8));
    }
    return macro_finish_hidden(h0, h1);
}

static void test_d16_fast_matches_scalar(TestCtx &ctx) {
    d16_mini_load_packed();
    macro_load_packed();
    std::mt19937 rng(20260913);
    int max_diff = 0;
    int max_macro = 0;
    for (int g = 0; g < 80; g++) {
        GlobalBoard board;
        for (int ply = 0; ply < 50; ply++) {
            int scalar = d16_evaluate_mini(board);
            int fast = d16_evaluate_mini_fast(board);
            max_diff = std::max(max_diff, std::abs(scalar - fast));
            int macro = evaluate_macro_fast(board);
            int stm = board.n_moves & 1;
            int macro_key = 0;
            for (int mb = 0; mb < 9; mb++) {
                int bit = 1 << mb;
                int cls = 0;
                if (board.mini_board_states[stm] & bit) cls = 1;
                else if (board.mini_board_states[stm ^ 1] & bit) cls = 2;
                else if (board.mini_board_states[2] & bit) cls = 3;
                macro_key |= cls << (2 * mb);
            }
            CHECK_EQ(
                macro,
                evaluate_macro_key(
                    d16_mini_board_constraint(board), macro_key));
            max_macro = std::max(max_macro, std::abs(macro));
            std::vector<Move> legal = board.getLegalMoves();
            if (legal.empty() || board.checkWinner() != -1) break;
            board.makeMove(legal[rng() % legal.size()]);
        }
    }
    CHECK(max_diff <= 8);
    CHECK(max_macro <= MACRO_CLIP);
}

static uint64_t play_book_child_hash(GlobalBoard b, Move m) {
    b.makeMove(m);
    int t;
    return pb_canonical(b, t);
}

static void test_play_book(TestCtx &ctx) {
    crossfish_nnue_load_once();
    // The payload is coded with this net's move ordering: its fingerprint must
    // match the net, and a different evaluator must be refused before the walk.
    auto nnue = [](const PbView &v, int c) { return b64::evaluate_board(v, c); };
    auto other = [](const PbView &v, int c) { return b64::evaluate_board(v, c) + (v.n_moves & 1 ? 1 : -1); };
    CHECK_EQ(pb_eval_fingerprint(nnue), PLAY_BOOK_EVAL_FINGERPRINT);
    CHECK(pb_eval_fingerprint(other) != PLAY_BOOK_EVAL_FINGERPRINT);
    CHECK((pb_init<GlobalBoard, Move>(nnue)));
    CHECK_EQ((int)PB_TABLE.size(), PLAY_BOOK_ENTRIES);
    // Pinned like the network payload hashes: regenerating the book changes it.
    std::vector<std::pair<uint64_t, uint8_t>> rows(PB_TABLE.begin(), PB_TABLE.end());
    std::sort(rows.begin(), rows.end());
    uint64_t h = 1469598103934665603ull;
    for (auto &r : rows) {
        for (int i = 0; i < 8; i++) { h ^= (r.first >> (8 * i)) & 0xff; h *= 1099511628211ull; }
        h ^= r.second;
        h *= 1099511628211ull;
    }
    CHECK_EQ(h, 17441813851168678777ull);

    // Both roots assume the first player's center-center; moving second, the
    // book always has our reply to it.
    {
        GlobalBoard root;
        root.makeMove(Move{4, 4});
        Move bm;
        CHECK(pb_lookup(root, bm));
    }
    // The book covers only some replies, so random games leave it early; every
    // move it does supply is legal and agrees across all 8 orientations.
    std::mt19937 rng(20260923);
    Move buf[81];
    for (int g = 0; g < 400; g++) {
        bool book_first = g % 2 == 0;
        GlobalBoard b;
        std::vector<int> hist;
        if (book_first) { b.makeMove(Move{4, 4}); hist.push_back(40); }
        int used = 0;
        while (b.checkWinner() == -1 && b.n_moves < 16) {
            bool mine = (b.n_moves % 2) == (book_first ? 0 : 1);
            int n = b.fillLegalMoves(buf);
            Move m = buf[rng() % n];
            if (mine) {
                Move bm;
                if (!pb_lookup(b, bm)) break;
                bool legal = false;
                for (int i = 0; i < n; i++) legal |= buf[i].mini_board == bm.mini_board && buf[i].square == bm.square;
                CHECK(legal);
                uint64_t want = play_book_child_hash(b, bm);
                for (int t = 1; t < 8; t++) {
                    GlobalBoard tb;
                    for (int packed : hist) {
                        int q = pb_map_move(t, packed, false);
                        tb.makeMove(Move{q / 9, q % 9});
                    }
                    Move tm;
                    CHECK(pb_lookup(tb, tm));
                    CHECK_EQ(play_book_child_hash(tb, tm), want);
                }
                m = bm;
                used++;
            }
            b.makeMove(m);
            hist.push_back(m.mini_board * 9 + m.square);
        }
        CHECK(used <= 40);
    }
}

static void test_cjk14_decoder(TestCtx &ctx) {
    unsigned char decoded[16]{};
    // Bytes 0..9 from tools/nnue_cjk14.py encode_u15, with a wrap newline to skip.
    int count = d16_mini_cjk_decode(
        "\u3400\u7480\u9480\n\u8460\u6c40\u5800", decoded, (int)sizeof(decoded));
    CHECK_EQ(count, 11);  // 90 bits: the ten bytes plus one byte of zero padding
    for (int i = 0; i < 10; i++) {
        CHECK_EQ((int)decoded[i], i);
    }
    CHECK_EQ((int)decoded[10], 0);
    // Top of the alphabet, in the private-use range: seven 0xFF bytes.
    count = d16_mini_cjk_decode(
        "\uf3ff\uf3ff\uf3ff\uf3f0", decoded, (int)sizeof(decoded));
    CHECK_EQ(count, 7);
    for (int i = 0; i < count; i++) {
        CHECK_EQ((int)decoded[i], 255);
    }
    CHECK_EQ(d16_mini_cjk_decode("\uf3ff\uf3ff", decoded, 2), -1);

    auto fnv1a = [](const unsigned char *data, int size) {
        uint64_t hash = 0xcbf29ce484222325ULL;
        for (int i = 0; i < size; i++) {
            hash ^= data[i];
            hash *= 0x100000001b3ULL;
        }
        return hash;
    };
    std::vector<unsigned char> payload(65536);
    count = d16_mini_cjk_decode(
        D16_MINI_PACK_CJK, payload.data(), (int)payload.size());
    CHECK_EQ(count, 42855);
    CHECK_EQ(fnv1a(payload.data(), count), 0xe35e987c17a453cfULL);
    count = d16_mini_cjk_decode(
        MACRO_PACK_CJK, payload.data(), (int)payload.size());
    CHECK_EQ(count, 3076);
    CHECK_EQ(fnv1a(payload.data(), count), 0x626e29f3a8d65679ULL);
}

static void test_lut_capture_block_tiar(TestCtx &ctx) {
    CrossfishDev::init_mini_lut();
    CrossfishDev dev;
    std::mt19937 rng(7);
    for (int g = 0; g < 50; g++) {
        GlobalBoard board;
        for (int ply = 0; ply < 40; ply++) {
            if (board.checkWinner() != -1) break;
            dev.init_hce_acc(board);
            CHECK_EQ(dev.evaluate_hce_incremental(board), dev.evaluate_hce(board));
            Move legal[81];
            int n = board.fillLegalMoves(legal);
            Move fast_legal[81];
            int fast_n = dev.fill_legal_moves_fast(board, fast_legal);
            CHECK_EQ(fast_n, n);
            for (int i = 0; i < n; i++) {
                CHECK(same_move(fast_legal[i], legal[i]));
            }
            CHECK_EQ(dev.check_winner_fast(board), board.checkWinner());
            if (n == 0) break;
            for (int i = 0; i < n; i++) {
                CHECK_EQ(dev.is_capture_avx(board, legal[i]), board.is_capture_avx(legal[i]));
                int opp = (board.n_moves + 1) % 2;
                int opp_m = board.mini_boards[legal[i].mini_board].markers[opp] | (1 << legal[i].square);
                CHECK_EQ(dev.is_block_avx(board, legal[i]), scalar_has_win(opp_m));
                int ours = board.mini_boards[legal[i].mini_board].markers[board.n_moves % 2] | (1 << legal[i].square);
                int occ = board.mini_boards[legal[i].mini_board].markers[0]
                        | board.mini_boards[legal[i].mini_board].markers[1];
                bool tiar = false;
                for (int k = 0; k < CrossfishDev::N_TIAR_MASKS / 2; k++) {
                    int pair = CrossfishDev::two_in_a_row_masks[k * 2];
                    int third = CrossfishDev::two_in_a_row_masks[k * 2 + 1];
                    if (((ours & pair) == pair) && ((occ & third) == 0)) {
                        tiar = true;
                        break;
                    }
                }
                CHECK_EQ(dev.creates_two_in_a_row(board, legal[i]), tiar);
            }
            Move caps_a[81];
            Move caps_b[81];
            int na = board.fillCaptures(caps_a);
            int nb = dev.fill_captures_lut(board, caps_b);
            CHECK_EQ(na, nb);
            for (int i = 0; i < na; i++) {
                CHECK(same_move(caps_a[i], caps_b[i]));
            }
            Move chosen = legal[rng() % n];
            GlobalBoard fast_board = board;
            GlobalBoard slow_board = board;
            dev.make_move_fast(fast_board, chosen);
            slow_board.makeMove(chosen);
            CHECK(snap_eq(take_snap(fast_board), take_snap(slow_board)));
            CHECK_EQ(dev.check_winner_fast(fast_board), slow_board.checkWinner());
            CHECK_EQ(dev.evaluate_hce_incremental(fast_board),
                     dev.evaluate_hce(fast_board));
            dev.unmake_move_fast(fast_board);
            CHECK(snap_eq(take_snap(fast_board), take_snap(board)));
            CHECK_EQ(dev.evaluate_hce_incremental(fast_board),
                     dev.evaluate_hce(fast_board));
            board.makeMove(chosen);
        }
    }
}

// ---------------------------------------------------------------- NNUE (nnue_b64.hpp)

// The table hash of the verified CodinGame builds (FNV-1a with offset basis 1469598103934665603);
// tools/nnue_emit_b64_header.py --check prints the same hashes.
static uint64_t nnue_table_hash(const void *p, size_t n) {
    uint64_t h = 1469598103934665603ull;
    const unsigned char *c = (const unsigned char *)p;
    for (size_t i = 0; i < n; i++) {
        h ^= c[i];
        h *= 1099511628211ull;
    }
    return h;
}

// load() must bake exactly the tables of the verified build of the committed net (r13w_11, the build of
// improvement log section 64's ship match; the hashes are nnue_emit_b64_header.py --check's): same payload,
// same float bake, same quantization.
static void test_nnue_tables_match_verified_build(TestCtx &ctx) {
    b64::load();
    CHECK_EQ(B64_QA, 9);
    CHECK_EQ(B64_QPS, 12);
    CHECK_EQ(B64_QB, 13);
    CHECK_EQ(B64_Q2, 13);
    CHECK_EQ(B64_QO, 10);
    const struct {
        const char *name;
        const void *p;
        size_t n;
        uint64_t want;
    } tables[] = {
        {"T", b64::T, sizeof(b64::T), 0x8467a8b43b1781abull},
        {"TP", b64::TP, sizeof(b64::TP), 0xd115d9d2344a015cull},
        {"F", b64::F, sizeof(b64::F), 0xcdae0bbc33217c1bull},
        {"FP", b64::FP, sizeof(b64::FP), 0x0827752cb06824cbull},
        {"DEC", b64::DEC, sizeof(b64::DEC), 0x10b25653b45b4759ull},
        {"DECP", b64::DECP, sizeof(b64::DECP), 0x25e158f8276b3d58ull},
        {"CON", b64::CON, sizeof(b64::CON), 0x3538d69f7a3a74e7ull},
        {"CONP", b64::CONP, sizeof(b64::CONP), 0x71329dd7cab3cabdull},
        {"BIAS", b64::BIAS, sizeof(b64::BIAS), 0x27b847b7560d6d92ull},
        {"BIASP", &b64::BIASP, sizeof(b64::BIASP), 0x9a691300c548b8fbull},
        {"W1p", b64::W1p, sizeof(b64::W1p), 0x4ffb93373ab96169ull},
        {"B1", b64::B1, sizeof(b64::B1), 0x9650ef9b37e383d3ull},
        {"W2p", b64::W2p, sizeof(b64::W2p), 0x7155982c2fbf7f0full},
        {"B2", b64::B2, sizeof(b64::B2), 0x5435d739cc7b6f4bull},
        {"WO", b64::WO, sizeof(b64::WO), 0x084874555f3d0ae2ull},
        {"BO", &b64::BO, sizeof(b64::BO), 0x64be4e773b169f15ull},
    };
    for (const auto &t : tables) {
        const uint64_t got = nnue_table_hash(t.p, t.n);
        if (got != t.want) {
            std::cerr << "  FAIL " << ctx.name << ": table " << t.name << " hash " << std::hex << got << " != "
                      << t.want << std::dec << std::endl;
            ctx.fails++;
        }
    }
}

// The same integers as the AVX2 head, computed independently: int32 accumulators rebuilt from the tables,
// dense layers in int64 without the sparse pair loop, the final division instead of the sign-symmetric shift.
template <class Board>
static int nnue_scalar_eval(const Board &b, int c) {
    using namespace b64;
    int32_t acc[2][A], ps[2];
    const int oop = b.mini_board_states[0] | b.mini_board_states[1] | b.mini_board_states[2];
    for (int P = 0; P < 2; P++) {
        ps[P] = BIASP;
        for (int i = 0; i < A; i++) acc[P][i] = BIAS[i];
        for (int mb = 0; mb < 9; mb++) {
            const int16_t *row;
            if (oop >> mb & 1) {
                const int st = (b.mini_board_states[2] >> mb & 1) ? 2 : ((b.mini_board_states[0] >> mb & 1) ? 0 : 1);
                const int d = mb * 3 + (st == 2 ? 2 : (st == P ? 0 : 1));
                row = DEC[d];
                ps[P] += DECP[d];
            } else {
                int pat = 0;
                for (int sq = 8; sq >= 0; sq--) {
                    const int cell = (b.mini_boards[mb].markers[P] >> sq & 1) ? 1
                                   : (b.mini_boards[mb].markers[P ^ 1] >> sq & 1) ? 2 : 0;
                    pat = pat * 3 + cell;
                }
                row = T[mb][pat];
                ps[P] += TP[mb][pat];
            }
            for (int i = 0; i < A; i++) acc[P][i] += row[i];
        }
    }
    const int stm = b.n_moves & 1;
    int64_t act[2 * A];
    int32_t psd = ps[stm] - ps[stm ^ 1] + CONP[c] - CONP[10 + c];
    for (int v = 0; v < 2; v++) {
        const int P = v ? stm ^ 1 : stm;
        const int16_t *con = CON[v ? 10 + c : c];
        int fpat = -1;
        if (c < 9) {
            fpat = 0;
            for (int sq = 8; sq >= 0; sq--) {
                const int cell = (b.mini_boards[c].markers[P] >> sq & 1) ? 1
                               : (b.mini_boards[c].markers[P ^ 1] >> sq & 1) ? 2 : 0;
                fpat = fpat * 3 + cell;
            }
            psd += v ? -FP[fpat] : FP[fpat];
        }
        for (int i = 0; i < A; i++) {
            const int32_t x = acc[P][i] + con[i] + (fpat >= 0 ? F[fpat][i] : 0);
            act[v * A + i] = x < 0 ? 0 : (x > (1 << QA) ? (1 << QA) : x);
        }
    }
    auto lo = [](int32_t w) { return (int64_t)(int16_t)(w & 0xFFFF); };
    auto hi = [](int32_t w) { return (int64_t)(int16_t)((uint32_t)w >> 16); };
    int64_t h1[L1], out = BO;
    for (int k = 0; k < L1; k++) {
        int64_t s = B1[k];
        for (int j = 0; j < 2 * A; j++) s += act[j] * (j & 1 ? hi(W1p[j / 2][k]) : lo(W1p[j / 2][k]));
        s = s < 0 ? 0 : (s > H1MAX ? H1MAX : s);
        h1[k] = (s + H1_ROUND) >> H1_SHIFT;
    }
    for (int k = 0; k < L2; k++) {
        int64_t s = B2[k];
        for (int j = 0; j < L1; j++) s += h1[j] * (j & 1 ? hi(W2p[j / 2][k]) : lo(W2p[j / 2][k]));
        s = s < 0 ? 0 : (s > H2MAX ? H2MAX : s);
        out += ((s + H2_ROUND) >> H2_SHIFT) * WO[k];
    }
    const int64_t x = 1000 * (out * OUT_MUL + (int64_t)psd * PS_MUL);
    return (int)(x / (1LL << FIN_SHIFT));
}

// Fixed positions: the empty board, the centre opening, positions at fixed plies of seeded random games,
// then the first position of a seeded game with a property (a drawn miniboard; one after ply 40; a free
// move; ply 60 or more). None is a finished game.
static bool nnue_position_ok(const GlobalBoard &b, int kind) {
    const int oop = b.mini_board_states[0] | b.mini_board_states[1] | b.mini_board_states[2];
    const bool free_move = b.n_moves > 0 && (oop >> b.move_history.top().square & 1);
    switch (kind) {
    case 0: return b.mini_board_states[2] != 0;
    case 1: return b.mini_board_states[2] != 0 && b.n_moves >= 40;
    case 2: return free_move;
    default: return b.n_moves >= 60;
    }
}

static void nnue_position(int i, GlobalBoard &b) {
    b = GlobalBoard();
    if (i == 0) return;
    if (i == 1) {
        b.makeMove({4, 4});
        return;
    }
    if (i < 12) {
        static const int plies[] = {3, 7, 12, 18, 24, 30, 36, 42, 48, 54};
        std::mt19937 rng(20260927u + (unsigned)i);
        while (b.n_moves < plies[i - 2] && b.checkWinner() == -1) {
            std::vector<Move> legal = b.getLegalMoves();
            b.makeMove(legal[rng() % legal.size()]);
        }
        return;
    }
    for (unsigned seed = 1;; seed++) {
        std::mt19937 rng(seed * 7919u + (unsigned)i);
        b = GlobalBoard();
        while (b.checkWinner() == -1) {
            if (nnue_position_ok(b, i - 12)) return;
            std::vector<Move> legal = b.getLegalMoves();
            b.makeMove(legal[rng() % legal.size()]);
        }
    }
}

// The committed net's (r13w_11) evals on those positions, checked against the float net in PyTorch (mean
// |d| 8.9, max 31; r12_M2's were 4.9 / 28 with a 20,000-position parity of 5.8 / 166): a change to the net,
// the bake, the quantization or the kernels shows up here.
static void test_nnue_fixed_positions(TestCtx &ctx) {
    static const int want[16] = {1529, -1513, -346,  -92,   386,  832,  580,  1518,
                                 2290, -2430, 12107, 16910, 9430, 10518, 6017, 15700};
    CrossfishDev dev;
    int drawn = 0, free_moves = 0, decided = 0;
    for (int i = 0; i < 16; i++) {
        GlobalBoard b;
        nnue_position(i, b);
        CHECK_EQ(b.checkWinner(), -1);
        const int c = d16_mini_board_constraint(b);
        CHECK_EQ(b64::evaluate_board(b, c), want[i]);
        CHECK_EQ(nnue_scalar_eval(b, c), want[i]);
        CHECK_EQ(dev.evaluate(b), want[i]);
        drawn += b.mini_board_states[2] != 0;
        free_moves += c == 9 && b.n_moves > 0;
        decided += (b.mini_board_states[0] | b.mini_board_states[1]) != 0;
    }
    CHECK(drawn >= 2 && free_moves >= 3 && decided >= 6);
    // g_force_hce_eval still gives the HCE (test_bots' HCE tools).
    GlobalBoard b;
    nnue_position(8, b);
    g_force_hce_eval = true;
    CHECK_EQ(dev.evaluate(b), dev.evaluate_hce(b));
    g_force_hce_eval = false;
}

// The fields of a board b64::Stack reads, with a Zobrist key in the role of the engine's tt_hash. Like the
// engine's, the empty board's key is 0, which is why the stack marks empty entries with kNoKey.
struct NnueWalkBoard {
    std::array<MiniBoard, 9> mini_boards;
    std::array<int, 3> mini_board_states;
    int n_moves;
    uint64_t tt_hash;
    int active_board;
};

struct NnueWalkKeys {
    uint64_t stone[2][9][9], con[10], stm;
    NnueWalkKeys() {
        uint64_t x = 0x243F6A8885A308D3ull;
        auto next = [&x]() {  // splitmix64
            uint64_t z = (x += 0x9E3779B97F4A7C15ull);
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
            return z ^ (z >> 31);
        };
        for (auto &p : stone)
            for (auto &mb : p)
                for (uint64_t &k : mb) k = next();
        for (int c = 0; c < 9; c++) con[c] = next();
        con[9] = 0;
        stm = next();
    }
};

static NnueWalkBoard nnue_walk_board(const GlobalBoard &g, const NnueWalkKeys &keys) {
    NnueWalkBoard b;
    for (int mb = 0; mb < 9; mb++) b.mini_boards[mb] = g.mini_boards[mb];
    for (int s = 0; s < 3; s++) b.mini_board_states[s] = g.mini_board_states[s];
    b.n_moves = g.n_moves;
    b.active_board = d16_mini_board_constraint(g);
    b.tt_hash = keys.con[b.active_board] ^ ((g.n_moves & 1) ? keys.stm : 0);
    for (int p = 0; p < 2; p++)
        for (int mb = 0; mb < 9; mb++)
            for (int sq = 0; sq < 9; sq++)
                if (g.mini_boards[mb].markers[p] >> sq & 1) b.tt_hash ^= keys.stone[p][mb][sq];
    return b;
}

struct NnueWalk {
    TestCtx &ctx;
    const NnueWalkKeys &keys;
    b64::Stack &stack;
    std::mt19937 &rng;
    long long evals = 0;

    void check(const NnueWalkBoard &b) {
        const int c = b.active_board;
        const int keyed = stack.evaluate_keyed(b, c, b.tt_hash);
        const int direct = stack.evaluate(b, c);
        const int scratch = b64::evaluate_board(b, c);
        const int scalar = nnue_scalar_eval(b, c);
        if (keyed != scratch || direct != scratch || scalar != scratch) {
            std::cerr << "  FAIL " << ctx.name << ": n_moves " << b.n_moves << " constraint " << c << ": keyed "
                      << keyed << " incremental " << direct << " scratch " << scratch << " scalar " << scalar
                      << std::endl;
            ctx.fails++;
        }
        evals++;
    }

    // A search-like walk: make records each move on the stack (before n_moves++, as make_move_fast does),
    // siblings overwrite the deeper entries, nodes are evaluated before and after their children.
    void walk(GlobalBoard &g, int depth) {
        const NnueWalkBoard b = nnue_walk_board(g, keys);
        if (rng() % 10 < 7) check(b);
        if (depth == 0 || g.checkWinner() != -1) return;
        std::vector<Move> legal = g.getLegalMoves();
        const int stm = g.n_moves & 1;
        for (int k = 0; k < 3 && !legal.empty(); k++) {
            const size_t pick = rng() % legal.size();
            const Move m = legal[pick];
            legal.erase(legal.begin() + (long)pick);
            GlobalBoard child = g;
            child.makeMove(m);
            NnueWalkBoard hook = nnue_walk_board(child, keys);
            hook.n_moves = g.n_moves;
            int decided = -1;
            for (int s = 0; s < 3; s++)
                if ((child.mini_board_states[s] & ~g.mini_board_states[s]) >> m.mini_board & 1) decided = s;
            stack.on_make(hook, m.mini_board, m.square, stm, decided, g.mini_boards[m.mini_board].markers[stm],
                          b.tt_hash);
            walk(child, depth - 1);
        }
        if (rng() % 10 < 3) check(b);
    }
};

// Incremental (the lazy keyed stack, with its eval cache) == from scratch == the scalar reference over a
// search-like walk of about 20,000 evaluations from 300 random roots.
static void test_nnue_incremental_matches_scratch(TestCtx &ctx) {
    static NnueWalkKeys keys;
    std::unique_ptr<b64::Stack> stack(new b64::Stack());
    std::mt19937 rng(20260927);
    NnueWalk w{ctx, keys, *stack, rng};
    GlobalBoard empty;
    CHECK_EQ(nnue_walk_board(empty, keys).tt_hash, 0ull);
    // A new stack used at the empty board (key 0) and below it with no refresh_root: an entry holding no
    // position must not pass for the empty board (the kNoKey sentinel).
    w.check(nnue_walk_board(empty, keys));
    w.walk(empty, 2);
    int roots = 0;
    for (int r = 0; r < 300; r++) {
        GlobalBoard root;
        const int plies = r == 0 ? 0 : (int)(rng() % 64);
        while (root.n_moves < plies && root.checkWinner() == -1) {
            std::vector<Move> legal = root.getLegalMoves();
            root.makeMove(legal[rng() % legal.size()]);
        }
        if (root.checkWinner() != -1) continue;
        // Every fifth root is not refreshed: the stack must see from the keys that none of its entries holds
        // this position or an ancestor, and rebuild from scratch.
        if (r % 5 != 4) stack->refresh_root(nnue_walk_board(root, keys));
        w.walk(root, 4);
        roots++;
    }
    CHECK(roots >= 250);
    CHECK(w.evals >= 10000);
}

using TestFn = void (*)(TestCtx &);

int main() {
    const std::pair<const char *, TestFn> tests[] = {
        {"startpos_counts", test_startpos_counts},
        {"send_to_same_board", test_send_to_same_board},
        {"send_to_other_board", test_send_to_other_board},
        {"grid_coord_roundtrip", test_grid_coord_roundtrip},
        {"scalar_vs_avx_wins", test_scalar_vs_avx_wins},
        {"miniboard_win_and_draw", test_miniboard_win_and_draw},
        {"free_move_when_sent_to_finished", test_free_move_when_sent_to_finished},
        {"global_win_by_three_miniboards", test_global_win_by_three_miniboards},
        {"count_win_when_all_decided", test_count_win_when_all_decided},
        {"fill_vs_vector", test_fill_vs_vector},
        {"oracle_agrees_random_games", test_oracle_agrees_random_games},
        {"make_unmake_restores", test_make_unmake_restores},
        {"copy_and_two_boards_same_hash", test_copy_and_two_boards_same_hash},
        {"pass_unpass_hash", test_pass_unpass_hash},
        {"perft_startpos", test_perft_startpos},
        {"perft_after_first_moves", test_perft_after_first_moves},
        {"mini_index_and_lut", test_mini_index_and_lut},
        {"eval_consistency", test_eval_consistency},
        {"eval_free_move_bonus", test_eval_free_move_bonus},
        {"search_returns_legal", test_search_returns_legal},
        {"search_takes_instant_win", test_search_takes_instant_win},
        {"ttentry_layout_matches_store", test_ttentry_layout_matches_store},
        {"mini_avx_matches_scalar", test_mini_avx_matches_scalar},
        {"mini_fast_matches_scalar", test_mini_fast_matches_scalar},
        {"d16_fast_matches_scalar", test_d16_fast_matches_scalar},
        {"cjk14_decoder", test_cjk14_decoder},
        {"play_book", test_play_book},
        {"lut_capture_block_tiar", test_lut_capture_block_tiar},
        {"nnue_tables_match_verified_build", test_nnue_tables_match_verified_build},
        {"nnue_fixed_positions", test_nnue_fixed_positions},
        {"nnue_incremental_matches_scratch", test_nnue_incremental_matches_scratch},
    };

    int passed = 0;
    int failed = 0;
    for (const auto &t : tests) {
        TestCtx ctx{t.first};
        t.second(ctx);
        if (ctx.fails == 0) {
            std::cout << "PASS  " << t.first << std::endl;
            passed++;
        } else {
            std::cout << "FAIL  " << t.first << " (" << ctx.fails << " checks)" << std::endl;
            failed++;
        }
    }
    std::cout << passed << " passed, " << failed << " failed, "
              << (passed + failed) << " total" << std::endl;
    return failed ? 1 : 0;
}
