#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <mutex>
#include <random>
#include <stack>
#include <string>
#include <vector>

#include "global_board.hpp"
#include "crossfish_dev.hpp"

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

static void test_lut_capture_block_tiar(TestCtx &ctx) {
    CrossfishDev::init_mini_lut();
    CrossfishDev dev;
    std::mt19937 rng(7);
    for (int g = 0; g < 50; g++) {
        GlobalBoard board;
        for (int ply = 0; ply < 40; ply++) {
            if (board.checkWinner() != -1) break;
            Move legal[81];
            int n = board.fillLegalMoves(legal);
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
            board.makeMove(legal[rng() % n]);
        }
    }
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
        {"lut_capture_block_tiar", test_lut_capture_block_tiar},
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
