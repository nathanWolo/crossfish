#pragma GCC optimize("O3")
#pragma GCC optimization("unroll-loops")
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <immintrin.h>
#include <random>
#include <string>
#include <vector>
#pragma GCC option("arch=native", "tune=native", "no-zero-upper")
#pragma GCC target("avx2,bmi,bmi2,lzcnt,popcnt")
// Eval parity of the NNUE runtime (nnue_b64.hpp) with any payload (any encoder width): the integer evaluation
// of every record of an eval_data file (tools/eval_data.py REC, 128 bytes: the 93-character position string
// first), to compare line for line with tools/nnue_emit_b64_header.py --int-eval (the exporter's integer
// reference) and with the float net (datasets/nnue2/r16/eval/build_r16enc.py).
//
//   nnue_parity IN.cfdg OUT [scratch|incremental] [SEED]   (OUT: .txt one eval per line, else little-endian int32)
//   nnue_parity --info                                     (the net's widths, scales, table bytes, load() time)
//
// scratch (the default) evaluates each position from scratch (b64::evaluate_board). incremental replays it through
// a b64::Stack from the empty board: its stones board by board in a random interleaving (SEED, default 1), a won
// board's deciding stone last (one whose removal leaves no line), so every board is decided by the move that
// decides it in a game; it evaluates the stack at random plies on the way (about one in four, each checked against
// evaluate_board, so deciding moves land both at and before the evaluated ply) and at the end. A record whose stones
// cannot be replayed that way (or whose side to move does not match its stone count) gets the scratch eval and is
// counted as not replayed. The header line it prints gives the counts; a mismatch on the way is an error.
// Undecodable records give INT32_MIN in both modes.
//
// The pragmas are the bot's: the same file builds with clang -O3 (the bot's flags or the native toolchain's) and,
// minified (tools/cg_minify.py --inline-local), with CodinGame's g++ command line (no -O).
#include "nnue_b64.hpp"

struct ParityMiniBoard {
    int markers[2];
};
struct ParityBoard {
    ParityMiniBoard mini_boards[9];
    int mini_board_states[3];
    int n_moves;
    uint64_t tt_hash;
    int active_board;
};

#pragma pack(push, 1)
struct ParityRec {
    char s[93];
    uint8_t rest[35];
};
#pragma pack(pop)
static_assert(sizeof(ParityRec) == 128, "eval_data REC is 128 bytes");

// The position string: 81 cells (0 empty, 1 / 2 the absolute players), 9 board states (0 live, 1 / 2 won,
// 3 drawn), the side to move (1 / 2), the constraint (0..8 forced, 9 free), a trailing field.
static bool decode(const ParityRec &r, ParityBoard &b, int &constraint, int &stm) {
    std::memset(&b, 0, sizeof(b));
    for (int i = 0; i < 81; i++) {
        const int v = r.s[i] - '0';
        if (v < 0 || v > 2) return false;
        if (v) b.mini_boards[i / 9].markers[v - 1] |= 1 << (i % 9);
    }
    for (int m = 0; m < 9; m++) {
        const int v = r.s[81 + m] - '0';
        if (v < 0 || v > 3) return false;
        if (v) b.mini_board_states[v - 1] |= 1 << m;
    }
    stm = r.s[90] - '0';
    if (stm != 1 && stm != 2) return false;
    stm--;
    b.n_moves = stm;  // scratch reads only its parity
    b.active_board = 9;
    constraint = r.s[91] - '0';
    if (constraint < 0 || constraint > 9) return false;
    return true;
}

static const int LINES9[8] = {7, 56, 448, 73, 146, 292, 273, 84};
static bool has_line(int m) {
    for (int l : LINES9)
        if ((m & l) == l) return true;
    return false;
}

struct Replayer {
    b64::Stack *stack;
    uint64_t key[2][9][9], stm_key;
    std::mt19937_64 rng;
    long long evals = 0, mismatches = 0;

    explicit Replayer(uint64_t seed) : stack(new b64::Stack()), rng(seed) {
        std::mt19937_64 k(0x243F6A8885A308D3ull);
        for (auto &p : key)
            for (auto &m : p)
                for (uint64_t &x : m) x = k();
        stm_key = k();
    }

    // One board's stones in a replay order (player, square), the deciding stone last; false if impossible.
    bool board_order(const ParityBoard &t, int m, std::vector<std::pair<int, int>> &out) {
        const int a = t.mini_boards[m].markers[0], c = t.mini_boards[m].markers[1];
        const int bit = 1 << m;
        const int st = (t.mini_board_states[0] & bit) ? 1 : (t.mini_board_states[1] & bit) ? 2 : (t.mini_board_states[2] & bit) ? 3 : 0;
        std::vector<std::pair<int, int>> v;
        for (int sq = 0; sq < 9; sq++) {
            if (a >> sq & 1) v.push_back({0, sq});
            if (c >> sq & 1) v.push_back({1, sq});
        }
        for (size_t i = v.size(); i > 1; i--) std::swap(v[i - 1], v[rng() % i]);
        const int occ = a | c;
        if (st == 0) {
            if (has_line(a) || has_line(c) || occ == 511) return false;
        } else if (st == 3) {
            if (has_line(a) || has_line(c) || occ != 511) return false;
        } else {
            const int w = st - 1, mw = w ? c : a, mo = w ? a : c;
            if (!has_line(mw) || has_line(mo)) return false;
            size_t last = v.size();
            for (size_t i = 0; i < v.size(); i++)
                if (v[i].first == w && !has_line(mw & ~(1 << v[i].second))) {
                    last = i;
                    break;
                }
            if (last == v.size()) return false;
            std::swap(v[last], v.back());
        }
        out = v;
        return true;
    }

    // Replays target t; returns false (nothing evaluated) when it cannot.
    bool replay(const ParityBoard &t, int stm, int constraint, int &eval) {
        std::vector<std::pair<int, int>> q[9];
        int total = 0;
        for (int m = 0; m < 9; m++) {
            if (!board_order(t, m, q[m])) return false;
            total += (int)q[m].size();
        }
        if ((total & 1) != stm || total >= b64::MAXPLY) return false;
        ParityBoard b;
        std::memset(&b, 0, sizeof(b));
        b.active_board = 9;
        stack->refresh_root(b);
        size_t pos[9] = {};
        for (int ply = 0; ply < total; ply++) {
            int m;
            do m = (int)(rng() % 9);
            while (pos[m] == q[m].size());
            const auto [p, sq] = q[m][pos[m]++];
            const int before = b.mini_boards[m].markers[p];
            const uint64_t pkey = b.tt_hash;
            b.mini_boards[m].markers[p] |= 1 << sq;
            b.tt_hash ^= key[p][m][sq] ^ stm_key;
            int decided = -1;
            if (has_line(b.mini_boards[m].markers[p])) decided = p;
            else if ((b.mini_boards[m].markers[0] | b.mini_boards[m].markers[1]) == 511) decided = 2;
            if (decided >= 0) b.mini_board_states[decided] |= 1 << m;
            stack->on_make(b, m, sq, p, decided, before, pkey);
            b.n_moves++;
            if (ply + 1 < total && rng() % 4 == 0) {
                const int c = (int)(rng() % 10);
                const int inc = stack->evaluate(b, c), scr = b64::evaluate_board(b, c);
                evals++;
                if (inc != scr) {
                    mismatches++;
                    std::fprintf(stderr, "incremental %d != scratch %d at ply %d\n", inc, scr, b.n_moves);
                }
            }
        }
        for (int s = 0; s < 3; s++)
            if (b.mini_board_states[s] != t.mini_board_states[s]) return false;
        eval = stack->evaluate(b, constraint);
        const int scr = b64::evaluate_board(b, constraint);
        evals++;
        if (eval != scr) {
            mismatches++;
            std::fprintf(stderr, "incremental %d != scratch %d at the end (ply %d)\n", eval, scr, b.n_moves);
        }
        return true;
    }
};

int main(int argc, char **argv) {
    using clk = std::chrono::steady_clock;
    const auto t0 = clk::now();
    b64::load();
    const double load_ms = std::chrono::duration<double, std::milli>(clk::now() - t0).count();
    if (argc == 2 && std::strcmp(argv[1], "--info") == 0) {
        std::printf("nnue_parity A=%d enc=%d,%d,%d scales=%d,%d,%d,%d,%d tables_mb=%.1f load_ms=%.1f\n", b64::A,
                    b64::ENC0, b64::ENC1, b64::E, b64::QA, b64::QPS, b64::QB, b64::Q2, b64::QO,
                    (sizeof(b64::T) + sizeof(b64::TP) + sizeof(b64::F) + sizeof(b64::FP)) / 1048576.0, load_ms);
        return 0;
    }
    if (argc < 3 || argc > 5) {
        std::fprintf(stderr, "usage: nnue_parity IN.cfdg OUT [scratch|incremental] [SEED] | --info\n");
        return 2;
    }
    const bool incremental = argc >= 4 && std::strcmp(argv[3], "incremental") == 0;
    if (argc >= 4 && !incremental && std::strcmp(argv[3], "scratch") != 0) {
        std::fprintf(stderr, "mode must be scratch or incremental\n");
        return 2;
    }
    std::FILE *in = std::fopen(argv[1], "rb");
    if (!in) {
        std::fprintf(stderr, "cannot read %s\n", argv[1]);
        return 1;
    }
    Replayer rep(argc == 5 ? std::stoull(argv[4]) : 1);
    std::vector<int32_t> out;
    ParityRec r;
    long long bad = 0, replayed = 0, not_replayed = 0;
    while (std::fread(&r, sizeof(r), 1, in) == 1) {
        ParityBoard b;
        int c, stm;
        if (!decode(r, b, c, stm)) {
            bad++;
            out.push_back(-2147483647 - 1);  // INT32_MIN (a literal: the minifier keeps no macro names)
            continue;
        }
        int e;
        if (incremental && rep.replay(b, stm, c, e)) {
            replayed++;
        } else {
            not_replayed += incremental;
            e = b64::evaluate_board(b, c);
        }
        out.push_back(e);
    }
    std::fclose(in);
    const size_t olen = std::strlen(argv[2]);
    const bool text = olen >= 4 && std::strcmp(argv[2] + olen - 4, ".txt") == 0;
    std::FILE *f = std::fopen(argv[2], "wb");
    if (!f) {
        std::fprintf(stderr, "cannot write %s\n", argv[2]);
        return 1;
    }
    if (text)
        for (int32_t v : out) std::fprintf(f, "%d\n", v);
    else
        std::fwrite(out.data(), sizeof(int32_t), out.size(), f);
    std::fclose(f);
    std::printf("nnue_parity mode=%s records=%zu undecodable=%lld load_ms=%.1f", incremental ? "incremental" : "scratch",
                out.size(), bad, load_ms);
    if (incremental)
        std::printf(" replayed=%lld not_replayed=%lld stack_evals=%lld mismatches=%lld", replayed, not_replayed, rep.evals,
                    rep.mismatches);
    std::printf("\n");
    return rep.mismatches ? 3 : 0;
}
