#pragma once
// EXPERIMENT (round ten): Stockfish-style full-evaluation NNUE, float
// reference inference. Features per perspective P:
//   cells of live miniboards: (mb*9+sq)*2 + (owner != P)      0..161
//   decided miniboards:       162 + mb*3 + {P won, other, draw} 162..188
//   constraint:               189 + (0..8 forced, 9 free)       189..198
// acc = sum W0[f] + b0 (A); h = [crelu(acc_stm), crelu(acc_ntm)]
// -> L1 (32) -> crelu -> L2 (1) * 1000, stm-relative eval units.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

struct FullNnueFloat {
    static constexpr int NF = 200, A = 256, L1 = 32;
    static inline float W0[NF][A], B0[A], W1[L1][2 * A], B1[L1], W2[L1], B2;
    static inline bool ready = false;
    static inline std::once_flag once;

    static void load() {
        std::call_once(once, [] {
            const char *path = std::getenv("FULLNNUE_PATH");
            FILE *f = path ? std::fopen(path, "rb") : nullptr;
            if (!f) { std::fprintf(stderr, "FULLNNUE_PATH not set or unreadable\n"); std::exit(1); }
            char magic[4]; int a, l1;
            bool ok = std::fread(magic, 1, 4, f) == 4 && std::memcmp(magic, "FNN1", 4) == 0
                && std::fread(&a, 4, 1, f) == 1 && std::fread(&l1, 4, 1, f) == 1 && a == A && l1 == L1;
            ok = ok && std::fread(W0, 4, NF * A, f) == (size_t)NF * A
                && std::fread(B0, 4, A, f) == (size_t)A
                && std::fread(W1, 4, L1 * 2 * A, f) == (size_t)L1 * 2 * A
                && std::fread(B1, 4, L1, f) == (size_t)L1
                && std::fread(W2, 4, L1, f) == (size_t)L1
                && std::fread(&B2, 4, 1, f) == 1;
            std::fclose(f);
            if (!ok) { std::fprintf(stderr, "bad FULLNNUE file\n"); std::exit(1); }
            ready = true;
        });
    }

    // markers[p][mb] (9-bit), states[0..2] (9-bit masks), constraint 0..9, stm 0/1
    static int evaluate(const int markers[2][9], const int states[3], int constraint, int stm) {
        if (!ready) load();
        float acc[2][A];
        const int decided = states[0] | states[1] | states[2];
        for (int view = 0; view < 2; view++) {
            const int P = view == 0 ? stm : stm ^ 1;
            float *ac = acc[view];
            for (int i = 0; i < A; i++) ac[i] = B0[i];
            auto add = [&](int f) { for (int i = 0; i < A; i++) ac[i] += W0[f][i]; };
            for (int mb = 0; mb < 9; mb++) {
                if (decided >> mb & 1) {
                    int cls = (states[2] >> mb & 1) ? 2 : ((states[P] >> mb & 1) ? 0 : 1);
                    add(162 + mb * 3 + cls);
                    continue;
                }
                for (int owner = 0; owner < 2; owner++) {
                    int m = markers[owner][mb];
                    while (m) {
                        int sq = __builtin_ctz(m); m &= m - 1;
                        add((mb * 9 + sq) * 2 + (owner != P));
                    }
                }
            }
            add(189 + constraint);
        }
        float h[2 * A];
        for (int v = 0; v < 2; v++)
            for (int i = 0; i < A; i++) h[v * A + i] = acc[v][i] < 0 ? 0 : (acc[v][i] > 1 ? 1 : acc[v][i]);
        float out = B2;
        for (int j = 0; j < L1; j++) {
            float s = B1[j];
            for (int i = 0; i < 2 * A; i++) s += W1[j][i] * h[i];
            s = s < 0 ? 0 : (s > 1 ? 1 : s);
            out += W2[j] * s;
        }
        return (int)(out * 1000.0f);
    }

    template <typename Board>
    static int evaluate_board(const Board &b, int constraint) {
        int markers[2][9];
        for (int mb = 0; mb < 9; mb++) {
            markers[0][mb] = b.mini_boards[mb].markers[0];
            markers[1][mb] = b.mini_boards[mb].markers[1];
        }
        int states[3] = {b.mini_board_states[0], b.mini_board_states[1], b.mini_board_states[2]};
        return evaluate(markers, states, constraint, b.n_moves & 1);
    }
};
