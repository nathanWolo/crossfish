#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <immintrin.h>
#include <random>
#include <string>
#include <vector>

#include "global_board.hpp"

// STM-perspective NNUE: 199 sparse features, H=32, concat STM+NSM -> 64 -> 32 -> 1.
// First layer int16, rest int8. Accumulators are int32 to avoid overflow.
struct NnueNet {
    static constexpr int N_SQ = 81;
    static constexpr int N_SQ_F = 81 * 2;
    static constexpr int N_SUPER = 9 * 3;
    static constexpr int N_CONSTR = 10;
    static constexpr int N_FEAT = N_SQ_F + N_SUPER + N_CONSTR; // 199
    static constexpr int H = 32;
    static constexpr int L2 = 32;
    static constexpr int QA = 127;
    static constexpr int QB = 64;
    static constexpr int SUPER_BASE = N_SQ_F;
    static constexpr int CONSTR_BASE = N_SQ_F + N_SUPER;
    static constexpr int UNDO_MAX = 128;

    int32_t acc[2][H]{};
    int16_t w0[N_FEAT * H]{};
    int16_t b0[H]{};
    int8_t w1[L2 * (2 * H)]{};
    int32_t b1[L2]{};
    int8_t w2[L2]{};
    int32_t b2 = 0;
    int scale = 2410;
    int crelu_max = QA;
    int asinh_s = 0;
    int constraint = 9;
    int undo_n = 0;
    int32_t undo_acc[UNDO_MAX][2][H]{};
    int undo_constraint[UNDO_MAX]{};
    bool weights_ready = false;

    static int board_constraint(const GlobalBoard &b) {
        if (b.n_moves == 0 || b.prev_move_was_pass) return 9;
        int sent = b.move_history.top().square;
        int oop = b.mini_board_states[0] | b.mini_board_states[1] | b.mini_board_states[2];
        if ((oop & (1 << sent)) != 0) return 9;
        return sent;
    }

    static int sq_index(int mb, int sq) { return mb * 9 + sq; }

    static int sq_feat(int pers, int mb, int sq, int owner) {
        int mine = (owner == pers) ? 0 : 1;
        return sq_index(mb, sq) * 2 + mine;
    }

    static int super_feat(int pers, int mb, int winner) {
        // winner: 0=p0, 1=p1, 2=draw
        int cls = 2;
        if (winner != 2) cls = (winner == pers) ? 0 : 1;
        return SUPER_BASE + mb * 3 + cls;
    }

    static int constr_feat(int c) { return CONSTR_BASE + c; }

    void add_feat(int pers, int f) {
        const int16_t *w = w0 + f * H;
        for (int h = 0; h < H; h++) acc[pers][h] += w[h];
    }

    void sub_feat(int pers, int f) {
        const int16_t *w = w0 + f * H;
        for (int h = 0; h < H; h++) acc[pers][h] -= w[h];
    }

    void add_both_constr(int c) {
        add_feat(0, constr_feat(c));
        add_feat(1, constr_feat(c));
    }

    void sub_both_constr(int c) {
        sub_feat(0, constr_feat(c));
        sub_feat(1, constr_feat(c));
    }

    void reset_bias() {
        for (int p = 0; p < 2; p++) {
            for (int h = 0; h < H; h++) acc[p][h] = b0[h];
        }
    }

    void refresh(const GlobalBoard &b) {
        reset_bias();
        for (int mb = 0; mb < 9; mb++) {
            int p0 = b.mini_boards[mb].markers[0];
            int p1 = b.mini_boards[mb].markers[1];
            for (int sq = 0; sq < 9; sq++) {
                if (p0 & (1 << sq)) {
                    add_feat(0, sq_feat(0, mb, sq, 0));
                    add_feat(1, sq_feat(1, mb, sq, 0));
                }
                if (p1 & (1 << sq)) {
                    add_feat(0, sq_feat(0, mb, sq, 1));
                    add_feat(1, sq_feat(1, mb, sq, 1));
                }
            }
            if (b.mini_board_states[0] & (1 << mb)) {
                add_feat(0, super_feat(0, mb, 0));
                add_feat(1, super_feat(1, mb, 0));
            } else if (b.mini_board_states[1] & (1 << mb)) {
                add_feat(0, super_feat(0, mb, 1));
                add_feat(1, super_feat(1, mb, 1));
            } else if (b.mini_board_states[2] & (1 << mb)) {
                add_feat(0, super_feat(0, mb, 2));
                add_feat(1, super_feat(1, mb, 2));
            }
        }
        constraint = board_constraint(b);
        add_both_constr(constraint);
        undo_n = 0;
    }

    void make(const GlobalBoard &after, Move m) {
        if (undo_n >= UNDO_MAX) {
            refresh(after);
            return;
        }
        std::memcpy(undo_acc[undo_n][0], acc[0], sizeof(acc[0]));
        std::memcpy(undo_acc[undo_n][1], acc[1], sizeof(acc[1]));
        undo_constraint[undo_n] = constraint;
        undo_n++;

        int who = (after.n_moves - 1) & 1;
        int mb = m.mini_board;
        int sq = m.square;
        add_feat(0, sq_feat(0, mb, sq, who));
        add_feat(1, sq_feat(1, mb, sq, who));

        if (after.mini_board_states[0] & (1 << mb)) {
            add_feat(0, super_feat(0, mb, 0));
            add_feat(1, super_feat(1, mb, 0));
        } else if (after.mini_board_states[1] & (1 << mb)) {
            add_feat(0, super_feat(0, mb, 1));
            add_feat(1, super_feat(1, mb, 1));
        } else if (after.mini_board_states[2] & (1 << mb)) {
            add_feat(0, super_feat(0, mb, 2));
            add_feat(1, super_feat(1, mb, 2));
        }

        int nc = board_constraint(after);
        if (nc != constraint) {
            sub_both_constr(constraint);
            add_both_constr(nc);
            constraint = nc;
        }
    }

    void unmake() {
        if (undo_n <= 0) return;
        undo_n--;
        std::memcpy(acc[0], undo_acc[undo_n][0], sizeof(acc[0]));
        std::memcpy(acc[1], undo_acc[undo_n][1], sizeof(acc[1]));
        constraint = undo_constraint[undo_n];
    }

    int crelu(int32_t x) const {
        if (x < 0) return 0;
        if (x > crelu_max) return crelu_max;
        return (int)x;
    }

    int evaluate_stm(int stm) const {
        int c[2 * H];
        const int32_t *a_stm = acc[stm];
        const int32_t *a_nsm = acc[stm ^ 1];
        for (int h = 0; h < H; h++) {
            c[h] = crelu(a_stm[h]);
            c[H + h] = crelu(a_nsm[h]);
        }
        int32_t h2[L2];
        for (int j = 0; j < L2; j++) {
            int32_t s = b1[j];
            const int8_t *row = w1 + j * (2 * H);
            for (int i = 0; i < 2 * H; i++) {
                s += c[i] * (int32_t)row[i];
            }
            h2[j] = crelu(s / QB);
        }
        int32_t out = b2;
        for (int j = 0; j < L2; j++) {
            out += h2[j] * (int32_t)w2[j];
        }
        if (asinh_s > 0) {
            double z = (double)out / (double)(QB * QA);
            return (int)std::lround((double)asinh_s * std::sinh(z));
        }
        return (int)((out * (int64_t)scale) / (int64_t)(QB * QA));
    }

    int evaluate(const GlobalBoard &b) const {
        return evaluate_stm(b.n_moves & 1);
    }

    bool acc_equal(const NnueNet &o) const {
        for (int p = 0; p < 2; p++) {
            for (int h = 0; h < H; h++) {
                if (acc[p][h] != o.acc[p][h]) return false;
            }
        }
        return constraint == o.constraint;
    }

    void copy_weights_from(const NnueNet &o) {
        std::memcpy(w0, o.w0, sizeof(w0));
        std::memcpy(b0, o.b0, sizeof(b0));
        std::memcpy(w1, o.w1, sizeof(w1));
        std::memcpy(b1, o.b1, sizeof(b1));
        std::memcpy(w2, o.w2, sizeof(w2));
        b2 = o.b2;
        scale = o.scale;
        crelu_max = o.crelu_max;
        asinh_s = o.asinh_s;
        weights_ready = o.weights_ready;
    }

    void init_random(uint32_t seed) {
        std::mt19937 rng(seed);
        std::uniform_int_distribution<int> d16(-48, 48);
        std::uniform_int_distribution<int> d8(-16, 16);
        for (int i = 0; i < N_FEAT * H; i++) w0[i] = (int16_t)d16(rng);
        for (int h = 0; h < H; h++) b0[h] = (int16_t)d16(rng);
        for (int i = 0; i < L2 * (2 * H); i++) w1[i] = (int8_t)d8(rng);
        for (int j = 0; j < L2; j++) b1[j] = d8(rng) * QA;
        for (int j = 0; j < L2; j++) w2[j] = (int8_t)d8(rng);
        b2 = d8(rng) * QA;
        scale = 2410;
        crelu_max = QA;
        asinh_s = 0;
        weights_ready = true;
    }

    void encode_state(const GlobalBoard &b, char *s93) const {
        for (int i = 0; i < 81; i++) {
            int mb = i / 9;
            int sq = i % 9;
            if (b.mini_boards[mb].markers[0] & (1 << sq)) s93[i] = '1';
            else if (b.mini_boards[mb].markers[1] & (1 << sq)) s93[i] = '2';
            else s93[i] = '0';
        }
        for (int mb = 0; mb < 9; mb++) {
            if (b.mini_board_states[0] & (1 << mb)) s93[81 + mb] = '1';
            else if (b.mini_board_states[1] & (1 << mb)) s93[81 + mb] = '2';
            else if (b.mini_board_states[2] & (1 << mb)) s93[81 + mb] = '3';
            else s93[81 + mb] = '0';
        }
        s93[90] = ((b.n_moves & 1) == 0) ? '1' : '2';
        s93[91] = (char)('0' + board_constraint(b));
        s93[92] = '0';
    }
};

#if defined(__has_include)
#if __has_include("nnue_weights.inc")
#include "nnue_weights.inc"
#define NNUE_HAS_COMPILED_WEIGHTS 1
#endif
#endif

inline int nnue_b64_decode(const char *s, unsigned char *out, int out_max) {
    auto val = [](char c) -> int {
        if (c >= 'A' && c <= 'Z') return c - 'A';
        if (c >= 'a' && c <= 'z') return c - 'a' + 26;
        if (c >= '0' && c <= '9') return c - '0' + 52;
        if (c == '+') return 62;
        if (c == '/') return 63;
        return -1;
    };
    int n = 0;
    int acc = 0;
    int bits = 0;
    for (const char *p = s; *p; p++) {
        int d = val(*p);
        if (d < 0) continue;
        acc = (acc << 6) | d;
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            if (n >= out_max) return -1;
            out[n++] = (unsigned char)((acc >> bits) & 255);
        }
    }
    return n;
}

inline bool nnue_unpack_blob(NnueNet &n, const unsigned char *p, int n_bytes) {
    const int need = 4 + 4 + 4 + NnueNet::N_FEAT * NnueNet::H * 2 + NnueNet::H * 2
                     + NnueNet::L2 * (2 * NnueNet::H) + NnueNet::L2 * 4 + NnueNet::L2;
    if (n_bytes < need) return false;
    if (p[0] != 'C' || p[1] != 'F' || p[2] != 'N' || p[3] != '1') return false;
    int off = 4;
    auto rd32 = [&]() {
        int32_t v;
        std::memcpy(&v, p + off, 4);
        off += 4;
        return v;
    };
    n.scale = rd32();
    n.b2 = rd32();
    std::memcpy(n.w0, p + off, sizeof(n.w0));
    off += (int)sizeof(n.w0);
    std::memcpy(n.b0, p + off, sizeof(n.b0));
    off += (int)sizeof(n.b0);
    std::memcpy(n.w1, p + off, sizeof(n.w1));
    off += (int)sizeof(n.w1);
    std::memcpy(n.b1, p + off, sizeof(n.b1));
    off += (int)sizeof(n.b1);
    std::memcpy(n.w2, p + off, sizeof(n.w2));
    n.weights_ready = true;
    return true;
}

inline void nnue_load_compiled(NnueNet &n) {
#ifdef NNUE_HAS_COMPILED_WEIGHTS
    std::memcpy(n.w0, NNUE_W0, sizeof(NNUE_W0));
    std::memcpy(n.b0, NNUE_B0, sizeof(NNUE_B0));
    std::memcpy(n.w1, NNUE_W1, sizeof(NNUE_W1));
    std::memcpy(n.b1, NNUE_B1, sizeof(NNUE_B1));
    std::memcpy(n.w2, NNUE_W2, sizeof(NNUE_W2));
    n.b2 = NNUE_B2;
    n.scale = NNUE_SCALE;
    n.crelu_max = NNUE_CRELU_MAX;
    n.asinh_s = NNUE_ASINH_S;
    n.weights_ready = true;
#else
    n.init_random(20260814u);
#endif
}

inline bool nnue_load_file(NnueNet &n, const char *path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    in.seekg(0, std::ios::end);
    const std::streamoff n_bytes = in.tellg();
    in.seekg(0);
    if (n_bytes < 8) return false;
    std::vector<unsigned char> buf((size_t)n_bytes);
    in.read(reinterpret_cast<char *>(buf.data()), n_bytes);
    if (!in) return false;
    const unsigned char *p = buf.data();
    if (n_bytes >= 4 && p[0] == 'C' && p[1] == 'F' && p[2] == 'N' && p[3] == '2') {
        const int need = 4 + 16 + NnueNet::N_FEAT * NnueNet::H * 2 + NnueNet::H * 2
                         + NnueNet::L2 * (2 * NnueNet::H) + NnueNet::L2 * 4 + NnueNet::L2;
        if ((int)n_bytes < need) return false;
        int off = 4;
        auto rd32 = [&]() {
            int32_t v;
            std::memcpy(&v, p + off, 4);
            off += 4;
            return v;
        };
        n.scale = rd32();
        n.b2 = rd32();
        n.crelu_max = rd32();
        n.asinh_s = rd32();
        std::memcpy(n.w0, p + off, sizeof(n.w0));
        off += (int)sizeof(n.w0);
        std::memcpy(n.b0, p + off, sizeof(n.b0));
        off += (int)sizeof(n.b0);
        std::memcpy(n.w1, p + off, sizeof(n.w1));
        off += (int)sizeof(n.w1);
        std::memcpy(n.b1, p + off, sizeof(n.b1));
        off += (int)sizeof(n.b1);
        std::memcpy(n.w2, p + off, sizeof(n.w2));
        n.weights_ready = true;
        return true;
    }
    if (!nnue_unpack_blob(n, p, (int)n_bytes)) return false;
    n.crelu_max = NnueNet::QA;
    n.asinh_s = 0;
    return true;
}

// Mini-index embedding net. Shared 3^9 LUT analogue + super/location/constraint.
// Weights live in g_nnue_mini; evaluate is from-scratch (speed is not the gate).
struct MiniNnue {
    static constexpr int N_IDX = 19683;
    static constexpr int D_MAX = 64;
    static constexpr int H_MAX = 256;
    static constexpr float CRELU = 1.0e9f;

    int d = 0;
    int h = 0;
    std::vector<float> emb;
    std::vector<float> super;
    std::vector<float> loc;
    std::vector<float> constr;
    std::vector<float> active;
    std::vector<float> w1;
    std::vector<float> b1;
    std::vector<float> w2;
    float b2 = 0;
    std::vector<float> v_emb;
    std::vector<float> v_super;
    std::vector<float> v_loc;
    std::vector<float> v_constr;
    std::vector<float> v_active;
    std::vector<float> v_win_loc;
    std::vector<uint8_t> has_mine_tiar;
    std::vector<uint8_t> has_opp_tiar;
    bool has_lut = false;
    bool ready = false;

    void fill_tiar_flags() {
        static const int tiar[] = {
            (1 << 0) + (1 << 1),  (1 << 2),
            (1 << 1) + (1 << 2), (1 << 0),
            (1 << 3) + (1 << 4), (1 << 5),
            (1 << 4) + (1 << 5), (1 << 3),
            (1 << 6) + (1 << 7), (1 << 8),
            (1 << 7) + (1 << 8), (1 << 6),
            (1 << 0) + (1 << 3), (1 << 6),
            (1 << 3) + (1 << 6), (1 << 0),
            (1 << 1) + (1 << 4), (1 << 7),
            (1 << 4) + (1 << 7), (1 << 1),
            (1 << 2) + (1 << 5), (1 << 8),
            (1 << 5) + (1 << 8), (1 << 2),
            (1 << 0) + (1 << 4), (1 << 8),
            (1 << 4) + (1 << 8), (1 << 0),
            (1 << 2) + (1 << 4), (1 << 6),
            (1 << 4) + (1 << 6), (1 << 2),
            (1 << 0) + (1 << 2), (1 << 1),
            (1 << 3) + (1 << 5), (1 << 4),
            (1 << 6) + (1 << 8), (1 << 7),
            (1 << 0) + (1 << 6), (1 << 3),
            (1 << 1) + (1 << 7), (1 << 4),
            (1 << 2) + (1 << 8), (1 << 5),
            (1 << 0) + (1 << 8), (1 << 4),
            (1 << 2) + (1 << 6), (1 << 4)
        };
        has_mine_tiar.assign(N_IDX, 0);
        has_opp_tiar.assign(N_IDX, 0);
        for (int idx = 0; idx < N_IDX; idx++) {
            int p0 = 0, p1 = 0, t = idx;
            for (int s = 0; s < 9; s++) {
                int cell = t % 3;
                t /= 3;
                if (cell == 1) p0 |= (1 << s);
                else if (cell == 2) p1 |= (1 << s);
            }
            int p0_tiar = 0, p1_tiar = 0;
            for (int i = 0; i < 24; i++) {
                int two = tiar[i * 2];
                int third = tiar[i * 2 + 1];
                p0_tiar += ((__builtin_popcount(p0 & two) - __builtin_popcount(p1 & third)) / 2);
                p1_tiar += ((__builtin_popcount(p1 & two) - __builtin_popcount(p0 & third)) / 2);
            }
            has_mine_tiar[(size_t)idx] = (uint8_t)(p0_tiar != 0);
            has_opp_tiar[(size_t)idx] = (uint8_t)(p1_tiar != 0);
        }
    }

    static int mini_index(int mine, int opp) {
        int idx = 0;
        int mul = 1;
        for (int s = 0; s < 9; s++) {
            int v = 0;
            if (mine & (1 << s)) v = 1;
            else if (opp & (1 << s)) v = 2;
            idx += v * mul;
            mul *= 3;
        }
        return idx;
    }

    bool load(const char *path) {
        ready = false;
        std::ifstream in(path, std::ios::binary);
        if (!in) return false;
        char magic[4];
        in.read(magic, 4);
        if (!in || magic[0] != 'C' || magic[1] != 'F' || magic[2] != 'M' ||
            (magic[3] != '1' && magic[3] != '2')) {
            return false;
        }
        const bool cfm2 = magic[3] == '2';
        int32_t dd = 0, hh = 0;
        in.read(reinterpret_cast<char *>(&dd), 4);
        in.read(reinterpret_cast<char *>(&hh), 4);
        if (!in || dd < 1 || hh < 1 || dd > D_MAX || hh > H_MAX) return false;
        d = dd;
        h = hh;
        const int in_dim = 10 * d;
        auto rd = [&](std::vector<float> &v, int n) {
            v.assign((size_t)n, 0.f);
            in.read(reinterpret_cast<char *>(v.data()), (std::streamsize)n * 4);
            return (bool)in;
        };
        if (!rd(emb, N_IDX * d)) return false;
        if (!rd(super, 4 * d)) return false;
        if (!rd(loc, 9 * d)) return false;
        if (!rd(constr, 10 * d)) return false;
        if (!rd(active, 2 * d)) return false;
        if (!rd(w1, h * in_dim)) return false;
        if (!rd(b1, h)) return false;
        if (!rd(w2, h)) return false;
        in.read(reinterpret_cast<char *>(&b2), 4);
        if (!in) return false;
        has_lut = false;
        v_emb.assign(N_IDX, 0.f);
        v_super.assign(4, 0.f);
        v_loc.assign(9, 0.f);
        v_constr.assign(10, 0.f);
        v_active.assign(2, 0.f);
        v_win_loc.assign(9, 0.f);
        if (cfm2) {
            if (!rd(v_emb, N_IDX)) return false;
            if (!rd(v_super, 4)) return false;
            if (!rd(v_loc, 9)) return false;
            if (!rd(v_constr, 10)) return false;
            if (!rd(v_active, 2)) return false;
            if (!rd(v_win_loc, 9)) return false;
            int32_t bake_flag = 0;
            in.read(reinterpret_cast<char *>(&bake_flag), 4);
            if (!in) return false;
            has_lut = bake_flag != 0;
            if (has_lut) fill_tiar_flags();
        }
        ready = true;
        return true;
    }

    static float hsum256(__m256 v) {
        __m128 lo = _mm256_castps256_ps128(v);
        __m128 hi = _mm256_extractf128_ps(v, 1);
        __m128 s = _mm_add_ps(lo, hi);
        s = _mm_add_ps(s, _mm_movehl_ps(s, s));
        s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
        return _mm_cvtss_f32(s);
    }

    static void add4_avx(float *dst, const float *e, const float *s, const float *l, const float *a, int d) {
        int i = 0;
        for (; i + 8 <= d; i += 8) {
            __m256 v = _mm256_loadu_ps(e + i);
            v = _mm256_add_ps(v, _mm256_loadu_ps(s + i));
            v = _mm256_add_ps(v, _mm256_loadu_ps(l + i));
            v = _mm256_add_ps(v, _mm256_loadu_ps(a + i));
            _mm256_storeu_ps(dst + i, v);
        }
        for (; i < d; i++) dst[i] = e[i] + s[i] + l[i] + a[i];
    }

    int evaluate_fast_d8(const GlobalBoard &b) const {
        const int stm = b.n_moves & 1;
        const int c = NnueNet::board_constraint(b);
        alignas(32) float x[80];
        for (int mb = 0; mb < 9; mb++) {
            const int mine = b.mini_boards[mb].markers[stm];
            const int opp = b.mini_boards[mb].markers[stm ^ 1];
            const int idx = mini_index(mine, opp);
            int super_cls = 0;
            if (b.mini_board_states[stm] & (1 << mb)) super_cls = 1;
            else if (b.mini_board_states[stm ^ 1] & (1 << mb)) super_cls = 2;
            else if (b.mini_board_states[2] & (1 << mb)) super_cls = 3;
            const int act = (c == mb) ? 1 : 0;
            __m256 v = _mm256_loadu_ps(emb.data() + idx * 8);
            v = _mm256_add_ps(v, _mm256_loadu_ps(super.data() + super_cls * 8));
            v = _mm256_add_ps(v, _mm256_loadu_ps(loc.data() + mb * 8));
            v = _mm256_add_ps(v, _mm256_loadu_ps(active.data() + act * 8));
            _mm256_store_ps(x + mb * 8, v);
        }
        _mm256_store_ps(x + 72, _mm256_loadu_ps(constr.data() + c * 8));
        float out = b2;
        for (int j = 0; j < h; j++) {
            const float *row = w1.data() + j * 80;
            __m256 vacc = _mm256_setzero_ps();
            for (int k = 0; k < 10; k++) {
                vacc = _mm256_fmadd_ps(_mm256_load_ps(x + k * 8), _mm256_loadu_ps(row + k * 8), vacc);
            }
            float s = b1[(size_t)j] + hsum256(vacc);
            if (s > 0) out += w2[(size_t)j] * s;
        }
        return (int)std::lround(out);
    }

    int evaluate(const GlobalBoard &b) const {
        if (!ready || d <= 0 || h <= 0) return 0;
        if (!has_lut && d == 8) return evaluate_fast_d8(b);
        const int stm = b.n_moves & 1;
        const int c = NnueNet::board_constraint(b);
        const int in_dim = 10 * d;
        alignas(32) float x[10 * D_MAX];
        float lut = has_lut ? v_constr[(size_t)c] : 0.f;
        int mine_tiar_map = 0;
        int opp_tiar_map = 0;
        for (int mb = 0; mb < 9; mb++) {
            const int mine = b.mini_boards[mb].markers[stm];
            const int opp = b.mini_boards[mb].markers[stm ^ 1];
            const int idx = mini_index(mine, opp);
            int super_cls = 0;
            if (b.mini_board_states[stm] & (1 << mb)) super_cls = 1;
            else if (b.mini_board_states[stm ^ 1] & (1 << mb)) super_cls = 2;
            else if (b.mini_board_states[2] & (1 << mb)) super_cls = 3;
            const int act = (c == mb) ? 1 : 0;
            add4_avx(x + mb * d,
                     emb.data() + idx * d,
                     super.data() + super_cls * d,
                     loc.data() + mb * d,
                     active.data() + act * d,
                     d);
            if (has_lut) {
                lut += v_super[(size_t)super_cls];
                if (super_cls == 0) {
                    lut += v_emb[(size_t)idx] + v_loc[(size_t)mb] + v_active[(size_t)act];
                    if (has_mine_tiar[(size_t)idx]) mine_tiar_map |= (1 << mb);
                    if (has_opp_tiar[(size_t)idx]) opp_tiar_map |= (1 << mb);
                } else if (super_cls == 1) {
                    lut += v_win_loc[(size_t)mb];
                } else if (super_cls == 2) {
                    lut -= v_win_loc[(size_t)mb];
                }
            }
        }
        if (has_lut) {
            static const int masks[] = {
                (1 << 0) + (1 << 1),  (1 << 2),
                (1 << 1) + (1 << 2), (1 << 0),
                (1 << 3) + (1 << 4), (1 << 5),
                (1 << 4) + (1 << 5), (1 << 3),
                (1 << 6) + (1 << 7), (1 << 8),
                (1 << 7) + (1 << 8), (1 << 6),
                (1 << 0) + (1 << 3), (1 << 6),
                (1 << 3) + (1 << 6), (1 << 0),
                (1 << 1) + (1 << 4), (1 << 7),
                (1 << 4) + (1 << 7), (1 << 1),
                (1 << 2) + (1 << 5), (1 << 8),
                (1 << 5) + (1 << 8), (1 << 2),
                (1 << 0) + (1 << 4), (1 << 8),
                (1 << 4) + (1 << 8), (1 << 0),
                (1 << 2) + (1 << 4), (1 << 6),
                (1 << 4) + (1 << 6), (1 << 2),
                (1 << 0) + (1 << 2), (1 << 1),
                (1 << 3) + (1 << 5), (1 << 4),
                (1 << 6) + (1 << 8), (1 << 7),
                (1 << 0) + (1 << 6), (1 << 3),
                (1 << 1) + (1 << 7), (1 << 4),
                (1 << 2) + (1 << 8), (1 << 5),
                (1 << 0) + (1 << 8), (1 << 4),
                (1 << 2) + (1 << 6), (1 << 4)
            };
            const int mine_b = b.mini_board_states[stm];
            const int opp_b = b.mini_board_states[stm ^ 1];
            int g = 0;
            for (int i = 0; i < 24; i++) {
                const int two = masks[i * 2];
                const int third = masks[i * 2 + 1];
                g += 1316 * ((__builtin_popcount(mine_b & two) - __builtin_popcount(opp_b & third)) / 2
                    - (__builtin_popcount(opp_b & two) - __builtin_popcount(mine_b & third)) / 2);
                g += 424 * ((__builtin_popcount((mine_tiar_map | mine_b) & two) - __builtin_popcount(opp_b & third)) / 2
                    - (__builtin_popcount((opp_tiar_map | opp_b) & two) - __builtin_popcount(mine_b & third)) / 2);
            }
            lut += (float)g;
        }
        const float *ce = constr.data() + c * d;
        std::memcpy(x + 9 * d, ce, (size_t)d * sizeof(float));
        float out = b2 + lut;
        for (int j = 0; j < h; j++) {
            const float *row = w1.data() + j * in_dim;
            __m256 vacc = _mm256_setzero_ps();
            int i = 0;
            for (; i + 8 <= in_dim; i += 8) {
                vacc = _mm256_add_ps(vacc, _mm256_mul_ps(_mm256_loadu_ps(row + i), _mm256_loadu_ps(x + i)));
            }
            float s = b1[(size_t)j] + hsum256(vacc);
            for (; i < in_dim; i++) s += row[i] * x[i];
            if (s > 0) out += w2[(size_t)j] * s;
        }
        return (int)std::lround(out);
    }

    struct Acc {
        static constexpr int UNDO = 96;
        bool ok = false;
        int c = 9;
        int undo_n = 0;
        alignas(32) float x[2][10 * D_MAX]{};
        alignas(32) float hid[2][H_MAX]{};
        struct Undo {
            int mb;
            int old_c;
            int new_c;
            alignas(32) float xmb[2][D_MAX];
            alignas(32) float xoldc[2][D_MAX];
            alignas(32) float xnewc[2][D_MAX];
            alignas(32) float xc[2][D_MAX];
            alignas(32) float hid[2][H_MAX];
        } u[UNDO];
    };

    void fill_pers(const GlobalBoard &b, int pers, float *xout) const {
        const int c = NnueNet::board_constraint(b);
        for (int mb = 0; mb < 9; mb++) {
            const int mine = b.mini_boards[mb].markers[pers];
            const int opp = b.mini_boards[mb].markers[pers ^ 1];
            const int idx = mini_index(mine, opp);
            int super_cls = 0;
            if (b.mini_board_states[pers] & (1 << mb)) super_cls = 1;
            else if (b.mini_board_states[pers ^ 1] & (1 << mb)) super_cls = 2;
            else if (b.mini_board_states[2] & (1 << mb)) super_cls = 3;
            const int act = (c == mb) ? 1 : 0;
            add4_avx(xout + mb * d,
                     emb.data() + idx * d,
                     super.data() + super_cls * d,
                     loc.data() + mb * d,
                     active.data() + act * d,
                     d);
        }
        std::memcpy(xout + 9 * d, constr.data() + c * d, (size_t)d * sizeof(float));
    }

    void fill_one(const GlobalBoard &b, int pers, int mb, int c, float *dst) const {
        const int mine = b.mini_boards[mb].markers[pers];
        const int opp = b.mini_boards[mb].markers[pers ^ 1];
        const int idx = mini_index(mine, opp);
        int super_cls = 0;
        if (b.mini_board_states[pers] & (1 << mb)) super_cls = 1;
        else if (b.mini_board_states[pers ^ 1] & (1 << mb)) super_cls = 2;
        else if (b.mini_board_states[2] & (1 << mb)) super_cls = 3;
        const int act = (c == mb) ? 1 : 0;
        add4_avx(dst,
                 emb.data() + idx * d,
                 super.data() + super_cls * d,
                 loc.data() + mb * d,
                 active.data() + act * d,
                 d);
    }

    void add_hid_delta(float *hid, int off, const float *old_s, const float *new_s) const {
        const int in_dim = 10 * d;
        alignas(32) float dx[D_MAX];
        for (int i = 0; i < d; i++) dx[i] = new_s[i] - old_s[i];
        for (int j = 0; j < h; j++) {
            hid[j] += dot_slice(w1.data() + j * in_dim + off, dx, d);
        }
    }

    void hid_from_x(const float *x, float *hid_out) const {
        const int in_dim = 10 * d;
        for (int j = 0; j < h; j++) {
            const float *row = w1.data() + j * in_dim;
            __m256 vacc = _mm256_setzero_ps();
            int i = 0;
            for (; i + 8 <= in_dim; i += 8) {
                vacc = _mm256_add_ps(vacc, _mm256_mul_ps(_mm256_loadu_ps(row + i), _mm256_loadu_ps(x + i)));
            }
            float s = b1[(size_t)j] + hsum256(vacc);
            for (; i < in_dim; i++) s += row[i] * x[i];
            hid_out[j] = s;
        }
    }

    static float dot_slice(const float *a, const float *b, int n) {
        __m256 vacc = _mm256_setzero_ps();
        int i = 0;
        for (; i + 8 <= n; i += 8) {
            vacc = _mm256_add_ps(vacc, _mm256_mul_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i)));
        }
        float s = hsum256(vacc);
        for (; i < n; i++) s += a[i] * b[i];
        return s;
    }

    void refresh(const GlobalBoard &b, Acc &acc) const {
        if (!ready || d <= 0 || h <= 0 || has_lut) {
            acc.ok = false;
            return;
        }
        acc.c = NnueNet::board_constraint(b);
        acc.undo_n = 0;
        for (int pers = 0; pers < 2; pers++) {
            fill_pers(b, pers, acc.x[pers]);
            hid_from_x(acc.x[pers], acc.hid[pers]);
        }
        acc.ok = true;
    }

    void make(const GlobalBoard &after, int mb, Acc &acc) const {
        if (!ready || has_lut) {
            acc.ok = false;
            return;
        }
        if (!acc.ok || acc.undo_n >= Acc::UNDO || d <= 0 || mb < 0 || mb > 8) {
            refresh(after, acc);
            return;
        }
        const int old_c = acc.c;
        const int new_c = NnueNet::board_constraint(after);
        Acc::Undo &u = acc.u[acc.undo_n++];
        u.mb = mb;
        u.old_c = old_c;
        u.new_c = new_c;
        alignas(32) float nslice[D_MAX];
        for (int pers = 0; pers < 2; pers++) {
            std::memcpy(u.xmb[pers], acc.x[pers] + mb * d, (size_t)d * sizeof(float));
            std::memcpy(u.xc[pers], acc.x[pers] + 9 * d, (size_t)d * sizeof(float));
            if (old_c < 9) {
                std::memcpy(u.xoldc[pers], acc.x[pers] + old_c * d, (size_t)d * sizeof(float));
            }
            if (new_c < 9) {
                std::memcpy(u.xnewc[pers], acc.x[pers] + new_c * d, (size_t)d * sizeof(float));
            }
            std::memcpy(u.hid[pers], acc.hid[pers], (size_t)h * sizeof(float));

            fill_one(after, pers, mb, new_c, nslice);
            add_hid_delta(acc.hid[pers], mb * d, acc.x[pers] + mb * d, nslice);
            std::memcpy(acc.x[pers] + mb * d, nslice, (size_t)d * sizeof(float));

            if (old_c < 9 && old_c != mb) {
                fill_one(after, pers, old_c, new_c, nslice);
                add_hid_delta(acc.hid[pers], old_c * d, acc.x[pers] + old_c * d, nslice);
                std::memcpy(acc.x[pers] + old_c * d, nslice, (size_t)d * sizeof(float));
            }
            if (new_c < 9 && new_c != mb && new_c != old_c) {
                fill_one(after, pers, new_c, new_c, nslice);
                add_hid_delta(acc.hid[pers], new_c * d, acc.x[pers] + new_c * d, nslice);
                std::memcpy(acc.x[pers] + new_c * d, nslice, (size_t)d * sizeof(float));
            }

            const float *nc = constr.data() + new_c * d;
            add_hid_delta(acc.hid[pers], 9 * d, acc.x[pers] + 9 * d, nc);
            std::memcpy(acc.x[pers] + 9 * d, nc, (size_t)d * sizeof(float));
        }
        acc.c = new_c;
    }

    void unmake(Acc &acc) const {
        if (!acc.ok || acc.undo_n <= 0) {
            acc.ok = false;
            return;
        }
        const Acc::Undo &u = acc.u[--acc.undo_n];
        acc.c = u.old_c;
        for (int pers = 0; pers < 2; pers++) {
            std::memcpy(acc.x[pers] + u.mb * d, u.xmb[pers], (size_t)d * sizeof(float));
            std::memcpy(acc.x[pers] + 9 * d, u.xc[pers], (size_t)d * sizeof(float));
            if (u.old_c < 9 && u.old_c != u.mb) {
                std::memcpy(acc.x[pers] + u.old_c * d, u.xoldc[pers], (size_t)d * sizeof(float));
            }
            if (u.new_c < 9 && u.new_c != u.mb && u.new_c != u.old_c) {
                std::memcpy(acc.x[pers] + u.new_c * d, u.xnewc[pers], (size_t)d * sizeof(float));
            }
            std::memcpy(acc.hid[pers], u.hid[pers], (size_t)h * sizeof(float));
        }
    }

    int evaluate_acc(const Acc &acc, int stm) const {
        if (!acc.ok || !ready) return 0;
        float out = b2;
        const float *hid = acc.hid[stm & 1];
        for (int j = 0; j < h; j++) {
            if (hid[j] > 0) out += w2[(size_t)j] * hid[j];
        }
        return (int)std::lround(out);
    }
};

inline NnueNet g_nnue_sparse;
inline MiniNnue g_nnue_mini;

inline bool nnue_init_runtime() {
    nnue_load_compiled(g_nnue_sparse);
    if (g_nnue_mode == 1 && g_nnue_bin_path[0] != 0) {
        if (!nnue_load_file(g_nnue_sparse, g_nnue_bin_path)) {
            std::fprintf(stderr, "failed to load sparse net %s\n", g_nnue_bin_path);
            return false;
        }
    }
    if (g_nnue_mode == 2) {
        if (g_nnue_bin_path[0] == 0 || !g_nnue_mini.load(g_nnue_bin_path)) {
            std::fprintf(stderr, "failed to load mini net %s\n", g_nnue_bin_path);
            return false;
        }
    }
    return true;
}
