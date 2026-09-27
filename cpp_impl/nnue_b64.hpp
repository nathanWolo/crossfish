#pragma once
// The engine's evaluation: the B-64 pattern-generator NNUE (net B64_d5M_57ep). crossfish_dev.hpp and the
// CodinGame bot (codingame_nnue.cpp) include this same file.
//
// The net, per perspective P (rows W = 65 wide: A = 64 accumulator lanes, then 1 PSQT lane):
//   live miniboard m      T[m][p]      p = sum over its squares i of cell_i 3^i (0 empty, 1 P's, 2 the other's)
//   decided miniboard m   DEC[3m + s]  s: 0 won by P, 1 won by the other, 2 drawn
//   acc_P = BIAS + those 9 rows (the stored accumulator), and at evaluation time
//   constraint c          CON[c] if P is to move, else CON[10 + c]   (c: 0..8 the forced board, 9 a free move)
//   forced board c < 9    F[p]         the forced board's pattern in P's view
//   h = [clamp(acc_stm + rows, 0, 1)[:64], clamp(acc_other + rows, 0, 1)[:64]] -> 16 -> clamp -> 32 -> clamp
//   -> 1; eval = trunc(1000 * (out + (psqt_stm - psqt_other) / 2)), from the side to move's view.
// T and F (9 + 1 tables of 3^9 rows) are not stored. nnue_b64_net.hpp (tools/nnue_emit_b64_header.py)
// holds the generator that makes them, an encoder 27 -> 64 -> 64 -> 32 of the pattern, one projection per
// location and one for the forced board, with the other rows and the dense head: 35,243 parameters in
// 30,923 CJK14 characters. load() decodes it once, bakes the rows of the 11,093 patterns a live board can
// have in float (fixed operation order, no FMA, so every build bakes the same floats) and quantizes as it
// goes: lanes to int16 at 2^B64_QA, the PSQT lane to int16 at 2^B64_QPS in separate arrays, the dense head to
// int16/int32 at 2^B64_QB, 2^B64_Q2, 2^B64_QO. The emitter picks those scales so that no accumulator, no
// accumulator + constraint + forced-board row and no dense sum can overflow. 25.6 MB of static tables,
// about 50 ms at start-up.
//
// Kernels: int16 accumulators, AVX2; the first dense layer runs over nonzero activation pairs
// (_mm256_madd_epi16); the hidden layers use rounded shifts back to int16.
//
// Search interface, the same in Dev and in the CodinGame bot:
//   Stack::on_make(board, mb, sq, stm, decided, markers_before, parent tt_hash)
//                          in make_move_fast(FastBoard&) just before n_moves++; unmake does nothing
//   Stack::refresh_root(board)                      where a search starts
//   Stack::evaluate_keyed(board, constraint, key)   stand-pat and static eval (eval cache keyed by tt_hash)
//   evaluate_board(board, constraint)               from scratch (Dev's evaluate(GlobalBoard&), tests)
// The accumulator stack is lazy: make only records the move; an evaluation replays the recorded moves from
// the nearest entry that holds an ancestor, keyed by tt_hash, or refreshes from scratch. The empty board's
// tt_hash is 0, so an entry holding no position is marked kNoKey, not 0.
//
// The code is tools/experiments/fast_nnue/fast_nnue_b.hpp specialised to this net, with the tables baked at
// start-up; unit_tests.cpp checks the baked tables' hashes, incremental == from scratch == a scalar
// reference, and fixed-position evals. Never FMA: the bake must stay bit-exact.
#include "nnue_b64_net.hpp"

namespace b64 {

constexpr int A = 64, W = 65, E = 32, L1 = 16, L2 = 32, NPAT = 19683, MAXPLY = 96, CACHE_BITS = 14;
constexpr int QA = B64_QA, QPS = B64_QPS, QB = B64_QB, Q2 = B64_Q2, QO = B64_QO;
constexpr int H1BITS = QA + QB < 14 ? QA + QB : 14, H1_SHIFT = QA + QB - H1BITS;
constexpr int H2BITS = H1BITS + Q2 < 15 ? H1BITS + Q2 : 15, H2_SHIFT = H1BITS + Q2 - H2BITS;
constexpr int32_t H1MAX = 1 << (QA + QB), H2MAX = 1 << (H1BITS + Q2);
constexpr int32_t H1_ROUND = H1_SHIFT > 0 ? 1 << (H1_SHIFT - 1) : 0;
constexpr int32_t H2_ROUND = H2_SHIFT > 0 ? 1 << (H2_SHIFT - 1) : 0;
constexpr int OUT_SHIFT = H2BITS + QO, FIN_SHIFT = OUT_SHIFT > QPS + 1 ? OUT_SHIFT : QPS + 1;
constexpr int64_t OUT_MUL = 1LL << (FIN_SHIFT - OUT_SHIFT), PS_MUL = 1LL << (FIN_SHIFT - QPS - 1);

// The generator (the BGN1 layout: W[out][in], proj_w[m][k][lane]).
struct Gen {
    float e0w[64][27], e0b[64], e1w[64][64], e1b[64], e2w[32][64], e2b[32];
    float pw[9][E][W], pb[9][W], fw[E][W], fb[W];
    float bias[W], dec[27][W], con[20][W];
    float w1[L1][2 * A], b1[L1], w2[L2][L1], b2[L2], wo[L2], bo;
};

// Integer tables. Rows of patterns no live board can have stay 0.
alignas(64) static int16_t T[9][NPAT][A];
static int16_t TP[9][NPAT];
alignas(64) static int16_t F[NPAT][A];
static int16_t FP[NPAT];
alignas(64) static int16_t DEC[27][A], CON[20][A], BIAS[A], ZERO[A];
static int16_t DECP[27], CONP[20], BIASP;
alignas(64) static int32_t W1p[A][L1];  // pair p, output k: lo16 = W1q[k][2p], hi16 = W1q[k][2p+1]
alignas(64) static int32_t B1[L1];
alignas(64) static int32_t W2p[L1 / 2][L2];
alignas(64) static int32_t B2[L2];
alignas(64) static int32_t WO[L2];
static int32_t BO;
static bool READY = false;

static uint16_t TERN[512];  // TERN[mask] = sum over the mask's squares i of 3^i (filled by load())

// ---------------------------------------------------------------- payload

// MSB-first bit reader straight over the CJK14 characters (14 bits each, U+4E00 + value).
struct Bits {
    const unsigned char *p;
    uint64_t acc = 0;
    int n = 0;
    unsigned get(int k) {
        while (n < k) {
            while ((*p & 0xF0) != 0xE0) p++;
            acc = acc << 14 | ((((p[0] & 15u) << 12) | ((p[1] & 63u) << 6) | (p[2] & 63u)) - 0x4E00u);
            p += 3;
            n += 14;
        }
        n -= k;
        return (unsigned)(acc >> n) & ((1u << k) - 1);
    }
};

// One payload matrix (rows x cols, row-major): ns bf16 scales (row r uses scale r % ns: proj's 9
// locations share a scale per lane), the Rice parameters of the normal and the PSQT rows (row % 65 == 64
// of proj, fwd, dec, con), then Rice(zigzag(q)) per value; value = float(q) * scale.
static void read_mat(Bits &b, float *m, int rows, int cols, int ns, bool psqt) {
    float sc[W];
    for (int i = 0; i < ns; i++) {
        const uint32_t u = b.get(16) << 16;
        std::memcpy(&sc[i], &u, 4);
    }
    const int k0 = b.get(4), k1 = b.get(4);
    for (int r = 0; r < rows; r++) {
        const int k = psqt && r % W == A ? k1 : k0;
        for (int c = 0; c < cols; c++) {
            unsigned u = 0;
            while (b.get(1)) u++;
            u = u << k | b.get(k);
            *m++ = (float)((int)(u >> 1) ^ -(int)(u & 1)) * sc[r % ns];
        }
    }
}

// Decode the payload into the Generator layout.
static void unpack(Gen &g) {
    Bits b{(const unsigned char *)B64_NET_CJK};
    static float m[9 * W * 33];
    auto aug = [&](float *w, float *bias, int out, int in) {  // [bias | W] rows
        read_mat(b, m, out, in + 1, out, false);
        for (int o = 0; o < out; o++) {
            bias[o] = m[o * (in + 1)];
            for (int i = 0; i < in; i++) w[o * in + i] = m[o * (in + 1) + 1 + i];
        }
    };
    aug(&g.e0w[0][0], g.e0b, 64, 27);
    aug(&g.e1w[0][0], g.e1b, 64, 64);
    aug(&g.e2w[0][0], g.e2b, 32, 64);
    read_mat(b, m, 9 * W, 33, W, true);  // row 65 m + j: [proj_b[m][j] | proj_w[m][:, j]]
    for (int r = 0; r < 9 * W; r++) {
        g.pb[r / W][r % W] = m[r * 33];
        for (int k = 0; k < E; k++) g.pw[r / W][k][r % W] = m[r * 33 + 1 + k];
    }
    read_mat(b, m, W, 33, W, true);  // row j: [fwd_b[j] | fwd_w[:, j]]
    for (int j = 0; j < W; j++) {
        g.fb[j] = m[j * 33];
        for (int k = 0; k < E; k++) g.fw[k][j] = m[j * 33 + 1 + k];
    }
    read_mat(b, m, W, 27, W, true);  // dec and con transposed: one row per lane
    for (int j = 0; j < W; j++)
        for (int r = 0; r < 27; r++) g.dec[r][j] = m[j * 27 + r];
    read_mat(b, m, W, 20, W, true);
    for (int j = 0; j < W; j++)
        for (int r = 0; r < 20; r++) g.con[r][j] = m[j * 20 + r];
    read_mat(b, g.bias, 1, W, 1, false);
    aug(&g.w1[0][0], g.b1, L1, 2 * A);
    aug(&g.w2[0][0], g.b2, L2, L1);
    aug(g.wo, &g.bo, 1, L2);
}

// ---------------------------------------------------------------- bake + quantize

// Live pattern (no line for either side, not full): digit 1 / 2 masks.
static bool live(int p, int *d) {
    int m1 = 0, m2 = 0;
    for (int k = 0; k < 9; k++, p /= 3) {
        d[k] = p % 3;
        m1 |= (d[k] == 1) << k;
        m2 |= (d[k] == 2) << k;
    }
    static const int LINES[8] = {7, 56, 448, 73, 146, 292, 273, 84};
    for (int l : LINES)
        if ((m1 & l) == l || (m2 & l) == l) return false;
    return (m1 | m2) != 511;
}

// The encoder's embedding of a live pattern (the float operations in this order are the reference).
static void embed(const Gen &g, const int *d, float *e) {
    float h[64], h2[64];
    for (int o = 0; o < 64; o++) {  // one-hot input: 9 columns
        float s = g.e0b[o];
        for (int k = 0; k < 9; k++) s += g.e0w[o][3 * k + d[k]];
        h[o] = s;
    }
    for (int i = 0; i < 64; i++) h[i] = h[i] < 0.f ? 0.f : h[i];  // std::max(h, 0.f)
    for (int o = 0; o < 64; o++) {
        float s = g.e1b[o];
        for (int i = 0; i < 64; i++) s += g.e1w[o][i] * h[i];
        h2[o] = s;
    }
    for (int i = 0; i < 64; i++) h2[i] = h2[i] < 0.f ? 0.f : h2[i];
    for (int o = 0; o < E; o++) {
        float s = g.e2b[o];
        for (int i = 0; i < 64; i++) s += g.e2w[o][i] * h2[i];
        e[o] = s;
    }
}

// A projection: row = pb + sum_k e[k] * pw[k], lane by lane in k order.
static void project(const float (*pw)[W], const float *pb, const float *e, float *row) {
    for (int j = 0; j < W; j++) row[j] = pb[j];
    for (int k = 0; k < E; k++) {
        const float ek = e[k];
        for (int j = 0; j < W; j++) row[j] += ek * pw[k][j];
    }
}

// Round half to even of x * 2^bits.
static long long qz(float x, int bits) { return (long long)std::nearbyint((double)x * (double)(1LL << bits)); }

static void put(const float *src, int16_t *dst, int16_t *dps) {
    for (int i = 0; i < A; i++) dst[i] = (int16_t)qz(src[i], QA);
    *dps = (int16_t)qz(src[A], QPS);
}

static void load() {
    if (READY) return;
    for (int m = 0; m < 512; m++)
        for (int k = 0, p = 1; k < 9; k++, p *= 3)
            if (m >> k & 1) TERN[m] += p;
    static Gen g;
    unpack(g);
    float e[E], row[W];
    int d[9];
    for (int p = 0; p < NPAT; p++) {
        if (!live(p, d)) continue;
        embed(g, d, e);
        for (int m = 0; m < 9; m++) {
            project(g.pw[m], g.pb[m], e, row);
            put(row, T[m][p], &TP[m][p]);
        }
        project(g.fw, g.fb, e, row);
        put(row, F[p], &FP[p]);
    }
    for (int r = 0; r < 27; r++) put(g.dec[r], DEC[r], &DECP[r]);
    for (int r = 0; r < 20; r++) put(g.con[r], CON[r], &CONP[r]);
    put(g.bias, BIAS, &BIASP);
    int16_t w1[L1][2 * A], w2[L2][L1];
    for (int k = 0; k < L1; k++) {
        B1[k] = (int32_t)qz(g.b1[k], QA + QB);
        for (int i = 0; i < 2 * A; i++) w1[k][i] = (int16_t)qz(g.w1[k][i], QB);
    }
    for (int p = 0; p < A; p++)
        for (int k = 0; k < L1; k++)
            W1p[p][k] = (int32_t)((uint32_t)(uint16_t)w1[k][2 * p] | (uint32_t)(uint16_t)w1[k][2 * p + 1] << 16);
    for (int k = 0; k < L2; k++) {
        B2[k] = (int32_t)qz(g.b2[k], H1BITS + Q2);
        for (int j = 0; j < L1; j++) w2[k][j] = (int16_t)qz(g.w2[k][j], Q2);
    }
    for (int e2 = 0; e2 < L1 / 2; e2++)
        for (int k = 0; k < L2; k++)
            W2p[e2][k] = (int32_t)((uint32_t)(uint16_t)w2[k][2 * e2] | (uint32_t)(uint16_t)w2[k][2 * e2 + 1] << 16);
    for (int k = 0; k < L2; k++) WO[k] = (int32_t)qz(g.wo[k], QO);
    BO = (int32_t)qz(g.bo, QO + H2BITS);
    READY = true;
}

// ---------------------------------------------------------------- kernels

__attribute__((always_inline)) inline int pat_of(int mine, int theirs) { return TERN[mine] + 2 * TERN[theirs]; }
__attribute__((always_inline)) inline int dec_idx(int mb, int state, int P) {
    return mb * 3 + (state == 2 ? 2 : (state == P ? 0 : 1));
}

__attribute__((always_inline)) inline void sub_add2(const int16_t *p0, int16_t *c0, const int16_t *a0, const int16_t *s0,
                                                    const int16_t *p1, int16_t *c1, const int16_t *a1, const int16_t *s1) {
    for (int t = 0; t < A; t += 16) {
        const __m256i x0 = _mm256_sub_epi16(_mm256_add_epi16(_mm256_load_si256((const __m256i *)(p0 + t)),
                                                             _mm256_load_si256((const __m256i *)(a0 + t))),
                                            _mm256_load_si256((const __m256i *)(s0 + t)));
        const __m256i x1 = _mm256_sub_epi16(_mm256_add_epi16(_mm256_load_si256((const __m256i *)(p1 + t)),
                                                             _mm256_load_si256((const __m256i *)(a1 + t))),
                                            _mm256_load_si256((const __m256i *)(s1 + t)));
        _mm256_store_si256((__m256i *)(c0 + t), x0);
        _mm256_store_si256((__m256i *)(c1 + t), x1);
    }
}

__attribute__((always_inline)) inline int finish(int64_t out, int32_t psd) {
    const int64_t x = 1000 * (out * OUT_MUL + (int64_t)psd * PS_MUL);
    return (int)(x >= 0 ? (x >> FIN_SHIFT) : -((-x) >> FIN_SHIFT));
}

// AVX2 head: us/them stored accumulators, cu/ct constraint rows, fu/ft forced-board rows (ZERO on a
// free move), psd the PSQT difference (stm - other).
static int eval_avx(const int16_t *us, const int16_t *them, const int16_t *cu, const int16_t *ct,
                    const int16_t *fu, const int16_t *ft, int32_t psd) {
    alignas(32) int16_t act[2 * A];
    uint64_t mask[2] = {0, 0};
    const __m256i zero = _mm256_setzero_si256();
    const __m256i qa = _mm256_set1_epi16((int16_t)(1 << QA));
    for (int v = 0; v < 2; v++) {
        const int16_t *acc = v ? them : us, *c = v ? ct : cu, *f = v ? ft : fu;
        int16_t *out = act + v * A;
        for (int j = 0; j < A / 16; j++) {
            __m256i x = _mm256_add_epi16(_mm256_add_epi16(_mm256_load_si256((const __m256i *)(acc + 16 * j)),
                                                          _mm256_load_si256((const __m256i *)(c + 16 * j))),
                                         _mm256_load_si256((const __m256i *)(f + 16 * j)));
            x = _mm256_min_epi16(_mm256_max_epi16(x, zero), qa);
            _mm256_store_si256((__m256i *)(out + 16 * j), x);
            const int pb = v * (A / 2) + j * 8;  // first activation pair of this vector
            mask[pb >> 6] |= (uint64_t)(uint32_t)_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpgt_epi32(x, zero)))
                             << (pb & 63);
        }
    }
    __m256i s[L1 / 8];
    for (int r = 0; r < L1 / 8; r++) s[r] = _mm256_load_si256((const __m256i *)(B1 + 8 * r));
    const int32_t *act32 = (const int32_t *)act;
    for (int w = 0; w < 2; w++) {
        uint64_t m = mask[w];
        while (m) {
            const int p = w * 64 + __builtin_ctzll(m);
            m &= m - 1;
            const __m256i a = _mm256_set1_epi32(act32[p]);
            const __m256i *wp = (const __m256i *)W1p[p];
            for (int r = 0; r < L1 / 8; r++) s[r] = _mm256_add_epi32(s[r], _mm256_madd_epi16(a, _mm256_load_si256(wp + r)));
        }
    }
    const __m128i sh1 = _mm_cvtsi32_si128(H1_SHIFT), sh2 = _mm_cvtsi32_si128(H2_SHIFT);
    alignas(32) int32_t hp[L1 / 2];  // h1 as int16 pairs (2e, 2e+1)
    for (int r = 0; r < L1 / 16; r++) {
        const __m256i a = _mm256_sra_epi32(_mm256_add_epi32(_mm256_min_epi32(_mm256_max_epi32(s[2 * r], zero),
                                                                             _mm256_set1_epi32(H1MAX)),
                                                            _mm256_set1_epi32(H1_ROUND)), sh1);
        const __m256i b = _mm256_sra_epi32(_mm256_add_epi32(_mm256_min_epi32(_mm256_max_epi32(s[2 * r + 1], zero),
                                                                             _mm256_set1_epi32(H1MAX)),
                                                            _mm256_set1_epi32(H1_ROUND)), sh1);
        // packus interleaves the 128-bit halves; permute 0xD8 restores h[16r .. 16r+15] order
        _mm256_store_si256((__m256i *)(hp + 8 * r), _mm256_permute4x64_epi64(_mm256_packus_epi32(a, b), 0xD8));
    }
    __m256i t[L2 / 8];
    for (int r = 0; r < L2 / 8; r++) t[r] = _mm256_load_si256((const __m256i *)(B2 + 8 * r));
    for (int e = 0; e < L1 / 2; e++) {
        const __m256i a = _mm256_set1_epi32(hp[e]);
        const __m256i *wp = (const __m256i *)W2p[e];
        for (int r = 0; r < L2 / 8; r++) t[r] = _mm256_add_epi32(t[r], _mm256_madd_epi16(a, _mm256_load_si256(wp + r)));
    }
    __m256i o = _mm256_setzero_si256();
    for (int r = 0; r < L2 / 8; r++) {
        const __m256i a = _mm256_sra_epi32(_mm256_add_epi32(_mm256_min_epi32(_mm256_max_epi32(t[r], zero),
                                                                             _mm256_set1_epi32(H2MAX)),
                                                            _mm256_set1_epi32(H2_ROUND)), sh2);
        o = _mm256_add_epi32(o, _mm256_mullo_epi32(a, _mm256_load_si256((const __m256i *)(WO + 8 * r))));
    }
    __m128i o4 = _mm_add_epi32(_mm256_castsi256_si128(o), _mm256_extracti128_si256(o, 1));
    o4 = _mm_add_epi32(o4, _mm_shuffle_epi32(o4, 0x4E));
    o4 = _mm_add_epi32(o4, _mm_shuffle_epi32(o4, 0xB1));
    return finish((int64_t)_mm_cvtsi128_si32(o4) + BO, psd);
}

// One perspective's accumulator of a whole board, from scratch.
template <class Board>
__attribute__((always_inline)) inline void scratch(const Board &b, int P, int16_t *out, int32_t &ps) {
    const int16_t *rows[9];
    ps = BIASP;
    const int oop = b.mini_board_states[0] | b.mini_board_states[1] | b.mini_board_states[2];
    for (int mb = 0; mb < 9; mb++) {
        const int bit = 1 << mb;
        if (oop & bit) {
            const int st = (b.mini_board_states[2] & bit) ? 2 : ((b.mini_board_states[0] & bit) ? 0 : 1);
            const int d = dec_idx(mb, st, P);
            rows[mb] = DEC[d];
            ps += DECP[d];
        } else {
            const int pat = pat_of(b.mini_boards[mb].markers[P], b.mini_boards[mb].markers[P ^ 1]);
            rows[mb] = T[mb][pat];
            ps += TP[mb][pat];
        }
    }
    for (int t = 0; t < A; t += 16) {
        __m256i v = _mm256_load_si256((const __m256i *)(BIAS + t));
        for (int r = 0; r < 9; r++) v = _mm256_add_epi16(v, _mm256_load_si256((const __m256i *)(rows[r] + t)));
        _mm256_store_si256((__m256i *)(out + t), v);
    }
}

// Evaluation-time rows: the constraint rows per side, the forced board's pattern row in each
// perspective's view, and their PSQT difference.
template <class Board>
__attribute__((always_inline)) inline int eval_with(const Board &b, int stm, int c, const int16_t *us,
                                                    const int16_t *them, int32_t psd) {
    const int16_t *fu = ZERO, *ft = ZERO;
    psd += (int32_t)CONP[c] - CONP[10 + c];
    if (c < 9) {
        const int t0 = TERN[b.mini_boards[c].markers[0]], t1 = TERN[b.mini_boards[c].markers[1]];
        const int ps = stm ? t1 + 2 * t0 : t0 + 2 * t1;  // stm's view
        const int pt = stm ? t0 + 2 * t1 : t1 + 2 * t0;  // the other's view
        fu = F[ps];
        ft = F[pt];
        psd += FP[ps] - FP[pt];
    }
    return eval_avx(us, them, CON[c], CON[10 + c], fu, ft, psd);
}

// From scratch: what Stack::evaluate returns for the same board (Dev's evaluate(GlobalBoard&), tests).
template <class Board>
inline int evaluate_board(const Board &b, int constraint) {
    load();
    alignas(64) int16_t acc[2][A];
    int32_t ps[2];
    scratch(b, 0, acc[0], ps[0]);
    scratch(b, 1, acc[1], ps[1]);
    const int stm = b.n_moves & 1;
    return eval_with(b, stm, constraint, acc[stm], acc[stm ^ 1], ps[stm] - ps[stm ^ 1]);
}

// ---------------------------------------------------------------- incremental stack

struct Dirty {
    uint8_t mb, sq, stm;
    int8_t decided;      // -1, or the mini_board_states index the move set
    uint16_t before[2];  // the miniboard's markers before the move
    uint64_t pkey;       // tt_hash of the position the move was made from (MoveUndo.tt_hash)
    uint64_t ckey;       // tt_hash of the position it led to
};

// The accumulator stack: one entry per n_moves holding both absolute perspectives and keyed by the
// position it holds; make records the move, evaluate replays lazily from the nearest entry holding an
// ancestor (or refreshes from scratch), unmake does nothing. evaluate_keyed puts the direct-mapped eval
// cache (2^14 entries keyed by tt_hash) in front.
struct Stack {
    struct alignas(64) Entry {
        int16_t v[2][A];
        int32_t ps[2];
    };
    struct CacheEntry {
        uint64_t key;
        int32_t eval, pad;
    };
    static constexpr uint64_t CMASK = (1u << CACHE_BITS) - 1;
    Entry acc[MAXPLY];
    uint64_t pos_key[MAXPLY];
    Dirty dirty[MAXPLY];
    CacheEntry cache[1 << CACHE_BITS];

    // The empty board's tt_hash is 0, so 0 cannot mean "no position" in pos_key.
    static constexpr uint64_t kNoKey = 0x9E3779B97F4A7C15ull;

    Stack() {
        load();
        for (uint64_t &k : pos_key) k = kNoKey;
        std::memset(dirty, 0, sizeof(dirty));
        std::memset(cache, 0, sizeof(cache));
    }
    Stack(const Stack &) = default;
    Stack &operator=(const Stack &) = default;

    template <typename Board>
    __attribute__((always_inline)) void refresh_root(const Board &b) {
        const int ply = b.n_moves;
        scratch(b, 0, acc[ply].v[0], acc[ply].ps[0]);
        scratch(b, 1, acc[ply].v[1], acc[ply].ps[1]);
        pos_key[ply] = b.tt_hash;
    }

    __attribute__((always_inline)) static void prefetch_row(const int16_t *r) {
        _mm_prefetch((const char *)r, _MM_HINT_T0);
        _mm_prefetch((const char *)r + 64, _MM_HINT_T0);
    }

    // make_move_fast(FastBoard&) just before board.n_moves++ (the board is otherwise the child's, its
    // tt_hash final); parent_key is the parent's tt_hash (MoveUndo.tt_hash).
    template <typename Board>
    __attribute__((always_inline)) void on_make(const Board &b, int mb, int sq, int stm, int decided, int before_stm,
                                                uint64_t parent_key) {
        const int before_other = b.mini_boards[mb].markers[stm ^ 1];
        Dirty &d = dirty[b.n_moves + 1];
        d.mb = (uint8_t)mb;
        d.sq = (uint8_t)sq;
        d.stm = (uint8_t)stm;
        d.decided = (int8_t)decided;
        d.before[stm] = (uint16_t)before_stm;
        d.before[stm ^ 1] = (uint16_t)before_other;
        d.pkey = parent_key;
        d.ckey = b.tt_hash;
        _mm_prefetch((const char *)&cache[b.tt_hash & CMASK], _MM_HINT_T0);  // evaluate_keyed reads it first
        const int ts = TERN[before_stm], to = TERN[before_other];
        if (decided < 0) {  // the rows the child's update adds
            const int ts2 = ts + POW3_SQ[sq];
            prefetch_row(T[mb][ts2 + 2 * to]);
            prefetch_row(T[mb][to + 2 * ts2]);
        }
        const int c = b.active_board;  // the child's forced board
        if (c < 9) {
            const int t0 = TERN[b.mini_boards[c].markers[0]], t1 = TERN[b.mini_boards[c].markers[1]];
            prefetch_row(F[t0 + 2 * t1]);
            prefetch_row(F[t1 + 2 * t0]);
        }
    }
    static constexpr int POW3_SQ[9] = {1, 3, 9, 27, 81, 243, 729, 2187, 6561};

    __attribute__((always_inline)) void update(int j) {
        const Dirty &d = dirty[j];
        const Entry &par = acc[j - 1];
        Entry &ch = acc[j];
        const int t0 = TERN[d.before[0]], t1 = TERN[d.before[1]];
        const int old0 = t0 + 2 * t1, old1 = t1 + 2 * t0;
        const int16_t *s0 = T[d.mb][old0], *s1 = T[d.mb][old1];
        int32_t dp0 = -TP[d.mb][old0], dp1 = -TP[d.mb][old1];
        const int16_t *a0, *a1;
        if (d.decided < 0) {
            const int new0 = old0 + POW3_SQ[d.sq] * (d.stm == 0 ? 1 : 2);  // the new stone in player 0's view
            const int new1 = old1 + POW3_SQ[d.sq] * (d.stm == 1 ? 1 : 2);
            a0 = T[d.mb][new0];
            a1 = T[d.mb][new1];
            dp0 += TP[d.mb][new0];
            dp1 += TP[d.mb][new1];
        } else {
            const int r0 = dec_idx(d.mb, d.decided, 0), r1 = dec_idx(d.mb, d.decided, 1);
            a0 = DEC[r0];
            a1 = DEC[r1];
            dp0 += DECP[r0];
            dp1 += DECP[r1];
        }
        sub_add2(par.v[0], ch.v[0], a0, s0, par.v[1], ch.v[1], a1, s1);
        ch.ps[0] = par.ps[0] + dp0;
        ch.ps[1] = par.ps[1] + dp1;
    }

    // Make acc[b.n_moves] hold b: walk back while the recorded moves lead to the position needed,
    // replay from the first entry holding a parent, else refresh from scratch.
    template <typename Board>
    void sync(const Board &b) {
        const int cur = b.n_moves;
        uint64_t need = b.tt_hash;
        int k = cur;
        for (;;) {
            if (k <= 0 || dirty[k].ckey != need) {
                refresh_root(b);
                return;
            }
            need = dirty[k].pkey;
            if (pos_key[--k] == need) break;
        }
        for (int j = k + 1; j <= cur; j++) {
            update(j);
            pos_key[j] = dirty[j].ckey;
        }
    }

    template <typename Board>
    __attribute__((always_inline)) int evaluate(const Board &b, int constraint) {
        const int cur = b.n_moves;
        if (pos_key[cur] != b.tt_hash) sync(b);
        const int stm = cur & 1;
        const Entry &e = acc[cur];
        return eval_with(b, stm, constraint, e.v[stm], e.v[stm ^ 1], e.ps[stm] - e.ps[stm ^ 1]);
    }

    template <typename Board>
    __attribute__((always_inline)) int evaluate_keyed(const Board &b, int constraint, uint64_t key) {
        CacheEntry &ce = cache[key & CMASK];
        if (key && ce.key == key) return ce.eval;
        const int e = evaluate(b, constraint);
        ce.key = key;
        ce.eval = e;
        return e;
    }
};

}  // namespace b64
