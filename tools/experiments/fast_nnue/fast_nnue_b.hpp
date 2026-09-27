#pragma once
// Fast incremental integer inference for the pattern-generator ("B") nets (nnue2 stage 2).
//
// Net (BGN1 file, tools/experiments/fast_nnue/export_bgn.py; tools/experiments/nnue2/gen_nnue.py Gen),
// per perspective P, W = A + 1 wide rows (A accumulator lanes + 1 PSQT lane):
//   live miniboard m      T[m][p_P(m)]         p = sum_i cell_i 3^i, 0 empty / 1 P's stone / 2 the other's
//   decided miniboard m   dec[3m + s]          s: 0 won by P, 1 won by the other, 2 drawn
//   constraint c          con[c] if P is the side to move, else con[10 + c]      (evaluation time)
//   forced board (c < 9)  F[p_P(c)]            the forced board's pattern in P's view (evaluation time)
//   acc_P = bias + the rows; h = [crelu(acc_stm[:A]), crelu(acc_ntm[:A])] (2A)
//   -> L1 -> crelu -> L2 -> crelu -> 1 (or -> L1 -> crelu -> 1), eval = 1000 * (dense + (ps_stm - ps_ntm) / 2)
//   with ps_P = acc_P[A], truncated toward zero (stm-relative eval units).
//
// Integer scheme (powers of two, chosen at load so that nothing can overflow):
//   lanes         int16 rows round(x * 2^qa); int16 accumulators. qa is the largest value whose rigorous
//                 per-lane range fits int16: bias + per board the max/min over its 11,093 live pattern
//                 rows or its decided rows (exact per board: rounding is monotonic, so the extreme of the
//                 rounded rows is the rounded extreme), for the stored accumulator, and at evaluation
//                 time, per side: that plus con[9] (free move), or (c < 9) that with board c's term
//                 replaced by the joint max/min over live patterns p of T[c][p] + F[p] (the forced board is
//                 live and its F row and T row come from the same pattern; exact on the rounded integer
//                 rows), plus con[c]; all of it under the stone-count balance (the stone differences of
//                 the 9 boards, decided boards' hidden stones included, sum to -1..1: a dynamic program by
//                 partial sum, see quantize()). -DFASTNNUE_JOINT_ONLY drops the balance, and
//                 -DFASTNNUE_INDEP_BOUND restores stage 2's independent bound (board c's own extreme plus
//                 F's). crelu = clamp(acc + con + F, 0, 2^qa). Rows of patterns a live board cannot have (a
//                 line, or full) are stored as 0.
//   PSQT lane     int16 rows round(x * 2^qps) in separate arrays (TP, FP, ...; 354 KB for T), int32
//                 accumulators; qps is the largest (<= 20) with every used row in int16.
//   dense 2A->L1  int16 weights, int32 sums over nonzero activation pairs (_mm256_madd_epi16), as the
//                 per-cell FNN1 path; L1 crelu (clamp(s, 0, 2^(qa+qb)) + 2^(shift-1)) >> shift (round to
//                 nearest; -DFASTNNUE_FLOOR_SHIFTS: floor) to [0, 2^14] (int16, when a second hidden layer
//                 follows) or [0, 2^15]. The L2 crelu shift rounds the same way.
//   L1 -> L2      int16 weights W2q = round(w * 2^q2), the 16 h1 values packed into 8 int16 pairs,
//                 _mm256_madd_epi16 per pair; q2 the largest with 2^14 * sum|W2q| + |B2q| < 2^31.
//   output        int32 weights (mullo) as FNN1; out at 2^out_shift, PSQT difference at 2^(qps+1);
//                 eval = trunc(1000 * (out * 2^(S-out_shift) + psd * 2^(S-qps-1)) / 2^S) in int64.
//
// Incremental update (lazy, the stage-1 stack with stage 3's position keys: an entry is reused only
// for the position whose tt_hash it carries, see fast_nnue.hpp): make records the miniboard's markers
// before the move and the parent's and child's keys;
// every update is exactly one row out and one row in per perspective: T[m][old pattern] out, and
// T[m][new pattern] in, or dec[3m + s_P] in when the move decides the board. The PSQT lane follows in
// int32. The constraint row and the forced-board row are added in the evaluation transform (they
// change on almost every move; the forced board's pattern is read from the board there).
//
// Tables: T is 9 x 3^9 rows of A int16 (A=128: 45 MB, A=64: 23 MB), F 3^9 rows. Both are read at
// random, but a search touches few of them (about 1,200 distinct rows per engine and search: ~150 KB
// at A=64, ~300 KB at A=128, so they stay in L2/L3). -DFASTNNUE_B_PREFETCH=1 (default) prefetches at
// make the rows the child's update and evaluation will read (T new-pattern rows, F rows of the child's
// forced board), =2 also the rows the update subtracts, =0 none. Measured: neutral single-threaded,
// +1.4% for A=128 with 14 search threads sharing the L3, neutral at 6.
// -DFASTNNUE_B_COMPACT stores only the 11,093 live patterns per board (44% fewer rows, behind a 39 KB
// index table; measured slower).
//
// -DFASTNNUE_CHECK: at every evaluation the incremental accumulators (lanes and PSQT) must equal a
// scalar int32 from-scratch sum and the AVX2 head must equal the scalar reference (fast_nnue.hpp's
// CheckStats counters); it also records the distinct table rows each engine touches.
#include "fast_nnue.hpp"

#include <chrono>
#include <memory>
#include <string>

#ifndef FASTNNUE_B_PREFETCH
#define FASTNNUE_B_PREFETCH 1  // neutral at 1 thread, +1.4% for A=128 at 14 threads (bench_mt_b.log)
#endif

namespace fnnue {
namespace bnn {

constexpr int NPAT = 19683;
constexpr int NLIVE = 11093;  // live patterns: no line for either side, not full

struct Tern {  // t[mask] = sum over the mask's squares i of 3^i
    uint16_t t[512];
    Tern() {
        for (int m = 0; m < 512; m++) {
            int s = 0;
            for (int k = 0, p = 1; k < 9; k++, p *= 3)
                if (m >> k & 1) s += p;
            t[m] = (uint16_t)s;
        }
    }
};
inline const Tern g_tern;
// Pattern index of a miniboard from the view of the player owning `mine`.
inline int pat_of(int mine, int theirs) { return g_tern.t[mine] + 2 * g_tern.t[theirs]; }
inline int dec_idx(int mb, int state, int P) { return mb * 3 + (state == 2 ? 2 : (state == P ? 0 : 1)); }

// Live pattern -> compact row (FASTNNUE_B_COMPACT); -1 for patterns no live board can have.
struct LiveIndex {
    int16_t idx[NPAT];
    LiveIndex() {
        const std::vector<uint8_t> &live = live_patterns();
        int k = 0;
        for (int p = 0; p < NPAT; p++) idx[p] = live[p] ? (int16_t)k++ : (int16_t)-1;
    }
};
inline const LiveIndex &live_index() {
    static const LiveIndex li;
    return li;
}
#ifdef FASTNNUE_B_COMPACT
constexpr int TROWS = NLIVE;
inline int row_of(int pat) { return live_index().idx[pat]; }
#else
constexpr int TROWS = NPAT;
inline int row_of(int pat) { return pat; }
#endif

struct Header {
    int A, L1, L2, forced, ncon, has_gen, E, n_enc;
};

inline bool read_header(const char *path, Header &h) {
    FILE *f = path ? std::fopen(path, "rb") : nullptr;
    if (!f) return false;
    char magic[4];
    const bool ok = std::fread(magic, 1, 4, f) == 4 && std::memcmp(magic, "BGN1", 4) == 0
                    && std::fread(&h, sizeof(int), 8, f) == 8;
    std::fclose(f);
    return ok;
}

template <int A_, int L1_, int L2_>
struct Net {
    static constexpr int A = A_, L1 = L1_, L2 = L2_, IN = 2 * A_, NPAIR = A_;
    static constexpr int LO = L2_ ? L2_ : L1_;  // width feeding the output layer
    static constexpr int L2S = L2_ ? L2_ : 8;   // storage (no zero-length arrays)
    static constexpr int NH1P = L2_ ? L1_ / 2 : 1;
    static_assert(A_ % 64 == 0 && L1_ % 16 == 0 && LO % 8 == 0 && L2S % 8 == 0, "unsupported widths");
    int16_t *T = nullptr;   // [9][TROWS][A]
    int16_t *TP = nullptr;  // [9][TROWS] PSQT lane
    int16_t *F = nullptr;   // [TROWS][A] (forced), else nullptr
    int16_t *FP = nullptr;  // [TROWS]
    alignas(64) int16_t DEC[27][A];
    alignas(64) int16_t CON[20][A];  // [c] side to move, [10 + c] the other (shared: duplicated)
    alignas(64) int16_t BIAS[A];
    alignas(64) int16_t ZERO[A];     // forced-board row of a free move / a net without one
    int16_t DECP[27], CONP[20], BIASP;
    bool forced = false;
    alignas(64) int32_t W1p[NPAIR][L1];  // pair p, output k: lo16 = W1q[k][2p], hi16 = W1q[k][2p+1]
    alignas(64) int16_t W1[L1][IN];
    alignas(64) int32_t B1[L1];
    alignas(64) int32_t W2p[NH1P][L2S];  // h1 pair e, output k: lo16 = W2q[k][2e], hi16 = W2q[k][2e+1]
    alignas(64) int16_t W2[L2S][L1];
    alignas(64) int32_t B2[L2S];
    alignas(64) int32_t WO[LO];
    int32_t BO = 0;
    int qa = 0, qps = 0, qb = 0, q2 = 0, qo = 0, h1bits = 0, h2bits = 0;
    int16_t QA = 0;
    int32_t H1MAX = 0, H2MAX = 0;
    int h1_shift = 0, h2_shift = 0, out_shift = 0, fin_shift = 0;
    int32_t h1_round = 0, h2_round = 0;  // added before the h1 / h2 shifts (round_half)
    int64_t out_mul = 1, ps_mul = 1;  // 2^(S - out_shift), 2^(S - qps - 1)

    const int16_t *trow(int m, int pat) const { return T + ((size_t)m * TROWS + row_of(pat)) * A; }
    int32_t tps(int m, int pat) const { return TP[(size_t)m * TROWS + row_of(pat)]; }
    const int16_t *frow(int pat) const { return F + (size_t)row_of(pat) * A; }
    int32_t fps(int pat) const { return FP[row_of(pat)]; }
};

template <class N>
inline N g_bnet;

// ---------------------------------------------------------------- the generator (bake at startup)

struct Generator {  // export_bgn.py's generator section
    int E = 0;
    std::vector<int> dims;                        // 27, 64, 64, 32
    std::vector<std::vector<float>> Wl, bl;       // per encoder layer: W[out][in], b[out]
    std::vector<float> proj_w, proj_b, fwd_w, fwd_b;
};

// T[m][p] = enc(onehot(p)) @ proj_w[m] + proj_b[m] and F[p] = enc(onehot(p)) @ fwd_w + fwd_b in float,
// single thread, live patterns only (the rest stay 0). What a CodinGame build would run at startup
// instead of reading the 23-46 MB tables. T: [9][NPAT][W], F: [NPAT][W].
inline void bake(const Generator &g, int W, bool forced, float *T, float *F) {
    const std::vector<uint8_t> &live = live_patterns();
    const int E = g.E, nl = (int)g.Wl.size();
    std::vector<float> emb((size_t)NPAT * E, 0.f);
    std::vector<float> h(256), h2(256);
    for (int p = 0; p < NPAT; p++) {
        if (!live[p]) continue;
        int d[9];
        for (int k = 0, x = p; k < 9; k++, x /= 3) d[k] = x % 3;
        const int w0 = g.dims[1];
        for (int o = 0; o < w0; o++) {  // first layer on a one-hot: 9 columns
            const float *wr = g.Wl[0].data() + (size_t)o * 27;
            float s = g.bl[0][o];
            for (int k = 0; k < 9; k++) s += wr[3 * k + d[k]];
            h[o] = s;
        }
        int cur = w0;
        for (int l = 1; l < nl; l++) {
            for (int i = 0; i < cur; i++) h[i] = std::max(h[i], 0.f);  // ReLU between layers
            const int out = g.dims[l + 1];
            for (int o = 0; o < out; o++) {
                const float *wr = g.Wl[l].data() + (size_t)o * cur;
                float s = g.bl[l][o];
                for (int i = 0; i < cur; i++) s += wr[i] * h[i];
                h2[o] = s;
            }
            std::swap(h, h2);
            cur = out;
        }
        std::memcpy(emb.data() + (size_t)p * E, h.data(), sizeof(float) * E);
    }
    std::vector<float> row(W);
    auto project = [&](const float *pw, const float *pb, const float *e, float *dst) {
        for (int j = 0; j < W; j++) row[j] = pb[j];
        for (int k = 0; k < E; k++) {
            const float ek = e[k];
            const float *w = pw + (size_t)k * W;
            for (int j = 0; j < W; j++) row[j] += ek * w[j];
        }
        std::memcpy(dst, row.data(), sizeof(float) * W);
    };
    for (int m = 0; m < 9; m++)
        for (int p = 0; p < NPAT; p++) {
            float *dst = T + ((size_t)m * NPAT + p) * W;
            if (!live[p]) { std::memset(dst, 0, sizeof(float) * W); continue; }
            project(g.proj_w.data() + (size_t)m * E * W, g.proj_b.data() + (size_t)m * W, emb.data() + (size_t)p * E, dst);
        }
    if (forced)
        for (int p = 0; p < NPAT; p++) {
            float *dst = F + (size_t)p * W;
            if (!live[p]) { std::memset(dst, 0, sizeof(float) * W); continue; }
            project(g.fwd_w.data(), g.fwd_b.data(), emb.data() + (size_t)p * E, dst);
        }
}

// ---------------------------------------------------------------- loading / quantization

struct FloatNet {
    Header h{};
    std::vector<float> bias, T, dec, con, F, W1, b1, W2, b2, WO;
    float bo = 0;
    Generator gen;
};

inline bool read_floats(FILE *f, std::vector<float> &v, size_t n) {
    v.resize(n);
    return n == 0 || std::fread(v.data(), 4, n, f) == n;
}

inline bool read_bgn(const char *path, FloatNet &fn) {
    FILE *f = std::fopen(path, "rb");
    if (!f) return false;
    char magic[4];
    Header &h = fn.h;
    bool ok = std::fread(magic, 1, 4, f) == 4 && std::memcmp(magic, "BGN1", 4) == 0 && std::fread(&h, 4, 8, f) == 8;
    const size_t W = (size_t)h.A + 1;
    const int LO = h.L2 ? h.L2 : h.L1;
    ok = ok && read_floats(f, fn.bias, W) && read_floats(f, fn.T, (size_t)9 * NPAT * W)
         && read_floats(f, fn.dec, 27 * W) && read_floats(f, fn.con, (size_t)h.ncon * W)
         && read_floats(f, fn.F, h.forced ? (size_t)NPAT * W : 0)
         && read_floats(f, fn.W1, (size_t)h.L1 * 2 * h.A) && read_floats(f, fn.b1, h.L1)
         && read_floats(f, fn.W2, (size_t)h.L2 * h.L1) && read_floats(f, fn.b2, h.L2)
         && read_floats(f, fn.WO, LO) && std::fread(&fn.bo, 4, 1, f) == 1;
    if (ok && h.has_gen) {
        Generator &g = fn.gen;
        g.E = h.E;
        g.dims.resize(h.n_enc + 1);
        ok = std::fread(g.dims.data(), 4, g.dims.size(), f) == g.dims.size();
        g.Wl.resize(h.n_enc);
        g.bl.resize(h.n_enc);
        for (int l = 0; ok && l < h.n_enc; l++)
            ok = read_floats(f, g.Wl[l], (size_t)g.dims[l + 1] * g.dims[l]) && read_floats(f, g.bl[l], g.dims[l + 1]);
        ok = ok && read_floats(f, g.proj_w, (size_t)9 * h.E * W) && read_floats(f, g.proj_b, 9 * W)
             && read_floats(f, g.fwd_w, h.forced ? (size_t)h.E * W : 0) && read_floats(f, g.fwd_b, h.forced ? W : 0);
    }
    std::fclose(f);
    return ok;
}

// Stone differences d = #mine - #theirs (-9..9, stored as d + 9) by pattern, and the ones each decided
// state allows (a superset: any pattern with a line for the winner and none for the loser, or full
// with no line when drawn). Used by the stone-balance bound in quantize().
struct StoneSets {
    static constexpr int ND = 19;
    uint8_t di[NPAT];      // d + 9 of each pattern (digit 1 = mine, 2 = theirs)
    uint32_t dec_mask[3];  // bit d + 9: won by mine, won by theirs, drawn
    StoneSets() {
        static const int LINES[8] = {7, 56, 448, 73, 146, 292, 273, 84};
        dec_mask[0] = dec_mask[1] = dec_mask[2] = 0;
        for (int p = 0; p < NPAT; p++) {
            int m[3] = {0, 0, 0};
            for (int k = 0, x = p; k < 9; k++, x /= 3) m[x % 3] |= 1 << k;
            const int d = __builtin_popcount(m[1]) - __builtin_popcount(m[2]);
            di[p] = (uint8_t)(d + 9);
            bool l1 = false, l2 = false;
            for (int l : LINES) {
                l1 |= (m[1] & l) == l;
                l2 |= (m[2] & l) == l;
            }
            if (l1 && !l2) dec_mask[0] |= 1u << (d + 9);
            if (l2 && !l1) dec_mask[1] |= 1u << (d + 9);
            if (m[0] == 0 && !l1 && !l2) dec_mask[2] |= 1u << (d + 9);
        }
    }
};
inline const StoneSets &stone_sets() {
    static const StoneSets s;
    return s;
}

template <class N>
inline bool quantize(N &n, const FloatNet &fn, const char *path) {
    constexpr int A = N::A, L1 = N::L1, L2 = N::L2, IN = N::IN, LO = N::LO;
    const int W = A + 1;
    const Header &h = fn.h;
    const bool forced = h.forced != 0;
    const std::vector<uint8_t> &live = live_patterns();
    // round half to even of x * 2^bits (exact scaling; nearbyint is one instruction, llround + ldexp
    // are library calls: the table fill is ~23M values). Monotonic, so the extreme argument holds.
    auto q = [](double x, int bits) { return (long long)std::nearbyint(x * (double)(1LL << bits)); };
    auto con_row = [&](int side, int c) { return fn.con.data() + (size_t)((h.ncon == 20 ? 10 * side : 0) + c) * W; };

    // Per board and lane, the float extremes over live patterns (rounding is monotonic, so the
    // extremes of the rounded rows are the rounded extremes); the same for F. By stone difference
    // d = #mine - #theirs of the pattern (index d + 9): tdh / tdl, the extremes of T[m][p], and jdh /
    // jdl, those of the forced-board term T[m][p] + F[p] (F = 0 without forced rows), whose float
    // extremes bracket the exact integer ones within +-1.
    const StoneSets &ss = stone_sets();
    constexpr int ND = StoneSets::ND;
    std::vector<double> tmax((size_t)9 * A, -1e300), tmin((size_t)9 * A, 1e300), fmax(A, 0.0), fmin(A, 0.0);
    std::vector<double> tdh((size_t)9 * ND * A, -1e300), tdl((size_t)9 * ND * A, 1e300);
    std::vector<double> jdh((size_t)9 * ND * A, -1e300), jdl((size_t)9 * ND * A, 1e300);
    double ps_abs = std::fabs(fn.bias[A]);
    for (int m = 0; m < 9; m++)
        for (int p = 0; p < NPAT; p++) {
            if (!live[p]) continue;
            const float *r = fn.T.data() + ((size_t)m * NPAT + p) * W;
            const float *fr = forced ? fn.F.data() + (size_t)p * W : nullptr;
            const size_t o = ((size_t)m * ND + ss.di[p]) * A;
            for (int i = 0; i < A; i++) {
                tmax[m * A + i] = std::max(tmax[m * A + i], (double)r[i]);
                tmin[m * A + i] = std::min(tmin[m * A + i], (double)r[i]);
                tdh[o + i] = std::max(tdh[o + i], (double)r[i]);
                tdl[o + i] = std::min(tdl[o + i], (double)r[i]);
                const double j = (double)r[i] + (fr ? (double)fr[i] : 0.0);
                jdh[o + i] = std::max(jdh[o + i], j);
                jdl[o + i] = std::min(jdl[o + i], j);
            }
            ps_abs = std::max(ps_abs, std::fabs((double)r[A]));
        }
    if (forced) {
        for (int i = 0; i < A; i++) { fmax[i] = -1e300; fmin[i] = 1e300; }
        for (int p = 0; p < NPAT; p++) {
            if (!live[p]) continue;
            const float *r = fn.F.data() + (size_t)p * W;
            for (int i = 0; i < A; i++) {
                fmax[i] = std::max(fmax[i], (double)r[i]);
                fmin[i] = std::min(fmin[i], (double)r[i]);
            }
            ps_abs = std::max(ps_abs, std::fabs((double)r[A]));
        }
    }
    for (int r = 0; r < 27; r++) ps_abs = std::max(ps_abs, std::fabs((double)fn.dec[(size_t)r * W + A]));
    for (int r = 0; r < h.ncon; r++) ps_abs = std::max(ps_abs, std::fabs((double)fn.con[(size_t)r * W + A]));

    // qa: the largest with every used row, the stored accumulator and acc + con + F in int16, over
    // every position the features can express. Three bounds (bound_mode):
    //   0 (-DFASTNNUE_INDEP_BOUND, stage 2): per-board extremes added independently, plus F's extreme.
    //   1 (-DFASTNNUE_JOINT_ONLY): the forced board c is live and its F row and T row come from the
    //     same pattern, so board c contributes max over live p of T[c][p] + F[p] (exact on integers).
    //   2 (default): 1, and the stone counts balance. Without passes, player 0 has as many stones as
    //     player 1 (player 0 to move) or one more, so the stone difference d = #mine - #theirs summed
    //     over the 9 boards (live boards' patterns and decided boards' hidden stones alike) is -1 or 0
    //     from the side to move's view, 0 or +1 from the other's, and -1..1 for a stored accumulator.
    //     A dynamic program over the boards, by partial sum of d, gives the extremes under that
    //     constraint; a decided board may have any d its state allows (StoneSets: any pattern with a
    //     line for the winner and none for the loser, or full and lineless when drawn: a superset).
    //     Exact on the rounded integer rows. The -DFASTNNUE_CHECK builds verify the premises (a forced
    //     board is live, the stones balance) and the int16 range at every evaluation.
#if defined(FASTNNUE_INDEP_BOUND)
    constexpr int bound_mode = 0;
#elif defined(FASTNNUE_JOINT_ONLY)
    constexpr int bound_mode = 1;
#else
    constexpr int bound_mode = 2;
#endif
    constexpr bool use_j = bound_mode >= 1, balance = bound_mode == 2;
    constexpr long long NEG = std::numeric_limits<long long>::min() / 4, POS = std::numeric_limits<long long>::max() / 4;
    constexpr int NS = 9 * (ND - 1) + 1, OFF = NS / 2;  // partial sums of d: -81..81
    // Exact integer extremes of q(T[m][p]) + q(F[p]) over live p, by board, d and lane, at `bits`.
    std::vector<long long> jqh((size_t)9 * ND * A), jql((size_t)9 * ND * A);
    auto exact_joint = [&](int bits) {
        std::fill(jqh.begin(), jqh.end(), NEG);
        std::fill(jql.begin(), jql.end(), POS);
        std::vector<long long> qf(A, 0);
        for (int p = 0; p < NPAT; p++) {
            if (!live[p]) continue;
            if (forced)
                for (int i = 0; i < A; i++) qf[i] = q(fn.F[(size_t)p * W + i], bits);
            for (int m = 0; m < 9; m++) {
                const float *r = fn.T.data() + ((size_t)m * NPAT + p) * W;
                const size_t o = ((size_t)m * ND + ss.di[p]) * A;
                for (int i = 0; i < A; i++) {
                    const long long v = q(r[i], bits) + qf[i];
                    jqh[o + i] = std::max(jqh[o + i], v);
                    jql[o + i] = std::min(jql[o + i], v);
                }
            }
        }
    };
    // Allowed totals of d: stored accumulator, side to move, the other side (bit t + 1 for t = -1..1).
    const int allow_stored = 7, allow_side[2] = {3, 6};
    auto best_total = [&](const long long *v, int allow, bool hi) {  // v indexed by total + OFF
        long long b = hi ? NEG : POS;
        if (!balance) {
            for (int s = 0; s < NS; s++) b = hi ? std::max(b, v[s]) : std::min(b, v[s]);
        } else {
            for (int t = -1; t <= 1; t++)
                if (allow >> (t + 1) & 1) b = hi ? std::max(b, v[OFF + t]) : std::min(b, v[OFF + t]);
        }
        return b;
    };
    // Best total over the allowed ones of x (boards <= c, by partial d) plus u (boards > c).
    auto combine = [&](const long long *x, const long long *u, int allow, bool hi) {
        long long b = hi ? NEG : POS;
        if (!balance) {
            long long bx = hi ? NEG : POS, bu = hi ? NEG : POS;
            for (int s = 0; s < NS; s++) {
                bx = hi ? std::max(bx, x[s]) : std::min(bx, x[s]);
                bu = hi ? std::max(bu, u[s]) : std::min(bu, u[s]);
            }
            return (hi ? (bx > NEG && bu > NEG) : (bx < POS && bu < POS)) ? bx + bu : b;
        }
        for (int t = -1; t <= 1; t++) {
            if (!(allow >> (t + 1) & 1)) continue;
            for (int s = 0; s < NS; s++) {
                const int w = 2 * OFF + t - s;  // u's index: its partial total is t - (s - OFF)
                if (w < 0 || w >= NS) continue;
                if (hi ? (x[s] > NEG && u[w] > NEG) : (x[s] < POS && u[w] < POS))
                    b = hi ? std::max(b, x[s] + u[w]) : std::min(b, x[s] + u[w]);
            }
        }
        return b;
    };
    // pre[k] / suf[k]: best sum of boards < k / >= k by partial d total, for the max (hi) and min (lo).
    std::vector<long long> preh((size_t)10 * NS), prel((size_t)10 * NS), sufh((size_t)10 * NS), sufl((size_t)10 * NS);
    std::vector<long long> xh(NS), xl(NS);
    int qa = 14, exact_passes = 0;
    long long b_lo = 0, b_hi = 0, e_lo = 0, e_hi = 0, max_row = 0;
    for (; qa >= 1; qa--) {
        bool ok = true;
        long long slo = 0, shi = 0, flo = 0, fhi = 0, mr = 0;
        const double sc = (double)(1LL << qa);
        // pass 0: the joint terms at their smallest possible magnitude given the float extremes (if
        // that does not fit, the exact one cannot); pass 1: the exact integer joint terms.
        for (int pass = 0; pass < (use_j ? 2 : 1) && ok; pass++) {
            if (pass == 1) { exact_joint(qa); exact_passes++; }
            slo = shi = flo = fhi = mr = 0;
            for (int i = 0; i < A && ok; i++) {
                const long long bias = q(fn.bias[i], qa);
                mr = std::max(mr, std::llabs(bias));
                // board options by d: a live pattern, or a decided row with any d its state allows
                long long oh[9][ND], ol[9][ND];
                for (int m = 0; m < 9; m++) {
                    mr = std::max({mr, std::llabs(q(tmax[m * A + i], qa)), std::llabs(q(tmin[m * A + i], qa))});
                    for (int di = 0; di < ND; di++) {
                        const size_t o = ((size_t)m * ND + di) * A + i;
                        const bool any = tdh[o] > -1e299;
                        oh[m][di] = any ? q(tdh[o], qa) : NEG;
                        ol[m][di] = any ? q(tdl[o], qa) : POS;
                    }
                    for (int s = 0; s < 3; s++) {
                        const long long d = q(fn.dec[(size_t)(3 * m + s) * W + i], qa);
                        mr = std::max(mr, std::llabs(d));
                        for (int di = 0; di < ND; di++)
                            if (ss.dec_mask[s] >> di & 1) {
                                oh[m][di] = std::max(oh[m][di], d);
                                ol[m][di] = std::min(ol[m][di], d);
                            }
                    }
                }
                std::fill(preh.begin(), preh.begin() + NS, NEG);
                std::fill(prel.begin(), prel.begin() + NS, POS);
                preh[OFF] = prel[OFF] = 0;
                for (int k = 0; k < 9; k++) {
                    long long *nh = &preh[(size_t)(k + 1) * NS], *nl = &prel[(size_t)(k + 1) * NS];
                    const long long *ch = &preh[(size_t)k * NS], *cl = &prel[(size_t)k * NS];
                    std::fill(nh, nh + NS, NEG);
                    std::fill(nl, nl + NS, POS);
                    for (int s = 0; s < NS; s++)
                        for (int di = 0; di < ND; di++) {
                            const int t = s + di - (ND / 2);
                            if (t < 0 || t >= NS) continue;
                            if (ch[s] > NEG && oh[k][di] > NEG) nh[t] = std::max(nh[t], ch[s] + oh[k][di]);
                            if (cl[s] < POS && ol[k][di] < POS) nl[t] = std::min(nl[t], cl[s] + ol[k][di]);
                        }
                }
                std::fill(sufh.begin() + (size_t)9 * NS, sufh.end(), NEG);
                std::fill(sufl.begin() + (size_t)9 * NS, sufl.end(), POS);
                sufh[(size_t)9 * NS + OFF] = sufl[(size_t)9 * NS + OFF] = 0;
                for (int k = 8; k >= 0; k--) {
                    long long *nh = &sufh[(size_t)k * NS], *nl = &sufl[(size_t)k * NS];
                    const long long *ch = &sufh[(size_t)(k + 1) * NS], *cl = &sufl[(size_t)(k + 1) * NS];
                    std::fill(nh, nh + NS, NEG);
                    std::fill(nl, nl + NS, POS);
                    for (int s = 0; s < NS; s++)
                        for (int di = 0; di < ND; di++) {
                            const int t = s + di - (ND / 2);
                            if (t < 0 || t >= NS) continue;
                            if (ch[s] > NEG && oh[k][di] > NEG) nh[t] = std::max(nh[t], ch[s] + oh[k][di]);
                            if (cl[s] < POS && ol[k][di] < POS) nl[t] = std::min(nl[t], cl[s] + ol[k][di]);
                        }
                }
                const long long *allh = &preh[(size_t)9 * NS], *alll = &prel[(size_t)9 * NS];
                shi = std::max(shi, bias + best_total(allh, allow_stored, true));
                slo = std::min(slo, bias + best_total(alll, allow_stored, false));
                const long long fh = forced ? q(fmax[i], qa) : 0, fl = forced ? q(fmin[i], qa) : 0;
                mr = std::max({mr, std::llabs(fh), std::llabs(fl)});
                long long eh[2], el[2];
                for (int side = 0; side < 2; side++) {
                    const long long v9 = q(con_row(side, 9)[i], qa);
                    mr = std::max(mr, std::llabs(v9));
                    eh[side] = bias + best_total(allh, allow_side[side], true) + v9;
                    el[side] = bias + best_total(alll, allow_side[side], false) + v9;
                }
                for (int c = 0; c < 9; c++) {
                    const long long v[2] = {q(con_row(0, c)[i], qa), q(con_row(1, c)[i], qa)};
                    mr = std::max({mr, std::llabs(v[0]), std::llabs(v[1])});
                    if (!use_j) {
                        for (int side = 0; side < 2; side++) {
                            eh[side] = std::max(eh[side], bias + best_total(allh, allow_side[side], true) + v[side] + fh);
                            el[side] = std::min(el[side], bias + best_total(alll, allow_side[side], false) + v[side] + fl);
                        }
                        continue;
                    }
                    // boards < c, then board c's forced-board options (x), then boards > c (u)
                    long long jh[ND], jl[ND];
                    for (int di = 0; di < ND; di++) {
                        const size_t o = ((size_t)c * ND + di) * A + i;
                        const bool any = jdh[o] > -1e299;
                        jh[di] = !any ? NEG : pass ? jqh[o] : (long long)std::ceil(jdh[o] * sc - 1.0 - 1e-9);
                        jl[di] = !any ? POS : pass ? jql[o] : (long long)std::floor(jdl[o] * sc + 1.0 + 1e-9);
                    }
                    const long long *ph = &preh[(size_t)c * NS], *pl = &prel[(size_t)c * NS];
                    std::fill(xh.begin(), xh.end(), NEG);
                    std::fill(xl.begin(), xl.end(), POS);
                    for (int s = 0; s < NS; s++)
                        for (int di = 0; di < ND; di++) {
                            const int t = s + di - (ND / 2);
                            if (t < 0 || t >= NS) continue;
                            if (ph[s] > NEG && jh[di] > NEG) xh[t] = std::max(xh[t], ph[s] + jh[di]);
                            if (pl[s] < POS && jl[di] < POS) xl[t] = std::min(xl[t], pl[s] + jl[di]);
                        }
                    const long long *uh = &sufh[(size_t)(c + 1) * NS], *ul = &sufl[(size_t)(c + 1) * NS];
                    for (int side = 0; side < 2; side++) {
                        eh[side] = std::max(eh[side], bias + combine(xh.data(), uh, allow_side[side], true) + v[side]);
                        el[side] = std::min(el[side], bias + combine(xl.data(), ul, allow_side[side], false) + v[side]);
                    }
                }
                fhi = std::max({fhi, eh[0], eh[1]});
                flo = std::min({flo, el[0], el[1]});
                ok = mr <= 32767 && shi <= 32767 && slo >= -32768 && fhi <= 32767 && flo >= -32768;
            }
        }
        if (ok) { b_lo = slo; b_hi = shi; e_lo = flo; e_hi = fhi; max_row = mr; break; }
    }
    if (qa < 1) { std::fprintf(stderr, "fast_nnue_b: first layer does not fit int16\n"); return false; }
    int qps = 20;
    while (qps > 0 && std::ldexp(ps_abs, qps) >= 32767.0) qps--;
    n.qa = qa;
    n.qps = qps;
    n.QA = (int16_t)(1 << qa);
    n.forced = forced;

    // Tables.
    const size_t trows = (size_t)9 * TROWS, frows = TROWS;
    n.T = (int16_t *)_mm_malloc(trows * A * sizeof(int16_t), 64);
    n.TP = (int16_t *)_mm_malloc(trows * sizeof(int16_t), 64);
    if (!n.T || !n.TP) { std::fprintf(stderr, "fast_nnue_b: out of memory\n"); return false; }
    long long dead_rows = 0;
    auto put = [&](const float *src, int16_t *dst, int16_t *dps) {
        for (int i = 0; i < A; i++) dst[i] = (int16_t)q(src[i], qa);
        *dps = (int16_t)q(src[A], qps);
    };
    for (int m = 0; m < 9; m++)
        for (int p = 0; p < NPAT; p++) {
#ifdef FASTNNUE_B_COMPACT
            if (!live[p]) { dead_rows++; continue; }
#else
            if (!live[p]) {  // never read: no live board has this pattern
                std::memset(n.T + ((size_t)m * TROWS + p) * A, 0, sizeof(int16_t) * A);
                n.TP[(size_t)m * TROWS + p] = 0;
                dead_rows++;
                continue;
            }
#endif
            const size_t r = (size_t)m * TROWS + row_of(p);
            put(fn.T.data() + ((size_t)m * NPAT + p) * W, n.T + r * A, n.TP + r);
        }
    if (forced) {
        n.F = (int16_t *)_mm_malloc(frows * A * sizeof(int16_t), 64);
        n.FP = (int16_t *)_mm_malloc(frows * sizeof(int16_t), 64);
        if (!n.F || !n.FP) { std::fprintf(stderr, "fast_nnue_b: out of memory\n"); return false; }
        for (int p = 0; p < NPAT; p++) {
            if (!live[p]) {
#ifndef FASTNNUE_B_COMPACT
                std::memset(n.F + (size_t)p * A, 0, sizeof(int16_t) * A);
                n.FP[p] = 0;
#endif
                continue;
            }
            put(fn.F.data() + (size_t)p * W, n.F + (size_t)row_of(p) * A, n.FP + row_of(p));
        }
    }
    for (int r = 0; r < 27; r++) put(fn.dec.data() + (size_t)r * W, n.DEC[r], &n.DECP[r]);
    for (int side = 0; side < 2; side++)
        for (int c = 0; c < 10; c++) put(con_row(side, c), n.CON[10 * side + c], &n.CONP[10 * side + c]);
    put(fn.bias.data(), n.BIAS, &n.BIASP);
    std::memset(n.ZERO, 0, sizeof(n.ZERO));

    // Dense 2A -> L1: int16 weights, int32-safe sums (activations <= 2^qa).
    int qb = -1;
    long long max_w1 = 0, max_l1 = 0;
    for (int bits = 16; bits >= 0 && qb < 0; bits--) {
        bool ok = true;
        long long mw = 0, ml = 0;
        for (int k = 0; k < L1 && ok; k++) {
            long long s = 0;
            for (int i = 0; i < IN; i++) {
                const long long v = q(fn.W1[(size_t)k * IN + i], bits);
                mw = std::max(mw, std::llabs(v));
                s += std::llabs(v);
            }
            const long long tot = s * (1LL << qa) + std::llabs(q(fn.b1[k], bits + qa));
            ml = std::max(ml, tot);
            ok = mw <= 32767 && tot < (1LL << 31) - 1;
        }
        if (ok) { qb = bits; max_w1 = mw; max_l1 = ml; }
    }
    if (qb < 0) { std::fprintf(stderr, "fast_nnue_b: dense layer does not fit\n"); return false; }
    n.qb = qb;
    n.H1MAX = (int32_t)(1LL << (qa + qb));
    n.h1bits = std::min(L2 ? 14 : 15, qa + qb);
    n.h1_shift = qa + qb - n.h1bits;
    n.h1_round = round_half(n.h1_shift);  // (clamp + 2^(shift-1)) >> shift stays <= 2^h1bits
    for (int k = 0; k < L1; k++) {
        n.B1[k] = (int32_t)q(fn.b1[k], qa + qb);
        for (int i = 0; i < IN; i++) n.W1[k][i] = (int16_t)q(fn.W1[(size_t)k * IN + i], qb);
    }
    for (int p = 0; p < N::NPAIR; p++)
        for (int k = 0; k < L1; k++)
            n.W1p[p][k] = (int32_t)(((uint32_t)(uint16_t)n.W1[k][2 * p]) | ((uint32_t)(uint16_t)n.W1[k][2 * p + 1] << 16));

    // Second hidden layer L1 -> L2 (int16 weights, madd over h1 pairs).
    long long max_l2 = 0, max_w2 = 0;
    int in_bits = n.h1bits;
    if (L2) {
        int q2 = -1;
        for (int bits = 16; bits >= 0 && q2 < 0; bits--) {
            bool ok = true;
            long long mw = 0, ml = 0;
            for (int k = 0; k < L2 && ok; k++) {
                long long s = 0;
                for (int j = 0; j < L1; j++) {
                    const long long v = q(fn.W2[(size_t)k * L1 + j], bits);
                    mw = std::max(mw, std::llabs(v));
                    s += std::llabs(v);
                }
                const long long tot = s * (1LL << n.h1bits) + std::llabs(q(fn.b2[k], bits + n.h1bits));
                ml = std::max(ml, tot);
                ok = mw <= 32767 && tot < (1LL << 31) - 1;
            }
            if (ok) { q2 = bits; max_w2 = mw; max_l2 = ml; }
        }
        if (q2 < 0) { std::fprintf(stderr, "fast_nnue_b: second hidden layer does not fit\n"); return false; }
        n.q2 = q2;
        n.H2MAX = (int32_t)(1LL << (n.h1bits + q2));
        n.h2bits = std::min(15, n.h1bits + q2);
        n.h2_shift = n.h1bits + q2 - n.h2bits;
        n.h2_round = round_half(n.h2_shift);
        for (int k = 0; k < L2; k++) {
            n.B2[k] = (int32_t)q(fn.b2[k], n.h1bits + q2);
            for (int j = 0; j < L1; j++) n.W2[k][j] = (int16_t)q(fn.W2[(size_t)k * L1 + j], q2);
        }
        for (int e = 0; e < L1 / 2; e++)
            for (int k = 0; k < L2; k++)
                n.W2p[e][k] = (int32_t)(((uint32_t)(uint16_t)n.W2[k][2 * e]) | ((uint32_t)(uint16_t)n.W2[k][2 * e + 1] << 16));
        in_bits = n.h2bits;
    }

    // Output layer.
    int qo = -1;
    long long out_bound = 0;
    for (int bits = 20; bits >= 0 && qo < 0; bits--) {
        long long s = 0;
        for (int k = 0; k < LO; k++) s += std::llabs(q(fn.WO[k], bits));
        const long long tot = s * (1LL << in_bits) + std::llabs(q(fn.bo, bits + in_bits));
        if (tot < (1LL << 31) - 1) { qo = bits; out_bound = tot; }
    }
    if (qo < 0) { std::fprintf(stderr, "fast_nnue_b: output layer does not fit\n"); return false; }
    n.qo = qo;
    for (int k = 0; k < LO; k++) n.WO[k] = (int32_t)q(fn.WO[k], qo);
    n.BO = (int32_t)q(fn.bo, qo + in_bits);
    n.out_shift = in_bits + qo;
    n.fin_shift = std::max(n.out_shift, qps + 1);
    n.out_mul = 1LL << (n.fin_shift - n.out_shift);
    n.ps_mul = 1LL << (n.fin_shift - qps - 1);
    std::fprintf(stderr,
                 "fast_nnue_b: BGN1 %s A=%d dense %d->%d%s->1 forced=%d con=%d: QA=2^%d (max|row| %lld; acc bound "
                 "[%lld, %lld], with con+F [%lld, %lld] of int16, %s), PSQT 2^%d, QB=2^%d (max|W1q| %lld, L1 sum bound "
                 "%lld of 2^31), h1 2^%d (%s), ",
                 path, A, IN, L1, L2 ? (std::string("->") + std::to_string(L2)).c_str() : "", (int)forced, h.ncon, qa,
                 max_row, b_lo, b_hi, e_lo, e_hi,
                 (std::string(bound_mode == 0   ? "independent bound"
                              : bound_mode == 1 ? "joint forced-board bound"
                                                : "joint forced-board + stone-balance bound")
                  + (use_j ? ", " + std::to_string(exact_passes) + " exact pass(es)" : std::string()))
                     .c_str(),
                 qps, qb, max_w1, max_l1, n.h1bits, n.h1_round ? "rounded" : "floored");
    if (L2)
        std::fprintf(stderr, "Q2=2^%d (max|W2q| %lld, sum bound %lld of 2^31), h2 2^%d (%s), ", n.q2, max_w2, max_l2,
                     n.h2bits, n.h2_round ? "rounded" : "floored");
    std::fprintf(stderr, "QO=2^%d (out bound %lld of 2^31), final shift %d; %lld dead-pattern rows %s; tables %.1f MB\n",
                 qo, out_bound, n.fin_shift, dead_rows,
#ifdef FASTNNUE_B_COMPACT
                 "dropped (compact)",
#else
                 "zeroed",
#endif
                 (double)((trows + (forced ? frows : 0)) * (A + 1) * sizeof(int16_t)) / 1e6);
    return true;
}

template <class N>
inline bool load(N &n, const char *path) {
    using clk = std::chrono::steady_clock;
    auto ms = [](clk::time_point a, clk::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
    const auto t0 = clk::now();
    FloatNet fn;
    if (!read_bgn(path, fn)) { std::fprintf(stderr, "fast_nnue_b: cannot read %s\n", path); return false; }
    const Header &h = fn.h;
    if (h.A != N::A || h.L1 != N::L1 || h.L2 != N::L2 || (h.ncon != 10 && h.ncon != 20)) {
        std::fprintf(stderr, "fast_nnue_b: %s: A/L1/L2 %d/%d/%d do not match the instantiation %d/%d/%d\n", path, h.A,
                     h.L1, h.L2, N::A, N::L1, N::L2);
        return false;
    }
    const auto t1 = clk::now();
    const char *bk = std::getenv("FASTNNUE_BAKE");
    if (!(bk && bk[0] == '1')) {
        if (!quantize(n, fn, path)) return false;
        std::fprintf(stderr, "fast_nnue_b: read %.0f ms, quantize %.0f ms\n", ms(t0, t1), ms(t1, clk::now()));
        return true;
    }
    // FASTNNUE_BAKE=1: re-bake T and F from the generator section in C++ (what a build without the
    // tables would do at startup), play with those, and report how far they are from the file's.
    if (!h.has_gen) { std::fprintf(stderr, "fast_nnue_b: FASTNNUE_BAKE=1 but %s has no generator\n", path); return false; }
    const int W = h.A + 1;
    const std::vector<uint8_t> &live = live_patterns();
    std::vector<float> T((size_t)9 * NPAT * W), F(h.forced ? (size_t)NPAT * W : 0);
    const auto b0 = clk::now();
    bake(fn.gen, W, h.forced != 0, T.data(), F.data());
    const auto b1 = clk::now();
    double dt = 0, df = 0;
    for (int m = 0; m < 9; m++)
        for (int p = 0; p < NPAT; p++)
            if (live[p])
                for (int j = 0; j < W; j++) {
                    const size_t k = ((size_t)m * NPAT + p) * W + j;
                    dt = std::max(dt, (double)std::fabs(T[k] - fn.T[k]));
                }
    for (size_t k = 0; k < F.size(); k++)
        if (live[k / W]) df = std::max(df, (double)std::fabs(F[k] - fn.F[k]));
    fn.T.swap(T);  // quantize the C++-baked tables; T, F now hold the file's
    fn.F.swap(F);
    const auto q0 = clk::now();
    if (!quantize(n, fn, path)) return false;
    const auto q1 = clk::now();
    long long diff = 0;  // quantized values that differ from quantizing the file's tables
    const double sa = (double)(1LL << n.qa), sp = (double)(1LL << n.qps);
    for (int m = 0; m < 9; m++)
        for (int p = 0; p < NPAT; p++)
            if (live[p]) {
                const float *r = T.data() + ((size_t)m * NPAT + p) * W;
                const int16_t *qr = n.trow(m, p);
                for (int j = 0; j < N::A; j++) diff += qr[j] != (int16_t)std::nearbyint(r[j] * sa);
                diff += n.tps(m, p) != (int16_t)std::nearbyint(r[N::A] * sp);
            }
    for (int p = 0; p < NPAT && h.forced; p++)
        if (live[p]) {
            const float *r = F.data() + (size_t)p * W;
            const int16_t *qr = n.frow(p);
            for (int j = 0; j < N::A; j++) diff += qr[j] != (int16_t)std::nearbyint(r[j] * sa);
            diff += n.fps(p) != (int16_t)std::nearbyint(r[N::A] * sp);
        }
    std::fprintf(stderr, "fast_nnue_b: read %.0f ms, bake (C++, 1 thread, live patterns) %.0f ms [max |T - file T| %.1e, "
                 "|F - file F| %.1e; %lld of %lld quantized values differ from the file's], quantize %.0f ms\n",
                 ms(t0, t1), ms(b0, b1), dt, df, diff, (long long)(9 + (h.forced ? 1 : 0)) * NLIVE * W, ms(q0, q1));
    return true;
}

// ---------------------------------------------------------------- kernels

// child = parent + add - sub for both perspectives (every B update is one row out, one in).
template <int A>
inline void sub_add2(const int16_t *p0, int16_t *c0, const int16_t *a0, const int16_t *s0,
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

template <class N>
inline int finish(const N &n, int64_t out, int32_t psd) {
    const int64_t x = 1000 * (out * n.out_mul + (int64_t)psd * n.ps_mul);
    return (int)(x >= 0 ? (x >> n.fin_shift) : -((-x) >> n.fin_shift));
}

// Output layer on NV vectors of 8 int32 pre-activations: crelu, rounded shift, dot with WO, + BO.
template <class N, int NV>
inline int64_t output_layer(const N &n, const __m256i *x, int32_t hmax_v, int32_t round, int shift) {
    const __m256i zero = _mm256_setzero_si256();
    const __m256i hmax = _mm256_set1_epi32(hmax_v);
    const __m256i half = _mm256_set1_epi32(round);
    const __m128i sh = _mm_cvtsi32_si128(shift);
    __m256i o = _mm256_setzero_si256();
    for (int r = 0; r < NV; r++) {
        const __m256i a =
            _mm256_sra_epi32(_mm256_add_epi32(_mm256_min_epi32(_mm256_max_epi32(x[r], zero), hmax), half), sh);
        o = _mm256_add_epi32(o, _mm256_mullo_epi32(a, _mm256_load_si256((const __m256i *)(n.WO + 8 * r))));
    }
    __m128i o4 = _mm_add_epi32(_mm256_castsi256_si128(o), _mm256_extracti128_si256(o, 1));
    o4 = _mm_add_epi32(o4, _mm_shuffle_epi32(o4, 0x4E));
    o4 = _mm_add_epi32(o4, _mm_shuffle_epi32(o4, 0xB1));
    return (int64_t)_mm_cvtsi128_si32(o4) + n.BO;
}

// AVX2 head. us/them: stored accumulators; cu/ct: constraint rows; fu/ft: forced-board rows (ZERO
// on a free move); psd: PSQT difference (stm - other, int32 at 2^qps).
template <class N>
FNNUE_KERNEL inline int eval_avx(const N &n, const int16_t *us, const int16_t *them, const int16_t *cu,
                                 const int16_t *ct, const int16_t *fu, const int16_t *ft, int32_t psd) {
    constexpr int A = N::A, L1 = N::L1, L2 = N::L2, NW = A / 64;
    alignas(32) int16_t act[2 * A];
    uint64_t mask[NW];
    for (int w = 0; w < NW; w++) mask[w] = 0;
    const __m256i zero = _mm256_setzero_si256();
    const __m256i qa = _mm256_set1_epi16(n.QA);
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
    for (int r = 0; r < L1 / 8; r++) s[r] = _mm256_load_si256((const __m256i *)(n.B1 + 8 * r));
    const int32_t *act32 = (const int32_t *)act;
    for (int w = 0; w < NW; w++) {
        uint64_t m = mask[w];
        while (m) {
            const int p = w * 64 + __builtin_ctzll(m);
            m &= m - 1;
            const __m256i a = _mm256_set1_epi32(act32[p]);
            const __m256i *wp = (const __m256i *)n.W1p[p];
            for (int r = 0; r < L1 / 8; r++) s[r] = _mm256_add_epi32(s[r], _mm256_madd_epi16(a, _mm256_load_si256(wp + r)));
        }
    }
    int64_t out;
    if constexpr (L2 > 0) {
        const __m256i hmax = _mm256_set1_epi32(n.H1MAX);
        const __m256i half = _mm256_set1_epi32(n.h1_round);
        const __m128i sh = _mm_cvtsi32_si128(n.h1_shift);
        alignas(32) int32_t hp[L1 / 2];  // h1 as int16 pairs (2e, 2e+1)
        for (int r = 0; r < L1 / 16; r++) {
            const __m256i a = _mm256_sra_epi32(
                _mm256_add_epi32(_mm256_min_epi32(_mm256_max_epi32(s[2 * r], zero), hmax), half), sh);
            const __m256i b = _mm256_sra_epi32(
                _mm256_add_epi32(_mm256_min_epi32(_mm256_max_epi32(s[2 * r + 1], zero), hmax), half), sh);
            // packus interleaves the 128-bit halves; permute 0xD8 restores h[16r .. 16r+15] order
            _mm256_store_si256((__m256i *)(hp + 8 * r), _mm256_permute4x64_epi64(_mm256_packus_epi32(a, b), 0xD8));
        }
        __m256i t[L2 / 8];
        for (int r = 0; r < L2 / 8; r++) t[r] = _mm256_load_si256((const __m256i *)(n.B2 + 8 * r));
        for (int e = 0; e < L1 / 2; e++) {
            const __m256i a = _mm256_set1_epi32(hp[e]);
            const __m256i *wp = (const __m256i *)n.W2p[e];
            for (int r = 0; r < L2 / 8; r++) t[r] = _mm256_add_epi32(t[r], _mm256_madd_epi16(a, _mm256_load_si256(wp + r)));
        }
        out = output_layer<N, L2 / 8>(n, t, n.H2MAX, n.h2_round, n.h2_shift);
    } else {
        out = output_layer<N, L1 / 8>(n, s, n.H1MAX, n.h1_round, n.h1_shift);
    }
    return finish(n, out, psd);
}

// Scalar reference of the same integer math, dense (no sparsity), int32/int64.
template <class N>
inline int eval_ref(const N &n, const int32_t *us, const int32_t *them, const int16_t *cu, const int16_t *ct,
                    const int16_t *fu, const int16_t *ft, int32_t psd) {
    constexpr int A = N::A, L1 = N::L1, L2 = N::L2, IN = N::IN;
    int32_t act[IN];
    for (int i = 0; i < A; i++) {
        act[i] = std::min<int32_t>(std::max<int32_t>(us[i] + cu[i] + fu[i], 0), n.QA);
        act[A + i] = std::min<int32_t>(std::max<int32_t>(them[i] + ct[i] + ft[i], 0), n.QA);
    }
    int64_t h1[L1];
    for (int k = 0; k < L1; k++) {
        int64_t s = n.B1[k];
        for (int i = 0; i < IN; i++) s += (int64_t)act[i] * n.W1[k][i];
        h1[k] = (std::min<int64_t>(std::max<int64_t>(s, 0), n.H1MAX) + n.h1_round) >> n.h1_shift;
    }
    int64_t out = n.BO;
    if constexpr (L2 > 0) {
        for (int k = 0; k < L2; k++) {
            int64_t s = n.B2[k];
            for (int j = 0; j < L1; j++) s += h1[j] * n.W2[k][j];
            s = (std::min<int64_t>(std::max<int64_t>(s, 0), n.H2MAX) + n.h2_round) >> n.h2_shift;
            out += s * n.WO[k];
        }
    } else {
        for (int k = 0; k < L1; k++) out += h1[k] * n.WO[k];
    }
    return finish(n, out, psd);
}

// One perspective's rows of a whole board: 9 (a pattern row or a decided row per miniboard).
template <class N, class Board>
inline void board_rows(const N &n, const Board &b, int P, const int16_t **rows, int32_t &ps) {
    ps = n.BIASP;
    for (int mb = 0; mb < 9; mb++) {
        const int bit = 1 << mb;
        if ((b.mini_board_states[0] | b.mini_board_states[1] | b.mini_board_states[2]) & bit) {
            const int st = (b.mini_board_states[2] & bit) ? 2 : ((b.mini_board_states[0] & bit) ? 0 : 1);
            const int d = dec_idx(mb, st, P);
            rows[mb] = n.DEC[d];
            ps += n.DECP[d];
        } else {
            const int pat = pat_of(b.mini_boards[mb].markers[P], b.mini_boards[mb].markers[P ^ 1]);
            rows[mb] = n.trow(mb, pat);
            ps += n.tps(mb, pat);
        }
    }
}

template <class N, class Board>
inline void scratch_avx(const N &n, const Board &b, int P, int16_t *out, int32_t &ps) {
    const int16_t *rows[9];
    board_rows(n, b, P, rows, ps);
    for (int t = 0; t < N::A; t += 16) {
        __m256i v = _mm256_load_si256((const __m256i *)(n.BIAS + t));
        for (int r = 0; r < 9; r++) v = _mm256_add_epi16(v, _mm256_load_si256((const __m256i *)(rows[r] + t)));
        _mm256_store_si256((__m256i *)(out + t), v);
    }
}

template <class N, class Board>
inline void scratch_ref(const N &n, const Board &b, int P, int32_t *out, int32_t &ps) {
    const int16_t *rows[9];
    board_rows(n, b, P, rows, ps);
    for (int i = 0; i < N::A; i++) {
        int32_t s = n.BIAS[i];
        for (int r = 0; r < 9; r++) s += rows[r][i];
        out[i] = s;
    }
}

// Evaluation-time rows: constraint rows per side, forced-board rows (the forced board's pattern in
// each perspective's view) and the PSQT difference's constant part.
template <class N, class Board>
struct EvalRows {
    const int16_t *cu, *ct, *fu, *ft;
    int32_t ps_extra;  // (con + F PSQT of stm) - (the other's)
    EvalRows(const N &n, const Board &b, int stm, int c) {
        cu = n.CON[c];
        ct = n.CON[10 + c];
        ps_extra = (int32_t)n.CONP[c] - n.CONP[10 + c];
        if (c < 9 && n.forced) {
            const int m0 = b.mini_boards[c].markers[0], m1 = b.mini_boards[c].markers[1];
            const int t0 = g_tern.t[m0], t1 = g_tern.t[m1];
            const int ps = stm ? t1 + 2 * t0 : t0 + 2 * t1;   // stm's view
            const int pt = stm ? t0 + 2 * t1 : t1 + 2 * t0;   // the other's view
            fu = n.frow(ps);
            ft = n.frow(pt);
            ps_extra += n.fps(ps) - n.fps(pt);
        } else {
            fu = ft = n.ZERO;
        }
    }
};

#ifdef FASTNNUE_CHECK
// The first-layer bound's premises (a forced board is live; the side to move has as many stones as the
// other or one fewer) and its conclusion (every lane of acc + con + F fits int16), on the int32 reference.
template <class N, class Board>
inline void check_ranges(const Board &b, int stm, int constraint, const int32_t *us, const int32_t *them,
                         const EvalRows<N, Board> &er) {
    const int cur = b.n_moves;
    if (constraint < 9 && ((b.mini_board_states[0] | b.mini_board_states[1] | b.mini_board_states[2]) >> constraint & 1)) {
        g_check.range_violations++;
        std::fprintf(stderr, "fast_nnue_b CHECK: constraint %d names a decided board at n_moves=%d\n", constraint, cur);
        std::abort();
    }
    int d = 0;
    for (int mb = 0; mb < 9; mb++)
        d += __builtin_popcount(b.mini_boards[mb].markers[stm]) - __builtin_popcount(b.mini_boards[mb].markers[stm ^ 1]);
    if (d != 0 && d != -1) {
        g_check.range_violations++;
        std::fprintf(stderr, "fast_nnue_b CHECK: stone difference %d from the side to move at n_moves=%d\n", d, cur);
        std::abort();
    }
    for (int i = 0; i < N::A; i++) {
        const int32_t xu = us[i] + er.cu[i] + er.fu[i], xt = them[i] + er.ct[i] + er.ft[i];
        if (xu < -32768 || xu > 32767 || xt < -32768 || xt > 32767) {
            g_check.range_violations++;
            std::fprintf(stderr, "fast_nnue_b CHECK: acc + con + F (%d, %d) outside int16 at n_moves=%d\n", xu, xt, cur);
            std::abort();
        }
    }
}
#endif

// ---------------------------------------------------------------- check counters (B only)

struct TouchStats {
    std::atomic<long long> engines{0}, rows_sum{0}, rows_max{0};
    ~TouchStats() {
        if (engines.load() == 0) return;
        std::fprintf(stderr, "fast_nnue_b check: distinct T/F rows touched per engine: mean %.0f max %lld over %lld engines\n",
                     (double)rows_sum.load() / engines.load(), rows_max.load(), engines.load());
    }
};
inline TouchStats g_touch;

// ---------------------------------------------------------------- incremental stack

template <class N, typename Board>
inline int evaluate_board(const N &n, const Board &b, int constraint);

template <class N>
struct Stack {
    static constexpr int A = N::A;
    struct alignas(64) Entry {
        int16_t v[2][A];  // [absolute player P][lane]
        int32_t ps[2];    // PSQT lane per P
    };
    const N &n;
    std::vector<Entry> acc;
    // The empty board's tt_hash is 0, so 0 cannot mean "no position" in pos_key.
    static constexpr uint64_t kNoKey = 0x9E3779B97F4A7C15ull;
    uint64_t pos_key[MAXPLY];  // tt_hash of the position acc[ply] holds; kNoKey: none
    Dirty dirty[MAXPLY];       // dirty[j]: the last move recorded from ply j-1 to ply j
    int root = 0;
#if FASTNNUE_CACHE_BITS > 0
    struct CacheEntry {
        uint64_t key;
        int32_t eval, pad;
    };
    static constexpr uint64_t CMASK = (1u << FASTNNUE_CACHE_BITS) - 1;
    std::vector<CacheEntry> cache = std::vector<CacheEntry>(1u << FASTNNUE_CACHE_BITS, CacheEntry{0, 0, 0});
#endif
#ifdef FASTNNUE_CHECK
    std::vector<uint8_t> touched = std::vector<uint8_t>((size_t)10 * NPAT, 0);
    long long n_touched = 0;
    void touch(int m, int pat) {  // m = 9: F
        uint8_t &t = touched[(size_t)m * NPAT + pat];
        if (!t) { t = 1; n_touched++; }
    }
    ~Stack() {
        g_touch.engines++;
        g_touch.rows_sum += n_touched;
        long long seen = g_touch.rows_max.load();
        while (n_touched > seen && !g_touch.rows_max.compare_exchange_weak(seen, n_touched)) {}
    }
#endif

    explicit Stack(const N &net) : n(net), acc(MAXPLY) {
        for (uint64_t &k : pos_key) k = kNoKey;
        std::memset(dirty, 0, sizeof(dirty));
    }

    // From scratch at the board's own ply.
    template <typename Board>
    void refresh_at(const Board &b) {
        const int ply = b.n_moves;
        scratch_avx(n, b, 0, acc[ply].v[0], acc[ply].ps[0]);
        scratch_avx(n, b, 1, acc[ply].v[1], acc[ply].ps[1]);
        pos_key[ply] = b.tt_hash;
#ifdef FASTNNUE_CHECK
        for (int mb = 0; mb < 9; mb++)
            if (!((b.mini_board_states[0] | b.mini_board_states[1] | b.mini_board_states[2]) >> mb & 1))
                for (int P = 0; P < 2; P++) touch(mb, pat_of(b.mini_boards[mb].markers[P], b.mini_boards[mb].markers[P ^ 1]));
#endif
    }

    template <typename Board>
    void refresh_root(const Board &b) {
        root = b.n_moves;
        refresh_at(b);
#ifdef FASTNNUE_CHECK
        g_check.roots++;
#endif
    }

    static void prefetch_row(const int16_t *r) {
        for (int l = 0; l < A * 2; l += 64) _mm_prefetch((const char *)r + l, _MM_HINT_T0);
    }

    // make_move_fast(FastBoard&) just before board.n_moves++ (the board is otherwise the child's, its
    // tt_hash final); parent_key is the parent's tt_hash (MoveUndo.tt_hash).
    template <typename Board>
    void on_make(const Board &b, int mb, int sq, int stm, int decided, int before_stm, int before_other,
                 uint64_t parent_key) {
        const int nm = b.n_moves;
        Dirty &d = dirty[nm + 1];
        d.mb = (uint8_t)mb;
        d.sq = (uint8_t)sq;
        d.stm = (uint8_t)stm;
        d.decided = (int8_t)decided;
        d.before[stm] = (uint16_t)before_stm;
        d.before[stm ^ 1] = (uint16_t)before_other;
        d.pkey = parent_key;
        d.ckey = b.tt_hash;
#if FASTNNUE_CACHE_BITS > 0 && !defined(FASTNNUE_NO_CACHE_PREFETCH)
        _mm_prefetch((const char *)&cache[b.tt_hash & CMASK], _MM_HINT_T0);  // evaluate_keyed reads it first
#endif
#if FASTNNUE_B_PREFETCH >= 1
        {
            const int ts = g_tern.t[before_stm], to = g_tern.t[before_other];
            if (decided < 0) {  // the rows the child's update adds
                const int ts2 = ts + POW3_SQ[sq];
                prefetch_row(n.trow(mb, ts2 + 2 * to));
                prefetch_row(n.trow(mb, to + 2 * ts2));
            }
#if FASTNNUE_B_PREFETCH >= 2
            prefetch_row(n.trow(mb, ts + 2 * to));  // and the rows it subtracts
            prefetch_row(n.trow(mb, to + 2 * ts));
#endif
            const int c = b.active_board;  // the child's forced board
            if (c < 9 && n.forced) {
                const int t0 = g_tern.t[b.mini_boards[c].markers[0]], t1 = g_tern.t[b.mini_boards[c].markers[1]];
                prefetch_row(n.frow(t0 + 2 * t1));
                prefetch_row(n.frow(t1 + 2 * t0));
            }
        }
#endif
    }
    static constexpr int POW3_SQ[9] = {1, 3, 9, 27, 81, 243, 729, 2187, 6561};

    void update(int j) {
        const Dirty &d = dirty[j];
        const Entry &par = acc[j - 1];
        Entry &ch = acc[j];
        const int t0 = g_tern.t[d.before[0]], t1 = g_tern.t[d.before[1]];
        const int old0 = t0 + 2 * t1, old1 = t1 + 2 * t0;
        const int16_t *s0 = n.trow(d.mb, old0), *s1 = n.trow(d.mb, old1);
        int32_t dp0 = -n.tps(d.mb, old0), dp1 = -n.tps(d.mb, old1);
        const int16_t *a0, *a1;
        if (d.decided < 0) {
            const int add = POW3_SQ[d.sq] * (d.stm == 0 ? 1 : 2);  // the new stone in player 0's view
            const int new0 = old0 + add, new1 = old1 + POW3_SQ[d.sq] * (d.stm == 1 ? 1 : 2);
            a0 = n.trow(d.mb, new0);
            a1 = n.trow(d.mb, new1);
            dp0 += n.tps(d.mb, new0);
            dp1 += n.tps(d.mb, new1);
#ifdef FASTNNUE_CHECK
            touch(d.mb, new0);
            touch(d.mb, new1);
#endif
        } else {
            const int r0 = dec_idx(d.mb, d.decided, 0), r1 = dec_idx(d.mb, d.decided, 1);
            a0 = n.DEC[r0];
            a1 = n.DEC[r1];
            dp0 += n.DECP[r0];
            dp1 += n.DECP[r1];
        }
        sub_add2<A>(par.v[0], ch.v[0], a0, s0, par.v[1], ch.v[1], a1, s1);
        ch.ps[0] = par.ps[0] + dp0;
        ch.ps[1] = par.ps[1] + dp1;
    }

    // Make acc[b.n_moves] hold b (fast_nnue.hpp's Stack::sync: walk back while the recorded moves
    // lead to the position needed, replay from the first entry holding a parent, else refresh).
    template <typename Board>
    void sync(const Board &b) {
        const int cur = b.n_moves;
        uint64_t need = b.tt_hash;
        int k = cur;
        for (;;) {
            if (k <= 0 || dirty[k].ckey != need) {
                refresh_at(b);
#ifdef FASTNNUE_CHECK
                g_check.fallback_refresh++;
#endif
                return;
            }
            need = dirty[k].pkey;
            if (pos_key[--k] == need) break;
        }
#ifdef FASTNNUE_CHECK
        long long chain = cur - k, seen = g_check.max_chain.load();
        while (chain > seen && !g_check.max_chain.compare_exchange_weak(seen, chain)) {}
#endif
        for (int j = k + 1; j <= cur; j++) {
            update(j);
            pos_key[j] = dirty[j].ckey;
#ifdef FASTNNUE_CHECK
            g_check.lazy_updates++;
            if (dirty[j].decided >= 0) g_check.lazy_decided_updates++;
#endif
        }
    }

    template <typename Board>
    int evaluate(const Board &b, int constraint) {
        const int cur = b.n_moves;
        if (pos_key[cur] != b.tt_hash) sync(b);
#ifdef FASTNNUE_CHECK
        else g_check.key_reuse++;
#endif
        const int stm = cur & 1;
        const EvalRows<N, Board> er(n, b, stm, constraint);
        const Entry &e = acc[cur];
        const int32_t psd = e.ps[stm] - e.ps[stm ^ 1] + er.ps_extra;
        const int v = eval_avx(n, e.v[stm], e.v[stm ^ 1], er.cu, er.ct, er.fu, er.ft, psd);
#ifdef FASTNNUE_CHECK
        check_node(b, constraint, v);
#endif
        return v;
    }

    template <typename Board>
    int evaluate_keyed(const Board &b, int constraint, uint64_t key) {
#if FASTNNUE_CACHE_BITS > 0
        CacheEntry &ce = cache[key & CMASK];
        if (key && ce.key == key) {
#ifdef FASTNNUE_CHECK
            const int e = evaluate_board(n, b, constraint);  // from scratch: the stack sees the release calls
            g_check.cache_hits++;
            if (e != ce.eval) {
                g_check.cache_mismatch++;
                std::fprintf(stderr, "fast_nnue_b CHECK: cached eval %d != computed %d\n", ce.eval, e);
                std::abort();
            }
#endif
            return ce.eval;
        }
        const int e = evaluate(b, constraint);
        ce.key = key;
        ce.eval = e;
        return e;
#else
        (void)key;
        return evaluate(b, constraint);
#endif
    }

#ifdef FASTNNUE_CHECK
    template <typename Board>
    void check_node(const Board &b, int constraint, int v) {
        const int cur = b.n_moves, stm = cur & 1;
        int32_t ref[2][A], ps[2];
        scratch_ref(n, b, 0, ref[0], ps[0]);
        scratch_ref(n, b, 1, ref[1], ps[1]);
        for (int P = 0; P < 2; P++) {
            for (int i = 0; i < A; i++)
                if (ref[P][i] != acc[cur].v[P][i]) {
                    g_check.acc_mismatch++;
                    std::fprintf(stderr, "fast_nnue_b CHECK: accumulator mismatch at n_moves=%d P=%d lane=%d: "
                                 "incremental %d scratch %d (root %d)\n", cur, P, i, acc[cur].v[P][i], ref[P][i], root);
                    std::abort();
                }
            if (ps[P] != acc[cur].ps[P]) {
                g_check.acc_mismatch++;
                std::fprintf(stderr, "fast_nnue_b CHECK: PSQT mismatch at n_moves=%d P=%d: incremental %d scratch %d\n",
                             cur, P, acc[cur].ps[P], ps[P]);
                std::abort();
            }
        }
        const EvalRows<N, Board> er(n, b, stm, constraint);
        check_ranges(b, stm, constraint, ref[stm], ref[stm ^ 1], er);
        const int r = eval_ref(n, ref[stm], ref[stm ^ 1], er.cu, er.ct, er.fu, er.ft, ps[stm] - ps[stm ^ 1] + er.ps_extra);
        if (r != v) {
            g_check.out_mismatch++;
            std::fprintf(stderr, "fast_nnue_b CHECK: output mismatch at n_moves=%d: avx %d scalar %d\n", cur, v, r);
            std::abort();
        }
        if (constraint < 9) {
            const int m0 = b.mini_boards[constraint].markers[0], m1 = b.mini_boards[constraint].markers[1];
            touch(9, pat_of(m0, m1));
            touch(9, pat_of(m1, m0));
        }
        g_check.evals++;
        const int dec = b.mini_board_states[0] | b.mini_board_states[1] | b.mini_board_states[2];
        if (dec) g_check.decided_nodes++;
        if (b.mini_board_states[2]) g_check.drawn_nodes++;
        if (constraint == 9) g_check.free_nodes++; else g_check.forced_nodes++;
    }
#endif
};

// From scratch, for GlobalBoard callers (datagen label's evaluate()).
template <class N, typename Board>
inline int evaluate_board(const N &n, const Board &b, int constraint) {
    alignas(64) int16_t acc[2][N::A];
    int32_t ps[2];
    scratch_avx(n, b, 0, acc[0], ps[0]);
    scratch_avx(n, b, 1, acc[1], ps[1]);
    const int stm = b.n_moves & 1;
    const EvalRows<N, Board> er(n, b, stm, constraint);
    const int e = eval_avx(n, acc[stm], acc[stm ^ 1], er.cu, er.ct, er.fu, er.ft, ps[stm] - ps[stm ^ 1] + er.ps_extra);
#ifdef FASTNNUE_CHECK
    int32_t ref[2][N::A], rps[2];
    scratch_ref(n, b, 0, ref[0], rps[0]);
    scratch_ref(n, b, 1, ref[1], rps[1]);
    check_ranges(b, stm, constraint, ref[stm], ref[stm ^ 1], er);
    const int r = eval_ref(n, ref[stm], ref[stm ^ 1], er.cu, er.ct, er.fu, er.ft, rps[stm] - rps[stm ^ 1] + er.ps_extra);
    if (r != e) {
        g_check.out_mismatch++;
        std::fprintf(stderr, "fast_nnue_b CHECK: scratch output mismatch: avx %d scalar %d\n", e, r);
        std::abort();
    }
    g_check.scratch_evals++;
#endif
    return e;
}

}  // namespace bnn
}  // namespace fnnue
