#pragma once
// Fast incremental integer NNUE inference (nnue2 stage 1: the per-cell FNN1 net).
//
// Net (FNN1, tools/experiments/full_nnue/full_nnue_float.hpp): per perspective P
//   cells of live miniboards  (mb*9+sq)*2 + (owner != P)          rows   0..161
//   decided miniboards         162 + mb*3 + {P won, other won, draw} rows 162..188
//   constraint                 189 + (0..8 forced, 9 free)          rows 189..198
//   acc_P = B0 + sum rows (A = 256); h = [crelu(acc_stm), crelu(acc_ntm)] (2A = 512)
//   -> L1 (32) -> crelu -> 1, x1000, truncated to int (stm-relative eval units).
//
// Integer scheme (all scales powers of two, chosen at load so nothing can overflow):
//   first layer   int16 rows W0q = round(W0 * 2^qa), bias likewise; accumulators int16.
//                 qa is the largest value whose rigorous per-lane bound fits int16: every
//                 miniboard contributes either one of the 11,093 live patterns (no line,
//                 not full; enumerated at load on the quantized rows) or one decided row,
//                 plus a constraint row. crelu = clamp(acc + con_row, 0, 2^qa) as int16.
//                 (A per-cell worst case, every cell holding its worst stone, allowed only
//                 2^9; the pattern bound allows 2^10 for r10_aug_d8x5M_lr6e3.)
//   dense 2A->L1  int16 weights W1q = round(W1 * 2^qb); int32 sums via _mm256_madd_epi16,
//                 bias B1q = round(B1 * 2^(qa+qb)). qb is the largest value with every
//                 |W1q| <= 32767 and 2^qa * sum_i |W1q[k][i]| + |B1q[k]| < 2^31.
//                 The sum only visits nonzero activation PAIRS (about 80% of crelu
//                 outputs are 0 on real positions), so it is sparse and exact.
//   L1 crelu      clamp(s, 0, 2^(qa+qb)) >> (qa+qb-15): [0, 2^15] int32.
//   output        W2q = round(W2 * 2^qo), B2q = round(B2 * 2^(15+qo)), qo the largest with
//                 2^15 * sum|W2q| + |B2q| < 2^31; eval = trunc(1000 * out / 2^(15+qo)),
//                 truncation toward zero exactly like the float hook's (int) cast.
//
// Accumulator stack (Stack): one entry per board.n_moves holding BOTH absolute
// perspectives (player 0's and player 1's view), so a move never swaps them. make only
// records what changed (Dirty: square, mover, decided state, the miniboard's markers
// before the move, the parent's and the child's position keys); evaluate walks back to the
// nearest entry that holds an ancestor of the current position and replays the updates
// (lazy: a node cut before evaluation costs nothing). unmake does nothing: an entry is
// overwritten when a later evaluation at that ply needs another position.
// The constraint row is NOT in the accumulator; it is added at evaluation, so free moves,
// forced boards and the root's constraint never enter the incremental state.
// evaluate_keyed puts a per-engine direct-mapped cache (2^FASTNNUE_CACHE_BITS entries, key =
// the search's FastBoard tt_hash) in front of evaluate: a third to a half of the search's
// evaluations are repeats.
//
// Lane order: permute_fnn1.py rewrites an FNN1 file with its accumulator lanes paired by
// co-activation (same function, bit-identical quantized evals), cutting the nonzero pairs
// the dense kernel visits from ~96 to ~80 per evaluation.
//
// Position keys (stage 3): every accumulator entry carries the tt_hash of the position it
// holds, and every recorded move carries the tt_hash of the position it was made from
// (MoveUndo.tt_hash, `pkey`) and of the one it led to (`ckey`). evaluate reuses an entry
// only when its key is the current position's; the walk back follows recorded moves only
// while each one leads to the position needed, and stops at the first entry whose key is
// that move's parent. A search entered without refresh_root, or any other break in the
// chain, therefore ends in a from-scratch refresh (counted as a fallback in check builds)
// instead of replaying another position's accumulator. Keys are 64-bit Zobrist hashes, the
// same assumption as the eval cache (a collision would reuse a wrong entry).
//
// Build flags: -DFASTNNUE_CHECK verifies at every evaluation that the incremental
// accumulators equal a scalar from-scratch int32 recomputation and that the AVX2 output
// equals a scalar dense reference of the same integer math; mismatches abort, counters
// are printed at exit. Eval-cache hits are verified against a from-scratch evaluation, so
// the stack sees exactly the release build's calls.
// -DFASTNNUE_FLOOR_SHIFTS restores stage 1/2's floor in the hidden-layer shift (default:
// round to nearest); -DFASTNNUE_NO_CACHE_PREFETCH drops the eval-cache prefetch at make;
// -DFASTNNUE_NO_SYSV drops the sysv_abi attribute of the eval kernels (Win64 only).
//
// Net file and engine side (stage 4). Everything here lives in namespace fnnue (the B nets in
// fnnue::bnn), and the net is loaded once per namespace. The file is the environment variable
// kPathEnv ("FASTNNUE_PATH") if set, else the path a candidate compiled in as FASTNNUE_NET_FILE
// (make_cand_b.py --net), else start-up fails. The first load prints the side, the file, where
// its name came from, its size and its CRC-32 (zlib's) to stderr, once. A two-engine build
// (tools/experiments/fast_nnue/fast_pair.py) gives Prev its own renamed copy of these headers
// (prev_fast_nnue*.hpp): namespace fnnue_prev, kSide "Prev", and every FASTNNUE_* / FNNUE_* name
// but FASTNNUE_CHECK with a PREV_ prefix ("FASTNNUE_PREV_PATH", FASTNNUE_PREV_NET_FILE,
// FASTNNUE_PREV_KIND, FASTNNUE_PREV_FLOOR_SHIFTS, FNNUE_PREV_KERNEL, ...), so each engine has its
// own net, tables, counters, stack type and numerics flags, and the two can never share weights.
// In such a build the -D flags above apply to Dev only (Prev: -DFASTNNUE_PREV_...), except
// -DFASTNNUE_CHECK, which checks both copies.
//
// Stage 2 (pattern-generator "B" nets) adds a format behind load_net()'s magic dispatch,
// its own row functions (pattern row per live miniboard: sub old / add new pattern on a
// move; forced-board pattern row and per-side constraint rows added at evaluation) and a
// PSQT lane plus deeper head; the stack, transform and sparse dense kernel carry over.
#include <immintrin.h>

#include <algorithm>
#include <atomic>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <vector>

#if !defined(__AVX2__)
#error "fast_nnue.hpp needs AVX2 (-mavx2)"
#endif

#ifndef FASTNNUE_CACHE_BITS
#define FASTNNUE_CACHE_BITS 14  // log2 entries of Stack::evaluate_keyed cache (16 B each); 0 = none
#endif

// Win64's calling convention makes a non-inlined callee save and restore xmm6-15, which the
// eval kernels all use; sysv_abi (the Linux default, so a no-op on CodinGame) avoids it.
#if defined(_WIN64) && !defined(FASTNNUE_NO_SYSV)
#define FNNUE_KERNEL __attribute__((sysv_abi))
#else
#define FNNUE_KERNEL
#endif

namespace fnnue {

// This copy's engine side and net-file variable (fast_pair.py's Prev copy renames both).
inline constexpr const char *kSide = "Dev";
inline constexpr const char *kPathEnv = "FASTNNUE_PATH";

constexpr int A = 256;          // accumulator lanes per perspective
constexpr int L1 = 32;          // hidden width
constexpr int IN = 2 * A;       // dense input: [us, them]
constexpr int NPAIR = IN / 2;   // madd consumes activations in (2p, 2p+1) pairs
constexpr int NROWS = 200;      // FNN1 first-layer rows (199 used, 199 is zero)
constexpr int ROW_DEC = 162, ROW_CON = 189;
constexpr int MAXPLY = 96;      // indexed by board.n_moves (<= 81) + 1
constexpr int HBITS = 15;       // L1 activation scale after the shift

struct Net {
    alignas(64) int16_t W0[NROWS][A];
    alignas(64) int16_t B0[A];
    alignas(64) int32_t W1p[NPAIR][L1];  // pair p, output k: lo16 = W1q[k][2p], hi16 = W1q[k][2p+1]
    alignas(64) int16_t W1[L1][IN];      // logical layout, for the scalar reference
    alignas(64) int32_t B1[L1];
    alignas(64) int32_t W2[L1];
    int32_t B2 = 0;
    int qa = 0, qb = 0, qo = 0;
    int16_t QA = 0;       // crelu ceiling = 1.0
    int32_t H1MAX = 0;    // L1 crelu ceiling = 1.0 before the shift
    int h1_shift = 0;
    int32_t h1_round = 0; // added before the shift: 2^(h1_shift-1) (round to nearest), or 0 (floor)
    int out_shift = 0;    // eval = trunc(1000 * out / 2^out_shift)
};

// The rounding term added before an arithmetic right shift of a non-negative value.
inline int32_t round_half(int shift) {
#ifdef FASTNNUE_FLOOR_SHIFTS
    (void)shift;
    return 0;
#else
    return shift > 0 ? (int32_t)1 << (shift - 1) : 0;
#endif
}

inline Net g_net;
inline std::once_flag g_once;

inline int cell_row(int mb, int sq, int owner, int P) { return (mb * 9 + sq) * 2 + (owner != P); }
inline int dec_row(int mb, int state, int P) {
    return ROW_DEC + mb * 3 + (state == 2 ? 2 : (state == P ? 0 : 1));
}

// ---------------------------------------------------------------- loading / quantization

// Miniboard patterns (index sum_k digit_k 3^k, digit 0 empty / 1 player 0 / 2 player 1) that
// can be live: no three-in-a-row for either player and not full.
inline const std::vector<uint8_t> &live_patterns() {
    static const std::vector<uint8_t> t = [] {
        static const int LINES[8] = {7, 56, 448, 73, 146, 292, 273, 84};
        std::vector<uint8_t> v(19683);
        for (int p = 0; p < 19683; p++) {
            int m[3] = {0, 0, 0};
            for (int k = 0, x = p; k < 9; k++, x /= 3) m[x % 3] |= 1 << k;
            bool ok = m[0] != 0;
            for (int l : LINES)
                if ((m[1] & l) == l || (m[2] & l) == l) ok = false;
            v[p] = (uint8_t)ok;
        }
        return v;
    }();
    return t;
}

// Per lane, the max and min over live patterns of miniboard mb of the sum of its cell rows
// (perspective 0: player 0's stones are the "mine" rows; the live set is symmetric under
// swapping the players, so perspective 1 has the same range). W: rows x A, row-major.
template <typename T>
struct LiveRange {
    const T *W;
    int mb;
    std::vector<T> s;  // (10, A) running sums by depth
    T *hi, *lo;
    const uint8_t *live;
    void rec(int k, int pat, int pw) {
        const T *cur = s.data() + (size_t)k * A;
        if (k == 9) {
            if (live[pat])
                for (int i = 0; i < A; i++) {
                    hi[i] = std::max(hi[i], cur[i]);
                    lo[i] = std::min(lo[i], cur[i]);
                }
            return;
        }
        T *nxt = s.data() + (size_t)(k + 1) * A;
        for (int d = 0; d < 3; d++) {
            if (d == 0) {
                std::memcpy(nxt, cur, sizeof(T) * A);
            } else {
                const T *r = W + (size_t)((mb * 9 + k) * 2 + (d - 1)) * A;
                for (int i = 0; i < A; i++) nxt[i] = cur[i] + r[i];
            }
            rec(k + 1, pat + d * pw, pw * 3);
        }
    }
};

// Rigorous per-lane range of the stored accumulator (bias + for each miniboard either a live
// pattern's cells or one decided row) plus a constraint row, over every board the features
// can express; cross-board constraints (stone counts) are ignored, so it is an upper bound.
// Returns [lo, hi] over lanes of min(acc, acc + con) and max(acc, acc + con).
template <typename T>
inline void acc_range(const T *W, const T *B, double &lo_all, double &hi_all) {
    std::vector<T> hi(A), lo(A), mh(A), ml(A);
    for (int i = 0; i < A; i++) hi[i] = lo[i] = B[i];
    for (int mb = 0; mb < 9; mb++) {
        for (int i = 0; i < A; i++) {
            mh[i] = std::numeric_limits<T>::lowest();
            ml[i] = std::numeric_limits<T>::max();
        }
        LiveRange<T> lr{W, mb, std::vector<T>((size_t)10 * A, T(0)), mh.data(), ml.data(), live_patterns().data()};
        lr.rec(0, 0, 1);
        for (int i = 0; i < A; i++) {
            T dh = mh[i], dl = ml[i];
            for (int st = 0; st < 3; st++) {
                const T d = W[(size_t)(ROW_DEC + mb * 3 + st) * A + i];
                dh = std::max(dh, d);
                dl = std::min(dl, d);
            }
            hi[i] += dh;
            lo[i] += dl;
        }
    }
    lo_all = 0;
    hi_all = 0;
    for (int i = 0; i < A; i++) {
        T cmax = W[(size_t)ROW_CON * A + i], cmin = cmax;
        for (int c = 1; c < 10; c++) {
            cmax = std::max(cmax, W[(size_t)(ROW_CON + c) * A + i]);
            cmin = std::min(cmin, W[(size_t)(ROW_CON + c) * A + i]);
        }
        hi_all = std::max(hi_all, (double)hi[i] + std::max(T(0), cmax));
        lo_all = std::min(lo_all, (double)lo[i] + std::min(T(0), cmin));
    }
}

inline bool quantize_fnn1(Net &n, const std::vector<float> &W0f, const std::vector<float> &B0f,
                          const std::vector<float> &W1f, const std::vector<float> &B1f,
                          const std::vector<float> &W2f, float B2f, const char *path) {
    auto q = [](double x, int bits) { return (long long)std::llround(std::ldexp(x, bits)); };
    // First layer: the largest qa whose rigorous accumulator range (acc_range, exact on the
    // quantized integer rows) fits int16. A float estimate picks the start.
    double flo = 0, fhi = 0;
    {
        std::vector<double> Wd(W0f.begin(), W0f.end()), Bd(B0f.begin(), B0f.end());
        acc_range(Wd.data(), Bd.data(), flo, fhi);
    }
    const double fmax = std::max(-flo, fhi);
    int qa = std::min(13, (int)std::floor(std::log2(32767.0 / (fmax + 0.05))));
    long long bound_lo = 0, bound_hi = 0, max_w0 = 0;
    for (; qa >= 1; qa--) {
        std::vector<int64_t> Wq((size_t)NROWS * A), Bq(A);
        long long mw = 0;
        for (size_t i = 0; i < Wq.size(); i++) {
            Wq[i] = q(W0f[i], qa);
            mw = std::max(mw, (long long)std::llabs(Wq[i]));
        }
        for (int i = 0; i < A; i++) {
            Bq[i] = q(B0f[i], qa);
            mw = std::max(mw, (long long)std::llabs(Bq[i]));
        }
        double lo = 0, hi = 0;
        acc_range(Wq.data(), Bq.data(), lo, hi);
        if (mw <= 32767 && hi <= 32767 && lo >= -32768) {
            bound_lo = (long long)lo;
            bound_hi = (long long)hi;
            max_w0 = mw;
            break;
        }
    }
    if (qa < 1) { std::fprintf(stderr, "fast_nnue: first layer does not fit int16\n"); return false; }
    n.qa = qa;
    n.QA = (int16_t)(1 << qa);
    for (int r = 0; r < NROWS; r++)
        for (int i = 0; i < A; i++) n.W0[r][i] = (int16_t)q(W0f[(size_t)r * A + i], qa);
    for (int i = 0; i < A; i++) n.B0[i] = (int16_t)q(B0f[i], qa);

    // Dense layer: the largest qb with int16 weights and an int32-safe sum.
    int qb = -1;
    long long max_w1 = 0, max_l1 = 0;
    for (int bits = 16; bits >= 0 && qb < 0; bits--) {
        bool ok = true;
        long long mw = 0, ml = 0;
        for (int k = 0; k < L1 && ok; k++) {
            long long s = 0;
            for (int i = 0; i < IN; i++) {
                long long v = q(W1f[(size_t)k * IN + i], bits);
                mw = std::max(mw, std::llabs(v));
                s += std::llabs(v);
            }
            long long b = q(B1f[k], bits + qa);
            long long tot = s * (1LL << qa) + std::llabs(b);
            ml = std::max(ml, tot);
            ok = mw <= 32767 && tot < (1LL << 31) - 1;
        }
        if (ok) { qb = bits; max_w1 = mw; max_l1 = ml; }
    }
    if (qb < 0) { std::fprintf(stderr, "fast_nnue: dense layer does not fit\n"); return false; }
    n.qb = qb;
    n.H1MAX = (int32_t)(1LL << (qa + qb));
    const int hbits = std::min(HBITS, qa + qb);
    n.h1_shift = qa + qb - hbits;
    // (clamp(s) + 2^(shift-1)) >> shift is still <= 2^hbits: clamp(s) <= 2^(hbits+shift).
    n.h1_round = round_half(n.h1_shift);
    for (int k = 0; k < L1; k++) {
        n.B1[k] = (int32_t)q(B1f[k], qa + qb);
        for (int i = 0; i < IN; i++) n.W1[k][i] = (int16_t)q(W1f[(size_t)k * IN + i], qb);
    }
    for (int p = 0; p < NPAIR; p++)
        for (int k = 0; k < L1; k++)
            n.W1p[p][k] = (int32_t)(((uint32_t)(uint16_t)n.W1[k][2 * p])
                                    | ((uint32_t)(uint16_t)n.W1[k][2 * p + 1] << 16));

    // Output layer.
    int qo = -1;
    long long out_bound = 0;
    for (int bits = 20; bits >= 0 && qo < 0; bits--) {
        long long s = 0;
        for (int k = 0; k < L1; k++) s += std::llabs(q(W2f[k], bits));
        long long tot = s * (1LL << hbits) + std::llabs(q(B2f, bits + hbits));
        if (tot < (1LL << 31) - 1) { qo = bits; out_bound = tot; }
    }
    if (qo < 0) { std::fprintf(stderr, "fast_nnue: output layer does not fit\n"); return false; }
    n.qo = qo;
    for (int k = 0; k < L1; k++) n.W2[k] = (int32_t)q(W2f[k], qo);
    n.B2 = (int32_t)q(B2f, qo + hbits);
    n.out_shift = hbits + qo;
    long long max_w2 = 0;
    for (int k = 0; k < L1; k++) max_w2 = std::max(max_w2, (long long)std::abs(n.W2[k]));
    std::fprintf(stderr,
                 "fast_nnue: FNN1 %s quantized: QA=2^%d (max|W0q| %lld, live-pattern acc bound [%lld, %lld] of int16), "
                 "QB=2^%d (max|W1q| %lld, L1 sum bound %lld of 2^31), L1 act 2^%d (shift %s), QO=2^%d (max|W2q| %lld, "
                 "out bound %lld of 2^31); 0 weights clipped\n",
                 path, qa, max_w0, bound_lo, bound_hi, qb, max_w1, max_l1, hbits, n.h1_round ? "rounds" : "floors", qo,
                 max_w2, out_bound);
    return true;
}

inline bool load_net(Net &n, const char *path) {
    FILE *f = path ? std::fopen(path, "rb") : nullptr;
    if (!f) return false;
    char magic[4] = {0, 0, 0, 0};
    bool ok = std::fread(magic, 1, 4, f) == 4;
    if (ok && std::memcmp(magic, "FNN1", 4) == 0) {
        int a = 0, l1 = 0;
        ok = std::fread(&a, 4, 1, f) == 1 && std::fread(&l1, 4, 1, f) == 1 && a == A && l1 == L1;
        std::vector<float> W0f((size_t)NROWS * A), B0f(A), W1f((size_t)L1 * IN), B1f(L1), W2f(L1);
        float B2f = 0;
        ok = ok && std::fread(W0f.data(), 4, W0f.size(), f) == W0f.size()
             && std::fread(B0f.data(), 4, A, f) == (size_t)A
             && std::fread(W1f.data(), 4, W1f.size(), f) == W1f.size()
             && std::fread(B1f.data(), 4, L1, f) == (size_t)L1
             && std::fread(W2f.data(), 4, L1, f) == (size_t)L1
             && std::fread(&B2f, 4, 1, f) == 1;
        std::fclose(f);
        if (!ok) { std::fprintf(stderr, "fast_nnue: truncated FNN1 file %s (or A/L1 != %d/%d)\n", path, A, L1); return false; }
        return quantize_fnn1(n, W0f, B0f, W1f, B1f, W2f, B2f, path);
    }
    std::fclose(f);
    // Stage 2 adds the pattern-generator format here.
    std::fprintf(stderr, "fast_nnue: %s: unknown magic %.4s (supported: FNN1)\n", path, magic);
    return false;
}

// CRC-32 (IEEE 802.3, reflected, as Python's zlib.crc32) of a whole file; false if unreadable.
inline bool file_crc32(const char *path, uint32_t &crc, long long &bytes) {
    static const auto table = [] {
        std::vector<uint32_t> t(256);
        for (uint32_t i = 0; i < 256; i++) {
            uint32_t c = i;
            for (int k = 0; k < 8; k++) c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
            t[i] = c;
        }
        return t;
    }();
    FILE *f = std::fopen(path, "rb");
    if (!f) return false;
    std::vector<unsigned char> buf(1 << 20);
    uint32_t c = 0xFFFFFFFFu;
    bytes = 0;
    size_t got;
    while ((got = std::fread(buf.data(), 1, buf.size(), f)) > 0) {
        for (size_t i = 0; i < got; i++) c = table[(c ^ buf[i]) & 0xFF] ^ (c >> 8);
        bytes += (long long)got;
    }
    const bool ok = !std::ferror(f);
    std::fclose(f);
    crc = c ^ 0xFFFFFFFFu;
    return ok;
}

// This side's net file (see the top of this file): resolved, checked readable and announced once.
inline std::once_flag g_path_once;
inline const char *g_path = nullptr;

inline const char *net_path() {
    std::call_once(g_path_once, [] {
        const char *env = std::getenv(kPathEnv);
        const char *from = kPathEnv;
        g_path = (env && *env) ? env : nullptr;
#ifdef FASTNNUE_NET_FILE
        if (!g_path) { g_path = FASTNNUE_NET_FILE; from = "compiled in"; }
#endif
        if (!g_path) {
            std::fprintf(stderr, "fast_nnue [%s]: %s not set and no net file compiled in\n", kSide, kPathEnv);
            std::exit(1);
        }
        uint32_t crc = 0;
        long long bytes = 0;
        if (!file_crc32(g_path, crc, bytes)) {
            std::fprintf(stderr, "fast_nnue [%s]: cannot read net %s (%s)\n", kSide, g_path, from);
            std::exit(1);
        }
        std::fprintf(stderr, "fast_nnue [%s]: net %s (%s), %lld bytes, crc32 %08x\n", kSide, g_path, from, bytes,
                     (unsigned)crc);
    });
    return g_path;
}

inline const Net &net() {
    std::call_once(g_once, [] {
        const char *path = net_path();
        if (!load_net(g_net, path)) { std::fprintf(stderr, "fast_nnue: cannot load %s\n", path); std::exit(1); }
    });
    return g_net;
}

// ---------------------------------------------------------------- kernels

// child = parent + sum(add) - sum(sub), 4 vectors (64 lanes) per tile.
inline void apply_rows(const int16_t *parent, int16_t *child, const int16_t *const *add, int na,
                       const int16_t *const *sub, int ns) {
    for (int t = 0; t < A; t += 64) {
        __m256i v0 = _mm256_load_si256((const __m256i *)(parent + t));
        __m256i v1 = _mm256_load_si256((const __m256i *)(parent + t + 16));
        __m256i v2 = _mm256_load_si256((const __m256i *)(parent + t + 32));
        __m256i v3 = _mm256_load_si256((const __m256i *)(parent + t + 48));
        for (int r = 0; r < na; r++) {
            const int16_t *w = add[r] + t;
            v0 = _mm256_add_epi16(v0, _mm256_load_si256((const __m256i *)(w)));
            v1 = _mm256_add_epi16(v1, _mm256_load_si256((const __m256i *)(w + 16)));
            v2 = _mm256_add_epi16(v2, _mm256_load_si256((const __m256i *)(w + 32)));
            v3 = _mm256_add_epi16(v3, _mm256_load_si256((const __m256i *)(w + 48)));
        }
        for (int r = 0; r < ns; r++) {
            const int16_t *w = sub[r] + t;
            v0 = _mm256_sub_epi16(v0, _mm256_load_si256((const __m256i *)(w)));
            v1 = _mm256_sub_epi16(v1, _mm256_load_si256((const __m256i *)(w + 16)));
            v2 = _mm256_sub_epi16(v2, _mm256_load_si256((const __m256i *)(w + 32)));
            v3 = _mm256_sub_epi16(v3, _mm256_load_si256((const __m256i *)(w + 48)));
        }
        _mm256_store_si256((__m256i *)(child + t), v0);
        _mm256_store_si256((__m256i *)(child + t + 16), v1);
        _mm256_store_si256((__m256i *)(child + t + 32), v2);
        _mm256_store_si256((__m256i *)(child + t + 48), v3);
    }
}

// Both perspectives, one row each (the common, non-deciding move).
inline void add_row2(const int16_t *p0, int16_t *c0, const int16_t *r0,
                     const int16_t *p1, int16_t *c1, const int16_t *r1) {
    for (int t = 0; t < A; t += 16) {
        _mm256_store_si256((__m256i *)(c0 + t), _mm256_add_epi16(_mm256_load_si256((const __m256i *)(p0 + t)),
                                                                 _mm256_load_si256((const __m256i *)(r0 + t))));
        _mm256_store_si256((__m256i *)(c1 + t), _mm256_add_epi16(_mm256_load_si256((const __m256i *)(p1 + t)),
                                                                 _mm256_load_si256((const __m256i *)(r1 + t))));
    }
}

inline int finish_output(const Net &n, int64_t out) {
    const int64_t x = out * 1000;
    return (int)(x >= 0 ? (x >> n.out_shift) : -((-x) >> n.out_shift));
}

// L1 clipped ReLU, shift, output layer and final scaling (shared by both dense kernels).
inline int output_layer(const Net &n, __m256i s0, __m256i s1, __m256i s2, __m256i s3) {
    const __m256i zero = _mm256_setzero_si256();
    const __m256i hmax = _mm256_set1_epi32(n.H1MAX);
    const __m256i half = _mm256_set1_epi32(n.h1_round);
    const __m128i sh = _mm_cvtsi32_si128(n.h1_shift);
    auto act1 = [&](__m256i s, int k) {
        s = _mm256_sra_epi32(_mm256_add_epi32(_mm256_min_epi32(_mm256_max_epi32(s, zero), hmax), half), sh);
        return _mm256_mullo_epi32(s, _mm256_load_si256((const __m256i *)(n.W2 + k)));
    };
    __m256i o = _mm256_add_epi32(_mm256_add_epi32(act1(s0, 0), act1(s1, 8)),
                                 _mm256_add_epi32(act1(s2, 16), act1(s3, 24)));
    __m128i o4 = _mm_add_epi32(_mm256_castsi256_si128(o), _mm256_extracti128_si256(o, 1));
    o4 = _mm_add_epi32(o4, _mm_shuffle_epi32(o4, 0x4E));
    o4 = _mm_add_epi32(o4, _mm_shuffle_epi32(o4, 0xB1));
    return finish_output(n, (int64_t)_mm_cvtsi128_si32(o4) + n.B2);
}

// AVX2 head: crelu(acc + con) for both perspectives with one bit per nonzero activation
// pair (movemask of the int32 view: a pair is skipped only when both lanes are 0), then the
// sparse pairwise dense layer walking the 256-bit mask with tzcnt/blsr, then the output.
// About 80% of the crelu outputs are 0 on real positions: ~96 of 256 pairs are nonzero
// (~80 with the lane pairing of permute_fnn1.py). The dense part is bound by the 4
// vpmaddwd + 4 weight loads per pair.
FNNUE_KERNEL inline int eval_avx(const Net &n, const int16_t *us, const int16_t *them, int constraint) {
    alignas(32) int16_t act[IN];
    uint64_t mask[IN / 128];  // one bit per activation pair
    const int16_t *con = n.W0[ROW_CON + constraint];
    const __m256i zero = _mm256_setzero_si256();
    const __m256i qa = _mm256_set1_epi16(n.QA);
    for (int v = 0; v < 2; v++) {
        const int16_t *acc = v == 0 ? us : them;
        int16_t *out = act + v * A;
        for (int w = 0; w < A / 128; w++) {
            uint64_t m = 0;
            for (int j = 0; j < 8; j++) {
                const int t = w * 128 + j * 16;
                __m256i x = _mm256_add_epi16(_mm256_load_si256((const __m256i *)(acc + t)),
                                             _mm256_load_si256((const __m256i *)(con + t)));
                x = _mm256_min_epi16(_mm256_max_epi16(x, zero), qa);
                _mm256_store_si256((__m256i *)(out + t), x);
                m |= (uint64_t)(uint32_t)_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpgt_epi32(x, zero)))
                     << (8 * j);
            }
            mask[v * (A / 128) + w] = m;
        }
    }
    __m256i s0 = _mm256_load_si256((const __m256i *)(n.B1));
    __m256i s1 = _mm256_load_si256((const __m256i *)(n.B1 + 8));
    __m256i s2 = _mm256_load_si256((const __m256i *)(n.B1 + 16));
    __m256i s3 = _mm256_load_si256((const __m256i *)(n.B1 + 24));
    const int32_t *act32 = (const int32_t *)act;
    for (int w = 0; w < IN / 128; w++) {
        uint64_t m = mask[w];
        while (m) {
            const int p = w * 64 + __builtin_ctzll(m);
            m &= m - 1;
            const __m256i a = _mm256_set1_epi32(act32[p]);
            const __m256i *wp = (const __m256i *)n.W1p[p];
            s0 = _mm256_add_epi32(s0, _mm256_madd_epi16(a, _mm256_load_si256(wp)));
            s1 = _mm256_add_epi32(s1, _mm256_madd_epi16(a, _mm256_load_si256(wp + 1)));
            s2 = _mm256_add_epi32(s2, _mm256_madd_epi16(a, _mm256_load_si256(wp + 2)));
            s3 = _mm256_add_epi32(s3, _mm256_madd_epi16(a, _mm256_load_si256(wp + 3)));
        }
    }
    return output_layer(n, s0, s1, s2, s3);
}

// 8-bit movemask -> the positions of its set bits (Stockfish's find_nnz idea).
struct NnzLut {
    alignas(16) uint16_t idx[256][8];
    NnzLut() {
        for (int m = 0; m < 256; m++) {
            int k = 0;
            for (int b = 0; b < 8; b++) idx[m][b] = 0;
            for (int b = 0; b < 8; b++)
                if (m >> b & 1) idx[m][k++] = (uint16_t)b;
        }
    }
};
inline const NnzLut g_nnz_lut;

// Alternative kernel (fast_bench compares it; -DFASTNNUE_LIST_KERNEL makes the search use
// it): the nonzero pair indices gathered into a list while transforming (Stockfish's
// find_nnz with an 8-bit LUT), then two pairs per iteration. Measured slower than the
// bitmask walk here (the list stores' addresses depend on a popcount chain).
FNNUE_KERNEL inline int eval_avx_list(const Net &n, const int16_t *us, const int16_t *them, int constraint) {
    alignas(32) int16_t act[IN];
    alignas(16) uint16_t list[NPAIR + 8];
    int count = 0;
    const int16_t *con = n.W0[ROW_CON + constraint];
    const __m256i zero = _mm256_setzero_si256();
    const __m256i qa = _mm256_set1_epi16(n.QA);
    __m128i base = _mm_setzero_si128();
    const __m128i step = _mm_set1_epi16(8);
    for (int v = 0; v < 2; v++) {
        const int16_t *acc = v == 0 ? us : them;
        int16_t *out = act + v * A;
        for (int t = 0; t < A; t += 16) {
            __m256i x = _mm256_add_epi16(_mm256_load_si256((const __m256i *)(acc + t)),
                                         _mm256_load_si256((const __m256i *)(con + t)));
            x = _mm256_min_epi16(_mm256_max_epi16(x, zero), qa);
            _mm256_store_si256((__m256i *)(out + t), x);
            const unsigned m = (unsigned)_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpgt_epi32(x, zero)));
            _mm_storeu_si128((__m128i *)(list + count),
                             _mm_add_epi16(_mm_load_si128((const __m128i *)g_nnz_lut.idx[m]), base));
            count += __builtin_popcount(m);
            base = _mm_add_epi16(base, step);
        }
    }
    __m256i s0 = _mm256_load_si256((const __m256i *)(n.B1));
    __m256i s1 = _mm256_load_si256((const __m256i *)(n.B1 + 8));
    __m256i s2 = _mm256_load_si256((const __m256i *)(n.B1 + 16));
    __m256i s3 = _mm256_load_si256((const __m256i *)(n.B1 + 24));
    const int32_t *act32 = (const int32_t *)act;
    int i = 0;
    for (; i + 1 < count; i += 2) {
        const int p0 = list[i], p1 = list[i + 1];
        const __m256i a0 = _mm256_set1_epi32(act32[p0]);
        const __m256i a1 = _mm256_set1_epi32(act32[p1]);
        const __m256i *w0 = (const __m256i *)n.W1p[p0];
        const __m256i *w1 = (const __m256i *)n.W1p[p1];
        s0 = _mm256_add_epi32(s0, _mm256_add_epi32(_mm256_madd_epi16(a0, _mm256_load_si256(w0)),
                                                   _mm256_madd_epi16(a1, _mm256_load_si256(w1))));
        s1 = _mm256_add_epi32(s1, _mm256_add_epi32(_mm256_madd_epi16(a0, _mm256_load_si256(w0 + 1)),
                                                   _mm256_madd_epi16(a1, _mm256_load_si256(w1 + 1))));
        s2 = _mm256_add_epi32(s2, _mm256_add_epi32(_mm256_madd_epi16(a0, _mm256_load_si256(w0 + 2)),
                                                   _mm256_madd_epi16(a1, _mm256_load_si256(w1 + 2))));
        s3 = _mm256_add_epi32(s3, _mm256_add_epi32(_mm256_madd_epi16(a0, _mm256_load_si256(w0 + 3)),
                                                   _mm256_madd_epi16(a1, _mm256_load_si256(w1 + 3))));
    }
    if (i < count) {
        const int p = list[i];
        const __m256i a = _mm256_set1_epi32(act32[p]);
        const __m256i *wp = (const __m256i *)n.W1p[p];
        s0 = _mm256_add_epi32(s0, _mm256_madd_epi16(a, _mm256_load_si256(wp)));
        s1 = _mm256_add_epi32(s1, _mm256_madd_epi16(a, _mm256_load_si256(wp + 1)));
        s2 = _mm256_add_epi32(s2, _mm256_madd_epi16(a, _mm256_load_si256(wp + 2)));
        s3 = _mm256_add_epi32(s3, _mm256_madd_epi16(a, _mm256_load_si256(wp + 3)));
    }
    return output_layer(n, s0, s1, s2, s3);
}

// Scalar reference of the same integer math, dense over all 2A inputs (no sparsity).
inline int eval_ref(const Net &n, const int32_t *us, const int32_t *them, int constraint) {
    int32_t act[IN];
    const int16_t *con = n.W0[ROW_CON + constraint];
    for (int i = 0; i < A; i++) {
        act[i] = std::min<int32_t>(std::max<int32_t>(us[i] + con[i], 0), n.QA);
        act[A + i] = std::min<int32_t>(std::max<int32_t>(them[i] + con[i], 0), n.QA);
    }
    int64_t out = n.B2;
    for (int k = 0; k < L1; k++) {
        int64_t s = n.B1[k];
        for (int i = 0; i < IN; i++) s += (int64_t)act[i] * n.W1[k][i];
        s = (std::min<int64_t>(std::max<int64_t>(s, 0), n.H1MAX) + n.h1_round) >> n.h1_shift;
        out += s * n.W2[k];
    }
    return finish_output(n, out);
}

// Row lists of a whole board for perspective P (bias not included).
template <typename Board>
inline int board_rows(const Board &b, int P, int *rows) {
    int nr = 0;
    const int decided = b.mini_board_states[0] | b.mini_board_states[1] | b.mini_board_states[2];
    for (int mb = 0; mb < 9; mb++) {
        if (decided >> mb & 1) {
            const int st = (b.mini_board_states[2] >> mb & 1) ? 2 : ((b.mini_board_states[0] >> mb & 1) ? 0 : 1);
            rows[nr++] = dec_row(mb, st, P);
            continue;
        }
        for (int owner = 0; owner < 2; owner++) {
            int m = b.mini_boards[mb].markers[owner];
            while (m) {
                rows[nr++] = cell_row(mb, __builtin_ctz(m), owner, P);
                m &= m - 1;
            }
        }
    }
    return nr;
}

// Scalar int32 from-scratch accumulator (the check's reference).
template <typename Board>
inline void scratch_ref(const Net &n, const Board &b, int P, int32_t *out) {
    int rows[96];
    const int nr = board_rows(b, P, rows);
    for (int i = 0; i < A; i++) out[i] = n.B0[i];
    for (int r = 0; r < nr; r++)
        for (int i = 0; i < A; i++) out[i] += n.W0[rows[r]][i];
}

// AVX2 from-scratch accumulator (root refresh and evaluate_board).
template <typename Board>
inline void scratch_avx(const Net &n, const Board &b, int P, int16_t *out) {
    int rows[96];
    const int nr = board_rows(b, P, rows);
    const int16_t *ptr[96];
    for (int r = 0; r < nr; r++) ptr[r] = n.W0[rows[r]];
    apply_rows(n.B0, out, ptr, nr, nullptr, 0);
}

// ---------------------------------------------------------------- check counters

struct CheckStats {
    std::atomic<long long> evals{0}, acc_mismatch{0}, out_mismatch{0}, decided_nodes{0}, drawn_nodes{0},
        free_nodes{0}, forced_nodes{0}, lazy_updates{0}, lazy_decided_updates{0}, max_chain{0}, roots{0},
        scratch_evals{0}, fallback_refresh{0}, cache_hits{0}, cache_mismatch{0}, key_reuse{0}, range_violations{0};
    ~CheckStats() {
        if (evals.load() + scratch_evals.load() == 0) return;
        std::fprintf(stderr,
                     "fast_nnue check: evals=%lld scratch_evals=%lld roots=%lld acc_mismatch=%lld out_mismatch=%lld "
                     "| evaluated nodes with a decided board=%lld with a drawn board=%lld free-move=%lld forced=%lld "
                     "| lazy updates=%lld (deciding %lld) max chain=%lld fallback refreshes=%lld "
                     "| cache hits=%lld cache mismatches=%lld | entry already held the position=%lld "
                     "int16 range violations=%lld side=%s\n",
                     evals.load(), scratch_evals.load(), roots.load(), acc_mismatch.load(), out_mismatch.load(),
                     decided_nodes.load(), drawn_nodes.load(), free_nodes.load(), forced_nodes.load(),
                     lazy_updates.load(), lazy_decided_updates.load(), max_chain.load(), fallback_refresh.load(),
                     cache_hits.load(), cache_mismatch.load(), key_reuse.load(), range_violations.load(), kSide);
    }
};
inline CheckStats g_check;

// ---------------------------------------------------------------- incremental stack

struct Dirty {
    uint8_t mb, sq, stm;
    int8_t decided;       // -1, or the mini_board_states index the move set
    uint16_t before[2];   // the miniboard's markers before the move
    uint64_t pkey;        // tt_hash of the position the move was made from (MoveUndo.tt_hash)
    uint64_t ckey;        // tt_hash of the position it led to
};

struct alignas(64) AccEntry {
    int16_t v[2][A];      // [absolute player P][lane]
};

template <typename Board>
inline int evaluate_board(const Board &b, int constraint);

struct Stack {
    std::vector<AccEntry> acc;
    // The empty board's tt_hash is 0, so 0 cannot mean "no position" in pos_key.
    static constexpr uint64_t kNoKey = 0x9E3779B97F4A7C15ull;
    uint64_t pos_key[MAXPLY];  // tt_hash of the position acc[ply] holds; kNoKey: none
    Dirty dirty[MAXPLY];   // dirty[j]: the last move recorded from ply j-1 to ply j
    int root = 0;

#if FASTNNUE_CACHE_BITS > 0
    struct CacheEntry {
        uint64_t key;
        int32_t eval, pad;
    };
    static constexpr uint64_t CMASK = (1u << FASTNNUE_CACHE_BITS) - 1;
    std::vector<CacheEntry> cache = std::vector<CacheEntry>(1u << FASTNNUE_CACHE_BITS, CacheEntry{0, 0, 0});
#endif

    Stack() : acc(MAXPLY) {
        net();
        for (uint64_t &k : pos_key) k = kNoKey;
        std::memset(dirty, 0, sizeof(dirty));
    }

    // From scratch at the board's own ply.
    template <typename Board>
    void refresh_at(const Board &b) {
        const Net &n = net();
        const int ply = b.n_moves;
        scratch_avx(n, b, 0, acc[ply].v[0]);
        scratch_avx(n, b, 1, acc[ply].v[1]);
        pos_key[ply] = b.tt_hash;
    }

    template <typename Board>
    void refresh_root(const Board &b) {
        root = b.n_moves;
        refresh_at(b);
#ifdef FASTNNUE_CHECK
        g_check.roots++;
#endif
    }

    // Called by make_move_fast(FastBoard&) just before board.n_moves++: the board is otherwise the
    // child's (its tt_hash is final); parent_key is the parent's tt_hash (MoveUndo.tt_hash).
    template <typename Board>
    void on_make(const Board &b, int mb, int sq, int stm, int decided, int before_stm, int before_other,
                 uint64_t parent_key) {
        const int n = b.n_moves;
        Dirty &d = dirty[n + 1];
        d.mb = (uint8_t)mb;
        d.sq = (uint8_t)sq;
        d.stm = (uint8_t)stm;
        d.decided = (int8_t)decided;
        d.before[stm] = (uint16_t)before_stm;
        d.before[stm ^ 1] = (uint16_t)before_other;
        d.pkey = parent_key;
        d.ckey = b.tt_hash;
#if FASTNNUE_CACHE_BITS > 0 && !defined(FASTNNUE_NO_CACHE_PREFETCH)
        // evaluate_keyed at the child reads this entry first
        _mm_prefetch((const char *)&cache[b.tt_hash & CMASK], _MM_HINT_T0);
#endif
#if defined(FASTNNUE_EAGER)
        // Variant for the before/after measurement: update on every make (Stockfish before
        // lazy accumulators) instead of at evaluation.
        if (pos_key[n] == parent_key) {
            update(g_net, n + 1);
            pos_key[n + 1] = b.tt_hash;
        }
#elif defined(FASTNNUE_PREFETCH)
        // Variant: start loading the two rows a non-deciding move will add.
        if (decided < 0) {
            const int16_t *r0 = g_net.W0[(mb * 9 + sq) * 2];
            for (int l = 0; l < 2 * A; l += 32) _mm_prefetch((const char *)(r0 + l), _MM_HINT_T0);
        }
#endif
    }

    void update(const Net &n, int j) {
        const Dirty &d = dirty[j];
        const AccEntry &par = acc[j - 1];
        AccEntry &ch = acc[j];
        if (d.decided < 0) {
            const int base = (d.mb * 9 + d.sq) * 2;
            // perspective stm sees "mine" (even row), the other "theirs" (odd row)
            add_row2(par.v[0], ch.v[0], n.W0[base + (d.stm != 0)], par.v[1], ch.v[1], n.W0[base + (d.stm != 1)]);
            return;
        }
        for (int P = 0; P < 2; P++) {
            const int16_t *sub[9];
            int ns = 0;
            for (int owner = 0; owner < 2; owner++) {
                int m = d.before[owner];
                while (m) {
                    sub[ns++] = n.W0[cell_row(d.mb, __builtin_ctz(m), owner, P)];
                    m &= m - 1;
                }
            }
            const int16_t *add[1] = {n.W0[dec_row(d.mb, d.decided, P)]};
            apply_rows(par.v[P], ch.v[P], add, 1, sub, ns);
        }
    }

    // Make acc[b.n_moves] hold b. Walk back along the recorded moves while each one leads to the
    // position needed (its ckey), stop at the first entry that holds that move's parent (pkey),
    // and replay from there. Within a search that entry is at the latest at the refreshed root.
    // A chain that breaks first (a search entered without refresh_root, a board this stack never
    // saw) refreshes from scratch at the board's ply.
    template <typename Board>
    void sync(const Net &n, const Board &b) {
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
            update(n, j);
            pos_key[j] = dirty[j].ckey;
#ifdef FASTNNUE_CHECK
            g_check.lazy_updates++;
            if (dirty[j].decided >= 0) g_check.lazy_decided_updates++;
#endif
        }
    }

    template <typename Board>
    int evaluate(const Board &b, int constraint) {
        const Net &n = net();
        const int cur = b.n_moves;
        if (pos_key[cur] != b.tt_hash) sync(n, b);
#ifdef FASTNNUE_CHECK
        else g_check.key_reuse++;
#endif
        const int stm = cur & 1;
#ifdef FASTNNUE_LIST_KERNEL
        const int e = eval_avx_list(n, acc[cur].v[stm], acc[cur].v[stm ^ 1], constraint);
#else
        const int e = eval_avx(n, acc[cur].v[stm], acc[cur].v[stm ^ 1], constraint);
#endif
#ifdef FASTNNUE_CHECK
        check_node(n, b, constraint, e);
#endif
        return e;
    }

    // evaluate() behind a direct-mapped cache of (search key -> eval), per engine, kept across
    // searches (the net is fixed). KEY must determine the position as the features see it:
    // the FastBoard tt_hash does (stones of live boards, decided states, side to move and
    // the constraint; stones of decided boards are dropped from it exactly as the features
    // drop them). A hit skips the lazy updates as well as the head. 0 = no key, no cache.
    template <typename Board>
    int evaluate_keyed(const Board &b, int constraint, uint64_t key) {
#if FASTNNUE_CACHE_BITS > 0
        CacheEntry &ce = cache[key & CMASK];
        if (key && ce.key == key) {
#ifdef FASTNNUE_CHECK
            const int e = evaluate_board(b, constraint);  // from scratch: the stack sees the release calls
            g_check.cache_hits++;
            if (e != ce.eval) {
                g_check.cache_mismatch++;
                std::fprintf(stderr, "fast_nnue CHECK: cached eval %d != computed %d\n", ce.eval, e);
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
    void check_node(const Net &n, const Board &b, int constraint, int e) {
        const int cur = b.n_moves, stm = cur & 1;
        int32_t ref[2][A];
        scratch_ref(n, b, 0, ref[0]);
        scratch_ref(n, b, 1, ref[1]);
        for (int P = 0; P < 2; P++)
            for (int i = 0; i < A; i++) {
                if (ref[P][i] != acc[cur].v[P][i]) {
                    g_check.acc_mismatch++;
                    std::fprintf(stderr, "fast_nnue CHECK: accumulator mismatch at n_moves=%d P=%d lane=%d: "
                                 "incremental %d scratch %d (root %d)\n", cur, P, i, acc[cur].v[P][i], ref[P][i], root);
                    std::abort();
                }
                const int32_t x = ref[P][i] + n.W0[ROW_CON + constraint][i];  // what the int16 transform adds
                if (x < -32768 || x > 32767) {
                    g_check.range_violations++;
                    std::fprintf(stderr, "fast_nnue CHECK: acc + con = %d outside int16 at n_moves=%d\n", x, cur);
                    std::abort();
                }
            }
        const int r = eval_ref(n, ref[stm], ref[stm ^ 1], constraint);
        if (r != e) {
            g_check.out_mismatch++;
            std::fprintf(stderr, "fast_nnue CHECK: output mismatch at n_moves=%d: avx %d scalar %d\n", cur, e, r);
            std::abort();
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
template <typename Board>
inline int evaluate_board(const Board &b, int constraint) {
    const Net &n = net();
    alignas(64) int16_t acc[2][A];
    scratch_avx(n, b, 0, acc[0]);
    scratch_avx(n, b, 1, acc[1]);
    const int stm = b.n_moves & 1;
    const int e = eval_avx(n, acc[stm], acc[stm ^ 1], constraint);
#ifdef FASTNNUE_CHECK
    int32_t ref[2][A];
    scratch_ref(n, b, 0, ref[0]);
    scratch_ref(n, b, 1, ref[1]);
    const int r = eval_ref(n, ref[stm], ref[stm ^ 1], constraint);
    if (r != e) {
        g_check.out_mismatch++;
        std::fprintf(stderr, "fast_nnue CHECK: scratch output mismatch: avx %d scalar %d\n", e, r);
        std::abort();
    }
    g_check.scratch_evals++;
#endif
    return e;
}

}  // namespace fnnue
