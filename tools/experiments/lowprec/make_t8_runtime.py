#!/usr/bin/env python3
"""Write a speed-test copy of cpp_impl/nnue_b64.hpp whose pattern tables (T, F, DEC) are int8 with one
power-of-two step per lane (T8MODE 2), or the same rounded values kept in int16 (T8MODE 1, the control).

Both modes compute bit-identical evals (the int8 value times the lane's multiplier equals the control's int16
value, and int16 arithmetic wraps identically), so a fixed-depth search walks the same tree in both and the
time difference is the table format alone: half the bytes per row (one cache line instead of two) against a
widen-and-multiply per row. This is a measurement tool, not a ship candidate: the rounded tables are W1's
tables post-training-rounded, not a net trained for them.

  python3 make_t8_runtime.py SRC_CPP_DIR OUT_CPP_DIR    (a copy of cpp_impl; only nnue_b64.hpp changes)
"""
import re
import shutil
import sys
from pathlib import Path

src, out = Path(sys.argv[1]), Path(sys.argv[2])
if out.exists():
    shutil.rmtree(out)
shutil.copytree(src, out)
p = out / "nnue_b64.hpp"
s = p.read_text()


def sub(old, new, count=1):
    global s
    assert s.count(old) == count, (old, s.count(old))
    s = s.replace(old, new)


sub("alignas(64) static int16_t T[9][NPAT][A];",
    "#ifndef T8MODE\n#define T8MODE 1\n#endif\n#if T8MODE == 2\n"
    "alignas(64) static int8_t T[9][NPAT][A];\nalignas(64) static int8_t F[NPAT][A];\nalignas(64) static int8_t DEC8[27][A];\n"
    "alignas(64) static int16_t LMUL[A];\nstatic int16_t (*TB)[NPAT][A] = new int16_t[9][NPAT][A]();\n"
    "static int16_t (*FB)[A] = new int16_t[NPAT][A]();\n#else\nalignas(64) static int16_t T[9][NPAT][A];\n"
    "alignas(64) static int16_t F[NPAT][A];\n#define TB T\n#define FB F\n#endif")
sub("alignas(64) static int16_t F[NPAT][A];\nstatic int16_t FP[NPAT];", "static int16_t FP[NPAT];")
sub("put(row, T[m][p], &TP[m][p]);", "put(row, TB[m][p], &TP[m][p]);")
sub("put(row, F[p], &FP[p]);", "put(row, FB[p], &FP[p]);")
# quantize after DEC is built
sub("    put(g.bias, BIAS, &BIASP);\n",
    "    put(g.bias, BIAS, &BIASP);\n"
    "    for (int l = 0; l < A; l++) {  // per-lane power-of-two step so every T, F, DEC value fits int8\n"
    "        int amax = 0;\n"
    "        for (int m = 0; m < 9; m++) for (int q = 0; q < NPAT; q++) amax = std::max(amax, std::abs((int)TB[m][q][l]));\n"
    "        for (int q = 0; q < NPAT; q++) amax = std::max(amax, std::abs((int)FB[q][l]));\n"
    "        for (int r = 0; r < 27; r++) amax = std::max(amax, std::abs((int)DEC[r][l]));\n"
    "        int k = 0;\n"
    "        while (((amax + (k ? 1 << (k - 1) : 0)) >> k) > 127) k++;\n"
    "        auto rq = [&](int16_t &v) { int q = (v + (k ? 1 << (k - 1) : 0)) >> k; q = q > 127 ? 127 : (q < -128 ? -128 : q); v = (int16_t)(q << k); return q; };\n"
    "#if T8MODE == 2\n"
    "        LMUL[l] = (int16_t)(1 << k);\n"
    "        for (int m = 0; m < 9; m++) for (int q = 0; q < NPAT; q++) T[m][q][l] = (int8_t)rq(TB[m][q][l]);\n"
    "        for (int q = 0; q < NPAT; q++) F[q][l] = (int8_t)rq(FB[q][l]);\n"
    "        for (int r = 0; r < 27; r++) DEC8[r][l] = (int8_t)rq(DEC[r][l]);\n"
    "#else\n"
    "        for (int m = 0; m < 9; m++) for (int q = 0; q < NPAT; q++) rq(T[m][q][l]);\n"
    "        for (int q = 0; q < NPAT; q++) rq(F[q][l]);\n"
    "        for (int r = 0; r < 27; r++) rq(DEC[r][l]);\n"
    "#endif\n"
    "    }\n")
# scratch
sub("""    const int16_t *rows[9];
    ps = BIASP;""", """#if T8MODE == 2
    const int8_t *rows[9];
#else
    const int16_t *rows[9];
#endif
    ps = BIASP;""")
sub("            rows[mb] = DEC[d];", "#if T8MODE == 2\n            rows[mb] = DEC8[d];\n#else\n            rows[mb] = DEC[d];\n#endif")
sub("""    for (int t = 0; t < A; t += 16) {
        __m256i v = _mm256_load_si256((const __m256i *)(BIAS + t));
        for (int r = 0; r < 9; r++) v = _mm256_add_epi16(v, _mm256_load_si256((const __m256i *)(rows[r] + t)));
        _mm256_store_si256((__m256i *)(out + t), v);
    }""", """    for (int t = 0; t < A; t += 16) {
#if T8MODE == 2
        __m256i v = _mm256_setzero_si256();
        for (int r = 0; r < 9; r++) v = _mm256_add_epi16(v, _mm256_cvtepi8_epi16(_mm_load_si128((const __m128i *)(rows[r] + t))));
        v = _mm256_add_epi16(_mm256_load_si256((const __m256i *)(BIAS + t)), _mm256_mullo_epi16(v, _mm256_load_si256((const __m256i *)(LMUL + t))));
#else
        __m256i v = _mm256_load_si256((const __m256i *)(BIAS + t));
        for (int r = 0; r < 9; r++) v = _mm256_add_epi16(v, _mm256_load_si256((const __m256i *)(rows[r] + t)));
#endif
        _mm256_store_si256((__m256i *)(out + t), v);
    }""")
# eval_with: widen the forced rows
sub("""        fu = F[ps];
        ft = F[pt];""", """#if T8MODE == 2
        alignas(32) static thread_local int16_t fbuf[2][A];
        for (int t = 0; t < A; t += 16) {
            const __m256i mul = _mm256_load_si256((const __m256i *)(LMUL + t));
            _mm256_store_si256((__m256i *)(fbuf[0] + t), _mm256_mullo_epi16(_mm256_cvtepi8_epi16(_mm_load_si128((const __m128i *)(F[ps] + t))), mul));
            _mm256_store_si256((__m256i *)(fbuf[1] + t), _mm256_mullo_epi16(_mm256_cvtepi8_epi16(_mm_load_si128((const __m128i *)(F[pt] + t))), mul));
        }
        fu = fbuf[0];
        ft = fbuf[1];
#else
        fu = F[ps];
        ft = F[pt];
#endif""")
# prefetch: one line per int8 row
sub("""    __attribute__((always_inline)) static void prefetch_row(const int16_t *r) {
        _mm_prefetch((const char *)r, _MM_HINT_T0);
        _mm_prefetch((const char *)r + 64, _MM_HINT_T0);
    }""", """    __attribute__((always_inline)) static void prefetch_row(const int16_t *r) {
        _mm_prefetch((const char *)r, _MM_HINT_T0);
        _mm_prefetch((const char *)r + 64, _MM_HINT_T0);
    }
    __attribute__((always_inline)) static void prefetch_row(const int8_t *r) { _mm_prefetch((const char *)r, _MM_HINT_T0); }""")
# update
sub("        const int16_t *s0 = T[d.mb][old0], *s1 = T[d.mb][old1];",
    "#if T8MODE == 2\n        const int8_t *s0 = T[d.mb][old0], *s1 = T[d.mb][old1];\n#else\n"
    "        const int16_t *s0 = T[d.mb][old0], *s1 = T[d.mb][old1];\n#endif")
sub("        const int16_t *a0, *a1;", "#if T8MODE == 2\n        const int8_t *a0, *a1;\n#else\n        const int16_t *a0, *a1;\n#endif")
sub("""            a0 = DEC[r0];
            a1 = DEC[r1];""", """#if T8MODE == 2
            a0 = DEC8[r0];
            a1 = DEC8[r1];
#else
            a0 = DEC[r0];
            a1 = DEC[r1];
#endif""")
sub("        sub_add2(par.v[0], ch.v[0], a0, s0, par.v[1], ch.v[1], a1, s1);",
    """#if T8MODE == 2
        for (int t = 0; t < A; t += 16) {
            const __m256i mul = _mm256_load_si256((const __m256i *)(LMUL + t));
            const __m256i d0 = _mm256_sub_epi16(_mm256_cvtepi8_epi16(_mm_load_si128((const __m128i *)(a0 + t))),
                                                _mm256_cvtepi8_epi16(_mm_load_si128((const __m128i *)(s0 + t))));
            const __m256i d1 = _mm256_sub_epi16(_mm256_cvtepi8_epi16(_mm_load_si128((const __m128i *)(a1 + t))),
                                                _mm256_cvtepi8_epi16(_mm_load_si128((const __m128i *)(s1 + t))));
            _mm256_store_si256((__m256i *)(ch.v[0] + t), _mm256_add_epi16(_mm256_load_si256((const __m256i *)(par.v[0] + t)), _mm256_mullo_epi16(d0, mul)));
            _mm256_store_si256((__m256i *)(ch.v[1] + t), _mm256_add_epi16(_mm256_load_si256((const __m256i *)(par.v[1] + t)), _mm256_mullo_epi16(d1, mul)));
        }
#else
        sub_add2(par.v[0], ch.v[0], a0, s0, par.v[1], ch.v[1], a1, s1);
#endif""")
p.write_text(s)
print("wrote", p)
