#pragma once
// The three helpers of mini_eval_d16.hpp the CodinGame bot still needs now that its evaluation is the
// NNUE (nnue_b64.hpp) and not the MiniNet: the CJK14 decoder (macro_eval.hpp's table and play_book.hpp's
// book payloads), the constraint helper and the macro head's horizontal sum. Copied verbatim from
// mini_eval_d16.hpp, so that codingame_nnue.cpp does not carry the MiniNet's 24,000-character payload.
// Only the CodinGame bot includes this; a translation unit must not include it and mini_eval_d16.hpp
// together (the engines, crossfish_dev.hpp and crossfish_prev.hpp, include mini_eval_d16.hpp).

#include <cstdint>
#include <cstring>
#include <cmath>
#include <immintrin.h>

template <typename Board>
static int d16_mini_board_constraint(const Board &b) {
    if (b.n_moves == 0 || b.prev_move_was_pass) return 9;
    int sent = b.move_history.top().square;
    int oop = b.mini_board_states[0] | b.mini_board_states[1] | b.mini_board_states[2];
    if ((oop & (1 << sent)) != 0) return 9;
    return sent;
}

static int d16_mini_cjk_decode(
    const char *s, unsigned char *out, int out_max) {
    int n = 0;
    int bits = 0;
    uint32_t acc = 0;
    for (const unsigned char *p = (const unsigned char *)s; *p; p++) {
        if ((*p & 0xF0) != 0xE0) continue;
        uint32_t code = ((p[0] & 15u) << 12) | ((p[1] & 63u) << 6)
                      | (p[2] & 63u);
        p += 2;
        acc = (acc << 14) | (code - 0x4E00u);
        bits += 14;
        while (bits >= 8) {
            if (n >= out_max) return -1;
            bits -= 8;
            out[n++] = (unsigned char)(acc >> bits);
        }
    }
    return n;
}

__attribute__((always_inline)) static inline float d16_mini_hsum256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
    return _mm_cvtss_f32(s);
}
