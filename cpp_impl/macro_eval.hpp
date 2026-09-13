#pragma once
// Compact super-board/constraint residual head. The checkpoint's embeddings
// are preprojected through the hidden layer and packed as exact float32 data.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <immintrin.h>

static const char MACRO_PACK_B64[] = R"MACRO(
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
ww==
)MACRO";

alignas(32) static float MACRO_BASE[16];
alignas(32) static float MACRO_CONSTR[10][16];
alignas(32) static float MACRO_EMB[9][4][16];
alignas(32) static float MACRO_OUT[16];
static float MACRO_BIAS = 0;
static bool MACRO_READY = false;
static constexpr int MACRO_CLIP = 2000;
static constexpr int MACRO_KEY_STATES = 1 << 18;
alignas(64) static int16_t MACRO_SCORE[10][MACRO_KEY_STATES];

static int macro_finish_hidden(__m256 h0, __m256 h1) {
    const __m256 zero = _mm256_setzero_ps();
    h0 = _mm256_max_ps(h0, zero);
    h1 = _mm256_max_ps(h1, zero);
    h0 = _mm256_mul_ps(h0, _mm256_load_ps(MACRO_OUT));
    h1 = _mm256_mul_ps(h1, _mm256_load_ps(MACRO_OUT + 8));
    int value = (int)std::lround(
        MACRO_BIAS + d16_mini_hsum256(_mm256_add_ps(h0, h1)));
    return std::max(-MACRO_CLIP, std::min(MACRO_CLIP, value));
}

static bool macro_load_packed() {
    if (MACRO_READY) return true;
    static unsigned char buf[4096];
    int count = d16_mini_b64_decode(
        MACRO_PACK_B64, buf, (int)sizeof(buf));
    const int need = (16 + 10 * 16 + 9 * 4 * 16 + 16 + 1) * 4;
    if (count < need) return false;
    int off = 0;
    memcpy(MACRO_BASE, buf + off, sizeof(MACRO_BASE));
    off += sizeof(MACRO_BASE);
    memcpy(MACRO_CONSTR, buf + off, sizeof(MACRO_CONSTR));
    off += sizeof(MACRO_CONSTR);
    memcpy(MACRO_EMB, buf + off, sizeof(MACRO_EMB));
    off += sizeof(MACRO_EMB);
    memcpy(MACRO_OUT, buf + off, sizeof(MACRO_OUT));
    off += sizeof(MACRO_OUT);
    memcpy(&MACRO_BIAS, buf + off, sizeof(MACRO_BIAS));
    for (int constraint = 0; constraint < 10; constraint++) {
        for (int key = 0; key < MACRO_KEY_STATES; key++) {
            __m256 h0 = _mm256_add_ps(
                _mm256_load_ps(MACRO_BASE),
                _mm256_load_ps(MACRO_CONSTR[constraint]));
            __m256 h1 = _mm256_add_ps(
                _mm256_load_ps(MACRO_BASE + 8),
                _mm256_load_ps(MACRO_CONSTR[constraint] + 8));
            int packed = key;
            for (int mb = 0; mb < 9; mb++, packed >>= 2) {
                int cls = packed & 3;
                h0 = _mm256_add_ps(
                    h0, _mm256_load_ps(MACRO_EMB[mb][cls]));
                h1 = _mm256_add_ps(
                    h1, _mm256_load_ps(MACRO_EMB[mb][cls] + 8));
            }
            MACRO_SCORE[constraint][key] =
                (int16_t)macro_finish_hidden(h0, h1);
        }
    }
    MACRO_READY = true;
    return true;
}

static int evaluate_macro_key(int constraint, int key) {
    if (!MACRO_READY && !macro_load_packed()) return 0;
    return MACRO_SCORE[constraint][key];
}

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
