#!/usr/bin/env python3
"""Emit the packed AVX2 macro-context residual evaluator."""

from __future__ import annotations

import argparse
import base64
import struct
from pathlib import Path

import numpy as np
import torch

import nnue_emit_mininet_cg as legacy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument(
        "-o",
        "--out",
        type=Path,
        default=Path("cpp_impl/macro_eval.hpp"),
    )
    parser.add_argument("--scale", type=float, default=1.25)
    parser.add_argument("--clip", type=int, default=2000)
    args = parser.parse_args()

    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    state = checkpoint["state_dict"]
    embedding = state["embedding.weight"].numpy()
    constraint = state["constraint.weight"].numpy()
    hidden_weight = state["hidden.weight"].numpy()
    hidden_bias = state["hidden.bias"].numpy()
    output_weight = state["out.weight"].numpy().reshape(-1)
    output_bias = float(state["out.bias"])

    projected = (embedding @ hidden_weight.T).reshape(9, 4, -1)
    projected_constraint = constraint @ hidden_weight.T
    if projected.shape[-1] != 16:
        raise SystemExit(
            f"compact runtime expects H=16, got H={projected.shape[-1]}"
        )
    empty_hidden = (
        hidden_bias
        + projected[:, 0, :].sum(axis=0)
        + projected_constraint[9]
    )
    empty_raw = output_bias + float(
        output_weight @ np.maximum(empty_hidden, 0.0)
    )
    scaled_output = output_weight * args.scale
    scaled_bias = (output_bias - empty_raw) * args.scale

    arrays = (
        np.asarray(hidden_bias, dtype="<f4"),
        np.asarray(projected_constraint, dtype="<f4"),
        np.asarray(projected, dtype="<f4"),
        np.asarray(scaled_output, dtype="<f4"),
    )
    blob = b"".join(value.tobytes() for value in arrays)
    blob += struct.pack("<f", float(scaled_bias))
    packed_b64 = legacy.wrap_b64(
        base64.b64encode(blob).decode("ascii")
    )

    text = f'''#pragma once
// Compact super-board/constraint residual head. The checkpoint's embeddings
// are preprojected through the hidden layer and packed as exact float32 data.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <immintrin.h>

static const char MACRO_PACK_B64[] = R"MACRO(
{packed_b64}
)MACRO";

alignas(32) static float MACRO_BASE[16];
alignas(32) static float MACRO_CONSTR[10][16];
alignas(32) static float MACRO_EMB[9][4][16];
alignas(32) static float MACRO_OUT[16];
static float MACRO_BIAS = 0;
static bool MACRO_READY = false;
static constexpr int MACRO_CLIP = {args.clip};
static constexpr int MACRO_KEY_STATES = 1 << 18;
alignas(64) static int16_t MACRO_SCORE[10][MACRO_KEY_STATES];

static int macro_finish_hidden(__m256 h0, __m256 h1) {{
    const __m256 zero = _mm256_setzero_ps();
    h0 = _mm256_max_ps(h0, zero);
    h1 = _mm256_max_ps(h1, zero);
    h0 = _mm256_mul_ps(h0, _mm256_load_ps(MACRO_OUT));
    h1 = _mm256_mul_ps(h1, _mm256_load_ps(MACRO_OUT + 8));
    int value = (int)std::lround(
        MACRO_BIAS + d16_mini_hsum256(_mm256_add_ps(h0, h1)));
    return std::max(-MACRO_CLIP, std::min(MACRO_CLIP, value));
}}

static bool macro_load_packed() {{
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
    for (int constraint = 0; constraint < 10; constraint++) {{
        for (int key = 0; key < MACRO_KEY_STATES; key++) {{
            __m256 h0 = _mm256_add_ps(
                _mm256_load_ps(MACRO_BASE),
                _mm256_load_ps(MACRO_CONSTR[constraint]));
            __m256 h1 = _mm256_add_ps(
                _mm256_load_ps(MACRO_BASE + 8),
                _mm256_load_ps(MACRO_CONSTR[constraint] + 8));
            int packed = key;
            for (int mb = 0; mb < 9; mb++, packed >>= 2) {{
                int cls = packed & 3;
                h0 = _mm256_add_ps(
                    h0, _mm256_load_ps(MACRO_EMB[mb][cls]));
                h1 = _mm256_add_ps(
                    h1, _mm256_load_ps(MACRO_EMB[mb][cls] + 8));
            }}
            MACRO_SCORE[constraint][key] =
                (int16_t)macro_finish_hidden(h0, h1);
        }}
    }}
    MACRO_READY = true;
    return true;
}}

static int evaluate_macro_key(int constraint, int key) {{
    if (!MACRO_READY && !macro_load_packed()) return 0;
    return MACRO_SCORE[constraint][key];
}}

template <typename Board>
static int evaluate_macro_fast(const Board &board) {{
    if (!MACRO_READY && !macro_load_packed()) return 0;
    const int stm = board.n_moves & 1;
    const int constraint = d16_mini_board_constraint(board);
    __m256 h0 = _mm256_add_ps(
        _mm256_load_ps(MACRO_BASE),
        _mm256_load_ps(MACRO_CONSTR[constraint]));
    __m256 h1 = _mm256_add_ps(
        _mm256_load_ps(MACRO_BASE + 8),
        _mm256_load_ps(MACRO_CONSTR[constraint] + 8));
    for (int mb = 0; mb < 9; mb++) {{
        const int bit = 1 << mb;
        int cls = 0;
        if (board.mini_board_states[stm] & bit) cls = 1;
        else if (board.mini_board_states[stm ^ 1] & bit) cls = 2;
        else if (board.mini_board_states[2] & bit) cls = 3;
        h0 = _mm256_add_ps(
            h0, _mm256_load_ps(MACRO_EMB[mb][cls]));
        h1 = _mm256_add_ps(
            h1, _mm256_load_ps(MACRO_EMB[mb][cls] + 8));
    }}
    return macro_finish_hidden(h0, h1);
}}
'''
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8", newline="\n")
    print(
        f"wrote {args.out} chars={len(text)} blob={len(blob)} "
        f"scale={args.scale} clip={args.clip} empty_raw={empty_raw:.3f}"
    )


if __name__ == "__main__":
    main()
