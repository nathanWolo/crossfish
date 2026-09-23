#!/usr/bin/env python3
"""Emit the packed H8 MiniNet runtime header.

The CodinGame evaluator stores 19,683 local-board embeddings as 256
centroids.  Clustering is performed in first-layer projection space, while
centroids remain means in embedding space.  Code zero is reserved for the
empty local board so the most common state is reconstructed exactly.

The hot path preprojects the network's first layer and stores the common
live/inactive contribution as eight int32 lanes.  Super-board and active-board
effects are added as small deltas, avoiding feature reconstruction and a much
larger combined lookup table at every qsearch leaf.
"""

from __future__ import annotations

import argparse
import re
import struct
from pathlib import Path

import numpy as np

import nnue_emit_mininet_cg as legacy
from nnue_cjk14 import CJK14_DECODER, encode_cjk14, wrap_cjk14


def load_cfm2(path: Path):
    blob = path.read_bytes()
    if blob[:4] != b"CFM2":
        raise SystemExit(f"bad MiniNet magic in {path}")
    d, h = struct.unpack_from("<ii", blob, 4)
    off = 12

    def read_floats(count: int) -> np.ndarray:
        nonlocal off
        end = off + count * 4
        if end > len(blob):
            raise SystemExit(f"truncated MiniNet in {path}")
        value = np.frombuffer(blob[off:end], dtype="<f4").copy()
        off = end
        return value

    emb = read_floats(legacy.N_IDX * d).reshape(legacy.N_IDX, d)
    super_e = read_floats(4 * d)
    loc = read_floats(9 * d)
    constr = read_floats(10 * d)
    active = read_floats(2 * d)
    w1 = read_floats(h * 10 * d)
    b1 = read_floats(h)
    w2 = read_floats(h)
    if off + 4 > len(blob):
        raise SystemExit(f"truncated MiniNet bias in {path}")
    b2 = struct.unpack_from("<f", blob, off)[0]
    return d, h, emb, super_e, loc, constr, active, w1, b1, w2, b2


def mini_forward(
    d: int,
    h: int,
    emb: np.ndarray,
    super_e: np.ndarray,
    loc: np.ndarray,
    constr: np.ndarray,
    active: np.ndarray,
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: float,
    ids: np.ndarray,
    supers: np.ndarray,
    constraint: int,
) -> float:
    x = np.empty(10 * d, dtype=np.float32)
    for mb in range(9):
        is_active = int(constraint == mb)
        x[mb * d : (mb + 1) * d] = (
            emb[ids[mb]]
            + super_e[supers[mb] * d : (supers[mb] + 1) * d]
            + loc[mb * d : (mb + 1) * d]
            + active[is_active * d : (is_active + 1) * d]
        )
    x[9 * d :] = constr[constraint * d : (constraint + 1) * d]
    hidden = w1.reshape(h, 10 * d) @ x + b1
    return float(w2 @ np.maximum(hidden, 0.0) + b2)


def empty_output(
    d: int,
    h: int,
    emb: np.ndarray,
    super_e: np.ndarray,
    loc: np.ndarray,
    constr: np.ndarray,
    active: np.ndarray,
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: float,
) -> float:
    return mini_forward(
        d,
        h,
        emb,
        super_e,
        loc,
        constr,
        active,
        w1,
        b1,
        w2,
        b2,
        np.zeros(9, dtype=np.int32),
        np.zeros(9, dtype=np.int32),
        9,
    )


def projected_centroids(
    emb: np.ndarray,
    w1: np.ndarray,
    w2: np.ndarray,
    d: int,
    h: int,
    reserve_empty: bool,
) -> tuple[np.ndarray, np.ndarray]:
    rows = w1.reshape(h, 10, d)[:, :9, :]
    features = np.einsum("nd,hmd->nhm", emb, rows, optimize=True)
    features *= np.abs(w2)[None, :, None]
    features = features.reshape(len(emb), -1)

    if reserve_empty:
        _, rest_codes = legacy.kmeans(
            features[1:], legacy.K - 1, seed=0, iters=40
        )
        codes = np.empty(len(emb), dtype=np.uint8)
        codes[0] = 0
        codes[1:] = rest_codes + 1
    else:
        _, codes = legacy.kmeans(
            features, legacy.K, seed=0, iters=40
        )

    centroids = np.empty((legacy.K, d), dtype=np.float32)
    for code in range(legacy.K):
        members = codes == code
        if not members.any():
            raise SystemExit(f"empty centroid {code}")
        centroids[code] = emb[members].mean(axis=0)
    return centroids, codes


FACTOR_DECLS = """\
static constexpr int MN_FACTOR_SCALE = 192;
// Common live/inactive projection: 9*256*H*4 bytes.
alignas(64) static int32_t MN_FACTOR_CODE[9][MN_K][MN_H];
alignas(64) static int32_t MN_FACTOR_INIT[10][MN_H];
alignas(64) static int32_t MN_FACTOR_SUPER[9][4][MN_H];
alignas(64) static int32_t MN_FACTOR_ACTIVE[9][MN_H];
"""


FACTOR_BUILD = r'''

    for (int c = 0; c < 10; c++) {
        for (int h = 0; h < MN_H; h++) {
            MN_FACTOR_INIT[c][h] = (int32_t)std::lround(
                (MN_FAST_BASE[h] + MN_FAST_CONSTR[c][h])
                * MN_FACTOR_SCALE);
        }
    }
    for (int mb = 0; mb < 9; mb++) {
        for (int code = 0; code < MN_K; code++) {
            for (int h = 0; h < MN_H; h++) {
                MN_FACTOR_CODE[mb][code][h] =
                    (int32_t)std::lround(
                        MN_FAST_MB[mb][code][0][h]
                        * MN_FACTOR_SCALE);
            }
        }
        for (int super_cls = 0; super_cls < 4; super_cls++) {
            for (int h = 0; h < MN_H; h++) {
                MN_FACTOR_SUPER[mb][super_cls][h] =
                    (int32_t)std::lround(
                        (MN_FAST_MB[mb][0][super_cls * 2][h]
                         - MN_FAST_MB[mb][0][0][h])
                        * MN_FACTOR_SCALE);
            }
        }
        for (int h = 0; h < MN_H; h++) {
            MN_FACTOR_ACTIVE[mb][h] =
                (int32_t)std::lround(
                    (MN_FAST_MB[mb][0][1][h]
                     - MN_FAST_MB[mb][0][0][h])
                    * MN_FACTOR_SCALE);
        }
    }'''


FACTOR_EVAL = r'''template <typename Board>
static int evaluate_mini_fast(const Board &b) {
    if (!MN_READY && !mini_load_packed()) return 0;
    static_assert(MN_H % 8 == 0, "factored MiniNet requires AVX-width H");
    constexpr int N_CHUNKS = MN_H / 8;
    const int stm = b.n_moves & 1;
    const int c = mini_board_constraint(b);
    __m256i hidden[N_CHUNKS];
    for (int k = 0; k < N_CHUNKS; k++) {
        hidden[k] = _mm256_load_si256(
            (const __m256i *)(MN_FACTOR_INIT[c] + 8 * k));
    }
    for (int mb = 0; mb < 9; mb++) {
        const int mine = b.mini_boards[mb].markers[stm];
        const int opp = b.mini_boards[mb].markers[stm ^ 1];
        const int code = MN_MASK_CODE[(mine << 9) | opp];
        for (int k = 0; k < N_CHUNKS; k++) {
            hidden[k] = _mm256_add_epi32(
                hidden[k],
                _mm256_load_si256(
                    (const __m256i *)(MN_FACTOR_CODE[mb][code] + 8 * k)));
        }
        const int bit = 1 << mb;
        int super_cls = 0;
        if (b.mini_board_states[stm] & bit) super_cls = 1;
        else if (b.mini_board_states[stm ^ 1] & bit) super_cls = 2;
        else if (b.mini_board_states[2] & bit) super_cls = 3;
        if (super_cls) {
            for (int k = 0; k < N_CHUNKS; k++) {
                hidden[k] = _mm256_add_epi32(
                    hidden[k],
                    _mm256_load_si256(
                        (const __m256i *)(
                            MN_FACTOR_SUPER[mb][super_cls] + 8 * k)));
            }
        }
    }
    if (c < 9) {
        for (int k = 0; k < N_CHUNKS; k++) {
            hidden[k] = _mm256_add_epi32(
                hidden[k],
                _mm256_load_si256(
                    (const __m256i *)(MN_FACTOR_ACTIVE[c] + 8 * k)));
        }
    }
    __m256 value = _mm256_setzero_ps();
    for (int k = 0; k < N_CHUNKS; k++) {
        __m256 chunk = _mm256_cvtepi32_ps(hidden[k]);
        chunk = _mm256_max_ps(chunk, _mm256_setzero_ps());
        value = _mm256_add_ps(
            value,
            _mm256_mul_ps(chunk, _mm256_loadu_ps(MN_W2 + 8 * k)));
    }
    const float out =
        MN_B2 + mini_hsum256(value) / MN_FACTOR_SCALE;
    return (int)std::lround(out);
}
'''


def emit_header(
    template_path: Path,
    output_path: Path,
    packed_text: str,
    d: int,
    h: int,
    symbol_tag: str | None = None,
) -> None:
    if d not in (16, 19) or h not in (8, 16):
        raise SystemExit(
            f"factored runtime is verified for D=16/19 H=8/16, got D={d} H={h}"
        )
    source = template_path.read_text(encoding="utf-8")
    marker = 'static const char MINI_PACK_B64[] = R"MNUE(\n'
    start = source.index(marker) + len(marker)
    end = source.index('\n)MNUE";', start)
    source = source[:start] + packed_text + source[end:]
    source = source.replace("MINI_PACK_B64", "MINI_PACK_CJK")
    source = source.replace('R"MNUE(', 'R"~(', 1)
    source = source.replace(')MNUE"', ')~"', 1)

    decoder_start = source.index("static int mini_b64_decode(")
    decoder_end = source.index(
        "\n\nstatic constexpr int MN_N", decoder_start
    )
    source = (
        source[:decoder_start]
        + CJK14_DECODER.rstrip()
        + source[decoder_end:]
    )
    source = source.replace("mini_b64_decode", "mini_cjk_decode")
    source = source.replace(
        "// Packed MiniNet (D=8 H=4, 256-centroid emb) shared by Dev/Prev SPRT\n"
        "// and matching codingame_nnue.cpp. HCE+mini at qsearch; HCE for RFP.\n"
        "// evaluate_mini is the scalar reference; evaluate_mini_avx is the shipped path.",
        f"// Packed D{d}/H{h} MiniNet with projected 256-centroid embeddings.\n"
        "// Scalar evaluation is the reference; the factored int32 path is shipped.",
        1,
    )
    source = source.replace('#include "global_board.hpp"\n', "", 1)
    source = source.replace(
        "static constexpr int MN_D = 8;",
        f"static constexpr int MN_D = {d};",
        1,
    )
    source = source.replace(
        "static constexpr int MN_H = 4;",
        f"static constexpr int MN_H = {h};",
        1,
    )
    source = source.replace(
        "static constexpr int MN_IN = 80;",
        f"static constexpr int MN_IN = {10 * d};",
        1,
    )
    source = source.replace(
        "static unsigned char buf[40000];",
        "static unsigned char buf[65536];",
        1,
    )
    source = source.replace(
        "alignas(64) static float MN_FAST_MB[9][MN_K][8][MN_H];\n",
        "alignas(64) static float MN_FAST_MB[9][MN_K][8][MN_H];\n"
        + FACTOR_DECLS,
        1,
    )

    build_end = source.index("\n}\n\nstatic bool mini_load_packed()")
    source = source[:build_end] + FACTOR_BUILD + source[build_end:]

    avx_start = source.index(
        "static int evaluate_mini_avx(const GlobalBoard &b) {"
    )
    avx_end = source.index(
        "// First-layer projections are constant", avx_start
    )
    source = (
        source[:avx_start]
        + "static int evaluate_mini_avx(const GlobalBoard &b) {\n"
        + "    return evaluate_mini(b);\n"
        + "}\n\n"
        + source[avx_end:]
    )
    fast_start = source.index(
        "template <typename Board>\n"
        "static int evaluate_mini_fast(const Board &b) {"
    )
    source = source[:fast_start] + FACTOR_EVAL

    function_tag = symbol_tag or f"d{d}"
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9]*", function_tag):
        raise SystemExit(
            f"symbol tag must be an identifier fragment, got {function_tag!r}"
        )
    runtime_tag = function_tag.upper()
    source = source.replace("MINI_PACK_CJK", f"{runtime_tag}_MINI_PACK_CJK")
    source = re.sub(r"\bMN_", f"{runtime_tag}_MN_", source)
    for name in (
        "mini_board_constraint",
        "mini_cjk_decode",
        "mini_mask_index",
        "mini_build_fast_tables",
        "mini_load_packed",
        "evaluate_mini",
        "mini_hsum256",
        "evaluate_mini_avx",
        "evaluate_mini_fast",
    ):
        source = re.sub(rf"\b{name}\b", f"{function_tag}_{name}", source)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(source, encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("net", type=Path, help="D16/H8 or D19/H8 CFM2 checkpoint")
    parser.add_argument(
        "-o",
        "--out",
        type=Path,
        default=Path("cpp_impl/mini_eval_d16.hpp"),
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=Path("cpp_impl/mini_eval.hpp"),
    )
    parser.add_argument(
        "--no-reserve-empty",
        action="store_true",
        help="allow k-means to merge the empty local board",
    )
    parser.add_argument(
        "--tag",
        help="override generated symbol tag (for isolated candidate evaluators)",
    )
    args = parser.parse_args()

    (
        d,
        h,
        emb,
        super_e,
        loc,
        constr,
        active,
        w1,
        b1,
        w2,
        b2,
    ) = load_cfm2(args.net)
    centroids, codes = projected_centroids(
        emb, w1, w2, d, h, not args.no_reserve_empty
    )
    reconstructed = centroids[codes]
    original_empty = empty_output(
        d, h, emb, super_e, loc, constr, active, w1, b1, w2, b2
    )
    packed_empty = empty_output(
        d,
        h,
        reconstructed,
        super_e,
        loc,
        constr,
        active,
        w1,
        b1,
        w2,
        b2,
    )
    b2 = float(b2 + original_empty - packed_empty)
    shifted_empty = empty_output(
        d,
        h,
        reconstructed,
        super_e,
        loc,
        constr,
        active,
        w1,
        b1,
        w2,
        b2,
    )
    blob = legacy.pack_blob(
        codes,
        centroids,
        super_e,
        loc,
        constr,
        active,
        w1,
        b1,
        w2,
        b2,
    )
    packed_text = wrap_cjk14(encode_cjk14(blob))
    emit_header(args.template, args.out, packed_text, d, h, args.tag)
    error = emb - reconstructed
    print(
        f"wrote {args.out} chars={args.out.stat().st_size} "
        f"blob={len(blob)} emb_mae={np.abs(error).mean():.5f} "
        f"empty={original_empty:.3f}->{packed_empty:.3f}"
        f"->{shifted_empty:.3f}"
    )


if __name__ == "__main__":
    main()
