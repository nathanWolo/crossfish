#!/usr/bin/env python3
"""Inject the sparse 199-feature NNUE (not MiniNet) into crossfish.cpp.

MiniNet's CodinGame file is built by nnue_emit_mininet_cg.py instead.
"""

from __future__ import annotations

import argparse
import os
import re

NNUE_CLASS = r'''
struct NnueNet {
    static constexpr int N_SQ_F = 162;
    static constexpr int N_FEAT = 199;
    static constexpr int H = 32;
    static constexpr int L2 = 32;
    static constexpr int QA = 127;
    static constexpr int QB = 64;
    static constexpr int SUPER_BASE = 162;
    static constexpr int CONSTR_BASE = 189;
    static constexpr int UNDO_MAX = 128;
    int32_t acc[2][H]{};
    int16_t w0[N_FEAT * H]{};
    int16_t b0[H]{};
    int8_t w1[L2 * (2 * H)]{};
    int32_t b1[L2]{};
    int8_t w2[L2]{};
    int32_t b2 = 0;
    int scale = 2410;
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
    static int sq_feat(int pers, int mb, int sq, int owner) {
        return (mb * 9 + sq) * 2 + ((owner == pers) ? 0 : 1);
    }
    static int super_feat(int pers, int mb, int winner) {
        int cls = (winner == 2) ? 2 : ((winner == pers) ? 0 : 1);
        return SUPER_BASE + mb * 3 + cls;
    }
    void add_feat(int pers, int f) {
        const int16_t *w = w0 + f * H;
        for (int h = 0; h < H; h++) acc[pers][h] += w[h];
    }
    void sub_feat(int pers, int f) {
        const int16_t *w = w0 + f * H;
        for (int h = 0; h < H; h++) acc[pers][h] -= w[h];
    }
    void add_both_constr(int c) {
        add_feat(0, CONSTR_BASE + c);
        add_feat(1, CONSTR_BASE + c);
    }
    void sub_both_constr(int c) {
        sub_feat(0, CONSTR_BASE + c);
        sub_feat(1, CONSTR_BASE + c);
    }
    void reset_bias() {
        for (int p = 0; p < 2; p++) for (int h = 0; h < H; h++) acc[p][h] = b0[h];
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
        if (undo_n >= UNDO_MAX) { refresh(after); return; }
        memcpy(undo_acc[undo_n][0], acc[0], sizeof(acc[0]));
        memcpy(undo_acc[undo_n][1], acc[1], sizeof(acc[1]));
        undo_constraint[undo_n] = constraint;
        undo_n++;
        int who = (after.n_moves - 1) & 1;
        int mb = m.mini_board;
        int sq = m.square;
        add_feat(0, sq_feat(0, mb, sq, who));
        add_feat(1, sq_feat(1, mb, sq, who));
        if (after.mini_board_states[0] & (1 << mb)) {
            add_feat(0, super_feat(0, mb, 0)); add_feat(1, super_feat(1, mb, 0));
        } else if (after.mini_board_states[1] & (1 << mb)) {
            add_feat(0, super_feat(0, mb, 1)); add_feat(1, super_feat(1, mb, 1));
        } else if (after.mini_board_states[2] & (1 << mb)) {
            add_feat(0, super_feat(0, mb, 2)); add_feat(1, super_feat(1, mb, 2));
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
        memcpy(acc[0], undo_acc[undo_n][0], sizeof(acc[0]));
        memcpy(acc[1], undo_acc[undo_n][1], sizeof(acc[1]));
        constraint = undo_constraint[undo_n];
    }
    static int crelu(int32_t x) {
        if (x < 0) return 0;
        if (x > QA) return QA;
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
            for (int i = 0; i < 2 * H; i++) s += c[i] * (int32_t)row[i];
            h2[j] = crelu(s / QB);
        }
        int32_t out = b2;
        for (int j = 0; j < L2; j++) out += h2[j] * (int32_t)w2[j];
        return (int)((out * (int64_t)scale) / (int64_t)(QB * QA));
    }
    int evaluate(const GlobalBoard &b) const { return evaluate_stm(b.n_moves & 1); }
};

static int nnue_b64_decode(const char *s, unsigned char *out, int out_max) {
    auto val = [](char c) -> int {
        if (c >= 'A' && c <= 'Z') return c - 'A';
        if (c >= 'a' && c <= 'z') return c - 'a' + 26;
        if (c >= '0' && c <= '9') return c - '0' + 52;
        if (c == '+') return 62;
        if (c == '/') return 63;
        return -1;
    };
    int n = 0, acc = 0, bits = 0;
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

static bool nnue_unpack_blob(NnueNet &n, const unsigned char *p, int n_bytes) {
    const int need = 4 + 4 + 4 + NnueNet::N_FEAT * NnueNet::H * 2 + NnueNet::H * 2
                     + NnueNet::L2 * (2 * NnueNet::H) + NnueNet::L2 * 4 + NnueNet::L2;
    if (n_bytes < need) return false;
    if (p[0] != 'C' || p[1] != 'F' || p[2] != 'N' || p[3] != '1') return false;
    int off = 4;
    memcpy(&n.scale, p + off, 4); off += 4;
    memcpy(&n.b2, p + off, 4); off += 4;
    memcpy(n.w0, p + off, sizeof(n.w0)); off += (int)sizeof(n.w0);
    memcpy(n.b0, p + off, sizeof(n.b0)); off += (int)sizeof(n.b0);
    memcpy(n.w1, p + off, sizeof(n.w1)); off += (int)sizeof(n.w1);
    memcpy(n.b1, p + off, sizeof(n.b1)); off += (int)sizeof(n.b1);
    memcpy(n.w2, p + off, sizeof(n.w2));
    n.weights_ready = true;
    return true;
}

static unsigned char NNUE_BLOB_BUF[20000];
static bool nnue_load_packed(NnueNet &n) {
    if (n.weights_ready) return true;
    int nb = nnue_b64_decode(NNUE_PACK_B64, NNUE_BLOB_BUF, (int)sizeof(NNUE_BLOB_BUF));
    if (nb < 0) return false;
    return nnue_unpack_blob(n, NNUE_BLOB_BUF, nb);
}
'''


def inject(src: str, pack_inc: str, enable: bool) -> str:
    with open(pack_inc, encoding="utf-8") as f:
        pack = f.read()
    # drop pragma
    pack = re.sub(r"#pragma once\s*", "", pack).strip() + "\n"

    if "#include <cstring>" not in src:
        src = src.replace("#include <iostream>", "#include <iostream>\n#include <cstring>\n#include <cstdint>")

    marker = "// NNUE packed weights + inference\n"
    if marker in src:
        start = src.find(marker)
        end = src.find("class CrossfishDev {")
        src = src[:start] + src[end:]

    block = marker + pack + "\n" + NNUE_CLASS + "\n"
    src = src.replace("class CrossfishDev {", block + "class CrossfishDev {", 1)

    # flags / member
    if "static constexpr int USE_NNUE" not in src:
        src = src.replace(
            "static constexpr int FREE_MOVE_PAWNS = 30;",
            "static constexpr int FREE_MOVE_PAWNS = 30;\n"
            f"        static constexpr int USE_NNUE = {1 if enable else 0};\n"
            "        NnueNet nnue;",
            1,
        )
    else:
        src = re.sub(
            r"static constexpr int USE_NNUE = [01];",
            f"static constexpr int USE_NNUE = {1 if enable else 0};",
            src,
            count=1,
        )

    if "nnue_load_packed(nnue)" not in src:
        src = src.replace(
            "init_mini_lut();\n            thinking_time = thinking_time_passed;",
            "init_mini_lut();\n"
            "            if (!nnue.weights_ready) nnue_load_packed(nnue);\n"
            "            if constexpr (USE_NNUE) nnue.refresh(board);\n"
            "            thinking_time = thinking_time_passed;",
            1,
        )

    if "if constexpr (USE_NNUE) nnue.make(board, caps[i]);" not in src:
        src = src.replace(
            "board.makeMove(caps[i]);\n                val = -qsearch(board, -beta, -alpha, ply + 1);\n                board.unmakeMove();",
            "board.makeMove(caps[i]);\n"
            "                if constexpr (USE_NNUE) nnue.make(board, caps[i]);\n"
            "                val = -qsearch(board, -beta, -alpha, ply + 1);\n"
            "                if constexpr (USE_NNUE) nnue.unmake();\n"
            "                board.unmakeMove();",
            1,
        )
    if "if constexpr (USE_NNUE) nnue.make(board, legal_moves[i]);" not in src:
        src = src.replace(
            "board.makeMove(legal_moves[i]);\n                if (i == 0) {",
            "board.makeMove(legal_moves[i]);\n"
            "                if constexpr (USE_NNUE) nnue.make(board, legal_moves[i]);\n"
            "                if (i == 0) {",
            1,
        )
        src = src.replace(
            "                board.unmakeMove();\n                if (stopped) return min_val;\n                if (val > best_val) {",
            "                if constexpr (USE_NNUE) nnue.unmake();\n"
            "                board.unmakeMove();\n                if (stopped) return min_val;\n                if (val > best_val) {",
            1,
        )

    if "if constexpr (USE_NNUE)" not in src.split("int evaluate(GlobalBoard &board)")[1][:400]:
        src = src.replace(
            "        int evaluate(GlobalBoard &board) {\n            init_mini_lut();",
            "        int evaluate(GlobalBoard &board) {\n"
            "            if constexpr (USE_NNUE) {\n"
            "                return nnue.evaluate(board);\n"
            "            }\n"
            "            init_mini_lut();",
            1,
        )
    return src


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="cpp_impl/crossfish.cpp")
    ap.add_argument("--pack", default="cpp_impl/nnue_pack.inc")
    ap.add_argument("--enable", action="store_true")
    args = ap.parse_args()
    with open(args.src, encoding="utf-8") as f:
        text = f.read()
    new = inject(text, args.pack, args.enable)
    with open(args.src, "w", encoding="utf-8", newline="\n") as f:
        f.write(new)
    print(f"wrote {args.src} chars={len(new)} enable={args.enable}")
    if len(new) >= 100000:
        print("WARNING: over CodinGame 100k character cap")


if __name__ == "__main__":
    main()
