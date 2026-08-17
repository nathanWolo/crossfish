#!/usr/bin/env python3
"""Pack a trained MiniNet .bin into cpp_impl/codingame_nnue.cpp.

Builds the CodinGame file from crossfish.cpp + minires_d8h4.bin (or another
CFM2 MiniNet blob from nnue_train_mininet.py). The HCE-only CG source has a
packing marker just before CrossfishDev; this script inserts the clustered
MiniNet blob and switches RFP to HCE / qsearch leaves to HCE+mini.

The full 19683x8 embedding table does not fit the 100k-character CG cap.
Cluster the rows into 256 centroids (seed 0). Reconstruction corr vs the
float net is ~0.9997 on random boards. Empty residual is re-shifted to 0.

Runtime matches the winning SPRT: HCE for interior RFP/futility, HCE+mini
at qsearch leaves. From-scratch MiniNet, no incremental acc.
"""

from __future__ import annotations

import argparse
import base64
import os
import re
import struct
import sys

import numpy as np

N_IDX = 19683
D = 8
H = 4
K = 256
IN_DIM = 10 * D


def load_cfm2(path: str):
    blob = open(path, "rb").read()
    if blob[:4] != b"CFM2":
        raise SystemExit(f"bad magic in {path}")
    d, h = struct.unpack_from("<ii", blob, 4)
    if d != D or h != H:
        raise SystemExit(f"expected d={D} h={H}, got d={d} h={h}")
    off = 12

    def rd(n):
        nonlocal off
        v = np.frombuffer(blob[off : off + n * 4], dtype="<f4").copy()
        off += n * 4
        return v

    emb = rd(N_IDX * D).reshape(N_IDX, D)
    super_e = rd(4 * D)
    loc = rd(9 * D)
    constr = rd(10 * D)
    active = rd(2 * D)
    w1 = rd(H * IN_DIM)
    b1 = rd(H)
    w2 = rd(H)
    b2 = struct.unpack_from("<f", blob, off)[0]
    return emb, super_e, loc, constr, active, w1, b1, w2, b2


def kmeans(x: np.ndarray, k: int, seed: int = 0, iters: int = 40):
    rng = np.random.default_rng(seed)
    n, d = x.shape
    # k-means++
    cents = np.empty((k, d), dtype=np.float64)
    cents[0] = x[int(rng.integers(0, n))]
    closest = np.full(n, np.inf)
    for j in range(1, k):
        last = cents[j - 1]
        dist = ((x - last) ** 2).sum(axis=1)
        np.minimum(closest, dist, out=closest)
        p = closest / closest.sum()
        cents[j] = x[int(rng.choice(n, p=p))]
    cents = cents.astype(np.float32)
    labels = np.zeros(n, dtype=np.int32)
    for _ in range(iters):
        for i in range(0, n, 2048):
            sl = x[i : i + 2048]
            d2 = ((sl[:, None, :] - cents[None, :, :]) ** 2).sum(axis=2)
            labels[i : i + 2048] = d2.argmin(axis=1)
        for j in range(k):
            m = labels == j
            if m.any():
                cents[j] = x[m].mean(axis=0)
    return cents.astype(np.float32), labels.astype(np.uint8)


def mini_forward(emb, super_e, loc, constr, active, w1, b1, w2, b2, ids, supers, c):
    x = np.empty(IN_DIM, dtype=np.float32)
    for mb in range(9):
        act = 1 if c == mb else 0
        x[mb * D : (mb + 1) * D] = (
            emb[ids[mb]]
            + super_e[supers[mb] * D : (supers[mb] + 1) * D]
            + loc[mb * D : (mb + 1) * D]
            + active[act * D : (act + 1) * D]
        )
    x[9 * D :] = constr[c * D : (c + 1) * D]
    hid = w1.reshape(H, IN_DIM) @ x + b1
    hid = np.maximum(hid, 0)
    return float(w2 @ hid + b2)


def empty_out(emb, super_e, loc, constr, active, w1, b1, w2, b2):
    return mini_forward(
        emb, super_e, loc, constr, active, w1, b1, w2, b2,
        np.zeros(9, np.int32), np.zeros(9, np.int32), 9,
    )


def pack_blob(codes, cents, super_e, loc, constr, active, w1, b1, w2, b2) -> bytes:
    parts = [
        codes.astype(np.uint8).tobytes(),
        cents.astype("<f4").tobytes(),
        super_e.astype("<f4").tobytes(),
        loc.astype("<f4").tobytes(),
        constr.astype("<f4").tobytes(),
        active.astype("<f4").tobytes(),
        w1.astype("<f4").tobytes(),
        b1.astype("<f4").tobytes(),
        w2.astype("<f4").tobytes(),
        struct.pack("<f", float(b2)),
    ]
    return b"".join(parts)


def wrap_b64(s: str, width: int = 100) -> str:
    return "\n".join(s[i : i + width] for i in range(0, len(s), width))


MINI_HELPERS = r'''
static int mini_board_constraint(const GlobalBoard &b) {
    if (b.n_moves == 0 || b.prev_move_was_pass) return 9;
    int sent = b.move_history.top().square;
    int oop = b.mini_board_states[0] | b.mini_board_states[1] | b.mini_board_states[2];
    if ((oop & (1 << sent)) != 0) return 9;
    return sent;
}

static int mini_b64_decode(const char *s, unsigned char *out, int out_max) {
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

static constexpr int MN_N = 19683;
static constexpr int MN_K = 256;
static constexpr int MN_D = 8;
static constexpr int MN_H = 4;
static constexpr int MN_IN = 80;
static uint8_t MN_CODE[MN_N];
static float MN_CENT[MN_K * MN_D];
static float MN_SUPER[4 * MN_D];
static float MN_LOC[9 * MN_D];
static float MN_CONSTR[10 * MN_D];
static float MN_ACTIVE[2 * MN_D];
static float MN_W1[MN_H * MN_IN];
static float MN_B1[MN_H];
static float MN_W2[MN_H];
static float MN_B2 = 0;
static bool MN_READY = false;

static bool mini_load_packed() {
    if (MN_READY) return true;
    static unsigned char buf[40000];
    int nb = mini_b64_decode(MINI_PACK_B64, buf, (int)sizeof(buf));
    const int need = MN_N + MN_K * MN_D * 4 + (4 + 9 + 10 + 2) * MN_D * 4
                     + MN_H * MN_IN * 4 + MN_H * 4 + MN_H * 4 + 4;
    if (nb < need) return false;
    int off = 0;
    memcpy(MN_CODE, buf + off, MN_N); off += MN_N;
    memcpy(MN_CENT, buf + off, MN_K * MN_D * 4); off += MN_K * MN_D * 4;
    memcpy(MN_SUPER, buf + off, 4 * MN_D * 4); off += 4 * MN_D * 4;
    memcpy(MN_LOC, buf + off, 9 * MN_D * 4); off += 9 * MN_D * 4;
    memcpy(MN_CONSTR, buf + off, 10 * MN_D * 4); off += 10 * MN_D * 4;
    memcpy(MN_ACTIVE, buf + off, 2 * MN_D * 4); off += 2 * MN_D * 4;
    memcpy(MN_W1, buf + off, MN_H * MN_IN * 4); off += MN_H * MN_IN * 4;
    memcpy(MN_B1, buf + off, MN_H * 4); off += MN_H * 4;
    memcpy(MN_W2, buf + off, MN_H * 4); off += MN_H * 4;
    memcpy(&MN_B2, buf + off, 4);
    MN_READY = true;
    return true;
}

static int evaluate_mini(const GlobalBoard &b) {
    if (!MN_READY && !mini_load_packed()) return 0;
    const int stm = b.n_moves & 1;
    const int c = mini_board_constraint(b);
    float x[MN_IN];
    for (int mb = 0; mb < 9; mb++) {
        const int mine = b.mini_boards[mb].markers[stm];
        const int opp = b.mini_boards[mb].markers[stm ^ 1];
        int idx = 0;
        idx += (mine & 1) ? 1 : ((opp & 1) ? 2 : 0);
        idx += (mine & 2) ? 3 : ((opp & 2) ? 6 : 0);
        idx += (mine & 4) ? 9 : ((opp & 4) ? 18 : 0);
        idx += (mine & 8) ? 27 : ((opp & 8) ? 54 : 0);
        idx += (mine & 16) ? 81 : ((opp & 16) ? 162 : 0);
        idx += (mine & 32) ? 243 : ((opp & 32) ? 486 : 0);
        idx += (mine & 64) ? 729 : ((opp & 64) ? 1458 : 0);
        idx += (mine & 128) ? 2187 : ((opp & 128) ? 4374 : 0);
        idx += (mine & 256) ? 6561 : ((opp & 256) ? 13122 : 0);
        int super_cls = 0;
        if (b.mini_board_states[stm] & (1 << mb)) super_cls = 1;
        else if (b.mini_board_states[stm ^ 1] & (1 << mb)) super_cls = 2;
        else if (b.mini_board_states[2] & (1 << mb)) super_cls = 3;
        const int act = (c == mb) ? 1 : 0;
        const float *e = MN_CENT + (int)MN_CODE[idx] * MN_D;
        const float *s = MN_SUPER + super_cls * MN_D;
        const float *l = MN_LOC + mb * MN_D;
        const float *a = MN_ACTIVE + act * MN_D;
        float *dst = x + mb * MN_D;
        for (int i = 0; i < MN_D; i++) dst[i] = e[i] + s[i] + l[i] + a[i];
    }
    const float *ce = MN_CONSTR + c * MN_D;
    for (int i = 0; i < MN_D; i++) x[9 * MN_D + i] = ce[i];
    float out = MN_B2;
    for (int j = 0; j < MN_H; j++) {
        const float *row = MN_W1 + j * MN_IN;
        float s = MN_B1[j];
        for (int i = 0; i < MN_IN; i++) s += row[i] * x[i];
        if (s > 0) out += MN_W2[j] * s;
    }
    return (int)std::lround(out);
}

'''


def transform(src: str, b64: str, raw_bytes: int) -> str:
    src = src.replace(
        "// Texel-tuned eval weights (per-weight Adam). SPRT 20ms: +25 Elo, LLR +3.04.",
        "// Texel-tuned eval weights (per-weight Adam). SPRT 20ms: +25 Elo, LLR +3.04.\n"
        "// NNUE variant: minires_d8h4 residual (D=8 H=4, 256-centroid emb).\n"
        "// Equal-depth +54 Elo; 20ms +11; 95ms +7. HCE RFP, MiniNet at qsearch.",
        1,
    )
    start = src.find("\n// MiniNet packing marker for tools/nnue_emit_mininet_cg.py\n")
    end = src.find("\nclass CrossfishDev {")
    if start < 0 or end < 0:
        raise SystemExit("could not find MiniNet packing marker in crossfish.cpp")
    blob_decl = (
        f"// MiniNet residual weights (k-means-256 embeddings)\n"
        f"static const int MINI_PACK_RAW_BYTES = {raw_bytes};\n"
        f"static const char MINI_PACK_B64[] = R\"MNUE(\n{wrap_b64(b64)}\n)MNUE\";\n"
    )
    src = src[: start + 1] + blob_decl + MINI_HELPERS + src[end + 1 :]

    src = src.replace(
        "            init_mini_lut();\n            thinking_time = thinking_time_passed;",
        "            init_mini_lut();\n            mini_load_packed();\n            thinking_time = thinking_time_passed;",
        1,
    )

    src = src.replace(
        "            if (!pv_node) {\n                int stand_pat = evaluate(board);",
        "            if (!pv_node) {\n                int stand_pat = evaluate_hce(board);",
        1,
    )

    src = src.replace(
        "        int evaluate(GlobalBoard &board) {\n            init_mini_lut();",
        "        int evaluate_hce(GlobalBoard &board) {\n            init_mini_lut();",
        1,
    )
    # Close evaluate_hce then add evaluate = HCE + mini.
    src = src.replace(
        "            return score;\n\n        }\n\n};",
        "            return score;\n\n        }\n\n"
        "        int evaluate(GlobalBoard &board) {\n"
        "            return evaluate_hce(board) + evaluate_mini(board);\n"
        "        }\n\n};",
        1,
    )
    return src


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="cpp_impl/crossfish.cpp")
    ap.add_argument("--bin", default="cpp_impl/bin/nnue_exps/minires_d8h4.bin")
    ap.add_argument("--out", default="cpp_impl/codingame_nnue.cpp")
    args = ap.parse_args()

    emb, super_e, loc, constr, active, w1, b1, w2, b2 = load_cfm2(args.bin)
    print("k-means 256 on embeddings...", flush=True)
    cents, codes = kmeans(emb, K, seed=0, iters=40)
    recon = cents[codes]
    err = emb - recon
    print(
        f"emb mae={np.abs(err).mean():.4f} rmse={np.sqrt((err * err).mean()):.4f}",
        flush=True,
    )
    e0 = empty_out(emb, super_e, loc, constr, active, w1, b1, w2, b2)
    k0 = empty_out(recon, super_e, loc, constr, active, w1, b1, w2, b2)
    b2 = float(b2 + (e0 - k0))
    k0b = empty_out(recon, super_e, loc, constr, active, w1, b1, w2, b2)
    print(f"empty orig={e0:.3f} kmeans={k0:.3f} shifted={k0b:.3f}", flush=True)

    blob = pack_blob(codes, cents, super_e, loc, constr, active, w1, b1, w2, b2)
    b64 = base64.b64encode(blob).decode("ascii")
    src = open(args.src, encoding="utf-8").read()
    out = transform(src, b64, len(blob))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        f.write(out)
    print(f"wrote {args.out} chars={len(out)} blob={len(blob)} b64={len(b64)}")
    if len(out) >= 100000:
        print("WARNING: over CodinGame 100k character cap", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
