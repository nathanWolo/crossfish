#!/usr/bin/env python3
"""Emit cpp_impl/nnue_b64_net.hpp, the payload of the engine's evaluation (cpp_impl/nnue_b64.hpp).

  python tools/nnue_emit_b64_header.py NET.bin [-o cpp_impl/nnue_b64_net.hpp] [--label NAME]
      [--config enc=14,proj=12,...] [--refit-groups proj,fwd,dense] [--calib datasets/nnue2/d8_a.cfdg]
  python tools/nnue_emit_b64_header.py --check [HEADER]

NET is the lane-paired BGN1 export of a gen_nnue pattern-generator checkpoint
(`tools/experiments/fast_nnue/export_bgn.py export NAME OUT --perm`, which rebuilds
datasets/nnue2/fast/B64_d5M_57ep_perm.bin byte for byte from datasets/nnue2/probe/B64_d5M_57ep.pt); only its
generator section and its dense head are read. The net's widths come from the file: any A with A % 16 == 0
(the shipped net is B-64; B-96 and B-128 are verified, documentation/nnue_generic_a.md), an L1 x L2 head
(L1 % 16 == 0, L2 % 8 == 0) and a three-layer encoder. The header declares them (B64_A, B64_L1, B64_L2,
B64_E, B64_ENC0, B64_ENC1) and the runtime compiles for them; a header for another width goes under
cpp_impl/experimental/ and is built with `make NNUE_NET=experimental/NAME.hpp ...`. The shipped header
came from that file with every default below:

  python tools/nnue_emit_b64_header.py datasets/nnue2/fast/B64_d5M_57ep_perm.bin --label B64_d5M_57ep

which reproduces the verified CodinGame build (datasets/nnue2/cg/d5M57, `quant_gen.py emit --qexp
9,13,13,13,10 --refit-groups proj,fwd,dense --config enc=14,proj=12,fwd=12,dec=14,con=13,dense=14,psqt=14,
bias=14`) byte for byte: the same 54,114-byte payload, so the same decoded generator and baked tables.
Calibration reads datasets/nnue2/d8_a.cfdg (--calib), which is not in the repository. Numerical steps
(least squares, Cholesky) were checked with numpy 2.5.2.

--check decodes a header's payload, bakes and quantizes it exactly as nnue_b64.hpp load() does (float32,
same operation order), recomputes the integer scales and prints the FNV-1a hashes of the 16 integer tables;
it needs numpy only (tools/test_nnue_emit_b64_header.py runs it in CI).

The net (the B-64 numbers; W = A + 1 lanes: A accumulator lanes + 1 PSQT lane; 35,243 parameters at A = 64,
48,363 at 96, 61,483 at 128):
  encoder 27 -> 64 -> 64 -> 32 of one miniboard pattern (one-hot 3 x 9), ReLU between layers
  proj_w[9][32][65], proj_b[9][65]: T[m][p] = enc(p) @ proj_w[m] + proj_b[m]  (miniboard m, pattern p)
  fwd_w[32][65], fwd_b[65]:         F[p] = enc(p) @ fwd_w + fwd_b             (the forced board's pattern)
  bias[65], dec[27][65] (decided boards), con[20][65] (constraint rows, stm then the other side)
  dense head 128 -> 16 -> 32 -> 1

Payload: one MSB-first bit stream of 11 matrices, each row one output unit with its bias in column 0:
  enc0 64x28, enc1 64x65, enc2 32x65   [b | W] of the encoder layers
  proj 9Wx33    row W m + j: [proj_b[m][j] | proj_w[m][:, j]]  (lane j of location m; j = A is PSQT)
  fwd Wx33      row j: [fwd_b[j] | fwd_w[:, j]]
  dec Wx27, con Wx20   transposed: one row per lane
  bias 1xW
  d1 16x(2A+1), d2 32x17, do 1x33   [b | W] of the dense head
Per matrix: its scales as bf16 (16 bits each; one per row, except proj, whose 9 locations share one per
lane), two 4-bit Rice parameters (normal rows, PSQT rows), then every code row by row as
Rice(zigzag(q)). A value is float32(q) * scale. Bits per group (--config) set each row's grid; the PSQT
rows (lane 64 of proj, fwd, dec and con) get their own width and scale, since an error there moves the
eval 500 times itself. The bytes are packed 15 bits per character (the U15 alphabet, tools/nnue_cjk14.py).

Rounding: the encoder, proj and fwd are rounded with GPTQ over the 11,093 live patterns (weighted by their
frequency in --ncal rows of --calib, plus a floor), the dense head over --ncal calibration positions passed
through the quantized first layer; the groups in --refit-groups are first refit by least squares to the
float net's outputs given the already-quantized inputs; dec, con and bias are rounded to nearest. The
encoder is not refit by default: on B64_d5M_57ep its refit moved rare patterns' embeddings (max |d| 116
against 11 at 16 bits everywhere).

Integer scales (B64_QA, B64_QPS, B64_QB, B64_Q2, B64_QO): the loader quantizes the baked float tables to int16
at 2^QA (lanes) and 2^QPS (PSQT), the dense layers at 2^QB, 2^Q2, 2^QO. QA is the largest value with every
row, every stored accumulator and every accumulator + constraint row + forced-board row inside int16 over
all positions the features can express (per-board extremes by the pattern's stone difference, combined
under the stone-count balance of a real game: the side to move has as many stones as the other or one
fewer); the others are the largest with the int32 sums of their layer bounded. `scales()` is a port of the
experiment loader's choice (tools/experiments/fast_nnue/fast_nnue_b.hpp quantize()): 9, 13, 13, 13, 10 here,
and 10, 13, 14, 14, 11 on the first CodinGame build's B64_lr1e2, as that loader chose.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nnue_cjk14 import decode_u15, encode_u15, wrap_cjk14  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / "cpp_impl" / "nnue_b64_net.hpp"
CALIB = ROOT / "datasets" / "nnue2" / "d8_a.cfdg"
SAMPLE = ROOT / "datasets" / "nnue2" / "fnn1" / "parity_in.cfdg"

A, W, E, L1, L2 = 64, 65, 32, 16, 32  # the shipped B-64 shape; set_shape() rebinds them for another net
ENC = (64, 64, 32)  # encoder widths (27 -> ENC[0] -> ENC[1] -> E)
N_PAT = 19683
POW3 = 3 ** np.arange(9)
DIGITS = np.array([(p // POW3) % 3 for p in range(N_PAT)])
PAT_SWAP = ((np.where(DIGITS == 1, 2, np.where(DIGITS == 2, 1, 0))) * POW3).sum(1)
LINES = [7, 56, 448, 73, 146, 292, 273, 84]


def live_patterns():
    """(3^9,) bool: no three-in-a-row for either side and not full (the patterns a live board can have)."""
    m1 = ((DIGITS == 1) * (1 << np.arange(9))).sum(1)
    m2 = ((DIGITS == 2) * (1 << np.arange(9))).sum(1)
    ok = (DIGITS == 0).any(1)
    for ln in LINES:
        ok &= ((m1 & ln) != ln) & ((m2 & ln) != ln)
    return ok


LIVE = live_patterns()
ONEHOT = np.eye(27)[3 * np.arange(9)[None, :] + DIGITS].sum(1)  # (3^9, 27): 3*square + digit

# The shipped configuration (datasets/nnue2/cg/d5M57/NOTES.md section 2).
DEFAULT = dict(enc=14, proj=12, fwd=12, dec=14, con=13, dense=14, psqt=14, bias=14)
DEFAULT_REFIT = ("proj", "fwd", "dense")
REFIT_GROUPS = ("enc", "proj", "fwd", "dense")
MATS = ["enc0", "enc1", "enc2", "proj", "fwd", "dec", "con", "bias", "d1", "d2", "do"]
GROUP = dict(enc0="enc", enc1="enc", enc2="enc", proj="proj", fwd="fwd", dec="dec", con="con", bias="bias",
             d1="dense", d2="dense", do="dense")
SHAPES = {}
OPTS = dict(refit=True, gptq=True, ridge=1e-6, refit_groups=DEFAULT_REFIT)


def set_shape(a=64, l1=16, l2=32, enc=(64, 64, 32)):
    """Bind the module to a net shape: A accumulator lanes (W = A + 1 with the PSQT lane), an L1 x L2 head
    and a three-layer encoder 27 -> enc[0] -> enc[1] -> enc[2] = E. The runtime needs A % 16 == 0,
    L1 % 16 == 0 and L2 % 8 == 0 (nnue_b64.hpp's static_assert)."""
    global A, W, E, L1, L2, ENC, SHAPES
    enc = tuple(int(x) for x in enc)
    if len(enc) != 3:
        raise SystemExit(f"encoder depth {len(enc)}: the runtime bakes a three-layer encoder")
    if a % 16 or l1 % 16 or l2 % 8 or l2 <= 0:
        raise SystemExit(f"shape A={a} L1={l1} L2={l2}: the runtime needs A % 16 == 0, L1 % 16 == 0, L2 % 8 == 0")
    A, W, E, L1, L2, ENC = int(a), int(a) + 1, enc[2], int(l1), int(l2), enc
    SHAPES.clear()
    SHAPES.update(enc0=(ENC[0], 28), enc1=(ENC[1], 1 + ENC[0]), enc2=(E, 1 + ENC[1]), proj=(9 * W, 1 + E),
                  fwd=(W, 1 + E), dec=(W, 27), con=(W, 20), bias=(1, W), d1=(L1, 1 + 2 * A), d2=(L2, 1 + L1),
                  do=(1, 1 + L2))


def n_params():
    """Parameters of the current shape (35,243 for B-64)."""
    return sum(r * c for r, c in SHAPES.values())


set_shape()


def parse_config(s):
    c = dict(DEFAULT)
    for kv in filter(None, (s or "").split(",")):
        k, v = kv.split("=")
        if k not in c:
            raise SystemExit(f"unknown config key {k} (keys: {', '.join(c)})")
        c[k] = int(v)
    return c


# ---------------------------------------------------------------- the BGN1 file (generator section)
class BGN:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def read_bgn(path):
    """The parts of a BGN1 file the payload needs (the baked T and F tables are skipped)."""
    raw = Path(path).read_bytes()
    if raw[:4] != b"BGN1":
        raise SystemExit(f"{path}: not a BGN1 file")
    a, l1, l2, forced, ncon, has_gen, e, n_enc = struct.unpack("<8i", raw[4:36])
    if (forced, ncon, has_gen, n_enc) != (1, 20, 1, 3):
        raise SystemExit(f"{path}: shape A={a} L1={l1} L2={l2} forced={forced} ncon={ncon} gen={has_gen} "
                         f"E={e} n_enc={n_enc}; this emitter needs a forced-board row, 20 constraint rows, "
                         f"the generator section and a three-layer encoder")
    if not l2:
        raise SystemExit(f"{path}: a 2A -> L1 -> 1 head (L2 = 0) is not supported by the runtime")
    # the encoder dims sit after the dense head; read them first so that the shape is bound before W is used
    head_floats = (2 * a + 1) * l1 + (l1 + 1) * l2 + (l2 + 1)
    ed_off = 36 + 4 * ((a + 1) * (1 + 9 * N_PAT + 27 + ncon + N_PAT) + head_floats)
    ed = list(np.frombuffer(raw, dtype="<i4", count=n_enc + 1, offset=ed_off))
    if ed[0] != 27 or ed[-1] != e:
        raise SystemExit(f"{path}: encoder dims {ed} do not start at 27 and end at E={e}")
    set_shape(a, l1, l2, ed[1:])
    off = 36

    def take(*shape, dt="<f4"):
        nonlocal off
        n = int(np.prod(shape))
        x = np.frombuffer(raw, dtype=dt, count=n, offset=off).reshape(shape)
        off += 4 * n
        return x

    b = BGN()
    b.bias = take(W)
    take(9, N_PAT, W)  # T (baked; not needed)
    b.dec = take(27, W)
    b.con = take(ncon, W)
    take(N_PAT, W)  # F (baked; not needed)
    dims = [2 * A, L1, L2, 1]
    b.dense = [(take(o, i), take(o)) for i, o in zip(dims[:-1], dims[1:])]
    assert list(take(n_enc + 1, dt="<i4")) == ed
    b.enc = [(take(o, i), take(o)) for i, o in zip(ed[:-1], ed[1:])]
    b.proj_w = take(9, E, W)
    b.proj_b = take(9, W)
    b.fwd_w = take(E, W)
    b.fwd_b = take(W)
    if off != len(raw):
        raise SystemExit(f"{path}: {len(raw) - off} trailing bytes")
    return b


def gen_params(b):
    p = {f"enc{i}_{t}": np.array(x, np.float32) for i, (w, bb) in enumerate(b.enc) for t, x in (("w", w), ("b", bb))}
    p.update(proj_w=np.array(b.proj_w, np.float32), proj_b=np.array(b.proj_b, np.float32),
             fwd_w=np.array(b.fwd_w, np.float32), fwd_b=np.array(b.fwd_b, np.float32),
             bias=np.array(b.bias, np.float32), dec=np.array(b.dec, np.float32), con=np.array(b.con, np.float32))
    for (w, bb), k in zip(b.dense, ("1", "2", "o")):
        p[f"w{k}"], p[f"b{k}"] = np.array(w, np.float32), np.array(bb, np.float32)
    return p


def to_mats(p):
    aug = lambda w, b: np.concatenate([b[:, None], w], 1)  # noqa: E731
    return dict(enc0=aug(p["enc0_w"], p["enc0_b"]), enc1=aug(p["enc1_w"], p["enc1_b"]), enc2=aug(p["enc2_w"], p["enc2_b"]),
                proj=np.concatenate([aug(p["proj_w"][m].T, p["proj_b"][m]) for m in range(9)], 0),
                fwd=aug(p["fwd_w"].T, p["fwd_b"]), dec=p["dec"].T.copy(), con=p["con"].T.copy(), bias=p["bias"][None, :].copy(),
                d1=aug(p["w1"], p["b1"]), d2=aug(p["w2"], p["b2"]), do=aug(p["wo"], p["bo"]))


def from_mats(M):
    p = {}
    for i in range(3):
        p[f"enc{i}_w"], p[f"enc{i}_b"] = M[f"enc{i}"][:, 1:], M[f"enc{i}"][:, 0]
    pr = M["proj"].reshape(9, W, 1 + E)
    p["proj_w"], p["proj_b"] = pr[:, :, 1:].transpose(0, 2, 1), pr[:, :, 0]
    p["fwd_w"], p["fwd_b"] = M["fwd"][:, 1:].T, M["fwd"][:, 0]
    p["dec"], p["con"], p["bias"] = M["dec"].T, M["con"].T, M["bias"][0]
    for k, n in (("1", "d1"), ("2", "d2"), ("o", "do")):
        p[f"w{k}"], p[f"b{k}"] = M[n][:, 1:], M[n][:, 0]
    return {k: np.ascontiguousarray(v, np.float32) for k, v in p.items()}


def encoder_np(p, onehot=ONEHOT):
    h = onehot.astype(np.float64)
    for i in range(3):
        h = h @ p[f"enc{i}_w"].T.astype(np.float64) + p[f"enc{i}_b"]
        if i < 2:
            h = np.maximum(h, 0)
    return h


def to_bgn(p):
    """The float64 tables of a generator (for calibration and measurement; not the engine's floats)."""
    b = BGN(bias=p["bias"], dec=p["dec"], con=p["con"], dense=[(p["w1"], p["b1"]), (p["w2"], p["b2"]), (p["wo"], p["bo"])])
    h = encoder_np(p)
    b.T = np.einsum("pk,lkj->lpj", h, p["proj_w"].astype(np.float64)) + p["proj_b"][:, None, :]
    b.F = h @ p["fwd_w"].astype(np.float64) + p["fwd_b"]
    return b


# ---------------------------------------------------------------- positions (eval_data records)
def compact(rec):
    """-> (N, 91) uint8: cells relative to the side to move (0 empty, 1 mine, 2 theirs), board states
    (0 live, 1 mine, 2 theirs, 3 drawn), constraint (0..8 forced, 9 free)."""
    s = np.frombuffer(rec["s"].tobytes(), dtype=np.uint8).reshape(-1, 93).astype(np.int16) - 48
    stm = np.where(s[:, 90] == 1, 1, 2)[:, None]
    cells = s[:, :81]
    c = np.where(cells == 0, 0, np.where(cells == stm, 1, 2))
    sup = s[:, 81:90]
    u = np.where(sup == 0, 0, np.where(sup == 3, 3, np.where(sup == stm, 1, 2)))
    con = np.clip(s[:, 91:92], 0, 9)
    return np.concatenate([c, u, con], axis=1).astype(np.uint8)


def accumulators_np(b, xc):
    """(n, 91) compact rows -> float64 (acc_us, acc_them), each (n, W)."""
    n = len(xc)
    cells = xc[:, :81].astype(np.int64).reshape(n, 9, 9)
    sup = xc[:, 81:90].astype(np.int64)
    con = xc[:, 90].astype(np.int64)
    pat = (cells * POW3).sum(2)
    out = []
    for side in (0, 1):
        p = pat if side == 0 else PAT_SWAP[pat]
        s = sup if side == 0 else np.where((sup == 1) | (sup == 2), 3 - sup, sup)
        acc = np.repeat(b.bias.astype(np.float64)[None], n, 0)
        for m in range(9):
            live = s[:, m] == 0
            row = np.where(live[:, None], b.T[m, p[:, m]], b.dec[3 * m + np.clip(s[:, m] - 1, 0, 2)])
            acc += row
        acc += b.con[con + 10 * side]
        forced = con < 9
        pf = p[np.arange(n), np.clip(con, 0, 8)]
        acc += np.where(forced[:, None], b.F[pf], 0.0)
        out.append(acc)
    return out


def forward_np(b, xc):
    us, them = accumulators_np(b, xc)
    x = np.clip(np.concatenate([us[:, :A], them[:, :A]], 1), 0, 1)
    for i, (w, bb) in enumerate(b.dense):
        x = x @ w.T.astype(np.float64) + bb
        if i < len(b.dense) - 1:
            x = np.clip(x, 0, 1)
    return (x[:, 0] + 0.5 * (us[:, A] - them[:, A])) * 1000.0


def load_records(path, rows=0, offset=0):
    import eval_data
    src = eval_data.load(str(path))
    if not rows:
        return np.array(src)
    idx = np.linspace(0, len(src) - 1, rows).astype(np.int64) + offset
    return np.array(src[np.minimum(idx, len(src) - 1)])


# ---------------------------------------------------------------- quantizer
def psqt_rows(name, rows):
    m = np.zeros(rows, bool)
    if name == "proj":
        m[np.arange(9) * W + A] = True
    elif name in ("fwd", "dec", "con"):
        m[A] = True
    return m


def row_bits(name, rows, cfg):
    b = np.full(rows, cfg[GROUP[name]], np.int64)
    if name != "bias":
        b[psqt_rows(name, rows)] = cfg["psqt"]
    return b


def bf16_up(x):
    """Smallest bf16 >= x (x >= 0), as float32."""
    f = np.asarray(x, np.float32).copy()
    u = f.view(np.uint32)
    low = (u & 0xFFFF) != 0
    u[low] = (u[low] + 0x10000) & 0xFFFF0000
    return u.view(np.float32)


CLIP = 32767  # codes stay exact in float32 and int16


def scale_index(name, rows):
    """Row -> index of its scale: proj's 9 locations share one scale per lane (an absolute step per
    accumulator lane), every other matrix has one scale per row."""
    return np.arange(rows) % W if name == "proj" else np.arange(rows)


def row_steps(name, Mw, bits):
    """bf16 step per row: the smallest bf16 >= max|row group| / (2^(b-1) - 1)."""
    qmax = ((1 << bits) >> 1) - 1
    idx = scale_index(name, Mw.shape[0])
    need = np.zeros(idx.max() + 1)
    np.maximum.at(need, idx, np.abs(Mw).max(1) / qmax)
    return bf16_up(need)[idx], bf16_up(need)


def quant_col(w, s):
    q = np.where(s > 0, np.clip(np.rint(w / np.where(s > 0, s, 1)), -CLIP, CLIP), 0).astype(np.int64)
    return q, (q.astype(np.float32) * s.astype(np.float32)).astype(np.float64)


def gptq(Mw, X, wts, s, damp=0.01):
    """Round Mw (rows x cols) to the per-row steps s given calibration inputs X (N x cols) and sample weights:
    column by column, the rounding error fed back through the inverse input Hessian's Cholesky factor.
    X None: round to nearest."""
    Mw = Mw.astype(np.float64).copy()
    rows, cols = Mw.shape
    Q = np.zeros((rows, cols), np.int64)
    if X is None or not OPTS["gptq"]:
        for j in range(cols):
            Q[:, j], _ = quant_col(Mw[:, j], s)
        return Q
    H = (X * wts[:, None]).T @ X / wts.sum()
    dead = np.diag(H) == 0
    H[dead, dead] = 1
    H += damp * np.mean(np.diag(H)) * np.eye(cols)
    U = np.linalg.cholesky(np.linalg.inv(H)).T
    for j in range(cols):
        q, v = quant_col(Mw[:, j], s)
        Q[:, j] = q
        err = (Mw[:, j] - v) / U[j, j]
        Mw[:, j + 1:] -= np.outer(err, U[j, j + 1:])
    return Q


def refit(X, Z, wts, orig, group):
    """Weighted least squares Mw with X @ Mw.T ~ Z, for the groups in OPTS["refit_groups"]; else orig."""
    if not OPTS["refit"] or group not in OPTS["refit_groups"]:
        return orig
    Xw = X * wts[:, None]
    G = Xw.T @ X + OPTS["ridge"] * np.eye(X.shape[1]) * wts.sum()
    return np.linalg.solve(G, Xw.T @ Z).T


def quantize_mat(name, Mw, X, wts, cfg, Q, S, SC):
    """Steps for the matrix's rows (their bit widths from cfg), then GPTQ. Fills Q (codes), S (per-row step)
    and SC (the stored scales)."""
    bits = row_bits(name, Mw.shape[0], cfg)
    steps = np.zeros(Mw.shape[0], np.float32)
    SC[name] = None
    for b in np.unique(bits):  # a row's step comes from its own bit width (psqt rows: their own scale)
        sel = bits == b
        st, sc = row_steps(name, np.where(sel[:, None], Mw, 0), int(b))
        steps[sel] = st[sel]
        SC[name] = sc if SC[name] is None else np.where(sc > 0, sc, SC[name])
    assert np.array_equal(SC[name][scale_index(name, Mw.shape[0])], steps)
    S[name] = steps
    Q[name] = gptq(Mw, X, wts, steps)


def deq(Q, S):
    return (Q.astype(np.float32) * S[:, None]).astype(np.float32)


def pattern_weights(kind, calib, rows=40000):
    if kind == "uniform":
        return np.ones(int(LIVE.sum()))
    xc = compact(load_records(calib, rows))
    cells = xc[:, :81].astype(np.int64).reshape(-1, 9, 9)
    live = xc[:, 81:90] == 0
    pat = (cells * POW3).sum(2)
    cnt = np.bincount(np.concatenate([pat[live], PAT_SWAP[pat[live]]]), minlength=N_PAT).astype(np.float64)
    w = cnt[LIVE]
    return w / w.mean() + 0.05  # floor: every live pattern keeps some weight


def quantize(p, cfg, weights, calib, ncal):
    """-> codes {name: Q}, stored scales {name: SC}, the dequantized parameters."""
    M = to_mats(p)
    Q, S, SC = {}, {}, {}
    live1 = np.concatenate([np.ones((int(LIVE.sum()), 1)), ONEHOT[LIVE]], 1)
    wp = pattern_weights(weights, calib)
    h = ONEHOT[LIVE].astype(np.float64)  # float targets over the live patterns
    zf = []
    for i in range(3):
        z = h @ p[f"enc{i}_w"].T.astype(np.float64) + p[f"enc{i}_b"]
        zf.append(z)
        h = np.maximum(z, 0) if i < 2 else z
    Ef = h
    X = live1
    for i in range(3):
        name = f"enc{i}"
        Mw = M[name] if i == 0 else refit(X, zf[i], wp, M[name], "enc")
        quantize_mat(name, Mw, X, wp, cfg, Q, S, SC)
        z = X @ deq(Q[name], S[name]).T.astype(np.float64)
        X = np.concatenate([np.ones((len(z), 1)), np.maximum(z, 0) if i < 2 else z], 1)
    Eq1 = X  # [1 | quantized embedding] of the live patterns
    Tf = np.einsum("pk,lkj->lpj", Ef, p["proj_w"].astype(np.float64)) + p["proj_b"][:, None, :]
    Ff = Ef @ p["fwd_w"].astype(np.float64) + p["fwd_b"]
    Mp = np.concatenate([refit(Eq1, Tf[m], wp, M["proj"][m * W:(m + 1) * W], "proj") for m in range(9)], 0)
    quantize_mat("proj", Mp, Eq1, wp, cfg, Q, S, SC)  # GPTQ rows are independent: all 9 locations at once
    quantize_mat("fwd", refit(Eq1, Ff, wp, M["fwd"], "fwd"), Eq1, wp, cfg, Q, S, SC)
    for name in ("dec", "con", "bias"):
        quantize_mat(name, M[name], None, None, cfg, Q, S, SC)
    # dense head: calibration positions through the quantized first layer
    pq = from_mats({**M, **{n: deq(Q[n], S[n]) for n in Q}})
    xc = compact(load_records(calib, ncal, offset=7))  # not the pattern-frequency rows
    uf, tf = accumulators_np(to_bgn(p), xc)
    uq, tq = accumulators_np(to_bgn(pq), xc)
    xf = np.clip(np.concatenate([uf[:, :A], tf[:, :A]], 1), 0, 1)
    xq = np.clip(np.concatenate([uq[:, :A], tq[:, :A]], 1), 0, 1)
    one = np.ones((len(xc), 1))
    wd = np.ones(len(xc))
    for name, (wk, bk) in (("d1", ("w1", "b1")), ("d2", ("w2", "b2")), ("do", ("wo", "bo"))):
        zf_ = xf @ p[wk].T.astype(np.float64) + p[bk]
        Xq = np.concatenate([one, xq], 1)
        quantize_mat(name, refit(Xq, zf_, wd, M[name], "dense"), Xq, wd, cfg, Q, S, SC)
        zq = Xq @ deq(Q[name], S[name]).T.astype(np.float64)
        if name != "do":
            xf, xq = np.clip(zf_, 0, 1), np.clip(zq, 0, 1)
    return Q, SC, from_mats({n: deq(Q[n], S[n]) for n in MATS})


# ---------------------------------------------------------------- payload bits
def zigzag(q):
    q = np.asarray(q, np.int64)
    return (q << 1) ^ (q >> 63)


def rice_bits(u, k):
    return int(((u >> k) + 1 + k).sum())


def rice_k(u):
    return min(range(16), key=lambda k: rice_bits(u, k))


class BitWriter:
    def __init__(self):
        self.out, self.acc, self.nb = bytearray(), 0, 0

    def put(self, v, n):
        self.acc = (self.acc << n) | int(v)
        self.nb += n
        while self.nb >= 8:
            self.nb -= 8
            self.out.append((self.acc >> self.nb) & 0xFF)
        self.acc &= (1 << self.nb) - 1

    def done(self):
        if self.nb:
            self.put(0, 8 - self.nb)
        return bytes(self.out)


def pack(Q, SC):
    """Per matrix (MATS order): its scales as bf16 (16 bits each), the Rice parameters of its normal and PSQT
    rows (4 bits each; 0 for an empty group), then every code row by row as Rice(zigzag(q)): (u >> k)
    one-bits, a zero-bit, the low k bits of u."""
    w = BitWriter()
    for n in MATS:
        for s in SC[n]:
            u = int(np.float32(s).view(np.uint32))
            assert u & 0xFFFF == 0
            w.put(u >> 16, 16)
        ps = psqt_rows(n, Q[n].shape[0])
        ks = []
        for g in (~ps, ps):
            ks.append(rice_k(zigzag(Q[n][g]).ravel()) if g.any() else 0)
            w.put(ks[-1], 4)
        for r in range(Q[n].shape[0]):
            k = ks[1] if ps[r] else ks[0]
            for u in zigzag(Q[n][r]):
                u = int(u)
                for _ in range(u >> k):
                    w.put(1, 1)
                w.put(0, 1)
                w.put(u & ((1 << k) - 1), k)
    return w.done()


def unpack(data):
    """nnue_b64.hpp unpack() in numpy: payload bytes -> {name: dequantized matrix (float32)}, plus "_bits",
    the number of bits read."""
    bits = np.unpackbits(np.frombuffer(data, np.uint8))
    pos = 0

    def get(n):
        nonlocal pos
        v = 0
        for k in range(n):
            v = (v << 1) | int(bits[pos + k])
        pos += n
        return v

    out = {}
    for n in MATS:
        rows, cols = SHAPES[n]
        idx = scale_index(n, rows)
        sc = np.array([get(16) << 16 for _ in range(idx.max() + 1)], np.uint32).view(np.float32)
        ks = [get(4), get(4)]
        ps = psqt_rows(n, rows)
        Q = np.zeros((rows, cols), np.int64)
        for r in range(rows):
            k = ks[1] if ps[r] else ks[0]
            for c in range(cols):
                u = 0
                while get(1):
                    u += 1
                u = (u << k) | get(k)
                Q[r, c] = (u >> 1) ^ -(u & 1)
        out[n] = deq(Q, sc[idx])
    out["_bits"] = pos
    return out


# ---------------------------------------------------------------- what load() computes (float32, same order)
def bake_f32(p):
    """T (9, 3^9, 65) and F (3^9, 65) float32 exactly as nnue_b64.hpp load() bakes them: every sum is a
    float32 multiply then a float32 add in index order (no FMA), ReLU between encoder layers; rows of dead
    patterns are 0."""
    f = np.float32
    d = DIGITS[LIVE]
    e0w, e0b = p["enc0_w"].astype(f), p["enc0_b"].astype(f)
    h = np.repeat(e0b[None], len(d), 0)
    for k in range(9):  # one-hot input: 9 columns
        h = (h + e0w[:, 3 * k + d[:, k]].T).astype(f)
    h = np.where(h < 0, f(0), h)  # load(): h < 0 ? 0 : h (keeps -0.0, like std::max(h, 0.f))
    for i in (1, 2):
        w, b = p[f"enc{i}_w"].astype(f), p[f"enc{i}_b"].astype(f)
        s = np.repeat(b[None], len(d), 0)
        for j in range(w.shape[1]):
            s = (s + (w[:, j][None, :] * h[:, j:j + 1]).astype(f)).astype(f)
        h = np.where(s < 0, f(0), s) if i == 1 else s

    def project(pw, pb):
        row = np.repeat(pb.astype(f)[None], len(d), 0)
        for k in range(E):
            row = (row + (h[:, k:k + 1] * pw[k][None, :]).astype(f)).astype(f)
        return row

    T = np.zeros((9, N_PAT, W), f)
    for m in range(9):
        T[m, LIVE] = project(p["proj_w"][m].astype(f), p["proj_b"][m])
    F = np.zeros((N_PAT, W), f)
    F[LIVE] = project(p["fwd_w"].astype(f), p["fwd_b"])
    return T, F


def q_round(x, bits):
    """load()'s qz(): round half to even of x * 2^bits, in double."""
    return np.rint(np.asarray(x, np.float64) * float(1 << bits)).astype(np.int64)


def stone_sets():
    """di[p] = #mine - #theirs + 9 of each pattern; dec_mask[s]: the stone differences a board decided in
    state s (0 won by mine, 1 won by theirs, 2 drawn) can hide."""
    m = [((DIGITS == v) * (1 << np.arange(9))).sum(1) for v in range(3)]
    di = np.array([bin(x).count("1") for x in m[1]]) - np.array([bin(x).count("1") for x in m[2]]) + 9
    l1 = np.zeros(N_PAT, bool)
    l2 = np.zeros(N_PAT, bool)
    for ln in LINES:
        l1 |= (m[1] & ln) == ln
        l2 |= (m[2] & ln) == ln
    dec_mask = [set(di[l1 & ~l2]), set(di[l2 & ~l1]), set(di[(m[0] == 0) & ~l1 & ~l2])]
    return di, dec_mask


def scales(p, T, F):
    """(qa, qps, qb, q2, qo) and the bounds behind qa, for the dequantized generator p and its float32 tables:
    a port of the scale choice of the nnue2 experiment loader (fast_nnue_b.hpp quantize(), its default
    joint forced-board + stone-balance bound)."""
    ND, NS = 19, 9 * 18 + 1
    OFF = NS // 2
    di, dec_mask = stone_sets()
    live_idx = np.nonzero(LIVE)[0]
    dl = di[live_idx]
    Tl = T[:, live_idx, :A].astype(np.float64)  # (9, NL, A)
    Fl = F[live_idx, :A].astype(np.float64)
    dec = p["dec"].astype(np.float64)
    con = p["con"].astype(np.float64)
    bias = p["bias"].astype(np.float64)
    inf = np.inf
    # float extremes by board, stone difference and lane: T, and T + F (the forced board's joint term)
    tdh = np.full((9, ND, A), -inf)
    tdl = np.full((9, ND, A), inf)
    jdh = np.full((9, ND, A), -inf)
    jdl = np.full((9, ND, A), inf)
    for d in range(ND):
        sel = dl == d
        if sel.any():
            tdh[:, d] = Tl[:, sel].max(1)
            tdl[:, d] = Tl[:, sel].min(1)
            jdh[:, d] = (Tl[:, sel] + Fl[sel]).max(1)
            jdl[:, d] = (Tl[:, sel] + Fl[sel]).min(1)
    tmax, tmin = Tl.max(1), Tl.min(1)
    fmax, fmin = Fl.max(0), Fl.min(0)
    ps_abs = max(abs(float(bias[A])), float(np.abs(T[:, live_idx, A]).max()), float(np.abs(F[live_idx, A]).max()),
                 float(np.abs(dec[:, A]).max()), float(np.abs(con[:, A]).max()))

    def shift_add(cur, opt, hi):
        """cur (NS, A) partial sums by total d; opt (ND, A) one board's options by d -> (NS, A)."""
        out = np.full((NS, A), -inf if hi else inf)
        for d in range(ND):
            sh = d - ND // 2  # total t = s + d - 9
            lo_s, hi_s = max(0, -sh), min(NS, NS - sh)
            cand = cur[lo_s:hi_s] + opt[d][None, :]
            tgt = out[lo_s + sh:hi_s + sh]
            np.maximum(tgt, cand, out=tgt) if hi else np.minimum(tgt, cand, out=tgt)
        return out

    def best_total(v, allow):  # v (NS, A); allowed totals t = -1..1 (bit t + 1)
        return np.stack([v[OFF + t] for t in (-1, 0, 1) if allow >> (t + 1) & 1])

    def combine(x, u, allow, hi):  # best over allowed totals of x[s] + u[w], w = t - (s - OFF) + OFF
        vals = []
        for t in (-1, 0, 1):
            if not allow >> (t + 1) & 1:
                continue
            s = np.arange(NS)
            w = 2 * OFF + t - s
            ok = (w >= 0) & (w < NS)
            vals.append((x[s[ok]] + u[w[ok]]).max(0) if hi else (x[s[ok]] + u[w[ok]]).min(0))
        v = np.stack(vals)
        return v.max(0) if hi else v.min(0)

    allow_stored, allow_side = 7, (3, 6)
    for qa in range(14, 0, -1):
        sc = float(1 << qa)
        ok = True
        for pass_ in (0, 1):
            bq = q_round(bias[:A], qa).astype(np.float64)
            mr = np.abs(bq).max()
            oh = np.where(np.isfinite(tdh), q_round(np.where(np.isfinite(tdh), tdh, 0), qa), -inf)
            ol = np.where(np.isfinite(tdl), q_round(np.where(np.isfinite(tdl), tdl, 0), qa), inf)
            mr = max(mr, np.abs(q_round(tmax, qa)).max(), np.abs(q_round(tmin, qa)).max())
            for m in range(9):
                for s in range(3):
                    dq = q_round(dec[3 * m + s, :A], qa).astype(np.float64)
                    mr = max(mr, np.abs(dq).max())
                    for d in dec_mask[s]:
                        oh[m, d] = np.maximum(oh[m, d], dq)
                        ol[m, d] = np.minimum(ol[m, d], dq)
            preh = [np.full((NS, A), -inf)]
            prel = [np.full((NS, A), inf)]
            preh[0][OFF] = prel[0][OFF] = 0
            for k in range(9):
                preh.append(shift_add(preh[-1], oh[k], True))
                prel.append(shift_add(prel[-1], ol[k], False))
            sufh = [None] * 10
            sufl = [None] * 10
            sufh[9] = np.full((NS, A), -inf)
            sufl[9] = np.full((NS, A), inf)
            sufh[9][OFF] = sufl[9][OFF] = 0
            for k in range(8, -1, -1):
                sufh[k] = shift_add(sufh[k + 1], oh[k], True)
                sufl[k] = shift_add(sufl[k + 1], ol[k], False)
            shi = (bq + best_total(preh[9], allow_stored).max(0)).max()
            slo = (bq + best_total(prel[9], allow_stored).min(0)).min()
            fh, fl = q_round(fmax, qa), q_round(fmin, qa)
            mr = max(mr, np.abs(fh).max(), np.abs(fl).max())
            eh, el = [], []
            for side in (0, 1):
                v9 = q_round(con[10 * side + 9, :A], qa).astype(np.float64)
                mr = max(mr, np.abs(v9).max())
                eh.append(bq + best_total(preh[9], allow_side[side]).max(0) + v9)
                el.append(bq + best_total(prel[9], allow_side[side]).min(0) + v9)
            if pass_ == 1:
                qT = q_round(Tl, qa)
                qJ = qT + q_round(Fl, qa)[None]
                jqh = np.full((9, ND, A), -inf)
                jql = np.full((9, ND, A), inf)
                for d in range(ND):
                    sel = dl == d
                    if sel.any():
                        jqh[:, d] = qJ[:, sel].max(1)
                        jql[:, d] = qJ[:, sel].min(1)
            for c in range(9):
                v = [q_round(con[10 * side + c, :A], qa).astype(np.float64) for side in (0, 1)]
                mr = max(mr, np.abs(v[0]).max(), np.abs(v[1]).max())
                if pass_ == 0:
                    fin_h, fin_l = np.isfinite(jdh[c]), np.isfinite(jdl[c])
                    jh = np.where(fin_h, np.ceil(np.where(fin_h, jdh[c], 0) * sc - 1.0 - 1e-9), -inf)
                    jl = np.where(fin_l, np.floor(np.where(fin_l, jdl[c], 0) * sc + 1.0 + 1e-9), inf)
                else:
                    jh, jl = jqh[c], jql[c]
                xh = shift_add(preh[c], jh, True)
                xl = shift_add(prel[c], jl, False)
                for side in (0, 1):
                    eh[side] = np.maximum(eh[side], bq + combine(xh, sufh[c + 1], allow_side[side], True) + v[side])
                    el[side] = np.minimum(el[side], bq + combine(xl, sufl[c + 1], allow_side[side], False) + v[side])
            fhi = max(0.0, eh[0].max(), eh[1].max())
            flo = min(0.0, el[0].min(), el[1].min())
            shi, slo = max(0.0, shi), min(0.0, slo)
            ok = mr <= 32767 and shi <= 32767 and slo >= -32768 and fhi <= 32767 and flo >= -32768
            if not ok:
                break
        if ok:
            bounds = dict(max_row=int(mr), stored=(int(slo), int(shi)), with_con_f=(int(flo), int(fhi)))
            break
    else:
        raise SystemExit("first layer does not fit int16")
    qps = 20
    while qps > 0 and np.ldexp(ps_abs, qps) >= 32767.0:
        qps -= 1

    def fit(ws, bs, in_bits, top):
        for bits in range(top, -1, -1):
            qw = q_round(ws, bits)
            tot = np.abs(qw).sum(1) * (1 << in_bits) + np.abs(q_round(bs, bits + in_bits))
            if np.abs(qw).max() <= 32767 and (tot < (1 << 31) - 1).all():
                return bits
        raise SystemExit("dense layer does not fit")

    qb = fit(p["w1"], p["b1"], qa, 16)
    h1bits = min(14, qa + qb)
    q2 = fit(p["w2"], p["b2"], h1bits, 16)
    h2bits = min(15, h1bits + q2)
    qo = 20
    while qo >= 0:
        tot = np.abs(q_round(p["wo"], qo)).sum() * (1 << h2bits) + abs(int(q_round(p["bo"], qo + h2bits).reshape(-1)[0]))
        if tot < (1 << 31) - 1:
            break
        qo -= 1
    return (qa, qps, qb, q2, qo), bounds


def int_tables(p, T, F, qexp):
    """The 16 integer tables of load() at scales qexp, as numpy arrays in the C++ memory layout."""
    qa, qps, qb, q2, qo = qexp
    h1bits = min(14, qa + qb)
    h2bits = min(15, h1bits + q2)
    i16 = lambda x: np.asarray(x, np.int64).astype(np.int16)  # noqa: E731
    t = dict(T=i16(q_round(T[..., :A], qa)), TP=i16(q_round(T[..., A], qps)),
             F=i16(q_round(F[:, :A], qa)), FP=i16(q_round(F[:, A], qps)))
    for n, src in (("DEC", p["dec"]), ("CON", p["con"])):
        t[n] = i16(q_round(src[:, :A], qa))
        t[n + "P"] = i16(q_round(src[:, A], qps))
    t["BIAS"], t["BIASP"] = i16(q_round(p["bias"][:A], qa)), i16(q_round(p["bias"][A:], qps))
    pair = lambda w: (w[:, 0::2].astype(np.uint16).astype(np.uint32)  # noqa: E731
                      | (w[:, 1::2].astype(np.uint16).astype(np.uint32) << 16)).T.astype(np.int32)
    t["W1p"] = np.ascontiguousarray(pair(i16(q_round(p["w1"], qb))))  # [pair][out]
    t["B1"] = q_round(p["b1"], qa + qb).astype(np.int32)
    t["W2p"] = np.ascontiguousarray(pair(i16(q_round(p["w2"], q2))))
    t["B2"] = q_round(p["b2"], h1bits + q2).astype(np.int32)
    t["WO"] = q_round(p["wo"].reshape(-1), qo).astype(np.int32)
    t["BO"] = q_round(p["bo"].reshape(-1), qo + h2bits).astype(np.int32)
    return t


TABLE_ORDER = ["T", "TP", "F", "FP", "DEC", "DECP", "CON", "CONP", "BIAS", "BIASP", "W1p", "B1", "W2p", "B2", "WO", "BO"]


def fnv1a(data: bytes) -> int:
    """The table hash of the verified build's table_hash.cpp and of unit_tests.cpp: FNV-1a 64 with offset
    basis 1469598103934665603 (the standard basis is 14695981039346656037; the check only needs a fixed
    function, and this one matches the recorded hashes). Pure Python: about 2 s for T."""
    h = 1469598103934665603
    for byte in data:
        h = ((h ^ byte) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def table_hashes(t, kind="fnv1a"):
    """{table: hash of its bytes in the C++ layout}: fnv1a() (what unit_tests.cpp checks), or sha256
    (fast; what tools/test_nnue_emit_b64_header.py checks)."""
    raw = {n: np.ascontiguousarray(t[n]).tobytes() for n in TABLE_ORDER}
    if kind == "sha256":
        return {n: hashlib.sha256(raw[n]).hexdigest() for n in TABLE_ORDER}
    return {n: fnv1a(raw[n]) for n in TABLE_ORDER}


# ---------------------------------------------------------------- header
WIDTHS_RE = (r"B64_A = (\d+), B64_L1 = (\d+), B64_L2 = (\d+), B64_E = (\d+), B64_ENC0 = (\d+), "
             r"B64_ENC1 = (\d+);")


def parse_header(path):
    """-> (payload bytes, qexp) of a generated header; binds the module to the header's shape (a header
    from before the generic emitter declares no widths and is B-64)."""
    text = Path(path).read_text(encoding="utf-8")
    m = re.search(r"B64_QA = (\d+), B64_QPS = (\d+), B64_QB = (\d+), B64_Q2 = (\d+), B64_QO = (\d+);", text)
    body = re.search(r'B64_NET_CJK\[\] = R"~\((.*?)\)~"', text, re.S)
    if not m or not body:
        raise SystemExit(f"{path}: not a generated B64 net header")
    w = re.search(WIDTHS_RE, text)
    if w:
        a, l1, l2, e, enc0, enc1 = (int(x) for x in w.groups())
        set_shape(a, l1, l2, (enc0, enc1, e))
    else:
        set_shape()
    payload = decode_u15(body.group(1))  # may carry one zero byte of U15 padding
    payload = payload[:(unpack(payload)["_bits"] + 7) // 8]
    return payload, tuple(int(x) for x in m.groups())


def write_header(path, data, meta, qexp):
    text = encode_u15(data)
    assert decode_u15(text)[:len(data)] == data
    cfg = meta["config"]
    qa, qps, qb, q2, qo = qexp
    src = f"""#pragma once
// Generated by tools/nnue_emit_b64_header.py; do not edit. Net {meta["label"]} (lane-paired BGN1
// {Path(meta["net"]).name}): the B-{A} pattern generator, {n_params():,} parameters, GPTQ-rounded to bits
// {", ".join(f"{k} {v}" for k, v in cfg.items())}
// (pattern weights {meta["weights"]}, refit {meta["refit"]}), Rice-coded: {len(data):,} bytes = {len(text):,} U15 characters
// (payload sha256 {hashlib.sha256(data).hexdigest()[:16]}). Layout: the emitter's docstring; reader and runtime:
// nnue_b64.hpp. B64_A.. are the net's widths the runtime compiles for (A accumulator lanes, an L1 x L2
// head, a 27 -> ENC0 -> ENC1 -> E encoder); B64_QA..B64_QO are the power-of-two scales load() quantizes
// the baked tables and the dense head at; the emitter derives them from the dequantized net so that no
// int16 accumulator and no int32 sum can overflow.
#include <cmath>
#include <cstdint>
#include <cstring>
#include <immintrin.h>

static constexpr int B64_A = {A}, B64_L1 = {L1}, B64_L2 = {L2}, B64_E = {E}, B64_ENC0 = {ENC[0]}, B64_ENC1 = {ENC[1]};
static constexpr int B64_QA = {qa}, B64_QPS = {qps}, B64_QB = {qb}, B64_Q2 = {q2}, B64_QO = {qo};

static const char B64_NET_CJK[] = R"~(
{wrap_cjk14(text)}
)~";
"""
    Path(path).write_text(src, encoding="utf-8", newline="\n")
    return len(text)


def describe_tables(p, qexp=None, hashes=True):
    """Bake like load(), derive the scales, and (hashes) FNV-1a the integer tables at qexp (or the derived
    scales). -> (derived scales, bounds, hashes or None)."""
    t0 = time.time()
    T, F = bake_f32(p)
    got, bounds = scales(p, T, F)
    print(f"  baked T/F in float32 like load() and derived the scales in {time.time() - t0:.1f} s: "
          f"QA 2^{got[0]}, PSQT 2^{got[1]}, QB 2^{got[2]}, Q2 2^{got[3]}, QO 2^{got[4]}; max |row| "
          f"{bounds['max_row']}, stored accumulator [{bounds['stored'][0]}, {bounds['stored'][1]}], with constraint "
          f"and forced-board rows [{bounds['with_con_f'][0]}, {bounds['with_con_f'][1]}] (int16)", flush=True)
    hs = None
    if hashes:
        t0 = time.time()
        hs = table_hashes(int_tables(p, T, F, qexp or got))
        print(f"  integer table hashes (fnv1a) ({time.time() - t0:.0f} s):", flush=True)
        for i in range(0, len(TABLE_ORDER), 8):
            print("   " + " ".join(f"{n} {hs[n]:016x}" for n in TABLE_ORDER[i:i + 8]))
    return got, bounds, hs


def generator_params(dq):
    """The decoded matrices -> the parameter dict (the Gen layout of nnue_b64.hpp)."""
    return from_mats({n: dq[n] for n in MATS})


def cmd_check(path, hashes):
    payload, qexp = parse_header(path)
    dq = unpack(payload)  # parse_header() decoded it once already; cheap next to the bake
    p = generator_params(dq)
    print(f"{path}: payload {len(payload):,} bytes (sha256 {hashlib.sha256(payload).hexdigest()}), "
          f"shape A {A} head {L1}x{L2} encoder 27->{'->'.join(map(str, ENC))} ({n_params():,} parameters), "
          f"header scales {qexp}")
    got, _, _ = describe_tables(p, qexp, hashes)
    if tuple(got) != tuple(qexp):
        raise SystemExit(f"the header's scales {qexp} are not the ones this net needs {got}")
    print("scales OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("net", nargs="?", type=Path, help="lane-paired BGN1 file with a generator section")
    ap.add_argument("-o", "--out", type=Path, default=HEADER)
    ap.add_argument("--label", default=None, help="the net's name in the header comment (default: from the file name)")
    ap.add_argument("--config", default="", help="bits per group, e.g. enc=14,proj=12 (missing keys: the defaults)")
    ap.add_argument("--refit-groups", default=",".join(DEFAULT_REFIT),
                    help="groups refit by least squares before rounding (of enc,proj,fwd,dense; 'none')")
    ap.add_argument("--weights", default="freq", choices=["uniform", "freq"])
    ap.add_argument("--calib", type=Path, default=CALIB, help="eval_data records for the calibration statistics")
    ap.add_argument("--ncal", type=int, default=40000)
    ap.add_argument("--sample", type=Path, default=SAMPLE, help="records for the float error report (optional)")
    ap.add_argument("--no-hashes", action="store_true", help="skip the integer table hashes (slow in Python)")
    ap.add_argument("--check", nargs="?", const=HEADER, type=Path, metavar="HEADER",
                    help="decode, bake and check an existing header instead of emitting one")
    a = ap.parse_args()
    if a.check:
        cmd_check(a.check, not a.no_hashes)
        return
    if not a.net:
        ap.error("NET is required (or --check)")
    groups = tuple(g for g in a.refit_groups.split(",") if g and g != "none")
    if not set(groups) <= set(REFIT_GROUPS):
        ap.error(f"unknown refit group in {a.refit_groups}")
    OPTS.update(refit_groups=groups)
    t0 = time.time()
    p = gen_params(read_bgn(a.net))
    cfg = parse_config(a.config)
    print(f"{a.net}: A {A}, head {L1}x{L2}, encoder 27->{'->'.join(map(str, ENC))} ({n_params():,} parameters); "
          f"numpy {np.__version__}; config {cfg}; refit {','.join(groups) or 'none'}; weights {a.weights}; "
          f"calibration {a.ncal:,} rows of {a.calib}", flush=True)
    Q, SC, dq = quantize(p, cfg, a.weights, a.calib, a.ncal)
    data = pack(Q, SC)
    back = unpack(data)
    for n in MATS:
        assert np.array_equal(back[n], to_mats(dq)[n]), n
    print(f"  quantized and packed in {time.time() - t0:.0f} s: {len(data):,} bytes, round trip OK", flush=True)
    qexp, _, _ = describe_tables(dq, None, not a.no_hashes)
    label = a.label or a.net.stem.replace("_perm", "")
    meta = dict(net=str(a.net), label=label, config=cfg, weights=a.weights, refit=",".join(groups) or "none")
    chars = write_header(a.out, data, meta, qexp)
    print(f"wrote {a.out}: {len(data):,} bytes = {chars:,} U15 characters, scales {qexp}, payload sha256 "
          f"{hashlib.sha256(data).hexdigest()}")
    if a.sample and Path(a.sample).exists():
        xc = compact(load_records(a.sample))
        d = forward_np(to_bgn(dq), xc) - forward_np(to_bgn(p), xc)
        print(f"  float eval change on {len(xc):,} rows of {a.sample.name}: mean |d| {np.abs(d).mean():.3f}, "
              f"max |d| {np.abs(d).max():.1f}")
    print(json.dumps(dict(chars=chars, bytes=len(data), qexp=list(qexp))))


if __name__ == "__main__":
    main()
