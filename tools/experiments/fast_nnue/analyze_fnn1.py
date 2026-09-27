#!/usr/bin/env python3
"""Weight and activation statistics of an FNN1 per-cell net, for choosing the integer scales of
fast_nnue.hpp (stage 1 of the fast NNUE path).

  analyze_fnn1.py NET.bin [SAMPLE.cfdg]

Prints max |weight| per layer, a rigorous per-lane bound on the first-layer accumulator over every
board the feature set can express (each miniboard contributes either its cells or one decided row,
plus one constraint row), and on SAMPLE the observed accumulator range, the fraction of clipped-ReLU
outputs that are exactly 0 / exactly 1, and the fraction of all-zero 4-lane input chunks (what a
sparse dense-layer kernel could skip).
"""
import struct
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "nnue2"))
sys.path.insert(0, str(HERE.parents[1]))
import eval_data  # noqa: E402


def load_fnn1(path):
    b = Path(path).read_bytes()
    assert b[:4] == b"FNN1", "not an FNN1 file"
    A, L1 = struct.unpack("<ii", b[4:12])
    f = np.frombuffer(b[12:], dtype="<f4")
    o = 0

    def take(n, shape):
        nonlocal o
        x = f[o:o + n].reshape(shape)
        o += n
        return x
    W0 = take(200 * A, (200, A)); B0 = take(A, (A,)); W1 = take(L1 * 2 * A, (L1, 2 * A))
    B1 = take(L1, (L1,)); W2 = take(L1, (L1,)); B2 = float(take(1, (1,))[0])
    assert o == len(f)
    return A, L1, W0, B0, W1, B1, W2, B2


def features(rec):
    """Active feature lists (stm view, ntm view) per record, as fast_nnue.hpp builds them."""
    s = np.frombuffer(rec["s"].tobytes(), dtype=np.uint8).reshape(-1, 93).astype(np.int16) - 48
    out = []
    for r in s:
        stm = 1 if r[90] == 1 else 2  # UTTTAI: '1' / '2' players
        cells, sup, con = r[:81], r[81:90], min(max(int(r[91]), 0), 9)
        views = []
        for P in (stm, 3 - stm):
            f = []
            for mb in range(9):
                if sup[mb] != 0:
                    cls = 2 if sup[mb] == 3 else (0 if sup[mb] == P else 1)
                    f.append(162 + mb * 3 + cls)
                    continue
                for sq in range(9):
                    v = cells[mb * 9 + sq]
                    if v:
                        f.append((mb * 9 + sq) * 2 + (v != P))
            f.append(189 + con)
            views.append(f)
        out.append(views)
    return out


def main():
    A, L1, W0, B0, W1, B1, W2, B2 = load_fnn1(sys.argv[1])
    print(f"A={A} L1={L1}")
    for n, x in (("W0", W0[:199]), ("B0", B0), ("W1", W1), ("B1", B1), ("W2", W2)):
        print(f"  max|{n}| {np.abs(x).max():.4f}  mean|{n}| {np.abs(x).mean():.4f}")
    print(f"  B2 {B2:.4f}  sum|W2| {np.abs(W2).sum():.4f}")
    # rigorous accumulator bound per lane (without the constraint row, which fast_nnue adds at eval)
    hi, lo = B0.copy(), B0.copy()
    for mb in range(9):
        cells_hi = np.zeros(A); cells_lo = np.zeros(A)
        for sq in range(9):
            r = W0[(mb * 9 + sq) * 2:(mb * 9 + sq) * 2 + 2]
            cells_hi += np.maximum(0, r.max(0)); cells_lo += np.minimum(0, r.min(0))
        dec = W0[162 + mb * 3:165 + mb * 3]
        hi += np.maximum(cells_hi, dec.max(0)); lo += np.minimum(cells_lo, dec.min(0))
    con = W0[189:199]
    print(f"  acc bound (no constraint row): [{lo.min():.3f}, {hi.max():.3f}]; with constraint "
          f"[{(lo + con.min(0)).min():.3f}, {(hi + con.max(0)).max():.3f}]")
    print(f"  max |W1| row-sum over 2A (L1 pre-activation bound, excl. bias): "
          f"{np.abs(W1).sum(1).max():.3f}")
    if len(sys.argv) > 2:
        rec = np.fromfile(sys.argv[2], dtype=eval_data.REC)
        feats = features(rec)
        acc = np.zeros((len(rec), 2, A))
        for i, views in enumerate(feats):
            for v in range(2):
                acc[i, v] = B0 + W0[views[v]].sum(0)
        h = np.clip(acc, 0, 1).reshape(len(rec), 2 * A)
        z = (h == 0).mean(); o = (h == 1).mean()
        chunks = h.reshape(len(rec), -1, 4)
        zc = (chunks.max(2) == 0).mean()
        s = h @ W1.T + B1
        e = (np.clip(s, 0, 1) @ W2 + B2) * 1000.0
        print(f"sample {len(rec):,}: acc range [{acc.min():.3f}, {acc.max():.3f}], crelu zero {z:.3f}, "
              f"one {o:.3f}, zero 4-chunks {zc:.3f}")
        print(f"  L1 pre-activation range [{s.min():.3f}, {s.max():.3f}], zero {(s <= 0).mean():.3f}, "
              f"one {(s >= 1).mean():.3f}; eval std {e.std():.1f}, "
              f"|eval - file static_eval| mean {np.abs(np.trunc(e) - rec['static_eval']).mean():.3f}")


if __name__ == "__main__":
    main()
