#!/usr/bin/env python3
"""Reorder an FNN1 net's accumulator lanes so fast_nnue.hpp's sparse dense layer does less work.

  permute_fnn1.py NET.bin OUT.bin [--stats SAMPLE.cfdg] [--rows N] [--eval EVAL.cfdg]

fast_nnue feeds the 2A clipped-ReLU outputs to _mm256_madd_epi16 in lane pairs (2p, 2p+1) and skips
a pair only when both are zero. Permuting the A lanes (W0 columns, B0, and the matching W1 columns
of both halves) leaves the network's function unchanged -- the quantized integer eval is
bit-identical, since each weight quantizes on its own and every sum is exact -- but pairing lanes
that fire together means fewer nonzero pairs. Pairs are a greedy maximum-weight matching on the
joint activation frequency P(lane i > 0 and lane j > 0), measured on N rows (default 20,000) spread
over STATS (default datasets/nnue2/d8_a.cfdg, not the parity sample), pooled over both views.
Prints the mean nonzero pairs per evaluation before and after on EVAL (default parity_in.cfdg)
and checks the float outputs agree.
"""
import argparse
import struct
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "tools"))
import eval_data  # noqa: E402
from analyze_fnn1 import features, load_fnn1  # noqa: E402


def activations(net, rec):
    A, L1, W0, B0, W1, B1, W2, B2 = net
    acc = np.zeros((len(rec), 2, A), np.float64)
    for i, views in enumerate(features(rec)):
        for v in range(2):
            acc[i, v] = B0 + W0[views[v]].astype(np.float64).sum(0)
    return acc


def forward(net, acc):
    A, L1, W0, B0, W1, B1, W2, B2 = net
    h = np.clip(acc, 0, 1).reshape(len(acc), 2 * A)
    return np.trunc((np.clip(h @ W1.T.astype(np.float64) + B1, 0, 1) @ W2 + B2) * 1000.0)


def nonzero_pairs(act, perm):
    """Mean nonzero (2p, 2p+1) pairs per evaluation (both views), lanes in order PERM."""
    x = act[:, :, perm].reshape(len(act), 2, -1, 2).max(3)
    return x.sum((1, 2)).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("net")
    ap.add_argument("out")
    ap.add_argument("--stats", default=str(ROOT / "datasets/nnue2/d8_a.cfdg"))
    ap.add_argument("--rows", type=int, default=20000)
    ap.add_argument("--eval", default=str(ROOT / "datasets/nnue2/fnn1/parity_in.cfdg"))
    a = ap.parse_args()
    net = load_fnn1(a.net)
    A = net[0]
    src = eval_data.load(a.stats)
    idx = np.linspace(0, len(src) - 1, a.rows).astype(np.int64)
    act = activations(net, np.array(src[idx])) > 0
    flat = act.reshape(-1, A).astype(np.float64)
    J = flat.T @ flat / len(flat)
    np.fill_diagonal(J, -1.0)
    order = np.argsort(-J, axis=None)
    used = np.zeros(A, bool)
    perm = []
    for k in order:
        i, j = divmod(int(k), A)
        if i < j and not used[i] and not used[j]:
            used[i] = used[j] = True
            perm += [i, j]
        if len(perm) == A:
            break
    perm = np.array(perm)
    assert sorted(perm) == list(range(A))

    _, L1, W0, B0, W1, B1, W2, B2 = net
    W1p = np.concatenate([W1[:, :A][:, perm], W1[:, A:][:, perm]], 1)
    blob = b"FNN1" + struct.pack("<ii", A, L1)
    for x in (W0[:, perm], B0[perm], W1p, B1, W2):
        blob += np.ascontiguousarray(x, dtype="<f4").tobytes()
    blob += struct.pack("<f", B2)
    Path(a.out).write_bytes(blob)
    new = load_fnn1(a.out)

    ev = np.fromfile(a.eval, dtype=eval_data.REC)
    acc0 = activations(net, ev)
    acc1 = activations(new, ev)
    d = np.abs(forward(net, acc0) - forward(new, acc1))
    e = acc0 > 0
    print(f"wrote {a.out}: lanes paired on {a.rows:,} rows of {Path(a.stats).name}")
    print(f"  {Path(a.eval).name}: nonzero pairs per eval {nonzero_pairs(e, np.arange(A)):.2f} -> "
          f"{nonzero_pairs(e, perm):.2f} (active lanes {e.sum((1, 2)).mean():.2f}); "
          f"float output max |d| {d.max():.0f} (summation order only)")


if __name__ == "__main__":
    main()
