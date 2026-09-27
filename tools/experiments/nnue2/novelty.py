#!/usr/bin/env python3
"""Leakage check: which holdout positions also occur (as exact states, or as a
symmetric image) in the training rows: the 4.3M depth-14 training rows and
the first 5M records of datasets/nnue2/d8_a.cfdg.

Writes datasets/nnue2/probe/holdout_seen.npz with boolean masks over the
holdout rows (in load order): in_d14 / in_d14_sym / in_d8 / in_d8_sym.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe  # noqa: E402

ntb, eval_data = probe.ntb, probe.eval_data
W = np.random.default_rng(99).integers(1, 2 ** 63 - 1, size=(93,), dtype=np.uint64) | np.uint64(1)


def h64(s):
    """(n, 93) uint8 -> (n,) uint64 hash (wrapping multiply-add over the columns)."""
    h = np.zeros(len(s), dtype=np.uint64)
    for j in range(93):
        h = h * np.uint64(1099511628211) + s[:, j].astype(np.uint64) * W[j]
    return h


def states(rec):
    s = np.frombuffer(rec["s"].tobytes(), dtype=np.uint8).reshape(-1, 93).copy()
    s[:, 92] = 48
    return s


def main():
    probe.below_normal_priority()
    rec, _, hold = ntb.load_all([str(p) for p in probe.DATA], None)
    s = states(rec)
    tr_h = np.unique(h64(s[~hold]))
    va = s[hold]
    er = np.array(np.memmap(ROOT_D8, dtype=eval_data.REC, mode="r", shape=(5_000_000,)))
    er = er[(er["flags"] & eval_data.F_SEARCH) != 0]
    d8_h = np.unique(h64(states(er)))
    res = {}
    for name, ref in (("d14", tr_h), ("d8", d8_h)):
        exact = np.isin(h64(va), ref)
        sym = exact.copy()
        for g in range(1, 8):
            sym |= np.isin(h64(probe.transform_states(va, np.full(len(va), g))), ref)
        res[f"in_{name}"], res[f"in_{name}_sym"] = exact, sym
        probe.log(f"holdout positions also in {name} training rows: exact {exact.mean():.1%},"
                  f" up to symmetry {sym.mean():.1%}")
    both = res["in_d14_sym"] | res["in_d8_sym"]
    probe.log(f"novel (in neither, up to symmetry): {(~both).mean():.1%} of {len(va):,}")
    np.savez(probe.OUT / "holdout_seen.npz", **res)


ROOT_D8 = str(probe.ROOT / "datasets/nnue2/d8_a.cfdg")

if __name__ == "__main__":
    main()
