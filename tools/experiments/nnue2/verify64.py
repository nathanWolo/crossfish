#!/usr/bin/env python3
"""export_bgn.py verify in float64: is a BGN1 file's float net exact, or is the gap torch float32 rounding?

  verify64.py NAME [--bgn datasets/nnue2/fast/NAME_perm.bin] [--n 20000] [--log FILE]

export_bgn.py verify compares the file's baked float32 tables (evaluated in float64 by forward_np) with
PyTorch Gen.forward in float32 and fails above 0.05 eval units. Nets with large first-layer rows (long
training) exceed that on a few rows through float32 accumulation in Gen.forward alone. This script
evaluates the checkpoint with Gen.forward in float64 and compares all three: file (float64 arithmetic on
the float32 tables), torch float32 and torch float64, on the same parity_in.cfdg rows. It also compares
the integer-truncated outputs (what datagen / compare_bgn see). fast_nnue/export_net.sh falls back to it.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools/experiments/fast_nnue"))
from export_bgn import read_bgn, load_gen, forward_np, compact_rows  # noqa: E402

sys.path.insert(0, str(ROOT / "tools"))
import eval_data  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name")
    ap.add_argument("--bgn")
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--sample", default=str(ROOT / "datasets/nnue2/fnn1/parity_in.cfdg"))
    ap.add_argument("--log", help="also append the report to this file (the long-training round used"
                                  " datasets/nnue2/fast/long/verify64.log)")
    a = ap.parse_args()
    bgn = a.bgn or str(ROOT / "datasets/nnue2/fast" / f"{a.name}_perm.bin")
    b = read_bgn(bgn)
    m, fx, _ = load_gen(a.name)
    rec = np.fromfile(a.sample, dtype=eval_data.REC)[: a.n]
    xc = compact_rows(rec)
    with torch.no_grad():
        cells, sup, con = fx.split(torch.from_numpy(xc))
        f32 = m(cells, sup, con).numpy().astype(np.float64)
        m64 = m.double()
        f64 = m64(cells, sup, con).numpy().astype(np.float64)
    got = forward_np(b, xc)
    lines = [f"[{time.strftime('%H:%M:%S')}] {a.name} ({Path(bgn).name}), {len(xc):,} rows of {Path(a.sample).name}:"]
    for lab, x, y in (("file vs torch float32", got, f32), ("file vs torch float64", got, f64),
                      ("torch float32 vs torch float64", f32, f64)):
        d = np.abs(x - y)
        lines.append(f"  {lab}: max |d| {d.max():.2e}, mean |d| {d.mean():.2e}, rows > 0.05: {(d > 0.05).sum()},"
                     f" trunc mismatches {(np.trunc(x) != np.trunc(y)).sum()}")
    out = "\n".join(lines)
    print(out)
    if a.log:
        with open(a.log, "a", encoding="utf-8") as f:
            f.write(out + "\n")


if __name__ == "__main__":
    main()
