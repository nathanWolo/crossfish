#!/usr/bin/env python3
"""Test (b): the quantized fast NNUE against the float net, in eval units.

  compare_float.py [--net NET.bin] [--cand DIR] [--exe datagen.exe] [--tag TAG] [--sample parity_in.cfdg]
                   [--float-labeled F.cfdg]

Relabels SAMPLE's statics with DIR/datagen.exe (`label IN OUT 0 2`: static_eval = the candidate's
evaluate(), i.e. fast_nnue's from-scratch quantized path) into datasets/nnue2/fast/, computes the
float reference in numpy from the FNN1 file (the float C++ hook's exact math: truncation toward
zero), cross-checks that reference against a float-labeled file when one is given (the
cand_fnn1 / PyTorch parity file), and prints mean |d|, max |d|, mean d, percentiles and the
correlation, overall and by game phase. --exe picks another datagen build in DIR; --tag TAG writes
parity_q_<net>_<TAG>.cfdg instead of parity_q_<net>.cfdg.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "tools"))
import eval_data  # noqa: E402
from analyze_fnn1 import features, load_fnn1  # noqa: E402

TOOLCHAIN = ROOT / "toolchains" / "llvm-mingw-20260616-ucrt-x86_64" / "bin"


def float_eval(net, rec):
    A, L1, W0, B0, W1, B1, W2, B2 = load_fnn1(net)
    W0 = W0.astype(np.float64)
    out = np.zeros(len(rec))
    for i, views in enumerate(features(rec)):
        h = np.concatenate([np.clip(B0 + W0[v].sum(0), 0, 1) for v in views])
        s = np.clip(W1.astype(np.float64) @ h + B1, 0, 1)
        out[i] = (s @ W2 + B2) * 1000.0
    return np.trunc(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", default=str(ROOT / "datasets/nnue2/fnn1/r10_aug_d8x5M_lr6e3.bin"))
    ap.add_argument("--cand", default=str(ROOT / "cpp_impl/bin/cand_fast"))
    ap.add_argument("--sample", default=str(ROOT / "datasets/nnue2/fnn1/parity_in.cfdg"))
    ap.add_argument("--float-labeled", default=str(ROOT / "datasets/nnue2/fnn1/parity_r10_aug_d8x5M_lr6e3.cfdg"))
    ap.add_argument("--exe", default="datagen.exe")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    out = ROOT / "datasets/nnue2/fast" / f"parity_q_{Path(a.net).stem}{'_' + a.tag if a.tag else ''}.cfdg"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.unlink(missing_ok=True)
    env = dict(os.environ, PATH=str(TOOLCHAIN) + os.pathsep + os.environ.get("PATH", ""), FASTNNUE_PATH=a.net)
    subprocess.run([str(Path(a.cand) / a.exe), "label", a.sample, str(out), "0", "2"], check=True, env=env,
                   stdout=subprocess.DEVNULL)
    q = np.fromfile(out, dtype=eval_data.REC)
    rec = np.fromfile(a.sample, dtype=eval_data.REC)
    assert len(q) == len(rec)
    f = float_eval(a.net, rec)
    if a.float_labeled and Path(a.float_labeled).exists():
        fl = np.fromfile(a.float_labeled, dtype=eval_data.REC)["static_eval"]
        print(f"float reference vs {Path(a.float_labeled).name} (float C++ hook): max |d| {np.abs(f - fl).max():.0f}")
    qe = q["static_eval"].astype(np.float64)
    d = qe - f

    def report(name, m):
        if m.sum() < 2:
            return
        dd = d[m]
        corr = np.corrcoef(qe[m], f[m])[0, 1]
        p = np.percentile(np.abs(dd), [50, 90, 99, 99.9])
        print(f"{name:>14}: n {m.sum():6d}  mean|d| {np.abs(dd).mean():6.3f}  max|d| {np.abs(dd).max():5.0f}  "
              f"mean d {dd.mean():+6.3f}  |d| p50/p90/p99/p99.9 {p[0]:.0f}/{p[1]:.0f}/{p[2]:.0f}/{p[3]:.0f}  "
              f"corr {corr:.7f}  (float std {f[m].std():.0f})")

    print(f"quantized fast NNUE ({Path(a.net).name}; {Path(a.cand).name}/{a.exe}) vs float, {len(q):,} positions of "
          f"{Path(a.sample).name}:")
    report("all", np.ones(len(q), bool))
    ply = rec["ply"]
    for lo, hi in ((0, 20), (20, 40), (40, 82)):
        report(f"ply {lo}-{hi - 1}", (ply >= lo) & (ply < hi))


if __name__ == "__main__":
    main()
