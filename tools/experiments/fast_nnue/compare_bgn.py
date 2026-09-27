#!/usr/bin/env python3
"""Test (b) for the pattern-generator nets: the quantized fast path against PyTorch float, in eval units.

  compare_bgn.py NAME [--net NET.bin] [--cand DIR] [--exe datagen.exe] [--tag TAG] [--sample parity_in.cfdg] [--bake]

Relabels SAMPLE's statics with DIR/datagen.exe (`label IN OUT 0 2`: static_eval = the candidate's
evaluate(), i.e. fast_nnue_b's from-scratch quantized path) into datasets/nnue2/fast/parity_q_<net>.cfdg,
computes the float reference with PyTorch Gen.forward from the checkpoint NAME (datasets/nnue2/probe),
truncated toward zero like the C++ output, and prints mean |d|, max |d|, mean d, |d| percentiles and the
correlation, overall and by game phase. --net defaults to datasets/nnue2/fast/<NAME>_perm.bin.
--bake runs the candidate with FASTNNUE_BAKE=1 (tables re-baked in C++ from the file's generator section,
what a build without the tables would play with); output parity_q_<net>_bake.cfdg.
--exe picks another datagen build in DIR (a variant compiled with other -D flags); --tag TAG writes
parity_q_<net>[_bake]_<TAG>.cfdg instead (stage 3: new files next to the earlier stages' outputs).
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
from export_bgn import load_gen  # noqa: E402

sys.path.insert(0, str(ROOT / "tools"))
import eval_data  # noqa: E402
import probe  # noqa: E402

TOOLCHAIN = ROOT / "toolchains" / "llvm-mingw-20260616-ucrt-x86_64" / "bin"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--net")
    ap.add_argument("--cand", default=str(ROOT / "cpp_impl/bin/cand_fastb"))
    ap.add_argument("--sample", default=str(ROOT / "datasets/nnue2/fnn1/parity_in.cfdg"))
    ap.add_argument("--bake", action="store_true")
    ap.add_argument("--exe", default="datagen.exe")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    net = a.net or str(ROOT / "datasets/nnue2/fast" / f"{a.name}_perm.bin")
    out = ROOT / "datasets/nnue2/fast" / (f"parity_q_{Path(net).stem}{'_bake' if a.bake else ''}"
                                          f"{'_' + a.tag if a.tag else ''}.cfdg")
    out.unlink(missing_ok=True)
    env = dict(os.environ, PATH=str(TOOLCHAIN) + os.pathsep + os.environ.get("PATH", ""), FASTNNUE_PATH=net,
               FASTNNUE_BAKE="1" if a.bake else "0")
    subprocess.run([str(Path(a.cand) / a.exe), "label", a.sample, str(out), "0", "2"], check=True, env=env,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    q = np.fromfile(out, dtype=eval_data.REC)
    rec = np.fromfile(a.sample, dtype=eval_data.REC)
    assert len(q) == len(rec)
    m, fx, _ = load_gen(a.name)
    with torch.no_grad():
        cells, sup, con = fx.split(torch.from_numpy(probe.compact(rec)))
        f = np.trunc(m(cells, sup, con).numpy().astype(np.float64))
    qe = q["static_eval"].astype(np.float64)
    d = qe - f

    def report(name, sel):
        if sel.sum() < 2:
            return
        dd = d[sel]
        corr = np.corrcoef(qe[sel], f[sel])[0, 1]
        p = np.percentile(np.abs(dd), [50, 90, 99, 99.9])
        print(f"{name:>14}: n {sel.sum():6d}  mean|d| {np.abs(dd).mean():6.3f}  max|d| {np.abs(dd).max():5.0f}  "
              f"mean d {dd.mean():+6.3f}  |d| p50/p90/p99/p99.9 {p[0]:.0f}/{p[1]:.0f}/{p[2]:.0f}/{p[3]:.0f}  "
              f"corr {corr:.7f}  (float std {f[sel].std():.0f})")

    print(f"quantized fast NNUE ({Path(net).name}{', C++-baked tables' if a.bake else ''}; {Path(a.cand).name}/{a.exe}) "
          f"vs PyTorch float ({a.name}), {len(q):,} positions of {Path(a.sample).name}:")
    report("all", np.ones(len(q), bool))
    ply = rec["ply"]
    for lo, hi in ((0, 20), (20, 40), (40, 82)):
        report(f"ply {lo}-{hi - 1}", (ply >= lo) & (ply < hi))
    xc = probe.compact(rec)
    report("free move", xc[:, 90] == 9)
    report("forced move", xc[:, 90] < 9)


if __name__ == "__main__":
    main()
