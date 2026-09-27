#!/usr/bin/env python3
"""Holdout loss of the ENGINE's quantized static eval (fast_nnue_b, from-scratch path) next to the float net's.

  quant_loss.py NAME=CAND_DIR [NAME=CAND_DIR ...] [--threads 4] [--d8h-rows 300000]

NAME is a probe checkpoint (datasets/nnue2/probe/NAME.{pt,json}) exported to datasets/nnue2/fast/NAME_perm.bin;
CAND_DIR a candidate of the right kind (make_cand_b.py --kind b64|b128). The V2 holdout (all 482,136 rows)
and a fixed random sample of the scored D8H rows are written once as .cfdg (datasets/nnue2/stream/) and
relabeled with CAND_DIR/datagen.exe `label IN OUT 0 THREADS` (static_eval = the quantized eval) with
FASTNNUE_PATH = the net. Loss is gen_nnue's (win-probability MSE, K 1600) against the recorded search labels.
Merged into datasets/nnue2/probe/quant_loss.json.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gen_nnue_stream as S  # noqa: E402
import probe  # noqa: E402
import buckets  # noqa: E402
import eval_data  # noqa: E402
import nnue_train_blend as ntb  # noqa: E402

K = 1600.0
TOOLCHAIN = S.ROOT / "toolchains" / "llvm-mingw-20260616-ucrt-x86_64" / "bin"


def loss(e, s):
    return float(((1 / (1 + np.exp(-e / K)) - 1 / (1 + np.exp(-s / K))) ** 2).mean())


def sets(n_d8h):
    v2p, d8p = S.SDIR / "v2_holdout.cfdg", S.SDIR / f"d8h_sample{n_d8h}.cfdg"
    if not v2p.exists():
        rec, _, hold = ntb.load_all([str(p) for p in probe.DATA], None)
        rec[np.flatnonzero(hold)].tofile(v2p)
    if not d8p.exists():
        h = np.load(S.SDIR / "d8_holdout.npz")
        ok = np.flatnonzero(~((h["fidx"] == 0) & (h["recidx"] < S.BASE_ROWS)))
        pick = np.sort(np.random.default_rng(11).choice(ok, n_d8h, replace=False))
        parts = []
        for f in range(len(S.D8)):
            sel = pick[h["fidx"][pick] == f]
            parts.append(np.array(eval_data.load(str(S.D8[f]))[h["recidx"][sel]]))
        np.concatenate(parts).tofile(d8p)
    return {"V2": v2p, "D8H": d8p}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("specs", nargs="+")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--d8h-rows", type=int, default=300000)
    a = ap.parse_args()
    probe.below_normal_priority()
    torch.set_num_threads(2)
    files = sets(a.d8h_rows)
    recs = {k: np.fromfile(p, dtype=eval_data.REC) for k, p in files.items()}
    path = probe.OUT / "quant_loss.json"
    res = json.loads(path.read_text()) if path.exists() else {}
    res["_shipped"] = {k: loss(r["static_eval"].astype(np.float64), r["search"].astype(np.float64)) for k, r in recs.items()}
    res["_rows"] = {k: int(len(r)) for k, r in recs.items()}
    fx = probe.Feats(torch.device("cpu"))
    for spec in a.specs:
        name, cand = spec.split("=")
        net = S.ROOT / "datasets/nnue2/fast" / f"{name}_perm.bin"
        m, _ = buckets.load_model(name, fx)
        out = {}
        for k, r in recs.items():
            s = r["search"].astype(np.float64)
            with torch.no_grad():
                X = torch.from_numpy(probe.compact(r))
                ef = np.concatenate([m.fast(X[i:i + 65536]).numpy() for i in range(0, len(X), 65536)]).astype(np.float64)
            q = S.SDIR / f"_q_{name}_{k}.cfdg"
            q.unlink(missing_ok=True)
            env = dict(os.environ, PATH=str(TOOLCHAIN) + os.pathsep + os.environ["PATH"], FASTNNUE_PATH=str(net))
            subprocess.run([str(Path(cand) / "datagen.exe"), "label", str(files[k]), str(q), "0", str(a.threads)],
                           check=True, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            qr = np.fromfile(q, dtype=eval_data.REC)
            assert len(qr) == len(r) and (qr["s"] == r["s"]).all()
            eq = qr["static_eval"].astype(np.float64)
            q.unlink()
            lf, lq = loss(ef, s), loss(eq, s)
            ship = res["_shipped"][k]
            out[k] = dict(float_loss=lf, quant_loss=lq, float_vs_shipped_pct=100 * (lf / ship - 1),
                          quant_vs_shipped_pct=100 * (lq / ship - 1), mean_abs_diff=float(np.abs(eq - np.trunc(ef)).mean()))
            print(f"{name} {k}: float {lf:.6f} ({out[k]['float_vs_shipped_pct']:+.2f}%), quantized {lq:.6f}"
                  f" ({out[k]['quant_vs_shipped_pct']:+.2f}%), mean |q - float| {out[k]['mean_abs_diff']:.2f}", flush=True)
        res[name] = out
        path.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
