#!/usr/bin/env python3
"""Memorization check for the long-training round: each net's loss on rows it trained on next to rows from
the same sources it never saw (no augmentation, eval mode, gen_nnue's loss at K 1600).

  long_gap.py NAME... [--device cpu|dml] [--rows 1000000]

  eval2 train   a fixed random sample of the eval2 training rows (in-sample for every run here)
  V2            all of gen_nnue.py's holdout (10% of the eval2 games; out-of-sample for every run)
  d8_a 5M       a fixed random sample of the labeled rows among d8_a.cfdg's first 5M records (in-sample for
                the d5M runs, out-of-sample for B64_e2only_114st)
  D8H           a fixed random sample of the scored D8H rows (other games of the same depth-8 self-play;
                out-of-sample for every run)
gap = holdout loss - in-sample loss of the same source. Writes datasets/nnue2/probe/long_gap.json (merged).
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen_nnue_stream as S  # noqa: E402
import buckets  # noqa: E402
import probe  # noqa: E402
import eval_data  # noqa: E402

K = 1600.0


def loss(e, s):
    return float(((1 / (1 + np.exp(-e / K)) - 1 / (1 + np.exp(-s.astype(np.float64) / K))) ** 2).mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="+")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--rows", type=int, default=1_000_000)
    a = ap.parse_args()
    probe.below_normal_priority()
    torch.set_num_threads(a.threads)
    t0 = time.time()
    rng = np.random.default_rng(20260927)
    e2 = S.load_eval2()
    tr = np.flatnonzero(~e2["hold"])
    va = np.flatnonzero(e2["hold"])
    pick = np.sort(rng.choice(tr, min(a.rows, len(tr)), replace=False))
    sets = {"eval2 train": (e2["x"][pick], e2["search"][pick], e2["static"][pick]),
            "V2": (e2["x"][va], e2["search"][va], e2["static"][va])}
    del e2
    mm = np.array(eval_data.load(str(S.D8[0]))[:S.BASE_ROWS])
    mm = mm[(mm["flags"] & eval_data.F_SEARCH) != 0]
    p8 = np.sort(rng.choice(len(mm), min(a.rows, len(mm)), replace=False))
    r8 = mm[p8]
    del mm
    sets["d8_a 5M"] = (probe.compact(r8), r8["search"].astype(np.float32), r8["static_eval"].astype(np.float32))
    h = S.load_d8h(K)
    ph = np.sort(rng.choice(len(h["search"]), min(a.rows, len(h["search"])), replace=False))
    sets["D8H"] = (h["x"][ph], h["search"][ph], h["static"][ph])
    print(f"sets: " + ", ".join(f"{k} {len(v[1]):,}" for k, v in sets.items()) + f" ({time.time() - t0:.0f}s)", flush=True)
    dev = S.get_dev(a.device)
    fx = probe.Feats(dev)
    path = probe.OUT / "long_gap.json"
    res = json.loads(path.read_text()) if path.exists() else {}
    res["_shipped"] = {k: loss(v[2].astype(np.float64), v[1]) for k, v in sets.items()}
    res["_rows"] = {k: int(len(v[1])) for k, v in sets.items()}
    for n in a.names:
        m, _ = buckets.load_model(n, fx)
        out = {}
        for k, (x, s, st) in sets.items():
            e, _, _ = S.predict(m, torch.from_numpy(np.ascontiguousarray(x)), dev)
            out[k] = loss(e, s)
        out["gap_eval2"] = out["V2"] - out["eval2 train"]
        out["gap_d8"] = out["D8H"] - out["d8_a 5M"]
        res[n] = out
        print(f"{n}: " + ", ".join(f"{k} {v:.6f}" for k, v in out.items()) + f" ({time.time() - t0:.0f}s)", flush=True)
        path.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
