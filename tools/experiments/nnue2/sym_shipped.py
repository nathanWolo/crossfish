#!/usr/bin/env python3
"""Holdout loss of the shipped eval with its learned part (MiniNet + macro)
averaged over the 8 dihedral symmetries. The HCE is exactly symmetric
(probe.py symcheck), so E_sym = hce + mean_g (mini + macro)(g(p)).

Same holdout, target and K as probe.py. CPU, one thread, below-normal priority.
Writes datasets/nnue2/probe/sym_shipped.json.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe  # noqa: E402

ntb = probe.ntb


def main():
    probe.below_normal_priority()
    torch.set_num_threads(1)
    rec, _, hold = ntb.load_all([str(p) for p in probe.DATA], None)
    r = rec[hold]
    del rec
    k = 1600.0
    y = 1 / (1 + np.exp(-r["search"].astype(np.float64) / k))
    ev = ntb.Eval(ntb.load_shipped_mini(), ntb.load_shipped_macro(), "centroid").eval()
    s = np.frombuffer(r["s"].tobytes(), dtype=np.uint8).reshape(-1, 93)
    hce = torch.from_numpy(r["hce"].astype(np.float32))
    learned = []
    T = lambda a: torch.from_numpy(np.ascontiguousarray(a)).long()  # noqa: E731
    for g in range(8):
        t = r.copy()
        t["s"] = np.frombuffer(probe.transform_states(s, np.full(len(s), g)).tobytes(), dtype="S93")
        mi, su, co, sg = ntb.features(t)
        out = []
        with torch.no_grad():
            for i in range(0, len(t), 65536):
                sl = slice(i, i + 65536)
                e, _, _ = ev(T(mi[sl]), T(su[sl]), T(co[sl]), hce[sl], T(sg[sl]).float())
                out.append((e - hce[sl]).numpy())
        learned.append(np.concatenate(out).astype(np.float64))
        probe.log(f"symmetry {g} done")
    L = np.stack(learned)
    h = r["hce"].astype(np.float64)
    res = {}
    for name, e in (("shipped_python", h + L[0]), ("shipped_cpp", r["static_eval"].astype(np.float64)),
                    ("sym_mean8", h + L.mean(0)), ("hce_only", h)):
        p = 1 / (1 + np.exp(-e / k))
        res[name] = dict(loss=float(((p - y) ** 2).mean()), std=float(e.std()))
    res["learned_std"] = float(L[0].std())
    res["learned_asym_mean_abs"] = float(np.abs(L - L.mean(0)).mean())
    res["rows"] = int(len(r))
    out = probe.OUT / "sym_shipped.json"
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
