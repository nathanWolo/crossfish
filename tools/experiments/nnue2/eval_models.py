#!/usr/bin/env python3
"""Holdout breakdown for saved probe models (datasets/nnue2/probe/<name>.pt):
loss, output std and correlation on all / non-mate (|search| < 8000) / mate
(|search| = 20000) rows, next to the shipped eval and the HCE alone.

  eval_models.py NAME [NAME...] [--device cpu|dml]
Writes datasets/nnue2/probe/breakdown.json (merged with earlier runs).
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe  # noqa: E402


SEEN = probe.OUT / "holdout_seen.npz"


def breakdown(e, s, k=1600.0):
    y = 1 / (1 + np.exp(-s / k))
    p = 1 / (1 + np.exp(-e / k))
    out = {}
    subsets = [("all", np.ones(len(s), bool)), ("nonmate", np.abs(s) < 8000), ("mate", np.abs(s) >= 20000)]
    if SEEN.exists():  # novelty.py: holdout positions in no training set (up to symmetry)
        z = np.load(SEEN)
        subsets.append(("novel", ~(z["in_d14_sym"] | z["in_d8_sym"])))
    for name, m in subsets:
        d = dict(frac=float(m.mean()), loss=float(((p[m] - y[m]) ** 2).mean()), std=float(e[m].std()),
                 mean_abs=float(np.abs(e[m]).mean()))
        if name in ("all", "nonmate"):
            d["corr_search"] = float(np.corrcoef(e[m], s[m])[0, 1])
        elif name == "mate":  # sign agreement with the mate
            d["sign_agree"] = float((np.sign(e[m]) == np.sign(s[m])).mean())
        out[name] = d
    # what stretching the output (to the shipped eval's spread) costs in loss
    out["loss_at_scale"] = {str(a): float(((1 / (1 + np.exp(-a * e / k)) - y) ** 2).mean()) for a in (0.8, 1.0, 1.15, 1.3, 1.6)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    probe.below_normal_priority()
    torch.set_num_threads(1)
    x, search, static, hce, hold, _ = probe.load_data()
    va = np.flatnonzero(hold)
    s = search[va].astype(np.float64)
    path = probe.OUT / "breakdown.json"
    res = json.loads(path.read_text()) if path.exists() else {}
    res["shipped"] = breakdown(static[va].astype(np.float64), s)
    res["hce_only"] = breakdown(hce[va].astype(np.float64), s)
    if args.device == "dml":
        import torch_directml
        dev = torch_directml.device()
    else:
        dev = torch.device("cpu")
    fx = probe.Feats(dev)
    vx = torch.from_numpy(x[va])
    for name in args.names:
        meta = json.loads((probe.OUT / f"{name}.json").read_text())
        a = meta["args"]
        if a["arm"] == "r10":
            model = probe.R10(fx, a["A"], a["L1"])
        else:
            model = probe.Pat(fx, a["width"], a["L1"], active=a["arm"] == "patact")
        model.load_state_dict(torch.load(probe.OUT / f"{name}.pt"))
        model = model.to(dev).eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(vx), 65536):
                out.append(model(*fx.split(vx[i:i + 65536].to(dev))).cpu())
        e = torch.cat(out).numpy().astype(np.float64)
        res[name] = breakdown(e, s)
        res[name]["corr_static"] = float(np.corrcoef(e, static[va])[0, 1])
        probe.log(f"{name}: {json.dumps(res[name])}")
    path.write_text(json.dumps(res, indent=1))
    for n, r in res.items():
        print(f"{n:16s} loss all {r['all']['loss']:.6f} nonmate {r['nonmate']['loss']:.6f} mate {r['mate']['loss']:.6f}"
              f" | std all {r['all']['std']:.0f} nonmate {r['nonmate']['std']:.0f} mate {r['mate']['std']:.0f}"
              f" | corr nonmate {r['nonmate']['corr_search']:.3f} | mate |E| {r['mate']['mean_abs']:.0f}"
              f" sign {r['mate']['sign_agree']:.3f}" + (f" | novel {r['novel']['loss']:.6f}" if "novel" in r else ""))


if __name__ == "__main__":
    main()
