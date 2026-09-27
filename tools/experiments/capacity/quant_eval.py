#!/usr/bin/env python3
"""Held-out loss of capacity-probe MiniNets after the shipped 256-centroid packing.

For each checkpoint the full 19,683-row pattern table is re-clustered exactly
as tools/nnue_emit_mininet_header.py does (k-means in first-layer projection
space weighted by |w2|, empty board reserved as code 0, centroids = member means
in embedding space) and the holdout loss is measured before and after.

  ffull   datasets/eval2/train/F_full.cfm2 + .macro.pt (arm a's model)
  a2 b c d e   datasets/eval2/capacity/<arm>.pt

--export ARM also writes datasets/eval2/capacity/<ARM>_export.{cfm2,macro.pt,json,...}
through nnue_train_blend.export: a plain D16/H8 checkpoint the existing
emitter accepts.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import argparse  # noqa: E402
import json  # noqa: E402
import struct  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import capacity_probe as cp  # noqa: E402

ntb = cp.ntb
sys.path.insert(0, str(cp.ROOT / "tools"))
from nnue_emit_mininet_header import projected_centroids  # noqa: E402

CFG = cp.WIDE


def load_ffull():
    blob = (cp.ROOT / "datasets/eval2/train/F_full.cfm2").read_bytes()
    d, h = struct.unpack("<ii", blob[4:12])
    a = np.frombuffer(blob[12:], dtype="<f4")
    off = 0

    def take(n, shape):
        nonlocal off
        x = a[off:off + n].reshape(shape).copy()
        off += n
        return x

    table = take(ntb.N_IDX * d, (ntb.N_IDX, d))
    mini = dict(codes=np.arange(ntb.N_IDX) % 256, cents=np.zeros((256, d), np.float32),
                super_e=take(4 * d, (4, d)), loc=take(9 * d, (9, d)), constr=take(10 * d, (10, d)),
                active=take(2 * d, (2, d)), w1=take(h * 10 * d, (h, 10 * d)), b1=take(h, (h,)), w2=take(h, (h,)),
                b2=float(take(1, (1,))[0]))
    m = torch.load(cp.ROOT / "datasets/eval2/train/F_full.macro.pt")["state_dict"]
    macro = dict(base=m["hidden.bias"].numpy(), constr=m["constraint.weight"].numpy(),
                 emb=m["embedding.weight"].numpy().reshape(9, 4, 16), out=m["out.weight"].numpy().reshape(16),
                 bias=0.0)
    model = ntb.Eval(mini, macro, "full")
    with torch.no_grad():
        model.emb.copy_(torch.from_numpy(table))
    return model


def load_arm(arm):
    base = arm.split("_")[0]  # a2_lr3e3 -> a2
    if base == "a":
        model = ntb.Eval(ntb.load_shipped_mini(), ntb.load_shipped_macro(), "full")
    else:
        model = cp.Wide(ntb.load_shipped_mini(), ntb.load_shipped_macro(), **CFG[base])
    sd = torch.load(cp.OUT / f"{arm}.pt")
    missing = model.load_state_dict({**{k: v for k, v in model.state_dict().items()}, **sd}, strict=True)
    return model


def full_first_layer(model):
    """Pattern table (N, D), first layer (H, 10, D) and output weights (H,) over all units."""
    d0 = model.emb.shape[1]
    table = model.emb.detach()
    w1 = model.w1.detach().reshape(-1, 10, d0)
    w2 = model.w2.detach()
    dx = getattr(model, "dx", 0)
    if dx:
        table = torch.cat([table, model.emb_x.detach()], dim=1)
        w1 = torch.cat([w1, model.w1_ox.detach().reshape(-1, 10, dx)], dim=2)
    if getattr(model, "hx", 0):
        wn = model.w1n.detach()
        n_o = wn[:, :10 * d0].reshape(-1, 10, d0)
        parts = [n_o]
        if dx:
            parts.append(wn[:, 10 * d0:].reshape(-1, 10, dx))
        w1 = torch.cat([w1, torch.cat(parts, dim=2)], dim=0)
        w2 = torch.cat([w2, model.w2n.detach()])
    return table.numpy(), w1.numpy(), w2.numpy()


def quantize(model):
    table, w1, w2 = full_first_layer(model)
    h, _, d = w1.shape
    t0 = time.time()
    cents, codes = projected_centroids(table.astype(np.float32), w1.reshape(h, 10 * d), w2, d, h, True)
    q = torch.from_numpy(cents[codes])
    d0 = model.emb.shape[1]
    with torch.no_grad():
        model.emb.copy_(q[:, :d0])
        if getattr(model, "dx", 0):
            model.emb_x.copy_(q[:, d0:])
    return time.time() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="+", help="ffull and/or arm names with a saved .pt")
    ap.add_argument("--device", default="dml")
    ap.add_argument("--export", default="", help="a dedup D16/H8 arm (a2, a2_lr3e3, ...) to write as plain CFM2")
    args = ap.parse_args()
    cp.below_normal_priority()
    torch.set_num_threads(2)
    rec, _, hold = ntb.load_all([str(p) for p in cp.DATA], None)
    rec = rec[hold]
    mini_idx, super_idx, constr, stm_sign = ntb.features(rec)
    k = 1600.0
    Y = torch.from_numpy((1 / (1 + np.exp(-rec["search"].astype(np.float32) / k))).astype(np.float32))
    if args.device == "dml":
        import torch_directml
        dev = torch_directml.device()
    else:
        dev = torch.device(args.device)
    T = lambda a, dt=torch.long: torch.from_numpy(np.ascontiguousarray(a)).to(dt)  # noqa: E731
    X = [T(mini_idx), T(super_idx), T(constr), T(rec["hce"], torch.float32), T(stm_sign, torch.float32)]

    def evaluate(model):
        model = model.to(dev).eval()
        tot = 0.0
        with torch.no_grad():
            for i in range(0, len(Y), 65536):
                e = model(*(x[i:i + 65536].to(dev) for x in X))[0].cpu()
                tot += (torch.sigmoid(e / k) - Y[i:i + 65536]).pow(2).sum().item()
        model.cpu()
        return tot / len(Y)

    results = {}
    for name in args.models:
        model = load_ffull() if name == "ffull" else load_arm(name)
        full = evaluate(model)
        secs = quantize(model)
        packed = evaluate(model)
        results[name] = dict(float=full, packed=packed, packing_cost_pct=100 * (packed / full - 1),
                             vs_shipped_pct=100 * (packed / 0.046817 - 1), vs_ffull_float_pct=100 * (packed / 0.043527 - 1))
        print(f"[{time.strftime('%H:%M:%S')}] {name}: float {full:.6f}  packed-256 {packed:.6f}"
              f" ({results[name]['packing_cost_pct']:+.2f}%; {results[name]['vs_shipped_pct']:+.2f}% vs shipped;"
              f" kmeans {secs:.0f}s)", flush=True)
    (cp.OUT / "quant_eval.json").write_text(json.dumps(results, indent=1))

    if args.export:
        a2 = load_arm(args.export)
        _, w1, w2 = full_first_layer(a2)
        m = ntb.Eval(ntb.load_shipped_mini(), ntb.load_shipped_macro(), "full")
        with torch.no_grad():
            m.emb.copy_(a2.emb)
            for n in ("super_e", "loc", "constr", "active", "b2", "m_base", "m_constr", "m_emb", "m_out"):
                getattr(m, n).copy_(getattr(a2, n))
            m.w1.copy_(torch.from_numpy(w1.reshape(8, 160)))
            m.b1.copy_(torch.cat([a2.b1.detach(), a2.b1n.detach()]))
            m.w2.copy_(torch.from_numpy(w2))
        check = evaluate(m)
        print(f"{args.export} as a plain D16/H8 ntb.Eval: {check:.6f}", flush=True)
        ntb.export(m.cpu(), cp.OUT / f"{args.export}_export",
                   dict(k=k, source=f"capacity arm {args.export} (dedup D16/H8)", holdout_float=check))


if __name__ == "__main__":
    main()
