#!/usr/bin/env python3
"""Low-precision ("natively quantized") training for the pattern-generator NNUE.

The idea comes from Thinking Machines' end-to-end 4-bit training (NVFP4 forward, high-precision backward,
block scales, 4/6 scale selection, a high-precision tail): train with the numerics the deployed net will use,
so nothing has to be calibrated or rounded after training. Here the forward pass is quantized exactly as a
candidate runtime or payload would store the net, the backward pass is float32 through a straight-through
estimator (STE), and the master weights stay float32.

Two places in CrossFish could use a lower-precision net:

  payload  the CodinGame source carries the generator (encoder, projections, decided / constraint rows, head)
           at 12-14 bits per weight with per-row bf16 scales, GPTQ-rounded after training (W1: 42,682 of the
           100,000 characters). A natively 4-bit generator would need about a third of that.
  runtime  the engine's tables are int16 (25.6 MB) and the head's first layer is int16 x int16 madd. int8
           tables halve the cache footprint, and u8 x i8 (vpmaddubsw) doubles the first layer's throughput,
           but both lose precision that post-training rounding cannot recover (FINDINGS A20).

Quantizers (all symmetric, fake-quantized with an STE; --q chooses which parts):

  nvfp4    E2M1 values {0, .5, 1, 1.5, 2, 3, 4, 6} in blocks of 16 along the input dimension, one E4M3 scale
           per block and one fp32 scale per tensor; with 4/6 each block also tries the scale that maps its
           max to 4 and keeps the lower-error choice. 4.5 bits per weight.
  intK     K-bit integers with one bf16 scale per output row (the payload's present layout at K = 12-14).
  int8p2   int8 with a power-of-two scale (per tensor or per output row): what an integer kernel can use.

Parts: gen (encoder, projections, forced projection, decided and constraint rows: the payload's generator),
head (the dense layers), tab (the baked accumulator rows, int8 per lane with a power-of-two scale), act (the
accumulator activation as u8 0..127 for vpmaddubsw), l1 (the head's first layer as int8, power-of-two scale).

Commands:
  check                         float forward == gen_nnue.Gen.fast on real rows
  eval  --init NET [--q SPEC]   held-out loss of a checkpoint under a quantization spec (post-training)
  train --init NET --q SPEC ... fine-tune with the quantized forward (QAT); --q none is the float control
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "tools" / "experiments" / "nnue2"))
sys.path.insert(0, str(ROOT / "tools"))
import gen_nnue  # noqa: E402
import probe  # noqa: E402

K = 1600.0
F_RESULT, F_SEARCH = 1, 2
# D4 maps on compact rows (r13_common.COLPERM / CONMAP)
G81, G9, CONMAP = probe.G81, probe.G9, probe.CONMAP
COLPERM = np.concatenate([G81, 81 + G9, np.full((8, 1), 90)], 1)
FP4_GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


# ---------------------------------------------------------------- quantizers (forward value, STE gradient)
def ste(x, q):
    return x + (q - x).detach()


def round_fp4(x):
    """Round |x| <= 6 to the nearest E2M1 value (ties to the even code, as the hardware converts)."""
    a = x.abs().clamp(max=6.0)
    # nearest of the 8 magnitudes; boundaries are the midpoints
    g = FP4_GRID.to(x.device)
    idx = torch.bucketize(a, (g[1:] + g[:-1]) / 2)
    return torch.sign(x) * g[idx]


def e4m3(x):
    return x.to(torch.float8_e4m3fn).to(torch.float32)


def q_nvfp4(w, block=16, four_six=True):
    """NVFP4 fake quantization of a 2-D (out, in) weight, blocks of `block` along `in` (padded)."""
    o, i = w.shape
    pad = (-i) % block
    x = F.pad(w, (0, pad)).reshape(o, -1, block)
    amax = x.abs().amax(-1, keepdim=True)
    t_scale = (w.abs().max() / (6.0 * 448.0)).clamp(min=1e-12)  # tensor scale: block scales fit E4M3
    best = None
    for top in ((6.0, 4.0) if four_six else (6.0,)):
        s = e4m3((amax / top / t_scale).clamp(min=2 ** -9)) * t_scale
        q = round_fp4(x / s) * s
        err = (q - x).pow(2).sum(-1, keepdim=True)
        if best is None:
            best, best_err = q, err
        else:
            m = err < best_err
            best = torch.where(m, q, best)
            best_err = torch.where(m, err, best_err)
    return best.reshape(o, -1)[:, :i]


def bf16(x):
    return x.to(torch.bfloat16).to(torch.float32)


def q_int_row(w, bits):
    """K-bit symmetric integers, one bf16 scale per output row (max maps to 2^(K-1) - 1)."""
    qmax = 2 ** (bits - 1) - 1
    s = bf16((w.abs().amax(1, keepdim=True) / qmax).clamp(min=1e-12))
    return torch.clamp(torch.round(w / s), -qmax, qmax) * s


def pow2_scale(amax, qmax):
    """Largest power-of-two multiplier m with amax * m <= qmax."""
    return torch.exp2(torch.floor(torch.log2(qmax / amax.clamp(min=1e-12))))


def q_int8_p2(w, per_row=True, qmax=127):
    amax = w.abs().amax(1, keepdim=True) if per_row else w.abs().max()
    m = pow2_scale(amax, qmax)
    return torch.clamp(torch.round(w * m), -qmax - 1, qmax) / m


# ---------------------------------------------------------------- quantization spec
PARTS = ("gen", "head", "tab", "act", "l1")


def parse_q(spec):
    """'none' or comma list of part=kind, e.g. gen=nvfp4,head=int14 or tab=int8,act=u7,l1=int8p2."""
    out = {}
    if not spec or spec == "none":
        return out
    for tok in spec.split(","):
        k, v = tok.split("=")
        assert k in PARTS, k
        out[k] = v
    return out


def quant_matrix(w, kind):
    """Quantize a 2-D (out, in) weight with a named scheme."""
    if kind == "nvfp4":
        return q_nvfp4(w, 16, True)
    if kind == "nvfp4n":  # without 4/6
        return q_nvfp4(w, 16, False)
    if kind.startswith("int8p2"):
        return q_int8_p2(w, per_row=not kind.endswith("t"))
    if kind.startswith("int"):
        return q_int_row(w, int(kind[3:]))
    raise ValueError(kind)


def bits_per_value(kind, n_in):
    """Raw payload bits per weight for a scheme (before entropy coding), scales included."""
    if kind.startswith("nvfp4"):
        return 4.0 + 8.0 / 16
    if kind.startswith("int8p2"):
        return 8.0 + 5.0 / n_in
    if kind.startswith("int"):
        return int(kind[3:]) + 16.0 / n_in
    raise ValueError(kind)


# ---------------------------------------------------------------- the quantized forward
class QNet:
    """A gen_nnue.Gen and a quantization spec; forward() is the net the spec deploys."""

    def __init__(self, model, q):
        self.m, self.q = model, q
        self.fx = model.fx
        self.A = model.A

    def _w(self, w, part, rows_first=True):
        kind = self.q.get(part)
        if not kind:
            return w
        return ste(w, quant_matrix(w, kind) if rows_first else quant_matrix(w.t(), kind).t())

    def gen_tables(self):
        m = self.m
        h = m.onehot
        lins = [l for l in m.enc if isinstance(l, torch.nn.Linear)]
        for i, lin in enumerate(lins):
            h = F.linear(h, self._w(lin.weight, "gen"), lin.bias)
            if i < len(lins) - 1:
                h = F.relu(h)
        E, W = m.E, m.A + 1
        # proj: rows = (location, lane), inputs = the E embedding dims
        pw = m.proj_w.permute(0, 2, 1).reshape(9 * W, E)
        pw = self._w(pw, "gen").reshape(9, W, E).permute(0, 2, 1)
        T = torch.einsum("pk,lkj->lpj", h, pw) + m.proj_b[:, None, :]  # (9, 3^9, W)
        fw = self._w(m.fwd_w.t(), "gen").t()
        Fw = h @ fw + m.fwd_b  # (3^9, W)
        dec = self._w(m.dec.t(), "gen").t()  # one row per lane, as the payload stores it
        con = self._w(m.con.t(), "gen").t()
        bias = m.bias
        if "tab" in self.q:  # baked int8 rows, one power-of-two step per lane over every table
            allr = torch.cat([T.reshape(-1, W), Fw, dec, con, bias[None]], 0)
            amax = allr.detach().abs().amax(0)
            qmax = {"int8": 127, "int7": 63, "int6": 31}[self.q["tab"]]
            mult = pow2_scale(amax, qmax)

            def qt(x):
                return ste(x, torch.clamp(torch.round(x * mult), -qmax - 1, qmax) / mult)
            T, Fw, dec, con, bias = qt(T), qt(Fw), qt(dec), qt(con), qt(bias)
        return T, Fw, dec, con, bias

    def accumulators(self, xb, tabs):
        T, Fw, dec, con, bias = tabs
        fx = self.fx
        b = xb.shape[0]
        cells, sup, cn = xb[:, :81].long(), xb[:, 81:90].long(), xb[:, 90].long()
        pat = (cells.reshape(b, 9, 9) * fx.pow3).sum(2)
        loc = torch.arange(9, device=xb.device)
        outs = []
        for side in (0, 1):
            p = pat if side == 0 else fx.pat_swap[pat]
            s = sup if side == 0 else fx.sup_swap[sup]
            live = (s == 0)
            rows = T[loc[None, :], p]  # (b, 9, W)
            drow = dec[(3 * loc[None, :] + (s - 1).clamp(min=0))]
            acc = torch.where(live[..., None], rows, drow).sum(1) + bias
            acc = acc + con[cn + 10 * side]
            forced = cn < 9
            pf = torch.gather(p, 1, cn.clamp(max=8)[:, None]).squeeze(1)
            acc = acc + Fw[pf] * forced[:, None].float()
            outs.append(acc)
        return outs

    def head(self, us, them):
        m, A = self.m, self.A
        x = torch.clamp(torch.cat([us[:, :A], them[:, :A]], 1), 0, 1)
        if "act" in self.q:  # u8 0..qmax activations (vpmaddubsw needs <= 127)
            qa = {"u7": 127, "u8": 255, "u6": 63}[self.q["act"]]
            x = ste(x, torch.round(x * qa) / qa)
        for i, lin in enumerate(m.dense):
            w = lin.weight
            if i == 0 and "l1" in self.q:
                w = ste(w, quant_matrix(w, self.q["l1"]))
            elif "head" in self.q:
                w = ste(w, quant_matrix(w, self.q["head"]))
            x = F.linear(x, w, lin.bias)
            if i < len(m.dense) - 1:
                x = torch.clamp(x, 0, 1)
        psqt = 0.5 * (us[:, A] - them[:, A])
        return (x.squeeze(1) + psqt) * 1000.0, psqt * 1000.0

    def forward(self, xb, tabs=None):
        tabs = self.gen_tables() if tabs is None else tabs
        return self.head(*self.accumulators(xb, tabs))

    def payload_bits(self):
        """Raw payload bits under this spec (generator + head; biases and PSQT-free details at 14 bits)."""
        m = self.m
        E, W = m.E, m.A + 1
        g, hd = self.q.get("gen"), self.q.get("head")
        tot = 0.0
        mats = [(l.weight.shape[0], l.weight.shape[1], "gen") for l in m.enc if isinstance(l, torch.nn.Linear)]
        mats += [(9 * W, E, "gen"), (W, E, "gen"), (W, 27, "gen"), (W, 20, "gen")]
        mats += [(l.weight.shape[0], l.weight.shape[1], "head") for l in m.dense]
        for o, i, part in mats:
            kind = g if part == "gen" else hd
            tot += o * i * (bits_per_value(kind, i) if kind else 14.0 + 16.0 / i)
        nb = sum(int(p.numel()) for n, p in m.named_parameters() if n.endswith("bias") or n in ("proj_b", "fwd_b"))
        return tot + 14.0 * nb


# ---------------------------------------------------------------- data
COLS = {"x": (np.uint8, 91), "lab": (np.int32, 1), "res": (np.int8, 1), "flags": (np.uint8, 1)}


def read_part(prefix):
    out = {}
    for c, (dt, w) in COLS.items():
        ext = np.dtype(dt).str[1:]
        a = np.fromfile(f"{prefix}.{c}.{ext}", dtype=dt)
        out[c] = a.reshape(-1, w) if w > 1 else a
    return out


def wdl_keep(lab, res, flags, d=1136.6107739467577, s0=615.4195192408732, seed=14):
    """The round-14 WDL contradiction filter (one fixed draw; W1's sp14 fit)."""
    s = lab.astype(np.float64)
    pw = 1 / (1 + np.exp(-(s - d) / s0))
    pl = 1 / (1 + np.exp(-(-s - d) / s0))
    pd = np.clip(1 - pw - pl, 0, 1)
    p = np.where(res > 0, pw, np.where(res < 0, pl, pd))
    both = ((flags & F_SEARCH) != 0) & ((flags & F_RESULT) != 0)
    rng = np.random.default_rng(seed)
    return ~(both & (rng.random(len(lab)) >= p))


def target(lab, flags):
    return (1 / (1 + np.exp(-lab.astype(np.float64) / K))).astype(np.float32)


def load_train(data_dir):
    d = Path(data_dir)
    e2 = read_part(d / "e2b" / "train")
    sp = read_part(d / "sp14s")
    me = (e2["flags"] & F_SEARCH) != 0
    ms = ((sp["flags"] & F_SEARCH) != 0) & wdl_keep(sp["lab"], sp["res"], sp["flags"])
    srcs = {"e2b": (e2["x"][me], target(e2["lab"][me], None)), "sp14": (sp["x"][ms], target(sp["lab"][ms], None))}
    return srcs


def load_holdouts(data_dir, n):
    d = Path(data_dir)
    H = {}
    for name, pre in (("V2", d / "e2b" / "hold"), ("SPH13", d / "sp13" / "hold")):
        h = read_part(pre)
        m = (h["flags"] & F_SEARCH) != 0
        idx = np.flatnonzero(m)
        if n and len(idx) > n:
            idx = np.sort(np.random.default_rng(7).choice(idx, n, replace=False))
        H[name] = dict(x=h["x"][idx], lab=h["lab"][idx], res=h["res"][idx], flags=h["flags"][idx])
    return H


def augment(x, rng):
    g = rng.integers(0, 8, len(x))
    for k in range(1, 8):
        idx = np.flatnonzero(g == k)
        if len(idx):
            xi = x[idx][:, COLPERM[k]]
            xi[:, 90] = CONMAP[k][x[idx, 90]]
            x[idx] = xi
    return x


# ---------------------------------------------------------------- metrics
def sig(v):
    return 1 / (1 + np.exp(-np.asarray(v, np.float64) / K))


def metrics(e, h):
    y = sig(h["lab"])
    p = sig(e)
    mse = float(((p - y) ** 2).mean())
    # fitted eval scale (the search's margins are re-scaled per net, so a pure scale change is free)
    best = (mse, 1.0)
    lo, hi = 0.8, 1.25
    for _ in range(30):
        a, b = lo + (hi - lo) * 0.382, lo + (hi - lo) * 0.618
        fa = float(((sig(a * e) - y) ** 2).mean())
        fb = float(((sig(b * e) - y) ** 2).mean())
        if fa < fb:
            hi = b
        else:
            lo = a
        best = min(best, (fa, a), (fb, b))
    return dict(mse=mse, mse_fit=best[0], scale=best[1])


@torch.no_grad()
def predict(qn, x, bs=65536):
    tabs = qn.gen_tables()
    out = []
    for i in range(0, len(x), bs):
        xb = torch.from_numpy(np.ascontiguousarray(x[i:i + bs]))
        out.append(qn.forward(xb, tabs)[0].numpy())
    return np.concatenate(out).astype(np.float64)


def evaluate(qn, H, ref=None):
    res = {}
    for name, h in H.items():
        e = predict(qn, h["x"])
        m = metrics(e, h)
        if ref is not None:
            m["d_ref"] = float(np.abs(e - ref[name]).mean())
        res[name] = m
    return res


# ---------------------------------------------------------------- model io
def load_model(path):
    fx = probe.Feats("cpu")
    meta_p = Path(str(path).replace("_rs.pt", ".pt")).with_suffix(".json")
    a = json.loads(meta_p.read_text())["args"] if meta_p.exists() else dict(A=64, dense=[16, 32], enc=[128, 128, 32])
    m = gen_nnue.build(fx, a)
    m.load_state_dict(torch.load(path, map_location="cpu"))
    return m, a


# ---------------------------------------------------------------- commands
def cmd_check(args):
    m, _ = load_model(args.init)
    H = load_holdouts(args.data, 20000)
    x = torch.from_numpy(H["V2"]["x"][:20000])
    with torch.no_grad():
        ref = m.fast(x)
        got = QNet(m, {}).forward(x)[0]
    d = (ref - got).abs().max().item()
    print(f"float forward vs Gen.fast: max |diff| {d:.3e} eval units over {len(x)} rows")
    assert d < 0.1, d  # float32 summation order only
    for spec in ("gen=nvfp4", "tab=int8,act=u7,l1=int8p2"):
        q = parse_q(spec)
        with torch.no_grad():
            e = QNet(m, q).forward(x)[0]
        print(f"{spec:32s} mean |d eval| {float((e - ref).abs().mean()):.2f}")


def cmd_eval(args):
    m, _ = load_model(args.init)
    H = load_holdouts(args.data, args.hold_n)
    ref = {n: predict(QNet(m, {}), h["x"]) for n, h in H.items()}
    rows = []
    for spec in args.q:
        qn = QNet(m, parse_q(spec))
        r = evaluate(qn, H, ref)
        bits = qn.payload_bits()
        rows.append(dict(spec=spec, payload_chars=int(math.ceil(bits / 15)), **{f"{n}_{k}": v for n, mm in r.items()
                                                                                 for k, v in mm.items()}))
        print(json.dumps(rows[-1]), flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=1))


def cmd_train(args):
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    torch.set_num_threads(args.threads)
    m, a = load_model(args.init)
    q = parse_q(args.q)
    qn = QNet(m, q)
    srcs = load_train(args.data)
    H = load_holdouts(args.data, args.hold_n)
    names = list(srcs)
    w = np.array([0.4644, 0.5356])
    first = [p for n, p in m.named_parameters() if not n.startswith("dense.")]
    dense = [p for n, p in m.named_parameters() if n.startswith("dense.")]
    opt = torch.optim.AdamW([dict(params=first, weight_decay=0.0), dict(params=dense, weight_decay=1e-5)],
                            lr=args.lr, betas=(0.9, 0.999))
    total = args.steps
    warm = max(1, int(0.01 * total))

    def lr_at(s):
        if s < warm:
            return args.lr * (s + 1) / warm
        t = (s - warm) / max(1, total - warm)
        return args.lr_floor + 0.5 * (args.lr - args.lr_floor) * (1 + math.cos(math.pi * t))

    log = []
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / f"{args.name}.log", "a")

    def say(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    say(f"train {args.name}: init {args.init} q {args.q} steps {total} batch {args.batch} lr {args.lr} seed {args.seed}")
    r0 = evaluate(qn, H)
    say(f"step 0: {json.dumps(r0)}")
    log.append(dict(step=0, **r0))
    t0 = time.time()
    run = 0.0
    for s in range(total):
        for g in opt.param_groups:
            g["lr"] = lr_at(s)
        nb = rng.multinomial(args.batch, w)
        xs, ys = [], []
        for nm, k in zip(names, nb):
            X, Y = srcs[nm]
            idx = rng.integers(0, len(X), k)
            xs.append(X[idx])
            ys.append(Y[idx])
        xb = augment(np.concatenate(xs), rng)
        y = torch.from_numpy(np.concatenate(ys))
        e, ps = qn.forward(torch.from_numpy(xb))
        dlt = torch.sigmoid(e / K) - y
        l_main = (dlt.abs().pow(2.5) * (1.0 + 0.2 * (dlt > 0).float())).mean()
        l_ps = (torch.sigmoid(ps / K) - y).pow(2).mean()
        loss = l_main + 0.06878 * l_ps
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        run += float(l_main)
        if (s + 1) % args.log_every == 0:
            el = time.time() - t0
            say(f"step {s + 1}/{total} loss {run / args.log_every:.6f} lr {lr_at(s):.2e} "
                f"{(s + 1) * args.batch / el:,.0f} rows/s")
            run = 0.0
        if (s + 1) % args.eval_every == 0 or s + 1 == total:
            r = evaluate(qn, H)
            say(f"step {s + 1}: {json.dumps(r)}")
            log.append(dict(step=s + 1, **r))
    torch.save(m.state_dict(), out / f"{args.name}.pt")
    meta = dict(name=args.name, args=a, lowprec=dict(vars(args)), q=q, hist=log,
                payload_chars=int(math.ceil(qn.payload_bits() / 15)), minutes=(time.time() - t0) / 60)
    (out / f"{args.name}.json").write_text(json.dumps(meta, indent=1, default=str))
    say(f"done {args.name} in {(time.time() - t0) / 60:.1f} min")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("check", "eval", "train"):
        p = sub.add_parser(name)
        p.add_argument("--init", required=True, help="a gen_nnue checkpoint (.pt with its .json beside it)")
        p.add_argument("--data", required=True, help="dir with e2b/{train,hold}.*, sp13/hold.*, sp14s.*")
        p.add_argument("--hold-n", type=int, default=300000)
    sub.choices["eval"].add_argument("--q", nargs="+", default=["none"])
    sub.choices["eval"].add_argument("--out")
    t = sub.choices["train"]
    t.add_argument("--name", required=True)
    t.add_argument("--q", default="none")
    t.add_argument("--steps", type=int, default=2000)
    t.add_argument("--batch", type=int, default=16384)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--lr-floor", type=float, default=1e-5)
    t.add_argument("--seed", type=int, default=1)
    t.add_argument("--threads", type=int, default=4)
    t.add_argument("--log-every", type=int, default=50)
    t.add_argument("--eval-every", type=int, default=500)
    t.add_argument("--out-dir", default="lowprec_out")
    args = ap.parse_args()
    {"check": cmd_check, "eval": cmd_eval, "train": cmd_train}[args.cmd](args)


if __name__ == "__main__":
    main()
