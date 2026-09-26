#!/usr/bin/env python3
"""Offline capacity probe: how much held-out loss does more eval capacity buy?

Same data, split, target and K as F_full (datasets/eval2/cf_play_d14.cfdg +
uttt_sp_d14.cfdg; lam 1; K 1600; win-probability MSE; 10% game holdout from
nnue_train_blend.game_holdout). Everything is a residual on top of the fixed
shipped HCE. Data loading, features, the shipped MiniNet/macro decode and the
Eval module come from tools/nnue_train_blend.py; the training loop mirrors its
main() (Adam, OneCycle pct_start 0.05, per-epoch holdout eval, best epoch).

Arms (--arm):
  a     F_full reproduction: nnue_train_blend.Eval, mode full, D16/H8.
  a2    D16/H8 with the duplicated hidden units merged (the shipped H8 is four
        copies each of two units) and the freed units re-initialised
        (w2 = 0, so epoch 0 == shipped up to the merge rounding).
  b     D32/H8: shipped weights, 16 new embedding dims (small random), zero
        cross weights into the old units (epoch 0 == shipped).
  c     D16/H16: 8 new hidden units, random first layer, zero output weight.
  d     D32/H32: both expansions.
  e     a2 plus a second hidden layer: out += v . relu(A r + c) over the 8
        hidden units (8 -> 16 -> 1, v = 0 so epoch 0 == a2's epoch 0).
  f     9-miniboard-token transformer residual on the frozen shipped eval.
  g     pattern NNUE residual (width 128 accumulator, 128 -> 16 -> 1) on the
        frozen shipped eval.
  fj/gj f/g on top of a jointly trained full-mode MiniNet + macro (F_full-style)
        instead of the frozen shipped eval.

New (from-scratch) parameters get --new-lr; the shipped parameters keep --lr,
as in F_full. Outputs: datasets/eval2/capacity/<arm>.json (+ .log from the
driver).
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import argparse  # noqa: E402
import ctypes  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))
import eval_data  # noqa: E402
import nnue_train_blend as ntb  # noqa: E402

DATA = [ROOT / "datasets/eval2/cf_play_d14.cfdg", ROOT / "datasets/eval2/uttt_sp_d14.cfdg"]
OUT = ROOT / "datasets/eval2/capacity"
N_IDX = ntb.N_IDX
WIDE = dict(a2=dict(D=16, H=8, dedup=True), a2s=dict(D=16, H=8, dedup=True, side=True), b=dict(D=32, H=8),
            c=dict(D=16, H=16), d=dict(D=32, H=32), e=dict(D=16, H=8, dedup=True, layer2=16))


def below_normal_priority():
    if os.name == "nt":
        k32 = ctypes.windll.kernel32
        k32.SetPriorityClass(k32.GetCurrentProcess(), 0x4000)  # BELOW_NORMAL_PRIORITY_CLASS


def active_of(constr, b, device):
    loc = torch.arange(9, device=device).expand(b, 9)
    return loc, ((loc == constr[:, None]) & (constr[:, None] < 9)).long()


def dup_groups(w1, thresh=0.98):
    """Greedy groups of hidden units whose first-layer rows are near-identical."""
    c = np.corrcoef(w1)
    left, groups = list(range(len(w1))), []
    while left:
        g = [j for j in left if c[left[0], j] > thresh]
        groups.append(g)
        left = [j for j in left if j not in g]
    return groups


class Wide(ntb.Eval):
    """ntb.Eval (mode full) with extra embedding dims, extra hidden units and an
    optional second layer. The shipped tensors stay as they are (old lr); every
    addition lives in its own tensor (new lr) and starts with zero effect."""

    def __init__(self, mini, macro, D=16, H=8, dedup=False, layer2=0, side=False, seed=1234):
        mini = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in mini.items()}
        g = torch.Generator().manual_seed(seed)
        rnd = lambda *s: torch.randn(*s, generator=g)  # noqa: E731
        d0 = mini["cents"].shape[1]
        w1, b1, w2 = mini["w1"], mini["b1"], mini["w2"]
        keep, groups = list(range(len(w1))), None
        if dedup:
            groups = dup_groups(w1)
            keep = [grp[0] for grp in groups]
            w2 = np.array([w2[grp].sum() for grp in groups], dtype=np.float32)
            w1, b1 = w1[keep], b1[keep]
        h_old = len(keep)
        mini["w1"], mini["b1"], mini["w2"] = w1.astype(np.float32), b1.astype(np.float32), w2.astype(np.float32)
        super().__init__(mini, macro, "full")
        self.d0, self.dx, self.h_old, self.hx = d0, D - d0, h_old, H - h_old
        self.layer2 = layer2
        self.groups = groups
        self.new_names = []

        def P(name, t):
            setattr(self, name, nn.Parameter(t.float()))
            self.new_names.append(name)

        if self.dx:
            dx = self.dx
            P("emb_x", 0.1 * rnd(N_IDX, dx))
            P("super_x", 0.1 * rnd(4, dx))
            P("loc_x", 0.1 * rnd(9, dx))
            P("constr_x", 0.1 * rnd(10, dx))
            P("active_x", 0.1 * rnd(2, dx))
            P("w1_ox", torch.zeros(h_old, 10 * dx))  # old units <- new dims
        if self.hx:
            n_in = 10 * D
            P("w1n", rnd(self.hx, n_in) / math.sqrt(n_in) * 12.0)  # h std ~ 10, like the old units' scale
            P("b1n", torch.full((self.hx,), 0.5))
            P("w2n", torch.zeros(self.hx))
        if layer2:
            P("A", rnd(layer2, H) / math.sqrt(H))
            P("c", torch.full((layer2,), 0.1))
            P("v", torch.zeros(layer2))
        self.side = side
        if side:  # side-to-move (player 1 or 2) embedding into the hidden pre-activations, zero-init
            P("side_o", torch.zeros(2, h_old))
            if self.hx:
                P("side_n", torch.zeros(2, self.hx))

    def forward(self, mini_idx, super_idx, constr, hce, stm_sign, diffs=None):
        if not self.side:
            return super().forward(mini_idx, super_idx, constr, hce, stm_sign, diffs)
        dev = mini_idx.device
        ez = torch.zeros((1, 9), dtype=torch.long, device=dev)
        cz = torch.full((1,), 9, dtype=torch.long, device=dev)
        side = (stm_sign > 0).long()
        first = torch.ones(1, dtype=torch.long, device=dev)  # player 1 moves first on the empty board
        mini = self.mini(mini_idx, super_idx, constr, side) - self.mini(ez, ez, cz, first)
        macro = torch.clamp(self.macro_raw(super_idx, constr) - self.macro_raw(ez, cz), -ntb.MACRO_CLIP,
                            ntb.MACRO_CLIP)
        return hce + mini + macro, mini, macro

    def mini(self, mini_idx, super_idx, constr, side=None):
        b = mini_idx.shape[0]
        loc, active = active_of(constr, b, mini_idx.device)
        v = self.emb[mini_idx] + self.super_e[super_idx] + self.loc[loc] + self.active[active]
        xo = torch.cat([v.reshape(b, 9 * self.d0), self.constr[constr]], dim=1)
        h = xo @ self.w1.T + self.b1
        if side is not None:
            h = h + self.side_o[side]
        xs = [xo]
        if self.dx:
            vx = self.emb_x[mini_idx] + self.super_x[super_idx] + self.loc_x[loc] + self.active_x[active]
            xn = torch.cat([vx.reshape(b, 9 * self.dx), self.constr_x[constr]], dim=1)
            h = h + xn @ self.w1_ox.T
            xs.append(xn)
        r = torch.relu(h)
        out = (r * self.w2).sum(dim=1) + self.b2  # avoids addmv (CPU fallback on DML)
        if self.hx:
            hn = torch.cat(xs, dim=1) @ self.w1n.T + self.b1n
            rn = torch.relu(hn + self.side_n[side] if side is not None else hn)
            out = out + (rn * self.w2n).sum(dim=1)
            r = torch.cat([r, rn], dim=1)
        if self.layer2:
            out = out + (torch.relu(r @ self.A.T + self.c) * self.v).sum(dim=1)
        return out


class TokenTransformer(nn.Module):
    """9 miniboard tokens + 1 global token, 1 pre-LN layer, output from the global token."""

    def __init__(self, d=32, heads=2, ffn=64, out_scale=256.0, side=True):
        super().__init__()
        self.d, self.heads, self.out_scale, self.use_side = d, heads, out_scale, side
        self.pat = nn.Parameter(0.5 * torch.randn(N_IDX, d))
        self.loc = nn.Parameter(0.5 * torch.randn(9, d))
        self.sup = nn.Parameter(0.5 * torch.randn(4, d))
        self.act = nn.Parameter(0.5 * torch.randn(2, d))
        self.g_tok = nn.Parameter(0.5 * torch.randn(d))
        self.g_con = nn.Parameter(0.5 * torch.randn(10, d))
        if side:
            self.g_side = nn.Parameter(0.5 * torch.randn(2, d))
        self.ln1, self.ln2, self.lnf = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ff1, self.ff2 = nn.Linear(d, ffn), nn.Linear(ffn, d)
        self.head = nn.Linear(d, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, mini_idx, super_idx, constr, stm_sign):
        b = mini_idx.shape[0]
        loc, active = active_of(constr, b, mini_idx.device)
        t = self.pat[mini_idx] + self.loc[loc] + self.sup[super_idx] + self.act[active]
        g = self.g_tok + self.g_con[constr]
        if self.use_side:
            g = g + self.g_side[(stm_sign > 0).long()]
        gtok = g[:, None, :]
        x = torch.cat([t, gtok], dim=1)  # (b, 10, d)
        dh = self.d // self.heads
        q, k, v = self.qkv(self.ln1(x)).reshape(b, 10, 3, self.heads, dh).permute(2, 0, 3, 1, 4)
        att = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(dh), dim=-1)
        o = (att @ v).transpose(1, 2).reshape(b, 10, self.d)
        x = x + self.proj(o)
        x = x + self.ff2(torch.relu(self.ff1(self.ln2(x))))
        z = self.lnf(x[:, 9])
        return self.out_scale * self.head(z).squeeze(-1)


class PatternNNUE(nn.Module):
    """acc = bias + sum_m FT[(m, pattern_m)] + FS[(m, super_m)] + FC[constraint];
    clipped ReLU; 128 -> 16 (clipped ReLU) -> 1, zero-init output."""

    def __init__(self, width=128, l2=16, out_scale=256.0):
        super().__init__()
        self.out_scale = out_scale
        self.ft = nn.Parameter(0.05 * torch.randn(9 * N_IDX, width))
        self.fs = nn.Parameter(0.05 * torch.randn(36, width))
        self.fc = nn.Parameter(0.05 * torch.randn(10, width))
        self.fb = nn.Parameter(torch.full((width,), 0.5))
        self.l1 = nn.Linear(width, l2)
        self.out = nn.Linear(l2, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, mini_idx, super_idx, constr, stm_sign):
        b = mini_idx.shape[0]
        loc = torch.arange(9, device=mini_idx.device).expand(b, 9)
        acc = self.fb + self.ft[loc * N_IDX + mini_idx].sum(dim=1) + self.fs[loc * 4 + super_idx].sum(dim=1) \
            + self.fc[constr]
        h = torch.clamp(self.l1(torch.clamp(acc, 0.0, 1.0)), 0.0, 1.0)
        return self.out_scale * self.out(h).squeeze(-1)


class Residual(nn.Module):
    """E = base + res. base is either the frozen shipped eval (precomputed per
    row, passed in as `base`) or a trainable ntb.Eval."""

    def __init__(self, res, base_model=None):
        super().__init__()
        self.res = res
        self.base_model = base_model

    def forward(self, mini_idx, super_idx, constr, hce, stm_sign, base=None):
        if self.base_model is not None:
            e = self.base_model(mini_idx, super_idx, constr, hce, stm_sign)[0]
        else:
            e = base
        return e + self.res(mini_idx, super_idx, constr, stm_sign)


def n_params(ps):
    return int(sum(p.numel() for p in ps))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True, choices=["a", "a2", "a2s", "b", "c", "d", "e", "f", "fns", "g", "fj", "gj"])
    ap.add_argument("--k", type=float, default=1600.0)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=16384)
    ap.add_argument("--lr", type=float, default=3e-4, help="lr of the shipped (F_full) parameters")
    ap.add_argument("--new-lr", type=float, default=3e-3, help="lr of new MiniNet parameters (arms a2-e)")
    ap.add_argument("--res-lr", type=float, default=1e-3, help="lr of residual models (arms f, g, fj, gj)")
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--device", default="dml")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--tag", default="")
    ap.add_argument("--check-fit-k", action="store_true")
    args = ap.parse_args()
    below_normal_priority()
    torch.manual_seed(args.seed)
    torch.set_num_threads(min(2, args.threads))
    rng = np.random.default_rng(args.seed)
    name = args.arm + (("_" + args.tag) if args.tag else "")
    OUT.mkdir(parents=True, exist_ok=True)
    t_all = time.time()

    rec, _, hold = ntb.load_all([str(p) for p in DATA], None)
    if args.max_rows and len(rec) > args.max_rows:
        idx = np.sort(rng.choice(len(rec), args.max_rows, replace=False))
        rec, hold = rec[idx], hold[idx]
    mini_idx, super_idx, constr, stm_sign = ntb.features(rec)
    print(f"{len(rec):,} rows; holdout {hold.mean():.1%}", flush=True)
    search = rec["search"].astype(np.float32)
    valid = (rec["flags"] & eval_data.F_RESULT) != 0
    wdl = (rec["result"].astype(np.float32) + 1) / 2
    k = args.k
    if args.check_fit_k:
        kf, ll = ntb.fit_k(search[valid & ~hold], wdl[valid & ~hold])
        print(f"fit_k check: K = {kf:.0f} (log loss {ll:.4f}); using K = {k:.0f}", flush=True)
    target = 1 / (1 + np.exp(-search / k))  # lam = 1: search only
    Y = torch.from_numpy(target.astype(np.float32))
    S = torch.from_numpy(search)

    if args.device == "dml":
        import torch_directml
        dev = torch_directml.device()
    else:
        dev = torch.device(args.device)
    T = lambda a, dt=torch.long: torch.from_numpy(np.ascontiguousarray(a)).to(dt)  # noqa: E731
    X = dict(mini=T(mini_idx), sup=T(super_idx), con=T(constr), hce=T(rec["hce"], torch.float32),
             sgn=T(stm_sign, torch.float32))
    del rec
    mini_w, macro_w = ntb.load_shipped_mini(), ntb.load_shipped_macro()

    frozen_base = args.arm in ("f", "fns", "g")
    if frozen_base:
        ship = ntb.Eval(mini_w, macro_w, "centroid").to(dev).eval()
        base = torch.empty(len(Y))
        with torch.no_grad():
            for i in range(0, len(Y), 131072):
                sl = slice(i, i + 131072)
                base[sl] = ship(X["mini"][sl].to(dev), X["sup"][sl].to(dev), X["con"][sl].to(dev),
                                X["hce"][sl].to(dev), X["sgn"][sl].to(dev))[0].cpu()
        X["base"] = base
        del ship

    # Parameter lists are built after .to(dev): moving to DirectML replaces the Parameter objects.
    groups_info, new_names = None, set()
    if args.arm == "a":
        model = ntb.Eval(mini_w, macro_w, "full")
    elif args.arm in WIDE:
        cfg = WIDE[args.arm]
        model = Wide(mini_w, macro_w, **cfg)
        groups_info, new_names = model.groups, set(model.new_names)
    else:
        res = TokenTransformer(side=args.arm != "fns") if args.arm in ("f", "fns", "fj") else PatternNNUE()
        base_model = None if frozen_base else ntb.Eval(mini_w, macro_w, "full")
        model = Residual(res, base_model)
        new_names = {"res." + n for n, _ in res.named_parameters()}
    model = model.to(dev)
    named = [(n, p) for n, p in model.named_parameters() if not n.endswith("hce_w")]
    old = [p for n, p in named if n not in new_names]
    new = [p for n, p in named if n in new_names]
    assert len(new) == len(new_names), (len(new), new_names)
    new_lr = args.new_lr if args.arm in WIDE else args.res_lr
    groups = []
    if old:
        groups.append({"params": old, "lr": args.lr})
    if new:
        groups.append({"params": new, "lr": new_lr})
    print(f"arm {args.arm}: trainable params old {n_params(old):,} (lr {args.lr}) new {n_params(new):,}"
          f" (lr {new_lr}); dup groups {groups_info}", flush=True)

    def run(idx):
        a = (X["mini"][idx].to(dev), X["sup"][idx].to(dev), X["con"][idx].to(dev), X["hce"][idx].to(dev),
             X["sgn"][idx].to(dev))
        if frozen_base:
            return model(*a, base=X["base"][idx].to(dev))
        out = model(*a)
        return out[0] if isinstance(out, tuple) else out

    tr = np.flatnonzero(~hold)
    va = torch.from_numpy(np.flatnonzero(hold))

    def evaluate():
        model.eval()
        tot = dict(loss=0.0, mae=0.0, ll=0.0, n=0, nv=0)
        with torch.no_grad():
            for i in range(0, len(va), 65536):
                idx = va[i:i + 65536]
                e = run(idx).cpu()
                p = torch.sigmoid(e / k)
                tot["loss"] += (p - Y[idx]).pow(2).sum().item()
                tot["mae"] += (e - S[idx]).abs().clamp(max=20000).sum().item()
                v = torch.from_numpy(valid[idx.numpy()])
                y = torch.from_numpy(wdl[idx.numpy()])[v]
                pv = p[v].clamp(1e-6, 1 - 1e-6)
                tot["ll"] += -(y * torch.log(pv) + (1 - y) * torch.log(1 - pv)).sum().item()
                tot["n"] += len(idx)
                tot["nv"] += int(v.sum())
        model.train()
        return tot["loss"] / tot["n"], tot["mae"] / tot["n"], tot["ll"] / max(1, tot["nv"])

    base0 = evaluate()
    print(f"[{time.strftime('%H:%M:%S')}] epoch 0 on holdout: loss {base0[0]:.6f}  |E-search| {base0[1]:.0f}"
          f"  result ll {base0[2]:.4f}", flush=True)
    opt = torch.optim.Adam(groups)
    n_steps = math.ceil(len(tr) / args.batch)
    steps = max(1, args.epochs * n_steps)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in groups], total_steps=steps,
                                                pct_start=0.05)
    best = (base0[0], 0)
    log = [dict(epoch=0, val_loss=base0[0], mae=base0[1], ll=base0[2])]
    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        perm = torch.from_numpy(rng.permutation(tr))
        run_loss = 0.0
        for step, i in enumerate(range(0, len(perm), args.batch)):
            if step and step % max(1, n_steps // 4) == 0:
                done = (epoch - 1) * n_steps + step
                eta = (time.time() - t_start) / done * (steps - done)
                print(f"[{time.strftime('%H:%M:%S')}] epoch {epoch} step {step}/{n_steps}"
                      f" train {run_loss / (i or 1):.6f} eta {eta / 60:.1f}m", flush=True)
            idx = perm[i:i + args.batch]
            e = run(idx)
            loss = (torch.sigmoid(e / k) - Y[idx].to(dev)).pow(2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            run_loss += loss.item() * len(idx)
        val = evaluate()
        log.append(dict(epoch=epoch, train_loss=run_loss / len(tr), val_loss=val[0], mae=val[1], ll=val[2],
                        secs=time.time() - t0))
        print(f"[{time.strftime('%H:%M:%S')}] epoch {epoch}: train {run_loss / len(tr):.6f}  val {val[0]:.6f}"
              f"  |E-search| {val[1]:.0f}  result ll {val[2]:.4f}  ({time.time() - t0:.0f}s)", flush=True)
        if not math.isfinite(val[0]):
            print("non-finite validation loss; stopping", flush=True)
            break
        if val[0] < best[0]:
            best = (val[0], epoch)
            if args.arm in WIDE or args.arm == "a":  # keep the best epoch's weights for quant_eval.py
                best_state = {n: p.detach().cpu().clone() for n, p in model.named_parameters()}
    ship_loss, ffull = 0.046817, 0.043527
    summary = dict(arm=args.arm, tag=args.tag, best_val=best[0], best_epoch=best[1], epoch0=base0[0],
                   vs_shipped_pct=100 * (best[0] / ship_loss - 1), vs_ffull_pct=100 * (best[0] / ffull - 1),
                   params_old=n_params(old), params_new=n_params(new), lr=args.lr, new_lr=new_lr,
                   train_minutes=(time.time() - t_start) / 60, total_minutes=(time.time() - t_all) / 60,
                   rows=int(len(Y)), groups=groups_info, log=log, args=vars(args))
    (OUT / f"{name}.json").write_text(json.dumps(summary, indent=1))
    print(f"RESULT {name}: best val {best[0]:.6f} (epoch {best[1]}), {summary['vs_shipped_pct']:+.2f}% vs shipped,"
          f" {summary['vs_ffull_pct']:+.2f}% vs F_full; train {summary['train_minutes']:.1f} min", flush=True)
    if args.arm in WIDE or args.arm == "a":
        torch.save(best_state if best[1] else {n: p.detach().cpu() for n, p in model.named_parameters()},
                   OUT / f"{name}.pt")


if __name__ == "__main__":
    main()
