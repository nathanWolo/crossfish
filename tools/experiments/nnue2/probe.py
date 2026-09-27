#!/usr/bin/env python3
"""Offline "was the full NNUE trained wrong?" probe (full replacement, no HCE base).

Same data, split, target and K as F_full and the capacity probes
(datasets/eval2/cf_play_d14.cfdg + uttt_sp_d14.cfdg; 10% game holdout from
nnue_train_blend.game_holdout; lam 1; K 1600; win-probability MSE). Unlike the
capacity probes, every model here predicts the WHOLE eval from the position:
no HCE, MiniNet or macro term underneath.

Arms (--arm):
  r10   round-ten net (tools/experiments/full_nnue/train_full_nnue.py): 199
        sparse features per perspective (live-board stones mine/theirs, decided
        boards mine/theirs/drawn, constraint), shared accumulator A=256 per
        perspective, clipped ReLU, 2A -> 32 -> 1, output x1000.
  pat   pattern NNUE: per perspective acc = b + sum_m FT[m, pattern_m] +
        FS[m, board state_m] + FC[constraint] (width 128; a decided board's
        pattern is read as empty), clipped ReLU, 2x128 -> 32 -> 1, output x1000.
  patact  pat plus FA[pattern of the forced miniboard] (the MiniNet's "active"
        idea as an accumulator feature).
--aug: a random one of the 8 dihedral symmetries (applied to the macro and
every miniboard at once) per training row per step; the holdout is scored in
its original orientation.
--train-rows N: nested subsets of the training GAMES (so fewer rows means
fewer games), for the data-scaling study.

Subcommands:
  train ...                      one arm; writes datasets/nnue2/probe/<name>.json (+ .pt)
  symcheck-prep N                write N holdout records and their 7 transforms to probe/sym_in.cfdg
  symcheck-compare               compare after `datagen label sym_in.cfdg sym_out.cfdg 0 1`
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse  # noqa: E402
import ctypes  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch import nn  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))
import eval_data  # noqa: E402
import nnue_train_blend as ntb  # noqa: E402

DATA = [ROOT / "datasets/eval2/cf_play_d14.cfdg", ROOT / "datasets/eval2/uttt_sp_d14.cfdg"]
OUT = ROOT / "datasets/nnue2/probe"
SHIP_LOSS, FFULL_LOSS, A_LR3E3_LOSS = 0.046817, 0.043527, 0.038353
N_PAT = 19683
POW3 = 3 ** np.arange(9)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def repo_path(p):
    """A path given on the command line: absolute, or relative to the repository root (so the tools run from
    any working directory; the recorded args keep the relative form, e.g. datasets/nnue2/d8_a.cfdg)."""
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def below_normal_priority():
    if os.name == "nt":
        k32 = ctypes.windll.kernel32
        k32.SetPriorityClass(k32.GetCurrentProcess(), 0x4000)  # BELOW_NORMAL_PRIORITY_CLASS


# ---------------------------------------------------------------- symmetries
def dihedral():
    """P[g][p]: image of 3x3 position p = 3*row + col under symmetry g."""
    maps = []
    for rot in range(4):
        for flip in (False, True):
            m = []
            for p in range(9):
                r, c = divmod(p, 3)
                if flip:
                    c = 2 - c
                for _ in range(rot):
                    r, c = c, 2 - r
                m.append(3 * r + c)
            maps.append(m)
    P = np.array(maps, dtype=np.int64)
    assert len({tuple(m) for m in maps}) == 8
    lines = {frozenset(l) for l in eval_data.WIN_LINES}
    for m in P:  # a game symmetry maps winning lines to winning lines
        assert {frozenset(int(m[i]) for i in l) for l in lines} == lines
    return P


P9 = dihedral()
P9_INV = np.argsort(P9, axis=1)
# gather form: new[k] = old[G[g][k]]; cell index = mb * 9 + sq (the 93-char state layout)
G81 = np.array([[P9_INV[g][k // 9] * 9 + P9_INV[g][k % 9] for k in range(81)] for g in range(8)], dtype=np.int64)
G9 = P9_INV.copy()
CONMAP = np.concatenate([P9, np.full((8, 1), 9)], axis=1)  # constraint 9 = free move


def transform_states(s, g):
    """s: (n, 93) uint8 ASCII states; g: symmetry id per row."""
    out = s.copy()
    out[:, :81] = np.take_along_axis(s[:, :81], G81[g], axis=1)
    out[:, 81:90] = np.take_along_axis(s[:, 81:90], G9[g], axis=1)
    con = s[:, 91].astype(np.int64) - 48
    out[:, 91] = (CONMAP[g, np.clip(con, 0, 9)] + 48).astype(np.uint8)
    return out


# ---------------------------------------------------------------- data
def compact(rec):
    """-> (N, 91) uint8: cells rel. to side to move (0 empty, 1 mine, 2 theirs), board states
    (0 live, 1 mine, 2 theirs, 3 drawn), constraint (0..8 forced, 9 free)."""
    s = np.frombuffer(rec["s"].tobytes(), dtype=np.uint8).reshape(-1, 93).astype(np.int16) - 48
    stm = np.where(s[:, 90] == 1, 1, 2)[:, None]
    cells = s[:, :81]
    c = np.where(cells == 0, 0, np.where(cells == stm, 1, 2))
    sup = s[:, 81:90]
    u = np.where(sup == 0, 0, np.where(sup == 3, 3, np.where(sup == stm, 1, 2)))
    con = np.clip(s[:, 91:92], 0, 9)
    return np.concatenate([c, u, con], axis=1).astype(np.uint8)


def load_data(max_rows=0, seed=1):
    rec, _, hold = ntb.load_all([str(p) for p in DATA], None)
    # game key unique across both files (load_all concatenates cf_play then uttt_sp)
    n0 = int(((eval_data.load(str(DATA[0]))["flags"] & eval_data.F_SEARCH) != 0).sum())
    fidx = np.zeros(len(rec), dtype=np.int64)
    fidx[n0:] = 1
    game = rec["game"].astype(np.int64) + fidx * (1 << 32)
    if max_rows and len(rec) > max_rows:
        idx = np.sort(np.random.default_rng(seed).choice(len(rec), max_rows, replace=False))
        rec, hold, game = rec[idx], hold[idx], game[idx]
    x = compact(rec)
    search = rec["search"].astype(np.float32)
    static = rec["static_eval"].astype(np.float32)
    hce = rec["hce"].astype(np.float32)
    return x, search, static, hce, hold, game


# ---------------------------------------------------------------- models
class Feats:
    """Device-side constants shared by the models."""

    def __init__(self, dev):
        self.dev = dev
        self.cell_mb = torch.arange(81, device=dev) // 9
        self.pow3 = torch.tensor(POW3, dtype=torch.long, device=dev)
        d = np.array([(p // POW3) % 3 for p in range(N_PAT)])
        sw = np.where(d == 1, 2, np.where(d == 2, 1, 0))
        self.pat_swap = torch.tensor((sw * POW3).sum(1), dtype=torch.long, device=dev)
        self.sup_swap = torch.tensor([0, 2, 1, 3], dtype=torch.long, device=dev)
        self.ar10 = torch.arange(10, device=dev)
        self.loc = torch.arange(9, device=dev)
        self.G81 = torch.tensor(G81, device=dev)
        self.G9 = torch.tensor(G9, device=dev)
        self.CON = torch.tensor(CONMAP, device=dev)

    def split(self, xb):
        xb = xb.long()
        return xb[:, :81], xb[:, 81:90], xb[:, 90]

    def augment(self, cells, sup, con):
        g = torch.randint(0, 8, (cells.shape[0],), device=self.dev)
        cells = torch.gather(cells, 1, self.G81[g])
        sup = torch.gather(sup, 1, self.G9[g])
        con = self.CON[g, con]
        return cells, sup, con


class R10(nn.Module):
    """Round-ten network (train_full_nnue.Net) on dense multi-hot inputs."""

    def __init__(self, fx, A=256, L1=32):
        super().__init__()
        self.fx = fx
        self.W = nn.Parameter(0.05 * torch.randn(199, A))
        self.b0 = nn.Parameter(torch.zeros(A))
        self.l1 = nn.Linear(2 * A, L1)
        self.l2 = nn.Linear(L1, 1)

    def forward(self, cells, sup, con):
        live = torch.gather(sup, 1, self.fx.cell_mb.expand(cells.shape[0], 81)) == 0
        mine = ((cells == 1) & live).float()
        theirs = ((cells == 2) & live).float()
        b = cells.shape[0]
        s1, s2, s3 = (sup == 1).float(), (sup == 2).float(), (sup == 3).float()
        c = (con[:, None] == self.fx.ar10[None, :]).float()
        xs = torch.cat([torch.stack([mine, theirs], 2).reshape(b, 162), torch.stack([s1, s2, s3], 2).reshape(b, 27), c], 1)
        xn = torch.cat([torch.stack([theirs, mine], 2).reshape(b, 162), torch.stack([s2, s1, s3], 2).reshape(b, 27), c], 1)
        a = torch.clamp(xs @ self.W + self.b0, 0, 1)
        n = torch.clamp(xn @ self.W + self.b0, 0, 1)
        h = torch.clamp(self.l1(torch.cat([a, n], 1)), 0, 1)
        return self.l2(h).squeeze(1) * 1000.0


class Pat(nn.Module):
    """Pattern NNUE, two perspectives with shared weights."""

    def __init__(self, fx, width=128, L1=32, active=False):
        super().__init__()
        self.fx, self.active = fx, active
        self.ft = nn.Parameter(0.05 * torch.randn(9 * N_PAT, width))
        self.fs = nn.Parameter(0.05 * torch.randn(36, width))
        self.fc = nn.Parameter(0.05 * torch.randn(10, width))
        if active:
            self.fa = nn.Parameter(0.05 * torch.randn(N_PAT + 1, width))  # last row: free move
        self.fb = nn.Parameter(torch.full((width,), 0.5))
        self.l1 = nn.Linear(2 * width, L1)
        self.l2 = nn.Linear(L1, 1)

    def acc(self, pat, sup, con, pat_forced):
        a = self.fb + F.embedding(self.fx.loc * N_PAT + pat, self.ft).sum(1) \
            + F.embedding(self.fx.loc * 4 + sup, self.fs).sum(1) + F.embedding(con, self.fc)
        if self.active:
            a = a + F.embedding(pat_forced, self.fa)
        return torch.clamp(a, 0, 1)

    def forward(self, cells, sup, con):
        b = cells.shape[0]
        pat = (cells.reshape(b, 9, 9) * self.fx.pow3).sum(2)
        pat = torch.where(sup == 0, pat, torch.zeros_like(pat))
        pat_n = self.fx.pat_swap[pat]
        sup_n = self.fx.sup_swap[sup]
        if self.active:
            forced = con < 9
            cidx = torch.clamp(con, max=8)[:, None]
            pf = torch.gather(pat, 1, cidx).squeeze(1)
            pfn = torch.gather(pat_n, 1, cidx).squeeze(1)
            pf = torch.where(forced, pf, torch.full_like(pf, N_PAT))
            pfn = torch.where(forced, pfn, torch.full_like(pfn, N_PAT))
        else:
            pf = pfn = None
        a = self.acc(pat, sup, con, pf)
        n = self.acc(pat_n, sup_n, con, pfn)
        h = torch.clamp(self.l1(torch.cat([a, n], 1)), 0, 1)
        return self.l2(h).squeeze(1) * 1000.0


def n_params(m):
    return int(sum(p.numel() for p in m.parameters()))


# ---------------------------------------------------------------- train
def cmd_train(args):
    below_normal_priority()
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    rng = np.random.default_rng(args.seed)
    name = args.name or args.arm
    OUT.mkdir(parents=True, exist_ok=True)
    t_all = time.time()
    x, search, static, hce, hold, game = load_data(args.max_rows, args.seed)
    k = args.k
    n_extra = 0
    if args.extra:
        # extra TRAINING rows from another .cfdg (e.g. the depth-8 self-play file while it is
        # still being written: only its first --extra-rows whole records are mapped)
        mm = np.memmap(repo_path(args.extra), dtype=eval_data.REC, mode="r", shape=(args.extra_rows,))
        n0, cap = len(x), len(x) + args.extra_rows
        # preallocate and fill in place: concatenating 20M+ rows would double the peak memory
        grow = lambda a: np.concatenate([a, np.zeros((cap - n0,) + a.shape[1:], dtype=a.dtype)])  # noqa: E731
        x, search, static, hce, game = grow(x), grow(search), grow(static), grow(hce), grow(game)
        for i in range(0, args.extra_rows, 2_000_000):  # chunks keep compact()'s temporaries small
            er = np.array(mm[i:i + 2_000_000])
            er = er[(er["flags"] & eval_data.F_SEARCH) != 0]
            sl = slice(n0 + n_extra, n0 + n_extra + len(er))
            x[sl] = compact(er)
            search[sl] = er["search"]
            static[sl] = er["static_eval"]
            hce[sl] = er["hce"]
            game[sl] = er["game"].astype(np.int64) + (2 << 32)
            n_extra += len(er)
            del er
        del mm
        x, search, static, hce, game = (a[:n0 + n_extra] for a in (x, search, static, hce, game))
        hold = np.concatenate([hold, np.zeros(n_extra, dtype=bool)])
        log(f"extra: {n_extra:,} labeled rows from the first {args.extra_rows:,} records of {args.extra}"
            f" (search std {search[len(search) - n_extra:].std():.0f})")
    tr = np.flatnonzero(~hold)
    va = np.flatnonzero(hold)
    n_train_full = len(tr)
    if args.train_rows and args.train_rows < len(tr):
        # nested subsets of training games: a fixed random order of games, take a prefix
        g_tr = game[tr]
        ug, inv = np.unique(g_tr, return_inverse=True)
        order = np.random.default_rng(12345).permutation(len(ug))
        rank = np.empty(len(ug), dtype=np.int64)
        rank[order] = np.arange(len(ug))
        sizes = np.bincount(inv)[order]
        n_games = int(np.searchsorted(np.cumsum(sizes), args.train_rows)) + 1
        tr = tr[rank[inv] < n_games]
        log(f"train subset: {n_games:,} of {len(ug):,} games, {len(tr):,} rows")
    log(f"{len(x):,} rows; train {len(tr):,}, holdout {len(va):,}")

    if args.device == "dml":
        import torch_directml
        dev = torch_directml.device()
    else:
        dev = torch.device(args.device)
    fx = Feats(dev)
    X = torch.from_numpy(x)
    Y = torch.from_numpy((1 / (1 + np.exp(-search / k))).astype(np.float32))
    S = search
    if args.arm == "r10":
        model = R10(fx, args.A, args.L1)
    else:
        model = Pat(fx, args.width, args.L1, active=args.arm == "patact")
    model = model.to(dev)
    log(f"arm {args.arm}: {n_params(model):,} params, lr {args.lr}, batch {args.batch}, aug {args.aug}")

    vx = X[va]
    ship_std = float(static[va].std())
    ship_loss = float(((1 / (1 + np.exp(-static[va] / k)) - Y[va].numpy()) ** 2).mean())
    hce_loss = float(((1 / (1 + np.exp(-hce[va] / k)) - Y[va].numpy()) ** 2).mean())
    log(f"holdout: shipped static_eval loss {ship_loss:.6f} (std {ship_std:.0f}); HCE alone {hce_loss:.6f}"
        f" (std {hce[va].std():.0f}); search std {S[va].std():.0f}")

    def predict(xs):
        model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(xs), 65536):
                out.append(model(*fx.split(xs[i:i + 65536].to(dev))).cpu())
        model.train()
        return torch.cat(out).numpy().astype(np.float64)

    def stats(e):
        yv = Y[va].numpy().astype(np.float64)
        p = 1 / (1 + np.exp(-e / k))
        nm = np.abs(S[va]) < 8000
        return dict(val=float(((p - yv) ** 2).mean()), std=float(e.std()),
                    mae=float(np.abs(e - S[va]).clip(max=20000).mean()),
                    corr_search=float(np.corrcoef(e[nm], S[va][nm])[0, 1]),
                    corr_static=float(np.corrcoef(e, static[va])[0, 1]))

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, foreach=False)
    n_steps = len(tr) // args.batch
    steps = max(1, args.epochs * n_steps)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.05)
    best, hist, bad = None, [], 0
    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        perm = torch.from_numpy(rng.permutation(tr))
        run_loss, seen = 0.0, 0
        for step in range(n_steps):
            if step and step % max(1, n_steps // 4) == 0:
                done = (epoch - 1) * n_steps + step
                eta = (time.time() - t_start) / done * (steps - done)
                log(f"{name} epoch {epoch} step {step}/{n_steps} train {run_loss / seen:.6f} eta {eta / 60:.1f}m")
            idx = perm[step * args.batch:(step + 1) * args.batch]
            cells, sup, con = fx.split(X[idx].to(dev))
            if args.aug:
                cells, sup, con = fx.augment(cells, sup, con)
            e = model(cells, sup, con)
            loss = (torch.sigmoid(e / k) - Y[idx].to(dev)).pow(2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            run_loss += loss.item() * len(idx)
            seen += len(idx)
        st = stats(predict(vx))
        st.update(epoch=epoch, train=run_loss / max(1, seen), secs=time.time() - t0)
        hist.append(st)
        log(f"{name} epoch {epoch}: train {st['train']:.6f} val {st['val']:.6f} std {st['std']:.0f}"
            f" |E-search| {st['mae']:.0f} corr(search) {st['corr_search']:.3f} corr(static) {st['corr_static']:.3f}"
            f" ({st['secs']:.0f}s)")
        if not math.isfinite(st["val"]):
            break
        if best is None or st["val"] < best["val"]:
            best, bad = st, 0
            if args.save:
                torch.save({n: p.detach().cpu() for n, p in model.state_dict().items()}, OUT / f"{name}.pt")
        else:
            bad += 1
            if args.patience and bad >= args.patience:
                log(f"{name}: no improvement for {bad} epochs; stopping")
                break
    summary = dict(name=name, arm=args.arm, aug=args.aug, best=best, hist=hist, params=n_params(model),
                   train_rows=int(len(tr)), train_rows_full=int(n_train_full), extra_rows=int(n_extra), holdout_rows=int(len(va)),
                   shipped_loss=ship_loss, shipped_std=ship_std, hce_loss=hce_loss,
                   vs_shipped_pct=100 * (best["val"] / ship_loss - 1), vs_ffull_pct=100 * (best["val"] / FFULL_LOSS - 1),
                   train_minutes=(time.time() - t_start) / 60, total_minutes=(time.time() - t_all) / 60,
                   args={a: v for a, v in vars(args).items() if a != "fn"})
    (OUT / f"{name}.json").write_text(json.dumps(summary, indent=1))
    log(f"RESULT {name}: best val {best['val']:.6f} (epoch {best['epoch']}), {summary['vs_shipped_pct']:+.2f}% vs"
        f" shipped {ship_loss:.6f}, {summary['vs_ffull_pct']:+.2f}% vs F_full; std {best['std']:.0f} vs shipped"
        f" {ship_std:.0f}; train {summary['train_minutes']:.1f} min")


# ---------------------------------------------------------------- symmetry check
def cmd_symcheck_prep(args):
    OUT.mkdir(parents=True, exist_ok=True)
    r = np.array(eval_data.load(str(DATA[0])))
    rng = np.random.default_rng(7)
    idx = np.sort(rng.choice(len(r), args.n, replace=False))
    base = r[idx]
    parts = []
    s = np.frombuffer(base["s"].tobytes(), dtype=np.uint8).reshape(-1, 93)
    for g in range(8):
        t = base.copy()
        ts = transform_states(s, np.full(len(s), g))
        t["s"] = np.frombuffer(ts.tobytes(), dtype="S93")
        t["game"] = np.arange(len(t)) + g * len(t)
        parts.append(t)
    out = np.concatenate(parts)
    out.tofile(OUT / "sym_in.cfdg")
    print(f"{len(out):,} records (8 symmetries x {args.n}) -> {OUT / 'sym_in.cfdg'}; now run:\n"
          f"  cpp_impl/bin/datagen.exe label {OUT / 'sym_in.cfdg'} {OUT / 'sym_out.cfdg'} 0 1")


def cmd_symcheck_compare(args):
    r = np.array(eval_data.load(str(OUT / "sym_out.cfdg")))
    n = len(r) // 8
    se = r["static_eval"].reshape(8, n).astype(np.int64)
    hc = r["hce"].reshape(8, n).astype(np.int64)
    res = {}
    for name, a in (("static_eval", se), ("hce", hc)):
        d = np.abs(a[1:] - a[0][None, :])
        res[name] = dict(exact_frac=float((d == 0).mean()), mean_abs=float(d.mean()), p99=float(np.percentile(d, 99)),
                         max=int(d.max()), per_sym_mean_abs=[float(x) for x in d.mean(1)],
                         std=float(a[0].std()))
    # python-side decoded MiniNet + macro (ntb.Eval, shipped centroids) for the same records
    ev = ntb.Eval(ntb.load_shipped_mini(), ntb.load_shipped_macro(), "centroid").eval()
    mini_idx, super_idx, constr, sgn = ntb.features(r)
    T = lambda a: torch.from_numpy(np.ascontiguousarray(a)).long()  # noqa: E731
    with torch.no_grad():
        e, mini, macro = ev(T(mini_idx), T(super_idx), T(constr), torch.from_numpy(r["hce"].astype(np.float32)),
                            T(sgn).float())
    py = e.numpy()
    res["python_vs_cpp_static_mean_abs"] = float(np.abs(py - r["static_eval"]).mean())
    for nm, a in (("mini", mini.numpy()), ("macro", macro.numpy())):
        a = a.reshape(8, n)
        d = np.abs(a[1:] - a[0][None, :])
        res[nm] = dict(mean_abs=float(d.mean()), p99=float(np.percentile(d, 99)), std=float(a[0].std()))
    # the transform itself: the identity copy must reproduce the dataset's own static_eval up to Dev drift
    rin = np.array(eval_data.load(str(OUT / "sym_in.cfdg")))
    res["identity_vs_recorded_static_mean_abs"] = float(np.abs(se[0] - rin["static_eval"][:n]).mean())
    res["identity_vs_recorded_static_exact"] = float((se[0] == rin["static_eval"][:n]).mean())
    (OUT / "symcheck.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("train")
    p.add_argument("--arm", required=True, choices=["r10", "pat", "patact"])
    p.add_argument("--name", default="")
    p.add_argument("--aug", action="store_true")
    p.add_argument("--k", type=float, default=1600.0)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--patience", type=int, default=0)
    p.add_argument("--batch", type=int, default=16384)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--A", type=int, default=256)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--L1", type=int, default=32)
    p.add_argument("--train-rows", type=int, default=0)
    p.add_argument("--extra", default="", help="extra training .cfdg (read-only, first --extra-rows records)")
    p.add_argument("--extra-rows", type=int, default=5_000_000)
    p.add_argument("--max-rows", type=int, default=0, help="smoke tests: random subset of all rows")
    p.add_argument("--device", default="dml")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--save", action="store_true")
    p.set_defaults(fn=cmd_train)
    p = sub.add_parser("symcheck-prep")
    p.add_argument("n", type=int)
    p.set_defaults(fn=cmd_symcheck_prep)
    p = sub.add_parser("symcheck-compare")
    p.set_defaults(fn=cmd_symcheck_compare)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
