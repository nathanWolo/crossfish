#!/usr/bin/env python3
"""Train the evaluation in win-probability space on searched evals and game results.

Stockfish-style objective (documentation/eval_data.md):

    E(p)   = HCE(p) + MiniNet(p) + Macro(p)          engine units, side to move
    pred   = sigmoid(E / K)
    target = lam * sigmoid(search / K) + (1 - lam) * wdl        (wdl = 1, 0.5, 0)
             optionally blended with uttt.ai's value: (1 - mu) * target + mu * (v + 1) / 2
    loss   = |pred - target| ** power

K is fitted so that sigmoid(search / K) best predicts the game results. Rows
whose result is not valid (positions before a uniform-random move) use
lam = 1. Mate scores (clamped to +/-20000) become targets near 0 or 1 without
special handling.

The MiniNet and macro head start from the shipped headers (cpp_impl/
mini_eval_d16.hpp, cpp_impl/macro_eval.hpp), decoded exactly. In the default
centroid mode the MiniNet trains its 256 centroids with the shipped
pattern-to-centroid codes held fixed, so the trained model is exactly the
deployable one; `--mode full` trains all 19,683 pattern embeddings and
re-clusters at export. `--hce-weights DIFFS...` also trains the HCE weights
from datagen's eval_diffs sidecars (the pawn weight stays fixed: every search
margin is expressed in it).

The HCE term is each row's hce column, so the net is fitted on top of the Dev
that labeled the data. When that Dev carried source patches (the *_hd files:
the drawn-miniboard HCE fix), pass the same patch list as `--dev-patches FILE`
(eval_candidate.py's dev_patches.json format). It only goes into <out>.json
("dev_patches"), from which eval_screen_plan.py builds the candidate's Dev.

Validation holds out 10% of games, so positions of one game never straddle
the split.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import struct
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_data  # noqa: E402
from nnue_cjk14 import decode_cjk14  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
N_IDX = 19683
MACRO_CLIP = 2000
HCE_WEIGHTS = [2410, 836, 464, 1316, 534, 424, 33, 10, 33, 112]  # crossfish_dev.hpp eval_weights
PAWN_IDX = 7
TRAINABLE_HCE = [0, 1, 2, 3, 4, 5, 6, 8]


def _payload(header: Path, name: str) -> bytes:
    text = header.read_text(encoding="utf-8")
    m = re.search(name + r'\[\] = R"~\(\n(.*?)\n\)~";', text, re.S)
    if not m:
        raise SystemExit(f"no {name} payload in {header}")
    return decode_cjk14(m.group(1))


def load_shipped_mini(header: Path = ROOT / "cpp_impl/mini_eval_d16.hpp"):
    blob = _payload(header, "D16_MINI_PACK_CJK")
    d, h, k = 16, 8, 256
    off = 0

    def take(count, dtype="<f4"):
        nonlocal off
        size = count * np.dtype(dtype).itemsize
        out = np.frombuffer(blob[off:off + size], dtype=dtype).copy()
        off += size
        return out

    codes = take(N_IDX, "u1")
    cents = take(k * d).reshape(k, d)
    super_e = take(4 * d).reshape(4, d)
    loc = take(9 * d).reshape(9, d)
    constr = take(10 * d).reshape(10, d)
    active = take(2 * d).reshape(2, d)
    w1 = take(h * 10 * d).reshape(h, 10 * d)
    b1 = take(h)
    w2 = take(h)
    b2 = float(take(1)[0])
    return dict(codes=codes, cents=cents, super_e=super_e, loc=loc, constr=constr,
                active=active, w1=w1, b1=b1, w2=w2, b2=b2)


def load_shipped_macro(header: Path = ROOT / "cpp_impl/macro_eval.hpp"):
    blob = np.frombuffer(_payload(header, "MACRO_PACK_CJK"), dtype="<f4")
    base = blob[0:16]
    constr = blob[16:176].reshape(10, 16)
    emb = blob[176:176 + 576].reshape(9, 4, 16)
    out = blob[752:768]
    bias = float(blob[768])
    return dict(base=base.copy(), constr=constr.copy(), emb=emb.copy(), out=out.copy(), bias=bias)


def features(rec):
    """(N,) DgRec -> mini_idx (N,9) int16, super_idx (N,9) int8, constr (N,) int8, stm_sign (N,) int8."""
    s = np.frombuffer(rec["s"].tobytes(), dtype=np.uint8).reshape(-1, 93).astype(np.int16) - 48
    stm = np.where(s[:, 90] == 1, 1, 2)[:, None]  # player digit of the side to move
    cells = s[:, :81].reshape(-1, 9, 9)
    digit = np.where(cells == 0, 0, np.where(cells == stm[:, :, None], 1, 2))
    mini_idx = (digit * (3 ** np.arange(9))[None, None, :]).sum(axis=2).astype(np.int16)
    sup = s[:, 81:90]
    super_idx = np.where(sup == 0, 0, np.where(sup == 3, 3, np.where(sup == stm, 1, 2))).astype(np.int8)
    constr = np.clip(s[:, 91], 0, 9).astype(np.int8)
    stm_sign = np.where(s[:, 90] == 1, 1, -1).astype(np.int8)
    return mini_idx, super_idx, constr, stm_sign


class Eval(nn.Module):
    def __init__(self, mini, macro, mode="centroid", hce_train=False):
        super().__init__()
        self.mode = mode
        codes = torch.from_numpy(mini["codes"].astype(np.int64))
        self.register_buffer("codes", codes)
        if mode == "centroid":
            self.emb = nn.Parameter(torch.from_numpy(mini["cents"]))
        else:
            self.emb = nn.Parameter(torch.from_numpy(mini["cents"][mini["codes"]]))
        self.super_e = nn.Parameter(torch.from_numpy(mini["super_e"]))
        self.loc = nn.Parameter(torch.from_numpy(mini["loc"]))
        self.constr = nn.Parameter(torch.from_numpy(mini["constr"]))
        self.active = nn.Parameter(torch.from_numpy(mini["active"]))
        self.w1 = nn.Parameter(torch.from_numpy(mini["w1"]))
        self.b1 = nn.Parameter(torch.from_numpy(mini["b1"]))
        self.w2 = nn.Parameter(torch.from_numpy(mini["w2"]))
        self.b2 = nn.Parameter(torch.tensor(mini["b2"]))
        self.m_base = nn.Parameter(torch.from_numpy(macro["base"]))
        self.m_constr = nn.Parameter(torch.from_numpy(macro["constr"]))
        self.m_emb = nn.Parameter(torch.from_numpy(macro["emb"]))
        self.m_out = nn.Parameter(torch.from_numpy(macro["out"]))
        w0 = torch.tensor(HCE_WEIGHTS, dtype=torch.float32)
        self.register_buffer("hce_w0", w0)
        self.hce_w = nn.Parameter(w0.clone(), requires_grad=hce_train)
        mask = torch.zeros(10)
        mask[TRAINABLE_HCE] = 1.0
        self.register_buffer("hce_mask", mask)

    def table(self):
        return self.emb[self.codes] if self.mode == "centroid" else self.emb

    def mini(self, mini_idx, super_idx, constr):
        b = mini_idx.shape[0]
        loc = torch.arange(9, device=mini_idx.device).expand(b, 9)
        active = ((loc == constr[:, None]) & (constr[:, None] < 9)).long()
        v = self.table()[mini_idx] + self.super_e[super_idx] + self.loc[loc] + self.active[active]
        x = torch.cat([v.reshape(b, 144), self.constr[constr]], dim=1)
        return torch.relu(x @ self.w1.T + self.b1) @ self.w2 + self.b2

    def macro_raw(self, super_idx, constr):
        h = self.m_base + self.m_constr[constr]
        h = h + self.m_emb[torch.arange(9, device=super_idx.device)[None, :], super_idx].sum(dim=1)
        return torch.relu(h) @ self.m_out

    def forward(self, mini_idx, super_idx, constr, hce, stm_sign, diffs=None):
        dev = mini_idx.device
        ez = torch.zeros((1, 9), dtype=torch.long, device=dev)
        cz = torch.full((1,), 9, dtype=torch.long, device=dev)
        mini = self.mini(mini_idx, super_idx, constr) - self.mini(ez, ez, cz)
        macro = torch.clamp(self.macro_raw(super_idx, constr) - self.macro_raw(ez, cz), -MACRO_CLIP, MACRO_CLIP)
        e = hce + mini + macro
        if diffs is not None:
            dw = (self.hce_w - self.hce_w0) * self.hce_mask
            e = e + stm_sign * (diffs @ dw)
        return e, mini, macro


def fit_k(search, wdl):
    """K minimizing the log loss of sigmoid(search / K) against results."""
    s = torch.from_numpy(search.astype(np.float32))
    y = torch.from_numpy(wdl.astype(np.float32))
    best = None
    for k in np.arange(300, 6001, 50):
        p = torch.sigmoid(s / float(k)).clamp(1e-6, 1 - 1e-6)
        ll = -(y * torch.log(p) + (1 - y) * torch.log(1 - p)).mean().item()
        if best is None or ll < best[1]:
            best = (float(k), ll)
    return best


def game_holdout(rec, file_index, frac=0.1):
    key = rec["game"].astype(np.uint64) * np.uint64(2654435761) + np.uint64(file_index * 97531)
    return (key % np.uint64(1000)) < np.uint64(int(frac * 1000))


def load_all(paths, diff_paths):
    parts = []
    for i, p in enumerate(paths):
        r = np.array(eval_data.load(p))
        keep = (r["flags"] & eval_data.F_SEARCH) != 0
        d = None
        if diff_paths:
            d = np.fromfile(diff_paths[i], dtype=np.int8).reshape(-1, 10)
            assert len(d) == len(r), f"{diff_paths[i]} does not match {p}"
            d = d[keep]
        hold = game_holdout(r, i)[keep]
        parts.append((r[keep], d, hold))
        print(f"{p}: {keep.sum():,} labeled rows", flush=True)
    rec = np.concatenate([x[0] for x in parts])
    diffs = np.concatenate([x[1] for x in parts]) if diff_paths else None
    hold = np.concatenate([x[2] for x in parts])
    return rec, diffs, hold


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", nargs="+", required=True, help="labeled .cfdg files")
    ap.add_argument("--hce-weights", nargs="*", default=None, help="eval_diffs sidecars, one per --data file")
    ap.add_argument("--out", required=True, help="output prefix: <out>.cfm2, <out>.macro.pt, <out>.json")
    ap.add_argument("--mode", choices=["centroid", "full"], default="centroid")
    ap.add_argument("--lam", type=float, default=0.7, help="weight of the search score (1 = search only)")
    ap.add_argument("--uttt", type=float, default=0.0, help="weight of uttt.ai's value in the target")
    ap.add_argument("--k", type=float, default=0.0, help="sigmoid scale; 0 = fit to results")
    ap.add_argument("--power", type=float, default=2.0)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=16384)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--hce-lr", type=float, default=0.5, help="Adam lr for HCE weights (engine units)")
    ap.add_argument("--hce-l2", type=float, default=1e-3, help="pull on relative HCE weight change")
    ap.add_argument("--drop-mates", action="store_true", help="drop rows whose search score is a clamped mate")
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--device", default="cpu", help="cpu or dml (DirectML GPU)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--check-shipped", action="store_true",
                    help="only compare the decoded shipped eval with the C++ static eval and exit")
    ap.add_argument("--dev-patches", default=None,
                    help="dev_patches.json the data's Dev was built with; recorded in <out>.json")
    args = ap.parse_args()
    dev_patches = json.loads(Path(args.dev_patches).read_text()) if args.dev_patches else None
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    rng = np.random.default_rng(args.seed)

    rec, diffs, hold = load_all(args.data, args.hce_weights)
    if args.drop_mates:
        keep = np.abs(rec["search"]) < 20000
        rec, hold = rec[keep], hold[keep]
        diffs = diffs[keep] if diffs is not None else None
    if args.max_rows and len(rec) > args.max_rows:
        idx = np.sort(rng.choice(len(rec), args.max_rows, replace=False))
        rec, hold = rec[idx], hold[idx]
        diffs = diffs[idx] if diffs is not None else None
    t0 = time.time()
    mini_idx, super_idx, constr, stm_sign = features(rec)
    print(f"{len(rec):,} rows, features in {time.time() - t0:.0f}s; holdout {hold.mean():.1%}", flush=True)

    if args.device == "dml":
        import torch_directml
        dev = torch_directml.device()
    else:
        dev = torch.device(args.device)
    model = Eval(load_shipped_mini(), load_shipped_macro(), args.mode, hce_train=diffs is not None).to(dev)
    T = lambda a, dt=torch.long: torch.from_numpy(np.ascontiguousarray(a)).to(dt)  # noqa: E731
    X = dict(mini=T(mini_idx), sup=T(super_idx), con=T(constr), hce=T(rec["hce"], torch.float32),
             sgn=T(stm_sign, torch.float32), diffs=T(diffs, torch.float32) if diffs is not None else None)

    def run(idx, grad=False):
        d = X["diffs"][idx].to(dev) if X["diffs"] is not None else None
        return model(X["mini"][idx].to(dev), X["sup"][idx].to(dev), X["con"][idx].to(dev), X["hce"][idx].to(dev),
                     X["sgn"][idx].to(dev), d)

    if args.check_shipped:
        with torch.no_grad():
            idx = torch.arange(min(200000, len(rec)))
            e, mini, macro = (t.cpu() for t in run(idx))  # .numpy() needs CPU tensors (--device dml)
        diff = e.numpy() - rec["static_eval"][: len(idx)]
        print(f"python E - C++ static_eval: mean {diff.mean():+.2f}, mean|d| {np.abs(diff).mean():.2f},"
              f" max|d| {np.abs(diff).max():.0f}; |mini| p99 {np.percentile(np.abs(mini.numpy()), 99):.0f},"
              f" |macro| p99 {np.percentile(np.abs(macro.numpy()), 99):.0f}")
        return

    search = rec["search"].astype(np.float32)
    valid = (rec["flags"] & eval_data.F_RESULT) != 0
    wdl = (rec["result"].astype(np.float32) + 1) / 2
    if args.k > 0:
        k = args.k
    else:
        k, ll = fit_k(search[valid & ~hold], wdl[valid & ~hold])
        k_static, ll_static = fit_k(rec["static_eval"][valid & ~hold].astype(np.float32), wdl[valid & ~hold])
        print(f"fitted K = {k:.0f} for search (log loss {ll:.4f}); static eval would fit K = {k_static:.0f}"
              f" (log loss {ll_static:.4f})", flush=True)
    lam = np.where(valid, args.lam, 1.0).astype(np.float32)
    target = lam / (1 + np.exp(-search / k)) + (1 - lam) * wdl
    has_v = (rec["flags"] & eval_data.F_UTTT_V) != 0
    if args.uttt > 0:
        mu = np.where(has_v, args.uttt, 0.0).astype(np.float32)
        target = (1 - mu) * target + mu * (rec["uttt_v"] + 1) / 2
        print(f"uttt.ai value blended into {has_v.mean():.1%} of rows at mu = {args.uttt}")
    Y = torch.from_numpy(target.astype(np.float32))
    S = torch.from_numpy(search)

    tr = np.flatnonzero(~hold)
    va = torch.from_numpy(np.flatnonzero(hold))

    def evaluate():
        model.eval()
        tot = dict(loss=0.0, mae=0.0, ll=0.0, n=0, nv=0)
        with torch.no_grad():
            for i in range(0, len(va), 65536):
                idx = va[i:i + 65536]
                e, _, _ = run(idx)
                e = e.cpu()
                p = torch.sigmoid(e / k)
                tot["loss"] += (p - Y[idx]).abs().pow(args.power).sum().item()
                tot["mae"] += (e - S[idx]).abs().clamp(max=20000).sum().item()
                v = torch.from_numpy(valid[idx.numpy()])
                y = torch.from_numpy(wdl[idx.numpy()])[v]
                pv = p[v].clamp(1e-6, 1 - 1e-6)
                tot["ll"] += -(y * torch.log(pv) + (1 - y) * torch.log(1 - pv)).sum().item()
                tot["n"] += len(idx)
                tot["nv"] += int(v.sum())
        model.train()
        return tot["loss"] / tot["n"], tot["mae"] / tot["n"], tot["ll"] / max(1, tot["nv"])

    base = evaluate()
    print(f"shipped eval on holdout: loss {base[0]:.6f}  |E-search| {base[1]:.0f}  result log loss {base[2]:.4f}",
          flush=True)
    net_params = [p for n, p in model.named_parameters() if n != "hce_w"]
    groups = [{"params": net_params, "lr": args.lr}]
    if diffs is not None:
        groups.append({"params": [model.hce_w], "lr": args.hce_lr})
    opt = torch.optim.Adam(groups)
    steps = max(1, args.epochs * math.ceil(len(tr) / args.batch))  # --epochs 0 exports the shipped eval
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[g["lr"] for g in groups], total_steps=steps, pct_start=0.05)
    best = (base[0], None, 0)
    log = [dict(epoch=0, val_loss=base[0], mae=base[1], ll=base[2])]
    n_steps = math.ceil(len(tr) / args.batch)
    print(f"training: {args.epochs} epochs x {n_steps} steps, {len(tr):,} train rows, K {k:.0f}, lam {args.lam},"
          f" uttt {args.uttt}, mode {args.mode}, hce weights {'trained' if diffs is not None else 'fixed'}", flush=True)
    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        perm = torch.from_numpy(rng.permutation(tr))
        run_loss = 0.0
        for step, i in enumerate(range(0, len(perm), args.batch)):
            if step and step % max(1, n_steps // 5) == 0:
                done = (epoch - 1) * n_steps + step
                eta = (time.time() - t_start) / done * (args.epochs * n_steps - done)
                print(f"[{time.strftime('%H:%M:%S')}] epoch {epoch} step {step}/{n_steps}"
                      f" train {run_loss / (i or 1):.6f} eta {eta / 60:.1f}m", flush=True)
            idx = perm[i:i + args.batch]
            e, _, _ = run(idx)
            loss = (torch.sigmoid(e / k) - Y[idx].to(dev)).abs().pow(args.power).mean()
            if diffs is not None:
                rel = (model.hce_w - model.hce_w0) / model.hce_w0 * model.hce_mask
                loss = loss + args.hce_l2 * (rel ** 2).sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            run_loss += loss.item() * len(idx)
        val = evaluate()
        log.append(dict(epoch=epoch, train_loss=run_loss / len(tr), val_loss=val[0], mae=val[1], ll=val[2]))
        extra = ""
        if diffs is not None:
            extra = " hce_w " + " ".join(f"{w:.0f}" for w in model.hce_w.detach().tolist())
        print(f"epoch {epoch}: train {run_loss / len(tr):.6f}  val {val[0]:.6f}  |E-search| {val[1]:.0f}"
              f"  result ll {val[2]:.4f}  ({time.time() - t0:.0f}s){extra}", flush=True)
        if val[0] < best[0]:
            best = (val[0], {n: p.detach().clone() for n, p in model.state_dict().items()}, epoch)
    if best[1] is None:
        print("no epoch beat the shipped eval on the holdout; writing the last epoch anyway")
    else:
        model.load_state_dict(best[1])
        print(f"best epoch {best[2]}: val {best[0]:.6f} (shipped {base[0]:.6f}, {100 * (1 - best[0] / base[0]):.2f}% lower)")
    export(model.cpu(), args.out, dict(k=k, args=vars(args), log=log, shipped=base, dev_patches=dev_patches))


def export(model, out, meta):
    """<out>.cfm2 (MiniNet, full 19,683-row table), <out>.macro.pt, <out>.codes.npy, <out>.json."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ext = lambda suffix: Path(str(out) + suffix)  # noqa: E731  (appends: a dotted prefix keeps its dots)
    sd = {n: p.detach().cpu().numpy() for n, p in model.named_parameters()}
    table = model.table().detach().cpu().numpy()
    # Anchor: the empty-board MiniNet output is zero (the emitter re-checks after packing).
    with torch.no_grad():
        ez = torch.zeros((1, 9), dtype=torch.long)
        cz = torch.full((1,), 9, dtype=torch.long)
        empty = float(model.mini(ez, ez, cz)[0])
        m_empty = float(model.macro_raw(ez, cz)[0])
    b2 = float(sd["b2"]) - empty
    blob = bytearray(b"CFM2") + struct.pack("<ii", 16, 8)
    for a in (table, sd["super_e"], sd["loc"], sd["constr"], sd["active"], sd["w1"], sd["b1"], sd["w2"]):
        blob += np.asarray(a, dtype="<f4").tobytes()
    blob += struct.pack("<f", b2)
    ext(".cfm2").write_bytes(bytes(blob))
    np.save(ext(".codes.npy"), model.codes.cpu().numpy().astype(np.uint8))
    np.save(ext(".cents.npy"), sd["emb"] if model.mode == "centroid" else table)
    # Macro in the emitter's parametrisation: identity hidden layer over
    # hidden-space embeddings; the emitter subtracts the empty output itself.
    torch.save({"state_dict": {
        "embedding.weight": torch.from_numpy(sd["m_emb"].reshape(36, 16).copy()),
        "constraint.weight": torch.from_numpy(sd["m_constr"].copy()),
        "hidden.weight": torch.eye(16),
        "hidden.bias": torch.from_numpy(sd["m_base"].copy()),
        "out.weight": torch.from_numpy(sd["m_out"].reshape(1, 16).copy()),
        "out.bias": torch.tensor([0.0]),
    }, "d": 16, "h": 16, "macro_empty_raw": m_empty}, ext(".macro.pt"))
    meta["hce_weights"] = [round(w) for w in model.hce_w.detach().tolist()]
    meta["mode"] = model.mode
    ext(".json").write_text(json.dumps(meta, indent=1))
    print(f"wrote {out}.cfm2/.macro.pt/.codes.npy/.cents.npy/.json; HCE weights {meta['hce_weights']}")


if __name__ == "__main__":
    main()
