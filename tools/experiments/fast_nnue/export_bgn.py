#!/usr/bin/env python3
"""Export a pattern-generator ("B") net (tools/experiments/nnue2/gen_nnue.py Gen) to the BGN1 file
that fast_nnue_b.hpp loads, with its first layer BAKED into float tables, and verify the file.

  export_bgn.py export NAME|CKPT.pt OUT.bin [--perm] [--stats d8_a.cfdg] [--rows N] [--no-gen]
  export_bgn.py verify OUT.bin NAME|CKPT.pt [--sample parity_in.cfdg] [--n N]
  export_bgn.py same A.bin B.bin          (two files compute the same function: tables equal up to a lane
                                           permutation; used for plain vs --perm)

NAME is a run in datasets/nnue2/probe (<NAME>.pt + <NAME>.json, whose "args" rebuild the Gen); a .pt path
takes its config from the .json next to it.

BGN1 layout (little endian; W = A + 1: A accumulator lanes, then the PSQT lane):
  "BGN1", int32 hdr[8] = A, L1, L2 (0: no second hidden layer), forced, ncon (20, or 10 when
                         shared_con), has_gen, E (embedding width), n_enc (encoder layers)
  float bias[W]
  float T[9][3^9][W]     T[m][p] = enc(onehot27(p)) @ proj_w[m] + proj_b[m]    (Gen.table())
  float dec[27][W]       row 3*m + (state - 1): mine / theirs / drawn from the perspective
  float con[ncon][W]     [c] side to move, [10 + c] the other perspective (c = 0..8 forced, 9 free)
  float F[3^9][W]        only when forced: enc(onehot27(p)) @ fwd_w + fwd_b (forced board's pattern)
  float W1[L1][2A], b1[L1]; if L2: W2[L2][L1], b2[L2]; Wo[L2 or L1], bo      (Gen.dense)
  if has_gen: int32 dims[n_enc + 1] (27, 64, 64, 32), then per encoder layer W[out][in], b[out];
              proj_w[9][E][W], proj_b[9][W]; if forced: fwd_w[E][W], fwd_b[W]
              (the generator itself: fast_nnue_b.hpp can re-bake T and F from it and time that)
Pattern index p = sum_i cell_i 3^i over the miniboard's squares i, 0 empty / 1 mine / 2 theirs from the
perspective's view (probe.Pat's index; the opponent's view is probe.Feats.pat_swap of it).
Per perspective: acc = bias + sum over live boards T[m][p_m] + sum over decided boards dec[3m + s - 1]
+ con[c or 10 + c] + (c < 9: F[p_c]); eval = 1000 * (dense(clip(acc_us[:A]), clip(acc_them[:A]))
+ (acc_us[A] - acc_them[A]) / 2) (Gen.head).

--perm reorders the A lanes (every first-layer table, the generator's projections, and the matching
W1 columns of both halves) so lanes that fire together share a (2p, 2p+1) pair of the sparse dense
kernel: a greedy maximum-weight matching on P(lane i > 0 and lane j > 0) over N rows (default 20,000)
spread over --stats (default datasets/nnue2/d8_a.cfdg, not the parity sample), both perspectives.
Same function; the quantized evals are bit-identical (every weight quantizes on its own).

verify recomputes the eval of --sample rows (default parity_in.cfdg, all 20,000) in numpy float64 from
the file's baked tables only, and compares it with PyTorch Gen.forward (float32) on the same rows:
max |d| must be below 0.05 eval units (float summation order only).
"""
import argparse
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "tools" / "experiments" / "nnue2"))
sys.path.insert(0, str(ROOT / "tools"))

N_PAT = 19683
POW3 = 3 ** np.arange(9)
DIGITS = np.array([(p // POW3) % 3 for p in range(N_PAT)])
PAT_SWAP = ((np.where(DIGITS == 1, 2, np.where(DIGITS == 2, 1, 0))) * POW3).sum(1)
LINES = [7, 56, 448, 73, 146, 292, 273, 84]


def live_patterns():
    """(3^9,) bool: no three-in-a-row for either side and not full (the patterns a live board can have)."""
    m1 = ((DIGITS == 1) * (1 << np.arange(9))).sum(1)
    m2 = ((DIGITS == 2) * (1 << np.arange(9))).sum(1)
    ok = (DIGITS == 0).any(1)
    for ln in LINES:
        ok &= ((m1 & ln) != ln) & ((m2 & ln) != ln)
    return ok


def ckpt_paths(spec):
    p = Path(spec)
    if p.suffix == ".pt":
        return p, p.with_suffix(".json")
    base = ROOT / "datasets" / "nnue2" / "probe" / spec
    return base.with_suffix(".pt"), base.with_suffix(".json")


def load_gen(spec):
    import torch
    import gen_nnue
    import probe
    pt, js = ckpt_paths(spec)
    cfg = json.loads(js.read_text())["args"]
    fx = probe.Feats(torch.device("cpu"))
    m = gen_nnue.build(fx, cfg)
    m.load_state_dict(torch.load(pt, map_location="cpu", weights_only=False))
    m.eval()
    return m, fx, cfg


# ---------------------------------------------------------------- file format
class BGN:
    """The BGN1 contents as numpy float32 arrays (W = A + 1 wide first-layer rows)."""

    def __init__(self, **kw):
        self.__dict__.update(kw)

    @property
    def W(self):
        return self.A + 1


def from_gen(m):
    import torch
    with torch.no_grad():
        h = m.enc(m.onehot)
        T = m.table().numpy()
        F = (h @ m.fwd_w + m.fwd_b).numpy() if m.forced else None
        dense = [(lin.weight.numpy().copy(), lin.bias.numpy().copy()) for lin in m.dense]
        enc = [(lay.weight.numpy().copy(), lay.bias.numpy().copy()) for lay in m.enc if hasattr(lay, "weight")]
        b = BGN(A=m.A, forced=bool(m.forced), T=T, F=F, bias=m.bias.numpy().copy(), dec=m.dec.numpy().copy(),
                con=m.con.numpy().copy(), dense=dense, enc=enc, proj_w=m.proj_w.numpy().copy(),
                proj_b=m.proj_b.numpy().copy(),
                fwd_w=m.fwd_w.numpy().copy() if m.forced else None, fwd_b=m.fwd_b.numpy().copy() if m.forced else None)
    if len(dense) not in (2, 3):
        raise SystemExit(f"dense depth {len(dense)} not supported (2A -> L1 [-> L2] -> 1)")
    return b


def write_bgn(b, path, with_gen=True):
    L1 = b.dense[0][0].shape[0]
    L2 = b.dense[1][0].shape[0] if len(b.dense) == 3 else 0
    E = b.proj_w.shape[1]
    hdr = [b.A, L1, L2, int(b.forced), b.con.shape[0], int(with_gen), E, len(b.enc)]
    f32 = lambda x: np.ascontiguousarray(x, dtype="<f4").tobytes()  # noqa: E731
    with open(path, "wb") as f:
        f.write(b"BGN1" + struct.pack("<8i", *hdr))
        f.write(f32(b.bias))
        f.write(f32(b.T))
        f.write(f32(b.dec))
        f.write(f32(b.con))
        if b.forced:
            f.write(f32(b.F))
        for w, bb in b.dense:
            f.write(f32(w))
            f.write(f32(bb))
        if with_gen:
            dims = [b.enc[0][0].shape[1]] + [w.shape[0] for w, _ in b.enc]
            f.write(struct.pack(f"<{len(dims)}i", *dims))
            for w, bb in b.enc:
                f.write(f32(w))
                f.write(f32(bb))
            f.write(f32(b.proj_w))
            f.write(f32(b.proj_b))
            if b.forced:
                f.write(f32(b.fwd_w))
                f.write(f32(b.fwd_b))


def read_bgn(path):
    raw = Path(path).read_bytes()
    if raw[:4] != b"BGN1":
        raise SystemExit(f"{path}: not a BGN1 file")
    A, L1, L2, forced, ncon, has_gen, E, n_enc = struct.unpack("<8i", raw[4:36])
    W = A + 1
    off = 36

    def take(*shape, dt="<f4"):
        nonlocal off
        n = int(np.prod(shape))
        a = np.frombuffer(raw, dtype=dt, count=n, offset=off).reshape(shape)
        off += 4 * n
        return a

    b = BGN(A=A, forced=bool(forced))
    b.bias = take(W)
    b.T = take(9, N_PAT, W)
    b.dec = take(27, W)
    b.con = take(ncon, W)
    b.F = take(N_PAT, W) if forced else None
    dims = [2 * A, L1] + ([L2] if L2 else []) + [1]
    b.dense = [(take(o, i), take(o)) for i, o in zip(dims[:-1], dims[1:])]
    b.enc = []
    if has_gen:
        ed = list(take(n_enc + 1, dt="<i4"))
        b.enc = [(take(o, i), take(o)) for i, o in zip(ed[:-1], ed[1:])]
        b.proj_w = take(9, E, W)
        b.proj_b = take(9, W)
        b.fwd_w = take(E, W) if forced else None
        b.fwd_b = take(W) if forced else None
    if off != len(raw):
        raise SystemExit(f"{path}: {len(raw) - off} trailing bytes")
    return b


def permute(b, perm):
    """Same net with accumulator lanes reordered: new lane j = old lane perm[j] (PSQT lane stays last)."""
    full = np.concatenate([perm, [b.A]])
    c = BGN(**b.__dict__)
    c.bias, c.T, c.dec, c.con = b.bias[full], b.T[..., full], b.dec[:, full], b.con[:, full]
    c.F = b.F[:, full] if b.forced else None
    w1, b1 = b.dense[0]
    c.dense = [(np.concatenate([w1[:, :b.A][:, perm], w1[:, b.A:][:, perm]], 1), b1)] + list(b.dense[1:])
    if b.enc:
        c.proj_w, c.proj_b = b.proj_w[..., full], b.proj_b[:, full]
        if b.forced:
            c.fwd_w, c.fwd_b = b.fwd_w[:, full], b.fwd_b[full]
    return c


# ---------------------------------------------------------------- numpy forward from the tables
def compact_rows(rec):
    import probe
    return probe.compact(rec)


def accumulators_np(b, xc):
    """(n, 91) compact rows -> float64 (acc_us, acc_them), each (n, W), from the baked tables only."""
    n = len(xc)
    cells = xc[:, :81].astype(np.int64).reshape(n, 9, 9)
    sup = xc[:, 81:90].astype(np.int64)
    con = xc[:, 90].astype(np.int64)
    pat = (cells * POW3).sum(2)
    out = []
    for side in (0, 1):
        p = pat if side == 0 else PAT_SWAP[pat]
        s = sup if side == 0 else np.where((sup == 1) | (sup == 2), 3 - sup, sup)
        acc = np.repeat(b.bias.astype(np.float64)[None], n, 0)
        for m in range(9):
            live = s[:, m] == 0
            row = np.where(live[:, None], b.T[m, p[:, m]], b.dec[3 * m + np.clip(s[:, m] - 1, 0, 2)])
            acc += row
        ci = con + (10 * side if b.con.shape[0] == 20 else 0)
        acc += b.con[ci]
        if b.forced:
            forced = con < 9
            pf = p[np.arange(n), np.clip(con, 0, 8)]
            acc += np.where(forced[:, None], b.F[pf], 0.0)
        out.append(acc)
    return out


def head_np(b, us, them):
    A = b.A
    x = np.clip(np.concatenate([us[:, :A], them[:, :A]], 1), 0, 1)
    for i, (w, bb) in enumerate(b.dense):
        x = x @ w.T.astype(np.float64) + bb
        if i < len(b.dense) - 1:
            x = np.clip(x, 0, 1)
    return (x[:, 0] + 0.5 * (us[:, A] - them[:, A])) * 1000.0


def forward_np(b, xc):
    return head_np(b, *accumulators_np(b, xc))


# ---------------------------------------------------------------- commands
def lane_pairs(b, xc):
    """Greedy max-weight matching of lanes on joint activation frequency (both perspectives)."""
    us, them = accumulators_np(b, xc)
    act = np.concatenate([us[:, :b.A], them[:, :b.A]], 0) > 0
    fl = act.astype(np.float64)
    J = fl.T @ fl / len(fl)
    np.fill_diagonal(J, -1.0)
    used = np.zeros(b.A, bool)
    perm = []
    for k in np.argsort(-J, axis=None):
        i, j = divmod(int(k), b.A)
        if i < j and not used[i] and not used[j]:
            used[i] = used[j] = True
            perm += [i, j]
        if len(perm) == b.A:
            break
    perm = np.array(perm)
    assert sorted(perm) == list(range(b.A))
    return perm


def nonzero_pairs(b, xc):
    us, them = accumulators_np(b, xc)
    a = np.stack([us[:, :b.A], them[:, :b.A]], 1) > 0
    return a.reshape(len(xc), 2, -1, 2).any(3).sum((1, 2)).mean(), a.sum((1, 2)).mean()


def cmd_export(a):
    import eval_data
    t0 = time.time()
    m, fx, cfg = load_gen(a.ckpt)
    b = from_gen(m)
    L = live_patterns()
    print(f"{a.ckpt}: A {b.A}, dense {[w.shape for w, _ in b.dense]}, forced {b.forced}, con rows {b.con.shape[0]}, "
          f"enc {[w.shape for w, _ in b.enc]}; baked in {time.time() - t0:.1f}s")
    print(f"  live-pattern rows: max |T| lanes {np.abs(b.T[:, L, :b.A]).max():.3f} psqt {np.abs(b.T[:, L, b.A]).max():.3f}"
          f"; all 3^9 rows: {np.abs(b.T[..., :b.A]).max():.3f}"
          + (f"; max |F| live {np.abs(b.F[L, :b.A]).max():.3f}" if b.forced else ""))
    ev = compact_rows(np.fromfile(a.sample, dtype=eval_data.REC))
    if a.perm:
        src = eval_data.load(a.stats)
        idx = np.linspace(0, len(src) - 1, a.rows).astype(np.int64)
        xs = compact_rows(np.array(src[idx]))
        perm = lane_pairs(b, xs)
        before = nonzero_pairs(b, ev)
        b2 = permute(b, perm)
        after = nonzero_pairs(b2, ev)
        d = np.abs(forward_np(b, ev) - forward_np(b2, ev)).max()
        print(f"  lanes paired on {a.rows:,} rows of {Path(a.stats).name}: {Path(a.sample).name} nonzero pairs per eval "
              f"{before[0]:.2f} -> {after[0]:.2f} of {b.A} (active lanes {before[1]:.2f} of {2 * b.A}); "
              f"float max |d| {d:.2e}")
        b = b2
    else:
        nz = nonzero_pairs(b, ev)
        print(f"  {Path(a.sample).name}: nonzero pairs per eval {nz[0]:.2f} of {b.A} (active lanes {nz[1]:.2f} of {2 * b.A})")
    write_bgn(b, a.out, with_gen=not a.no_gen)
    print(f"wrote {a.out}: {Path(a.out).stat().st_size:,} bytes ({time.time() - t0:.1f}s)")


def cmd_verify(a):
    import torch
    import eval_data
    b = read_bgn(a.bgn)
    m, fx, cfg = load_gen(a.ckpt)
    rec = np.fromfile(a.sample, dtype=eval_data.REC)[: a.n]
    xc = compact_rows(rec)
    with torch.no_grad():
        cells, sup, con = fx.split(torch.from_numpy(xc))
        ref = m(cells, sup, con).numpy().astype(np.float64)
        fast = m.fast(torch.from_numpy(xc)).numpy().astype(np.float64)
    got = forward_np(b, xc)
    d = np.abs(got - ref)
    print(f"verify {Path(a.bgn).name} vs PyTorch Gen.forward ({Path(a.ckpt).name}), {len(xc):,} rows of "
          f"{Path(a.sample).name}: max |d| {d.max():.2e} mean |d| {d.mean():.2e} (eval units; float std {ref.std():.0f}); "
          f"Gen.fast vs forward max |d| {np.abs(fast - ref).max():.2e}; trunc mismatches {(np.trunc(got) != np.trunc(ref)).sum()}")
    if b.enc:  # the generator section re-bakes the tables
        h = np.eye(27)[3 * np.arange(9)[None, :] + DIGITS].sum(1)  # (3^9, 27) one-hot
        for i, (w, bb) in enumerate(b.enc):
            h = h @ w.T.astype(np.float64) + bb
            if i < len(b.enc) - 1:
                h = np.maximum(h, 0)
        T = np.einsum("pk,lkj->lpj", h, b.proj_w) + b.proj_b[:, None, :]
        msg = f"  generator section re-baked in numpy: max |T - file T| {np.abs(T - b.T).max():.2e}"
        if b.forced:
            msg += f", max |F - file F| {np.abs(h @ b.fwd_w + b.fwd_b - b.F).max():.2e}"
        print(msg)
    if d.max() > 0.05:
        sys.exit("baked tables do not reproduce Gen.forward")


def cmd_same(a):
    x, y = read_bgn(a.a), read_bgn(a.b)
    # recover the permutation from the bias + dec rows, then compare everything
    key = lambda b: np.concatenate([b.dec[:, :b.A], b.con[:, :b.A], b.bias[None, :b.A]], 0).T  # noqa: E731
    kx, ky = key(x), key(y)
    perm = np.array([int(np.argmin(np.abs(kx - ky[j]).sum(1))) for j in range(y.A)])
    ok = sorted(perm) == list(range(x.A)) and np.array_equal(permute(x, perm).T, y.T)
    print(f"{Path(a.b).name} is {Path(a.a).name} with lanes permuted: {ok}")
    if not ok:
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("export")
    p.add_argument("ckpt")
    p.add_argument("out")
    p.add_argument("--perm", action="store_true")
    p.add_argument("--stats", default=str(ROOT / "datasets/nnue2/d8_a.cfdg"))
    p.add_argument("--rows", type=int, default=20000)
    p.add_argument("--sample", default=str(ROOT / "datasets/nnue2/fnn1/parity_in.cfdg"))
    p.add_argument("--no-gen", action="store_true")
    p.set_defaults(fn=cmd_export)
    p = sub.add_parser("verify")
    p.add_argument("bgn")
    p.add_argument("ckpt")
    p.add_argument("--sample", default=str(ROOT / "datasets/nnue2/fnn1/parity_in.cfdg"))
    p.add_argument("--n", type=int, default=20000)
    p.set_defaults(fn=cmd_verify)
    p = sub.add_parser("same")
    p.add_argument("a")
    p.add_argument("b")
    p.set_defaults(fn=cmd_same)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
