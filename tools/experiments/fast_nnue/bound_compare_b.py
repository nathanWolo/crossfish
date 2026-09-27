#!/usr/bin/env python3
"""Stage 3: compare the first-layer bounds of a BGN1 net in float (x 2^qa, no rounding), to see which QA
each bound allows. quantize() in fast_nnue_b.hpp computes the chosen bound exactly on the integer rows;
this is the float prototype the stage-3 write-up quotes for the scales the loader does not pick.

  bound_compare_b.py NET.bin

Bounds: independent (stage 2), joint forced board (T[c][p] + F[p] of the same live pattern), and joint +
stone balance (the stone difference d = #mine - #theirs over the 9 boards, decided boards' hidden stones
included, is -1..0 for the side to move, 0..1 for the other, -1..1 for a stored accumulator).
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_bgn import read_bgn, live_patterns, DIGITS, LINES

path = sys.argv[1]
b = read_bgn(path)
A = b.A
live = live_patterns()
mine = (DIGITS == 1).sum(1)
theirs = (DIGITS == 2).sum(1)
d_all = mine - theirs
m1 = ((DIGITS == 1) * (1 << np.arange(9))).sum(1)
m2 = ((DIGITS == 2) * (1 << np.arange(9))).sum(1)
line1 = np.zeros(len(DIGITS), bool)
line2 = np.zeros(len(DIGITS), bool)
for ln in LINES:
    line1 |= (m1 & ln) == ln
    line2 |= (m2 & ln) == ln
full = (DIGITS != 0).all(1)
won_mine = line1 & ~line2
won_theirs = line2 & ~line1
drawn = full & ~line1 & ~line2
D = np.arange(-9, 10)
def dset(mask):
    return sorted(set(d_all[mask].tolist()))
print("live d:", dset(live), " won-by-mine d:", dset(won_mine), " won-by-theirs d:", dset(won_theirs), " drawn d:", dset(drawn))

T = b.T[:, :, :A].astype(np.float64)  # (9, NPAT, A)
F = b.F[:, :A].astype(np.float64) if b.forced else np.zeros((len(DIGITS), A))
dec = b.dec[:, :A].astype(np.float64)
bias = b.bias[:A].astype(np.float64)
ncon = b.con.shape[0]
con = b.con[:, :A].astype(np.float64)
con_side = [con[0:10], con[10:20] if ncon == 20 else con[0:10]]

NEG, POS = -1e18, 1e18
# per board, per d (index d+9), per lane: best live T; best live T+F; decided options
def per_d(vals, mask, fn, init):
    out = np.full((19, A), init)
    for d in range(-9, 10):
        sel = mask & (d_all == d)
        if sel.any():
            out[d + 9] = fn(vals[sel], axis=0)
    return out

S0 = 63  # sum offset
def dp(options_per_board, fn, init):
    # options_per_board: list of (19, A) arrays (init where impossible); returns (127, A) best by total d
    cur = np.full((127, A), init)
    cur[S0] = 0.0
    for opt in options_per_board:
        nxt = np.full((127, A), init)
        for di in range(19):
            d = di - 9
            v = opt[di]
            if np.all(v == init):
                continue
            lo, hi = max(0, -d), min(127, 127 - d)
            cand = cur[lo:hi] + v
            nxt[lo + d:hi + d] = fn(nxt[lo + d:hi + d], cand)
        cur = nxt
    return cur

for qa in (9, 10):
    s = 2.0 ** qa
    hi_opts, lo_opts, jhi_opts, jlo_opts = [], [], [], []
    ind_hi = bias.copy(); ind_lo = bias.copy()
    for m in range(9):
        th = per_d(T[m], live, np.max, NEG); tl = per_d(T[m], live, np.min, POS)
        for st, mask in ((0, won_mine), (1, won_theirs), (2, drawn)):
            for d in set(d_all[mask].tolist()):
                th[d + 9] = np.maximum(th[d + 9], dec[3 * m + st]); tl[d + 9] = np.minimum(tl[d + 9], dec[3 * m + st])
        hi_opts.append(th); lo_opts.append(tl)
        ind_hi += th.max(0); ind_lo += tl.min(0)
        TF = T[m] + F
        jhi_opts.append(per_d(TF, live, np.max, NEG)); jlo_opts.append(per_d(TF, live, np.min, POS))
    fmax = F[live].max(0); fmin = F[live].min(0)
    print(f"--- qa={qa}")
    print(f"stored, independent: [{ind_lo.min() * s:.0f}, {ind_hi.max() * s:.0f}]")
    # stored with balance: total d in {-1,0,1}
    H = dp(hi_opts, np.maximum, NEG); L = dp(lo_opts, np.minimum, POS)
    sel = [S0 - 1, S0, S0 + 1]
    print(f"stored, stone balance: [{(bias + L[sel].min(0)).min() * s:.0f}, {(bias + H[sel].max(0)).max() * s:.0f}]")
    # evaluation time
    e_ind_lo, e_ind_hi, e_j_lo, e_j_hi, e_b_lo, e_b_hi = [], [], [], [], [], []
    for side in (0, 1):
        cs = con_side[side]
        allowed = [S0 - 1, S0] if side == 0 else [S0, S0 + 1]
        for c in range(10):
            if c == 9:
                e_ind_hi.append(ind_hi + cs[9]); e_ind_lo.append(ind_lo + cs[9])
                e_j_hi.append(ind_hi + cs[9]); e_j_lo.append(ind_lo + cs[9])
                e_b_hi.append(bias + H[allowed].max(0) + cs[9]); e_b_lo.append(bias + L[allowed].min(0) + cs[9])
                continue
            e_ind_hi.append(ind_hi + cs[c] + fmax); e_ind_lo.append(ind_lo + cs[c] + fmin)
            e_j_hi.append(ind_hi - hi_opts[c].max(0) + jhi_opts[c].max(0) + cs[c])
            e_j_lo.append(ind_lo - lo_opts[c].min(0) + jlo_opts[c].min(0) + cs[c])
            oh = list(hi_opts); oh[c] = jhi_opts[c]
            ol = list(lo_opts); ol[c] = jlo_opts[c]
            Hc = dp(oh, np.maximum, NEG); Lc = dp(ol, np.minimum, POS)
            e_b_hi.append(bias + Hc[allowed].max(0) + cs[c]); e_b_lo.append(bias + Lc[allowed].min(0) + cs[c])
    for name, lo, hi in (("independent", e_ind_lo, e_ind_hi), ("joint forced board", e_j_lo, e_j_hi),
                         ("joint + stone balance", e_b_lo, e_b_hi)):
        print(f"eval time, {name}: [{np.min(lo) * s:.0f}, {np.max(hi) * s:.0f}]")
