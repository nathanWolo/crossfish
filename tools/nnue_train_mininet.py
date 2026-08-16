#!/usr/bin/env python3
"""Train MiniNet: nine mini-board embeddings + a tiny MLP, in HCE units.

This is the net behind codingame_nnue.cpp (shipped as D=8, H=4 residual).
The older 199-feature dual-accumulator lives in nnue_train_sparse199.py.

Architecture (float, STM-relative)
----------------------------------
Each of 9 minis:
    v_mb = emb[ternary_idx] + super[cls] + loc[mb] + active[0|1]     # all width D
Concatenate v_0..v_8 and constr[c] → 10*D, then
    score = Linear(ReLU(Linear(x, H)), 1)

--residual (the winning recipe):  pred = static_HCE + MiniNet
    MiniNet only has to learn what depth-N search knows that HCE does not.
    Empty-board residual is shifted to 0 after training (HCE tempo 112 stays).
--arch mini without --residual:   pred = MiniNet   (replace HCE; historically lost)

Dump (NNUEWDL1, after `test_bots dump annotate`)
-----------------------------------------------
Per position: 93-byte UTTTAI state + float32 static HCE + int32 search score.
Un-annotated dumps still have float32 = game WDL in [0,1]; we refuse those.

Export
------
--arch mini  → CFM2 float blob (nnue.hpp MiniNet / nnue_emit_mininet_cg.py).
--arch sparse → quantized 199-feat net (CFN2), using nnue_train_sparse199 helpers.
"""

from __future__ import annotations

import argparse
import os
import struct
import sys

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    sys.stderr.write("PyTorch is required: pip install torch\n")
    raise

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nnue_train_sparse199 as nt  # sparse 199-feat path + dump/quant helpers

# 3^9 live-mini patterns: each of 9 squares is empty(0) / mine(1) / opp(2).
N_IDX = 19683

# Unused by MiniNet (plain ReLU). Sparse --arch still reads --crelu-max instead.
CRELU = 4.0

# HCE "two-in-a-row" detector, copied from the C++ mini LUT.
# Each pair is (mask of two cells on a line, mask of the completing cell).
# Count += 1 iff both cells of `a` are ours AND the completing cell is not theirs:
#     (popcount(ours & a) - popcount(theirs & b)) // 2
# Used to (1) bake the HCE mini LUT into v_emb, and (2) flag minis that have a
# local threat, which the freeze-LUT head treats as a "virtual" super-stone.
TIAR_PAIRS = (
    ((1 << 0) + (1 << 1), 1 << 2),
    ((1 << 1) + (1 << 2), 1 << 0),
    ((1 << 3) + (1 << 4), 1 << 5),
    ((1 << 4) + (1 << 5), 1 << 3),
    ((1 << 6) + (1 << 7), 1 << 8),
    ((1 << 7) + (1 << 8), 1 << 6),
    ((1 << 0) + (1 << 3), 1 << 6),
    ((1 << 3) + (1 << 6), 1 << 0),
    ((1 << 1) + (1 << 4), 1 << 7),
    ((1 << 4) + (1 << 7), 1 << 1),
    ((1 << 2) + (1 << 5), 1 << 8),
    ((1 << 5) + (1 << 8), 1 << 2),
    ((1 << 0) + (1 << 4), 1 << 8),
    ((1 << 4) + (1 << 8), 1 << 0),
    ((1 << 2) + (1 << 4), 1 << 6),
    ((1 << 4) + (1 << 6), 1 << 2),
    ((1 << 0) + (1 << 2), 1 << 1),
    ((1 << 3) + (1 << 5), 1 << 4),
    ((1 << 6) + (1 << 8), 1 << 7),
    ((1 << 0) + (1 << 6), 1 << 3),
    ((1 << 1) + (1 << 7), 1 << 4),
    ((1 << 2) + (1 << 8), 1 << 5),
    ((1 << 0) + (1 << 8), 1 << 4),
    ((1 << 2) + (1 << 6), 1 << 4),
)


def load_records(path: str):
    """Parse a NNUEWDL1 dump into raw state bytes plus the two score fields.

    File layout:
        8 bytes magic "NNUEWDL1"
        uint64 N
        N records of (93-byte state, float32 field, int32 field)

    Returns
    -------
    s    : (N, 93) uint8   UTTTAI state (see extract_mini for byte meanings)
    y    : (N,) float32    first score field  — static HCE after annotate,
                           or WDL in [0,1] if you forgot to annotate
    hce  : (N,) float32    second score field — search score after annotate,
                           or static HCE on a raw WDL dump

    train() immediately remaps: static_hce = y, search = hce, and aborts if
    |y| looks like WDL.
    """
    with open(path, "rb") as f:
        blob = f.read()
    if blob[:8] != b"NNUEWDL1":
        raise SystemExit(f"bad magic in {path}")
    n = struct.unpack_from("<Q", blob, 8)[0]
    rec = 93 + 4 + 4
    need = 16 + n * rec
    if len(blob) < need:
        raise SystemExit(f"truncated dump: {len(blob)} < {need}")
    s = np.empty((n, 93), dtype=np.uint8)
    y = np.zeros((n,), dtype=np.float32)
    hce = np.zeros((n,), dtype=np.float32)
    off = 16
    for i in range(n):
        s[i] = np.frombuffer(blob[off : off + 93], dtype=np.uint8)
        yi, hi = struct.unpack_from("<fi", blob, off + 93)
        y[i] = yi
        hce[i] = float(hi)
        off += rec
        if i == 0 or (i + 1) % 250000 == 0:
            print(f"records {i + 1}/{n}", flush=True)
    return s, y, hce


def extract_mini(s: np.ndarray):
    """Turn UTTTAI state bytes into MiniNet integer keys (STM-relative).

    State bytes (ASCII digits):
        [0:81]  81 local squares, row-major over minis then squares.
                Mini mb occupies bytes [9*mb, 9*mb+9). Square 0 is top-left.
                '0' empty, '1' P0, '2' P1.
        [81:90] super-cell of each mini: '0' live, '1' P0 won, '2' P1 won, '3' draw
        [90]    side to move: '1' P0, '2' P1
        [91]    constraint: '0'..'8' forced mini, '9' free move
        [92]    game result (unused here)

    Returns
    -------
    mini_idx  : (N, 9) int32  ternary pattern 0..19682, STM-relative
                              idx = Σ_s d_s * 3^s, d=0 empty / 1 mine / 2 opp
    super_idx : (N, 9) int32  0 live, 1 I won, 2 opp won, 3 draw
    constr    : (N,) int32    0..9, clipped
    """
    n = s.shape[0]
    mini_idx = np.zeros((n, 9), dtype=np.int32)
    super_idx = np.zeros((n, 9), dtype=np.int32)
    constr = (s[:, 91].astype(np.int32) - 48)  # ASCII '0'..'9'
    stm = np.where(s[:, 90] == 49, 0, 1).astype(np.int32)  # '1' → P0
    for i in range(n):
        pers = int(stm[i])
        row = s[i]
        for mb in range(9):
            idx = 0
            mul = 1  # 3^sq, square 0 least significant — matches C++ mini_index
            base = mb * 9
            for sq in range(9):
                ch = int(row[base + sq])
                if ch == 49:
                    owner = 0
                elif ch == 50:
                    owner = 1
                else:
                    mul *= 3
                    continue
                # 1 if this stone is "mine" from STM's view, else 2.
                idx += (1 if owner == pers else 2) * mul
                mul *= 3
            mini_idx[i, mb] = idx
            ch = int(row[81 + mb])
            if ch == 48:
                super_idx[i, mb] = 0  # live
            elif ch == 51:
                super_idx[i, mb] = 3  # draw
            else:
                winner = 0 if ch == 49 else 1
                super_idx[i, mb] = 1 if winner == pers else 2
        if i == 0 or (i + 1) % 250000 == 0:
            print(f"mini features {i + 1}/{n}", flush=True)
    constr = np.clip(constr, 0, 9)
    return mini_idx, super_idx, constr


class MiniNet(nn.Module):
    """Factored mini-board net. Widths D (`d`) and hidden H (`h`) are CLI flags.

    Tables, all width D except the v_* scalars:
      emb      19683 × D   live-mini stone pattern (HCE LUT analogue)
      super_e      4 × D   live / mine-win / opp-win / draw
      loc          9 × D   which super-square this mini sits on
      active       2 × D   1 iff this mini is the send-to constraint
      constr      10 × D   appended once: forced mini 0..8, or 9 = free move

    MLP:  (9+1)*D  →  ReLU(H)  →  1

    v_* / v_win_loc are an optional *additive scalar head* used only when
    --freeze-lut bakes the real HCE into those tables and freezes them.
    The shipped d8h4 residual does NOT use that head (use_hce_bake=False);
    C++ evaluate_mini only runs the MLP path.

    learn_out_bias=False (residual training) freezes l2.bias at 0. After
    training we subtract the empty-board output from bias so residual is 0.
    """

    def __init__(self, d=16, h=64, learn_out_bias=False):
        super().__init__()
        self.d = d
        self.h = h
        self.emb = nn.Embedding(N_IDX, d)
        self.super_e = nn.Embedding(4, d)
        self.loc = nn.Embedding(9, d)
        self.constr = nn.Embedding(10, d)
        self.active = nn.Embedding(2, d)
        # Scalar HCE-clone tables. Zero until bake_hce_into_mini; still written
        # to the CFM2 blob so C++ can optionally add them.
        self.v_emb = nn.Embedding(N_IDX, 1)
        self.v_super = nn.Embedding(4, 1)
        self.v_loc = nn.Embedding(9, 1)
        self.v_constr = nn.Embedding(10, 1)
        self.v_active = nn.Embedding(2, 1)
        self.v_win_loc = nn.Embedding(9, 1)  # material of a won mini by location
        self.l1 = nn.Linear(10 * d, h)
        self.l2 = nn.Linear(h, 1)
        for p in (self.emb.weight, self.super_e.weight, self.loc.weight, self.constr.weight, self.active.weight):
            nn.init.uniform_(p, -0.02, 0.02)
        for p in (self.v_emb.weight, self.v_super.weight, self.v_loc.weight, self.v_constr.weight, self.v_active.weight, self.v_win_loc.weight):
            nn.init.zeros_(p)
        nn.init.uniform_(self.l1.weight, -0.08, 0.08)
        nn.init.constant_(self.l1.bias, 0.25)  # start slightly in the ReLU linear region
        nn.init.uniform_(self.l2.weight, -0.05, 0.05)
        nn.init.zeros_(self.l2.bias)
        if not learn_out_bias:
            self.l2.bias.requires_grad = False
        # Precompute, for every ternary index, whether STM / opp has a local TIAR.
        # Used only by the freeze-LUT superboard threat term.
        hm = np.zeros((N_IDX,), dtype=np.uint8)
        ho = np.zeros((N_IDX,), dtype=np.uint8)
        for idx in range(N_IDX):
            p0 = 0
            p1 = 0
            t = idx
            for s in range(9):
                cell = t % 3
                t //= 3
                if cell == 1:
                    p0 |= 1 << s
                elif cell == 2:
                    p1 |= 1 << s
            p0_tiar = 0
            p1_tiar = 0
            for a, b in TIAR_PAIRS:
                p0_tiar += (bin(p0 & a).count("1") - bin(p1 & b).count("1")) // 2
                p1_tiar += (bin(p1 & a).count("1") - bin(p0 & b).count("1")) // 2
            hm[idx] = 1 if p0_tiar else 0
            ho[idx] = 1 if p1_tiar else 0
        self.register_buffer("has_mine_tiar", torch.from_numpy(hm))
        self.register_buffer("has_opp_tiar", torch.from_numpy(ho))
        self.use_hce_bake = False

    def forward(self, mini_idx, super_idx, constr, parts=False):
        """mini_idx/super_idx: (B, 9) long; constr: (B,) long in 0..9.

        If parts=True, also return the MLP scalar alone (for --mlp-l2, so we
        can penalize the mixer without shrinking the baked LUT).
        """
        bsz = mini_idx.size(0)
        # Location is just 0..8, broadcast over the batch. Active is 1 only
        # when constraint equals that mini AND it is not a free move (c==9).
        loc = torch.arange(9, device=mini_idx.device).view(1, 9).expand(bsz, 9)
        active = ((loc == constr.view(-1, 1)) & (constr.view(-1, 1) < 9)).long()
        # Per-mini D-vector: four tables added, not concatenated.
        v = self.emb(mini_idx) + self.super_e(super_idx) + self.loc(loc) + self.active(active)
        c = self.constr(constr)
        x = torch.cat([v.reshape(bsz, 9 * self.d), c], dim=1)  # (B, 10*D)
        h = F.relu(self.l1(x))
        mlp = self.l2(h).squeeze(-1)
        if not self.use_hce_bake:
            if parts:
                return mlp, mlp
            return mlp

        # ---- freeze-LUT additive head (not used by shipped d8h4) ----
        # Reconstruct HCE as: live-mini LUT + won-mini material by location
        # + global two-in-a-rows on the superboard (won minis and local TIARs).
        live = (super_idx == 0).to(mlp.dtype)
        mine_win = (super_idx == 1).to(mlp.dtype)
        opp_win = (super_idx == 2).to(mlp.dtype)
        win_loc = self.v_win_loc(loc).squeeze(-1)
        lut = (self.v_emb(mini_idx).squeeze(-1) * live).sum(dim=1)
        lut = lut + self.v_super(super_idx).squeeze(-1).sum(dim=1)
        lut = lut + (self.v_loc(loc).squeeze(-1) * live).sum(dim=1)
        lut = lut + (self.v_active(active).squeeze(-1) * live).sum(dim=1)
        lut = lut + (win_loc * mine_win - win_loc * opp_win).sum(dim=1)
        lut = lut + self.v_constr(constr).squeeze(-1)
        pow2 = (1 << torch.arange(9, device=mini_idx.device, dtype=torch.int32))
        mine_b = ((super_idx == 1).to(torch.int32) * pow2).sum(dim=1)  # won-mini bitboard
        opp_b = ((super_idx == 2).to(torch.int32) * pow2).sum(dim=1)
        live_b = (super_idx == 0)
        hm = self.has_mine_tiar[mini_idx.long()].bool() & live_b
        ho = self.has_opp_tiar[mini_idx.long()].bool() & live_b
        mine_t = (hm.to(torch.int32) * pow2).sum(dim=1)  # live minis with a local TIAR
        opp_t = (ho.to(torch.int32) * pow2).sum(dim=1)

        def popc(bb, mask):
            v = bb & mask
            c = torch.zeros_like(v)
            for bit in range(9):
                c = c + ((v >> bit) & 1)
            return c

        # 1316 = super two-in-a-row, 424 = two-in-a-rows-lined-up (HCE globals).
        g = torch.zeros(bsz, device=mini_idx.device, dtype=mlp.dtype)
        for a, b in TIAR_PAIRS:
            p0g = torch.div(popc(mine_b, a) - popc(opp_b, b), 2, rounding_mode="trunc")
            p1g = torch.div(popc(opp_b, a) - popc(mine_b, b), 2, rounding_mode="trunc")
            p0l = torch.div(popc(mine_t | mine_b, a) - popc(opp_b, b), 2, rounding_mode="trunc")
            p1l = torch.div(popc(opp_t | opp_b, a) - popc(mine_b, b), 2, rounding_mode="trunc")
            g = g + 1316.0 * (p0g - p1g).to(mlp.dtype) + 424.0 * (p0l - p1l).to(mlp.dtype)
        lut = lut + g
        out = mlp + lut
        if parts:
            return out, mlp
        return out


def pack_cfn2(w0, b0, w1, b1, w2, b2, scale, crelu_max, asinh_s) -> bytes:
    """Binary for the *sparse* 199-feat net (magic CFN2). MiniNet uses CFM2 instead."""
    buf = bytearray()
    buf += b"CFN2"
    buf += struct.pack("<iiii", int(scale), int(b2), int(crelu_max), int(asinh_s))
    buf += w0.astype("<i2").tobytes()
    buf += b0.astype("<i2").tobytes()
    buf += w1.astype("i1").tobytes()
    buf += b1.astype("<i4").tobytes()
    buf += w2.astype("i1").tobytes()
    return bytes(buf)


def write_mini_bin(path, model: MiniNet, bake=False):
    """Write a CFM2 blob that C++ MiniNet::load / nnue_emit_mininet_cg.py consume.

    Layout (little-endian):
        "CFM2"  + int32 D + int32 H
        emb[N_IDX*D], super[4*D], loc[9*D], constr[10*D], active[2*D]   float32
        l1.weight[H, 10D], l1.bias[H], l2.weight[H], l2.bias             float32
        v_emb[N_IDX], v_super[4], v_loc[9], v_constr[10], v_active[2],
        v_win_loc[9]                                                     float32
        int32 bake_flag
    """
    d, h = model.d, model.h
    with open(path, "wb") as f:
        f.write(b"CFM2")
        f.write(struct.pack("<ii", d, h))
        def wr(t):
            f.write(t.detach().cpu().contiguous().numpy().astype("<f4").tobytes())
        wr(model.emb.weight)
        wr(model.super_e.weight)
        wr(model.loc.weight)
        wr(model.constr.weight)
        wr(model.active.weight)
        wr(model.l1.weight)
        wr(model.l1.bias)
        wr(model.l2.weight.reshape(-1))
        f.write(struct.pack("<f", float(model.l2.bias.detach().cpu().reshape(-1)[0])))
        wr(model.v_emb.weight.reshape(-1))
        wr(model.v_super.weight.reshape(-1))
        wr(model.v_loc.weight.reshape(-1))
        wr(model.v_constr.weight.reshape(-1))
        wr(model.v_active.weight.reshape(-1))
        wr(model.v_win_loc.weight.reshape(-1))
        f.write(struct.pack("<i", 1 if bake else 0))
    print(f"wrote {path} bake={int(bake)}", flush=True)


def hce_mini_scores():
    """Scalar HCE for every ternary mini index, STM-positive, matching C++ MiniLut.

    Terms (same coefficients as evaluate_hce's per-mini loop):
        534 * (mine TIAR − opp TIAR)
         33 * (center held)
         10 * (corners held)     # PAWN
         33 * (stones held)
    Superboard globals and tempo are NOT in this table; bake_hce_into_mini
    puts won-mini material in v_win_loc / v_super and tempo in l2.bias.
    """
    corners = (1 << 0) + (1 << 2) + (1 << 6) + (1 << 8)
    scores = np.zeros((N_IDX,), dtype=np.float32)
    for idx in range(N_IDX):
        p0 = 0
        p1 = 0
        t = idx
        for s in range(9):
            cell = t % 3
            t //= 3
            if cell == 1:
                p0 |= 1 << s
            elif cell == 2:
                p1 |= 1 << s
        p0_tiar = 0
        p1_tiar = 0
        for a, b in TIAR_PAIRS:
            p0_tiar += (bin(p0 & a).count("1") - bin(p1 & b).count("1")) // 2
            p1_tiar += (bin(p1 & a).count("1") - bin(p0 & b).count("1")) // 2
        scores[idx] = (
            534 * (p0_tiar - p1_tiar)
            + 33 * (((p0 >> 4) & 1) - ((p1 >> 4) & 1))
            + 10 * (bin(p0 & corners).count("1") - bin(p1 & corners).count("1"))
            + 33 * (bin(p0).count("1") - bin(p1).count("1"))
        )
    return scores


def bake_hce_into_mini(model: MiniNet):
    """Copy the linear HCE into the additive v_* head and freeze those tables.

    The MLP starts near-zero so the net is "HCE + tiny mixer". --freeze-lut
    replace nets still lost SPRT; the shipped residual leaves this off and
    lets C++ add real evaluate_hce instead.
    """
    sc = hce_mini_scores()
    with torch.no_grad():
        model.v_emb.weight[:, 0] = torch.from_numpy(sc)
        model.v_super.weight.zero_()
        model.v_super.weight[1, 0] = 2410.0   # I won this mini (PAWN * ~241)
        model.v_super.weight[2, 0] = -2410.0
        model.v_loc.weight.zero_()
        model.v_constr.weight.zero_()
        model.v_active.weight.zero_()
        model.v_win_loc.weight.zero_()
        # Won-mini location: corners 464, center 836, edges 0 (HCE material).
        model.v_win_loc.weight[:, 0] = torch.tensor(
            [464.0, 0.0, 464.0, 0.0, 836.0, 0.0, 464.0, 0.0, 464.0]
        )
        model.l2.bias.zero_()
        model.l2.bias[0] = 112.0  # HCE tempo; replace-net empty eval target
        model.l2.weight.zero_()
        model.l1.weight.mul_(0.05)
        model.l1.bias.zero_()
    model.v_emb.weight.requires_grad = False
    model.v_super.weight.requires_grad = False
    model.v_win_loc.weight.requires_grad = False
    model.use_hce_bake = True
    print("baked frozen HCE LUT + won-mini material into additive head", flush=True)


def empty_mini_tensors(device):
    """One empty UTTTAI position: all zeros, P0 to move, constraint 9 (free)."""
    s = np.full((1, 93), 48, dtype=np.uint8)  # ASCII '0'
    s[0, 90] = 49       # STM = P0
    s[0, 91] = 48 + 9   # free move
    s[0, 92] = 48
    mini_idx, super_idx, constr = extract_mini(s)
    return (
        torch.from_numpy(mini_idx).long().to(device),
        torch.from_numpy(super_idx).long().to(device),
        torch.from_numpy(constr).long().to(device),
    )


def train(args):
    s, y, search = load_records(args.data)
    # After annotate, y is static HCE (thousands). Raw WDL dumps have |y|~0.5.
    y_abs = float(np.mean(np.abs(y)))
    if y_abs < 2.0:
        raise SystemExit(
            f"{args.data} float field looks like WDL (mean|y|={y_abs:.3f}). "
            "Run: test_bots dump annotate in.bin out.bin"
        )
    static_hce = y.astype(np.float32)
    print(
        f"loaded {len(search)} pos mean|search|={float(np.mean(np.abs(search))):.1f} "
        f"mean|hce|={float(np.mean(np.abs(static_hce))):.1f} "
        f"search_vs_hce_corr={float(np.corrcoef(search, static_hce)[0, 1]):.3f}",
        flush=True,
    )

    # Mate scores dominate Huber; clip (winning recipe) or drop them.
    keep = np.abs(search) < args.mate_clip
    if args.drop_mates:
        n_drop = int((~keep).sum())
        s, static_hce, search = s[keep], static_hce[keep], search[keep]
        print(f"dropped {n_drop} |search|>={args.mate_clip} kept={len(search)}", flush=True)
    else:
        search = np.clip(search, -args.mate_clip, args.mate_clip)
        static_hce = np.clip(static_hce, -args.mate_clip, args.mate_clip)
        print(f"clipped labels to ±{args.mate_clip}", flush=True)

    resid = search - static_hce
    print(
        f"residual mean={float(resid.mean()):.1f} std={float(resid.std()):.1f} "
        f"median|r|={float(np.median(np.abs(resid))):.1f}",
        flush=True,
    )

    # Duplicate positions where |search-HCE| is large. Helped val corr, leaked
    # into the holdout and did not help SPRT; winning d8h4 run used upsample=0.
    if args.upsample > 0 and args.upsample_times > 1:
        hot = np.abs(resid) >= args.upsample
        extra = np.where(hot)[0]
        if len(extra):
            reps = np.repeat(extra, args.upsample_times - 1)
            s = np.concatenate([s, s[reps]], axis=0)
            static_hce = np.concatenate([static_hce, static_hce[reps]])
            search = np.concatenate([search, search[reps]])
            print(f"upsampled |r|>={args.upsample}: +{len(reps)} now {len(search)}", flush=True)

    if args.label == "asinh":
        # Compress mate tails so the last layer does not need huge weights.
        tgt = np.arcsinh(search / args.asinh_s).astype(np.float32)
        hce_t = np.arcsinh(static_hce / args.asinh_s).astype(np.float32)
        print(f"asinh S={args.asinh_s} tgt std={float(tgt.std()):.3f}", flush=True)
    else:
        tgt = search.astype(np.float32)
        hce_t = static_hce.astype(np.float32)

    rng = np.random.default_rng(42)
    perm = rng.permutation(len(tgt))
    split = int(len(tgt) * 0.9)
    tr, va = perm[:split], perm[split:]
    # The number to beat: how well static HCE already matches the search teacher.
    hce_va_mae = float(np.mean(np.abs(search[va] - static_hce[va])))
    hce_va_corr = float(np.corrcoef(search[va], static_hce[va])[0, 1])
    print(f"val HCE-only vs search mae={hce_va_mae:.1f} corr={hce_va_corr:.3f}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} arch={args.arch} residual={int(args.residual)}", flush=True)

    tag = f"n{len(s)}_m{int(args.mate_clip)}_u{int(args.upsample)}_{int(args.drop_mates)}_{args.upsample_times}"
    if args.arch == "mini":
        # Residual: freeze output bias (shifted to 0 after fit). Replace: learn it.
        model = MiniNet(d=args.mini_d, h=args.hidden, learn_out_bias=not args.residual).to(device)
        if args.freeze_lut:
            bake_hce_into_mini(model)
        cache = args.data + f".mini{args.mini_d}.{tag}.npz"
        if os.path.isfile(cache):
            z = np.load(cache)
            mini_idx, super_idx, constr = z["mini"], z["super"], z["constr"]
            print(f"loaded cache {cache}", flush=True)
        else:
            mini_idx, super_idx, constr = extract_mini(s)
            np.savez(cache, mini=mini_idx, super=super_idx, constr=constr)
            print(f"wrote cache {cache}", flush=True)
        feat = (
            torch.from_numpy(mini_idx),
            torch.from_numpy(super_idx),
            torch.from_numpy(constr),
        )

        def forward_idx(idx, parts=False):
            return model(
                feat[0][idx].long().to(device),
                feat[1][idx].long().to(device),
                feat[2][idx].long().to(device),
                parts=parts,
            )
    else:
        # 199-feature dual-accumulator, same QAT net as nnue_train_sparse199.
        cache = args.data + f".sparse.{tag}.npz"
        if os.path.isfile(cache):
            z = np.load(cache)
            stm_idx, nsm_idx = z["stm"], z["nsm"]
            print(f"loaded cache {cache}", flush=True)
        else:
            stm_idx = np.full((len(s), nt.MAX_FEAT), nt.PAD, dtype=np.int32)
            nsm_idx = np.full((len(s), nt.MAX_FEAT), nt.PAD, dtype=np.int32)
            for i in range(len(s)):
                raw = s[i].tobytes()
                pers = 0 if s[i, 90] == 49 else 1
                nt.fill_feats_into(raw, pers, stm_idx[i])
                nt.fill_feats_into(raw, pers ^ 1, nsm_idx[i])
                if i == 0 or (i + 1) % 250000 == 0:
                    print(f"sparse features {i + 1}/{len(s)}", flush=True)
            np.savez(cache, stm=stm_idx, nsm=nsm_idx)
            print(f"wrote cache {cache}", flush=True)
        model = nt.Nnue(use_scale=True, crelu_max=args.crelu_max, init_scale=20.0 if args.residual else 80.0).to(device)
        stm_t = torch.from_numpy(stm_idx)
        nsm_t = torch.from_numpy(nsm_idx)

        def forward_idx(idx):
            return model(stm_t[idx].long().to(device), nsm_t[idx].long().to(device))

    tgt_t = torch.from_numpy(tgt)
    hce_tt = torch.from_numpy(hce_t)
    opt = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.wd
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs))

    def run_epoch(indices, train_mode):
        if train_mode:
            model.train()
            rng.shuffle(indices)
        else:
            model.eval()
        total = 0.0
        seen = 0
        ctx = torch.enable_grad() if train_mode else torch.no_grad()
        with ctx:
            for start in range(0, len(indices), args.batch):
                sl = indices[start : start + args.batch]
                mlp = None
                if args.arch == "mini" and args.mlp_l2 > 0:
                    net, mlp = forward_idx(sl, parts=True)
                else:
                    net = forward_idx(sl)
                batch = tgt_t[sl].to(device)
                if args.residual:
                    # Teacher is search; net is the gap HCE missed.
                    pred = hce_tt[sl].to(device) + net
                    loss = F.huber_loss(pred, batch, delta=args.huber)
                    if args.res_l2 > 0:
                        # Shrink the residual so we do not unlearn HCE.
                        loss = loss + args.res_l2 * (net * net).mean()
                else:
                    loss = F.huber_loss(net, batch, delta=args.huber)
                    if args.hce_anchor > 0:
                        # Replace-only: stay close to HCE while fitting search.
                        loss = loss + args.hce_anchor * F.huber_loss(
                            net, hce_tt[sl].to(device), delta=args.huber
                        )
                if mlp is not None:
                    loss = loss + args.mlp_l2 * (mlp * mlp).mean()
                if train_mode:
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    if args.arch == "sparse":
                        with torch.no_grad():
                            model.emb.weight[nt.PAD].zero_()
                            model.l2.weight.clamp_(-127.0 / nt.QB, 127.0 / nt.QB)
                            model.l3.weight.clamp_(-127.0 / nt.QB, 127.0 / nt.QB)
                total += float(loss.item()) * len(sl)
                seen += len(sl)
        if train_mode:
            sched.step()
        return total / max(1, seen)

    best_val = 1e9
    best_state = None
    bad = 0
    init_va = run_epoch(va.copy(), False)
    print(f"init val={init_va:.6f}", flush=True)
    warmup = max(0, int(args.hce_warmup))
    if warmup > 0 and not args.residual:
        # Replace-only: first clone HCE, then switch the target to search.
        print(f"HCE warmup epochs={warmup}", flush=True)
        saved_tgt = tgt_t
        tgt_t = hce_tt
        for epoch in range(warmup):
            tr_loss = run_epoch(tr.copy(), True)
            va_loss = run_epoch(va.copy(), False)
            print(f"warmup {epoch:03d} train={tr_loss:.6f} val={va_loss:.6f}", flush=True)
        tgt_t = saved_tgt
        opt = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.wd
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs))
        print("warmup done, training on search labels", flush=True)

    for epoch in range(args.epochs):
        tr_loss = run_epoch(tr.copy(), True)
        va_loss = run_epoch(va.copy(), False)
        print(f"epoch {epoch:03d} train={tr_loss:.6f} val={va_loss:.6f}", flush=True)
        if (epoch % 2) == 0:
            model.eval()
            with torch.no_grad():
                probe = forward_idx(va[: min(1024, len(va))]).cpu()
            print(
                f"  net mean={float(probe.mean()):.2f} std={float(probe.std()):.2f} "
                f"median|net|={float(probe.abs().median()):.2f}",
                flush=True,
            )
        if va_loss + 1e-6 < best_val:
            best_val = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                print("early stop", flush=True)
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    # Holdout in HCE/search units (undo asinh if needed) vs the HCE baseline.
    model.eval()
    with torch.no_grad():
        sl = va[: min(4096, len(va))]
        net = forward_idx(sl).cpu().numpy()
    if args.label == "asinh":
        if args.residual:
            pred = args.asinh_s * np.sinh(np.arcsinh(static_hce[sl] / args.asinh_s) + net)
        else:
            pred = args.asinh_s * np.sinh(net)
    else:
        pred = (static_hce[sl] + net) if args.residual else net
    t = search[sl]
    err = pred - t
    corr = float(np.corrcoef(pred, t)[0, 1]) if t.std() > 0 else 0.0
    print(
        f"float val n={len(sl)} mae={float(np.mean(np.abs(err))):.1f} "
        f"median|e|={float(np.median(np.abs(err))):.1f} corr={corr:.3f} "
        f"HCE mae={float(np.mean(np.abs(static_hce[sl] - t))):.1f} "
        f"HCE corr={float(np.corrcoef(static_hce[sl], t)[0, 1]):.3f} "
        f"net mean={float(net.mean()):.1f} median|net|={float(np.median(np.abs(net))):.1f}",
        flush=True,
    )
    if corr <= hce_va_corr + 0.005 and args.residual:
        print("WARNING: net did not beat HCE correlation on this holdout", flush=True)
    if not args.residual and corr < 0.97:
        print("WARNING: replace net is a weak clone of the teacher (corr < 0.97)", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    if args.arch == "mini":
        with torch.no_grad():
            e0, e1, e2 = empty_mini_tensors(device)
            empty_v = float(model(e0, e1, e2).cpu().numpy()[0])
        if args.residual:
            # C++ does evaluate_hce + MiniNet; empty HCE is already tempo 112.
            model.l2.bias.data -= empty_v
            print(f"shifted mini b2 so empty residual {empty_v:.1f}->0", flush=True)
        else:
            model.l2.bias.data += 112.0 - empty_v
            print(f"shifted mini b2 so empty {empty_v:.1f}->112 (HCE tempo)", flush=True)
        write_mini_bin(args.out, model, bake=args.freeze_lut)
        return

    # Sparse 199-feat export (int16/int8), same as nnue_train_sparse199.
    w0, b0, w1, b1, w2, b2 = nt.quantize(model)
    crelu_imax = max(1, int(round(args.crelu_max * nt.QA)))
    if model.out_scale is not None:
        scale = max(1, int(round(float(model.out_scale.detach().abs()))))
    else:
        scale = nt.pin_scale(w0, b0, w1, b1, w2, b2, stm_idx, nsm_idx, search, crelu_max=crelu_imax)
    empty = b"0" * 90 + b"190"
    empty_stm = np.array([nt.pad_feats(nt.features_for_pers(empty, 0))], dtype=np.int32)
    empty_nsm = np.array([nt.pad_feats(nt.features_for_pers(empty, 1))], dtype=np.int32)
    empty_eval = int(
        nt.quant_forward_batch(
            w0, b0, w1, b1, w2, b2, empty_stm, empty_nsm, scale, crelu_max=crelu_imax
        )[0]
    )
    if args.residual:
        delta = int(round(empty_eval * (nt.QB * nt.QA) / max(1, scale)))
        b2 -= delta
        empty_eval = int(
            nt.quant_forward_batch(
                w0, b0, w1, b1, w2, b2, empty_stm, empty_nsm, scale, crelu_max=crelu_imax
            )[0]
        )
        print(f"empty residual -> {empty_eval} scale={scale} crelu={crelu_imax}", flush=True)
    nt.report_quant_sat(w1, w2)
    nt.report_quant_fit(
        w0, b0, w1, b1, w2, b2, scale, stm_idx[va], nsm_idx[va], search[va], crelu_max=crelu_imax
    )
    inc_path = os.path.splitext(args.out)[0] + ".inc"
    nt.write_inc(inc_path, w0, b0, w1, b1, w2, b2, scale, crelu_max=crelu_imax, asinh_s=0)
    blob = pack_cfn2(w0, b0, w1, b1, w2, b2, scale, crelu_imax, 0)
    with open(args.out, "wb") as f:
        f.write(blob)
    print(f"wrote {inc_path}")
    print(f"wrote {args.out} ({len(blob)} bytes)")


def main():
    ap = argparse.ArgumentParser(
        description="Train MiniNet (default) or the sparse 199-feature NNUE on search-score dumps."
    )
    ap.add_argument("--data", required=True, help="NNUEWDL1 dump, annotated (float=HCE, int32=search)")
    ap.add_argument("--out", required=True, help="CFM2 .bin (mini) or CFN2 .bin (sparse)")
    ap.add_argument("--arch", choices=["sparse", "mini"], default="mini",
                    help="mini = MiniNet (shipped); sparse = 199-feature dual-acc")
    ap.add_argument("--residual", action="store_true",
                    help="train HCE+net vs search; C++ eval = HCE + MiniNet")
    ap.add_argument("--label", choices=["linear", "asinh"], default="linear",
                    help="linear = Huber in HCE units; asinh compresses mate tails")
    ap.add_argument("--asinh-s", type=float, default=1000.0, help="S in asinh(score/S)")
    ap.add_argument("--mate-clip", type=float, default=8000.0,
                    help="clip (or drop) |search| above this before fitting")
    ap.add_argument("--drop-mates", action="store_true",
                    help="remove |search|>=mate-clip rows instead of clipping")
    ap.add_argument("--upsample", type=float, default=400.0,
                    help="duplicate rows with |search-HCE| >= this; 0 disables. Winning run: 0")
    ap.add_argument("--upsample-times", type=int, default=2,
                    help="total copies of each upsampled row (2 = one extra)")
    ap.add_argument("--res-l2", type=float, default=0.0, help="L2 on the residual output")
    ap.add_argument("--crelu-max", type=float, default=4.0, help="sparse arch CReLU clip only")
    ap.add_argument("--huber", type=float, default=1500.0, help="Huber delta in label units")
    ap.add_argument("--mini-d", type=int, default=32, help="embedding width D (shipped=8)")
    ap.add_argument("--hidden", type=int, default=128, help="MLP hidden H (shipped=4)")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--wd", type=float, default=0.0, help="Adam weight decay")
    ap.add_argument("--patience", type=int, default=8, help="early-stop epochs without val gain")
    ap.add_argument("--hce-anchor", type=float, default=0.0,
                    help="replace only: extra Huber(net, HCE) weight")
    ap.add_argument("--hce-warmup", type=int, default=0,
                    help="replace only: epochs fitting HCE before search labels")
    ap.add_argument("--freeze-lut", action="store_true",
                    help="bake frozen HCE mini LUT + won-board material into additive head")
    ap.add_argument("--mlp-l2", type=float, default=0.0,
                    help="L2 on the MLP head so the additive LUT stays in charge")
    train(ap.parse_args())


if __name__ == "__main__":
    main()
