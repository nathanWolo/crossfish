#!/usr/bin/env python3
"""Train the classic STM NNUE (199 sparse features) and export quantized weights.

This is NOT MiniNet (codingame_nnue.cpp / nnue_train_mininet.py). MiniNet uses
mini-index embeddings (D=8, H=4). This script trains the Stockfish-style net:

    199 features → int16 accumulator (H=32) → CReLU → concat STM+NTM
                 → int8 hidden (L2=32) → CReLU → int8 → 1

C++ eval (see NnueNet::evaluate) is roughly:

    out * NNUE_SCALE / (QB * QA)

so a raw last-layer of int8 can only move ~±64 around the bias unless we
learn an output scale or apply asinh.

Targets
-------
--target hce : regress the dump's int32 field (static HCE, or a fixed-depth
               search score written into that slot). Huber loss in score units,
               or asinh(score/S) if --label asinh.
--target wdl : sigmoid(pred/K) vs the dump's float WDL in [0, 1].
--residual   : WDL only. pred = HCE + net, then sigmoid. C++ eval = HCE + net.
               Empty-board residual is later shifted to 0 so tempo stays 112.

Dump format (NNUEWDL1)
----------------------
16-byte header: magic + uint64 count. Then per position:
    93-byte UTTTAI state + float32 WDL + int32 HCE/search score.
State bytes: [0:81] squares ('0' empty, '1' P0, '2' P1),
             [81:90] super cells ('0' live, '1' P0, '2' P1, '3' draw),
             [90] STM ('1' P0, '2' P1), [91] constraint '0'..'9', [92] result.
"""

from __future__ import annotations

import argparse
import base64
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

# ---------------------------------------------------------------------------
# Feature layout (must match cpp_impl/nnue.hpp NnueNet)
#
#   squares:  81 cells × 2 (mine vs opp from a chosen perspective) = 162
#             feature id = (mini*9 + sq)*2 + (0 if owner==pers else 1)
#   super:    9 minis × 3 classes (mine-win, opp-win, draw)         = 27
#             feature id = SUPER_BASE + mb*3 + cls
#   constr:   10 keys (forced mini 0..8, or 9 = free move)          = 10
#             feature id = CONSTR_BASE + c
#   total:    199 real features, plus a dummy PAD slot used to fill
#             unused entries in the dense index tensor (embedding row 199
#             is kept at 0 so summing pads is a no-op).
# ---------------------------------------------------------------------------
N_FEAT = 199
H = 32          # first-layer / accumulator width
L2 = 32         # second hidden width
QA = 127        # first-layer float→int16 scale (CReLU clip 1.0 → 127)
QB = 64         # later-layer float→int8 scale (weight 1.0 → 64, clip ±127)
SUPER_BASE = 162
CONSTR_BASE = 189
MAX_FEAT = 96   # dense padded length: 81 stones + 9 super + 1 constr, worst case
PAD = 199       # embedding index that means "no feature"; weight always 0


def fill_feats_into(s: bytes, pers: int, dest):
    """Write sparse feature ids for one perspective into dest[0:MAX_FEAT].

    pers=0 means “mine = P0”. pers=1 means the board is flipped so P1 is mine.
    dest is a length-MAX_FEAT int32 buffer; unused tail is filled with PAD.
    """
    n = 0
    # 81 local squares. ASCII '1' (49) = P0, '2' (50) = P1, else empty.
    for i in range(81):
        ch = s[i]
        if ch == 49:
            owner = 0
        elif ch == 50:
            owner = 1
        else:
            continue
        # Even id = mine occupies this cell; odd id = opponent occupies it.
        dest[n] = i * 2 + (0 if owner == pers else 1)
        n += 1
    # Superboard: which minis are already decided. '3' (51) = draw.
    for mb in range(9):
        ch = s[81 + mb]
        if ch == 49:
            winner = 0
        elif ch == 50:
            winner = 1
        elif ch == 51:
            winner = 2
        else:
            continue
        # cls 0 = I won it, 1 = opp won it, 2 = draw. Live minis are omitted.
        cls = 2 if winner == 2 else (0 if winner == pers else 1)
        dest[n] = SUPER_BASE + mb * 3 + cls
        n += 1
    # Constraint: last-move send-to mini, or 9 if the opponent may play anywhere.
    c = s[91] - 48  # ASCII '0'..'9' → 0..9
    if 0 <= c <= 9:
        dest[n] = CONSTR_BASE + c
        n += 1
    dest[n:] = PAD


def features_for_pers(s: bytes, pers: int) -> list[int]:
    """Same as fill_feats_into, returning a Python list (used for the empty board)."""
    dest = np.full((MAX_FEAT,), PAD, dtype=np.int32)
    fill_feats_into(s, pers, dest)
    return dest.tolist()


def pad_feats(feats: list[int]) -> list[int]:
    """Truncate or PAD-extend a feature list to exactly MAX_FEAT slots."""
    out = feats[:MAX_FEAT] + [PAD] * (MAX_FEAT - min(len(feats), MAX_FEAT))
    return out


def load_dump(path: str):
    """Load a NNUEWDL1 dump, with a .npz cache of already-expanded feature indices.

    Returns
    -------
    stm_idx : (N, MAX_FEAT) int32  feature ids from the side-to-move's view
    nsm_idx : (N, MAX_FEAT) int32  same position from the not-side-to-move's view
    y       : (N,) float32         game WDL in [0, 1] (STM win = 1)
    hce     : (N,) float32         int32 score field (HCE or search), STM-positive
    """
    cache = path + ".npz"
    if os.path.isfile(cache) and os.path.getmtime(cache) >= os.path.getmtime(path):
        z = np.load(cache)
        print(f"loaded cache {cache}", flush=True)
        return z["stm_idx"], z["nsm_idx"], z["y"], z["hce"]
    with open(path, "rb") as f:
        blob = f.read()
    if blob[:8] != b"NNUEWDL1":
        raise SystemExit(f"bad magic in {path}")
    n = struct.unpack_from("<Q", blob, 8)[0]
    rec = 93 + 4 + 4  # state + float WDL + int32 score
    need = 16 + n * rec
    if len(blob) < need:
        raise SystemExit(f"truncated dump: {len(blob)} < {need}")
    stm_idx = np.full((n, MAX_FEAT), PAD, dtype=np.int32)
    nsm_idx = np.full((n, MAX_FEAT), PAD, dtype=np.int32)
    y = np.zeros((n,), dtype=np.float32)
    hce = np.zeros((n,), dtype=np.float32)
    off = 16
    for i in range(n):
        s = blob[off : off + 93]
        yi, hi = struct.unpack_from("<fi", blob, off + 93)
        # Byte 90: '1' → P0 to move (pers 0), else P1 (pers 1).
        stm = 0 if s[90] == 49 else 1
        fill_feats_into(s, stm, stm_idx[i])
        fill_feats_into(s, stm ^ 1, nsm_idx[i])  # flipped perspective for the dual acc
        y[i] = yi
        hce[i] = float(hi)
        off += rec
        if i == 0 or (i + 1) % 250000 == 0:
            print(f"features {i + 1}/{n}", flush=True)
    np.savez(cache, stm_idx=stm_idx, nsm_idx=nsm_idx, y=y, hce=hce)
    print(f"wrote cache {cache}", flush=True)
    return stm_idx, nsm_idx, y, hce


def _ste_quant(w, scale, qmin, qmax):
    """Fake-quantize `w` for QAT, with a straight-through estimator.

    Forward:  clamp(round(w * scale), qmin, qmax) / scale
    Backward: identity through `w` (the (q-w) term is detached).

    So the optimizer sees float weights, but the forward pass looks like the
    int16/int8 inference the C++ engine will actually run.
    """
    q = torch.clamp(torch.round(w * scale), qmin, qmax) / scale
    return w + (q - w).detach()


class Nnue(nn.Module):
    """Dual-accumulator NNUE matching the C++ topology.

    Forward:
      acc_stm = sum_i emb[feat_i] + b0     # H floats, QAT as int16
      acc_ntm = same from the other view
      h0, h1  = CReLU(acc)                 # clip to [0, crelu_max]
      h2      = CReLU(W1 @ cat(h0,h1) + b1)
      raw     = W2 @ h2 + b2
      if use_scale: return raw * |out_scale|   # maps into HCE units
    """

    def __init__(self, use_scale=False, crelu_max=1.0, init_scale=80.0):
        super().__init__()
        self.crelu_max = float(crelu_max)
        # PAD+1 rows so index 199 is a real (zeroed) padding vector.
        self.emb = nn.Embedding(PAD + 1, H, padding_idx=PAD)
        self.b0 = nn.Parameter(torch.zeros(H))
        self.l2 = nn.Linear(2 * H, L2)  # 64 → 32
        self.l3 = nn.Linear(L2, 1)      # 32 → 1
        # Learned multiplier so int8 last-layer logits can span HCE magnitudes.
        self.out_scale = nn.Parameter(torch.tensor(float(init_scale))) if use_scale else None
        nn.init.uniform_(self.emb.weight, -0.05, 0.05)
        nn.init.uniform_(self.l2.weight, -0.05, 0.05)
        nn.init.zeros_(self.l2.bias)
        nn.init.uniform_(self.l3.weight, -0.05, 0.05)
        nn.init.zeros_(self.l3.bias)
        with torch.no_grad():
            self.emb.weight[PAD].zero_()

    def forward(self, stm_idx, nsm_idx):
        # Feature transformer: shared embedding, two accumulators.
        ew = _ste_quant(self.emb.weight, float(QA), -32767, 32767)
        b0 = _ste_quant(self.b0, float(QA), -32767, 32767)
        a0 = F.embedding(stm_idx, ew, padding_idx=PAD).sum(dim=1) + b0
        a1 = F.embedding(nsm_idx, ew, padding_idx=PAD).sum(dim=1) + b0
        m = self.crelu_max
        h0 = torch.clamp(a0, 0.0, m)
        h1 = torch.clamp(a1, 0.0, m)
        h = torch.cat([h0, h1], dim=1)
        # Hidden + output: QAT as int8 weights. Biases stay float here;
        # quantize() folds QA*QB into b1/b2 to match the C++ integer pipeline.
        w1 = _ste_quant(self.l2.weight, float(QB), -127, 127)
        h2 = torch.clamp(F.linear(h, w1, self.l2.bias), 0.0, m)
        w2 = _ste_quant(self.l3.weight, float(QB), -127, 127)
        raw = F.linear(h2, w2, self.l3.bias).squeeze(-1)
        if self.out_scale is not None:
            return raw * self.out_scale.abs()
        return raw


def sigmoid_np(x):
    """Numerically stable sigmoid for numpy WDL probabilities."""
    x = np.clip(x, -20, 20)
    return 1.0 / (1.0 + np.exp(-x))


def fit_k(hce: np.ndarray, y: np.ndarray) -> float:
    """Pick WDL temperature K minimizing MSE of sigmoid(hce/K) vs game outcomes.

    Same K is then used for the net: sigmoid(pred/K). A good K makes HCE a
    fair baseline; if the net cannot beat HCE-only MSE, it is a worse eval.
    """
    best_k, best = 400.0, 1e9
    for k in range(50, 8000, 50):
        p = np.clip(sigmoid_np(hce / k), 1e-6, 1 - 1e-6)
        loss = np.mean((p - y) ** 2)
        if loss < best:
            best, best_k = loss, float(k)
    return best_k


def quantize(model: Nnue):
    """Convert float QAT weights into the integer tensors C++ loads.

    Scaling conventions (Stockfish-style):
      W0, B0 : * QA, stored int16          (feature transformer)
      W1, W2 : * QB, stored int8           (affine layers)
      B1     : * QA * QB, stored int32     (so  h2 = (B1 + W1·c) / QB )
      B2     : * QA * QB, stored int32     (so  raw = B2 + W2·h2     )
    """
    w0 = (model.emb.weight.detach().cpu().numpy()[:N_FEAT] * QA).round()
    w0 = np.clip(w0, -32767, 32767).astype(np.int16)
    b0 = (model.b0.detach().cpu().numpy() * QA).round()
    b0 = np.clip(b0, -32767, 32767).astype(np.int16)
    w1 = (model.l2.weight.detach().cpu().numpy() * QB).round()
    w1 = np.clip(w1, -127, 127).astype(np.int8)
    b1 = (model.l2.bias.detach().cpu().numpy() * QB * QA).round().astype(np.int32)
    w2 = (model.l3.weight.detach().cpu().numpy().reshape(-1) * QB).round()
    w2 = np.clip(w2, -127, 127).astype(np.int8)
    b2 = int(np.round(model.l3.bias.detach().cpu().numpy().reshape(-1)[0] * QB * QA))
    return w0, b0, w1, b1, w2, b2


def quant_forward_batch(
    w0, b0, w1, b1, w2, b2, stm_idx, nsm_idx, scale: int, crelu_max=QA, asinh_s=0
):
    """Integer forward pass matching C++ NnueNet::evaluate (including scale/asinh).

    Used to pin the output scale, shift the empty-board bias, and report
    quantized fit — not for training.
    """
    n = stm_idx.shape[0]
    out = np.zeros((n,), dtype=np.int32)
    cmax = int(crelu_max)
    for i in range(n):
        acc0 = b0.astype(np.int32).copy()
        acc1 = b0.astype(np.int32).copy()
        for f in stm_idx[i]:
            if 0 <= f < N_FEAT:
                acc0 += w0[f].astype(np.int32)
        for f in nsm_idx[i]:
            if 0 <= f < N_FEAT:
                acc1 += w0[f].astype(np.int32)
        # CReLU on both accumulators, then concat → 2H ints in {0..cmax}.
        c = np.concatenate(
            [np.clip(acc0, 0, cmax), np.clip(acc1, 0, cmax)]
        ).astype(np.int32)
        h2 = np.empty((L2,), dtype=np.int32)
        for j in range(L2):
            s = int(b1[j]) + int(np.dot(c, w1[j].astype(np.int32)))
            h2[j] = int(np.clip(s // QB, 0, cmax))
        raw = int(b2) + int(np.dot(h2, w2.astype(np.int32)))
        if asinh_s > 0:
            # Train in asinh-space; invert so reported scores are HCE units.
            z = raw / float(QB * QA)
            out[i] = int(round(asinh_s * np.sinh(z)))
        else:
            out[i] = int(raw * scale // (QB * QA))
    return out


def pin_scale(w0, b0, w1, b1, w2, b2, stm_idx, nsm_idx, hce, crelu_max=QA, n_sample=4000):
    """Choose NNUE_SCALE so median |quantized net| ≈ median |HCE| on a sample.

    Without this, an HCE-regression net can be well correlated but in the
    wrong units, which wrecks search margins (RFP, futility, delta).
    Fallback 2410 is a historically-used default if the sample is degenerate.
    """
    rng = np.random.default_rng(0)
    n = min(n_sample, len(hce))
    pick = rng.choice(len(hce), size=n, replace=False)
    # scale=QA*QB → identity in quant_forward (raw * 1), i.e. unscaled logits.
    raw = quant_forward_batch(
        w0, b0, w1, b1, w2, b2, stm_idx[pick], nsm_idx[pick], QA * QB, crelu_max=crelu_max
    )
    mask = (np.abs(hce[pick]) > 0) & (np.abs(raw) > 0)
    if mask.sum() < 50:
        return 2410
    med_h = float(np.median(np.abs(hce[pick][mask])))
    med_n = max(1.0, float(np.median(np.abs(raw[mask]))))
    scale = int(round(med_h * QA * QB / med_n))
    return max(1, scale)


def write_inc(path, w0, b0, w1, b1, w2, b2, scale, crelu_max=QA, asinh_s=0):
    """Emit a C++ header with the quantized tables as compile-time arrays."""

    def csv_i16(arr):
        return ",".join(str(int(x)) for x in arr.flatten())

    def csv_i8(arr):
        return ",".join(str(int(x)) for x in arr.flatten())

    def csv_i32(arr):
        return ",".join(str(int(x)) for x in arr.flatten())

    with open(path, "w", encoding="utf-8") as f:
        f.write("#pragma once\n")
        f.write(f"static const int NNUE_SCALE = {int(scale)};\n")
        f.write(f"static const int NNUE_CRELU_MAX = {int(crelu_max)};\n")
        f.write(f"static const int NNUE_ASINH_S = {int(asinh_s)};\n")
        f.write(f"static const int32_t NNUE_B2 = {int(b2)};\n")
        f.write(f"static const int16_t NNUE_W0[{N_FEAT * H}] = {{ {csv_i16(w0)} }};\n")
        f.write(f"static const int16_t NNUE_B0[{H}] = {{ {csv_i16(b0)} }};\n")
        f.write(f"static const int8_t NNUE_W1[{L2 * 2 * H}] = {{ {csv_i8(w1)} }};\n")
        f.write(f"static const int32_t NNUE_B1[{L2}] = {{ {csv_i32(b1)} }};\n")
        f.write(f"static const int8_t NNUE_W2[{L2}] = {{ {csv_i8(w2)} }};\n")


def pack_blob(w0, b0, w1, b1, w2, b2, scale) -> bytes:
    """Binary blob the engine can mmap/decode (magic CFN1 + scale + tables)."""
    buf = bytearray()
    buf += b"CFN1"
    buf += struct.pack("<ii", int(scale), int(b2))
    buf += w0.astype("<i2").tobytes()
    buf += b0.astype("<i2").tobytes()
    buf += w1.astype("i1").tobytes()
    buf += b1.astype("<i4").tobytes()
    buf += w2.astype("i1").tobytes()
    return bytes(buf)


def write_pack_inc(path, blob: bytes):
    """CodinGame-friendly header: the CFN1 blob as a base64 raw-string."""
    b64 = base64.b64encode(blob).decode("ascii")
    with open(path, "w", encoding="utf-8") as f:
        f.write("#pragma once\n")
        f.write(f"static const int NNUE_PACK_RAW_BYTES = {len(blob)};\n")
        f.write("static const char NNUE_PACK_B64[] = R\"NNUE(")
        f.write(b64)
        f.write(")NNUE\";\n")


def report_quant_fit(
    w0, b0, w1, b1, w2, b2, scale, stm_idx, nsm_idx, hce, crelu_max=QA, asinh_s=0, n_sample=2500
):
    """Print quantized-net vs HCE error stats on a random sample (HCE units)."""
    rng = np.random.default_rng(1)
    n = min(n_sample, len(hce))
    pick = rng.choice(len(hce), size=n, replace=False)
    pred = quant_forward_batch(
        w0, b0, w1, b1, w2, b2, stm_idx[pick], nsm_idx[pick], scale,
        crelu_max=crelu_max, asinh_s=asinh_s,
    ).astype(np.float64)
    t = hce[pick].astype(np.float64)
    err = pred - t
    mae = float(np.mean(np.abs(err)))
    med = float(np.median(np.abs(err)))
    p90 = float(np.percentile(np.abs(err), 90))
    mse = float(np.mean(err ** 2))
    corr = float(np.corrcoef(pred, t)[0, 1]) if t.std() > 0 else 0.0
    mean_t = float(np.mean(t))
    mse_mean = float(np.mean((t - mean_t) ** 2))
    print(
        f"quant fit n={n} mae={mae:.1f} median|e|={med:.1f} p90|e|={p90:.1f} "
        f"mse={mse:.0f} corr={corr:.3f} baseline_mse_mean={mse_mean:.0f}",
        flush=True,
    )


def report_quant_sat(w1, w2):
    """How many int8 weights are glued to the ±127 rail (QAT saturation)."""
    w2a = np.abs(w2.astype(np.int32))
    w1a = np.abs(w1.astype(np.int32))
    print(
        f"quant sat W2 |w|>=120 {int((w2a >= 120).sum())}/{w2.size} "
        f"median|W2|={float(np.median(w2a)):.0f} "
        f"W1 |w|>=120 {int((w1a >= 120).sum())}/{w1.size} "
        f"median|W1|={float(np.median(w1a)):.0f}",
        flush=True,
    )


def train(args):
    stm_idx, nsm_idx, y, hce = load_dump(args.data)
    print(
        f"loaded {len(y)} positions from {args.data} target={args.target} "
        f"label={args.label} residual={int(args.residual)}",
        flush=True,
    )
    if args.residual and args.target != "wdl":
        raise SystemExit("--residual requires --target wdl")

    k = 400.0
    # Search mates are huge; clip so Huber/MSE is not dominated by ±mate.
    hce_clip = np.clip(hce, -20000.0, 20000.0)
    crelu_max = float(args.crelu_max)
    # Integer CReLU clip used at export: float max 1.0 → 127 when QA=127.
    crelu_imax = max(1, int(round(crelu_max * QA)))
    asinh_s = int(args.asinh_s) if args.label == "asinh" else 0

    if args.target == "hce":
        if args.label == "asinh":
            # Compress the long tails so the last layer does not have to span
            # thousands of HCE units with int8 weights.
            tgt = np.arcsinh(hce_clip / float(args.asinh_s)).astype(np.float32)
            print(
                f"HCE asinh S={args.asinh_s}: mean_z={float(tgt.mean()):.3f} std_z={float(tgt.std()):.3f} "
                f"HCE std={float(hce_clip.std()):.1f} Huber delta={args.huber} CReLU max={crelu_max} ({crelu_imax})",
                flush=True,
            )
        else:
            tgt = hce_clip
            print(
                f"HCE regression: mean={float(tgt.mean()):.1f} std={float(tgt.std()):.1f} "
                f"median|hce|={float(np.median(np.abs(tgt))):.1f} Huber delta={args.huber} "
                f"CReLU max={crelu_max} ({crelu_imax})",
                flush=True,
            )
    else:
        tgt = y
        k = fit_k(hce_clip, y)
        p_hce = sigmoid_np(hce_clip / k)
        hce_wdl_mse = float(np.mean((p_hce - y) ** 2))
        print(
            f"WDL sigmoid K={k:.0f} HCE-only MSE={hce_wdl_mse:.6f} "
            f"CReLU max={crelu_max} ({crelu_imax}) residual={int(args.residual)}",
            flush=True,
        )

    rng = np.random.default_rng(42)
    perm = rng.permutation(len(y))
    split = int(len(y) * 0.9)
    tr, va = perm[:split], perm[split:]
    # Dumb baseline: always predict the training-set mean of the target.
    baseline = float(np.mean((tgt[va] - float(np.mean(tgt[tr]))) ** 2))
    print(f"val baseline MSE (predict train mean)={baseline:.6f}", flush=True)
    if args.target == "wdl":
        p_hce_va = sigmoid_np(hce_clip[va] / k)
        print(f"val HCE-only WDL MSE={float(np.mean((p_hce_va - y[va]) ** 2)):.6f}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    # Residual and linear-HCE both need a learned output scale; asinh WDL-style
    # logits are already O(1) so the int8 layer is enough.
    use_scale = (args.target == "hce" and args.label != "asinh") or args.residual
    init_scale = 20.0 if args.residual else 80.0
    model = Nnue(use_scale=use_scale, crelu_max=crelu_max, init_scale=init_scale).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs))
    stm_t = torch.from_numpy(stm_idx)
    nsm_t = torch.from_numpy(nsm_idx)
    tgt_t = torch.from_numpy(tgt.astype(np.float32))
    hce_t = torch.from_numpy(hce_clip.astype(np.float32))

    def run_epoch(indices, train_mode):
        if train_mode:
            model.train()
            rng.shuffle(indices)
        else:
            model.eval()
        total = 0.0
        seen = 0
        bs = args.batch
        ctx = torch.enable_grad() if train_mode else torch.no_grad()
        with ctx:
            for start in range(0, len(indices), bs):
                sl = indices[start : start + bs]
                pred = model(stm_t[sl].long().to(device), nsm_t[sl].long().to(device))
                batch = tgt_t[sl].to(device)
                if args.residual:
                    # Net is a correction: HCE + residual, then WDL sigmoid.
                    pred = hce_t[sl].to(device) + pred
                    p = torch.sigmoid(pred / k)
                    loss = F.mse_loss(p, batch)
                elif args.target == "hce":
                    loss = F.huber_loss(pred, batch, delta=args.huber)
                else:
                    p = torch.sigmoid(pred / k)
                    loss = F.mse_loss(p, batch)
                if train_mode:
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    with torch.no_grad():
                        # Keep PAD a true zero; keep W1/W2 inside the int8 grid
                        # even if STE rounding is slightly off this step.
                        model.emb.weight[PAD].zero_()
                        model.l2.weight.clamp_(-127.0 / QB, 127.0 / QB)
                        model.l3.weight.clamp_(-127.0 / QB, 127.0 / QB)
                total += float(loss.item()) * len(sl)
                seen += len(sl)
        if train_mode:
            sched.step()
        return total / max(1, seen)

    best_val = 1e9
    best_state = None
    bad = 0
    for epoch in range(args.epochs):
        tr_loss = run_epoch(tr.copy(), True)
        va_loss = run_epoch(va.copy(), False)
        scale_s = ""
        if model.out_scale is not None:
            scale_s = f" out_scale={float(model.out_scale.detach().abs()):.1f}"
        print(f"epoch {epoch:03d} train={tr_loss:.6f} val={va_loss:.6f}{scale_s}", flush=True)
        if va_loss + 1e-6 < best_val:
            best_val = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                print("early stop")
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    # Float-net diagnostics on a val slice, before integer quantization.
    if args.target == "hce" or args.residual:
        model.eval()
        with torch.no_grad():
            sl = va[: min(4096, len(va))]
            pred = model(stm_t[sl].long().to(device), nsm_t[sl].long().to(device)).cpu().numpy()
        if args.residual:
            print(
                f"float val residual n={len(sl)} mean={float(pred.mean()):.1f} "
                f"std={float(pred.std()):.1f} median|r|={float(np.median(np.abs(pred))):.1f} "
                f"val_wdl_mse={best_val:.6f}",
                flush=True,
            )
        else:
            if asinh_s > 0:
                pred_hce = asinh_s * np.sinh(pred)
            else:
                pred_hce = pred
            t = hce_clip[sl]
            err = pred_hce - t
            corr = float(np.corrcoef(pred_hce, t)[0, 1]) if t.std() > 0 else 0.0
            print(
                f"float val n={len(sl)} mae={float(np.mean(np.abs(err))):.1f} "
                f"median|e|={float(np.median(np.abs(err))):.1f} corr={corr:.3f} (HCE units)",
                flush=True,
            )

    w0, b0, w1, b1, w2, b2 = quantize(model)
    report_quant_sat(w1, w2)
    if asinh_s > 0:
        scale = 1  # inversion happens inside evaluate via sinh, not a linear scale
    elif args.residual and model.out_scale is not None:
        scale = max(1, int(round(float(model.out_scale.detach().abs()))))
    else:
        scale = pin_scale(
            w0, b0, w1, b1, w2, b2, stm_idx, nsm_idx, hce_clip, crelu_max=crelu_imax
        )

    # Empty UTTTAI string: 90 zeros, STM='1' (P0), constraint='9' (free move), result='0'.
    empty = b"0" * 90 + b"190"
    empty_stm = np.array([pad_feats(features_for_pers(empty, 0))], dtype=np.int32)
    empty_nsm = np.array([pad_feats(features_for_pers(empty, 1))], dtype=np.int32)
    empty_eval = int(
        quant_forward_batch(
            w0, b0, w1, b1, w2, b2, empty_stm, empty_nsm, scale,
            crelu_max=crelu_imax, asinh_s=asinh_s,
        )[0]
    )
    if args.residual:
        # Shift B2 so empty residual is 0. HCE already has tempo 112; we do not
        # want the net to double-count opening "side to move".
        delta = int(round(empty_eval * (QB * QA) / max(1, scale)))
        b2 -= delta
        empty_eval2 = int(
            quant_forward_batch(
                w0, b0, w1, b1, w2, b2, empty_stm, empty_nsm, scale,
                crelu_max=crelu_imax, asinh_s=asinh_s,
            )[0]
        )
        print(
            f"quantized nnue_scale={scale} crelu_imax={crelu_imax} val_wdl_mse={best_val:.6f} "
            f"empty residual {empty_eval}->{empty_eval2} (keep HCE tempo 112, HCE+net ~ {112 + empty_eval2})",
            flush=True,
        )
    elif args.target != "hce":
        # Replace-HCE WDL net: empty eval should look like HCE tempo, not 0.
        target_empty = 112
        delta = int(round((empty_eval - target_empty) * (QB * QA) / max(1, scale)))
        b2 -= delta
        empty_eval2 = int(
            quant_forward_batch(
                w0, b0, w1, b1, w2, b2, empty_stm, empty_nsm, scale,
                crelu_max=crelu_imax, asinh_s=asinh_s,
            )[0]
        )
        print(
            f"quantized nnue_scale={scale} val_loss={best_val:.6f} empty {empty_eval}->{empty_eval2} (target {target_empty})",
            flush=True,
        )
    else:
        print(
            f"quantized nnue_scale={scale} crelu_imax={crelu_imax} asinh_s={asinh_s} "
            f"val_huber={best_val:.6f} empty NNUE={empty_eval} (HCE tempo=112)",
            flush=True,
        )
        report_quant_fit(
            w0, b0, w1, b1, w2, b2, scale, stm_idx[va], nsm_idx[va], hce_clip[va],
            crelu_max=crelu_imax, asinh_s=asinh_s,
        )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    write_inc(args.out, w0, b0, w1, b1, w2, b2, scale, crelu_max=crelu_imax, asinh_s=asinh_s)
    blob = pack_blob(w0, b0, w1, b1, w2, b2, scale)
    bin_path = args.bin or os.path.splitext(args.out)[0] + ".bin"
    with open(bin_path, "wb") as f:
        f.write(blob)
    print(f"wrote {args.out}")
    print(f"wrote {bin_path} ({len(blob)} bytes)")
    if args.pack:
        pack_path = args.pack
        write_pack_inc(pack_path, blob)
        print(f"wrote {pack_path} (b64 {len(base64.b64encode(blob))} chars)")


def main():
    ap = argparse.ArgumentParser(
        description="Train the 199-feature dual-accumulator NNUE (not MiniNet; see nnue_train_mininet.py)."
    )
    ap.add_argument("--data", default="datasets/nnue_pos.bin", help="NNUEWDL1 dump")
    ap.add_argument("--out", default="cpp_impl/nnue_weights.inc", help="C++ weight header")
    ap.add_argument("--bin", default="cpp_impl/nnue_weights.bin", help="CFN1 binary blob")
    ap.add_argument("--pack", default="", help="optional base64 .inc for CodinGame")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--wd", type=float, default=0.0, help="Adam weight decay")
    ap.add_argument("--patience", type=int, default=12, help="early-stop epochs without val gain")
    ap.add_argument("--target", choices=["wdl", "hce"], default="hce",
                    help="wdl = game outcome; hce = dump int32 score")
    ap.add_argument("--label", choices=["linear", "asinh"], default="linear",
                    help="hce target transform; asinh compresses mate tails")
    ap.add_argument("--asinh-s", type=float, default=1000.0, help="S in asinh(hce/S)")
    ap.add_argument("--crelu-max", type=float, default=1.0,
                    help="CReLU upper clip in float units (1.0 → 127 after *QA)")
    ap.add_argument("--huber", type=float, default=400.0, help="Huber delta for --target hce")
    ap.add_argument("--residual", action="store_true",
                    help="WDL only: train HCE+net, export residual (empty shifted to 0)")
    train(ap.parse_args())


if __name__ == "__main__":
    main()
