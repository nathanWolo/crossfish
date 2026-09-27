#!/usr/bin/env python3
"""Export probe.py R10 checkpoints to round ten's FNN1 float format, and check parity.

  export_fnn1.py export CKPT.pt OUT.bin
      FNN1 layout (tools/experiments/full_nnue/full_nnue_float.hpp): magic, A, L1,
      W0[200][A] (feature 199 unused, zero), B0[A], W1[L1][2A], B1[L1], W2[L1], B2.
      The probe's R10 uses the same 199 features in the same order and the same
      x1000 output, so no reordering is needed.
  export_fnn1.py parity CKPT.pt LABELED.cfdg
      LABELED is a sample relabelled at depth 0 by a candidate datagen whose
      evaluate() is the FNN1 net: its static_eval must equal PyTorch's R10 output.
"""
import struct
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
import probe  # noqa: E402
import eval_data  # noqa: E402


def export(ckpt, out):
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    W = sd["W"].numpy()
    A = W.shape[1]
    L1 = sd["l1.weight"].shape[0]
    W0 = np.zeros((200, A), dtype="<f4")
    W0[:199] = W
    blob = b"FNN1" + struct.pack("<ii", A, L1)
    for a in (W0, sd["b0"].numpy(), sd["l1.weight"].numpy(), sd["l1.bias"].numpy(),
              sd["l2.weight"].numpy().reshape(-1)):
        blob += np.asarray(a, dtype="<f4").tobytes()
    blob += struct.pack("<f", float(sd["l2.bias"].numpy().reshape(-1)[0]))
    Path(out).write_bytes(blob)
    print(f"wrote {out}: A={A} L1={L1} {len(blob):,} bytes")


def parity(ckpt, labeled):
    rec = np.fromfile(labeled, dtype=eval_data.REC)
    x = torch.from_numpy(probe.compact(rec).astype(np.int64))
    fx = probe.Feats(torch.device("cpu"))
    m = probe.R10(fx)
    m.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False))
    with torch.no_grad():
        e = m(x[:, :81], x[:, 81:90], x[:, 90]).numpy()
    d = np.trunc(e) - rec["static_eval"]  # C++ truncates the float to int
    print(f"parity {Path(ckpt).name}: {len(rec):,} positions, mean|d| {np.abs(d).mean():.3f}, max|d| {np.abs(d).max():.0f}")
    if np.abs(d).max() > 2:
        sys.exit("C++ FNN1 eval differs from PyTorch")


if __name__ == "__main__":
    {"export": export, "parity": parity}[sys.argv[1]](*sys.argv[2:])
