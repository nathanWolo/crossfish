#!/usr/bin/env python3
"""Train a compact super-board/constraint residual head.

The base MiniNet is deliberately local-board heavy.  This companion head
learns the residual left after HCE + MiniNet from only nine super-board
classes and the active-board constraint.  Its first layer can be preprojected
into tiny runtime tables.
"""

from __future__ import annotations

import argparse
import copy
import math
import struct
from pathlib import Path

import numpy as np
import torch
from torch import nn

import nnue_train_mininet as mininet


class MacroContext(nn.Module):
    def __init__(self, d: int, hidden: int):
        super().__init__()
        self.d = d
        self.h = hidden
        self.embedding = nn.Embedding(9 * 4, d)
        self.constraint = nn.Embedding(10, d)
        self.hidden = nn.Linear(d, hidden)
        self.out = nn.Linear(hidden, 1)
        self.register_buffer("offsets", torch.arange(9) * 4)
        nn.init.normal_(self.embedding.weight, std=0.02)
        nn.init.normal_(self.constraint.weight, std=0.02)
        nn.init.normal_(self.hidden.weight, std=0.02)
        nn.init.zeros_(self.hidden.bias)
        nn.init.normal_(self.out.weight, std=0.02)
        nn.init.zeros_(self.out.bias)

    def raw(self, super_state, constraint):
        value = self.embedding(super_state + self.offsets).sum(dim=1)
        value = value + self.constraint(constraint)
        return self.out(torch.relu(self.hidden(value))).squeeze(1)

    def forward(self, super_state, constraint):
        empty_super = torch.zeros(
            (1, 9), dtype=torch.long, device=super_state.device
        )
        empty_constraint = torch.full(
            (1,), 9, dtype=torch.long, device=super_state.device
        )
        return (
            self.raw(super_state, constraint)
            - self.raw(empty_super, empty_constraint)
        )


def load_base(path: Path) -> mininet.MiniNet:
    blob = path.read_bytes()
    if blob[:4] != b"CFM2":
        raise SystemExit(f"bad MiniNet magic in {path}")
    d, h = struct.unpack_from("<ii", blob, 4)
    model = mininet.MiniNet(d=d, h=h)
    mininet.load_mini_bin(str(path), model)
    model.eval()
    return model


def predict_base(
    model: mininet.MiniNet,
    mini: np.ndarray,
    supers: np.ndarray,
    constraint: np.ndarray,
) -> np.ndarray:
    output = np.empty(len(mini), dtype=np.float32)
    with torch.no_grad():
        for lo in range(0, len(output), 16384):
            hi = min(len(output), lo + 16384)
            output[lo:hi] = model(
                torch.from_numpy(mini[lo:hi]).long(),
                torch.from_numpy(supers[lo:hi]).long(),
                torch.from_numpy(constraint[lo:hi]).long(),
            ).numpy()
    return output


def report(name: str, target: np.ndarray, prediction: np.ndarray) -> None:
    error = target - prediction
    corr = (
        float(np.corrcoef(target, prediction)[0, 1])
        if np.std(prediction) > 0
        else 0.0
    )
    print(
        f"{name}: mae={np.mean(np.abs(error)):.3f} "
        f"rmse={math.sqrt(float(np.mean(error * error))):.3f} "
        f"corr={corr:.5f} |pred|={np.mean(np.abs(prediction)):.3f}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--net", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--d", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--huber", type=float, default=800.0)
    parser.add_argument("--target-clip", type=float, default=4000.0)
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()

    torch.manual_seed(20260913 + args.hidden)
    torch.set_num_threads(args.threads)
    rng = np.random.default_rng(20260913)

    states, hce, search = mininet.load_records(str(args.data))
    keep = np.abs(search) < 8000
    states, hce, search = states[keep], hce[keep], search[keep]
    cache = Path(
        str(args.data)
        + f".mini16.n{len(search)}_m8000_u0_1_2.npz"
    )
    if cache.exists():
        cached = np.load(cache)
        mini = cached["mini"]
        supers = cached["super"]
        constraint = cached["constr"]
    else:
        mini, supers, constraint = mininet.extract_mini(states)

    base = load_base(args.net)
    residual = search - hce - predict_base(
        base, mini, supers, constraint
    )
    keep = np.abs(residual) <= args.target_clip
    supers = supers[keep].astype(np.int64)
    constraint = constraint[keep].astype(np.int64)
    residual = residual[keep].astype(np.float32)
    print(
        f"usable={len(residual)} residual_mean={residual.mean():.2f} "
        f"residual_mae={np.mean(np.abs(residual)):.2f}",
        flush=True,
    )

    order = rng.permutation(len(residual))
    split = int(0.9 * len(order))
    train_idx = order[:split]
    valid_idx = order[split:]
    super_t = torch.from_numpy(supers)
    constraint_t = torch.from_numpy(constraint)
    target_t = torch.from_numpy(residual)

    model = MacroContext(args.d, args.hidden)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-4
    )
    loss_fn = nn.HuberLoss(delta=args.huber)
    best_state = copy.deepcopy(model.state_dict())
    best_mae = float("inf")
    stale = 0

    report(
        "uncorrected",
        residual[valid_idx],
        np.zeros(len(valid_idx), dtype=np.float32),
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        shuffled = rng.permutation(train_idx)
        for lo in range(0, len(shuffled), 4096):
            idx = shuffled[lo : lo + 4096]
            prediction = model(super_t[idx], constraint_t[idx])
            loss = loss_fn(prediction, target_t[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            prediction = model(
                super_t[valid_idx], constraint_t[valid_idx]
            )
            mae = torch.mean(
                torch.abs(target_t[valid_idx] - prediction)
            ).item()
        if mae + 0.01 < best_mae:
            best_mae = mae
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0:
            print(
                f"epoch={epoch:3d} loss={loss.item():.3f} "
                f"valid_mae={mae:.3f}",
                flush=True,
            )
        if stale >= args.patience:
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        prediction = model(
            super_t[valid_idx], constraint_t[valid_idx]
        ).numpy()
        all_prediction = model(super_t, constraint_t).numpy()
    report("macro-context", residual[valid_idx], prediction)
    print(
        f"range=[{all_prediction.min():.1f},{all_prediction.max():.1f}] "
        f"p99_abs={np.quantile(np.abs(all_prediction), 0.99):.1f}",
        flush=True,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "d": model.d,
            "h": model.h,
            "best_mae": best_mae,
            "base_net": str(args.net),
            "data": str(args.data),
        },
        args.out,
    )
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
