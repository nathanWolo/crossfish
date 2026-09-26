#!/usr/bin/env python3
"""Table of the capacity-probe results in datasets/eval2/capacity/*.json."""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[3] / "datasets/eval2/capacity"
SHIP, FFULL = 0.046817, 0.043527
rows = []
for f in sorted(OUT.glob("*.json")):
    d = json.loads(f.read_text())
    if not isinstance(d, dict) or "log" not in d or "params_old" not in d:
        continue
    curve = " ".join(f"{e['val_loss']:.5f}" for e in d["log"][1:])
    rows.append((f.stem, d["params_old"], d["params_new"], d["best_val"], d["best_epoch"], d["epoch0"],
                 100 * (d["best_val"] / SHIP - 1), 100 * (d["best_val"] / FFULL - 1), d["train_minutes"],
                 d["log"][-1].get("ll", float("nan")), curve))
print(f"{'arm':6} {'old':>9} {'new':>10} {'best':>9} {'ep':>3} {'ep0':>9} {'vs ship':>8} {'vs Ffull':>8}"
      f" {'min':>5} {'ll':>6}  curve")
for r in rows:
    print(f"{r[0]:6} {r[1]:>9,} {r[2]:>10,} {r[3]:.6f} {r[4]:>3} {r[5]:.6f} {r[6]:+7.2f}% {r[7]:+7.2f}%"
          f" {r[8]:5.1f} {r[9]:.4f}  {r[10]}")
