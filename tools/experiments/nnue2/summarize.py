#!/usr/bin/env python3
"""Table of every probe run in datasets/nnue2/probe/*.json (held-out loss vs the shipped eval and F_full)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe import FFULL_LOSS, OUT, SHIP_LOSS  # noqa: E402

print(f"shipped eval {SHIP_LOSS:.6f}, F_full {FFULL_LOSS:.6f}")
print(f"{'run':18s} {'params':>11s} {'rows':>10s} {'best':>9s} {'ep':>3s} {'vs ship':>8s} {'vs Ffull':>8s} {'std':>6s}"
      f" {'min':>5s}  curve")
for f in sorted(OUT.glob("*.json")):
    d = json.loads(f.read_text())
    if "best" not in d:
        continue
    b = d["best"]
    curve = " ".join(f"{h['val']:.4f}" for h in d["hist"])
    rows = d["train_rows"]
    print(f"{d['name']:18s} {d['params']:>11,} {rows:>10,} {b['val']:.6f} {b['epoch']:>3d} {d['vs_shipped_pct']:+7.2f}%"
          f" {d.get('vs_ffull_pct', 100 * (b['val'] / FFULL_LOSS - 1)):+7.2f}% {b['std']:>6.0f} {d['train_minutes']:>5.1f}  {curve}")
