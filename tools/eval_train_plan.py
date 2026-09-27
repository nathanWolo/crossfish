#!/usr/bin/env python3
"""Wait for the eval data pipeline, then train a sweep of variants: eval_train_plan.py [SWEEP].

Each run logs to <dir>/train/<name>.log (shown on the lab dashboard), ending
with "exit <rc>", and writes <dir>/train/<name>.{cfm2,macro.pt,codes.npy,
cents.npy,json}. Finished runs (log has the trainer's "wrote " line) are
skipped on restart. A run on data labeled by a patched Dev passes that patch
list as --dev-patches, so eval_screen_plan.py builds it that way (the finished
D_full run predates the option: its patch is the hand-placed cpp_impl/bin/
cand_D_full/dev_patches.json).
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "datasets" / "eval2"
DEPTH = 14
DATA = [str(D / f"cf_play_d{DEPTH}.cfdg"), str(D / f"uttt_sp_d{DEPTH}.cfdg")]
DIFFS = [str(D / f"cf_play_d{DEPTH}.diffs"), str(D / f"uttt_sp_d{DEPTH}.diffs")]
COMMON = ["--epochs", "10", "--batch", "16384", "--lr", "3e-4", "--device", "dml", "--threads", "2"]
SWEEPS = {
    # Sweep 1: target blend, uttt.ai value, HCE weights.
    "1": [
        ("A_search", ["--lam", "1.0"]),
        ("A_blend", ["--lam", "0.7"]),
        ("A_blend_uttt", ["--lam", "0.7", "--uttt", "0.25"]),
        ("B_blend_hce", ["--lam", "0.7", "--hce-weights", *DIFFS]),
    ],
    # Sweep 2: the search-only target (the sweep-1 screen winner) with each idea.
    "2": [
        ("S_uttt", ["--lam", "1.0", "--uttt", "0.25"]),
        ("S_hce", ["--lam", "1.0", "--hce-weights", *DIFFS]),
        ("S_full", ["--lam", "1.0", "--mode", "full"]),
        ("S_e20", ["--lam", "1.0", "--epochs", "20"]),
    ],
    # Sweep 3: after the free-move label fix (free-move rows relabeled with the
    # corrected loader, 61 games played under the bug excluded).
    "3": [
        ("F_search", ["--lam", "1.0"]),
        ("F_full", ["--lam", "1.0", "--mode", "full"]),
        ("F_hce", ["--lam", "1.0", "--hce-weights", *DIFFS]),
    ],
    # Sweep 4: the label fix moved the fitted K from 3150 to 1600, so sweep 3
    # changed the objective as well as the labels. Fix K explicitly.
    "4": [
        ("F_k3150", ["--lam", "1.0", "--k", "3150"]),
        ("F_k2400", ["--lam", "1.0", "--k", "2400"]),
    ],
    # Sweep 5: the full-table net on the drawn-miniboard HCE fix (statics
    # recomputed with that HCE by `datagen label ... 0`; search labels kept).
    "5": [
        ("D_full", ["--lam", "1.0", "--mode", "full",
                    "--data", *(d.replace(".cfdg", "_hd.cfdg") for d in DATA),
                    # the drawn-miniboard HCE patch the _hd statics were computed with
                    "--dev-patches", str(ROOT / "cpp_impl" / "bin" / "cand_hcedraw" / "dev_patches.json")]),
    ],
}


def pipeline_done():
    try:
        doc = json.loads((D / "pipeline.json").read_text())
    except (OSError, ValueError):
        return False
    return all(s["status"] == "done" for s in doc["stages"])


def main():
    sweep = sys.argv[1] if len(sys.argv) > 1 else "1"
    while not pipeline_done():
        time.sleep(60)
    (D / "train").mkdir(exist_ok=True)
    for name, extra in SWEEPS[sweep]:
        log = D / "train" / f"{name}.log"
        if log.exists() and "\nwrote " in log.read_text(encoding="utf-8", errors="replace"):
            continue
        cmd = [sys.executable, str(ROOT / "tools" / "nnue_train_blend.py"), "--data", *DATA,
               "--out", str(D / "train" / name), *COMMON, *extra]
        with open(log, "w", encoding="utf-8") as fh:
            fh.write("$ " + " ".join(cmd) + "\n")
            fh.flush()
            rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=ROOT).returncode
        with open(log, "a", encoding="utf-8") as fh:
            fh.write(f"exit {rc}\n")  # eval_screen_plan.py drops runs whose trainer failed
        print(f"{name}: exit {rc}", flush=True)
    print("training sweep complete")


if __name__ == "__main__":
    main()
