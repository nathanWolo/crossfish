#!/usr/bin/env python3
"""Screen each eval_train_plan.py result as soon as its training finishes.

For every run: emit headers, build the isolated A/B test_bots (with the Dev
patches recorded in <run>.json by nnue_train_blend.py --dev-patches, written
to the build dir as dev_patches.json), verify that the build equals the
trained model on the run's own training data, then play an equal-depth screen
against the shipped eval (datasets/eval2/sprt/<run>_d<depth>.log, shown on the
dashboard). A run whose training or any step fails is reported and skipped;
restart the plan to retry it.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "datasets" / "eval2"
RUNS = ["A_search", "A_blend", "A_blend_uttt", "B_blend_hce"]  # default; or pass run names
DEPTH, GAMES = 8, 4000


def tool(*args):
    subprocess.run([sys.executable, str(ROOT / "tools" / "eval_candidate.py"), *map(str, args)], check=True, cwd=ROOT)


def main():
    pending = sys.argv[1:] or list(RUNS)
    while pending:
        for name in list(pending):
            log = D / "train" / f"{name}.log"
            train = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
            if "\nwrote " not in train:  # finished as eval_train_plan.py counts it, whatever the exit code
                failed = re.search(r"\nexit (-?\d+)\s*$", train)  # eval_train_plan.py appends the trainer's exit code
                if failed:
                    pending.remove(name)
                    print(f"{name}: training failed (exit {failed.group(1)}); see {log}", flush=True)
                continue
            pending.remove(name)
            out = ROOT / "cpp_impl" / "bin" / f"cand_{name}"
            screen = D / "sprt" / f"{name}_d{DEPTH}.log"
            text = screen.read_text(encoding="utf-8", errors="replace") if screen.exists() else ""
            if re.search(r"^SPRT (PASS|FAIL|INCONCLUSIVE)", text, re.M):
                continue  # already screened (the log ends with the SPRT verdict line)
            prefix = D / "train" / name
            meta = json.loads((D / "train" / f"{name}.json").read_text())
            out.mkdir(parents=True, exist_ok=True)
            if "dev_patches" in meta:  # runs trained before --dev-patches keep a hand-placed file
                (out / "dev_patches.json").unlink(missing_ok=True)
                if meta["dev_patches"]:
                    (out / "dev_patches.json").write_text(json.dumps(meta["dev_patches"], indent=1))
            steps = [("emit", prefix, out, *(["--recluster"] if meta.get("mode") == "full" else [])),
                     ("build", out), ("verify", out, "--train-prefix", prefix),
                     ("match", out, f"{name}_d{DEPTH}", "--depth", DEPTH, "--games", GAMES)]
            try:
                for step in steps:
                    tool(*step)
            except subprocess.CalledProcessError:
                print(f"{name}: FAILED at {step[0]}", flush=True)
                continue
            print(f"{name}: screened", flush=True)
        time.sleep(30)
    print("screens complete")


if __name__ == "__main__":
    main()
