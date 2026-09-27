#!/usr/bin/env python3
"""Parallel builds and a worker-aware queue for fast-NNUE round robins (fast_pair.py rr-build / rr-run). --root DIR (before the command) keeps the
tournaments somewhere other than datasets/eval2/rr, as round_robin.py --root does.
Written for the long-training play stage's nnue_long_20ms (then named long_play_rr.py).

  build NAME [--jobs 6]
      fast_pair.py rr-build, in parallel: every pairing that is not up to date is built by
      `fast_pair.py pair A=DIR_A B=DIR_B --out PAIRDIR` (the same build_pairing rr-build calls, one
      process per pairing, output in the tournament's build/A__B.log) and stamped with rr_build.json
      exactly as rr-build stamps it (round_robin's digest and fast_pair's fast digest, both taken before
      the build). Planned pairs whose build is current become "built".
  run NAME [--worker-exclude REGEX] [fast_pair.py rr-run / round_robin.py run options ...]
      fast_pair.py rr-run NAME ... with one change to round_robin's queue (Runner.next_pair) when REGEX
      is given: the worker never takes a pair in which either engine's name matches REGEX, and the local
      executor takes those pairs first, then helps with the rest. The recorded runs used '^B128_': B-128
      nets lose relative speed on the laptop's smaller L3 (nps_probe.py). Everything else (stamps check,
      FASTNNUE_* dropped, fast_worker.py on the worker, logs, resume, refits) is rr-run's.

Run with toolchains/py312-dml's Python.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FAST = HERE
sys.path.insert(0, str(FAST))
sys.path.insert(0, str(ROOT / "tools"))
import fast_pair as fp  # noqa: E402
rr = fp.rr


def say(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def cmd_build(a):
    plan = rr.load_plan(a.name)
    todo = [p for p in plan["pairs"] if not fp.stamp_ok(plan, p)]
    say(f"{a.name}: {len(plan['pairs']) - len(todo)} of {len(plan['pairs'])} pairings up to date;"
        f" building {len(todo)} with {a.jobs} jobs")
    build_logs = rr.tour_dir(a.name) / "build"
    build_logs.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    def one(p):
        key = rr.pair_key(p)
        d = rr.absdir(p["dir"])
        a_dir, b_dir = rr.absdir(plan["candidates"][p["a"]]), rr.absdir(plan["candidates"][p["b"]])
        d.mkdir(parents=True, exist_ok=True)
        (d / "rr_build.json").unlink(missing_ok=True)
        digest, fdigest = rr.build_digest(plan, p), fp.fast_digest(a_dir, b_dir)
        log = build_logs / f"{key}.log"
        tb = time.time()
        with open(log, "w", encoding="utf-8") as fh:
            r = subprocess.run([sys.executable, str(FAST / "fast_pair.py"), "pair", f"{p['a']}={a_dir}",
                                f"{p['b']}={b_dir}", "--out", str(d)], stdout=fh, stderr=subprocess.STDOUT, cwd=ROOT)
        if r.returncode:
            raise RuntimeError(f"{key}: build failed (exit {r.returncode}), see {log}")
        (d / "rr_build.json").write_text(json.dumps(dict(
            digest=digest, fast_digest=fdigest, dev=plan["candidates"][p["a"]], prev=plan["candidates"][p["b"]],
            built=time.time(), builder="tools/experiments/fast_nnue/fast_pair.py (rr_parallel.py build)"), indent=1))
        return key, time.time() - tb, log.read_text(encoding="utf-8").strip().splitlines()[-1]

    bad = []
    with ThreadPoolExecutor(a.jobs) as ex:
        futs = [ex.submit(one, p) for p in todo]
        for i, f in enumerate(as_completed(futs), 1):
            try:
                key, dt, last = f.result()
                el = time.time() - t0
                say(f"built {key} in {dt:.0f} s ({i}/{len(todo)}, ETA {rr.fmt_dur(el / i * (len(todo) - i))}): {last}")
            except Exception as e:  # noqa: BLE001
                bad.append(str(e))
                say(f"FAILED: {e}")
    plan = rr.load_plan(a.name)  # re-read: a run may have updated states meanwhile
    for p in plan["pairs"]:
        if p["state"] == "planned" and fp.stamp_ok(plan, p):
            p["state"] = "built"
    rr.save_plan(plan)
    ok = sum(fp.stamp_ok(plan, p) for p in plan["pairs"])
    say(f"{ok} of {len(plan['pairs'])} pairings up to date after {rr.fmt_dur(time.time() - t0)}")
    if bad:
        raise SystemExit(f"{len(bad)} build(s) failed")


def install_queue(exclude):
    pat = re.compile(exclude)

    def worker_ok(p):
        return not (pat.search(p["a"]) or pat.search(p["b"]))

    def next_pair(self, where):
        """round_robin.Runner.next_pair with the worker restricted to worker_ok pairs and the local
        executor taking the others first (otherwise unchanged: plan order, stale pairs reported)."""
        stale = []
        with self.lock:
            if self.stop.is_set():
                return None, None
            plan = rr.load_plan(self.name)
            todo = [p for p in plan["pairs"] if rr.pair_key(p) not in self.claimed
                    and rr.games_done(self.dir / f"{rr.pair_key(p)}.log") < p["games"] and p["state"] != "failed"]
            running = [p for p in todo if p["state"] == "running"]  # only the worker's survive cmd_run's reset
            rest = [p for p in todo if p["state"] != "running"]
            if where == "worker":
                order = [p for p in running + rest if worker_ok(p)]
            else:
                order = [p for p in rest if not worker_ok(p)] + [p for p in rest if worker_ok(p)]
            for p in order:
                if rr.is_built(plan, p):
                    self.claimed.add(rr.pair_key(p))
                    return plan, p
                stale.append(rr.pair_key(p))
        if stale:
            self.say(f"{where}: {len(stale)} unfinished pair(s) not built or out of date ({', '.join(stale)});"
                     f" run the build, then `run` again")
        return plan, None

    rr.Runner.next_pair = next_pair


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, allow_abbrev=False)
    ap.add_argument("--root", help="tournaments directory, before the command (default datasets/eval2/rr)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("build", allow_abbrev=False); p.add_argument("name"); p.add_argument("--jobs", type=int, default=6)
    p = sub.add_parser("run", allow_abbrev=False); p.add_argument("name")
    p.add_argument("--worker-exclude", help="regex of engine names the worker must not play (e.g. '^B128_')")
    a, rest = ap.parse_known_args()
    if a.root:
        rr.RR_ROOT = rr.absdir(a.root)
    if a.cmd == "build":
        if rest:
            ap.error(f"unrecognized arguments: {' '.join(rest)}")
        cmd_build(a)
    else:
        if a.worker_exclude:
            install_queue(a.worker_exclude)
            say(f"run {a.name}: worker excludes pairs matching {a.worker_exclude!r}; local takes those first")
        fp.cmd_rr_run(argparse.Namespace(name=a.name), rest)


if __name__ == "__main__":
    main()
