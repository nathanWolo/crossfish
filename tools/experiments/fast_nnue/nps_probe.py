#!/usr/bin/env python3
"""Single-thread speed of a two-net pairing's Dev against its Prev on one machine: is a net relatively
slower on the laptop worker than on the desktop? Written for the full-data play stage, where B128_d5M_57ep
rated 16 Elo lower in nnue_full_20ms's laptop pairs than in its desktop pairs (z -2.2): the B-128 tables are
about twice the B-64 ones, and the laptop's i7-1365U has 12 MB of L3 against the desktop's 32 MB.
bench_ab `nodes 400 9` (Dev and Prev search the same 400 positions to depth 9) in each pairing, ROUNDS times.
  nps_probe.py laptop PAIRDIR... [--rounds 3] [--log FILE]    ships each pairing (fast_worker.py setup;
                                                              CROSSFISH_WORKER), g++ bench_ab there
  nps_probe.py desktop PAIRDIR... [--rounds 3] [--log FILE]   clang bench_ab.exe in PAIRDIR
Lines "[HH:MM:SS] MACHINE PAIR round R: dev nps D prev nps P ratio dev/prev X%", then the median ratio per
pairing; also appended to FILE with --log. Recorded: B-64 at 116% of B-128's nps on the desktop, 131% on
the laptop (datasets/nnue2/fast/play/nps_probe.log).
"""
import argparse
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "tools"))
import fast_worker as fw  # noqa: E402

LOG = None
PAT = re.compile(r"prev nodes=(\d+) secs=([\d.]+) nps=(\d+) dev nodes=(\d+) secs=([\d.]+) nps=(\d+)")


def say(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if LOG:
        with open(LOG, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def parse(out):
    m = PAT.search(" ".join(out.split()))
    if not m:
        raise SystemExit(f"no bench_ab result in:\n{out[-1500:]}")
    return int(m.group(6)), int(m.group(3)), int(m.group(4)), int(m.group(1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("machine", choices=["laptop", "desktop"])
    ap.add_argument("pairdirs", nargs="+")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--log", help="also append the lines to this file")
    a = ap.parse_args()
    global LOG
    LOG = a.log
    ratios = {}
    for pd in a.pairdirs:
        pd = fw.absdir(pd)
        _, doc = fw.pairing(pd)
        key = f"{doc['dev']['name']}__{doc['prev']['name']}"
        if a.machine == "laptop":
            name = f"nps_{key}"
            subprocess.run([sys.executable, str(HERE / "fast_worker.py"), "setup", name, str(pd)], check=True)
            rd = fw.remote_dir(name)
            fw.ssh(f"cd ~/{rd}/cpp_impl && g++ {fw.sw.FLAGS} -I. -o ../cand/bench_ab ../cand/bench_ab.cpp")
            _, env = fw.ssh(f"cat ~/{rd}/cand/nets.env")
            run = lambda: fw.ssh(f"cd ~/{rd}/cpp_impl && env {' '.join(env.split())} ../cand/bench_ab nodes 400 9 2>&1")[1]
        else:
            exe = pd / "bench_ab.exe"
            tc = ROOT / "toolchains" / "llvm-mingw-20260616-ucrt-x86_64" / "bin"
            flags = ["-O3", "-std=c++17", "-mavx2", "-mbmi", "-mbmi2", "-mlzcnt", "-mpopcnt", "-pthread",
                     "-Wno-unknown-pragmas", "-Wno-ignored-attributes"]
            subprocess.run([str(tc / "g++.exe"), *flags, f"-I{ROOT / 'cpp_impl'}", "-o", str(exe),
                            str(pd / "bench_ab.cpp")], check=True)
            env = {k: v for k, v in os.environ.items() if not k.upper().startswith("FASTNNUE_")}
            env["PATH"] = str(tc) + os.pathsep + env.get("PATH", "")
            run = lambda: subprocess.run([str(exe), "nodes", "400", "9"], env=env, cwd=ROOT / "cpp_impl",
                                         capture_output=True, text=True, errors="replace").stdout
        for r in range(a.rounds):
            dn, pn, dnodes, pnodes = parse(run())
            ratios.setdefault(key, []).append(dn / pn)
            say(f"{a.machine} {key} round {r + 1}: dev nps {dn:,} prev nps {pn:,} (nodes {dnodes:,} / {pnodes:,})"
                f" ratio dev/prev {100 * dn / pn:.1f}%")
        if a.machine == "laptop":
            fw.ssh(f"rm -rf ~/{fw.remote_dir(f'nps_{key}')}", check=False)
    for k, v in ratios.items():
        say(f"{a.machine} {k}: median nps ratio dev/prev {100 * statistics.median(v):.1f}% over {len(v)} rounds")


if __name__ == "__main__":
    main()
