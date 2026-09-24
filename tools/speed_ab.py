#!/usr/bin/env python3
"""Repeated paired Dev-vs-Prev timing with a confidence interval.

One `bench_ab sat` run is a few seconds and swings by several percent on a
shared host even when Dev and Prev are the same code. This runs it many times,
checks that node counts are identical every time (a speed-only change must not
change the tree), and reports the mean dev/prev time ratio with a 95% CI over
runs. A gain whose CI straddles the A/A noise band is not a measured gain.

usage: speed_ab.py [--bin cpp_impl/bin/bench_ab] [--runs 20]
                   [--games 6] [--plies 40] [--depth 10]
"""
import argparse
import math
import re
import statistics
import subprocess
import sys

# Pin to one core so the scheduler cannot migrate a run mid-measurement.
PIN = ["taskset", "-c", "2"]


def one_run(binary, games, plies, depth):
    out = subprocess.run(PIN + [binary, "sat", str(games), str(plies), str(depth)],
                         check=True, capture_output=True, text=True).stdout
    nodes = re.findall(r"(prev|dev)\s+nodes=(\d+).*?secs=([\d.]+)", out)
    d = {k: (int(n), float(s)) for k, n, s in nodes}
    return d["prev"], d["dev"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", default="cpp_impl/bin/bench_ab")
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--games", type=int, default=6)
    ap.add_argument("--plies", type=int, default=40)
    ap.add_argument("--depth", type=int, default=10)
    a = ap.parse_args()
    ratios = []
    identical = True
    for i in range(a.runs):
        (pn, ps), (dn, ds) = one_run(a.bin, a.games, a.plies, a.depth)
        if pn != dn:
            identical = False
        ratios.append(ds / ps)
        print(f"run {i + 1:2d}: prev {ps:.3f}s dev {ds:.3f}s "
              f"ratio {ds / ps:.4f} nodes {'==' if pn == dn else '!='}",
              flush=True)
    m = statistics.mean(ratios)
    sd = statistics.stdev(ratios) if len(ratios) > 1 else 0.0
    ci = 1.96 * sd / math.sqrt(len(ratios))
    med = statistics.median(ratios)
    print(f"time ratio dev/prev: mean {m:.4f} +/- {ci:.4f} (95% CI), "
          f"median {med:.4f}, n={len(ratios)}")
    print(f"=> dev speedup {100 * (1 / m - 1):+.2f}% "
          f"[{100 * (1 / (m + ci) - 1):+.2f}%, {100 * (1 / (m - ci) - 1):+.2f}%]")
    print("node counts:", "IDENTICAL every run" if identical else "DIFFER")
    return 0 if identical else 1


if __name__ == "__main__":
    sys.exit(main())
