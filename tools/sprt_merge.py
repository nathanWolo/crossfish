#!/usr/bin/env python3
"""Combine test_bots SPRT shards (one per machine) into one pentanomial SPRT.

Each shard is an ordinary `test_bots` run over its own opening range
(SPRT_GAME_OFFSET) with both engines on the same machine, so every pair is a
fair comparison and pairs from different machines can be pooled. This sums the
last pentanomial line of each log and recomputes the LLR and Elo exactly as
test_bots does (sprt_pentanomial, calc_pentanomial_elo), then appends a
test_bots-style line to OUT, plus "SPRT PASS/FAIL" once |LLR| reaches the bound.

  sprt_merge.py OUT LOG [LOG ...] [--elo0 0] [--elo1 5] [--bound 2.94]
"""
from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

LINE = re.compile(r"^N: (\d+) W: (\d+) D: (\d+) L: (\d+) Penta=(\d+),(\d+),(\d+),(\d+),(\d+)")
TIMEOUTS = re.compile(r"timeouts Prev=(\d+) Dev=(\d+)")


def logistic_score(elo):
    return 1.0 / (1.0 + 10.0 ** (-elo / 400.0))


def mle_expected(results, expected):
    """test_bots pentanomial_mle_expected: MLE of the five outcome probabilities
    under the constraint that the expected pair score equals `expected`."""
    counts = [r if r else 1e-3 for r in results]
    total = sum(counts)
    probs = [c / total for c in counts]
    deltas = [i / 4.0 - expected for i in range(5)]
    lower, upper = -1.0 / max(deltas), -1.0 / min(deltas)
    lower += 1e-12 * max(1.0, abs(lower))
    upper -= 1e-12 * max(1.0, abs(upper))
    secular = lambda x: sum(p * d / (1.0 + x * d) for p, d in zip(probs, deltas))  # noqa: E731
    for _ in range(200):
        mid = (lower + upper) * 0.5
        if secular(mid) > 0:
            lower = mid
        else:
            upper = mid
    root = (lower + upper) * 0.5
    return [p / (1.0 + root * d) for p, d in zip(probs, deltas)]


def llr(results, elo0, elo1):
    if sum(results) == 0:
        return 0.0
    m0 = mle_expected(results, logistic_score(elo0))
    m1 = mle_expected(results, logistic_score(elo1))
    return sum((r if r else 1e-3) * math.log(a / b) for r, a, b in zip(results, m1, m0))


def pentanomial_elo(results):
    pairs = sum(results)
    games = 2.0 * pairs
    mean = sum(r * (i / 2.0) for i, r in enumerate(results)) / games
    var = sum(r * (i / 2.0 - 2.0 * mean) ** 2 for i, r in enumerate(results)) / games
    se = math.sqrt(var / games)
    z = 1.959963984540054
    to_elo = lambda s: -400.0 * math.log10(1.0 / min(max(s, 1e-9), 1 - 1e-9) - 1.0)  # noqa: E731
    return to_elo(mean), (to_elo(mean + z * se) - to_elo(mean - z * se)) / 2.0


def last_line(path):
    last = None
    try:
        for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
            if LINE.match(line.strip()):
                last = line.strip()
    except OSError:
        return None
    return last


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--elo0", type=float, default=0.0)
    ap.add_argument("--elo1", type=float, default=5.0)
    ap.add_argument("--bound", type=float, default=2.94)
    a = ap.parse_args()
    wdl, penta, tp, td, parts = [0, 0, 0], [0] * 5, 0, 0, []
    for log in a.logs:
        line = last_line(log)
        if not line:
            parts.append(f"{Path(log).stem}: -")
            continue
        m = LINE.match(line)
        g = [int(x) for x in m.groups()]
        wdl = [x + y for x, y in zip(wdl, g[1:4])]
        penta = [x + y for x, y in zip(penta, g[4:9])]
        t = TIMEOUTS.search(line)
        if t:
            tp, td = tp + int(t.group(1)), td + int(t.group(2))
        parts.append(f"{Path(log).stem}: {g[0]}")
    n = sum(wdl)
    if n == 0:
        print("no games yet")
        return 0
    value = llr(penta, a.elo0, a.elo1)
    elo, ci = pentanomial_elo(penta)
    line = (f"N: {n} W: {wdl[0]} D: {wdl[1]} L: {wdl[2]} Penta={','.join(map(str, penta))} "
            f"Elo diff: {elo:.4f} +/- {ci:.4f} LLR: {value:.5f} timeouts Prev={tp} Dev={td} "
            f"shards [{'; '.join(parts)}]")
    with open(a.out, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")
        if value >= a.bound:
            fh.write(f"SPRT PASS: H1 {a.elo1} accepted (combined shards)\n")
        elif value <= -a.bound:
            fh.write(f"SPRT FAIL: H0 {a.elo0} accepted (combined shards)\n")
    print(line)
    return 2 if abs(value) >= a.bound else 0


if __name__ == "__main__":
    sys.exit(main())
