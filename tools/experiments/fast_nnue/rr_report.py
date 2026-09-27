#!/usr/bin/env python3
"""Markdown tables of a round robin (tools/round_robin.py plan / fast_pair.py rr-run), from its pair logs.
Written for the long-training play stage's nnue_long_20ms (then named long_play_report.py); the machine
comparison of that write-up is rr_machine_check.py.

  rr_report.py NAME [--anchor noop] [--ref ENGINE] [--compare OTHER] [--json FILE] [--root DIR]

  * ratings (round_robin.wls on every finished pair) with ANCHOR = 0, and with REF = 0 when --ref is given;
  * the head-to-head matrix (row's Elo against the column);
  * with --ref, every engine's head to head against REF next to its rating difference in the full fit;
  * every pair: openings, machine (desktop = local, laptop = worker), W/D/L, pentanomial, Elo and 95% CI,
    timeouts and the largest move time of each side;
  * with --compare OTHER, each engine's rating here next to its rating in tournament OTHER's ratings.json.
The numbers also go to FILE as JSON with --json. Standard library + tools/round_robin.py.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))
import round_robin as rr  # noqa: E402

Z = rr.Z95
MAXMS = re.compile(r"timeouts Prev=(\d+) Dev=(\d+) max_ms Prev=([\d.]+) Dev=([\d.]+)")


def cif(f, n):
    return Z * f[n][1] if f.get(n, (None, None))[1] is not None else float("nan")


def rt(f, n):
    return f[n][0] if f.get(n, (None, None))[0] is not None else float("nan")


def load_pairs(name):
    plan = rr.load_plan(name)
    d = rr.tour_dir(name)
    pairs = []
    for p in plan["pairs"]:
        log = d / f"{rr.pair_key(p)}.log"
        r = rr.last_result(log)
        if not r:
            continue
        elo, ci, se, _ = rr.pair_estimate(r[2])
        last = [ln for ln in log.read_text(encoding="utf-8", errors="replace").splitlines() if ln.startswith("N: ")][-1]
        m = MAXMS.search(last)
        pairs.append(dict(a=p["a"], b=p["b"], where="laptop" if p.get("where") == "worker" else "desktop",
                          games=r[0], wdl=r[1], penta=r[2], elo=elo, ci=ci, se=se, offset=p["offset"],
                          timeouts=[int(m.group(2)), int(m.group(1))] if m else None,
                          max_ms=[float(m.group(4)), float(m.group(3))] if m else None))
    return plan, pairs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name")
    ap.add_argument("--anchor", default="noop")
    ap.add_argument("--ref", help="a second anchor: every engine's rating and head to head against it")
    ap.add_argument("--compare", help="another tournament whose ratings.json to set beside these ratings")
    ap.add_argument("--json", help="write the numbers here")
    ap.add_argument("--root", help="tournaments directory (round_robin.py --root; default datasets/eval2/rr)")
    a = ap.parse_args()
    if a.root:
        rr.RR_ROOT = Path(a.root).resolve()
    plan, pairs = load_pairs(a.name)
    names = list(plan["candidates"])
    if a.anchor not in names or (a.ref and a.ref not in names):
        raise SystemExit(f"--anchor / --ref must be engines of {a.name}: {', '.join(names)}")
    rows = [(p["a"], p["b"], p["elo"], p["se"]) for p in pairs]
    out = dict(name=a.name, anchor=a.anchor, ref=a.ref, pairs=pairs)

    fit, chi2, dof = rr.wls(names, a.anchor, rows)
    order = sorted(names, key=lambda n: -(fit[n][0] if fit[n][0] is not None else -1e9))
    fit_r = rr.wls(names, a.ref, rows)[0] if a.ref else None
    out["fit"] = dict(ratings={n: [fit[n][0], cif(fit, n)] for n in names}, chi2=chi2, dof=dof)
    if fit_r:
        out["fit_ref"] = {n: [fit_r[n][0], cif(fit_r, n)] for n in names}
    games = {n: sum(p["games"] for p in pairs if n in (p["a"], p["b"])) for n in names}
    print(f"## ratings ({a.name}: {len(pairs)} pairs, {sum(p['games'] for p in pairs):,} games;"
          f" fit chi2 {chi2:.1f} on {dof} dof)\n")
    head = f"| # | engine | vs {a.anchor} | 95% CI |" + (f" vs {a.ref} | 95% CI |" if fit_r else "") + " games |"
    print(head)
    print("| ---: | --- | ---: | ---: |" + (" ---: | ---: |" if fit_r else "") + " ---: |")
    for i, n in enumerate(order):
        ref_cells = f" {rt(fit_r, n):+.1f} | {cif(fit_r, n):.1f} |" if fit_r else ""
        print(f"| {i + 1} | {n} | {rt(fit, n):+.1f} | {cif(fit, n):.1f} |{ref_cells} {games[n]:,} |")

    h2h = {}
    for p in pairs:
        h2h[(p["a"], p["b"])] = (p["elo"], p["ci"])
        h2h[(p["b"], p["a"])] = (-p["elo"], p["ci"])
    per_pair = sorted({p["games"] for p in pairs})
    print(f"\n## head to head (row's Elo against the column; {'/'.join(f'{g:,}' for g in per_pair)} games per pair)\n")
    print("| | " + " | ".join(order) + " |\n| --- | " + " | ".join("---:" for _ in order) + " |")
    for n in order:
        print(f"| **{n}** | " + " | ".join("." if m == n else f"{h2h[(n, m)][0]:+.0f}" if (n, m) in h2h else "-"
                                         for m in order) + " |")

    if fit_r:
        print(f"\n## each engine against {a.ref}\n")
        print("| engine | head to head (Elo, 95% CI) | W/D/L | where | rating difference in the full fit |")
        print("| --- | --- | --- | --- | ---: |")
        out["vs_ref"] = {}
        for n in order:
            p = next((p for p in pairs if {p["a"], p["b"]} == {n, a.ref}), None)
            if n == a.ref or not p:
                continue
            sgn = 1 if p["a"] == n else -1
            wdl = p["wdl"] if sgn == 1 else [p["wdl"][2], p["wdl"][1], p["wdl"][0]]
            print(f"| {n} | {sgn * p['elo']:+.1f} +/- {p['ci']:.1f} | {'/'.join(map(str, wdl))} | {p['where']} |"
                  f" {rt(fit_r, n):+.1f} +/- {cif(fit_r, n):.1f} |")
            out["vs_ref"][n] = dict(h2h=sgn * p["elo"], ci=p["ci"], wdl=wdl, where=p["where"], fit=fit_r[n][0],
                                    fit_ci=cif(fit_r, n))

    print("\n## every pair (Dev = A, Prev = B; Elo from A's side)\n")
    print("| pair | openings | played on | W/D/L | pentanomial | Elo | 95% CI | timeouts Dev/Prev | max ms Dev/Prev |")
    print("| --- | --- | --- | --- | --- | ---: | ---: | --- | --- |")
    for p in pairs:
        to = f"{p['timeouts'][0]}/{p['timeouts'][1]}" if p["timeouts"] else "-"
        mx = f"{p['max_ms'][0]:.1f}/{p['max_ms'][1]:.1f}" if p["max_ms"] else "-"
        print(f"| {p['a']} vs {p['b']} | {p['offset']}+{p['games'] // 2} | {p['where']} | {'/'.join(map(str, p['wdl']))} |"
              f" {', '.join(map(str, p['penta']))} | {p['elo']:+.1f} | {p['ci']:.1f} | {to} | {mx} |")
    tot = sum(sum(p["timeouts"]) for p in pairs if p["timeouts"])
    mx = max((max(p["max_ms"]) for p in pairs if p["max_ms"]), default=float("nan"))
    print(f"\ntimeouts in all pairs: {tot}; largest move time {mx:.1f} ms")
    out["timeouts"], out["max_ms"] = tot, mx

    if a.compare:
        other = json.loads((rr.tour_dir(a.compare) / "ratings.json").read_text(encoding="utf-8"))
        fr = {e["name"]: (e["rating"], e["ci"]) for e in other["engines"]}
        print(f"\n## ratings here and in {a.compare} (its anchor: {other.get('anchor')})\n")
        print(f"| engine | here | {a.compare} |\n| --- | ---: | ---: |")
        out["compare"] = {}
        for n in order:
            if n in fr:
                print(f"| {n} | {rt(fit, n):+.1f} +/- {cif(fit, n):.1f} | {fr[n][0]:+.1f} +/- {fr[n][1]:.1f} |")
                out["compare"][n] = [fit[n][0], fr[n][0]]
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
