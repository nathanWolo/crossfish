#!/usr/bin/env python3
"""nnue2 stage 4, isolation test (ii): compare a pairing's pair_statics output with each net's references.

  pair_statics_check.py OUT.txt DEV_REF.cfdg PREV_REF.cfdg [--label L] [--json SUMMARY.jsonl]

DEV_REF and PREV_REF are the single-net builds' `datagen label POSITIONS REF DEPTH 1` outputs (static
eval and fixed-depth search per record, one thread) for the net each side of the pairing should have.
Every Dev value must equal DEV_REF's and every Prev value PREV_REF's. The report also counts the
positions where the two references differ from each other: a pairing whose sides shared one net would
fail on about that many positions, so that count is the test's power. Exits 1 on any difference.
Standard library only.
"""
import argparse
import json
import struct
import sys
from pathlib import Path

REC = 128
NONE = 99999999


def load_ref(path):
    data = Path(path).read_bytes()
    out = []
    for i in range(len(data) // REC):
        static, search = struct.unpack_from("<ii", data, i * REC + 108)
        out.append((static, search))
    return out


def load_out(path):
    rows = {}
    for line in Path(path).read_text().splitlines():
        f = line.split()
        if len(f) == 5:
            rows[int(f[0])] = tuple(int(x) for x in f[1:])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("dev_ref")
    ap.add_argument("prev_ref")
    ap.add_argument("--label", default="")
    ap.add_argument("--json")
    a = ap.parse_args()
    rows, dref, pref = load_out(a.out), load_ref(a.dev_ref), load_ref(a.prev_ref)
    n = len(rows)
    res = dict(label=a.label or Path(a.out).stem, positions=n, dev_ref=Path(a.dev_ref).name,
               prev_ref=Path(a.prev_ref).name)
    for side, col_s, col_q, ref in (("dev", 0, 2, dref), ("prev", 1, 3, pref)):
        res[f"{side}_static_diff"] = sum(1 for i, r in rows.items() if r[col_s] != ref[i][0])
        res[f"{side}_search_diff"] = sum(1 for i, r in rows.items() if r[col_q] != NONE and r[col_q] != ref[i][1])
        res[f"{side}_unsearched"] = sum(1 for r in rows.values() if r[col_q] == NONE)
    res["refs_static_differ"] = sum(1 for i in rows if dref[i][0] != pref[i][0])
    res["refs_search_differ"] = sum(1 for i in rows if dref[i][1] != pref[i][1])
    ok = all(res[k] == 0 for k in ("dev_static_diff", "dev_search_diff", "prev_static_diff", "prev_search_diff",
                                   "dev_unsearched", "prev_unsearched")) and n > 0
    res["ok"] = ok
    print(f"{res['label']}: {n} positions; Dev vs {res['dev_ref']}: static {n - res['dev_static_diff']}/{n} equal,"
          f" search {n - res['dev_search_diff'] - res['dev_unsearched']}/{n} equal; Prev vs {res['prev_ref']}: static"
          f" {n - res['prev_static_diff']}/{n} equal, search {n - res['prev_search_diff'] - res['prev_unsearched']}/{n}"
          f" equal; the two references differ on {res['refs_static_differ']} statics and {res['refs_search_differ']}"
          f" searches -> {'OK' if ok else 'FAIL'}")
    if a.json:
        with open(a.json, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(res) + "\n")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
