#!/usr/bin/env python3
"""Negative tests of fast_worker.py's checks (nnue2 full-data play stage).

  fw_negative.py NAME PAIRDIR        (the worker: CROSSFISH_WORKER=user@host, as for fast_worker.py)
      NAME is a worker directory that `fast_worker.py setup NAME PAIRDIR` prepared. Runs its remote
      pair_statics with the nets as set up, swapped (FASTNNUE_PATH <-> FASTNNUE_PREV_PATH) and shared
      (both sides on Dev's net), and counts the positions whose Dev / Prev static differs from the
      candidates' own single local builds: the crosscheck passes only the first. Then feeds
      fast_pair.check_text net lines that must pass (the planned local path; the worker path from
      the side's own variable) and lines that must fail (wrong CRC directory, the other side's
      variable, another file name, a wrong size, the other net's CRC).
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fast_worker as fw  # noqa: E402
import fast_pair as fp  # noqa: E402


def main():
    name, pairdir = sys.argv[1], sys.argv[2]
    pd, doc = fw.pairing(pairdir)
    ref = {s: fw.reference(doc[s]) for s in ("dev", "prev")}
    n = len(ref["dev"])
    _, env = fw.ssh(f"cat ~/{fw.remote_dir(name)}/cand/nets.env")
    kv = dict(line.split("=", 1) for line in env.split())
    cases = [("as set up", f"FASTNNUE_PATH={kv['FASTNNUE_PATH']} FASTNNUE_PREV_PATH={kv['FASTNNUE_PREV_PATH']}", True),
             ("swapped", f"FASTNNUE_PATH={kv['FASTNNUE_PREV_PATH']} FASTNNUE_PREV_PATH={kv['FASTNNUE_PATH']}", False),
             ("shared (Dev's net)", f"FASTNNUE_PATH={kv['FASTNNUE_PATH']} FASTNNUE_PREV_PATH={kv['FASTNNUE_PATH']}", False)]
    bad = 0
    for label, e, should_pass in cases:
        _, out = fw.ssh(f"cd ~/{fw.remote_dir(name)}/cpp_impl && env {e} ../cand/pair_statics ../cand/positions.cfdg"
                        f" {n} 0 ../neg.txt > /dev/null 2>&1; cat ../neg.txt")
        rows = {int(f[0]): (int(f[1]), int(f[2])) for f in (x.split() for x in out.splitlines()) if len(f) == 5}
        dd = sum(rows.get(i, (None, None))[0] != ref["dev"][i] for i in range(n))
        pdd = sum(rows.get(i, (None, None))[1] != ref["prev"][i] for i in range(n))
        passes = dd + pdd == 0 and len(rows) == n
        ok = passes == should_pass
        bad += not ok
        print(f"statics {label:<20} Dev differs on {dd:>4}/{n}, Prev on {pdd:>4}/{n}: crosscheck"
              f" {'passes' if passes else 'fails'} ({'as it must' if ok else 'WRONG'})")

    dev, prev = doc["dev"], doc["prev"]
    want = {"Dev": (dev["net"], dev["crc32"]), "Prev": (prev["net"], prev["crc32"])}
    size = {s: Path(doc[s]["net"]).stat().st_size for s in ("dev", "prev")}
    _, remote_home = fw.ssh("echo $HOME")
    home = f"{remote_home.strip()}/{fw.REMOTE_NETS}"  # where fast_worker.py keeps the nets

    def line(side, path, src, nbytes, crc):
        return f"fast_nnue [{side}]: net {path} ({src}), {nbytes} bytes, crc32 {crc}\n"

    dname, pname = Path(dev["net"]).name, Path(prev["net"]).name
    good_dev = line("Dev", f"{home}/{dev['crc32']}/{dname}", "FASTNNUE_PATH", size["dev"], dev["crc32"])
    good_prev = line("Prev", f"{home}/{prev['crc32']}/{pname}", "FASTNNUE_PREV_PATH", size["prev"], prev["crc32"])
    local = (line("Dev", dev["net"], "compiled in", size["dev"], dev["crc32"])
             + line("Prev", prev["net"], "compiled in", size["prev"], prev["crc32"]))
    logs = [
        ("local compiled-in paths", local, True),
        ("worker paths", good_dev + good_prev, True),
        ("local then resumed on the worker", local + good_dev + good_prev, True),
        ("worker Dev net in the wrong CRC directory",
         line("Dev", f"{home}/{prev['crc32']}/{dname}", "FASTNNUE_PATH", size["dev"], dev["crc32"]) + good_prev, False),
        ("worker Dev net from Prev's variable",
         line("Dev", f"{home}/{dev['crc32']}/{dname}", "FASTNNUE_PREV_PATH", size["dev"], dev["crc32"]) + good_prev,
         False),
        ("worker Dev net under another file name",
         line("Dev", f"{home}/{dev['crc32']}/other.bin", "FASTNNUE_PATH", size["dev"], dev["crc32"]) + good_prev, False),
        ("worker Dev net with a wrong size",
         line("Dev", f"{home}/{dev['crc32']}/{dname}", "FASTNNUE_PATH", size["dev"] + 1, dev["crc32"]) + good_prev,
         False),
        ("worker nets swapped",
         line("Dev", f"{home}/{prev['crc32']}/{pname}", "FASTNNUE_PATH", size["prev"], prev["crc32"])
         + line("Prev", f"{home}/{dev['crc32']}/{dname}", "FASTNNUE_PREV_PATH", size["dev"], dev["crc32"]), False),
        ("Prev net line missing", good_dev, False),
    ]
    for label, text, should_pass in logs:
        _, problems = fp.check_text(text, want)
        passes = not problems
        ok = passes == should_pass
        bad += not ok
        print(f"check-logs {label:<42} {'passes' if passes else 'fails'} ({'as it must' if ok else 'WRONG'})")
    print("all negative tests behave as they must" if not bad else f"{bad} WRONG")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
