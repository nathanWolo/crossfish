#!/usr/bin/env python3
"""Negative tests of fast_worker.py's review-5 fix (long-training play stage).

The laptop's login environment is simulated as polluted: every worker command runs after
`export FASTNNUE_BAKE=1 FASTNNUE_PREV_BAKE=1;` (and, in one test, a stray SPRT_ELO1=7). On one real pairing:

  A  new fast_worker, polluted: setup's statics crosscheck must pass, start's log header and the running
     test_bots's /proc/PID/environ must hold exactly the planned FASTNNUE_* / SPRT_* variables, and the
     finished log's loader lines must be the normal (not re-baked) ones.
  B  (with --old FILE: a fast_worker.py from before the fix) the old worker, polluted, same start: shows
     the hole: /proc has FASTNNUE_BAKE=1 and FASTNNUE_PREV_BAKE=1, its header does not say so, and
     the engines load in bake mode.
  C  new fast_worker with a stray SPRT_ELO1=7 in the login environment: start must refuse and stop the run.
  D  pair_statics old-style (`env NETS`) vs new-style (`env -u ... NETS`) under the pollution, against
     the candidates' own local statics: the power of the statics crosscheck against a re-bake.

  fw_env_test.py --pair PAIRDIR [--old OLD_FAST_WORKER.py] [--log FILE]
      PAIRDIR is a fast_pair.py pairing of two fast candidates (the recorded run used
      cpp_impl/bin/rrlong_B64_d5M_57ep__B64_lr1e2). The worker is CROSSFISH_WORKER=user@host, as for
      fast_worker.py; it runs 8 games of 20 ms in ~/crossfish_worker/fwenv_test/, removed at the end.
"""
from __future__ import annotations

import argparse
import importlib.util
import re
import sys
import time
from argparse import Namespace
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fast_worker as new  # noqa: E402

POLLUTE = "export FASTNNUE_BAKE=1 FASTNNUE_PREV_BAKE=1; "
REMOTE = "fwenv_test"
LOG = None


def out(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if LOG:
        with open(LOG, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def polluted(mod, prefix):
    real = new.ssh.__wrapped__ if hasattr(new.ssh, "__wrapped__") else new.ssh

    def ssh(cmd, check=True, capture=True, input=None):
        return real(prefix + cmd, check=check, capture=capture, input=input)
    ssh.__wrapped__ = real
    mod.ssh = ssh
    return real


def wait_done(real, rd, limit=120):
    t0 = time.time()
    while time.time() - t0 < limit:
        _, o = real(f"kill -0 $(cat ~/{rd}/sprt.pid) 2>/dev/null && echo alive || echo dead", check=False)
        if "dead" in o:
            break
        time.sleep(2)
    _, log = real(f"cat ~/{rd}/sprt.log", check=False)
    return log


def proc_env(real, rd):
    _, o = real(f"tr '\\0' '\\n' < /proc/$(cat ~/{rd}/sprt.pid)/environ | grep -E '^(FASTNNUE|SPRT)_' | LC_ALL=C sort",
                check=False)
    return sorted(o.split())


def loader_modes(log):
    return {"bake": len(re.findall(r"fast_nnue_b: read \d+ ms, bake \(C\+\+", log)),
            "normal": len(re.findall(r"fast_nnue_b: read \d+ ms, quantize \d+ ms", log))}


def test_b(old_path, start, real, rd):
    """B: the fast_worker.py at OLD_PATH (before the fix), same pollution, same pairing (sources already
    shipped by A): the hole must show."""
    spec = importlib.util.spec_from_file_location("fast_worker_before", old_path)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    out(f"B: old fast_worker ({old_path}), same pollution, same pairing (sources already shipped)")
    polluted(old, POLLUTE)
    old.cmd_start(Namespace(**start))
    env_b = proc_env(real, rd)
    log = wait_done(real, rd)
    head = log.splitlines()[0] if log else ""
    modes = loader_modes(log)
    b_shows = "FASTNNUE_BAKE=1" in env_b and "FASTNNUE_BAKE" not in head and modes["bake"] == 2
    out(f"B: /proc environ {env_b}")
    out(f"B: header {head!r}")
    out(f"B: loader lines {modes}; the hole {'REPRODUCED' if b_shows else 'NOT reproduced'}"
        f" (engines re-baked, header silent)")
    return b_shows


def main():
    global LOG
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pair", required=True, help="a fast_pair.py pairing directory")
    ap.add_argument("--old", help="a fast_worker.py from before the review-5 fix: test B reproduces its hole")
    ap.add_argument("--log", help="also write the report lines to this file")
    a = ap.parse_args()
    LOG = a.log
    if LOG:
        Path(LOG).write_text("", encoding="utf-8")
    rd = new.remote_dir(REMOTE)
    start = dict(name=REMOTE, ms=20, threads=2, offset=49900, depth=0, games=8, env=[])
    ok = True

    out(f"A: new fast_worker, login environment polluted with {POLLUTE.strip()}")
    real = polluted(new, POLLUTE)
    new.cmd_setup(Namespace(name=REMOTE, pairdir=a.pair))
    new.cmd_start(Namespace(**start))
    log = wait_done(real, rd)
    head = log.splitlines()[0] if log else ""
    modes = loader_modes(log)
    a_ok = "BAKE" not in head and modes == {"bake": 0, "normal": 2}
    out(f"A: header {head!r}")
    out(f"A: loader lines {modes}; {'PASS' if a_ok else 'FAIL'}")
    ok &= a_ok

    if a.old:
        ok &= test_b(a.old, start, real, rd)
    else:
        out("B: skipped (no --old fast_worker.py given)")

    out("C: new fast_worker, stray SPRT_ELO1=7 in the login environment")
    polluted(new, POLLUTE + "export SPRT_ELO1=7; ")
    try:
        new.cmd_start(Namespace(**start))
        out("C: start accepted the stray variable: FAIL")
        ok = False
    except SystemExit as e:
        out(f"C: start refused: {str(e)[:400]}; PASS")
    _, o = real(f"sleep 1; kill -0 $(cat ~/{rd}/sprt.pid) 2>/dev/null && echo alive || echo dead", check=False)
    out(f"C: the refused run is {o.strip()}")
    ok &= "dead" in o

    out("D: pair_statics under the pollution, old-style vs new-style environment, against the local references")
    pd, doc = new.pairing(a.pair)
    ref = {s: new.reference(doc[s]) for s in ("dev", "prev")}
    _, nets = real(f"cat ~/{rd}/cand/nets.env")
    nenv = " ".join(nets.split())
    for label, envcmd in (("old env NETS", f"env {nenv}"), ("new env -u ... NETS", f"{new.CLEAN_ENV} {nenv}")):
        _, o = real(f"{POLLUTE}cd ~/{rd}/cpp_impl && {envcmd} ../cand/pair_statics ../cand/positions.cfdg 1000 0"
                    f" ../statics_{len(label)}.txt > /dev/null 2>&1; cat ../statics_{len(label)}.txt", check=False)
        rows = {}
        for line in o.splitlines():
            f = line.split()
            if len(f) == 5:
                rows[int(f[0])] = (int(f[1]), int(f[2]))
        dd = sum(rows.get(i, (None, None))[0] != ref["dev"][i] for i in range(1000))
        pdd = sum(rows.get(i, (None, None))[1] != ref["prev"][i] for i in range(1000))
        out(f"D: {label}: {len(rows)} rows; Dev statics differ from its single build on {dd}, Prev on {pdd}")
    real(f"rm -rf ~/{rd}", check=False)
    out(f"removed ~/{rd}; overall {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
