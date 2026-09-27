#!/usr/bin/env python3
"""A fixed-length match of two fast-NNUE candidates (NEW as Dev against OPP as Prev), with one opening range
split between the local machine and the Linux worker, pooled into one log. The long-training play stage's
90 ms match (B64_d5M_114ep_lr5e3 vs B64_d5M_57ep) was played with it (then named long90.py).

  split_match.py NEW --opp OPP [--new-cand DIR] [--opp-cand DIR] [--games 1200] [--offset 48000]
            [--desktop-share 0.6] [--no-laptop] [--ms 90] [--desktop-threads 6] [--laptop-threads 4]
            [--tag TAG] [--log-dir DIR]

  * Pairing cpp_impl/bin/pair_split_NEW__OPP, built fresh by `fast_pair.py pair` (Dev = NEW, Prev = OPP,
    both nets compiled in) from --new-cand / --opp-cand (default cpp_impl/bin/cand_full_{NEW,OPP}, as
    export_net.sh names them). check-log compares each shard's net lines with the nets in the pairing's
    fast_pair.json.
  * Openings OFFSET .. OFFSET + GAMES/2 - 1, each played with both colours. The desktop plays the first
    DESKTOP_SHARE of them (6 threads, clang), the laptop the rest (4 threads, g++ 11.4, through
    fast_worker.py setup/start: nets by CRC, statics crosscheck, clean and recorded engine environment).
    With --no-laptop the desktop plays them all.
  * Not an early-stopping SPRT: SPRT_LLR_BOUND=100 and SPRT_MAX_GAMES = the shard's games; test_bots's
    default H0 0 / H1 +5 only label the LLR column. Every FASTNNUE_* variable and SPRT_ELO0/1 are unset.
  * Logs: datasets/nnue2/matches/NEW__vs_OPP__90_ms[_TAG].log: "[HH:MM:SS] start ..." first, then every
    30 s a pooled test_bots-style line "N: .. W: .. D: .. L: .. Penta=.. Elo diff: .. +/- .. LLR: ..
    timeouts Prev=.. Dev=.. max_ms Prev=.. Dev=.. shards [...]" (pentanomial sums of the two shards'
    last lines, Elo and LLR as test_bots / sprt_merge.py compute them) with progress and ETA as comment
    lines, the shards' check-log results, and "[HH:MM:SS] exit N" last. The shards' own logs are
    ..._90_ms[_TAG].desktop.log and ..._90_ms[_TAG].laptop.log next to it.
  * The laptop shard needs CROSSFISH_WORKER=user@host (as fast_worker.py); --no-laptop does not.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CPP = ROOT / "cpp_impl"
FAST = HERE
TOOLCHAIN = ROOT / "toolchains" / "llvm-mingw-20260616-ucrt-x86_64" / "bin"
PY = sys.executable
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(FAST))
from sprt_merge import LINE, llr, pentanomial_elo  # noqa: E402
import fast_worker as fw  # noqa: E402
import sprt_worker as sw  # noqa: E402

MAXMS = re.compile(r"max_ms Prev=([0-9.]+) Dev=([0-9.]+)")
TIMEOUTS = re.compile(r"timeouts Prev=(\d+) Dev=(\d+)")
VERDICT = re.compile(r"^SPRT (PASS|FAIL|INCONCLUSIVE)", re.M)


def stamp():
    return time.strftime("%H:%M:%S")


def last(path):
    """(games, wdl, penta, timeouts (prev, dev), max_ms (prev, dev)) of a test_bots log's last result line."""
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    got = None
    for line in text.splitlines():
        m = LINE.match(line.strip())
        if m:
            g = [int(x) for x in m.groups()]
            t = TIMEOUTS.search(line)
            x = MAXMS.search(line)
            got = (g[0], g[1:4], g[4:9], tuple(int(v) for v in t.groups()) if t else (0, 0),
                   tuple(float(v) for v in x.groups()) if x else (0.0, 0.0))
    return got


def pooled(shards):
    wdl, penta, to, mx, parts = [0, 0, 0], [0] * 5, [0, 0], [0.0, 0.0], []
    for name, path, planned in shards:
        r = last(path)
        parts.append(f"{name}: {r[0] if r else 0}/{planned}")
        if not r:
            continue
        wdl = [a + b for a, b in zip(wdl, r[1])]
        penta = [a + b for a, b in zip(penta, r[2])]
        to = [a + b for a, b in zip(to, r[3])]
        mx = [max(a, b) for a, b in zip(mx, r[4])]
    n = sum(wdl)
    if not n:
        return 0, None
    elo, ci = pentanomial_elo(penta)
    return n, (f"N: {n} W: {wdl[0]} D: {wdl[1]} L: {wdl[2]} Penta={','.join(map(str, penta))} "
               f"Elo diff: {elo:.4f} +/- {ci:.4f} LLR: {llr(penta, 0.0, 5.0):.5f} timeouts Prev={to[0]} Dev={to[1]} "
               f"max_ms Prev={mx[0]:.4f} Dev={mx[1]:.4f} shards [{'; '.join(parts)}]")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                 allow_abbrev=False)
    ap.add_argument("new")
    ap.add_argument("--opp", required=True, help="the Prev candidate's name")
    ap.add_argument("--new-cand", help="NEW's candidate directory (default cpp_impl/bin/cand_full_NEW)")
    ap.add_argument("--opp-cand", help="OPP's candidate directory (default cpp_impl/bin/cand_full_OPP)")
    ap.add_argument("--games", type=int, default=1200)
    ap.add_argument("--offset", type=int, default=48000)
    ap.add_argument("--desktop-share", type=float, default=0.6)
    ap.add_argument("--no-laptop", action="store_true")
    ap.add_argument("--ms", type=int, default=90)
    ap.add_argument("--desktop-threads", type=int, default=6)
    ap.add_argument("--laptop-threads", type=int, default=4)
    ap.add_argument("--tag", default="")
    ap.add_argument("--log-dir", help="write the logs here instead of datasets/nnue2/matches (smoke tests)")
    a = ap.parse_args()
    if a.games % 2:
        raise SystemExit("--games must be even")
    openings = a.games // 2
    d_open = openings if a.no_laptop else int(round(openings * a.desktop_share))
    l_open = openings - d_open
    base = f"{a.new}__vs_{a.opp}__{a.ms}_ms" + (f"_{a.tag}" if a.tag else "")
    mdir = Path(a.log_dir) if a.log_dir else ROOT / "datasets" / "nnue2" / "matches"
    mdir.mkdir(parents=True, exist_ok=True)
    log = mdir / f"{base}.log"
    dlog, llog = mdir / f"{base}.desktop.log", mdir / f"{base}.laptop.log"
    pair = CPP / "bin" / f"pair_split_{a.new}__{a.opp}"
    new_cand = fw.absdir(a.new_cand) if a.new_cand else CPP / "bin" / f"cand_full_{a.new}"
    opp_cand = fw.absdir(a.opp_cand) if a.opp_cand else CPP / "bin" / f"cand_full_{a.opp}"
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith("FASTNNUE_")
           and k not in ("SPRT_ELO0", "SPRT_ELO1", "SPRT_RESUME_WINS", "SPRT_RESUME_DRAWS", "SPRT_RESUME_LOSSES",
                         "SPRT_RESUME_PENTA")}
    env["PATH"] = str(TOOLCHAIN) + os.pathsep + env.get("PATH", "")

    def note(msg):
        line = f"[{stamp()}] {msg}"
        print(line, flush=True)
        with open(log, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def comment(msg):
        print(f"# {msg}", flush=True)
        with open(log, "a", encoding="utf-8") as fh:
            fh.write(f"# {msg}\n")

    split = (f"desktop openings {a.offset}..{a.offset + d_open - 1} ({2 * d_open} games, {a.desktop_threads} threads,"
             f" clang)" + ("" if a.no_laptop else f" + laptop openings {a.offset + d_open}..{a.offset + openings - 1}"
                                                   f" ({2 * l_open} games, {a.laptop_threads} threads, g++)"))
    log.write_text(f"[{stamp()}] start {base}: {a.new} (Dev) vs {a.opp} (Prev), two-net pairing {pair.name}, both nets"
                   f" compiled in; {a.ms} ms per move, {a.games} games fixed length (SPRT_LLR_BOUND=100, no early stop),"
                   f" {split}\n", encoding="utf-8")
    t0 = time.time()
    b = subprocess.run([PY, str(FAST / "fast_pair.py"), "pair", f"{a.new}={new_cand}", f"{a.opp}={opp_cand}",
                        "--out", str(pair)], cwd=ROOT, env=env, capture_output=True, text=True, errors="replace")
    (mdir / f"{base}.build.log").write_text(b.stdout + b.stderr, encoding="utf-8")
    if b.returncode:
        note(f"build of {pair.name} failed; see {base}.build.log")
        note("exit 1")
        return 1
    for line in (b.stdout.strip().splitlines())[-4:]:
        comment(f"build: {line.strip()}")
    _, doc = fw.pairing(pair)
    dnet, pnet = doc["dev"].get("net"), doc["prev"].get("net")  # None: that side has no fast NNUE

    shards = [("desktop", dlog, 2 * d_open)] + ([] if a.no_laptop else [("laptop", llog, 2 * l_open)])
    rname = f"split_{a.new}__{a.opp}" + (f"_{a.tag}" if a.tag else "")
    rd = fw.remote_dir(rname)
    if not a.no_laptop:
        os.environ["FAST_WORKER_LOG"] = str(mdir / f"{base}.worker.log")
        try:
            fw.cmd_setup(argparse.Namespace(name=rname, pairdir=str(pair)))
            fw.cmd_start(argparse.Namespace(name=rname, ms=a.ms, threads=a.laptop_threads, offset=a.offset + d_open,
                                            depth=0, games=2 * l_open, env=[]))
        except SystemExit as e:
            note(f"laptop shard not started: {e}")
            note("exit 1")
            return 1
        comment(f"laptop shard: fast_worker setup/start log {base}.worker.log")

    denv = dict(env, SPRT_THINK_MS=str(a.ms), SPRT_THREADS=str(a.desktop_threads), SPRT_GAME_OFFSET=str(a.offset),
                SPRT_LLR_BOUND="100", SPRT_MAX_GAMES=str(2 * d_open))
    shown = " ".join(f"{k}={denv[k]}" for k in sorted(denv) if k.startswith(("SPRT_", "FASTNNUE_")))
    with open(dlog, "w", encoding="utf-8") as fh:
        fh.write(f"# desktop: {shown} {pair / 'test_bots.exe'}\n")
        fh.flush()
        proc = subprocess.Popen([str(pair / "test_bots.exe")], cwd=CPP, env=denv, stdout=fh, stderr=subprocess.STDOUT)
    comment(f"desktop shard: pid {proc.pid}, {shown}")

    def laptop_alive():
        _, o = fw.ssh(f"kill -0 $(cat ~/{rd}/sprt.pid) 2>/dev/null && echo alive || echo dead", check=False)
        return "alive" in o

    def fetch():
        subprocess.run([sw.SCP, *sw.OPTS, "-q", f"{sw.HOST}:{rd}/sprt.log", str(llog)], check=False,
                       stdin=subprocess.DEVNULL)

    shown_n = -1
    while True:
        d_done = proc.poll() is not None
        l_done = True
        if not a.no_laptop:
            fetch()
            l_done = not laptop_alive()
            if l_done:
                fetch()
        n, line = pooled(shards)
        if line and n != shown_n:
            with open(log, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
            shown_n = n
            el = time.time() - t0
            comment(f"[{stamp()}] {n}/{a.games} games; elapsed {el / 60:.1f} min"
                    + (f", ETA {(a.games - n) * el / n / 60:.1f} min" if 0 < n < a.games else ""))
        if d_done and l_done:
            break
        time.sleep(30)

    rc = proc.returncode
    for name, path, planned in shards:
        r = last(path)
        text = Path(path).read_text(encoding="utf-8", errors="replace") if Path(path).exists() else ""
        v = VERDICT.search(text)
        elo = pentanomial_elo(r[2]) if r else (0, 0)
        comment(f"{name} shard: {r[0] if r else 0}/{planned} games, W/D/L {'/'.join(map(str, r[1])) if r else '-'},"
                f" penta {','.join(map(str, r[2])) if r else '-'}, Elo {elo[0]:+.1f} +/- {elo[1]:.1f},"
                f" timeouts {r[3] if r else '-'}, max_ms Prev/Dev {r[4] if r else '-'}; {v.group(0) if v else 'no verdict'}")
        nets = (["--dev", str(dnet)] if dnet else []) + (["--prev", str(pnet)] if pnet else [])
        c = subprocess.run([PY, str(FAST / "fast_pair.py"), "check-log", str(path), *nets], cwd=ROOT,
                           capture_output=True, text=True, errors="replace")
        for line in (c.stdout + c.stderr).strip().splitlines():
            comment(f"{name} check-log: {line.strip()}")
        if c.returncode or not r or r[0] != planned or not v:
            rc = rc or 1
    if not a.no_laptop:
        _, head = fw.ssh(f"head -n 1 ~/{rd}/sprt.log", check=False)
        comment(f"laptop shard header: {head.strip()}")
        fw.ssh(f"rm -rf ~/{rd}", check=False)
    n, line = pooled(shards)
    comment(f"final: {n}/{a.games} games pooled over the shards; fixed length, no early stop")
    with open(log, "a", encoding="utf-8") as fh:
        fh.write((line or "no games") + "\n")
    note(f"exit {rc}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
