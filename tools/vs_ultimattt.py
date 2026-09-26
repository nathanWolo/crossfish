#!/usr/bin/env python3
"""Crossfish (the CodinGame bot) vs nelhage/ultimattt's minimax engine, scored under CodinGame rules.

  vs_ultimattt.py PAIRS CF_MS UT_MS WORKERS LOG [--plies 4] [--seed 1]

Each pair plays one opening twice with colours swapped. Openings are PLIES
uniform-random legal moves from a seeded generator. Crossfish runs through its
match protocol (crossfish_cg_meta.exe: NEW / APPLY / GO ms), ultimattt through
its JSON worker (`ultimattt worker`: newgame / getmove with a time limit), and
the uttt.ai fork's utttpy judges every move and the result, including the
CodinGame miniboard-count tiebreak. ultimattt itself plays standard rules (a
full board with no line is a draw to it). Progress with an ETA goes to LOG.
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import random
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
# The uttt.ai fork and ultimattt are expected next to this repository (override with env vars).
UTTTAI = Path(os.environ.get("UTTTAI_ROOT", str(HERE.parent.parent / "utttai")))
sys.path.insert(0, str(UTTTAI))
from sprt_merge import pentanomial_elo  # noqa: E402
from utttpy.game.action import Action  # noqa: E402
from utttpy.game.ultimate_tic_tac_toe import UltimateTicTacToe  # noqa: E402

# crossfish_cg_meta.exe is built by the fork's cg/tools/make_crossfish_variants.py.
CF_EXE = Path(os.environ.get("CF_EXE", str(UTTTAI / "cg" / "crossfish_cg_meta.exe")))
UT_EXE = Path(os.environ.get("UT_EXE", str(HERE.parent.parent / "ultimattt" / "target" / "release" / "ultimattt.exe")))
TOOLCHAIN = HERE.parent / "toolchains" / "llvm-mingw-20260616-ucrt-x86_64" / "bin"


def ut_board(u: UltimateTicTacToe) -> str:
    """ultimattt notation: side;global (with @ on the forced board);9 local boards of 9 cells, a..i row-major."""
    sym = {0: ".", 1: "X", 2: "O"}
    glob = "".join({0: ".", 1: "X", 2: "O", 3: "#"}[v] for v in u.state[81:90])
    if u.constraint != 9:
        glob = glob[:u.constraint] + "@" + glob[u.constraint + 1:]
    locs = "/".join("".join(sym[u.state[b * 9 + s]] for s in range(9)) for b in range(9))
    return f"{'X' if u.next_symbol == 1 else 'O'};{glob};{locs}"


class Engines:
    def __init__(self, cf_ms, ut_ms):
        env = dict(os.environ, PATH=str(TOOLCHAIN) + os.pathsep + os.environ.get("PATH", ""))
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self.cf = self.ut = None
        self.cf_ms, self.ut_ms = cf_ms, ut_ms
        self.ut_limit = {"secs": ut_ms // 1000, "nanos": (ut_ms % 1000) * 1_000_000}
        try:
            self.cf = subprocess.Popen([str(CF_EXE), "match"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                       stderr=subprocess.DEVNULL, text=True, bufsize=1, env=env, creationflags=flags)
            self.ut = subprocess.Popen([str(UT_EXE), "worker"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                       stderr=subprocess.DEVNULL, text=True, bufsize=1, creationflags=flags)
            # ultimattt allocates its 1 GiB table when it builds its searcher, and builds a new one
            # whenever the config (the time limit included) changes, so warm up with the games' limit.
            self.ut_call({"op": "getmove", "body": {"id": "warmup", "board": ut_board(UltimateTicTacToe()),
                                                  "limit": self.ut_limit, "max_depth": None}})
        except BaseException:
            self.close()
            raise

    def cf_send(self, line):
        self.cf.stdin.write(line + "\n")
        self.cf.stdin.flush()

    def ut_call(self, cmd):
        self.ut.stdin.write(json.dumps(cmd) + "\n")
        self.ut.stdin.flush()
        line = self.ut.stdout.readline()
        if not line:
            raise RuntimeError("ultimattt exited")
        return json.loads(line)

    def play(self, opening, cf_is_x, game_id):
        u = UltimateTicTacToe()
        self.cf_send("NEW")
        if self.cf.stdout.readline().strip() != "READY":
            raise RuntimeError("crossfish did not answer NEW")
        self.ut_call({"op": "newgame", "body": {"id": str(game_id), "player": "X" if not cf_is_x else "O"}})
        for idx in opening:
            u.execute(Action(symbol=u.next_symbol, index=idx))
            self.cf_send(f"APPLY {idx // 9} {idx % 9}")
        times = {"cf": [], "ut": []}
        while not u.is_terminated():
            cf_turn = (u.next_symbol == 1) == cf_is_x
            t0 = time.perf_counter()
            if cf_turn:
                self.cf_send(f"GO {self.cf_ms}")
                reply = self.cf.stdout.readline().split()
                if len(reply) < 2:
                    raise RuntimeError(f"crossfish exited or sent {reply!r}")
                mb, sq = int(reply[0]), int(reply[1])
                idx = mb * 9 + sq if 0 <= mb < 9 and 0 <= sq < 9 else -1
            else:
                r = self.ut_call({"op": "getmove", "body": {"id": str(game_id), "board": ut_board(u),
                                                          "limit": self.ut_limit, "max_depth": None}})
                if r.get("op") != "move":
                    raise RuntimeError(f"ultimattt error: {r}")
                mv = r["body"]["move"]
                ok = len(mv) == 2 and all("a" <= c <= "i" for c in mv)
                idx = (ord(mv[0]) - 97) * 9 + (ord(mv[1]) - 97) if ok else -1
            times["cf" if cf_turn else "ut"].append((time.perf_counter() - t0) * 1000)
            # Judge before crossfish sees ultimattt's move: its match loop applies whatever it is sent.
            if idx not in u.get_legal_indexes():
                return ("illegal", "cf" if cf_turn else "ut", times)
            if not cf_turn:
                self.cf_send(f"APPLY {idx // 9} {idx % 9}")
            u.execute(Action(symbol=u.next_symbol, index=idx))
        if u.is_result_draw():
            return (0.5, None, times)
        cf_won = u.is_result_X() == cf_is_x
        return (1.0 if cf_won else 0.0, None, times)

    def close(self):
        for p in (self.cf, self.ut):
            if p is None:
                continue
            try:
                p.kill()
                p.wait(timeout=10)
            except (OSError, subprocess.TimeoutExpired):
                pass


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pairs", type=int)
    ap.add_argument("cf_ms", type=int)
    ap.add_argument("ut_ms", type=int)
    ap.add_argument("workers", type=int)
    ap.add_argument("log")
    ap.add_argument("--plies", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    openings = []
    for _ in range(a.pairs):
        u = UltimateTicTacToe()
        seq = []
        for _ in range(a.plies):
            idx = rng.choice(u.get_legal_indexes())
            u.execute(Action(symbol=u.next_symbol, index=idx))
            seq.append(idx)
        openings.append(seq)
    jobs = queue.Queue()
    for k, op in enumerate(openings):
        jobs.put((k, op))
    lock = threading.Lock()
    penta = [0] * 5
    wdl = [0, 0, 0]
    illegal = []
    maxt = {"cf": 0.0, "ut": 0.0}
    tsum = {"cf": [0.0, 0], "ut": [0.0, 0]}
    over = {"cf": 0, "ut": 0}
    t_start = time.time()
    log = open(a.log, "a", encoding="utf-8")

    def out(line):
        line = f"[{time.strftime('%H:%M:%S')}] {line}"
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    out(f"crossfish {a.cf_ms} ms (CodinGame bot) vs ultimattt minimax {a.ut_ms} ms: {a.pairs} pairs,"
        f" {a.plies}-ply random openings, {a.workers} workers, CodinGame rules")

    failed = []  # (pair, error): an engine died or answered garbage; the pair is not scored

    def worker():
        eng = None
        try:
            while True:
                try:
                    k, op = jobs.get_nowait()
                except queue.Empty:
                    return
                try:
                    if eng is None:
                        eng = Engines(a.cf_ms, a.ut_ms)
                    results = [eng.play(op, cf_is_x, 2 * k + (0 if cf_is_x else 1)) for cf_is_x in (True, False)]
                except Exception as e:  # restart both engines and go on with the next pair
                    if eng is not None:
                        eng.close()
                        eng = None
                    with lock:
                        failed.append((k, repr(e)))
                        out(f"pair {k} failed: {e!r}; engines restarted")
                    continue
                scores = []
                for cf_is_x, (s, who, times) in zip((True, False), results):
                    with lock:
                        for side in ("cf", "ut"):
                            if times[side]:
                                maxt[side] = max(maxt[side], max(times[side]))
                                tsum[side][0] += sum(times[side])
                                tsum[side][1] += len(times[side])
                                budget = a.cf_ms if side == "cf" else a.ut_ms
                                over[side] += sum(t > budget + 100 for t in times[side])
                    if s == "illegal":
                        with lock:
                            illegal.append((k, cf_is_x, who))
                        s = 0.0 if who == "cf" else 1.0
                    scores.append(s)
                with lock:
                    for s in scores:
                        wdl[0 if s == 1.0 else 1 if s == 0.5 else 2] += 1
                    penta[int(round(sum(scores) * 2))] += 1
                    n = sum(penta)
                    if n % 10 == 0 or n + len(failed) == a.pairs:
                        elo, ci = pentanomial_elo(penta)
                        rate = (n + len(failed)) / max(1e-9, time.time() - t_start)
                        eta = (a.pairs - n - len(failed)) / rate
                        out(f"{2 * n}/{2 * a.pairs} games  crossfish W/D/L {wdl[0]}/{wdl[1]}/{wdl[2]}"
                            f"  Elo {elo:+.1f} +/- {ci:.1f}  penta {','.join(map(str, penta))}"
                            f"  move ms avg/max cf {tsum['cf'][0] / max(1, tsum['cf'][1]):.0f}/{maxt['cf']:.0f}"
                            f" ut {tsum['ut'][0] / max(1, tsum['ut'][1]):.0f}/{maxt['ut']:.0f}"
                            f"  illegal {len(illegal)}  failed pairs {len(failed)}  ETA {eta / 60:.0f}m")
        finally:
            if eng is not None:
                eng.close()

    threads = [threading.Thread(target=worker) for _ in range(a.workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if not sum(penta):
        out(f"FINAL crossfish vs ultimattt: no pair finished; failed pairs {failed}")
        return 1
    elo, ci = pentanomial_elo(penta)
    out(f"FINAL crossfish vs ultimattt: {2 * sum(penta)} games, W/D/L {wdl[0]}/{wdl[1]}/{wdl[2]},"
        f" Elo {elo:+.1f} +/- {ci:.1f} (CodinGame rules); illegal moves {illegal}; moves > budget+100ms {over};"
        f" failed pairs {failed}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
