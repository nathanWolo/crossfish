#!/usr/bin/env python3
"""Round-robin ranking of eval candidates.

Every pair of candidates plays one fixed-length paired match (no SPRT stop) on
its own range of book openings, and one set of ratings is fitted to all the
pair results at once, so each candidate's rating uses every game it played,
not only its games against the shipped eval.

  plan NAME CAND... [--depth D | --ms MS] [--games-per-pair N] [--base-offset K]
      writes datasets/eval2/rr/NAME/plan.json: every unordered pair (A, B) with
      its own opening range (SPRT_GAME_OFFSET, in opening pairs; N games use
      N/2 openings). CAND is a candidate name (cpp_impl/bin/cand_CAND), a
      candidate directory, or NAME=DIR. Planning an existing tournament again
      with more candidates adds their pairs after the used openings.
  build NAME [--tools]
      builds cpp_impl/bin/rr_A__B/test_bots.exe for every pair: Dev is A,
      Prev is B (eval_candidate.py build --prev-dir). Up-to-date pairs are
      skipped (rr_build.json holds a digest of every input).
  run NAME [--local-threads 6] [--worker] [--worker-threads 4] [--no-local]
      plays the unfinished pairs, one at a time locally and, with --worker,
      one at a time on the Linux worker (tools/sprt_worker.py). Each pair logs
      to datasets/eval2/rr/NAME/A__B.log; a pair whose log has N games is
      done and an interrupted pair resumes from its last line. Progress goes
      to the console and datasets/eval2/rr/NAME/run.log, and ratings.json is
      refitted after every pair.
  rate NAME [--anchor noop]
      fits the ratings (weighted least squares on the pair Elo differences,
      anchor fixed at 0) and writes datasets/eval2/rr/NAME/ratings.json, which
      the lab dashboard shows. Without --anchor the previous anchor is kept
      (noop at first).
  status NAME
      the plan with each pair's state and games.

--root DIR (before the command) keeps tournaments somewhere other than
datasets/eval2/rr.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CPP = ROOT / "cpp_impl"
BIN = CPP / "bin"
TOOLCHAIN = ROOT / "toolchains" / "llvm-mingw-20260616-ucrt-x86_64" / "bin"
sys.path.insert(0, str(HERE))
from sprt_merge import LINE, pentanomial_elo  # noqa: E402

RR_ROOT = ROOT / "datasets" / "eval2" / "rr"
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
EMITTED = ("mini_eval_d16.hpp", "macro_eval.hpp", "hce_weights.json", "dev_patches.json")
VERDICT = re.compile(r"^SPRT (PASS|FAIL|INCONCLUSIVE)", re.M)
Z95 = 1.959963984540054
POLL_LOCAL, POLL_WORKER = 30, 60  # seconds between progress checks
# SPRT_LLR_BOUND for a fixed-length pair. 100 (the screens' value) is reached within a few
# thousand games by a pair a few hundred Elo apart; the pair would stop short, be marked
# failed, and stop again at once on every resume.
NO_STOP_LLR = "1000000000"


# ---------------------------------------------------------------- plan

def book_openings():
    """Openings in cpp_impl/opening_book.bin (CFBOOK header: magic[8], u32 count)."""
    data = (CPP / "opening_book.bin").read_bytes()[:12]
    if not data.startswith(b"CFBOOK"):
        raise SystemExit("cpp_impl/opening_book.bin is not a CFBOOK file")
    return struct.unpack_from("<I", data, 8)[0]


def tour_dir(name):
    return RR_ROOT / name


def rel(p):
    p = Path(p).resolve()
    try:
        return p.relative_to(ROOT).as_posix()
    except ValueError:
        return str(p)


def absdir(s):
    p = Path(s)
    return p if p.is_absolute() else ROOT / p


def load_plan(name):
    path = tour_dir(name) / "plan.json"
    if not path.exists():
        raise SystemExit(f"no plan {path}; run plan first")
    for attempt in range(20):  # a writer may be replacing it right now
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (PermissionError, ValueError):
            if attempt == 19:
                raise
            time.sleep(0.25)


def write_json(path, doc):
    """Replace PATH atomically (readers never see half a file)."""
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    for _ in range(20):  # on Windows a reader (the dashboard) may hold the file open for a moment
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.25)
    os.replace(tmp, path)


def save_plan(plan):
    path = tour_dir(plan["name"]) / "plan.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, plan)


def parse_candidate(spec):
    if "=" in spec:
        name, d = spec.split("=", 1)
        d = absdir(d)
    else:
        d = absdir(spec)
        if d.is_dir() and (d / "mini_eval_d16.hpp").exists():
            name = d.name[5:] if d.name.startswith("cand_") else d.name
        else:
            name, d = spec, BIN / f"cand_{spec}"
    if not NAME_RE.match(name) or "__" in name:
        raise SystemExit(f"candidate name {name!r}: use letters, digits, '_', '.', '-' and no '__'")
    for f in ("mini_eval_d16.hpp", "macro_eval.hpp"):
        if not (d / f).exists():
            raise SystemExit(f"candidate {name}: {d / f} missing (eval_candidate.py emit first)")
    return name, d


def pair_key(p):
    return f"{p['a']}__{p['b']}"


def cmd_plan(a):
    if not NAME_RE.match(a.name):
        raise SystemExit(f"tournament name {a.name!r}: use letters, digits, '_', '.', '-'")
    if a.games_per_pair < 2 or a.games_per_pair % 2:
        raise SystemExit("--games-per-pair must be even (each opening is played with both colours)")
    if a.base_offset < 0:  # test_bots clamps SPRT_GAME_OFFSET at 0, so pair ranges would overlap
        raise SystemExit("--base-offset must be >= 0")
    mode = f"depth {a.depth}" if a.depth else f"{a.ms} ms"
    cands = dict(parse_candidate(s) for s in a.cand)
    if len(cands) != len(a.cand):
        raise SystemExit("duplicate candidate names")
    path = tour_dir(a.name) / "plan.json"
    if path.exists():
        plan = json.loads(path.read_text(encoding="utf-8"))
        if a.base_offset and a.base_offset != plan["base_offset"]:
            print(f"note: {a.name} exists; new pairs follow its used openings, --base-offset is ignored")
        if plan["mode"] != mode or plan["games_per_pair"] != a.games_per_pair:
            raise SystemExit(f"{path} is {plan['mode']}, {plan['games_per_pair']} games per pair; use another NAME")
        for n, d in cands.items():
            if n in plan["candidates"] and absdir(plan["candidates"][n]).resolve() != d.resolve():
                raise SystemExit(f"{n} is {plan['candidates'][n]} in the existing plan, not {rel(d)}")
    else:
        plan = dict(name=a.name, created=time.time(), mode=mode, depth=a.depth or None, ms=None if a.depth else a.ms,
                    games_per_pair=a.games_per_pair, base_offset=a.base_offset, candidates={}, pairs=[])
    for n, d in cands.items():
        plan["candidates"].setdefault(n, rel(d))
    names = list(plan["candidates"])
    have = {frozenset((p["a"], p["b"])) for p in plan["pairs"]}
    half = a.games_per_pair // 2
    nxt = max([p["offset"] + p["games"] // 2 for p in plan["pairs"]], default=plan["base_offset"])
    added = []
    for x, y in itertools.combinations(names, 2):
        if frozenset((x, y)) in have:
            continue
        p = dict(a=x, b=y, offset=nxt, games=a.games_per_pair, dir=rel(BIN / f"rr_{x}__{y}"), state="planned")
        nxt += half
        plan["pairs"].append(p)
        added.append(p)
    book = book_openings()
    if nxt > book:
        raise SystemExit(f"{len(plan['pairs'])} pairs x {half} openings from offset {plan['base_offset']} need"
                         f" openings up to {nxt}, but the book has {book:,} (book wrap is off)")
    save_plan(plan)
    print(f"{path}: {len(names)} candidates, {len(plan['pairs'])} pairs ({len(added)} new), {plan['mode']},"
          f" {plan['games_per_pair']} games per pair, openings {plan['base_offset']}..{nxt - 1} of {book:,}")
    for p in added:
        print(f"  {p['a']} (Dev) vs {p['b']} (Prev): openings {p['offset']}..{p['offset'] + half - 1}")


# ---------------------------------------------------------------- build

INCLUDE = re.compile(r'^\s*#\s*include\s+"([^"]+)"', re.M)


def engine_sources():
    """The cpp_impl files a pairing build compiles: test_bots.cpp, datagen.cpp and bench_ab.cpp,
    the two engine headers (included through macros) and every quoted include of those,
    recursively. The eval headers come from the candidates instead. Unrelated sources (the
    play book, the shipped eval headers) are left out, so editing them does not make every
    pairing out of date in the middle of a tournament."""
    todo, seen = ["test_bots.cpp", "datagen.cpp", "bench_ab.cpp", "crossfish_dev.hpp", "crossfish_prev.hpp"], []
    while todo:
        f = todo.pop()
        if f in seen or f in EMITTED or not (CPP / f).is_file():
            continue
        seen.append(f)
        todo += INCLUDE.findall((CPP / f).read_text(encoding="utf-8", errors="replace"))
    return sorted(seen)


def build_inputs(plan, p):
    """Every file a pairing build reads, as (label, path)."""
    ins = []
    for side in ("a", "b"):
        d = absdir(plan["candidates"][p[side]])
        ins += [(f"{side}/{f}", d / f) for f in EMITTED if (d / f).exists()]
    ins += [(f"cpp/{f}", CPP / f) for f in engine_sources()]
    ins.append(("tools/eval_candidate.py", HERE / "eval_candidate.py"))
    return ins


def build_digest(plan, p):
    h = hashlib.sha256()
    for label, path in build_inputs(plan, p):
        h.update(label.encode() + b"\0" + hashlib.sha256(path.read_bytes()).digest())
    return h.hexdigest()


def is_built(plan, p):
    d = absdir(p["dir"])
    try:
        stamp = json.loads((d / "rr_build.json").read_text())
    except (OSError, ValueError):
        return False
    return (d / "test_bots.exe").exists() and stamp.get("digest") == build_digest(plan, p)


def build_pair(plan, p, tools=False):
    d = absdir(p["dir"])
    src = absdir(plan["candidates"][p["a"]])
    digest = build_digest(plan, p)
    d.mkdir(parents=True, exist_ok=True)
    (d / "rr_build.json").unlink(missing_ok=True)
    for f in EMITTED:  # Dev is A exactly as emitted (a patch file A lacks must not linger)
        if (src / f).exists():
            shutil.copy(src / f, d / f)
        else:
            (d / f).unlink(missing_ok=True)
    cmd = [sys.executable, str(HERE / "eval_candidate.py"), "build", str(d),
           "--prev-dir", str(absdir(plan["candidates"][p["b"]]))] + ([] if tools else ["--no-tools"])
    subprocess.run(cmd, check=True, cwd=ROOT)
    (d / "rr_build.json").write_text(json.dumps(dict(digest=digest, dev=plan["candidates"][p["a"]],
                                                     prev=plan["candidates"][p["b"]], built=time.time()), indent=1))


def cmd_build(a):
    plan = load_plan(a.name)
    todo = [p for p in plan["pairs"] if not is_built(plan, p)]
    print(f"{len(plan['pairs']) - len(todo)} of {len(plan['pairs'])} pairings up to date")
    t0 = time.time()
    for i, p in enumerate(todo):
        eta = f", ETA {fmt_dur((time.time() - t0) / i * (len(todo) - i))}" if i else ""
        print(f"[{stamp()}] building {pair_key(p)} ({i + 1}/{len(todo)}{eta})", flush=True)
        build_pair(plan, p, a.tools)
    plan = load_plan(a.name)  # re-read: a run may have updated states meanwhile
    for p in plan["pairs"]:
        if p["state"] == "planned" and is_built(plan, p):
            p["state"] = "built"
    save_plan(plan)
    print(f"built {len(todo)} pairings")


# ---------------------------------------------------------------- logs

def last_result(path):
    """(games, [W, D, L], penta) of the last result line of a test_bots log, or None."""
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    last = None
    for line in text.splitlines():
        m = LINE.match(line.strip())
        if m:
            last = [int(x) for x in m.groups()]
    return (last[0], last[1:4], last[4:9]) if last else None


def games_done(path):
    r = last_result(path)
    return r[0] if r else 0


def stamp():
    return time.strftime("%H:%M:%S")


def fmt_dur(s):
    s = int(max(0, s))
    return f"{s // 3600}h{s % 3600 // 60:02d}m" if s >= 3600 else f"{s // 60}m{s % 60:02d}s"


class Runner:
    """Shared state of one `run`: the plan on disk, a console + run.log printer and the queue."""

    def __init__(self, name):
        self.name = name
        self.dir = tour_dir(name)
        self.lock = threading.Lock()
        self.log = open(self.dir / "run.log", "a", encoding="utf-8")
        self.claimed = set()
        self.stop = threading.Event()  # Ctrl-C: finish no more pairs, leave remote ones running
        self.t0 = time.time()
        self.games0 = None

    def say(self, msg):
        line = f"[{stamp()}] {msg}"
        with self.lock:
            print(line, flush=True)
            self.log.write(line + "\n")
            self.log.flush()

    def update(self, key, **fields):
        with self.lock:
            plan = load_plan(self.name)
            for p in plan["pairs"]:
                if pair_key(p) == key:
                    p.update(fields)
            save_plan(plan)

    def next_pair(self, where):
        """Claim the next unfinished, built pair in plan order (pairs added to the plan while
        this runs are picked up too). The worker first takes back a pair it left running."""
        stale = []
        with self.lock:
            if self.stop.is_set():
                return None, None
            plan = load_plan(self.name)
            todo = [p for p in plan["pairs"] if pair_key(p) not in self.claimed
                    and games_done(self.dir / f"{pair_key(p)}.log") < p["games"] and p["state"] != "failed"]
            running = [p for p in todo if p["state"] == "running"]  # only the worker's survive cmd_run's reset
            mine = running if where == "worker" else []
            for p in mine + [p for p in todo if p["state"] != "running"]:
                if is_built(plan, p):
                    self.claimed.add(pair_key(p))
                    return plan, p
                stale.append(pair_key(p))
        if stale:  # the sources changed after `build` (say() takes the lock, so outside it)
            self.say(f"{where}: {len(stale)} unfinished pair(s) not built or out of date ({', '.join(stale)});"
                     f" run `round_robin.py build {self.name}`, then `run` again")
        return plan, None

    def release(self, key):
        with self.lock:
            self.claimed.discard(key)

    def progress(self, plan):
        """Overall games played / planned and an ETA from this run's rate."""
        total = sum(p["games"] for p in plan["pairs"])
        done = sum(min(p["games"], games_done(self.dir / f"{pair_key(p)}.log")) for p in plan["pairs"])
        if self.games0 is None:
            self.games0 = done
        speed = (done - self.games0) / max(1.0, time.time() - self.t0)
        eta = f", tournament ETA {fmt_dur((total - done) / speed)}" if speed > 0 else ""
        return f"tournament {done:,}/{total:,} games{eta}"

    def rerate(self):
        try:
            with self.lock:
                rate(self.name, None, quiet=True)
        except Exception as e:  # a refit must not stop the games
            self.say(f"rate failed: {e}")


def resume_env(log, p):
    """Environment that continues LOG's pair where it stopped (test_bots plays batches of whole
    openings in order, so a log's last line always covers openings offset..offset+pairs-1)."""
    r = last_result(log)
    if not r:
        return {}, 0
    n, wdl, penta = r
    return dict(SPRT_RESUME_WINS=str(wdl[0]), SPRT_RESUME_DRAWS=str(wdl[1]), SPRT_RESUME_LOSSES=str(wdl[2]),
                SPRT_RESUME_PENTA=",".join(map(str, penta))), n


def run_local(R, plan, p, threads):
    key, log = pair_key(p), R.dir / f"{pair_key(p)}.log"
    extra, done = resume_env(log, p)
    env = dict(os.environ, PATH=str(TOOLCHAIN) + os.pathsep + os.environ.get("PATH", ""))
    env.update(extra, SPRT_LLR_BOUND=NO_STOP_LLR, SPRT_MAX_GAMES=str(p["games"]), SPRT_THREADS=str(threads),
               SPRT_GAME_OFFSET=str(p["offset"] + done // 2))
    cmd = [str((absdir(p["dir"]) / "test_bots.exe").resolve())]
    if plan["depth"]:
        cmd += ["depth", str(plan["depth"])]
    else:
        env["SPRT_THINK_MS"] = str(plan["ms"])
    shown = {k: env[k] for k in sorted(extra) + ["SPRT_GAME_OFFSET", "SPRT_MAX_GAMES", "SPRT_LLR_BOUND",
                                                 "SPRT_THREADS", "SPRT_THINK_MS"] if k in env}
    (R.dir / f"{key}.log.prefix").unlink(missing_ok=True)  # a worker copy's prefix, now in the log itself
    R.update(key, state="running", where="local", started=time.time())
    R.say(f"local: {key} {'resumes at ' + str(done) if done else 'starts'} ({p['games']} games, {plan['mode']},"
          f" {threads} threads) -> {log}")
    with open(log, "a" if done else "w", encoding="utf-8") as fh:
        fh.write(("# resumed\n" if done else "") + "$ " + " ".join(f"{k}={v}" for k, v in shown.items()) + " "
                 + " ".join(cmd) + "\n")
        fh.flush()
        proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=CPP)
        t0, n0, shown_at = time.time(), done, time.time()
        while proc.poll() is None:
            if R.stop.wait(1):
                proc.terminate()
                proc.wait()
                break
            if time.time() - shown_at < POLL_LOCAL:
                continue
            shown_at = time.time()
            n = games_done(log)
            speed = (n - n0) / max(1.0, time.time() - t0)
            if proc.poll() is None:
                eta = f", ETA {fmt_dur((p['games'] - n) / speed)}" if speed > 0 else ""
                R.say(f"local: {key} {n}/{p['games']} games{eta}; {R.progress(load_plan(R.name))}")
    n = games_done(log)
    ok = n >= p["games"] and VERDICT.search(log.read_text(encoding="utf-8", errors="replace"))
    state = "done" if ok else "interrupted" if R.stop.is_set() else "failed"
    R.update(key, state=state, where="local", finished=time.time())
    R.say(f"local: {key} {state if state != 'failed' else f'FAILED (exit {proc.returncode})'} at {n} games;"
          f" {summary(log)}")
    R.rerate()


def summary(log):
    r = last_result(log)
    if not r:
        return "no games"
    elo, ci = pentanomial_elo(r[2])
    return f"N {r[0]} W/D/L {'/'.join(map(str, r[1]))} Elo {elo:+.1f} +/- {ci:.1f}"


# The remote side: tools/sprt_worker.py ships and builds a pairing dir, then runs test_bots there.
def worker_call(*args):
    subprocess.run([sys.executable, str(HERE / "sprt_worker.py"), *map(str, args)], check=True, cwd=ROOT)


def remote_name(tname, key):
    return f"rr_{tname}__{key}"


class WorkerUnreachable(RuntimeError):
    """The worker did not answer; its pair may still be playing and keeps its running state."""


def run_worker(R, plan, p, threads):
    import sprt_worker as sw
    key, log = pair_key(p), R.dir / f"{pair_key(p)}.log"
    prefix_file = R.dir / f"{key}.log.prefix"
    rname = remote_name(plan["name"], key)
    rd = sw.remote_dir(rname)

    def alive():
        """True / False, or None when the worker cannot be reached."""
        out = sw.ssh(f"kill -0 $(cat ~/{rd}/sprt.pid) 2>/dev/null && echo rr-alive || echo rr-dead", check=False,
                     capture=True) or ""
        return True if "rr-alive" in out else False if "rr-dead" in out else None

    def fetch():
        """Local log = what came before the remote run (the prefix) + the remote log."""
        tmp = R.dir / f"{key}.log.remote"
        r = subprocess.run([sw.SCP, *sw.OPTS, "-q", f"{sw.HOST}:{rd}/sprt.log", str(tmp)], check=False)
        if r.returncode == 0 and tmp.exists():
            pre = prefix_file.read_text(encoding="utf-8") if prefix_file.exists() else ""
            log.write_text(pre + tmp.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")
            tmp.unlink()
        return games_done(log)

    def finish(n):
        text = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
        ok = n >= p["games"] and VERDICT.search(text)
        if ok:
            prefix_file.unlink(missing_ok=True)
            sw.ssh(f"rm -rf ~/{rd}", check=False)  # the pairing's remote sources and binaries; the log is local now
        R.update(key, state="done" if ok else "failed", where="worker", finished=time.time())
        R.say(f"worker: {key} {'done' if ok else f'FAILED (remote run ended at {n} games)'}; {summary(log)}")
        R.rerate()

    up = alive()  # this pair's own remote directory: a live process there is this pair's run
    if up is None:
        raise WorkerUnreachable(f"cannot reach the worker ({sw.HOST})")
    if not up and p["state"] == "running" and p.get("where") == "worker" and prefix_file.exists():
        # Left running remotely by an earlier run that has since ended or died: its remote log
        # may hold games the local copy lacks (all of them, if it finished). Collect it before
        # setup below deletes the remote directory. Nothing else touched this pair meanwhile:
        # only the worker takes a pair in the running state.
        n = fetch()
        if n >= p["games"]:
            R.say(f"worker: {key} ended remotely while detached; collected {n} games")
            finish(n)
            return
    if up:
        R.update(key, state="running", where="worker", remote=rname)
        R.say(f"worker: {key} still running remotely ({rname}); reattaching")
    else:
        extra, done = resume_env(log, p)
        prefix = (log.read_text(encoding="utf-8", errors="replace") + "# resumed on the worker\n") if done else ""
        prefix_file.write_text(prefix, encoding="utf-8")
        R.say(f"worker: {key} setup ({rname})")
        worker_call("setup", rname, absdir(p["dir"]))
        mode = ["--depth", plan["depth"]] if plan["depth"] else ["--ms", plan["ms"]]
        extra = dict(extra, SPRT_LLR_BOUND=NO_STOP_LLR)  # replaces sprt_worker.py's default of 100
        envs = [x for k, v in extra.items() for x in ("--env", f"{k}={v}")]
        worker_call("start", rname, *mode, "--threads", threads, "--offset", p["offset"] + done // 2,
                    "--games", p["games"], *envs)
        R.update(key, state="running", where="worker", remote=rname, started=time.time())
        R.say(f"worker: {key} {'resumes at ' + str(done) if done else 'starts'} ({p['games']} games,"
              f" {plan['mode']}, {threads} threads) -> {log}")
    t0, n0, unreachable = time.time(), None, 0
    while True:
        if R.stop.wait(POLL_WORKER):
            fetch()
            R.say(f"worker: {key} left running remotely ({rname}); a new run with --worker reattaches")
            return
        n = fetch()
        n0 = n if n0 is None else n0
        text = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
        finished = n >= p["games"] and VERDICT.search(text)
        up = alive()
        unreachable = unreachable + 1 if up is None else 0
        if unreachable >= 30:  # half an hour without an answer
            raise WorkerUnreachable(f"no answer from the worker for 30 polls; {rname} may still be running")
        if finished or up is False:
            n = fetch()
            break
        speed = (n - n0) / max(1.0, time.time() - t0)
        eta = f", ETA {fmt_dur((p['games'] - n) / speed)}" if speed > 0 else ""
        R.say(f"worker: {key} {n}/{p['games']} games{eta}; {R.progress(load_plan(R.name))}")
    finish(n)


def executor(R, where, fn, threads):
    """One pair at a time. A pair that plays out but falls short is marked failed and the next one
    starts; an error in the machinery (SSH, worker setup, a missing binary) hands the pair back to
    the queue and stops this executor, so an unreachable worker cannot fail the whole plan."""
    while True:
        plan, p = R.next_pair(where)
        if p is None:
            return
        try:
            fn(R, plan, p, threads)
        except Exception as e:  # noqa: BLE001
            if not isinstance(e, WorkerUnreachable):  # a pair that may still be playing stays "running"
                R.update(pair_key(p), state="interrupted", where=where)
            R.release(pair_key(p))
            R.say(f"{where}: {pair_key(p)} not finished: {type(e).__name__}: {e}; the {where} executor stops")
            return


def cmd_run(a):
    plan = load_plan(a.name)
    R = Runner(a.name)
    for p in plan["pairs"]:
        # A new run retries failed pairs from their logs. A local pair still marked running
        # belonged to a run that died (its games resume from the log); a worker pair may still
        # be playing remotely, and the worker executor reattaches to it.
        if p["state"] == "failed" or (p["state"] == "running" and p.get("where") != "worker"):
            R.update(pair_key(p), state="interrupted")
        if p["state"] == "running" and p.get("where") == "worker" and not a.worker:
            print(f"note: {pair_key(p)} was running on the worker; pass --worker to reattach to it")
    plan = load_plan(a.name)
    unbuilt = [pair_key(p) for p in plan["pairs"]
               if games_done(R.dir / f"{pair_key(p)}.log") < p["games"] and not is_built(plan, p)]
    if unbuilt:
        raise SystemExit(f"not built or out of date: {', '.join(unbuilt)}; run `round_robin.py build {a.name}`")
    R.say(f"run {a.name}: {len(plan['pairs'])} pairs, {plan['mode']}; {R.progress(plan)}"
          f"{'; local ' + str(a.local_threads) + ' threads' if not a.no_local else ''}"
          f"{'; worker ' + str(a.worker_threads) + ' threads' if a.worker else ''}")
    threads = []
    if a.worker:
        threads.append(threading.Thread(target=executor, args=(R, "worker", run_worker, a.worker_threads)))
    if not a.no_local:
        threads.append(threading.Thread(target=executor, args=(R, "local", run_local, a.local_threads)))
    for t in threads:
        t.daemon = True
        t.start()
    try:
        while any(t.is_alive() for t in threads):
            time.sleep(0.5)
    except KeyboardInterrupt:
        R.stop.set()
        R.say("interrupted: stopping the local game, leaving any worker pair running")
        for t in threads:
            t.join()
    plan = load_plan(a.name)
    left = [pair_key(p) for p in plan["pairs"] if games_done(R.dir / f"{pair_key(p)}.log") < p["games"]]
    R.say(f"run {a.name} stopped: {len(plan['pairs']) - len(left)}/{len(plan['pairs'])} pairs complete"
          + (f"; unfinished: {', '.join(left)}" if left else ""))
    rate(a.name, None)
    if left:
        sys.exit(1)


# ---------------------------------------------------------------- rate

def pair_estimate(penta):
    """(elo, ci, se_for_fit, regularized): the pair's Elo and 95% half-width exactly as
    sprt_merge.pentanomial_elo (test_bots's calc_pentanomial_elo) computes them; the
    fit uses se = ci / 1.96. A degenerate sample (every pair the same outcome, or a
    0% / 100% score) has no usable variance, so it is fitted with half a pair added
    to every outcome."""
    elo, ci = pentanomial_elo(penta)
    if ci > 0 and math.isfinite(ci) and abs(elo) < 1500:
        return elo, ci, ci / Z95, False
    e2, c2 = pentanomial_elo([x + 0.5 for x in penta])
    return elo, ci, c2 / Z95, True


def wls(names, anchor, obs):
    """Ratings r (r[anchor] = 0) minimizing sum w (y - (r_a - r_b))^2 over OBS = [(a, b, y, se)].
    Returns ({name: (rating, se)}, chi2, dof); engines with no path of games to the anchor get None."""
    import numpy as np
    adj = {n: set() for n in names}
    for a, b, _, _ in obs:
        adj[a].add(b)
        adj[b].add(a)
    seen, stack = {anchor}, [anchor]
    while stack:
        for m in adj[stack.pop()] - seen:
            seen.add(m)
            stack.append(m)
    free = [n for n in names if n in seen and n != anchor]
    idx = {n: i for i, n in enumerate(free)}
    rows = [(a, b, y, se) for a, b, y, se in obs if a in seen and b in seen]
    out = {n: (None, None) for n in names}
    out[anchor] = (0.0, 0.0)
    if not free:
        return out, 0.0, 0
    X = np.zeros((len(rows), len(free)))
    y = np.array([r[2] for r in rows])
    w = np.array([1.0 / r[3] ** 2 for r in rows])
    for k, (a, b, _, _) in enumerate(rows):
        if a in idx:
            X[k, idx[a]] = 1.0
        if b in idx:
            X[k, idx[b]] = -1.0
    A = X.T @ (w[:, None] * X)
    cov = np.linalg.inv(A)
    beta = cov @ (X.T @ (w * y))
    resid = y - X @ beta
    for n, i in idx.items():
        out[n] = (float(beta[i]), float(math.sqrt(cov[i, i])))
    return out, float((w * resid ** 2).sum()), len(rows) - len(free)


def rate(name, anchor, quiet=False):
    plan = load_plan(name)
    d = tour_dir(name)
    names = list(plan["candidates"])
    if not anchor:  # keep an anchor chosen with `rate --anchor` through run's refits
        try:
            anchor = json.loads((d / "ratings.json").read_text(encoding="utf-8")).get("anchor")
        except (OSError, ValueError, AttributeError):
            anchor = None
        anchor = anchor if anchor in names else None
    anchor = anchor or ("noop" if "noop" in names else names[0])
    if anchor not in names:
        raise SystemExit(f"anchor {anchor} is not in {name}: {', '.join(names)}")
    pairs, obs, games = [], [], {n: 0 for n in names}
    for p in plan["pairs"]:
        r = last_result(d / f"{pair_key(p)}.log")
        entry = dict(a=p["a"], b=p["b"], games=0, target=p["games"], elo=None, ci=None, penta=None,
                     state=p.get("state"))
        if r and r[0] > 0:
            elo, ci, se, reg = pair_estimate(r[2])
            entry.update(games=r[0], elo=round(elo, 2), ci=round(ci, 2), penta=r[2], wdl=r[1])
            if reg:
                entry["regularized"] = True
            fit_elo = pentanomial_elo([x + 0.5 for x in r[2]])[0] if reg else elo
            if reg:
                entry["fit_elo"] = round(fit_elo, 2)
            obs.append((p["a"], p["b"], fit_elo, se))
            games[p["a"]] += r[0]
            games[p["b"]] += r[0]
        pairs.append(entry)
    fit, chi2, dof = wls(names, anchor, obs)
    engines = [dict(name=n, rating=None if fit[n][0] is None else round(fit[n][0], 2),
                    ci=None if fit[n][1] is None else round(Z95 * fit[n][1], 2), games=games[n]) for n in names]
    engines.sort(key=lambda e: (e["rating"] is None, -(e["rating"] or 0)))
    doc = dict(name=name, anchor=anchor, mode=plan["mode"], games_per_pair=plan["games_per_pair"], engines=engines,
               pairs=pairs, fit=dict(chi2=round(chi2, 3), dof=dof), updated=time.time())
    write_json(d / "ratings.json", doc)
    if not quiet:
        print_ratings(doc)
    return doc


def print_ratings(doc):
    played = sum(1 for p in doc["pairs"] if p["games"])
    full = sum(1 for p in doc["pairs"] if p["games"] >= p["target"])
    print(f"{doc['name']}: {doc['mode']}, anchor {doc['anchor']} = 0; {played}/{len(doc['pairs'])} pairs played,"
          f" {full} complete; fit chi2 {doc['fit']['chi2']:.2f} on {doc['fit']['dof']} dof")
    w = max(len(e["name"]) for e in doc["engines"])
    print(f"  {'#':>2}  {'engine':<{w}}  {'rating':>7}  {'95% CI':>7}  {'games':>7}")
    for i, e in enumerate(doc["engines"]):
        r = "   n/a" if e["rating"] is None else f"{e['rating']:+7.1f}"
        c = "" if e["ci"] is None else f"+/-{e['ci']:.1f}"
        print(f"  {i + 1:>2}  {e['name']:<{w}}  {r:>7}  {c:>7}  {e['games']:>7}")
    order = [e["name"] for e in doc["engines"]]
    h2h = {}
    for p in doc["pairs"]:
        if p["games"]:
            h2h[(p["a"], p["b"])] = p["elo"]
            h2h[(p["b"], p["a"])] = -p["elo"]
    short = [n[:6] for n in order]
    print("  head to head (row's Elo vs column):")
    print(f"  {'':<{w}}  " + " ".join(f"{s:>6}" for s in short))
    for n in order:
        cells = ["     ." if m == n else (f"{h2h[(n, m)]:+6.0f}" if (n, m) in h2h else "     -") for m in order]
        print(f"  {n:<{w}}  " + " ".join(cells))


def cmd_rate(a):
    rate(a.name, a.anchor)


def cmd_status(a):
    plan = load_plan(a.name)
    d = tour_dir(a.name)
    print(f"{a.name}: {plan['mode']}, {plan['games_per_pair']} games per pair, {len(plan['candidates'])} candidates")
    for p in plan["pairs"]:
        log = d / f"{pair_key(p)}.log"
        print(f"  {pair_key(p):<40} {p['state']:<8} {p.get('where', ''):<6} openings {p['offset']:>6}+"
              f"{p['games'] // 2:<5} {games_done(log):>6}/{p['games']:<6} {summary(log) if log.exists() else ''}")


def main():
    global RR_ROOT
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", help="tournament directory (default datasets/eval2/rr)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan"); p.add_argument("name"); p.add_argument("cand", nargs="+")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--depth", type=int, default=0, help="equal-depth games (default: 90 ms timed games)")
    g.add_argument("--ms", type=int, default=90)
    p.add_argument("--games-per-pair", type=int, default=1000)
    p.add_argument("--base-offset", type=int, default=0, help="first opening (in pairs) of the first pair")
    p.set_defaults(fn=cmd_plan)
    p = sub.add_parser("build"); p.add_argument("name")
    p.add_argument("--tools", action="store_true", help="also compile datagen and bench_ab")
    p.set_defaults(fn=cmd_build)
    p = sub.add_parser("run"); p.add_argument("name")
    p.add_argument("--local-threads", type=int, default=6)
    p.add_argument("--worker", action="store_true", help="also play pairs on the Linux worker (sprt_worker.py)")
    p.add_argument("--worker-threads", type=int, default=4)
    p.add_argument("--no-local", action="store_true", help="play on the worker only")
    p.set_defaults(fn=cmd_run)
    p = sub.add_parser("rate"); p.add_argument("name"); p.add_argument("--anchor")
    p.set_defaults(fn=cmd_rate)
    p = sub.add_parser("status"); p.add_argument("name"); p.set_defaults(fn=cmd_status)
    a = ap.parse_args()
    if a.root:
        RR_ROOT = absdir(a.root)
    if a.cmd == "run" and a.no_local and not a.worker:
        raise SystemExit("--no-local needs --worker")
    if a.cmd == "run" and a.worker and not os.environ.get("CROSSFISH_WORKER"):
        raise SystemExit("--worker needs CROSSFISH_WORKER=user@host (see tools/sprt_worker.py)")
    a.fn(a)


if __name__ == "__main__":
    main()
