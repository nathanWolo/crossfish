#!/usr/bin/env python3
"""Fast-NNUE pairings on the Linux laptop worker (nnue2, full-data play stage).

round_robin.py run --worker plays a pairing remotely through tools/sprt_worker.py, which ships the
pairing's sources but not its nets, and passes only SPRT_* variables to test_bots. A fast pairing
(fast_pair.py) compiles its nets in as Windows paths, so on Linux each engine needs its net from the
environment. `fast_pair.py rr-run NAME --worker ...` swaps round_robin's worker_call for this script,
so that for every pairing the worker does the following.

  setup NAME PAIRDIR
      1. Nets, by content. Each fast side's net (from PAIRDIR/fast_pair.json; its CRC-32 must still
         be the one the pairing was built with) goes to ~/crossfish_worker/nets/CRC32/FILE. A net
         already there, or in the old flat ~/crossfish_worker/nets/FILE with that CRC, is not copied
         again; a copied one is CRC-checked on arrival before it is renamed into place.
      2. Sources. cpp_impl's engine sources and the pairing directory (Dev's fast_nnue*.hpp, Prev's
         renamed prev_fast_nnue*.hpp, the generated crossfish_{dev,prev}.hpp, test_bots.cpp), plus
         pair_statics.cpp, the 1,000 test positions and cand/nets.env:
            FASTNNUE_PATH=<Dev's net>   FASTNNUE_PREV_PATH=<Prev's net>   (only for a fast side)
         to ~/crossfish_worker/NAME/ (the directory round_robin's worker loop polls).
      3. g++ builds of test_bots and pair_statics (sprt_worker.FLAGS), in parallel.
      4. test_bots verify (the self-tests) with nets.env.
      5. Statics. pair_statics (fresh Dev and Prev engines, static eval of each of 1,000 positions,
         nets.env) must give, for every position, Dev's static equal to the Dev candidate's own
         single local build (`datagen label POS OUT 0 1` of its candidate directory) and Prev's
         equal to the Prev candidate's. Both engines' net lines must name the planned file and CRC.
         The two references differ on most positions, so a swapped or shared net cannot pass.
      Any failure exits non-zero: round_robin hands the pairing back and stops the worker executor.
  start NAME [--ms MS | --depth D] [--threads 4] [--offset K] [--games N] [--env SPRT_X=V ...]
      sprt_worker.py start's arguments. Starts test_bots under nohup with those SPRT_* settings and
      nets.env; log ~/crossfish_worker/NAME/sprt.log, pid in sprt.pid: where round_robin's worker loop
      looks for them. The log's first line is "# worker HOST (g++ V): VARS test_bots ARGS", where
      VARS is every FASTNNUE_* and SPRT_* variable in test_bots's own environment, sorted, as written
      by the launcher that then execs test_bots (so it is the environment the engines read, not the
      command line). start then compares VARS, and the running process's /proc/PID/environ, with
      exactly the planned settings plus nets.env, and stops the run if anything else is there.

Every remote engine run (verify, pair_statics, test_bots) starts from the worker's login environment
with the four variables the engines read removed (`env -u FASTNNUE_BAKE -u FASTNNUE_PREV_BAKE -u
FASTNNUE_PATH -u FASTNNUE_PREV_PATH`); nets.env then sets the net path of each fast side. A
FASTNNUE_BAKE exported on the laptop would otherwise make Dev re-bake its tables in C++ (review 5).
  stop NAME
  refs PAIRDIR
      only the local reference statics of PAIRDIR's two sides (step 5's references), cached in
      datasets/nnue2/fast/worker/refs/.

Host and key as sprt_worker.py: CROSSFISH_WORKER=user@host (required) and CROSSFISH_WORKER_KEY (default
~/.ssh/crossfish_worker).
Progress lines also go to $FAST_WORKER_LOG when set (fast_pair.py rr-run sets it to the tournament's
worker.log). Run with the Python that has numpy (toolchains/py312-dml) like fast_pair.py; this script
itself needs only the standard library.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import shlex
import struct
import subprocess
import sys
import tarfile
import time
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CPP = ROOT / "cpp_impl"
TOOLCHAIN = ROOT / "toolchains" / "llvm-mingw-20260616-ucrt-x86_64" / "bin"
sys.path.insert(0, str(ROOT / "tools"))
import sprt_worker as sw  # noqa: E402  (reads CROSSFISH_WORKER at import)

POSITIONS = ROOT / "datasets" / "nnue2" / "fast" / "iso_pos1000.cfdg"  # made by ensure_positions() if missing
PARITY = ROOT / "datasets" / "nnue2" / "fnn1" / "parity_in.cfdg"
REF_DIR = ROOT / "datasets" / "nnue2" / "fast" / "worker" / "refs"
REMOTE_NETS = "crossfish_worker/nets"
REC = 128
NET_LINE = re.compile(r"fast_nnue \[(Dev|Prev)\]: net (.*) \((.*)\), (\d+) bytes, crc32 ([0-9a-f]{8})")
SIDE_ENV = {"dev": "FASTNNUE_PATH", "prev": "FASTNNUE_PREV_PATH"}
KSIDE = {"dev": "Dev", "prev": "Prev"}
# The environment variables the two engines read (fast_nnue.hpp: FASTNNUE_PATH; fast_nnue_b.hpp:
# FASTNNUE_BAKE; Prev's copies under the PREV_ names). Every remote engine run drops them first; nets.env
# then sets the path of each fast side. (review 5: the worker passed its login environment through.)
ENGINE_ENV = ("FASTNNUE_BAKE", "FASTNNUE_PREV_BAKE", "FASTNNUE_PATH", "FASTNNUE_PREV_PATH")
CLEAN_ENV = "env " + " ".join(f"-u {v}" for v in ENGINE_ENV)
RECORDED = re.compile(r"^(?:FASTNNUE|SPRT)_[A-Za-z0-9_]*=")
# Written to ~/crossfish_worker/NAME/cand/launch.sh by start and run from cpp_impl/ under nohup: the
# first log line records the FASTNNUE_* / SPRT_* environment of this very process, which then becomes
# test_bots (exec keeps the pid and the environment).
LAUNCH = """#!/bin/sh
{ printf '# worker %s (g++ %s): ' '@HOST@' "$(cat ../cand/gxx_version)"
  env | grep -E '^(FASTNNUE|SPRT)_' | LC_ALL=C sort | tr '\\n' ' '
  printf 'test_bots@ARGS@\\n'; } > ../sprt.log
exec ../cand/test_bots@ARGS@ >> ../sprt.log 2>&1 < /dev/null
"""

# Runs on the worker (python3 -): for each CRC/FILE argument, is ~/crossfish_worker/nets/CRC/FILE in
# place with that CRC? Takes the old flat nets/FILE, or a finished upload nets/CRC/FILE.part, when its
# CRC matches. Prints "home DIR" then "have|copied|arrived|need CRC/FILE" per net.
NET_SCRIPT = r'''
import os, shutil, sys, zlib
def crc(p):
    c = 0
    with open(p, "rb") as f:
        while True:
            b = f.read(1 << 22)
            if not b:
                return "%08x" % (c & 0xffffffff)
            c = zlib.crc32(b, c)
home = os.path.expanduser("~")
base = os.path.join(home, "crossfish_worker", "nets")
print("home", home)
for arg in sys.argv[1:]:
    want, name = arg.split("/", 1)
    dst = os.path.join(base, want, name)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst) and crc(dst) == want:
        print("have", arg)
        continue
    part = dst + ".part"
    if os.path.exists(part) and crc(part) == want:
        os.replace(part, dst)
        print("arrived", arg)
        continue
    flat = os.path.join(base, name)
    if os.path.exists(flat) and crc(flat) == want:
        shutil.copyfile(flat, part)
        os.replace(part, dst)
        print("copied", arg)
        continue
    print("need", arg)
'''


def stamp():
    return time.strftime("%H:%M:%S")


def say(msg):
    line = f"[{stamp()}] fast_worker: {msg}"
    print(line, flush=True)
    if os.environ.get("FAST_WORKER_LOG"):
        with open(os.environ["FAST_WORKER_LOG"], "a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def absdir(s):
    p = Path(s)
    return p if p.is_absolute() else ROOT / p


def file_crc(path):
    c = 0
    with open(path, "rb") as fh:
        while chunk := fh.read(1 << 22):
            c = zlib.crc32(chunk, c)
    return f"{c & 0xFFFFFFFF:08x}"


def require_host():
    if not sw.HOST:
        raise SystemExit("no worker: set CROSSFISH_WORKER=user@host (and CROSSFISH_WORKER_KEY if the key is not"
                         " ~/.ssh/crossfish_worker), as for tools/sprt_worker.py")


def ssh(cmd, check=True, capture=True, input=None):
    """Run CMD on the worker; returns (exit code, stdout+stderr text)."""
    require_host()
    io_args = dict(input=input) if input is not None else dict(stdin=subprocess.DEVNULL)
    r = subprocess.run([sw.SSH, *sw.OPTS, "-o", "ServerAliveInterval=30", sw.HOST, cmd], **io_args,
                       stdout=subprocess.PIPE if capture else None, stderr=subprocess.STDOUT if capture else None)
    out = r.stdout.decode("utf-8", "replace") if capture and r.stdout is not None else ""
    if check and r.returncode:
        raise SystemExit(f"worker command failed (exit {r.returncode}): {cmd[:200]}\n{out[-2000:]}")
    return r.returncode, out


def pairing(pairdir):
    """PAIRDIR/fast_pair.json's two sides, each with its net checked against the build-time CRC."""
    pd = absdir(pairdir)
    f = pd / "fast_pair.json"
    if not f.exists():
        raise SystemExit(f"{pd}: no fast_pair.json (build it with fast_pair.py pair / rr-build)")
    doc = json.loads(f.read_text(encoding="utf-8"))
    for side in ("dev", "prev"):
        s = doc[side]
        if s.get("fast"):
            if not s.get("net"):
                raise SystemExit(f"{pd}: {side} has no compiled-in net; nothing to ship")
            if not Path(s["net"]).exists():
                raise SystemExit(f"{pd}: {side}'s net {s['net']} is missing")
            now = file_crc(s["net"])
            if now != s["crc32"]:
                raise SystemExit(f"{pd}: {side}'s net {s['net']} is crc32 {now}, but the pairing was built "
                                 f"for {s['crc32']}; rebuild the pairing")
    return pd, doc


def remote_net(s):
    return f"{s['crc32']}/{Path(s['net']).name}"


def ship_nets(doc):
    """Every fast side's net in ~/crossfish_worker/nets/CRC/FILE; returns (remote home, {side: path})."""
    nets = {side: remote_net(doc[side]) for side in ("dev", "prev") if doc[side].get("fast")}
    args = " ".join(shlex.quote(n) for n in sorted(set(nets.values())))

    def check():
        _, out = ssh(f"python3 - {args}", input=NET_SCRIPT.encode())
        home = next(line.split(" ", 1)[1] for line in out.splitlines() if line.startswith("home "))
        state = {line.split(" ", 1)[1]: line.split(" ", 1)[0] for line in out.splitlines()
                 if line.split(" ", 1)[0] in ("have", "copied", "arrived", "need")}
        return home, state

    home, state = check()
    local = {remote_net(doc[s]): doc[s]["net"] for s in nets}
    for n, st in sorted(state.items()):
        if st == "need":
            t0 = time.time()
            subprocess.run([sw.SCP, *sw.OPTS, "-q", local[n], f"{sw.HOST}:{REMOTE_NETS}/{n}.part"], check=True,
                           stdin=subprocess.DEVNULL)
            say(f"net {n} copied to the worker ({os.path.getsize(local[n]) / 1e6:.0f} MB, {time.time() - t0:.0f} s)")
        elif st != "have":
            say(f"net {n}: {st} (from the flat nets directory)")
    if any(st == "need" for st in state.values()):
        home, state = check()
    bad = [n for n in nets.values() if state.get(n) not in ("have", "arrived", "copied")]
    if bad:
        raise SystemExit(f"nets not in place on the worker after copying: {', '.join(bad)}")
    return home, {side: f"{home}/{REMOTE_NETS}/{n}" for side, n in nets.items()}


def net_env(paths):
    return " ".join(f"{SIDE_ENV[side]}={p}" for side, p in sorted(paths.items()))


# ---------------------------------------------------------------- local reference statics

def ensure_positions():
    """The 1,000 crosscheck positions: every 20th record of the 20,000-position parity sample (stage 4's
    iso_pos1000.cfdg, byte for byte)."""
    if not POSITIONS.exists():
        if not PARITY.exists():
            raise SystemExit(f"{POSITIONS} is missing and so is {PARITY}, from which it is made")
        data = PARITY.read_bytes()
        POSITIONS.parent.mkdir(parents=True, exist_ok=True)
        POSITIONS.write_bytes(b"".join(data[i * REC:(i + 1) * REC] for i in range(0, len(data) // REC, 20)))
        say(f"wrote {POSITIONS.name}: every 20th record of {PARITY.name}")
    return POSITIONS


def reference(s):
    """Static evals of POSITIONS by candidate S's own single build (its datagen.exe, compiled-in net):
    a list of ints. Cached by the executable's and the net's content."""
    d = absdir(s["dir"])
    exe = d / "datagen.exe"
    if not exe.exists():
        raise SystemExit(f"{d}: no datagen.exe for the reference statics (eval_candidate.py build {d})")
    h = hashlib.sha256(exe.read_bytes())
    h.update(f"{s.get('crc32', 'shipped')}\0{file_crc(ensure_positions())}".encode())
    out = REF_DIR / f"{s['name']}__{h.hexdigest()[:12]}.cfdg"
    if not out.exists():
        REF_DIR.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".tmp")
        tmp.unlink(missing_ok=True)
        env = {k: v for k, v in os.environ.items() if not k.upper().startswith("FASTNNUE_")}
        env["PATH"] = str(TOOLCHAIN) + os.pathsep + env.get("PATH", "")
        r = subprocess.run([str(exe), "label", str(POSITIONS), str(tmp), "0", "1"], env=env, cwd=CPP,
                           capture_output=True, text=True, errors="replace")
        if r.returncode:
            raise SystemExit(f"{exe} label failed (exit {r.returncode}): {r.stderr[-1000:]}")
        got = {m.group(1): m.group(5) for m in NET_LINE.finditer(r.stderr)}
        if s.get("fast") and got.get("Dev") != s["crc32"]:
            raise SystemExit(f"{exe}: its net line shows crc32 {got.get('Dev')}, expected {s['crc32']}")
        if not s.get("fast") and got:
            raise SystemExit(f"{exe}: loaded a fast NNUE net ({got}) but {s['name']} has none")
        os.replace(tmp, out)
    data = out.read_bytes()
    return [struct.unpack_from("<i", data, i * REC + 108)[0] for i in range(len(data) // REC)]


def cmd_refs(a):
    pd, doc = pairing(a.pairdir)
    ra, rb = reference(doc["dev"]), reference(doc["prev"])
    print(f"{doc['dev']['name']}: {len(ra)} statics; {doc['prev']['name']}: {len(rb)} statics;"
          f" they differ on {sum(x != y for x, y in zip(ra, rb))}")


# ---------------------------------------------------------------- setup

def remote_dir(name):
    return sw.remote_dir(name)


def cmd_setup(a):
    t0 = time.time()
    pd, doc = pairing(a.pairdir)
    rd = remote_dir(a.name)
    desc = ", ".join(f"{KSIDE[s]} {doc[s]['name']}" + (f" ({Path(doc[s]['net']).name} crc32 {doc[s]['crc32']})"
                                                       if doc[s].get("fast") else " (no fast NNUE)")
                     for s in ("dev", "prev"))
    say(f"setup {a.name} from {pd.name}: {desc}")
    ref = {s: reference(doc[s]) for s in ("dev", "prev")}
    power = sum(x != y for x, y in zip(ref["dev"], ref["prev"]))
    home, paths = ship_nets(doc)
    env = net_env(paths)

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for p in sorted(CPP.iterdir()):
            if p.is_file() and p.suffix in (".hpp", ".cpp", ".inc", ".h") or p.name == "opening_book.bin":
                tar.add(p, arcname=f"cpp_impl/{p.name}")
        for p in sorted((CPP / "compat").iterdir()):
            tar.add(p, arcname=f"cpp_impl/compat/{p.name}")
        for p in sorted(pd.iterdir()):
            if p.is_file() and p.suffix in (".hpp", ".cpp", ".json"):
                tar.add(p, arcname=f"cand/{p.name}")
        if not (pd / "pair_statics.cpp").exists():
            tar.add(HERE / "pair_statics.cpp", arcname="cand/pair_statics.cpp")
        tar.add(ensure_positions(), arcname="cand/positions.cfdg")
        data = "".join(f"{SIDE_ENV[s]}={p}\n" for s, p in sorted(paths.items())).encode()
        info = tarfile.TarInfo("cand/nets.env")
        info.size, info.mtime = len(data), int(time.time())
        tar.addfile(info, io.BytesIO(data))
    ssh(f"rm -rf ~/{rd} && mkdir -p ~/{rd} && tar -xzf - -C ~/{rd}", input=buf.getvalue())
    say(f"shipped {len(buf.getvalue()) // 1024} KiB of sources to {sw.HOST}:~/{rd}; nets.env: {env or '(none)'}")

    build = (f"cd ~/{rd}/cpp_impl && g++ -dumpfullversion > ../cand/gxx_version && "
             f"(g++ {sw.FLAGS} -I. -o ../cand/test_bots ../cand/test_bots.cpp 2> ../cand/test_bots.err & p1=$!; "
             f"g++ {sw.FLAGS} -I. -o ../cand/pair_statics ../cand/pair_statics.cpp 2> ../cand/pair_statics.err & p2=$!; "
             f"wait $p1; r1=$?; wait $p2; r2=$?; echo \"build $r1 $r2 warnings $(cat ../cand/*.err | grep -c warning:)"
             f" gxx $(cat ../cand/gxx_version)\"; tail -n 5 ../cand/*.err; [ $r1 = 0 ] && [ $r2 = 0 ])")
    tb = time.time()
    rc, out = ssh(build, check=False)
    if rc:
        raise SystemExit(f"worker build of {a.name} failed:\n{out[-3000:]}")
    summary = next((line for line in out.splitlines() if line.startswith("build ")), "build ?")
    say(f"built test_bots and pair_statics with g++ in {time.time() - tb:.0f} s ({summary})")

    _, out = ssh(f"cd ~/{rd}/cpp_impl && {CLEAN_ENV} {env} ../cand/test_bots verify 2>&1 | tail -n 16", check=False)
    oks = [line for line in out.splitlines() if line.rstrip().endswith("OK") or ": OK" in line]
    if not oks or "mismatch" in out.lower() or "failed" in out.lower():
        raise SystemExit(f"worker self-tests of {a.name} did not pass:\n{out}")
    say(f"test_bots verify: {len(oks)} self-tests OK")

    rc, out = ssh(f"cd ~/{rd}/cpp_impl && {CLEAN_ENV} {env} ../cand/pair_statics ../cand/positions.cfdg {len(ref['dev'])} 0"
                  f" ../statics.txt > ../statics.out 2>&1; echo rc=$?; cat ../statics.out; echo ---; cat ../statics.txt",
                  check=False)
    head, _, table = out.partition("\n---\n")
    if "rc=0" not in head:
        raise SystemExit(f"worker pair_statics of {a.name} failed:\n{head[-2000:]}")
    problems = []
    got = {}
    for m in NET_LINE.finditer(head):
        got.setdefault(m.group(1), set()).add((m.group(2), m.group(3), int(m.group(4)), m.group(5)))
    for s in ("dev", "prev"):
        k = KSIDE[s]
        if not doc[s].get("fast"):
            if got.get(k):
                problems.append(f"{k} loaded {sorted(got[k])} but {doc[s]['name']} has no fast NNUE")
            continue
        want = (paths[s], SIDE_ENV[s], os.path.getsize(doc[s]["net"]), doc[s]["crc32"])
        if got.get(k) != {want}:
            problems.append(f"{k} net lines {sorted(got.get(k, []))}, expected {want}")
    rows = {}
    for line in table.splitlines():
        f = line.split()
        if len(f) == 5:
            rows[int(f[0])] = (int(f[1]), int(f[2]))
    n = len(ref["dev"])
    dd = sum(rows.get(i, (None, None))[0] != ref["dev"][i] for i in range(n))
    pdiff = sum(rows.get(i, (None, None))[1] != ref["prev"][i] for i in range(n))
    if len(rows) != n or dd or pdiff:
        problems.append(f"statics: {len(rows)}/{n} rows; Dev differs from {doc['dev']['name']}'s single build on"
                        f" {dd}, Prev from {doc['prev']['name']}'s on {pdiff}")
    if problems:
        raise SystemExit(f"worker crosscheck of {a.name} failed: " + "; ".join(problems))
    say(f"statics crosscheck: Dev = {doc['dev']['name']}'s single build on {n - dd}/{n}, Prev ="
        f" {doc['prev']['name']}'s on {n - pdiff}/{n} (the two references differ on {power});"
        f" net lines " + "; ".join(f"{k} {Path(x[0]).parent.name}/{Path(x[0]).name} crc32 {x[3]}"
                                    for k, v in sorted(got.items()) for x in v))
    json_doc = dict(name=a.name, pairdir=str(pd), host=sw.HOST, home=home, nets=paths, env=f"{CLEAN_ENV} {env}", statics=n,
                    dev_equal=n - dd, prev_equal=n - pdiff, power=power, setup_s=round(time.time() - t0, 1),
                    at=time.time())
    ssh(f"cat > ~/{rd}/worker_setup.json", input=json.dumps(json_doc, indent=1).encode())
    say(f"setup {a.name} done in {time.time() - t0:.0f} s")


# ---------------------------------------------------------------- start / stop

def cmd_start(a):
    rd = remote_dir(a.name)
    env = dict(SPRT_THINK_MS=a.ms, SPRT_THREADS=a.threads, SPRT_GAME_OFFSET=a.offset, SPRT_LLR_BOUND=100)
    args = ""
    if a.depth:
        del env["SPRT_THINK_MS"]
        args = f" depth {a.depth}"
    if a.games:
        env["SPRT_MAX_GAMES"] = a.games
    for kv in a.env:
        if not re.fullmatch(r"SPRT_[A-Z_]+=[0-9,.]+", kv):
            raise SystemExit(f"--env {kv!r}: expected SPRT_NAME=number[,number...]")
        k, v = kv.split("=", 1)
        env[k] = v
    envs = " ".join(f"{k}={v}" for k, v in env.items())
    _, nets = ssh(f"cat ~/{rd}/cand/nets.env")
    nenv = " ".join(nets.split())
    for kv in nenv.split():
        if not re.fullmatch(r"FASTNNUE_(PREV_)?PATH=/[A-Za-z0-9_./-]+", kv):
            raise SystemExit(f"~/{rd}/cand/nets.env: unexpected entry {kv!r}; run setup again")
    want = sorted(f"{k}={v}" for k, v in env.items()) + sorted(nenv.split())
    want = sorted(want)
    launch = LAUNCH.replace("@HOST@", sw.HOST).replace("@ARGS@", args)
    ssh(f"cat > ~/{rd}/cand/launch.sh", input=launch.encode())
    rc, out = ssh(f"cd ~/{rd}/cpp_impl && rm -f ../sprt.log ../sprt.pid && (nohup {CLEAN_ENV} {envs} {nenv}"
                  f" sh ../cand/launch.sh > /dev/null 2>&1 < /dev/null & echo $! > ../sprt.pid)"
                  f" && sleep 1 && echo started pid $(cat ../sprt.pid) && echo '--- header' && head -n 1 ../sprt.log"
                  f" && echo '--- proc' && (tr '\\0' '\\n' < /proc/$(cat ../sprt.pid)/environ || echo proc-unreadable)"
                  f" | grep -E '^(FASTNNUE|SPRT)_|proc-unreadable' | LC_ALL=C sort")
    started, _, rest = out.partition("--- header\n")
    header, _, proc = rest.partition("--- proc\n")
    header = header.strip()
    m = re.match(r"^# worker \S+ \(g\+\+ [^)]*\): (.*)test_bots", header)
    recorded = sorted(t for t in (m.group(1).split() if m else []) if RECORDED.match(t))
    in_proc = sorted(t for t in proc.split() if RECORDED.match(t))
    problems = []
    if recorded != want:
        problems.append(f"log header records {recorded or header!r}")
    if "proc-unreadable" not in proc and in_proc != want:  # test_bots may already have exited (tiny runs)
        problems.append(f"/proc environ has {in_proc}")
    if problems:
        ssh(f"kill $(cat ~/{rd}/sprt.pid) 2>/dev/null", check=False)
        raise SystemExit(f"start {a.name}: the engine environment is not the planned one ({'; '.join(problems)};"
                         f" planned {want}); stopped the run")
    say(f"start {a.name}: {started.strip()}; environment checked (log header and /proc: {len(want)} FASTNNUE_*/SPRT_*"
        f" variables, exactly the planned ones): {header}")


def cmd_stop(a):
    rd = remote_dir(a.name)
    _, out = ssh(f"kill $(cat ~/{rd}/sprt.pid) 2>/dev/null; sleep 1; pgrep -ax test_bots || echo stopped", check=False)
    print(out.strip())


# ---------------------------------------------------------------- round_robin hook

def worker_call(*args):
    """round_robin.worker_call's replacement: the same commands, through this script."""
    subprocess.run([sys.executable, str(Path(__file__).resolve()), *map(str, args)], check=True, cwd=ROOT)


def install(rr_module, log_path=None):
    """Make round_robin's worker executor use this script (fast_pair.py rr-run --worker)."""
    rr_module.worker_call = worker_call
    if log_path:
        os.environ["FAST_WORKER_LOG"] = str(log_path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("setup"); p.add_argument("name"); p.add_argument("pairdir"); p.set_defaults(fn=cmd_setup)
    p = sub.add_parser("start"); p.add_argument("name"); p.add_argument("--ms", type=int, default=90)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--offset", type=int, default=25000)
    p.add_argument("--depth", type=int, default=0)
    p.add_argument("--games", type=int, default=0)
    p.add_argument("--env", action="append", default=[])
    p.set_defaults(fn=cmd_start)
    p = sub.add_parser("stop"); p.add_argument("name"); p.set_defaults(fn=cmd_stop)
    p = sub.add_parser("refs"); p.add_argument("pairdir"); p.set_defaults(fn=cmd_refs)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
