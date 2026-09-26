#!/usr/bin/env python3
"""Run SPRT shards of an eval_candidate.py build on a remote Linux worker over SSH.

The worker builds the same sources natively (g++) and plays its own opening
range, both engines on the same machine; tools/sprt_merge.py pools its
pentanomial counts with the local shard's.

  sprt_worker.py setup NAME DIR            ship cpp_impl sources + candidate DIR, build, run self-tests
  sprt_worker.py crosscheck NAME SAMPLE    remote static evals must equal the local build's on SAMPLE
  sprt_worker.py start NAME [--ms 90 | --depth D] [--threads 8] [--offset 25000] [--games N] [--env SPRT_X=V ...]
  sprt_worker.py sync NAME LOCAL_LOG [--interval 60]
      copies the remote log to datasets/eval2/sprt/NAME_worker.log, appends the
      pooled result to datasets/eval2/sprt/NAME_combined.log, and stops the
      remote shard and exits once the pooled LLR reaches +/-2.94.
  sprt_worker.py stop NAME

Host and key: CROSSFISH_WORKER (user@host, required) and
CROSSFISH_WORKER_KEY (default ~/.ssh/crossfish_worker). Remote files live in
~/crossfish_worker/NAME/.
"""
from __future__ import annotations

import argparse
import io
import os
import re
import subprocess
import sys
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CPP = ROOT / "cpp_impl"
SPRT_DIR = ROOT / "datasets" / "eval2" / "sprt"
HOST = os.environ.get("CROSSFISH_WORKER", "")  # user@host of the Linux worker
KEY = os.path.expanduser(os.environ.get("CROSSFISH_WORKER_KEY", "~/.ssh/crossfish_worker"))
GIT_BIN = Path(r"C:\Program Files\Git\usr\bin")
SSH = str(GIT_BIN / "ssh.exe") if (GIT_BIN / "ssh.exe").exists() else "ssh"
SCP = str(GIT_BIN / "scp.exe") if (GIT_BIN / "scp.exe").exists() else "scp"
OPTS = ["-i", KEY, "-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]
FLAGS = "-O3 -std=c++17 -mavx2 -mbmi -mbmi2 -mlzcnt -mpopcnt -pthread -Wno-unknown-pragmas -Wno-ignored-attributes"


def ssh(cmd, check=True, capture=False):
    r = subprocess.run([SSH, *OPTS, HOST, cmd], check=check, text=True,
                       stdout=subprocess.PIPE if capture else None, stderr=subprocess.STDOUT if capture else None)
    return r.stdout if capture else r.returncode


def remote_dir(name):
    return f"crossfish_worker/{name}"


def cmd_setup(a):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for p in sorted(CPP.iterdir()):
            if p.is_file() and p.suffix in (".hpp", ".cpp", ".inc", ".h") or p.name == "opening_book.bin":
                tar.add(p, arcname=f"cpp_impl/{p.name}")
        for p in sorted((CPP / "compat").iterdir()):
            tar.add(p, arcname=f"cpp_impl/compat/{p.name}")
        for p in sorted(Path(a.dir).iterdir()):
            if p.is_file() and p.suffix in (".hpp", ".cpp", ".json"):
                tar.add(p, arcname=f"cand/{p.name}")
    rd = remote_dir(a.name)
    r = subprocess.run([SSH, *OPTS, HOST, f"rm -rf ~/{rd} && mkdir -p ~/{rd} && tar -xzf - -C ~/{rd}"],
                       input=buf.getvalue(), check=True)
    print(f"shipped {len(buf.getvalue()) // 1024} KiB to {HOST}:~/{rd}")
    ssh(f"cd ~/{rd}/cpp_impl && g++ {FLAGS} -I. -o ../cand/test_bots ../cand/test_bots.cpp"
        f" && g++ {FLAGS} -I. -o ../cand/datagen ../cand/datagen.cpp && echo built")
    out = ssh(f"cd ~/{rd}/cpp_impl && ../cand/test_bots verify 2>&1 | tail -n 12", capture=True)
    print(out)
    if "OK" not in out or "mismatch" in out or "failed" in out:
        sys.exit("remote self-tests did not pass")


def cmd_crosscheck(a):
    import numpy as np
    sys.path.insert(0, str(ROOT / "tools"))
    import eval_data
    rd = remote_dir(a.name)
    sample = Path(a.sample)
    subprocess.run([SCP, *OPTS, str(sample), f"{HOST}:{rd}/sample.cfdg"], check=True)
    ssh(f"cd ~/{rd} && rm -f sample_out.cfdg && cand/datagen label sample.cfdg sample_out.cfdg 0 2 > /dev/null")
    local_out = SPRT_DIR.parent / "tmp" / f"{a.name}_crosscheck.cfdg"
    remote_copy = SPRT_DIR.parent / "tmp" / f"{a.name}_crosscheck_remote.cfdg"
    local_out.parent.mkdir(parents=True, exist_ok=True)
    local_out.unlink(missing_ok=True)
    env = dict(os.environ, PATH=str(ROOT / "toolchains" / "llvm-mingw-20260616-ucrt-x86_64" / "bin") + os.pathsep
               + os.environ.get("PATH", ""))
    subprocess.run([str(Path(a.local_dir) / "datagen.exe"), "label", str(sample), str(local_out), "0", "2"],
                   check=True, env=env, stdout=subprocess.DEVNULL)
    subprocess.run([SCP, *OPTS, f"{HOST}:{rd}/sample_out.cfdg", str(remote_copy)], check=True)
    x, y = np.fromfile(local_out, dtype=eval_data.REC), np.fromfile(remote_copy, dtype=eval_data.REC)
    dh = np.abs(x["hce"].astype(int) - y["hce"]).max()
    ds = np.abs(x["static_eval"].astype(int) - y["static_eval"]).max()
    print(f"crosscheck {len(x):,} positions: max |hce diff| {dh}, max |static diff| {ds}")
    if dh or ds > 1:
        sys.exit("remote build evaluates differently from the local build")


def cmd_start(a):
    rd = remote_dir(a.name)
    # the pooled LLR decides; this shard never stops on its own
    env = dict(SPRT_THINK_MS=a.ms, SPRT_THREADS=a.threads, SPRT_GAME_OFFSET=a.offset, SPRT_LLR_BOUND=100)
    args = ""
    if a.depth:  # an equal-depth match (tools/round_robin.py): no clock
        del env["SPRT_THINK_MS"]
        args = f" depth {a.depth}"
    if a.games:
        env["SPRT_MAX_GAMES"] = a.games
    for kv in a.env:  # a name given here replaces the default above
        if not re.fullmatch(r"SPRT_[A-Z_]+=[0-9,.]+", kv):
            sys.exit(f"--env {kv!r}: expected SPRT_NAME=number[,number...]")
        k, v = kv.split("=", 1)
        env[k] = v
    env = " ".join(f"{k}={v}" for k, v in env.items())
    ssh(f"cd ~/{rd}/cpp_impl && (nohup env {env} ../cand/test_bots{args} > ../sprt.log 2>&1 & echo $! > ../sprt.pid)"
        f" && sleep 1 && echo started pid $(cat ../sprt.pid)")


def cmd_stop(a):
    rd = remote_dir(a.name)
    ssh(f"kill $(cat ~/{rd}/sprt.pid) 2>/dev/null; sleep 1; pgrep -ax test_bots || echo stopped", check=False)


def cmd_sync(a):
    sys.path.insert(0, str(ROOT / "tools"))
    rd = remote_dir(a.name)
    worker_log = SPRT_DIR / f"{a.name}_worker.log"
    combined = SPRT_DIR / f"{a.name}_combined.log"
    SPRT_DIR.mkdir(parents=True, exist_ok=True)
    while True:
        subprocess.run([SCP, *OPTS, "-q", f"{HOST}:{rd}/sprt.log", str(worker_log)], check=False)
        rc = subprocess.run([sys.executable, str(ROOT / "tools" / "sprt_merge.py"), str(combined),
                             a.local_log, str(worker_log)], stdout=subprocess.PIPE, text=True).returncode
        if rc == 2:
            cmd_stop(a)
            print("pooled SPRT reached its bound; remote shard stopped")
            return
        time.sleep(a.interval)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("setup"); p.add_argument("name"); p.add_argument("dir"); p.set_defaults(fn=cmd_setup)
    p = sub.add_parser("crosscheck"); p.add_argument("name"); p.add_argument("sample")
    p.add_argument("--local-dir", required=True, help="the candidate DIR whose datagen.exe is the reference")
    p.set_defaults(fn=cmd_crosscheck)
    p = sub.add_parser("start"); p.add_argument("name"); p.add_argument("--ms", type=int, default=90)
    p.add_argument("--threads", type=int, default=4)  # 8 caused timeouts on the laptop
    p.add_argument("--offset", type=int, default=25000)
    p.add_argument("--depth", type=int, default=0, help="equal-depth games instead of --ms")
    p.add_argument("--games", type=int, default=0, help="stop after this many games (SPRT_MAX_GAMES)")
    p.add_argument("--env", action="append", default=[], help="extra SPRT_*=value for test_bots (repeatable)")
    p.set_defaults(fn=cmd_start)
    p = sub.add_parser("stop"); p.add_argument("name"); p.set_defaults(fn=cmd_stop)
    p = sub.add_parser("sync"); p.add_argument("name"); p.add_argument("local_log")
    p.add_argument("--interval", type=int, default=60); p.set_defaults(fn=cmd_sync)
    a = ap.parse_args()
    if not HOST:
        sys.exit("set CROSSFISH_WORKER=user@host (the Linux worker) first")
    a.fn(a)


if __name__ == "__main__":
    main()
