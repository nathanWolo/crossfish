#!/usr/bin/env python3
"""Run the eval-retraining data pipeline stage by stage and publish its status.

Status goes to <dir>/pipeline.json, which the lab dashboard (the uttt.ai fork's
cg/viewer) reads together with each stage's log to show progress and ETA.
Every stage is resumable: datagen play and label append to their outputs.
A stage is done when its outputs are complete (N_PLAY self-play records, a
label file as long as its input, ten diff bytes per input record, a uttt.ai
value on every row), so a new --depth or a regenerated input is never taken
for finished work. A stage that runs first moves the outputs of the stages
fed by it aside to <name>.stale, so they are rebuilt from its new output.
While a stage's child runs, <dir>/<log stem>.lock holds its pid, and a
restarted driver waits for that process instead of starting a second copy.

  python tools/eval_pipeline.py [--dir datasets/eval2] [--depth 14] [--threads 16]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from eval_data import F_UTTT_V, REC, UTTTAI_TRAIN_DIR

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
TOOLCHAIN = ROOT / "toolchains" / "llvm-mingw-20260616-ucrt-x86_64" / "bin"
DATAGEN = ROOT / "cpp_impl" / "bin" / "datagen.exe"
# uttt.ai net4 from the fork's cg-net4 release (nathanWolo/utttai).
ONNX = Path(os.environ.get("UTTTAI_ONNX", str(ROOT.parent / "utttai" / "cg" / "train" / "net4.onnx"))).resolve()
N_PLAY = 3500000
DIFF_BYTES = 10  # datagen diffs: int8[10] per record
# A stage log with no lock file (an older driver, a manual run) this fresh may still be running.
LIVE_WINDOW = 600
END = "pipeline: stage "  # the driver's closing line in a stage log


def size(p: Path) -> int:
    try:
        return p.stat().st_size
    except OSError:
        return 0


def records(p: Path) -> int:
    return size(p) // REC.itemsize


def same_size(out: Path, src: Path) -> bool:
    return size(src) > 0 and size(out) == size(src)


def uttt_filled(p: Path) -> bool:
    if records(p) == 0 or size(p) % REC.itemsize:
        return False
    return bool(np.all(np.memmap(p, dtype=REC, mode="r")["flags"] & F_UTTT_V))


def stages(a):
    d, dep = a.dir, a.depth
    play, uttt = d / "cf_play.cfdg", d / "uttt_sp.cfdg"
    play_l, uttt_l = d / f"cf_play_d{dep}.cfdg", d / f"uttt_sp_d{dep}.cfdg"
    diffs = [(play_l, d / f"cf_play_d{dep}.diffs"), (uttt_l, d / f"uttt_sp_d{dep}.diffs")]
    # ins/outs are the files the pipeline makes, for invalidating the stages downstream of one that runs.
    return [
        dict(key="play", title="Crossfish self-play", log="cf_play.log", unit="positions", ins=[], outs=[play],
             done=lambda: records(play) >= N_PLAY,
             cmds=[[DATAGEN, "play", play, str(N_PLAY), "40", str(a.threads), "20260925", d / "uttt_openings.txt"]]),
        dict(key="label_cf", title=f"Depth-{dep} search labels: Crossfish games", log=f"label_cf_d{dep}.log",
             unit="positions", ins=[play], outs=[play_l], done=lambda: same_size(play_l, play),
             cmds=[[DATAGEN, "label", play, play_l, str(dep), str(a.threads)]]),
        dict(key="label_uttt", title=f"Depth-{dep} search labels: uttt.ai games", log=f"label_uttt_d{dep}.log",
             unit="positions", ins=[uttt], outs=[uttt_l], done=lambda: same_size(uttt_l, uttt),
             cmds=[[DATAGEN, "label", uttt, uttt_l, str(dep), str(a.threads)]]),
        dict(key="diffs", title="HCE feature counts", log=f"diffs_d{dep}.log", unit=None,
             ins=[play_l, uttt_l], outs=[o for _, o in diffs],
             done=lambda: all(records(i) > 0 and size(o) == DIFF_BYTES * records(i) for i, o in diffs),
             cmds=[[DATAGEN, "diffs", i, o] for i, o in diffs]),
        # Fills play_l in place, so it has no outputs of its own.
        dict(key="uttt_value", title="uttt.ai net4 values", log=f"uttt_value_d{dep}.log", unit="positions",
             ins=[play_l], outs=[], done=lambda: uttt_filled(play_l),
             cmds=[[PY, ROOT / "tools" / "eval_data.py", "uttt-value", play_l, ONNX]]),
    ]


def dependents(st, s):
    """The stages after s that read its outputs, directly or through each other."""
    made, out = set(s["outs"]), []
    for t in st[st.index(s) + 1:]:
        if made & set(t["ins"]):
            out.append(t)
            made |= set(t["outs"])
    return out


def check_inputs(a, st):
    """Fail now, not hours in, when a stage still to run lacks an input the pipeline does not make."""
    todo = {s["key"] for s in st if s["status"] != "done"}
    missing = []
    if todo - {"uttt_value"} and not DATAGEN.is_file():
        missing.append(f"{DATAGEN} (make -C cpp_impl datagen)")
    openings = a.dir / "uttt_openings.txt"
    if "play" in todo:
        try:  # datagen play silently plays without openings when it reads no 93-char lines
            text = openings.read_text(encoding="ascii", errors="replace")
        except OSError:
            text = ""
        if not any(len(line.rstrip("\r ")) == 93 for line in text.splitlines()):
            missing.append(f"{openings} with 93-char states (tools/eval_data.py openings)")
    uttt = a.dir / "uttt_sp.cfdg"
    if "label_uttt" in todo and (records(uttt) == 0 or size(uttt) % REC.itemsize):
        missing.append(f"{uttt} of 128-byte records (tools/eval_data.py import-utttai)")
    # eval_data.py uttt-value imports onnx_weights.py from UTTTAI_TRAIN_DIR, else from the ONNX file's directory.
    train_dir = ROOT / os.environ.get("UTTTAI_TRAIN_DIR", UTTTAI_TRAIN_DIR)  # the child runs in ROOT
    if "uttt_value" in todo and not ONNX.is_file():
        missing.append(f"{ONNX} (or set UTTTAI_ONNX)")
    if "uttt_value" in todo and not any((p / "onnx_weights.py").is_file() for p in (train_dir, ONNX.parent)):
        missing.append(f"the fork's onnx_weights.py in {train_dir} (UTTTAI_TRAIN_DIR) or {ONNX.parent}")
    if missing:
        raise SystemExit("missing pipeline inputs:\n  " + "\n  ".join(missing))


def pid_alive(pid: int, started: float) -> bool:
    """Whether pid is running and is the process started at `started`, not a later one reusing its pid."""
    if pid <= 0:
        return False
    if os.name != "nt":
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            pass
        return True
    # Not os.kill(pid, 0): on Windows that terminates the process.
    import ctypes
    k32 = ctypes.WinDLL("kernel32")
    k32.OpenProcess.restype = ctypes.c_void_p
    h = k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not h:
        return False
    try:
        code = ctypes.c_ulong()
        if not k32.GetExitCodeProcess(ctypes.c_void_p(h), ctypes.byref(code)) or code.value != 259:  # STILL_ACTIVE
            return False
        t = [ctypes.c_ulonglong() for _ in range(4)]  # creation, exit, kernel, user FILETIMEs (100 ns since 1601)
        k32.GetProcessTimes(ctypes.c_void_p(h), *map(ctypes.byref, t))
        return (t[0].value - 116444736000000000) / 1e7 <= started + 5
    finally:
        k32.CloseHandle(ctypes.c_void_p(h))


def lock_path(a, s) -> Path:
    return a.dir / (Path(s["log"]).stem + ".lock")


def running_elsewhere(a, s) -> bool:
    """Whether a process this driver did not start (an earlier driver's child, another driver) runs stage s."""
    lock = lock_path(a, s)
    try:
        info = json.loads(lock.read_text(encoding="utf-8"))
    except (OSError, ValueError):  # none, or caught half-written
        info = None
    if info is not None:
        if pid_alive(int(info.get("pid", 0)), float(info.get("started", 0))):
            return True
        lock.unlink(missing_ok=True)  # its process has exited
        return False
    # No readable lock: a log still being written, unless this driver closed it.
    log = a.dir / s["log"]
    try:
        with open(log, "rb") as fh:
            fh.seek(max(0, fh.seek(0, 2) - 256))
            last = fh.read().rstrip().rsplit(b"\n", 1)[-1]
        return time.time() - log.stat().st_mtime < LIVE_WINDOW and END.encode() not in last
    except OSError:
        return False


# Rough records per second on this machine (16 threads), for queued-stage estimates.
RATE = dict(play=670, label_cf=630, label_uttt=630, diffs=150000, uttt_value=30000)
EXPECTED_ROWS = dict(play=3500000, label_cf=3500000, label_uttt=1324283, diffs=4824283, uttt_value=3500000)


def write_status(a, st, current):
    doc = dict(name="Eval retraining data", dir=str(a.dir), updated=time.time(), current=current,
               stages=[dict(key=s["key"], title=s["title"], log=str(a.dir / s["log"]), unit=s["unit"],
                            status=s["status"], rows=EXPECTED_ROWS[s["key"]],
                            estimate_s=EXPECTED_ROWS[s["key"]] / RATE[s["key"]]) for s in st])
    tmp = a.dir / "pipeline.json.tmp"
    tmp.write_text(json.dumps(doc, indent=1))
    # A reader holding pipeline.json open (the dashboard polls it) blocks os.replace on Windows.
    for _ in range(50):
        try:
            os.replace(tmp, a.dir / "pipeline.json")
            return
        except PermissionError:
            time.sleep(0.02)
    os.replace(tmp, a.dir / "pipeline.json")


def run_stage(a, st, s, env):
    """Move aside the outputs of the stages s feeds, then run its commands, each under the stage's lock."""
    log, lock = a.dir / s["log"], lock_path(a, s)
    with open(log, "a", encoding="utf-8") as fh:
        end = "failed"
        try:
            for t in dependents(st, s):
                t["status"] = "queued"
                for p in t["outs"]:
                    if p.exists():
                        os.replace(p, p.with_name(p.name + ".stale"))
                        fh.write(f"[{time.strftime('%H:%M:%S')}] pipeline: moved {p.name}, made before this"
                                 f" {s['key']} run, to {p.name}.stale\n")
            s["status"] = "running"
            write_status(a, st, s["key"])
            for cmd in s["cmds"]:
                fh.write(f"[{time.strftime('%H:%M:%S')}] $ {' '.join(map(str, cmd))}\n")
                fh.flush()
                proc = subprocess.Popen([str(c) for c in cmd], stdout=fh, stderr=subprocess.STDOUT, env=env,
                                        cwd=ROOT)
                try:
                    lock.write_text(json.dumps(dict(pid=proc.pid, started=time.time())), encoding="utf-8")
                    rc = proc.wait()
                finally:
                    if proc.poll() is None:  # interrupted: do not leave it writing with no driver
                        proc.kill()
                        proc.wait()
                    lock.unlink(missing_ok=True)
                if rc != 0:
                    raise SystemExit(f"{s['key']} failed with exit code {rc}; see {log}")
            if not s["done"]():
                raise SystemExit(f"{s['key']} exited cleanly but its outputs are incomplete; see {log}")
            end = "done"
        except KeyboardInterrupt:
            end = "interrupted"
            raise
        finally:
            fh.write(f"[{time.strftime('%H:%M:%S')}] {END}{s['key']} {end}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=ROOT / "datasets" / "eval2")
    ap.add_argument("--depth", type=int, default=14)
    ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args()
    a.dir = a.dir.resolve()  # the children run in ROOT
    env = dict(os.environ, PATH=str(TOOLCHAIN) + os.pathsep + os.environ.get("PATH", ""))
    st = stages(a)
    for s in st:
        s["status"] = "done" if s["done"]() else "queued"
    for s in st:  # a stage still to run rebuilds the stages it feeds (run_stage moves their outputs aside)
        if s["status"] != "done":
            for t in dependents(st, s):
                t["status"] = "queued"
    if a.dir.is_dir():
        write_status(a, st, None)
    check_inputs(a, st)
    for s in st:
        try:
            # A stage run by a process this driver did not start: wait for it.
            while not s["done"]() and running_elsewhere(a, s):
                s["status"] = "running"
                write_status(a, st, s["key"])
                time.sleep(20)
            if not s["done"]():
                run_stage(a, st, s, env)
            s["status"] = "done"
        except BaseException as e:
            s["status"] = "interrupted" if isinstance(e, KeyboardInterrupt) else "failed"
            raise
        finally:
            write_status(a, st, None)
    print("pipeline complete")


if __name__ == "__main__":
    main()
