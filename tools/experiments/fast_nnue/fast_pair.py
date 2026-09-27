#!/usr/bin/env python3
"""Two-net builds of the fast NNUE (nnue2 stage 4): Dev and Prev of one test_bots each load their own net.

The fast NNUE headers keep one net per namespace (fnnue, and fnnue::bnn for the B nets), loaded once
per process, and #pragma once admits each header once per build. tools/eval_candidate.py build
--prev-dir OTHER applies OTHER's dev_patches.json to crossfish_prev.hpp, so with two fast candidates
both engines would include the same fast_nnue_any.hpp and share one net (make_cand.ENGINE_GUARD now
makes that a compile error instead). This script builds such a pairing:

  Dev   candidate A exactly as eval_candidate.py builds it: A's eval headers, HCE weights, patches
        and fast_nnue*.hpp in the build directory. Net: FASTNNUE_PATH, else A's FASTNNUE_NET_FILE.
  Prev  B's fast headers renamed into the build directory as prev_fast_nnue*.hpp (namespace
        fnnue_prev, kSide "Prev", and every FASTNNUE_* / FNNUE_* macro or environment name except
        FASTNNUE_CHECK with the prefix FASTNNUE_PREV_ / FNNUE_PREV_: FASTNNUE_PREV_KIND, _NET_FILE,
        _ENGINE, _PATH, _BAKE, _CACHE_BITS, _FLOOR_SHIFTS, ..., FNNUE_PREV_KERNEL), and a Prev view of
        B (DIR/prev_view: B's eval headers and HCE weights, its dev_patches.json under the same
        renames) given to eval_candidate.py build --prev-dir.
        Net: FASTNNUE_PREV_PATH, else B's FASTNNUE_NET_FILE.
  A candidate without fast NNUE patches (the shipped noop, say) is passed through unchanged.

Build flags and environment act on one side only. A candidate's own #defines travel with its side
(Prev's under the PREV_ names), so B's numerics variant (FASTNNUE_FLOOR_SHIFTS in its patches, say)
reaches Prev and never Dev. A -D numerics or tuning flag given on the compiler command line
(-DFASTNNUE_FLOOR_SHIFTS, -DFASTNNUE_CACHE_BITS=N, -DFASTNNUE_B_PREFETCH=N, -DFASTNNUE_NO_SYSV, ...)
applies to Dev only; give Prev the same with the PREV_ name (-DFASTNNUE_PREV_FLOOR_SHIFTS). Likewise
the environment: FASTNNUE_PATH and FASTNNUE_BAKE are Dev's, FASTNNUE_PREV_PATH and FASTNNUE_PREV_BAKE
Prev's. The one shared switch is -DFASTNNUE_CHECK: a check build verifies both copies, each with its
own counters (side=Dev / side=Prev).

After the build, check_sources rejects a generated source that reaches the other side's copy or
still names a Dev-side FASTNNUE_* / FNNUE_* macro on Prev's side, and check_symbols (llvm-nm)
requires fnnue_prev:: symbols in test_bots.exe exactly when Prev has a fast NNUE (fnnue:: likewise
for Dev). Both sides must carry stage-4 or later headers (kSide, net_path); an older fast candidate
is refused.

Each engine prints "fast_nnue [Dev|Prev]: net PATH (SOURCE), BYTES bytes, crc32 XXXXXXXX" to stderr
once, and DIR/fast_pair.json records which net each side was built for.

  pair A B [--out DIR] [--tools]
      builds DIR (default cpp_impl/bin/pair_A__B): Dev = candidate A, Prev = candidate B. A and B are
      candidate specs as round_robin.py takes them (NAME for cpp_impl/bin/cand_NAME, a directory, or
      NAME=DIR). --tools also builds datagen.exe and bench_ab.exe.
  rr-build NAME [--tools]
      `round_robin.py build NAME` for tournaments with fast NNUE candidates: every pairing built as
      `pair` builds it and stamped so that `round_robin.py run` accepts it. A pairing is up to date
      when round_robin's own digest and this script's fast digest (both candidates' fast headers,
      this script, the nets' CRC-32) match.
  rr-run NAME [round_robin.py run options...]
      checks that every unfinished pairing is up to date, drops every FASTNNUE_* variable
      (FASTNNUE_PATH, FASTNNUE_PREV_PATH, FASTNNUE_BAKE, ...) from the environment (each engine then
      loads its compiled-in net as built) and runs round_robin.py run (in this process).
      With --worker (and --worker-threads T, --no-local), the pairings the Linux worker takes go
      through fast_worker.py instead of sprt_worker.py: it ships the two nets by content (CRC-32),
      builds the pairing with g++ there, checks both engines' statics against the candidates' own
      local builds, and runs test_bots with FASTNNUE_PATH / FASTNNUE_PREV_PATH set to the shipped
      nets (the compiled-in Windows paths do not exist on Linux). Its setup lines also go to the
      tournament's worker.log. The worker is CROSSFISH_WORKER=user@host (required, with
      CROSSFISH_WORKER_KEY, default ~/.ssh/crossfish_worker), as for tools/sprt_worker.py.
  check-logs NAME
      every pairing log of tournament NAME must show each fast side loading its candidate's net
      (path and CRC-32) and no net line for a side without one. A worker's net line passes when it
      names the same file in the worker's ~/crossfish_worker/nets/CRC32/ with that CRC and size,
      taken from that side's variable (FASTNNUE_PATH for Dev, FASTNNUE_PREV_PATH for Prev).

--root DIR (before the command) keeps tournaments somewhere other than datasets/eval2/rr, as
round_robin.py --root does.
  check-log LOG [--dev NET] [--prev NET]
      the same for one test_bots log: Dev (Prev) must load NET, or nothing if the option is omitted.
  crc NET...
      the CRC-32 an engine prints for each net file.

Run with the Python that has numpy (toolchains/py312-dml): eval_candidate.py needs it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
TOOLS = ROOT / "tools"
BIN = ROOT / "cpp_impl" / "bin"
sys.path.insert(0, str(TOOLS))
import round_robin as rr  # noqa: E402

FAST_HEADERS = ("fast_nnue.hpp", "fast_nnue_b.hpp", "fast_nnue_any.hpp")
EMITTED = rr.EMITTED  # mini_eval_d16.hpp, macro_eval.hpp, hce_weights.json, dev_patches.json
INCLUDE_FAST = re.compile(r'#\s*include\s+"(fast_nnue(?:_b|_any)?\.hpp)"')
NET_FILE = re.compile(r'#define FASTNNUE_NET_FILE "([^"]*)"')
NET_LINE = re.compile(r"fast_nnue \[(Dev|Prev)\]: net (.*) \((.*)\), (\d+) bytes, crc32 ([0-9a-f]{8})")

# Prev's copy of the fast NNUE: every name that could tie the two engines together. Since stage 5 that
# is EVERY FASTNNUE_* and FNNUE_* identifier (macros, and the environment names "FASTNNUE_PATH" and
# "FASTNNUE_BAKE", which the same pattern renames inside their string literals). Stage 4 renamed only
# KIND / NET_FILE / ENGINE / "FASTNNUE_PATH", so a numerics macro that one candidate #defines in its
# patches (FASTNNUE_FLOOR_SHIFTS) leaked into the other engine's copy (review3). The one exception is
# FASTNNUE_CHECK, the build-wide switch that makes both copies verify themselves.
PREV_RENAMES = [
    (re.compile(r'"(fast_nnue(?:_b|_any)?\.hpp)"'), r'"prev_\1"'),        # its own header files
    (re.compile(r"\bfnnue\b"), "fnnue_prev"),                             # namespace (bnn is nested in it)
    (re.compile(r"\bFASTNNUE_(?!PREV_|CHECK\b)([A-Z0-9_]+)\b"), r"FASTNNUE_PREV_\1"),  # macros, env names
    (re.compile(r"\bFNNUE_(?!PREV_)([A-Z0-9_]+)\b"), r"FNNUE_PREV_\1"),  # FNNUE_KERNEL, FNNUE_ANY
    (re.compile(r'\bkSide = "Dev"'), 'kSide = "Prev"'),                   # its name in the log lines
]
# A Dev-side name left on Prev's side: what the renames above must leave nothing of.
DEV_NAME = re.compile(r"\bfnnue\b|\bFASTNNUE_(?!PREV_|CHECK\b)[A-Z0-9_]+|\bFNNUE_(?!PREV_)[A-Z0-9_]+"
                      r'|#\s*include\s+"fast_nnue')
# A Prev-side name in Dev's source.
PREV_NAME = re.compile(r"\bfnnue_prev\b|\bFASTNNUE_PREV_[A-Z0-9_]+|\bFNNUE_PREV_[A-Z0-9_]+|prev_fast_nnue")


def prev_rename(text):
    for pat, rep in PREV_RENAMES:
        text = pat.sub(rep, text)
    return text


def rel(p):
    return rr.rel(p)


def read_patches(d):
    f = Path(d) / "dev_patches.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else []


def fast_includes(d):
    """The fast NNUE headers candidate D's patches include (empty: not a fast NNUE candidate)."""
    return sorted({m for p in read_patches(d) for m in INCLUDE_FAST.findall(p["new"])})


def header_closure(d):
    """The fast headers candidate D needs: its patches' includes and theirs, all present in D."""
    todo, seen = fast_includes(d), []
    while todo:
        h = todo.pop()
        if h in seen:
            continue
        if not (Path(d) / h).exists():
            raise SystemExit(f"{d}: its patches need {h}, which is not in the candidate directory")
        seen.append(h)
        todo += INCLUDE_FAST.findall((Path(d) / h).read_text(encoding="utf-8"))
    return sorted(seen)


def baked_net(d):
    nets = {m for p in read_patches(d) for m in NET_FILE.findall(p["new"])}
    if len(nets) > 1:
        raise SystemExit(f"{d}: more than one FASTNNUE_NET_FILE in dev_patches.json")
    return next(iter(nets), None)


_crc_cache = {}


def net_crc(path):
    """(CRC-32 as the engines print it, size) of a net file."""
    path = str(Path(path).resolve())
    st = os.stat(path)
    key = (path, st.st_size, st.st_mtime_ns)
    if key not in _crc_cache:
        crc = 0
        with open(path, "rb") as fh:
            while chunk := fh.read(1 << 22):
                crc = zlib.crc32(chunk, crc)
        _crc_cache[key] = (f"{crc & 0xFFFFFFFF:08x}", st.st_size)
    return _crc_cache[key]


def side_info(name, d):
    fast = bool(fast_includes(d))
    info = dict(name=name, dir=rel(d), fast=fast)
    if fast:
        net = baked_net(d)
        info["net"] = net
        if net:
            info["crc32"], info["bytes"] = net_crc(net)
    return info


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fast_digest(a_dir, b_dir):
    """Everything a pairing's fast NNUE depends on beyond round_robin's own digest."""
    h = hashlib.sha256()
    for side, d in (("a", a_dir), ("b", b_dir)):
        for f in FAST_HEADERS:
            if (Path(d) / f).exists():
                h.update(f"{side}/{f}\0{sha(Path(d) / f)}\n".encode())
        net = baked_net(d) if fast_includes(d) else None
        if net:
            h.update(f"{side}/net\0{net}\0{net_crc(net)}\n".encode())
    h.update(f"fast_pair.py\0{sha(__file__)}\n".encode())
    return h.hexdigest()


def _leftovers(pat, text):
    return sorted({m.group(0) for m in pat.finditer(text)})


def check_sources(out, dev_fast, prev_fast):
    """The generated engine sources must each reach only their own fast NNUE copy: no Prev name in
    Dev's source, and no Dev name (namespace fnnue, fast_nnue*.hpp, or any FASTNNUE_* / FNNUE_* but
    FASTNNUE_CHECK) in Prev's source or its prev_fast_nnue*.hpp."""
    dev = (out / "crossfish_dev.hpp").read_text(encoding="utf-8")
    prev = (out / "crossfish_prev.hpp").read_text(encoding="utf-8")
    bad = []
    if left := _leftovers(PREV_NAME, dev):
        bad.append(f"crossfish_dev.hpp names Prev's fast NNUE ({', '.join(left[:6])})")
    if dev_fast and not (re.search(r"\bfnnue::", dev) and re.search(r'#include "fast_nnue', dev)):
        bad.append("crossfish_dev.hpp does not use its fast NNUE")
    if left := _leftovers(DEV_NAME, prev):
        bad.append(f"crossfish_prev.hpp names Dev's fast NNUE ({', '.join(left[:6])})")
    if prev_fast and not ("fnnue_prev::" in prev and '#include "prev_fast_nnue' in prev):
        bad.append("crossfish_prev.hpp does not use its fast NNUE")
    for h in FAST_HEADERS:
        p = out / f"prev_{h}"
        if p.exists():
            if left := _leftovers(DEV_NAME, p.read_text(encoding="utf-8")):
                bad.append(f"prev_{h} still names Dev's fast NNUE ({', '.join(left[:6])})")
    if bad:
        raise SystemExit(f"{out}: " + "; ".join(bad))


def find_nm():
    """llvm-nm of the repository's toolchain, else llvm-nm / nm on PATH."""
    for d in sorted((ROOT / "toolchains").glob("llvm-mingw-*/bin"), reverse=True):
        for n in ("llvm-nm.exe", "llvm-nm"):
            if (d / n).exists():
                return str(d / n)
    for n in ("llvm-nm", "nm"):
        if found := shutil.which(n):
            return found
    raise SystemExit("check_symbols: no llvm-nm or nm found (toolchains/llvm-mingw-*/bin or PATH)")


def symbol_counts(exe):
    """(fnnue:: symbols, fnnue_prev:: symbols) in EXE's symbol table (demangled)."""
    r = subprocess.run([find_nm(), "-C", str(exe)], check=True, capture_output=True, text=True, errors="replace")
    dev = prev = 0
    for line in r.stdout.splitlines():
        if "fnnue_prev::" in line:
            prev += 1
        elif "fnnue::" in line:
            dev += 1
    return dev, prev


def check_symbols(out, dev_fast, prev_fast):
    """The linked test_bots.exe must hold Prev's renamed copy exactly when Prev has a fast NNUE (and
    Dev's exactly when Dev has one): a pairing whose Prev silently used Dev's copy has no fnnue_prev::
    symbols. Returns the counts."""
    exe = out / "test_bots.exe"
    dev, prev = symbol_counts(exe)
    bad = []
    if bool(prev_fast) != (prev > 0):
        bad.append(f"{prev} fnnue_prev:: symbols but Prev {'has' if prev_fast else 'has no'} fast NNUE")
    if bool(dev_fast) != (dev > 0):
        bad.append(f"{dev} fnnue:: symbols but Dev {'has' if dev_fast else 'has no'} fast NNUE")
    if bad:
        raise SystemExit(f"{exe}: " + "; ".join(bad))
    return dev, prev


def require_stage4(d, side):
    """A fast side's headers must be stage 4 or later (kSide, net_path: the net line check-logs reads)."""
    f = Path(d) / "fast_nnue.hpp"
    if not f.exists() or 'kSide = "Dev"' not in f.read_text(encoding="utf-8"):
        raise SystemExit(f"{d}: {side}'s fast NNUE headers predate stage 4 (no kSide / net_path); "
                         "regenerate the candidate with make_cand_b.py --net")


def build_pairing(a_name, a_dir, b_name, b_dir, out, tools=False):
    a_dir, b_dir, out = Path(a_dir).resolve(), Path(b_dir).resolve(), Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "fast_pair.json").unlink(missing_ok=True)
    if out != a_dir:  # Dev is A exactly as emitted (a file A lacks must not linger)
        for f in EMITTED + FAST_HEADERS:
            if (a_dir / f).exists():
                shutil.copy(a_dir / f, out / f)
            else:
                (out / f).unlink(missing_ok=True)
    dev_fast, prev_fast = fast_includes(a_dir), fast_includes(b_dir)
    if dev_fast:
        header_closure(out)
        require_stage4(a_dir, "Dev")
    if prev_fast:
        require_stage4(b_dir, "Prev")
    view = out / "prev_view"
    shutil.rmtree(view, ignore_errors=True)
    for h in FAST_HEADERS:
        (out / f"prev_{h}").unlink(missing_ok=True)
    prev_dir = b_dir
    if prev_fast:
        view.mkdir()
        for f in ("mini_eval_d16.hpp", "macro_eval.hpp", "hce_weights.json"):
            if (b_dir / f).exists():
                shutil.copy(b_dir / f, view / f)
        renamed = [dict(p, old=prev_rename(p["old"]), new=prev_rename(p["new"])) for p in read_patches(b_dir)]
        (view / "dev_patches.json").write_text(json.dumps(renamed, indent=1), encoding="utf-8")
        for h in header_closure(b_dir):
            text = prev_rename((b_dir / h).read_text(encoding="utf-8"))
            (out / f"prev_{h}").write_text(text, encoding="utf-8", newline="\n")
        if (out / "prev_fast_nnue.hpp").exists():
            t = (out / "prev_fast_nnue.hpp").read_text(encoding="utf-8")
            if 'kSide = "Prev"' not in t or '"FASTNNUE_PREV_PATH"' not in t:
                raise SystemExit(f"{b_dir}/fast_nnue.hpp: the Prev renames found no kSide / FASTNNUE_PATH;"
                                 " rebuild the candidate with the stage-4 headers")
        prev_dir = view
    differ = [h for h in FAST_HEADERS if (a_dir / h).exists() and (b_dir / h).exists()
              and (a_dir / h).read_bytes() != (b_dir / h).read_bytes()]
    if differ and dev_fast and prev_fast:
        print(f"note: {', '.join(differ)} differ between {a_name} and {b_name} (each side compiles its own)")
    cmd = [sys.executable, str(TOOLS / "eval_candidate.py"), "build", str(out), "--prev-dir", str(prev_dir)]
    subprocess.run(cmd + ([] if tools else ["--no-tools"]), check=True, cwd=ROOT)
    check_sources(out, dev_fast, prev_fast)
    nsym = check_symbols(out, dev_fast, prev_fast)
    doc = dict(dev=side_info(a_name, a_dir), prev=side_info(b_name, b_dir), built=time.time(),
               builder="tools/experiments/fast_nnue/fast_pair.py",
               symbols=dict(dev_fnnue=nsym[0], prev_fnnue_prev=nsym[1]))
    (out / "fast_pair.json").write_text(json.dumps(doc, indent=1), encoding="utf-8")
    for side in ("dev", "prev"):
        s = doc[side]
        net = ("no fast NNUE" if not s["fast"] else f"net {s['net']} (crc32 {s['crc32']})" if s.get("net")
               else "net from the environment (FASTNNUE_PATH / FASTNNUE_PREV_PATH), none compiled in")
        print(f"  {side:<4} {s['name']}: {net}")
    print(f"  symbols: {nsym[0]} fnnue:: (Dev), {nsym[1]} fnnue_prev:: (Prev)")
    return doc


# ---------------------------------------------------------------- commands

def cmd_pair(a):
    an, ad = rr.parse_candidate(a.a)
    bn, bd = rr.parse_candidate(a.b)
    out = Path(a.out) if a.out else BIN / f"pair_{an}__{bn}"
    out = out if out.is_absolute() else ROOT / out
    build_pairing(an, ad, bn, bd, out, a.tools)
    print(f"built {out}: Dev {an}, Prev {bn}")


def stamp_ok(plan, p):
    d = rr.absdir(p["dir"])
    try:
        st = json.loads((d / "rr_build.json").read_text())
    except (OSError, ValueError):
        return False
    a_dir, b_dir = rr.absdir(plan["candidates"][p["a"]]), rr.absdir(plan["candidates"][p["b"]])
    return ((d / "test_bots.exe").exists() and st.get("digest") == rr.build_digest(plan, p)
            and st.get("fast_digest") == fast_digest(a_dir, b_dir))


def cmd_rr_build(a):
    plan = rr.load_plan(a.name)
    todo = [p for p in plan["pairs"] if not stamp_ok(plan, p)]
    print(f"{len(plan['pairs']) - len(todo)} of {len(plan['pairs'])} pairings up to date")
    t0 = time.time()
    for i, p in enumerate(todo):
        eta = f", ETA {rr.fmt_dur((time.time() - t0) / i * (len(todo) - i))}" if i else ""
        print(f"[{rr.stamp()}] building {rr.pair_key(p)} ({i + 1}/{len(todo)}{eta})", flush=True)
        d = rr.absdir(p["dir"])
        a_dir, b_dir = rr.absdir(plan["candidates"][p["a"]]), rr.absdir(plan["candidates"][p["b"]])
        (d / "rr_build.json").unlink(missing_ok=True)
        digest = rr.build_digest(plan, p)  # before building: the inputs as they are now
        fdigest = fast_digest(a_dir, b_dir)
        build_pairing(p["a"], a_dir, p["b"], b_dir, d, a.tools)
        (d / "rr_build.json").write_text(json.dumps(dict(
            digest=digest, fast_digest=fdigest, dev=plan["candidates"][p["a"]], prev=plan["candidates"][p["b"]],
            built=time.time(), builder="tools/experiments/fast_nnue/fast_pair.py"), indent=1))
    plan = rr.load_plan(a.name)  # re-read: a run may have updated states meanwhile
    for p in plan["pairs"]:
        if p["state"] == "planned" and rr.is_built(plan, p):
            p["state"] = "built"
    rr.save_plan(plan)
    print(f"built {len(todo)} pairings")


def cmd_rr_run(a, rest):
    plan = rr.load_plan(a.name)
    tdir = rr.tour_dir(a.name)
    stale = [rr.pair_key(p) for p in plan["pairs"]
             if rr.games_done(tdir / f"{rr.pair_key(p)}.log") < p["games"] and not stamp_ok(plan, p)]
    if stale:
        raise SystemExit(f"not built or out of date: {', '.join(stale)}; run `fast_pair.py rr-build {a.name}`")
    dropped = sorted(k for k in os.environ if k.upper().startswith("FASTNNUE_"))
    for k in dropped:  # round_robin runs in this process; its local games inherit os.environ
        del os.environ[k]
    if dropped:
        print(f"note: ignoring {', '.join(dropped)} (each engine loads its compiled-in net, as built)")
    if "--worker" in rest:
        if not os.environ.get("CROSSFISH_WORKER"):
            raise SystemExit("--worker needs CROSSFISH_WORKER=user@host (and CROSSFISH_WORKER_KEY if the key is not"
                             " ~/.ssh/crossfish_worker), as tools/sprt_worker.py does")
        import fast_worker  # noqa: E402  (this directory is on sys.path as the script's own)
        fast_worker.install(rr, rr.tour_dir(a.name) / "worker.log")
        print(f"worker pairings through fast_worker.py ({os.environ['CROSSFISH_WORKER']}); setup log"
              f" {rel(rr.tour_dir(a.name) / 'worker.log')}")
    sys.argv = ["round_robin.py", "--root", str(rr.RR_ROOT), "run", a.name, *rest]
    rr.main()


WORKER_NET = re.compile(r"^/.*/crossfish_worker/nets/([0-9a-f]{8})/([^/]+)$")
SIDE_VAR = {"Dev": "FASTNNUE_PATH", "Prev": "FASTNNUE_PREV_PATH"}


def log_nets(text):
    """{side: set((path, source, bytes, crc))} of a test_bots log's fast NNUE net lines (a worker's
    POSIX path is kept as printed)."""
    got = {"Dev": set(), "Prev": set()}
    for m in NET_LINE.finditer(text):
        p = m.group(2)
        p = p if p.startswith("/") else Path(p).resolve().as_posix()
        got[m.group(1)].add((p, m.group(3), int(m.group(4)), m.group(5)))
    return got


def net_ok(side, g, path, crc):
    """Is net line G = (path, source, bytes, crc) SIDE's planned net PATH with CRC? Locally: that very
    path. On the worker (fast_worker.py): the same file name under ~/crossfish_worker/nets/CRC/, from
    SIDE's environment variable, with the planned size."""
    if g[3] != crc:
        return False
    if g[0] == Path(path).resolve().as_posix():
        return True
    m = WORKER_NET.match(g[0])
    return bool(m and m.group(1) == crc and m.group(2) == Path(path).name and g[1] == SIDE_VAR[side]
                and g[2] == net_crc(path)[1])


def check_text(text, want):
    """Problems of one log against WANT = {side: (net path, crc) or None}."""
    got, bad = log_nets(text), []
    for side in ("Dev", "Prev"):
        if want[side] is None:
            if got[side]:
                bad.append(f"{side} loaded {sorted(got[side])} but should have no fast NNUE")
            continue
        path, crc = want[side]
        if not got[side]:
            bad.append(f"{side}: no net line")
        for g in got[side]:
            if not net_ok(side, g, path, crc):
                bad.append(f"{side} loaded {g[0]} ({g[1]}) crc32 {g[3]}, expected {path} crc32 {crc}")
    return got, bad


def want_side(d):
    if not fast_includes(d):
        return None
    net = baked_net(d)
    if not net:
        raise SystemExit(f"{d}: no FASTNNUE_NET_FILE compiled in; its net comes from the environment")
    return net, net_crc(net)[0]


def cmd_check_logs(a):
    plan = rr.load_plan(a.name)
    tdir = rr.tour_dir(a.name)
    nbad = 0
    for p in plan["pairs"]:
        log = tdir / f"{rr.pair_key(p)}.log"
        if not log.exists():
            print(f"  {rr.pair_key(p)}: no log yet")
            continue
        want = {"Dev": want_side(rr.absdir(plan["candidates"][p["a"]])),
                "Prev": want_side(rr.absdir(plan["candidates"][p["b"]]))}
        got, bad = check_text(log.read_text(encoding="utf-8", errors="replace"), want)
        nbad += len(bad)
        desc = "; ".join(f"{s} {Path(next(iter(g))[0]).name} crc32 {next(iter(g))[3]} x{len(g)}" if g else f"{s} none"
                         for s, g in got.items())
        print(f"  {rr.pair_key(p)}: {'OK' if not bad else 'BAD'} ({desc})" + "".join(f"\n    {b}" for b in bad))
    if nbad:
        raise SystemExit(f"{nbad} problem(s)")
    print("all pairing logs load the planned nets")


def cmd_check_log(a):
    want = {"Dev": (a.dev, net_crc(a.dev)[0]) if a.dev else None,
            "Prev": (a.prev, net_crc(a.prev)[0]) if a.prev else None}
    got, bad = check_text(Path(a.log).read_text(encoding="utf-8", errors="replace"), want)
    for s, g in got.items():
        for x in sorted(g):
            print(f"  {s}: {x[0]} ({x[1]}), {x[2]} bytes, crc32 {x[3]}")
    if bad:
        raise SystemExit("; ".join(bad))
    print(f"{a.log}: OK")


def cmd_crc(a):
    for n in a.net:
        crc, size = net_crc(n)
        print(f"{crc}  {size:>11,}  {Path(n).resolve().as_posix()}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", help="tournament directory (default datasets/eval2/rr)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pair"); p.add_argument("a"); p.add_argument("b"); p.add_argument("--out")
    p.add_argument("--tools", action="store_true"); p.set_defaults(fn=cmd_pair)
    p = sub.add_parser("rr-build"); p.add_argument("name"); p.add_argument("--tools", action="store_true")
    p.set_defaults(fn=cmd_rr_build)
    p = sub.add_parser("rr-run"); p.add_argument("name"); p.set_defaults(fn=None)
    p = sub.add_parser("check-logs"); p.add_argument("name"); p.set_defaults(fn=cmd_check_logs)
    p = sub.add_parser("check-log"); p.add_argument("log"); p.add_argument("--dev"); p.add_argument("--prev")
    p.set_defaults(fn=cmd_check_log)
    p = sub.add_parser("crc"); p.add_argument("net", nargs="+"); p.set_defaults(fn=cmd_crc)
    a, rest = ap.parse_known_args()
    if a.root:
        rr.RR_ROOT = rr.absdir(a.root)
    if a.cmd == "rr-run":
        cmd_rr_run(a, rest)
    elif rest:
        ap.error(f"unrecognized arguments: {' '.join(rest)}")
    else:
        a.fn(a)


if __name__ == "__main__":
    main()
