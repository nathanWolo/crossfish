"""CodinGame performance gate: build the paste file the way CodinGame does and
compare it with a base revision.

The Dev-vs-Prev SPRT compiles both engines at -O3, so it cannot see anything
that only CodinGame's compiler does. CodinGame builds C++ with g++ 11 and
`-std=gnu++17 -Werror=return-type -g -pthread`, no -O flag: the submission is
only fast because of its pragmas and always_inline attributes (section 47 of
the improvement log). This gate checks the paste file that would actually be
submitted:

  1. fresh:    cpp_impl/cg_input.cpp is the minifier's current output of
               codingame_nnue.cpp (not stale, not hand-edited)
  2. size:     under the 100,000-character cap (UTF-16 units)
  3. speed:    nodes per millisecond over full-budget replies, candidate vs
               base, both built with CodinGame's flags, over paired protocol
               games against a random opponent; fails on a slowdown that is
               both material (more than --tol) and significant (95% interval
               below 1)
  4. inlining: the candidate built with CodinGame's flags vs the same source
               at -O3; a large gap means a hot helper is not always_inline
  5. latency:  first reply under 1,000 ms; 99th percentile of later replies
               under 95 ms (CodinGame forfeits a reply after 100 ms)
  6. smoke:    candidate vs base, 90 ms, random openings (roundrobin match
               mode); fails on timeouts or a significantly worse score. The
               Elo is informational: strength is the SPRT's job.
  7. book:     tools/play_book_protocol_check.py on the candidate build

Exit status 1 if any check fails. Needs a real GCC for 3-5 (clang ignores
GCC's optimize pragmas, so its -O0 build says nothing about CodinGame);
--allow-non-gcc runs anyway and labels the result.

usage: python tools/cg_perf_gate.py [--base origin/main] [--cxx g++-11]
           [--speed-games 16] [--smoke-games 200] [--workers N] [--tol 0.05]
           [--skip-smoke] [--skip-book] [--allow-non-gcc] [--summary FILE]
"""
from __future__ import annotations

import argparse
import math
import os
import random
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from python_impl.board import board_obj  # noqa: E402
from python_impl.operations import ops  # noqa: E402

CPP = ROOT / "cpp_impl"
CAP = 100_000
FULL_MS = 80  # a later reply this long used (nearly) the whole 90 ms budget
CG_FLAGS = ["-std=gnu++17", "-Werror=return-type", "-g", "-pthread"]
O3_FLAGS = ["-O3", "-std=c++17", "-mavx2", "-mbmi", "-mbmi2", "-mlzcnt", "-mpopcnt", "-pthread",
            "-Wno-unknown-pragmas", "-Wno-ignored-attributes"]
WINDOWS = os.name == "nt"
EXE = ".exe" if WINDOWS else ""
LIBS = [] if WINDOWS else ["-lm", "-lpthread", "-ldl", "-lcrypt"]
STACK = ["-Wl,--stack,16777216"] if WINDOWS else []

results: list[tuple[str, bool, str]] = []  # (check, passed, detail)


def record(check: str, passed: bool, detail: str) -> None:
    results.append((check, passed, detail))
    print(f"[{'PASS' if passed else 'FAIL'}] {check}: {detail}", flush=True)


def utf16_len(text: str) -> int:
    return len(text.encode("utf-16-le")) // 2


def pick_compiler(requested: str | None) -> str:
    for cand in ([requested] if requested else ["g++-11", "g++"]):
        if cand and shutil.which(cand):
            return cand
    sys.exit(f"no compiler found (tried {requested or 'g++-11, g++'}); pass --cxx")


def is_gcc(cxx: str) -> bool:
    out = subprocess.run([cxx, "--version"], capture_output=True, text=True).stdout
    return "Free Software Foundation" in out and "clang" not in out.lower()


def build(cxx: str, src: Path, out: Path, flags: list[str]) -> None:
    cmd = [cxx, *flags, *STACK, "-o", str(out), str(src), *LIBS]
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=CPP)
    if p.returncode != 0:
        sys.exit(f"build failed: {' '.join(cmd)}\n{p.stderr[-3000:]}")


# ---------------------------------------------------------------- CodinGame protocol games
def protocol_game(bot: Path, bot_first: bool, seed: int, max_ply: int = 40) -> dict:
    """One game through the real CodinGame protocol against a seeded random
    opponent. Returns per searched move: nodes, depth; and every reply's latency."""
    rng = random.Random(seed)
    proc = subprocess.Popen([str(bot)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL, text=True, bufsize=1)
    board = board_obj()
    last = (-1, -1)
    nodes, depths, latencies = [], [], []
    try:
        for ply in range(max_ply):
            if ops.check_game_finished(board):
                break
            valid = ops.get_valid_moves(board)
            if (ply % 2 == 0) == bot_first:
                t0 = time.perf_counter()
                proc.stdin.write(f"{last[0]} {last[1]}\n{len(valid)}\n")
                proc.stdin.write("".join(f"{r} {c}\n" for r, c in valid))
                proc.stdin.flush()
                line = proc.stdout.readline()
                latencies.append((time.perf_counter() - t0) * 1000)
                reply = line.split()
                if len(reply) < 2:
                    raise RuntimeError(f"{bot.name}: no move at ply {ply} (exit code {proc.poll()})")
                move = (int(reply[0]), int(reply[1]))
                if move not in valid:
                    raise RuntimeError(f"{bot.name}: illegal move {move} at ply {ply}")
                n = [int(t[1:]) for t in reply[2:] if t.startswith("N")]
                d = [int(t[1:]) for t in reply[2:] if t.startswith("D")]
                if n and len(latencies) > 1:  # later turns only: the first has a longer budget
                    nodes.append((n[0], latencies[-1]))
                if d:
                    depths.append(d[0])
            else:
                move = rng.choice(valid)
                last = move
            ops.make_move(board, move)
    finally:
        proc.kill()
    return dict(nodes=nodes, depths=depths, latencies=latencies)


def ratio_ci(pairs: list[tuple[float, float]]) -> tuple[float, float, float]:
    """Geometric-mean ratio b/a over paired games and a 95% interval (t on log ratios)."""
    logs = [math.log(b / a) for a, b in pairs if a > 0 and b > 0]
    m = statistics.mean(logs)
    if len(logs) < 2:
        return math.exp(m), math.exp(m), math.exp(m)
    se = statistics.stdev(logs) / math.sqrt(len(logs))
    t = 2.13 if len(logs) <= 16 else 2.0
    return math.exp(m), math.exp(m - t * se), math.exp(m + t * se)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="origin/main", help="git revision to compare against")
    ap.add_argument("--cxx", default=None, help="compiler for the CodinGame build (default g++-11, else g++)")
    ap.add_argument("--speed-games", type=int, default=16)
    ap.add_argument("--smoke-games", type=int, default=200)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2 - 1))
    ap.add_argument("--tol", type=float, default=0.05, help="tolerated slowdown in nodes per millisecond")
    ap.add_argument("--skip-smoke", action="store_true")
    ap.add_argument("--skip-book", action="store_true")
    ap.add_argument("--allow-non-gcc", action="store_true")
    ap.add_argument("--summary", default=os.environ.get("GITHUB_STEP_SUMMARY"), help="append a markdown summary here")
    args = ap.parse_args()

    cxx = pick_compiler(args.cxx)
    gcc = is_gcc(cxx)
    if not gcc and not args.allow_non_gcc:
        sys.exit(f"{cxx} is not GCC: its -O0 build ignores the source's optimize pragmas and says "
                 "nothing about CodinGame. Use a GCC, or --allow-non-gcc for a labelled dry run.")
    note = "" if gcc else " (NOT CodinGame-accurate: non-GCC compiler)"
    if not gcc:
        # clang also ignores `#pragma GCC target`, so a dry run needs the ISA on the command line.
        CG_FLAGS.extend(["-mavx2", "-mbmi", "-mbmi2", "-mlzcnt", "-mpopcnt",
                         "-Wno-unknown-pragmas", "-Wno-ignored-attributes"])
    work = Path(tempfile.mkdtemp(prefix="cg_gate_"))
    print(f"cg_perf_gate: compiler {cxx}{note}, base {args.base}, work dir {work}", flush=True)

    # 1. fresh
    cand_src = CPP / "cg_input.cpp"
    regen = work / "cg_input_regen.cpp"
    subprocess.run([sys.executable, str(ROOT / "tools/cg_minify.py"), str(CPP / "codingame_nnue.cpp"),
                    "-o", str(regen), "--inline-local"], check=True, capture_output=True, cwd=ROOT)
    norm = lambda p: p.read_text(encoding="utf-8").replace("\r\n", "\n")
    record("fresh", norm(regen) == norm(cand_src),
           "cg_input.cpp matches the minifier's output" if norm(regen) == norm(cand_src)
           else "cg_input.cpp is stale or hand-edited: run make -C cpp_impl cg-input")

    # 2. size
    size = utf16_len(norm(cand_src))
    base_text = subprocess.run(["git", "show", f"{args.base}:cpp_impl/cg_input.cpp"], capture_output=True,
                               text=True, encoding="utf-8", cwd=ROOT)
    if base_text.returncode != 0:
        sys.exit(f"cannot read cpp_impl/cg_input.cpp at {args.base}: {base_text.stderr.strip()}")
    base_size = utf16_len(base_text.stdout.replace("\r\n", "\n"))
    record("size", size <= CAP, f"{size:,} characters ({CAP - size:,} left; base {base_size:,}, {size - base_size:+,})")

    # builds (the base keeps its own source; both land in the work dir)
    base_src = work / "cg_input_base.cpp"
    base_src.write_text(base_text.stdout, encoding="utf-8")
    bins = {"base": work / f"base_cg{EXE}", "cand": work / f"cand_cg{EXE}", "cand_o3": work / f"cand_o3{EXE}"}
    build(cxx, base_src, bins["base"], CG_FLAGS)
    build(cxx, cand_src, bins["cand"], CG_FLAGS)
    build(cxx, cand_src, bins["cand_o3"], O3_FLAGS)
    print("built base and candidate with CodinGame's flags, and the candidate at -O3", flush=True)

    # 3-5. paired protocol games, interleaved so drift hits every build alike
    per = {k: [] for k in bins}
    for g in range(args.speed_games):
        for k in bins:
            per[k].append(protocol_game(bins[k], bot_first=(g % 2 == 0), seed=1000 + g))
    # Speed is nodes per millisecond over full-budget replies (at least FULL_MS of
    # the 90 ms budget). Nodes per move is not speed: a search that stops early
    # (a proven result, or a mate "reached depth 50") searches fewer nodes
    # without being slower, and a fix that stops it quitting early would look
    # faster.
    def nps(game):
        full = [(n, ms) for n, ms in game["nodes"] if ms >= FULL_MS]
        return sum(n for n, _ in full) / sum(ms for _, ms in full) if full else None

    def paired(a, b):
        return [(x, y) for x, y in zip(map(nps, per[a]), map(nps, per[b])) if x and y]

    def full_share(k):
        moves = [ms for g in per[k] for _, ms in g["nodes"]]
        return sum(ms >= FULL_MS for ms in moves) / max(1, len(moves))

    all_depth = {k: [d for r in per[k] for d in r["depths"]] for k in bins}
    pairs = paired("base", "cand")
    r, lo, hi = ratio_ci(pairs)
    speed_ok = not (r < 1 - args.tol and hi < 1.0)
    record("speed" + note, speed_ok,
           f"nodes/ms at full budget: candidate {statistics.mean(y for _, y in pairs):,.0f} vs base "
           f"{statistics.mean(x for x, _ in pairs):,.0f}, ratio {r:.3f} [{lo:.3f}, {hi:.3f}] over "
           f"{len(pairs)} paired games; full-budget replies {full_share('cand'):.0%} vs "
           f"{full_share('base'):.0%}; mean depth {statistics.mean(all_depth['cand']):.1f} vs "
           f"{statistics.mean(all_depth['base']):.1f}")
    ri, loi, hii = ratio_ci(paired("cand_o3", "cand"))
    record("inlining" + note, ri >= 0.85,
           f"CodinGame-flags build at {ri:.0%} [{loi:.0%}, {hii:.0%}] of the -O3 build's nodes/ms "
           f"(below 85% means a hot helper is not always_inline)")
    first = [r["latencies"][0] for r in per["cand"] if r["latencies"]]
    later = sorted(x for r in per["cand"] for x in r["latencies"][1:])
    p99 = later[min(len(later) - 1, int(0.99 * len(later)))] if later else 0.0
    over = sum(x >= 100 for x in later)
    record("latency" + note, max(first) < 1000 and p99 < 95,
           f"first reply max {max(first):.0f} ms; later replies p99 {p99:.1f} ms, max {later[-1]:.1f} ms, "
           f"{over} of {len(later)} at or over 100 ms")

    # 6. smoke match
    if not args.skip_smoke:
        from roundrobin import calc_elo, run_pair
        res = run_pair("candidate", [str(bins["cand"]), "match"], "base", [str(bins["base"]), "match"],
                       args.smoke_games, 90, args.workers)
        n = res.wins + res.draws + res.losses
        elo, ci = calc_elo(res.wins, res.losses, res.draws)
        timeout_ok = res.timeouts1 <= max(1, n // 100)
        record("smoke" + note, timeout_ok and elo + ci >= 0,
               f"candidate vs base {res.wins}-{res.draws}-{res.losses} (N={n}), Elo {elo:+.0f} +/- {ci:.0f} "
               f"(informational); timeouts candidate {res.timeouts1} / base {res.timeouts2}; "
               f"max reply {res.max_move_ms1:.1f} / {res.max_move_ms2:.1f} ms")

    # 7. book coverage and protocol timing
    if not args.skip_book:
        p = subprocess.run([sys.executable, str(ROOT / "tools/play_book_protocol_check.py"), str(bins["cand"]), "40"],
                           capture_output=True, text=True, cwd=ROOT)
        tail = [l for l in p.stdout.strip().splitlines() if l][-2:]
        record("book", p.returncode == 0, " / ".join(tail) if tail else p.stderr.strip()[-300:])

    ok = all(passed for _, passed, _ in results)
    lines = [f"### CodinGame performance gate: {'PASS' if ok else 'FAIL'}", "",
             f"Compiler `{cxx}`{note}; base `{args.base}`.", "", "| Check | Result | Detail |", "|---|---|---|"]
    lines += [f"| {c} | {'PASS' if p else '**FAIL**'} | {d} |" for c, p, d in results]
    text = "\n".join(lines) + "\n"
    print("\n" + text)
    if args.summary:
        with open(args.summary, "a", encoding="utf-8") as f:
            f.write(text)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
