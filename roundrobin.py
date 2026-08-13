#!/usr/bin/env python3
"""Round-robin match harness for current C++, Legend C++, and legacy Python.

Uses a persistent-process match protocol (NEW / APPLY / GO) so 10k-game
pairs are feasible. Default time control is 20ms/move, the same budget used
for SPRT eval-tuning, not the 95ms CodinGame clock.
"""
from __future__ import annotations

import argparse
import math
import os
import queue
import subprocess
import sys
import time
from dataclasses import dataclass
from multiprocessing import Process, Queue
from pathlib import Path

import numpy as np

from python_impl.board import board_obj
from python_impl.operations import ops

ROOT = Path(__file__).resolve().parent
BIN = ROOT / "cpp_impl" / "bin"


def cpp_to_py(mb: int, sq: int) -> tuple[int, int]:
    return (mb // 3) * 3 + sq // 3, (mb % 3) * 3 + sq % 3


def py_to_cpp(row: int, col: int) -> tuple[int, int]:
    return (row // 3) * 3 + (col // 3), (row % 3) * 3 + (col % 3)


def calc_elo(wins: int, losses: int, draws: int) -> tuple[float, float]:
    total = wins + losses + draws
    if total == 0:
        return float("nan"), float("nan")
    win_rate = wins / total
    draw_rate = draws / total
    loss_rate = losses / total
    e = win_rate + 0.5 * draw_rate
    if e <= 0 or e >= 1:
        elo = math.inf if e >= 1 else -math.inf
    else:
        elo = -400 * math.log10(1 / e - 1)

    percentage = (wins + draws / 2) / total
    wins_dev = win_rate * (1 - percentage) ** 2
    draws_dev = draw_rate * (0.5 - percentage) ** 2
    losses_dev = loss_rate * (0 - percentage) ** 2
    std_dev = math.sqrt(wins_dev + draws_dev + losses_dev) / math.sqrt(total)
    z = 1.959963984540054
    lo = percentage - z * std_dev
    hi = percentage + z * std_dev

    def _elo(p: float) -> float:
        if p <= 0:
            return -math.inf
        if p >= 1:
            return math.inf
        return -400 * math.log10(1 / p - 1)

    ci = (_elo(hi) - _elo(lo)) / 2
    return elo, ci


def calc_los(wins: int, losses: int) -> float:
    if wins + losses == 0:
        return 50.0
    return 100 * (0.5 + 0.5 * math.erf((wins - losses) / math.sqrt(2.0 * (wins + losses))))


class MatchBot:
    def __init__(self, cmd: list[str], name: str):
        self.name = name
        self.cmd = cmd
        self.proc = None
        self.start()

    def start(self):
        self.close()
        self.proc = subprocess.Popen(
            self.cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
            cwd=str(ROOT),
        )

    def close(self):
        if self.proc is None:
            return
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        self.proc = None

    def _send(self, line: str):
        if self.proc is None or self.proc.poll() is not None:
            self.start()
        assert self.proc.stdin is not None
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def new(self):
        self._send("NEW")

    def apply(self, mb: int, sq: int):
        self._send(f"APPLY {mb} {sq}")

    def go(self, ms: int) -> tuple[int, int]:
        self._send(f"GO {ms}")
        assert self.proc.stdout is not None
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError(f"{self.name} exited during GO (code {self.proc.poll()})")
        parts = line.split()
        return int(parts[0]), int(parts[1])


def random_opening(rng: np.random.Generator) -> list[tuple[int, int]]:
    board = board_obj()
    n = int(rng.integers(4, 9))
    moves = []
    for i in range(n):
        if ops.check_game_finished(board):
            break
        if i == 0 and rng.random() < 0.3:
            mv = (4, 4)
        else:
            legal = ops.get_valid_moves(board)
            mv = legal[int(rng.integers(0, len(legal)))]
        ops.make_move(board, mv)
        moves.append(py_to_cpp(int(mv[0]), int(mv[1])))
    return moves


def play_one(b1: MatchBot, b2: MatchBot, opening: list[tuple[int, int]], think_ms: int, first_is_b1: bool) -> int:
    """Return 1 if b1 wins, 0 draw, -1 if b1 loses."""
    board = board_obj()
    b1.new()
    b2.new()
    for mb, sq in opening:
        rowcol = cpp_to_py(mb, sq)
        ops.make_move(board, rowcol)
        b1.apply(mb, sq)
        b2.apply(mb, sq)

    while not ops.check_game_finished(board):
        stm_is_b1 = ((board.n_moves % 2 == 0) == first_is_b1)
        bot = b1 if stm_is_b1 else b2
        other = b2 if stm_is_b1 else b1
        try:
            mb, sq = bot.go(think_ms)
        except Exception:
            return -1 if stm_is_b1 else 1
        rowcol = cpp_to_py(mb, sq)
        if not ops.check_move_is_valid(board, rowcol):
            return -1 if stm_is_b1 else 1
        ops.make_move(board, rowcol)
        other.apply(mb, sq)

    winner = ops.get_winner(board)
    if "stale" in winner:
        return 0
    agent1_won = "agent 1" in winner
    p0_is_b1 = first_is_b1
    if agent1_won:
        return 1 if p0_is_b1 else -1
    return -1 if p0_is_b1 else 1


def worker(task_q: Queue, result_q: Queue, cmd1: list[str], cmd2: list[str], name1: str, name2: str, think_ms: int):
    b1 = MatchBot(cmd1, name1)
    b2 = MatchBot(cmd2, name2)
    try:
        while True:
            task = task_q.get()
            if task is None:
                break
            seed = task
            rng = np.random.default_rng(seed)
            opening = random_opening(rng)
            r1 = play_one(b1, b2, opening, think_ms, True)
            r2 = play_one(b1, b2, opening, think_ms, False)
            result_q.put((r1, r2))
    finally:
        b1.close()
        b2.close()


@dataclass
class PairResult:
    name1: str
    name2: str
    wins: int
    draws: int
    losses: int
    seconds: float
    think_ms: int

    def summary(self) -> str:
        elo, ci = calc_elo(self.wins, self.losses, self.draws)
        los = calc_los(self.wins, self.losses)
        n = self.wins + self.draws + self.losses
        gps = n / self.seconds if self.seconds > 0 else 0
        return (
            f"{self.name1} vs {self.name2}  N={n}  "
            f"W {self.wins} / D {self.draws} / L {self.losses}  "
            f"Elo {elo:+.1f} +/- {ci:.1f}  LOS {los:.1f}%  "
            f"{gps:.1f} games/s  {self.think_ms}ms"
        )


def run_pair(name1: str, cmd1: list[str], name2: str, cmd2: list[str], games: int, think_ms: int, workers: int) -> PairResult:
    if games % 2:
        games += 1
    openings = games // 2
    task_q: Queue = Queue()
    result_q: Queue = Queue()
    procs = [
        Process(target=worker, args=(task_q, result_q, cmd1, cmd2, name1, name2, think_ms))
        for _ in range(workers)
    ]
    for p in procs:
        p.start()
    for i in range(openings):
        task_q.put(10007 + i * 997)
    for _ in procs:
        task_q.put(None)

    wins = draws = losses = 0
    done = 0
    t0 = time.time()
    print(f"== {name1} vs {name2}: {games} games, {think_ms}ms, {workers} workers ==", flush=True)
    while done < games:
        r1, r2 = result_q.get()
        for r in (r1, r2):
            if r > 0:
                wins += 1
            elif r < 0:
                losses += 1
            else:
                draws += 1
            done += 1
        if done % 200 == 0 or done == games:
            elo, ci = calc_elo(wins, losses, draws)
            sec = time.time() - t0
            print(
                f"  {done}/{games}  W {wins} D {draws} L {losses}  "
                f"Elo {elo:+.1f} +/- {ci:.1f}  {done / max(sec, 1e-6):.1f} g/s",
                flush=True,
            )
    for p in procs:
        p.join()
    return PairResult(name1, name2, wins, draws, losses, time.time() - t0, think_ms)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=10000)
    parser.add_argument("--ms", type=int, default=20)
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args()

    current = [str(BIN / "crossfish"), "match"]
    legend = [str(BIN / "cg_legend"), "match"]
    python = [sys.executable, str(ROOT / "python_impl" / "match_bot.py")]

    for path in (BIN / "crossfish", BIN / "cg_legend"):
        if not path.exists():
            raise SystemExit(f"missing {path}; run `make -C cpp_impl compile-cg compile-legend`")

    pairs = [
        ("current", current, "legend", legend),
        ("current", current, "python", python),
        ("legend", legend, "python", python),
    ]
    results = []
    t0 = time.time()
    for n1, c1, n2, c2 in pairs:
        results.append(run_pair(n1, c1, n2, c2, args.games, args.ms, args.workers))

    lines = [
        f"round robin  games/pair={args.games}  think={args.ms}ms  workers={args.workers}",
        f"elapsed {time.time() - t0:.1f}s",
        "",
    ]
    for r in results:
        lines.append(r.summary())
    text = "\n".join(lines) + "\n"
    print()
    print(text)
    if args.out:
        Path(args.out).write_text(text)


if __name__ == "__main__":
    main()
