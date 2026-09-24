"""Compare search speed of CodinGame binaries through the real protocol.

CodinGame compiles C++ with `g++ -std=gnu++17 -Werror=return-type -g -pthread`
and no -O flag, so a submission is only as fast as its pragmas and
always_inline attributes make it. Build each candidate that way
(`make -C cpp_impl cg-flags`) and compare the node counts the bot prints for
searched moves against a random opponent.

usage: python tools/cg_speed_check.py <bot> [<bot> ...] [games=4]
"""

import random
import statistics
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from python_impl.board import board_obj  # noqa: E402
from python_impl.operations import ops  # noqa: E402


def play(bot, bot_first, rng, max_ply, out):
    proc = subprocess.Popen([bot], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL, text=True)
    board = board_obj()
    last = (-1, -1)
    try:
        for ply in range(max_ply):
            if ops.check_game_finished(board):
                break
            valid = ops.get_valid_moves(board)
            if (ply % 2 == 0) == bot_first:
                proc.stdin.write(f"{last[0]} {last[1]}\n{len(valid)}\n")
                proc.stdin.write("".join(f"{r} {c}\n" for r, c in valid))
                proc.stdin.flush()
                reply = proc.stdout.readline().split()
                move = (int(reply[0]), int(reply[1]))
                if move not in valid:
                    raise AssertionError(f"illegal move {move} at ply {ply}")
                out.extend(int(t[1:]) for t in reply[2:] if t.startswith("N"))
            else:
                move = rng.choice(valid)
                last = move
            ops.make_move(board, move)
    finally:
        proc.kill()


def main():
    args = sys.argv[1:]
    games = int(args.pop()) if args and args[-1].isdigit() else 4
    for bot in args:
        nodes = []
        for g in range(games):
            play(bot, g % 2 == 0, random.Random(1000 + g), 40, nodes)
        print(f"{bot}: searched moves {len(nodes)}  mean nodes {statistics.mean(nodes):.0f}"
              f"  median {statistics.median(nodes):.0f}")


if __name__ == "__main__":
    main()
