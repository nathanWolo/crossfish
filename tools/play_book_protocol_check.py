"""Drive the CodinGame binary through real protocol games and check the book.

Against a uniformly random opponent (the harshest test of full coverage),
every game must contain exactly PLAY_BOOK_DEPTH_FIRST book moves when the bot
moves first and PLAY_BOOK_DEPTH_SECOND when it moves second, and every reply
must be legal. Response times are reported against the 100 ms referee.

usage: python tools/play_book_protocol_check.py <bot.exe> [games=40] [max_ply=24]
"""

import random
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from python_impl.board import board_obj  # noqa: E402
from python_impl.operations import ops  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def book_depths():
    text = (ROOT / "cpp_impl" / "play_book_data.hpp").read_text(encoding="utf-8")
    first = int(re.search(r"PLAY_BOOK_DEPTH_FIRST = (\d+)", text).group(1))
    second = int(re.search(r"PLAY_BOOK_DEPTH_SECOND = (\d+)", text).group(1))
    return first, second


def play(bot, bot_first, rng, max_ply):
    proc = subprocess.Popen([bot], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL, text=True)
    board = board_obj()
    last = (-1, -1)
    book = 0
    times = []
    try:
        for ply in range(max_ply):
            if ops.check_game_finished(board):
                break
            valid = ops.get_valid_moves(board)
            if (ply % 2 == 0) == bot_first:
                proc.stdin.write(f"{last[0]} {last[1]}\n{len(valid)}\n")
                proc.stdin.write("".join(f"{r} {c}\n" for r, c in valid))
                proc.stdin.flush()
                t0 = time.perf_counter()
                reply = proc.stdout.readline().split()
                times.append((time.perf_counter() - t0) * 1000)
                if len(reply) < 2:
                    raise RuntimeError(
                        f"bot gave no move at ply {ply} (exit code {proc.poll()}); "
                        "on Windows, check its runtime DLLs are on PATH")
                move = (int(reply[0]), int(reply[1]))
                if move not in valid:
                    raise AssertionError(f"illegal move {move} at ply {ply}")
                book += len(reply) > 2 and reply[2] == "BOOK"
            else:
                move = rng.choice(valid)
                last = move
            ops.make_move(board, move)
    finally:
        proc.kill()
    return book, times


def main():
    bot = sys.argv[1]
    games = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    max_ply = int(sys.argv[3]) if len(sys.argv) > 3 else 24
    want = book_depths()
    rng = random.Random(2026)
    failures = 0
    all_times = []
    first_turn = []
    for g in range(games):
        bot_first = g % 2 == 0
        book, times = play(bot, bot_first, rng, max_ply)
        expected = want[0] if bot_first else want[1]
        all_times += times[1:]
        first_turn.append(times[0])
        status = "ok" if book == expected else "FAIL"
        failures += book != expected
        print(f"{time.strftime('%H:%M:%S')}  game {g + 1}/{games}  bot moves "
              f"{'first' if bot_first else 'second'}: {book} book moves (want {expected})  {status}",
              flush=True)
    all_times.sort()
    print(f"\n{games - failures}/{games} games had exactly the expected book moves")
    print(f"first turn: max {max(first_turn):.1f} ms; later moves: median "
          f"{all_times[len(all_times) // 2]:.1f} ms, max {all_times[-1]:.1f} ms "
          f"over {len(all_times)} replies")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
