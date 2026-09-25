"""Drive the CodinGame binary through real protocol games and check the book.

The book covers selected lines after the first player's center-center (see
documentation/play_book.md). Every reply must be legal, and response times are
reported against the 100 ms referee.

With --book <text book> (the "S" lines play_book_pack consumed), the check is
exact: at every one of the bot's turns it must play from the book (its reply
carries "BOOK") exactly when the position is one of the book's positions,
merged under the 8 board symmetries. The opponent then prefers replies that
stay in the book, so games follow book lines deeply. Without the text book the
opponent plays uniformly at random, and the bot moving second must still play
from the book at its first move (after center-center).

usage: python tools/play_book_protocol_check.py <bot.exe> [games=40] [max_ply=24] [--book <text book>]
"""

import random
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from python_impl.board import board_obj  # noqa: E402
from python_impl.operations import ops  # noqa: E402


def _sym_rc(t, n, r, c):
    m = n - 1
    return [(r, c), (c, m - r), (m - r, m - c), (m - c, r), (r, m - c), (m - r, c), (c, r), (m - c, m - r)][t]


def key_of(cells, active):
    """Symmetry-canonical key: cells[r][c] in {0, 1, 2} and the board to play (9 = any)."""
    best = None
    for t in range(8):
        k = [0] * 81
        for r in range(9):
            for c in range(9):
                if cells[r][c]:
                    orow, ocol = _sym_rc(t, 9, r, c)
                    k[orow * 9 + ocol] = cells[r][c]
        if active == 9:
            a = 9
        else:
            ar, ac = _sym_rc(t, 3, active // 3, active % 3)
            a = ar * 3 + ac
        key = bytes(k) + bytes([a])
        if best is None or key < best:
            best = key
    return best


def active_board(valid):
    boards = {(r // 3) * 3 + c // 3 for r, c in valid}
    return boards.pop() if len(boards) == 1 else 9


def load_book(path):
    """Canonical keys of the book's positions, from "S <seq> <move>" lines."""
    keys = set()
    for line in open(path, encoding="utf-8"):
        parts = line.split()
        if len(parts) != 3 or parts[0] != "S":
            continue
        board, cells = board_obj(), [[0] * 9 for _ in range(9)]
        for i, cell in enumerate([] if parts[1] == "-" else parts[1].split(",")):
            cell = int(cell)
            mb, sq = divmod(cell, 9)
            r, c = (mb // 3) * 3 + sq // 3, (mb % 3) * 3 + sq % 3
            cells[r][c] = 1 + i % 2
            ops.make_move(board, (r, c))
        keys.add(key_of(cells, active_board(ops.get_valid_moves(board))))
    return keys


def play(bot, bot_first, rng, max_ply, book_keys):
    proc = subprocess.Popen([bot], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL, text=True)
    board = board_obj()
    cells = [[0] * 9 for _ in range(9)]
    moves_so_far = []
    last = (-1, -1)
    book, errors, times = 0, [], []
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
                from_book = len(reply) > 2 and reply[2] == "BOOK"
                book += from_book
                if book_keys is not None and ply > 0:
                    want = key_of(cells, active_board(valid)) in book_keys
                    if from_book != want:
                        errors.append(f"ply {ply}: {'book move' if from_book else 'no book move'} but position "
                                      f"{'is' if want else 'is not'} in the book")
                if book_keys is None and not bot_first and ply == 1 and last == (4, 4) and not from_book:
                    errors.append("no book move at our reply to center-center")
            else:
                if ply == 0 and book_keys is not None:
                    move = (4, 4) if rng.random() < 0.9 else rng.choice(valid)
                elif book_keys is not None and rng.random() < 0.9:
                    stay = []
                    for m in valid:
                        cells[m[0]][m[1]] = 1 + ply % 2
                        b2 = board_obj()
                        for mm in moves_so_far + [m]:
                            ops.make_move(b2, mm)
                        if not ops.check_game_finished(b2) and \
                                key_of(cells, active_board(ops.get_valid_moves(b2))) in book_keys:
                            stay.append(m)
                        cells[m[0]][m[1]] = 0
                    move = rng.choice(stay) if stay else rng.choice(valid)
                else:
                    move = rng.choice(valid)
                last = move
            ops.make_move(board, move)
            cells[move[0]][move[1]] = 1 + ply % 2
            moves_so_far.append(move)
    finally:
        proc.kill()
    return book, times, errors


def main():
    args = sys.argv[1:]
    book_path = None
    if "--book" in args:
        k = args.index("--book")
        book_path = args[k + 1]
        del args[k:k + 2]
    bot = args[0]
    games = int(args[1]) if len(args) > 1 else 40
    max_ply = int(args[2]) if len(args) > 2 else 24
    book_keys = load_book(book_path) if book_path else None
    if book_keys is not None:
        print(f"text book: {len(book_keys)} positions")
    rng = random.Random(2026)
    failures = 0
    all_times, first_turn, book_moves = [], [], {True: [], False: []}
    for g in range(games):
        bot_first = g % 2 == 0
        book, times, errors = play(bot, bot_first, rng, max_ply, book_keys)
        all_times += times[1:]
        first_turn.append(times[0])
        book_moves[bot_first].append(book)
        failures += bool(errors)
        print(f"{time.strftime('%H:%M:%S')}  game {g + 1}/{games}  bot moves "
              f"{'first' if bot_first else 'second'}: {book} book moves  {'ok' if not errors else 'FAIL: ' + errors[0]}",
              flush=True)
    all_times.sort()
    mean = lambda xs: sum(xs) / max(1, len(xs))
    print(f"\n{games - failures}/{games} games played from the book exactly where expected "
          f"(book moves per game: first {mean(book_moves[True]):.1f}, second {mean(book_moves[False]):.1f})")
    print(f"first turn: max {max(first_turn):.1f} ms; later moves: median "
          f"{all_times[len(all_times) // 2]:.1f} ms, max {all_times[-1]:.1f} ms "
          f"over {len(all_times)} replies")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
