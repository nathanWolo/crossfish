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

No book (the shipped default since 2026-10-09; documentation/play_book.md): pass
--book with the empty text book (cpp_impl/play_book.txt, or the payload dumped by
play_book_text_dump). That is the exact no-book check: no position is in the
book, so every reply must be a search move (none marked "BOOK"), and the
opponent, finding no book reply to steer to, plays uniformly at random after its
opening move (center-center with probability 0.9). Without --book the check is
not a no-book check: the opponent's first move is uniformly random, and a book
reply is demanded only in the bot-second games where it happens to open
center-center (about 1 in 81), so a no-book build passes or fails by chance.
verify.sh, the Makefile and cg_perf_gate.py all pass --book.

In every mode, the bot moving first must open center-center (4 4): both book
roots assume it, and the bot plays it hard-coded.

Positions with several book moves (payload format 2: several "S" lines of one
position with the same seq list its moves, in file order, the primary first):
  - a game fails if a BOOK move leads to none of the positions after the
    position's listed moves (canonical keys, so a symmetric twin counts);
  - the bot's choices at such positions are counted and summarized before the
    result lines (the bot picks uniformly at random, a new draw per game);
  - an opponent steering into the book (90% of its moves) picks, with
    probability 0.75, a reply into such a position when there is one. A book
    without such positions draws no extra random numbers, so its games are
    those of a single-move book.

The last two lines are the result (verify.sh and cg_perf_gate.py read them).

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

STEER_MULTI = 0.75  # chance that a steering opponent picks a reply into a multi-move position


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


def rc_of(cell):
    mb, sq = divmod(cell, 9)
    return (mb // 3) * 3 + sq // 3, (mb % 3) * 3 + sq % 3


def load_book(path):
    """From "S <seq> <move>" lines: the canonical keys of the book's positions; per position, the
    canonical keys after each listed move (file order, distinct); and labels for the summary.
    As in the packer (play_book_text.hpp), only lines with the seq of a position's first line add
    moves; a line reaching the position by another seq leaves its moves as they are."""
    keys, children, labels, first_seq = set(), {}, {}, {}
    for line in open(path, encoding="utf-8"):
        parts = line.split()
        if len(parts) != 3 or parts[0] != "S":
            continue
        board, cells = board_obj(), [[0] * 9 for _ in range(9)]
        seq = [] if parts[1] == "-" else [int(cell) for cell in parts[1].split(",")]
        for i, cell in enumerate(seq):
            r, c = rc_of(cell)
            cells[r][c] = 1 + i % 2
            ops.make_move(board, (r, c))
        key = key_of(cells, active_board(ops.get_valid_moves(board)))
        keys.add(key)
        r, c = rc_of(int(parts[2]))
        cells[r][c] = 1 + len(seq) % 2
        ops.make_move(board, (r, c))
        child = key_of(cells, active_board(ops.get_valid_moves(board)))
        kids = children.setdefault(key, [])
        if child in kids or first_seq.setdefault(key, seq) != seq:
            continue  # the same move again, a symmetric twin, or another seq: ignored, as by the packer
        kids.append(child)
        seq_label = ", ".join("%d %d" % rc_of(cell) for cell in seq) or "the empty board"
        lab = labels.setdefault(key, {"seq": seq_label, "ply": len(seq), "moves": []})
        lab["moves"].append(f"{r} {c}")
    return keys, children, labels


def play(bot, bot_first, rng, max_ply, book, choices):
    book_keys, children, multi = (None, {}, set()) if book is None else book
    proc = subprocess.Popen([bot], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL, text=True)
    board = board_obj()
    cells = [[0] * 9 for _ in range(9)]
    moves_so_far = []
    last = (-1, -1)
    n_book, errors, times = 0, [], []
    try:
        for ply in range(max_ply):
            if ops.check_game_finished(board):
                break
            valid = ops.get_valid_moves(board)
            book_pos = None  # the canonical key of the position where the bot just played from the book
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
                if ply == 0 and move != (4, 4):
                    errors.append(f"opened {move[0]} {move[1]}, not center-center (4 4)")
                from_book = len(reply) > 2 and reply[2] == "BOOK"
                n_book += from_book
                if book_keys is not None and ply > 0:
                    pos = key_of(cells, active_board(valid))
                    want = pos in book_keys
                    if from_book != want:
                        errors.append(f"ply {ply}: {'book move' if from_book else 'no book move'} but position "
                                      f"{'is' if want else 'is not'} in the book")
                    if from_book and want:
                        book_pos = pos
                if book_keys is None and not bot_first and ply == 1 and last == (4, 4) and not from_book:
                    errors.append("no book move at our reply to center-center")
            else:
                if ply == 0 and book_keys is not None:
                    move = (4, 4) if rng.random() < 0.9 else rng.choice(valid)
                elif book_keys is not None and rng.random() < 0.9:
                    stay, into_multi = [], []
                    for m in valid:
                        cells[m[0]][m[1]] = 1 + ply % 2
                        b2 = board_obj()
                        for mm in moves_so_far + [m]:
                            ops.make_move(b2, mm)
                        if not ops.check_game_finished(b2):
                            k = key_of(cells, active_board(ops.get_valid_moves(b2)))
                            if k in book_keys:
                                stay.append(m)
                                if k in multi:
                                    into_multi.append(m)
                        cells[m[0]][m[1]] = 0
                    if into_multi and rng.random() < STEER_MULTI:
                        move = rng.choice(into_multi)
                    else:
                        move = rng.choice(stay) if stay else rng.choice(valid)
                else:
                    move = rng.choice(valid)
                last = move
            ops.make_move(board, move)
            cells[move[0]][move[1]] = 1 + ply % 2
            moves_so_far.append(move)
            if book_pos is not None:  # the bot's book move must lead to one of the listed moves' positions
                child = key_of(cells, active_board(ops.get_valid_moves(board)))
                kids = children[book_pos]
                if child not in kids:
                    errors.append(f"ply {ply}: book move {move[0]} {move[1]} is none of the position's "
                                  f"{len(kids)} listed move(s)")
                elif book_pos in multi:
                    choices[book_pos][kids.index(child)] += 1
    finally:
        proc.kill()
    return n_book, times, errors


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
    book, labels, choices = None, {}, {}
    if book_path:
        keys, children, labels = load_book(book_path)
        multi = {k for k, kids in children.items() if len(kids) > 1}
        book = (keys, children, multi)
        choices = {k: [0] * len(children[k]) for k in multi}
        print(f"text book: {len(keys)} positions, {len(multi)} with several moves "
              f"({sum(len(children[k]) == 2 for k in multi)} with 2, {sum(len(children[k]) == 3 for k in multi)} "
              f"with 3, {sum(len(children[k]) > 3 for k in multi)} with more)")
        if not keys:
            print("no book: the text book is empty, so every reply must be a search move (0 BOOK replies)")
    rng = random.Random(2026)
    failures = 0
    all_times, first_turn, book_moves = [], [], {True: [], False: []}
    for g in range(games):
        bot_first = g % 2 == 0
        n_book, times, errors = play(bot, bot_first, rng, max_ply, book, choices)
        all_times += times[1:]
        first_turn.append(times[0])
        book_moves[bot_first].append(n_book)
        failures += bool(errors)
        print(f"{time.strftime('%H:%M:%S')}  game {g + 1}/{games}  bot moves "
              f"{'first' if bot_first else 'second'}: {n_book} book moves  {'ok' if not errors else 'FAIL: ' + errors[0]}",
              flush=True)
    all_times.sort()
    mean = lambda xs: sum(xs) / max(1, len(xs))
    if choices:
        print("\npositions with several book moves: the bot's choices (moves as row col, in the orientation of "
              "the position's first line; the first move is the primary)")
        for k in sorted(choices, key=lambda k: (labels[k]["ply"], labels[k]["seq"])):
            lab, counts = labels[k], choices[k]
            moves = ", ".join(f"{m}: {n}" for m, n in zip(lab["moves"], counts))
            print(f"  after {lab['seq']} (ply {lab['ply']}): {moves}  ({sum(counts)} visits)")
    print(f"\n{games - failures}/{games} games played from the book exactly where expected "
          f"(book moves per game: first {mean(book_moves[True]):.1f}, second {mean(book_moves[False]):.1f})")
    first_turn.sort()
    print(f"first turn: max {first_turn[-1]:.1f} ms (median {first_turn[len(first_turn) // 2]:.1f}); later moves: "
          f"median {all_times[len(all_times) // 2]:.1f} ms, max {all_times[-1]:.1f} ms "
          f"over {len(all_times)} replies")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
