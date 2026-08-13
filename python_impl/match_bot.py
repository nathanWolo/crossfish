#!/usr/bin/env python3
"""Match-protocol wrapper for the legacy Python engine (crossfish_v17)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from python_impl.crossfish import board_obj, crossfish_v17


def cpp_to_py(mb, sq):
    return (mb // 3) * 3 + sq // 3, (mb % 3) * 3 + sq % 3


def py_to_cpp(row, col):
    return (row // 3) * 3 + (col // 3), (row % 3) * 3 + (col % 3)


def main():
    sys.stdout.reconfigure(line_buffering=True)
    engine = crossfish_v17()
    board = board_obj()
    for raw in sys.stdin:
        parts = raw.split()
        if not parts:
            continue
        cmd = parts[0]
        if cmd == "NEW":
            engine = crossfish_v17()
            board = board_obj()
        elif cmd == "APPLY":
            mb, sq = int(parts[1]), int(parts[2])
            engine.make_move(board, cpp_to_py(mb, sq))
        elif cmd == "GO":
            ms = int(parts[1])
            engine.thinking_time = max(ms, 1) / 1000.0
            move = engine.get_best_move(board)
            engine.make_move(board, move)
            mb, sq = py_to_cpp(int(move[0]), int(move[1]))
            print(mb, sq, flush=True)


if __name__ == "__main__":
    main()
