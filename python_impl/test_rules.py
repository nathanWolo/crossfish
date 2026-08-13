"""Independent UTTT rules tests for the legacy Python board.

Perft counts are frozen against cpp_impl/unit_tests.cpp. If either suite
changes, the other must be updated in the same change.
"""
import unittest

from python_impl.board import board_obj
from python_impl.operations import ops

# Same table as STARTPOS_PERFT in cpp_impl/unit_tests.cpp
STARTPOS_PERFT = {
    1: 81,
    2: 720,
    3: 6336,
    4: 55080,
    5: 473256,
}


def perft(board, depth):
    if depth == 0:
        return 1
    if ops.check_game_finished(board):
        return 0
    moves = ops.get_valid_moves(board)
    if not moves:
        return 0
    if depth == 1:
        return len(moves)
    nodes = 0
    for move in moves:
        ops.make_move(board, move)
        nodes += perft(board, depth - 1)
        ops.undo_move(board)
    return nodes


def cpp_to_py(mini_board, square):
    row = (mini_board // 3) * 3 + (square // 3)
    col = (mini_board % 3) * 3 + (square % 3)
    return row, col


def py_to_cpp(row, col):
    mini_board = (row // 3) * 3 + (col // 3)
    square = (row % 3) * 3 + (col % 3)
    return mini_board, square


class TestCoordinateMap(unittest.TestCase):
    def test_roundtrip(self):
        for row in range(9):
            for col in range(9):
                mb, sq = py_to_cpp(row, col)
                self.assertEqual(cpp_to_py(mb, sq), (row, col))


class TestStartpos(unittest.TestCase):
    def test_first_move_count(self):
        board = board_obj()
        self.assertEqual(len(ops.get_valid_moves(board)), 81)
        self.assertFalse(ops.check_game_finished(board))
        self.assertEqual(ops.get_winner(board), "game is ongoing")

    def test_center_center_sends_to_same_board(self):
        board = board_obj()
        ops.make_move(board, (4, 4))
        moves = ops.get_valid_moves(board)
        self.assertEqual(len(moves), 8)
        for r, c in moves:
            self.assertEqual((r // 3, c // 3), (1, 1))
            self.assertNotEqual((r, c), (4, 4))

    def test_send_to_empty_other_board(self):
        board = board_obj()
        # (0, 1) is miniboard 0, square 1 -> opponent plays in miniboard 1.
        ops.make_move(board, (0, 1))
        moves = ops.get_valid_moves(board)
        self.assertEqual(len(moves), 9)
        for r, c in moves:
            self.assertEqual((r // 3, c // 3), (0, 1))


class TestMakeUndo(unittest.TestCase):
    def test_undo_restores_startpos(self):
        board = board_obj()
        ops.make_move(board, (0, 0))
        ops.undo_move(board)
        self.assertEqual(board.n_moves, 0)
        self.assertEqual(len(ops.get_valid_moves(board)), 81)
        self.assertFalse(board.markers.any())

    def test_random_play_undo_stack(self):
        import numpy as np
        rng = np.random.default_rng(123)
        board = board_obj()
        history = []
        for _ in range(30):
            if ops.check_game_finished(board):
                break
            moves = ops.get_valid_moves(board)
            if not moves:
                break
            move = moves[int(rng.integers(0, len(moves)))]
            history.append((move, board.n_moves, board.markers.copy(), board.miniboxes.copy()))
            ops.make_move(board, move)
        while history:
            move, n, markers, boxes = history.pop()
            ops.undo_move(board)
            self.assertEqual(board.n_moves, n)
            self.assertTrue((board.markers == markers).all())
            self.assertTrue((board.miniboxes == boxes).all())


class TestWinners(unittest.TestCase):
    def test_minibox_line_finishes_game_on_global_line(self):
        board = board_obj()
        board.miniboxes[0, 0, 0] = True
        board.miniboxes[0, 1, 0] = True
        board.miniboxes[0, 2, 0] = True
        board.n_moves = 1
        self.assertTrue(ops.check_game_finished(board))
        self.assertEqual(ops.get_winner(board), "agent 1 wins")

    def test_more_miniboards_wins_full_macro(self):
        board = board_obj()
        board.miniboxes[:, :, 0] = False
        board.miniboxes[:, :, 1] = False
        board.miniboxes[:, :, 2] = False
        # 4-3-2 split, no 3-in-a-row for either player.
        for r, c in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            board.miniboxes[r, c, 0] = True
        for r, c in [(0, 2), (1, 2), (2, 0)]:
            board.miniboxes[r, c, 1] = True
        board.miniboxes[2, 1, 2] = True
        board.miniboxes[2, 2, 2] = True
        self.assertTrue(ops.check_game_finished(board))
        self.assertEqual(ops.get_winner(board), "agent 1 wins")


class TestPerft(unittest.TestCase):
    def test_startpos_perft(self):
        for depth, expected in STARTPOS_PERFT.items():
            board = board_obj()
            self.assertEqual(perft(board, depth), expected, msg=f"perft({depth})")
            self.assertEqual(board.n_moves, 0)

    def test_divide_matches_perft2(self):
        board = board_obj()
        total = 0
        for move in ops.get_valid_moves(board):
            ops.make_move(board, move)
            total += perft(board, 1)
            ops.undo_move(board)
        self.assertEqual(total, STARTPOS_PERFT[2])


if __name__ == "__main__":
    unittest.main()
