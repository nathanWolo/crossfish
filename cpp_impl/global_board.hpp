#pragma once

#include <array>
#include <bitset>
#include <cstdlib>
#include <iostream>
#include <random>
#include <stack>
#include <vector>
#include <immintrin.h>

// Shared board used by local SPRT (test_bots.cpp) and unit tests.
// Codingame submissions stay self-contained in crossfish.cpp / cg_legend_hce.cpp.

struct MiniBoard {
    std::array<int, 2> markers = {0, 0};
};

struct Move {
    int mini_board = 99;
    int square = 99;
};

struct TTEntry {
    int depth;
    int score;
    int flag;
    uint64_t zobrist_hash;
    Move best_move;
};

class GlobalBoard {
    public:
        int miniboard_mask = (1 << 9) - 1;
        /*
        0 1 2
        3 4 5
        6 7 8
        */
        std::array<int, 8> win_masks = {(1 << 0) + (1 << 1) + (1 << 2),
                                            (1 << 3) + (1 << 4) + (1 << 5),
                                            (1 << 6) + (1 << 7) + (1 << 8),
                                            (1 << 0) + (1 << 3) + (1 << 6),
                                            (1 << 1) + (1 << 4) + (1 << 7),
                                            (1 << 2) + (1 << 5) + (1 << 8),
                                            (1 << 0) + (1 << 4) + (1 << 8),
                                            (1 << 2) + (1 << 4) + (1 << 6)};
        std::array<MiniBoard, 9> mini_boards;
        std::array<int, 3> mini_board_states = {0, 0, 0}; // 0 = p0, 1 = p1, 2 = draw
        std::stack<Move> move_history;
        uint64_t zobrist_hash = 0;
        std::array<std::array<std::array<uint64_t, 9>, 9>, 2> move_hashes; //player, mini board, square
        std::array<std::array<uint64_t, 9>, 3> mini_board_hashes; //p0/p1/draw, mini board
        std::array<uint64_t, 9> legal_mini_board_hashes;
        uint64_t player_to_move_hash;
        int n_moves = 0;
        bool prev_move_was_pass = false;
        void pass() {
            n_moves++;
            zobrist_hash ^= player_to_move_hash;
            prev_move_was_pass = true;
        }
        void unpass() {
            n_moves--;
            zobrist_hash ^= player_to_move_hash;
            prev_move_was_pass = false;
        }
        bool is_capture_avx(Move &move) {
            int miniboard_markers = mini_boards[move.mini_board].markers[n_moves % 2];
            miniboard_markers |= (1 << move.square);

            __m256i markers_vec = _mm256_set1_epi32(miniboard_markers);
            __m256i win_masks_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(win_masks.data()));
            __m256i result_vec = _mm256_and_si256(markers_vec, win_masks_vec);
            result_vec = _mm256_cmpeq_epi32(result_vec, win_masks_vec);
            int mask = _mm256_movemask_ps(_mm256_castsi256_ps(result_vec));
            return mask != 0;
        }
        void makeMove(Move move) {
            int occupied = mini_boards[move.mini_board].markers[0] | mini_boards[move.mini_board].markers[1];
            int out_of_play = mini_board_states[0] | mini_board_states[1] | mini_board_states[2];
            if (((occupied & (1 << move.square)) != 0)
            || ((out_of_play & (1 << move.mini_board)) != 0)
            || move.mini_board > 8 || move.square > 8 || move.mini_board < 0 || move.square < 0)
            {
                std::cerr << "ILLEGAL MOVE MADE: " << move.mini_board << " " << move.square << std::endl;
                if (!move_history.empty()) {
                    std::cerr << "Last move: " << move_history.top().mini_board << " " << move_history.top().square << std::endl;
                }
                std::cerr << "First illegal move block" << std::endl;

                if ((occupied & (1 << move.square)) != 0) {
                    std::cerr << "Square occupied" << std::endl;
                }
                if ((out_of_play & (1 << move.mini_board)) != 0) {
                    std::cerr << "Board out of play" << std::endl;
                    std::cerr << "Out of play: " << std::bitset<9>(out_of_play) << std::endl;
                    std::cerr << "Mini board: " << std::bitset<9>(mini_boards[move.mini_board].markers[0] | mini_boards[move.mini_board].markers[1]) << std::endl;
                    bool won_by_p0 = (mini_board_states[0] & (1 << move.mini_board)) != 0;
                    bool won_by_p1 = (mini_board_states[1] & (1 << move.mini_board)) != 0;
                    bool drawn = (mini_board_states[2] & (1 << move.mini_board)) != 0;
                    std::cerr << "Won by p0: " << won_by_p0 << " Won by p1: " << won_by_p1 << " Drawn: " << drawn << std::endl;
                }
                if (move.mini_board > 8 || move.square > 8 || move.mini_board < 0 || move.square < 0) {
                    std::cerr << "Move out of bounds" << std::endl;
                }

                print_board();
                std::exit(EXIT_FAILURE);
            }
            if (n_moves > 0) {
                Move prevMove = move_history.top();

                if (n_moves > 0 && ((out_of_play & (1 << prevMove.square)) == 0) && (move.mini_board != prevMove.square) && !prev_move_was_pass)
                    {
                        std::cerr << "ILLEGAL MOVE MADE: " << move.mini_board << " " << move.square << std::endl;
                        std::cerr << "Last move: " << move_history.top().mini_board << " " << move_history.top().square << std::endl;
                        std::cerr << "Second illegal move block" << std::endl;
                        print_board();
                        std::exit(EXIT_FAILURE);

                    }
            }
            if (n_moves > 0) {
                zobrist_hash ^= legal_mini_board_hashes[move_history.top().square];
            }
            move_history.push(move);
            mini_boards[move.mini_board].markers[n_moves % 2] |= (1 << move.square);
            mini_boards[move.mini_board].markers[n_moves % 2] &= miniboard_mask;
            zobrist_hash ^= move_hashes[n_moves % 2][move.mini_board][move.square];
            zobrist_hash ^= legal_mini_board_hashes[move.square];

            if(is_capture_avx(move)) {
                mini_board_states[n_moves % 2] |= (1 << move.mini_board);
                zobrist_hash ^= mini_board_hashes[n_moves % 2][move.mini_board];
            }

            else if (((mini_boards[move.mini_board].markers[0] | mini_boards[move.mini_board].markers[1]) & miniboard_mask) == miniboard_mask) {
                mini_board_states[2] |= (1 << move.mini_board);
                zobrist_hash ^= mini_board_hashes[2][move.mini_board];
            }
            zobrist_hash ^= player_to_move_hash;
            n_moves++;
        }
        void unmakeMove() {
            if (n_moves == 0) {
                std::cerr << "No moves to unmake" << std::endl;
                return;
            }
            n_moves--;
            zobrist_hash ^= player_to_move_hash;
            Move move = move_history.top();
            move_history.pop();

            if ((mini_board_states[0] & (1 << move.mini_board)) != 0) {
                zobrist_hash ^= mini_board_hashes[0][move.mini_board];
                mini_board_states[0] &= ~(1 << move.mini_board);
            }
            else if ((mini_board_states[1] & (1 << move.mini_board)) != 0) {
                zobrist_hash ^= mini_board_hashes[1][move.mini_board];
                mini_board_states[1] &= ~(1 << move.mini_board);
            }
            else if ((mini_board_states[2] & (1 << move.mini_board)) != 0) {
                zobrist_hash ^= mini_board_hashes[2][move.mini_board];
                mini_board_states[2] &= ~(1 << move.mini_board);
            }

            mini_boards[move.mini_board].markers[n_moves % 2] &= ~(1 << move.square);
            mini_boards[move.mini_board].markers[n_moves % 2] &= miniboard_mask;
            zobrist_hash ^= move_hashes[n_moves % 2][move.mini_board][move.square];
            zobrist_hash ^= legal_mini_board_hashes[move.square];
            if (n_moves > 0) {
                zobrist_hash ^= legal_mini_board_hashes[move_history.top().square];
            }
        }

        bool won_avx(int player) {
            int markers = mini_board_states[player];
            __m256i markers_vec = _mm256_set1_epi32(markers);
            __m256i win_masks_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(win_masks.data()));
            __m256i result_vec = _mm256_and_si256(markers_vec, win_masks_vec);
            result_vec = _mm256_cmpeq_epi32(result_vec, win_masks_vec);
            int mask = _mm256_movemask_ps(_mm256_castsi256_ps(result_vec));
            return mask != 0;
        }

        int checkWinner() {
            if (won_avx(0)) {
                return 0;
            }
            else if (won_avx(1)) {
                return 1;
            }
            else if ((mini_board_states[0] | mini_board_states[1] | mini_board_states[2]) == miniboard_mask) {
                if (__builtin_popcount(mini_board_states[0]) > __builtin_popcount(mini_board_states[1])) {
                    return 0;
                }
                else if (__builtin_popcount(mini_board_states[0]) < __builtin_popcount(mini_board_states[1])) {
                    return 1;
                }
                else {
                    return 2;
                }
            }
            return -1;
        }

        int fillCaptures(Move* dst) {
            int n = 0;
            if (n_moves == 0) {
                return 0;
            }
            int active_square = move_history.top().square;
            int out_of_play = mini_board_states[0] | mini_board_states[1] | mini_board_states[2];

            if ((out_of_play & (1 << active_square)) == 0 ) {
                int marked = mini_boards[active_square].markers[0] | mini_boards[active_square].markers[1];
                for (int i = 0; i < 9; i++) {
                    if ((marked & (1 << i)) == 0)
                    {
                        Move move = {active_square, i};
                        if (is_capture_avx(move)) {
                            dst[n++] = move;
                        }
                    }
                }
            }
            else {
                for (int i = 0; i < 9; i++) {
                    if ((out_of_play & (1 << i)) == 0)
                    {
                        int marked = mini_boards[i].markers[0] | mini_boards[i].markers[1];
                        for (int j = 0; j < 9; j++) {
                            if ((marked & (1 << j)) == 0)
                            {
                                Move move = {i, j};
                                if (is_capture_avx(move)) {
                                    dst[n++] = move;
                                }
                            }
                        }
                    }
                }
            }
            return n;
        }

        int fillLegalMoves(Move* dst) {
            int n = 0;
            if (n_moves == 0) {
                for (int i = 0; i < 9; i++) {
                    for (int j = 0; j < 9; j++) {
                        dst[n++] = Move{i, j};
                    }
                }
            } else {
                int active_square = move_history.top().square;
                int out_of_play = mini_board_states[0] | mini_board_states[1] | mini_board_states[2];
                if (((out_of_play & (1 << active_square)) != 0) || prev_move_was_pass) {
                    for (int i = 0; i < 9; i++) {
                        if ((out_of_play & (1 << i)) == 0)
                        {
                            int marked = mini_boards[i].markers[0] | mini_boards[i].markers[1];
                            for (int j = 0; j < 9; j++) {
                                if ((marked & (1 << j)) == 0)
                                {
                                    dst[n++] = Move{i, j};
                                }
                            }
                        }
                    }
                } else {
                    int marked = mini_boards[active_square].markers[0] | mini_boards[active_square].markers[1];
                    for (int i = 0; i < 9; i++) {
                        if ((marked & (1 << i)) == 0 )
                        {
                            dst[n++] = Move{active_square, i};
                        }
                    }
                }
            }
            return n;
        }

        std::vector<Move> get_captures() {
            Move buf[81];
            int n = fillCaptures(buf);
            return std::vector<Move>(buf, buf + n);
        }
        std::vector<Move> getLegalMoves() {
            std::vector<Move> legal_moves;
            legal_moves.reserve(9);
            if (n_moves == 0) {
                for (int i = 0; i < 9; i++) {
                    for (int j = 0; j < 9; j++) {
                        Move move = {i, j};
                        legal_moves.push_back(move);
                    }
                }
            } else {
                int active_square = move_history.top().square;
                int out_of_play = mini_board_states[0] | mini_board_states[1] | mini_board_states[2];
                if (((out_of_play & (1 << active_square)) != 0) || prev_move_was_pass) {
                    for (int i = 0; i < 9; i++) {
                        if ((out_of_play & (1 << i)) == 0)
                        {
                            for (int j = 0; j < 9; j++) {
                                int marked = mini_boards[i].markers[0] | mini_boards[i].markers[1];
                                if ((marked & (1 << j)) == 0)
                                {
                                    Move move = {i, j};
                                    legal_moves.push_back(move);
                                }
                            }
                        }
                    }
                } else {
                    int marked = mini_boards[active_square].markers[0] | mini_boards[active_square].markers[1];
                    for (int i = 0; i < 9; i++) {
                        if ((marked & (1 << i)) == 0 )
                        {
                            Move move = {active_square, i};
                            legal_moves.push_back(move);
                        }
                    }
                }
            }
            return legal_moves;
        }

        void print_board() {
            for (int row = 0; row < 9; row++) {
                for (int col = 0; col < 9; col++) {
                    int mini_board_index = (row /3) *3 + (col/3);
                    std::cerr << " ";
                    int square_index = (row % 3) * 3 + (col % 3);
                    if (mini_boards[mini_board_index].markers[0] & (1 << square_index)) {
                        std::cerr <<  'O';
                    } else if (mini_boards[mini_board_index].markers[1] & (1 << square_index)) {
                        std::cerr <<  'X';
                    }
                    else {
                        std::cerr << '.';
                    }
                    if (col % 3 ==  2) {
                        std::cerr << " |";
                    }
                }
                std::cerr << std::endl;
                if (row % 3 ==  2) {
                    std::cerr << "---------------------" << std::endl;
                }
            }
        }

    GlobalBoard(const GlobalBoard& other) = default;
    GlobalBoard& operator=(const GlobalBoard& other) = default;

    GlobalBoard() {
        for (int i = 0; i < 9; i++) {
            mini_boards[i] = MiniBoard();
        }

        std::mt19937_64 rng(69420);
        std::uniform_int_distribution<uint64_t> dist(1ull, UINT64_MAX - 1);
        player_to_move_hash = dist(rng);
        for (int p = 0; p < 2; p++) {
            for (int m = 0; m < 9; m++) {
                for (int s = 0; s < 9; s++) {
                    move_hashes[p][m][s] = dist(rng);
                }
            }
        }
        for (int st = 0; st < 3; st++) {
            for (int m = 0; m < 9; m++) {
                mini_board_hashes[st][m] = dist(rng);
            }
        }
        for (int s = 0; s < 9; s++) {
            legal_mini_board_hashes[s] = dist(rng);
        }
    }

};
