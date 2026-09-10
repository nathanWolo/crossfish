#include <iostream>
#include <cstring>
#include <cstdint>
#include <vector>
#include <stdlib.h>
#include <unistd.h>
#include <chrono>
#include <array>
#include <cmath>
#include <string>
#include <random>
#include <stack>
#include <future>
#include <numeric>
#include <thread>
#include <bitset>
#include <limits>
#include <mutex>
#include <immintrin.h>
#pragma GCC optimize("O3")
#pragma GCC option("arch=native", "tune=native", "no-zero-upper")
#pragma GCC optimization("unroll-loops")
#pragma GCC target("avx2,bmi,bmi2,lzcnt,popcnt")
// Strength patch vs the original legend submission (SPRT at 20ms, 569 games):
// W 349 / D 61 / L 159, +121 Elo, LLR +3.00. Changes: working Zobrist keys,
// TT cutoffs with matching bound flags, skip finished miniboards in eval,
// history heuristic, 2^18 TT, less frequent time checks.
// Alloc-free movegen: stack Move[81]/int[81] in search/qsearch instead of std::vector.
// 3x3 LUT replaces the inner miniboard eval loop (same features, ~2x NPS).
// Texel-tuned eval weights (per-weight Adam). SPRT 20ms: +25 Elo, LLR +3.04.
// NNUE variant: minires_d8h4 residual (D=8 H=4, 256-centroid emb).
// Equal-depth +54 Elo; 20ms +11; 95ms +7. HCE RFP, MiniNet at qsearch.
// Skip MiniNet at qsearch when HCE already fail-highs. SPRT vs previous
// codingame_nnue: 20ms N 2048 W 869 D 492 L 687, +31.0 +/- 13.2 Elo, LLR +3.01;
// 95ms N 3272 W 1257 D 942 L 1073, +19.6 +/- 10.1 Elo, LLR +3.01.
// NPS bundle on top of that: AVX MiniNet, skip MiniNet when HCE cannot
// reach alpha even at +8000, LUT capture/block/tiar. SPRT vs MiniNet-skip
// baseline: 20ms N 2152 W 887 D 558 L 707, +29.1 +/- 12.7 Elo, LLR +3.01;
// 95ms N 2416 W 939 D 715 L 762, +25.5 +/- 11.6 Elo, LLR +3.01.
// Correction history + logarithmic LMR vs the frozen #8 speed engine.
// Independent 20ms SPRT: N 1920 W 784 D 533 L 603, +32.85 +/- 13.25, LLR +3.10.
// Author 95ms fixed-sample: N 8016, +30.64 +/- 6.49, LLR 11.98.
// Hot-path LUTs + compact TT + persist corrhist + global-win order + incremental HCE
// vs that engine. Independent 20ms SPRT: N 1536 W 642 D 427 L 467,
// +39.76 +/- 14.83, LLR +3.04. Prev NPS 15.9M, Dev NPS 24.7M.
// Round-3 bundle: aspiration reset, ternary-index LUT, compact search board,
// canonical decided-miniboard TT keys, and root history aging. Direct 20ms
// SPRT vs PR #10: N 3328 W 1573 D 817 L 938, +67.12 +/- 10.38 Elo;
// LLR +3.05 for H0=+50 / H1=+55.
// Child TT prefetch: 20ms N 8032 W 2990 D 2268 L 2774,
// +9.35 +/- 6.44 Elo, LLR +3.01.
// Round-5 direct 95ms SPRT vs origin/main 0c50c95:
// N 7744 W 3282 D 2385 L 2077, +54.51 +/- 6.48 Elo,
// LLR +3.00274 for H0=+50 / H1=+55.
//a struct representing a 3x3 board with 16 bit integers
struct MiniBoard {
    std::array<int, 2> markers = {0, 0};
};

struct Move {
    int mini_board = 99;
    int square = 99;
};

struct TTEntry {
    int8_t depth;
    int8_t flag;
    int score;
    uint64_t zobrist_hash;
    Move best_move;
};

class GlobalBoard {
    private:
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
        //random 64 bit numbers used to update zobrist hash
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

            // Prepare a vector of miniboard_markers
            __m256i markers_vec = _mm256_set1_epi32(miniboard_markers);

            // Load win_masks into a vector
            __m256i win_masks_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(win_masks.data()));

            // Perform AND and compare operations
            __m256i result_vec = _mm256_and_si256(markers_vec, win_masks_vec);
            //Check if any of our results are equal to the win masks
            result_vec = _mm256_cmpeq_epi32(result_vec, win_masks_vec);

            // Aggregate results: if any of the win conditions is fully met, result is true
            int mask = _mm256_movemask_ps(_mm256_castsi256_ps(result_vec));
            return mask != 0;
        }
        void makeMove(Move move) {
            // make sure move is legal
            // int occupied = mini_boards[move.mini_board].markers[0] | mini_boards[move.mini_board].markers[1];
            // int out_of_play = mini_board_states[0] | mini_board_states[1] | mini_board_states[2];
            // if (((occupied & (1 << move.square)) != 0)
            // || ((out_of_play & (1 << move.mini_board)) != 0)
            // || move.mini_board > 8 || move.square > 8 || move.mini_board < 0 || move.square < 0)
            // {
            //     std::cerr << "ILLEGAL MOVE MADE: " << move.mini_board << " " << move.square << std::endl;
            //     std::cerr << "Last move: " << move_history.top().mini_board << " " << move_history.top().square << std::endl;
            //     std::cerr << "First illegal move block" << std::endl;

            //     //print which illegal move condition was met
            //     if ((occupied & (1 << move.square)) != 0) {
            //         std::cerr << "Square occupied" << std::endl;
            //     }
            //     if ((out_of_play & (1 << move.mini_board)) != 0) {
            //         std::cerr << "Board out of play" << std::endl;
            //         //print binary rep of out of play
            //         std::cerr << "Out of play: " << std::bitset<9>(out_of_play) << std::endl;
            //         //print binary rep of miniboard we tried to play in 
            //         std::cerr << "Mini board: " << std::bitset<9>(mini_boards[move.mini_board].markers[0] | mini_boards[move.mini_board].markers[1]) << std::endl;
            //         bool won_by_p0 = (mini_board_states[0] & (1 << move.mini_board)) != 0;
            //         bool won_by_p1 = (mini_board_states[1] & (1 << move.mini_board)) != 0;
            //         bool drawn = (mini_board_states[2] & (1 << move.mini_board)) != 0;
            //         std::cerr << "Won by p0: " << won_by_p0 << " Won by p1: " << won_by_p1 << " Drawn: " << drawn << std::endl;
            //     }
            //     if (move.mini_board > 8 || move.square > 8 || move.mini_board < 0 || move.square < 0) {
            //         std::cerr << "Move out of bounds" << std::endl;
            //     }

            //     print_board();
            //     std::exit(EXIT_FAILURE); // Terminate the program
            // }
            // if (n_moves > 0) {
            //     Move prevMove = move_history.top();

            //     if (n_moves > 0 && ((out_of_play & (1 << prevMove.square)) == 0) && (move.mini_board != prevMove.square) && !prev_move_was_pass) //we were not sent to a won or drawn board
            //         {
            //             std::cerr << "ILLEGAL MOVE MADE: " << move.mini_board << " " << move.square << std::endl;
            //             std::cerr << "Last move: " << move_history.top().mini_board << " " << move_history.top().square << std::endl;
            //             std::cerr << "Second illegal move block" << std::endl;
            //             print_board();
            //             std::exit(EXIT_FAILURE); // Terminate the program
                    
            //         }
            // }
                        if (n_moves > 0) {
                zobrist_hash ^= legal_mini_board_hashes[move_history.top().square];
            }

            move_history.push(move); //add the move to the list of moves
            mini_boards[move.mini_board].markers[n_moves % 2] |= (1 << move.square); //set the bit at the square to 1
            mini_boards[move.mini_board].markers[n_moves % 2] &= miniboard_mask; //make sure that only the last 9 bits are in use
            zobrist_hash ^= move_hashes[n_moves % 2][move.mini_board][move.square];
            zobrist_hash ^= legal_mini_board_hashes[move.square];

            if(is_capture_avx(move)) {
                mini_board_states[n_moves % 2] |= (1 << move.mini_board);
                zobrist_hash ^= mini_board_hashes[n_moves % 2][move.mini_board];
            }

            //check if the mini board is drawn
            else if (((mini_boards[move.mini_board].markers[0] | mini_boards[move.mini_board].markers[1]) & miniboard_mask) == miniboard_mask) {
                mini_board_states[2] |= (1 << move.mini_board);
                zobrist_hash ^= mini_board_hashes[2][move.mini_board];
            }
            zobrist_hash ^= player_to_move_hash;
            n_moves++;
        }
        void unmakeMove() {
            // if (n_moves == 0) {
            //     std::cerr << "No moves to unmake" << std::endl;
            //     return;
            // }
            n_moves--; //dec the number of moves so that the index is the same as when the move was made
            zobrist_hash ^= player_to_move_hash;
            Move move = move_history.top();
            move_history.pop();

            //check if that board was won, if it was, invert the state for it in the zobrist hash
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

            mini_boards[move.mini_board].markers[n_moves % 2] &= ~(1 << move.square); //remove marker
            mini_boards[move.mini_board].markers[n_moves % 2] &= miniboard_mask; //make sure that only the last 9 bits are in use
            zobrist_hash ^= move_hashes[n_moves % 2][move.mini_board][move.square];
            zobrist_hash ^= legal_mini_board_hashes[move.square];
                        if (n_moves > 0) {
                zobrist_hash ^= legal_mini_board_hashes[move_history.top().square];
            }
        }

        bool won_avx(int player) {
            //check if the player has won
            int markers = mini_board_states[player];
            // Prepare a vector of miniboard_markers
            __m256i markers_vec = _mm256_set1_epi32(markers);

            // Load win_masks into a vector
            __m256i win_masks_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(win_masks.data()));

            // Perform AND and compare operations
            __m256i result_vec = _mm256_and_si256(markers_vec, win_masks_vec);
            //Check if any of our results are equal to the win masks
            result_vec = _mm256_cmpeq_epi32(result_vec, win_masks_vec);

            // Aggregate results: if any of the win conditions is fully met, result is true
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
                // return 2;
                //winner has more won miniboards
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
            int8_t active_square = move_history.top().square;
            int out_of_play = mini_board_states[0] | mini_board_states[1] | mini_board_states[2];

            if ((out_of_play & (1 << active_square)) == 0 ) {
                int marked = mini_boards[active_square].markers[0] | mini_boards[active_square].markers[1];
                for (int8_t i = 0; i < 9; i++) {
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
                for (int8_t i = 0; i < 9; i++) {
                    if ((out_of_play & (1 << i)) == 0)
                    {
                        int marked = mini_boards[i].markers[0] | mini_boards[i].markers[1];
                        for (int8_t j = 0; j < 9; j++) {
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
                for (int8_t i = 0; i < 9; i++) {
                    for (int8_t j = 0; j < 9; j++) {
                        dst[n++] = Move{i, j};
                    }
                }
            } else {
                int8_t active_square = move_history.top().square;
                int out_of_play = mini_board_states[0] | mini_board_states[1] | mini_board_states[2];
                if (((out_of_play & (1 << active_square)) != 0) || prev_move_was_pass) {
                    for (int8_t i = 0; i < 9; i++) {
                        if ((out_of_play & (1 << i)) == 0)
                        {
                            int marked = mini_boards[i].markers[0] | mini_boards[i].markers[1];
                            for (int8_t j = 0; j < 9; j++) {
                                if ((marked & (1 << j)) == 0)
                                {
                                    dst[n++] = Move{i, j};
                                }
                            }
                        }
                    }
                } else {
                    int marked = mini_boards[active_square].markers[0] | mini_boards[active_square].markers[1];
                    for (int8_t i = 0; i < 9; i++) {
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
            Move buf[81];
            int n = fillLegalMoves(buf);
            return std::vector<Move>(buf, buf + n);
        }
        void print_board() {
            for (int row = 0; row < 9; row++) {
                for (int col = 0; col < 9; col++) {
                    int mini_board_index = (row /3) *3 + (col/3);
                    std::cerr << " ";
                    int square_index = (row % 3) * 3 + (col % 3);
                    // char symbol = mini_board_index;
                    if (mini_boards[mini_board_index].markers[0] & (1 << square_index)) {
                        std::cerr <<  'O'; // Player 0
                    } else if (mini_boards[mini_board_index].markers[1] & (1 << square_index)) {
                        std::cerr <<  'X'; // Player 1
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

// MiniNet residual weights (k-means-256 embeddings)
static const int MINI_PACK_RAW_BYTES = 29991;
static const char MINI_PACK_B64[] = R"MNUE(
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
6+8yte+1lsaEdgvVdhjbMVFcUe/QFe8JZ1j+AhW77xUVfILTaZKLc8qLKRJw4OxdkQlMQfqz+H89NRGxbwXps6j9bisFN58QUO+7
FRU3AV/pne/vFRXvfxUXK8sgLxUwKyRfFRXv7xUW3A3rFe8V7+8VFRVnFu8FGBWlCHcV9xVBlu8L1F9/FRUTFe/vVnr2NG1DtYwQ
+hk+HPDdyycdeAb1yANkPxVxwoay86mdJTodcnApzZNNqzinqQOCfGSpWZ0rWAETSLGJd0cCfXBDbfRl9CrV4KrFLwPmDsV26kuR
495aABVMDlrNJ0Gg+Al1FxO2aQOcJxVFicquVjm3MAPh97PUGgtEwTHGfc7psgVncyRY5f4MT8IVpU72y7e1PU+c1OTURNHqspXa
kF+2mEG5yB27DbkNIGSi1+7pLBXRon7jA4a57+8Jxj1kd6ELCB009xAXVANAGIRZ9iQJfBWEf+9kIZBN9HuQz6IrvjrKIIs4JAmV
fLi/Q39zf+/cAgN+0QmgwWOinLkDHuHaKs4mE1b37SG22KCdIw/xdtKLSa97Y4Rbm6/evaEgaap1XX4AU/8D+rWGkHknRkOkLxCd
6actysTNNKZwczEdihWk+SuATzZGq8o6Dl3vm2RE73Pun+oXTxV+pRVr4XrB0AuFUQI0ldAVehUO7xXKc79ZOu8QFRWAvxV/3RVl
Ae/kEe8VFM3We7yss0r3i++/iu+lk8F/lLoVJkP6LvnvE+9Z7xUVdRWqFe/vFRUVse8VAe9/Fe8Vqgzvt+/v7+8Vyu+lFRUVFRUV
shUVqxUIFRWRnO8V8hV3Ce+RAwOSFRUV7xXvFRUJxRW8ZBViNOHv9xC/Yr3wtDTPqhW7HRVygEO7+mLFOIWhTIsVJRU3jBWn5u+Q
CRUhEO+1Je9N0hbyUMZc3ZsV7nroFEJWqG3JncuLqxW7CX+trItdhIvxgAMVAJEL7xVoARUK7xUV7+8VOBUV94i7Mpjw3MEV0e+a
FRVwKkep7xVNFRVz6hVzfxX1QrJDzhAV+iD9ynBefA63Ke/KfxUB653o6kkqIRCcZojvtBW37xVHzRUWFRUV7+/v7xUVeu8VZO8V
qasV7+8VFe8V7xUV7xXvFe8VFRXvFRXvFe8VFRXvNxW8EBX5tB2cFRUVFe8V7+8VtKt3fBWq6e0V2hWvzk+qevroixVeEBW4t++A
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
YvE+Y0L+vmo76L5Dc0c/xOlAP/wGPr9NgUq/HIJFvxSoSr935U8/fno5P69ZhL9SYIS/pi6FP9JrhD9W7YQ/KISEPxdWhL82WoW/
Z2m+Pmcouz6R4Le+OcS+vpTfu74toL++9eXAPqOYtj71Cvu/1KD7v5tx+j9CXvs/lA78P8KB+z83rfu/Izr7v7JpgT9KcX4/Yid6
v1ZmgL8/y3y/n7yBv2F/gz88CXY/uc09v+uPPb/8bD8/PEk+P5eoPj/YGD4/k849vy75P7/kcrY/KXC2Pzj6tb85xLW/QgO2vzaF
tr9TBrU/fpG2PzEtkL6oOYu+ITmPPiVTlz56KJQ+2jyUPn1glr5wc4q+6M6XP6Lklj+CnJa/TPGXv2dll7/uLpi/snaYP5Qhlj9S
dZo/rwCdP7zMnL/2eZm/J3yavwkcmr8O0JU/F5meP6oZhz/3pog/gv2Hv7b4hb9YCYe/ReOGvw4ohD9rdYk/6WcawIlrGsBwhxpA
mC4aQDpuGkBCHhpAseEawPlBGsCQYna/u691v+XLdT9CqHQ/FkN2PyisdT/Sy3e/M4l1vyJgIr9mPhy/zcUZP0+wJj/3diQ/XPAj
PxSYMb/DsBe/gdjgv7I24b/ue+E/I/PgPwX/4D/6K+E/HMnhvxOK4b/gsri9m9nGvSxr2j3Xo7A9USLBPftotT1yJaa9lo3jvcsw
Ej+fLBg/UbMZvwgzFL/z8hW/Z58Sv9TfBT82QSA/NCOHO6P6Pjs4VAa7uI/PO4h9Ljs15g+7M24Luq620bl4WVi/jdpYv4EeWT/e
llg/ujxZP5qlWD+Hb1i/aMZZv6QOBD8CugI/qRMBv2lJ/L61h/u+RYMBv20EAT81C/0+mljSvzyg0r8P9tI/MYfSPzL10j+EjtI/
heHSv6zy0r9kL2w+2JKCPkjpg76oxFG+XIFevjD0Yb7EBDE+6fmJPnOWzT8zAc4//vzMv/Iuzb9W98y/vv/Nv2a9zD9WQM0/5vsf
QLK5H0Ay9h/A2MsfwP/rH8DpSh/AYosdQDgWIEAP0iI/4AAqP0JuLL/ppCi/mh8qv2YwJb8imBg/RlwzP94yFL9rCRa/NgkXP1tq
FD9sBxY/M00UPwmqE79o5Ri/cIs2Pu+nJT7+bkC+gQ5XvpcvTb6Q2j6+0XH6PV2GPD6amWM/WiVhP5CEXb/qOWG/LSVfv514Y7/0
HGY/PX1bP3DKrr4F5qa+DAinPj1ssz7l+K4+61GwPhtAvr6UVqC+TQeTv8gQk7+805M/gsqSP08Hkz9f1JI/AduSv3TMk79NQvw/
9az8P3r++7/1Qfu/HOL7v028/L8nkPs/j5n7P+Ff7z8bB+8/Mdvtvx1s7r/lbe6/7QDvv1wl7j+Nke4/uoq1v3JFtb9llLU/V0a1
P+V5tT9+TLU/aOu1v61vtb+Fa5I/7UKVP2L1lL9XG5K/OlaTvwhzkr+lvo0/Q22XP5mfOD49mkg+P8g7vmoIBL7MMA++PC0hvj6l
+T03/jo+NBJ4P15Ydz+2BXa/0bF3vzG2dr8jDHi/uwp3P9O7dT+lnQFA0uABQGbTAcDgcQHADaoBwLAAAsDs6QBA7tABQFsLpj/T
xKU/cCSlv6+Wpb9dWqW/bPSlv+ZtpT/q3qQ/yPvGvplNy75yUM0+ZjvEPqfLxz6UhcY+gurCvrHhzr6S/K+/TG+vv5W8rz+Ak68/
Wt+vP1i2rz+O3bC/y6avv0AXNL5DqSO+fBkkPpl3Tz4M2UY+DDNAPh94YL6Q4xe+zYRwP+zycT/YZXK/A6Ruv/P/b78I02+/7Oxo
P3M9dD+bsdI/tDXTP1Nu0r9Q/tK/Na3Sv7ow078+t9E/iv7SP+B0mj792pE+TMGMvoJIkr5rrom+D/uavig7nj5STX8+bH1aP5f4
WD8pFVa/NA5Yv+GzVr+XOlq/Zc9bP+u8VD8D1my/aR1nv3YQZz/1MHA/poNuP26Lbj90mne/srBkv0bACL8yBQO/41wCP7hRDz8F
1Qw/N50LP4zXF79N4gC/3QFCwOu4QcDnCEJAt4dBQJjHQECxLEJAAABBwH8UQsCubtA9GYqsPSsXgL1NJba92DCcvULlxr0J2gA+
bns5PeR5CsC1WQrAFBEKQI99CkDLVgpAxLkKQJQWCsAeMgrAIfsyPr0/Hj4pkRC+U8Ytvm3TG76fEDa+6xJGPtEv/T3dIIM+C+qE
PjUjj75cd4q+e6aKvhvbhb69tlE+y8qRPjx9ZL2XpYe9jGOfPWBYiD2CQZg94KlxPbrYR708bLa9d5bbPv312D7I7NS+Y2/QvgVu
z76PA9i+uWHUPja50D5+NH6/9Pp+vwV4gD97030/edN/P5QMfj/JZHu/5ueAv2WBazwB94q8xXyqvCB1Yr0ASTK9fh3cvKR2JL3m
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
W1y/cpJbvy8XUz9Mw2E/DDIQPnZbMT5UpkK+kjEdvrkjKL4QVxq+O4i8PfulXj5O9Ay/GHYOv194Dj80pQw/gSUOPzq+DD8bfA2/
a9UPv4sFZD9eymU/2nlmvxx5Yr9fymO/ucdjv7lgXD/C22g/ocuOPvDKnD5DEaC+WiOLvmgCkL4wmo2+bD1nPjMJqz4wpUa/9NFA
v04OOz8zsEM/iVVCP5IjRD93kFO/6y83vyOGAr5dmye+wKstPjEQJT5MODQ+0fALPucnD74htlW+q28TP0ljEz8oHxG/KW0PvxW9
D7+azhG/SugQP9XfED9S0zg/B4w9PwxWPr9iYja/Y6M4v9K+OL/Hjys/KotCP8lrlj0lWqM9Ld2YvYaSnL2gBZq9IDiYvUDFkD0Z
I6s9xPhTv1hQT79nlkk/MDZQP80NUD9ZalE/xRNhv8xLR79mZxS/MzEQv4GHDD9EIxg/5F0WP3SdFj+yRCW/qF4Lv1SQMz36CJE9
nFqpvUSLJb0GP1O9ogo7va9hOrzkHtk9KHyVPkYQlz7sqZO+2rmcvlddnb7uk5a+rkCbPhgrmz75KqE9KQnTPQaW6b1ntpe9y4+t
vaPmob16fOI8fwkIPhla7D3KcQ8+NpwYvp5V3L3WGPK9HUHrvSZbgj3uzio+FLoYv5ZUGr812R8/hq4cPxeuHj89LRo/71IUv9/U
Ir9qznM/orJxPwj1br/2Q3O/6Plwv2/xc78eK3Y/vn9tP3wyoj80X6E/PA6hvynyob+ma6G/fmeiv+s0oj8WLaA/VDMbPwYZFj9x
fhC/kmEZv4LJFL/lTBy/pFkhP129Cz9gF1G/hl1Qv6ewUT9RElE/D5BRP/toUT+IVVC/HJxRv07xTz7ihVY+QoVMvgiLTb52hk2+
6QNQvppmTz4RMVQ+qgPIPxwOyD9r98a/ZvPGvwOfxr9oEMi/4OLGPw7zxj+5koo/HVuMPy6ri7+FsYm/XdGKv0uMir9d54c/M1yN
P5rrMz8VtDE/MVkvv/tKLr+4fCy/4gczv1gAMz/IMSs/9Q1yvn1Hdb7Utno+4aN3Pl71dz4OD3U+Wdl1vuzRer6RsUw/6IJQP003
Ub9ZpUm/Xt9Lv8c5TL/ZUEA/Dr9UPxQonz9PraE/Joihv2z3nb9J+56/7a+ev99vmj95O6M/xFsjv5ycI7/OOyk/oGcnP8fxKD9L
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
)MNUE";

template <typename Board>
static int mini_board_constraint(const Board &b) {
    if (b.n_moves == 0 || b.prev_move_was_pass) return 9;
    int sent = b.move_history.top().square;
    int oop = b.mini_board_states[0] | b.mini_board_states[1] | b.mini_board_states[2];
    if ((oop & (1 << sent)) != 0) return 9;
    return sent;
}

static int mini_b64_decode(const char *s, unsigned char *out, int out_max) {
    auto val = [](char c) -> int {
        if (c >= 'A' && c <= 'Z') return c - 'A';
        if (c >= 'a' && c <= 'z') return c - 'a' + 26;
        if (c >= '0' && c <= '9') return c - '0' + 52;
        if (c == '+') return 62;
        if (c == '/') return 63;
        return -1;
    };
    int n = 0, acc = 0, bits = 0;
    for (const char *p = s; *p; p++) {
        int d = val(*p);
        if (d < 0) continue;
        acc = (acc << 6) | d;
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            if (n >= out_max) return -1;
            out[n++] = (unsigned char)((acc >> bits) & 255);
        }
    }
    return n;
}

static constexpr int MN_N = 19683;
static constexpr int MN_K = 256;
static constexpr int MN_D = 8;
static constexpr int MN_H = 4;
static constexpr int MN_IN = 80;
static uint8_t MN_CODE[MN_N];
static float MN_CENT[MN_K * MN_D];
static float MN_SUPER[4 * MN_D];
static float MN_LOC[9 * MN_D];
static float MN_CONSTR[10 * MN_D];
static float MN_ACTIVE[2 * MN_D];
static float MN_W1[MN_H * MN_IN];
static float MN_B1[MN_H];
static float MN_W2[MN_H];
static float MN_B2 = 0;
static bool MN_READY = false;
static uint8_t MN_MASK_CODE[1 << 18];
alignas(64) static float MN_FAST_BASE[MN_H], MN_FAST_CONSTR[10][MN_H];
alignas(64) static float MN_FAST_MB[9][MN_K][8][MN_H];

static int mini_mask_index(int mine, int opp) {
    int idx = 0;
    idx += (mine & 1) ? 1 : ((opp & 1) ? 2 : 0);
    idx += (mine & 2) ? 3 : ((opp & 2) ? 6 : 0);
    idx += (mine & 4) ? 9 : ((opp & 4) ? 18 : 0);
    idx += (mine & 8) ? 27 : ((opp & 8) ? 54 : 0);
    idx += (mine & 16) ? 81 : ((opp & 16) ? 162 : 0);
    idx += (mine & 32) ? 243 : ((opp & 32) ? 486 : 0);
    idx += (mine & 64) ? 729 : ((opp & 64) ? 1458 : 0);
    idx += (mine & 128) ? 2187 : ((opp & 128) ? 4374 : 0);
    idx += (mine & 256) ? 6561 : ((opp & 256) ? 13122 : 0);
    return idx;
}

static void mini_build_fast_tables() {
    for (int mine = 0; mine < 512; mine++) for (int opp = 0; opp < 512; opp++)
        MN_MASK_CODE[(mine << 9) | opp] =
            (mine & opp) ? 0 : MN_CODE[mini_mask_index(mine, opp)];
    for (int h = 0; h < MN_H; h++) {
        const float *row = MN_W1 + h * MN_IN;
        float base = MN_B1[h];
        for (int mb = 0; mb < 9; mb++) for (int d = 0; d < MN_D; d++)
            base += row[mb * MN_D + d] * MN_LOC[mb * MN_D + d];
        MN_FAST_BASE[h] = base;
        for (int c = 0; c < 10; c++) {
            float v = 0;
            for (int d = 0; d < MN_D; d++)
                v += row[9 * MN_D + d] * MN_CONSTR[c * MN_D + d];
            MN_FAST_CONSTR[c][h] = v;
        }
    }
    for (int mb = 0; mb < 9; mb++) for (int code = 0; code < MN_K; code++)
        for (int sc = 0; sc < 4; sc++) for (int act = 0; act < 2; act++)
            for (int h = 0; h < MN_H; h++) {
                const float *row = MN_W1 + h * MN_IN + mb * MN_D;
                float v = 0;
                for (int d = 0; d < MN_D; d++)
                    v += row[d] * (MN_CENT[code * MN_D + d]
                        + MN_SUPER[sc * MN_D + d] + MN_ACTIVE[act * MN_D + d]);
                MN_FAST_MB[mb][code][sc * 2 + act][h] = v;
            }
}

static bool mini_load_packed() {
    if (MN_READY) return true;
    static unsigned char buf[40000];
    int nb = mini_b64_decode(MINI_PACK_B64, buf, (int)sizeof(buf));
    const int need = MN_N + MN_K * MN_D * 4 + (4 + 9 + 10 + 2) * MN_D * 4
                     + MN_H * MN_IN * 4 + MN_H * 4 + MN_H * 4 + 4;
    if (nb < need) return false;
    int off = 0;
    memcpy(MN_CODE, buf + off, MN_N); off += MN_N;
    memcpy(MN_CENT, buf + off, MN_K * MN_D * 4); off += MN_K * MN_D * 4;
    memcpy(MN_SUPER, buf + off, 4 * MN_D * 4); off += 4 * MN_D * 4;
    memcpy(MN_LOC, buf + off, 9 * MN_D * 4); off += 9 * MN_D * 4;
    memcpy(MN_CONSTR, buf + off, 10 * MN_D * 4); off += 10 * MN_D * 4;
    memcpy(MN_ACTIVE, buf + off, 2 * MN_D * 4); off += 2 * MN_D * 4;
    memcpy(MN_W1, buf + off, MN_H * MN_IN * 4); off += MN_H * MN_IN * 4;
    memcpy(MN_B1, buf + off, MN_H * 4); off += MN_H * 4;
    memcpy(MN_W2, buf + off, MN_H * 4); off += MN_H * 4;
    memcpy(&MN_B2, buf + off, 4);
    mini_build_fast_tables();
    MN_READY = true;
    return true;
}

template <typename Board>
static int evaluate_mini(const Board &b) {
    if (!MN_READY && !mini_load_packed()) return 0;
    const int stm = b.n_moves & 1;
    const int c = mini_board_constraint(b);
    float h0 = MN_FAST_BASE[0] + MN_FAST_CONSTR[c][0];
    float h1 = MN_FAST_BASE[1] + MN_FAST_CONSTR[c][1];
    float h2 = MN_FAST_BASE[2] + MN_FAST_CONSTR[c][2];
    float h3 = MN_FAST_BASE[3] + MN_FAST_CONSTR[c][3];
    for (int mb = 0; mb < 9; mb++) {
        const int mine = b.mini_boards[mb].markers[stm];
        const int opp = b.mini_boards[mb].markers[stm ^ 1];
        int sc = 0, bit = 1 << mb;
        if (b.mini_board_states[stm] & bit) sc = 1;
        else if (b.mini_board_states[stm ^ 1] & bit) sc = 2;
        else if (b.mini_board_states[2] & bit) sc = 3;
        const float *v = MN_FAST_MB[mb][MN_MASK_CODE[(mine << 9) | opp]][sc * 2 + (c == mb)];
        h0 += v[0]; h1 += v[1]; h2 += v[2]; h3 += v[3];
    }
    float out = MN_B2;
    if (h0 > 0) out += MN_W2[0] * h0;
    if (h1 > 0) out += MN_W2[1] * h1;
    if (h2 > 0) out += MN_W2[2] * h2;
    if (h3 > 0) out += MN_W2[3] * h3;
    return (int)std::lround(out);
}

enum TTFlag { TT_EXACT = 0, TT_UPPER = 1, TT_LOWER = 2 };
static int g_fixed_search_depth = 0;
static bool g_disable_eval_prune = false;
static bool g_force_hce_eval = false;

class CrossfishDev {
       private:
        struct FastMoveStack {
            std::array<Move, 128> moves{};
            int count = 0;

            Move &top() { return moves[count - 1]; }
            const Move &top() const { return moves[count - 1]; }
            void push(const Move &move) { moves[count++] = move; }
            void pop() { count--; }
            bool empty() const { return count == 0; }
        };

        struct FastBoard {
            using MarkerHashTable =
                std::array<std::array<std::array<uint64_t, 512>, 9>, 2>;

            std::array<MiniBoard, 9> mini_boards;
            std::array<int, 3> mini_board_states;
            FastMoveStack move_history;
            uint64_t zobrist_hash;
            uint64_t tt_hash;
            const decltype(GlobalBoard::move_hashes) &move_hashes;
            const decltype(GlobalBoard::mini_board_hashes) &mini_board_hashes;
            const decltype(GlobalBoard::legal_mini_board_hashes) &legal_mini_board_hashes;
            const uint64_t &player_to_move_hash;
            const MarkerHashTable &marker_hashes;
            int n_moves;
            bool prev_move_was_pass;

            static const MarkerHashTable &get_marker_hashes(const GlobalBoard &board) {
                static const MarkerHashTable hashes = [&board] {
                    MarkerHashTable table{};
                    for (int p = 0; p < 2; p++) {
                        for (int mb = 0; mb < 9; mb++) {
                            for (int mask = 1; mask < 512; mask++) {
                                int prev = mask & (mask - 1);
                                int sq = __builtin_ctz(mask);
                                table[p][mb][mask] =
                                    table[p][mb][prev] ^ board.move_hashes[p][mb][sq];
                            }
                        }
                    }
                    return table;
                }();
                return hashes;
            }

            explicit FastBoard(const GlobalBoard &board)
                : mini_boards(board.mini_boards),
                  mini_board_states(board.mini_board_states),
                  zobrist_hash(board.zobrist_hash),
                  tt_hash(board.zobrist_hash),
                  move_hashes(board.move_hashes),
                  mini_board_hashes(board.mini_board_hashes),
                  legal_mini_board_hashes(board.legal_mini_board_hashes),
                  player_to_move_hash(board.player_to_move_hash),
                  marker_hashes(get_marker_hashes(board)),
                  n_moves(board.n_moves),
                  prev_move_was_pass(board.prev_move_was_pass) {
                auto history = board.move_history;
                move_history.count = (int)history.size();
                for (int i = move_history.count - 1; i >= 0; i--) {
                    move_history.moves[i] = history.top();
                    history.pop();
                }
                // Stones inside a decided miniboard can never affect play again.
                // Remove them from the search key so transpositions that reached
                // the same won/drawn miniboard through different move orders merge.
                int decided = mini_board_states[0]
                            | mini_board_states[1]
                            | mini_board_states[2];
                while (decided) {
                    int mb = __builtin_ctz(decided);
                    decided &= decided - 1;
                    for (int p = 0; p < 2; p++) {
                        tt_hash ^= marker_hashes[p][mb][mini_boards[mb].markers[p]];
                    }
                }
            }
        };

        static void xor_tt_hash(FastBoard &board, uint64_t value) {
            board.tt_hash ^= value;
        }

        static void xor_tt_hash(GlobalBoard &, uint64_t) {}

        static void xor_marker_hashes(FastBoard &board, int mb) {
            board.tt_hash ^= board.marker_hashes[0][mb][board.mini_boards[mb].markers[0]]
                          ^ board.marker_hashes[1][mb][board.mini_boards[mb].markers[1]];
        }

        static void xor_marker_hashes(GlobalBoard &, int) {}

        std::chrono::milliseconds thinking_time = std::chrono::milliseconds(95);
        Move root_best_move;
        std::chrono::time_point<std::chrono::high_resolution_clock> start_time =  std::chrono::high_resolution_clock::now();
        int min_val = -99999;
        int max_val = 99999;
        bool stopped = false;
    public:
        int root_score;
        int nodes;
        std::array<std::array<int, 9>, 128> killer_moves;
        std::array<std::array<std::array<int, 9>, 9>, 2> history_table{};
        Move counter_move[9][9];
        bool counters_ready = false;
        // Correction history: [stm][constrained miniboard, 9 = free choice][decided-miniboard mask].
        // A small exact index beats hashing the zobrist here: no collisions, no mixing
        // in the hot path, and it separates the two structural facts the fixed eval
        // misprices most - game phase and being locked into one miniboard.
        static constexpr int CORR_MB = 10;
        static constexpr int CORR_MASKS = 512;
        // Stored units are CORR_GRAIN x eval units. The gravity term bounds |entry| at
        // CORR_SCALE, so the applied shift never exceeds CORR_SCALE / CORR_GRAIN.
        static constexpr int CORR_SCALE = 16384;
        static constexpr int CORR_GRAIN = 24;
        static constexpr int CORR_MAX = 16384;
        // CORR_DIFF_MAX * CORR_MAX_WEIGHT <= CORR_SCALE keeps the update a contraction.
        static constexpr int CORR_DIFF_MAX = 1024;
        static constexpr int CORR_MAX_WEIGHT = 8;
        // Mate scores are ±(max_val - ply), i.e. distances to mate rather than
        // quantities the static eval can be measured against.
        static constexpr int CORR_MATE_BOUND = 90000;
        std::array<std::array<std::array<int, CORR_MASKS>, CORR_MB>, 2> corr_hist{};

        struct HceUndo {
            int16_t score = 0;
            uint8_t flags = 0;
        };
        int hce_local_score = 0;
        std::array<int, 2> hce_tiar_maps{};
        std::array<int16_t, 9> hce_mb_scores{};
        std::array<uint8_t, 9> hce_mb_flags{};
        std::array<HceUndo, 128> hce_undo{};
        bool hce_acc_ready = false;

        struct CompactTTEntry {
            uint64_t zobrist_hash = 0;
            int score = 0;
            int16_t depth = 0;
            int8_t flag = TT_EXACT;
            uint8_t best_move = 255;
        };
        static_assert(sizeof(CompactTTEntry) == 16);

        struct alignas(32) CompactTTBucket {
            CompactTTEntry entries[2];
        };
        static_assert(sizeof(CompactTTBucket) == 32);

        static uint8_t pack_tt_move(Move move) {
            if (move.mini_board < 0 || move.mini_board >= 9
                || move.square < 0 || move.square >= 9) {
                return 255;
            }
            return (uint8_t)(move.mini_board * 9 + move.square);
        }

        static Move unpack_tt_move(uint8_t packed) {
            if (packed >= 81) return Move{99, 99};
            return Move{packed / 9, packed % 9};
        }

        using FastMove = uint8_t;
        static constexpr FastMove NO_FAST_MOVE = 255;

        static FastMove pack_fast_move(int mb, int sq) {
            return (FastMove)((mb << 4) | sq);
        }

        static Move unpack_fast_move(FastMove move) {
            return Move{move >> 4, move & 15};
        }

        static FastMove tt_to_fast_move(uint8_t packed) {
            if (packed >= 81) return NO_FAST_MOVE;
            return pack_fast_move(packed / 9, packed % 9);
        }

        static uint8_t pack_tt_move(FastMove move) {
            if (move == NO_FAST_MOVE) return 255;
            return (uint8_t)((move >> 4) * 9 + (move & 15));
        }

        static const int tt_bucket_count = 1 << 17;
        std::vector<CompactTTBucket> transposition_table =
            std::vector<CompactTTBucket>(tt_bucket_count);

        static constexpr int N_TIAR_MASKS = 48;
        static constexpr int two_in_a_row_masks[N_TIAR_MASKS] = {
            (1 << 0) + (1 << 1),  (1 << 2),
            (1 << 1) + (1 << 2), (1 << 0),
            (1 << 3) + (1 << 4), (1 << 5),
            (1 << 4) + (1 << 5), (1 << 3),
            (1 << 6) + (1 << 7), (1 << 8),
            (1 << 7) + (1 << 8), (1 << 6),
            (1 << 0) + (1 << 3), (1 << 6),
            (1 << 3) + (1 << 6), (1 << 0),
            (1 << 1) + (1 << 4), (1 << 7),
            (1 << 4) + (1 << 7), (1 << 1),
            (1 << 2) + (1 << 5), (1 << 8),
            (1 << 5) + (1 << 8), (1 << 2),
            (1 << 0) + (1 << 4), (1 << 8),
            (1 << 4) + (1 << 8), (1 << 0),
            (1 << 2) + (1 << 4), (1 << 6),
            (1 << 4) + (1 << 6), (1 << 2),
            (1 << 0) + (1 << 2), (1 << 1),
            (1 << 3) + (1 << 5), (1 << 4),
            (1 << 6) + (1 << 8), (1 << 7),
            (1 << 0) + (1 << 6), (1 << 3),
            (1 << 1) + (1 << 7), (1 << 4),
            (1 << 2) + (1 << 8), (1 << 5),
            (1 << 0) + (1 << 8), (1 << 4),
            (1 << 2) + (1 << 6), (1 << 4)
        };
        int depth = 1;

        struct MiniLut {
            int8_t dead;
            int8_t p0_tiar;
            int8_t p1_tiar;
            int8_t p0_win1;
            int8_t p1_win1;
            int8_t p0_center;
            int8_t p1_center;
            int8_t p0_corner;
            int8_t p1_corner;
            int8_t p0_sq;
            int8_t p1_sq;
        };
        static constexpr int MINI_LUT_SIZE = 19683;
        static inline MiniLut mini_lut[MINI_LUT_SIZE];
        static inline int16_t mini_score[MINI_LUT_SIZE];
        static inline int16_t fast_local_score[1 << 18];
        // Sum of powers of three for each occupied square in a 9-bit mask.
        // For disjoint masks, ternary(p0, p1) = table[p0] + 2*table[p1].
        static inline uint16_t fast_mini_index[1 << 9];
        // Bit 0: player 0 has a live local two-in-a-row; bit 1: player 1.
        static inline uint8_t fast_tiar_flags[1 << 18];
        // Local wins depend only on one player's 9-bit occupancy. These tiny
        // tables replace repeated AVX line tests in make/unmake and ordering.
        static inline uint16_t fast_win_moves[1 << 9];
        static inline uint8_t fast_has_win[1 << 9];
        // Number of global two-in-a-row masks in `ours` whose third square
        // is not occupied by `theirs`, indexed as (ours << 9) | theirs.
        static inline uint8_t fast_threat_count[1 << 18];
        // Separate from MiniLut so HCE's hot LUT stays compact. Bit s set iff
        // playing square s wins / makes a 2-in-a-row for that player.
        static inline uint16_t mini_win_sq[MINI_LUT_SIZE][2];
        static inline uint16_t mini_tiar_sq[MINI_LUT_SIZE][2];
        static inline std::once_flag mini_lut_once;
        // One pawn = one extra corner square on a live miniboard (smallest HCE feature).
        // Kept at 10, not 1, so other terms can be tenths of a pawn. Texel freezes PAWN_IDX.
        static constexpr int PAWN_IDX = 7;
        static constexpr int PAWN = 10;
        static constexpr int ASP_PAWNS = 40;
        static constexpr int RFP_PAWNS = 50;
        static constexpr int FP_PAWNS = 80;
        static constexpr int QDELTA_PAWNS = 350;
        static constexpr int FREE_MOVE_PAWNS = 30;
        static constexpr int LUT_W_TIAR = 534;
        static constexpr int LUT_W_CENTER_SQ = 33;
        static constexpr int LUT_W_CORNER_SQ = PAWN;
        static constexpr int LUT_W_SQUARES = 33;
        // Random-play MiniNet residuals reached ~4500; slack for search positions.
        static constexpr int MINI_MAX = 8000;
        CrossfishDev() {
            mini_load_packed();
        }

        static constexpr int W_FREE_MOVE = FREE_MOVE_PAWNS * PAWN;

        static int mini_index(int p0, int p1) {
            return fast_mini_index[p0] + 2 * fast_mini_index[p1];
        }

        static void init_mini_lut() {
            std::call_once(mini_lut_once, []() {
                const int win[8] = {
                    (1 << 0) + (1 << 1) + (1 << 2),
                    (1 << 3) + (1 << 4) + (1 << 5),
                    (1 << 6) + (1 << 7) + (1 << 8),
                    (1 << 0) + (1 << 3) + (1 << 6),
                    (1 << 1) + (1 << 4) + (1 << 7),
                    (1 << 2) + (1 << 5) + (1 << 8),
                    (1 << 0) + (1 << 4) + (1 << 8),
                    (1 << 2) + (1 << 4) + (1 << 6)
                };
                const int tiar[] = {
                    (1 << 0) + (1 << 1),  (1 << 2),
                    (1 << 1) + (1 << 2), (1 << 0),
                    (1 << 3) + (1 << 4), (1 << 5),
                    (1 << 4) + (1 << 5), (1 << 3),
                    (1 << 6) + (1 << 7), (1 << 8),
                    (1 << 7) + (1 << 8), (1 << 6),
                    (1 << 0) + (1 << 3), (1 << 6),
                    (1 << 3) + (1 << 6), (1 << 0),
                    (1 << 1) + (1 << 4), (1 << 7),
                    (1 << 4) + (1 << 7), (1 << 1),
                    (1 << 2) + (1 << 5), (1 << 8),
                    (1 << 5) + (1 << 8), (1 << 2),
                    (1 << 0) + (1 << 4), (1 << 8),
                    (1 << 4) + (1 << 8), (1 << 0),
                    (1 << 2) + (1 << 4), (1 << 6),
                    (1 << 4) + (1 << 6), (1 << 2),
                    (1 << 0) + (1 << 2), (1 << 1),
                    (1 << 3) + (1 << 5), (1 << 4),
                    (1 << 6) + (1 << 8), (1 << 7),
                    (1 << 0) + (1 << 6), (1 << 3),
                    (1 << 1) + (1 << 7), (1 << 4),
                    (1 << 2) + (1 << 8), (1 << 5),
                    (1 << 0) + (1 << 8), (1 << 4),
                    (1 << 2) + (1 << 6), (1 << 4)
                };
                auto has_win = [&](int markers) {
                    for (int i = 0; i < 8; i++) {
                        if ((markers & win[i]) == win[i]) return true;
                    }
                    return false;
                };
                for (int markers = 0; markers < 512; markers++) {
                    int ternary = 0;
                    int pow3 = 1;
                    for (int s = 0; s < 9; s++) {
                        if (markers & (1 << s)) ternary += pow3;
                        pow3 *= 3;
                    }
                    fast_mini_index[markers] = (uint16_t)ternary;
                    fast_has_win[markers] = (uint8_t)has_win(markers);
                    int wins = 0;
                    for (int s = 0; s < 9; s++) {
                        int bit = 1 << s;
                        if ((markers & bit) == 0 && has_win(markers | bit)) {
                            wins |= bit;
                        }
                    }
                    fast_win_moves[markers] = (uint16_t)wins;
                }
                const int corners = (1 << 0) + (1 << 2) + (1 << 6) + (1 << 8);
                const int n_pairs = (int)(sizeof(tiar) / sizeof(tiar[0]) / 2);
                for (int idx = 0; idx < MINI_LUT_SIZE; idx++) {
                    int p0 = 0;
                    int p1 = 0;
                    int t = idx;
                    for (int s = 0; s < 9; s++) {
                        int cell = t % 3;
                        t /= 3;
                        if (cell == 1) p0 |= (1 << s);
                        else if (cell == 2) p1 |= (1 << s);
                    }
                    MiniLut e{};
                    bool p0_can = false;
                    bool p1_can = false;
                    for (int i = 0; i < 8; i++) {
                        if ((p1 & win[i]) == 0) p0_can = true;
                        if ((p0 & win[i]) == 0) p1_can = true;
                    }
                    e.dead = (!p0_can && !p1_can) ? 1 : 0;
                    int occ = p0 | p1;
                    uint16_t p0w = 0, p1w = 0, p0t = 0, p1t = 0;
                    for (int s = 0; s < 9; s++) {
                        if (occ & (1 << s)) continue;
                        if (has_win(p0 | (1 << s))) {
                            e.p0_win1 = 1;
                            p0w = (uint16_t)(p0w | (1 << s));
                        }
                        if (has_win(p1 | (1 << s))) {
                            e.p1_win1 = 1;
                            p1w = (uint16_t)(p1w | (1 << s));
                        }
                        int ours0 = p0 | (1 << s);
                        int ours1 = p1 | (1 << s);
                        for (int i = 0; i < n_pairs; i++) {
                            int pair = tiar[i * 2];
                            int third = tiar[i * 2 + 1];
                            if (((ours0 & pair) == pair) && ((occ & third) == 0)) {
                                p0t = (uint16_t)(p0t | (1 << s));
                            }
                            if (((ours1 & pair) == pair) && ((occ & third) == 0)) {
                                p1t = (uint16_t)(p1t | (1 << s));
                            }
                        }
                    }
                    mini_win_sq[idx][0] = p0w;
                    mini_win_sq[idx][1] = p1w;
                    mini_tiar_sq[idx][0] = p0t;
                    mini_tiar_sq[idx][1] = p1t;
                    for (int i = 0; i < n_pairs; i++) {
                        e.p0_tiar = (int8_t)(e.p0_tiar + ((__builtin_popcount(p0 & tiar[i * 2]) - __builtin_popcount(p1 & tiar[i * 2 + 1])) / 2));
                        e.p1_tiar = (int8_t)(e.p1_tiar + ((__builtin_popcount(p1 & tiar[i * 2]) - __builtin_popcount(p0 & tiar[i * 2 + 1])) / 2));
                    }
                    e.p0_center = (p0 >> 4) & 1;
                    e.p1_center = (p1 >> 4) & 1;
                    e.p0_corner = (int8_t)__builtin_popcount(p0 & corners);
                    e.p1_corner = (int8_t)__builtin_popcount(p1 & corners);
                    e.p0_sq = (int8_t)__builtin_popcount(p0);
                    e.p1_sq = (int8_t)__builtin_popcount(p1);
                    mini_lut[idx] = e;
                    int s = LUT_W_TIAR * (e.p0_tiar - e.p1_tiar)
                          + LUT_W_CENTER_SQ * (e.p0_center - e.p1_center)
                          + LUT_W_CORNER_SQ * (e.p0_corner - e.p1_corner)
                          + LUT_W_SQUARES * (e.p0_sq - e.p1_sq);
                    if (s > 32767) s = 32767;
                    if (s < -32768) s = -32768;
                    mini_score[idx] = (int16_t)s;
                }
                for (int p0 = 0; p0 < 512; p0++) {
                    for (int p1 = 0; p1 < 512; p1++) {
                        int packed = (p0 << 9) | p1;
                        if ((p0 & p1) == 0) {
                            int idx = mini_index(p0, p1);
                            fast_local_score[packed] = mini_score[idx];
                            fast_tiar_flags[packed] =
                                (mini_lut[idx].p0_tiar != 0)
                                | ((mini_lut[idx].p1_tiar != 0) << 1);
                        }
                        int threats = 0;
                        for (int i = 0; i < N_TIAR_MASKS / 2; i++) {
                            int pair = two_in_a_row_masks[i * 2];
                            int third = two_in_a_row_masks[i * 2 + 1];
                            threats += ((p0 & pair) == pair) && ((p1 & third) == 0);
                        }
                        fast_threat_count[packed] = (uint8_t)threats;
                    }
                }
            });
        }

        // LMR amounts in hundredths, so they can be retuned as integers.
        static constexpr int LMR_BASE = 55;
        static constexpr int LMR_DIV = 100;
        static constexpr int LMR_MAX_DEPTH = 64;
        static constexpr int LMR_MAX_MOVES = 81;
        static inline int lmr_table[LMR_MAX_DEPTH][LMR_MAX_MOVES];
        static inline std::once_flag lmr_table_once;

        // Precomputed so no logs run in the move loop. Row 0 stays zero (log(0)).
        static void init_lmr_table() {
            std::call_once(lmr_table_once, []() {
                for (int d = 1; d < LMR_MAX_DEPTH; d++) {
                    for (int i = 0; i < LMR_MAX_MOVES; i++) {
                        // Both terms are in hundredths of a ply: LMR_BASE directly, and
                        // 10000*ln(d)*ln(i+1)/LMR_DIV for the growth term.
                        int r = (int)((LMR_BASE + 10000.0 * std::log((double)d)
                                       * std::log((double)(i + 1)) / LMR_DIV) / 100.0);
                        lmr_table[d][i] = std::max(0, r);
                    }
                }
            });
        }

        bool time_up() {
            if (stopped) return true;
            if ((nodes & 255) == 0) {
                if (std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::high_resolution_clock::now() - start_time) > thinking_time) {
                    stopped = true;
                }
            }
            return stopped;
        }

        // Live entry for this position; the constraint test mirrors fillLegalMoves.
        int &corr_entry(FastBoard &board) {
            int out_of_play = board.mini_board_states[0] | board.mini_board_states[1] | board.mini_board_states[2];
            int mb = 9;
            if (board.n_moves > 0 && !board.prev_move_was_pass) {
                int active = board.move_history.top().square;
                if ((out_of_play & (1 << active)) == 0) mb = active;
            }
            return corr_hist[board.n_moves % 2][mb][out_of_play];
        }

        // Clamped clear of the decided-game band, leaving room for the MiniNet
        // residual that qsearch adds on top of this value.
        static constexpr int CORR_EVAL_LIMIT = CORR_MATE_BOUND - MINI_MAX - 1;
        int corrected_eval(FastBoard &board, int static_eval) {
            int v = static_eval + corr_entry(board) / CORR_GRAIN;
            if (v > CORR_EVAL_LIMIT) v = CORR_EVAL_LIMIT;
            if (v < -CORR_EVAL_LIMIT) v = -CORR_EVAL_LIMIT;
            return v;
        }

        void update_corr_hist(FastBoard &board, int diff, int d) {
            if (diff > CORR_DIFF_MAX) diff = CORR_DIFF_MAX;
            if (diff < -CORR_DIFF_MAX) diff = -CORR_DIFF_MAX;
            int w = std::min(d, CORR_MAX_WEIGHT);
            int &e = corr_entry(board);
            e += diff * w - e * abs(diff) * w / CORR_SCALE;
            if (e > CORR_MAX) e = CORR_MAX;
            if (e < -CORR_MAX) e = -CORR_MAX;
        }

        template <typename Board>
        int check_winner_fast(Board &board) {
            int p0 = board.mini_board_states[0];
            int p1 = board.mini_board_states[1];
            if (fast_has_win[p0]) return 0;
            if (fast_has_win[p1]) return 1;
            if ((p0 | p1 | board.mini_board_states[2]) == 511) {
                int n0 = __builtin_popcount(p0);
                int n1 = __builtin_popcount(p1);
                return n0 > n1 ? 0 : (n1 > n0 ? 1 : 2);
            }
            return -1;
        }

        template <typename Board>
        int fill_legal_moves_fast(Board &board, Move *dst) {
            int n = 0;
            if (board.n_moves == 0) {
                for (int mb = 0; mb < 9; mb++) {
                    for (int sq = 0; sq < 9; sq++) {
                        dst[n++] = Move{mb, sq};
                    }
                }
                return n;
            }
            int active = board.move_history.top().square;
            int out_of_play = board.mini_board_states[0]
                            | board.mini_board_states[1]
                            | board.mini_board_states[2];
            auto add_from_mb = [&](int mb) {
                int occupied = board.mini_boards[mb].markers[0]
                             | board.mini_boards[mb].markers[1];
                int empty = (~occupied) & 511;
                while (empty) {
                    int sq = __builtin_ctz(empty);
                    empty &= empty - 1;
                    dst[n++] = Move{mb, sq};
                }
            };
            if (!board.prev_move_was_pass && (out_of_play & (1 << active)) == 0) {
                add_from_mb(active);
            } else {
                int live = (~out_of_play) & 511;
                while (live) {
                    int mb = __builtin_ctz(live);
                    live &= live - 1;
                    add_from_mb(mb);
                }
            }
            return n;
        }

        template <typename Board>
        void set_hce_mb(Board &board, int mb) {
            int bit = 1 << mb;
            hce_local_score -= hce_mb_scores[mb];
            hce_tiar_maps[0] &= ~bit;
            hce_tiar_maps[1] &= ~bit;

            int score = 0;
            int flags = 0;
            int out_of_play = board.mini_board_states[0]
                            | board.mini_board_states[1]
                            | board.mini_board_states[2];
            if ((out_of_play & bit) == 0) {
                int packed = (board.mini_boards[mb].markers[0] << 9)
                           | board.mini_boards[mb].markers[1];
                score = fast_local_score[packed];
                flags = fast_tiar_flags[packed];
            }
            hce_mb_scores[mb] = (int16_t)score;
            hce_mb_flags[mb] = (uint8_t)flags;
            hce_local_score += score;
            hce_tiar_maps[0] |= (flags & 1) << mb;
            hce_tiar_maps[1] |= ((flags >> 1) & 1) << mb;
        }

        template <typename Board>
        void init_hce_acc(Board &board) {
            hce_local_score = 0;
            hce_tiar_maps = {};
            hce_mb_scores = {};
            hce_mb_flags = {};
            for (int mb = 0; mb < 9; mb++) {
                set_hce_mb(board, mb);
            }
            hce_acc_ready = true;
        }

        void restore_hce_mb(int ply, int mb) {
            int bit = 1 << mb;
            hce_local_score -= hce_mb_scores[mb];
            hce_tiar_maps[0] &= ~bit;
            hce_tiar_maps[1] &= ~bit;
            HceUndo old = hce_undo[ply];
            hce_mb_scores[mb] = old.score;
            hce_mb_flags[mb] = old.flags;
            hce_local_score += old.score;
            hce_tiar_maps[0] |= (old.flags & 1) << mb;
            hce_tiar_maps[1] |= ((old.flags >> 1) & 1) << mb;
        }

        template <typename Board>
        void make_move_fast(Board &board, const Move &move) {
            int stm = board.n_moves & 1;
            int bit = 1 << move.square;
            int mb_bit = 1 << move.mini_board;
            int before = board.mini_boards[move.mini_board].markers[stm];
            if (hce_acc_ready) {
                hce_undo[board.n_moves] = {
                    hce_mb_scores[move.mini_board],
                    hce_mb_flags[move.mini_board]
                };
            }
            if (board.n_moves > 0) {
                board.zobrist_hash ^= board.legal_mini_board_hashes[board.move_history.top().square];
                xor_tt_hash(board, board.legal_mini_board_hashes[board.move_history.top().square]);
            }
            board.move_history.push(move);
            board.mini_boards[move.mini_board].markers[stm] = before | bit;
            board.zobrist_hash ^= board.move_hashes[stm][move.mini_board][move.square];
            xor_tt_hash(board, board.move_hashes[stm][move.mini_board][move.square]);
            board.zobrist_hash ^= board.legal_mini_board_hashes[move.square];
            xor_tt_hash(board, board.legal_mini_board_hashes[move.square]);
            bool decided = false;
            if (fast_win_moves[before] & bit) {
                board.mini_board_states[stm] |= mb_bit;
                board.zobrist_hash ^= board.mini_board_hashes[stm][move.mini_board];
                xor_tt_hash(board, board.mini_board_hashes[stm][move.mini_board]);
                decided = true;
            } else {
                int occupied = board.mini_boards[move.mini_board].markers[0]
                             | board.mini_boards[move.mini_board].markers[1];
                if (occupied == 511) {
                    board.mini_board_states[2] |= mb_bit;
                    board.zobrist_hash ^= board.mini_board_hashes[2][move.mini_board];
                    xor_tt_hash(board, board.mini_board_hashes[2][move.mini_board]);
                    decided = true;
                }
            }
            if (decided) {
                xor_marker_hashes(board, move.mini_board);
            }
            board.zobrist_hash ^= board.player_to_move_hash;
            xor_tt_hash(board, board.player_to_move_hash);
            board.n_moves++;
            if (hce_acc_ready) {
                set_hce_mb(board, move.mini_board);
            }
        }

        template <typename Board>
        void unmake_move_fast(Board &board) {
            board.n_moves--;
            board.zobrist_hash ^= board.player_to_move_hash;
            xor_tt_hash(board, board.player_to_move_hash);
            Move move = board.move_history.top();
            board.move_history.pop();
            int mb_bit = 1 << move.mini_board;
            bool was_decided = false;
            if (board.mini_board_states[0] & mb_bit) {
                board.mini_board_states[0] &= ~mb_bit;
                board.zobrist_hash ^= board.mini_board_hashes[0][move.mini_board];
                xor_tt_hash(board, board.mini_board_hashes[0][move.mini_board]);
                was_decided = true;
            } else if (board.mini_board_states[1] & mb_bit) {
                board.mini_board_states[1] &= ~mb_bit;
                board.zobrist_hash ^= board.mini_board_hashes[1][move.mini_board];
                xor_tt_hash(board, board.mini_board_hashes[1][move.mini_board]);
                was_decided = true;
            } else if (board.mini_board_states[2] & mb_bit) {
                board.mini_board_states[2] &= ~mb_bit;
                board.zobrist_hash ^= board.mini_board_hashes[2][move.mini_board];
                xor_tt_hash(board, board.mini_board_hashes[2][move.mini_board]);
                was_decided = true;
            }
            if (was_decided) {
                xor_marker_hashes(board, move.mini_board);
            }
            int stm = board.n_moves & 1;
            board.mini_boards[move.mini_board].markers[stm] &= ~(1 << move.square);
            board.zobrist_hash ^= board.move_hashes[stm][move.mini_board][move.square];
            xor_tt_hash(board, board.move_hashes[stm][move.mini_board][move.square]);
            board.zobrist_hash ^= board.legal_mini_board_hashes[move.square];
            xor_tt_hash(board, board.legal_mini_board_hashes[move.square]);
            if (board.n_moves > 0) {
                board.zobrist_hash ^= board.legal_mini_board_hashes[board.move_history.top().square];
                xor_tt_hash(board, board.legal_mini_board_hashes[board.move_history.top().square]);
            }
            if (hce_acc_ready) {
                restore_hce_mb(board.n_moves, move.mini_board);
            }
        }

        Move getMove(GlobalBoard input_board, std::chrono::milliseconds thinking_time_passed = std::chrono::milliseconds(95)) {
            FastBoard board(input_board);
            init_mini_lut();
            init_lmr_table();
            thinking_time = thinking_time_passed;
            nodes = 0;
            stopped = false;
            root_score = 0;
            Move root_moves[81];
            fill_legal_moves_fast(board, root_moves);
            root_best_move = root_moves[0];
            init_hce_acc(board);
            killer_moves = std::array<std::array<int, 9>, 128>();
            for (auto &by_player : history_table) {
                for (auto &by_miniboard : by_player) {
                    for (int &h : by_miniboard) {
                        h /= 2;
                    }
                }
            }
            if (!counters_ready) {
                for (int i = 0; i < 9; i++) {
                    for (int j = 0; j < 9; j++) {
                        counter_move[i][j] = Move{99, 99};
                    }
                }
                counters_ready = true;
            }
            start_time = std::chrono::high_resolution_clock::now();
            if (g_fixed_search_depth > 0) {
                thinking_time = std::chrono::milliseconds(24 * 60 * 60 * 1000);
                depth = g_fixed_search_depth;
                search(board, g_fixed_search_depth, 0, min_val, max_val);
                return root_best_move;
            }
            depth = 1;
            int alpha = min_val;
            int beta = max_val;
            int aspiration_window = ASP_PAWNS * eval_weights[PAWN_IDX];
            while (!time_up() && (depth < 50)) {
                int eval = search(board, depth, 0, alpha, beta);
                if (stopped) break;
                if (eval <= alpha ) {
                    aspiration_window *= 3;
                    alpha -= aspiration_window;
                }
                else if (eval >= beta) {
                    aspiration_window *= 3;
                    beta += aspiration_window;
                }
                else {
                    aspiration_window = ASP_PAWNS * eval_weights[PAWN_IDX];
                    alpha = eval - aspiration_window;
                    beta = eval + aspiration_window;
                    depth++;
                }
            }
            return root_best_move;
        }

        // One full-window search at `d`. No aspiration, no time cutoff.
        // Returns false on timeout/unfinished sentinel. Mates clamp to ±20000.
        static constexpr int SEARCH_SCORE_CLAMP = 20000;
        bool search_fixed_depth(GlobalBoard &input_board, int d, int &out_score) {
            FastBoard board(input_board);
            init_mini_lut();
            init_lmr_table();
            thinking_time = std::chrono::milliseconds(24 * 60 * 60 * 1000);
            nodes = 0;
            stopped = false;
            root_score = 0;
            depth = d;
            init_hce_acc(board);
            killer_moves = std::array<std::array<int, 9>, 128>();
            corr_hist = {};
            start_time = std::chrono::high_resolution_clock::now();
            int eval = search(board, d, 0, min_val, max_val);
            if (stopped || eval == min_val) return false;
            if (eval > SEARCH_SCORE_CLAMP) eval = SEARCH_SCORE_CLAMP;
            if (eval < -SEARCH_SCORE_CLAMP) eval = -SEARCH_SCORE_CLAMP;
            out_score = eval;
            return true;
        }

        int qsearch(FastBoard &board, int alpha, int beta, int ply) {
            if (time_up()) return min_val;
            nodes++;

            int winner = check_winner_fast(board);
            if (winner != -1){
                if (winner == 2) {
                    return 0;
                }
                else {
                    if (winner == board.n_moves % 2) {
                        return max_val - ply;
                    }
                    else {
                        return min_val + ply;
                    }
                }
            }

            int hce = corrected_eval(board, evaluate_hce_incremental(board));
            if (hce >= beta) {
                return beta;
            }
            int stand_pat;
            if (hce + MINI_MAX < alpha) {
                // MiniNet is clamped near ±2000; it cannot raise alpha or fail high.
                stand_pat = hce + MINI_MAX;
            } else {
                stand_pat = hce + evaluate_mini(board);
            }
            if (stand_pat >= beta) {
                return beta;
            }
            if (alpha < stand_pat) {
                alpha = stand_pat;
            }
            if (!g_disable_eval_prune && stand_pat + QDELTA_PAWNS * eval_weights[PAWN_IDX] < alpha) {
                return alpha;
            }

            FastMove caps[81];
            int scores[81];
            int n_caps = fill_fast_captures(board, caps);
            get_fast_move_scores(caps, n_caps, board, ply, scores, true);
            sort_fast_moves(caps, scores, n_caps);
            int val;
            for (int i = 0; i < n_caps; i++) {
                Move move = unpack_fast_move(caps[i]);
                make_move_fast(board, move);
                val = -qsearch(board, -beta, -alpha, ply + 1);
                unmake_move_fast(board);
                if (stopped) return min_val;
                alpha = std::max(alpha, val);
                if (alpha >= beta) {
                    break;
                }
            }
            return alpha;
        }

        int search(FastBoard &board, int depth, int ply, int alpha, int beta) {
            if (time_up()) return min_val;
            nodes++;
            int winner = check_winner_fast(board);
            if (winner != -1){
                if (winner == 2) {
                    return 0;
                }
                else {
                    if (winner == board.n_moves % 2) {
                        return max_val - ply;
                    }
                    else {
                        return min_val + ply;
                    }
                }
            }
            bool pv_node = (beta - alpha > 1);
            CompactTTBucket &tt_bucket =
                transposition_table[board.tt_hash & (tt_bucket_count - 1)];
            CompactTTEntry entry = tt_bucket.entries[0];
            if (entry.zobrist_hash != board.tt_hash) {
                entry = tt_bucket.entries[1];
            }
            bool tt_hit = (entry.zobrist_hash == board.tt_hash)
                       && (board.tt_hash != 0);
            FastMove tt_move = tt_hit ? tt_to_fast_move(entry.best_move) : NO_FAST_MOVE;
            if (tt_hit && (entry.depth >= depth)) {
                // Flags match the original store: 0 exact, 1 upper (fail low), 2 lower (fail high).
                if (entry.flag == TT_EXACT && (!pv_node || ply > 0)) {
                    return entry.score;
                }
                else if (!pv_node && entry.flag == TT_LOWER) {
                    if (entry.score >= beta) return entry.score;
                }
                else if (!pv_node && entry.flag == TT_UPPER) {
                    if (entry.score <= alpha) return entry.score;
                }
            }

            if (depth <= 0) {
                return qsearch(board, alpha, beta, ply);
            }
            bool can_futility_prune = false;
            int static_eval = 0;
            bool have_static = false;
            if (!pv_node && !g_disable_eval_prune) {
                static_eval = corrected_eval(board, evaluate_hce_incremental(board));
                have_static = true;

                int reverse_futility_margin = RFP_PAWNS * eval_weights[PAWN_IDX];
                if (static_eval - reverse_futility_margin * depth >= beta) {
                    return beta;
                }

                int futility_margin = FP_PAWNS * eval_weights[PAWN_IDX];
                can_futility_prune = (static_eval + futility_margin * depth <= alpha);
            }
            if (pv_node && !tt_hit && depth > 2) {
                search(board, 1, ply, alpha, beta);
                if (stopped) return min_val;
                CompactTTBucket &iid_bucket =
                    transposition_table[board.tt_hash & (tt_bucket_count - 1)];
                entry = iid_bucket.entries[0];
                if (entry.zobrist_hash != board.tt_hash) {
                    entry = iid_bucket.entries[1];
                }
                tt_hit = (entry.zobrist_hash == board.tt_hash) && (board.tt_hash != 0);
                tt_move = tt_hit ? tt_to_fast_move(entry.best_move) : NO_FAST_MOVE;
            }

            bool singular =
                tt_hit && entry.depth >= depth - 3
                && (entry.flag == TT_LOWER || entry.flag == TT_EXACT);

            FastMove legal_moves[81];
            int scores[81];
            int nmoves = fill_fast_legal_moves(board, legal_moves);
            bool defer_move_scores = false;
            int tt_index = -1;
            for (int i = 0; i < nmoves; i++) {
                if (legal_moves[i] == tt_move) {
                    tt_index = i;
                    break;
                }
            }
            if (tt_index >= 0) {
                FastMove hash_move = legal_moves[tt_index];
                for (int i = tt_index; i > 0; i--) {
                    legal_moves[i] = legal_moves[i - 1];
                }
                legal_moves[0] = hash_move;
                scores[0] = 1000;
                defer_move_scores = true;
            } else {
                get_fast_move_scores(legal_moves, nmoves, board, ply, scores, false);
                sort_fast_moves(legal_moves, scores, nmoves);
            }

            FastMove best_move = legal_moves[0];
            int best_val = min_val;
            int alpha_orig = alpha;
            int val;
            int stm = board.n_moves & 1;
            int opponent_global_targets =
                fast_win_moves[board.mini_board_states[stm ^ 1]];
            for (int i = 0; i < nmoves; i++) {
                if (i == 1 && defer_move_scores) {
                    get_fast_move_scores(legal_moves + 1, nmoves - 1,
                                         board, ply, scores + 1, false);
                    sort_fast_moves(legal_moves + 1, scores + 1, nmoves - 1);
                }
                FastMove fast_move = legal_moves[i];
                Move move = unpack_fast_move(fast_move);
                bool capture = is_fast_capture(board, fast_move);
                if (can_futility_prune && i > 0 && !capture) {
                    continue;
                }
                int extension = 0;
                if (nmoves == 1 || (singular && fast_move == tt_move)) {
                    extension = 1;
                }

                make_move_fast(board, move);
                if (opponent_global_targets
                    && has_immediate_global_win(board, opponent_global_targets)) {
                    val = min_val + ply + 2;
                }
                else if (has_forced_global_win_after_reply(board, stm)) {
                    val = max_val - ply - 3;
                }
                else if (i == 0) {
                    __builtin_prefetch(
                        &transposition_table[
                            board.tt_hash & (tt_bucket_count - 1)], 0, 1);
                    val = -search(board, depth - 1 + extension, ply + 1, -beta, -alpha);
                }
                else {
                    __builtin_prefetch(
                        &transposition_table[
                            board.tt_hash & (tt_bucket_count - 1)], 0, 1);
                    int reduction = 0;
                    bool do_lmr = (scores[i] < 0 || (i >= 2 && !capture));
                    if (do_lmr) {
                        reduction = lmr_table[std::min(depth, LMR_MAX_DEPTH - 1)][std::min(i, LMR_MAX_MOVES - 1)];
                        if (pv_node && reduction > 0) reduction--;
                    }
                    if (reduction > depth - 1) reduction = std::max(0, depth - 1);
                    val = -search(board, depth - 1 - reduction + extension, ply + 1, -alpha - 1, -alpha);
                    // Reduced searches are not allowed to fail high unchallenged.
                    if (val > alpha) {
                        val = -search(board, depth - 1 + extension, ply + 1, -alpha - 1, -alpha);
                        if (val > alpha && val < beta) {
                            val = -search(board, depth - 1 + extension, ply + 1, -beta, -alpha);
                        }
                    }
                }
                unmake_move_fast(board);
                if (stopped) return min_val;
                if (val > best_val) {
                    best_val = val;
                    best_move = fast_move;
                    if (ply == 0 && abs(best_val) != abs(min_val)) {
                        root_best_move = move;
                        root_score = best_val;
                    }
                }
                alpha = std::max(alpha, best_val);
                if (alpha >= beta) {
                    int mb = fast_move >> 4;
                    int sq = fast_move & 15;
                    killer_moves[ply][sq] = 1;
                    int &h = history_table[board.n_moves % 2][mb][sq];
                    int bonus = depth * depth;
                    h += bonus - h * bonus / 10000;
                    int stm = board.n_moves % 2;
                    for (int j = 0; j < i; j++) {
                        FastMove prior = legal_moves[j];
                        if (is_fast_capture(board, prior)) continue;
                        int &hj = history_table[stm][prior >> 4][prior & 15];
                        int malus = 2 * bonus;
                        hj -= malus + hj * malus / 10000;
                        if (hj < -10000) hj = -10000;
                    }
                    if (board.n_moves > 0) {
                        Move prev = board.move_history.top();
                        counter_move[prev.mini_board][prev.square] = move;
                    }
                    break;
                }
            }
            if (!stopped) {
                int flag = TT_EXACT;
                if (best_val <= alpha_orig) {
                    flag = TT_UPPER;
                }
                else if (best_val >= beta) {
                    flag = TT_LOWER;
                }
                CompactTTEntry new_entry = {
                    board.tt_hash,
                    best_val,
                    (int16_t)depth,
                    (int8_t)flag,
                    pack_tt_move(best_move)
                };
                CompactTTBucket &store_bucket =
                    transposition_table[board.tt_hash & (tt_bucket_count - 1)];
                int replace = 0;
                if (store_bucket.entries[0].zobrist_hash == board.tt_hash) {
                    replace = 0;
                }
                else if (store_bucket.entries[1].zobrist_hash == board.tt_hash) {
                    replace = 1;
                }
                else if (store_bucket.entries[0].zobrist_hash == 0) {
                    replace = 0;
                }
                else if (store_bucket.entries[1].zobrist_hash == 0) {
                    replace = 1;
                }
                else if (store_bucket.entries[1].depth
                         < store_bucket.entries[0].depth) {
                    replace = 1;
                }
                store_bucket.entries[replace] = new_entry;
                // Only a bound that actually contradicts the static eval carries information.
                if (have_static && abs(best_val) < CORR_MATE_BOUND
                    && (flag == TT_EXACT
                        || (flag == TT_LOWER && best_val > static_eval)
                        || (flag == TT_UPPER && best_val < static_eval))) {
                    update_corr_hist(board, best_val - static_eval, depth);
                }
            }

            return best_val;
        }

        void sort_fast_moves(FastMove* moves, int* scores, int n) {
            for (int i = 1; i < n; i++) {
                int key = scores[i];
                FastMove key_move = moves[i];
                int j = i - 1;
                while (j >= 0 && scores[j] < key) {
                    scores[j + 1] = scores[j];
                    moves[j + 1] = moves[j];
                    j = j - 1;
                }
                scores[j + 1] = key;
                moves[j + 1] = key_move;
            }
        }

        template <typename Board>
        int fill_fast_legal_moves(Board &board, FastMove *dst) {
            int n = 0;
            if (board.n_moves == 0) {
                for (int mb = 0; mb < 9; mb++) {
                    for (int sq = 0; sq < 9; sq++) {
                        dst[n++] = pack_fast_move(mb, sq);
                    }
                }
                return n;
            }
            int active = board.move_history.top().square;
            int out_of_play = board.mini_board_states[0]
                            | board.mini_board_states[1]
                            | board.mini_board_states[2];
            auto add_from_mb = [&](int mb) {
                int occupied = board.mini_boards[mb].markers[0]
                             | board.mini_boards[mb].markers[1];
                int empty = (~occupied) & 511;
                while (empty) {
                    int sq = __builtin_ctz(empty);
                    empty &= empty - 1;
                    dst[n++] = pack_fast_move(mb, sq);
                }
            };
            if (!board.prev_move_was_pass && (out_of_play & (1 << active)) == 0) {
                add_from_mb(active);
            } else {
                int live = (~out_of_play) & 511;
                while (live) {
                    int mb = __builtin_ctz(live);
                    live &= live - 1;
                    add_from_mb(mb);
                }
            }
            return n;
        }

        template <typename Board>
        int fill_fast_captures(Board &board, FastMove *dst) {
            int n = 0;
            if (board.n_moves == 0) return 0;
            int active_square = board.move_history.top().square;
            int out_of_play = board.mini_board_states[0]
                            | board.mini_board_states[1]
                            | board.mini_board_states[2];
            int stm = board.n_moves % 2;
            auto add_from_mb = [&](int mb) {
                int mine = board.mini_boards[mb].markers[stm];
                int occupied = board.mini_boards[mb].markers[0]
                             | board.mini_boards[mb].markers[1];
                int wins = fast_win_moves[mine] & ~occupied & 511;
                while (wins) {
                    int sq = __builtin_ctz(wins);
                    wins &= wins - 1;
                    dst[n++] = pack_fast_move(mb, sq);
                }
            };
            if ((out_of_play & (1 << active_square)) == 0) {
                add_from_mb(active_square);
            } else {
                int live = (~out_of_play) & 511;
                while (live) {
                    int mb = __builtin_ctz(live);
                    live &= live - 1;
                    add_from_mb(mb);
                }
            }
            return n;
        }

        template <typename Board>
        bool is_fast_capture(Board &board, FastMove move) {
            int stm = board.n_moves % 2;
            int mb = move >> 4;
            int sq = move & 15;
            int mine = board.mini_boards[mb].markers[stm];
            return (fast_win_moves[mine] & (1 << sq)) != 0;
        }

        template <typename Board>
        bool has_immediate_global_win(Board &board, int targets) {
            int stm = board.n_moves & 1;
            if (fast_has_win[board.mini_board_states[stm ^ 1]]) return false;
            int out_of_play = board.mini_board_states[0]
                            | board.mini_board_states[1]
                            | board.mini_board_states[2];
            int live = (~out_of_play) & 511;
            targets &= live;
            if (targets == 0) return false;
            if (board.n_moves > 0 && !board.prev_move_was_pass) {
                int active = board.move_history.top().square;
                if (live & (1 << active)) {
                    targets &= 1 << active;
                }
            }
            while (targets) {
                int mb = __builtin_ctz(targets);
                targets &= targets - 1;
                int occupied = board.mini_boards[mb].markers[0]
                             | board.mini_boards[mb].markers[1];
                if (fast_win_moves[board.mini_boards[mb].markers[stm]]
                    & ~occupied & 511) {
                    return true;
                }
            }
            return false;
        }

        template <typename Board>
        bool has_forced_global_win_after_reply(Board &board, int player) {
            if ((board.n_moves & 1) == player
                || fast_has_win[board.mini_board_states[player]]) {
                return false;
            }
            int targets = fast_win_moves[board.mini_board_states[player]];
            int out_of_play = board.mini_board_states[0]
                            | board.mini_board_states[1]
                            | board.mini_board_states[2];
            targets &= (~out_of_play) & 511;
            int winning_targets = 0;
            int remaining = targets;
            while (remaining) {
                int mb = __builtin_ctz(remaining);
                remaining &= remaining - 1;
                int occupied = board.mini_boards[mb].markers[0]
                             | board.mini_boards[mb].markers[1];
                if (fast_win_moves[board.mini_boards[mb].markers[player]]
                    & ~occupied & 511) {
                    winning_targets |= 1 << mb;
                }
            }
            if (winning_targets == 0) return false;

            int opponent = player ^ 1;
            bool any_reply = false;
            auto miniboard_refutes = [&](int mb) {
                int mb_bit = 1 << mb;
                int occupied = board.mini_boards[mb].markers[0]
                             | board.mini_boards[mb].markers[1];
                int empty = (~occupied) & 511;
                if (empty == 0) return false;
                any_reply = true;

                int captures =
                    fast_win_moves[
                        board.mini_boards[mb].markers[opponent]] & empty;
                int draws = 0;
                if ((empty & (empty - 1)) == 0) {
                    draws = empty & ~captures;
                }
                int decided = captures | draws;
                int safe = 0;

                int targets_after_decision = winning_targets & ~mb_bit;
                if (targets_after_decision
                    && (out_of_play | mb_bit) != 511) {
                    int safe_destinations =
                        out_of_play | targets_after_decision | mb_bit;
                    int safe_decided = decided & safe_destinations;
                    if (fast_has_win[
                            board.mini_board_states[opponent] | mb_bit]) {
                        safe_decided &= ~captures;
                    }
                    safe |= safe_decided;
                }

                int nondeciding = empty & ~decided;
                int safe_nondeciding =
                    nondeciding & (out_of_play | winning_targets);
                if (winning_targets & mb_bit) {
                    int player_wins =
                        fast_win_moves[
                            board.mini_boards[mb].markers[player]] & empty;
                    if (player_wins
                        && (player_wins & (player_wins - 1)) == 0) {
                        int blocked_reply = nondeciding & player_wins;
                        safe_nondeciding &= ~blocked_reply;
                        int remaining_targets = winning_targets & ~mb_bit;
                        if (remaining_targets) {
                            safe_nondeciding |=
                                blocked_reply
                                & (out_of_play | remaining_targets);
                        }
                    }
                }
                safe |= safe_nondeciding;
                return (empty & ~safe) != 0;
            };

            if (board.n_moves > 0 && !board.prev_move_was_pass) {
                int active = board.move_history.top().square;
                if ((out_of_play & (1 << active)) == 0) {
                    if (miniboard_refutes(active)) return false;
                    return any_reply;
                }
            }
            int live = (~out_of_play) & 511;
            while (live) {
                int mb = __builtin_ctz(live);
                live &= live - 1;
                if (miniboard_refutes(mb)) return false;
            }
            return any_reply;
        }

        template <typename Board>
        void get_fast_move_scores(FastMove* moves, int n, Board &board, int &ply,
                                  int* scores, bool qs = false) {
            if (n <= 1) {
                if (n == 1) scores[0] = 0;
                return;
            }
            int out_of_play = board.mini_board_states[0]
                            | board.mini_board_states[1]
                            | board.mini_board_states[2];
            int stm = board.n_moves % 2;
            FastMove cm = NO_FAST_MOVE;
            if (board.n_moves > 0) {
                Move prev = board.move_history.top();
                Move counter = counter_move[prev.mini_board][prev.square];
                if (counter.mini_board < 9 && counter.square < 9) {
                    cm = pack_fast_move(counter.mini_board, counter.square);
                }
            }
            int last_mb = -1;
            int last_idx = 0;
            int capture_mask = 0;
            int block_mask = 0;
            int tiar_mask = 0;
            int global_win_bonus = 0;
            for (int i = 0; i < n; i++) {
                FastMove move = moves[i];
                int mb = move >> 4;
                int sq = move & 15;
                if (mb != last_mb) {
                    last_mb = mb;
                    last_idx = mini_index(board.mini_boards[mb].markers[0],
                                          board.mini_boards[mb].markers[1]);
                    capture_mask = fast_win_moves[board.mini_boards[mb].markers[stm]];
                    block_mask = fast_win_moves[board.mini_boards[mb].markers[stm ^ 1]];
                    tiar_mask = mini_tiar_sq[last_idx][stm];
                    global_win_bonus =
                        800 * fast_has_win[board.mini_board_states[stm] | (1 << mb)];
                }
                int capture = (capture_mask >> sq) & 1;
                scores[i] =
                    25 * killer_moves[ply][sq]
                    + 40 * (cm == move)
                    + capture * (global_win_bonus + 100 * !qs)
                    + 75 * ((block_mask >> sq) & 1)
                    + 50 * ((tiar_mask >> sq) & 1)
                    - 250 * ((out_of_play >> sq) & 1)
                    + history_table[stm][mb][sq] / 20;
            }
        }

        template <typename Board>
        int fill_captures_lut(Board &board, Move* dst) {
            int n = 0;
            if (board.n_moves == 0) return 0;
            int active_square = board.move_history.top().square;
            int out_of_play = board.mini_board_states[0] | board.mini_board_states[1] | board.mini_board_states[2];
            int stm = board.n_moves % 2;
            auto add_from_mb = [&](int mb) {
                int mine = board.mini_boards[mb].markers[stm];
                int occupied = board.mini_boards[mb].markers[0]
                             | board.mini_boards[mb].markers[1];
                int wins = fast_win_moves[mine] & ~occupied & 511;
                while (wins) {
                    int s = __builtin_ctz(wins);
                    wins &= wins - 1;
                    dst[n++] = Move{mb, s};
                }
            };
            if ((out_of_play & (1 << active_square)) == 0) {
                add_from_mb(active_square);
            } else {
                for (int i = 0; i < 9; i++) {
                    if ((out_of_play & (1 << i)) == 0) add_from_mb(i);
                }
            }
            return n;
        }

        template <typename Board>
        bool is_capture_avx(Board &board, Move &move) {
            int stm = board.n_moves % 2;
            int mine = board.mini_boards[move.mini_board].markers[stm];
            return (fast_win_moves[mine] & (1 << move.square)) != 0;
        }

        template <typename Board>
        bool is_block_avx(Board &board, Move &move) {
            int opp = (board.n_moves + 1) % 2;
            int theirs = board.mini_boards[move.mini_board].markers[opp];
            return (fast_win_moves[theirs] & (1 << move.square)) != 0;
        }

        template <typename Board>
        bool creates_two_in_a_row(Board &board, Move &move) {
            int idx = mini_index(
                board.mini_boards[move.mini_board].markers[0],
                board.mini_boards[move.mini_board].markers[1]);
            int stm = board.n_moves % 2;
            return (mini_tiar_sq[idx][stm] & (1 << move.square)) != 0;
        }

        template <typename Board>
        void get_move_scores(Move* moves, int n, Board &board, int &ply,
                             int* scores, bool qs = false) {
            if (n <= 1) {
                if (n == 1) scores[0] = 0;
                return;
            }
            int out_of_play = board.mini_board_states[0] | board.mini_board_states[1] | board.mini_board_states[2];
            int stm = board.n_moves % 2;
            Move cm{99, 99};
            if (board.n_moves > 0) {
                Move prev = board.move_history.top();
                cm = counter_move[prev.mini_board][prev.square];
            }
            int last_mb = -1;
            int last_idx = 0;
            int capture_mask = 0;
            int block_mask = 0;
            int tiar_mask = 0;
            int global_win_bonus = 0;
            for (int i = 0; i < n; i++) {
                int mb = moves[i].mini_board;
                int sq = moves[i].square;
                if (mb != last_mb) {
                    last_mb = mb;
                    last_idx = mini_index(board.mini_boards[mb].markers[0], board.mini_boards[mb].markers[1]);
                    capture_mask = fast_win_moves[board.mini_boards[mb].markers[stm]];
                    block_mask = fast_win_moves[board.mini_boards[mb].markers[stm ^ 1]];
                    tiar_mask = mini_tiar_sq[last_idx][stm];
                    global_win_bonus =
                        800 * fast_has_win[board.mini_board_states[stm] | (1 << mb)];
                }
                int capture = (capture_mask >> sq) & 1;
                int move_score =
                    25 * killer_moves[ply][sq]
                    + 40 * (cm.mini_board == mb && cm.square == sq)
                    + capture * (global_win_bonus + 100 * !qs)
                    + 75 * ((block_mask >> sq) & 1)
                    + 50 * ((tiar_mask >> sq) & 1)
                    - 250 * ((out_of_play >> sq) & 1)
                    + history_table[stm][mb][sq] / 20;
                scores[i] = move_score;
            }
        }

        static constexpr int N_EVAL_WEIGHTS = 10;
        int eval_weights[N_EVAL_WEIGHTS] = {2410, 836, 464, 1316, 534, 424, 33, PAWN, 33, 112};

        void eval_diffs(GlobalBoard &board, int *d) {
            init_mini_lut();
            int p0_miniboards_held = __builtin_popcount(board.mini_board_states[0]);
            int p1_miniboards_held = __builtin_popcount(board.mini_board_states[1]);
            int out_of_play = board.mini_board_states[0] | board.mini_board_states[1] | board.mini_board_states[2];
            int p0_two_in_a_row = 0;
            int p1_two_in_a_row = 0;
            int p0_center_squares_held = 0;
            int p1_center_squares_held = 0;
            int p0_corner_squares_held = 0;
            int p1_corner_squares_held = 0;
            int p0_squares_held = 0;
            int p1_squares_held = 0;
            int p0_two_in_a_row_map = 0;
            int p1_two_in_a_row_map = 0;
            int corners = (1 << 0) + (1 << 2) + (1 << 6) + (1 << 8);

            for (int miniboard = 0; miniboard < 9; miniboard++) {
                if ((out_of_play & (1 << miniboard)) != 0) {
                    continue;
                }
                const MiniLut &e = mini_lut[mini_index(
                    board.mini_boards[miniboard].markers[0],
                    board.mini_boards[miniboard].markers[1])];
                p0_two_in_a_row += e.p0_tiar;
                p1_two_in_a_row += e.p1_tiar;
                p0_two_in_a_row_map |= ((1 << miniboard) * (e.p0_tiar != 0));
                p1_two_in_a_row_map |= ((1 << miniboard) * (e.p1_tiar != 0));
                p0_center_squares_held += e.p0_center;
                p1_center_squares_held += e.p1_center;
                p0_corner_squares_held += e.p0_corner;
                p1_corner_squares_held += e.p1_corner;
                p0_squares_held += e.p0_sq;
                p1_squares_held += e.p1_sq;
            }

            int p0_miniboards = board.mini_board_states[0];
            int p1_miniboards = board.mini_board_states[1];
            int p0_center_miniboard_held = __builtin_popcount(p0_miniboards & (1 << 4));
            int p1_center_miniboard_held = __builtin_popcount(p1_miniboards & (1 << 4));
            int p0_corner_miniboards_held = __builtin_popcount(p0_miniboards & corners);
            int p1_corner_miniboards_held = __builtin_popcount(p1_miniboards & corners);
            int p0_global_two_in_a_row = 0;
            int p1_global_two_in_a_row = 0;
            int p0_two_in_a_rows_lined_up = 0;
            int p1_two_in_a_rows_lined_up = 0;
            for(int i = 0; i < N_TIAR_MASKS / 2; i++) {
                p0_global_two_in_a_row += ((__builtin_popcount(p0_miniboards & two_in_a_row_masks[i * 2]) - __builtin_popcount(p1_miniboards & two_in_a_row_masks[i * 2 + 1])) /2);
                p1_global_two_in_a_row += ((__builtin_popcount(p1_miniboards & two_in_a_row_masks[i * 2]) - __builtin_popcount(p0_miniboards & two_in_a_row_masks[i * 2 + 1])) /2);
                p0_two_in_a_rows_lined_up += ((__builtin_popcount((p0_two_in_a_row_map | p0_miniboards) & two_in_a_row_masks[i * 2]) - __builtin_popcount(p1_miniboards & two_in_a_row_masks[i * 2 + 1]))  / 2);
                p1_two_in_a_rows_lined_up += ((__builtin_popcount((p1_two_in_a_row_map | p1_miniboards) & two_in_a_row_masks[i * 2]) - __builtin_popcount(p0_miniboards & two_in_a_row_masks[i * 2 + 1]))   / 2);
            }
            d[0] = p0_miniboards_held - p1_miniboards_held;
            d[1] = p0_center_miniboard_held - p1_center_miniboard_held;
            d[2] = p0_corner_miniboards_held - p1_corner_miniboards_held;
            d[3] = p0_global_two_in_a_row - p1_global_two_in_a_row;
            d[4] = p0_two_in_a_row - p1_two_in_a_row;
            d[5] = p0_two_in_a_rows_lined_up - p1_two_in_a_rows_lined_up;
            d[6] = p0_center_squares_held - p1_center_squares_held;
            d[7] = p0_corner_squares_held - p1_corner_squares_held;
            d[8] = p0_squares_held - p1_squares_held;
            d[9] = 0;
        }

        // Frozen global terms + tempo, in evaluate() units: stm*global + tempo.
        // Live miniboard LUT indices are written to idx_out.
        void eval_parts(GlobalBoard &board, int16_t *idx_out, int &n_out, int &base_out) {
            init_mini_lut();
            int stm_sign = (board.n_moves % 2 == 0) ? 1 : -1;
            int out_of_play = board.mini_board_states[0] | board.mini_board_states[1] | board.mini_board_states[2];
            int p0_two_in_a_row_map = 0;
            int p1_two_in_a_row_map = 0;
            int corners = (1 << 0) + (1 << 2) + (1 << 6) + (1 << 8);
            n_out = 0;
            for (int miniboard = 0; miniboard < 9; miniboard++) {
                if ((out_of_play & (1 << miniboard)) != 0) {
                    continue;
                }
                int idx = mini_index(
                    board.mini_boards[miniboard].markers[0],
                    board.mini_boards[miniboard].markers[1]);
                idx_out[n_out++] = (int16_t)idx;
                const MiniLut &e = mini_lut[idx];
                p0_two_in_a_row_map |= ((1 << miniboard) * (e.p0_tiar != 0));
                p1_two_in_a_row_map |= ((1 << miniboard) * (e.p1_tiar != 0));
            }

            int p0_miniboards = board.mini_board_states[0];
            int p1_miniboards = board.mini_board_states[1];
            int p0_global_two_in_a_row = 0;
            int p1_global_two_in_a_row = 0;
            int p0_two_in_a_rows_lined_up = 0;
            int p1_two_in_a_rows_lined_up = 0;
            for(int i = 0; i < N_TIAR_MASKS / 2; i++) {
                int third = two_in_a_row_masks[i * 2 + 1];
                p0_global_two_in_a_row += ((__builtin_popcount(p0_miniboards & two_in_a_row_masks[i * 2]) - __builtin_popcount(p1_miniboards & third)) /2);
                p1_global_two_in_a_row += ((__builtin_popcount(p1_miniboards & two_in_a_row_masks[i * 2]) - __builtin_popcount(p0_miniboards & third)) /2);
                p0_two_in_a_rows_lined_up += ((__builtin_popcount((p0_two_in_a_row_map | p0_miniboards) & two_in_a_row_masks[i * 2]) - __builtin_popcount(p1_miniboards & third))  / 2);
                p1_two_in_a_rows_lined_up += ((__builtin_popcount((p1_two_in_a_row_map | p1_miniboards) & two_in_a_row_masks[i * 2]) - __builtin_popcount(p0_miniboards & third))   / 2);
            }
            int g = eval_weights[0] * (__builtin_popcount(p0_miniboards) - __builtin_popcount(p1_miniboards));
            g += eval_weights[1] * (__builtin_popcount(p0_miniboards & (1 << 4)) - __builtin_popcount(p1_miniboards & (1 << 4)));
            g += eval_weights[2] * (__builtin_popcount(p0_miniboards & corners) - __builtin_popcount(p1_miniboards & corners));
            g += eval_weights[3] * (p0_global_two_in_a_row - p1_global_two_in_a_row);
            g += eval_weights[5] * (p0_two_in_a_rows_lined_up - p1_two_in_a_rows_lined_up);
            base_out = stm_sign * g + eval_weights[9];
        }

        // STM-centric bonus on top of the linear/LUT eval.
        template <typename Board>
        int eval_extra(Board &board) {
            if (board.n_moves > 0) {
                int out_of_play = board.mini_board_states[0] | board.mini_board_states[1] | board.mini_board_states[2];
                if (board.prev_move_was_pass || ((out_of_play & (1 << board.move_history.top().square)) != 0)) {
                    return FREE_MOVE_PAWNS * eval_weights[PAWN_IDX];
                }
            }
            return 0;
        }

        template <typename Board>
        int finish_hce(Board &board, int local,
                       int p0_two_in_a_row_map,
                       int p1_two_in_a_row_map) {
            int stm_sign = (board.n_moves % 2 == 0) ? 1 : -1;
            int p0_miniboards = board.mini_board_states[0];
            int p1_miniboards = board.mini_board_states[1];
            const int corners = (1 << 0) | (1 << 2) | (1 << 6) | (1 << 8);
            int global = eval_weights[0]
                * (__builtin_popcount(p0_miniboards) - __builtin_popcount(p1_miniboards));
            global += eval_weights[1]
                * (((p0_miniboards >> 4) & 1) - ((p1_miniboards >> 4) & 1));
            global += eval_weights[2]
                * (__builtin_popcount(p0_miniboards & corners)
                   - __builtin_popcount(p1_miniboards & corners));
            global += eval_weights[3]
                * ((int)fast_threat_count[(p0_miniboards << 9) | p1_miniboards]
                   - (int)fast_threat_count[(p1_miniboards << 9) | p0_miniboards]);
            global += eval_weights[5]
                * ((int)fast_threat_count[
                       ((p0_miniboards | p0_two_in_a_row_map) << 9) | p1_miniboards]
                   - (int)fast_threat_count[
                       ((p1_miniboards | p1_two_in_a_row_map) << 9) | p0_miniboards]);
            return stm_sign * (global + local) + eval_weights[9] + eval_extra(board);
        }

        template <typename Board>
        int evaluate_hce(Board &board) {
            int out_of_play = board.mini_board_states[0]
                            | board.mini_board_states[1]
                            | board.mini_board_states[2];
            int live = (~out_of_play) & 511;
            int local = 0;
            int p0_two_in_a_row_map = 0;
            int p1_two_in_a_row_map = 0;
            while (live) {
                int miniboard = __builtin_ctz(live);
                live &= live - 1;
                int packed = (board.mini_boards[miniboard].markers[0] << 9)
                           | board.mini_boards[miniboard].markers[1];
                local += fast_local_score[packed];
                int flags = fast_tiar_flags[packed];
                p0_two_in_a_row_map |= (flags & 1) << miniboard;
                p1_two_in_a_row_map |= ((flags >> 1) & 1) << miniboard;
            }
            return finish_hce(board, local, p0_two_in_a_row_map,
                              p1_two_in_a_row_map);
        }

        template <typename Board>
        int evaluate_hce_incremental(Board &board) {
            if (!hce_acc_ready) {
                return evaluate_hce(board);
            }
            return finish_hce(board, hce_local_score, hce_tiar_maps[0],
                              hce_tiar_maps[1]);
        }

        int evaluate(GlobalBoard &board) {
            if (g_force_hce_eval) {
                return evaluate_hce(board);
            }
            return evaluate_hce(board) + evaluate_mini(board);
        }

};

Move grid_coord_to_move(int row, int col) {
    int8_t mini_board = (row / 3) * 3 + (col / 3);
    int8_t square = (row % 3) * 3 + (col % 3);
    Move move = {mini_board, square};
    return move;
}

std::array<int, 2> move_to_grid_coord(Move move) {
    int8_t row = (move.mini_board / 3) * 3 + (move.square / 3);
    int8_t col = (move.mini_board % 3) * 3 + (move.square % 3);
    std::array<int, 2> grid_coord = {row, col};
    return grid_coord;
}



// Function to aggregate results from multiple runs
std::array<int, 3> aggregate_results(std::vector<std::future<std::array<int, 3>>>& futures) {
    std::array<int, 3> total = {0, 0, 0};
    for (auto& future : futures) {
        const auto result = future.get();
        total[0] += result[0];
        total[1] += result[1];
        total[2] += result[2];
    }
    return total;
}

static int run_match() {
    CrossfishDev engine;
    GlobalBoard board;
    std::string cmd;
    std::cout << std::unitbuf;
    while (std::cin >> cmd) {
        if (cmd == "NEW") {
            engine = CrossfishDev();
            board = GlobalBoard();
        } else if (cmd == "APPLY") {
            int mb, sq;
            std::cin >> mb >> sq;
            board.makeMove({(int8_t)mb, (int8_t)sq});
        } else if (cmd == "GO") {
            int ms;
            std::cin >> ms;
            if (ms < 1) ms = 20;
            Move best = engine.getMove(board, std::chrono::milliseconds(ms));
            board.makeMove(best);
            std::cout << (int)best.mini_board << " " << (int)best.square << std::endl;
        }
    }
    return 0;
}

//main function for codingame
int main(int argc, char** argv)
{
    if (argc >= 2 && std::string(argv[1]) == "match") {
        return run_match();
    }

    CrossfishDev crossfish;
    GlobalBoard board;
    // game loop
    while (1) {
        int opponent_row;
        int opponent_col;
        std::cin >> opponent_row >> opponent_col; std::cin.ignore();
        int valid_action_count;
        std::cin >> valid_action_count; std::cin.ignore();
        for (int i = 0; i < valid_action_count; i++) {
            int row;
            int col;
            std::cin >> row >> col; std::cin.ignore();
        }
        if (opponent_row != -1) {
            Move opponent_move = grid_coord_to_move(opponent_row, opponent_col);
            board.makeMove(opponent_move);
            // std::cerr << "Opponent move: " << opponent_move.mini_board << " " << opponent_move.square << std::endl;
        }
        
        if (opponent_row == -1) {
            crossfish.getMove(board, std::chrono::milliseconds(800));
            std::cout << 4 << " " << 4 << std::endl;
            board.makeMove({4, 4});
        }
        else {
            Move best_move = crossfish.getMove(board);
            board.makeMove(best_move);
            std::array<int, 2> grid_coord = move_to_grid_coord(best_move);
            std::cout << grid_coord[0] << " " << grid_coord[1] << " D" << crossfish.depth << " E" << crossfish.root_score <<
            " N" << crossfish.nodes << std::endl;
        }
    }
}
