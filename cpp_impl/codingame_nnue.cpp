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
// Round-7 eval stack vs merged round-6: D16/H8 local net, macro residual,
// and exact macro-score lookup. 95ms N 5152 W 2012 D 1591 L 1549,
// +31.31 +/- 7.91 Elo, LLR +3.063 for H0=+20 / H1=+25.
// Round-8 exact macro-state correction history with a lazy trained prior.
// Independent 90ms book SPRT: N 4672 W 1564 D 1941 L 1167,
// +29.59 +/- 7.63 Elo, LLR +3.022 for H0=+20 / H1=+25; zero timeouts.
// Round-9 tree-identical hot-path rewrite: branchless AVX2 rank sort over one
// packed move key, O(1) cached terminal answer, pre-XORed move/destination/
// side-to-move hash, wide move emit, direct macro-key set/clear, cached
// MiniNet centroid codes, cached global HCE term, inline integer lround.
// The search tree is bit-identical to round eight; only its cost changed.
// Official 90ms SPRT vs the round-eight freeze: N 2366, 723-1048-595,
// +18.81 +/- 10.18 Elo, LLR +3.010 (H0=0, H1=+5) PASS, zero timeouts.
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

// Packed evaluation headers are kept separate for normal builds and inlined by cg_minify.
#include "mini_eval_d16.hpp"
#include "macro_eval.hpp"

enum TTFlag { TT_EXACT = 0, TT_UPPER = 1, TT_LOWER = 2 };
static int g_fixed_search_depth = 0;
static bool g_disable_eval_prune = false;
static bool g_force_hce_eval = false;
// Leave enough of CodinGame's 100 ms turn for setup, scheduler jitter,
// recursive-search unwinding, and flushing the selected move.
static constexpr int CODINGAME_MOVE_MS = 90;

class CrossfishDev {
       private:
        using FastMove = uint8_t;
        static constexpr FastMove NO_FAST_MOVE = 255;

        // Two bytes per history entry instead of two ints: the search board's
        // history is 256 bytes rather than 1 KiB, so FastBoard stays cache hot.
        struct FastHistoryMove {
            uint8_t mini_board = 0;
            uint8_t square = 0;

            FastHistoryMove &operator=(const Move &move) {
                mini_board = (uint8_t)move.mini_board;
                square = (uint8_t)move.square;
                return *this;
            }

            operator Move() const {
                return Move{mini_board, square};
            }
        };
        // The 256-byte history depends on this: padding here would silently
        // double FastBoard's history and cost NPS with nothing failing.
        static_assert(sizeof(FastHistoryMove) == 2);

        struct FastMoveStack {
            std::array<FastHistoryMove, 128> moves{};
            int count = 0;

            FastHistoryMove &top() { return moves[count - 1]; }
            const FastHistoryMove &top() const { return moves[count - 1]; }
            void push(const Move &move) { moves[count++] = move; }
            void pop() { count--; }
            bool empty() const { return count == 0; }
        };

        struct FastBoard {
            using MarkerHashTable =
                std::array<std::array<std::array<uint64_t, 512>, 9>, 2>;
            // move_hashes[p][mb][sq] ^ legal_mini_board_hashes[sq]
            // ^ player_to_move_hash: the three terms every make/unmake always
            // folds in together, pre-XORed into one 1296-byte table.
            using ComboHashTable =
                std::array<std::array<std::array<uint64_t, 9>, 9>, 2>;

            std::array<MiniBoard, 9> mini_boards;
            std::array<int, 3> mini_board_states;
            int out_of_play;
            FastMoveStack move_history;
            uint64_t tt_hash;
            const decltype(GlobalBoard::move_hashes) &move_hashes;
            const decltype(GlobalBoard::mini_board_hashes) &mini_board_hashes;
            const decltype(GlobalBoard::legal_mini_board_hashes) &legal_mini_board_hashes;
            const uint64_t &player_to_move_hash;
            const MarkerHashTable &marker_hashes;
            const ComboHashTable &combo_hashes;
            int n_moves;
            bool prev_move_was_pass;
            uint32_t macro_key[2]{};
            // MiniNet centroid code per (perspective, miniboard); only the
            // played miniboard's two bytes change on a move.
            uint8_t mini_code[2][9]{};
            uint8_t active_board = 9;
            // check_winner_fast's answer for the current position. Only a move
            // that decides a miniboard can change it, so make/unmake refresh it
            // and every node reads one byte instead of redoing the test.
            int8_t terminal = -1;
            std::array<uint8_t, 128> active_board_undo{};

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

            static const ComboHashTable &get_combo_hashes(
                const GlobalBoard &board) {
                static const ComboHashTable hashes = [&board] {
                    ComboHashTable table{};
                    for (int p = 0; p < 2; p++) {
                        for (int mb = 0; mb < 9; mb++) {
                            for (int sq = 0; sq < 9; sq++) {
                                table[p][mb][sq] =
                                    board.move_hashes[p][mb][sq]
                                    ^ board.legal_mini_board_hashes[sq]
                                    ^ board.player_to_move_hash;
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
                  out_of_play(board.mini_board_states[0]
                            | board.mini_board_states[1]
                            | board.mini_board_states[2]),
                  tt_hash(board.zobrist_hash),
                  move_hashes(board.move_hashes),
                  mini_board_hashes(board.mini_board_hashes),
                  legal_mini_board_hashes(board.legal_mini_board_hashes),
                  player_to_move_hash(board.player_to_move_hash),
                  marker_hashes(get_marker_hashes(board)),
                  combo_hashes(get_combo_hashes(board)),
                  n_moves(board.n_moves),
                  prev_move_was_pass(board.prev_move_was_pass) {
                auto history = board.move_history;
                move_history.count = (int)history.size();
                for (int i = move_history.count - 1; i >= 0; i--) {
                    move_history.moves[i] = history.top();
                    history.pop();
                }
                if (n_moves > 0 && !prev_move_was_pass) {
                    int sent = move_history.top().square;
                    if ((out_of_play & (1 << sent)) == 0) {
                        active_board = (uint8_t)sent;
                    }
                }
                // Seed the per-miniboard MiniNet codes; make/unmake keep them
                // current from here on.
                for (int mb = 0; mb < 9; mb++) {
                    const int t0 = fast_mini_index[mini_boards[mb].markers[0]];
                    const int t1 = fast_mini_index[mini_boards[mb].markers[1]];
                    mini_code[0][mb] = D16_MN_CODE[t0 + 2 * t1];
                    mini_code[1][mb] = D16_MN_CODE[t1 + 2 * t0];
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

        static int out_of_play_mask(const FastBoard &board) {
            return board.out_of_play;
        }

        template <typename Board>
        static int out_of_play_mask(const Board &board) {
            return board.mini_board_states[0]
                 | board.mini_board_states[1]
                 | board.mini_board_states[2];
        }

        static int active_board_index(const FastBoard &board) {
            return board.active_board;
        }

        template <typename Board>
        static int active_board_index(const Board &board) {
            return d16_mini_board_constraint(board);
        }

        static void update_active_board(FastBoard &board, int sent) {
            board.active_board_undo[board.n_moves] = board.active_board;
            board.active_board =
                (!board.prev_move_was_pass
                 && (board.out_of_play & (1 << sent)) == 0)
                ? (uint8_t)sent
                : (uint8_t)9;
        }

        template <typename Board>
        static void update_active_board(Board &, int) {}

        static void restore_active_board(FastBoard &board) {
            board.active_board =
                board.active_board_undo[board.n_moves];
        }

        template <typename Board>
        static void restore_active_board(Board &) {}

        template <typename Board>
        static int cached_mini_key(
            const Board &board, int mb, int perspective = 0) {
            return mini_index(
                board.mini_boards[mb].markers[perspective],
                board.mini_boards[mb].markers[perspective ^ 1]);
        }

        // The MiniNet centroid code of a miniboard changes only when that
        // miniboard changes, so it is maintained here (two bytes per move)
        // instead of being looked up nine times per evaluated leaf. Search
        // markers are disjoint, and for disjoint masks the packed table obeys
        // D16_MN_MASK_CODE[(mine << 9) | opp] == D16_MN_CODE[ternary], where
        // the ternary index is the sum fast_mini_index already tabulates. So
        // this reads the same byte the eval used to read, out of a 19 KiB table
        // through a 1 KiB one rather than out of the 256 KiB packed table.
        static void update_mini_code(FastBoard &board, int mb) {
            const int t0 = fast_mini_index[board.mini_boards[mb].markers[0]];
            const int t1 = fast_mini_index[board.mini_boards[mb].markers[1]];
            board.mini_code[0][mb] = D16_MN_CODE[t0 + 2 * t1];
            board.mini_code[1][mb] = D16_MN_CODE[t1 + 2 * t0];
        }

        template <typename Board>
        static void update_mini_code(Board &, int) {}

        static void add_out_of_play(FastBoard &board, int bit) {
            board.out_of_play |= bit;
        }

        template <typename Board>
        static void add_out_of_play(Board &, int) {}

        static void remove_out_of_play(FastBoard &board, int bit) {
            board.out_of_play &= ~bit;
        }

        template <typename Board>
        static void remove_out_of_play(Board &, int) {}

        static void xor_position_hash(FastBoard &board, uint64_t value) {
            board.tt_hash ^= value;
        }

        static void xor_position_hash(GlobalBoard &board, uint64_t value) {
            board.zobrist_hash ^= value;
        }

        static void xor_move_combo(FastBoard &board, int stm, int mb, int sq) {
            board.tt_hash ^= board.combo_hashes[stm][mb][sq];
        }

        static void xor_move_combo(GlobalBoard &board, int stm, int mb,
                                   int sq) {
            board.zobrist_hash ^= board.move_hashes[stm][mb][sq]
                                ^ board.legal_mini_board_hashes[sq]
                                ^ board.player_to_move_hash;
        }

        static void xor_marker_hashes(FastBoard &board, int mb) {
            board.tt_hash ^= board.marker_hashes[0][mb][board.mini_boards[mb].markers[0]]
                          ^ board.marker_hashes[1][mb][board.mini_boards[mb].markers[1]];
        }

        static void xor_marker_hashes(GlobalBoard &, int) {}

        static void sync_macro_key_mb(FastBoard &board, int mb) {
            uint32_t shift = (uint32_t)(2 * mb);
            uint32_t clear = ~(3u << shift);
            int bit = 1 << mb;
            int cls0 = 0;
            int cls1 = 0;
            if (board.mini_board_states[0] & bit) {
                cls0 = 1;
                cls1 = 2;
            } else if (board.mini_board_states[1] & bit) {
                cls0 = 2;
                cls1 = 1;
            } else if (board.mini_board_states[2] & bit) {
                cls0 = cls1 = 3;
            }
            board.macro_key[0] =
                (board.macro_key[0] & clear) | ((uint32_t)cls0 << shift);
            board.macro_key[1] =
                (board.macro_key[1] & clear) | ((uint32_t)cls1 << shift);
        }

        template <typename Board>
        static void sync_macro_key_mb(Board &, int) {}

        // Same result as sync_macro_key_mb when mini_board_states[state] has
        // just gained mb, without re-reading the three state masks.
        static void set_macro_key_mb(FastBoard &board, int mb, int state) {
            static constexpr uint32_t CLS[2][3] = {{1, 2, 3}, {2, 1, 3}};
            uint32_t shift = (uint32_t)(2 * mb);
            uint32_t clear = ~(3u << shift);
            board.macro_key[0] =
                (board.macro_key[0] & clear) | (CLS[0][state] << shift);
            board.macro_key[1] =
                (board.macro_key[1] & clear) | (CLS[1][state] << shift);
        }

        template <typename Board>
        static void set_macro_key_mb(Board &, int, int) {}

        static void clear_macro_key_mb(FastBoard &board, int mb) {
            uint32_t clear = ~(3u << (uint32_t)(2 * mb));
            board.macro_key[0] &= clear;
            board.macro_key[1] &= clear;
        }

        template <typename Board>
        static void clear_macro_key_mb(Board &, int) {}

        static void init_macro_key(FastBoard &board) {
            board.macro_key[0] = board.macro_key[1] = 0;
            for (int mb = 0; mb < 9; mb++) {
                sync_macro_key_mb(board, mb);
            }
        }

        static int evaluate_macro_cached(const FastBoard &board) {
            int stm = board.n_moves & 1;
            int constraint = active_board_index(board);
            return evaluate_macro_key(
                constraint, board.macro_key[stm]);
        }

        // std::lround(float) compiles to an out-of-line lroundf@plt call, one
        // per evaluated leaf. glibc's lroundf is pure integer bit manipulation
        // on the float's encoding; running the same steps inline returns the
        // identical int with no call. Verified equal to (int)std::lround for
        // all 2,650,800,126 finite floats with |x| < 2^31.
        static int lround_bits(float x) {
            uint32_t bits;
            memcpy(&bits, &x, sizeof(bits));
            const int exponent = (int)((bits >> 23) & 0xff) - 127;
            const int sign = ((int32_t)bits >> 31) | 1;
            if (exponent >= 31) return (int)(long)x;
            if (exponent < 0) return (exponent == -1) ? sign : 0;
            uint32_t mantissa = (bits & 0x7fffffu) | 0x800000u;
            if (exponent > 22) {
                return sign * (int)(mantissa << (exponent - 23));
            }
            mantissa += 0x400000u >> exponent;
            return sign * (int)(mantissa >> (23 - exponent));
        }

        // Same network and the same arithmetic as d16_evaluate_mini_fast, but
        // the per-miniboard centroid codes and the super-board classes come out
        // of FastBoard's incremental caches instead of nine probes into the
        // 256 KiB packed table plus three state tests per miniboard.
        static int evaluate_mini_cached(const FastBoard &board) {
            const int stm = board.n_moves & 1;
            const int c = active_board_index(board);
            __m256i hidden = _mm256_load_si256(
                (const __m256i *)D16_MN_FACTOR_INIT[c]);
            uint32_t macro = board.macro_key[stm];
            for (int mb = 0; mb < 9; mb++, macro >>= 2) {
                const int code = board.mini_code[stm][mb];
                hidden = _mm256_add_epi32(
                    hidden,
                    _mm256_load_si256(
                        (const __m256i *)D16_MN_FACTOR_CODE[mb][code]));
                const int super_cls = macro & 3;
                if (super_cls) {
                    hidden = _mm256_add_epi32(
                        hidden,
                        _mm256_load_si256(
                            (const __m256i *)
                                    D16_MN_FACTOR_SUPER[mb][super_cls]));
                }
            }
            if (c < 9) {
                hidden = _mm256_add_epi32(
                    hidden,
                    _mm256_load_si256(
                        (const __m256i *)D16_MN_FACTOR_ACTIVE[c]));
            }
            __m256 value = _mm256_cvtepi32_ps(hidden);
            value = _mm256_max_ps(value, _mm256_setzero_ps());
            value = _mm256_mul_ps(value, _mm256_loadu_ps(D16_MN_W2));
            const float out =
                D16_MN_B2
                + d16_mini_hsum256(value) / D16_MN_FACTOR_SCALE;
            return lround_bits(out);
        }

        std::chrono::milliseconds thinking_time =
            std::chrono::milliseconds(CODINGAME_MOVE_MS);
        Move root_best_move;
        std::chrono::time_point<std::chrono::high_resolution_clock> start_time =  std::chrono::high_resolution_clock::now();
        int min_val = -99999;
        int max_val = 99999;
        bool stopped = false;
    public:
        int root_score;
        int nodes;
        std::array<std::array<uint8_t, 9>, 128> killer_moves;
        std::array<std::array<std::array<int, 9>, 9>, 2> history_table{};
        FastMove counter_move[9][9];
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
        static constexpr int CORR_LOCAL_KEYS = 19683 + 1;
        static constexpr int CORR_LOCAL_GRAIN = 48;
        static constexpr int CORR_MACRO_KEYS = 1 << 18;
        static constexpr int CORR_MACRO_GRAIN = 12;
        // Mate scores are +/-(max_val - ply), i.e. distances to mate rather than
        // quantities the static eval can be measured against.
        static constexpr int CORR_MATE_BOUND = 90000;
        struct CorrEntry {
            int16_t raw = 0;
            int16_t applied = 0;
        };
        static_assert(sizeof(CorrEntry) == 4);
        std::array<std::array<std::array<CorrEntry, CORR_MASKS>, CORR_MB>, 2>
            corr_hist{};
        // A second correction learns residuals by the exact STM-relative shape
        // of the forced miniboard. The final slot represents a free move.
        std::array<CorrEntry, CORR_LOCAL_KEYS> corr_local_hist{};
        // The incrementally maintained macro key is already an exact compact
        // description of all nine decided-miniboard states from the side to
        // move's perspective, so use it directly rather than hashing it.
        std::array<CorrEntry, CORR_MACRO_KEYS> corr_macro_hist{};

        struct HceUndo {
            int16_t score = 0;
            uint8_t flags = 0;
        };
        int hce_local_score = 0;
        int hce_global_score = 0;
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

        // fill_fast_legal_moves emits a miniboard's squares 0..7 with one
        // 8-byte store, so its output buffer needs slack past the 81 slots a
        // position can legally use. 80 real moves + a 7-byte tail = index 86.
        static constexpr int FAST_MOVE_SLACK = 15;

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
        // fast_win_moves, forced to zero once the mask already contains a line.
        // Used at the macro level so the "already won" guard costs no branch.
        static inline uint16_t fast_win_moves_open[1 << 9];
        // The set bits of an 8-bit mask as a little-endian list of byte
        // indices, so a whole miniboard's empty squares are emitted with one
        // 8-byte store instead of one store per square.
        static inline uint64_t fast_empty_squares[1 << 8];
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
        static constexpr int OPP_LATENT_CAPTURE_BONUS = -800;
        static constexpr int LUT_W_TIAR = 534;
        static constexpr int LUT_W_CENTER_SQ = 33;
        static constexpr int LUT_W_CORNER_SQ = PAWN;
        static constexpr int LUT_W_SQUARES = 33;
        // Random-play MiniNet residuals reached ~4500; slack for search positions.
        static constexpr int MINI_MAX = 8000;
        CrossfishDev() {
            d16_mini_load_packed();
            macro_load_packed();
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
                    fast_win_moves_open[markers] =
                        fast_has_win[markers] ? 0 : (uint16_t)wins;
                }
                for (int mask = 0; mask < 256; mask++) {
                    uint64_t list = 0;
                    int k = 0;
                    for (int b = 0; b < 8; b++) {
                        if (mask & (1 << b)) {
                            list |= (uint64_t)b << (8 * k++);
                        }
                    }
                    fast_empty_squares[mask] = list;
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
            if ((nodes & 127) == 0) {
                if (std::chrono::high_resolution_clock::now() - start_time
                    >= thinking_time) {
                    stopped = true;
                }
            }
            return stopped;
        }

        // Live entry for this position; the constraint test mirrors fillLegalMoves.
        CorrEntry &corr_entry(FastBoard &board) {
            int out_of_play = out_of_play_mask(board);
            int mb = active_board_index(board);
            return corr_hist[(board.n_moves & 1)][mb][out_of_play];
        }

        CorrEntry &corr_local_entry(FastBoard &board) {
            int key = MINI_LUT_SIZE;
            int mb = active_board_index(board);
            if (mb < 9) {
                int stm = board.n_moves & 1;
                key = cached_mini_key(board, mb, stm);
            }
            return corr_local_hist[key];
        }

        CorrEntry &corr_macro_entry(FastBoard &board) {
            uint32_t key = board.macro_key[board.n_moves & 1];
            CorrEntry &entry = corr_macro_hist[key];
            if (entry.raw == 0 && entry.applied == 0
                && __builtin_popcount((unsigned)board.out_of_play) >= 3) {
                int raw = evaluate_macro_key(9, key) * CORR_MACRO_GRAIN;
                if (raw > CORR_MAX) raw = CORR_MAX;
                if (raw < -CORR_MAX) raw = -CORR_MAX;
                // Keep a zero-valued prior distinguishable from untouched
                // storage so it is initialized only once per exact state.
                entry.raw = (int16_t)(raw == 0 ? 1 : raw);
                entry.applied =
                    (int16_t)(entry.raw / CORR_MACRO_GRAIN);
            }
            return entry;
        }

        struct CorrRefs {
            CorrEntry *structural = nullptr;
            CorrEntry *local = nullptr;
            CorrEntry *macro = nullptr;
        };

        CorrRefs corr_refs(FastBoard &board) {
            return CorrRefs{
                &corr_entry(board),
                &corr_local_entry(board),
                &corr_macro_entry(board)
            };
        }

        // Clamped clear of the decided-game band, leaving room for the MiniNet
        // residual that qsearch adds on top of this value.
        static constexpr int CORR_EVAL_LIMIT = CORR_MATE_BOUND - MINI_MAX - 1;
        int corrected_eval(int static_eval, CorrRefs refs) {
            int v = static_eval
                  + refs.structural->applied
                  + refs.local->applied
                  + refs.macro->applied;
            if (v > CORR_EVAL_LIMIT) v = CORR_EVAL_LIMIT;
            if (v < -CORR_EVAL_LIMIT) v = -CORR_EVAL_LIMIT;
            return v;
        }

        void update_corr_entry(CorrEntry &e, int diff, int w, int grain) {
            int raw = e.raw;
            raw += diff * w - raw * abs(diff) * w / CORR_SCALE;
            if (raw > CORR_MAX) raw = CORR_MAX;
            if (raw < -CORR_MAX) raw = -CORR_MAX;
            e.raw = (int16_t)raw;
            e.applied = (int16_t)(raw / grain);
        }

        void update_corr_hist(CorrRefs refs, int diff, int d) {
            if (diff > CORR_DIFF_MAX) diff = CORR_DIFF_MAX;
            if (diff < -CORR_DIFF_MAX) diff = -CORR_DIFF_MAX;
            int w = std::min(d, CORR_MAX_WEIGHT);
            update_corr_entry(*refs.structural, diff, w, CORR_GRAIN);
            update_corr_entry(*refs.local, diff, w, CORR_LOCAL_GRAIN);
            update_corr_entry(*refs.macro, diff, w, CORR_MACRO_GRAIN);
        }

        // Recompute the cached terminal answer. Called only where a miniboard
        // was just decided or undecided, which is the only way it can change.
        static void sync_terminal(FastBoard &board) {
            int p0 = board.mini_board_states[0];
            int p1 = board.mini_board_states[1];
            int t = -1;
            if (fast_has_win[p0]) {
                t = 0;
            } else if (fast_has_win[p1]) {
                t = 1;
            } else if (board.out_of_play == 511) {
                int n0 = __builtin_popcount((unsigned)p0);
                int n1 = __builtin_popcount((unsigned)p1);
                t = n0 > n1 ? 0 : (n1 > n0 ? 1 : 2);
            }
            board.terminal = (int8_t)t;
        }

        template <typename Board>
        static void sync_terminal(Board &) {}

        int check_winner_fast(FastBoard &board) {
            return board.terminal;
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
            int active = active_board_index(board);
            int out_of_play = out_of_play_mask(board);
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
            if (active < 9) {
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
            int out_of_play = out_of_play_mask(board);
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
            hce_global_score = evaluate_hce_global(board);
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
                xor_position_hash(
                    board,
                    board.legal_mini_board_hashes[board.move_history.top().square]);
            }
            board.move_history.push(move);
            board.mini_boards[move.mini_board].markers[stm] = before | bit;
            update_mini_code(board, move.mini_board);
            // One lookup folds the stone, the destination-constraint term and
            // the side-to-move flip that used to be three separate XORs.
            xor_move_combo(board, stm, move.mini_board, move.square);
            int decided_state = -1;
            if (fast_win_moves[before] & bit) {
                board.mini_board_states[stm] |= mb_bit;
                xor_position_hash(
                    board, board.mini_board_hashes[stm][move.mini_board]);
                decided_state = stm;
            } else {
                int occupied = board.mini_boards[move.mini_board].markers[0]
                             | board.mini_boards[move.mini_board].markers[1];
                if (occupied == 511) {
                    board.mini_board_states[2] |= mb_bit;
                    xor_position_hash(
                        board, board.mini_board_hashes[2][move.mini_board]);
                    decided_state = 2;
                }
            }
            if (decided_state >= 0) {
                add_out_of_play(board, mb_bit);
                xor_marker_hashes(board, move.mini_board);
                // The deciding state is already known here, so there is no
                // need to re-test all three mini_board_states masks.
                set_macro_key_mb(board, move.mini_board, decided_state);
                sync_terminal(board);
            }
            update_active_board(board, move.square);
            board.n_moves++;
            if (hce_acc_ready) {
                set_hce_mb(board, move.mini_board);
                // The global terms read only mini_board_states, which change
                // here exactly when this move decided a miniboard.
                if (decided_state >= 0) {
                    hce_global_score = evaluate_hce_global(board);
                }
            }
        }

        template <typename Board>
        void unmake_move_fast(Board &board) {
            board.n_moves--;
            Move move = board.move_history.top();
            board.move_history.pop();
            int mb_bit = 1 << move.mini_board;
            bool was_decided = false;
            if (board.mini_board_states[0] & mb_bit) {
                board.mini_board_states[0] &= ~mb_bit;
                xor_position_hash(
                    board, board.mini_board_hashes[0][move.mini_board]);
                was_decided = true;
            } else if (board.mini_board_states[1] & mb_bit) {
                board.mini_board_states[1] &= ~mb_bit;
                xor_position_hash(
                    board, board.mini_board_hashes[1][move.mini_board]);
                was_decided = true;
            } else if (board.mini_board_states[2] & mb_bit) {
                board.mini_board_states[2] &= ~mb_bit;
                xor_position_hash(
                    board, board.mini_board_hashes[2][move.mini_board]);
                was_decided = true;
            }
            if (was_decided) {
                remove_out_of_play(board, mb_bit);
                xor_marker_hashes(board, move.mini_board);
                // Undoing a decision always returns the slot to "undecided".
                clear_macro_key_mb(board, move.mini_board);
                sync_terminal(board);
            }
            int stm = board.n_moves & 1;
            board.mini_boards[move.mini_board].markers[stm] &= ~(1 << move.square);
            update_mini_code(board, move.mini_board);
            xor_move_combo(board, stm, move.mini_board, move.square);
            if (board.n_moves > 0) {
                xor_position_hash(
                    board,
                    board.legal_mini_board_hashes[board.move_history.top().square]);
            }
            if (hce_acc_ready) {
                restore_hce_mb(board.n_moves, move.mini_board);
                if (was_decided) {
                    hce_global_score = evaluate_hce_global(board);
                }
            }
            restore_active_board(board);
        }

        Move getMove(
            GlobalBoard input_board,
            std::chrono::milliseconds thinking_time_passed =
                std::chrono::milliseconds(CODINGAME_MOVE_MS)) {
            thinking_time = thinking_time_passed;
            start_time = std::chrono::high_resolution_clock::now();
            init_mini_lut();
            FastBoard board(input_board);
            init_lmr_table();
            nodes = 0;
            stopped = false;
            root_score = 0;
            Move root_moves[81];
            fill_legal_moves_fast(board, root_moves);
            root_best_move = root_moves[0];
            init_hce_acc(board);
            init_macro_key(board);
            sync_terminal(board);
            killer_moves = {};
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
                        counter_move[i][j] = NO_FAST_MOVE;
                    }
                }
                counters_ready = true;
            }
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
        // Returns false on timeout/unfinished sentinel. Mates clamp to +/-20000.
        static constexpr int SEARCH_SCORE_CLAMP = 20000;
        bool search_fixed_depth(GlobalBoard &input_board, int d, int &out_score) {
            init_mini_lut();
            FastBoard board(input_board);
            init_lmr_table();
            thinking_time = std::chrono::milliseconds(24 * 60 * 60 * 1000);
            nodes = 0;
            stopped = false;
            root_score = 0;
            depth = d;
            init_hce_acc(board);
            init_macro_key(board);
            sync_terminal(board);
            killer_moves = {};
            history_table = {};
            for (int mb = 0; mb < 9; mb++) {
                for (int sq = 0; sq < 9; sq++) {
                    counter_move[mb][sq] = NO_FAST_MOVE;
                }
            }
            counters_ready = true;
            corr_hist = {};
            corr_local_hist = {};
            corr_macro_hist = {};
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
                    if (winner == (board.n_moves & 1)) {
                        return max_val - ply;
                    }
                    else {
                        return min_val + ply;
                    }
                }
            }

            CorrEntry &qstruct = corr_entry(board);
            int hce = evaluate_hce_incremental(board)
                    + qstruct.applied;
            if (hce - 640 >= beta) {
                return beta;
            }
            int stand_pat;
            if (hce + MINI_MAX + MACRO_CLIP < alpha) {
                stand_pat = hce + MINI_MAX + MACRO_CLIP;
            } else {
                stand_pat = hce + evaluate_mini_cached(board)
                          + evaluate_macro_cached(board);
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
            int cap_keys[81];
            int n_caps = fill_fast_captures(board, caps);
            get_fast_move_scores(caps, n_caps, board, ply, cap_keys, true);
            sort_move_keys(cap_keys, n_caps);
            int val;
            for (int i = 0; i < n_caps; i++) {
                Move move = unpack_fast_move(move_from_key(cap_keys[i]));
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
                    if (winner == (board.n_moves & 1)) {
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
            CorrRefs static_corr_refs{};
            if (!pv_node && !g_disable_eval_prune) {
                static_corr_refs = corr_refs(board);
                static_eval = corrected_eval(
                    evaluate_hce_incremental(board), static_corr_refs);
                have_static = true;

                int reverse_futility_margin = RFP_PAWNS * eval_weights[PAWN_IDX];
                if (static_eval - reverse_futility_margin * depth >= beta) {
                    return beta;
                }
                if (depth == 1
                    && static_eval + 2500 - reverse_futility_margin >= beta
                    && static_eval + evaluate_mini_cached(board)
                       - reverse_futility_margin >= beta) {
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

            // The wide emit in fill_fast_legal_moves stores 8 bytes at a time,
            // so the move buffer carries slack past the 81 real slots.
            FastMove legal_moves[81 + FAST_MOVE_SLACK];
            int move_keys[81];
            int nmoves = fill_fast_legal_moves(board, legal_moves);
            bool defer_move_scores = false;
            int tt_index = -1;
            // Generated moves are grouped by miniboard, so when the position is
            // forced the only candidate slot for the tt move is that miniboard:
            // two compares replace a whole pass over the move list.
            if (tt_move != NO_FAST_MOVE) {
                int forced = active_board_index(board);
                if (forced == 9 || (tt_move >> 4) == forced) {
                    for (int i = 0; i < nmoves; i++) {
                        if (legal_moves[i] == tt_move) {
                            tt_index = i;
                            break;
                        }
                    }
                }
            }
            if (tt_index >= 0) {
                FastMove hash_move = legal_moves[tt_index];
                for (int i = tt_index; i > 0; i--) {
                    legal_moves[i] = legal_moves[i - 1];
                }
                legal_moves[0] = hash_move;
                move_keys[0] = pack_move_key(1000, hash_move);
                defer_move_scores = true;
            } else {
                get_fast_move_scores(legal_moves, nmoves, board, ply, move_keys, false);
                sort_move_keys(move_keys, nmoves);
            }

            FastMove best_move = move_from_key(move_keys[0]);
            int best_val = min_val;
            int alpha_orig = alpha;
            int val;
            int stm = board.n_moves & 1;
            int opponent_global_targets =
                fast_win_moves[board.mini_board_states[stm ^ 1]];
            for (int i = 0; i < nmoves; i++) {
                if (i == 1 && defer_move_scores) {
                    get_fast_move_scores(legal_moves + 1, nmoves - 1,
                                         board, ply, move_keys + 1, false);
                    sort_move_keys(move_keys + 1, nmoves - 1);
                }
                FastMove fast_move = move_from_key(move_keys[i]);
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
                    bool do_lmr =
                        (move_keys[i] > MOVE_KEY_ZERO || (i >= 2 && !capture));
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
                    int &h = history_table[board.n_moves & 1][mb][sq];
                    int bonus = depth * depth;
                    h += bonus - h * bonus / 10000;
                    int stm = board.n_moves & 1;
                    for (int j = 0; j < i; j++) {
                        FastMove prior = move_from_key(move_keys[j]);
                        if (is_fast_capture(board, prior)) continue;
                        int &hj = history_table[stm][prior >> 4][prior & 15];
                        int malus = 2 * bonus;
                        hj -= malus + hj * malus / 10000;
                        if (hj < -10000) hj = -10000;
                    }
                    if (board.n_moves > 0) {
                        Move prev = board.move_history.top();
                        counter_move[prev.mini_board][prev.square] = fast_move;
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
                    update_corr_hist(
                        static_corr_refs, best_val - static_eval, depth);
                }
            }

            return best_val;
        }

        // Scratch for the wide path. Members, not locals, so the recursive
        // search frame does not grow by hundreds of bytes per ply.
        int sort_val[88]{};
        int sort_rank[88]{};

        // Branchless rank sort of the packed move keys, replacing the stable
        // insertion sort. Keys inside one group are unique, so a key's rank is
        // exactly its sorted index and scattering by rank reproduces the
        // insertion sort's permutation with no data-dependent branch at all.
        // Lanes past n read as INT_MAX, which exceeds every real key and so
        // never contributes to a rank. Measured on real search data, 93% of
        // ordered move lists have n <= 8.
        void sort_move_keys(int *keys, int n) {
            if (n < 2) return;
            if (n <= 8) {
                const __m256i mask = _mm256_cmpgt_epi32(
                    _mm256_set1_epi32(n),
                    _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7));
                const __m256i v = _mm256_blendv_epi8(
                    _mm256_set1_epi32(0x7fffffff),
                    _mm256_maskload_epi32(keys, mask), mask);
                int val[8];
                int rank[8];
                _mm256_storeu_si256((__m256i *)val, v);
                __m256i r = _mm256_setzero_si256();
                for (int j = 0; j < n; j++) {
                    r = _mm256_sub_epi32(
                        r, _mm256_cmpgt_epi32(
                               v, _mm256_set1_epi32(val[j])));
                }
                _mm256_storeu_si256((__m256i *)rank, r);
                for (int i = 0; i < n; i++) keys[rank[i]] = val[i];
                return;
            }
            const int padded = (n + 7) & ~7;
            for (int i = 0; i < n; i++) sort_val[i] = keys[i];
            for (int i = n; i < padded; i++) sort_val[i] = 0x7fffffff;
            for (int b = 0; b < n; b += 8) {
                const __m256i v = _mm256_loadu_si256(
                    (const __m256i *)(sort_val + b));
                __m256i r = _mm256_setzero_si256();
                for (int j = 0; j < n; j++) {
                    r = _mm256_sub_epi32(
                        r, _mm256_cmpgt_epi32(
                               v, _mm256_set1_epi32(sort_val[j])));
                }
                _mm256_storeu_si256((__m256i *)(sort_rank + b), r);
            }
            for (int i = 0; i < n; i++) keys[sort_rank[i]] = sort_val[i];
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
            int active = active_board_index(board);
            // Squares 0..7 of a miniboard leave in one 8-byte store: the index
            // list comes from the table and the miniboard tag is broadcast into
            // the high nibble of every byte. Bytes past the count are either
            // overwritten by the next group or never read, so `dst` only needs
            // FAST_MOVE_SLACK bytes of tail room.
            auto add_from_mb = [&](int mb) {
                int occupied = board.mini_boards[mb].markers[0]
                             | board.mini_boards[mb].markers[1];
                int empty = (~occupied) & 511;
                uint64_t packed = fast_empty_squares[empty & 255]
                                | (uint64_t)(mb << 4) * 0x0101010101010101ull;
                __builtin_memcpy(dst + n, &packed, sizeof(packed));
                n += __builtin_popcount((unsigned)(empty & 255));
                dst[n] = pack_fast_move(mb, 8);
                n += (empty >> 8) & 1;
            };
            if (active < 9) {
                add_from_mb(active);
            } else {
                int live = (~out_of_play_mask(board)) & 511;
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
            int active_square = active_board_index(board);
            // `& 1` rather than `% 2`: n_moves is never negative, and signed
            // remainder costs four extra instructions per use.
            int stm = board.n_moves & 1;
            auto add_from_mb = [&](int mb) {
                // fast_win_moves never marks a square the mover already holds,
                // so masking out the opponent's stones is the whole occupancy
                // test and the 9-bit clamp is a no-op.
                int wins = fast_win_moves[board.mini_boards[mb].markers[stm]]
                         & ~board.mini_boards[mb].markers[stm ^ 1];
                while (wins) {
                    int sq = __builtin_ctz(wins);
                    wins &= wins - 1;
                    dst[n++] = pack_fast_move(mb, sq);
                }
            };
            if (active_square < 9) {
                add_from_mb(active_square);
            } else {
                int live = (~out_of_play_mask(board)) & 511;
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
            int stm = board.n_moves & 1;
            int mine = board.mini_boards[move >> 4].markers[stm];
            return (fast_win_moves[mine] & (1 << (move & 15))) != 0;
        }

        template <typename Board>
        bool has_immediate_global_win(Board &board, int targets) {
            int stm = board.n_moves & 1;
            if (fast_has_win[board.mini_board_states[stm ^ 1]]) return false;
            // hce_tiar_maps only ever holds bits for live miniboards, so the
            // out-of-play intersection it used to be combined with is implied
            // and the mask no longer has to be loaded or inverted here.
            targets &= hce_tiar_maps[stm];
            int active = active_board_index(board);
            if (active < 9) {
                targets &= 1 << active;
            }
            return targets != 0;
        }

        template <typename Board>
        bool has_forced_global_win_after_reply(Board &board, int player) {
            if ((board.n_moves & 1) == player) {
                return false;
            }
            // fast_win_moves_open is zero exactly when `player` already owns a
            // macro line, folding the old fast_has_win guard into this lookup,
            // and hce_tiar_maps is already confined to live miniboards. This is
            // the hot path: 97.5% of calls leave with no winning target.
            int winning_targets =
                fast_win_moves_open[board.mini_board_states[player]]
                & hce_tiar_maps[player];
            if (winning_targets == 0) return false;
            int out_of_play = out_of_play_mask(board);

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

            int active = active_board_index(board);
            if (active < 9) {
                if (miniboard_refutes(active)) return false;
                return any_reply;
            }
            int live = (~out_of_play) & 511;
            while (live) {
                int mb = __builtin_ctz(live);
                live &= live - 1;
                if (miniboard_refutes(mb)) return false;
            }
            return any_reply;
        }

        // One 32-bit ordering key per move: ((BIAS - score) << 8) | packed move.
        // Ascending key order == descending score, ties broken by the smaller
        // packed move. Both move generators emit (mini_board, square) in
        // increasing order and the hash-move rotation only deletes one element,
        // so "smaller packed move" is always "smaller original index": sorting
        // these keys ascending reproduces the stable insertion sort's exact
        // permutation. Keys are unique, so any total-order sort works.
        static constexpr int MOVE_KEY_BIAS = 1 << 20;
        // Keys with score < 0, i.e. (BIAS - score) > BIAS, all exceed this.
        static constexpr int MOVE_KEY_ZERO = (MOVE_KEY_BIAS << 8) | 255;
        static int pack_move_key(int score, FastMove move) {
            return ((MOVE_KEY_BIAS - score) << 8) | move;
        }
        static FastMove move_from_key(int key) {
            return (FastMove)(key & 255);
        }
        template <typename Board>
        void get_fast_move_scores(FastMove* moves, int n, Board &board, int ply,
                                  int* keys, bool qs = false) {
            if (n <= 1) {
                if (n == 1) keys[0] = pack_move_key(0, moves[0]);
                return;
            }
            int out_of_play = out_of_play_mask(board);
            int stm = (board.n_moves & 1);
            FastMove cm = NO_FAST_MOVE;
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
                FastMove move = moves[i];
                int mb = move >> 4;
                int sq = move & 15;
                if (mb != last_mb) {
                    last_mb = mb;
                    last_idx = cached_mini_key(board, mb);
                    capture_mask = fast_win_moves[board.mini_boards[mb].markers[stm]];
                    block_mask = fast_win_moves[board.mini_boards[mb].markers[stm ^ 1]];
                    tiar_mask = mini_tiar_sq[last_idx][stm];
                    global_win_bonus =
                        800 * fast_has_win[board.mini_board_states[stm] | (1 << mb)];
                }
                int capture = (capture_mask >> sq) & 1;
                keys[i] = pack_move_key(
                    25 * killer_moves[ply][sq]
                    + 40 * (cm == move)
                    + capture * (global_win_bonus + 100 * !qs)
                    + 75 * ((block_mask >> sq) & 1)
                    + 50 * ((tiar_mask >> sq) & 1)
                    - 250 * ((out_of_play >> sq) & 1)
                    + history_table[stm][mb][sq] / 20,
                    move);
            }
        }

        template <typename Board>
        int fill_captures_lut(Board &board, Move* dst) {
            int n = 0;
            if (board.n_moves == 0) return 0;
            int active_square = active_board_index(board);
            int out_of_play = out_of_play_mask(board);
            int stm = (board.n_moves & 1);
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
            if (active_square < 9) {
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
            int stm = (board.n_moves & 1);
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
            int idx = cached_mini_key(board, move.mini_board);
            int stm = (board.n_moves & 1);
            return (mini_tiar_sq[idx][stm] & (1 << move.square)) != 0;
        }

        template <typename Board>
        void get_move_scores(Move* moves, int n, Board &board, int &ply,
                             int* scores, bool qs = false) {
            if (n <= 1) {
                if (n == 1) scores[0] = 0;
                return;
            }
            int out_of_play = out_of_play_mask(board);
            int stm = (board.n_moves & 1);
            Move cm{99, 99};
            if (board.n_moves > 0) {
                Move prev = board.move_history.top();
                FastMove counter = counter_move[prev.mini_board][prev.square];
                if (counter != NO_FAST_MOVE) cm = unpack_fast_move(counter);
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
                    last_idx = cached_mini_key(board, mb);
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
            int out_of_play = out_of_play_mask(board);
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
                const MiniLut &e =
                    mini_lut[cached_mini_key(board, miniboard)];
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
            int stm_sign = ((board.n_moves & 1) == 0) ? 1 : -1;
            int out_of_play = out_of_play_mask(board);
            int p0_two_in_a_row_map = 0;
            int p1_two_in_a_row_map = 0;
            int corners = (1 << 0) + (1 << 2) + (1 << 6) + (1 << 8);
            n_out = 0;
            for (int miniboard = 0; miniboard < 9; miniboard++) {
                if ((out_of_play & (1 << miniboard)) != 0) {
                    continue;
                }
                int idx = cached_mini_key(board, miniboard);
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

        template <typename Board>
        int eval_extra_from_maps(Board &board,
                                 int p0_two_in_a_row_map,
                                 int p1_two_in_a_row_map) {
            int extra = 0;
            if (board.n_moves > 0 && active_board_index(board) == 9) {
                extra += FREE_MOVE_PAWNS * eval_weights[PAWN_IDX];
            }
            int live = (~out_of_play_mask(board)) & 511;
            // Only the side that is not to move is tested, so build that one
            // mask rather than both.
            const int other = (board.n_moves & 1) ^ 1;
            const int other_map =
                other ? p1_two_in_a_row_map : p0_two_in_a_row_map;
            bool opponent_has_latent_capture =
                (fast_win_moves[board.mini_board_states[other]]
                 & other_map & live) != 0;
            if (opponent_has_latent_capture) {
                extra += OPP_LATENT_CAPTURE_BONUS;
            }
            return extra;
        }

        // STM-centric bonuses on top of the linear/LUT eval. This public
        // reference path reconstructs the local threat maps for consistency
        // tests; the search passes its incremental maps directly below.
        template <typename Board>
        int eval_extra(Board &board) {
            int live = (~out_of_play_mask(board)) & 511;
            int p0_two_in_a_row_map = 0;
            int p1_two_in_a_row_map = 0;
            while (live) {
                int miniboard = __builtin_ctz(live);
                live &= live - 1;
                int packed = (board.mini_boards[miniboard].markers[0] << 9)
                           | board.mini_boards[miniboard].markers[1];
                int flags = fast_tiar_flags[packed];
                p0_two_in_a_row_map |= (flags & 1) << miniboard;
                p1_two_in_a_row_map |= ((flags >> 1) & 1) << miniboard;
            }
            return eval_extra_from_maps(
                board, p0_two_in_a_row_map, p1_two_in_a_row_map);
        }

        template <typename Board>
        int evaluate_hce_global(Board &board) {
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
            return global;
        }

        template <typename Board>
        int finish_hce_with_global(Board &board, int local,
                                   int p0_two_in_a_row_map,
                                   int p1_two_in_a_row_map,
                                   int global) {
            int stm_sign = ((board.n_moves & 1) == 0) ? 1 : -1;
            int p0_miniboards = board.mini_board_states[0];
            int p1_miniboards = board.mini_board_states[1];
            global += eval_weights[5]
                * ((int)fast_threat_count[
                       ((p0_miniboards | p0_two_in_a_row_map) << 9) | p1_miniboards]
                   - (int)fast_threat_count[
                       ((p1_miniboards | p1_two_in_a_row_map) << 9) | p0_miniboards]);
            return stm_sign * (global + local) + eval_weights[9]
                 + eval_extra_from_maps(
                       board, p0_two_in_a_row_map, p1_two_in_a_row_map);
        }

        template <typename Board>
        int finish_hce(Board &board, int local,
                       int p0_two_in_a_row_map,
                       int p1_two_in_a_row_map) {
            return finish_hce_with_global(
                board, local, p0_two_in_a_row_map, p1_two_in_a_row_map,
                evaluate_hce_global(board));
        }

        template <typename Board>
        int evaluate_hce(Board &board) {
            int out_of_play = out_of_play_mask(board);
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
            // hce_global_score is maintained on every miniboard decision, so
            // the four popcounts, four multiplies and two 256 KiB threat-table
            // reads of evaluate_hce_global run once per decided move instead of
            // once per evaluated node (about 168k times instead of 1.2M in a
            // 2.1M-node search).
            return finish_hce_with_global(
                board, hce_local_score, hce_tiar_maps[0],
                hce_tiar_maps[1], hce_global_score);
        }

        int evaluate(GlobalBoard &board) {
            if (g_force_hce_eval) {
                return evaluate_hce(board);
            }
            return evaluate_hce(board)
                 + d16_evaluate_mini_fast(board)
                 + evaluate_macro_fast(board);
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
            std::cout << "READY" << std::endl;
        } else if (cmd == "APPLY") {
            int mb, sq;
            std::cin >> mb >> sq;
            board.makeMove({(int8_t)mb, (int8_t)sq});
        } else if (cmd == "SYNC") {
            std::cout << "READY" << std::endl;
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
            // The opening is fixed, so this search only warms the persistent
            // tables and search state. Keep it within the normal move budget:
            // eager NNUE/macro initialization also counts against CodinGame's
            // one-second first-turn deadline.
            crossfish.getMove(
                board, std::chrono::milliseconds(CODINGAME_MOVE_MS));
            std::cout << 4 << " " << 4 << std::endl;
            board.makeMove({4, 4});
        }
        else {
            Move best_move = crossfish.getMove(
                board, std::chrono::milliseconds(CODINGAME_MOVE_MS));
            board.makeMove(best_move);
            std::array<int, 2> grid_coord = move_to_grid_coord(best_move);
            std::cout << grid_coord[0] << " " << grid_coord[1] << " D" << crossfish.depth << " E" << crossfish.root_score <<
            " N" << crossfish.nodes << std::endl;
        }
    }
}
