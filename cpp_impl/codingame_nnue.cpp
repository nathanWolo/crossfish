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
//a struct representing a 3x3 board with 16 bit integers
struct MiniBoard {
    std::array<int, 2> markers = {0, 0};
};

struct Move {
    int8_t mini_board = 99;
    int8_t square = 99;
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
CXe/Ynh3P4E2dT87As2/rOnMv9VFzT8r98w/a5jNPyTMzD85ns2/mhTNvzJjrT/7sK0/tlmsv/+SrL9FNq2/DxKtv2VjrT9Yp60/
kJOOvyyRjr9J3o8/kTiNP9oBjj8Qzo0/FbONv4Igj78u3FQ4ZvSDOZo3DTvsDzk830bWOgU8bjtnv7M5V8yPupk9Xb/SmFa/W2tT
P4mxYj/+ZV8/u1RfP5BUcL9ip1G/XEEUvxc6Fr+nVRQ/YmsWP4FYFz9kDxU/ztMYv9eNF7+JMAVA6oAFQFufBcDnSgXApmkFwAup
BcCvNARAe+gFQOEwsj/0KrI/etGwv++asL8IiLC/ZPKxv7AhsT8qS7E/dYotwMV7LcDEZS1AyEstQI6MLEDiNy5A2EAswBabLMBL
P5Y/DrmWP6Ikl7+xtpe/022Xv4Xhlr8RbZY/2a6XP6Q6PD/zLz8/pDpDv45zPb8Kdj+/4zg9vwGnLD/5+0U/qKcTQDuYE0BVDRTA
7vwSwFGPEsCBBRTAJjsTQL49E0BJkq4+5VS5Pobivr58EbW+NQG3vs6Gr748npY+Rj/KPoGInb+eiZ2/cuWdP5kWnj8SJZ4/3Zmd
PwrUnb8fWZ6/EHM0P7ddOD8J7za/AKMrv0F3Lr/w/TC/tPUmPzlLOj9h1ka/cWBHv454Sj/G+Uk/GbBLP1wNSD8NxUW/BEVLv0Ax
jr7uzY2+wmOOPnIGcj6wB3g+VCOEPhhqfr7DJoq+0v0dP8EJGj+jexG/RDEdv0+eGL/wciC/otkoPy52Dj87Ka+/pDuuv/5xrj/C
h64/ftKuPzKyrj/KPrC/2kOuv4H35L9zIeW/i/XkPxWj5T+m6+Q/moblP7OR578le+W/1dY0vy9lM7/ATjU/Ajs2P/ecNT/59jY/
oGQ4v4OWNb+/ecm92w+tve7csD0laxk+rrUTPnh28D3oACG+sj6+veNuTz9c+E8/BqxMv1DcTL+r0ku/9UNPv12eUT8fe00/rejG
vn7bzb453Ms+lr++PkQEwz61UcU+kNnGvv0yzL4ag8Y/S5DKP7Kuyr/UyMS/XEnHvximxr8HMMA/V6rNP3355z5l6fQ+y3jzvmD7
075/Ct6+bH/cvqMoyD7Ysv4+sG4pPz9jKj+7xCq/Jd4nv2LCKL8N9yi/UZwjP/qMLD98PtU924TjPb/R273+jf29R3D0vdM67L1C
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
)MNUE";

static int mini_board_constraint(const GlobalBoard &b) {
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
    MN_READY = true;
    return true;
}

static int evaluate_mini(const GlobalBoard &b) {
    if (!MN_READY && !mini_load_packed()) return 0;
    const int stm = b.n_moves & 1;
    const int c = mini_board_constraint(b);
    alignas(32) float x[MN_IN];
    for (int mb = 0; mb < 9; mb++) {
        const int mine = b.mini_boards[mb].markers[stm];
        const int opp = b.mini_boards[mb].markers[stm ^ 1];
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
        int super_cls = 0;
        if (b.mini_board_states[stm] & (1 << mb)) super_cls = 1;
        else if (b.mini_board_states[stm ^ 1] & (1 << mb)) super_cls = 2;
        else if (b.mini_board_states[2] & (1 << mb)) super_cls = 3;
        const int act = (c == mb) ? 1 : 0;
        __m256 v = _mm256_loadu_ps(MN_CENT + (int)MN_CODE[idx] * MN_D);
        v = _mm256_add_ps(v, _mm256_loadu_ps(MN_SUPER + super_cls * MN_D));
        v = _mm256_add_ps(v, _mm256_loadu_ps(MN_LOC + mb * MN_D));
        v = _mm256_add_ps(v, _mm256_loadu_ps(MN_ACTIVE + act * MN_D));
        _mm256_store_ps(x + mb * MN_D, v);
    }
    _mm256_store_ps(x + 9 * MN_D, _mm256_loadu_ps(MN_CONSTR + c * MN_D));
    float out = MN_B2;
    for (int j = 0; j < MN_H; j++) {
        const float *row = MN_W1 + j * MN_IN;
        __m256 vacc = _mm256_setzero_ps();
        for (int k = 0; k < 10; k++) {
            vacc = _mm256_add_ps(_mm256_mul_ps(_mm256_load_ps(x + k * 8),
                                                _mm256_loadu_ps(row + k * 8)), vacc);
        }
        __m128 lo = _mm256_castps256_ps128(vacc);
        __m128 hi = _mm256_extractf128_ps(vacc, 1);
        __m128 s4 = _mm_add_ps(lo, hi);
        s4 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
        s4 = _mm_add_ss(s4, _mm_shuffle_ps(s4, s4, 1));
        float s = MN_B1[j] + _mm_cvtss_f32(s4);
        if (s > 0) out += MN_W2[j] * s;
    }
    return (int)std::lround(out);
}

class CrossfishDev {
       private:
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
        static const int tt_size = 1 << 18;
        std::vector<TTEntry, std::allocator<TTEntry>> transposition_table = std::vector<TTEntry>(tt_size);

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
        static constexpr int PAWN_IDX = 7;
        static constexpr int PAWN = 10;
        static constexpr int ASP_PAWNS = 50;
        static constexpr int RFP_PAWNS = 50;
        static constexpr int FP_PAWNS = 80;
        static constexpr int QDELTA_PAWNS = 400;
        static constexpr int FREE_MOVE_PAWNS = 30;
        static constexpr int MINI_MAX = 8000;

        struct MiniLut {
            int8_t p0_tiar;
            int8_t p1_tiar;
            int8_t p0_center;
            int8_t p1_center;
            int8_t p0_corner;
            int8_t p1_corner;
            int8_t p0_sq;
            int8_t p1_sq;
        };
        static constexpr int MINI_LUT_SIZE = 19683;
        MiniLut mini_lut[MINI_LUT_SIZE];
        uint16_t mini_win_sq[MINI_LUT_SIZE][2];
        uint16_t mini_tiar_sq[MINI_LUT_SIZE][2];
        bool mini_lut_ready = false;

        static int mini_index(int p0, int p1) {
            int idx = 0;
            idx += (p0 & 1) ? 1 : ((p1 & 1) ? 2 : 0);
            idx += (p0 & 2) ? 3 : ((p1 & 2) ? 6 : 0);
            idx += (p0 & 4) ? 9 : ((p1 & 4) ? 18 : 0);
            idx += (p0 & 8) ? 27 : ((p1 & 8) ? 54 : 0);
            idx += (p0 & 16) ? 81 : ((p1 & 16) ? 162 : 0);
            idx += (p0 & 32) ? 243 : ((p1 & 32) ? 486 : 0);
            idx += (p0 & 64) ? 729 : ((p1 & 64) ? 1458 : 0);
            idx += (p0 & 128) ? 2187 : ((p1 & 128) ? 4374 : 0);
            idx += (p0 & 256) ? 6561 : ((p1 & 256) ? 13122 : 0);
            return idx;
        }

        void init_mini_lut() {
            if (mini_lut_ready) return;
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
            const int corners = (1 << 0) + (1 << 2) + (1 << 6) + (1 << 8);
            const int n_pairs = (int)(sizeof(tiar) / sizeof(tiar[0]) / 2);
            auto has_win = [&](int markers) {
                for (int i = 0; i < 8; i++) {
                    if ((markers & win[i]) == win[i]) return true;
                }
                return false;
            };
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
                int occ = p0 | p1;
                uint16_t p0w = 0, p1w = 0, p0t = 0, p1t = 0;
                for (int s = 0; s < 9; s++) {
                    if (occ & (1 << s)) continue;
                    if (has_win(p0 | (1 << s))) p0w = (uint16_t)(p0w | (1 << s));
                    if (has_win(p1 | (1 << s))) p1w = (uint16_t)(p1w | (1 << s));
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
                mini_lut[idx] = e;
            }
            mini_lut_ready = true;
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
        Move getMove(GlobalBoard board, std::chrono::milliseconds thinking_time_passed = std::chrono::milliseconds(95)) {
            init_mini_lut();
            mini_load_packed();
            thinking_time = thinking_time_passed;
            nodes = 0;
            stopped = false;
            root_score = 0;
            Move root_moves[81];
            board.fillLegalMoves(root_moves);
            root_best_move = root_moves[0];

            //clear killers
            killer_moves = std::array<std::array<int, 9>, 128>();
            if (!counters_ready) {
                for (int i = 0; i < 9; i++) {
                    for (int j = 0; j < 9; j++) {
                        counter_move[i][j] = Move{99, 99};
                    }
                }
                counters_ready = true;
            }
            // history_table = std::array<std::array<std::array<int, 9>, 9>, 2>();
            start_time = std::chrono::high_resolution_clock::now();
            depth = 1;
            int alpha = min_val;
            int beta = max_val;
            int aspiration_window = ASP_PAWNS * PAWN;
            int searches = 0;
            int researches = 0;
            while (!time_up()
            && (depth < 50)) {
                int eval = search(board, depth, 0, alpha, beta);
                if (stopped) break;
                if (eval <= alpha ) {
                    //fail low
                    researches++;
                    aspiration_window *= 3;
                    alpha -= aspiration_window;

                }
                else if (eval >= beta) {
                    //fail high
                    researches++;
                    aspiration_window *= 3;
                    beta += aspiration_window;
                }
                else {
                    alpha = eval - aspiration_window;
                    beta = eval + aspiration_window;
                    depth++;
                }
                // depth++;
                searches++;
            }
            // std::cerr << "Depth: " << depth << " Best Move: " << root_best_move.mini_board << " " << root_best_move.square << 
            // " Score: " << root_score << " Nodes: " << nodes << std::endl;
            // std::cerr << "Searches: " << searches << " Researches: " << researches << std::endl;
            return root_best_move;
        }

        int qsearch(GlobalBoard &board, int alpha, int beta, int ply) {
            if (time_up()) {
                return min_val;
            }
            nodes++;
            //should we stop searching?

            int winner = board.checkWinner();
            if (winner != -1){
                if (winner == 2) {
                    return 0;
                }
                else {
                    if (winner == board.n_moves % 2) {
                        return max_val - ply; //current player won
                    }
                    else {
                        return min_val + ply; //previous player won
                    }
                }
            }

            int hce = evaluate_hce(board);
            if (hce >= beta) {
                return beta;
            }
            int stand_pat;
            if (hce + MINI_MAX < alpha) {
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
            if (stand_pat + QDELTA_PAWNS * PAWN < alpha) {
                return alpha;
            }

            //get and sort moves
            Move caps[81];
            int scores[81];
            int n_caps = fill_captures_lut(board, caps);
            get_move_scores(caps, n_caps, {99, 99}, board, ply, scores, true);
            sort_moves(caps, scores, n_caps);
            int val;
            for (int i = 0; i < n_caps; i++) {
                board.makeMove(caps[i]);
                val = -qsearch(board, -beta, -alpha, ply + 1);
                board.unmakeMove();
                alpha = std::max(alpha, val);
                if (alpha >= beta) {
                    break;
                }
            }
            return alpha;



        }

        int search(GlobalBoard &board, int8_t depth, int ply, int alpha, int beta,  bool can_null = true) {
            if (time_up()) {
                return min_val;
            }
            nodes++;
            int winner = board.checkWinner();
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
            TTEntry entry = transposition_table[board.zobrist_hash & (tt_size - 1)];
            bool tt_hit = (entry.zobrist_hash == board.zobrist_hash) && (board.zobrist_hash != 0);
            if (tt_hit && (entry.depth >= depth) && !pv_node) {
                // 0 exact, 1 upper (fail low), 2 lower (fail high)
                if (entry.flag == 0) {
                    return entry.score;
                } else if (entry.flag == 2) {
                    if (entry.score >= beta) return entry.score;
                } else if (entry.flag == 1) {
                    if (entry.score <= alpha) return entry.score;
                }
            }

            if (depth <= 0) {
                return qsearch(board, alpha, beta, ply);
            }
            bool can_futility_prune = false;
            if (!pv_node) {
                int stand_pat = evaluate_hce(board);

                int reverse_futility_margin = RFP_PAWNS * PAWN;
                if (stand_pat - reverse_futility_margin * depth >= beta) {
                    return beta;
                }

                int futility_margin = FP_PAWNS * PAWN;
                can_futility_prune = (stand_pat + futility_margin * depth <= alpha);
            }
            if (pv_node && !tt_hit && depth > 2) {
                search(board, 1, ply, alpha, beta, false);
                if (stopped) return min_val;
                entry = transposition_table[board.zobrist_hash & (tt_size - 1)];
                tt_hit = (entry.zobrist_hash == board.zobrist_hash) && (board.zobrist_hash != 0);
            }

            bool singular = (tt_hit && entry.depth >= depth - 3 && (entry.flag == 2 || entry.flag == 0));
        
            Move legal_moves[81];
            int scores[81];
            int nmoves = board.fillLegalMoves(legal_moves);
            get_move_scores(legal_moves, nmoves, entry.best_move, board, ply, scores, false);
            sort_moves(legal_moves, scores, nmoves);

            Move best_move = legal_moves[0];
            int best_val = min_val;
            int alpha_orig = alpha;
            int val;
            for (int i = 0; i < nmoves; i++) {
                bool capture = is_capture_avx(board, legal_moves[i]);
                if (can_futility_prune && i > 0 && !capture) { //dont search quiet moves in already losing positions
                    continue;
                }
                int extension = 0;
                //one reply extension or singular extension
                if (nmoves==1 || (singular && legal_moves[i].mini_board == entry.best_move.mini_board && legal_moves[i].square == entry.best_move.square)) {
                    extension = 1;
                }

                board.makeMove(legal_moves[i]);
                if (i == 0) {
                    val = -search(board, depth - 1 + extension, ply + 1, -beta, -alpha, can_null);
                }
                else {
                    int reduction = 0;
                    if (scores[i] < 0 || (i >= 3 && !capture)) {
                        reduction = i / 3; //late move reduction
                    }
                    if (reduction > depth - 1) reduction = std::max(0, depth - 1);
                    val = -search(board, depth - 1 - reduction + extension, ply + 1, -alpha - 1, -alpha, can_null);
                    if (val > alpha) {
                        val = -search(board, depth - 1 + extension, ply + 1, -alpha - 1, -alpha, can_null);
                        if (val > alpha && val < beta) {
                            val = -search(board, depth - 1 + extension, ply + 1, -beta, -alpha, can_null);
                        }
                    }
                }
                board.unmakeMove();
                if (stopped) return min_val;
                if (val > best_val) {
                    best_val = val;
                    best_move = legal_moves[i];
                    if (ply == 0 && abs(best_val) != abs(min_val)) {
                        root_best_move = best_move;
                        root_score = best_val;
                    }
                }
                alpha = std::max(alpha, best_val);
                if (alpha >= beta) {
                    killer_moves[ply][legal_moves[i].square] = 1;
                    int &h = history_table[board.n_moves % 2][legal_moves[i].mini_board][legal_moves[i].square];
                    int bonus = depth * depth;
                    h += bonus - h * bonus / 10000;
                    int stm = board.n_moves % 2;
                    for (int j = 0; j < i; j++) {
                        if (is_capture_avx(board, legal_moves[j])) continue;
                        int &hj = history_table[stm][legal_moves[j].mini_board][legal_moves[j].square];
                        hj -= bonus;
                        if (hj < 0) hj = 0;
                    }
                    if (board.n_moves > 0) {
                        Move prev = board.move_history.top();
                        counter_move[prev.mini_board][prev.square] = legal_moves[i];
                    }
                    break;
                }
            }
            if (!stopped) {
                int8_t flag = 0;
                if (best_val <= alpha_orig) {
                    flag = 1;
                }
                else if (best_val >= beta) {
                    flag = 2;
                }
                TTEntry new_entry = {depth, flag, best_val, board.zobrist_hash, best_move};
                transposition_table[board.zobrist_hash & (tt_size - 1)] = new_entry;
            }

            return best_val;

        }

        void sort_moves(Move* moves, int* scores, int n) {
            for (int i = 1; i < n; i++) {
                int key = scores[i];
                Move key_move = moves[i];
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

        // bool miniboard_is_winnable(GlobalBoard &board, int mb, int player) {
            
        //     //check if any of the win masks satisfy
        //     // (other player's markers) & (win mask) == 0
        //     bool result = false;
        //     int opp_markers = board.mini_boards[mb].markers[(player + 1) % 2];
        //     for (int i = 0; i < board.win_masks.size(); i++) {
        //         result = result || ((opp_markers & board.win_masks[i]) == 0);
        //     }
        //     return result;

        // }

        int fill_captures_lut(GlobalBoard &board, Move* dst) {
            int n = 0;
            if (board.n_moves == 0) return 0;
            int active_square = board.move_history.top().square;
            int out_of_play = board.mini_board_states[0] | board.mini_board_states[1] | board.mini_board_states[2];
            int stm = board.n_moves % 2;
            auto add_from_mb = [&](int mb) {
                int idx = mini_index(board.mini_boards[mb].markers[0], board.mini_boards[mb].markers[1]);
                int wins = mini_win_sq[idx][stm];
                while (wins) {
                    int s = __builtin_ctz(wins);
                    wins &= wins - 1;
                    dst[n++] = Move{(int8_t)mb, (int8_t)s};
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

        bool is_capture_avx(GlobalBoard &board, Move &move) {
            int idx = mini_index(
                board.mini_boards[move.mini_board].markers[0],
                board.mini_boards[move.mini_board].markers[1]);
            int stm = board.n_moves % 2;
            return (mini_win_sq[idx][stm] & (1 << move.square)) != 0;
        }

        bool is_block_avx(GlobalBoard &board, Move &move) {
            int idx = mini_index(
                board.mini_boards[move.mini_board].markers[0],
                board.mini_boards[move.mini_board].markers[1]);
            int opp = (board.n_moves + 1) % 2;
            return (mini_win_sq[idx][opp] & (1 << move.square)) != 0;
        }

        bool creates_two_in_a_row(GlobalBoard &board, Move &move) {
            int idx = mini_index(
                board.mini_boards[move.mini_board].markers[0],
                board.mini_boards[move.mini_board].markers[1]);
            int stm = board.n_moves % 2;
            return (mini_tiar_sq[idx][stm] & (1 << move.square)) != 0;
        }

        void get_move_scores(Move* moves, int n, Move tt_move, GlobalBoard &board, int &ply, int* scores, bool qs = false) {
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
            for (int i = 0; i < n; i++) {
                int mb = moves[i].mini_board;
                int sq = moves[i].square;
                if (mb == tt_move.mini_board && sq == tt_move.square) {
                    scores[i] = 1000;
                    continue;
                }
                if (mb != last_mb) {
                    last_mb = mb;
                    last_idx = mini_index(board.mini_boards[mb].markers[0], board.mini_boards[mb].markers[1]);
                }
                int move_score = 0;
                if (killer_moves[ply][sq] == 1) {
                    move_score += 25;
                }
                if (cm.mini_board == mb && cm.square == sq) {
                    move_score += 40;
                }
                if (!qs && (mini_win_sq[last_idx][stm] & (1 << sq))) {
                    move_score += 100;
                }
                if (mini_win_sq[last_idx][stm ^ 1] & (1 << sq)) {
                    move_score += 75;
                }
                if (mini_tiar_sq[last_idx][stm] & (1 << sq)) {
                    move_score += 50;
                }
                if ((out_of_play & (1 << sq)) != 0) {
                    move_score -= 250;
                }
                move_score += history_table[stm][mb][sq] / 20;
                scores[i] = move_score;
            }
        }

        int evaluate_hce(GlobalBoard &board) {
            init_mini_lut();
            /*use bitscan to count number of won miniboards for both players*/
            int p0_miniboards_held = __builtin_popcount(board.mini_board_states[0]);
            int p1_miniboards_held = __builtin_popcount(board.mini_board_states[1]);
            int out_of_play = board.mini_board_states[0] | board.mini_board_states[1] | board.mini_board_states[2]; 
            //count two in a rows for both players
            int p0_two_in_a_row = 0;
            int p1_two_in_a_row = 0;
            //square counts
            int p0_center_squares_held = 0;
            int p1_center_squares_held = 0;
            int p0_corner_squares_held = 0;
            int p1_corner_squares_held = 0;
            int p0_squares_held = 0;
            int p1_squares_held = 0;
            //idea, keep a map of two in a rows. Two in a rows that form two in a rows with other two in a rows are worth more
            int p0_two_in_a_row_map = 0;
            int p1_two_in_a_row_map = 0;
            //corner mask
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


            //also check for 2 in a rows in the out of play miniboards
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
                //check for global two in a rows
                p0_global_two_in_a_row += ((__builtin_popcount(p0_miniboards & two_in_a_row_masks[i * 2]) - __builtin_popcount(p1_miniboards & two_in_a_row_masks[i * 2 + 1])) /2);
                p1_global_two_in_a_row += ((__builtin_popcount(p1_miniboards & two_in_a_row_masks[i * 2]) - __builtin_popcount(p0_miniboards & two_in_a_row_masks[i * 2 + 1])) /2);
                //check for two in a rows that line up
                p0_two_in_a_rows_lined_up += ((__builtin_popcount((p0_two_in_a_row_map | p0_miniboards) & two_in_a_row_masks[i * 2]) - __builtin_popcount(p1_miniboards & two_in_a_row_masks[i * 2 + 1]))  / 2);
                p1_two_in_a_rows_lined_up += ((__builtin_popcount((p1_two_in_a_row_map | p1_miniboards) & two_in_a_row_masks[i * 2]) - __builtin_popcount(p0_miniboards & two_in_a_row_masks[i * 2 + 1]))   / 2);
            }
            // Texel-tuned (per-weight Adam). SPRT vs round numbers at 20ms:
            // N: 2688 W: 1148 D: 584 L: 956 Elo +24.86 LLR 3.04
            // Free-move +300. SPRT 20ms: N: 4640 W: 1929 D: 993 L: 1718 Elo +15.81 LLR 3.13
            int val = (p0_miniboards_held - p1_miniboards_held) * 2410;
            val += (p0_center_miniboard_held - p1_center_miniboard_held) * 836;
            val += (p0_corner_miniboards_held - p1_corner_miniboards_held) * 464;
            val += (p0_global_two_in_a_row - p1_global_two_in_a_row) * 1316;
            val += (p0_two_in_a_row - p1_two_in_a_row) * 534;
            val += (p0_two_in_a_rows_lined_up - p1_two_in_a_rows_lined_up) * 424;
            val += (p0_center_squares_held - p1_center_squares_held) * 33;
            val += (p0_corner_squares_held - p1_corner_squares_held) * PAWN;
            val += (p0_squares_held - p1_squares_held)* 33;

            int stm_sign = (board.n_moves % 2 == 0) ? 1 : -1;
            val += stm_sign * 112;
            int score = stm_sign * val;
            if (board.n_moves > 0) {
                int out_of_play = board.mini_board_states[0] | board.mini_board_states[1] | board.mini_board_states[2];
                if (board.prev_move_was_pass || ((out_of_play & (1 << board.move_history.top().square)) != 0)) {
                    score += FREE_MOVE_PAWNS * PAWN;
                }
            }
            return score;

        }

        int evaluate(GlobalBoard &board) {
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