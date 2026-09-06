#include "mini_eval.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <mutex>
#include <vector>

// Landed engine: HCE at RFP/futility, HCE + packed MiniNet residual at qsearch
// leaves. Free-move +300. Failed SEARCH/EVAL/NNUE experiment stubs removed.
#ifndef CROSSFISH_TTFLAG
#define CROSSFISH_TTFLAG
enum TTFlag { TT_EXACT = 0, TT_UPPER = 1, TT_LOWER = 2 };
#endif

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
        // Correction history: [stm][constrained miniboard, 9 = free choice][decided-miniboard mask].
        // A small exact index beats hashing the zobrist here: no collisions, no mixing
        // in the hot path, and it separates the two structural facts the fixed eval
        // misprices most - game phase and being locked into one miniboard.
        static constexpr int CORR_MB = 10;
        static constexpr int CORR_MASKS = 512;
        // Stored units are CORR_GRAIN x eval units. The gravity term bounds |entry| at
        // CORR_SCALE, so the applied shift never exceeds CORR_SCALE / CORR_GRAIN = 512.
        static constexpr int CORR_SCALE = 16384;
        static constexpr int CORR_GRAIN = 32;
        static constexpr int CORR_MAX = 16384;
        // CORR_DIFF_MAX * CORR_MAX_WEIGHT <= CORR_SCALE keeps the update a contraction.
        static constexpr int CORR_DIFF_MAX = 1024;
        static constexpr int CORR_MAX_WEIGHT = 8;
        // Mate scores are ±(max_val - ply), i.e. distances to mate rather than
        // quantities the static eval can be measured against.
        static constexpr int CORR_MATE_BOUND = 90000;
        std::array<std::array<std::array<int, CORR_MASKS>, CORR_MB>, 2> corr_hist{};
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
        static constexpr int ASP_PAWNS = 50;
        static constexpr int RFP_PAWNS = 50;
        static constexpr int FP_PAWNS = 80;
        static constexpr int QDELTA_PAWNS = 400;
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
        int &corr_entry(GlobalBoard &board) {
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
        int corrected_eval(GlobalBoard &board, int static_eval) {
            int v = static_eval + corr_entry(board) / CORR_GRAIN;
            if (v > CORR_EVAL_LIMIT) v = CORR_EVAL_LIMIT;
            if (v < -CORR_EVAL_LIMIT) v = -CORR_EVAL_LIMIT;
            return v;
        }

        void update_corr_hist(GlobalBoard &board, int diff, int d) {
            if (diff > CORR_DIFF_MAX) diff = CORR_DIFF_MAX;
            if (diff < -CORR_DIFF_MAX) diff = -CORR_DIFF_MAX;
            int w = std::min(d, CORR_MAX_WEIGHT);
            int &e = corr_entry(board);
            e += diff * w - e * abs(diff) * w / CORR_SCALE;
            if (e > CORR_MAX) e = CORR_MAX;
            if (e < -CORR_MAX) e = -CORR_MAX;
        }

        int check_winner_fast(GlobalBoard &board) {
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

        int fill_legal_moves_fast(GlobalBoard &board, Move *dst) {
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

        void make_move_fast(GlobalBoard &board, const Move &move) {
            int stm = board.n_moves & 1;
            int bit = 1 << move.square;
            int mb_bit = 1 << move.mini_board;
            int before = board.mini_boards[move.mini_board].markers[stm];
            if (board.n_moves > 0) {
                board.zobrist_hash ^= board.legal_mini_board_hashes[board.move_history.top().square];
            }
            board.move_history.push(move);
            board.mini_boards[move.mini_board].markers[stm] = before | bit;
            board.zobrist_hash ^= board.move_hashes[stm][move.mini_board][move.square];
            board.zobrist_hash ^= board.legal_mini_board_hashes[move.square];
            if (fast_win_moves[before] & bit) {
                board.mini_board_states[stm] |= mb_bit;
                board.zobrist_hash ^= board.mini_board_hashes[stm][move.mini_board];
            } else {
                int occupied = board.mini_boards[move.mini_board].markers[0]
                             | board.mini_boards[move.mini_board].markers[1];
                if (occupied == 511) {
                    board.mini_board_states[2] |= mb_bit;
                    board.zobrist_hash ^= board.mini_board_hashes[2][move.mini_board];
                }
            }
            board.zobrist_hash ^= board.player_to_move_hash;
            board.n_moves++;
        }

        void unmake_move_fast(GlobalBoard &board) {
            board.n_moves--;
            board.zobrist_hash ^= board.player_to_move_hash;
            Move move = board.move_history.top();
            board.move_history.pop();
            int mb_bit = 1 << move.mini_board;
            if (board.mini_board_states[0] & mb_bit) {
                board.mini_board_states[0] &= ~mb_bit;
                board.zobrist_hash ^= board.mini_board_hashes[0][move.mini_board];
            } else if (board.mini_board_states[1] & mb_bit) {
                board.mini_board_states[1] &= ~mb_bit;
                board.zobrist_hash ^= board.mini_board_hashes[1][move.mini_board];
            } else if (board.mini_board_states[2] & mb_bit) {
                board.mini_board_states[2] &= ~mb_bit;
                board.zobrist_hash ^= board.mini_board_hashes[2][move.mini_board];
            }
            int stm = board.n_moves & 1;
            board.mini_boards[move.mini_board].markers[stm] &= ~(1 << move.square);
            board.zobrist_hash ^= board.move_hashes[stm][move.mini_board][move.square];
            board.zobrist_hash ^= board.legal_mini_board_hashes[move.square];
            if (board.n_moves > 0) {
                board.zobrist_hash ^= board.legal_mini_board_hashes[board.move_history.top().square];
            }
        }

        Move getMove(GlobalBoard board, std::chrono::milliseconds thinking_time_passed = std::chrono::milliseconds(95)) {
            init_mini_lut();
            init_lmr_table();
            thinking_time = thinking_time_passed;
            nodes = 0;
            stopped = false;
            root_score = 0;
            Move root_moves[81];
            fill_legal_moves_fast(board, root_moves);
            root_best_move = root_moves[0];
            killer_moves = std::array<std::array<int, 9>, 128>();
            corr_hist = {};
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
        bool search_fixed_depth(GlobalBoard &board, int d, int &out_score) {
            init_mini_lut();
            init_lmr_table();
            thinking_time = std::chrono::milliseconds(24 * 60 * 60 * 1000);
            nodes = 0;
            stopped = false;
            root_score = 0;
            depth = d;
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

        int qsearch(GlobalBoard &board, int alpha, int beta, int ply) {
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

            int hce = corrected_eval(board, evaluate_hce(board));
            if (hce >= beta) {
                return beta;
            }
            int stand_pat;
            if (hce + MINI_MAX < alpha) {
                // MiniNet is clamped near ±2000; it cannot raise alpha or fail high.
                stand_pat = hce + MINI_MAX;
            } else {
                stand_pat = hce + evaluate_mini_fast(board);
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

            Move caps[81];
            int scores[81];
            int n_caps = fill_captures_lut(board, caps);
            get_move_scores(caps, n_caps, {99, 99}, board, ply, scores, true);
            sort_moves(caps, scores, n_caps);
            int val;
            for (int i = 0; i < n_caps; i++) {
                make_move_fast(board, caps[i]);
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

        int search(GlobalBoard &board, int depth, int ply, int alpha, int beta) {
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
            TTEntry entry = transposition_table[board.zobrist_hash & (tt_size - 1)];
            bool tt_hit = (entry.zobrist_hash == board.zobrist_hash) && (board.zobrist_hash != 0);
            if (tt_hit && (entry.depth >= depth) && !pv_node) {
                // Flags match the original store: 0 exact, 1 upper (fail low), 2 lower (fail high).
                if (entry.flag == TT_EXACT) {
                    return entry.score;
                }
                else if (entry.flag == TT_LOWER) {
                    if (entry.score >= beta) return entry.score;
                }
                else if (entry.flag == TT_UPPER) {
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
                static_eval = corrected_eval(board, evaluate_hce(board));
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
                entry = transposition_table[board.zobrist_hash & (tt_size - 1)];
                tt_hit = (entry.zobrist_hash == board.zobrist_hash) && (board.zobrist_hash != 0);
            }

            bool singular = (tt_hit && entry.depth >= depth - 3 && (entry.flag == TT_LOWER || entry.flag == TT_EXACT));

            Move legal_moves[81];
            int scores[81];
            int nmoves = fill_legal_moves_fast(board, legal_moves);
            get_move_scores(legal_moves, nmoves, tt_hit ? entry.best_move : Move{99, 99}, board, ply, scores, false);
            sort_moves(legal_moves, scores, nmoves);

            Move best_move = legal_moves[0];
            int best_val = min_val;
            int alpha_orig = alpha;
            int val;
            for (int i = 0; i < nmoves; i++) {
                bool capture = is_capture_avx(board, legal_moves[i]);
                if (can_futility_prune && i > 0 && !capture) {
                    continue;
                }
                int extension = 0;
                if (nmoves==1 || (singular && legal_moves[i].mini_board == entry.best_move.mini_board && legal_moves[i].square == entry.best_move.square)) {
                    extension = 1;
                }

                make_move_fast(board, legal_moves[i]);
                if (i == 0) {
                    val = -search(board, depth - 1 + extension, ply + 1, -beta, -alpha);
                }
                else {
                    int reduction = 0;
                    bool do_lmr = (scores[i] < 0 || (i >= 3 && !capture));
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
                int flag = TT_EXACT;
                if (best_val <= alpha_orig) {
                    flag = TT_UPPER;
                }
                else if (best_val >= beta) {
                    flag = TT_LOWER;
                }
                TTEntry new_entry = {depth, best_val, flag, board.zobrist_hash, best_move};
                transposition_table[board.zobrist_hash & (tt_size - 1)] = new_entry;
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

        int fill_captures_lut(GlobalBoard &board, Move* dst) {
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

        bool is_capture_avx(GlobalBoard &board, Move &move) {
            int stm = board.n_moves % 2;
            int mine = board.mini_boards[move.mini_board].markers[stm];
            return (fast_win_moves[mine] & (1 << move.square)) != 0;
        }

        bool is_block_avx(GlobalBoard &board, Move &move) {
            int opp = (board.n_moves + 1) % 2;
            int theirs = board.mini_boards[move.mini_board].markers[opp];
            return (fast_win_moves[theirs] & (1 << move.square)) != 0;
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
                if (!qs && (fast_win_moves[board.mini_boards[mb].markers[stm]] & (1 << sq))) {
                    move_score += 100;
                }
                if (fast_win_moves[board.mini_boards[mb].markers[stm ^ 1]] & (1 << sq)) {
                    move_score += 75;
                }
                if (mini_tiar_sq[last_idx][stm] & (1 << sq)) {
                    move_score += 50;
                }
                if ((out_of_play & (1 << sq)) != 0) {
                    move_score -= 250;
                }
                int hs = history_table[stm][mb][sq] / 20;
                move_score += hs;
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
        int eval_extra(GlobalBoard &board) {
            if (board.n_moves > 0) {
                int out_of_play = board.mini_board_states[0] | board.mini_board_states[1] | board.mini_board_states[2];
                if (board.prev_move_was_pass || ((out_of_play & (1 << board.move_history.top().square)) != 0)) {
                    return FREE_MOVE_PAWNS * eval_weights[PAWN_IDX];
                }
            }
            return 0;
        }

        int evaluate_hce(GlobalBoard &board) {
            int stm_sign = (board.n_moves % 2 == 0) ? 1 : -1;
            int p0_miniboards = board.mini_board_states[0];
            int p1_miniboards = board.mini_board_states[1];
            int out_of_play = p0_miniboards | p1_miniboards | board.mini_board_states[2];
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

        int evaluate(GlobalBoard &board) {
            if (g_force_hce_eval) {
                return evaluate_hce(board);
            }
            return evaluate_hce(board) + evaluate_mini_avx(board);
        }

};
