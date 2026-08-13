// Free-move +300 in baseline. Experiment extras on top.
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
        static const int tt_size = 1 << 18;
        std::vector<TTEntry, std::allocator<TTEntry>> transposition_table = std::vector<TTEntry>(tt_size);

        std::vector<int> two_in_a_row_masks = {
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
        static inline std::once_flag mini_lut_once;
        static constexpr int WIN_IN_ONE_WEIGHT = 300;
        static constexpr int LUT_W_TIAR = 534;
        static constexpr int LUT_W_CENTER_SQ = 33;
        static constexpr int LUT_W_CORNER_SQ = 10;
        static constexpr int LUT_W_SQUARES = 33;
        // 0=free-move baseline, 2=off-board forks, 3=live-third global threats
        static constexpr int EVAL_EXPERIMENT = 0;
        static constexpr int W_FREE_MOVE = 300;
        static constexpr int W_FORK = 400;

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
                    for (int s = 0; s < 9; s++) {
                        if (occ & (1 << s)) continue;
                        if (has_win(p0 | (1 << s))) e.p0_win1 = 1;
                        if (has_win(p1 | (1 << s))) e.p1_win1 = 1;
                    }
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

        Move getMove(GlobalBoard board, std::chrono::milliseconds thinking_time_passed = std::chrono::milliseconds(95)) {
            init_mini_lut();
            thinking_time = thinking_time_passed;
            nodes = 0;
            stopped = false;
            root_score = 0;
            Move root_moves[81];
            board.fillLegalMoves(root_moves);
            root_best_move = root_moves[0];
            killer_moves = std::array<std::array<int, 9>, 128>();
            start_time = std::chrono::high_resolution_clock::now();
            depth = 1;
            int alpha = min_val;
            int beta = max_val;
            int aspiration_window = 500;
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

        int qsearch(GlobalBoard &board, int alpha, int beta, int ply) {
            if (time_up()) return min_val;
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

            int stand_pat = evaluate(board);
            if (stand_pat >= beta) {
                return beta;
            }
            if (alpha < stand_pat) {
                alpha = stand_pat;
            }

            Move caps[81];
            int scores[81];
            int n_caps = board.fillCaptures(caps);
            get_move_scores(caps, n_caps, {99, 99}, board, ply, scores);
            sort_moves(caps, scores, n_caps);
            int val;
            for (int i = 0; i < n_caps; i++) {
                board.makeMove(caps[i]);
                val = -qsearch(board, -beta, -alpha, ply + 1);
                board.unmakeMove();
                if (stopped) return min_val;
                alpha = std::max(alpha, val);
                if (alpha >= beta) {
                    break;
                }
            }
            return alpha;
        }

        int search(GlobalBoard &board, int depth, int ply, int alpha, int beta,  bool can_null = true) {
            if (time_up()) return min_val;
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
            if (!pv_node) {
                int stand_pat = evaluate(board);

                int reverse_futility_margin = 650;
                if (stand_pat - reverse_futility_margin * depth >= beta) {
                    return beta;
                }

                int futility_margin = 800;
                can_futility_prune = (stand_pat + futility_margin * depth <= alpha);
            }
            if (pv_node && !tt_hit && depth > 2) {
                search(board, 1, ply, alpha, beta, false);
                if (stopped) return min_val;
                entry = transposition_table[board.zobrist_hash & (tt_size - 1)];
                tt_hit = (entry.zobrist_hash == board.zobrist_hash) && (board.zobrist_hash != 0);
            }

            bool singular = (tt_hit && entry.depth >= depth - 3 && (entry.flag == TT_LOWER || entry.flag == TT_EXACT));

            Move legal_moves[81];
            int scores[81];
            int nmoves = board.fillLegalMoves(legal_moves);
            get_move_scores(legal_moves, nmoves, tt_hit ? entry.best_move : Move{99, 99}, board, ply, scores);
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

                board.makeMove(legal_moves[i]);
                if (i == 0) {
                    val = -search(board, depth - 1 + extension, ply + 1, -beta, -alpha, can_null);
                }
                else {
                    int reduction = 0;
                    if (scores[i] < 0) {
                        reduction = i/4;
                    }
                    if (reduction > depth - 1) reduction = std::max(0, depth - 1);
                    val = -search(board, depth - 1 - reduction + extension, ply + 1, -alpha - 1, -alpha, can_null);
                    // Reduced searches are not allowed to fail high unchallenged.
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
                    h += depth * depth;
                    if (h > 10000) h = 10000;
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

        bool is_capture_avx(GlobalBoard &board, Move &move) {
            int miniboard_markers = board.mini_boards[move.mini_board].markers[board.n_moves % 2];
            miniboard_markers |= (1 << move.square);
            __m256i markers_vec = _mm256_set1_epi32(miniboard_markers);
            __m256i win_masks_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(board.win_masks.data()));
            __m256i result_vec = _mm256_and_si256(markers_vec, win_masks_vec);
            result_vec = _mm256_cmpeq_epi32(result_vec, win_masks_vec);
            int mask = _mm256_movemask_ps(_mm256_castsi256_ps(result_vec));
            return mask != 0;
        }

        bool is_block_avx(GlobalBoard &board, Move &move) {
            int opp_markers = board.mini_boards[move.mini_board].markers[(board.n_moves + 1) % 2];
            opp_markers |= (1 << move.square);
            __m256i markers_vec = _mm256_set1_epi32(opp_markers);
            __m256i win_masks_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(board.win_masks.data()));
            __m256i result_vec = _mm256_and_si256(markers_vec, win_masks_vec);
            result_vec = _mm256_cmpeq_epi32(result_vec, win_masks_vec);
            int mask = _mm256_movemask_ps(_mm256_castsi256_ps(result_vec));
            return mask != 0;
        }

        bool creates_two_in_a_row(GlobalBoard &board, Move &move) {
            int our_markers = board.mini_boards[move.mini_board].markers[board.n_moves % 2];
            int opp_markers = board.mini_boards[move.mini_board].markers[(board.n_moves + 1) % 2];
            bool result = false;
            for (int mask = 0; mask < (int)two_in_a_row_masks.size() / 2; mask++) {
                result = result || ((((our_markers | (1 << move.square)) & two_in_a_row_masks[mask * 2]) == two_in_a_row_masks[mask * 2]) &&
                (((opp_markers | our_markers) & two_in_a_row_masks[mask * 2 + 1]) == 0));
            }
            return result;
        }

        void get_move_scores(Move* moves, int n, Move tt_move, GlobalBoard &board, int &ply, int* scores) {
            for (int i = 0; i < n; i++) {
                int move_score = 0;
                if (moves[i].mini_board == tt_move.mini_board && moves[i].square == tt_move.square) {
                    move_score += 1000;
                    scores[i] = move_score;
                    continue;
                }

                if (killer_moves[ply][moves[i].square] == 1) {
                    move_score += 25;
                }

                if (is_capture_avx(board, moves[i])) {
                    move_score += 100;
                }

                if (is_block_avx(board, moves[i])) {
                    move_score += 75;
                }

                if (creates_two_in_a_row(board, moves[i])) {
                    move_score += 50;
                }

                int out_of_play = board.mini_board_states[0] | board.mini_board_states[1] | board.mini_board_states[2];
                if ((out_of_play & (1 << moves[i].square)) != 0) {
                    move_score -= 250;
                }
                move_score += history_table[board.n_moves % 2][moves[i].mini_board][moves[i].square] / 20;
                scores[i] = move_score;
            }
        }

        static constexpr int N_EVAL_WEIGHTS = 10;
        int eval_weights[N_EVAL_WEIGHTS] = {2410, 836, 464, 1316, 534, 424, 33, 10, 33, 112};

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
            for(int i = 0; i < (int)two_in_a_row_masks.size() / 2; i++) {
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
            for(int i = 0; i < (int)two_in_a_row_masks.size() / 2; i++) {
                int third = two_in_a_row_masks[i * 2 + 1];
                if constexpr (EVAL_EXPERIMENT == 3) {
                    if ((board.mini_board_states[2] & third) != 0) {
                        continue;
                    }
                }
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
            int extra = 0;
            if (board.n_moves > 0) {
                int out_of_play = board.mini_board_states[0] | board.mini_board_states[1] | board.mini_board_states[2];
                if (board.prev_move_was_pass || ((out_of_play & (1 << board.move_history.top().square)) != 0)) {
                    extra += W_FREE_MOVE;
                }
            }
            if constexpr (EVAL_EXPERIMENT == 2) {
                init_mini_lut();
                int out_of_play = board.mini_board_states[0] | board.mini_board_states[1] | board.mini_board_states[2];
                int active = -1;
                if (board.n_moves > 0 && !board.prev_move_was_pass) {
                    int sent = board.move_history.top().square;
                    if ((out_of_play & (1 << sent)) == 0) {
                        active = sent;
                    }
                }
                int p0_w1 = 0;
                int p1_w1 = 0;
                for (int mb = 0; mb < 9; mb++) {
                    if ((out_of_play & (1 << mb)) != 0) continue;
                    if (mb == active) continue;
                    const MiniLut &e = mini_lut[mini_index(
                        board.mini_boards[mb].markers[0],
                        board.mini_boards[mb].markers[1])];
                    p0_w1 += e.p0_win1;
                    p1_w1 += e.p1_win1;
                }
                int p0_fork = (p0_w1 > 1) ? (p0_w1 - 1) : 0;
                int p1_fork = (p1_w1 > 1) ? (p1_w1 - 1) : 0;
                int stm_sign = (board.n_moves % 2 == 0) ? 1 : -1;
                extra += stm_sign * W_FORK * (p0_fork - p1_fork);
            }
            return extra;
        }

        int evaluate(GlobalBoard &board) {
            int16_t idx[9];
            int n = 0;
            int base = 0;
            eval_parts(board, idx, n, base);
            int stm_sign = (board.n_moves % 2 == 0) ? 1 : -1;
            int local = 0;
            for (int i = 0; i < n; i++) {
                local += mini_score[idx[i]];
            }
            return base + stm_sign * local + eval_extra(board);
        }

};
