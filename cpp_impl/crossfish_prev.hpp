// Frozen alloc-free baseline (5840146) plus LMR fail-high re-search.
#ifndef CROSSFISH_TTFLAG
#define CROSSFISH_TTFLAG
enum TTFlag { TT_EXACT = 0, TT_UPPER = 1, TT_LOWER = 2 };
#endif

class CrossfishPrev {
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

        int evaluate(GlobalBoard &board) {
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

            int p0_tiar_temp;
            int p1_tiar_temp;
            int p0_markers;
            int p1_markers;
            for (int miniboard = 0; miniboard < 9; miniboard++) {
                if ((out_of_play & (1 << miniboard)) != 0) {
                    continue;
                }
                p0_markers = board.mini_boards[miniboard].markers[0];
                p1_markers = board.mini_boards[miniboard].markers[1];

                for (int i = 0; i < (int)two_in_a_row_masks.size() / 2; i++) {
                    p0_tiar_temp = ((__builtin_popcount(p0_markers & two_in_a_row_masks[i * 2]) - __builtin_popcount(p1_markers & two_in_a_row_masks[i * 2 + 1])) /2);
                    p1_tiar_temp = ((__builtin_popcount(p1_markers & two_in_a_row_masks[i * 2]) - __builtin_popcount(p0_markers & two_in_a_row_masks[i * 2 + 1])) /2);
                    p0_two_in_a_row += p0_tiar_temp;
                    p1_two_in_a_row += p1_tiar_temp;

                    p0_two_in_a_row_map |= ((1 << miniboard) * p0_tiar_temp);
                    p1_two_in_a_row_map |= ((1 << miniboard) * p1_tiar_temp);
                }

                p0_center_squares_held += __builtin_popcount(p0_markers & (1 << 4));
                p1_center_squares_held += __builtin_popcount(p1_markers & (1 << 4));
                p0_corner_squares_held += __builtin_popcount(p0_markers & corners);
                p1_corner_squares_held += __builtin_popcount(p1_markers & corners);
                p0_squares_held += __builtin_popcount(p0_markers);
                p1_squares_held += __builtin_popcount(p1_markers);
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
            int val = (p0_miniboards_held - p1_miniboards_held) * 2000;
            val += (p0_center_miniboard_held - p1_center_miniboard_held) * 1000;
            val += (p0_corner_miniboards_held - p1_corner_miniboards_held) * 500;
            val += (p0_global_two_in_a_row - p1_global_two_in_a_row) * 1500;
            val += (p0_two_in_a_row - p1_two_in_a_row) * 500;
            val += (p0_two_in_a_rows_lined_up - p1_two_in_a_rows_lined_up) * 500;
            val += (p0_center_squares_held - p1_center_squares_held) * 20;
            val += (p0_corner_squares_held - p1_corner_squares_held) * 10;
            val += (p0_squares_held - p1_squares_held)* 20;

            int stm_sign = (board.n_moves % 2 == 0) ? 1 : -1;
            val += stm_sign * 50;
            return stm_sign * val;
        }

};
