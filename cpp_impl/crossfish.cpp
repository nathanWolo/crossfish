#include <iostream>
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
#include <array>
#pragma GCC optimize("O3")
#pragma GCC optimization("Ofast,unroll-loops")
#pragma GCC target("avx2,bmi,bmi2,lzcnt,popcnt")
/*
This is a port of crossfish.py to C++.
Crossfish was my entry to the UVic AI Ultimate Tic Tac Toe competition in 2024, where it won first place.    
*/

//a struct representing a 3x3 board with 16 bit integers
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
        void makeMove(Move move) {
            //make sure move is legal
            // int occupied = mini_boards[move.mini_board].markers[0] | mini_boards[move.mini_board].markers[1];
            // int out_of_play = mini_board_states[0] | mini_board_states[1] | mini_board_states[2];
            // if (((occupied & (1 << move.square)) != 0)
            // || ((out_of_play & (1 << move.mini_board)) != 0)
            // || move.mini_board > 8 || move.square > 8 || move.mini_board < 0 || move.square < 0)
            // {
            //     std::cerr << "ILLEGAL MOVE MADE: " << move.mini_board << " " << move.square << std::endl;
            //     std::cerr << "Last move: " << move_history.top().mini_board << " " << move_history.top().square << std::endl;
            //     print_board();
            //     std::exit(EXIT_FAILURE); // Terminate the program
            // }
            // if (n_moves > 0) {
            //     Move prevMove = move_history.top();

            //     if (n_moves > 0 && ((out_of_play & (1 << prevMove.square)) == 0) && (move.mini_board != prevMove.square)) //we were not sent to a won or drawn board
            //         {
            //             std::cerr << "ILLEGAL MOVE MADE: " << move.mini_board << " " << move.square << std::endl;
            //             std::cerr << "Last move: " << move_history.top().mini_board << " " << move_history.top().square << std::endl;
            //             print_board();
            //             std::exit(EXIT_FAILURE); // Terminate the program
                    
            //         }
            // }

            move_history.push(move); //add the move to the list of moves
            mini_boards[move.mini_board].markers[n_moves % 2] |= (1 << move.square); //set the bit at the square to 1
            mini_boards[move.mini_board].markers[n_moves % 2] &= miniboard_mask; //make sure that only the last 9 bits are in use
            zobrist_hash ^= move_hashes[n_moves % 2][move.mini_board][move.square];
            zobrist_hash ^= legal_mini_board_hashes[move.square];

            //check if the miniboard is won
            for (int i = 0; i < win_masks.size(); i++) {
                if ((mini_boards[move.mini_board].markers[n_moves % 2] & win_masks[i]) == win_masks[i]) {
                    mini_board_states[n_moves % 2] |= (1 << move.mini_board);
                    zobrist_hash ^= mini_board_hashes[n_moves % 2][move.mini_board];
                    break;
                }
            }
            //check if the mini board is drawn
            if (((mini_boards[move.mini_board].markers[0] | mini_boards[move.mini_board].markers[1]) & miniboard_mask) == miniboard_mask) {
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
        }

        int checkWinner() {
            for (int i = 0; i < 8; i++) {
                if ((mini_board_states[0] & win_masks[i]) == win_masks[i]) {
                    return 0;
                }
                if ((mini_board_states[1] & win_masks[i]) == win_masks[i]) {
                    return 1;
                }
            }
            if ((mini_board_states[0] | mini_board_states[1] | mini_board_states[2]) == miniboard_mask) {
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

        std::vector<Move> getLegalMoves() {
            std::vector<Move> legal_moves;
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
                //check if we were sent to a won or drawn board
                if ((out_of_play & (1 << active_square)) != 0) {
                    // we were
                    for (int i = 0; i < 9; i++) {
                        if ((out_of_play & (1 << i)) == 0) //check if board i is not out of play
                        {
                            for (int j = 0; j < 9; j++) {
                                int marked = mini_boards[i].markers[0] | mini_boards[i].markers[1];
                                if ((marked & (1 << j)) == 0) 
                                { //if the square is not taken
                                    Move move = {i, j};
                                    legal_moves.push_back(move);
                                }
                            }
                        }
                    }
                } else {
                    int marked = mini_boards[active_square].markers[0] | mini_boards[active_square].markers[1];
                    for (int i = 0; i < 9; i++) {
                        if ((marked & (1 << i)) == 0 ) //if the square is not taken
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

                // Copy constructor
    GlobalBoard(const GlobalBoard& other)
        : mini_boards(other.mini_boards), // std::array supports deep copy by default
          miniboard_mask(other.miniboard_mask),
          win_masks(other.win_masks),
          mini_board_states(other.mini_board_states),
          n_moves(other.n_moves) {
        // Manually copy the stack, if needed. However, std::stack also supports deep copy.
        move_history = other.move_history;
    }

    //default constructor
    GlobalBoard() {
        for (int i = 0; i < 9; i++) {
            mini_boards[i] = MiniBoard();
        }
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

    
        // 64-bit Mersenne Twister Generator
        std::mt19937_64 rng(69420);
        
        // Uniform distribution across uint64_t range
        std::uniform_int_distribution<uint64_t> dist(99999, UINT64_MAX - 1);
        std::array<std::array<std::array<uint64_t, 9>, 9>, 2> move_hashes; //player, mini board, square
        std::array<std::array<uint64_t, 9>, 3> mini_board_hashes; //p0/p1/draw, mini board
        std::array<uint64_t, 9> legal_mini_board_hashes;
        uint64_t player_to_move_hash = dist(rng);
        for (int p = 0; p < 2; p++) {
            for (int m = 0; m < 9; m++) {
                for (int s = 0; s < 9; s++) {
                    move_hashes[p][m][s] = dist(rng);
                    mini_board_hashes[s % 3][m] = dist(rng);
                    legal_mini_board_hashes[s] = dist(rng);
                }
            }
        }
    }

};

class CrossfishDev {
       private:
        std::chrono::milliseconds thinking_time = std::chrono::milliseconds(95);
        Move root_best_move;
        std::chrono::time_point<std::chrono::high_resolution_clock> start_time =  std::chrono::high_resolution_clock::now();
        int min_val = -9999;
        int max_val = 9999;
    public:
        int root_score;
        int nodes;
        std::array<Move, 128> killer_moves;
        static const int tt_size = 1 << 23;
        std::vector<TTEntry, std::allocator<TTEntry>> transposition_table = std::vector<TTEntry>(tt_size);

        //0 1 2
        //3 4 5
        //6 7 8
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

        Move getMove(GlobalBoard board, std::chrono::milliseconds thinking_time_passed = std::chrono::milliseconds(95)) {
            thinking_time = thinking_time_passed;
            nodes = 0;
            root_score = 0;
            root_best_move = board.getLegalMoves()[0];

            //clear killers
            killer_moves = std::array<Move, 128>();
            start_time = std::chrono::high_resolution_clock::now();
            int depth = 1;
            int alpha = min_val;
            int beta = max_val;
            int aspiration_window = 200;
            int searches = 0;
            int researches = 0;
            while ((std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_time) < thinking_time)
            && (depth < 50)) {
                int eval = search(board, depth, 0, alpha, beta);
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

        int qsearch(GlobalBoard board, int alpha, int beta, int ply) {
            //quiescence search
            if (std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_time) > thinking_time) {
                return min_val;
            }
            nodes++;
            int stand_pat = evaluate(board);
            if (stand_pat >= beta) {
                return beta;
            }
            if (alpha < stand_pat) {
                alpha = stand_pat;
            }
            std::vector<Move> legal_moves = board.getLegalMoves();
            std::vector<int> scores = get_move_scores(legal_moves, {99, 99}, board, ply);

            Move best_move = legal_moves[0];
            int val;
            for (int i = 0; i < legal_moves.size(); i++) {
                if (scores[i] < 100) {
                    continue;
                }
                board.makeMove(legal_moves[i]);
                val = -qsearch(board, -beta, -alpha, ply + 1);
                board.unmakeMove();
                alpha = std::max(alpha, val);
                if (alpha >= beta) {
                    break;
                }
            }
            return alpha;



        }

        int search(GlobalBoard board, int depth, int ply, int alpha, int beta) {
            /*A simple negamax search*/
            //check out of time
            if (std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_time) > thinking_time) {
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
                        return max_val - ply; //current player won
                    }
                    else {
                        return min_val + ply; //previous player won
                    }
                }
                
            }
            TTEntry entry = transposition_table[board.zobrist_hash % tt_size];
            if ((entry.zobrist_hash == board.zobrist_hash ) && (entry.depth >= depth) && (board.zobrist_hash != 0)) {
                if (entry.flag == 1) {
                    alpha = std::max(alpha, entry.score);
                }
                else if (entry.flag == 2) {
                    beta = std::min(beta, entry.score);
                }
                else {
                    return entry.score;
                }
                if (alpha >= beta) {
                    return entry.score;
                }
            }
            if (depth == 0) {
                // return evaluate(board);
                return qsearch(board, alpha, beta, ply);
            }
            std::vector<Move> legal_moves = board.getLegalMoves();
            if (legal_moves.empty()){
                std::cerr << "LEGAL MOVES EMPTY. SHOULD NEVER REACH HERE " << "BOARD WINNER: " << board.checkWinner() << std::endl;
                std::cerr << "Player to move: " << board.n_moves % 2 << "Last move: " << board.move_history.top().mini_board << ", " << board.move_history.top().square << std::endl;
                board.print_board();
                // std::cout << board.checkWinner() << std::endl;
            }

            int stand_pat = evaluate(board);

            int reverse_futility_margin = 100;
            if (stand_pat - reverse_futility_margin * depth >= beta) {
                return beta;
            }

            int futility_margin = 100;
            bool can_futility_prune = (stand_pat + futility_margin * depth <= alpha); 
            
            std::vector<int> scores = get_move_scores(legal_moves, entry.best_move, board, ply);
            //sort on moves and scores, with scores as the key
            for (int i = 1; i < legal_moves.size(); i++) {
                int key = scores[i];
                Move key_move = legal_moves[i];
                int j = i - 1;
                while (j >= 0 && scores[j] < key) {
                    scores[j + 1] = scores[j];
                    legal_moves[j + 1] = legal_moves[j];
                    j = j - 1;
                }
                scores[j + 1] = key;
                legal_moves[j + 1] = key_move;
            }

            Move best_move = legal_moves[0];
            int best_val = min_val;
            int alpha_orig = alpha;
            for (int i = 0; i < legal_moves.size(); i++) {
                if (can_futility_prune && i > 0 && scores[i] == 0) { //dont search quiet moves in already losing positions
                    continue;
                }
                board.makeMove(legal_moves[i]);
                int val = -search(board, depth - 1, ply + 1, -beta, -alpha);
                board.unmakeMove();
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
                    killer_moves[ply] = legal_moves[i];
                    break;
                }
            }
            int flag = 0;
            if (best_val <= alpha_orig) {
                flag = 1;
            }
            else if (best_val >= beta) {
                flag = 2;
            }
            TTEntry new_entry = {depth, best_val, flag, board.zobrist_hash, best_move};
            transposition_table[board.zobrist_hash % tt_size] = new_entry;

            return best_val;

        }
        std::vector<int> get_move_scores(std::vector<Move> moves, Move tt_move, GlobalBoard board, int ply) {
            std::vector<int> scores = std::vector<int>(moves.size(), 0);
            for (int i = 0; i < moves.size(); i++) {
                int move_score = 0;
                if (moves[i].mini_board == tt_move.mini_board && moves[i].square == tt_move.square) {
                    move_score += 1000;
                    scores[i] = move_score;
                    continue;
                }

                //is it a killer move?
                if (moves[i].mini_board == killer_moves[ply].mini_board && moves[i].square == killer_moves[ply].square) {
                    move_score += 900;
                    scores[i] = move_score;
                    continue;
                }

                //if it wins a miniboard
                int miniboard_markers = board.mini_boards[moves[i].mini_board].markers[board.n_moves % 2];
                int opp_markers = board.mini_boards[moves[i].mini_board].markers[(board.n_moves + 1) % 2];
                for (int mask = 0; mask < board.win_masks.size(); mask++) {
                    if (((miniboard_markers | (1 << moves[i].square)) & board.win_masks[mask]) == board.win_masks[mask]) {
                        move_score += 100;
                        break;
                    }
                }

                //if it blocks a win
                for (int mask = 0; mask < board.win_masks.size(); mask++) {
                    if (((opp_markers | (1 << moves[i].square)) & board.win_masks[mask]) == board.win_masks[mask]) {
                        move_score += 75;
                        break;
                    }
                }

                //if it creates an unblocked 2 in a row
                for (int mask = 0; mask < two_in_a_row_masks.size() / 2; mask++) {
                    if (((miniboard_markers | (1 << moves[i].square)) & two_in_a_row_masks[mask * 2]) == two_in_a_row_masks[mask * 2]) {
                        if ((opp_markers & two_in_a_row_masks[mask * 2 + 1]) == 0 &&
                        (miniboard_markers & two_in_a_row_masks[mask * 2 + 1]) == 0) {
                            move_score += 50;
                            break;
                        }
                    }
                }

                //if it sends the opponent to a won or drawn miniboard
                int out_of_play = board.mini_board_states[0] | board.mini_board_states[1] | board.mini_board_states[2];
                if ((out_of_play & (1 << moves[i].square)) != 0) {
                    move_score -= 250;
                }
                scores[i] = move_score;
            }
            return scores;
            
        }

        int evaluate(GlobalBoard board) {
            /*use bitscan to count number of won miniboards for both players*/
            int p0_won = __builtin_popcount(board.mini_board_states[0]);
            int p1_won = __builtin_popcount(board.mini_board_states[1]);
            int out_of_play = board.mini_board_states[0] | board.mini_board_states[1] | board.mini_board_states[2]; 
            //count two in a rows for both players
            int p0_two_in_a_row = 0;
            int p1_two_in_a_row = 0;
            int p0_center_squares_held = 0;
            int p1_center_squares_held = 0;
            
            //idea, keep a map of two in a rows. Two in a rows that form two in a rows with other two in a rows are worth more
            int p0_two_in_a_row_map = 0;
            int p1_two_in_a_row_map = 0;

            for (int miniboard = 0; miniboard < 8; miniboard++) {
                if ((out_of_play & ( 1<< miniboard) != 0)) {
                    continue;
                }
                int p0_markers = board.mini_boards[miniboard].markers[0];
                int p1_markers = board.mini_boards[miniboard].markers[1];

                //make sure board is in play


                for (int i = 0; i < two_in_a_row_masks.size() / 2; i++) {
                    if (((p0_markers & two_in_a_row_masks[i * 2]) == two_in_a_row_masks[i * 2])
                    && ((p1_markers & two_in_a_row_masks[i * 2 + 1]) == 0)) {
                        p0_two_in_a_row++;
                        p0_two_in_a_row_map |= (1 << miniboard);
                    }
                    if (((p1_markers & two_in_a_row_masks[i * 2]) == two_in_a_row_masks[i * 2])
                    && ((p0_markers & two_in_a_row_masks[i * 2 + 1]) == 0)){
                        p1_two_in_a_row++;
                        p1_two_in_a_row_map |= (1 << miniboard);
                    }
                }

                if ((p0_markers & (1 << 4)) != 0) {
                    p0_center_squares_held++;
                } 
                else if ((p1_markers & (1 << 4)) != 0) {
                    p1_center_squares_held++;
                }
                
            }


            //also check for 2 in a rows in the out of play miniboards
            int p0_miniboards = board.mini_board_states[0];
            int p1_miniboards = board.mini_board_states[1];
            int p0_global_two_in_a_row = 0;
            int p1_global_two_in_a_row = 0;
            int p0_two_in_a_rows_lined_up = 0;
            int p1_two_in_a_rows_lined_up = 0;
            for(int i = 0; i < two_in_a_row_masks.size() / 2; i++) {
                //check for global two in a rows
                if (((p0_miniboards & two_in_a_row_masks[i * 2]) == two_in_a_row_masks[i * 2])
                && ((p1_miniboards & two_in_a_row_masks[i * 2 + 1]) == 0)) {
                    p0_global_two_in_a_row += 1;
                }
                if (((p1_miniboards & two_in_a_row_masks[i * 2]) == two_in_a_row_masks[i * 2])
                && ((p0_miniboards & two_in_a_row_masks[i * 2 + 1]) == 0)){
                    p1_global_two_in_a_row += 1;
                }
                //check for two in a rows that line up
                if ((((p0_two_in_a_row_map | p0_miniboards) & two_in_a_row_masks[i * 2]) == two_in_a_row_masks[i * 2])
                && ((p1_miniboards & two_in_a_row_masks[i * 2 + 1]) == 0)) {
                    p0_two_in_a_rows_lined_up++;
                }
                if ((((p1_two_in_a_row_map | p1_miniboards) & two_in_a_row_masks[i * 2]) == two_in_a_row_masks[i * 2])
                && ((p0_miniboards & two_in_a_row_masks[i * 2 + 1]) == 0)){
                    p1_two_in_a_rows_lined_up++;
                }
            }

            int val = (p0_won - p1_won) * 200;
            val += (p0_global_two_in_a_row - p1_global_two_in_a_row) * 100;
            val += (p0_two_in_a_row - p1_two_in_a_row) * 50;
            val += (p0_two_in_a_rows_lined_up - p1_two_in_a_rows_lined_up) * 25;
            return pow(-1, board.n_moves) * val;

        }
};

Move grid_coord_to_move(int row, int col) {
    int mini_board = (row / 3) * 3 + (col / 3);
    int square = (row % 3) * 3 + (col % 3);
    Move move = {mini_board, square};
    return move;
}

std::array<int, 2> move_to_grid_coord(Move move) {
    int row = (move.mini_board / 3) * 3 + (move.square / 3);
    int col = (move.mini_board % 3) * 3 + (move.square % 3);
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

//main function for codingame
int main()
{

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
            crossfish.getMove(board, std::chrono::milliseconds(400));
            std::cout << 4 << " " << 4 << std::endl;
            board.makeMove({4, 4});
        }
        else {
            Move best_move = crossfish.getMove(board);
            board.makeMove(best_move);
            std::array<int, 2> grid_coord = move_to_grid_coord(best_move);
            std::cout << grid_coord[0] << " " << grid_coord[1] << std::endl;
        }
    }
}