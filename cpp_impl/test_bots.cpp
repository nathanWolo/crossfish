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
#include <bitset>
#include <limits>
#include <immintrin.h>
#pragma GCC optimize("O3")
#pragma GCC optimization("Ofast,unroll-loops")
#pragma GCC target("avx2,bmi,bmi2,lzcnt,popcnt")
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
            int occupied = mini_boards[move.mini_board].markers[0] | mini_boards[move.mini_board].markers[1];
            int out_of_play = mini_board_states[0] | mini_board_states[1] | mini_board_states[2];
            if (((occupied & (1 << move.square)) != 0)
            || ((out_of_play & (1 << move.mini_board)) != 0)
            || move.mini_board > 8 || move.square > 8 || move.mini_board < 0 || move.square < 0)
            {
                std::cerr << "ILLEGAL MOVE MADE: " << move.mini_board << " " << move.square << std::endl;
                std::cerr << "Last move: " << move_history.top().mini_board << " " << move_history.top().square << std::endl;
                std::cerr << "First illegal move block" << std::endl;

                //print which illegal move condition was met
                if ((occupied & (1 << move.square)) != 0) {
                    std::cerr << "Square occupied" << std::endl;
                }
                if ((out_of_play & (1 << move.mini_board)) != 0) {
                    std::cerr << "Board out of play" << std::endl;
                    //print binary rep of out of play
                    std::cerr << "Out of play: " << std::bitset<9>(out_of_play) << std::endl;
                    //print binary rep of miniboard we tried to play in 
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
                std::exit(EXIT_FAILURE); // Terminate the program
            }
            if (n_moves > 0) {
                Move prevMove = move_history.top();

                if (n_moves > 0 && ((out_of_play & (1 << prevMove.square)) == 0) && (move.mini_board != prevMove.square) && !prev_move_was_pass) //we were not sent to a won or drawn board
                    {
                        std::cerr << "ILLEGAL MOVE MADE: " << move.mini_board << " " << move.square << std::endl;
                        std::cerr << "Last move: " << move_history.top().mini_board << " " << move_history.top().square << std::endl;
                        std::cerr << "Second illegal move block" << std::endl;
                        print_board();
                        std::exit(EXIT_FAILURE); // Terminate the program
                    
                    }
            }
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

        std::vector<Move> get_captures() {
            std::vector<Move> captures;

            //check if we were sent to one miniboard or if we were sent to a won or drawn miniboard
            int active_square = move_history.top().square;
            int out_of_play = mini_board_states[0] | mini_board_states[1] | mini_board_states[2];

            if ((out_of_play & (1 << active_square)) == 0 ) { //we were not sent to a won or drawn board
                int marked = mini_boards[active_square].markers[0] | mini_boards[active_square].markers[1];
                //find all moves that capture this square
                for (int i = 0; i < 9; i++) {
                    if ((marked & (1 << i)) == 0) //if the square is not taken
                    {
                        Move move = {active_square, i};
                        if (is_capture_avx(move)) {
                            captures.push_back(move);
                        }
                    }
                }
            }
            else {
                //we were sent to a won or drawn board
                for (int i = 0; i < 9; i++) {
                    if ((out_of_play & (1 << i)) == 0) //check if board i is not out of play
                    {
                        int marked = mini_boards[i].markers[0] | mini_boards[i].markers[1];
                        for (int j = 0; j < 9; j++) {
                            if ((marked & (1 << j)) == 0) //if the square is not taken
                            {
                                Move move = {i, j};
                                if (is_capture_avx(move)) {
                                    captures.push_back(move);
                                }
                            }
                        }
                    }
                }
            }
            return captures;
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
                if (((out_of_play & (1 << active_square)) != 0) || prev_move_was_pass) {
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


class CrossfishPrev {
       private:
        std::chrono::milliseconds thinking_time = std::chrono::milliseconds(95);
        Move root_best_move;
        std::chrono::time_point<std::chrono::high_resolution_clock> start_time =  std::chrono::high_resolution_clock::now();
        int min_val = -99999;
        int max_val = 99999;
    public:
        int root_score;
        int nodes;
        std::array<Move, 128> killer_moves;
        std::array<std::array<std::array<int, 9>, 9>, 2> history_table; //player, mini board, square
        static const int tt_size = 1 << 23;
        std::vector<TTEntry, std::allocator<TTEntry>> transposition_table = std::vector<TTEntry>(tt_size);

        //0 1 2
        //3 4 5
        //6 7 8
        std::vector<int> two_in_a_row_masks = { //should this be an array instead?
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
        Move getMove(GlobalBoard board, std::chrono::milliseconds thinking_time_passed = std::chrono::milliseconds(95)) {
            thinking_time = thinking_time_passed;
            nodes = 0;
            root_score = 0;
            root_best_move = board.getLegalMoves()[0];

            //clear killers
            killer_moves = std::array<Move, 128>();
            history_table = std::array<std::array<std::array<int, 9>, 9>, 2>();
            start_time = std::chrono::high_resolution_clock::now();
            depth = 1;
            int alpha = min_val;
            int beta = max_val;
            int aspiration_window = 1500;
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

        int qsearch(GlobalBoard &board, int alpha, int beta, int ply) {
            //quiescence search
            if (std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_time) > thinking_time) {
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

            int stand_pat = evaluate(board);
            if (stand_pat >= beta) {
                return beta;
            }
            if (alpha < stand_pat) {
                alpha = stand_pat;
            }

            //get and sort moves
            std::vector<Move> caps = board.get_captures();
            std::vector<int> scores = get_move_scores(caps, {99, 99}, board, ply);
            sort_moves(caps, scores);
            int val;
            for (int i = 0; i < caps.size(); i++) {
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

        int search(GlobalBoard board, int depth, int ply, int alpha, int beta,  bool can_null = true) {
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
            bool pv_node = (beta - alpha > 1);
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
            //internal iterative deepening
            if (pv_node && entry.zobrist_hash != board.zobrist_hash && depth > 2) {
                search(board, 1, ply, alpha, beta, false);
                entry = transposition_table[board.zobrist_hash % tt_size];
            }

            //singular extensions condition
            // if we got a tt hit, and the depth on the entry isn't too low, and the entry is a lower bound or exact score
            bool singular = (entry.zobrist_hash == board.zobrist_hash && entry.depth >= depth - 3 && (entry.flag == 1 || entry.flag == 0));
        
            std::vector<Move> legal_moves = board.getLegalMoves();
            if (legal_moves.empty()){
                std::cerr << "LEGAL MOVES EMPTY. SHOULD NEVER REACH HERE " << "BOARD WINNER: " << board.checkWinner() << std::endl;
                std::cerr << "Player to move: " << board.n_moves % 2 << "Last move: " << board.move_history.top().mini_board << ", " << board.move_history.top().square << std::endl;
                board.print_board();
                // std::cout << board.checkWinner() << std::endl;
            }
            
            std::vector<int> scores = get_move_scores(legal_moves, entry.best_move, board, ply);
            //sort on moves and scores, with scores as the key
            sort_moves(legal_moves, scores);

            Move best_move = legal_moves[0];
            int best_val = min_val;
            int alpha_orig = alpha;
            int val;
            int nmoves = legal_moves.size();
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
                    if (scores[i] < 0) {
                        reduction = i/4; //late move reduction
                    }
                    val = -search(board, depth - 1 - reduction + extension, ply + 1, -alpha - 1, -alpha, can_null);
                    if (val > alpha && val < beta) {
                        val = -search(board, depth - 1, ply + 1, -beta, -alpha, can_null);
                    }
                }
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
                    // history_table[board.n_moves % 2][legal_moves[i].mini_board][legal_moves[i].square] += (1 << depth);
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

        void sort_moves(std::vector<Move>& moves, std::vector<int>& scores) {
            //sort on moves and scores, with scores as the key, in place
            for (int i = 1; i < moves.size(); i++) {
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

            // Prepare a vector of miniboard_markers
            __m256i markers_vec = _mm256_set1_epi32(miniboard_markers);

            // Load win_masks into a vector
            __m256i win_masks_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(board.win_masks.data()));

            // Perform AND and compare operations
            __m256i result_vec = _mm256_and_si256(markers_vec, win_masks_vec);
            //Check if any of our results are equal to the win masks
            result_vec = _mm256_cmpeq_epi32(result_vec, win_masks_vec);

            // Aggregate results: if any of the win conditions is fully met, result is true
            int mask = _mm256_movemask_ps(_mm256_castsi256_ps(result_vec));
            return mask != 0;
        }

        bool is_block_avx(GlobalBoard &board, Move &move) {
            int opp_markers = board.mini_boards[move.mini_board].markers[(board.n_moves + 1) % 2];
            opp_markers |= (1 << move.square);

            // Prepare a vector of opp_markers
            __m256i markers_vec = _mm256_set1_epi32(opp_markers);

            // Load win_masks into a vector
            __m256i win_masks_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(board.win_masks.data()));

            // Perform AND and compare operations
            __m256i result_vec = _mm256_and_si256(markers_vec, win_masks_vec);
            result_vec = _mm256_cmpeq_epi32(result_vec, win_masks_vec);

            // Aggregate results: if any of the win conditions is fully met, result is true
            int mask = _mm256_movemask_ps(_mm256_castsi256_ps(result_vec));
            return mask != 0;
        }


        bool creates_two_in_a_row(GlobalBoard &board, Move &move) {
            int our_markers = board.mini_boards[move.mini_board].markers[board.n_moves % 2];
            int opp_markers = board.mini_boards[move.mini_board].markers[(board.n_moves + 1) % 2];
            bool result = false;

            for (int mask = 0; mask < two_in_a_row_masks.size() / 2; mask++) {
                result = result || ((((our_markers | (1 << move.square)) & two_in_a_row_masks[mask * 2]) == two_in_a_row_masks[mask * 2]) && 
                (((opp_markers | our_markers) & two_in_a_row_masks[mask * 2 + 1]) == 0));
            }

            return result;

        }


        std::vector<int> get_move_scores(std::vector<Move> &moves, Move tt_move, GlobalBoard &board, int &ply) {
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
                if (is_capture_avx(board, moves[i])) {
                    move_score += 100;
                }

                //if it blocks a win
                if (is_block_avx(board, moves[i])) {
                    move_score += 75;
                }

                //if it creates an unblocked 2 in a row
                if (creates_two_in_a_row(board, moves[i])) {
                    move_score += 50;
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

        int evaluate(GlobalBoard &board) {
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

            int p0_tiar_temp;
            int p1_tiar_temp;
            int p0_markers;
            int p1_markers;
            for (int miniboard = 0; miniboard < 8; miniboard++) {
                if ((out_of_play & ( 1<< miniboard) != 0)) {
                    continue;
                }
                p0_markers = board.mini_boards[miniboard].markers[0];
                p1_markers = board.mini_boards[miniboard].markers[1];

                //make sure board is in play

                //find and keep track of two in a rows
                for (int i = 0; i < two_in_a_row_masks.size() / 2; i++) {
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
                //total squares
                p0_squares_held += __builtin_popcount(p0_markers);
                p1_squares_held += __builtin_popcount(p1_markers);
                
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
            for(int i = 0; i < two_in_a_row_masks.size() / 2; i++) {
                //check for global two in a rows
                p0_global_two_in_a_row += ((__builtin_popcount(p0_miniboards & two_in_a_row_masks[i * 2]) - __builtin_popcount(p1_miniboards & two_in_a_row_masks[i * 2 + 1])) /2);
                p1_global_two_in_a_row += ((__builtin_popcount(p1_miniboards & two_in_a_row_masks[i * 2]) - __builtin_popcount(p0_miniboards & two_in_a_row_masks[i * 2 + 1])) /2);
                //check for two in a rows that line up
                p0_two_in_a_rows_lined_up += ((__builtin_popcount((p0_two_in_a_row_map | p0_miniboards) & two_in_a_row_masks[i * 2]) - __builtin_popcount(p1_miniboards & two_in_a_row_masks[i * 2 + 1]))  / 2);
                p1_two_in_a_rows_lined_up += ((__builtin_popcount((p1_two_in_a_row_map | p1_miniboards) & two_in_a_row_masks[i * 2]) - __builtin_popcount(p0_miniboards & two_in_a_row_masks[i * 2 + 1]))   / 2);
            }
            //should tune these coefficients
            int val = (p0_miniboards_held - p1_miniboards_held) * 2000;
            val += (p0_center_miniboard_held - p1_center_miniboard_held) * 1000;
            val += (p0_corner_miniboards_held - p1_corner_miniboards_held) * 500;
            val += (p0_global_two_in_a_row - p1_global_two_in_a_row) * 1500;
            val += (p0_two_in_a_row - p1_two_in_a_row) * 500;
            val += (p0_two_in_a_rows_lined_up - p1_two_in_a_rows_lined_up) * 500;
            val += (p0_center_squares_held - p1_center_squares_held) * 20;
            val += (p0_corner_squares_held - p1_corner_squares_held) * 10;
            val += (p0_squares_held - p1_squares_held)* 20;

            //tempo bonus to help with aspiration windows
            val += pow(-1, board.n_moves) * 50;
            return pow(-1, board.n_moves) * val;

        }

};

class CrossfishDev {
       private:
        std::chrono::milliseconds thinking_time = std::chrono::milliseconds(95);
        Move root_best_move;
        std::chrono::time_point<std::chrono::high_resolution_clock> start_time =  std::chrono::high_resolution_clock::now();
        int min_val = -99999;
        int max_val = 99999;
    public:
        int root_score;
        int nodes;
        std::array<Move, 128> killer_moves;
        std::array<std::array<std::array<int, 9>, 9>, 2> history_table; //player, mini board, square
        static const int tt_size = 1 << 23;
        std::vector<TTEntry, std::allocator<TTEntry>> transposition_table = std::vector<TTEntry>(tt_size);

        //0 1 2
        //3 4 5
        //6 7 8
        std::vector<int> two_in_a_row_masks = { //should this be an array instead?
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
        Move getMove(GlobalBoard board, std::chrono::milliseconds thinking_time_passed = std::chrono::milliseconds(95)) {
            thinking_time = thinking_time_passed;
            nodes = 0;
            root_score = 0;
            root_best_move = board.getLegalMoves()[0];

            //clear killers
            killer_moves = std::array<Move, 128>();
            history_table = std::array<std::array<std::array<int, 9>, 9>, 2>();
            start_time = std::chrono::high_resolution_clock::now();
            depth = 1;
            int alpha = min_val;
            int beta = max_val;
            int aspiration_window = 1500;
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

        int qsearch(GlobalBoard &board, int alpha, int beta, int ply) {
            //quiescence search
            if (std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_time) > thinking_time) {
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

            int stand_pat = evaluate(board);
            if (stand_pat >= beta) {
                return beta;
            }
            if (alpha < stand_pat) {
                alpha = stand_pat;
            }

            //get and sort moves
            std::vector<Move> caps = board.get_captures();
            std::vector<int> scores = get_move_scores(caps, {99, 99}, board, ply);
            sort_moves(caps, scores);
            int val;
            for (int i = 0; i < caps.size(); i++) {
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

        int search(GlobalBoard board, int depth, int ply, int alpha, int beta,  bool can_null = true) {
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
            bool pv_node = (beta - alpha > 1);
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
            //internal iterative deepening
            if (pv_node && entry.zobrist_hash != board.zobrist_hash && depth > 2) {
                search(board, 1, ply, alpha, beta, false);
                entry = transposition_table[board.zobrist_hash % tt_size];
            }

            //singular extensions condition
            // if we got a tt hit, and the depth on the entry isn't too low, and the entry is a lower bound or exact score
            bool singular = (entry.zobrist_hash == board.zobrist_hash && entry.depth >= depth - 3 && (entry.flag == 1 || entry.flag == 0));
        
            std::vector<Move> legal_moves = board.getLegalMoves();
            if (legal_moves.empty()){
                std::cerr << "LEGAL MOVES EMPTY. SHOULD NEVER REACH HERE " << "BOARD WINNER: " << board.checkWinner() << std::endl;
                std::cerr << "Player to move: " << board.n_moves % 2 << "Last move: " << board.move_history.top().mini_board << ", " << board.move_history.top().square << std::endl;
                board.print_board();
                // std::cout << board.checkWinner() << std::endl;
            }
            
            std::vector<int> scores = get_move_scores(legal_moves, entry.best_move, board, ply);
            //sort on moves and scores, with scores as the key
            sort_moves(legal_moves, scores);

            Move best_move = legal_moves[0];
            int best_val = min_val;
            int alpha_orig = alpha;
            int val;
            int nmoves = legal_moves.size();
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
                    if (scores[i] < 0) {
                        reduction = i/4; //late move reduction
                    }
                    val = -search(board, depth - 1 - reduction + extension, ply + 1, -alpha - 1, -alpha, can_null);
                    if (val > alpha && val < beta) {
                        val = -search(board, depth - 1, ply + 1, -beta, -alpha, can_null);
                    }
                }
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
                    // history_table[board.n_moves % 2][legal_moves[i].mini_board][legal_moves[i].square] += (1 << depth);
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

        void sort_moves(std::vector<Move>& moves, std::vector<int>& scores) {
            //sort on moves and scores, with scores as the key, in place
            for (int i = 1; i < moves.size(); i++) {
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

            // Prepare a vector of miniboard_markers
            __m256i markers_vec = _mm256_set1_epi32(miniboard_markers);

            // Load win_masks into a vector
            __m256i win_masks_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(board.win_masks.data()));

            // Perform AND and compare operations
            __m256i result_vec = _mm256_and_si256(markers_vec, win_masks_vec);
            //Check if any of our results are equal to the win masks
            result_vec = _mm256_cmpeq_epi32(result_vec, win_masks_vec);

            // Aggregate results: if any of the win conditions is fully met, result is true
            int mask = _mm256_movemask_ps(_mm256_castsi256_ps(result_vec));
            return mask != 0;
        }

        bool is_block_avx(GlobalBoard &board, Move &move) {
            int opp_markers = board.mini_boards[move.mini_board].markers[(board.n_moves + 1) % 2];
            opp_markers |= (1 << move.square);

            // Prepare a vector of opp_markers
            __m256i markers_vec = _mm256_set1_epi32(opp_markers);

            // Load win_masks into a vector
            __m256i win_masks_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(board.win_masks.data()));

            // Perform AND and compare operations
            __m256i result_vec = _mm256_and_si256(markers_vec, win_masks_vec);
            result_vec = _mm256_cmpeq_epi32(result_vec, win_masks_vec);

            // Aggregate results: if any of the win conditions is fully met, result is true
            int mask = _mm256_movemask_ps(_mm256_castsi256_ps(result_vec));
            return mask != 0;
        }


        bool creates_two_in_a_row(GlobalBoard &board, Move &move) {
            int our_markers = board.mini_boards[move.mini_board].markers[board.n_moves % 2];
            int opp_markers = board.mini_boards[move.mini_board].markers[(board.n_moves + 1) % 2];
            bool result = false;

            for (int mask = 0; mask < two_in_a_row_masks.size() / 2; mask++) {
                result = result || ((((our_markers | (1 << move.square)) & two_in_a_row_masks[mask * 2]) == two_in_a_row_masks[mask * 2]) && 
                (((opp_markers | our_markers) & two_in_a_row_masks[mask * 2 + 1]) == 0));
            }

            return result;

        }


        std::vector<int> get_move_scores(std::vector<Move> &moves, Move tt_move, GlobalBoard &board, int &ply) {
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
                if (is_capture_avx(board, moves[i])) {
                    move_score += 100;
                }

                //if it blocks a win
                if (is_block_avx(board, moves[i])) {
                    move_score += 75;
                }

                //if it creates an unblocked 2 in a row
                if (creates_two_in_a_row(board, moves[i])) {
                    move_score += 50;
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

        int evaluate(GlobalBoard &board) {
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

            int p0_tiar_temp;
            int p1_tiar_temp;
            int p0_markers;
            int p1_markers;
            for (int miniboard = 0; miniboard < 8; miniboard++) {
                if ((out_of_play & ( 1<< miniboard) != 0)) {
                    continue;
                }
                p0_markers = board.mini_boards[miniboard].markers[0];
                p1_markers = board.mini_boards[miniboard].markers[1];

                //make sure board is in play

                //find and keep track of two in a rows
                for (int i = 0; i < two_in_a_row_masks.size() / 2; i++) {
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
                //total squares
                p0_squares_held += __builtin_popcount(p0_markers);
                p1_squares_held += __builtin_popcount(p1_markers);
                
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
            for(int i = 0; i < two_in_a_row_masks.size() / 2; i++) {
                //check for global two in a rows
                p0_global_two_in_a_row += ((__builtin_popcount(p0_miniboards & two_in_a_row_masks[i * 2]) - __builtin_popcount(p1_miniboards & two_in_a_row_masks[i * 2 + 1])) /2);
                p1_global_two_in_a_row += ((__builtin_popcount(p1_miniboards & two_in_a_row_masks[i * 2]) - __builtin_popcount(p0_miniboards & two_in_a_row_masks[i * 2 + 1])) /2);
                //check for two in a rows that line up
                p0_two_in_a_rows_lined_up += ((__builtin_popcount((p0_two_in_a_row_map | p0_miniboards) & two_in_a_row_masks[i * 2]) - __builtin_popcount(p1_miniboards & two_in_a_row_masks[i * 2 + 1]))  / 2);
                p1_two_in_a_rows_lined_up += ((__builtin_popcount((p1_two_in_a_row_map | p1_miniboards) & two_in_a_row_masks[i * 2]) - __builtin_popcount(p0_miniboards & two_in_a_row_masks[i * 2 + 1]))   / 2);
            }
            //should tune these coefficients
            int val = (p0_miniboards_held - p1_miniboards_held) * 2000;
            val += (p0_center_miniboard_held - p1_center_miniboard_held) * 1000;
            val += (p0_corner_miniboards_held - p1_corner_miniboards_held) * 500;
            val += (p0_global_two_in_a_row - p1_global_two_in_a_row) * 1500;
            val += (p0_two_in_a_row - p1_two_in_a_row) * 500;
            val += (p0_two_in_a_rows_lined_up - p1_two_in_a_rows_lined_up) * 500;
            val += (p0_center_squares_held - p1_center_squares_held) * 20;
            val += (p0_corner_squares_held - p1_corner_squares_held) * 10;
            val += (p0_squares_held - p1_squares_held)* 20;

            //tempo bonus to help with aspiration windows
            val += pow(-1, board.n_moves) * 50;
            return pow(-1, board.n_moves) * val;

        }

};

struct EloResult {
    double elo_diff;
    double ci;
};

double norm_ppf(double p) {
    // An approximation of the inverse of the cumulative distribution function for the standard normal distribution.
    // Constants are from a simplified version of the Abramowitz and Stegun formula (26.2.23).
    // This approximation is not as accurate as those provided by statistical libraries but is sufficient for basic needs.
    const double a1 = -39.6968302866538, a2 = 220.946098424521, a3 = -275.928510446969;
    const double a4 = 138.357751867269, a5 = -30.6647980661472, a6 = 2.50662827745924;
    const double b1 = -54.4760987982241, b2 = 161.585836858041, b3 = -155.698979859887;
    const double b4 = 66.8013118877197, b5 = -13.2806815528857, c1 = -7.78489400243029E-03;
    const double c2 = -0.322396458041136, c3 = -2.40075827716184, c4 = -2.54973253934373;
    const double c5 = 4.37466414146497, c6 = 2.93816398269878, d1 = 7.78469570904146E-03;
    const double d2 = 0.32246712907004, d3 = 2.445134137143, d4 = 3.75440866190742;
    const double p_low = 0.02425, p_high = 1 - p_low;
    double q, r;

    if (p < 0 || p > 1) {
        return std::numeric_limits<double>::quiet_NaN();
    } else if (p < p_low) {
        q = sqrt(-2*log(p));
        return (((((c1*q+c2)*q+c3)*q+c4)*q+c5)*q+c6) / ((((d1*q+d2)*q+d3)*q+d4)*q+1);
    } else if (p <= p_high) {
        q = p - 0.5;
        r = q*q;
        return (((((a1*r+a2)*r+a3)*r+a4)*r+a5)*r+a6)*q / (((((b1*r+b2)*r+b3)*r+b4)*r+b5)*r+1);
    } else {
        q = sqrt(-2*log(1-p));
        return -(((((c1*q+c2)*q+c3)*q+c4)*q+c5)*q+c6) / ((((d1*q+d2)*q+d3)*q+d4)*q+1);
    }
}


EloResult calc_elo_diff(int wins, int losses, int draws) {
    int total_games = wins + losses + draws;
    double win_rate = static_cast<double>(wins) / total_games;
    double draw_rate = static_cast<double>(draws) / total_games;
    double loss_rate = static_cast<double>(losses) / total_games;
    double E = win_rate + 0.5 * draw_rate;
    double elo_diff;

    try {
        if (E == 1) {
            elo_diff = std::numeric_limits<double>::infinity();
        } else {
            elo_diff = -400 * log10(1 / E - 1);
        }
    } catch (...) {
        elo_diff = std::numeric_limits<double>::infinity();
    }

    // CI formula
    double percentage = (wins + static_cast<double>(draws) / 2) / total_games;
    
    double wins_dev = win_rate * std::pow(1 - percentage, 2);
    double draws_dev = draw_rate * std::pow(0.5 - percentage, 2);
    double losses_dev = loss_rate * std::pow(0 - percentage, 2);

    double std_dev = sqrt(wins_dev + draws_dev + losses_dev) / sqrt(total_games);

    double confidence = 0.95;
    double min_confidence = (1 - confidence) / 2;
    double max_confidence = 1 - min_confidence;

    double min_dev = percentage + norm_ppf(min_confidence) * std_dev;
    double max_dev = percentage + norm_ppf(max_confidence) * std_dev;

    double diff;

    try {
        if (max_dev == 1 || min_dev == 1) {
            diff = std::numeric_limits<double>::infinity();
        } else {
            diff = ((-400 * log10(1 / max_dev - 1)) - (-400 * log10(1 / min_dev - 1)))/2;
        }
    } catch (...) {
        diff = std::numeric_limits<double>::infinity();
    }

    return {elo_diff, diff};
}


class RandomMover {
    public:
        Move getMove(GlobalBoard board) {
            std::vector<Move> legal_moves = board.getLegalMoves();
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<int> dis(0, legal_moves.size() - 1);
            return legal_moves[dis(gen)];
        }
};

class HumanPlayer {
    public:
        Move getMove(GlobalBoard board) {
            int mini_board;
            int square;
            std::cout << "Enter mini board and square: ";
            std::cin >> mini_board >> square;
            Move move = {mini_board, square};
            return move;
        }
};

std::array<int, 3> global_total = {0, 0, 0}; //wins, draws, losses
std::mutex global_mutex;
std::atomic<int> completed_tasks(0);

std::array<double, 3> eloToWDL(double elo, double dlo) {
    std::array<double, 3> probabilities;
    
    double w = 1 / (1 + std::pow(10, (-elo + dlo) / 400)); // win probability
    double l = 1 / (1 + std::pow(10, (+elo + dlo) / 400)); // loss probability
    double d = 1 - w - l;                                  // draw probability

    probabilities[0] = w;
    probabilities[1] = d;
    probabilities[2] = l;
    
    return probabilities;
}

std::pair<double, double> wdlToElo(double w, double d, double l) {
    double elo = 200 * std::log10((w / l) * ((1 - l) / (1 - w)));
    double dlo = 200 * std::log10(((1 - l) / l) * ((1 - w) / w));
    return {elo, dlo};
}

double sprt(int wins, int draws, int losses) {
    if (wins == 0 || losses == 0 || draws == 0) {
        return 0;
    }
    //testing that we're gaining 5 or more elo
    double elo0 = 0;
    double elo1 = 5;
    
    double n = wins + draws + losses;

    double dlo = wdlToElo(wins / n, draws / n, losses / n).second;

    std::array<double, 3> probabilities0 = eloToWDL(elo0, dlo);
    std::array<double, 3> probabilities1 = eloToWDL(elo1, dlo);

    return (double)wins * log(probabilities1[0] / probabilities0[0]) 
        + (double)draws * log(probabilities1[1] / probabilities0[1])
        + (double)losses * log(probabilities1[2] / probabilities0[2]); 
}

void play_game(){
    //play two games from the same start position, alternating who goes first
    RandomMover random_mover;

    //update these bots to test new changes
    CrossfishDev bot2;
    CrossfishPrev bot1;

    GlobalBoard board;
    //game loop
    //first 4-8 moves are random
    int num_random_moves = 4 + rand() % 5;
    for (int i = 0; i < num_random_moves; i++) {
        if (i == 0) {
            //30% chance of first move being very center
            if (rand() % 10 < 3) {
                Move m = {4, 4};
                board.makeMove(m);
                continue;
            }
        }
        Move m = random_mover.getMove(board);
        board.makeMove(m);
    }
    GlobalBoard startpos = GlobalBoard(board);
    //play two games, alternating who goes first
    for (int i = 0; i < 2; i++) {
        int bot1_player;
        int bot2_player;
        while (board.checkWinner() == -1){
            if (board.n_moves % 2 == i) {
                bot1_player = board.n_moves % 2;
                Move m = bot1.getMove(board);
                board.makeMove(m);
            }
            else {
                bot2_player = board.n_moves % 2;
                Move best_move = bot2.getMove(board);
                board.makeMove(best_move);
            }
        }
        //update global total
        int winner = board.checkWinner();
        if (winner  == bot1_player) {
            global_mutex.lock();
            global_total[2]++; //loss
            global_mutex.unlock();
        }
        else if (winner  == bot2_player) {
            global_mutex.lock();
            global_total[0]++;  //win
            global_mutex.unlock();
        }
        else {
            global_mutex.lock();
            global_total[1]++; //draw
            global_mutex.unlock();
        }
        EloResult elo = calc_elo_diff(global_total[0], global_total[2], global_total[1]);
        int total_games = global_total[0] + global_total[1] + global_total[2];
        std::cout << "N: " << total_games << " W: " << global_total[0] 
                << " D: " << global_total[1] << " L: " << global_total[2] 
                << " Elo diff: " << elo.elo_diff << " +/- " << elo.ci << " LLR: " << sprt(global_total[0], global_total[1], global_total[2]) << std::endl;
        //reset board
        board = GlobalBoard(startpos);
    }
}

int main() {
    const unsigned int n_threads = std::thread::hardware_concurrency(); // Get the number of threads supported by the hardware
    // const unsigned int n_threads = 6;
    std::cout << "Number of threads: " << n_threads << std::endl;
    double llr = 0;

    //benchmark NPS from startpos for Prev and Dev
    CrossfishPrev prev;
    CrossfishDev dev;
    GlobalBoard board;
    std::chrono::milliseconds thinking_time = std::chrono::milliseconds(1000);//1 second
    prev.getMove(board, thinking_time);
    int prev_nps = prev.nodes;
    dev.getMove(board, thinking_time);
    int dev_nps = dev.nodes;
    std::cout << "Prev NPS: " << prev_nps << " Dev NPS: " << dev_nps << std::endl;
    while(abs(llr) < 3) {
        std::vector<std::future<void>> futures;
        for (unsigned int i = 0; i < n_threads; ++i) {
            futures.push_back(std::async(std::launch::async, play_game));
        }
        for (auto& f : futures) {
            f.get();
        }
        llr = sprt(global_total[0], global_total[1], global_total[2]);
    }
    return 0;
}