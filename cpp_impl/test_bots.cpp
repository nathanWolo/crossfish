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
#include <cmath>
#include <limits>
#pragma GCC optimize("O3")
#pragma GCC optimization("Ofast,unroll-loops")
#pragma GCC target("avx2,bmi,bmi2,lzcnt,popcnt")
//a struct representing a 3x3 board with 16 bit integers
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
        std::array<MiniBoard, 9> mini_boards;
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
    public:
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
            int occupied = mini_boards[move.mini_board].markers[0] | mini_boards[move.mini_board].markers[1];
            int out_of_play = mini_board_states[0] | mini_board_states[1] | mini_board_states[2];
            if (((occupied & (1 << move.square)) != 0)
            || ((out_of_play & (1 << move.mini_board)) != 0)
            || move.mini_board > 8 || move.square > 8 || move.mini_board < 0 || move.square < 0)
            {
                std::cerr << "ILLEGAL MOVE MADE: " << move.mini_board << " " << move.square << std::endl;
                std::cerr << "Last move: " << move_history.top().mini_board << " " << move_history.top().square << std::endl;
                print_board();
                std::exit(EXIT_FAILURE); // Terminate the program
            }
            if (n_moves > 0) {
                Move prevMove = move_history.top();

                if (n_moves > 0 && ((out_of_play & (1 << prevMove.square)) == 0) && (move.mini_board != prevMove.square)) //we were not sent to a won or drawn board
                    {
                        std::cerr << "ILLEGAL MOVE MADE: " << move.mini_board << " " << move.square << std::endl;
                        std::cerr << "Last move: " << move_history.top().mini_board << " " << move_history.top().square << std::endl;
                        print_board();
                        std::exit(EXIT_FAILURE); // Terminate the program
                    
                    }
            }

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
                return 2;
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
        static const int tt_size = 1 << 24;
        std::vector<TTEntry, std::allocator<TTEntry>> transposition_table = std::vector<TTEntry>(tt_size);
        Move getMove(GlobalBoard board) {
            nodes = 0;
            root_score = 0;
            root_best_move = board.getLegalMoves()[0];
            start_time = std::chrono::high_resolution_clock::now();
            int depth = 1;
            while ((std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_time) < thinking_time)
            && (depth < 50)) {
                search(board, depth, 0, min_val, max_val);
                depth++;
            }
            // std::cerr << "Depth: " << depth << " Best Move: " << root_best_move.mini_board << " " << root_best_move.square << 
            // " Score: " << root_score << " Nodes: " << nodes << std::endl;
            return root_best_move;
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
                    return evaluate(board)*10;
                }
                else {
                    return min_val + ply; //previous player won
                }
                
            }
            TTEntry entry = transposition_table[board.zobrist_hash % tt_size];
            if ((entry.zobrist_hash == board.zobrist_hash ) && (entry.depth >= depth) && (board.zobrist_hash != 0)) {
                std::cerr << "TT HIT, SEARCH DEPTH: " << depth << " ENTRY DEPTH: " << entry.depth << std::endl;
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
                return evaluate(board);
            }
            std::vector<Move> legal_moves = board.getLegalMoves();
            if (legal_moves.empty()){
                std::cerr << "LEGAL MOVES EMPTY. SHOULD NEVER REACH HERE " << "BOARD WINNER: " << board.checkWinner() << std::endl;
                std::cerr << "Player to move: " << board.n_moves % 2 << "Last move: " << board.move_history.top().mini_board << ", " << board.move_history.top().square << std::endl;
                board.print_board();
                // std::cout << board.checkWinner() << std::endl;
            }

            //check if the TT move is in the legal moves, and if it is, make it the first move
            if (entry.best_move.mini_board != 99) {
                for (int i = 0; i < legal_moves.size(); i++) {
                    if (legal_moves[i].mini_board == entry.best_move.mini_board && legal_moves[i].square == entry.best_move.square) {
                        Move temp = legal_moves[0];
                        legal_moves[0] = legal_moves[i];
                        legal_moves[i] = temp;
                        break;
                    }
                }
            }

            Move best_move = legal_moves[0];
            int best_val = min_val;
            int alpha_orig = alpha;
            for (int i = 0; i < legal_moves.size(); i++) {
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
        int evaluate(GlobalBoard board) {
            /*use bitscan to count number of won miniboards for both players*/
            int p0_won = __builtin_popcount(board.mini_board_states[0]);
            int p1_won = __builtin_popcount(board.mini_board_states[1]);
            int val = p0_won - p1_won;

            return pow(-1, board.n_moves) * val;

        }
};


class CrossfishPrev {
    private:
        std::chrono::milliseconds thinking_time = std::chrono::milliseconds(95);
        Move root_best_move;
        std::chrono::time_point<std::chrono::high_resolution_clock> start_time =  std::chrono::high_resolution_clock::now();
        int min_val = -9999;
        int max_val = 9999;
    public:
        int root_score;
        int nodes;
        Move getMove(GlobalBoard board) {
            nodes = 0;
            root_score = 0;
            root_best_move = board.getLegalMoves()[0];
            start_time = std::chrono::high_resolution_clock::now();
            int depth = 1;
            while ((std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start_time) < thinking_time)
            && (depth < 50)) {
                search(board, depth, 0, min_val, max_val);
                depth++;
            }
            // std::cerr << "Depth: " << depth << " Best Move: " << root_best_move.mini_board << " " << root_best_move.square << 
            // " Score: " << root_score << " Nodes: " << nodes << std::endl;
            return root_best_move;
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
                    return evaluate(board)*10;
                }
                else {
                    return min_val + ply; //previous player won
                }
                
            }
            if (depth == 0) {
                return evaluate(board);
            }
            std::vector<Move> legal_moves = board.getLegalMoves();
            if (legal_moves.empty()){
                std::cerr << "LEGAL MOVES EMPTY. SHOULD NEVER REACH HERE " << "BOARD WINNER: " << board.checkWinner() << std::endl;
                std::cerr << "Player to move: " << board.n_moves % 2 << "Last move: " << board.move_history.top().mini_board << ", " << board.move_history.top().square << std::endl;
                board.print_board();
                // std::cout << board.checkWinner() << std::endl;
            }
            Move best_move = legal_moves[0];
            int best_val = min_val;
            for (int i = 0; i < legal_moves.size(); i++) {
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
                    break;
                }
            }
            return best_val;

        }
        int evaluate(GlobalBoard board) {
            /*use bitscan to count number of won miniboards for both players*/
            int p0_won = __builtin_popcount(board.mini_board_states[0]);
            int p1_won = __builtin_popcount(board.mini_board_states[1]);
            int val = p0_won - p1_won;

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
            diff = (-400 * log10(1 / max_dev - 1)) - (-400 * log10(1 / min_dev - 1));
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
        std::cout << "Game : " << total_games << " Wins: " << global_total[0] 
                << " Draws: " << global_total[1] << " Losses: " << global_total[2] 
                << " Elo diff: " << elo.elo_diff << " +/- " << elo.ci << " LLR: " << sprt(global_total[0], global_total[1], global_total[2]) << std::endl;
        //reset board
        board = GlobalBoard(startpos);
    }
}

int main() {
    const unsigned int n_threads = std::thread::hardware_concurrency(); // Get the number of threads supported by the hardware
    // const unsigned int n_threads = 1;
    std::cout << "Number of threads: " << n_threads << std::endl;
    double llr = 0;
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