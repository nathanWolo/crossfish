#include <iostream>
#include <vector>
#include <stdlib.h>
#include <unistd.h>
#include <chrono>
#include <array>
#include <cmath> // For pow
#include <string>
#include <random>
#include <stack>
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

class GlobalBoard {
    private:
        std::array<MiniBoard, 9> mini_boards;
        uint16_t miniboard_mask = (1 << 9) - 1;
        /*
        0 1 2 
        3 4 5
        6 7 8
        */
        std::array<uint16_t, 8> win_masks = {(1 << 0) + (1 << 1) + (1 << 2), 
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

                if (n_moves > 0 && (out_of_play & (1 << prevMove.square) == 0) && (move.mini_board != prevMove.square)) //we were not sent to a won or drawn board
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
            
            //check if the miniboard is won
            for (int i = 0; i < 8; i++) {
                if ((mini_boards[move.mini_board].markers[n_moves % 2] & win_masks[i]) == win_masks[i]) {
                    mini_board_states[n_moves % 2] |= (1 << move.mini_board);
                    break;
                }
            }
            //check if the mini board is drawn
            if (((mini_boards[move.mini_board].markers[0] | mini_boards[move.mini_board].markers[1]) & miniboard_mask) == miniboard_mask) {
                mini_board_states[2] |= (1 << move.mini_board);
            }
            n_moves++;
        }
        void unmakeMove() {
            if (n_moves == 0) {
                std::cerr << "No moves to unmake" << std::endl;
                return;
            }
            n_moves--; //dec the number of moves so that the index is the same as when the move was made
            Move move = move_history.top();
            move_history.pop();
            mini_boards[move.mini_board].markers[n_moves % 2] &= ~(1 << move.square); //remove marker
            mini_boards[move.mini_board].markers[n_moves % 2] &= miniboard_mask; //make sure that only the last 9 bits are in use

            //open up that miniboard (this was either the last move on the board, or the board was open anyways)
            mini_board_states[0] &= ~(1 << move.mini_board);
            mini_board_states[1] &= ~(1 << move.mini_board);
            mini_board_states[2] &= ~(1 << move.mini_board);
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

};

class Crossfish {
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
            std::cerr << "Depth: " << depth << " Best Move: " << root_best_move.mini_board << " " << root_best_move.square << 
            " Score: " << root_score << " Nodes: " << nodes << std::endl;
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



int main()
{
    int wins = 0;
    int draws = 0;
    int losses = 0;
    for (int games = 0; games < 100; games++) {
    Crossfish crossfish;
    GlobalBoard board;
    // game loop
    while (board.checkWinner() == -1){
        if (board.n_moves % 2 == 0) {
            //get human input from stdin in the form minibox square
            // int minibox;
            // int square;
            // std::cin >> minibox >> square;
            // Move playerMove = {minibox, square};
            // board.makeMove(playerMove);
            // make a random move
            std::vector<Move> legal_moves = board.getLegalMoves();
            std::random_device rd;
            int low = 0;
            int high = legal_moves.size() - 1;
            std::mt19937 eng(rd()); // Mersenne Twister engine

            // Distribution in range [low, high]
            std::uniform_int_distribution<> distr(low, high);
            Move m = legal_moves[distr(eng)];
            board.makeMove(m);

        }
        else {
            Move best_move = crossfish.getMove(board);
            board.makeMove(best_move);
        }
        board.print_board();

    }
    //print winner 
    int winner = board.checkWinner();
    if (winner == 0) {
        std::cerr << "Player 0 wins" << std::endl;
        losses++;
    }
    else if (winner == 1) {
        std::cerr << "Player 1 wins" << std::endl;
        wins++;
    }
    else {
        std::cerr << "Draw" << std::endl;
        draws++;
    }
    }
    std::cerr << "Wins: " << wins << " Draws: " << draws << " Losses: " << losses << std::endl;
    return 0;
} 