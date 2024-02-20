from python_impl.board import board_obj
from python_impl.operations import ops
import matplotlib.pyplot as plt
import python_impl.vis_tools as vis_tools
import matplotlib
import numpy as np
import copy
import matplotlib.pyplot as plt
import matplotlib
import math
from scipy import stats
from python_impl.operations import ops
from python_impl.board import board_obj
import python_impl.vis_tools as vis_tools
import time
import sys
import os
import fcntl
import select
import subprocess
from subprocess import PIPE
'''Purpose: test codingame UTTT submissions written in different languages against each other'''
def play_random_moves(b: board_obj, n_moves: int):
    ''' plays n_moves random moves on board b '''
    for i in range(n_moves):
        legal_moves = ops.get_valid_moves(b)
        move = legal_moves[np.random.choice(len(legal_moves))]
        ops.make_move(b, move)
    return b
class cg_bot_wrapper:
    '''A class to wrap running a cg bot process.
    Bots take in their opponents move via stdin and return their move in stdout.'''
    def __init__(self, bot_cmd):
        # Initialize the command to run the CodinGame bot
        self.cmd = bot_cmd
        # Start the bot process
        self.process = subprocess.Popen(self.cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE, text=True, bufsize=1)
        
    def send_move(self, move):
        '''Send a move to the bot and get its response.'''
        if self.process.poll() is not None:
            raise Exception("Bot process has terminated unexpectedly.")
        
        # Send the move to the bot's stdin and flush to ensure it's sent
        print(move, file=self.process.stdin, flush=True)
        self.print_stderr()
        time.sleep(0.11)
        # Read the bot's move from its stdout
        bot_move = self.process.stdout.readline().strip()
        return bot_move


    def set_non_blocking(self, fd):
        """Set the file descriptor to non-blocking mode."""
        flags = fcntl.fcntl(fd, fcntl.F_GETFL)
        fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)

    def print_stderr(self):
        '''Prints the bot's stderr output if available, without blocking.'''
        # Set stderr to non-blocking
        self.set_non_blocking(self.process.stderr.fileno())

        while True:
            ready, _, _ = select.select([self.process.stderr], [], [], 0.1)
            if not ready:  # If stderr is not ready, it's empty, so break
                break
            for stream in ready:
                # Read and print all available lines
                while True:
                    line = stream.readline()
                    if not line:  # If line is empty, no more data to read
                        break
                    print(line.strip(), file=sys.stderr)
    
    def close(self):
        '''Terminate the bot process gracefully.'''
        if self.process.poll() is None:
            self.process.terminate()
        self.process.wait()



#run 20 games
def faceoff_sequential(agent1_cmd, agent2_cmd, ngames=100, visualize=False):
    win_counter = 0 # quick integer check to see whether line bot wins more than loses
    loss_counter = 0
    draw_counter = 0
    if visualize:
        matplotlib.use('Qt5Agg')
        plt.ion()
        fig, ax = plt.subplots()
        score_text = ax.text(0,-2, '', transform=ax.transAxes) 
    # last_start_pos = board_obj()
    for j in range(ngames):
        my_board = board_obj()
        agent1 = cg_bot_wrapper(agent1_cmd)
        agent2 = cg_bot_wrapper(agent2_cmd)
        prev_move = '-1 -1'
        for i in range(81): # up to 81 moves per game.
            #alternating who goes first
            # time.sleep(1)
            if j % 2 == i % 2:
                move = agent1.send_move(prev_move)

                prev_move = move
                move_row, move_col = [int(i) for i in move.split()]
                #check move is legal
                if not ops.check_move_is_valid(my_board, (move_row, move_col)):
                    print(f'AGENT1 MADE AN ILLEGAL MOVE {(move_row, move_col)}')
            else:
                move = agent2.send_move(prev_move)
                prev_move = move
                move_row, move_col = [int(i) for i in move.split()]
                # print(f'agent2 makes move: {move_row, move_col}')
                #check move is legal
                # print(f'agent2 legal moves: {ops.get_valid_moves(my_board)}')
                if not ops.check_move_is_valid(my_board, (move_row, move_col)):
                    print(f'AGENT2 MADE AN ILLEGAL MOVE')
            ops.make_move(my_board, (move_row, move_col))
            if visualize:
                ax.clear()
                score_text = ax.text(1,1, '', transform=ax.transAxes)
                score_text.set_text(f'Agent 1 Wins: {win_counter}\nAgent 2 Wins: {loss_counter}\nDraws: {draw_counter}\nAgent 1 plays: {"O" if j % 2 == 0 % 2 else "X"}')
                vis_tools.fancy_draw_board(my_board)  # Draw new state
                plt.draw()
                plt.pause(0.1)
            if ops.check_game_finished(my_board):
                agent1.close()
                agent2.close()
                #check if agent1 won
                if j % 2 == i % 2:
                    win_counter += 1
                else:
                    #check if agent2 won
                    winner = ops.get_winner(my_board)
                    if 'stale' in winner:
                        draw_counter += 1
                    else:
                        loss_counter += 1
                break
        if visualize:
            ax.clear()
            if j == ngames - 1:
                plt.close(fig)  # Close the plot after each game
            
        print(f'game:{j} completed, agent1 wins: {win_counter}, agent2 wins: {loss_counter}, draws: {draw_counter}')

                
    
    # Calculate Elo difference
    total_games = win_counter + loss_counter + draw_counter
    win_rate = win_counter / total_games
    draw_rate = draw_counter / total_games
    loss_rate = loss_counter / total_games
    E = win_rate + 0.5 * draw_rate
    elo_diff = -400 * math.log10(1 / E - 1)

    #ci formula from view-source:https://3dkingdoms.com/chess/elo.htm
    percentage = (win_counter + draw_counter / 2) / total_games
    
    wins_dev = win_rate * (1- percentage)**2
    draws_dev = draw_rate * (0.5 - percentage)**2
    losses_dev = loss_rate * (0 - percentage)**2

    std_dev = math.sqrt(wins_dev + draws_dev + losses_dev) / math.sqrt(total_games)

    confidence = 0.95
    min_confidence = (1- confidence) / 2
    max_confidence = 1 - min_confidence

    min_dev = percentage + stats.norm.ppf(min_confidence) * std_dev
    max_dev = percentage + stats.norm.ppf(max_confidence) * std_dev
    try:
        diff = (-400 * math.log10(1 / max_dev - 1)) - (-400 * math.log10(1 / min_dev - 1)) 
    except ValueError:
        diff = np.inf
    d = {'win':win_counter, 'loss':loss_counter, 'draw':draw_counter, 'elo_diff':elo_diff, 'elo_diff_ci +/': diff}
    print(d)
    return {'win':win_counter, 'loss':loss_counter, 'draw':draw_counter, 'elo_diff':elo_diff, 'elo_diff_ci +/': diff}


faceoff_sequential(['./cpp_impl/crossfish'], ['python', 'python_impl/cg_random.py'], ngames=100, visualize=True)