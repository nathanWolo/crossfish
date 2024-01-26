from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import math
from scipy import stats
from operations import ops
from board import board_obj
import vis_tools
from bots import random_bot, line_completer_bot, minimax_ref, ab_pruning_ref, transposition_table, two_in_a_row_eval, tt_cutoffs, tt_move_ordering
def play_random_moves(b: board_obj, n_moves: int):
    ''' plays n_moves random moves on board b '''
    for i in range(n_moves):
        legal_moves = ops.get_valid_moves(b)
        move = legal_moves[np.random.choice(len(legal_moves))]
        ops.make_move(b, move)
    return b
def faceoff_sequential(agent1, agent2, ngames=100, visualize=False, n_random_moves=4):
    win_counter = 0 # quick integer check to see whether line bot wins more than loses
    loss_counter = 0
    draw_counter = 0
    if visualize:
        matplotlib.use('Qt5Agg')
        plt.ion()
        fig, ax = plt.subplots()
        score_text = ax.text(0,-2, '', transform=ax.transAxes) 
    for j in range(ngames):
        my_board = board_obj()
        #make first 4 moves randomly
        play_random_moves(my_board, n_random_moves)

        for i in range(81): # up to 81 moves per game.
            d = ops.pull_dictionary(my_board)
            #alternating who goes first
            if j % 2 == i % 2:
                move = agent1.move(d)
            else:
                move = agent2.move(d)
            ops.make_move(my_board, move)
            if visualize:
                ax.clear()
                score_text = ax.text(1,1, '', transform=ax.transAxes)
                score_text.set_text(f'Agent 1 Wins: {win_counter}\nAgent 2 Wins: {loss_counter}\nDraws: {draw_counter}\nAgent 1 plays: {"O" if j % 2 == 0 % 2 else "X"}')
                vis_tools.fancy_draw_board(my_board)  # Draw new state
                plt.draw()
                plt.pause(0.05)
            if ops.check_game_finished(my_board):
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


from joblib import Parallel, delayed

def play_single_game(agent1, agent2):
    win, draw, loss = 0, 0, 0
    my_board = board_obj()
    #make first 4 moves randomly
    for _ in range(4):
        legal_moves = ops.get_valid_moves(my_board)
        move = legal_moves[np.random.choice(len(legal_moves))]
        ops.make_move(my_board, move)

    for _ in range(81): # up to 81 moves per game.
        ''' ------ agent 1 turn ------'''
        # get dictionary 
        temp_dict = ops.pull_dictionary(my_board)
        # give dict to agent, calculate move
        agent1_move = agent1.move(temp_dict)
        # validate the move
        if not ops.check_move_is_valid(my_board, agent1_move):
            raise Exception(f'invalid move selected by p1, {agent1_move}')

        # make the move
        ops.make_move(my_board, agent1_move)
        # check whether game is finished
        if ops.check_game_finished(my_board):
            if 'agent 1' in ops.get_winner(my_board):
                win += 1
            else:
                draw += 1
            break

        ''' agent 2 turn '''
        # get dictionary 
        temp_dict = ops.pull_dictionary(my_board)
        # give dict to agent, calculate move
        agent2_move = agent2.move(temp_dict)

        # validate the move
        if not ops.check_move_is_valid(my_board, agent2_move):
            raise Exception(f'invalid move selected by p2, {agent2_move}')
        # make the move
        ops.make_move(my_board, agent2_move)
        # check whether game is finished
        if ops.check_game_finished(my_board):
            if 'agent 2' in ops.get_winner(my_board):
                loss += 1
            else:
                draw += 1
            break
    return win, loss, draw
import os
def calc_elo_diff(wins,losses,draws):
    # Calculate Elo difference
    total_games = wins + losses + draws
    win_rate = wins / total_games
    draw_rate = draws / total_games
    loss_rate = losses / total_games
    E = win_rate + 0.5 * draw_rate
    try:
        elo_diff = -400 * math.log10(1 / E - 1)
    except ValueError:
        elo_diff = np.inf

    #ci formula from view-source:https://3dkingdoms.com/chess/elo.htm
    percentage = (wins + draws / 2) / total_games
    
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
    return {'elo_diff':elo_diff, 'elo_diff_ci +/': diff}
def faceoff_parallel(agent1, agent2, ngames=100, njobs=-1):
    # Parallel execution with tqdm progress update
    # results = Parallel(n_jobs=njobs, backend="loky")(delayed(play_single_game)(agent1(), agent2()) for _ in tqdm(range(ngames)))
    #run games in batches of (core count), and report results after batch finishes and repeat
    results = []
    batch_size = njobs
    if njobs == -1:
        batch_size = os.cpu_count()
    for i in range(ngames // batch_size):
        results.extend(Parallel(n_jobs=batch_size, backend="loky")(delayed(play_single_game)(agent1(), agent2()) for _ in range(batch_size)))
        print(f'batch {i}/{ngames//batch_size} completed, agent1 wins: {sum(result[0] for result in results)}, agent2 wins: {sum(result[1] for result in results)}, draws: {sum(result[2] for result in results)}, elo_diff: {calc_elo_diff(sum(result[0] for result in results), sum(result[1] for result in results), sum(result[2] for result in results))}')

    # Aggregate results
    total_wins = sum(result[0] for result in results)
    total_losses = sum(result[1] for result in results)
    total_draws = sum(result[2] for result in results)

    # Calculate Elo difference
    total_games = total_wins + total_losses + total_draws
    win_rate = total_wins / total_games
    draw_rate = total_draws / total_games
    loss_rate = total_losses / total_games
    E = win_rate + 0.5 * draw_rate
    elo_diff = -400 * math.log10(1 / E - 1)

    #ci formula from view-source:https://3dkingdoms.com/chess/elo.htm
    percentage = (total_wins + total_draws / 2) / total_games
    
    wins_dev = win_rate * (1- percentage)**2
    draws_dev = draw_rate * (0.5 - percentage)**2
    losses_dev = loss_rate * (0 - percentage)**2

    std_dev = math.sqrt(wins_dev + draws_dev + losses_dev) / math.sqrt(total_games)

    confidence = 0.95
    min_confidence = (1- confidence) / 2
    max_confidence = 1 - min_confidence

    min_dev = percentage + stats.norm.ppf(min_confidence) * std_dev
    max_dev = percentage + stats.norm.ppf(max_confidence) * std_dev
    diff = (-400 * math.log10(1 / max_dev - 1)) - (-400 * math.log10(1 / min_dev - 1)) 
    

    d = {'win':total_wins, 'loss':total_losses, 'draw':total_draws, 'elo_diff':elo_diff, 'elo_conf_interval +/-': diff/2}
    print(d)
    return d
# faceoff_sequential(tt_move_ordering(), line_completer_bot(), ngames=20, visualize=True, n_random_moves=4)
faceoff_parallel(tt_move_ordering, tt_cutoffs, ngames=10000, njobs=-1)