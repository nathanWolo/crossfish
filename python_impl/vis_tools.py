'''
two things here
1) the fancy plotting (original matplotlib plotting)
2) conversion tools for nathan's visualization

'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
checkerboard_cmap = LinearSegmentedColormap.from_list('mycmap', ['lightgrey', 'white'])

from python_impl.operations import ops
from python_impl.board import board_obj

def checkerboard(shape):
    # from https://stackoverflow.com/questions/2169478/how-to-make-a-checkerboard-in-numpy
    # for visualization
    return np.indices(shape).sum(axis=0) % 2

def _check_line(box):
    '''
    box is a (3,3) array
    returns True if a line is found
    '''
    for i in range(3):
        if abs(sum(box[:,i])) == 3: return True # horizontal
        if abs(sum(box[i,:])) == 3: return True # vertical

    # diagonals
    if abs(box.trace()) == 3: return True
    if abs(np.rot90(box).trace()) == 3: return True
    # no line found
    return False

def calc_finished_boxes(temp_dict):
    ''' only used in plotting '''
    temp_finished = np.zeros((3,3,3))
    for i in range(3):
        for j in range(3):
            temp_box = temp_dict['board_state'][i*3:(i+1)*3][:,j*3:(j+1)*3]
            # p1
            temp_box_p1 = (temp_box == 1)
            temp_finished[i,j,0] = _check_line(np.clip(temp_box_p1,0,1))
            # p2
            temp_box_p2 = (temp_box == -1)
            temp_finished[i,j,1] = _check_line(np.clip(temp_box_p2,0,1))
            # stale
            if sum(temp_finished[i,j]) == 0:
                temp_finished[i,j,2] = abs(temp_box).sum() == 9
    return temp_finished

def fancy_draw_board(board_obj, marker_size: int = 100) -> None:

    # pull board information (marker and miniboxes)
    temp_dict = ops.pull_dictionary(board_obj)
    
    # flip board state for visualization of p2's turn.
    if board_obj.n_moves%2 == 1:
        temp_dict['board_state'] *= -1
        
    plt.imshow(checkerboard((9,9)), cmap=checkerboard_cmap, origin='lower')
    for i in [-0.5,2.5,5.5, 8.5]:
        plt.axvline(i,c='k')
        plt.axhline(i,c='k')
    plt.xticks(np.arange(9))
    plt.yticks(np.arange(9))

    # markers
    plt.scatter(*np.where(temp_dict['board_state'] == -1),
                marker='x',s=marker_size,c='tab:blue')
    plt.scatter(*np.where(temp_dict['board_state'] == 1),
                marker='o',s=marker_size,c='tab:orange')

    # miniboard markers
    finished_boxes = calc_finished_boxes(temp_dict)
    x_boxes = np.where(finished_boxes[:,:,1] == 1)
    o_boxes = np.where(finished_boxes[:,:,0] == 1)
    plt.scatter(x_boxes[0]*3+1,x_boxes[1]*3+1,
                marker='s',s=marker_size*50,alpha=0.6,c='tab:blue')
    plt.scatter(o_boxes[0]*3+1,o_boxes[1]*3+1,
                marker='s',s=marker_size*50,alpha=0.6,c='tab:orange')

    stale_boxes = np.where(finished_boxes[:,:,2] == 1)
    plt.scatter(stale_boxes[0]*3+1,stale_boxes[1]*3+1,
                marker='s',s=marker_size*50,alpha=0.3,c='k')
    
    # add markers for row and column

def add_valid_moves(board_obj):
    plt.scatter(*np.array(ops.get_valid_moves(board_obj)).T,
                marker='s',
                c='purple')


''' ---- part 2: nathans visualization ----- '''

def unconvert_log(symbol):
    ''' part of loading a game from a string '''
    offset = 32
    int_position = ord(symbol) - offset
    return (int_position//9, int_position%9)

def cvt_vis_app(board_obj) -> str:
    ''' converts a game board to a string for visualization with the app '''
    return ''.join([(chr(_p[0]*9 + _p[1] + 32)) for _p in board_obj.hist[:board_obj.n_moves]])

def load_game(game_string: str) -> board_obj:
    temp_board = board_obj()
    debug_log = list()
    ''' plays through a game up to the end of the game log '''
    for idx, encoded_position in enumerate(game_string):
        _move = unconvert_log(encoded_position)
        if not ops.check_move_is_valid(temp_board,_move):
            played_moves = ''.join(debug_log)
            raise Exception(f'valid moves played - {played_moves} - invalid move - {encoded_position}')
        ops.make_move(temp_board, _move)
        debug_log.append(encoded_position)
    return temp_board
