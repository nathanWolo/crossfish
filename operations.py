import numpy as np
from board import board_obj

class ops():
    lines_mask = np.array([[1,1,1,0,0,0,0,0,0], # horizontals
                       [0,0,0,1,1,1,0,0,0],
                       [0,0,0,0,0,0,1,1,1],
                       [1,0,0,1,0,0,1,0,0], # verticals
                       [0,1,0,0,1,0,0,1,0],
                       [0,0,1,0,0,1,0,0,1],
                       [1,0,0,0,1,0,0,0,1], # diagonals
                       [0,0,1,0,1,0,1,0,0]],dtype=bool).reshape(-1,3,3)
        
    @staticmethod
    def get_player(board_obj: board_obj) -> int:
        return board_obj.n_moves%2 
    
    @staticmethod
    def make_move(board_obj: board_obj, move:tuple) -> None:
        # NOTE: there is no safety check in here, we deal with error handling elsewhere.
        
        # update move history
        board_obj.hist[board_obj.n_moves] = move
        
        # update board for player
        board_obj.markers[move[0],move[1], board_obj.n_moves%2] = True

        # if check line, update finished
        if ops.check_minibox_lines(board_obj, move):
            board_obj.miniboxes[move[0]//3, move[1]//3, board_obj.n_moves%2] = True
            board_obj.n_moves += 1
            return

        # check stale
        mini_board_idx = move[0]//3, move[1]//3

        if np.all(np.any(board_obj.markers[mini_board_idx[0]*3:(mini_board_idx[0]+1)*3,
                                           mini_board_idx[1]*3:(mini_board_idx[1]+1)*3],axis=2)):
            board_obj.miniboxes[mini_board_idx[0],mini_board_idx[1],2] = True
            
        # update history index
        board_obj.n_moves += 1
    
    @staticmethod
    def undo_move(board_obj: board_obj) -> None:
        if board_obj.n_moves == 0:
            print('no moves, returning null')
            return
        # update history and index
        _move = np.copy(board_obj.hist[board_obj.n_moves-1])

        board_obj.hist[board_obj.n_moves-1] = [0,0]
        
        # clear player markers (don't need to check for players)
        board_obj.markers[_move[0],_move[1],:] = False
        
        # open that miniboard (the move was either the last move on that board or it was already open)
        board_obj.miniboxes[_move[0]//3,_move[1]//3,:] = False
        
        # update index
        board_obj.n_moves -= 1
        

    @staticmethod
    def get_valid_moves(board_obj:board_obj) -> list[tuple[int]]:
        
        # all non-markered positions
        all_valid = (np.any(board_obj.markers,axis=2) == False)
        
        # initialization problem
        if board_obj.n_moves == 0:
            return list(zip(*np.where(all_valid)))

        # calculate last move's relative position
        _last_move = board_obj.hist[board_obj.n_moves-1]
        _rel_pos = _last_move[0] % 3, _last_move[1] % 3 # which minibox position is this
        
        # ---- 'play anywhere' branch -----
        # if minibox is finished
        if np.any(board_obj.miniboxes[_rel_pos[0],_rel_pos[1]]):
            # create "finished_box mask"
            finished_mask = np.zeros((9,9),dtype=bool)
            # loop through each finished box
            temp_in = np.any(board_obj.miniboxes,axis=2)
            for _box_finished_x, _box_finished_y, _flag in zip(np.arange(9)//3,np.arange(9)%3,temp_in.flatten()):
                if ~_flag:
                    finished_mask[_box_finished_x*3:(_box_finished_x+1)*3,
                                  _box_finished_y*3:(_box_finished_y+1)*3] = True
            return list(zip(*np.where(all_valid & finished_mask)))
        
        # mask to miniboard
        mini_mask = np.zeros((9,9),dtype=bool)
        mini_mask[_rel_pos[0]*3:(_rel_pos[0]+1)*3,
                  _rel_pos[1]*3:(_rel_pos[1]+1)*3] = True
        
        return list(zip(*np.where(all_valid & mini_mask)))

    @staticmethod
    def check_move_is_valid(board_obj: board_obj, move: tuple) -> bool:
        return move in ops.get_valid_moves(board_obj)
    
    @staticmethod
    def check_minibox_lines(board_obj: board_obj, move: int) -> bool:
        ''' checks whether the last move created a line '''
        # get player channel by move number
        _player_channel = board_obj.n_moves%2
        
        # select the minibox and relative position
        _temp_minibox_idx = move[0]//3, move[1]//3
        _rel_pos = move[0]%9, move[1]%9
        
        # the nested index below reduces the number of things to loop over
        _temp_mini = board_obj.markers[_temp_minibox_idx[0]*3:(_temp_minibox_idx[0]+1)*3,
                                       _temp_minibox_idx[1]*3:(_temp_minibox_idx[1]+1)*3,
                                       _player_channel]

        # check lines in that miniboard
        for _line in ops.lines_mask:
            if np.all(_temp_mini & _line == _line):
                return True
                
        return False
    
    @staticmethod
    def check_game_finished(board_obj: board_obj) -> bool:
        ''' not a check whether it IS finished, but if the most recent move finished it '''
        
        _player_channel = (board_obj.n_moves-1)%2
        
        # check if last active player made a line in the miniboxes
        for _line in ops.lines_mask:
            if np.all(board_obj.miniboxes[:,:,_player_channel] * _line == _line):
                # game is finished
                return True
        
        # all miniboxes filled
        # (if all of them are filled will return true, otherwise will return false
        return np.all(np.any(board_obj.miniboxes,axis=2))
    
    @staticmethod
    def pull_dictionary(board_obj: board_obj) -> dict:
        # dictionary, active miniboard, valid moves in the original format
        temp_dict = {}

        # make array (the main thing)
        temp_array = np.zeros((9,9))
        temp_array[board_obj.markers[:,:,0]] = 1
        temp_array[board_obj.markers[:,:,1]] = -1
        temp_dict['board_state'] = temp_array
        if board_obj.n_moves%2 == 1:
            temp_dict['board_state'] *= -1 # flip perspectives based on player
        
        # calculate active miniboard
        _last_move = board_obj.hist[board_obj.n_moves-1]
        _rel_pos = _last_move[0] % 3, _last_move[1] % 3

        if np.any(board_obj.miniboxes[_rel_pos[0],_rel_pos[1]]):
            temp_dict['active_box'] = (-1,-1)
        else:
            temp_dict['active_box'] = (_rel_pos[0],_rel_pos[1])

        # valid moves (converted to tuples)
        temp_dict['valid_moves'] = ops.get_valid_moves(board_obj)
        temp_dict['history'] = board_obj.hist
        temp_dict['n_moves'] = board_obj.n_moves
        temp_dict['markers'] = board_obj.markers
        temp_dict['miniboxes'] = board_obj.miniboxes
        return temp_dict
    
    @staticmethod # to be used infrequently, not efficient and rarely needed
    def get_winner(board_obj: board_obj) -> str:
        # check agent 1
        for _line in ops.lines_mask:
            if np.all(board_obj.miniboxes[:,:,0] * _line == _line):
                return 'agent 1 wins'

        # check agent 2
        for _line in ops.lines_mask:
            if np.all(board_obj.miniboxes[:,:,1] * _line == _line):
                return 'agent 2 wins'

        # check stale
        if np.all(np.any(board_obj.miniboxes,axis=2)): # if all miniboards are filled with something
            # return 'stale'
            #check who has more miniboards
            if np.sum(board_obj.miniboxes[:,:,0]) > np.sum(board_obj.miniboxes[:,:,1]):
                return 'agent 1 wins'
            elif np.sum(board_obj.miniboxes[:,:,0]) < np.sum(board_obj.miniboxes[:,:,1]):
                return 'agent 2 wins'
            else:
                return 'stale'

        return 'game is ongoing'
