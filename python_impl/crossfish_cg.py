import time
import numpy as np
import sys #used in codingame environment, not used here
class board_obj:
    def __init__(self):
        # the full board: two channels, one per player
        self.markers = np.zeros((9,9,2)).astype(bool)
        # an "open" location is calculated by ORing
        
        # the overall miniboard status
        self.miniboxes = np.zeros((3,3,3)).astype(bool)
        # channels: p1, p2, stale
        
        # board history
        self.hist = np.zeros((82,2),dtype=np.uint8)
        self.n_moves = 0
    def build_from_dict_gamestate(self, gamestate: dict):
        self.markers = gamestate['markers']
        self.miniboxes = gamestate['miniboxes']
        self.hist = gamestate['history']
        #add one extra element at the end of the history, to support null move pruning
        row_to_append = np.array([[0, 0]])
        # Append row to arr
        self.hist = np.vstack((self.hist, row_to_append))
        self.n_moves = gamestate['n_moves']
class crossfish:
    '''
    removed complicated hce in favor of simple minibox score
    test result:  W: 62974, L: 52620, D: 4814, elo diff: 29.95 +/- 3.86, LOS: 100.00
    '''
    def __init__(self):
        self.thinking_time = 0.095
        self.root_best_move = None
        self.start_time = None
        self.score = 0
        self.transposition_table = dict()
        self.killer_moves = np.zeros((128, 2), dtype=np.int8) #depth, move
        self.history_table = np.zeros((2, 9, 9), dtype=np.uint16) #player (0 or 1), miniboard, move
        self.nodes = 0
        self.lines_mask = np.array([[1,1,1,0,0,0,0,0,0], # horizontals
                                    [0,0,0,1,1,1,0,0,0],
                                    [0,0,0,0,0,0,1,1,1],
                                    [1,0,0,1,0,0,1,0,0], # verticals
                                    [0,1,0,0,1,0,0,1,0],
                                    [0,0,1,0,0,1,0,0,1],
                                    [1,0,0,0,1,0,0,0,1], # diagonals
                                    [0,0,1,0,1,0,1,0,0]],dtype=bool).reshape(-1,3,3)
                                             
    def move(self, board_dict: dict) -> tuple:
        ''' wrapper
        apply the logic and returns the desired move
        '''
        b_obj = board_obj()
        b_obj.build_from_dict_gamestate(board_dict)
        return self.get_best_move(b_obj)
    
    def get_best_move(self, board: board_obj):
        self.nodes = 0
        depth = 1
        self.start_time = time.time()
        aspiration = 2
        alpha = -np.inf
        beta = np.inf
        while time.time() - self.start_time < self.thinking_time and depth < 40:
            self.search(board, depth, 0, alpha, beta, True)
            aspiration *= 2
            if self.score <= alpha:
                alpha -= aspiration
            elif self.score >= beta:
                beta += aspiration
            else:
                aspiration = 2
                alpha = self.score - aspiration
                beta = self.score + aspiration
                depth += 1

        # print(f'depth: {depth-1}, score: {self.score}, node: {self.nodes}', file=sys.stderr, flush=True)
        return self.root_best_move
    
    def search(self, board:board_obj, depth:int, ply: int, alpha: int, beta: int, can_null:bool) -> int:
        '''negamax with alpha beta pruning'''
        if time.time() - self.start_time > self.thinking_time:
            return -np.inf
        
        self.nodes += 1
        pv_node = (alpha + 0.1 == beta)
        can_null = can_null and ply > 0
        
        if self.check_game_finished(board):
            if self.check_win_via_line(board.miniboxes, (board.n_moves + 1) % 2):
                return -100 + ply
            else:
                #if global board is stale, winner is the player with the most boxes
                return 50 *((-1)**(board.n_moves) * (np.sum(board.miniboxes[:, :, 0]) - np.sum(board.miniboxes[:, :, 1]))) 
        
        tt_move = None
        try:
            tt_entry = self.transposition_table[self.hash_position(board)]
            tt_move = tt_entry[3]
            if tt_entry[1] >= depth and not pv_node: 
                if (tt_entry[2] == 0): #exact score 
                    return tt_entry[0]
                elif tt_entry[2] == 2:
                    beta = min(beta, tt_entry[0])
                elif tt_entry[2] == 1:
                    alpha = max(alpha, tt_entry[0])
                if alpha >= beta:
                    return tt_entry[0]
        except KeyError:
            pass
        
        if depth == 0:
            return self.evaluate(board)
        
        can_futility_prune = False
        
        #dont do risky pruning stuff inside PV nodes
        if not pv_node:

            stand_pat = self.evaluate(board)

            #reverse futility pruning
            rfp_margin = 1
            if stand_pat - rfp_margin * depth >= beta:
                return beta

            # #futility pruning
            f_margin = 1
            can_futility_prune = (stand_pat + f_margin * depth) <= alpha
            
            #Null Move Pruning
            #TODO: twiddle with depth reduction formula
            if depth > 2 and can_null:
                self.make_null_move(board)
                null_move_score = -self.search(board, depth//2, ply+1, -beta, -alpha, can_null=False)
                self.undo_move(board)
                if null_move_score >= beta:
                    return beta

        legal_moves_and_scores = self.get_sorted_moves_and_scores(board, tt_move, ply)
        best_move = legal_moves_and_scores[0][0]
        alpha_orig = alpha
        max_val = -np.inf
        
        #main search loop
        for move_idx in range(len(legal_moves_and_scores)):
            move = legal_moves_and_scores[move_idx][0]
            move_score = legal_moves_and_scores[move_idx][1]
            if can_futility_prune and move_idx > 0 and move_score <= 0:
                continue
            self.make_move(board, move)
            if move_idx == 0:
                val = -self.search(board, depth-1, ply+1, -beta, -alpha, can_null=can_null)
            else:
                lmr_val = 1 #late move reductions, TODO: twiddle with this
                #idea: moves that are at the end of the sorted list are likely to be bad, so reduce their depth
                if move_score <= 0 and depth > 2:
                    lmr_val += depth // 2
                val = -self.search(board, max(0, depth - lmr_val), ply+1, -alpha-0.1, -alpha, can_null=can_null)
                if alpha < val and val < beta:
                    val = -self.search(board, depth-1, ply+1, -beta, -alpha, can_null=can_null)
            self.undo_move(board)
            if val > max_val:
                max_val = val
                best_move = move
                if ply == 0 and abs(val) != np.inf:
                    self.root_best_move = move
                    self.score = max_val
            alpha = max(alpha, max_val)
            if alpha >= beta:
                self.killer_moves[ply, :] = move
                self.history_table[board.n_moves % 2, move[0], move[1]] += depth ** 2
                break
        entry_bound_flag = 0
        if val <= alpha_orig:
            entry_bound_flag = 1
        elif val >= beta:
            entry_bound_flag = 2
        self.transposition_table[self.hash_position(board)] = (val, depth, entry_bound_flag, best_move)
        return alpha
    
    def evaluate(self, board):
        '''simple evaluation function'''
        return self.minibox_score(board)
    
    def minibox_score(self, board):
        score = (np.sum(board.miniboxes[:, :, 0]) - np.sum(board.miniboxes[:, :, 1])) * 3 * ((-1) ** board.n_moves)   
        return score
   
    def hash_position(self, board: board_obj) -> bytes:
        int_markers = board.markers.astype(np.int8)
        uniboard = int_markers[:,:,0] - int_markers[:,:, 1]
        #information needed to fully describe a board: markers and active square
        return uniboard.tobytes() + board.hist[board.n_moves-1][1].tobytes()    
    
    def pull_mini_boards(self, markers: np.array) -> np.array:
        ''' returns a (3,3,2) array of the miniboards '''
        # Reshape and transpose the markers array to get the desired shape
        return markers.reshape(3, 3, 3, 3, 2).transpose(0, 2, 1, 3, 4).reshape(9,3,3,2)
    
    def check_win_via_line(self, miniboxes:np.ndarray, p:int) -> bool:
        '''Check if player p has won the game'''
        for i in range(3):
            if np.sum(miniboxes[:,i, p]) == 3: return True # horizontal
            if np.sum(miniboxes[i,:, p]) == 3: return True # vertical

        # diagonals
        if miniboxes[:, :, p].trace() == 3: return True
        if np.rot90(miniboxes[:, :, p]).trace() == 3: return True
        return False
    
    def get_sorted_moves_and_scores(self, board: board_obj, tt_move: tuple, ply:int) -> list:
        legal_moves = self.get_valid_moves(board)
        legal_moves_and_scores = [(move, self.score_move(board, move, tt_move, ply)) for move in legal_moves]
        legal_moves_and_scores.sort(key=lambda x: x[1], reverse=True)
        return legal_moves_and_scores
    
    def score_move(self, board: board_obj, move: tuple, tt_move:tuple, ply:int) -> int:
        if move == tt_move:
            return np.inf
        if np.all(move == self.killer_moves[ply]):
            return 9999999
        score = 0
        score += self.history_table[board.n_moves % 2, move[0], move[1]]
        #if move completes a box, or blocks a line, give it more score
        markers = board.markers
        target_miniboard = markers[move[0], :, :].reshape(3,3,2)
        #check if sum along row for move of either player is 2
        if np.sum(target_miniboard[move[1] // 3, :, :]) == 2:
            score += 10
        #check if sum along column for move of either player is 2
        if np.sum(target_miniboard[:, move[1] % 3, :]) == 2:
            score += 10
        # 0 1 2
        # 3 4 5
        # 6 7 8
        if move[1] % 4 == 0 and np.sum(target_miniboard.diagonal()) == 2: #diagonal, left to right
            score += 10
        elif move[1] % 2 == 0 and np.sum(np.fliplr(target_miniboard).diagonal()) == 2: #diagonal, right to left
            score += 10
        #if a move sends the opponent to a completed box (i.e lets them go anywhere), give it a negative score
        done_boxes = board.miniboxes[:, :, 0] & board.miniboxes[:, :, 1] & board.miniboxes[:,:, 2]
        if done_boxes.flatten()[move[1]]:
            score -= 50
        
        return score
   
    def get_player(self, board_obj: board_obj) -> int:
        return board_obj.n_moves%2 
    
    def make_move(self, board_obj: board_obj, move:tuple) -> None:
        # NOTE: there is no safety check in here, we deal with error handling elsewhere.
        
        # update move history
        board_obj.hist[board_obj.n_moves] = move
        
        # update board for player
        board_obj.markers[move[0],move[1], board_obj.n_moves%2] = True

        # if check line, update finished
        if self.check_minibox_lines(board_obj, move):
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
    
    def make_null_move(self, board_obj: board_obj) -> None:
        '''for the purpose of null move pruning'''
        board_obj.n_moves += 1

    
    def unmake_null_move(self, board_obj: board_obj) -> None:
        '''for the purpose of null move pruning'''
        board_obj.n_moves -= 1
        


    def undo_move(self, board_obj: board_obj) -> None:
        if board_obj.n_moves == 0:
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
        

    def get_valid_moves(self, board_obj:board_obj) -> list[tuple[int]]:
        
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

    def check_move_is_valid(self, board_obj: board_obj, move: tuple) -> bool:
        return move in self.get_valid_moves(board_obj)

    def check_minibox_lines(self, board_obj: board_obj, move: int) -> bool:
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
        for _line in self.lines_mask:
            if np.all(_temp_mini & _line == _line):
                return True
                
        return False

    def check_game_finished(self, board_obj: board_obj) -> bool:
        ''' not a check whether it IS finished, but if the most recent move finished it '''
        
        _player_channel = (board_obj.n_moves-1)%2
        
        # check if last active player made a line in the miniboxes
        for _line in self.lines_mask:
            if np.all(board_obj.miniboxes[:,:,_player_channel] * _line == _line):
                # game is finished
                return True
        
        # all miniboxes filled
        # (if all of them are filled will return true, otherwise will return false
        return np.all(np.any(board_obj.miniboxes,axis=2))

    def pull_dictionary(self, board_obj: board_obj) -> dict:
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
        temp_dict['valid_moves'] = self.get_valid_moves(board_obj)
        temp_dict['history'] = board_obj.hist
        temp_dict['n_moves'] = board_obj.n_moves
        temp_dict['markers'] = board_obj.markers
        temp_dict['miniboxes'] = board_obj.miniboxes
        return temp_dict

    # to be used infrequently, not efficient and rarely needed
    def get_winner(self, board_obj: board_obj) -> str:
        # check agent 1
        for _line in self.lines_mask:
            if np.all(board_obj.miniboxes[:,:,0] * _line == _line):
                return 'agent 1 wins'

        # check agent 2
        for _line in self.lines_mask:
            if np.all(board_obj.miniboxes[:,:,1] * _line == _line):
                return 'agent 2 wins'
        #check stale
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

    

    

    
    
# game loop
b = board_obj()
crossfish = crossfish()
while True:
    opponent_row, opponent_col = [int(i) for i in input().split()]
    # valid_action_count = int(input())
    # valid_actions = []
    # for i in range(valid_action_count):
    #     row, col = [int(j) for j in input().split()]
    #     valid_actions.append((row, col))
    if opponent_row == -1:
        crossfish.make_move(b, (4, 4))
        print(4, 4)

    else:
        #select a random move
        crossfish.make_move(b, (opponent_row, opponent_col))
        move = crossfish.get_best_move(b)
        crossfish.make_move(b, move)
        print(move[0], move[1])
