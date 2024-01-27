import numpy as np
from operations import ops
from board import board_obj
class random_bot:
    '''
    this bot selects a random valid move
    '''
    def __init__(self, name = 'beep-boop'):
        self.name = name
    def move(self, board_dict):
        # print(board_dict['valid_moves'])
        b = board_obj()
        b.build_from_dict_gamestate(board_dict)
        # print(b.miniboxes)
        
        random_index = np.random.choice(len(board_dict['valid_moves']))
        return board_dict['valid_moves'][random_index]


class line_completer_bot:
    '''
    tries to complete lines, otherwise it plays randomly
    designed to show how to implement a relatively simple strategy
    '''
    
    ''' ------------------ required function ---------------- '''
    
    def __init__(self,name: str = 'Chekhov') -> None:
        self.name = name
        self.box_probs = np.ones((3,3)) # edges
        self.box_probs[1,1] = 4 # center
        self.box_probs[0,0] = self.box_probs[0,2] = self.box_probs[2,0] = self.box_probs[2,2] = 2 # corners
        
    def move(self, board_dict: dict) -> tuple:
        ''' wrapper
        apply the logic and returns the desired move
        '''
        return tuple(self.heuristic_mini_to_major(board_state = board_dict['board_state'],
                                                  active_box = board_dict['active_box'],
                                                  valid_moves = board_dict['valid_moves']))
    
    
    ''' --------- generally useful bot functions ------------ '''
    
    def _check_line(self, box: np.array) -> bool:
        '''
        box is a (3,3) array
        returns True if a line is found, else returns False '''
        for i in range(3):
            if abs(sum(box[:,i])) == 3: return True # horizontal
            if abs(sum(box[i,:])) == 3: return True # vertical

        # diagonals
        if abs(box.trace()) == 3: return True
        if abs(np.rot90(box).trace()) == 3: return True
        return False

    def _check_line_playerwise(self, box: np.array, player: int = None):
        ''' returns true if the given player has a line in the box, else false
        if no player is given, it checks for whether any player has a line in the box'''
        if player == None:
            return self._check_line(box)
        if player == -1:
            box = box * -1
        box = np.clip(box,0,1)
        return self._check_line(box)
    
    def pull_mini_board(self, board_state: np.array, mini_board_index: tuple) -> np.array:
        ''' extracts a mini board from the 9x9 given the its index'''
        temp = board_state[mini_board_index[0]*3:(mini_board_index[0]+1)*3,
                           mini_board_index[1]*3:(mini_board_index[1]+1)*3]
        return temp

    def get_valid(self, mini_board: np.array) -> np.array:
        ''' gets valid moves in the miniboard'''
#        print(mini_board)
#        print(np.where(mini_board == 0))
#        return np.where(mini_board == 0)
        return np.where(abs(mini_board) != 1)

    def get_finished(self, board_state: np.array) -> np.array:
        ''' calculates the completed boxes'''
        self_boxes = np.zeros((3,3))
        opp_boxes = np.zeros((3,3))
        stale_boxes = np.zeros((3,3))
        # look at each miniboard separately
        for _r in range(3):
            for _c in range(3):
                player_finished = False
                mini_board = self.pull_mini_board(board_state, (_r,_c))
                if self._check_line_playerwise(mini_board, player = 1):
                    self_boxes[_r,_c] = 1
                    player_finished = True
                if self._check_line_playerwise(mini_board, player = -1):
                    opp_boxes[_r,_c] = 1
                    player_finished = True
                if (sum(abs(mini_board.flatten())) == 9) and not player_finished:
                    stale_boxes[_r,_c] = 1

        # return finished boxes (separated by their content)
        return (self_boxes, opp_boxes, stale_boxes)
    
    def complete_line(self, mini_board: np.array) -> list:
        if sum(abs(mini_board.flatten())) == 9:
            print('invalid mini_board') # should never reach here
        # works as expected, however mini-board sometimes is sometimes invalid
        ''' completes a line if available '''
        # loop through valid moves with hypothetic self position there.
        # if it makes a line it's an imminent win
        imminent = list()
        valid_moves = self.get_valid(mini_board)
        for _valid in zip(*valid_moves):
            # create temp valid pattern
            valid_filter = np.zeros((3,3))
            valid_filter[_valid[0],_valid[1]] = 1
            if self._check_line(mini_board + valid_filter):
                imminent.append(_valid)
        return imminent
    
    def get_probs(self, valid_moves: list) -> np.array:
        ''' match the probability with the valid moves to weight the random choice '''
        valid_moves = np.array(valid_moves)
        probs = list()
        for _valid in np.array(valid_moves).reshape(-1,2):
            
            probs.append(self.box_probs[_valid[0],_valid[1]])
        probs /= sum(probs) # normalize
        return probs
    
    ''' ------------------ bot specific logic ---------------- '''
    
    def heuristic_mini_to_major(self,
                                board_state: np.array,
                                active_box: tuple,
                                valid_moves: list) -> tuple:
        '''
        either applies the heuristic to the mini-board or selects a mini-board (then applies the heuristic to it)
        '''
        if active_box != (-1,-1):
            # look just at the mini board
            mini_board = self.pull_mini_board(board_state, active_box)
            # look using the logic, select a move
            move = self.mid_heuristic(mini_board)
            # project back to original board space
            return (move[0] + 3 * active_box[0],
                    move[1] + 3 * active_box[1])

        else:
        #    print(np.array(valid_moves).shape) # sometimes the miniboard i'm sent to has no valid moves
        
            # use heuristic on finished boxes to select which box to play in
            imposed_active_box = self.major_heuristic(board_state)
#            print(self.pull_mini_board(board_state, imposed_active_box),'\n')
#            print('\n')

            # call this function with the self-imposed active box
            return self.heuristic_mini_to_major(board_state = board_state,
                                                active_box = imposed_active_box,
                                                valid_moves = valid_moves)

    def major_heuristic(self, board_state: np.array) -> tuple:
        '''
        determines which miniboard to play on
        note: having stale boxes was causing issues where the logic wanted to block
              the opponent but that mini-board was already finished (it was stale)
        '''
        z = self.get_finished(board_state)
        # finished boxes is a tuple of 3 masks: self, opponent, stale 
        self_boxes  = z[0]
        opp_boxes   = z[1]
        stale_boxes = z[2]
#        print('self:\n',self_boxes)
#        print('opp :\n',opp_boxes)
#        print('stale:\n',stale_boxes)
        
        # ----- identify imminent wins -----
        imminent_wins = self.complete_line(self_boxes)
#        print('len imminent win:',len(imminent_wins))
        # remove imminent wins that point to stale boxes (or opponent)
        stale_boxes_idxs = zip(*np.where(stale_boxes))
        for stale_box in stale_boxes_idxs:
            if stale_box in imminent_wins:
                imminent_wins.remove(stale_box)
        opp_boxes_idx = zip(*np.where(opp_boxes))
        for opp_box in opp_boxes_idx:
            if opp_box in imminent_wins:
                imminent_wins.remove(opp_box)
        # if it can complete a line, do it
        if len(imminent_wins) > 0: 
#            print('returning line')
#            print('len imminent win:',len(imminent_wins))
            return imminent_wins[np.random.choice(len(imminent_wins), p=self.get_probs(imminent_wins))]

        # ------ attempt to block -----
        imminent_loss = self.complete_line(opp_boxes)
        # make new list to remove imminent wins that point to stale boxes
        stale_boxes_idx = zip(*np.where(stale_boxes))
        for stale_box in stale_boxes_idx:
            if stale_box in imminent_loss:
                imminent_loss.remove(stale_box)
        self_boxes_idx = zip(*np.where(self_boxes))
        for self_box in self_boxes_idx:
            if self_box in imminent_loss:
                imminent_loss.remove(self_box)
        if len(imminent_loss) > 0:
#            print('returning block')
            return imminent_loss[np.random.choice(len(imminent_loss), p=self.get_probs(imminent_loss))]

        # ------ else take random ------
#        print('returning random')
        internal_valid = np.array(list(zip(*self.get_valid(self_boxes + opp_boxes + stale_boxes))))
        return tuple(internal_valid[np.random.choice(len(internal_valid), p=self.get_probs(internal_valid))])
        
    def mid_heuristic(self, mini_board: np.array) -> tuple:
        ''' main mini-board logic '''
        # try to complete a line on this miniboard
        imminent_wins = self.complete_line(mini_board)
        if len(imminent_wins) > 0:
            return imminent_wins[np.random.choice(len(imminent_wins))]

        ''' attempt to block'''
        imminent_wins = self.complete_line(mini_board * -1) # pretend to make lines from opponent's perspective
        if len(imminent_wins) > 0:
            return imminent_wins[np.random.choice(len(imminent_wins))]

        # else play randomly
        valid_moves = np.array(list(zip(*self.get_valid(mini_board))))
        return tuple(valid_moves[np.random.choice(len(valid_moves), p=self.get_probs(valid_moves))])


import time
class minimax_ref:
    def __init__(self,name: str = 'Minimax Reference') -> None:
        self.name = name
        self.thinking_time = 0.1
        self.root_best_move = None
        self.start_time = None
        self.score = 0
        self.maximizing_idx = 0
    def move(self, board_dict: dict) -> tuple:
        ''' wrapper
        apply the logic and returns the desired move
        '''
        b_obj = board_obj()
        b_obj.build_from_dict_gamestate(board_dict)
        self.maximizing_idx = b_obj.n_moves % 2
        return self.get_best_move(b_obj)
    
    def get_best_move(self, board: board_obj):
        depth = 1
        self.start_time = time.time()
        while time.time() - self.start_time < self.thinking_time:
            self.search(board, depth, True, 0)
            depth += 1
        # print(f'reached depth {depth-1} in {time.time() - self.start_time} seconds with score {self.score}')
        return self.root_best_move
    
    def search(self, board:board_obj, depth:int, maximizing_player:bool, ply: int) -> int:
        '''simple minimax search'''
        if ops.check_game_finished(board):
            if np.all(np.any(board.miniboxes,axis=2)):
                return 0 #draw
            else:
                if maximizing_player:
                    return -100 + ply
                else:
                    return 100 - ply
        if depth == 0:
            return self.evaluate(board)
        if time.time() - self.start_time > self.thinking_time:
            #want to immediately return and ignore results when out of time, so just turn node into a cutoff for its parent
            if maximizing_player:
                return np.inf
            else:
                return -np.inf
        if maximizing_player:
            max_value = -np.inf
            legal_moves = ops.get_valid_moves(board)
            for move in legal_moves:
                ops.make_move(board, move)
                new_value = self.search(board, depth-1, False, ply+1)
                if new_value > max_value:
                    max_value = new_value
                    if ply == 0:
                        self.root_best_move = move
                        self.score = max_value
                ops.undo_move(board)
            return max_value
        else:
            value = np.inf
            legal_moves = ops.get_valid_moves(board)
            for move in legal_moves:
                ops.make_move(board, move)
                value = min(value, self.search(board, depth-1, True, ply+1))
                ops.undo_move(board)
            return value
    def evaluate(self, board):
        '''simple evaluation function'''
        return self.minibox_score(board)
    def minibox_score(self, board):
        scores = [np.sum(board.miniboxes[:, :, p]) for p in range(2)]
        minibox_scores = scores[self.maximizing_idx] - scores[(self.maximizing_idx + 1) % 2]
        return minibox_scores

        
class ab_pruning_ref(minimax_ref):
    def get_best_move(self, board: board_obj):
        depth = 1
        self.start_time = time.time()
        while time.time() - self.start_time < self.thinking_time:
            self.search(board, depth, True, 0)
            depth += 1
        # print(f'reached depth {depth-1} in {time.time() - self.start_time} seconds with score {self.score}')
        return self.root_best_move
    def search(self, board:board_obj, depth:int, maximizing_player:bool, ply: int, alpha: int = -np.inf, beta: int = np.inf) -> int:
        '''simple minimax search'''
        if ops.check_game_finished(board):
            if np.all(np.any(board.miniboxes,axis=2)):
                return 0 #draw
            else:
                if maximizing_player:
                    return -100 + ply
                else:
                    return 100 - ply
        if depth == 0:
            return self.evaluate(board)
        if time.time() - self.start_time > self.thinking_time:
            if maximizing_player:
                return np.inf
            else:
                return -np.inf
        if maximizing_player:
            max_value = -np.inf
            legal_moves = ops.get_valid_moves(board)
            for move in legal_moves:
                ops.make_move(board, move)
                max_value = max(max_value, self.search(board, depth-1, False, ply+1, alpha, beta))
                ops.undo_move(board)
                if max_value > beta:
                    break
                if max_value > alpha:
                    alpha = max_value
                    if ply == 0:
                        self.root_best_move = move
                        self.score = max_value
            return max_value
        else:
            value = np.inf
            legal_moves = ops.get_valid_moves(board)
            for move in legal_moves:
                ops.make_move(board, move)
                value = min(value, self.search(board, depth-1, True, ply+1, alpha, beta))
                ops.undo_move(board)
                if value < alpha:
                    break
                beta = min(beta, value)
            return value

class transposition_table(ab_pruning_ref):
    def get_best_move(self, board: board_obj):
        depth = 1
        self.start_time = time.time()
        while time.time() - self.start_time < self.thinking_time:
            self.search(board, depth, True, 0)
            depth += 1
        # print(f'reached depth {depth-1} in {time.time() - self.start_time} seconds with score {self.score}')
        return self.root_best_move
    def __init__(self, name: str = 'Transposition Table'):
        self.name = name
        self.thinking_time = 0.1
        self.root_best_move = None
        self.start_time = None
        self.score = 0
        self.maximizing_idx = 0
        self.transposition_table = dict()
    def hash_position(self, board: board_obj) -> bytes:
        return board.markers.tobytes() + board.hist[board.n_moves-1][1].tobytes()
    def search(self, board:board_obj, depth:int, maximizing_player:bool, ply: int, alpha: int = -np.inf, beta: int = np.inf) -> int:
        '''simple minimax search'''
        if ops.check_game_finished(board):
            if np.all(np.any(board.miniboxes,axis=2)):
                return 0 #draw
            else:
                if maximizing_player:
                    return -100 + ply
                else:
                    return 100 - ply
        try:
            tt_entry = self.transposition_table[self.hash_position(board)]
            if tt_entry[1] >= depth and (tt_entry[2] == 0 #exact score
                                          #or tt_entry[2] == 2 and tt_entry[0] >= beta #lower bound
                                          #or tt_entry[2] == 1 and tt_entry[0] <= alpha
                                          ): #upper bound
                return tt_entry[0]
        except KeyError:
            pass
        
        if depth == 0:
            return self.evaluate(board)
        if time.time() - self.start_time > self.thinking_time:
            if maximizing_player:
                return np.inf
            else:
                return -np.inf
        if maximizing_player:
            max_value = -np.inf
            legal_moves = ops.get_valid_moves(board)
            for move in legal_moves:
                ops.make_move(board, move)
                max_value = max(max_value, self.search(board, depth-1, False, ply+1, alpha, beta))
                ops.undo_move(board)
                if max_value > beta:
                    break
                if max_value > alpha:
                    alpha = max_value
                    if ply == 0:
                        self.root_best_move = move
                        self.score = max_value
            entry_bound_flag = 0
            if max_value <= alpha:
                entry_bound_flag = 1
            elif max_value >= beta:
                entry_bound_flag = 2
            self.transposition_table[self.hash_position(board)] = (max_value, depth, entry_bound_flag)
            return max_value
        else:
            value = np.inf
            legal_moves = ops.get_valid_moves(board)
            for move in legal_moves:
                ops.make_move(board, move)
                value = min(value, self.search(board, depth-1, True, ply+1, alpha, beta))
                ops.undo_move(board)
                if value < alpha:
                    break
                beta = min(beta, value)
            entry_bound_flag = 0
            if value <= alpha:
                entry_bound_flag = 1
            elif value >= beta:
                entry_bound_flag = 2
            self.transposition_table[self.hash_position(board)] = (value, depth, entry_bound_flag)
            return value
    
class two_in_a_row_eval(transposition_table):
    def get_best_move(self, board: board_obj):
        depth = 1
        self.start_time = time.time()
        while time.time() - self.start_time < self.thinking_time:
            self.search(board, depth, True, 0)
            depth += 1
        # print(f'reached depth {depth-1} in {time.time() - self.start_time} seconds with score {self.score}')
        return self.root_best_move
    def pull_mini_boards(self, markers: np.array) -> np.array:
        ''' returns a (3,3,2) array of the miniboards '''
        # Reshape and transpose the markers array to get the desired shape
        return markers.reshape(3, 3, 3, 3, 2).transpose(0, 2, 1, 3, 4).reshape(9,3,3,2)


    def evaluate(self, board):
        '''simple evaluation function'''
        return self.minibox_score(board)

    def minibox_score(self, board):
        scores = [np.sum(board.miniboxes[:, :, p]) for p in range(2)]
        minibox_scores = scores[self.maximizing_idx] - scores[(self.maximizing_idx + 1) % 2]
        #add in a bonus for unblocked two in a rows
        #structure: first element is the attacking pattern (two in a row), second element is the blocking pattern
        two_in_a_row_masks = np.array([
                                        [[[1,1,0],
                                        [0,0,0],
                                        [0,0,0]],
                                        [[0,0,1],
                                        [0,0,0],
                                        [0,0,0]]
                                        ],
                                        [[[1,0,1],
                                        [0,0,0],
                                        [0,0,0]],
                                        [[0,1,0],
                                        [0,0,0],
                                        [0,0,0]]
                                        ],

                                        [[[0,0,0],
                                        [1,1,0],
                                        [0,0,0]],
                                        [[0,0,0],
                                        [0,0,1],
                                        [0,0,0]]
                                        ],
                                        [[[0,0,0],
                                        [1,0,1],
                                        [0,0,0]],
                                        [[0,0,0],
                                        [0,1,0],
                                        [0,0,0]]
                                        ],

                                        [[[0,0,0],
                                        [0,0,0],
                                        [1,1,0]],
                                        [[0,0,0],
                                        [0,0,0],
                                        [0,0,1]]],

                                        [[[0,0,0],
                                        [0,0,0],
                                        [1,0,1]],
                                        [[0,0,0],
                                        [0,0,0],
                                        [0,1,0]]],

                                        [[[0,1,1],
                                        [0,0,0],
                                        [0,0,0]],
                                        [[1,0,0],
                                        [0,0,0],
                                        [0,0,0]]],
                                        
               
                                        [[[0,0,0],
                                        [0,1,1],
                                        [0,0,0]],
                                        [[0,0,0],
                                        [1,0,0],
                                        [0,0,0]]],

                                        [[[0,0,0],
                                        [0,0,0],
                                        [0,1,1]],
                                        [[0,0,0],
                                        [0,0,0],
                                        [1,0,0]]],

                                        [[[1,0,0],
                                         [1,0,0],       
                                         [0,0,0]],
                                         [[0,0,0],
                                         [0,0,0],
                                         [1,0,0]]],

                                        [[[0,1,0],
                                         [0,1,0],
                                         [0,0,0]],
                                         [[0,0,0],
                                         [0,0,0],
                                         [0,1,0]]],
                                        [[[0,1,0],
                                         [0,0,0],
                                         [0,1,0]],
                                         [[0,0,0],
                                         [0,1,0],
                                         [0,0,0]]],

                                        [[[0,0,1],
                                         [0,0,1],
                                         [0,0,0]],
                                         [[0,0,0],
                                         [0,0,0],
                                         [0,0,1]],],

                                        [[[0,0,1],
                                         [0,0,0],
                                         [0,0,1]],
                                         [[0,0,0],
                                         [0,0,1],
                                         [0,0,0]],],

                                         [[[0,0,0],
                                         [1,0,0],
                                         [1,0,0]],
                                         [[1,0,0],
                                         [0,0,0],
                                         [0,0,0]]],

                                         [[[1,0,0],
                                         [0,0,0],
                                         [1,0,0]],
                                         [[0,0,0],
                                         [1,0,0],
                                         [0,0,0]]],

                                        [[[0,0,0],
                                         [0,1,0],
                                         [0,1,0]],
                                         [[0,1,0],
                                         [0,0,0],
                                         [0,0,0]],],

                                        [[[0,0,0],
                                         [0,0,1],
                                         [0,0,1]],
                                         [[0,0,1],
                                         [0,0,0],
                                         [0,0,0]]],

                                        [[[1,0,0],
                                        [0,1,0],
                                        [0,0,0]],
                                        [[0,0,0],
                                        [0,0,0],
                                        [0,0,1]]],

                                        [[[0,0,0],
                                        [0,1,0],
                                        [0,0,1]],
                                        [[1,0,0],
                                        [0,0,0],
                                        [0,0,0]]],

                                        [[[0,0,1],
                                        [0,1,0],
                                        [0,0,0]],
                                       [[0,0,0],
                                        [0,0,0],
                                        [1,0,0]] ],

                                        [[[0,0,1],
                                        [0,0,0],
                                        [1,0,0]],
                                       [[0,0,0],
                                        [0,1,0],
                                        [0,0,0]] ],

                                        [[[0,0,0],
                                        [0,1,0],
                                        [1,0,0]],
                                        [[0,0,1],
                                        [0,0,0],
                                        [0,0,0]]],

                                        [[[0,0,1],
                                        [0,0,0],
                                        [1,0,0]],
                                        [[0,0,0],
                                        [0,1,0],
                                        [0,0,0]]],

                                         ]).astype(bool)
        miniboards = self.pull_mini_boards(board.markers)
        close_to_win_scores = np.zeros(2)
        boxes_in_play = ~board.miniboxes[:,:,0] & ~board.miniboxes[:,:,1] & ~board.miniboxes[:,:,2]
        '''TODO: FIND A WAY TO OPTIMIZE THIS'''
        for p in range(2):
            for miniboard in miniboards[boxes_in_play.flatten()]:
                for mask in two_in_a_row_masks:
                    attacker = mask[0]
                    blocker = mask[1]
                    if np.all(miniboard[:,:, p][attacker]) and not np.any(miniboard[:,:,(p + 1) % 2][blocker]):
                        close_to_win_scores[p] += 0.5
                        break
        minibox_scores += close_to_win_scores[self.maximizing_idx] - close_to_win_scores[(self.maximizing_idx + 1) % 2]

        
        return minibox_scores

class tt_cutoffs:
    def __init__(self, name: str = 'Transposition Table'):
        self.name = name
        self.thinking_time = 0.1
        self.root_best_move = None
        self.start_time = None
        self.score = 0
        self.maximizing_idx = 0
        self.transposition_table = dict()
    def move(self, board_dict: dict) -> tuple:
        ''' wrapper
        apply the logic and returns the desired move
        '''
        b_obj = board_obj()
        b_obj.build_from_dict_gamestate(board_dict)
        self.maximizing_idx = b_obj.n_moves % 2
        return self.get_best_move(b_obj)
    
    def get_best_move(self, board: board_obj):
        depth = 1
        self.start_time = time.time()
        while time.time() - self.start_time < self.thinking_time:
            self.search(board, depth, True, 0)
            depth += 1
        # print(f'reached depth {depth-1} in {time.time() - self.start_time} seconds with score {self.score}')
        return self.root_best_move
    
    def hash_position(self, board: board_obj) -> bytes:
        # int_markers = board.markers.astype(np.uint8)
        # uniboard = int_markers[:,:,self.maximizing_idx] - int_markers[:,:,((self.maximizing_idx + 1) % 2)]
        # return uniboard.tobytes() + board.hist[board.n_moves-1][1].tobytes()
        return board.markers.tobytes() + board.hist[board.n_moves-1][1].tobytes() #consider switching to above on performance testing. Uses less space but more time
    def pull_mini_boards(self, markers: np.array) -> np.array:
        ''' returns a (3,3,2) array of the miniboards '''
        # Reshape and transpose the markers array to get the desired shape
        return markers.reshape(3, 3, 3, 3, 2).transpose(0, 2, 1, 3, 4).reshape(9,3,3,2)

    
    def search(self, board:board_obj, depth:int, maximizing_player:bool, ply: int, alpha: int = -np.inf, beta: int = np.inf) -> int:
        '''simple minimax search'''
        if ops.check_game_finished(board):
            if np.all(np.any(board.miniboxes,axis=2)):
                return 0 #draw
            else:
                if maximizing_player:
                    return -100 + ply
                else:
                    return 100 - ply
        try:
            tt_entry = self.transposition_table[self.hash_position(board)]
            if tt_entry[1] >= depth and (tt_entry[2] == 0 #exact score
                                          or tt_entry[2] == 2 and tt_entry[0] >= beta #lower bound
                                          or tt_entry[2] == 1 and tt_entry[0] <= alpha #upper bound
                                          ): 
                return tt_entry[0]
        except KeyError:
            pass
        
        if depth == 0:
            return self.evaluate(board)
        if time.time() - self.start_time > self.thinking_time:
            if maximizing_player:
                return np.inf
            else:
                return -np.inf
        if maximizing_player:
            max_value = -np.inf
            legal_moves = ops.get_valid_moves(board)
            for move in legal_moves:
                ops.make_move(board, move)
                max_value = max(max_value, self.search(board, depth-1, False, ply+1, alpha, beta))
                ops.undo_move(board)
                if max_value > beta:
                    break
                if max_value > alpha:
                    alpha = max_value
                    if ply == 0:
                        self.root_best_move = move
                        self.score = max_value
            entry_bound_flag = 0
            if max_value <= alpha:
                entry_bound_flag = 1
            elif max_value >= beta:
                entry_bound_flag = 2
            self.transposition_table[self.hash_position(board)] = (max_value, depth, entry_bound_flag)
            return max_value
        else:
            value = np.inf
            legal_moves = ops.get_valid_moves(board)
            for move in legal_moves:
                ops.make_move(board, move)
                value = min(value, self.search(board, depth-1, True, ply+1, alpha, beta))
                ops.undo_move(board)
                if value < alpha:
                    break
                beta = min(beta, value)
            entry_bound_flag = 0
            if value <= alpha:
                entry_bound_flag = 1
            elif value >= beta:
                entry_bound_flag = 2
            self.transposition_table[self.hash_position(board)] = (value, depth, entry_bound_flag)
            return value
    def evaluate(self, board):
        '''simple evaluation function'''
        return self.minibox_score(board)

    def minibox_score(self, board):
        scores = [np.sum(board.miniboxes[:, :, p]) for p in range(2)]
        minibox_scores = scores[self.maximizing_idx] - scores[(self.maximizing_idx + 1) % 2]
        #add in a bonus for unblocked two in a rows
        #structure: first element is the attacking pattern (two in a row), second element is the blocking pattern
        two_in_a_row_masks = np.array([
                                        [[[1,1,0],
                                        [0,0,0],
                                        [0,0,0]],
                                        [[0,0,1],
                                        [0,0,0],
                                        [0,0,0]]
                                        ],
                                        [[[1,0,1],
                                        [0,0,0],
                                        [0,0,0]],
                                        [[0,1,0],
                                        [0,0,0],
                                        [0,0,0]]
                                        ],

                                        [[[0,0,0],
                                        [1,1,0],
                                        [0,0,0]],
                                        [[0,0,0],
                                        [0,0,1],
                                        [0,0,0]]
                                        ],
                                        [[[0,0,0],
                                        [1,0,1],
                                        [0,0,0]],
                                        [[0,0,0],
                                        [0,1,0],
                                        [0,0,0]]
                                        ],

                                        [[[0,0,0],
                                        [0,0,0],
                                        [1,1,0]],
                                        [[0,0,0],
                                        [0,0,0],
                                        [0,0,1]]],

                                        [[[0,0,0],
                                        [0,0,0],
                                        [1,0,1]],
                                        [[0,0,0],
                                        [0,0,0],
                                        [0,1,0]]],

                                        [[[0,1,1],
                                        [0,0,0],
                                        [0,0,0]],
                                        [[1,0,0],
                                        [0,0,0],
                                        [0,0,0]]],
                                        
               
                                        [[[0,0,0],
                                        [0,1,1],
                                        [0,0,0]],
                                        [[0,0,0],
                                        [1,0,0],
                                        [0,0,0]]],

                                        [[[0,0,0],
                                        [0,0,0],
                                        [0,1,1]],
                                        [[0,0,0],
                                        [0,0,0],
                                        [1,0,0]]],

                                        [[[1,0,0],
                                         [1,0,0],       
                                         [0,0,0]],
                                         [[0,0,0],
                                         [0,0,0],
                                         [1,0,0]]],

                                        [[[0,1,0],
                                         [0,1,0],
                                         [0,0,0]],
                                         [[0,0,0],
                                         [0,0,0],
                                         [0,1,0]]],
                                        [[[0,1,0],
                                         [0,0,0],
                                         [0,1,0]],
                                         [[0,0,0],
                                         [0,1,0],
                                         [0,0,0]]],

                                        [[[0,0,1],
                                         [0,0,1],
                                         [0,0,0]],
                                         [[0,0,0],
                                         [0,0,0],
                                         [0,0,1]],],

                                        [[[0,0,1],
                                         [0,0,0],
                                         [0,0,1]],
                                         [[0,0,0],
                                         [0,0,1],
                                         [0,0,0]],],

                                         [[[0,0,0],
                                         [1,0,0],
                                         [1,0,0]],
                                         [[1,0,0],
                                         [0,0,0],
                                         [0,0,0]]],

                                         [[[1,0,0],
                                         [0,0,0],
                                         [1,0,0]],
                                         [[0,0,0],
                                         [1,0,0],
                                         [0,0,0]]],

                                        [[[0,0,0],
                                         [0,1,0],
                                         [0,1,0]],
                                         [[0,1,0],
                                         [0,0,0],
                                         [0,0,0]],],

                                        [[[0,0,0],
                                         [0,0,1],
                                         [0,0,1]],
                                         [[0,0,1],
                                         [0,0,0],
                                         [0,0,0]]],

                                        [[[1,0,0],
                                        [0,1,0],
                                        [0,0,0]],
                                        [[0,0,0],
                                        [0,0,0],
                                        [0,0,1]]],

                                        [[[0,0,0],
                                        [0,1,0],
                                        [0,0,1]],
                                        [[1,0,0],
                                        [0,0,0],
                                        [0,0,0]]],

                                        [[[0,0,1],
                                        [0,1,0],
                                        [0,0,0]],
                                       [[0,0,0],
                                        [0,0,0],
                                        [1,0,0]] ],

                                        [[[0,0,1],
                                        [0,0,0],
                                        [1,0,0]],
                                       [[0,0,0],
                                        [0,1,0],
                                        [0,0,0]] ],

                                        [[[0,0,0],
                                        [0,1,0],
                                        [1,0,0]],
                                        [[0,0,1],
                                        [0,0,0],
                                        [0,0,0]]],

                                        [[[0,0,1],
                                        [0,0,0],
                                        [1,0,0]],
                                        [[0,0,0],
                                        [0,1,0],
                                        [0,0,0]]],

                                         ]).astype(bool)
        miniboards = self.pull_mini_boards(board.markers)
        close_to_win_scores = np.zeros(2)
        boxes_in_play = ~board.miniboxes[:,:,0] & ~board.miniboxes[:,:,1] & ~board.miniboxes[:,:,2]
        '''TODO: FIND A WAY TO OPTIMIZE THIS'''
        for p in range(2):
            for miniboard in miniboards[boxes_in_play.flatten()]:
                for mask in two_in_a_row_masks:
                    attacker = mask[0]
                    blocker = mask[1]
                    if np.all(miniboard[:,:, p][attacker]) and not np.any(miniboard[:,:,(p + 1) % 2][blocker]):
                        close_to_win_scores[p] += 0.5
                        break
        minibox_scores += close_to_win_scores[self.maximizing_idx] - close_to_win_scores[(self.maximizing_idx + 1) % 2]

        
        return minibox_scores

class tt_move_ordering:
    def __init__(self, name: str = 'Transposition Table'):
        self.name = name
        self.thinking_time = 0.1
        self.root_best_move = None
        self.start_time = None
        self.score = 0
        self.maximizing_idx = 0
        self.transposition_table = dict()
    def move(self, board_dict: dict) -> tuple:
        ''' wrapper
        apply the logic and returns the desired move
        '''
        b_obj = board_obj()
        b_obj.build_from_dict_gamestate(board_dict)
        self.maximizing_idx = b_obj.n_moves % 2
        return self.get_best_move(b_obj)
    
    def get_best_move(self, board: board_obj):
        depth = 1
        self.start_time = time.time()
        while time.time() - self.start_time < self.thinking_time:
            self.search(board, depth, True, 0)
            depth += 1
        # print(f'reached depth {depth-1} in {time.time() - self.start_time} seconds with score {self.score}')
        return self.root_best_move
    
    def hash_position(self, board: board_obj) -> bytes:
        # int_markers = board.markers.astype(np.uint8)
        # uniboard = int_markers[:,:,self.maximizing_idx] - int_markers[:,:,((self.maximizing_idx + 1) % 2)]
        # return uniboard.tobytes() + board.hist[board.n_moves-1][1].tobytes()
        return board.markers.tobytes() + board.hist[board.n_moves-1][1].tobytes() #consider switching to above on performance testing. Uses less space but more time
    def pull_mini_boards(self, markers: np.array) -> np.array:
        ''' returns a (3,3,2) array of the miniboards '''
        # Reshape and transpose the markers array to get the desired shape
        return markers.reshape(3, 3, 3, 3, 2).transpose(0, 2, 1, 3, 4).reshape(9,3,3,2)

    
    def search(self, board:board_obj, depth:int, maximizing_player:bool, ply: int, alpha: int = -np.inf, beta: int = np.inf) -> int:
        '''simple minimax search'''
        if ops.check_game_finished(board):
            if np.all(np.any(board.miniboxes,axis=2)):
                return 0 #draw
            else:
                if maximizing_player:
                    return -100 + ply
                else:
                    return 100 - ply
        tt_move = None
        try:
            tt_entry = self.transposition_table[self.hash_position(board)]
            tt_move = tt_entry[3]
            if tt_entry[1] >= depth and (tt_entry[2] == 0 #exact score
                                          or tt_entry[2] == 2 and tt_entry[0] >= beta #lower bound
                                          or tt_entry[2] == 1 and tt_entry[0] <= alpha #upper bound
                                          ): 
                return tt_entry[0]
        except KeyError:
            pass
        
        if depth == 0:
            return self.evaluate(board)
        if time.time() - self.start_time > self.thinking_time:
            if maximizing_player:
                return np.inf
            else:
                return -np.inf
        legal_moves = ops.get_valid_moves(board)
        
        if tt_move is not None:
            for i in range(len(legal_moves)):
                if legal_moves[i] == tt_move:
                    legal_moves[0], legal_moves[i] = legal_moves[i], legal_moves[0]
                    break
        
        if maximizing_player:
            max_value = -np.inf
            best_move = legal_moves[0]
            for move in legal_moves:
                ops.make_move(board, move)
                max_value = max(max_value, self.search(board, depth-1, False, ply+1, alpha, beta))
                ops.undo_move(board)
                if max_value > beta:
                    break
                if max_value > alpha:
                    alpha = max_value
                    best_move = move
                    if ply == 0:
                        self.root_best_move = move
                        self.score = max_value
            entry_bound_flag = 0
            if max_value <= alpha:
                entry_bound_flag = 1
            elif max_value >= beta:
                entry_bound_flag = 2
            self.transposition_table[self.hash_position(board)] = (max_value, depth, entry_bound_flag, best_move)
            return max_value
        
        else:
            value = np.inf
            best_move = legal_moves[0]
            for move in legal_moves:
                ops.make_move(board, move)
                value = min(value, self.search(board, depth-1, True, ply+1, alpha, beta))
                ops.undo_move(board)
                if value < alpha:
                    break
                # beta = min(beta, value)
                if value < beta:
                    beta = value
                    best_move = move
            entry_bound_flag = 0
            if value <= alpha:
                entry_bound_flag = 1
            elif value >= beta:
                entry_bound_flag = 2
            self.transposition_table[self.hash_position(board)] = (value, depth, entry_bound_flag, best_move)
            return value
    
    def evaluate(self, board):
        '''simple evaluation function'''
        return self.minibox_score(board)

    def minibox_score(self, board):
        scores = [np.sum(board.miniboxes[:, :, p]) for p in range(2)]
        minibox_scores = scores[self.maximizing_idx] - scores[(self.maximizing_idx + 1) % 2]
        #add in a bonus for unblocked two in a rows
        #structure: first element is the attacking pattern (two in a row), second element is the blocking pattern
        two_in_a_row_masks = np.array([
                                        [[[1,1,0],
                                        [0,0,0],
                                        [0,0,0]],
                                        [[0,0,1],
                                        [0,0,0],
                                        [0,0,0]]
                                        ],
                                        [[[1,0,1],
                                        [0,0,0],
                                        [0,0,0]],
                                        [[0,1,0],
                                        [0,0,0],
                                        [0,0,0]]
                                        ],

                                        [[[0,0,0],
                                        [1,1,0],
                                        [0,0,0]],
                                        [[0,0,0],
                                        [0,0,1],
                                        [0,0,0]]
                                        ],
                                        [[[0,0,0],
                                        [1,0,1],
                                        [0,0,0]],
                                        [[0,0,0],
                                        [0,1,0],
                                        [0,0,0]]
                                        ],

                                        [[[0,0,0],
                                        [0,0,0],
                                        [1,1,0]],
                                        [[0,0,0],
                                        [0,0,0],
                                        [0,0,1]]],

                                        [[[0,0,0],
                                        [0,0,0],
                                        [1,0,1]],
                                        [[0,0,0],
                                        [0,0,0],
                                        [0,1,0]]],

                                        [[[0,1,1],
                                        [0,0,0],
                                        [0,0,0]],
                                        [[1,0,0],
                                        [0,0,0],
                                        [0,0,0]]],
                                        
               
                                        [[[0,0,0],
                                        [0,1,1],
                                        [0,0,0]],
                                        [[0,0,0],
                                        [1,0,0],
                                        [0,0,0]]],

                                        [[[0,0,0],
                                        [0,0,0],
                                        [0,1,1]],
                                        [[0,0,0],
                                        [0,0,0],
                                        [1,0,0]]],

                                        [[[1,0,0],
                                         [1,0,0],       
                                         [0,0,0]],
                                         [[0,0,0],
                                         [0,0,0],
                                         [1,0,0]]],

                                        [[[0,1,0],
                                         [0,1,0],
                                         [0,0,0]],
                                         [[0,0,0],
                                         [0,0,0],
                                         [0,1,0]]],
                                        [[[0,1,0],
                                         [0,0,0],
                                         [0,1,0]],
                                         [[0,0,0],
                                         [0,1,0],
                                         [0,0,0]]],

                                        [[[0,0,1],
                                         [0,0,1],
                                         [0,0,0]],
                                         [[0,0,0],
                                         [0,0,0],
                                         [0,0,1]],],

                                        [[[0,0,1],
                                         [0,0,0],
                                         [0,0,1]],
                                         [[0,0,0],
                                         [0,0,1],
                                         [0,0,0]],],

                                         [[[0,0,0],
                                         [1,0,0],
                                         [1,0,0]],
                                         [[1,0,0],
                                         [0,0,0],
                                         [0,0,0]]],

                                         [[[1,0,0],
                                         [0,0,0],
                                         [1,0,0]],
                                         [[0,0,0],
                                         [1,0,0],
                                         [0,0,0]]],

                                        [[[0,0,0],
                                         [0,1,0],
                                         [0,1,0]],
                                         [[0,1,0],
                                         [0,0,0],
                                         [0,0,0]],],

                                        [[[0,0,0],
                                         [0,0,1],
                                         [0,0,1]],
                                         [[0,0,1],
                                         [0,0,0],
                                         [0,0,0]]],

                                        [[[1,0,0],
                                        [0,1,0],
                                        [0,0,0]],
                                        [[0,0,0],
                                        [0,0,0],
                                        [0,0,1]]],

                                        [[[0,0,0],
                                        [0,1,0],
                                        [0,0,1]],
                                        [[1,0,0],
                                        [0,0,0],
                                        [0,0,0]]],

                                        [[[0,0,1],
                                        [0,1,0],
                                        [0,0,0]],
                                       [[0,0,0],
                                        [0,0,0],
                                        [1,0,0]] ],

                                        [[[0,0,1],
                                        [0,0,0],
                                        [1,0,0]],
                                       [[0,0,0],
                                        [0,1,0],
                                        [0,0,0]] ],

                                        [[[0,0,0],
                                        [0,1,0],
                                        [1,0,0]],
                                        [[0,0,1],
                                        [0,0,0],
                                        [0,0,0]]],

                                        [[[0,0,1],
                                        [0,0,0],
                                        [1,0,0]],
                                        [[0,0,0],
                                        [0,1,0],
                                        [0,0,0]]],

                                         ]).astype(bool)
        miniboards = self.pull_mini_boards(board.markers)
        close_to_win_scores = np.zeros(2)
        boxes_in_play = ~board.miniboxes[:,:,0] & ~board.miniboxes[:,:,1] & ~board.miniboxes[:,:,2]
        '''TODO: FIND A WAY TO OPTIMIZE THIS'''
        for p in range(2):
            for miniboard in miniboards[boxes_in_play.flatten()]:
                for mask in two_in_a_row_masks:
                    attacker = mask[0]
                    blocker = mask[1]
                    if np.all(miniboard[:,:, p][attacker]) and not np.any(miniboard[:,:,(p + 1) % 2][blocker]):
                        close_to_win_scores[p] += 0.5
                        break
        minibox_scores += close_to_win_scores[self.maximizing_idx] - close_to_win_scores[(self.maximizing_idx + 1) % 2]

        
        return minibox_scores
    


class non_tt_move_ordering:
    def __init__(self, name: str = 'Transposition Table'):
        self.name = name
        self.thinking_time = 0.1
        self.root_best_move = None
        self.start_time = None
        self.score = 0
        self.maximizing_idx = 0
        self.transposition_table = dict()
        self.two_in_a_row_masks = np.array([
                                        [[[1,1,0],
                                        [0,0,0],
                                        [0,0,0]],
                                        [[0,0,1],
                                        [0,0,0],
                                        [0,0,0]]
                                        ],
                                        [[[1,0,1],
                                        [0,0,0],
                                        [0,0,0]],
                                        [[0,1,0],
                                        [0,0,0],
                                        [0,0,0]]
                                        ],

                                        [[[0,0,0],
                                        [1,1,0],
                                        [0,0,0]],
                                        [[0,0,0],
                                        [0,0,1],
                                        [0,0,0]]
                                        ],
                                        [[[0,0,0],
                                        [1,0,1],
                                        [0,0,0]],
                                        [[0,0,0],
                                        [0,1,0],
                                        [0,0,0]]
                                        ],

                                        [[[0,0,0],
                                        [0,0,0],
                                        [1,1,0]],
                                        [[0,0,0],
                                        [0,0,0],
                                        [0,0,1]]],

                                        [[[0,0,0],
                                        [0,0,0],
                                        [1,0,1]],
                                        [[0,0,0],
                                        [0,0,0],
                                        [0,1,0]]],

                                        [[[0,1,1],
                                        [0,0,0],
                                        [0,0,0]],
                                        [[1,0,0],
                                        [0,0,0],
                                        [0,0,0]]],
                                        
               
                                        [[[0,0,0],
                                        [0,1,1],
                                        [0,0,0]],
                                        [[0,0,0],
                                        [1,0,0],
                                        [0,0,0]]],

                                        [[[0,0,0],
                                        [0,0,0],
                                        [0,1,1]],
                                        [[0,0,0],
                                        [0,0,0],
                                        [1,0,0]]],

                                        [[[1,0,0],
                                         [1,0,0],       
                                         [0,0,0]],
                                         [[0,0,0],
                                         [0,0,0],
                                         [1,0,0]]],

                                        [[[0,1,0],
                                         [0,1,0],
                                         [0,0,0]],
                                         [[0,0,0],
                                         [0,0,0],
                                         [0,1,0]]],
                                        [[[0,1,0],
                                         [0,0,0],
                                         [0,1,0]],
                                         [[0,0,0],
                                         [0,1,0],
                                         [0,0,0]]],

                                        [[[0,0,1],
                                         [0,0,1],
                                         [0,0,0]],
                                         [[0,0,0],
                                         [0,0,0],
                                         [0,0,1]],],

                                        [[[0,0,1],
                                         [0,0,0],
                                         [0,0,1]],
                                         [[0,0,0],
                                         [0,0,1],
                                         [0,0,0]],],

                                         [[[0,0,0],
                                         [1,0,0],
                                         [1,0,0]],
                                         [[1,0,0],
                                         [0,0,0],
                                         [0,0,0]]],

                                         [[[1,0,0],
                                         [0,0,0],
                                         [1,0,0]],
                                         [[0,0,0],
                                         [1,0,0],
                                         [0,0,0]]],

                                        [[[0,0,0],
                                         [0,1,0],
                                         [0,1,0]],
                                         [[0,1,0],
                                         [0,0,0],
                                         [0,0,0]],],

                                        [[[0,0,0],
                                         [0,0,1],
                                         [0,0,1]],
                                         [[0,0,1],
                                         [0,0,0],
                                         [0,0,0]]],

                                        [[[1,0,0],
                                        [0,1,0],
                                        [0,0,0]],
                                        [[0,0,0],
                                        [0,0,0],
                                        [0,0,1]]],

                                        [[[0,0,0],
                                        [0,1,0],
                                        [0,0,1]],
                                        [[1,0,0],
                                        [0,0,0],
                                        [0,0,0]]],

                                        [[[0,0,1],
                                        [0,1,0],
                                        [0,0,0]],
                                       [[0,0,0],
                                        [0,0,0],
                                        [1,0,0]] ],

                                        [[[0,0,1],
                                        [0,0,0],
                                        [1,0,0]],
                                       [[0,0,0],
                                        [0,1,0],
                                        [0,0,0]] ],

                                        [[[0,0,0],
                                        [0,1,0],
                                        [1,0,0]],
                                        [[0,0,1],
                                        [0,0,0],
                                        [0,0,0]]],

                                        [[[0,0,1],
                                        [0,0,0],
                                        [1,0,0]],
                                        [[0,0,0],
                                        [0,1,0],
                                        [0,0,0]]],

                                         ]).astype(bool)
        self.flattened_attack_masks = np.stack([mask[0].flatten() for mask in self.two_in_a_row_masks])
        self.flattened_blocking_masks = np.stack([mask[1].flatten() for mask in self.two_in_a_row_masks])
    def move(self, board_dict: dict) -> tuple:
        ''' wrapper
        apply the logic and returns the desired move
        '''
        b_obj = board_obj()
        b_obj.build_from_dict_gamestate(board_dict)
        self.maximizing_idx = b_obj.n_moves % 2
        return self.get_best_move(b_obj)
    
    def get_best_move(self, board: board_obj):
        depth = 1
        self.start_time = time.time()
        while time.time() - self.start_time < self.thinking_time:
            self.search(board, depth, True, 0)
            depth += 1
        # print(f'reached depth {depth-1} in {time.time() - self.start_time} seconds with score {self.score}')
        return self.root_best_move
    
    def hash_position(self, board: board_obj) -> bytes:
        # int_markers = board.markers.astype(np.uint8)
        # uniboard = int_markers[:,:,self.maximizing_idx] - int_markers[:,:,((self.maximizing_idx + 1) % 2)]
        # return uniboard.tobytes() + board.hist[board.n_moves-1][1].tobytes()
        return board.markers.tobytes() + board.hist[board.n_moves-1][1].tobytes() #consider switching to above on performance testing. Uses less space but more time
    
    def pull_mini_boards(self, markers: np.array) -> np.array:
        ''' returns a (3,3,2) array of the miniboards '''
        # Reshape and transpose the markers array to get the desired shape
        return markers.reshape(3, 3, 3, 3, 2).transpose(0, 2, 1, 3, 4).reshape(9,3,3,2)

    def sort_moves(self, board, tt_move):
        legal_moves = ops.get_valid_moves(board)
        legal_moves.sort(key=lambda move: self.score_move(board, move, tt_move), reverse=True)
        return legal_moves
    
    def score_move(self, board: board_obj, move: tuple, tt_move:tuple) -> int:
        if move == tt_move:
            return 100
        
        score = 0
        #if move completes a box, or blocks a line, give it more score
        # markers = board.markers
        # target_miniboard = markers[move[0], :, :]
        # for masks in self.two_in_a_row_masks:
        #     attack_mask = masks[0]
        #     blocking_mask = masks[1]
        #     if (
        #         (np.all(target_miniboard[:, 0][attack_mask.flatten()]) or np.all(target_miniboard[:, 1][attack_mask.flatten()])) 
        #         and blocking_mask.flatten()[move[1]] 
        #         and not (target_miniboard[:, 0].flatten()[move[1]] and target_miniboard[:, 0].flatten()[move[1]])
        #         ):
        #         score += 50
        #         break
            
        #if a move sends the opponent to a completed box (i.e lets them go anywhere), give it a negative score
        done_boxes = board.miniboxes[:, :, 0] & board.miniboxes[:, :, 1] & board.miniboxes[:,:, 2]
        if done_boxes.flatten()[move[1]]:
            score -= 50
        
        return score

    def search(self, board:board_obj, depth:int, maximizing_player:bool, ply: int, alpha: int = -np.inf, beta: int = np.inf) -> int:
        '''simple minimax search'''
        if ops.check_game_finished(board):
            if np.all(np.any(board.miniboxes,axis=2)):
                return 0 #draw
            else:
                if maximizing_player:
                    return -100 + ply
                else:
                    return 100 - ply
        tt_move = None
        try:
            tt_entry = self.transposition_table[self.hash_position(board)]
            tt_move = tt_entry[3]
            if tt_entry[1] >= depth and (tt_entry[2] == 0 #exact score
                                          or tt_entry[2] == 2 and tt_entry[0] >= beta #lower bound
                                          or tt_entry[2] == 1 and tt_entry[0] <= alpha #upper bound
                                          ): 
                return tt_entry[0]
        except KeyError:
            pass
        
        if depth == 0:
            return self.evaluate(board)
        if time.time() - self.start_time > self.thinking_time:
            if maximizing_player:
                return np.inf
            else:
                return -np.inf
            
        legal_moves = self.sort_moves(board, tt_move)
        
        if maximizing_player:
            max_value = -np.inf
            best_move = legal_moves[0]
            for move in legal_moves:
                ops.make_move(board, move)
                max_value = max(max_value, self.search(board, depth-1, False, ply+1, alpha, beta))
                ops.undo_move(board)
                if max_value > beta:
                    break
                if max_value > alpha:
                    alpha = max_value
                    best_move = move
                    if ply == 0:
                        self.root_best_move = move
                        self.score = max_value
            entry_bound_flag = 0
            if max_value <= alpha:
                entry_bound_flag = 1
            elif max_value >= beta:
                entry_bound_flag = 2
            self.transposition_table[self.hash_position(board)] = (max_value, depth, entry_bound_flag, best_move)
            return max_value
        
        else:
            value = np.inf
            best_move = legal_moves[0]
            for move in legal_moves:
                ops.make_move(board, move)
                value = min(value, self.search(board, depth-1, True, ply+1, alpha, beta))
                ops.undo_move(board)
                if value < alpha:
                    break
                # beta = min(beta, value)
                if value < beta:
                    beta = value
                    best_move = move
            entry_bound_flag = 0
            if value <= alpha:
                entry_bound_flag = 1
            elif value >= beta:
                entry_bound_flag = 2
            self.transposition_table[self.hash_position(board)] = (value, depth, entry_bound_flag, best_move)
            return value
    
    def evaluate(self, board):
        '''simple evaluation function'''
        return self.minibox_score(board)

    def minibox_score(self, board):
        scores = [np.sum(board.miniboxes[:, :, p]) for p in range(2)]
        minibox_scores = scores[self.maximizing_idx] - scores[(self.maximizing_idx + 1) % 2]
        #add in a bonus for unblocked two in a rows
        #structure: first element is the attacking pattern (two in a row), second element is the blocking pattern
        
        miniboards = self.pull_mini_boards(board.markers)
        close_to_win_scores = np.zeros(2)
        boxes_in_play = ~board.miniboxes[:,:,0] & ~board.miniboxes[:,:,1] & ~board.miniboxes[:,:,2]
        '''TODO: FIND A WAY TO OPTIMIZE THIS'''
        for p in range(2):
            for miniboard in miniboards[boxes_in_play.flatten()]:
                for mask in self.two_in_a_row_masks:
                    attacker = mask[0]
                    blocker = mask[1]
                    #if we have already placed markers in the attacking spots and the blocking spot is unmarked
                    if np.all(miniboard[:,:, p][attacker]) and not np.any(miniboard[:,:,(p + 1) % 2][blocker]):
                        close_to_win_scores[p] += 0.5
                        break
        minibox_scores += close_to_win_scores[self.maximizing_idx] - close_to_win_scores[(self.maximizing_idx + 1) % 2]

        
        return minibox_scores

class move_ordering_v3:
    def __init__(self, name: str = 'Transposition Table'):
        self.name = name
        self.thinking_time = 0.1
        self.root_best_move = None
        self.start_time = None
        self.score = 0
        self.maximizing_idx = 0
        self.transposition_table = dict()
        self.two_in_a_row_masks = np.array([
                                        [[[1,1,0],
                                        [0,0,0],
                                        [0,0,0]],
                                        [[0,0,1],
                                        [0,0,0],
                                        [0,0,0]]
                                        ],
                                        [[[1,0,1],
                                        [0,0,0],
                                        [0,0,0]],
                                        [[0,1,0],
                                        [0,0,0],
                                        [0,0,0]]
                                        ],

                                        [[[0,0,0],
                                        [1,1,0],
                                        [0,0,0]],
                                        [[0,0,0],
                                        [0,0,1],
                                        [0,0,0]]
                                        ],
                                        [[[0,0,0],
                                        [1,0,1],
                                        [0,0,0]],
                                        [[0,0,0],
                                        [0,1,0],
                                        [0,0,0]]
                                        ],

                                        [[[0,0,0],
                                        [0,0,0],
                                        [1,1,0]],
                                        [[0,0,0],
                                        [0,0,0],
                                        [0,0,1]]],

                                        [[[0,0,0],
                                        [0,0,0],
                                        [1,0,1]],
                                        [[0,0,0],
                                        [0,0,0],
                                        [0,1,0]]],

                                        [[[0,1,1],
                                        [0,0,0],
                                        [0,0,0]],
                                        [[1,0,0],
                                        [0,0,0],
                                        [0,0,0]]],
                                        
               
                                        [[[0,0,0],
                                        [0,1,1],
                                        [0,0,0]],
                                        [[0,0,0],
                                        [1,0,0],
                                        [0,0,0]]],

                                        [[[0,0,0],
                                        [0,0,0],
                                        [0,1,1]],
                                        [[0,0,0],
                                        [0,0,0],
                                        [1,0,0]]],

                                        [[[1,0,0],
                                         [1,0,0],       
                                         [0,0,0]],
                                         [[0,0,0],
                                         [0,0,0],
                                         [1,0,0]]],

                                        [[[0,1,0],
                                         [0,1,0],
                                         [0,0,0]],
                                         [[0,0,0],
                                         [0,0,0],
                                         [0,1,0]]],
                                        [[[0,1,0],
                                         [0,0,0],
                                         [0,1,0]],
                                         [[0,0,0],
                                         [0,1,0],
                                         [0,0,0]]],

                                        [[[0,0,1],
                                         [0,0,1],
                                         [0,0,0]],
                                         [[0,0,0],
                                         [0,0,0],
                                         [0,0,1]],],

                                        [[[0,0,1],
                                         [0,0,0],
                                         [0,0,1]],
                                         [[0,0,0],
                                         [0,0,1],
                                         [0,0,0]],],

                                         [[[0,0,0],
                                         [1,0,0],
                                         [1,0,0]],
                                         [[1,0,0],
                                         [0,0,0],
                                         [0,0,0]]],

                                         [[[1,0,0],
                                         [0,0,0],
                                         [1,0,0]],
                                         [[0,0,0],
                                         [1,0,0],
                                         [0,0,0]]],

                                        [[[0,0,0],
                                         [0,1,0],
                                         [0,1,0]],
                                         [[0,1,0],
                                         [0,0,0],
                                         [0,0,0]],],

                                        [[[0,0,0],
                                         [0,0,1],
                                         [0,0,1]],
                                         [[0,0,1],
                                         [0,0,0],
                                         [0,0,0]]],

                                        [[[1,0,0],
                                        [0,1,0],
                                        [0,0,0]],
                                        [[0,0,0],
                                        [0,0,0],
                                        [0,0,1]]],

                                        [[[0,0,0],
                                        [0,1,0],
                                        [0,0,1]],
                                        [[1,0,0],
                                        [0,0,0],
                                        [0,0,0]]],

                                        [[[0,0,1],
                                        [0,1,0],
                                        [0,0,0]],
                                       [[0,0,0],
                                        [0,0,0],
                                        [1,0,0]] ],

                                        [[[0,0,1],
                                        [0,0,0],
                                        [1,0,0]],
                                       [[0,0,0],
                                        [0,1,0],
                                        [0,0,0]] ],

                                        [[[0,0,0],
                                        [0,1,0],
                                        [1,0,0]],
                                        [[0,0,1],
                                        [0,0,0],
                                        [0,0,0]]],

                                        [[[0,0,1],
                                        [0,0,0],
                                        [1,0,0]],
                                        [[0,0,0],
                                        [0,1,0],
                                        [0,0,0]]],

                                         ]).astype(bool)
        self.flattened_attack_masks = np.stack([mask[0].flatten() for mask in self.two_in_a_row_masks])
        self.flattened_blocking_masks = np.stack([mask[1].flatten() for mask in self.two_in_a_row_masks])
    def move(self, board_dict: dict) -> tuple:
        ''' wrapper
        apply the logic and returns the desired move
        '''
        b_obj = board_obj()
        b_obj.build_from_dict_gamestate(board_dict)
        self.maximizing_idx = b_obj.n_moves % 2
        return self.get_best_move(b_obj)
    
    def get_best_move(self, board: board_obj):
        depth = 1
        self.start_time = time.time()
        while time.time() - self.start_time < self.thinking_time:
            self.search(board, depth, True, 0)
            depth += 1
            # print(f'reached depth {depth-1} in {time.time() - self.start_time} seconds with score {self.score}')
        return self.root_best_move
    
    def hash_position(self, board: board_obj) -> bytes:
        # int_markers = board.markers.astype(np.uint8)
        # uniboard = int_markers[:,:,self.maximizing_idx] - int_markers[:,:,((self.maximizing_idx + 1) % 2)]
        # return uniboard.tobytes() + board.hist[board.n_moves-1][1].tobytes()
        return board.markers.tobytes() + board.hist[board.n_moves-1][1].tobytes() #consider switching to above on performance testing. Uses less space but more time
    
    def pull_mini_boards(self, markers: np.array) -> np.array:
        ''' returns a (3,3,2) array of the miniboards '''
        # Reshape and transpose the markers array to get the desired shape
        return markers.reshape(3, 3, 3, 3, 2).transpose(0, 2, 1, 3, 4).reshape(9,3,3,2)

    def sort_moves(self, board, tt_move):
        legal_moves = ops.get_valid_moves(board)
        legal_moves.sort(key=lambda move: self.score_move(board, move, tt_move), reverse=True)
        return legal_moves
    
    def score_move(self, board: board_obj, move: tuple, tt_move:tuple) -> int:
        if move == tt_move:
            return 100
        
        score = 0
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
        if move[1] % 4 == 0 and np.sum(target_miniboard.diagonal()) == 2:
            score += 10
        elif move[1] % 2 == 0 and np.sum(np.fliplr(target_miniboard).diagonal()) == 2:
            score += 10
        #if a move sends the opponent to a completed box (i.e lets them go anywhere), give it a negative score
        done_boxes = board.miniboxes[:, :, 0] & board.miniboxes[:, :, 1] & board.miniboxes[:,:, 2]
        if done_boxes.flatten()[move[1]]:
            score -= 50
        
        return score

    def search(self, board:board_obj, depth:int, maximizing_player:bool, ply: int, alpha: int = -np.inf, beta: int = np.inf) -> int:
        '''simple minimax search'''
        if ops.check_game_finished(board):
            if np.all(np.any(board.miniboxes,axis=2)):
                return 0 #draw
            else:
                if maximizing_player:
                    return -100 + ply
                else:
                    return 100 - ply
        tt_move = None
        try:
            tt_entry = self.transposition_table[self.hash_position(board)]
            tt_move = tt_entry[3]
            if tt_entry[1] >= depth and (tt_entry[2] == 0 #exact score
                                          or tt_entry[2] == 2 and tt_entry[0] >= beta #lower bound
                                          or tt_entry[2] == 1 and tt_entry[0] <= alpha #upper bound
                                          ): 
                return tt_entry[0]
        except KeyError:
            pass
        
        if depth == 0:
            return self.evaluate(board)
        if time.time() - self.start_time > self.thinking_time:
            if maximizing_player:
                return np.inf
            else:
                return -np.inf
            
        legal_moves = self.sort_moves(board, tt_move)
        
        if maximizing_player:
            max_value = -np.inf
            best_move = legal_moves[0]
            for move in legal_moves:
                ops.make_move(board, move)
                max_value = max(max_value, self.search(board, depth-1, False, ply+1, alpha, beta))
                ops.undo_move(board)
                if max_value > beta:
                    break
                if max_value > alpha:
                    alpha = max_value
                    best_move = move
                    if ply == 0:
                        self.root_best_move = move
                        self.score = max_value
            entry_bound_flag = 0
            if max_value <= alpha:
                entry_bound_flag = 1
            elif max_value >= beta:
                entry_bound_flag = 2
            self.transposition_table[self.hash_position(board)] = (max_value, depth, entry_bound_flag, best_move)
            return max_value
        
        else:
            value = np.inf
            best_move = legal_moves[0]
            for move in legal_moves:
                ops.make_move(board, move)
                value = min(value, self.search(board, depth-1, True, ply+1, alpha, beta))
                ops.undo_move(board)
                if value < alpha:
                    break
                # beta = min(beta, value)
                if value < beta:
                    beta = value
                    best_move = move
            entry_bound_flag = 0
            if value <= alpha:
                entry_bound_flag = 1
            elif value >= beta:
                entry_bound_flag = 2
            self.transposition_table[self.hash_position(board)] = (value, depth, entry_bound_flag, best_move)
            return value
    
    def evaluate(self, board):
        '''simple evaluation function'''
        return self.minibox_score(board)

    def minibox_score(self, board):
        scores = [np.sum(board.miniboxes[:, :, p]) for p in range(2)]
        minibox_scores = scores[self.maximizing_idx] - scores[(self.maximizing_idx + 1) % 2]
        #add in a bonus for unblocked two in a rows
        #structure: first element is the attacking pattern (two in a row), second element is the blocking pattern
        
        miniboards = self.pull_mini_boards(board.markers)
        close_to_win_scores = np.zeros(2)
        boxes_in_play = ~board.miniboxes[:,:,0] & ~board.miniboxes[:,:,1] & ~board.miniboxes[:,:,2]
        '''TODO: FIND A WAY TO OPTIMIZE THIS'''
        for p in range(2):
            for miniboard in miniboards[boxes_in_play.flatten()]:
                for mask in self.two_in_a_row_masks:
                    attacker = mask[0]
                    blocker = mask[1]
                    #if we have already placed markers in the attacking spots and the blocking spot is unmarked
                    if np.all(miniboard[:,:, p][attacker]) and not np.any(miniboard[:,:,(p + 1) % 2][blocker]):
                        close_to_win_scores[p] += 0.5
                        break
        minibox_scores += close_to_win_scores[self.maximizing_idx] - close_to_win_scores[(self.maximizing_idx + 1) % 2]

        
        return minibox_scores
    


class smaller_tt_entries_v1:
    def __init__(self, name: str = 'Transposition Table'):
        self.name = name
        self.thinking_time = 0.1
        self.root_best_move = None
        self.start_time = None
        self.score = 0
        self.maximizing_idx = 0
        self.transposition_table = dict()
        self.two_in_a_row_masks = np.array([
                                        [[[1,1,0],
                                        [0,0,0],
                                        [0,0,0]],
                                        [[0,0,1],
                                        [0,0,0],
                                        [0,0,0]]
                                        ],
                                        [[[1,0,1],
                                        [0,0,0],
                                        [0,0,0]],
                                        [[0,1,0],
                                        [0,0,0],
                                        [0,0,0]]
                                        ],

                                        [[[0,0,0],
                                        [1,1,0],
                                        [0,0,0]],
                                        [[0,0,0],
                                        [0,0,1],
                                        [0,0,0]]
                                        ],
                                        [[[0,0,0],
                                        [1,0,1],
                                        [0,0,0]],
                                        [[0,0,0],
                                        [0,1,0],
                                        [0,0,0]]
                                        ],

                                        [[[0,0,0],
                                        [0,0,0],
                                        [1,1,0]],
                                        [[0,0,0],
                                        [0,0,0],
                                        [0,0,1]]],

                                        [[[0,0,0],
                                        [0,0,0],
                                        [1,0,1]],
                                        [[0,0,0],
                                        [0,0,0],
                                        [0,1,0]]],

                                        [[[0,1,1],
                                        [0,0,0],
                                        [0,0,0]],
                                        [[1,0,0],
                                        [0,0,0],
                                        [0,0,0]]],
                                        
               
                                        [[[0,0,0],
                                        [0,1,1],
                                        [0,0,0]],
                                        [[0,0,0],
                                        [1,0,0],
                                        [0,0,0]]],

                                        [[[0,0,0],
                                        [0,0,0],
                                        [0,1,1]],
                                        [[0,0,0],
                                        [0,0,0],
                                        [1,0,0]]],

                                        [[[1,0,0],
                                         [1,0,0],       
                                         [0,0,0]],
                                         [[0,0,0],
                                         [0,0,0],
                                         [1,0,0]]],

                                        [[[0,1,0],
                                         [0,1,0],
                                         [0,0,0]],
                                         [[0,0,0],
                                         [0,0,0],
                                         [0,1,0]]],
                                        [[[0,1,0],
                                         [0,0,0],
                                         [0,1,0]],
                                         [[0,0,0],
                                         [0,1,0],
                                         [0,0,0]]],

                                        [[[0,0,1],
                                         [0,0,1],
                                         [0,0,0]],
                                         [[0,0,0],
                                         [0,0,0],
                                         [0,0,1]],],

                                        [[[0,0,1],
                                         [0,0,0],
                                         [0,0,1]],
                                         [[0,0,0],
                                         [0,0,1],
                                         [0,0,0]],],

                                         [[[0,0,0],
                                         [1,0,0],
                                         [1,0,0]],
                                         [[1,0,0],
                                         [0,0,0],
                                         [0,0,0]]],

                                         [[[1,0,0],
                                         [0,0,0],
                                         [1,0,0]],
                                         [[0,0,0],
                                         [1,0,0],
                                         [0,0,0]]],

                                        [[[0,0,0],
                                         [0,1,0],
                                         [0,1,0]],
                                         [[0,1,0],
                                         [0,0,0],
                                         [0,0,0]],],

                                        [[[0,0,0],
                                         [0,0,1],
                                         [0,0,1]],
                                         [[0,0,1],
                                         [0,0,0],
                                         [0,0,0]]],

                                        [[[1,0,0],
                                        [0,1,0],
                                        [0,0,0]],
                                        [[0,0,0],
                                        [0,0,0],
                                        [0,0,1]]],

                                        [[[0,0,0],
                                        [0,1,0],
                                        [0,0,1]],
                                        [[1,0,0],
                                        [0,0,0],
                                        [0,0,0]]],

                                        [[[0,0,1],
                                        [0,1,0],
                                        [0,0,0]],
                                       [[0,0,0],
                                        [0,0,0],
                                        [1,0,0]] ],

                                        [[[0,0,1],
                                        [0,0,0],
                                        [1,0,0]],
                                       [[0,0,0],
                                        [0,1,0],
                                        [0,0,0]] ],

                                        [[[0,0,0],
                                        [0,1,0],
                                        [1,0,0]],
                                        [[0,0,1],
                                        [0,0,0],
                                        [0,0,0]]],

                                        [[[0,0,1],
                                        [0,0,0],
                                        [1,0,0]],
                                        [[0,0,0],
                                        [0,1,0],
                                        [0,0,0]]],

                                         ]).astype(bool)
        self.flattened_attack_masks = np.stack([mask[0].flatten() for mask in self.two_in_a_row_masks])
        self.flattened_blocking_masks = np.stack([mask[1].flatten() for mask in self.two_in_a_row_masks])
    def move(self, board_dict: dict) -> tuple:
        ''' wrapper
        apply the logic and returns the desired move
        '''
        b_obj = board_obj()
        b_obj.build_from_dict_gamestate(board_dict)
        self.maximizing_idx = b_obj.n_moves % 2
        return self.get_best_move(b_obj)
    
    def get_best_move(self, board: board_obj):
        depth = 1
        self.start_time = time.time()
        while time.time() - self.start_time < self.thinking_time:
            self.search(board, depth, True, 0)
            depth += 1
            # print(f'reached depth {depth-1} in {time.time() - self.start_time} seconds with score {self.score}')
        return self.root_best_move
    
    def hash_position(self, board: board_obj) -> bytes:
        int_markers = board.markers.astype(np.int8)
        uniboard = int_markers[:,:,self.maximizing_idx] - int_markers[:,:,((self.maximizing_idx + 1) % 2)]
        return uniboard.tobytes() + board.hist[board.n_moves-1][1].tobytes()    
    def pull_mini_boards(self, markers: np.array) -> np.array:
        ''' returns a (3,3,2) array of the miniboards '''
        # Reshape and transpose the markers array to get the desired shape
        return markers.reshape(3, 3, 3, 3, 2).transpose(0, 2, 1, 3, 4).reshape(9,3,3,2)

    def sort_moves(self, board, tt_move):
        legal_moves = ops.get_valid_moves(board)
        legal_moves.sort(key=lambda move: self.score_move(board, move, tt_move), reverse=True)
        return legal_moves
    
    def score_move(self, board: board_obj, move: tuple, tt_move:tuple) -> int:
        if move == tt_move:
            return 100
        
        score = 0
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
        if move[1] % 4 == 0 and np.sum(target_miniboard.diagonal()) == 2:
            score += 10
        elif move[1] % 2 == 0 and np.sum(np.fliplr(target_miniboard).diagonal()) == 2:
            score += 10
        #if a move sends the opponent to a completed box (i.e lets them go anywhere), give it a negative score
        done_boxes = board.miniboxes[:, :, 0] & board.miniboxes[:, :, 1] & board.miniboxes[:,:, 2]
        if done_boxes.flatten()[move[1]]:
            score -= 50
        
        return score

    def search(self, board:board_obj, depth:int, maximizing_player:bool, ply: int, alpha: int = -np.inf, beta: int = np.inf) -> int:
        '''simple minimax search'''
        if ops.check_game_finished(board):
            if np.all(np.any(board.miniboxes,axis=2)):
                return 0 #draw
            else:
                if maximizing_player:
                    return -100 + ply
                else:
                    return 100 - ply
        tt_move = None
        try:
            tt_entry = self.transposition_table[self.hash_position(board)]
            tt_move = tt_entry[3]
            if tt_entry[1] >= depth and (tt_entry[2] == 0 #exact score
                                          or tt_entry[2] == 2 and tt_entry[0] >= beta #lower bound
                                          or tt_entry[2] == 1 and tt_entry[0] <= alpha #upper bound
                                          ): 
                return tt_entry[0]
        except KeyError:
            pass
        
        if depth == 0:
            return self.evaluate(board)
        if time.time() - self.start_time > self.thinking_time:
            if maximizing_player:
                return np.inf
            else:
                return -np.inf
            
        legal_moves = self.sort_moves(board, tt_move)
        
        if maximizing_player:
            max_value = -np.inf
            best_move = legal_moves[0]
            for move in legal_moves:
                ops.make_move(board, move)
                max_value = max(max_value, self.search(board, depth-1, False, ply+1, alpha, beta))
                ops.undo_move(board)
                if max_value > beta:
                    break
                if max_value > alpha:
                    alpha = max_value
                    best_move = move
                    if ply == 0:
                        self.root_best_move = move
                        self.score = max_value
            entry_bound_flag = 0
            if max_value <= alpha:
                entry_bound_flag = 1
            elif max_value >= beta:
                entry_bound_flag = 2
            self.transposition_table[self.hash_position(board)] = (max_value, depth, entry_bound_flag, best_move)
            return max_value
        
        else:
            value = np.inf
            best_move = legal_moves[0]
            for move in legal_moves:
                ops.make_move(board, move)
                value = min(value, self.search(board, depth-1, True, ply+1, alpha, beta))
                ops.undo_move(board)
                if value < alpha:
                    break
                # beta = min(beta, value)
                if value < beta:
                    beta = value
                    best_move = move
            entry_bound_flag = 0
            if value <= alpha:
                entry_bound_flag = 1
            elif value >= beta:
                entry_bound_flag = 2
            self.transposition_table[self.hash_position(board)] = (value, depth, entry_bound_flag, best_move)
            return value
    
    def evaluate(self, board):
        '''simple evaluation function'''
        return self.minibox_score(board)

    def minibox_score(self, board):
        scores = [np.sum(board.miniboxes[:, :, p]) for p in range(2)]
        minibox_scores = scores[self.maximizing_idx] - scores[(self.maximizing_idx + 1) % 2]
        #add in a bonus for unblocked two in a rows
        #structure: first element is the attacking pattern (two in a row), second element is the blocking pattern
        
        miniboards = self.pull_mini_boards(board.markers)
        close_to_win_scores = np.zeros(2)
        boxes_in_play = ~board.miniboxes[:,:,0] & ~board.miniboxes[:,:,1] & ~board.miniboxes[:,:,2]
        '''TODO: FIND A WAY TO OPTIMIZE THIS'''
        for p in range(2):
            for miniboard in miniboards[boxes_in_play.flatten()]:
                for mask in self.two_in_a_row_masks:
                    attacker = mask[0]
                    blocker = mask[1]
                    #if we have already placed markers in the attacking spots and the blocking spot is unmarked
                    if np.all(miniboard[:,:, p][attacker]) and not np.any(miniboard[:,:,(p + 1) % 2][blocker]):
                        close_to_win_scores[p] += 0.5
                        break
        minibox_scores += close_to_win_scores[self.maximizing_idx] - close_to_win_scores[(self.maximizing_idx + 1) % 2]

        
        return minibox_scores