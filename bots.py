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

class negamax_v1:
    def __init__(self, name: str = 'Transposition Table'):
        self.name = name
        self.thinking_time = 0.1
        self.root_best_move = None
        self.start_time = None
        self.score = 0
        self.root_player = 0
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
        self.root_player = b_obj.n_moves % 2
        return self.get_best_move(b_obj)
    
    def get_best_move(self, board: board_obj):
        depth = 1
        self.start_time = time.time()
        while time.time() - self.start_time < self.thinking_time:
            self.search(board, depth, 0, -999, 999)
            depth += 1
            # print(f'depth: {depth-1} in {time.time() - self.start_time:.4f}s, score: {self.score}')
        return self.root_best_move
    
    def search(self, board:board_obj, depth:int, ply: int, alpha: int, beta: int) -> int:
        '''negamax with alpha beta pruning'''
        if time.time() - self.start_time > self.thinking_time:
            return -np.inf
        if ops.check_game_finished(board):
            if self.check_win(board.miniboxes, (board.n_moves + 1) % 2):
                return -100 + ply
            else:
                return 0
        tt_move = None
        try:
            tt_entry = self.transposition_table[self.hash_position(board)]
            tt_move = tt_entry[3]
            if tt_entry[1] >= depth: 
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

        legal_moves = self.sort_moves(board, tt_move)
        best_move = legal_moves[0]
        alpha_orig = alpha
        max_val = -np.inf
        for move in legal_moves:
            ops.make_move(board, move)
            val = -self.search(board, depth-1, ply+1, -beta, -alpha)
            ops.undo_move(board)
            if val > max_val:
                max_val = val
                best_move = move
                if ply == 0 and abs(val) != np.inf:
                    self.root_best_move = move
                    self.score = max_val
            alpha = max(alpha, max_val)
            if alpha >= beta:
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
        score = np.sum(board.miniboxes[:, :, 0]) - np.sum(board.miniboxes[:, :, 1])
        # miniboards = self.pull_mini_boards(board.markers)
        # boxes_in_play = ~board.miniboxes[:,:,0] & ~board.miniboxes[:,:,1] & ~board.miniboxes[:,:,2]
        # '''TODO: FIND A WAY TO OPTIMIZE THIS'''
        # for p in range(2):
        #     for miniboard in miniboards[boxes_in_play.flatten()]:
        #         for mask in self.two_in_a_row_masks:
        #             attacker = mask[0]
        #             blocker = mask[1]
        #             #if we have already placed markers in the attacking spots and the blocking spot is unmarked
        #             if np.all(miniboard[:,:, p][attacker]) and not np.any(miniboard[:,:,(p + 1) % 2][blocker]):
        #                 # close_to_win_scores[p] += 0.5
        #                 score += 0.5
        #                 break  
        if board.n_moves % 2 == 1:
            score = -score      
        return score
   
    def hash_position(self, board: board_obj) -> bytes:
        int_markers = board.markers.astype(np.int8)
        uniboard = int_markers[:,:,self.root_player] - int_markers[:,:,((self.root_player + 1) % 2)]
        return uniboard.tobytes() + board.hist[board.n_moves-1][1].tobytes()    
    
    def pull_mini_boards(self, markers: np.array) -> np.array:
        ''' returns a (3,3,2) array of the miniboards '''
        # Reshape and transpose the markers array to get the desired shape
        return markers.reshape(3, 3, 3, 3, 2).transpose(0, 2, 1, 3, 4).reshape(9,3,3,2)
    
    def check_win(self, miniboxes:np.ndarray, p:int) -> bool:
        '''Check if player p has won the game'''
        for i in range(3):
            if np.sum(miniboxes[:,i, p]) == 3: return True # horizontal
            if np.sum(miniboxes[i,:, p]) == 3: return True # vertical

        # diagonals
        if miniboxes[:, :, p].trace() == 3: return True
        if np.rot90(miniboxes[:, :, p]).trace() == 3: return True
        return False
    
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
'''
putting the 2 in a row eval back in
'''
class negamax_v2:
    def __init__(self, name: str = 'Transposition Table'):
        self.name = name
        self.thinking_time = 0.1
        self.root_best_move = None
        self.start_time = None
        self.score = 0
        self.root_player = 0
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
        self.root_player = b_obj.n_moves % 2
        return self.get_best_move(b_obj)
    
    def get_best_move(self, board: board_obj):
        depth = 1
        self.start_time = time.time()
        while time.time() - self.start_time < self.thinking_time:
            self.search(board, depth, 0, -999, 999)
            depth += 1
            # print(f'depth: {depth-1} in {time.time() - self.start_time:.4f}s, score: {self.score}')
        return self.root_best_move
    
    def search(self, board:board_obj, depth:int, ply: int, alpha: int, beta: int) -> int:
        '''negamax with alpha beta pruning'''
        if time.time() - self.start_time > self.thinking_time:
            return -np.inf
        if ops.check_game_finished(board):
            if self.check_win(board.miniboxes, (board.n_moves + 1) % 2):
                return -100 + ply
            else:
                return 0
        tt_move = None
        try:
            tt_entry = self.transposition_table[self.hash_position(board)]
            tt_move = tt_entry[3]
            if tt_entry[1] >= depth: 
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

        legal_moves = self.sort_moves(board, tt_move)
        best_move = legal_moves[0]
        alpha_orig = alpha
        max_val = -np.inf
        for move in legal_moves:
            ops.make_move(board, move)
            val = -self.search(board, depth-1, ply+1, -beta, -alpha)
            ops.undo_move(board)
            if val > max_val:
                max_val = val
                best_move = move
                if ply == 0 and abs(val) != np.inf:
                    self.root_best_move = move
                    self.score = max_val
            alpha = max(alpha, max_val)
            if alpha >= beta:
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
        score = np.sum(board.miniboxes[:, :, 0]) - np.sum(board.miniboxes[:, :, 1])
        '''TODO: FIND A WAY TO OPTIMIZE THIS'''
        miniboards = self.pull_mini_boards(board.markers)
        boxes_in_play = ~board.miniboxes[:,:,0] & ~board.miniboxes[:,:,1] & ~board.miniboxes[:,:,2]
        two_in_a_rows = np.zeros(2)
        for p in range(2):
            for miniboard in miniboards[boxes_in_play.flatten()]:
                for mask in self.two_in_a_row_masks:
                    attacker = mask[0]
                    blocker = mask[1]
                    #if we have already placed markers in the attacking spots and the blocking spot is unmarked
                    if np.all(miniboard[:,:, p][attacker]) and not np.any(miniboard[:,:,(p + 1) % 2][blocker]):
                        # close_to_win_scores[p] += 0.5
                        two_in_a_rows[p] += 0.5
                        break
        score += two_in_a_rows[0] - two_in_a_rows[1]
        if board.n_moves % 2 == 1:
            score = -score      
        return score
   
    def hash_position(self, board: board_obj) -> bytes:
        int_markers = board.markers.astype(np.int8)
        uniboard = int_markers[:,:,self.root_player] - int_markers[:,:,((self.root_player + 1) % 2)]
        return uniboard.tobytes() + board.hist[board.n_moves-1][1].tobytes()    
    
    def pull_mini_boards(self, markers: np.array) -> np.array:
        ''' returns a (3,3,2) array of the miniboards '''
        # Reshape and transpose the markers array to get the desired shape
        return markers.reshape(3, 3, 3, 3, 2).transpose(0, 2, 1, 3, 4).reshape(9,3,3,2)
    
    def check_win(self, miniboxes:np.ndarray, p:int) -> bool:
        '''Check if player p has won the game'''
        for i in range(3):
            if np.sum(miniboxes[:,i, p]) == 3: return True # horizontal
            if np.sum(miniboxes[i,:, p]) == 3: return True # vertical

        # diagonals
        if miniboxes[:, :, p].trace() == 3: return True
        if np.rot90(miniboxes[:, :, p]).trace() == 3: return True
        return False
    
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
    

class killers_v1:
    def __init__(self, name: str = 'Transposition Table'):
        self.name = name
        self.thinking_time = 0.1
        self.root_best_move = None
        self.start_time = None
        self.score = 0
        self.root_player = 0
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
        self.killer_moves = np.zeros((128, 2), dtype=np.int8)
                                             
    def move(self, board_dict: dict) -> tuple:
        ''' wrapper
        apply the logic and returns the desired move
        '''
        b_obj = board_obj()
        b_obj.build_from_dict_gamestate(board_dict)
        self.root_player = b_obj.n_moves % 2
        return self.get_best_move(b_obj)
    
    def get_best_move(self, board: board_obj):
        depth = 1
        self.start_time = time.time()
        while time.time() - self.start_time < self.thinking_time and depth < 40:
            self.search(board, depth, 0, -999, 999)
            depth += 1
            # print(f'depth: {depth-1} in {time.time() - self.start_time:.4f}s, score: {self.score}')
        return self.root_best_move
    
    def search(self, board:board_obj, depth:int, ply: int, alpha: int, beta: int) -> int:
        '''negamax with alpha beta pruning'''
        if time.time() - self.start_time > self.thinking_time:
            return -np.inf
        if ops.check_game_finished(board):
            if self.check_win(board.miniboxes, (board.n_moves + 1) % 2):
                return -100 + ply
            else:
                return 0
        tt_move = None
        # pv_node = beta != alpha + 1
        try:
            tt_entry = self.transposition_table[self.hash_position(board)]
            tt_move = tt_entry[3]
            if tt_entry[1] >= depth: 
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

        legal_moves = self.sort_moves(board, tt_move, depth)
        best_move = legal_moves[0]
        alpha_orig = alpha
        max_val = -np.inf
        for move in legal_moves:
            ops.make_move(board, move)
            val = -self.search(board, depth-1, ply+1, -beta, -alpha)
            ops.undo_move(board)
            if val > max_val:
                max_val = val
                best_move = move
                if ply == 0 and abs(val) != np.inf:
                    self.root_best_move = move
                    self.score = max_val
            alpha = max(alpha, max_val)
            if alpha >= beta:
                self.killer_moves[depth, :] = move
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
        score = np.sum(board.miniboxes[:, :, 0]) - np.sum(board.miniboxes[:, :, 1])
        '''TODO: FIND A WAY TO OPTIMIZE THIS'''
        miniboards = self.pull_mini_boards(board.markers)
        boxes_in_play = ~board.miniboxes[:,:,0] & ~board.miniboxes[:,:,1] & ~board.miniboxes[:,:,2]
        two_in_a_rows = np.zeros(2)
        for p in range(2):
            for miniboard in miniboards[boxes_in_play.flatten()]:
                for mask in self.two_in_a_row_masks:
                    attacker = mask[0]
                    blocker = mask[1]
                    #if we have already placed markers in the attacking spots and the blocking spot is unmarked
                    if np.all(miniboard[:,:, p][attacker]) and not np.any(miniboard[:,:,(p + 1) % 2][blocker]):
                        two_in_a_rows[p] += 0.5
                        break
        score += two_in_a_rows[0] - two_in_a_rows[1]
        if board.n_moves % 2 == 1:
            score = -score      
        return score
   
    def hash_position(self, board: board_obj) -> bytes:
        int_markers = board.markers.astype(np.int8)
        uniboard = int_markers[:,:,self.root_player] - int_markers[:,:,((self.root_player + 1) % 2)]
        return uniboard.tobytes() + board.hist[board.n_moves-1][1].tobytes()    
    
    def pull_mini_boards(self, markers: np.array) -> np.array:
        ''' returns a (3,3,2) array of the miniboards '''
        # Reshape and transpose the markers array to get the desired shape
        return markers.reshape(3, 3, 3, 3, 2).transpose(0, 2, 1, 3, 4).reshape(9,3,3,2)
    
    def check_win(self, miniboxes:np.ndarray, p:int) -> bool:
        '''Check if player p has won the game'''
        for i in range(3):
            if np.sum(miniboxes[:,i, p]) == 3: return True # horizontal
            if np.sum(miniboxes[i,:, p]) == 3: return True # vertical

        # diagonals
        if miniboxes[:, :, p].trace() == 3: return True
        if np.rot90(miniboxes[:, :, p]).trace() == 3: return True
        return False
    
    def sort_moves(self, board: board_obj, tt_move: tuple, depth:int) -> list:
        legal_moves = ops.get_valid_moves(board)
        legal_moves.sort(key=lambda move: self.score_move(board, move, tt_move, depth), reverse=True)
        return legal_moves
    
    def score_move(self, board: board_obj, move: tuple, tt_move:tuple, depth:int) -> int:
        if move == tt_move:
            return 100
        if np.all(move == self.killer_moves[depth]):
            return 90
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
    
class history_v1:
    def __init__(self, name: str = 'Transposition Table'):
        self.name = name
        self.thinking_time = 0.1
        self.root_best_move = None
        self.start_time = None
        self.score = 0
        self.root_player = 0
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
        self.killer_moves = np.zeros((128, 2), dtype=np.int8) #depth, move
        self.history_table = np.zeros((2, 9, 9), dtype=np.uint16) #player (0 or 1), miniboard, move
                                             
    def move(self, board_dict: dict) -> tuple:
        ''' wrapper
        apply the logic and returns the desired move
        '''
        b_obj = board_obj()
        b_obj.build_from_dict_gamestate(board_dict)
        self.root_player = b_obj.n_moves % 2
        return self.get_best_move(b_obj)
    
    def get_best_move(self, board: board_obj):
        depth = 1
        self.start_time = time.time()
        while time.time() - self.start_time < self.thinking_time and depth < 40:
            self.search(board, depth, 0, -999, 999)
            depth += 1
            # print(f'depth: {depth-1} in {time.time() - self.start_time:.4f}s, score: {self.score}')
        return self.root_best_move
    
    def search(self, board:board_obj, depth:int, ply: int, alpha: int, beta: int) -> int:
        '''negamax with alpha beta pruning'''
        if time.time() - self.start_time > self.thinking_time:
            return -np.inf
        if ops.check_game_finished(board):
            if self.check_win(board.miniboxes, (board.n_moves + 1) % 2):
                return -100 + ply
            else:
                return 0
        tt_move = None
        # pv_node = beta != alpha + 1
        try:
            tt_entry = self.transposition_table[self.hash_position(board)]
            tt_move = tt_entry[3]
            if tt_entry[1] >= depth: 
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

        legal_moves = self.sort_moves(board, tt_move, depth)
        best_move = legal_moves[0]
        alpha_orig = alpha
        max_val = -np.inf
        for move_idx in range(len(legal_moves)):
            ops.make_move(board, legal_moves[move_idx])
            val = -self.search(board, depth-1, ply+1, -beta, -alpha)
            ops.undo_move(board)
            if val > max_val:
                max_val = val
                best_move = legal_moves[move_idx]
                if ply == 0 and abs(val) != np.inf:
                    self.root_best_move = legal_moves[move_idx]
                    self.score = max_val
            alpha = max(alpha, max_val)
            if alpha >= beta:
                self.killer_moves[depth, :] = legal_moves[move_idx]
                self.history_table[board.n_moves % 2, legal_moves[move_idx][0], legal_moves[move_idx][1]] += depth ** 2
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
        score = np.sum(board.miniboxes[:, :, 0]) - np.sum(board.miniboxes[:, :, 1])
        '''TODO: FIND A WAY TO OPTIMIZE THIS'''
        miniboards = self.pull_mini_boards(board.markers)
        boxes_in_play = ~board.miniboxes[:,:,0] & ~board.miniboxes[:,:,1] & ~board.miniboxes[:,:,2]
        two_in_a_rows = np.zeros(2)
        for p in range(2):
            for miniboard in miniboards[boxes_in_play.flatten()]:
                for mask in self.two_in_a_row_masks:
                    attacker = mask[0]
                    blocker = mask[1]
                    #if we have already placed markers in the attacking spots and the blocking spot is unmarked
                    if np.all(miniboard[:,:, p][attacker]) and not np.any(miniboard[:,:,(p + 1) % 2][blocker]):
                        two_in_a_rows[p] += 0.5
                        break
        score += two_in_a_rows[0] - two_in_a_rows[1]
        if board.n_moves % 2 == 1:
            score = -score      
        return score
   
    def hash_position(self, board: board_obj) -> bytes:
        int_markers = board.markers.astype(np.int8)
        uniboard = int_markers[:,:,self.root_player] - int_markers[:,:,((self.root_player + 1) % 2)]
        return uniboard.tobytes() + board.hist[board.n_moves-1][1].tobytes()    
    
    def pull_mini_boards(self, markers: np.array) -> np.array:
        ''' returns a (3,3,2) array of the miniboards '''
        # Reshape and transpose the markers array to get the desired shape
        return markers.reshape(3, 3, 3, 3, 2).transpose(0, 2, 1, 3, 4).reshape(9,3,3,2)
    
    def check_win(self, miniboxes:np.ndarray, p:int) -> bool:
        '''Check if player p has won the game'''
        for i in range(3):
            if np.sum(miniboxes[:,i, p]) == 3: return True # horizontal
            if np.sum(miniboxes[i,:, p]) == 3: return True # vertical

        # diagonals
        if miniboxes[:, :, p].trace() == 3: return True
        if np.rot90(miniboxes[:, :, p]).trace() == 3: return True
        return False
    
    def sort_moves(self, board: board_obj, tt_move: tuple, depth:int) -> list:
        legal_moves = ops.get_valid_moves(board)
        legal_moves.sort(key=lambda move: self.score_move(board, move, tt_move, depth), reverse=True)
        return legal_moves
    
    def score_move(self, board: board_obj, move: tuple, tt_move:tuple, depth:int) -> int:
        if move == tt_move:
            return np.inf
        if np.all(move == self.killer_moves[depth]):
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
        if move[1] % 4 == 0 and np.sum(target_miniboard.diagonal()) == 2:
            score += 10
        elif move[1] % 2 == 0 and np.sum(np.fliplr(target_miniboard).diagonal()) == 2:
            score += 10
        #if a move sends the opponent to a completed box (i.e lets them go anywhere), give it a negative score
        done_boxes = board.miniboxes[:, :, 0] & board.miniboxes[:, :, 1] & board.miniboxes[:,:, 2]
        if done_boxes.flatten()[move[1]]:
            score -= 50
        
        return score
class faster_eval_v1:
    def __init__(self, name: str = 'Transposition Table'):
        self.name = name
        self.thinking_time = 0.1
        self.root_best_move = None
        self.start_time = None
        self.score = 0
        self.root_player = 0
        self.transposition_table = dict()
        self.killer_moves = np.zeros((128, 2), dtype=np.int8) #depth, move
        self.history_table = np.zeros((2, 9, 9), dtype=np.uint16) #player (0 or 1), miniboard, move
        self.nodes = 0
                                             
    def move(self, board_dict: dict) -> tuple:
        ''' wrapper
        apply the logic and returns the desired move
        '''
        b_obj = board_obj()
        b_obj.build_from_dict_gamestate(board_dict)
        self.root_player = b_obj.n_moves % 2
        return self.get_best_move(b_obj)
    
    def get_best_move(self, board: board_obj):
        self.nodes = 0
        depth = 1
        self.start_time = time.time()
        while time.time() - self.start_time < self.thinking_time and depth < 40:
            self.search(board, depth, 0, -999, 999)
            depth += 1
            # print(f'depth: {depth-1} in {time.time() - self.start_time:.4f}s, score: {self.score}, nps: {self.nodes / (time.time() - self.start_time + 1e-5):.2f}')
        return self.root_best_move
    
    def search(self, board:board_obj, depth:int, ply: int, alpha: int, beta: int) -> int:
        '''negamax with alpha beta pruning'''
        if time.time() - self.start_time > self.thinking_time:
            return -np.inf
        self.nodes += 1
        if ops.check_game_finished(board):
            if self.check_win(board.miniboxes, (board.n_moves + 1) % 2):
                return -100 + ply
            else:
                return 0
        tt_move = None
        try:
            tt_entry = self.transposition_table[self.hash_position(board)]
            tt_move = tt_entry[3]
            if tt_entry[1] >= depth: 
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

        legal_moves = self.sort_moves(board, tt_move, depth)
        best_move = legal_moves[0]
        alpha_orig = alpha
        max_val = -np.inf
        for move_idx in range(len(legal_moves)):
            ops.make_move(board, legal_moves[move_idx])
            val = -self.search(board, depth-1, ply+1, -beta, -alpha)
            ops.undo_move(board)
            if val > max_val:
                max_val = val
                best_move = legal_moves[move_idx]
                if ply == 0 and abs(val) != np.inf:
                    self.root_best_move = legal_moves[move_idx]
                    self.score = max_val
            alpha = max(alpha, max_val)
            if alpha >= beta:
                self.killer_moves[depth, :] = legal_moves[move_idx]
                self.history_table[board.n_moves % 2, legal_moves[move_idx][0], legal_moves[move_idx][1]] += depth ** 2
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
        score = (np.sum(board.miniboxes[:, :, 0]) - np.sum(board.miniboxes[:, :, 1])) * 2
        # '''TODO: FIND A WAY TO OPTIMIZE THIS'''
        miniboards = self.pull_mini_boards(board.markers)
        boxes_in_play = ~board.miniboxes[:,:,0] & ~board.miniboxes[:,:,1] & ~board.miniboxes[:,:,2]
        two_in_a_row_eval = np.zeros(2, dtype=np.int8)
        for p in range(2):
            for b in miniboards[boxes_in_play.flatten()]:
                if b[:, :, p].trace() - b[:, :, (p + 1) % 2].trace() == 2:
                    two_in_a_row_eval[p] += 1
                    continue
                if np.rot90(b[:, :, p]).trace() - np.rot90(b[:, :, (p + 1) % 2]).trace() == 2:
                    two_in_a_row_eval[p] += 1
                    continue
                for i in range(3):
                    if np.sum(b[:,i,p]) - np.sum(b[:,i,(p + 1) % 2]) == 2:
                        two_in_a_row_eval[p] += 1
                        break
                    if np.sum(b[i,:,p]) - np.sum(b[i,:,(p + 1) %2]) == 2:
                        two_in_a_row_eval[p] += 1
                        break
        score += two_in_a_row_eval[0] - two_in_a_row_eval[1]
        if board.n_moves % 2 == 1:
            score = -score      
        return score
   
    def hash_position(self, board: board_obj) -> bytes:
        int_markers = board.markers.astype(np.int8)
        uniboard = int_markers[:,:,self.root_player] - int_markers[:,:,((self.root_player + 1) % 2)]
        return uniboard.tobytes() + board.hist[board.n_moves-1][1].tobytes()    
    
    def pull_mini_boards(self, markers: np.array) -> np.array:
        ''' returns a (3,3,2) array of the miniboards '''
        # Reshape and transpose the markers array to get the desired shape
        return markers.reshape(3, 3, 3, 3, 2).transpose(0, 2, 1, 3, 4).reshape(9,3,3,2)
    
    def check_win(self, miniboxes:np.ndarray, p:int) -> bool:
        '''Check if player p has won the game'''
        for i in range(3):
            if np.sum(miniboxes[:,i, p]) == 3: return True # horizontal
            if np.sum(miniboxes[i,:, p]) == 3: return True # vertical

        # diagonals
        if miniboxes[:, :, p].trace() == 3: return True
        if np.rot90(miniboxes[:, :, p]).trace() == 3: return True
        return False
    
    def sort_moves(self, board: board_obj, tt_move: tuple, depth:int) -> list:
        legal_moves = ops.get_valid_moves(board)
        legal_moves.sort(key=lambda move: self.score_move(board, move, tt_move, depth), reverse=True)
        return legal_moves
    
    def score_move(self, board: board_obj, move: tuple, tt_move:tuple, depth:int) -> int:
        if move == tt_move:
            return np.inf
        if np.all(move == self.killer_moves[depth]):
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
        if move[1] % 4 == 0 and np.sum(target_miniboard.diagonal()) == 2:
            score += 10
        elif move[1] % 2 == 0 and np.sum(np.fliplr(target_miniboard).diagonal()) == 2:
            score += 10
        #if a move sends the opponent to a completed box (i.e lets them go anywhere), give it a negative score
        done_boxes = board.miniboxes[:, :, 0] & board.miniboxes[:, :, 1] & board.miniboxes[:,:, 2]
        if done_boxes.flatten()[move[1]]:
            score -= 50
        
        return score
    
class faster_eval_v2:
    def __init__(self, name: str = 'Transposition Table'):
        self.name = name
        self.thinking_time = 0.1
        self.root_best_move = None
        self.start_time = None
        self.score = 0
        self.root_player = 0
        self.transposition_table = dict()
        self.killer_moves = np.zeros((128, 2), dtype=np.int8) #depth, move
        self.history_table = np.zeros((2, 9, 9), dtype=np.uint16) #player (0 or 1), miniboard, move
        self.nodes = 0
                                             
    def move(self, board_dict: dict) -> tuple:
        ''' wrapper
        apply the logic and returns the desired move
        '''
        b_obj = board_obj()
        b_obj.build_from_dict_gamestate(board_dict)
        self.root_player = b_obj.n_moves % 2
        return self.get_best_move(b_obj)
    
    def get_best_move(self, board: board_obj):
        self.nodes = 0
        depth = 1
        self.start_time = time.time()
        while time.time() - self.start_time < self.thinking_time and depth < 40:
            self.search(board, depth, 0, -np.inf, np.inf)
            depth += 1
            # print(f'depth: {depth-1} in {time.time() - self.start_time:.4f}s, score: {self.score}, nps: {self.nodes / (time.time() - self.start_time + 1e-5):.2f}')
        return self.root_best_move
    
    def search(self, board:board_obj, depth:int, ply: int, alpha: int, beta: int) -> int:
        '''negamax with alpha beta pruning'''
        if time.time() - self.start_time > self.thinking_time:
            return -np.inf
        self.nodes += 1
        if ops.check_game_finished(board):
            if self.check_win(board.miniboxes, (board.n_moves + 1) % 2):
                return -100 + ply
            else:
                return 0
        tt_move = None
        try:
            tt_entry = self.transposition_table[self.hash_position(board)]
            tt_move = tt_entry[3]
            if tt_entry[1] >= depth: 
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

        legal_moves = self.sort_moves(board, tt_move, depth)
        best_move = legal_moves[0]
        alpha_orig = alpha
        max_val = -np.inf
        for move_idx in range(len(legal_moves)):
            ops.make_move(board, legal_moves[move_idx])
            val = -self.search(board, depth-1, ply+1, -beta, -alpha)
            ops.undo_move(board)
            if val > max_val:
                max_val = val
                best_move = legal_moves[move_idx]
                if ply == 0 and abs(val) != np.inf:
                    self.root_best_move = legal_moves[move_idx]
                    self.score = max_val
            alpha = max(alpha, max_val)
            if alpha >= beta:
                self.killer_moves[depth, :] = legal_moves[move_idx]
                self.history_table[board.n_moves % 2, legal_moves[move_idx][0], legal_moves[move_idx][1]] += depth ** 2
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
        score = (np.sum(board.miniboxes[:, :, 0]) - np.sum(board.miniboxes[:, :, 1])) * 2
        # '''TODO: FIND A WAY TO OPTIMIZE THIS'''
        miniboards = self.pull_mini_boards(board.markers)
        boxes_in_play = ~board.miniboxes[:,:,0] & ~board.miniboxes[:,:,1] & ~board.miniboxes[:,:,2]
        two_in_a_row_eval = np.zeros(2, dtype=np.int8)
        for p in range(2):
            q = (p + 1) % 2  # Opponent's index
            for b in miniboards[boxes_in_play.flatten()]:
                b_p = b[:, :, p]
                b_q = b[:, :, q]

                # Precompute sums
                sum_b_p_rows = np.sum(b_p, axis=1)
                sum_b_q_rows = np.sum(b_q, axis=1)
                sum_b_p_cols = np.sum(b_p, axis=0)
                sum_b_q_cols = np.sum(b_q, axis=0)

                # Diagonal checks
                if np.trace(b_p) - np.trace(b_q) == 2 or \
                np.trace(np.fliplr(b_p)) - np.trace(np.fliplr(b_q)) == 2:
                    two_in_a_row_eval[p] += 1
                    continue

                # Row and Column checks
                for i in range(3):
                    if (sum_b_p_rows[i] - sum_b_q_rows[i] == 2) or \
                    (sum_b_p_cols[i] - sum_b_q_cols[i] == 2):
                        two_in_a_row_eval[p] += 1
                        break
        
        score += two_in_a_row_eval[0] - two_in_a_row_eval[1]
        if board.n_moves % 2 == 1:
            score = -score      
        return score
   
    def hash_position(self, board: board_obj) -> bytes:
        int_markers = board.markers.astype(np.int8)
        uniboard = int_markers[:,:,self.root_player] - int_markers[:,:,((self.root_player + 1) % 2)]
        return uniboard.tobytes() + board.hist[board.n_moves-1][1].tobytes()    
    
    def pull_mini_boards(self, markers: np.array) -> np.array:
        ''' returns a (3,3,2) array of the miniboards '''
        # Reshape and transpose the markers array to get the desired shape
        return markers.reshape(3, 3, 3, 3, 2).transpose(0, 2, 1, 3, 4).reshape(9,3,3,2)
    
    def check_win(self, miniboxes:np.ndarray, p:int) -> bool:
        '''Check if player p has won the game'''
        for i in range(3):
            if np.sum(miniboxes[:,i, p]) == 3: return True # horizontal
            if np.sum(miniboxes[i,:, p]) == 3: return True # vertical

        # diagonals
        if miniboxes[:, :, p].trace() == 3: return True
        if np.rot90(miniboxes[:, :, p]).trace() == 3: return True
        return False
    
    def sort_moves(self, board: board_obj, tt_move: tuple, depth:int) -> list:
        legal_moves = ops.get_valid_moves(board)
        legal_moves.sort(key=lambda move: self.score_move(board, move, tt_move, depth), reverse=True)
        return legal_moves
    
    def score_move(self, board: board_obj, move: tuple, tt_move:tuple, depth:int) -> int:
        if move == tt_move:
            return np.inf
        if np.all(move == self.killer_moves[depth]):
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
        if move[1] % 4 == 0 and np.sum(target_miniboard.diagonal()) == 2:
            score += 10
        elif move[1] % 2 == 0 and np.sum(np.fliplr(target_miniboard).diagonal()) == 2:
            score += 10
        #if a move sends the opponent to a completed box (i.e lets them go anywhere), give it a negative score
        done_boxes = board.miniboxes[:, :, 0] & board.miniboxes[:, :, 1] & board.miniboxes[:,:, 2]
        if done_boxes.flatten()[move[1]]:
            score -= 50
        
        return score
    

class crossfish_v1:
    def __init__(self, name: str = 'Crossfish'):
        self.name = name
        self.thinking_time = 0.1
        self.root_best_move = None
        self.start_time = None
        self.score = 0
        self.root_player = 0
        self.transposition_table = dict()
        self.killer_moves = np.zeros((128, 2), dtype=np.int8) #depth, move
        self.history_table = np.zeros((2, 9, 9), dtype=np.uint16) #player (0 or 1), miniboard, move
        self.nodes = 0
        self.instance_b = board_obj()
                                             
    def move(self, board_dict: dict) -> tuple:
        ''' wrapper
        apply the logic and returns the desired move
        '''
        b_obj = board_obj()
        b_obj.build_from_dict_gamestate(board_dict)
        self.root_player = b_obj.n_moves % 2
        return self.get_best_move(b_obj)
    
    def get_best_move(self, board: board_obj):
        self.nodes = 0
        depth = 1
        self.start_time = time.time()
        while time.time() - self.start_time < self.thinking_time and depth < 40:
            self.search(board, depth, 0, -np.inf, np.inf)
            depth += 1
            # print(f'depth: {depth-1} in {time.time() - self.start_time:.4f}s, score: {self.score}, nps: {self.nodes / (time.time() - self.start_time + 1e-5):.2f}')
        return self.root_best_move
    
    def search(self, board:board_obj, depth:int, ply: int, alpha: int, beta: int) -> int:
        '''negamax with alpha beta pruning'''
        if time.time() - self.start_time > self.thinking_time:
            return -np.inf
        self.nodes += 1
        if ops.check_game_finished(board):
            if self.check_win(board.miniboxes, (board.n_moves + 1) % 2):
                return -100 + ply
            else:
                return 0
        tt_move = None
        try:
            tt_entry = self.transposition_table[self.hash_position(board)]
            tt_move = tt_entry[3]
            if tt_entry[1] >= depth: 
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

        legal_moves = self.sort_moves(board, tt_move, depth)
        best_move = legal_moves[0]
        alpha_orig = alpha
        max_val = -np.inf
        for move_idx in range(len(legal_moves)):
            ops.make_move(board, legal_moves[move_idx])
            val = -self.search(board, depth-1, ply+1, -beta, -alpha)
            ops.undo_move(board)
            if val > max_val:
                max_val = val
                best_move = legal_moves[move_idx]
                if ply == 0 and abs(val) != np.inf:
                    self.root_best_move = legal_moves[move_idx]
                    self.score = max_val
            alpha = max(alpha, max_val)
            if alpha >= beta:
                self.killer_moves[depth, :] = legal_moves[move_idx]
                self.history_table[board.n_moves % 2, legal_moves[move_idx][0], legal_moves[move_idx][1]] += depth ** 2
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
        score = (np.sum(board.miniboxes[:, :, 0]) - np.sum(board.miniboxes[:, :, 1])) * 2
        # '''TODO: FIND A WAY TO OPTIMIZE THIS'''
        miniboards = self.pull_mini_boards(board.markers)
        boxes_in_play = ~board.miniboxes[:,:,0] & ~board.miniboxes[:,:,1] & ~board.miniboxes[:,:,2]
        two_in_a_row_eval = np.zeros(2, dtype=np.int8)
        for p in range(2):
            q = (p + 1) % 2  # Opponent's index
            for b in miniboards[boxes_in_play.flatten()]:
                b_p = b[:, :, p]
                b_q = b[:, :, q]

                # Precompute sums
                sum_b_p_rows = np.sum(b_p, axis=1)
                sum_b_q_rows = np.sum(b_q, axis=1)
                sum_b_p_cols = np.sum(b_p, axis=0)
                sum_b_q_cols = np.sum(b_q, axis=0)

                # Diagonal checks
                if np.trace(b_p) - np.trace(b_q) == 2 or \
                np.trace(np.fliplr(b_p)) - np.trace(np.fliplr(b_q)) == 2:
                    two_in_a_row_eval[p] += 1
                    continue

                # Row and Column checks
                for i in range(3):
                    if (sum_b_p_rows[i] - sum_b_q_rows[i] == 2) or \
                    (sum_b_p_cols[i] - sum_b_q_cols[i] == 2):
                        two_in_a_row_eval[p] += 1
                        break
        
        score += two_in_a_row_eval[0] - two_in_a_row_eval[1]
        if board.n_moves % 2 == 1:
            score = -score      
        return score
   
    def hash_position(self, board: board_obj) -> bytes:
        int_markers = board.markers.astype(np.int8)
        uniboard = int_markers[:,:,self.root_player] - int_markers[:,:,((self.root_player + 1) % 2)]
        return uniboard.tobytes() + board.hist[board.n_moves-1][1].tobytes()    
    
    def pull_mini_boards(self, markers: np.array) -> np.array:
        ''' returns a (3,3,2) array of the miniboards '''
        # Reshape and transpose the markers array to get the desired shape
        return markers.reshape(3, 3, 3, 3, 2).transpose(0, 2, 1, 3, 4).reshape(9,3,3,2)
    
    def check_win(self, miniboxes:np.ndarray, p:int) -> bool:
        '''Check if player p has won the game'''
        for i in range(3):
            if np.sum(miniboxes[:,i, p]) == 3: return True # horizontal
            if np.sum(miniboxes[i,:, p]) == 3: return True # vertical

        # diagonals
        if miniboxes[:, :, p].trace() == 3: return True
        if np.rot90(miniboxes[:, :, p]).trace() == 3: return True
        return False
    
    def sort_moves(self, board: board_obj, tt_move: tuple, depth:int) -> list:
        legal_moves = ops.get_valid_moves(board)
        legal_moves.sort(key=lambda move: self.score_move(board, move, tt_move, depth), reverse=True)
        return legal_moves
    
    def score_move(self, board: board_obj, move: tuple, tt_move:tuple, depth:int) -> int:
        if move == tt_move:
            return np.inf
        if np.all(move == self.killer_moves[depth]):
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
        if move[1] % 4 == 0 and np.sum(target_miniboard.diagonal()) == 2:
            score += 10
        elif move[1] % 2 == 0 and np.sum(np.fliplr(target_miniboard).diagonal()) == 2:
            score += 10
        #if a move sends the opponent to a completed box (i.e lets them go anywhere), give it a negative score
        done_boxes = board.miniboxes[:, :, 0] & board.miniboxes[:, :, 1] & board.miniboxes[:,:, 2]
        if done_boxes.flatten()[move[1]]:
            score -= 50
        
        return score

class crossfish_v2:
    def __init__(self, name: str = 'Crossfish'):
        self.name = name
        self.thinking_time = 0.1
        self.root_best_move = None
        self.start_time = None
        self.score = 0
        self.root_player = 0
        self.transposition_table = dict()
        self.killer_moves = np.zeros((128, 2), dtype=np.int8) #depth, move
        self.history_table = np.zeros((2, 9, 9), dtype=np.uint16) #player (0 or 1), miniboard, move
        self.nodes = 0
        self.instance_b = board_obj()
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
        self.root_player = b_obj.n_moves % 2
        return self.get_best_move(b_obj)
    
    def get_best_move(self, board: board_obj):
        self.nodes = 0
        depth = 1
        self.start_time = time.time()
        while time.time() - self.start_time < self.thinking_time and depth < 40:
            self.search(board, depth, 0, -np.inf, np.inf)
            depth += 1
            # print(f'depth: {depth-1} in {time.time() - self.start_time:.4f}s, score: {self.score}, nps: {self.nodes / (time.time() - self.start_time + 1e-5):.2f}')
        return self.root_best_move
    
    def search(self, board:board_obj, depth:int, ply: int, alpha: int, beta: int) -> int:
        '''negamax with alpha beta pruning'''
        if time.time() - self.start_time > self.thinking_time:
            return -np.inf
        self.nodes += 1
        if self.check_game_finished(board):
            if self.check_win(board.miniboxes, (board.n_moves + 1) % 2):
                return -100 + ply
            else:
                return 0
        tt_move = None
        try:
            tt_entry = self.transposition_table[self.hash_position(board)]
            tt_move = tt_entry[3]
            if tt_entry[1] >= depth: 
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

        legal_moves = self.sort_moves(board, tt_move, depth)
        best_move = legal_moves[0]
        alpha_orig = alpha
        max_val = -np.inf
        for move_idx in range(len(legal_moves)):
            self.make_move(board, legal_moves[move_idx])
            val = -self.search(board, depth-1, ply+1, -beta, -alpha)
            self.undo_move(board)
            if val > max_val:
                max_val = val
                best_move = legal_moves[move_idx]
                if ply == 0 and abs(val) != np.inf:
                    self.root_best_move = legal_moves[move_idx]
                    self.score = max_val
            alpha = max(alpha, max_val)
            if alpha >= beta:
                self.killer_moves[depth, :] = legal_moves[move_idx]
                self.history_table[board.n_moves % 2, legal_moves[move_idx][0], legal_moves[move_idx][1]] += depth ** 2
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
        score = (np.sum(board.miniboxes[:, :, 0]) - np.sum(board.miniboxes[:, :, 1])) * 2
        # '''TODO: FIND A WAY TO OPTIMIZE THIS'''
        miniboards = self.pull_mini_boards(board.markers)
        boxes_in_play = ~board.miniboxes[:,:,0] & ~board.miniboxes[:,:,1] & ~board.miniboxes[:,:,2]
        two_in_a_row_eval = np.zeros(2, dtype=np.int8)
        for p in range(2):
            q = (p + 1) % 2  # Opponent's index
            for b in miniboards[boxes_in_play.flatten()]:
                b_p = b[:, :, p]
                b_q = b[:, :, q]

                # Precompute sums
                sum_b_p_rows = np.sum(b_p, axis=1)
                sum_b_q_rows = np.sum(b_q, axis=1)
                sum_b_p_cols = np.sum(b_p, axis=0)
                sum_b_q_cols = np.sum(b_q, axis=0)

                # Diagonal checks
                if np.trace(b_p) - np.trace(b_q) == 2 or \
                np.trace(np.fliplr(b_p)) - np.trace(np.fliplr(b_q)) == 2:
                    two_in_a_row_eval[p] += 1
                    continue

                # Row and Column checks
                for i in range(3):
                    if (sum_b_p_rows[i] - sum_b_q_rows[i] == 2) or \
                    (sum_b_p_cols[i] - sum_b_q_cols[i] == 2):
                        two_in_a_row_eval[p] += 1
                        break
        
        score += two_in_a_row_eval[0] - two_in_a_row_eval[1]
        if board.n_moves % 2 == 1:
            score = -score      
        return score
   
    def hash_position(self, board: board_obj) -> bytes:
        int_markers = board.markers.astype(np.int8)
        uniboard = int_markers[:,:,self.root_player] - int_markers[:,:,((self.root_player + 1) % 2)]
        return uniboard.tobytes() + board.hist[board.n_moves-1][1].tobytes()    
    
    def pull_mini_boards(self, markers: np.array) -> np.array:
        ''' returns a (3,3,2) array of the miniboards '''
        # Reshape and transpose the markers array to get the desired shape
        return markers.reshape(3, 3, 3, 3, 2).transpose(0, 2, 1, 3, 4).reshape(9,3,3,2)
    
    def check_win(self, miniboxes:np.ndarray, p:int) -> bool:
        '''Check if player p has won the game'''
        for i in range(3):
            if np.sum(miniboxes[:,i, p]) == 3: return True # horizontal
            if np.sum(miniboxes[i,:, p]) == 3: return True # vertical

        # diagonals
        if miniboxes[:, :, p].trace() == 3: return True
        if np.rot90(miniboxes[:, :, p]).trace() == 3: return True
        return False
    
    def sort_moves(self, board: board_obj, tt_move: tuple, depth:int) -> list:
        legal_moves = self.get_valid_moves(board)
        legal_moves.sort(key=lambda move: self.score_move(board, move, tt_move, depth), reverse=True)
        return legal_moves
    
    def score_move(self, board: board_obj, move: tuple, tt_move:tuple, depth:int) -> int:
        if move == tt_move:
            return np.inf
        if np.all(move == self.killer_moves[depth]):
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
        if move[1] % 4 == 0 and np.sum(target_miniboard.diagonal()) == 2:
            score += 10
        elif move[1] % 2 == 0 and np.sum(np.fliplr(target_miniboard).diagonal()) == 2:
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

    def undo_move(self, board_obj: board_obj) -> None:
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

        # check stale
        if np.all(np.any(board_obj.miniboxes,axis=2)): # if all miniboards are filled with something
            return 'stale'

        return 'game is ongoing'
    
class crossfish_v3:
    '''This version fixes an error where killers were indexed by depth instead of ply. 
    Test Result: W: 1038, L: 778, D: 488, elo diff: 39.37 +/- 25.32, LOS: 100.00'''
    def __init__(self, name: str = 'Crossfish'):
        self.name = name
        self.thinking_time = 0.1
        self.root_best_move = None
        self.start_time = None
        self.score = 0
        self.root_player = 0
        self.transposition_table = dict()
        self.killer_moves = np.zeros((128, 2), dtype=np.int8) #depth, move
        self.history_table = np.zeros((2, 9, 9), dtype=np.uint16) #player (0 or 1), miniboard, move
        self.nodes = 0
        self.instance_b = board_obj()
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
        self.root_player = b_obj.n_moves % 2
        return self.get_best_move(b_obj)
    
    def get_best_move(self, board: board_obj):
        self.nodes = 0
        depth = 1
        self.start_time = time.time()
        while time.time() - self.start_time < self.thinking_time and depth < 40:
            self.search(board, depth, 0, -np.inf, np.inf)
            depth += 1
            # print(f'depth: {depth-1} in {time.time() - self.start_time:.4f}s, score: {self.score}, nps: {self.nodes / (time.time() - self.start_time + 1e-5):.2f}')
        return self.root_best_move
    
    def search(self, board:board_obj, depth:int, ply: int, alpha: int, beta: int) -> int:
        '''negamax with alpha beta pruning'''
        if time.time() - self.start_time > self.thinking_time:
            return -np.inf
        self.nodes += 1
        if self.check_game_finished(board):
            if self.check_win(board.miniboxes, (board.n_moves + 1) % 2):
                return -100 + ply
            else:
                return 0
        tt_move = None
        try:
            tt_entry = self.transposition_table[self.hash_position(board)]
            tt_move = tt_entry[3]
            if tt_entry[1] >= depth: 
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

        legal_moves = self.get_sorted_moves(board, tt_move, ply)
        best_move = legal_moves[0]
        alpha_orig = alpha
        max_val = -np.inf
        for move_idx in range(len(legal_moves)):
            self.make_move(board, legal_moves[move_idx])
            val = -self.search(board, depth-1, ply+1, -beta, -alpha)
            self.undo_move(board)
            if val > max_val:
                max_val = val
                best_move = legal_moves[move_idx]
                if ply == 0 and abs(val) != np.inf:
                    self.root_best_move = legal_moves[move_idx]
                    self.score = max_val
            alpha = max(alpha, max_val)
            if alpha >= beta:
                self.killer_moves[ply, :] = legal_moves[move_idx]
                self.history_table[board.n_moves % 2, legal_moves[move_idx][0], legal_moves[move_idx][1]] += depth ** 2
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
        score = (np.sum(board.miniboxes[:, :, 0]) - np.sum(board.miniboxes[:, :, 1])) * 2
        # '''TODO: FIND A WAY TO OPTIMIZE THIS'''
        miniboards = self.pull_mini_boards(board.markers)
        boxes_in_play = ~board.miniboxes[:,:,0] & ~board.miniboxes[:,:,1] & ~board.miniboxes[:,:,2]
        two_in_a_row_eval = np.zeros(2, dtype=np.int8)
        for p in range(2):
            q = (p + 1) % 2  # Opponent's index
            for b in miniboards[boxes_in_play.flatten()]:
                b_p = b[:, :, p]
                b_q = b[:, :, q]

                # Precompute sums
                sum_b_p_rows = np.sum(b_p, axis=1)
                sum_b_q_rows = np.sum(b_q, axis=1)
                sum_b_p_cols = np.sum(b_p, axis=0)
                sum_b_q_cols = np.sum(b_q, axis=0)

                # Diagonal checks
                if np.trace(b_p) - np.trace(b_q) == 2 or \
                np.trace(np.fliplr(b_p)) - np.trace(np.fliplr(b_q)) == 2:
                    two_in_a_row_eval[p] += 1
                    continue

                # Row and Column checks
                for i in range(3):
                    if (sum_b_p_rows[i] - sum_b_q_rows[i] == 2) or \
                    (sum_b_p_cols[i] - sum_b_q_cols[i] == 2):
                        two_in_a_row_eval[p] += 1
                        break
        
        score += two_in_a_row_eval[0] - two_in_a_row_eval[1]
        if board.n_moves % 2 == 1:
            score = -score      
        return score
   
    def hash_position(self, board: board_obj) -> bytes:
        int_markers = board.markers.astype(np.int8)
        uniboard = int_markers[:,:,self.root_player] - int_markers[:,:,((self.root_player + 1) % 2)]
        return uniboard.tobytes() + board.hist[board.n_moves-1][1].tobytes()    
    
    def pull_mini_boards(self, markers: np.array) -> np.array:
        ''' returns a (3,3,2) array of the miniboards '''
        # Reshape and transpose the markers array to get the desired shape
        return markers.reshape(3, 3, 3, 3, 2).transpose(0, 2, 1, 3, 4).reshape(9,3,3,2)
    
    def check_win(self, miniboxes:np.ndarray, p:int) -> bool:
        '''Check if player p has won the game'''
        for i in range(3):
            if np.sum(miniboxes[:,i, p]) == 3: return True # horizontal
            if np.sum(miniboxes[i,:, p]) == 3: return True # vertical

        # diagonals
        if miniboxes[:, :, p].trace() == 3: return True
        if np.rot90(miniboxes[:, :, p]).trace() == 3: return True
        return False
    
    def get_sorted_moves(self, board: board_obj, tt_move: tuple, ply:int) -> list:
        legal_moves = self.get_valid_moves(board)
        legal_moves.sort(key=lambda move: self.score_move(board, move, tt_move, ply), reverse=True)
        return legal_moves
    
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
        if move[1] % 4 == 0 and np.sum(target_miniboard.diagonal()) == 2:
            score += 10
        elif move[1] % 2 == 0 and np.sum(np.fliplr(target_miniboard).diagonal()) == 2:
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

    def undo_move(self, board_obj: board_obj) -> None:
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

        # check stale
        if np.all(np.any(board_obj.miniboxes,axis=2)): # if all miniboards are filled with something
            return 'stale'

        return 'game is ongoing'

class crossfish_v4:
    '''This version adds an eval term for the state of the won miniboards.
    Test Result: W: 1465, L: 694, D: 529, elo diff: 102.53 +/- 24.32, LOS: 100.00'''
    def __init__(self, name: str = 'Crossfish'):
        self.name = name
        self.thinking_time = 0.1
        self.root_best_move = None
        self.start_time = None
        self.score = 0
        self.root_player = 0
        self.transposition_table = dict()
        self.killer_moves = np.zeros((128, 2), dtype=np.int8) #depth, move
        self.history_table = np.zeros((2, 9, 9), dtype=np.uint16) #player (0 or 1), miniboard, move
        self.nodes = 0
        self.instance_b = board_obj()
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
        self.root_player = b_obj.n_moves % 2
        return self.get_best_move(b_obj)
    
    def get_best_move(self, board: board_obj):
        self.nodes = 0
        depth = 1
        self.start_time = time.time()
        while time.time() - self.start_time < self.thinking_time and depth < 40:
            self.search(board, depth, 0, -np.inf, np.inf)
            depth += 1
            # print(f'depth: {depth-1} in {time.time() - self.start_time:.4f}s, score: {self.score}, nps: {self.nodes / (time.time() - self.start_time + 1e-5):.2f}')
        return self.root_best_move
    
    def search(self, board:board_obj, depth:int, ply: int, alpha: int, beta: int) -> int:
        '''negamax with alpha beta pruning'''
        if time.time() - self.start_time > self.thinking_time:
            return -np.inf
        self.nodes += 1
        if self.check_game_finished(board):
            if self.check_win(board.miniboxes, (board.n_moves + 1) % 2):
                return -100 + ply
            else:
                return 0
        tt_move = None
        try:
            tt_entry = self.transposition_table[self.hash_position(board)]
            tt_move = tt_entry[3]
            if tt_entry[1] >= depth: 
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

        legal_moves = self.get_sorted_moves(board, tt_move, ply)
        best_move = legal_moves[0]
        alpha_orig = alpha
        max_val = -np.inf
        for move_idx in range(len(legal_moves)):
            self.make_move(board, legal_moves[move_idx])
            val = -self.search(board, depth-1, ply+1, -beta, -alpha)
            self.undo_move(board)
            if val > max_val:
                max_val = val
                best_move = legal_moves[move_idx]
                if ply == 0 and abs(val) != np.inf:
                    self.root_best_move = legal_moves[move_idx]
                    self.score = max_val
            alpha = max(alpha, max_val)
            if alpha >= beta:
                self.killer_moves[ply, :] = legal_moves[move_idx]
                self.history_table[board.n_moves % 2, legal_moves[move_idx][0], legal_moves[move_idx][1]] += depth ** 2
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
        score = (np.sum(board.miniboxes[:, :, 0]) - np.sum(board.miniboxes[:, :, 1])) * 2
        # '''TODO: FIND A WAY TO OPTIMIZE THIS'''
        miniboards = self.pull_mini_boards(board.markers)
        boxes_in_play = ~board.miniboxes[:,:,0] & ~board.miniboxes[:,:,1] & ~board.miniboxes[:,:,2]
        two_in_a_row_eval = np.zeros(2, dtype=np.int8)
        for p in range(2):
            q = (p + 1) % 2  # Opponent's index
            for b in miniboards[boxes_in_play.flatten()]:
                b_p = b[:, :, p]
                b_q = b[:, :, q]

                # Precompute sums
                sum_b_p_rows = np.sum(b_p, axis=1)
                sum_b_q_rows = np.sum(b_q, axis=1)
                sum_b_p_cols = np.sum(b_p, axis=0)
                sum_b_q_cols = np.sum(b_q, axis=0)

                # Diagonal checks
                if np.trace(b_p) - np.trace(b_q) == 2 or \
                np.trace(np.fliplr(b_p)) - np.trace(np.fliplr(b_q)) == 2:
                    two_in_a_row_eval[p] += 1
                    continue

                # Row and Column checks
                for i in range(3):
                    if (sum_b_p_rows[i] - sum_b_q_rows[i] == 2) or \
                    (sum_b_p_cols[i] - sum_b_q_cols[i] == 2):
                        two_in_a_row_eval[p] += 1
                        break
            #check if there are any two in a row among the completed miniboards
            b_p = board.miniboxes[:, :, p]
            b_q = board.miniboxes[:, :, q]
            # Precompute sums
            sum_b_p_rows = np.sum(b_p, axis=1)
            sum_b_q_rows = np.sum(b_q, axis=1)
            sum_b_p_cols = np.sum(b_p, axis=0)
            sum_b_q_cols = np.sum(b_q, axis=0)
            # Diagonal checks
            if np.trace(b_p) - np.trace(b_q) == 2 or \
            np.trace(np.fliplr(b_p)) - np.trace(np.fliplr(b_q)) == 2:
                two_in_a_row_eval[p] += 1
            # Row and Column checks
            for i in range(3):
                if (sum_b_p_rows[i] - sum_b_q_rows[i] == 2) or \
                (sum_b_p_cols[i] - sum_b_q_cols[i] == 2):
                    two_in_a_row_eval[p] += 1

        
        score += two_in_a_row_eval[0] - two_in_a_row_eval[1]
        if board.n_moves % 2 == 1:
            score = -score      
        return score
   
    def hash_position(self, board: board_obj) -> bytes:
        int_markers = board.markers.astype(np.int8)
        uniboard = int_markers[:,:,self.root_player] - int_markers[:,:,((self.root_player + 1) % 2)]
        return uniboard.tobytes() + board.hist[board.n_moves-1][1].tobytes()    
    
    def pull_mini_boards(self, markers: np.array) -> np.array:
        ''' returns a (3,3,2) array of the miniboards '''
        # Reshape and transpose the markers array to get the desired shape
        return markers.reshape(3, 3, 3, 3, 2).transpose(0, 2, 1, 3, 4).reshape(9,3,3,2)
    
    def check_win(self, miniboxes:np.ndarray, p:int) -> bool:
        '''Check if player p has won the game'''
        for i in range(3):
            if np.sum(miniboxes[:,i, p]) == 3: return True # horizontal
            if np.sum(miniboxes[i,:, p]) == 3: return True # vertical

        # diagonals
        if miniboxes[:, :, p].trace() == 3: return True
        if np.rot90(miniboxes[:, :, p]).trace() == 3: return True
        return False
    
    def get_sorted_moves(self, board: board_obj, tt_move: tuple, ply:int) -> list:
        legal_moves = self.get_valid_moves(board)
        legal_moves.sort(key=lambda move: self.score_move(board, move, tt_move, ply), reverse=True)
        return legal_moves
    
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
        if move[1] % 4 == 0 and np.sum(target_miniboard.diagonal()) == 2:
            score += 10
        elif move[1] % 2 == 0 and np.sum(np.fliplr(target_miniboard).diagonal()) == 2:
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

        # check stale
        if np.all(np.any(board_obj.miniboxes,axis=2)): # if all miniboards are filled with something
            return 'stale'

        return 'game is ongoing'

class crossfish_v5:
    '''This version adds aspiration windows.
    test results: W: 1711, L: 1386, D: 767, elo diff: 29.29 +/- 19.67, LOS: 100.00'''
    def __init__(self, name: str = 'Crossfish'):
        self.name = name
        self.thinking_time = 0.095
        self.root_best_move = None
        self.start_time = None
        self.score = 0
        self.root_player = 0
        self.transposition_table = dict()
        self.killer_moves = np.zeros((128, 2), dtype=np.int8) #depth, move
        self.history_table = np.zeros((2, 9, 9), dtype=np.uint16) #player (0 or 1), miniboard, move
        self.nodes = 0
        self.instance_b = board_obj()
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
        self.root_player = b_obj.n_moves % 2
        return self.get_best_move(b_obj)
    
    def get_best_move(self, board: board_obj):
        self.nodes = 0
        depth = 1
        self.start_time = time.time()
        aspiration = 2
        alpha = -np.inf
        beta = np.inf
        while time.time() - self.start_time < self.thinking_time and depth < 40:
            self.search(board, depth, 0, alpha, beta)
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

            # print(f'depth: {depth-1} in {time.time() - self.start_time:.4f}s, score: {self.score}, nps: {self.nodes / (time.time() - self.start_time + 1e-5):.2f}')
        return self.root_best_move
    
    def search(self, board:board_obj, depth:int, ply: int, alpha: int, beta: int) -> int:
        '''negamax with alpha beta pruning'''
        if time.time() - self.start_time > self.thinking_time:
            return -np.inf
        self.nodes += 1
        if self.check_game_finished(board):
            if self.check_win(board.miniboxes, (board.n_moves + 1) % 2):
                return -100 + ply
            else:
                return 0
        tt_move = None
        try:
            tt_entry = self.transposition_table[self.hash_position(board)]
            tt_move = tt_entry[3]
            if tt_entry[1] >= depth: 
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

        legal_moves = self.get_sorted_moves(board, tt_move, ply)
        best_move = legal_moves[0]
        alpha_orig = alpha
        max_val = -np.inf
        for move_idx in range(len(legal_moves)):
            self.make_move(board, legal_moves[move_idx])
            val = -self.search(board, depth-1, ply+1, -beta, -alpha)
            self.undo_move(board)
            if val > max_val:
                max_val = val
                best_move = legal_moves[move_idx]
                if ply == 0 and abs(val) != np.inf:
                    self.root_best_move = legal_moves[move_idx]
                    self.score = max_val
            alpha = max(alpha, max_val)
            if alpha >= beta:
                self.killer_moves[ply, :] = legal_moves[move_idx]
                self.history_table[board.n_moves % 2, legal_moves[move_idx][0], legal_moves[move_idx][1]] += depth ** 2
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
        score = (np.sum(board.miniboxes[:, :, 0]) - np.sum(board.miniboxes[:, :, 1])) * 2
        # '''TODO: FIND A WAY TO OPTIMIZE THIS'''
        miniboards = self.pull_mini_boards(board.markers)
        boxes_in_play = ~board.miniboxes[:,:,0] & ~board.miniboxes[:,:,1] & ~board.miniboxes[:,:,2]
        two_in_a_row_eval = np.zeros(2, dtype=np.int8)
        for p in range(2):
            q = (p + 1) % 2  # Opponent's index
            for b in miniboards[boxes_in_play.flatten()]:
                b_p = b[:, :, p]
                b_q = b[:, :, q]

                # Precompute sums
                sum_b_p_rows = np.sum(b_p, axis=1)
                sum_b_q_rows = np.sum(b_q, axis=1)
                sum_b_p_cols = np.sum(b_p, axis=0)
                sum_b_q_cols = np.sum(b_q, axis=0)

                # Diagonal checks
                if np.trace(b_p) - np.trace(b_q) == 2 or \
                np.trace(np.fliplr(b_p)) - np.trace(np.fliplr(b_q)) == 2:
                    two_in_a_row_eval[p] += 1
                    continue

                # Row and Column checks
                for i in range(3):
                    if (sum_b_p_rows[i] - sum_b_q_rows[i] == 2) or \
                    (sum_b_p_cols[i] - sum_b_q_cols[i] == 2):
                        two_in_a_row_eval[p] += 1
                        break
            #check if there are any two in a row among the completed miniboards
            b_p = board.miniboxes[:, :, p]
            b_q = board.miniboxes[:, :, q]
            # Precompute sums
            sum_b_p_rows = np.sum(b_p, axis=1)
            sum_b_q_rows = np.sum(b_q, axis=1)
            sum_b_p_cols = np.sum(b_p, axis=0)
            sum_b_q_cols = np.sum(b_q, axis=0)
            # Diagonal checks
            if np.trace(b_p) - np.trace(b_q) == 2 or \
            np.trace(np.fliplr(b_p)) - np.trace(np.fliplr(b_q)) == 2:
                two_in_a_row_eval[p] += 1
            # Row and Column checks
            for i in range(3):
                if (sum_b_p_rows[i] - sum_b_q_rows[i] == 2) or \
                (sum_b_p_cols[i] - sum_b_q_cols[i] == 2):
                    two_in_a_row_eval[p] += 1

        
        score += two_in_a_row_eval[0] - two_in_a_row_eval[1]
        if board.n_moves % 2 == 1:
            score = -score      
        return score
   
    def hash_position(self, board: board_obj) -> bytes:
        int_markers = board.markers.astype(np.int8)
        uniboard = int_markers[:,:,self.root_player] - int_markers[:,:,((self.root_player + 1) % 2)]
        return uniboard.tobytes() + board.hist[board.n_moves-1][1].tobytes()    
    
    def pull_mini_boards(self, markers: np.array) -> np.array:
        ''' returns a (3,3,2) array of the miniboards '''
        # Reshape and transpose the markers array to get the desired shape
        return markers.reshape(3, 3, 3, 3, 2).transpose(0, 2, 1, 3, 4).reshape(9,3,3,2)
    
    def check_win(self, miniboxes:np.ndarray, p:int) -> bool:
        '''Check if player p has won the game'''
        for i in range(3):
            if np.sum(miniboxes[:,i, p]) == 3: return True # horizontal
            if np.sum(miniboxes[i,:, p]) == 3: return True # vertical

        # diagonals
        if miniboxes[:, :, p].trace() == 3: return True
        if np.rot90(miniboxes[:, :, p]).trace() == 3: return True
        return False
    
    def get_sorted_moves(self, board: board_obj, tt_move: tuple, ply:int) -> list:
        legal_moves = self.get_valid_moves(board)
        legal_moves.sort(key=lambda move: self.score_move(board, move, tt_move, ply), reverse=True)
        return legal_moves
    
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
        if move[1] % 4 == 0 and np.sum(target_miniboard.diagonal()) == 2:
            score += 10
        elif move[1] % 2 == 0 and np.sum(np.fliplr(target_miniboard).diagonal()) == 2:
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

        # check stale
        if np.all(np.any(board_obj.miniboxes,axis=2)): # if all miniboards are filled with something
            return 'stale'

        return 'game is ongoing'

class crossfish_v6:
    '''This version null move pruning.
    Test result: W: 378, L: 237, D: 105, elo diff: 68.93 +/- 47.76, LOS: 100.00'''
    def __init__(self, name: str = 'Crossfish'):
        self.name = name
        self.thinking_time = 0.3
        self.root_best_move = None
        self.start_time = None
        self.score = 0
        self.root_player = 0
        self.transposition_table = dict()
        self.killer_moves = np.zeros((128, 2), dtype=np.int8) #depth, move
        self.history_table = np.zeros((2, 9, 9), dtype=np.uint16) #player (0 or 1), miniboard, move
        self.nodes = 0
        self.instance_b = board_obj()
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
        self.root_player = b_obj.n_moves % 2
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

            # print(f'depth: {depth-1} in {time.time() - self.start_time:.4f}s, score: {self.score}, nps: {self.nodes / (time.time() - self.start_time + 1e-5):.2f}')
        return self.root_best_move
    
    def search(self, board:board_obj, depth:int, ply: int, alpha: int, beta: int, can_null:bool) -> int:
        '''negamax with alpha beta pruning'''
        if time.time() - self.start_time > self.thinking_time:
            return -np.inf
        self.nodes += 1
        pv_node = (alpha + 0.1 == beta)
        can_null = can_null and ply > 0 and not pv_node
        if self.check_game_finished(board):
            if self.check_win(board.miniboxes, (board.n_moves + 1) % 2):
                return -100 + ply
            else:
                return 0
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
        if not pv_node:
            # stand_pat = self.evaluate(board)

            # #reverse futility pruning
            # r_margin = 1
            # if stand_pat - r_margin * depth >= beta:
            #     return beta

            # #futility pruning
            # f_margin = 1
            # can_futility_prune = (stand_pat + f_margin * depth) <= alpha

            #null move pruning
            # if not can_futility_prune:
            if depth > 2 and can_null:
                self.make_null_move(board)
                null_move_score = -self.search(board, depth//2, ply+1, -beta, -beta+1, can_null=False)
                self.undo_move(board)
                if null_move_score >= beta:
                    return beta


        legal_moves_and_scores = self.get_sorted_moves_and_scores(board, tt_move, ply)
        best_move = legal_moves_and_scores[0][0]
        alpha_orig = alpha
        max_val = -np.inf
        for move_idx in range(len(legal_moves_and_scores)):
            move = legal_moves_and_scores[move_idx][0]
            move_score = legal_moves_and_scores[move_idx][1]
            if can_futility_prune and move_idx > 0 and move_score <= 0:
                continue
            self.make_move(board, move)
            if move_idx == 0:
                val = -self.search(board, depth-1, ply+1, -beta, -alpha, can_null=can_null)
            else:
                val = -self.search(board, depth-1, ply+1, -alpha-0.1, -alpha, can_null=can_null)
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
        score = (np.sum(board.miniboxes[:, :, 0]) - np.sum(board.miniboxes[:, :, 1])) * 2
        # '''TODO: FIND A WAY TO OPTIMIZE THIS'''
        miniboards = self.pull_mini_boards(board.markers)
        boxes_in_play = ~board.miniboxes[:,:,0] & ~board.miniboxes[:,:,1] & ~board.miniboxes[:,:,2]
        two_in_a_row_eval = np.zeros(2, dtype=np.int8)
        for p in range(2):
            q = (p + 1) % 2  # Opponent's index
            for b in miniboards[boxes_in_play.flatten()]:
                b_p = b[:, :, p]
                b_q = b[:, :, q]

                # Precompute sums
                sum_b_p_rows = np.sum(b_p, axis=1)
                sum_b_q_rows = np.sum(b_q, axis=1)
                sum_b_p_cols = np.sum(b_p, axis=0)
                sum_b_q_cols = np.sum(b_q, axis=0)

                # Diagonal checks
                if np.trace(b_p) - np.trace(b_q) == 2 or \
                np.trace(np.fliplr(b_p)) - np.trace(np.fliplr(b_q)) == 2:
                    two_in_a_row_eval[p] += 1
                    continue

                # Row and Column checks
                for i in range(3):
                    if (sum_b_p_rows[i] - sum_b_q_rows[i] == 2) or \
                    (sum_b_p_cols[i] - sum_b_q_cols[i] == 2):
                        two_in_a_row_eval[p] += 1
                        break
            #check if there are any two in a row among the completed miniboards
            b_p = board.miniboxes[:, :, p]
            b_q = board.miniboxes[:, :, q]
            # Precompute sums
            sum_b_p_rows = np.sum(b_p, axis=1)
            sum_b_q_rows = np.sum(b_q, axis=1)
            sum_b_p_cols = np.sum(b_p, axis=0)
            sum_b_q_cols = np.sum(b_q, axis=0)
            # Diagonal checks
            if np.trace(b_p) - np.trace(b_q) == 2 or \
            np.trace(np.fliplr(b_p)) - np.trace(np.fliplr(b_q)) == 2:
                two_in_a_row_eval[p] += 1
            # Row and Column checks
            for i in range(3):
                if (sum_b_p_rows[i] - sum_b_q_rows[i] == 2) or \
                (sum_b_p_cols[i] - sum_b_q_cols[i] == 2):
                    two_in_a_row_eval[p] += 1

        
        score += two_in_a_row_eval[0] - two_in_a_row_eval[1]
        if board.n_moves % 2 == 1:
            score = -score      
        return score
   
    def hash_position(self, board: board_obj) -> bytes:
        int_markers = board.markers.astype(np.int8)
        uniboard = int_markers[:,:,self.root_player] - int_markers[:,:,((self.root_player + 1) % 2)]
        return uniboard.tobytes() + board.hist[board.n_moves-1][1].tobytes()    
    
    def pull_mini_boards(self, markers: np.array) -> np.array:
        ''' returns a (3,3,2) array of the miniboards '''
        # Reshape and transpose the markers array to get the desired shape
        return markers.reshape(3, 3, 3, 3, 2).transpose(0, 2, 1, 3, 4).reshape(9,3,3,2)
    
    def check_win(self, miniboxes:np.ndarray, p:int) -> bool:
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
        # legal_moves.sort(key=lambda move: self.score_move(board, move, tt_move, ply), reverse=True)
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
        if move[1] % 4 == 0 and np.sum(target_miniboard.diagonal()) == 2:
            score += 10
        elif move[1] % 2 == 0 and np.sum(np.fliplr(target_miniboard).diagonal()) == 2:
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

        # check stale
        if np.all(np.any(board_obj.miniboxes,axis=2)): # if all miniboards are filled with something
            return 'stale'

        return 'game is ongoing'

class crossfish_v7:
    '''This version adds futility pruning.
    Test result: W: 559, L: 456, D: 185, elo diff: 29.90 +/- 36.30, LOS: 99.94'''
    def __init__(self, name: str = 'Crossfish'):
        self.name = name
        self.thinking_time = 0.3
        self.root_best_move = None
        self.start_time = None
        self.score = 0
        self.root_player = 0
        self.transposition_table = dict()
        self.killer_moves = np.zeros((128, 2), dtype=np.int8) #depth, move
        self.history_table = np.zeros((2, 9, 9), dtype=np.uint16) #player (0 or 1), miniboard, move
        self.nodes = 0
        self.instance_b = board_obj()
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
        self.root_player = b_obj.n_moves % 2
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

            # print(f'depth: {depth-1} in {time.time() - self.start_time:.4f}s, score: {self.score}, nps: {self.nodes / (time.time() - self.start_time + 1e-5):.2f}')
        return self.root_best_move
    
    def search(self, board:board_obj, depth:int, ply: int, alpha: int, beta: int, can_null:bool) -> int:
        '''negamax with alpha beta pruning'''
        if time.time() - self.start_time > self.thinking_time:
            return -np.inf
        self.nodes += 1
        pv_node = (alpha + 0.1 == beta)
        can_null = can_null and ply > 0 and not pv_node
        if self.check_game_finished(board):
            if self.check_win(board.miniboxes, (board.n_moves + 1) % 2):
                return -100 + ply
            else:
                return 0
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
        if not pv_node:
            stand_pat = self.evaluate(board)

            # #reverse futility pruning
            # r_margin = 1
            # if stand_pat - r_margin * depth >= beta:
            #     return beta

            #futility pruning
            f_margin = 1
            can_futility_prune = (stand_pat + f_margin * depth) <= alpha

            #null move pruning
            # if not can_futility_prune:
            if depth > 2 and can_null:
                self.make_null_move(board)
                null_move_score = -self.search(board, depth//2, ply+1, -beta, -beta+1, can_null=False)
                self.undo_move(board)
                if null_move_score >= beta:
                    return beta


        legal_moves_and_scores = self.get_sorted_moves_and_scores(board, tt_move, ply)
        best_move = legal_moves_and_scores[0][0]
        alpha_orig = alpha
        max_val = -np.inf
        for move_idx in range(len(legal_moves_and_scores)):
            move = legal_moves_and_scores[move_idx][0]
            move_score = legal_moves_and_scores[move_idx][1]
            if can_futility_prune and move_idx > 0 and move_score <= 0:
                continue
            self.make_move(board, move)
            if move_idx == 0:
                val = -self.search(board, depth-1, ply+1, -beta, -alpha, can_null=can_null)
            else:
                val = -self.search(board, depth-1, ply+1, -alpha-0.1, -alpha, can_null=can_null)
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
        score = (np.sum(board.miniboxes[:, :, 0]) - np.sum(board.miniboxes[:, :, 1])) * 2
        # '''TODO: FIND A WAY TO OPTIMIZE THIS'''
        miniboards = self.pull_mini_boards(board.markers)
        boxes_in_play = ~board.miniboxes[:,:,0] & ~board.miniboxes[:,:,1] & ~board.miniboxes[:,:,2]
        two_in_a_row_eval = np.zeros(2, dtype=np.int8)
        for p in range(2):
            q = (p + 1) % 2  # Opponent's index
            for b in miniboards[boxes_in_play.flatten()]:
                b_p = b[:, :, p]
                b_q = b[:, :, q]

                # Precompute sums
                sum_b_p_rows = np.sum(b_p, axis=1)
                sum_b_q_rows = np.sum(b_q, axis=1)
                sum_b_p_cols = np.sum(b_p, axis=0)
                sum_b_q_cols = np.sum(b_q, axis=0)

                # Diagonal checks
                if np.trace(b_p) - np.trace(b_q) == 2 or \
                np.trace(np.fliplr(b_p)) - np.trace(np.fliplr(b_q)) == 2:
                    two_in_a_row_eval[p] += 1
                    continue

                # Row and Column checks
                for i in range(3):
                    if (sum_b_p_rows[i] - sum_b_q_rows[i] == 2) or \
                    (sum_b_p_cols[i] - sum_b_q_cols[i] == 2):
                        two_in_a_row_eval[p] += 1
                        break
            #check if there are any two in a row among the completed miniboards
            b_p = board.miniboxes[:, :, p]
            b_q = board.miniboxes[:, :, q]
            # Precompute sums
            sum_b_p_rows = np.sum(b_p, axis=1)
            sum_b_q_rows = np.sum(b_q, axis=1)
            sum_b_p_cols = np.sum(b_p, axis=0)
            sum_b_q_cols = np.sum(b_q, axis=0)
            # Diagonal checks
            if np.trace(b_p) - np.trace(b_q) == 2 or \
            np.trace(np.fliplr(b_p)) - np.trace(np.fliplr(b_q)) == 2:
                two_in_a_row_eval[p] += 1
            # Row and Column checks
            for i in range(3):
                if (sum_b_p_rows[i] - sum_b_q_rows[i] == 2) or \
                (sum_b_p_cols[i] - sum_b_q_cols[i] == 2):
                    two_in_a_row_eval[p] += 1

        
        score += two_in_a_row_eval[0] - two_in_a_row_eval[1]
        if board.n_moves % 2 == 1:
            score = -score      
        return score
   
    def hash_position(self, board: board_obj) -> bytes:
        int_markers = board.markers.astype(np.int8)
        uniboard = int_markers[:,:,self.root_player] - int_markers[:,:,((self.root_player + 1) % 2)]
        return uniboard.tobytes() + board.hist[board.n_moves-1][1].tobytes()    
    
    def pull_mini_boards(self, markers: np.array) -> np.array:
        ''' returns a (3,3,2) array of the miniboards '''
        # Reshape and transpose the markers array to get the desired shape
        return markers.reshape(3, 3, 3, 3, 2).transpose(0, 2, 1, 3, 4).reshape(9,3,3,2)
    
    def check_win(self, miniboxes:np.ndarray, p:int) -> bool:
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
        # legal_moves.sort(key=lambda move: self.score_move(board, move, tt_move, ply), reverse=True)
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
        if move[1] % 4 == 0 and np.sum(target_miniboard.diagonal()) == 2:
            score += 10
        elif move[1] % 2 == 0 and np.sum(np.fliplr(target_miniboard).diagonal()) == 2:
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

        # check stale
        if np.all(np.any(board_obj.miniboxes,axis=2)): # if all miniboards are filled with something
            return 'stale'

        return 'game is ongoing'

class crossfish_v8:
    '''This version adds reverse futility pruning.
    Test result: W: 1027, L: 743, D: 390, elo diff: 45.95 +/- 26.72, LOS: 100.00'''
    def __init__(self, name: str = 'Crossfish'):
        self.name = name
        self.thinking_time = 0.1
        self.root_best_move = None
        self.start_time = None
        self.score = 0
        self.root_player = 0
        self.transposition_table = dict()
        self.killer_moves = np.zeros((128, 2), dtype=np.int8) #depth, move
        self.history_table = np.zeros((2, 9, 9), dtype=np.uint16) #player (0 or 1), miniboard, move
        self.nodes = 0
        self.instance_b = board_obj()
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
        self.root_player = b_obj.n_moves % 2
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

            # print(f'depth: {depth-1} in {time.time() - self.start_time:.4f}s, score: {self.score}, nps: {self.nodes / (time.time() - self.start_time + 1e-5):.2f}')
        return self.root_best_move
    
    def search(self, board:board_obj, depth:int, ply: int, alpha: int, beta: int, can_null:bool) -> int:
        '''negamax with alpha beta pruning'''
        if time.time() - self.start_time > self.thinking_time:
            return -np.inf
        self.nodes += 1
        pv_node = (alpha + 0.1 == beta)
        can_null = can_null and ply > 0 and not pv_node
        if self.check_game_finished(board):
            if self.check_win(board.miniboxes, (board.n_moves + 1) % 2):
                return -100 + ply
            else:
                return 0
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
        if not pv_node:
            stand_pat = self.evaluate(board)

            # #reverse futility pruning
            r_margin = 1
            if stand_pat - r_margin * depth >= beta:
                return beta

            #futility pruning
            f_margin = 1
            can_futility_prune = (stand_pat + f_margin * depth) <= alpha

            #null move pruning
            # if not can_futility_prune:
            if depth > 2 and can_null:
                self.make_null_move(board)
                null_move_score = -self.search(board, depth//2, ply+1, -beta, -beta+1, can_null=False)
                self.undo_move(board)
                if null_move_score >= beta:
                    return beta


        legal_moves_and_scores = self.get_sorted_moves_and_scores(board, tt_move, ply)
        best_move = legal_moves_and_scores[0][0]
        alpha_orig = alpha
        max_val = -np.inf
        for move_idx in range(len(legal_moves_and_scores)):
            move = legal_moves_and_scores[move_idx][0]
            move_score = legal_moves_and_scores[move_idx][1]
            if can_futility_prune and move_idx > 0 and move_score <= 0:
                continue
            self.make_move(board, move)
            if move_idx == 0:
                val = -self.search(board, depth-1, ply+1, -beta, -alpha, can_null=can_null)
            else:
                val = -self.search(board, depth-1, ply+1, -alpha-0.1, -alpha, can_null=can_null)
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
        score = (np.sum(board.miniboxes[:, :, 0]) - np.sum(board.miniboxes[:, :, 1])) * 2
        # '''TODO: FIND A WAY TO OPTIMIZE THIS'''
        miniboards = self.pull_mini_boards(board.markers)
        boxes_in_play = ~board.miniboxes[:,:,0] & ~board.miniboxes[:,:,1] & ~board.miniboxes[:,:,2]
        two_in_a_row_eval = np.zeros(2, dtype=np.int8)
        for p in range(2):
            q = (p + 1) % 2  # Opponent's index
            for b in miniboards[boxes_in_play.flatten()]:
                b_p = b[:, :, p]
                b_q = b[:, :, q]

                # Precompute sums
                sum_b_p_rows = np.sum(b_p, axis=1)
                sum_b_q_rows = np.sum(b_q, axis=1)
                sum_b_p_cols = np.sum(b_p, axis=0)
                sum_b_q_cols = np.sum(b_q, axis=0)

                # Diagonal checks
                if np.trace(b_p) - np.trace(b_q) == 2 or \
                np.trace(np.fliplr(b_p)) - np.trace(np.fliplr(b_q)) == 2:
                    two_in_a_row_eval[p] += 1
                    continue

                # Row and Column checks
                for i in range(3):
                    if (sum_b_p_rows[i] - sum_b_q_rows[i] == 2) or \
                    (sum_b_p_cols[i] - sum_b_q_cols[i] == 2):
                        two_in_a_row_eval[p] += 1
                        break
            #check if there are any two in a row among the completed miniboards
            b_p = board.miniboxes[:, :, p]
            b_q = board.miniboxes[:, :, q]
            # Precompute sums
            sum_b_p_rows = np.sum(b_p, axis=1)
            sum_b_q_rows = np.sum(b_q, axis=1)
            sum_b_p_cols = np.sum(b_p, axis=0)
            sum_b_q_cols = np.sum(b_q, axis=0)
            # Diagonal checks
            if np.trace(b_p) - np.trace(b_q) == 2 or \
            np.trace(np.fliplr(b_p)) - np.trace(np.fliplr(b_q)) == 2:
                two_in_a_row_eval[p] += 1
            # Row and Column checks
            for i in range(3):
                if (sum_b_p_rows[i] - sum_b_q_rows[i] == 2) or \
                (sum_b_p_cols[i] - sum_b_q_cols[i] == 2):
                    two_in_a_row_eval[p] += 1

        
        score += two_in_a_row_eval[0] - two_in_a_row_eval[1]
        if board.n_moves % 2 == 1:
            score = -score      
        return score
   
    def hash_position(self, board: board_obj) -> bytes:
        int_markers = board.markers.astype(np.int8)
        uniboard = int_markers[:,:,self.root_player] - int_markers[:,:,((self.root_player + 1) % 2)]
        return uniboard.tobytes() + board.hist[board.n_moves-1][1].tobytes()    
    
    def pull_mini_boards(self, markers: np.array) -> np.array:
        ''' returns a (3,3,2) array of the miniboards '''
        # Reshape and transpose the markers array to get the desired shape
        return markers.reshape(3, 3, 3, 3, 2).transpose(0, 2, 1, 3, 4).reshape(9,3,3,2)
    
    def check_win(self, miniboxes:np.ndarray, p:int) -> bool:
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
        # legal_moves.sort(key=lambda move: self.score_move(board, move, tt_move, ply), reverse=True)
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
        if move[1] % 4 == 0 and np.sum(target_miniboard.diagonal()) == 2:
            score += 10
        elif move[1] % 2 == 0 and np.sum(np.fliplr(target_miniboard).diagonal()) == 2:
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

        # check stale
        if np.all(np.any(board_obj.miniboxes,axis=2)): # if all miniboards are filled with something
            return 'stale'

        return 'game is ongoing'
    

class crossfish_v9:
    '''This version adapts to the rules on codingame.
    unlike the hackathon, the winner on a full board without a line is the player with more miniboards
    Test Results:W: 776, L: 537, D: 103, elo diff: 59.21 +/- 35.35, LOS: 100.00
    '''
    def __init__(self, name: str = 'Crossfish'):
        self.name = name
        self.thinking_time = 0.095
        self.root_best_move = None
        self.start_time = None
        self.score = 0
        self.root_player = 0
        self.transposition_table = dict()
        self.killer_moves = np.zeros((128, 2), dtype=np.int8) #depth, move
        self.history_table = np.zeros((2, 9, 9), dtype=np.uint16) #player (0 or 1), miniboard, move
        self.nodes = 0
        self.instance_b = board_obj()
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
        self.root_player = b_obj.n_moves % 2
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

            # print(f'depth: {depth-1} in {time.time() - self.start_time:.4f}s, score: {self.score}, nps: {self.nodes / (time.time() - self.start_time + 1e-5):.2f}')
        return self.root_best_move
    
    def search(self, board:board_obj, depth:int, ply: int, alpha: int, beta: int, can_null:bool) -> int:
        '''negamax with alpha beta pruning'''
        if time.time() - self.start_time > self.thinking_time:
            return -np.inf
        self.nodes += 1
        # pv_node = (alpha + 0.1 == beta)
        can_null = can_null and ply > 0
        if self.check_game_finished(board):
            if self.check_win_via_line(board.miniboxes, (board.n_moves + 1) % 2):
                return -100 + ply
            else:
                #if global board is stale, winner is the player with the most boxes
                return (-1)**(board.n_moves) * (np.sum(board.miniboxes[:, :, 0]) - np.sum(board.miniboxes[:, :, 1]))
        tt_move = None
        try:
            tt_entry = self.transposition_table[self.hash_position(board)]
            tt_move = tt_entry[3]
            if tt_entry[1] >= depth: 
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
        # can_futility_prune = False
        # if not pv_node:
            # stand_pat = self.evaluate(board)

            # #reverse futility pruning
            # r_margin = 1
            # if stand_pat - r_margin * depth >= beta:
            #     return beta

            # #futility pruning
            # f_margin = 1
            # can_futility_prune = (stand_pat + f_margin * depth) <= alpha

            #null move pruning
            # if not can_futility_prune:
        # if depth > 2 and can_null:
        #     self.make_null_move(board)
        #     null_move_score = -self.search(board, depth//2, ply+1, -beta, -beta+1, can_null=False)
        #     self.undo_move(board)
        #     if null_move_score >= beta:
        #         return beta


        legal_moves_and_scores = self.get_sorted_moves_and_scores(board, tt_move, ply)
        best_move = legal_moves_and_scores[0][0]
        alpha_orig = alpha
        max_val = -np.inf
        for move_idx in range(len(legal_moves_and_scores)):
            move = legal_moves_and_scores[move_idx][0]
            # move_score = legal_moves_and_scores[move_idx][1]
            # if can_futility_prune and move_idx > 0 and move_score <= 0:
            #     continue
            self.make_move(board, move)
            # if move_idx == 0:
            val = -self.search(board, depth-1, ply+1, -beta, -alpha, can_null=can_null)
            # else:
            #     val = -self.search(board, depth-1, ply+1, -alpha-0.1, -alpha, can_null=can_null)
            #     if alpha < val and val < beta:
            #         val = -self.search(board, depth-1, ply+1, -beta, -alpha, can_null=can_null)
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
        score = (np.sum(board.miniboxes[:, :, 0]) - np.sum(board.miniboxes[:, :, 1])) * 2
        # '''TODO: FIND A WAY TO OPTIMIZE THIS'''
        miniboards = self.pull_mini_boards(board.markers)
        boxes_in_play = ~board.miniboxes[:,:,0] & ~board.miniboxes[:,:,1] & ~board.miniboxes[:,:,2]
        two_in_a_row_eval = np.zeros(2, dtype=np.int8)
        for p in range(2):
            q = (p + 1) % 2  # Opponent's index
            for b in miniboards[boxes_in_play.flatten()]:
                b_p = b[:, :, p]
                b_q = b[:, :, q]

                # Precompute sums
                sum_b_p_rows = np.sum(b_p, axis=1)
                sum_b_q_rows = np.sum(b_q, axis=1)
                sum_b_p_cols = np.sum(b_p, axis=0)
                sum_b_q_cols = np.sum(b_q, axis=0)

                # Diagonal checks
                if np.trace(b_p) - np.trace(b_q) == 2 or \
                np.trace(np.fliplr(b_p)) - np.trace(np.fliplr(b_q)) == 2:
                    two_in_a_row_eval[p] += 1
                    continue

                # Row and Column checks
                for i in range(3):
                    if (sum_b_p_rows[i] - sum_b_q_rows[i] == 2) or \
                    (sum_b_p_cols[i] - sum_b_q_cols[i] == 2):
                        two_in_a_row_eval[p] += 1
                        break
            #check if there are any two in a row among the completed miniboards
            b_p = board.miniboxes[:, :, p]
            b_q = board.miniboxes[:, :, q]
            # Precompute sums
            sum_b_p_rows = np.sum(b_p, axis=1)
            sum_b_q_rows = np.sum(b_q, axis=1)
            sum_b_p_cols = np.sum(b_p, axis=0)
            sum_b_q_cols = np.sum(b_q, axis=0)
            # Diagonal checks
            if np.trace(b_p) - np.trace(b_q) == 2 or \
            np.trace(np.fliplr(b_p)) - np.trace(np.fliplr(b_q)) == 2:
                two_in_a_row_eval[p] += 1
            # Row and Column checks
            for i in range(3):
                if (sum_b_p_rows[i] - sum_b_q_rows[i] == 2) or \
                (sum_b_p_cols[i] - sum_b_q_cols[i] == 2):
                    two_in_a_row_eval[p] += 1

        
        score += two_in_a_row_eval[0] - two_in_a_row_eval[1]
        if board.n_moves % 2 == 1:
            score = -score      
        return score
   
    def hash_position(self, board: board_obj) -> bytes:
        int_markers = board.markers.astype(np.int8)
        uniboard = int_markers[:,:,self.root_player] - int_markers[:,:,((self.root_player + 1) % 2)]
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
        # legal_moves.sort(key=lambda move: self.score_move(board, move, tt_move, ply), reverse=True)
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
        if move[1] % 4 == 0 and np.sum(target_miniboard.diagonal()) == 2:
            score += 10
        elif move[1] % 2 == 0 and np.sum(np.fliplr(target_miniboard).diagonal()) == 2:
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
    

class crossfish_v10:
    '''This version adds futility pruning
    test results: W: 1713, L: 1373, D: 226, elo diff: 35.79 +/- 22.96, LOS: 100.00
    '''
    def __init__(self, name: str = 'Crossfish'):
        self.name = name
        self.thinking_time = 0.095
        self.root_best_move = None
        self.start_time = None
        self.score = 0
        self.root_player = 0
        self.transposition_table = dict()
        self.killer_moves = np.zeros((128, 2), dtype=np.int8) #depth, move
        self.history_table = np.zeros((2, 9, 9), dtype=np.uint16) #player (0 or 1), miniboard, move
        self.nodes = 0
        self.instance_b = board_obj()
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
        self.root_player = b_obj.n_moves % 2
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

            # print(f'depth: {depth-1} in {time.time() - self.start_time:.4f}s, score: {self.score}, nps: {self.nodes / (time.time() - self.start_time + 1e-5):.2f}')
        return self.root_best_move
    
    def search(self, board:board_obj, depth:int, ply: int, alpha: int, beta: int, can_null:bool) -> int:
        '''negamax with alpha beta pruning'''
        if time.time() - self.start_time > self.thinking_time:
            return -np.inf
        self.nodes += 1
        # pv_node = (alpha + 0.1 == beta)
        can_null = can_null and ply > 0
        if self.check_game_finished(board):
            if self.check_win_via_line(board.miniboxes, (board.n_moves + 1) % 2):
                return -100 + ply
            else:
                #if global board is stale, winner is the player with the most boxes
                return (-1)**(board.n_moves) * (np.sum(board.miniboxes[:, :, 0]) - np.sum(board.miniboxes[:, :, 1]))
        tt_move = None
        try:
            tt_entry = self.transposition_table[self.hash_position(board)]
            tt_move = tt_entry[3]
            if tt_entry[1] >= depth: 
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
        # if not pv_node:
        stand_pat = self.evaluate(board)

            # #reverse futility pruning
            # r_margin = 1
            # if stand_pat - r_margin * depth >= beta:
            #     return beta

            # #futility pruning
        f_margin = 2
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
        for move_idx in range(len(legal_moves_and_scores)):
            move = legal_moves_and_scores[move_idx][0]
            move_score = legal_moves_and_scores[move_idx][1]
            if can_futility_prune and move_idx > 0 and move_score <= 0:
                continue
            self.make_move(board, move)
            # if move_idx == 0:
            val = -self.search(board, depth-1, ply+1, -beta, -alpha, can_null=can_null)
            # else:
            #     val = -self.search(board, depth-1, ply+1, -alpha-0.1, -alpha, can_null=can_null)
            #     if alpha < val and val < beta:
            #         val = -self.search(board, depth-1, ply+1, -beta, -alpha, can_null=can_null)
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
        score = (np.sum(board.miniboxes[:, :, 0]) - np.sum(board.miniboxes[:, :, 1])) * 2
        # '''TODO: FIND A WAY TO OPTIMIZE THIS'''
        miniboards = self.pull_mini_boards(board.markers)
        boxes_in_play = ~board.miniboxes[:,:,0] & ~board.miniboxes[:,:,1] & ~board.miniboxes[:,:,2]
        two_in_a_row_eval = np.zeros(2, dtype=np.int8)
        for p in range(2):
            q = (p + 1) % 2  # Opponent's index
            for b in miniboards[boxes_in_play.flatten()]:
                b_p = b[:, :, p]
                b_q = b[:, :, q]

                # Precompute sums
                sum_b_p_rows = np.sum(b_p, axis=1)
                sum_b_q_rows = np.sum(b_q, axis=1)
                sum_b_p_cols = np.sum(b_p, axis=0)
                sum_b_q_cols = np.sum(b_q, axis=0)

                # Diagonal checks
                if np.trace(b_p) - np.trace(b_q) == 2 or \
                np.trace(np.fliplr(b_p)) - np.trace(np.fliplr(b_q)) == 2:
                    two_in_a_row_eval[p] += 1
                    continue

                # Row and Column checks
                for i in range(3):
                    if (sum_b_p_rows[i] - sum_b_q_rows[i] == 2) or \
                    (sum_b_p_cols[i] - sum_b_q_cols[i] == 2):
                        two_in_a_row_eval[p] += 1
                        break
            #check if there are any two in a row among the completed miniboards
            b_p = board.miniboxes[:, :, p]
            b_q = board.miniboxes[:, :, q]
            # Precompute sums
            sum_b_p_rows = np.sum(b_p, axis=1)
            sum_b_q_rows = np.sum(b_q, axis=1)
            sum_b_p_cols = np.sum(b_p, axis=0)
            sum_b_q_cols = np.sum(b_q, axis=0)
            # Diagonal checks
            if np.trace(b_p) - np.trace(b_q) == 2 or \
            np.trace(np.fliplr(b_p)) - np.trace(np.fliplr(b_q)) == 2:
                two_in_a_row_eval[p] += 1
            # Row and Column checks
            for i in range(3):
                if (sum_b_p_rows[i] - sum_b_q_rows[i] == 2) or \
                (sum_b_p_cols[i] - sum_b_q_cols[i] == 2):
                    two_in_a_row_eval[p] += 1

        
        score += two_in_a_row_eval[0] - two_in_a_row_eval[1]
        if board.n_moves % 2 == 1:
            score = -score      
        return score
   
    def hash_position(self, board: board_obj) -> bytes:
        int_markers = board.markers.astype(np.int8)
        uniboard = int_markers[:,:,self.root_player] - int_markers[:,:,((self.root_player + 1) % 2)]
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
        # legal_moves.sort(key=lambda move: self.score_move(board, move, tt_move, ply), reverse=True)
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
        if move[1] % 4 == 0 and np.sum(target_miniboard.diagonal()) == 2:
            score += 10
        elif move[1] % 2 == 0 and np.sum(np.fliplr(target_miniboard).diagonal()) == 2:
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
    

class crossfish_v11:
    '''This version adds reverse futility pruning
    test results: W: 2028, L: 1641, D: 291, elo diff: 34.06 +/- 20.93, LOS: 100.00
    '''
    def __init__(self, name: str = 'Crossfish'):
        self.name = name
        self.thinking_time = 0.095
        self.root_best_move = None
        self.start_time = None
        self.score = 0
        self.root_player = 0
        self.transposition_table = dict()
        self.killer_moves = np.zeros((128, 2), dtype=np.int8) #depth, move
        self.history_table = np.zeros((2, 9, 9), dtype=np.uint16) #player (0 or 1), miniboard, move
        self.nodes = 0
        self.instance_b = board_obj()
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
        self.root_player = b_obj.n_moves % 2
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

            # print(f'depth: {depth-1} in {time.time() - self.start_time:.4f}s, score: {self.score}, nps: {self.nodes / (time.time() - self.start_time + 1e-5):.2f}')
        return self.root_best_move
    
    def search(self, board:board_obj, depth:int, ply: int, alpha: int, beta: int, can_null:bool) -> int:
        '''negamax with alpha beta pruning'''
        if time.time() - self.start_time > self.thinking_time:
            return -np.inf
        self.nodes += 1
        # pv_node = (alpha + 0.1 == beta)
        can_null = can_null and ply > 0
        if self.check_game_finished(board):
            if self.check_win_via_line(board.miniboxes, (board.n_moves + 1) % 2):
                return -100 + ply
            else:
                #if global board is stale, winner is the player with the most boxes
                return (-1)**(board.n_moves) * (np.sum(board.miniboxes[:, :, 0]) - np.sum(board.miniboxes[:, :, 1]))
        tt_move = None
        try:
            tt_entry = self.transposition_table[self.hash_position(board)]
            tt_move = tt_entry[3]
            if tt_entry[1] >= depth: 
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
        # if not pv_node:
        stand_pat = self.evaluate(board)

        #reverse futility pruning
        rfp_margin = 2
        if stand_pat - rfp_margin * depth >= beta:
            return beta

        # #futility pruning
        f_margin = 2
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
        for move_idx in range(len(legal_moves_and_scores)):
            move = legal_moves_and_scores[move_idx][0]
            move_score = legal_moves_and_scores[move_idx][1]
            if can_futility_prune and move_idx > 0 and move_score <= 0:
                continue
            self.make_move(board, move)
            # if move_idx == 0:
            val = -self.search(board, depth-1, ply+1, -beta, -alpha, can_null=can_null)
            # else:
            #     val = -self.search(board, depth-1, ply+1, -alpha-0.1, -alpha, can_null=can_null)
            #     if alpha < val and val < beta:
            #         val = -self.search(board, depth-1, ply+1, -beta, -alpha, can_null=can_null)
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
        score = (np.sum(board.miniboxes[:, :, 0]) - np.sum(board.miniboxes[:, :, 1])) * 2
        # '''TODO: FIND A WAY TO OPTIMIZE THIS'''
        miniboards = self.pull_mini_boards(board.markers)
        boxes_in_play = ~board.miniboxes[:,:,0] & ~board.miniboxes[:,:,1] & ~board.miniboxes[:,:,2]
        two_in_a_row_eval = np.zeros(2, dtype=np.int8)
        for p in range(2):
            q = (p + 1) % 2  # Opponent's index
            for b in miniboards[boxes_in_play.flatten()]:
                b_p = b[:, :, p]
                b_q = b[:, :, q]

                # Precompute sums
                sum_b_p_rows = np.sum(b_p, axis=1)
                sum_b_q_rows = np.sum(b_q, axis=1)
                sum_b_p_cols = np.sum(b_p, axis=0)
                sum_b_q_cols = np.sum(b_q, axis=0)

                # Diagonal checks
                if np.trace(b_p) - np.trace(b_q) == 2 or \
                np.trace(np.fliplr(b_p)) - np.trace(np.fliplr(b_q)) == 2:
                    two_in_a_row_eval[p] += 1
                    continue

                # Row and Column checks
                for i in range(3):
                    if (sum_b_p_rows[i] - sum_b_q_rows[i] == 2) or \
                    (sum_b_p_cols[i] - sum_b_q_cols[i] == 2):
                        two_in_a_row_eval[p] += 1
                        break
            #check if there are any two in a row among the completed miniboards
            b_p = board.miniboxes[:, :, p]
            b_q = board.miniboxes[:, :, q]
            # Precompute sums
            sum_b_p_rows = np.sum(b_p, axis=1)
            sum_b_q_rows = np.sum(b_q, axis=1)
            sum_b_p_cols = np.sum(b_p, axis=0)
            sum_b_q_cols = np.sum(b_q, axis=0)
            # Diagonal checks
            if np.trace(b_p) - np.trace(b_q) == 2 or \
            np.trace(np.fliplr(b_p)) - np.trace(np.fliplr(b_q)) == 2:
                two_in_a_row_eval[p] += 1
            # Row and Column checks
            for i in range(3):
                if (sum_b_p_rows[i] - sum_b_q_rows[i] == 2) or \
                (sum_b_p_cols[i] - sum_b_q_cols[i] == 2):
                    two_in_a_row_eval[p] += 1

        
        score += two_in_a_row_eval[0] - two_in_a_row_eval[1]
        if board.n_moves % 2 == 1:
            score = -score      
        return score
   
    def hash_position(self, board: board_obj) -> bytes:
        int_markers = board.markers.astype(np.int8)
        uniboard = int_markers[:,:,self.root_player] - int_markers[:,:,((self.root_player + 1) % 2)]
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
        # legal_moves.sort(key=lambda move: self.score_move(board, move, tt_move, ply), reverse=True)
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
        if move[1] % 4 == 0 and np.sum(target_miniboard.diagonal()) == 2:
            score += 10
        elif move[1] % 2 == 0 and np.sum(np.fliplr(target_miniboard).diagonal()) == 2:
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

class crossfish_v12:
    '''This version adds PVS
    Test Result: W: 4115, L: 3486, D: 511, elo diff: 26.99 +/- 14.68, LOS: 100.00
    '''
    def __init__(self, name: str = 'Crossfish'):
        self.name = name
        self.thinking_time = 0.095
        self.root_best_move = None
        self.start_time = None
        self.score = 0
        self.root_player = 0
        self.transposition_table = dict()
        self.killer_moves = np.zeros((128, 2), dtype=np.int8) #depth, move
        self.history_table = np.zeros((2, 9, 9), dtype=np.uint16) #player (0 or 1), miniboard, move
        self.nodes = 0
        self.instance_b = board_obj()
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
        self.root_player = b_obj.n_moves % 2
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

            # print(f'depth: {depth-1} in {time.time() - self.start_time:.4f}s, score: {self.score}, nps: {self.nodes / (time.time() - self.start_time + 1e-5):.2f}')
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
                return (-1)**(board.n_moves) * (np.sum(board.miniboxes[:, :, 0]) - np.sum(board.miniboxes[:, :, 1]))
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
        if not pv_node:

            # if not pv_node:
            stand_pat = self.evaluate(board)

            #reverse futility pruning
            rfp_margin = 2
            if stand_pat - rfp_margin * depth >= beta:
                return beta

            # #futility pruning
            f_margin = 2
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
        for move_idx in range(len(legal_moves_and_scores)):
            move = legal_moves_and_scores[move_idx][0]
            move_score = legal_moves_and_scores[move_idx][1]
            if can_futility_prune and move_idx > 0 and move_score <= 0:
                continue
            self.make_move(board, move)
            if move_idx == 0:
                val = -self.search(board, depth-1, ply+1, -beta, -alpha, can_null=can_null)
            else:
                val = -self.search(board, depth-1, ply+1, -alpha-0.1, -alpha, can_null=can_null)
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
        score = (np.sum(board.miniboxes[:, :, 0]) - np.sum(board.miniboxes[:, :, 1])) * 2
        # '''TODO: FIND A WAY TO OPTIMIZE THIS'''
        miniboards = self.pull_mini_boards(board.markers)
        boxes_in_play = ~board.miniboxes[:,:,0] & ~board.miniboxes[:,:,1] & ~board.miniboxes[:,:,2]
        two_in_a_row_eval = np.zeros(2, dtype=np.int8)
        for p in range(2):
            q = (p + 1) % 2  # Opponent's index
            for b in miniboards[boxes_in_play.flatten()]:
                b_p = b[:, :, p]
                b_q = b[:, :, q]

                # Precompute sums
                sum_b_p_rows = np.sum(b_p, axis=1)
                sum_b_q_rows = np.sum(b_q, axis=1)
                sum_b_p_cols = np.sum(b_p, axis=0)
                sum_b_q_cols = np.sum(b_q, axis=0)

                # Diagonal checks
                if np.trace(b_p) - np.trace(b_q) == 2 or \
                np.trace(np.fliplr(b_p)) - np.trace(np.fliplr(b_q)) == 2:
                    two_in_a_row_eval[p] += 1
                    continue

                # Row and Column checks
                for i in range(3):
                    if (sum_b_p_rows[i] - sum_b_q_rows[i] == 2) or \
                    (sum_b_p_cols[i] - sum_b_q_cols[i] == 2):
                        two_in_a_row_eval[p] += 1
                        break
            #check if there are any two in a row among the completed miniboards
            b_p = board.miniboxes[:, :, p]
            b_q = board.miniboxes[:, :, q]
            # Precompute sums
            sum_b_p_rows = np.sum(b_p, axis=1)
            sum_b_q_rows = np.sum(b_q, axis=1)
            sum_b_p_cols = np.sum(b_p, axis=0)
            sum_b_q_cols = np.sum(b_q, axis=0)
            # Diagonal checks
            if np.trace(b_p) - np.trace(b_q) == 2 or \
            np.trace(np.fliplr(b_p)) - np.trace(np.fliplr(b_q)) == 2:
                two_in_a_row_eval[p] += 1
            # Row and Column checks
            for i in range(3):
                if (sum_b_p_rows[i] - sum_b_q_rows[i] == 2) or \
                (sum_b_p_cols[i] - sum_b_q_cols[i] == 2):
                    two_in_a_row_eval[p] += 1

        
        score += two_in_a_row_eval[0] - two_in_a_row_eval[1]
        if board.n_moves % 2 == 1:
            score = -score      
        return score
   
    def hash_position(self, board: board_obj) -> bytes:
        int_markers = board.markers.astype(np.int8)
        uniboard = int_markers[:,:,self.root_player] - int_markers[:,:,((self.root_player + 1) % 2)]
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
        # legal_moves.sort(key=lambda move: self.score_move(board, move, tt_move, ply), reverse=True)
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
        if move[1] % 4 == 0 and np.sum(target_miniboard.diagonal()) == 2:
            score += 10
        elif move[1] % 2 == 0 and np.sum(np.fliplr(target_miniboard).diagonal()) == 2:
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

    
class crossfish_v13:
    '''Eval tweaks. Increased value of a won minibox from 2 to 3. Fixed a typo with the eval in the case of a stale global board.
    Test Result: W: 1811, L: 1345, D: 204, elo diff: 48.50 +/- 22.99, LOS: 100.00
    '''
    def __init__(self, name: str = 'Crossfish'):
        self.name = name
        self.thinking_time = 0.095
        self.root_best_move = None
        self.start_time = None
        self.score = 0
        self.root_player = 0
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
        self.root_player = b_obj.n_moves % 2
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

            # print(f'depth: {depth-1} in {time.time() - self.start_time:.4f}s, score: {self.score}, nps: {self.nodes / (time.time() - self.start_time + 1e-5):.2f}')
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
        if not pv_node:

            # if not pv_node:
            stand_pat = self.evaluate(board)

            #reverse futility pruning
            rfp_margin = 2
            if stand_pat - rfp_margin * depth >= beta:
                return beta

            # #futility pruning
            f_margin = 2
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
        for move_idx in range(len(legal_moves_and_scores)):
            move = legal_moves_and_scores[move_idx][0]
            move_score = legal_moves_and_scores[move_idx][1]
            if can_futility_prune and move_idx > 0 and move_score <= 0:
                continue
            self.make_move(board, move)
            if move_idx == 0:
                val = -self.search(board, depth-1, ply+1, -beta, -alpha, can_null=can_null)
            else:
                val = -self.search(board, depth-1, ply+1, -alpha-0.1, -alpha, can_null=can_null)
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
        score = (np.sum(board.miniboxes[:, :, 0]) - np.sum(board.miniboxes[:, :, 1])) * 3
        # '''TODO: FIND A WAY TO OPTIMIZE THIS'''
        miniboards = self.pull_mini_boards(board.markers)
        boxes_in_play = ~board.miniboxes[:,:,0] & ~board.miniboxes[:,:,1] & ~board.miniboxes[:,:,2]
        two_in_a_row_eval = np.zeros(2, dtype=np.int8)
        for p in range(2):
            q = (p + 1) % 2  # Opponent's index
            for b in miniboards[boxes_in_play.flatten()]:
                b_p = b[:, :, p]
                b_q = b[:, :, q]

                # Precompute sums
                sum_b_p_rows = np.sum(b_p, axis=1)
                sum_b_q_rows = np.sum(b_q, axis=1)
                sum_b_p_cols = np.sum(b_p, axis=0)
                sum_b_q_cols = np.sum(b_q, axis=0)

                # Diagonal checks
                if np.trace(b_p) - np.trace(b_q) == 2 or \
                np.trace(np.fliplr(b_p)) - np.trace(np.fliplr(b_q)) == 2:
                    two_in_a_row_eval[p] += 1
                    continue

                # Row and Column checks
                for i in range(3):
                    if (sum_b_p_rows[i] - sum_b_q_rows[i] == 2) or \
                    (sum_b_p_cols[i] - sum_b_q_cols[i] == 2):
                        two_in_a_row_eval[p] += 1
                        break
            #check if there are any two in a row among the completed miniboards
            b_p = board.miniboxes[:, :, p]
            b_q = board.miniboxes[:, :, q]
            # Precompute sums
            sum_b_p_rows = np.sum(b_p, axis=1)
            sum_b_q_rows = np.sum(b_q, axis=1)
            sum_b_p_cols = np.sum(b_p, axis=0)
            sum_b_q_cols = np.sum(b_q, axis=0)
            # Diagonal checks
            if np.trace(b_p) - np.trace(b_q) == 2 or \
            np.trace(np.fliplr(b_p)) - np.trace(np.fliplr(b_q)) == 2:
                two_in_a_row_eval[p] += 1
            # Row and Column checks
            for i in range(3):
                if (sum_b_p_rows[i] - sum_b_q_rows[i] == 2) or \
                (sum_b_p_cols[i] - sum_b_q_cols[i] == 2):
                    two_in_a_row_eval[p] += 1

        
        score += two_in_a_row_eval[0] - two_in_a_row_eval[1]
        if board.n_moves % 2 == 1:
            score = -score      
        return score
   
    def hash_position(self, board: board_obj) -> bytes:
        int_markers = board.markers.astype(np.int8)
        uniboard = int_markers[:,:,self.root_player] - int_markers[:,:,((self.root_player + 1) % 2)]
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
        # legal_moves.sort(key=lambda move: self.score_move(board, move, tt_move, ply), reverse=True)
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
        if move[1] % 4 == 0 and np.sum(target_miniboard.diagonal()) == 2:
            score += 10
        elif move[1] % 2 == 0 and np.sum(np.fliplr(target_miniboard).diagonal()) == 2:
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

    
class crossfish_v14:
    '''
    tweak fp margin 2->1
    test result: W: 1892, L: 1430, D: 230, elo diff: 45.45 +/- 22.28, LOS: 100.00
    '''
    def __init__(self, name: str = 'Crossfish'):
        self.name = name
        self.thinking_time = 0.095
        self.root_best_move = None
        self.start_time = None
        self.score = 0
        self.root_player = 0
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
        self.root_player = b_obj.n_moves % 2
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

            # print(f'depth: {depth-1} in {time.time() - self.start_time:.4f}s, score: {self.score}, nps: {self.nodes / (time.time() - self.start_time + 1e-5):.2f}')
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
        if not pv_node:

            # if not pv_node:
            stand_pat = self.evaluate(board)

            #reverse futility pruning
            rfp_margin = 2
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
        for move_idx in range(len(legal_moves_and_scores)):
            move = legal_moves_and_scores[move_idx][0]
            move_score = legal_moves_and_scores[move_idx][1]
            if can_futility_prune and move_idx > 0 and move_score <= 0:
                continue
            self.make_move(board, move)
            if move_idx == 0:
                val = -self.search(board, depth-1, ply+1, -beta, -alpha, can_null=can_null)
            else:
                val = -self.search(board, depth-1, ply+1, -alpha-0.1, -alpha, can_null=can_null)
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
        score = (np.sum(board.miniboxes[:, :, 0]) - np.sum(board.miniboxes[:, :, 1])) * 3
        # '''TODO: FIND A WAY TO OPTIMIZE THIS'''
        miniboards = self.pull_mini_boards(board.markers)
        boxes_in_play = ~board.miniboxes[:,:,0] & ~board.miniboxes[:,:,1] & ~board.miniboxes[:,:,2]
        two_in_a_row_eval = np.zeros(2, dtype=np.int8)
        for p in range(2):
            q = (p + 1) % 2  # Opponent's index
            for b in miniboards[boxes_in_play.flatten()]:
                b_p = b[:, :, p]
                b_q = b[:, :, q]

                # Precompute sums
                sum_b_p_rows = np.sum(b_p, axis=1)
                sum_b_q_rows = np.sum(b_q, axis=1)
                sum_b_p_cols = np.sum(b_p, axis=0)
                sum_b_q_cols = np.sum(b_q, axis=0)

                # Diagonal checks
                if np.trace(b_p) - np.trace(b_q) == 2 or \
                np.trace(np.fliplr(b_p)) - np.trace(np.fliplr(b_q)) == 2:
                    two_in_a_row_eval[p] += 1
                    continue

                # Row and Column checks
                for i in range(3):
                    if (sum_b_p_rows[i] - sum_b_q_rows[i] == 2) or \
                    (sum_b_p_cols[i] - sum_b_q_cols[i] == 2):
                        two_in_a_row_eval[p] += 1
                        break
            #check if there are any two in a row among the completed miniboards
            b_p = board.miniboxes[:, :, p]
            b_q = board.miniboxes[:, :, q]
            # Precompute sums
            sum_b_p_rows = np.sum(b_p, axis=1)
            sum_b_q_rows = np.sum(b_q, axis=1)
            sum_b_p_cols = np.sum(b_p, axis=0)
            sum_b_q_cols = np.sum(b_q, axis=0)
            # Diagonal checks
            if np.trace(b_p) - np.trace(b_q) == 2 or \
            np.trace(np.fliplr(b_p)) - np.trace(np.fliplr(b_q)) == 2:
                two_in_a_row_eval[p] += 1
            # Row and Column checks
            for i in range(3):
                if (sum_b_p_rows[i] - sum_b_q_rows[i] == 2) or \
                (sum_b_p_cols[i] - sum_b_q_cols[i] == 2):
                    two_in_a_row_eval[p] += 1

        
        score += two_in_a_row_eval[0] - two_in_a_row_eval[1]
        if board.n_moves % 2 == 1:
            score = -score      
        return score
   
    def hash_position(self, board: board_obj) -> bytes:
        int_markers = board.markers.astype(np.int8)
        uniboard = int_markers[:,:,self.root_player] - int_markers[:,:,((self.root_player + 1) % 2)]
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
        # legal_moves.sort(key=lambda move: self.score_move(board, move, tt_move, ply), reverse=True)
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
        if move[1] % 4 == 0 and np.sum(target_miniboard.diagonal()) == 2:
            score += 10
        elif move[1] % 2 == 0 and np.sum(np.fliplr(target_miniboard).diagonal()) == 2:
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
    

class crossfish_v15:
    '''
    tweak rfp margin: 2->3
    test result: W: 3035, L: 2450, D: 371, elo diff: 34.82 +/- 17.31, LOS: 100.00
    '''
    def __init__(self, name: str = 'Crossfish'):
        self.name = name
        self.thinking_time = 0.095
        self.root_best_move = None
        self.start_time = None
        self.score = 0
        self.root_player = 0
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
        self.root_player = b_obj.n_moves % 2
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

            # print(f'depth: {depth-1} in {time.time() - self.start_time:.4f}s, score: {self.score}, nps: {self.nodes / (time.time() - self.start_time + 1e-5):.2f}')
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
        if not pv_node:

            # if not pv_node:
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
        for move_idx in range(len(legal_moves_and_scores)):
            move = legal_moves_and_scores[move_idx][0]
            move_score = legal_moves_and_scores[move_idx][1]
            if can_futility_prune and move_idx > 0 and move_score <= 0:
                continue
            self.make_move(board, move)
            if move_idx == 0:
                val = -self.search(board, depth-1, ply+1, -beta, -alpha, can_null=can_null)
            else:
                val = -self.search(board, depth-1, ply+1, -alpha-0.1, -alpha, can_null=can_null)
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
        score = (np.sum(board.miniboxes[:, :, 0]) - np.sum(board.miniboxes[:, :, 1])) * 3
        # '''TODO: FIND A WAY TO OPTIMIZE THIS'''
        miniboards = self.pull_mini_boards(board.markers)
        boxes_in_play = ~board.miniboxes[:,:,0] & ~board.miniboxes[:,:,1] & ~board.miniboxes[:,:,2]
        two_in_a_row_eval = np.zeros(2, dtype=np.int8)
        for p in range(2):
            q = (p + 1) % 2  # Opponent's index
            for b in miniboards[boxes_in_play.flatten()]:
                b_p = b[:, :, p]
                b_q = b[:, :, q]

                # Precompute sums
                sum_b_p_rows = np.sum(b_p, axis=1)
                sum_b_q_rows = np.sum(b_q, axis=1)
                sum_b_p_cols = np.sum(b_p, axis=0)
                sum_b_q_cols = np.sum(b_q, axis=0)

                # Diagonal checks
                if np.trace(b_p) - np.trace(b_q) == 2 or \
                np.trace(np.fliplr(b_p)) - np.trace(np.fliplr(b_q)) == 2:
                    two_in_a_row_eval[p] += 1
                    continue

                # Row and Column checks
                for i in range(3):
                    if (sum_b_p_rows[i] - sum_b_q_rows[i] == 2) or \
                    (sum_b_p_cols[i] - sum_b_q_cols[i] == 2):
                        two_in_a_row_eval[p] += 1
                        break
            #check if there are any two in a row among the completed miniboards
            b_p = board.miniboxes[:, :, p]
            b_q = board.miniboxes[:, :, q]
            # Precompute sums
            sum_b_p_rows = np.sum(b_p, axis=1)
            sum_b_q_rows = np.sum(b_q, axis=1)
            sum_b_p_cols = np.sum(b_p, axis=0)
            sum_b_q_cols = np.sum(b_q, axis=0)
            # Diagonal checks
            if np.trace(b_p) - np.trace(b_q) == 2 or \
            np.trace(np.fliplr(b_p)) - np.trace(np.fliplr(b_q)) == 2:
                two_in_a_row_eval[p] += 1
            # Row and Column checks
            for i in range(3):
                if (sum_b_p_rows[i] - sum_b_q_rows[i] == 2) or \
                (sum_b_p_cols[i] - sum_b_q_cols[i] == 2):
                    two_in_a_row_eval[p] += 1

        
        score += two_in_a_row_eval[0] - two_in_a_row_eval[1]
        if board.n_moves % 2 == 1:
            score = -score      
        return score
   
    def hash_position(self, board: board_obj) -> bytes:
        int_markers = board.markers.astype(np.int8)
        uniboard = int_markers[:,:,self.root_player] - int_markers[:,:,((self.root_player + 1) % 2)]
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
        # legal_moves.sort(key=lambda move: self.score_move(board, move, tt_move, ply), reverse=True)
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
        if move[1] % 4 == 0 and np.sum(target_miniboard.diagonal()) == 2:
            score += 10
        elif move[1] % 2 == 0 and np.sum(np.fliplr(target_miniboard).diagonal()) == 2:
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

    