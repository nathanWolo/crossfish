import numpy as np
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

