
import numpy as np


class board_obj():
    def __init__(self):
        # the full board: two channels, one per player
        self.markers = np.zeros((9,9,2)).astype(bool)
        # an "open" location is calculated by ORing
        
        # the overall miniboard status
        self.miniboxes = np.zeros((3,3,3)).astype(bool)
        # channels: p1, p2, stale
        
        # board history
        self.hist = np.zeros((81,2),dtype=np.uint8)
        self.n_moves = 0
    def build_from_dict_gamestate(self, gamestate: dict):
        self.markers = gamestate['markers']
        self.miniboxes = gamestate['miniboxes']
        self.hist = gamestate['history']
        self.n_moves = gamestate['n_moves']