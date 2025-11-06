import random
import numpy as np
from typing import List


class robot_on_gridworld:
    """
    Use coordinates to represent states on a gridworld.
    Coordinates are (row, col), 0-indexed, top-left origin.

    Start state = 1, Goal = 25, Water = {7, 19, 23} by default. Here we also give obstacles coordinates.
    Actions: 'U','D','L','R'
    Transition rules: 0.80 intended, 0.05 right-turn, 0.05 left-turn, 0.10 no-move.
    Reward: +10 on entering goal; -10 on entering any water coordinate; 0 otherwise.
    Episode ends at the goal.
    """

    def __init__(self,
                 water:List[int],
                 obstacles:List[int],
                 start_state,
                 rows=5, cols=5,   
                 goal_state=25,
                 gamma=0.9,):
        self.rows = rows
        self.cols = cols
        self.n_states = rows * cols

        self.start_state = start_state
        self.goal_state = goal_state
        self.water = water
        self.obstacles = obstacles
        self.gamma = gamma

        self.actions = ['U', 'D', 'L', 'R']

        self.s = self.start_state

    # ---------- helpers ----------
    def _state_to_coord(self, s):
        """state index -> coordinate (row, col)."""
        s0 = s - 1
        return s0 // self.cols, s0 % self.cols

    def _coord_to_state(self, row, col):
        """coordinate (row, col) -> state index."""
        return row * self.cols + col + 1

    def _move(self, row, col, a):
        """one-step cardinal proposal from a coordinate."""
        if a == 'U': row -= 1
        elif a == 'D': row += 1
        elif a == 'L': col -= 1
        elif a == 'R': col += 1
        return row, col

    # ---------- action mapping ----------
    def _turn_right(self, a): return {'U':'R','R':'D','D':'L','L':'U'}[a]
    def _turn_left (self, a): return {'U':'L','L':'D','D':'R','R':'U'}[a]

    def _forbidden_coordinate(self, row, col):
        """
        True if the agent is outside the grid bounds OR hit an obstacle.
        """
        in_bounds = (0 <= row < self.rows) and (0 <= col < self.cols)
        if not in_bounds:
            return True
        s = self._coord_to_state(row, col)
        return s in self.obstacles

    def reset(self):
        self.s = self.start_state
        return self.s

    def transition(self, a):
        """
        Transition rules under action a and state s.
        """

        # stochastic direction
        u = random.random()
        if u < 0.80:      a_eff = a
        elif u < 0.85:    a_eff = self._turn_right(a)
        elif u < 0.90:    a_eff = self._turn_left(a)
        else:             a_eff = None  

        current_row, current_col = self._state_to_coord(self.s)
        next_row, next_col = current_row, current_col

        if a_eff is not None:
            temp_row, temp_col = self._move(current_row, current_col, a_eff)
            if not self._forbidden_coordinate(temp_row, temp_col):   
                next_row, next_col = temp_row, temp_col

        s_next = self._coord_to_state(next_row, next_col)

        # reward by next coordinate/state
        if s_next == self.goal_state:
            reward = +10
        elif s_next in self.water:
            reward = -10
        else:
            reward = 0

        self.s = s_next
        done = (self.s == self.goal_state)
        return s_next, reward, done