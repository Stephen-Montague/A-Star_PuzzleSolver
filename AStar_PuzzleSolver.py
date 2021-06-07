
"""
Stephen Montague
22 Jan 2020
Spring 2020 Term 1
AI - 1: Machine Problem 1 - A* for Sliding Puzzle

Created on Thu Jan  3 14:24:06 2019
Updated by student on 22 Jan 2020

@author: Piotr Szczurek, with 4 functions implemented by student "Stephen" as marked.

This program implements A* for solving a sliding 8 tile puzzle.

"""
import numpy as np
import queue


def _compute_sum_manhattan_distance(abs_distances):  # Implemented by Stephen
    """Takes a 2d array of absolute value distances to a Solved Puzzle state and returns a sum Manhattan distance """
    for i, array in enumerate(abs_distances):
        for j, element in enumerate(array):
            if element < 3:
                continue
            elif element == 3:
                abs_distances[i, j] = 1
            else:
                abs_distances[i, j] = (element // 3) + (element % 3)
    sum_manhattan_distance = np.sum(abs_distances)
    return sum_manhattan_distance


class PuzzleState:
    SOLVED_PUZZLE = np.arange(9).reshape((3, 3))

    def __init__(self, config, g, predState, predAction):  # Stephen: Added 4th parameter 'predAction' (predecessor action)
        self.puzzle = config
        self.zeroloc = np.argwhere(self.puzzle == 0)[0]  # Define blank puzzle
        self.gcost = g
        self.hcost = self._compute_heuristic_cost()
        self.fcost = self.gcost + self.hcost
        self.pred = predState
        self.action_from_pred = predAction

    def __hash__(self):
        return tuple(self.puzzle.ravel()).__hash__()

    def _compute_heuristic_cost(self):  # Implemented by Stephen
        """ Returns the sum Manhattan distance for all tiles to reach a Solved Puzzle state"""
        if self.is_goal():
            return 0
        else:
            absolute_distances = np.abs(np.subtract(PuzzleState.SOLVED_PUZZLE, self.puzzle))
            absolute_distances[self.zeroloc[0]] = 0  # Trim abs_dist to ignore '0' tile
            sum_manhattan_distance = _compute_sum_manhattan_distance(absolute_distances)
            return sum_manhattan_distance

    def _trim_abs_dist(self):
        target_value_of_zeroloc = PuzzleState.SOLVED_PUZZLE[self.zeroloc]
        print("Zero should be", target_value_of_zeroloc)

    def is_goal(self):
        return np.array_equal(PuzzleState.SOLVED_PUZZLE, self.puzzle)

    def __eq__(self, other):
        return np.array_equal(self.puzzle, other.puzzle)

    def __lt__(self, other):
        return self.fcost < other.fcost

    def __str__(self):
        return np.str(self.puzzle)

    move = 0

    def show_path(self):
        if self.pred is not None:
            self.pred.show_path()

        if PuzzleState.move == 0:
            print('START')
        else:
            print('Move', PuzzleState.move, 'ACTION:', self.action_from_pred)
        PuzzleState.move = PuzzleState.move + 1
        print(self)

    def can_move(self, direction):  # Implemented by Stephen
        if direction == 'up':
            return True if self.zeroloc[0] > 0 else False
        elif direction == 'down':
            return True if self.zeroloc[0] < 2 else False
        elif direction == 'left':
            return True if self.zeroloc[1] > 0 else False
        elif direction == 'right':
            return True if self.zeroloc[1] < 2 else False

    def gen_next_state(self, direction):  # Implemented by Stephen
        """ Generates a new puzzle state, given a valid 'direction' parameter """
        new_puzzle = np.copy(self.puzzle)
        zero_index_row = self.zeroloc[0]  # Row of '0' in 2d array
        zero_index_col = self.zeroloc[1]  # Column of '0'
        direction_values = {'up': -1, 'down': 1, 'left': -1, 'right': 1}

        if direction == 'up' or direction == 'down':  # Swap '0' with index above or below in 2d array
            new_puzzle[zero_index_row + direction_values[direction], zero_index_col] = 0
            new_puzzle[zero_index_row, zero_index_col] = \
                self.puzzle[zero_index_row + direction_values[direction], zero_index_col]
        else:  # Swap '0' with index left or right
            new_puzzle[zero_index_row, zero_index_col + direction_values[direction]] = 0
            new_puzzle[zero_index_row, zero_index_col] = \
                self.puzzle[zero_index_row, zero_index_col + direction_values[direction]]

        return PuzzleState(new_puzzle, self.gcost + 1, self, direction)


print('Artificial Intelligence')
print('MP1: A* for Sliding Puzzle')
print('SEMESTER: Spring 2020 Term 1')
print('NAME: Stephen Montague')
print()

# load random start state onto frontier priority queue
frontier = queue.PriorityQueue()
a = np.loadtxt('mp1input.txt', dtype=np.int32)
start_state = PuzzleState(a, 0, None, None)

frontier.put(start_state)

closed_set = set()

num_states = 0
while not frontier.empty():
    #  choose state at front of priority queue
    next_state = frontier.get()
    num_states = num_states + 1

    #  if goal then quit and return path
    if next_state.is_goal():
        next_state.show_path()
        break

    # Add state chosen for expansion to closed_set
    closed_set.add(next_state)

    # Expand state (up to 4 moves possible): block implemented by Stephen
    possible_moves = ['up', 'down', 'left', 'right']
    for move in possible_moves:
        if next_state.can_move(move):
            neighbor = next_state.gen_next_state(move)
            if neighbor in closed_set:
                continue
            if neighbor not in frontier.queue:
                frontier.put(neighbor)
            # If it's already in the frontier, it's guaranteed to have lower cost, so no need to update

print('\nNumber of states visited =', num_states)
