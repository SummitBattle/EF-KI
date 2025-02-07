import math
from board import *
class Node:
    def __init__(self, game_state, done, parent_node, child_board_state, action_index, reward):

        self.parent = parent_node
        self.child_nodes = None
        self.visits = 0
        self.node_value = 0
        self.game_state = game_state
        self.done = done
        self.action_index = action_index
        self.c = 1.5
        self.child_board_state = child_board_state
        self.reward = reward



    def getUCTscore(self):

        if self.visits == 0:
            return float('inf')

        top_node = self

        if top_node.parent:
            self.parent = top_node

        # UCT formula
        return (self.reward / self.visits) + self.c * math.sqrt(math.log(top_node.visits / self.visits))

    def create_child(self):

        valid_moves = []
        new_boards = []

        # Create child board with valid move
        for i in range(getValidMoves(self.game_state)):
            valid_moves.append(i)
            child_boards = deepcopy(self.game_state)
            new_boards.append(child_boards)

        child_nodes = {}

        # Do move on child boards and create child nodes
        for action, board in zip(valid_moves,new_boards):
            child_board_state = makeMove(board, action)
            done = gameIsOver(board)

            if not done:
                reward = 0

            elif findFours(child_board_state):
                reward = 1
            else:
                reward = -1

            child_nodes[action] = Node(self.game_state, done, self, child_board_state, action, reward)

        self.child_nodes = child_nodes







