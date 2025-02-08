import pygame
import random


from board import *
from copy import deepcopy
from minimaxAlphaBeta import MiniMaxAlphaBeta

AI_PLAYER = 'o'
OPPONENT = 'x'
class Node:
    def __init__(self, game_state, done, parent_node, action_index, reward):
        self.parent = parent_node
        self.child_nodes = {}
        self.visits = 0
        self.node_value = 0  # Accumulated value from simulations
        self.game_state = game_state
        self.done = done
        self.action_index = action_index
        self.c = 0.5
        self.reward = reward

    def getUCTscore(self):
        """Calculate the UCT score for this node."""
        if self.visits == 0:
            return float('inf')  # Encourage unexplored nodes

        parent_visits = max(1, self.parent.visits) if self.parent else 1  # Prevent division by zero

        # UCT formula: Exploitation + Exploration
        return (self.node_value / self.visits) + self.c * math.sqrt(math.log(parent_visits) / self.visits)

    def create_child_nodes(self):
        valid_moves = getValidMoves(self.game_state)
        for action in valid_moves:
            result = makeMove(deepcopy(self.game_state), action, AI_PLAYER)
            child_board_state = result[0]

            done = gameIsOver(child_board_state)
            reward = utilityValue(child_board_state, AI_PLAYER)
            self.child_nodes[action] = Node(child_board_state, done, self, action, reward)

    def explore(self, minimax_depth=3):
        """Select the best child node based on UCT or expand a new node if possible."""
        current = self

        while current.child_nodes:
            child_scores = {a: c.getUCTscore() for a, c in current.child_nodes.items()}
            max_U = max(child_scores.values())
            best_actions = [a for a, score in child_scores.items() if score == max_U]

            action = random.choice(best_actions)
            current = current.child_nodes[action]

        if current.visits == 0:
            if minimax_depth > 0:
                _, current.reward = MiniMaxAlphaBeta(current.game_state, minimax_depth, AI_PLAYER)
            else:
                current.reward = current.rollout()

        current.create_child_nodes()
        if current.child_nodes:
            current = random.choice(list(current.child_nodes.values()))

        current.visits += 1
        parent = current
        while parent.parent:
            parent = parent.parent
            parent.visits += 1
            parent.node_value += current.reward

    def rollout(self):
        """Simulate a random game until the end and return the result."""
        if self.done:
            return self.reward

        new_board = deepcopy(self.game_state)

        while True:
            valid_moves = getValidMoves(new_board)
            if not valid_moves:
                return 0  # Draw
            action = random.choice(valid_moves)
            new_board, _, _ = makeMove(new_board, action, AI_PLAYER)

            if gameIsOver(new_board):
                return utilityValue(new_board, AI_PLAYER)

    def next(self):
        """Return the best child based on visit count."""
        if self.done:
            raise ValueError("Game has ended")

        if not self.child_nodes:
            raise ValueError("No children found and the game hasn't ended")

        max_visits = max(node.visits for node in self.child_nodes.values())
        best_children = [c for c in self.child_nodes.values() if c.visits == max_visits]

        best_child = random.choice(best_children)
        print(best_child, best_child.action_index)
        return best_child, best_child.action_index
