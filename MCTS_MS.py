import random
import time
import logging


from board import *
from copy import deepcopy

from utility_functions import utilityValue, gameIsOver, AI_PLAYER, HUMAN_PLAYER, countSequence


class Node:
    def __init__(self, game_state, done, parent_node, action_index, ):
        self.parent = parent_node
        self.child_nodes = {}
        self.visits = 0
        self.node_value = 0  # Accumulated value from simulations
        self.game_state = game_state
        self.done = done
        self.action_index = action_index
        self.c = 1.4 # Exploration constant
        self.reward = 0

    def getUCTscore(self):
        """Calculate the UCT score for this node."""
        if self.visits == 0:
            return float('inf')  # If the node has not been visited, prioritize exploration

        parent_visits = max(1, self.parent.visits) if self.parent else 1  # Prevent division by zero

        # UCT formula: Exploitation + Exploration
        uct_score = (self.node_value / self.visits) + self.c * math.sqrt(math.log(parent_visits) / self.visits)
        return uct_score

    def create_child_nodes(self):
        valid_moves = getValidMoves(self.game_state)
        for action in valid_moves:
            result = makeMove(deepcopy(self.game_state), action, AI_PLAYER)
            child_board_state = result[0]

            done = gameIsOver(child_board_state)
            self.child_nodes[action] = Node(child_board_state, done, self, action)

    def explore(self, minimax_depth=3, min_rollouts=500, min_time=0.0):
        """Select the best child node based on UCT or expand a new node if possible."""
        start_time = time.time()
        rollouts = 0

        while rollouts < min_rollouts or (time.time() - start_time) < min_time:
            current = self

            # **Selection Phase**: Traverse down using UCT scores
            while current.child_nodes:
                child_scores = {a: c.getUCTscore() for a, c in current.child_nodes.items()}
                max_U = max(child_scores.values())
                best_actions = [a for a, score in child_scores.items() if score == max_U]

                action = random.choice(best_actions)
                current = current.child_nodes[action]

            # **Expansion Phase**: Expand if not terminal
            if not current.done:
                current.create_child_nodes()
                if current.child_nodes:
                    current = random.choice(list(current.child_nodes.values()))

                    # **Simulation Phase**: Rollout to estimate value
            current.reward = current.rollout(minimax_depth)

            # **Backpropagation Phase**: Update parent nodes
            parent = current
            while parent:
                parent.visits += 1
                parent.node_value += current.reward
                parent = parent.parent

            rollouts += 1

        return self

    def rollout(self, minimax_depth):
        new_board = deepcopy(self.game_state)
        current_player = AI_PLAYER

        while not gameIsOver(new_board):
            valid_moves = getValidMoves(new_board)

            if not valid_moves:
                return 0.5  # Draw (no valid moves left)

            # Use a random move
            action = random.choice(valid_moves)

            # Apply the chosen move
            new_board, _, _ = makeMove(new_board, action, current_player)

            # Check if the game is over
            if gameIsOver(new_board):
                return EndValue(new_board, AI_PLAYER)  # Score from AI's perspective

            # Switch players
            current_player = HUMAN_PLAYER if current_player == AI_PLAYER else AI_PLAYER



        return EndValue(new_board, AI_PLAYER)

    def next(self):
        """Return the best child node based on the highest visit count."""
        if self.done:
            raise ValueError("Game has ended. No next move available.")

        if not self.child_nodes:
            raise ValueError("No children found. Ensure exploration has been performed.")

        # Find the most visited child node
        best_child = max(self.child_nodes.values(), key=lambda x: x.visits)
        print(self.child_nodes)
        print(self.child_nodes)
        print(self.child_nodes)
        print(self.child_nodes)
        print(self.child_nodes)

        print(f"Selected Best Action: {best_child.action_index} with {best_child.visits} visits.")
        printBoard(best_child.game_state)  # Assuming `printBoard` prints the board state
        print("=" * 30)  # Separator for readability
        print(self.child_nodes)
        print(self.child_nodes)


        return best_child, best_child.action_index
