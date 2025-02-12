import random
import time
import logging

from board import *
from copy import deepcopy

from minimaxAlphaBeta import MiniMaxAlphaBeta
from utility_functions import utilityValue, gameIsOver, AI_PLAYER, HUMAN_PLAYER, countSequence


class Node:
    def __init__(self, game_state, done, parent_node, action_index, starting_player):
        self.parent = parent_node
        self.child_nodes = {}
        self.visits = 0
        self.node_value = 0  # Accumulated value from simulations
        self.game_state = game_state
        self.done = done
        self.action_index = action_index
        self.c = 1.4  # Exploration constant
        self.reward = 0
        self.starting_player = starting_player

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

        # Count moves to determine whose turn it is
        ai_moves = sum(row.count(AI_PLAYER) for row in self.game_state)
        human_moves = sum(row.count(HUMAN_PLAYER) for row in self.game_state)

        # Determine the next player based on move count and who started first
        if self.starting_player == AI_PLAYER:
            current_player = AI_PLAYER if ai_moves == human_moves else HUMAN_PLAYER
        else:
            current_player = HUMAN_PLAYER if human_moves == ai_moves else AI_PLAYER

        for action in valid_moves:
            new_board = deepcopy(self.game_state)
            result = makeMove(new_board, action, current_player)
            child_board_state = result[0]

            done = gameIsOver(child_board_state)

            self.child_nodes[action] = Node(child_board_state, done, self, action, self.starting_player)

    def explore(self, minimax_depth=3, min_rollouts=100000, min_time=0.0, max_time=8.0):
        """Select the best child node based on UCT or expand a new node if possible.

        Stops when min_rollouts are done OR min_time is reached, but enforces a hard stop at max_time.
        """
        start_time = time.time()
        rollouts = 0

        while (rollouts < min_rollouts or (time.time() - start_time) < min_time) and (
                time.time() - start_time) < max_time:
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
        end_time = time.time()  # Capture the end time
        elapsed_time = end_time - start_time  # Calculate elapsed time
        print(f"Total exploration time: {elapsed_time:.2f} seconds")  # Print elapsed time
        print(rollouts)

        return self

    def rollout(self, minimax_depth: int = 2) -> float:
        """Simulate a game using Minimax during rollouts."""
        ai_moves = sum(row.count(AI_PLAYER) for row in self.game_state)
        human_moves = sum(row.count(HUMAN_PLAYER) for row in self.game_state)
        current_player = AI_PLAYER if (ai_moves == human_moves and self.starting_player == AI_PLAYER) else HUMAN_PLAYER
        new_board = deepcopy(self.game_state)

        # If the game is over at this node, return the result
        if gameIsOver(new_board):
            return EndValue(new_board, AI_PLAYER)

        # Use Minimax for the first move in the rollout
        best_move, _ = MiniMaxAlphaBeta(new_board, minimax_depth, current_player)
        if best_move is not None:
            new_board, _, _ = makeMove(new_board, best_move, current_player)
            if gameIsOver(new_board):
                return EndValue(new_board, AI_PLAYER)

        # Continue with random moves if Minimax doesn't end the game
        current_player = HUMAN_PLAYER if current_player == AI_PLAYER else AI_PLAYER
        while not gameIsOver(new_board):
            valid_moves = getValidMoves(new_board)
            if not valid_moves:
                return 0.5  # Draw if no moves are available
            action = random.choice(valid_moves)
            new_board, _, _ = makeMove(new_board, action, current_player)
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

        print(f"Selected Best Action: {best_child.action_index} with {best_child.visits} visits.")
        print("=" * 30)  # Separator for readability

        return best_child, best_child.action_index

    def movePlayer(self, playerMove):
        """Update the node tree with the player's move and return the new root."""
        # Use `setdefault` to either get the existing child node or create a new one
        new_root = self.child_nodes.setdefault(playerMove,
                                               Node(*makeMove(self.game_state, playerMove, HUMAN_PLAYER)[:1],
                                                    gameIsOver(self.game_state),
                                                    self,
                                                    playerMove,
                                                    self.starting_player))
        return new_root
