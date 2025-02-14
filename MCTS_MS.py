import random
import time
import math
from copy import deepcopy

from board import *
from minimaxAlphaBeta import MiniMaxAlphaBeta
from utility_functions import utilityValue, gameIsOver, AI_PLAYER, HUMAN_PLAYER, countSequence, EndValue


class Node:
    def __init__(self, game_state, done, parent_node, action_index, starting_player):
        self.parent = parent_node
        self.child_nodes = {}  # Dictionary mapping actions to child nodes
        self.visits = 0
        self.node_value = 0  # Accumulated reward from simulations
        self.game_state = game_state
        self.done = done  # Indicates if the game ended at this state
        self.action_index = action_index
        self.c = 1.4  # Exploration constant for UCT
        self.reward = 0  # Reward from the last simulation
        self.starting_player = starting_player
        self._cached_log_parent_visits = None  # For caching parent's log(visit_count)

    def getUCTscore(self):
        """Calculate the UCT score for this node."""
        # If never visited, prioritize exploration.
        if self.visits == 0:
            return float('inf')

        # Get parent's visit count (or use 1 if root)
        parent_visits = self.parent.visits if self.parent else 1

        # Cache the parent's logarithm value to avoid recalculating.
        if self.parent:
            if self.parent._cached_log_parent_visits is None or self.parent._cached_log_parent_visits != math.log(
                    parent_visits):
                self.parent._cached_log_parent_visits = math.log(parent_visits)
            log_parent_visits = self.parent._cached_log_parent_visits
        else:
            log_parent_visits = 0  # log(1)=0

        exploitation = self.node_value / self.visits
        exploration = self.c * math.sqrt(log_parent_visits / self.visits)
        return exploitation + exploration

    def create_child_nodes(self):
        """Generate all valid child nodes from the current state."""
        valid_moves = getValidMoves(self.game_state)

        # Precompute counts for each player to determine whose turn it is.
        ai_moves = sum(row.count(AI_PLAYER) for row in self.game_state)
        human_moves = sum(row.count(HUMAN_PLAYER) for row in self.game_state)

        # Determine current player based on who started and the move counts.
        if self.starting_player == AI_PLAYER:
            current_player = AI_PLAYER if ai_moves == human_moves else HUMAN_PLAYER
        else:
            current_player = HUMAN_PLAYER if human_moves == ai_moves else AI_PLAYER

        for action in valid_moves:
            new_board = deepcopy(self.game_state)
            new_board, _, _ = makeMove(new_board, action, current_player)
            done = gameIsOver(new_board)  # Cache game-over check
            self.child_nodes[action] = Node(new_board, done, self, action, self.starting_player)

    def explore(self, minimax_depth=3, min_rollouts=100000, min_time=0.0, max_time=8.0):
        """
        Perform exploration (selection, expansion, simulation, backpropagation)
        until a minimum number of rollouts or minimum time is reached, but never beyond max_time.
        """
        start_time = time.time()
        rollouts = 0

        while ((rollouts < min_rollouts or (time.time() - start_time) < min_time)
               and (time.time() - start_time) < max_time):

            current = self
            # **Selection Phase:** Traverse the tree by selecting the best UCT child.
            while current.child_nodes:
                child_scores = {action: child.getUCTscore() for action, child in current.child_nodes.items()}
                max_score = max(child_scores.values())
                # In case of ties, randomly choose among the best.
                best_actions = [action for action, score in child_scores.items() if score == max_score]
                action = random.choice(best_actions)
                current = current.child_nodes[action]

            # **Expansion Phase:** If the node is not terminal, create its children.
            if not current.done:
                current.create_child_nodes()
                if current.child_nodes:
                    # Randomly select one child for simulation.
                    current = random.choice(list(current.child_nodes.values()))

            # **Simulation Phase:** Rollout the game from the current node.
            current.reward = current.rollout(minimax_depth)

            # **Backpropagation Phase:** Update the statistics for the nodes along the path.
            node_to_update = current
            while node_to_update:
                node_to_update.visits += 1
                node_to_update.node_value += current.reward
                node_to_update = node_to_update.parent

            rollouts += 1

        elapsed_time = time.time() - start_time
        print(f"Total exploration time: {elapsed_time:.2f} seconds")
        print(f"Rollouts performed: {rollouts}")
        return self

    def rollout(self, minimax_depth: int = 2) -> float:
        """
        Simulate a game rollout. Use a Minimax-guided first move, then follow with random moves.
        """
        # Determine the current player based on board state.
        ai_moves = sum(row.count(AI_PLAYER) for row in self.game_state)
        human_moves = sum(row.count(HUMAN_PLAYER) for row in self.game_state)
        current_player = AI_PLAYER if (ai_moves == human_moves and self.starting_player == AI_PLAYER) else HUMAN_PLAYER

        new_board = deepcopy(self.game_state)

        # If the game is already over at this node, return the outcome.
        if gameIsOver(new_board):
            return EndValue(new_board, AI_PLAYER)

        # **Minimax-Guided Move:** Attempt a more informed first move.
        best_move, _ = MiniMaxAlphaBeta(new_board, minimax_depth, current_player)
        if best_move is not None:
            new_board, _, _ = makeMove(new_board, best_move, current_player)
            if gameIsOver(new_board):
                return EndValue(new_board, AI_PLAYER)

        # Switch player for subsequent moves.
        current_player = HUMAN_PLAYER if current_player == AI_PLAYER else AI_PLAYER

        # **Random Rollout:** Continue playing random moves until the game ends.
        while not gameIsOver(new_board):
            valid_moves = getValidMoves(new_board)
            if not valid_moves:
                return 0.5  # Return a draw value if no moves are available.
            action = random.choice(valid_moves)
            new_board, _, _ = makeMove(new_board, action, current_player)
            current_player = HUMAN_PLAYER if current_player == AI_PLAYER else AI_PLAYER

        return EndValue(new_board, AI_PLAYER)

    def next(self):
        """
        Select and return the best child node based on visit counts.
        Raises an error if no valid move exists (e.g., game over or exploration not performed).
        """
        if self.done:
            raise ValueError("Game has ended. No next move available.")
        if not self.child_nodes:
            raise ValueError("No children found. Ensure exploration has been performed.")

        best_child = max(self.child_nodes.values(), key=lambda child: child.visits)
        print(f"Selected Best Action: {best_child.action_index} with {best_child.visits} visits.")
        print("=" * 30)
        return best_child, best_child.action_index

    def movePlayer(self, playerMove):
        """
        Update the current tree with the player's move.
        Returns the corresponding child node (creating it if necessary).
        """
        if playerMove in self.child_nodes:
            new_root = self.child_nodes[playerMove]
        else:
            new_board, _, _ = makeMove(deepcopy(self.game_state), playerMove, HUMAN_PLAYER)
            done = gameIsOver(new_board)
            new_root = Node(new_board, done, self, playerMove, self.starting_player)
            self.child_nodes[playerMove] = new_root
        return new_root
