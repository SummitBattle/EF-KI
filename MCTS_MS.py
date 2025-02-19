import random
import time
import math
from copy import deepcopy
import concurrent.futures
import numpy as np
from tensorflow.keras.models import load_model

from utility_functions import gameIsOver, AI_PLAYER, HUMAN_PLAYER, EndValue

import logging
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # '3' suppresses all messages except errors.

# Set up logging configuration to log to a file.
logging.basicConfig(
    filename='rollouts.log',  # File where log will be saved.
    format='%(asctime)s - %(message)s',  # Format for each log entry.
    level=logging.INFO  # Change to DEBUG for more detailed logs.
)

# Global variable for the model (each process will load it once).
model = None


def get_model():
    """Lazily load and return the trained model."""
    global model
    if model is None:
        model = load_model('connect4_model.h5')
    return model


def board_to_input(board_state):
    if hasattr(board_state, "get_board_matrix"):
        board_matrix = board_state.get_board_matrix()

    elif hasattr(board_state, "board1") and hasattr(board_state, "board2"):

        board_matrix = np.zeros((6, 7), dtype=int)
        for r in range(6):
            for c in range(7):
                idx = r * 7 + c  # Adjust the mapping if needed.
                if (board_state.board1 >> idx) & 1:
                    board_matrix[r, c] = 1
                elif (board_state.board2 >> idx) & 1:
                    board_matrix[r, c] = 2

    # Continue processing to prepare the model input.
    player1 = (board_matrix == 1).astype(np.float32)
    player2 = (board_matrix == 2).astype(np.float32)
    input_data = np.stack([player1, player2], axis=-1)

    return input_data


def model_evaluation(game_state, ai_player, starting_player):
    """
    Evaluate the given board state using the trained model.

    The model outputs probabilities in the order:
      [2nd player win chance, tie, 1st player win chance]

    Expected value is computed as:
      - If the AI is the starting player: value = (p1 - p2)
      - Otherwise: value = (p2 - p1)
    """
    mdl = get_model()
    board = board_to_input(game_state)

    board_flat = board.reshape(-1, 42)  # Now shape (1,42) if you had one board

    probabilities = mdl.predict(board_flat)[0]  # Expected shape: (3,)

    p2, tie, p1 = probabilities

    if not starting_player:
        value = p1 - p2

    else:
        value = p2 - p1

    return value


def copy_board(bitboard):
    """Fast board copy: use a built-in copy method if available."""

    if hasattr(bitboard, 'copy'):

        return bitboard.copy()

    return deepcopy(bitboard)


def count_moves(bitboard):
    """
    Count the number of moves for each player using the bitboard representation.
    Assumes:
      - bitboard.board1 holds the first player's moves.
      - bitboard.board2 holds the second player's moves.
    """

    human_moves = bin(bitboard.board1).count("1")
    ai_moves = bin(bitboard.board2).count("1")

    return ai_moves, human_moves


def get_move_column(move):
    """
    Extract the column index from a move.
    If move is a tuple, its second element is the column.
    If move is an int, it is the column.
    """

    if isinstance(move, int):

        return move
    elif isinstance(move, tuple):

        return move[1]

    return move


def biased_random_move(valid_moves, center_col=3, bias_strength=0.09):
    """
    Select a move with a slight center bias.
    For Connect 4 (7 columns), the center column is 3.
    Moves closer to the center receive a higher weight.
    """

    weights = []
    max_distance = 3  # Maximum distance from the center (columns: 0-6).
    for move in valid_moves:
        col = get_move_column(move)
        weight = 1 + bias_strength * (max_distance - abs(col - center_col))
        weights.append(weight)

    chosen_move = random.choices(valid_moves, weights=weights, k=1)[0]

    return chosen_move


def rollout_simulation(game_state, starting_player):
    """
    Perform a simulation from a given board state.

    Instead of running a minimax search or random rollout,
    we simply evaluate the board using the trained model.
    """

    board_state = copy_board(game_state)
    if gameIsOver(board_state):

        return EndValue(board_state, AI_PLAYER)
    result = model_evaluation(board_state, AI_PLAYER, starting_player)

    return result


class Node:
    __slots__ = ('parent', 'child_nodes', 'visits', 'node_value', 'game_state', 'done',
                 'action_index', 'c', 'reward', 'starting_player')

    def __init__(self, game_state, done, parent_node, action_index, starting_player):

        self.parent = parent_node
        self.child_nodes = {}  # Dictionary mapping action -> Node.
        self.visits = 0
        self.node_value = 0.0
        self.game_state = game_state  # The board state.
        self.done = done
        self.action_index = action_index  # The move that led to this node.
        self.c = 1.6  # Exploration constant.
        self.reward = 0.0
        self.starting_player = starting_player

    def getUCTscore(self, center_col=3, bias_strength=0.09):
        """
        Compute the UCT (Upper Confidence Bound for Trees) score for the node.
        A bonus center bias is added based on the column of the move.
        """

        if self.visits == 0:

            return float('inf')
        parent_visits = self.parent.visits if self.parent else 1

        # UCT formula.
        uct_score = (self.node_value / self.visits) + self.c * math.sqrt(math.log(parent_visits) / self.visits)
        if isinstance(self.action_index, tuple):
            move_col = self.action_index[1]  # Assume move is (row, col).
        else:
            move_col = self.action_index
        max_distance = 3  # For a 7-column board.
        center_bias = bias_strength * (max_distance - abs(move_col - center_col))
        final_score = uct_score + center_bias

        return final_score

    def create_child_nodes(self):
        """
        Expand the current node by creating child nodes for each valid move.
        """

        valid_moves = self.game_state.get_valid_moves()

        for action in valid_moves:
            new_board = copy_board(self.game_state)
            new_board.play_move(action)
            done = gameIsOver(new_board)

            self.child_nodes[action] = Node(new_board, done, self, action, self.starting_player)


    def explore(self, min_rollouts=200000, min_time=5.0, max_time=5.0, batch_size=32):
        """
        Explore the tree using parallel rollouts.

        Each rollout now simply evaluates a board (after expansion) using the model.
        """

        start_time = time.perf_counter()
        rollouts = 0

        rand_choice = random.choice
        get_time = time.perf_counter
        batch = []  # List of tuples: (future, node).

        with concurrent.futures.ProcessPoolExecutor() as executor:
            while True:
                elapsed = get_time() - start_time
                if (rollouts >= min_rollouts and elapsed >= min_time) or (elapsed >= max_time):
                    print(f"Node.explore: Stopping exploration after {rollouts} rollouts and {elapsed:.2f} seconds")
                    break

                current = self
                # --- Selection Phase ---
                while current.child_nodes:
                    best_score = -float('inf')
                    best_children = []
                    for child in current.child_nodes.values():
                        score = child.getUCTscore()
                        if score > best_score:
                            best_score = score
                            best_children = [child]
                        elif score == best_score:
                            best_children.append(child)
                    current = rand_choice(best_children)


                # --- Expansion Phase ---
                if not current.done:
                    current.create_child_nodes()
                    if current.child_nodes:
                        current = rand_choice(list(current.child_nodes.values()))


                # Schedule the rollout simulation in parallel.
                future = executor.submit(rollout_simulation, current.game_state, self.starting_player)
                batch.append((future, current))
                rollouts += 1


                # Process batch once full.
                if len(batch) >= batch_size:
                    for future, node in batch:
                        try:
                            reward = future.result()

                        except Exception as e:

                            reward = 0.0  # Fallback if simulation fails.
                        # --- Backpropagation Phase ---
                        node_to_update = node
                        while node_to_update:
                            node_to_update.visits += 1
                            node_to_update.node_value += reward
                            node_to_update = node_to_update.parent
                    batch.clear()

            # Process any remaining futures.
            for future, node in batch:
                try:
                    reward = future.result()
                except Exception as e:
                    reward = 0.0
                node_to_update = node
                while node_to_update:
                    node_to_update.visits += 1
                    node_to_update.node_value += reward
                    node_to_update = node_to_update.parent


        logging.info(f"Number of rollouts: {rollouts}")
        return self

    def rollout(self) -> float:
        """
        Evaluate the current node's board state using the trained model.
        """

        board_state = copy_board(self.game_state)
        if gameIsOver(board_state):

            return EndValue(board_state, AI_PLAYER)
        result = model_evaluation(board_state, AI_PLAYER, self.starting_player)

        return result

    def next(self):
        """
        Retrieve the child node with the highest visit count.
        """

        if self.done:
            raise ValueError("Game has ended. No next move available.")
        if not self.child_nodes:
            raise ValueError("No children available. Ensure exploration has been performed.")
        best_child = max(self.child_nodes.values(), key=lambda child: child.visits)


        return best_child, best_child.action_index

    def movePlayer(self, playerMove):
        """
        Update the current node based on a player's move.
        """

        if playerMove in self.child_nodes:

            new_root = self.child_nodes[playerMove]
        else:

            new_board = copy_board(self.game_state)
            new_board.play_move(playerMove)
            done = gameIsOver(new_board)
            new_root = Node(new_board, done, self, playerMove, self.starting_player)
            self.child_nodes[playerMove] = new_root

        return new_root
