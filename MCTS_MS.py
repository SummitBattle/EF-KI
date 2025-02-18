import random
import time
import math
from copy import deepcopy
import logging
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

from utility_functions import gameIsOver

# Enable XLA for potential speed improvements.
tf.config.optimizer.set_jit(True)

# -------------------------------------------------------
# Logging configuration – set to INFO or higher for production.
# -------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    filename='mcts_debug.log',
    filemode='w'
)

# -------------------------------------------------------
# Constants for demonstration.
# -------------------------------------------------------
AI_PLAYER = 1
HUMAN_PLAYER = -1

# -------------------------------------------------------
# Load the trained Keras model.
# -------------------------------------------------------
try:
    model = load_model('connect4_model.h5')
    logging.info("Keras model loaded successfully.")
except Exception as e:
    logging.exception("Failed to load model: %s", e)
    raise

# -------------------------------------------------------
# Convert the Keras model to a TFLite model to reduce inference overhead.
# -------------------------------------------------------
try:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open("model.tflite", "wb") as f:
        f.write(tflite_model)
    logging.info("TFLite model converted and saved.")
except Exception as e:
    logging.exception("Failed to convert model to TFLite: %s", e)
    raise

# -------------------------------------------------------
# Set up the TFLite interpreter.
# -------------------------------------------------------
tflite_interpreter = tf.lite.Interpreter(model_path="model.tflite")
tflite_interpreter.allocate_tensors()
input_details = tflite_interpreter.get_input_details()
output_details = tflite_interpreter.get_output_details()


def copy_board(bitboard):
    """Fast board copy using built-in copy if available."""
    return bitboard.copy() if hasattr(bitboard, 'copy') else deepcopy(bitboard)
def tflite_batched_predict(inputs: np.ndarray) -> np.ndarray:
    """
    Perform batched prediction using the TFLite interpreter.
    Resizes the input tensor only if necessary.
    """
    current_shape = tflite_interpreter.get_tensor(input_details[0]['index']).shape
    if tuple(inputs.shape) != tuple(current_shape):
        tflite_interpreter.resize_tensor_input(input_details[0]['index'], inputs.shape)
        tflite_interpreter.allocate_tensors()
    tflite_interpreter.set_tensor(input_details[0]['index'], inputs)
    tflite_interpreter.invoke()
    return tflite_interpreter.get_tensor(output_details[0]['index'])


# -------------------------------------------------------
# Helper Functions and Caches
# -------------------------------------------------------
_bitboard_cache: Dict[int, List[int]] = {}

def bitboard_to_array(bitboard_int: int) -> List[int]:
    """
    Convert an integer bitboard into a list of 42 bits.
    Optimized using bit shifting instead of string conversion.
    """
    if bitboard_int in _bitboard_cache:
        return _bitboard_cache[bitboard_int]
    # Use bit shifting to extract each bit (most significant bit first)
    arr = [(bitboard_int >> shift) & 1 for shift in range(41, -1, -1)]
    _bitboard_cache[bitboard_int] = arr
    return arr

def convert_bitboard_to_feature_array(game_state: Any) -> List[int]:
    """
    Convert the game's bitboard representation into a feature array of length 42.
      - game_state.board1: human moves (encoded as -1)
      - game_state.board2: AI moves (encoded as 1)
      - Empty cells: 0
    """
    human_arr = bitboard_to_array(game_state.board1)
    ai_arr = bitboard_to_array(game_state.board2)
    # Inline conditional: if there's an AI move, use 1; if human move, use -1; else 0.
    feature_arr = [1 if a == 1 else (-1 if h == 1 else 0)
                   for h, a in zip(human_arr, ai_arr)]
    return feature_arr

_prediction_cache: Dict[Tuple[int, ...], float] = {}

# -------------------------------------------------------
# MCTS Node Class with Batched Simulation and TFLite Inference
# -------------------------------------------------------
class Node:
    __slots__ = ('parent', 'child_nodes', 'visits', 'node_value', 'game_state', 'done',
                 'action_index', 'c', 'reward', 'starting_player', 'feature_arr', 'feature_tuple')

    def __init__(self, game_state: Any, done: bool, parent_node: Optional['Node'],
                 action_index: Any, starting_player: int):
        self.parent = parent_node
        self.child_nodes: Dict[Any, 'Node'] = {}  # Mapping: action -> Node
        self.visits: int = 0
        self.node_value: float = 0.0
        self.game_state = game_state
        self.done: bool = done
        self.action_index = action_index  # The move that led to this node
        self.c: float = 1.6  # Exploration constant
        self.reward: float = 0.0
        self.starting_player: int = starting_player
        # Precompute the feature array and tuple for caching.
        self.feature_arr: List[int] = convert_bitboard_to_feature_array(game_state)
        self.feature_tuple: Tuple[int, ...] = tuple(self.feature_arr)

    def getUCTscore_inline(self, parent_visits: int) -> float:
        """
        Compute the UCT score using parent's visits.
        Adds a small center bias based on the column index.
        """
        if self.visits == 0:
            return float('inf')
        avg_value = self.node_value / self.visits
        exploration_term = self.c * math.sqrt(math.log(parent_visits) / self.visits)
        # Determine column from the move; supports tuple moves or single integer moves.
        move_col = self.action_index[1] if isinstance(self.action_index, tuple) else self.action_index
        center_bias = 0.009 * (3 - abs(move_col - 3))
        return avg_value + exploration_term + center_bias

    def create_child_nodes(self) -> None:
        """Generate child nodes for all valid moves from the current state."""
        valid_moves = self.game_state.get_valid_moves()
        for action in valid_moves:
            new_board = copy_board(self.game_state)
            new_board.play_move(action)
            done = gameIsOver(new_board)
            self.child_nodes[action] = Node(new_board, done, self, action, self.starting_player)

    def _process_batch(self, batch_nodes: List['Node']) -> None:
        """
        Process a batch of nodes using TFLite predictions and backpropagate rewards.
        """
        batch_size = len(batch_nodes)
        # Preallocate a NumPy array for batched board inputs.
        board_inputs = np.empty((batch_size, 42), dtype=np.float32)
        keys = [node.feature_tuple for node in batch_nodes]
        for i, node in enumerate(batch_nodes):
            board_inputs[i, :] = np.array(node.feature_arr, dtype=np.float32)
        try:
            predictions = tflite_batched_predict(board_inputs)
        except Exception as e:
            logging.exception("Batched TFLite prediction failed: %s", e)
            predictions = None

        if predictions is not None:
            pred_indices = np.argmax(predictions, axis=1)
        else:
            pred_indices = [None] * batch_size

        for i, node in enumerate(batch_nodes):
            key = keys[i]
            if key in _prediction_cache:
                reward = _prediction_cache[key]
            else:
                # Map prediction to a reward:
                #   index 1 -> win (1)
                #   index 0 -> loss (-1)
                #   else -> tie/uncertain (0.5)
                reward = 1 if pred_indices[i] == 1 else (-1 if pred_indices[i] == 0 else 0.5)
                _prediction_cache[key] = reward

            # Backpropagation: propagate reward up the tree.
            node_to_update = node
            while node_to_update is not None:
                node_to_update.visits += 1
                node_to_update.node_value += reward
                node_to_update = node_to_update.parent

    def explore(self, min_rollouts: int = 50000, min_time: float = 5.0, max_time: float = 5.0,
                batch_size: int = 32) -> 'Node':
        """
        Explore the MCTS tree using batched rollouts.
        Stops when the minimum rollouts/time have been reached or maximum time exceeded.
        """
        start_time = time.perf_counter()
        rollouts = 0
        batch_nodes: List[Node] = []

        while True:
            elapsed = time.perf_counter() - start_time
            if (rollouts >= min_rollouts and elapsed >= min_time) or (elapsed >= max_time):
                logging.info("Stopping exploration: %d rollouts, %.2f sec elapsed.", rollouts, elapsed)
                break

            current = self
            # --- Selection Phase ---
            while current.child_nodes:
                parent_visits = current.visits if current.visits > 0 else 1
                best_score = -float('inf')
                best_children: List[Node] = []
                for child in current.child_nodes.values():
                    score = child.getUCTscore_inline(parent_visits)
                    if score > best_score:
                        best_score = score
                        best_children = [child]
                    elif score == best_score:
                        best_children.append(child)
                current = random.choice(best_children)

            # --- Expansion Phase ---
            if not current.done:
                current.create_child_nodes()
                if current.child_nodes:
                    current = random.choice(list(current.child_nodes.values()))

            batch_nodes.append(current)

            # --- Batched Simulation Phase ---
            if len(batch_nodes) >= batch_size:
                self._process_batch(batch_nodes)
                rollouts += len(batch_nodes)
                batch_nodes.clear()  # Reset batch
                if rollouts % 1000 < batch_size:  # Log progress every ~1000 rollouts.
                    logging.info("Rollouts: %d, elapsed time: %.2f sec", rollouts, elapsed)

        logging.info("Total rollouts performed: %d", rollouts)
        return self

    def next(self) -> Tuple['Node', Any]:
        """
        Choose the next move based on the child node with the highest average reward (Q-value).
        """
        if self.done:
            raise ValueError("Game has ended. No next move available.")
        if not self.child_nodes:
            raise ValueError("No children available. Ensure exploration has been performed.")
        best_child = max(
            self.child_nodes.values(),
            key=lambda child: (child.node_value / child.visits) if child.visits > 0 else float('-inf')
        )
        avg_q = best_child.node_value / best_child.visits if best_child.visits > 0 else 0.0
        logging.info("Next move selected: %s with average Q-value: %.3f", best_child.action_index, avg_q)
        return best_child, best_child.action_index

    def movePlayer(self, playerMove: Any) -> 'Node':
        """
        Update the tree based on the player's move.
        """
        logging.info("Player made move: %s", playerMove)
        if playerMove in self.child_nodes:
            new_root = self.child_nodes[playerMove]
        else:
            new_board = copy_board(self.game_state)
            new_board.play_move(playerMove)
            done = gameIsOver(new_board)
            new_root = Node(new_board, done, self, playerMove, self.starting_player)
            self.child_nodes[playerMove] = new_root
        return new_root
