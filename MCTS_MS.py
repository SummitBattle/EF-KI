import random
import time
import math
from copy import deepcopy

from utility_functions import gameIsOver, AI_PLAYER, HUMAN_PLAYER, EndValue
import numpy as np

from tensorflow.keras.models import load_model

# Set up logging configuration to log to a file


# Load the trained model (Make sure the path to the model is correct)
model = load_model('connect4_model.h5')


def copy_board(bitboard):
    """Fast board copy: use a built-in copy method if available."""
    if hasattr(bitboard, 'copy'):
        return bitboard.copy()
    return deepcopy(bitboard)


def count_moves(bitboard):
    """
    Count the number of moves for each player using the bitboard representation.
    We assume:
      - bitboard.board1 holds the human moves ('x')
      - bitboard.board2 holds the AI moves ('o')
    """
    human_moves = bin(bitboard.board1).count("1")
    ai_moves = bin(bitboard.board2).count("1")
    return ai_moves, human_moves


def get_move_column(move):
    """
    Helper to extract the column index from a move.
    Assumes that if the move is a tuple, its second element is the column.
    If the move is an int, it is the column.
    """
    if isinstance(move, int):
        return move
    elif isinstance(move, tuple):
        return move[1]
    return move


def biased_random_move(valid_moves, center_col=3, bias_strength=0.001):
    """
    Select a move with a slight center bias.
    For Connect 4 (7 columns), the center column is 3.
    Moves closer to the center receive a higher weight.
    """
    weights = []
    max_distance = 3  # For a standard Connect 4 board
    for move in valid_moves:
        col = get_move_column(move)
        weight = 1 + bias_strength * (max_distance - abs(col - center_col))
        weights.append(weight)
    chosen_move = random.choices(valid_moves, weights=weights, k=1)[0]

    return chosen_move


# --- New helper functions to convert the bitboard into a feature array ---

def bitboard_to_array(bitboard_int):
    """
    Convert an integer bitboard into a list of 42 bits.
    Assumes that the bitboard is represented with at most 42 relevant bits.
    """
    # Convert to binary string, remove the "0b" prefix, and pad with zeros to length 42.
    binary_str = bin(bitboard_int)[2:].zfill(42)
    return [int(b) for b in binary_str]


def convert_bitboard_to_feature_array(game_state):
    """
    Convert the game's bitboard representation into a feature array of length 42.
    We assume:
      - game_state.board1 holds human moves (will be encoded as -1)
      - game_state.board2 holds AI moves (will be encoded as 1)
      - Empty cells are 0.
    """
    human_arr = bitboard_to_array(game_state.board1)
    ai_arr = bitboard_to_array(game_state.board2)
    feature_arr = []
    for h, a in zip(human_arr, ai_arr):
        if a == 1:
            feature_arr.append(1)
        elif h == 1:
            feature_arr.append(-1)
        else:
            feature_arr.append(0)

    return feature_arr


def rollout_simulation(game_state):
    """
    Evaluate the given game_state using the pre-trained model.
    This function now converts the bitboard to a feature vector of length 42,
    which is then fed to the model.

    Returns:
      - 1 if the model predicts an AI win,
      - -1 if the model predicts a human win,
      - 0.5 for a tie or uncertainty.
    """
    board_state = copy_board(game_state)

    if gameIsOver(board_state):
        outcome = EndValue(board_state, AI_PLAYER)

        return outcome

    # Convert the bitboard to the feature array expected by the model.
    feature_arr = convert_bitboard_to_feature_array(board_state)
    board_input = np.array(feature_arr).reshape(1, -1)  # Shape: (1, 42)


    try:
        prediction = model.predict(board_input)
    except Exception as e:

        return 0.0

    predicted_winner = np.argmax(prediction)


    if predicted_winner == 1:  # AI wins
        return 1
    elif predicted_winner == 0:  # Human wins
        return -1
    else:
        return 0.5  # Tie or uncertain


class Node:
    __slots__ = ('parent', 'child_nodes', 'visits', 'node_value', 'game_state', 'done',
                 'action_index', 'c', 'reward', 'starting_player')

    def __init__(self, game_state, done, parent_node, action_index, starting_player):
        self.parent = parent_node
        self.child_nodes = {}  # action -> Node
        self.visits = 0
        self.node_value = 0.0
        self.game_state = game_state  # BitBoard instance
        self.done = done
        self.action_index = action_index  # The move that led to this node (tuple or int)
        self.c = 1.41  # Exploration constant (~sqrt(2))
        self.reward = 0.0
        self.starting_player = starting_player

    def getUCTscore(self, center_col=3, bias_strength=0.001):
        if self.visits == 0:
            return float('inf')
        parent_visits = self.parent.visits if self.parent else 1

        # Standard UCT formula
        uct_score = (self.node_value / self.visits) + self.c * math.sqrt(math.log(parent_visits) / self.visits)

        if isinstance(self.action_index, tuple):
            move_col = self.action_index[1]
        else:
            move_col = self.action_index

        max_distance = 3  # For a 7-column board
        center_bias = bias_strength * (max_distance - abs(move_col - center_col))

        score = uct_score + center_bias

        return score

    def create_child_nodes(self):
        valid_moves = self.game_state.get_valid_moves()

        for action in valid_moves:
            new_board = copy_board(self.game_state)
            new_board.play_move(action)
            done = gameIsOver(new_board)
            self.child_nodes[action] = Node(new_board, done, self, action, self.starting_player)

    def explore(self, min_rollouts=20000, min_time=4.0, max_time=4.0):
        """
        Explore the MCTS tree using rollouts that evaluate the board state
        directly with the pre-trained model.
        NOTE: The stopping conditions depend on both time and rollout count.
        If you're only seeing ~100 rollouts per move, consider increasing max_time.
        """
        start_time = time.perf_counter()
        rollouts = 0

        while True:
            elapsed = time.perf_counter() - start_time
            if (rollouts >= min_rollouts and elapsed >= min_time) or (elapsed >= max_time):
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
                current = random.choice(best_children)



            if not current.done:
                current.create_child_nodes()
                if current.child_nodes:
                    current = random.choice(list(current.child_nodes.values()))



            try:
                reward = rollout_simulation(current.game_state)

            except Exception as e:

                reward = 0.0


            node_to_update = current
            while node_to_update:
                node_to_update.visits += 1
                node_to_update.node_value += reward

                node_to_update = node_to_update.parent

            rollouts += 1


        return self

    def next(self):
        if self.done:
            raise ValueError("Game has ended. No next move available.")
        if not self.child_nodes:
            raise ValueError("No children available. Ensure exploration has been performed.")

        best_child = max(self.child_nodes.values(), key=lambda child: child.visits)

        best_child.game_state.print_board()
        return best_child, best_child.action_index

    def movePlayer(self, playerMove):
        if playerMove in self.child_nodes:
            new_root = self.child_nodes[playerMove]
        else:
            new_board = copy_board(self.game_state)
            new_board.play_move(playerMove)
            done = gameIsOver(new_board)
            new_root = Node(new_board, done, self, playerMove, self.starting_player)
            self.child_nodes[playerMove] = new_root

        return new_root

# (Optional) Remove or comment out any legacy rollout function that uses minimax.
