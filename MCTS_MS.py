import random
import time
import math
from copy import deepcopy
import concurrent.futures
import logging

from minimaxAlphaBeta import MiniMaxAlphaBeta
from utility_functions import gameIsOver, AI_PLAYER, HUMAN_PLAYER, EndValue

# Logging configuration
logging.basicConfig(
    filename='rollouts.log',
    format='%(asctime)s - %(message)s',
    level=logging.INFO  # Change to DEBUG for more detailed logs
)

# Global constants for bias calculations
CENTER_COL = 3
MAX_DISTANCE = 3
BIAS_STRENGTH = 0.05

# Precompute bias lookup for a standard 7-column board
bias_lookup = {
    col: 1 + BIAS_STRENGTH * (MAX_DISTANCE - abs(col - CENTER_COL))
    for col in range(7)
}


def copy_board(bitboard):
    """
    Creates a copy of the given bitboard.
    Uses the built-in copy() method if available, else falls back to deepcopy.
    """
    if hasattr(bitboard, 'copy'):
        return bitboard.copy()
    return deepcopy(bitboard)


def count_moves(bitboard):
    """
    Counts the moves made by each player based on the bitboard representation.
    Assumes:
      - bitboard.board1 contains the human ('x') moves
      - bitboard.board2 contains the AI ('o') moves
    """
    human_moves = bin(bitboard.board1).count("1")
    ai_moves = bin(bitboard.board2).count("1")
    return ai_moves, human_moves


def get_move_column(move):
    """
    Extracts the column index from a move.
    For a tuple move, returns the second element (the column).
    For an int move, returns it directly.
    """
    if isinstance(move, int):
        return move
    elif isinstance(move, tuple):
        return move[1]
    return move


def biased_random_move(valid_moves):
    """
    Chooses a move from the valid moves list with a slight bias toward the center.
    Uses a precomputed bias lookup table for faster access.
    """
    weights = [bias_lookup[get_move_column(move)] for move in valid_moves]
    return random.choices(valid_moves, weights=weights, k=1)[0]


def rollout_simulation(game_state, minimax_depth):
    """
    Performs a rollout simulation starting from a given game state.
    Uses an initial minimax move followed by a random rollout with center bias.
    """
    board_state = copy_board(game_state)

    # If the game is already over, return its terminal evaluation.
    if gameIsOver(board_state):
        return EndValue(board_state, AI_PLAYER)

    # --- First move via Minimax ---
    best_move, _ = MiniMaxAlphaBeta(board_state, minimax_depth, board_state.current_player)
    if best_move is not None:
        board_state.play_move(best_move)
        if gameIsOver(board_state):
            return EndValue(board_state, AI_PLAYER)

    # Switch the active player for the subsequent random rollout phase.
    current_player = HUMAN_PLAYER if board_state.current_player == AI_PLAYER else AI_PLAYER

    # --- Random Rollout Phase with Center Bias ---
    while not gameIsOver(board_state):
        valid_moves = board_state.get_valid_moves()
        if not valid_moves:
            return 0.5  # Return draw value if no moves are available.
        action = biased_random_move(valid_moves)
        board_state.current_player = current_player
        board_state.play_move(action)
        current_player = HUMAN_PLAYER if current_player == AI_PLAYER else AI_PLAYER

    return EndValue(board_state, AI_PLAYER)


class Node:
    __slots__ = ('parent', 'child_nodes', 'visits', 'node_value', 'game_state', 'done',
                 'action_index', 'c', 'starting_player')

    def __init__(self, game_state, done, parent_node, action_index, starting_player):
        self.parent = parent_node
        self.child_nodes = {}  # Dictionary mapping action -> Node
        self.visits = 0
        self.node_value = 0.0
        self.game_state = game_state  # Instance of the bitboard
        self.done = done
        self.action_index = action_index  # The move that led to this node (e.g. (row, col))
        self.c = 1.2  # Exploration constant
        self.starting_player = starting_player

    def getUCTscore(self):
        """
        Computes the UCT score for this node.
        Adds a center bias to favor moves near the center.
        """
        if self.visits == 0:
            return float('inf')
        parent_visits = self.parent.visits if self.parent else 1
        log_parent = math.log(parent_visits) if parent_visits > 0 else 0
        exploitation = self.node_value / self.visits
        exploration = self.c * math.sqrt(log_parent / self.visits)
        move_col = get_move_column(self.action_index)
        center_bias = BIAS_STRENGTH * (MAX_DISTANCE - abs(move_col - CENTER_COL))
        return exploitation + exploration + center_bias

    def create_child_nodes(self):
        """
        Expands the current node by generating all possible child nodes
        based on the valid moves from the current game state.
        """
        valid_moves = self.game_state.get_valid_moves()
        for action in valid_moves:
            new_board = copy_board(self.game_state)
            new_board.play_move(action)
            done = gameIsOver(new_board)
            self.child_nodes[action] = Node(new_board, done, self, action, self.starting_player)

    def explore(self, minimax_depth=2, min_rollouts=50000000, min_time=0.0, max_time=6.0, batch_size=32):
        """
        Explores the search tree using parallel rollouts.
        Instead of running each rollout synchronously, batches of rollouts are executed in parallel.
        """
        start_time = time.perf_counter()
        rollouts = 0
        batch = []  # List of tuples: (future, node)

        # Using ProcessPoolExecutor for true parallelism in CPU-bound tasks.
        with concurrent.futures.ProcessPoolExecutor() as executor:
            while True:
                elapsed = time.perf_counter() - start_time
                if (rollouts >= min_rollouts and elapsed >= min_time) or (elapsed >= max_time):
                    break

                current = self
                # --- Selection ---
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

                # --- Expansion ---
                if not current.done:
                    current.create_child_nodes()
                    if current.child_nodes:
                        current = random.choice(list(current.child_nodes.values()))

                # Schedule the rollout simulation in parallel.
                future = executor.submit(rollout_simulation, current.game_state, minimax_depth)
                batch.append((future, current))
                rollouts += 1

                # Process the batch once the batch size is reached.
                if len(batch) >= batch_size:
                    for future, node in batch:
                        try:
                            reward = future.result()
                        except Exception:
                            reward = 0.0  # Fallback if simulation fails.
                        # --- Backpropagation ---
                        node_to_update = node
                        while node_to_update:
                            node_to_update.visits += 1
                            node_to_update.node_value += reward
                            node_to_update = node_to_update.parent
                    batch.clear()

            # Process any remaining futures in the batch.
            for future, node in batch:
                try:
                    reward = future.result()
                except Exception:
                    reward = 0.0
                node_to_update = node
                while node_to_update:
                    node_to_update.visits += 1
                    node_to_update.node_value += reward
                    node_to_update = node_to_update.parent

        logging.info(f"Rollouts completed: {rollouts}")
        return self

    def rollout(self, minimax_depth: int = 2) -> float:
        """
        Executes a single rollout with a center-biased move selection.
        """
        board_state = copy_board(self.game_state)
        if gameIsOver(board_state):
            return EndValue(board_state, AI_PLAYER)

        best_move, _ = MiniMaxAlphaBeta(board_state, minimax_depth, board_state.current_player)
        if best_move is not None:
            board_state.play_move(best_move)
            if gameIsOver(board_state):
                return EndValue(board_state, AI_PLAYER)

        current_player = HUMAN_PLAYER if board_state.current_player == AI_PLAYER else AI_PLAYER

        while not gameIsOver(board_state):
            valid_moves = board_state.get_valid_moves()
            if not valid_moves:
                return 0.5  # Return draw value if no moves available.
            action = biased_random_move(valid_moves)
            board_state.current_player = current_player
            board_state.play_move(action)
            current_player = HUMAN_PLAYER if current_player == AI_PLAYER else AI_PLAYER

        return EndValue(board_state, AI_PLAYER)

    def next(self):
        """
        Selects the best next move based on the highest visit count.
        """
        if self.done:
            raise ValueError("Game over. No next move available.")
        if not self.child_nodes:
            raise ValueError("No child nodes available. Run exploration first.")

        best_child = max(self.child_nodes.values(), key=lambda child: child.visits)
        best_child.game_state.print_board()
        return best_child, best_child.action_index

    def movePlayer(self, playerMove):
        """
        Updates the search tree according to the player's move.
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
