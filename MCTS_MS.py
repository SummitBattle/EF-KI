import random
import time
import math
from copy import deepcopy
import concurrent.futures
import logging

from board import BitBoard
from minimaxAlphaBeta import minimax_alpha_beta
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
BIAS_STRENGTH = 0.13 # Increased for stronger bias toward the center


def copy_board(bitboard):
    """
    Creates a copy of the given bitboard efficiently.
    """
    return deepcopy(bitboard)


def get_move_column(move):
    """
    Extracts the column index from a move.
    """
    return move if isinstance(move, int) else move[1]


def guided_rollout_move(valid_moves, board_state):
    """
    Chooses a move based on heuristic-guided rollouts.
    - Prefers forming two-in-a-row or blocking threats.
    - Still biases towards the center.
    """
    best_move = None
    best_score = -float('inf')
    for move in valid_moves:
        new_board = copy_board(board_state)
        new_board.play_move(move)
        score = heuristic_evaluation(new_board)
        if score > best_score:
            best_score = score
            best_move = move
    return best_move if best_move else random.choice(valid_moves)


def heuristic_evaluation(board_state):
    """
    Evaluates the board to favor strong early and mid-game moves.
    """
    return sum(bin(board_state.board2).count("1") - bin(board_state.board1).count("1"))


def rollout_simulation(game_state, minimax_depth):
    """
    Performs a rollout simulation using hybrid rollouts.
    """
    board_state = copy_board(game_state)

    if gameIsOver(board_state):
        return EndValue(board_state, AI_PLAYER)

    best_move, _ = minimax_alpha_beta(board_state, 2, -math.inf, math.inf, board_state.current_player)
    if best_move is not None:
        board_state.play_move(best_move)
        if gameIsOver(board_state):
            return EndValue(board_state, AI_PLAYER)

    current_player = HUMAN_PLAYER if board_state.current_player == AI_PLAYER else AI_PLAYER

    while not gameIsOver(board_state):
        valid_moves = board_state.get_valid_moves()
        if not valid_moves:
            return 0.5
        action = guided_rollout_move(valid_moves, board_state)
        board_state.current_player = current_player
        board_state.play_move(action)
        current_player = HUMAN_PLAYER if current_player == AI_PLAYER else AI_PLAYER

    return EndValue(board_state, AI_PLAYER)


class Node:
    def __init__(self, game_state, done, parent_node, action_index, starting_player):
        self.parent = parent_node
        self.child_nodes = {}
        self.visits = 0
        self.node_value = 0.0
        self.game_state = game_state
        self.done = done
        self.action_index = action_index
        self.c = 1.3 # Adjusted exploration factor
        self.starting_player = starting_player

    def getUCTscore(self):
        if self.visits == 0:
            return float('inf')
        parent_visits = self.parent.visits if self.parent else 1
        log_parent = math.log(parent_visits) if parent_visits > 0 else 0
        exploitation = self.node_value / self.visits
        exploration = self.c * math.sqrt(log_parent / self.visits)
        move_col = get_move_column(self.action_index)
        center_bias = BIAS_STRENGTH * (MAX_DISTANCE - abs(move_col - CENTER_COL))
        return exploitation + exploration + center_bias

    def explore(self, minimax_depth=3, min_rollouts=100000, max_time=6.0, batch_size=32):
        """
        Explores using parallelized guided rollouts with an immediate threat detection check.
        """

        start_time = time.perf_counter()
        rollouts = 0
        batch = []

        with concurrent.futures.ThreadPoolExecutor() as executor:
            while True:
                elapsed = time.perf_counter() - start_time
                if rollouts >= min_rollouts or elapsed >= max_time:
                    break

                current = self
                while current.child_nodes:
                    current = max(current.child_nodes.values(), key=lambda c: c.getUCTscore())

                if not current.done:
                    valid_moves = current.game_state.get_valid_moves()
                    for action in valid_moves:
                        new_board = copy_board(current.game_state)
                        new_board.play_move(action)
                        done = gameIsOver(new_board)
                        current.child_nodes[action] = Node(new_board, done, current, action, self.starting_player)
                future = executor.submit(rollout_simulation, current.game_state, minimax_depth)
                batch.append((future, current))
                rollouts += 1

                if len(batch) >= batch_size:
                    for future, node in batch:
                        try:
                            reward = future.result()
                        except Exception:
                            reward = 0.0
                        while node:
                            node.visits += 1
                            node.node_value += reward
                            node = node.parent
                    batch.clear()

        logging.info(f"Rollouts completed: {rollouts}")
        return self

    def next(self):
        """
        Selects the best next move based on the highest visit count.
        Returns both the best child node and the corresponding action.
        """
        if self.done:
            raise ValueError("Game over. No next move available.")
        if not self.child_nodes:
            raise ValueError("No child nodes available. Run exploration first.")

        best_child = max(self.child_nodes.values(), key=lambda child: child.visits)

        return best_child, best_child.action_index  # Ensure both Node and move are returned

    def movePlayer(self, playerMove):
        if playerMove in self.child_nodes:
            return self.child_nodes[playerMove]
        new_board = copy_board(self.game_state)
        new_board.play_move(playerMove)
        done = gameIsOver(new_board)
        new_node = Node(new_board, done, self, playerMove, self.starting_player)
        self.child_nodes[playerMove] = new_node
